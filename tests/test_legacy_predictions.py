from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _formal_context import seed_legacy_snapshot_source
from _training_refs import seed_training_references

from football_data_platform.domain.ids import MatchId, ModelRunId, RawAssetId, TeamId
from football_data_platform.domain.predictions import (
    ScorePrediction,
    _build_score_prediction,
    parse_prediction_payload,
    prediction_payload,
    score_grid_composition_payload,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
    snapshot_payload,
)
from football_data_platform.domain.training import (
    MODEL_OUTPUT_REF_PREFIX,
    ModelRunArtifact,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contribution,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.models.score_grid import DixonColesGrid
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.legacy_predictions import (
    LegacyModelLineageLimitation,
    LegacyPredictionAuditError,
    LegacyPredictionAuditLimitation,
    LegacyScorePredictionV3,
    load_legacy_prediction_v3_for_audit,
)
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)

SNAPSHOT_AS_OF = datetime(2025, 8, 22, 14, 0, tzinfo=UTC)
GENERATED_AT = datetime(2026, 7, 17, 8, 0, tzinfo=UTC)
REFERENCE_TIME = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
KICKOFF = SNAPSHOT_AS_OF + timedelta(days=1)
MATCH_ID = MatchId("match:legacy-prediction")
HOME_TEAM_ID = TeamId("team:legacy-home")
AWAY_TEAM_ID = TeamId("team:legacy-away")
MODEL_VERSION = "dixon-coles-composed/1"
REAL_V3_REFERENCE = "prediction:191527126b3278b2a5e788b356a9dd40c33b2a479f3c9616352a761d94b339c2"


@dataclass(frozen=True, slots=True)
class LegacyFixture:
    layout: DataLayout
    reference: str
    prediction_path: Path
    composition_path: Path
    composition_manifest_path: Path
    snapshot_manifest_path: Path
    training_raw_object_path: Path
    model_artifact_path: Path
    model_manifest_path: Path
    snapshot: Any


def test_composed_v3_loads_as_calibration_unverifiable_audit_view(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")

    prediction = load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)

    assert isinstance(prediction, LegacyScorePredictionV3)
    assert not isinstance(prediction, ScorePrediction)
    assert prediction.audit_only is True
    assert (
        prediction.model_lineage_limitation
        is LegacyModelLineageLimitation.TRAINING_DATASET_POLICY_UNVERIFIABLE
    )
    assert (
        prediction.audit_limitation
        is LegacyPredictionAuditLimitation.CALIBRATION_POLICY_UNVERIFIABLE
    )
    assert len(prediction.score_cells) == 144
    assert tuple(market[0] for market in prediction.markets) == (
        "result_90",
        "home_handicap_3way",
        "total_goals",
    )
    with pytest.raises(FrozenInstanceError):
        prediction.lambda_home = 9.0  # type: ignore[misc]


def test_legacy_inline_v3_is_permanently_provenance_incomplete(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="legacy-inline/1")

    prediction = load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)

    assert prediction.audit_limitation is LegacyPredictionAuditLimitation.PROVENANCE_INCOMPLETE
    assert prediction.contribution_keys == ()
    assert prediction.contribution_multipliers == ()
    assert prediction.baseline_lambda_home == prediction.lambda_home
    assert prediction.baseline_lambda_away == prediction.lambda_away


def test_real_repository_v3_remains_loadable() -> None:
    data_root = Path(__file__).parents[1] / "data"
    layout = DataLayout(data_root)
    prediction_path = _content_path(layout.derived, "predictions", REAL_V3_REFERENCE)
    if not data_root.is_dir() or not prediction_path.is_file():
        pytest.skip("repository-local legacy v3 audit fixture is unavailable")

    prediction = load_legacy_prediction_v3_for_audit(layout, REAL_V3_REFERENCE)

    assert prediction.id == REAL_V3_REFERENCE
    assert prediction.audit_only is True
    assert (
        prediction.model_lineage_limitation
        is LegacyModelLineageLimitation.TRAINING_DATASET_POLICY_UNVERIFIABLE
    )


def test_v3_audit_view_cannot_enter_current_prediction_contract(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    legacy = load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)
    payload = _read_json(fixture.prediction_path)

    with pytest.raises(ValueError, match="unsupported prediction schema_version 3"):
        parse_prediction_payload(payload, snapshot=None, snapshot_validator=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="audit-only"):
        DerivedArchive(fixture.layout).write_prediction(legacy, snapshot=fixture.snapshot)  # type: ignore[arg-type]


def test_v4_with_legacy_snapshot_is_audit_readable_but_not_formal(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    derived = DerivedArchive(fixture.layout)
    baseline = next(
        feature for feature in fixture.snapshot.features if feature.name == "team_baseline"
    )
    context = next(
        feature for feature in fixture.snapshot.features if feature.name == "match_context"
    )
    composition = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        (context_contribution(context.value, source_ref=context.source_ref),),
    )
    legacy_v3_payload = _read_json(fixture.prediction_path)
    prediction = _build_score_prediction(
        snapshot=fixture.snapshot,
        snapshot_validator=derived,
        model_run_id=ModelRunId(legacy_v3_payload["model_run_id"]),
        model_version=legacy_v3_payload["model_version"],
        generated_at=GENERATED_AT,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.1,
        max_goals=11,
        input_refs=(),
        model_run_validator=None,
        expected_goals=composition,
        composition=None,
        composition_artifact_ref=None,
        calibration_versions=None,
        enforce_current=False,
    )
    composition_document = score_grid_composition_payload(prediction)
    _write_json(
        derived.score_grid_composition_path(prediction.composition_artifact_ref),
        {"id": prediction.composition_artifact_ref, **composition_document},
    )
    composition_manifest = DerivedArtifactManifest.create(
        artifact_type="score-grid-composition",
        schema_version=composition_document["schema_version"],
        payload=composition_document,
        generated_at=prediction.generated_at,
        transform_version=prediction.composition_version,
        code_version="football-data-platform/0.1.0",
        input_refs=prediction.input_refs,
        output_refs=(prediction.composition_artifact_ref,),
        status="succeeded",
        quality=prediction.snapshot_quality_status,
    )
    _write_json(
        derived.artifact_manifest_path(composition_manifest.id),
        composition_manifest.to_payload(),
    )
    document = prediction_payload(prediction)
    _write_json(_content_path(fixture.layout.derived, "predictions", prediction.id.value), document)
    prediction_manifest = DerivedArtifactManifest.create(
        artifact_type="prediction",
        schema_version=prediction.schema_version,
        payload=document,
        generated_at=prediction.generated_at,
        transform_version=prediction.model_version,
        code_version="football-data-platform/0.1.0",
        input_refs=(*prediction.input_refs, prediction.composition_artifact_ref),
        output_refs=(prediction.id.value,),
        status="succeeded",
        quality=prediction.snapshot_quality_status,
    )
    _write_json(
        derived.artifact_manifest_path(prediction_manifest.id),
        prediction_manifest.to_payload(),
    )
    store = TrainingArtifactStore(fixture.layout)

    assert store.load_prediction_for_audit(prediction.id.value) == prediction
    with pytest.raises(TrainingArtifactConflict, match="audit-only"):
        store.load_verified_prediction(prediction.id.value)
    with pytest.raises(ValueError, match="audit-only"):
        derived.write_prediction(prediction, snapshot=fixture.snapshot)


def test_v3_loader_rejects_tampered_prediction_grid(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    payload = _read_json(fixture.prediction_path)
    payload["score_cells"][0]["probability"] = 0.99
    _write_json(fixture.prediction_path, payload)

    with pytest.raises(LegacyPredictionAuditError, match="score cells"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_tampered_composition_manifest(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    manifest = _read_json(fixture.composition_manifest_path)
    manifest["quality"] = "preview"
    _write_json(fixture.composition_manifest_path, manifest)

    with pytest.raises(LegacyPredictionAuditError, match="manifest"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_missing_snapshot_manifest(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    fixture.snapshot_manifest_path.unlink()

    with pytest.raises(LegacyPredictionAuditError, match="exactly one immutable manifest"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_tampered_snapshot_manifest(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    manifest = _read_json(fixture.snapshot_manifest_path)
    manifest["quality"] = "preview"
    _write_json(fixture.snapshot_manifest_path, manifest)

    with pytest.raises(LegacyPredictionAuditError, match="manifest"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_missing_training_raw_evidence(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    fixture.training_raw_object_path.unlink()

    with pytest.raises(LegacyPredictionAuditError, match="training dataset input reference"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_rehashed_training_label_extra_key(tmp_path: Path) -> None:
    fixture = _legacy_fixture(
        tmp_path,
        composition_version="expected-goals-composition/1",
        training_label_extra_key=True,
    )

    with pytest.raises(LegacyPredictionAuditError, match="must contain exactly"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_unknown_model_evaluation_cohort(tmp_path: Path) -> None:
    fixture = _legacy_fixture(
        tmp_path,
        composition_version="expected-goals-composition/1",
        model_evaluation_cohort=("sample:missing",),
    )

    with pytest.raises(LegacyPredictionAuditError, match="cohort contains unknown samples"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


@pytest.mark.parametrize(
    ("model_feature_version", "model_label_version", "message"),
    (
        ("score-features/2", "result-90/1", "feature_version"),
        ("score-features/1", "result-90/2", "label_version"),
    ),
)
def test_v3_loader_rejects_model_dataset_version_mismatch(
    tmp_path: Path,
    model_feature_version: str,
    model_label_version: str,
    message: str,
) -> None:
    fixture = _legacy_fixture(
        tmp_path,
        composition_version="expected-goals-composition/1",
        model_feature_version=model_feature_version,
        model_label_version=model_label_version,
    )

    with pytest.raises(LegacyPredictionAuditError, match=message):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_calibration_or_model_lineage_tampering(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    payload = _read_json(fixture.prediction_path)
    payload["calibration_versions"] = []
    _write_json(fixture.prediction_path, payload)

    with pytest.raises(LegacyPredictionAuditError, match="calibration_versions"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)

    fixture = _legacy_fixture(tmp_path / "model", composition_version="legacy-inline/1")
    fixture.model_artifact_path.write_bytes(b"tampered model")
    with pytest.raises(LegacyPredictionAuditError, match="model content hash"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)

    fixture = _legacy_fixture(tmp_path / "manifest", composition_version="legacy-inline/1")
    manifest = _read_json(fixture.model_manifest_path)
    manifest["status"] = "failed"
    _write_json(fixture.model_manifest_path, manifest)
    with pytest.raises(LegacyPredictionAuditError, match="manifest"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


@pytest.mark.parametrize(
    ("target", "version", "message"),
    (
        ("prediction", 4, "unsupported legacy prediction schema_version 4"),
        ("composition", 2, "unsupported legacy composition schema_version 2"),
    ),
)
def test_v3_loader_rejects_unknown_schema(
    tmp_path: Path, target: str, version: int, message: str
) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    path = fixture.prediction_path if target == "prediction" else fixture.composition_path
    payload = _read_json(path)
    payload["schema_version"] = version
    _write_json(path, payload)

    with pytest.raises(LegacyPredictionAuditError, match=message):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def test_v3_loader_rejects_noncanonical_shape_and_reference(tmp_path: Path) -> None:
    fixture = _legacy_fixture(tmp_path, composition_version="expected-goals-composition/1")
    payload = _read_json(fixture.prediction_path)
    payload["unexpected"] = True
    _write_json(fixture.prediction_path, payload)

    with pytest.raises(LegacyPredictionAuditError, match="shape mismatch"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)

    fixture = _legacy_fixture(tmp_path / "typed-ref", composition_version="legacy-inline/1")
    payload = _read_json(fixture.prediction_path)
    payload["input_refs"][-1] = "untyped"
    _write_json(fixture.prediction_path, payload)
    with pytest.raises(LegacyPredictionAuditError, match="typed reference"):
        load_legacy_prediction_v3_for_audit(fixture.layout, fixture.reference)


def _legacy_fixture(
    tmp_path: Path,
    *,
    composition_version: str,
    model_evaluation_cohort: tuple[str, ...] | None = None,
    model_feature_version: str = "score-features/1",
    model_label_version: str = "result-90/1",
    training_label_extra_key: bool = False,
) -> LegacyFixture:
    layout = DataLayout(tmp_path / "legacy-data")
    raw = RawArchive(layout)
    derived = DerivedArchive(layout)
    asset = raw.archive(
        b"legacy prediction evidence",
        source="test-source",
        source_id="legacy-prediction-evidence",
        url="fixture://legacy-prediction-evidence",
        observed_at=GENERATED_AT,
        target_event_time=SNAPSHOT_AS_OF,
        collector_version="legacy-fixture/1",
        media_type="application/octet-stream",
    )
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                match_id="legacy-baseline-match",
                home_team_id=HOME_TEAM_ID.value,
                away_team_id=AWAY_TEAM_ID.value,
                kickoff_at=SNAPSHOT_AS_OF - timedelta(days=30),
                known_at=SNAPSHOT_AS_OF - timedelta(days=1),
                home_xg=1.7,
                away_xg=0.8,
                source_ref=asset.id.value,
            ),
        ),
        as_of=SNAPSHOT_AS_OF,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    baseline_home, baseline_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=HOME_TEAM_ID.value,
        away_team_id=AWAY_TEAM_ID.value,
    )
    baseline_value = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": baseline_home,
        "lambda_away": baseline_away,
    }
    baseline_source = seed_legacy_snapshot_source(
        derived,
        value=baseline_value,
        input_refs=(asset.id,),
        transform_version="team-baseline-input/2",
        generated_at=GENERATED_AT,
    )
    context_value = {"days_since_previous_match": 7.0}
    context_source = seed_legacy_snapshot_source(
        derived,
        value=context_value,
        input_refs=(asset.id,),
        transform_version="match-context-input/1",
        generated_at=GENERATED_AT,
    )
    snapshot = build_snapshot(
        match_id=MATCH_ID,
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        as_of=SNAPSHOT_AS_OF,
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="prematch-features/1",
        features=(
            SnapshotFeature(
                name="team_baseline",
                value=baseline_value,
                known_at=SNAPSHOT_AS_OF - timedelta(days=1),
                source_ref=baseline_source,
                contribution_key="team-baseline",
            ),
            SnapshotFeature(
                name="match_context",
                value=context_value,
                known_at=SNAPSHOT_AS_OF - timedelta(days=1),
                source_ref=context_source,
                contribution_key="context:rest-days",
            ),
        ),
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
        source_validator=derived,
    )
    snapshot_document = snapshot_payload(snapshot, source_validator=derived)
    _write_json(derived.snapshot_path(snapshot), snapshot_document)
    snapshot_manifest = DerivedArtifactManifest.create(
        artifact_type="prematch-snapshot",
        schema_version=snapshot.schema_version,
        payload=snapshot_document,
        generated_at=snapshot.observed_at,
        started_at=snapshot.observed_at,
        ended_at=snapshot.observed_at,
        transform_version=snapshot.feature_spec_version,
        code_version="football-data-platform/0.1.0",
        input_refs=snapshot.input_refs,
        output_refs=(snapshot.id.value,),
        status="succeeded",
        quality=snapshot.quality_status,
    )
    derived.write_artifact_manifest(snapshot_manifest)
    snapshot_manifest_path = derived.artifact_manifest_path(snapshot_manifest.id)
    (
        model_run,
        model_artifact_path,
        model_manifest_path,
        training_raw_object_path,
    ) = _write_model_run(
        layout,
        evaluation_cohort=model_evaluation_cohort,
        feature_version=model_feature_version,
        label_version=model_label_version,
        training_label_extra_key=training_label_extra_key,
    )

    if composition_version == "legacy-inline/1":
        baseline_home, baseline_away = 1.45, 0.95
        lambda_home, lambda_away = baseline_home, baseline_away
        contributions: list[dict[str, Any]] = []
        contribution_keys: list[str] = []
        calibration_versions: list[str] = []
        contribution_refs: set[str] = set()
    else:
        multiplier = math.exp(0.02 * (context_value["days_since_previous_match"] - 5.0))
        contributions = [
            {
                "contribution_key": "context:rest-days:match-context-calibration/1",
                "lambda_home_multiplier": multiplier,
                "lambda_away_multiplier": multiplier,
                "source_ref": context_source,
                "version": "match-context-calibration/1",
            }
        ]
        contribution_keys = [contributions[0]["contribution_key"]]
        calibration_versions = ["match-context-calibration/1"]
        contribution_refs = {context_source}
        lambda_home = baseline_home * multiplier
        lambda_away = baseline_away * multiplier

    input_refs = sorted(
        {
            snapshot.id.value,
            model_run.model_run_id,
            baseline_artifact.artifact_id,
            *contribution_refs,
        }
    )
    grid = DixonColesGrid(lambda_home, lambda_away, rho=-0.1, max_goals=11)
    score_cells = list(grid.score_cells())
    grid_payload = {
        "lambda_home": grid.lambda_home,
        "lambda_away": grid.lambda_away,
        "rho": grid.rho,
        "max_goals": grid.max_goals,
        "normalization_residual": grid.normalization_residual,
        "score_cells": score_cells,
    }
    composition_payload = {
        "schema_version": 1,
        "artifact_type": "score-grid-composition",
        "match_id": MATCH_ID.value,
        "snapshot_id": snapshot.id.value,
        "model_run_id": model_run.model_run_id,
        "model_version": MODEL_VERSION,
        "generated_at": _timestamp(GENERATED_AT),
        "snapshot_as_of": _timestamp(SNAPSHOT_AS_OF),
        "baseline_lambda_home": float(baseline_home),
        "baseline_lambda_away": float(baseline_away),
        "contribution_keys": contribution_keys,
        "contribution_multipliers": contributions,
        "composition_version": composition_version,
        "calibration_versions": calibration_versions,
        **grid_payload,
        "input_refs": input_refs,
        "grid": grid_payload,
    }
    composition_ref = "score-grid-composition:" + _digest(composition_payload)
    prediction_identity = {
        "schema_version": 3,
        "match_id": MATCH_ID.value,
        "snapshot_id": snapshot.id.value,
        "capture_mode": snapshot.capture_mode.value,
        "snapshot_quality_status": snapshot.quality_status,
        "model_run_id": model_run.model_run_id,
        "model_version": MODEL_VERSION,
        "generated_at": _timestamp(GENERATED_AT),
        "snapshot_as_of": _timestamp(SNAPSHOT_AS_OF),
        "lambda_home": grid.lambda_home,
        "lambda_away": grid.lambda_away,
        "rho": grid.rho,
        "max_goals": grid.max_goals,
        "score_cells": score_cells,
        "markets": _market_payloads(grid),
        "normalization_residual": grid.normalization_residual,
        "input_refs": input_refs,
        "baseline_lambda_home": float(baseline_home),
        "baseline_lambda_away": float(baseline_away),
        "contribution_keys": contribution_keys,
        "contribution_multipliers": contributions,
        "composition_version": composition_version,
        "calibration_versions": calibration_versions,
        "composition_artifact_ref": composition_ref,
    }
    prediction_ref = "prediction:" + _digest(prediction_identity)
    prediction_payload = {"id": prediction_ref, **prediction_identity}

    prediction_path = _content_path(layout.derived, "predictions", prediction_ref)
    composition_path = _content_path(layout.derived, "compositions", composition_ref)
    _write_json(prediction_path, prediction_payload)
    _write_json(composition_path, {"id": composition_ref, **composition_payload})

    composition_manifest = DerivedArtifactManifest.create(
        artifact_type="score-grid-composition",
        schema_version=1,
        payload=composition_payload,
        generated_at=GENERATED_AT,
        transform_version=composition_version,
        code_version="football-data-platform/0.1.0",
        input_refs=input_refs,
        output_refs=(composition_ref,),
        status="succeeded",
        quality="ready",
    )
    prediction_manifest = DerivedArtifactManifest.create(
        artifact_type="prediction",
        schema_version=3,
        payload=prediction_payload,
        generated_at=GENERATED_AT,
        transform_version=MODEL_VERSION,
        code_version="football-data-platform/0.1.0",
        input_refs=(*input_refs, composition_ref),
        output_refs=(prediction_ref,),
        status="succeeded",
        quality="ready",
    )
    composition_manifest_path = derived.artifact_manifest_path(composition_manifest.id)
    _write_json(composition_manifest_path, composition_manifest.to_payload())
    _write_json(
        derived.artifact_manifest_path(prediction_manifest.id), prediction_manifest.to_payload()
    )
    return LegacyFixture(
        layout=layout,
        reference=prediction_ref,
        prediction_path=prediction_path,
        composition_path=composition_path,
        composition_manifest_path=composition_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
        training_raw_object_path=training_raw_object_path,
        model_artifact_path=model_artifact_path,
        model_manifest_path=model_manifest_path,
        snapshot=snapshot,
    )


def _write_model_run(
    layout: DataLayout,
    *,
    evaluation_cohort: tuple[str, ...] | None,
    feature_version: str,
    label_version: str,
    training_label_extra_key: bool,
) -> tuple[ModelRunArtifact, Path, Path, Path]:
    feature_ref, label_ref = seed_training_references(layout)
    train = TrainingSample(
        sample_id="sample:legacy-train",
        as_of=REFERENCE_TIME - timedelta(days=2),
        feature_known_at=REFERENCE_TIME - timedelta(days=3),
        label_known_at=REFERENCE_TIME,
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(feature_ref,),
        label_ref=label_ref,
        features={"strength": 1.0},
        label={"home_goals": 2, "away_goals": 1},
        split="train",
    )
    holdout = TrainingSample(
        sample_id="sample:legacy-holdout",
        as_of=REFERENCE_TIME - timedelta(days=1),
        feature_known_at=REFERENCE_TIME - timedelta(days=2),
        label_known_at=REFERENCE_TIME,
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(feature_ref,),
        label_ref=label_ref,
        features={"strength": 1.0},
        label={"home_goals": 2, "away_goals": 1},
        split="test",
    )
    dataset = TrainingDatasetManifest.create(
        dataset_version="legacy-prediction-dataset/1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=REFERENCE_TIME,
        split_strategy="forward-chaining/1",
        samples=(train, holdout),
        generated_at=REFERENCE_TIME + timedelta(hours=1),
        transform_version="training-dataset/1",
        code_version="git:legacy-fixture",
    )
    store = TrainingArtifactStore(layout)
    store.write_dataset(dataset)
    dataset_id = dataset.dataset_id
    if training_label_extra_key:
        dataset_payload = dataset.to_payload()
        sample_payload = dataset_payload["samples"][0]
        sample_payload["label"]["extra"] = "forged"
        sample_payload["label_hash"] = _digest(sample_payload["label"])
        dataset_identity = dict(dataset_payload)
        dataset_identity.pop("id")
        dataset_id = "training-dataset:" + _digest(dataset_identity)
        dataset_payload["id"] = dataset_id
        _write_json(store.dataset_path(dataset_id), dataset_payload)
        forged_dataset_manifest = DerivedArtifactManifest.create(
            artifact_type="training-dataset",
            schema_version=dataset.schema_version,
            payload=dataset_payload,
            generated_at=dataset.generated_at,
            transform_version=dataset.transform_version,
            code_version=dataset.code_version,
            input_refs=dataset.input_refs,
            output_refs=(dataset_id,),
            status="succeeded",
            quality=dataset.status.value,
        )
        _write_json(
            store.derived.artifact_manifest_path(forged_dataset_manifest.id),
            forged_dataset_manifest.to_payload(),
        )
    model_ref = store.write_model_artifact(b"legacy-model")
    output_ref = store.write_model_output(b"legacy-output")
    model_run = ModelRunArtifact.create(
        model_version=MODEL_VERSION,
        run_role="challenger",
        task="score-model",
        dataset_id=dataset_id,
        feature_version=feature_version,
        label_version=label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:legacy-fixture",
        environment_version="python-3.11:legacy-fixture",
        started_at=GENERATED_AT - timedelta(seconds=2),
        ended_at=GENERATED_AT - timedelta(seconds=1),
        random_seed=17,
        model_artifact_refs=(model_ref,),
        evaluation_cohort=(holdout.sample_id,) if evaluation_cohort is None else evaluation_cohort,
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_ref,),
        output_hashes=(output_ref.removeprefix(MODEL_OUTPUT_REF_PREFIX),),
    )
    if (
        model_run.feature_version == dataset.feature_version
        and model_run.label_version == dataset.label_version
        and model_run.evaluation_cohort == (holdout.sample_id,)
        and model_run.dataset_id == dataset.dataset_id
    ):
        store.write_model_run_for_audit(model_run)
        manifest = store.derived._load_artifact_manifest_for_output_ref(model_run.model_run_id)
    else:
        _write_json(store.model_run_path(model_run.model_run_id), model_run.to_payload())
        invalid_manifest = DerivedArtifactManifest.create(
            artifact_type="model-run",
            schema_version=model_run.schema_version,
            payload=model_run.to_payload(),
            generated_at=model_run.ended_at,
            transform_version=model_run.model_version,
            code_version=model_run.code_version,
            input_refs=(
                model_run.dataset_id,
                *model_run.model_artifact_refs,
                *model_run.evaluation_cohort,
            ),
            output_refs=(model_run.model_run_id, *model_run.output_refs),
            status="succeeded",
            quality=model_run.status.value,
        )
        _write_json(
            store.derived.artifact_manifest_path(invalid_manifest.id),
            invalid_manifest.to_payload(),
        )
        manifest = invalid_manifest
    source = store.derived.validate_snapshot_source(feature_ref)
    raw_reference = next(
        reference for reference in source.input_refs if reference.startswith("raw-asset:")
    )
    raw_asset = RawArchive(layout).load(RawAssetId(raw_reference))
    return (
        model_run,
        store.model_artifact_path(model_ref),
        store.derived.artifact_manifest_path(manifest.id),
        layout.raw_object_path(raw_asset.checksum),
    )


def _market_payloads(grid: DixonColesGrid) -> list[dict[str, Any]]:
    return [
        _market("result_90", None, grid.result_probabilities()),
        _market("home_handicap_3way", "-1", grid.handicap_probabilities(-1)),
        _market("total_goals", "0-6,7+", grid.total_goals_probabilities()),
    ]


def _market(
    market_type: str, parameter: str | None, probabilities: dict[str, float]
) -> dict[str, Any]:
    return {
        "market_type": market_type,
        "parameter": parameter,
        "outcomes": [
            {"outcome": outcome, "probability": probability}
            for outcome, probability in probabilities.items()
        ],
    }


def _content_path(root: Path, directory: str, reference: str) -> Path:
    digest = reference.split(":", 1)[1]
    return root / directory / digest[:2] / f"{digest}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
