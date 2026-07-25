from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _formal_context import seed_formal_context

from football_data_platform.domain.ids import MatchId, ModelRunId
from football_data_platform.domain.lifecycle import Qualification
from football_data_platform.domain.predictions import (
    _build_score_prediction,
    build_score_prediction,
    prediction_payload,
    score_grid_composition_payload,
)
from football_data_platform.domain.snapshots import (
    CURRENT_FEATURE_SPEC_VERSION,
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    MODEL_OUTPUT_REF_PREFIX,
    ModelRunArtifact,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contributions,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
)
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)

OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


class _StaticModelRunValidator:
    def __init__(self, artifact: ModelRunArtifact) -> None:
        self.artifact = artifact

    def load_model_run(self, model_run_id: str) -> ModelRunArtifact:
        assert model_run_id == self.artifact.model_run_id
        return self.artifact


def _seed_formal_training(layout: DataLayout) -> tuple[MatchId, MatchId, str, str, str]:
    kickoff = datetime(2026, 5, 20, 15, 0, tzinfo=UTC)
    as_of = kickoff - timedelta(hours=24)
    fixture = seed_formal_context(
        layout,
        as_of=as_of,
        kickoff=kickoff,
        key="training-qualification",
        observed_at=OBSERVED_AT,
    )
    target_result = CanonicalFactStore(fixture.canonical).append_result_90(
        match_id=fixture.match_id,
        match_version=fixture.match_version,
        home_goals=1,
        away_goals=1,
        known_at=kickoff + timedelta(hours=2),
        observed_at=OBSERVED_AT,
        raw_asset_id=fixture.schedule_raw_id,
    )
    with fixture.canonical.connect() as connection:
        rows = connection.execute(
            "SELECT result.record_id, result.match_id, result.known_at, "
            "matches.home_team_id, matches.away_team_id, version.kickoff_at "
            "FROM match_results_90 AS result JOIN matches "
            "ON matches.match_id = result.match_id JOIN match_versions AS version "
            "ON version.match_id = result.match_id AND version.version = result.match_version "
            "WHERE result.match_id != ? ORDER BY version.kickoff_at",
            (fixture.match_id.value,),
        ).fetchall()
    observations = tuple(
        TeamMatchProcess(
            match_id=row["match_id"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_at=datetime.fromisoformat(row["kickoff_at"].replace("Z", "+00:00")),
            known_at=datetime.fromisoformat(row["known_at"].replace("Z", "+00:00")),
            home_xg=1.5 + index * 0.1,
            away_xg=0.9 + index * 0.1,
            source_ref=row["record_id"],
        )
        for index, row in enumerate(rows)
    )
    baseline = build_team_baseline(
        observations,
        as_of=as_of,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    fixture.derived.write_team_baseline(baseline, generated_at=OBSERVED_AT)
    baseline_ref = fixture.derived.write_team_baseline_source(
        match_id=fixture.match_id,
        match_version=fixture.match_version,
        as_of=as_of,
        baseline_artifact_id=baseline.artifact_id,
    )
    baseline_validation = fixture.derived.validate_snapshot_source(baseline_ref)
    snapshot = build_snapshot(
        match_id=fixture.match_id,
        match_version=fixture.match_version,
        snapshot_type=SnapshotType.T24H,
        as_of=as_of,
        scheduled_kickoff_used=kickoff,
        feature_spec_version=CURRENT_FEATURE_SPEC_VERSION,
        features=(
            SnapshotFeature(
                name="team_baseline",
                value=baseline_validation.value,
                known_at=baseline_validation.known_at,
                source_ref=baseline_ref,
                contribution_key="team-baseline",
            ),
            SnapshotFeature(
                name="match_context",
                value=fixture.context_value,
                known_at=fixture.context_known_at,
                source_ref=fixture.context_ref,
                contribution_key="match-context",
            ),
        ),
        home_team_id=fixture.home_team_id,
        away_team_id=fixture.away_team_id,
        source_validator=fixture.derived,
    )
    fixture.derived.write_snapshot(snapshot)
    first = rows[0]
    return (
        MatchId(first["match_id"]),
        fixture.match_id,
        first["record_id"],
        target_result.record_id,
        snapshot.id.value,
    )


def _formal_dataset(
    store: TrainingArtifactStore, sample: TrainingSample
) -> TrainingDatasetManifest:
    return TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/qualification-contract-1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        as_of=sample.as_of,
        split_strategy="forward-chaining/1",
        samples=(sample,),
        generated_at=OBSERVED_AT,
        code_version="git:test",
    )


def test_formal_training_qualification_replays_and_closes_forgery_paths(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data").ensure()
    first_match, second_match, first_result, second_result, snapshot_ref = _seed_formal_training(
        layout
    )
    store = TrainingArtifactStore(layout)

    qualification = store.create_training_qualification(
        match_id=second_match,
        match_version=1,
        qualification=Qualification.SCORE_MODEL,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=snapshot_ref,
        result_ref=second_result,
    )
    assert qualification.passed
    assert store.load_training_qualification(qualification.qualification_id) == qualification

    sample = store.build_formal_score_sample(
        sample_id=f"sample:{second_match.value}:score-model",
        qualification_ref=qualification.qualification_id,
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        split="holdout",
    )
    assert sample.feature_version == CURRENT_SCORE_FEATURE_PROJECTION_VERSION
    assert sample.features["projection_version"] == CURRENT_SCORE_FEATURE_PROJECTION_VERSION
    with pytest.raises(TrainingArtifactConflict, match="current feature projection"):
        store.build_formal_score_sample(
            sample_id="sample:forged-feature-version",
            qualification_ref=qualification.qualification_id,
            feature_version="attacker-score-features/1",
            split="holdout",
        )
    forged_version_sample = replace(
        sample,
        feature_version="attacker-score-features/1",
    )
    forged_version_dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/forged-projection-1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=forged_version_sample.feature_version,
        label_version="result-90/1",
        as_of=forged_version_sample.as_of,
        split_strategy="forward-chaining/1",
        samples=(forged_version_sample,),
        generated_at=OBSERVED_AT,
        code_version="git:test",
    )
    with pytest.raises(TrainingArtifactConflict, match="current score feature projection"):
        store.write_formal_dataset(forged_version_dataset)
    dataset = _formal_dataset(store, sample)
    store.write_formal_dataset(dataset)
    assert store.load_formal_dataset(dataset.dataset_id) == dataset

    forged_features = replace(sample, features={"lambda_home": 99.0, "lambda_away": 0.01})
    with pytest.raises(TrainingArtifactConflict, match="replayed evidence"):
        store.write_formal_dataset(_formal_dataset(store, forged_features))

    forged_match = replace(sample, match_id=first_match.value)
    with pytest.raises(TrainingArtifactConflict, match="replayed evidence"):
        store.write_formal_dataset(_formal_dataset(store, forged_match))

    with pytest.raises(TrainingArtifactConflict, match="different match or version"):
        store.create_training_qualification(
            match_id=second_match,
            match_version=1,
            qualification=Qualification.SCORE_MODEL,
            ruleset_version="readiness/1",
            evaluated_at=OBSERVED_AT,
            snapshot_ref=snapshot_ref,
            result_ref=first_result,
        )

    missing_snapshot = store.create_training_qualification(
        match_id=first_match,
        match_version=1,
        qualification=Qualification.SCORE_MODEL,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=None,
        result_ref=first_result,
    )
    assert not missing_snapshot.passed
    assert "missing_bound_prematch_snapshot" in missing_snapshot.reason_codes
    excluded = store.build_formal_score_sample(
        sample_id=f"sample:{first_match.value}:score-model",
        qualification_ref=missing_snapshot.qualification_id,
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        split="train",
        as_of=datetime(2025, 8, 14, 20, 0, tzinfo=UTC),
    )
    forged_pass = replace(excluded, qualification_passed=True, exclusion_reasons=())
    with pytest.raises(TrainingArtifactConflict, match="replayed evidence"):
        store.write_formal_dataset(_formal_dataset(store, forged_pass))

    legacy = TrainingDatasetManifest.create(
        dataset_version="legacy/1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=sample.feature_version,
        label_version=sample.label_version,
        as_of=sample.as_of,
        split_strategy="forward-chaining/1",
        samples=(
            replace(
                sample,
                match_id=None,
                match_version=None,
                snapshot_ref=None,
                qualification_ref=None,
                feature_refs=(snapshot_ref,),
            ),
        ),
        generated_at=OBSERVED_AT,
    )
    with pytest.raises(TrainingArtifactConflict, match="audit-only"):
        store.write_formal_dataset(legacy)

    legacy_train = replace(
        sample,
        sample_id="sample:legacy-self-reported-train",
        split="train",
        match_id=None,
        match_version=None,
        snapshot_ref=None,
        qualification_ref=None,
        feature_refs=(snapshot_ref,),
    )
    legacy_holdout = replace(
        legacy_train,
        sample_id="sample:legacy-self-reported-holdout",
        as_of=legacy_train.as_of + timedelta(minutes=1),
        split="holdout",
    )
    legacy_dataset = TrainingDatasetManifest.create(
        dataset_version="legacy-self-reported/1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=legacy_train.feature_version,
        label_version=legacy_train.label_version,
        as_of=legacy_holdout.as_of,
        split_strategy="forward-chaining/1",
        samples=(legacy_train, legacy_holdout),
        generated_at=OBSERVED_AT,
        code_version="git:test",
    )
    store.write_dataset(legacy_dataset)
    model_ref = store.write_model_artifact(b"legacy-self-reported-model")
    output_ref = store.write_model_output(b"legacy-self-reported-output")
    legacy_run = ModelRunArtifact.create(
        model_version="dixon-coles/legacy-self-reported-1",
        run_role="challenger",
        task="score-model",
        dataset_id=legacy_dataset.dataset_id,
        feature_version=legacy_dataset.feature_version,
        label_version=legacy_dataset.label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:test",
        environment_version="python:test",
        started_at=OBSERVED_AT,
        ended_at=OBSERVED_AT,
        random_seed=17,
        model_artifact_refs=(model_ref,),
        evaluation_cohort=(legacy_holdout.sample_id,),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_ref,),
        output_hashes=(output_ref.removeprefix(MODEL_OUTPUT_REF_PREFIX),),
    )
    with pytest.raises(TrainingArtifactConflict, match="audit-only"):
        store.write_model_run(legacy_run)
    store.write_model_run_for_audit(legacy_run)

    snapshot = store._load_qualification_snapshot(snapshot_ref)
    baseline = next(feature for feature in snapshot.features if feature.name == "team_baseline")
    context = next(feature for feature in snapshot.features if feature.name == "match_context")
    expected_goals = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        context_contributions(
            context.value,
            home_team_id=snapshot.home_team_id.value,
            away_team_id=snapshot.away_team_id.value,
            source_ref=context.source_ref,
            source_validator=store.derived,
        ),
    )
    pre_kickoff_run = ModelRunArtifact.create(
        model_version="dixon-coles/wrong-projection-1",
        run_role="challenger",
        task="score-model",
        dataset_id="training-dataset:" + "a" * 64,
        feature_version="attacker-score-features/1",
        label_version="result-90/1",
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:test",
        environment_version="python:test",
        started_at=snapshot.as_of + timedelta(hours=1),
        ended_at=snapshot.as_of + timedelta(hours=2),
        random_seed=17,
        model_artifact_refs=("model-artifact:" + "b" * 64,),
        evaluation_cohort=("sample:projection-holdout",),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=("model-output:" + "c" * 64,),
        output_hashes=("c" * 64,),
    )
    with pytest.raises(ValueError, match="current score feature projection"):
        build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=store.derived,
            model_run_id=ModelRunId(pre_kickoff_run.model_run_id),
            model_version=pre_kickoff_run.model_version,
            generated_at=snapshot.as_of + timedelta(hours=3),
            lambda_home=expected_goals.lambda_home,
            lambda_away=expected_goals.lambda_away,
            rho=-0.1,
            max_goals=11,
            input_refs=(baseline.value["artifact_id"], *expected_goals.input_refs),
            model_run_validator=_StaticModelRunValidator(pre_kickoff_run),
            expected_goals=expected_goals,
            calibration_versions=expected_goals.calibration_versions,
        )

    prediction = _build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=store.derived,
        model_run_id=ModelRunId(legacy_run.model_run_id),
        model_version=legacy_run.model_version,
        generated_at=OBSERVED_AT,
        lambda_home=expected_goals.lambda_home,
        lambda_away=expected_goals.lambda_away,
        rho=-0.1,
        max_goals=11,
        input_refs=(baseline.value["artifact_id"], *expected_goals.input_refs),
        model_run_validator=None,
        expected_goals=expected_goals,
        composition=None,
        composition_artifact_ref=None,
        calibration_versions=expected_goals.calibration_versions,
        enforce_current=False,
    )
    formal_archive = DerivedArchive(layout, model_run_validator=store)
    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        formal_archive.write_prediction(prediction, snapshot=snapshot)
    composition_document = score_grid_composition_payload(prediction)
    store.derived._write_json(
        store.derived.score_grid_composition_path(prediction.composition_artifact_ref),
        {"id": prediction.composition_artifact_ref, **composition_document},
    )
    composition_manifest = DerivedArtifactManifest.create(
        artifact_type="score-grid-composition",
        schema_version=composition_document["schema_version"],
        payload=composition_document,
        generated_at=prediction.generated_at,
        transform_version=prediction.composition_version,
        code_version="git:test",
        input_refs=prediction.input_refs,
        output_refs=(prediction.composition_artifact_ref,),
        quality=prediction.snapshot_quality_status,
    )
    store.derived._write_json(
        store.derived.artifact_manifest_path(composition_manifest.artifact_id),
        composition_manifest.to_payload(),
    )
    prediction_document = prediction_payload(prediction)
    prediction_digest = prediction.id.value.removeprefix("prediction:")
    store.derived._write_json(
        layout.derived / "predictions" / prediction_digest[:2] / f"{prediction_digest}.json",
        prediction_document,
    )
    prediction_manifest = DerivedArtifactManifest.create(
        artifact_type="prediction",
        schema_version=prediction.schema_version,
        payload=prediction_document,
        generated_at=prediction.generated_at,
        transform_version=prediction.model_version,
        code_version="git:test",
        input_refs=(*prediction.input_refs, prediction.composition_artifact_ref),
        output_refs=(prediction.id.value,),
        quality=prediction.snapshot_quality_status,
    )
    store.derived._write_json(
        store.derived.artifact_manifest_path(prediction_manifest.artifact_id),
        prediction_manifest.to_payload(),
    )
    assert store.load_prediction_for_audit(prediction.id.value) == prediction
    with pytest.raises(TrainingArtifactConflict, match="audit-only"):
        store.load_verified_prediction(prediction.id.value)
    with pytest.raises(TrainingArtifactConflict, match="audit-only"):
        store.load_verified_prediction_context(prediction.id.value)

    model_manifest = store.derived._load_artifact_manifest_for_output_ref(legacy_run.model_run_id)
    store.derived.artifact_manifest_path(model_manifest.artifact_id).unlink()
    with pytest.raises(TrainingArtifactConflict, match="manifest|unavailable|invalid"):
        store.load_prediction_for_audit(prediction.id.value)


def test_non_score_qualifications_and_captured_snapshots_fail_closed(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data").ensure()
    _, second_match, _, second_result, snapshot_ref = _seed_formal_training(layout)
    store = TrainingArtifactStore(layout)

    team_baseline = store.create_training_qualification(
        match_id=second_match,
        match_version=1,
        qualification=Qualification.TEAM_BASELINE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=snapshot_ref,
        result_ref=second_result,
    )
    assert not team_baseline.passed
    assert any(reason.startswith("missing_team_stat:") for reason in team_baseline.reason_codes)
    assert "typed_team_fact_replay_unavailable" not in team_baseline.reason_codes

    player_profile = store.create_training_qualification(
        match_id=second_match,
        match_version=1,
        qualification=Qualification.PLAYER_PROFILE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=snapshot_ref,
        result_ref=second_result,
    )
    assert not player_profile.passed
    assert "typed_player_fact_replay_unavailable" not in player_profile.reason_codes
    assert "missing_typed_player_fact_batch" in player_profile.reason_codes

    derived = DerivedArchive(layout)
    snapshot_path = next(
        path
        for path in (layout.derived / "snapshots").rglob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["id"] == snapshot_ref
    )
    captured_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    captured_payload["capture_mode"] = CaptureMode.CAPTURED.value
    identity = dict(captured_payload)
    identity.pop("id")
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    captured_ref = f"snapshot:{digest}"
    captured_payload["id"] = captured_ref
    captured_path = layout.derived / "snapshots" / digest[:2] / f"{digest}.json"
    derived._write_json(captured_path, captured_payload)
    original_manifest = derived._load_artifact_manifest_for_output_ref(snapshot_ref)
    derived.write_artifact_manifest(
        DerivedArtifactManifest.create(
            artifact_type="prematch-snapshot",
            schema_version=original_manifest.schema_version,
            payload=captured_payload,
            generated_at=original_manifest.generated_at,
            transform_version=original_manifest.transform_version,
            code_version=original_manifest.code_version,
            input_refs=original_manifest.input_refs,
            output_refs=(captured_ref,),
            quality=original_manifest.quality,
        )
    )
    with pytest.raises(TrainingArtifactConflict, match="trusted capture-run"):
        store.create_training_qualification(
            match_id=second_match,
            match_version=1,
            qualification=Qualification.SCORE_MODEL,
            ruleset_version="readiness/1",
            evaluated_at=OBSERVED_AT,
            snapshot_ref=captured_ref,
            result_ref=second_result,
        )
