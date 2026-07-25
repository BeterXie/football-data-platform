"""Read-only audit support for immutable prediction schema-v3 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import (
    MatchId,
    ModelRunId,
    PredictionId,
    RawAssetId,
    SnapshotId,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    PreMatchSnapshot,
    parse_snapshot_payload,
)
from football_data_platform.domain.training import (
    DatasetStatus,
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
)
from football_data_platform.models.score_grid import DixonColesGrid
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DerivedArchive,
    DerivedArtifactManifest,
    _parse_artifact_manifest,
)
from football_data_platform.storage.facts import load_verified_match_result
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactStore,
    parse_model_run_payload,
    parse_training_dataset_payload,
)

PREDICTION_V3_SCHEMA_VERSION = 3
COMPOSITION_V1_SCHEMA_VERSION = 1
LEGACY_INLINE_COMPOSITION_VERSION = "legacy-inline/1"
EXPECTED_GOALS_V1_COMPOSITION_VERSION = "expected-goals-composition/1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PREDICTION_KEYS = frozenset(
    """id schema_version match_id snapshot_id capture_mode snapshot_quality_status
    model_run_id model_version generated_at snapshot_as_of lambda_home lambda_away rho max_goals
    score_cells markets normalization_residual input_refs baseline_lambda_home baseline_lambda_away
    contribution_keys contribution_multipliers composition_version calibration_versions
    composition_artifact_ref""".split()
)
_CONTRIBUTION_KEYS = frozenset(
    "contribution_key lambda_home_multiplier lambda_away_multiplier source_ref version".split()
)
_COMPOSITION_KEYS = frozenset(
    """schema_version artifact_type match_id snapshot_id model_run_id model_version generated_at
    snapshot_as_of baseline_lambda_home baseline_lambda_away contribution_keys
    contribution_multipliers composition_version calibration_versions lambda_home lambda_away rho
    max_goals normalization_residual score_cells input_refs grid""".split()
)
_GRID_FIELDS = (
    "lambda_home",
    "lambda_away",
    "rho",
    "max_goals",
    "normalization_residual",
    "score_cells",
)
_COMPOSITION_CROSS_FIELDS = (
    "match_id",
    "snapshot_id",
    "model_run_id",
    "model_version",
    "generated_at",
    "snapshot_as_of",
    "baseline_lambda_home",
    "baseline_lambda_away",
    "contribution_keys",
    "contribution_multipliers",
    "composition_version",
    "calibration_versions",
    *_GRID_FIELDS,
    "input_refs",
)


class LegacyPredictionAuditError(ValueError):
    """The requested historical prediction cannot be audited safely."""


class LegacyPredictionAuditLimitation(StrEnum):
    PROVENANCE_INCOMPLETE = "provenance_incomplete"
    CALIBRATION_POLICY_UNVERIFIABLE = "calibration_policy_unverifiable"


class LegacyModelLineageLimitation(StrEnum):
    """Dataset bytes are preserved, but current policy cannot recertify historical labels."""

    TRAINING_DATASET_POLICY_UNVERIFIABLE = "training_dataset_policy_unverifiable"


LegacyScoreCellV3 = tuple[int, int, float]
LegacyOutcomeProbabilityV3 = tuple[str, float]
LegacyMarketViewV3 = tuple[str, str | None, tuple[LegacyOutcomeProbabilityV3, ...]]
LegacyPredictionContributionV3 = tuple[str, float, float, str, str]


@dataclass(frozen=True, slots=True)
class LegacyScorePredictionV3:
    """Immutable audit view deliberately incompatible with the current prediction type."""

    id: str
    schema_version: int
    match_id: str
    snapshot_id: str
    capture_mode: CaptureMode
    snapshot_quality_status: str
    model_run_id: str
    model_version: str
    generated_at: datetime
    snapshot_as_of: datetime
    lambda_home: float
    lambda_away: float
    rho: float
    max_goals: int
    score_cells: tuple[LegacyScoreCellV3, ...]
    markets: tuple[LegacyMarketViewV3, ...]
    normalization_residual: float
    input_refs: tuple[str, ...]
    baseline_lambda_home: float
    baseline_lambda_away: float
    contribution_keys: tuple[str, ...]
    contribution_multipliers: tuple[LegacyPredictionContributionV3, ...]
    composition_version: str
    calibration_versions: tuple[str, ...]
    composition_artifact_ref: str
    prediction_manifest_id: str
    composition_manifest_id: str
    audit_limitation: LegacyPredictionAuditLimitation
    model_lineage_limitation: LegacyModelLineageLimitation

    @property
    def audit_only(self) -> bool:
        return True


def load_legacy_prediction_v3_for_audit(
    layout: DataLayout,
    reference: str,
) -> LegacyScorePredictionV3:
    """Load one v3 prediction for audit without rewriting or promoting it."""

    if not isinstance(layout, DataLayout):
        raise TypeError("layout must be a DataLayout")
    if not all(path.is_dir() for path in (layout.raw, layout.canonical, layout.derived)):
        raise LegacyPredictionAuditError("legacy audit requires an existing data layout")

    try:
        archive = DerivedArchive(layout)
        payload = _read_referenced_json(layout.derived, "predictions", reference, "prediction")
        generated_at, snapshot_as_of, limitation = _validate_prediction(payload, reference)

        composition_ref = payload["composition_artifact_ref"]
        stored_composition = _read_referenced_json(
            layout.derived, "compositions", composition_ref, "score-grid-composition"
        )
        if stored_composition.get("schema_version") != COMPOSITION_V1_SCHEMA_VERSION:
            raise LegacyPredictionAuditError(
                "unsupported legacy composition schema_version "
                f"{stored_composition.get('schema_version')!r}"
            )
        composition_payload = archive.load_score_grid_composition_payload(composition_ref)
        _validate_composition(composition_payload, payload)

        prediction_manifest = _load_manifest_for_output_ref(archive, reference)
        _validate_manifest(
            prediction_manifest,
            artifact_type="prediction",
            schema_version=PREDICTION_V3_SCHEMA_VERSION,
            transform_version=payload["model_version"],
            generated_at=generated_at,
            input_refs=tuple(sorted({*payload["input_refs"], composition_ref})),
            output_refs=(reference,),
            quality=payload["snapshot_quality_status"],
            payload=payload,
        )
        composition_manifest = _load_manifest_for_output_ref(archive, composition_ref)
        _validate_manifest(
            composition_manifest,
            artifact_type="score-grid-composition",
            schema_version=COMPOSITION_V1_SCHEMA_VERSION,
            transform_version=payload["composition_version"],
            generated_at=generated_at,
            input_refs=tuple(payload["input_refs"]),
            output_refs=(composition_ref,),
            quality=payload["snapshot_quality_status"],
            payload=composition_payload,
        )

        snapshot_document = _read_referenced_json(
            layout.derived, "snapshots", payload["snapshot_id"], "snapshot"
        )
        snapshot = parse_snapshot_payload(snapshot_document, source_validator=archive)
        _validate_snapshot(payload, snapshot, snapshot_as_of, limitation)
        _validate_snapshot_manifest(archive, snapshot, snapshot_document)

        model_document = _read_referenced_json(
            layout.derived, "model-runs", payload["model_run_id"], "model-run"
        )
        model_run = parse_model_run_payload(model_document)
        if (
            model_run.model_run_id != payload["model_run_id"]
            or model_run.status is not ModelRunStatus.SUCCEEDED
            or model_run.task != "score-model"
            or model_run.model_version != payload["model_version"]
            or model_run.ended_at > generated_at
        ):
            raise LegacyPredictionAuditError("legacy prediction model run does not match")
        model_manifest = _load_manifest_for_output_ref(archive, model_run.model_run_id)
        _validate_manifest(
            model_manifest,
            artifact_type="model-run",
            schema_version=model_run.schema_version,
            transform_version=model_run.model_version,
            generated_at=model_run.ended_at,
            input_refs=tuple(
                sorted(
                    {
                        model_run.dataset_id,
                        *model_run.model_artifact_refs,
                        *model_run.evaluation_cohort,
                    }
                )
            ),
            output_refs=tuple(sorted({model_run.model_run_id, *model_run.output_refs})),
            quality=model_run.status.value,
            payload=model_run.to_payload(),
        )
        model_store = TrainingArtifactStore(layout)
        for model_ref in model_run.model_artifact_refs:
            model_store.verify_model_artifact(model_ref)
        for output_ref, output_hash in zip(
            model_run.output_refs, model_run.output_hashes, strict=True
        ):
            model_store.verify_model_output(output_ref)
            if output_ref.rsplit(":", 1)[-1] != output_hash:
                raise LegacyPredictionAuditError("model output reference hash does not match")
        _validate_training_dataset(archive, model_run)
    except LegacyPredictionAuditError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise LegacyPredictionAuditError(f"legacy prediction v3 audit failed: {error}") from error

    return _build_view(
        payload,
        generated_at=generated_at,
        snapshot_as_of=snapshot_as_of,
        prediction_manifest_id=prediction_manifest.id,
        composition_manifest_id=composition_manifest.id,
        limitation=limitation,
    )


def _validate_prediction(
    payload: dict[str, Any], reference: str
) -> tuple[datetime, datetime, LegacyPredictionAuditLimitation]:
    _require_keys(payload, _PREDICTION_KEYS, "prediction")
    if _strict_int(payload["schema_version"], "schema_version") != PREDICTION_V3_SCHEMA_VERSION:
        raise LegacyPredictionAuditError(
            f"unsupported legacy prediction schema_version {payload['schema_version']!r}"
        )
    if payload["id"] != reference:
        raise LegacyPredictionAuditError("prediction ID does not match requested path")
    PredictionId(reference)
    MatchId(_text(payload["match_id"], "match_id"))
    SnapshotId(_text(payload["snapshot_id"], "snapshot_id"))
    ModelRunId(_text(payload["model_run_id"], "model_run_id"))
    _digest_ref(payload["snapshot_id"], "snapshot")
    _digest_ref(payload["model_run_id"], "model-run")
    _digest_ref(payload["composition_artifact_ref"], "score-grid-composition")

    generated_at = _utc(payload["generated_at"], "generated_at")
    snapshot_as_of = _utc(payload["snapshot_as_of"], "snapshot_as_of")
    if generated_at < snapshot_as_of:
        raise LegacyPredictionAuditError("prediction predates its snapshot")
    try:
        CaptureMode(payload["capture_mode"])
    except (TypeError, ValueError) as error:
        raise LegacyPredictionAuditError("prediction capture_mode is invalid") from error
    if payload["snapshot_quality_status"] != "ready":
        raise LegacyPredictionAuditError("legacy prediction must cite a ready snapshot")
    _text(payload["model_version"], "model_version")

    lambda_home = _float(payload["lambda_home"], "lambda_home", positive=True)
    lambda_away = _float(payload["lambda_away"], "lambda_away", positive=True)
    rho = _float(payload["rho"], "rho")
    max_goals = _strict_int(payload["max_goals"], "max_goals")
    if max_goals < 1:
        raise LegacyPredictionAuditError("max_goals must be positive")
    baseline_home = _float(payload["baseline_lambda_home"], "baseline_lambda_home", positive=True)
    baseline_away = _float(payload["baseline_lambda_away"], "baseline_lambda_away", positive=True)
    _float(payload["normalization_residual"], "normalization_residual")

    input_refs = _text_list(payload["input_refs"], "input_refs")
    if payload["snapshot_id"] not in input_refs or payload["model_run_id"] not in input_refs:
        raise LegacyPredictionAuditError("prediction input_refs omit snapshot or model run")
    if any(":" not in item or any(char.isspace() for char in item) for item in input_refs):
        raise LegacyPredictionAuditError("input_refs must contain typed references")
    contribution_keys = _text_list(payload["contribution_keys"], "contribution_keys")
    calibration_versions = _text_list(payload["calibration_versions"], "calibration_versions")
    contributions = _contributions(payload["contribution_multipliers"])
    if contribution_keys != tuple(item["contribution_key"] for item in contributions):
        raise LegacyPredictionAuditError("contribution keys do not match contribution details")
    if any(item["source_ref"] not in input_refs for item in contributions):
        raise LegacyPredictionAuditError("contribution source is missing from input_refs")

    version = payload["composition_version"]
    if version == LEGACY_INLINE_COMPOSITION_VERSION:
        if (
            contributions
            or contribution_keys
            or calibration_versions
            or baseline_home != lambda_home
            or baseline_away != lambda_away
        ):
            raise LegacyPredictionAuditError("legacy-inline composition is not canonical")
        limitation = LegacyPredictionAuditLimitation.PROVENANCE_INCOMPLETE
    elif version == EXPECTED_GOALS_V1_COMPOSITION_VERSION:
        contribution_versions = tuple(sorted({item["version"] for item in contributions}))
        if calibration_versions != contribution_versions:
            raise LegacyPredictionAuditError(
                "calibration_versions do not match contribution versions"
            )
        composed_home = baseline_home * math.prod(
            item["lambda_home_multiplier"] for item in contributions
        )
        composed_away = baseline_away * math.prod(
            item["lambda_away_multiplier"] for item in contributions
        )
        if not math.isclose(composed_home, lambda_home, rel_tol=1e-12, abs_tol=1e-12):
            raise LegacyPredictionAuditError("home lambda is not reproducible")
        if not math.isclose(composed_away, lambda_away, rel_tol=1e-12, abs_tol=1e-12):
            raise LegacyPredictionAuditError("away lambda is not reproducible")
        limitation = LegacyPredictionAuditLimitation.CALIBRATION_POLICY_UNVERIFIABLE
    else:
        raise LegacyPredictionAuditError(f"unsupported legacy composition_version {version!r}")

    grid = DixonColesGrid(lambda_home, lambda_away, rho=rho, max_goals=max_goals)
    if payload["score_cells"] != list(grid.score_cells()):
        raise LegacyPredictionAuditError("score cells do not match the Dixon-Coles grid")
    if payload["markets"] != _market_payloads(grid):
        raise LegacyPredictionAuditError("market views do not match the Dixon-Coles grid")
    if payload["normalization_residual"] != grid.normalization_residual:
        raise LegacyPredictionAuditError("normalization residual does not match the grid")

    identity = dict(payload)
    identity.pop("id")
    expected_id = "prediction:" + hashlib.sha256(_canonical_json(identity)).hexdigest()
    if reference != expected_id:
        raise LegacyPredictionAuditError("prediction identity hash does not match content")
    return generated_at, snapshot_as_of, limitation


def _validate_composition(composition: dict[str, Any], prediction: dict[str, Any]) -> None:
    _require_keys(composition, _COMPOSITION_KEYS, "score-grid-composition")
    if (
        _strict_int(composition["schema_version"], "composition schema_version")
        != COMPOSITION_V1_SCHEMA_VERSION
    ):
        raise LegacyPredictionAuditError(
            f"unsupported legacy composition schema_version {composition['schema_version']!r}"
        )
    if composition["artifact_type"] != "score-grid-composition":
        raise LegacyPredictionAuditError("composition artifact_type is invalid")
    if any(composition[field] != prediction[field] for field in _COMPOSITION_CROSS_FIELDS):
        raise LegacyPredictionAuditError("composition fields do not match prediction")
    expected_grid = {field: composition[field] for field in _GRID_FIELDS}
    if composition["grid"] != expected_grid:
        raise LegacyPredictionAuditError("nested composition grid does not match top-level fields")


def _validate_manifest(
    manifest: DerivedArtifactManifest,
    *,
    artifact_type: str,
    schema_version: int,
    transform_version: str,
    generated_at: datetime,
    input_refs: tuple[str, ...],
    output_refs: tuple[str, ...],
    quality: str,
    payload: dict[str, Any],
) -> None:
    if (
        manifest.artifact_type != artifact_type
        or manifest.schema_version != schema_version
        or manifest.transform_version != transform_version
        or manifest.generated_at != generated_at
        or manifest.started_at != generated_at
        or manifest.ended_at != generated_at
        or manifest.input_refs != input_refs
        or manifest.output_refs != output_refs
        or manifest.status != "succeeded"
        or manifest.error is not None
        or manifest.quality != quality
        or manifest.payload != payload
    ):
        raise LegacyPredictionAuditError(
            f"legacy {artifact_type} manifest does not match persisted content"
        )


def _load_manifest_for_output_ref(
    archive: DerivedArchive, output_ref: str
) -> DerivedArtifactManifest:
    # Current reference resolution would recertify historical inputs under newer policies.
    matches: list[DerivedArtifactManifest] = []
    root = archive.layout.derived / "manifests" / "artifacts"
    for path in root.rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        if output_ref not in raw:
            continue
        document = json.loads(raw)
        if not isinstance(document, dict) or output_ref not in document.get("output_refs", ()):
            continue
        manifest = _parse_artifact_manifest(document)
        if path != archive.artifact_manifest_path(manifest.id):
            raise LegacyPredictionAuditError("derived manifest path does not match its ID")
        matches.append(manifest)
    if len(matches) != 1:
        raise LegacyPredictionAuditError(
            f"legacy output ref {output_ref} requires exactly one immutable manifest"
        )
    return matches[0]


def _validate_snapshot(
    prediction: dict[str, Any],
    snapshot: PreMatchSnapshot,
    snapshot_as_of: datetime,
    limitation: LegacyPredictionAuditLimitation,
) -> None:
    if (
        snapshot.id.value != prediction["snapshot_id"]
        or snapshot.match_id.value != prediction["match_id"]
        or snapshot.as_of != snapshot_as_of
        or snapshot.capture_mode.value != prediction["capture_mode"]
        or snapshot.quality_status != prediction["snapshot_quality_status"]
    ):
        raise LegacyPredictionAuditError("prediction does not match its snapshot")
    sources = {feature.source_ref for feature in snapshot.features}
    if any(
        contribution["source_ref"] not in sources
        for contribution in prediction["contribution_multipliers"]
    ):
        raise LegacyPredictionAuditError("contribution source is outside the verified snapshot")
    if limitation is LegacyPredictionAuditLimitation.CALIBRATION_POLICY_UNVERIFIABLE:
        baselines = [
            feature.value for feature in snapshot.features if feature.name == "team_baseline"
        ]
        if len(baselines) != 1 or not isinstance(baselines[0], dict):
            raise LegacyPredictionAuditError("composed v1 prediction lacks a snapshot baseline")
        if (
            baselines[0].get("lambda_home") != prediction["baseline_lambda_home"]
            or baselines[0].get("lambda_away") != prediction["baseline_lambda_away"]
        ):
            raise LegacyPredictionAuditError("prediction baseline does not match snapshot")


def _validate_snapshot_manifest(
    archive: DerivedArchive,
    snapshot: PreMatchSnapshot,
    document: dict[str, Any],
) -> None:
    manifest = _load_manifest_for_output_ref(archive, snapshot.id.value)
    _validate_manifest(
        manifest,
        artifact_type="prematch-snapshot",
        schema_version=snapshot.schema_version,
        transform_version=snapshot.feature_spec_version,
        generated_at=snapshot.observed_at,
        input_refs=snapshot.input_refs,
        output_refs=(snapshot.id.value,),
        quality=snapshot.quality_status,
        payload=document,
    )


def _validate_training_dataset(
    archive: DerivedArchive,
    model_run: ModelRunArtifact,
) -> None:
    reference = model_run.dataset_id
    document = _read_referenced_json(
        archive.layout.derived, "training-datasets", reference, "training-dataset"
    )
    dataset = parse_training_dataset_payload(document)
    if dataset.dataset_id != reference:
        raise LegacyPredictionAuditError("training dataset ID does not match model run")
    if dataset.status is not DatasetStatus.SUCCEEDED or dataset.error is not None:
        raise LegacyPredictionAuditError("model run cites an unsuccessful training dataset")
    if model_run.task != dataset.task:
        raise LegacyPredictionAuditError("model run task does not match training dataset")
    if model_run.feature_version != dataset.feature_version:
        raise LegacyPredictionAuditError(
            "model run feature_version does not match training dataset"
        )
    if model_run.label_version != dataset.label_version:
        raise LegacyPredictionAuditError("model run label_version does not match training dataset")
    _verify_legacy_model_dataset_contract(model_run, dataset)

    dataset_manifest = _load_manifest_for_output_ref(archive, reference)
    _validate_manifest(
        dataset_manifest,
        artifact_type="training-dataset",
        schema_version=dataset.schema_version,
        transform_version=dataset.transform_version,
        generated_at=dataset.generated_at,
        input_refs=dataset.input_refs,
        output_refs=(reference,),
        quality=dataset.status.value,
        payload=document,
    )
    _validate_training_references(archive, dataset)


def _verify_legacy_model_dataset_contract(
    model_run: ModelRunArtifact,
    dataset: TrainingDatasetManifest,
) -> None:
    """Replay the schema-v1 cross-contract without applying later time policy."""

    if model_run.dataset_id != dataset.dataset_id:
        raise LegacyPredictionAuditError("model run references a different training dataset")
    by_id = {sample.sample_id: sample for sample in dataset.samples}
    missing = sorted(set(model_run.evaluation_cohort) - set(by_id))
    if missing:
        raise LegacyPredictionAuditError("model evaluation cohort contains unknown samples")
    cohort = [by_id[sample_id] for sample_id in model_run.evaluation_cohort]
    if any(not sample.eligible for sample in cohort):
        raise LegacyPredictionAuditError("model evaluation cohort contains excluded samples")
    if any(sample.feature_version != model_run.feature_version for sample in cohort):
        raise LegacyPredictionAuditError("model evaluation cohort feature version mismatch")
    if any(sample.label_version != model_run.label_version for sample in cohort):
        raise LegacyPredictionAuditError("model evaluation cohort label version mismatch")
    if any(sample.capture_mode is not model_run.evaluation_capture_mode for sample in cohort):
        raise LegacyPredictionAuditError(
            "model evaluation cohort capture mode does not match manifest"
        )
    if any(sample.split not in {"validation", "test", "holdout"} for sample in cohort):
        raise LegacyPredictionAuditError(
            "model evaluation cohort must use an out-of-time split, not train"
        )
    if model_run.status in {ModelRunStatus.SUCCEEDED, ModelRunStatus.PARTIAL}:
        strategy = dataset.split_strategy.lower()
        if not any(marker in strategy for marker in ("forward", "rolling", "temporal")):
            raise LegacyPredictionAuditError(
                "model evaluation requires a forward or temporal split strategy"
            )
        training = [
            sample for sample in dataset.samples if sample.eligible and sample.split == "train"
        ]
        if not training:
            raise LegacyPredictionAuditError(
                "model run requires at least one eligible training sample"
            )
        latest_training_as_of = max(sample.as_of for sample in training)
        if any(sample.as_of <= latest_training_as_of for sample in cohort):
            raise LegacyPredictionAuditError(
                "model evaluation cohort must be strictly later than the training window"
            )


def _validate_training_references(
    archive: DerivedArchive,
    dataset: TrainingDatasetManifest,
) -> None:
    raw = RawArchive(archive.layout)
    canonical = CanonicalStore(archive.layout.canonical / "platform.sqlite3")
    results: dict[str, Any] = {}

    for reference in dataset.input_refs:
        try:
            if reference.startswith("raw-asset:"):
                raw.verify(RawAssetId(reference))
            elif reference.startswith("derived-source:"):
                archive.validate_snapshot_source(reference)
            elif reference.startswith("snapshot:"):
                document = _read_referenced_json(
                    archive.layout.derived, "snapshots", reference, "snapshot"
                )
                snapshot = parse_snapshot_payload(document, source_validator=archive)
                _validate_snapshot_manifest(archive, snapshot, document)
            elif reference.startswith("team-baseline:"):
                baseline = archive.load_team_baseline(reference)
                document = json.loads(
                    archive.team_baseline_path(reference).read_text(encoding="utf-8")
                )
                manifest = _load_manifest_for_output_ref(archive, reference)
                _validate_manifest(
                    manifest,
                    artifact_type="team-baseline",
                    schema_version=baseline.schema_version,
                    transform_version=baseline.transform_version,
                    generated_at=manifest.generated_at,
                    input_refs=baseline.input_refs,
                    output_refs=(reference,),
                    quality=baseline.quality_status,
                    payload=document,
                )
            elif reference.startswith(("fact:", "canonical:")):
                if not canonical.path.is_file():
                    raise LegacyPredictionAuditError(
                        "training dataset canonical store is unavailable"
                    )
                results[reference] = load_verified_match_result(
                    reference,
                    archive=raw,
                    canonical=canonical,
                )
            else:
                raise LegacyPredictionAuditError(
                    f"unsupported training dataset input reference: {reference}"
                )
        except LegacyPredictionAuditError:
            raise
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise LegacyPredictionAuditError(
                f"training dataset input reference is unavailable or invalid: {reference}"
            ) from error

    if dataset.task == "score-model" and dataset.label_version == "result-90/1":
        for sample in dataset.samples:
            result = results.get(sample.label_ref)
            if result is None:
                raise LegacyPredictionAuditError(
                    f"score-model sample {sample.sample_id} lacks a verified result"
                )
            if (
                not isinstance(sample.label, dict)
                or type(sample.label.get("home_goals")) is not int
                or type(sample.label.get("away_goals")) is not int
                or sample.label["home_goals"] != result.home_goals
                or sample.label["away_goals"] != result.away_goals
            ):
                raise LegacyPredictionAuditError(
                    f"score-model sample {sample.sample_id} label does not match its result"
                )
            if result.known_at > sample.label_known_at:
                raise LegacyPredictionAuditError(
                    f"score-model sample {sample.sample_id} claims its result too early"
                )


def _contributions(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise LegacyPredictionAuditError("contribution_multipliers must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise LegacyPredictionAuditError("contributions must be objects")
        _require_keys(item, _CONTRIBUTION_KEYS, "contribution")
        _text(item["contribution_key"], "contribution_key")
        _float(item["lambda_home_multiplier"], "lambda_home_multiplier", positive=True)
        _float(item["lambda_away_multiplier"], "lambda_away_multiplier", positive=True)
        source_ref = _text(item["source_ref"], "source_ref")
        if ":" not in source_ref or any(char.isspace() for char in source_ref):
            raise LegacyPredictionAuditError("contribution source_ref must be typed")
        _text(item["version"], "contribution version")
        result.append(item)
    keys = tuple(item["contribution_key"] for item in result)
    if keys != tuple(sorted(set(keys))):
        raise LegacyPredictionAuditError("contributions must be unique and sorted")
    return tuple(result)


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


def _read_referenced_json(
    root: Path, directory: str, reference: str, prefix: str
) -> dict[str, Any]:
    digest = _digest_ref(reference, prefix)
    path = root / directory / digest[:2] / f"{digest}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LegacyPredictionAuditError(f"cannot read legacy {prefix}: {path}") from error
    if not isinstance(payload, dict) or payload.get("id") != reference:
        raise LegacyPredictionAuditError(f"legacy {prefix} ID does not match requested path")
    return payload


def _digest_ref(reference: Any, prefix: str) -> str:
    reference = _text(reference, f"{prefix} reference")
    marker = f"{prefix}:"
    digest = reference.removeprefix(marker) if reference.startswith(marker) else ""
    if not _DIGEST.fullmatch(digest):
        raise LegacyPredictionAuditError(f"{prefix} reference must contain a SHA-256 digest")
    return digest


def _require_keys(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise LegacyPredictionAuditError(
            f"{name} shape mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LegacyPredictionAuditError(f"{name} must be non-empty canonical text")
    return value


def _text_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LegacyPredictionAuditError(f"{name} must be a list")
    result = tuple(_text(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise LegacyPredictionAuditError(f"{name} must be unique and sorted")
    return result


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LegacyPredictionAuditError(f"{name} must be an integer")
    return value


def _float(value: Any, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise LegacyPredictionAuditError(f"{name} must be a finite JSON float")
    if positive and value <= 0:
        raise LegacyPredictionAuditError(f"{name} must be positive")
    return value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise LegacyPredictionAuditError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LegacyPredictionAuditError(f"{name} is not a valid timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LegacyPredictionAuditError(f"{name} must be timezone-aware UTC")
    parsed = parsed.astimezone(UTC)
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise LegacyPredictionAuditError(f"{name} is not a canonical UTC timestamp")
    return parsed


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _build_view(
    payload: dict[str, Any],
    *,
    generated_at: datetime,
    snapshot_as_of: datetime,
    prediction_manifest_id: str,
    composition_manifest_id: str,
    limitation: LegacyPredictionAuditLimitation,
) -> LegacyScorePredictionV3:
    return LegacyScorePredictionV3(
        id=payload["id"],
        schema_version=payload["schema_version"],
        match_id=payload["match_id"],
        snapshot_id=payload["snapshot_id"],
        capture_mode=CaptureMode(payload["capture_mode"]),
        snapshot_quality_status=payload["snapshot_quality_status"],
        model_run_id=payload["model_run_id"],
        model_version=payload["model_version"],
        generated_at=generated_at,
        snapshot_as_of=snapshot_as_of,
        lambda_home=payload["lambda_home"],
        lambda_away=payload["lambda_away"],
        rho=payload["rho"],
        max_goals=payload["max_goals"],
        score_cells=tuple(
            (item["home_goals"], item["away_goals"], item["probability"])
            for item in payload["score_cells"]
        ),
        markets=tuple(
            (
                market["market_type"],
                market["parameter"],
                tuple(
                    (outcome["outcome"], outcome["probability"]) for outcome in market["outcomes"]
                ),
            )
            for market in payload["markets"]
        ),
        normalization_residual=payload["normalization_residual"],
        input_refs=tuple(payload["input_refs"]),
        baseline_lambda_home=payload["baseline_lambda_home"],
        baseline_lambda_away=payload["baseline_lambda_away"],
        contribution_keys=tuple(payload["contribution_keys"]),
        contribution_multipliers=tuple(
            (
                item["contribution_key"],
                item["lambda_home_multiplier"],
                item["lambda_away_multiplier"],
                item["source_ref"],
                item["version"],
            )
            for item in payload["contribution_multipliers"]
        ),
        composition_version=payload["composition_version"],
        calibration_versions=tuple(payload["calibration_versions"]),
        composition_artifact_ref=payload["composition_artifact_ref"],
        prediction_manifest_id=prediction_manifest_id,
        composition_manifest_id=composition_manifest_id,
        audit_limitation=limitation,
        model_lineage_limitation=(
            LegacyModelLineageLimitation.TRAINING_DATASET_POLICY_UNVERIFIABLE
        ),
    )
