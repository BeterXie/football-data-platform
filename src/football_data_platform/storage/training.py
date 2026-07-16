"""Immutable storage for versioned training datasets and model runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import RawAssetId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.domain.training import (
    MODEL_ARTIFACT_REF_PREFIX,
    MODEL_OUTPUT_REF_PREFIX,
    DatasetStatus,
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
    verify_model_run_artifact,
    verify_training_dataset,
)
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive


class TrainingArtifactConflict(ArchiveConflictError):
    """Raised when a training artifact conflicts with immutable storage."""


class TrainingArtifactStore:
    """Persist datasets and model runs as content-addressed derived records."""

    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout.ensure()
        self.derived = DerivedArchive(self.layout)

    def write_dataset(self, dataset: TrainingDatasetManifest) -> Path:
        verify_training_dataset(dataset)
        self._validate_sample_references(dataset)
        self._validate_capture_evidence(dataset)
        path = self._write_json(self.dataset_path(dataset.dataset_id), dataset.to_payload())
        self.derived.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="training-dataset",
                schema_version=dataset.schema_version,
                payload=dataset.to_payload(),
                generated_at=dataset.generated_at,
                transform_version=dataset.transform_version,
                code_version=dataset.code_version or DERIVED_CODE_VERSION,
                input_refs=dataset.input_refs,
                output_refs=(dataset.dataset_id,),
                status=_manifest_status(dataset.status),
                error=(
                    dataset.error
                    if dataset.status in {DatasetStatus.FAILED, DatasetStatus.PARTIAL}
                    else None
                ),
                quality=dataset.status.value,
            )
        )
        return path

    def write_training_dataset(self, dataset: TrainingDatasetManifest) -> Path:
        """Alias used by pipeline callers."""

        return self.write_dataset(dataset)

    def create_dataset(self, **kwargs: Any) -> TrainingDatasetManifest:
        dataset = TrainingDatasetManifest.create(**kwargs)
        self.write_dataset(dataset)
        return dataset

    def load_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        path = self.dataset_path(dataset_id)
        payload = self._read_json(path)
        dataset = parse_training_dataset_payload(payload)
        if dataset.dataset_id != dataset_id:
            raise TrainingArtifactConflict("training dataset path and ID disagree")
        self._validate_sample_references(dataset)
        self._validate_capture_evidence(dataset)
        return dataset

    def load_training_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        return self.load_dataset(dataset_id)

    def write_model_run(self, artifact: ModelRunArtifact) -> Path:
        dataset = self._load_dataset_for_run(artifact)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)
        path = self._write_json(self.model_run_path(artifact.model_run_id), artifact.to_payload())
        input_refs = tuple(
            sorted(
                {
                    artifact.dataset_id,
                    *artifact.model_artifact_refs,
                    *artifact.evaluation_cohort,
                }
            )
        )
        self.derived.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="model-run",
                schema_version=artifact.schema_version,
                payload=artifact.to_payload(),
                generated_at=artifact.ended_at,
                transform_version=artifact.model_version,
                code_version=artifact.code_version or DERIVED_CODE_VERSION,
                input_refs=input_refs,
                output_refs=(artifact.model_run_id, *artifact.output_refs),
                status=_manifest_status(artifact.status),
                error=(
                    artifact.error
                    if artifact.status in {ModelRunStatus.FAILED, ModelRunStatus.PARTIAL}
                    else None
                ),
                quality=artifact.status.value,
            )
        )
        return path

    def write_model_run_artifact(self, artifact: ModelRunArtifact) -> Path:
        return self.write_model_run(artifact)

    def create_model_run(self, **kwargs: Any) -> ModelRunArtifact:
        artifact = ModelRunArtifact.create(**kwargs)
        self.write_model_run(artifact)
        return artifact

    def load_model_run(self, model_run_id: str) -> ModelRunArtifact:
        path = self.model_run_path(model_run_id)
        payload = self._read_json(path)
        artifact = parse_model_run_payload(payload)
        if artifact.model_run_id != model_run_id:
            raise TrainingArtifactConflict("model run path and ID disagree")
        dataset = self._load_dataset_for_run(artifact)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)
        return artifact

    def load_model_run_artifact(self, model_run_id: str) -> ModelRunArtifact:
        return self.load_model_run(model_run_id)

    def verify_model_run(self, artifact: ModelRunArtifact) -> None:
        dataset = self._load_dataset_for_run(artifact)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)

    def write_model_artifact(self, payload: bytes) -> str:
        """Persist immutable model bytes and return their content reference."""

        return self._write_content_bytes(payload, prefix=MODEL_ARTIFACT_REF_PREFIX)

    def write_model_output(self, payload: bytes) -> str:
        """Persist immutable evaluation/output bytes and return their content reference."""

        return self._write_content_bytes(payload, prefix=MODEL_OUTPUT_REF_PREFIX)

    def verify_model_artifact(self, reference: str) -> None:
        self._verify_content_bytes(reference, prefix=MODEL_ARTIFACT_REF_PREFIX)

    def verify_model_output(self, reference: str) -> None:
        self._verify_content_bytes(reference, prefix=MODEL_OUTPUT_REF_PREFIX)

    def model_artifact_path(self, reference: str) -> Path:
        digest = _digest_id(reference, MODEL_ARTIFACT_REF_PREFIX)
        return self.layout.derived / "model-artifacts" / "sha256" / digest[:2] / digest

    def model_output_path(self, reference: str) -> Path:
        digest = _digest_id(reference, MODEL_OUTPUT_REF_PREFIX)
        return self.layout.derived / "model-outputs" / "sha256" / digest[:2] / digest

    def dataset_path(self, dataset_id: str) -> Path:
        digest = _digest_id(dataset_id, "training-dataset:")
        return self.layout.derived / "training-datasets" / digest[:2] / f"{digest}.json"

    def model_run_path(self, model_run_id: str) -> Path:
        digest = _digest_id(model_run_id, "model-run:")
        return self.layout.derived / "model-runs" / digest[:2] / f"{digest}.json"

    def _load_dataset_for_run(self, artifact: ModelRunArtifact) -> TrainingDatasetManifest | None:
        if not self.dataset_path(artifact.dataset_id).exists():
            if artifact.status is ModelRunStatus.FAILED:
                return None
            raise TrainingArtifactConflict(
                "successful model runs require a persisted training dataset"
            )
        return self.load_dataset(artifact.dataset_id)

    def _validate_capture_evidence(self, dataset: TrainingDatasetManifest) -> None:
        raw = RawArchive(self.layout)
        for sample in dataset.samples:
            if sample.capture_mode is not CaptureMode.CAPTURED:
                continue
            assert sample.capture_evidence_ref is not None
            try:
                asset_id = RawAssetId(sample.capture_evidence_ref)
                raw.verify(asset_id)
                asset = raw.load(asset_id)
            except (OSError, ValueError, ArchiveConflictError) as error:
                raise TrainingArtifactConflict(
                    f"captured sample {sample.sample_id} lacks verifiable raw evidence"
                ) from error
            if asset.observed_at > sample.as_of:
                raise TrainingArtifactConflict(
                    f"captured sample {sample.sample_id} raw evidence follows as_of"
                )
            if (
                sample.capture_observed_at is not None
                and asset.observed_at > sample.capture_observed_at
            ):
                raise TrainingArtifactConflict(
                    f"captured sample {sample.sample_id} capture timestamp precedes raw evidence"
                )

    def _validate_sample_references(self, dataset: TrainingDatasetManifest) -> None:
        """Resolve every feature/label reference through an immutable store."""

        references = {
            reference
            for sample in dataset.samples
            for reference in (*sample.feature_refs, sample.label_ref)
        }
        for reference in sorted(references):
            try:
                self._verify_training_reference(reference)
            except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                raise TrainingArtifactConflict(
                    f"training sample reference is unavailable or invalid: {reference}"
                ) from error

    def _verify_training_reference(self, reference: str) -> None:
        if not isinstance(reference, str) or not reference or reference.strip() != reference:
            raise ValueError("reference must be non-empty text")
        if reference.startswith("raw-asset:"):
            RawArchive(self.layout).verify(RawAssetId(reference))
            return
        if reference.startswith("derived-source:"):
            _require_digest_reference(reference, "derived-source:")
            self.derived.validate_snapshot_source(reference)
            return
        if reference.startswith("team-baseline:"):
            self.derived.load_team_baseline(reference)
            self._verify_manifest_output(reference, "team-baseline")
            return
        if reference.startswith("snapshot:"):
            self._verify_json_artifact(reference, "snapshot:", "snapshots", "prematch-snapshot")
            return
        if reference.startswith("prediction:"):
            self._verify_prediction_reference(reference)
            return
        if reference.startswith("score-grid-composition:"):
            self.derived.load_score_grid_composition_payload(reference)
            self._verify_manifest_output(reference, "score-grid-composition")
            return
        if reference.startswith("derived-artifact:"):
            self.derived.load_artifact_manifest(reference)
            return
        if reference.startswith("training-dataset:"):
            self.load_dataset(reference)
            return
        if reference.startswith("model-run:"):
            self.load_model_run(reference)
            return
        if reference.startswith("model-artifact:"):
            self.verify_model_artifact(reference)
            return
        if reference.startswith("model-output:"):
            self.verify_model_output(reference)
            return
        if reference.startswith("fact:") or reference.startswith("canonical:"):
            self._verify_canonical_reference(reference)
            return
        raise ValueError(f"unsupported training reference type: {reference}")

    def _verify_manifest_output(self, reference: str, artifact_type: str) -> None:
        manifest = self.derived._load_artifact_manifest_for_output_ref(reference)
        if manifest.artifact_type != artifact_type or manifest.status != "succeeded":
            raise TrainingArtifactConflict(
                f"reference {reference} does not resolve to a succeeded {artifact_type} artifact"
            )

    def _verify_json_artifact(
        self,
        reference: str,
        prefix: str,
        directory: str,
        artifact_type: str,
    ) -> dict[str, Any]:
        digest = _digest_id(reference, prefix)
        path = self.layout.derived / directory / digest[:2] / f"{digest}.json"
        payload = self._read_json(path)
        if payload.get("id") != reference:
            raise TrainingArtifactConflict(f"{reference} ID does not match stored bytes")
        identity = dict(payload)
        identity.pop("id", None)
        if hashlib.sha256(_canonical_json(identity)).hexdigest() != digest:
            raise TrainingArtifactConflict(f"{reference} content hash does not match its ID")
        manifest = self.derived._load_artifact_manifest_for_output_ref(reference)
        if (
            manifest.artifact_type != artifact_type
            or manifest.status != "succeeded"
            or manifest.payload != payload
        ):
            raise TrainingArtifactConflict(f"{reference} manifest does not match stored bytes")
        return payload

    def _verify_prediction_reference(self, reference: str) -> None:
        payload = self._verify_json_artifact(reference, "prediction:", "predictions", "prediction")
        composition_ref = payload.get("composition_artifact_ref")
        if not isinstance(composition_ref, str):
            raise TrainingArtifactConflict("prediction lacks a composition artifact reference")
        composition = self.derived.load_score_grid_composition_payload(composition_ref)
        composition_manifest = self.derived._load_artifact_manifest_for_output_ref(composition_ref)
        if (
            composition_manifest.artifact_type != "score-grid-composition"
            or composition_manifest.status != "succeeded"
            or composition_manifest.payload != composition
        ):
            raise TrainingArtifactConflict("prediction composition manifest is invalid")
        expected = _prediction_composition_payload(payload)
        if composition != expected:
            raise TrainingArtifactConflict("prediction composition does not match prediction")

    def _verify_canonical_reference(self, reference: str) -> None:
        path = self.layout.canonical / "platform.sqlite3"
        if not path.exists():
            raise TrainingArtifactConflict("canonical store is unavailable")
        with sqlite3.connect(path) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            for (table_name,) in tables:
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
                if "record_id" not in columns:
                    continue
                if connection.execute(
                    f"SELECT 1 FROM {table_name} WHERE record_id = ? LIMIT 1", (reference,)
                ).fetchone():
                    return
        raise TrainingArtifactConflict(f"canonical record does not exist: {reference}")

    def _validate_model_bytes(self, artifact: ModelRunArtifact) -> None:
        for reference in artifact.model_artifact_refs:
            self.verify_model_artifact(reference)
        for reference, expected_hash in zip(
            artifact.output_refs, artifact.output_hashes, strict=True
        ):
            self.verify_model_output(reference)
            if reference.removeprefix(MODEL_OUTPUT_REF_PREFIX) != expected_hash:
                raise TrainingArtifactConflict(
                    "model output reference does not match its declared hash"
                )

    def _write_content_bytes(self, payload: bytes, *, prefix: str) -> str:
        if not isinstance(payload, bytes):
            raise TypeError("model artifact payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        reference = prefix + digest
        path = (
            self.model_artifact_path(reference)
            if prefix == MODEL_ARTIFACT_REF_PREFIX
            else self.model_output_path(reference)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise TrainingArtifactConflict(f"model content conflicts at {path}")
            return reference
        try:
            with path.open("xb") as destination:
                destination.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise TrainingArtifactConflict(f"model content conflicts at {path}") from None
        return reference

    def _verify_content_bytes(self, reference: str, *, prefix: str) -> None:
        path = (
            self.model_artifact_path(reference)
            if prefix == MODEL_ARTIFACT_REF_PREFIX
            else self.model_output_path(reference)
        )
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise TrainingArtifactConflict(
                f"model content is unavailable for {reference}"
            ) from error
        actual = hashlib.sha256(payload).hexdigest()
        if reference != prefix + actual:
            raise TrainingArtifactConflict(f"model content hash mismatch for {reference}")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingArtifactConflict(f"cannot read training artifact: {path}") from error
        if not isinstance(payload, dict):
            raise TrainingArtifactConflict(f"training artifact must be an object: {path}")
        return payload

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        encoded = (
            json.dumps(
                payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != encoded:
                raise TrainingArtifactConflict(f"training artifact conflicts at {path}")
            return path
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise TrainingArtifactConflict(f"training artifact conflicts at {path}") from None
        return path


def parse_training_dataset_payload(payload: dict[str, Any]) -> TrainingDatasetManifest:
    if not isinstance(payload, dict):
        raise TrainingArtifactConflict("training dataset must be an object")
    try:
        samples = tuple(_parse_sample(item) for item in payload["samples"])
        dataset = TrainingDatasetManifest(
            dataset_id=str(payload["id"]),
            schema_version=_strict_int(payload["schema_version"], "schema_version"),
            dataset_version=str(payload["dataset_version"]),
            task=str(payload["task"]),
            qualification=str(payload["qualification"]),
            qualification_ruleset_version=str(payload["qualification_ruleset_version"]),
            feature_version=str(payload["feature_version"]),
            label_version=str(payload["label_version"]),
            as_of=_parse_datetime(payload["as_of"], "dataset as_of"),
            split_strategy=str(payload["split_strategy"]),
            samples=samples,
            generated_at=_parse_datetime(payload["generated_at"], "dataset generated_at"),
            input_refs=tuple(str(item) for item in payload["input_refs"]),
            transform_version=str(payload["transform_version"]),
            code_version=str(payload["code_version"]),
            status=DatasetStatus(str(payload["status"])),
            error=None if payload.get("error") is None else str(payload["error"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TrainingArtifactConflict(f"invalid training dataset: {error}") from error
    try:
        verify_training_dataset(dataset)
    except (TypeError, ValueError) as error:
        raise TrainingArtifactConflict(str(error)) from error
    if payload != dataset.to_payload():
        raise TrainingArtifactConflict("training dataset is not canonical")
    return dataset


def parse_model_run_payload(payload: dict[str, Any]) -> ModelRunArtifact:
    if not isinstance(payload, dict):
        raise TrainingArtifactConflict("model run must be an object")
    try:
        artifact = ModelRunArtifact(
            model_run_id=str(payload["id"]),
            schema_version=_strict_int(payload["schema_version"], "schema_version"),
            model_version=str(payload["model_version"]),
            run_role=str(payload["run_role"]),
            task=str(payload["task"]),
            dataset_id=str(payload["dataset_id"]),
            feature_version=str(payload["feature_version"]),
            label_version=str(payload["label_version"]),
            algorithm=str(payload["algorithm"]),
            parameters=payload["parameters"],
            code_version=str(payload["code_version"]),
            environment_version=str(payload["environment_version"]),
            started_at=_parse_datetime(payload["started_at"], "model started_at"),
            ended_at=_parse_datetime(payload["ended_at"], "model ended_at"),
            random_seed=(
                None
                if payload.get("random_seed") is None
                else _strict_int(payload["random_seed"], "random_seed")
            ),
            model_artifact_refs=tuple(str(item) for item in payload["model_artifact_refs"]),
            evaluation_cohort=tuple(str(item) for item in payload["evaluation_cohort"]),
            evaluation_capture_mode=CaptureMode(str(payload["evaluation_capture_mode"])),
            output_refs=tuple(str(item) for item in payload["output_refs"]),
            output_hashes=tuple(str(item) for item in payload["output_hashes"]),
            status=ModelRunStatus(str(payload["status"])),
            error=None if payload.get("error") is None else str(payload["error"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TrainingArtifactConflict(f"invalid model run: {error}") from error
    try:
        verify_model_run_artifact(artifact)
    except (TypeError, ValueError) as error:
        raise TrainingArtifactConflict(str(error)) from error
    if payload != artifact.to_payload():
        raise TrainingArtifactConflict("model run is not canonical")
    return artifact


def _parse_sample(payload: Any) -> TrainingSample:
    if not isinstance(payload, dict):
        raise ValueError("training sample must be an object")
    return TrainingSample(
        sample_id=str(payload["sample_id"]),
        as_of=_parse_datetime(payload["as_of"], "sample as_of"),
        feature_known_at=_parse_datetime(payload["feature_known_at"], "feature_known_at"),
        label_known_at=_parse_datetime(payload["label_known_at"], "label_known_at"),
        capture_mode=CaptureMode(str(payload["capture_mode"])),
        qualification=str(payload["qualification"]),
        qualification_passed=_strict_bool(payload["qualification_passed"], "qualification_passed"),
        feature_version=str(payload["feature_version"]),
        label_version=str(payload["label_version"]),
        feature_refs=tuple(str(item) for item in payload["feature_refs"]),
        label_ref=str(payload["label_ref"]),
        features=payload["features"],
        label=payload["label"],
        exclusion_reasons=tuple(str(item) for item in payload.get("exclusion_reasons", ())),
        split=str(payload.get("split", "train")),
        capture_evidence_ref=(
            None
            if payload.get("capture_evidence_ref") is None
            else str(payload["capture_evidence_ref"])
        ),
        capture_observed_at=(
            None
            if payload.get("capture_observed_at") is None
            else _parse_datetime(payload["capture_observed_at"], "capture_observed_at")
        ),
    )


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_utc(parsed, field_name)
    return parsed.astimezone(UTC)


def _strict_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _digest_id(value: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"invalid {prefix} ID")
    digest = value.removeprefix(prefix)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid {prefix} ID")
    return digest


def _require_digest_reference(value: str, prefix: str) -> str:
    return _digest_id(value, prefix)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _prediction_composition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "artifact_type",
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
        "lambda_home",
        "lambda_away",
        "rho",
        "max_goals",
        "normalization_residual",
        "score_cells",
        "input_refs",
    )
    try:
        projected = {name: payload[name] for name in fields}
    except KeyError as error:
        raise TrainingArtifactConflict(
            f"prediction payload is missing composition field: {error.args[0]}"
        ) from error
    projected["schema_version"] = 1
    projected["artifact_type"] = "score-grid-composition"
    projected["grid"] = {
        "lambda_home": projected["lambda_home"],
        "lambda_away": projected["lambda_away"],
        "rho": projected["rho"],
        "max_goals": projected["max_goals"],
        "normalization_residual": projected["normalization_residual"],
        "score_cells": projected["score_cells"],
    }
    return projected


def _manifest_status(status: DatasetStatus | ModelRunStatus) -> str:
    if status.value == "succeeded":
        return "succeeded"
    if status.value == "partial":
        return "partial"
    return "failed"


# Compatibility aliases for callers that use archive/manifest terminology.
TrainingArtifactArchive = TrainingArtifactStore
DerivedTrainingArchive = TrainingArtifactStore
parse_dataset_payload = parse_training_dataset_payload
parse_model_artifact_payload = parse_model_run_payload
