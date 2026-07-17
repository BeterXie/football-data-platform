"""Versioned training-data and model-run contracts.

Training artifacts are derived records, but their provenance is part of the
model contract.  The records in this module intentionally keep the per-sample
qualification and capture mode instead of reducing them to one global flag.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode

TRAINING_DATASET_SCHEMA_VERSION = 1
MODEL_RUN_ARTIFACT_SCHEMA_VERSION = 1
_DATASET_PREFIX = "training-dataset:"
_MODEL_RUN_PREFIX = "model-run:"
MODEL_ARTIFACT_REF_PREFIX = "model-artifact:"
MODEL_OUTPUT_REF_PREFIX = "model-output:"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_DATASET_STATUSES = frozenset({"succeeded", "partial", "failed"})
_KNOWN_MODEL_STATUSES = frozenset({"succeeded", "partial", "failed"})
_KNOWN_SPLITS = frozenset({"train", "validation", "test", "holdout"})
_OUT_OF_TIME_SPLITS = frozenset({"validation", "test", "holdout"})
_TEMPORAL_SPLIT_MARKERS = ("forward", "rolling", "temporal")
_RESULT_90_LABEL_FIELDS = frozenset({"home_goals", "away_goals"})


class TrainingArtifactConflict(ValueError):
    """Raised when a training artifact is not canonical or content addressed."""


class DatasetStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class ModelRunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


def validate_training_label(label_version: str, label: Any) -> None:
    """Validate the known schema for one versioned training label."""

    _require_text(label_version, "label_version")
    _ensure_json(label, "label")
    if label_version != "result-90/1":
        return
    if not isinstance(label, dict) or set(label) != _RESULT_90_LABEL_FIELDS:
        raise ValueError("result-90/1 label must contain exactly home_goals and away_goals")
    if any(type(label[field]) is not int or label[field] < 0 for field in _RESULT_90_LABEL_FIELDS):
        raise ValueError("result-90/1 label goals must be non-negative integers")


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One auditable feature/label pair.

    ``features`` and ``label`` are retained in the manifest so an offline
    replay does not depend on a mutable feature service.  A sample can be
    present but excluded; exclusion is represented explicitly rather than by
    dropping the row.
    """

    sample_id: str
    as_of: datetime
    feature_known_at: datetime
    label_known_at: datetime
    capture_mode: CaptureMode
    qualification: str
    qualification_passed: bool
    feature_version: str
    label_version: str
    feature_refs: tuple[str, ...]
    label_ref: str
    features: Any
    label: Any
    exclusion_reasons: tuple[str, ...] = ()
    split: str = "train"
    capture_evidence_ref: str | None = None
    capture_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.qualification, "qualification")
        _require_text(self.feature_version, "feature_version")
        _require_text(self.label_version, "label_version")
        _require_text(self.label_ref, "label_ref")
        if self.split not in _KNOWN_SPLITS:
            raise ValueError(f"unsupported training split {self.split!r}")
        for name, value in (
            ("as_of", self.as_of),
            ("feature_known_at", self.feature_known_at),
            ("label_known_at", self.label_known_at),
        ):
            require_utc(value, name)
        if not isinstance(self.capture_mode, CaptureMode):
            raise TypeError("capture_mode must be a CaptureMode")
        if not isinstance(self.qualification_passed, bool):
            raise TypeError("qualification_passed must be a bool")
        refs = _normalize_refs(self.feature_refs, "feature_refs")
        if refs != self.feature_refs:
            raise ValueError("feature_refs must be sorted and unique")
        reasons = _normalize_reasons(self.exclusion_reasons)
        if reasons != self.exclusion_reasons:
            raise ValueError("exclusion_reasons must be sorted and unique")
        if self.feature_known_at > self.as_of and self.qualification_passed:
            raise ValueError("eligible sample feature is known after sample as_of")
        if self.label_known_at <= self.as_of and self.qualification_passed:
            raise ValueError("eligible sample label is known at or before sample as_of")
        if self.qualification_passed and reasons:
            raise ValueError("eligible samples cannot contain exclusion_reasons")
        if not self.qualification_passed and not reasons:
            raise ValueError("excluded samples require exclusion_reasons")
        if self.capture_mode is CaptureMode.CAPTURED:
            if self.capture_evidence_ref is None:
                raise ValueError("captured samples require capture_evidence_ref")
            _require_text(self.capture_evidence_ref, "capture_evidence_ref")
            if not self.capture_evidence_ref.startswith("raw-asset:"):
                raise ValueError("capture_evidence_ref must reference immutable raw evidence")
            if self.capture_observed_at is None:
                raise ValueError("captured samples require capture_observed_at")
            require_utc(self.capture_observed_at, "capture_observed_at")
            if self.capture_observed_at > self.as_of:
                raise ValueError("capture_observed_at cannot follow sample as_of")
        elif self.capture_evidence_ref is not None or self.capture_observed_at is not None:
            raise ValueError("reconstructed samples cannot carry captured evidence metadata")
        _ensure_json(self.features, "features")
        validate_training_label(self.label_version, self.label)

    @property
    def eligible(self) -> bool:
        return self.qualification_passed and not self.exclusion_reasons

    @property
    def feature_values(self) -> Any:
        """Compatibility alias for callers using feature_values terminology."""

        return self.features

    @property
    def label_value(self) -> Any:
        """Compatibility alias for callers using label_value terminology."""

        return self.label

    @property
    def input_refs(self) -> tuple[str, ...]:
        refs = set(self.feature_refs) | {self.label_ref}
        if self.capture_evidence_ref is not None:
            refs.add(self.capture_evidence_ref)
        return tuple(sorted(refs))

    @property
    def feature_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.features)).hexdigest()

    @property
    def label_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.label)).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingDatasetManifest:
    """Content-addressed dataset manifest with an auditable sample ledger."""

    dataset_id: str
    schema_version: int
    dataset_version: str
    task: str
    qualification: str
    qualification_ruleset_version: str
    feature_version: str
    label_version: str
    as_of: datetime
    split_strategy: str
    samples: tuple[TrainingSample, ...]
    generated_at: datetime
    input_refs: tuple[str, ...]
    transform_version: str
    code_version: str
    status: DatasetStatus = DatasetStatus.SUCCEEDED
    error: str | None = None

    @classmethod
    def create(
        cls,
        *,
        dataset_version: str,
        task: str,
        qualification: str,
        qualification_ruleset_version: str,
        feature_version: str,
        label_version: str,
        as_of: datetime,
        split_strategy: str,
        samples: Sequence[TrainingSample],
        generated_at: datetime,
        input_refs: Sequence[str] = (),
        transform_version: str = "training-dataset/1",
        code_version: str = "unknown",
        status: DatasetStatus | str | None = None,
        error: str | None = None,
    ) -> TrainingDatasetManifest:
        normalized_samples = _normalize_samples(samples)
        normalized_input_refs = tuple(
            sorted(
                set(_normalize_refs(input_refs, "input_refs"))
                | set(ref for sample in normalized_samples for ref in sample.input_refs)
            )
        )
        inferred_status = _infer_dataset_status(normalized_samples)
        normalized_status = DatasetStatus(inferred_status if status is None else status)
        if status is not None and normalized_status is not inferred_status:
            raise ValueError(
                "dataset status must match per-sample eligibility; it cannot claim ready samples"
            )
        normalized_error = error
        if normalized_status is DatasetStatus.FAILED and normalized_error is None:
            normalized_error = "no_eligible_training_samples"
        fields = _dataset_fields(
            dataset_version=dataset_version,
            task=task,
            qualification=qualification,
            qualification_ruleset_version=qualification_ruleset_version,
            feature_version=feature_version,
            label_version=label_version,
            as_of=as_of,
            split_strategy=split_strategy,
            samples=normalized_samples,
            generated_at=generated_at,
            input_refs=normalized_input_refs,
            transform_version=transform_version,
            code_version=code_version,
            status=normalized_status,
            error=normalized_error,
        )
        digest = hashlib.sha256(
            _canonical_json({"schema_version": TRAINING_DATASET_SCHEMA_VERSION, **fields})
        ).hexdigest()
        return cls(
            dataset_id=_DATASET_PREFIX + digest,
            schema_version=TRAINING_DATASET_SCHEMA_VERSION,
            **fields,
        )

    @property
    def id(self) -> str:
        return self.dataset_id

    @property
    def content_hash(self) -> str:
        return self.dataset_id.removeprefix(_DATASET_PREFIX)

    @property
    def included_samples(self) -> tuple[TrainingSample, ...]:
        return tuple(sample for sample in self.samples if sample.eligible)

    @property
    def excluded_samples(self) -> tuple[TrainingSample, ...]:
        return tuple(sample for sample in self.samples if not sample.eligible)

    @property
    def capture_mode_counts(self) -> dict[str, int]:
        counts = {mode.value: 0 for mode in CaptureMode}
        for sample in self.samples:
            counts[sample.capture_mode.value] += 1
        return counts

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.dataset_id, **dataset_identity_payload(self)}


@dataclass(frozen=True, slots=True)
class ModelRunArtifact:
    """Immutable model run metadata tied to one persisted dataset."""

    model_run_id: str
    schema_version: int
    model_version: str
    run_role: str
    task: str
    dataset_id: str
    feature_version: str
    label_version: str
    algorithm: str
    parameters: Any
    code_version: str
    environment_version: str
    started_at: datetime
    ended_at: datetime
    random_seed: int | None
    model_artifact_refs: tuple[str, ...]
    evaluation_cohort: tuple[str, ...]
    evaluation_capture_mode: CaptureMode
    output_refs: tuple[str, ...]
    output_hashes: tuple[str, ...]
    status: ModelRunStatus = ModelRunStatus.SUCCEEDED
    error: str | None = None

    @classmethod
    def create(
        cls,
        *,
        model_version: str,
        run_role: str,
        task: str,
        dataset_id: str,
        feature_version: str,
        label_version: str,
        algorithm: str,
        parameters: Any,
        code_version: str,
        environment_version: str,
        started_at: datetime,
        ended_at: datetime,
        random_seed: int | None,
        model_artifact_refs: Sequence[str],
        evaluation_cohort: Sequence[str],
        evaluation_capture_mode: CaptureMode,
        output_refs: Sequence[str],
        output_hashes: Sequence[str],
        status: ModelRunStatus | str = ModelRunStatus.SUCCEEDED,
        error: str | None = None,
    ) -> ModelRunArtifact:
        fields = _model_run_fields(
            model_version=model_version,
            run_role=run_role,
            task=task,
            dataset_id=dataset_id,
            feature_version=feature_version,
            label_version=label_version,
            algorithm=algorithm,
            parameters=parameters,
            code_version=code_version,
            environment_version=environment_version,
            started_at=started_at,
            ended_at=ended_at,
            random_seed=random_seed,
            model_artifact_refs=model_artifact_refs,
            evaluation_cohort=evaluation_cohort,
            evaluation_capture_mode=evaluation_capture_mode,
            output_refs=output_refs,
            output_hashes=output_hashes,
            status=ModelRunStatus(status),
            error=error,
        )
        digest = hashlib.sha256(
            _canonical_json({"schema_version": MODEL_RUN_ARTIFACT_SCHEMA_VERSION, **fields})
        ).hexdigest()
        return cls(
            model_run_id=_MODEL_RUN_PREFIX + digest,
            schema_version=MODEL_RUN_ARTIFACT_SCHEMA_VERSION,
            **fields,
        )

    @property
    def id(self) -> str:
        return self.model_run_id

    @property
    def content_hash(self) -> str:
        return self.model_run_id.removeprefix(_MODEL_RUN_PREFIX)

    @property
    def training_dataset_id(self) -> str:
        return self.dataset_id

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.model_run_id, **model_run_identity_payload(self)}


def verify_training_sample(sample: TrainingSample) -> None:
    """Re-run sample-level anti-leakage and qualification checks."""

    if not isinstance(sample, TrainingSample):
        raise TypeError("sample must be a TrainingSample")
    # __post_init__ contains the invariant checks; calling it explicitly also
    # catches a forged object produced through object.__new__.
    TrainingSample(
        sample_id=sample.sample_id,
        as_of=sample.as_of,
        feature_known_at=sample.feature_known_at,
        label_known_at=sample.label_known_at,
        capture_mode=sample.capture_mode,
        qualification=sample.qualification,
        qualification_passed=sample.qualification_passed,
        feature_version=sample.feature_version,
        label_version=sample.label_version,
        feature_refs=sample.feature_refs,
        label_ref=sample.label_ref,
        features=sample.features,
        label=sample.label,
        exclusion_reasons=sample.exclusion_reasons,
        split=sample.split,
        capture_evidence_ref=sample.capture_evidence_ref,
        capture_observed_at=sample.capture_observed_at,
    )


def verify_training_dataset(dataset: TrainingDatasetManifest) -> None:
    """Verify dataset identity, versions, sample eligibility and lineage."""

    if not isinstance(dataset, TrainingDatasetManifest):
        raise TypeError("dataset must be a TrainingDatasetManifest")
    if dataset.schema_version != TRAINING_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported training dataset schema_version")
    if not isinstance(dataset.status, DatasetStatus):
        raise TypeError("dataset status must be a DatasetStatus")
    fields = _dataset_fields(
        dataset_version=dataset.dataset_version,
        task=dataset.task,
        qualification=dataset.qualification,
        qualification_ruleset_version=dataset.qualification_ruleset_version,
        feature_version=dataset.feature_version,
        label_version=dataset.label_version,
        as_of=dataset.as_of,
        split_strategy=dataset.split_strategy,
        samples=dataset.samples,
        generated_at=dataset.generated_at,
        input_refs=dataset.input_refs,
        transform_version=dataset.transform_version,
        code_version=dataset.code_version,
        status=dataset.status,
        error=dataset.error,
    )
    expected = (
        _DATASET_PREFIX
        + hashlib.sha256(
            _canonical_json({"schema_version": dataset.schema_version, **fields})
        ).hexdigest()
    )
    if dataset.dataset_id != expected:
        raise ValueError("training dataset identity does not match canonical content")


def verify_model_run_artifact(
    artifact: ModelRunArtifact,
    *,
    dataset: TrainingDatasetManifest | None = None,
) -> None:
    """Verify model-run metadata and, when supplied, its dataset cohort."""

    if not isinstance(artifact, ModelRunArtifact):
        raise TypeError("artifact must be a ModelRunArtifact")
    if artifact.schema_version != MODEL_RUN_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported model run schema_version")
    if not isinstance(artifact.status, ModelRunStatus):
        raise TypeError("model run status must be a ModelRunStatus")
    fields = _model_run_fields(
        model_version=artifact.model_version,
        run_role=artifact.run_role,
        task=artifact.task,
        dataset_id=artifact.dataset_id,
        feature_version=artifact.feature_version,
        label_version=artifact.label_version,
        algorithm=artifact.algorithm,
        parameters=artifact.parameters,
        code_version=artifact.code_version,
        environment_version=artifact.environment_version,
        started_at=artifact.started_at,
        ended_at=artifact.ended_at,
        random_seed=artifact.random_seed,
        model_artifact_refs=artifact.model_artifact_refs,
        evaluation_cohort=artifact.evaluation_cohort,
        evaluation_capture_mode=artifact.evaluation_capture_mode,
        output_refs=artifact.output_refs,
        output_hashes=artifact.output_hashes,
        status=artifact.status,
        error=artifact.error,
    )
    expected = (
        _MODEL_RUN_PREFIX
        + hashlib.sha256(
            _canonical_json({"schema_version": artifact.schema_version, **fields})
        ).hexdigest()
    )
    if artifact.model_run_id != expected:
        raise ValueError("model run identity does not match canonical content")
    if dataset is not None:
        verify_training_dataset(dataset)
        if artifact.dataset_id != dataset.dataset_id:
            raise ValueError("model run references a different training dataset")
        if dataset.generated_at > artifact.started_at:
            raise ValueError("model run cannot start before its training dataset is generated")
        by_id = {sample.sample_id: sample for sample in dataset.samples}
        missing = sorted(set(artifact.evaluation_cohort) - set(by_id))
        if missing:
            raise ValueError("model evaluation cohort contains unknown samples")
        cohort = [by_id[sample_id] for sample_id in artifact.evaluation_cohort]
        if any(not sample.eligible for sample in cohort):
            raise ValueError("model evaluation cohort contains excluded samples")
        if any(sample.feature_version != artifact.feature_version for sample in cohort):
            raise ValueError("model evaluation cohort feature version mismatch")
        if any(sample.label_version != artifact.label_version for sample in cohort):
            raise ValueError("model evaluation cohort label version mismatch")
        if any(sample.capture_mode is not artifact.evaluation_capture_mode for sample in cohort):
            raise ValueError("model evaluation cohort capture mode does not match manifest")
        if any(sample.split not in _OUT_OF_TIME_SPLITS for sample in cohort):
            raise ValueError("model evaluation cohort must use an out-of-time split, not train")
        if artifact.status in {ModelRunStatus.SUCCEEDED, ModelRunStatus.PARTIAL}:
            strategy = dataset.split_strategy.lower()
            if not any(marker in strategy for marker in _TEMPORAL_SPLIT_MARKERS):
                raise ValueError("model evaluation requires a forward or temporal split strategy")
            training = [
                sample for sample in dataset.samples if sample.eligible and sample.split == "train"
            ]
            if not training:
                raise ValueError("model run requires at least one eligible training sample")
            latest_training_as_of = max(sample.as_of for sample in training)
            if any(sample.as_of <= latest_training_as_of for sample in cohort):
                raise ValueError(
                    "model evaluation cohort must be strictly later than the training window"
                )


def training_sample_payload(sample: TrainingSample) -> dict[str, Any]:
    verify_training_sample(sample)
    return {
        "sample_id": sample.sample_id,
        "as_of": _timestamp(sample.as_of),
        "feature_known_at": _timestamp(sample.feature_known_at),
        "label_known_at": _timestamp(sample.label_known_at),
        "capture_mode": sample.capture_mode.value,
        "qualification": sample.qualification,
        "qualification_passed": sample.qualification_passed,
        "feature_version": sample.feature_version,
        "label_version": sample.label_version,
        "feature_refs": list(sample.feature_refs),
        "label_ref": sample.label_ref,
        "features": sample.features,
        "label": sample.label,
        "feature_hash": sample.feature_hash,
        "label_hash": sample.label_hash,
        "exclusion_reasons": list(sample.exclusion_reasons),
        "split": sample.split,
        "capture_evidence_ref": sample.capture_evidence_ref,
        "capture_observed_at": (
            None if sample.capture_observed_at is None else _timestamp(sample.capture_observed_at)
        ),
    }


def dataset_identity_payload(dataset: TrainingDatasetManifest) -> dict[str, Any]:
    verify_training_dataset(dataset)
    return {
        "schema_version": dataset.schema_version,
        "dataset_version": dataset.dataset_version,
        "task": dataset.task,
        "qualification": dataset.qualification,
        "qualification_ruleset_version": dataset.qualification_ruleset_version,
        "feature_version": dataset.feature_version,
        "label_version": dataset.label_version,
        "as_of": _timestamp(dataset.as_of),
        "split_strategy": dataset.split_strategy,
        "samples": [training_sample_payload(sample) for sample in dataset.samples],
        "generated_at": _timestamp(dataset.generated_at),
        "input_refs": list(dataset.input_refs),
        "transform_version": dataset.transform_version,
        "code_version": dataset.code_version,
        "status": dataset.status.value,
        "error": dataset.error,
    }


def model_run_identity_payload(artifact: ModelRunArtifact) -> dict[str, Any]:
    verify_model_run_artifact(artifact)
    return {
        "schema_version": artifact.schema_version,
        "model_version": artifact.model_version,
        "run_role": artifact.run_role,
        "task": artifact.task,
        "dataset_id": artifact.dataset_id,
        "feature_version": artifact.feature_version,
        "label_version": artifact.label_version,
        "algorithm": artifact.algorithm,
        "parameters": artifact.parameters,
        "code_version": artifact.code_version,
        "environment_version": artifact.environment_version,
        "started_at": _timestamp(artifact.started_at),
        "ended_at": _timestamp(artifact.ended_at),
        "random_seed": artifact.random_seed,
        "model_artifact_refs": list(artifact.model_artifact_refs),
        "evaluation_cohort": list(artifact.evaluation_cohort),
        "evaluation_capture_mode": artifact.evaluation_capture_mode.value,
        "output_refs": list(artifact.output_refs),
        "output_hashes": list(artifact.output_hashes),
        "status": artifact.status.value,
        "error": artifact.error,
    }


def feature_label_payload(sample: TrainingSample) -> dict[str, Any]:
    """Return the versioned feature/label identity for one sample."""

    verify_training_sample(sample)
    return {
        "sample_id": sample.sample_id,
        "feature_version": sample.feature_version,
        "label_version": sample.label_version,
        "feature_hash": sample.feature_hash,
        "label_hash": sample.label_hash,
        "feature_refs": list(sample.feature_refs),
        "label_ref": sample.label_ref,
    }


def build_training_dataset(**kwargs: Any) -> TrainingDatasetManifest:
    """Compatibility factory for command-line and pipeline callers."""

    return TrainingDatasetManifest.create(**kwargs)


def build_model_run_artifact(**kwargs: Any) -> ModelRunArtifact:
    """Compatibility factory for model runners."""

    return ModelRunArtifact.create(**kwargs)


def _dataset_fields(**kwargs: Any) -> dict[str, Any]:
    for name in (
        "dataset_version",
        "task",
        "qualification",
        "qualification_ruleset_version",
        "feature_version",
        "label_version",
        "split_strategy",
        "transform_version",
        "code_version",
    ):
        _require_text(kwargs[name], name)
    require_utc(kwargs["as_of"], "dataset as_of")
    require_utc(kwargs["generated_at"], "dataset generated_at")
    if kwargs["generated_at"] < kwargs["as_of"]:
        raise ValueError("dataset generated_at cannot precede as_of")
    samples = _normalize_samples(kwargs["samples"])
    if not samples:
        raise ValueError("training dataset requires at least one sample")
    status = DatasetStatus(kwargs["status"])
    if status is not _infer_dataset_status(samples):
        raise ValueError("dataset status must match per-sample eligibility")
    if status is DatasetStatus.FAILED and not kwargs["error"]:
        raise ValueError("failed training dataset requires error")
    if status is DatasetStatus.SUCCEEDED and kwargs["error"] is not None:
        raise ValueError("succeeded training dataset cannot contain error")
    refs = _normalize_refs(kwargs["input_refs"], "input_refs")
    sample_refs = {ref for sample in samples for ref in sample.input_refs}
    if not sample_refs <= set(refs):
        raise ValueError("dataset input_refs must include every sample input reference")
    for sample in samples:
        if sample.qualification != kwargs["qualification"]:
            raise ValueError("sample qualification does not match dataset qualification")
        if sample.feature_version != kwargs["feature_version"]:
            raise ValueError("sample feature_version does not match dataset")
        if sample.label_version != kwargs["label_version"]:
            raise ValueError("sample label_version does not match dataset")
        if sample.as_of > kwargs["as_of"]:
            raise ValueError("sample as_of cannot follow dataset as_of")
        for field_name, known_at in (
            ("feature_known_at", sample.feature_known_at),
            ("label_known_at", sample.label_known_at),
            ("capture_observed_at", sample.capture_observed_at),
        ):
            if known_at is not None and known_at > kwargs["generated_at"]:
                raise ValueError(f"dataset generated_at cannot precede sample {field_name}")
    return {
        "dataset_version": kwargs["dataset_version"],
        "task": kwargs["task"],
        "qualification": kwargs["qualification"],
        "qualification_ruleset_version": kwargs["qualification_ruleset_version"],
        "feature_version": kwargs["feature_version"],
        "label_version": kwargs["label_version"],
        "as_of": kwargs["as_of"],
        "split_strategy": kwargs["split_strategy"],
        "samples": samples,
        "generated_at": kwargs["generated_at"],
        "input_refs": refs,
        "transform_version": kwargs["transform_version"],
        "code_version": kwargs["code_version"],
        "status": status,
        "error": kwargs["error"],
    }


def _model_run_fields(**kwargs: Any) -> dict[str, Any]:
    for name in (
        "model_version",
        "run_role",
        "task",
        "feature_version",
        "label_version",
        "algorithm",
        "code_version",
        "environment_version",
    ):
        _require_text(kwargs[name], name)
    _validate_digest_id(kwargs["dataset_id"], _DATASET_PREFIX, "dataset_id")
    for name in ("started_at", "ended_at"):
        require_utc(kwargs[name], name)
    if kwargs["ended_at"] < kwargs["started_at"]:
        raise ValueError("model run ended_at cannot precede started_at")
    if kwargs["random_seed"] is not None and (
        not isinstance(kwargs["random_seed"], int) or isinstance(kwargs["random_seed"], bool)
    ):
        raise ValueError("random_seed must be an integer when present")
    if not isinstance(kwargs["evaluation_capture_mode"], CaptureMode):
        raise TypeError("evaluation_capture_mode must be a CaptureMode")
    model_refs = _normalize_refs(kwargs["model_artifact_refs"], "model_artifact_refs")
    cohort = _normalize_unique_refs(kwargs["evaluation_cohort"], "evaluation_cohort")
    output_refs = _normalize_refs(kwargs["output_refs"], "output_refs")
    hashes = tuple(kwargs["output_hashes"])
    if len(output_refs) != len(hashes) or any(not _DIGEST.fullmatch(str(item)) for item in hashes):
        raise ValueError("output_refs and output_hashes must have matching SHA-256 entries")
    for reference in model_refs:
        _validate_digest_id(reference, MODEL_ARTIFACT_REF_PREFIX, "model_artifact_refs")
    for reference, digest in zip(output_refs, hashes, strict=True):
        _validate_digest_id(reference, MODEL_OUTPUT_REF_PREFIX, "output_refs")
        if reference.removeprefix(MODEL_OUTPUT_REF_PREFIX) != str(digest):
            raise ValueError("output reference digest must match its output_hash")
    status = ModelRunStatus(kwargs["status"])
    if kwargs["run_role"] in {"formal", "champion"} and (
        kwargs["evaluation_capture_mode"] is not CaptureMode.CAPTURED
    ):
        raise ValueError("formal and champion model runs require a captured evaluation cohort")
    if status is ModelRunStatus.SUCCEEDED and (not model_refs or not cohort or not output_refs):
        raise ValueError("successful model runs require model, cohort, and output references")
    if status is ModelRunStatus.FAILED and not kwargs["error"]:
        raise ValueError("failed model runs require error")
    _ensure_json(kwargs["parameters"], "parameters")
    return {
        "model_version": kwargs["model_version"],
        "run_role": kwargs["run_role"],
        "task": kwargs["task"],
        "dataset_id": kwargs["dataset_id"],
        "feature_version": kwargs["feature_version"],
        "label_version": kwargs["label_version"],
        "algorithm": kwargs["algorithm"],
        "parameters": _json_safe(kwargs["parameters"]),
        "code_version": kwargs["code_version"],
        "environment_version": kwargs["environment_version"],
        "started_at": kwargs["started_at"],
        "ended_at": kwargs["ended_at"],
        "random_seed": kwargs["random_seed"],
        "model_artifact_refs": model_refs,
        "evaluation_cohort": cohort,
        "evaluation_capture_mode": kwargs["evaluation_capture_mode"],
        "output_refs": output_refs,
        "output_hashes": tuple(str(item) for item in hashes),
        "status": status,
        "error": kwargs["error"],
    }


def _normalize_samples(samples: Sequence[TrainingSample]) -> tuple[TrainingSample, ...]:
    if isinstance(samples, (str, bytes)):
        raise ValueError("samples must be a sequence of TrainingSample values")
    try:
        normalized = tuple(samples)
    except TypeError as error:
        raise ValueError("samples must be a sequence of TrainingSample values") from error
    if any(not isinstance(sample, TrainingSample) for sample in normalized):
        raise TypeError("samples must contain TrainingSample values")
    for sample in normalized:
        verify_training_sample(sample)
    ordered = tuple(sorted(normalized, key=lambda item: item.sample_id))
    if len({sample.sample_id for sample in ordered}) != len(ordered):
        raise ValueError("training sample IDs must be unique")
    return ordered


def _infer_dataset_status(samples: Sequence[TrainingSample]) -> DatasetStatus:
    eligible = sum(sample.eligible for sample in samples)
    if eligible == len(samples):
        return DatasetStatus.SUCCEEDED
    if eligible:
        return DatasetStatus.PARTIAL
    return DatasetStatus.FAILED


def _normalize_refs(refs: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(refs, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of references")
    try:
        normalized = tuple(refs)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of references") from error
    if any(not isinstance(item, str) or not item or item.strip() != item for item in normalized):
        raise ValueError(f"{field_name} must contain non-empty references")
    return tuple(sorted(set(normalized)))


def _normalize_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    normalized = _normalize_refs(reasons, "exclusion_reasons")
    return normalized


def _normalize_unique_refs(refs: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(refs, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of references")
    try:
        values = tuple(refs)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of references") from error
    normalized = _normalize_refs(values, field_name)
    if len(normalized) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate references")
    return normalized


def _validate_digest_id(value: str, prefix: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{field_name} must start with {prefix!r}")
    if not _DIGEST.fullmatch(value.removeprefix(prefix)):
        raise ValueError(f"{field_name} must contain a SHA-256 digest")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


def _ensure_json(value: Any, field_name: str) -> None:
    try:
        json.dumps(_json_safe(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite JSON data") from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        require_utc(value, "training payload datetime")
        return _timestamp(value)
    if isinstance(value, TrainingSample):
        return training_sample_payload(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("training payload contains a non-finite number")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _timestamp(value: datetime) -> str:
    require_utc(value, "training timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# Discoverable compatibility aliases.
TrainingDataset = TrainingDatasetManifest
TrainingDatasetArtifact = TrainingDatasetManifest
FeatureLabelSample = TrainingSample
ModelRun = ModelRunArtifact
ModelArtifactManifest = ModelRunArtifact
verify_model_run = verify_model_run_artifact
