"""Immutable storage for versioned training datasets and model runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.lifecycle import (
    Qualification,
    QualificationResult,
    SnapshotAvailability,
    assess_lifecycle,
)
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    ScorePrediction,
    parse_prediction_payload,
    verify_prediction_snapshot,
)
from football_data_platform.domain.snapshots import (
    CURRENT_FEATURE_SPEC_VERSION,
    CaptureMode,
    PreMatchSnapshot,
    SnapshotSourceValidation,
    parse_snapshot_payload,
)
from football_data_platform.domain.training import (
    LEGACY_TRAINING_DATASET_SCHEMA_VERSION,
    MODEL_ARTIFACT_REF_PREFIX,
    MODEL_OUTPUT_REF_PREFIX,
    TRAINING_DATASET_SCHEMA_VERSION,
    DatasetStatus,
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
    verify_model_run_artifact,
    verify_training_dataset,
)
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
    TrainingQualification,
    parse_training_qualification_payload,
    score_feature_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
)
from football_data_platform.storage.facts import (
    CanonicalFactStore,
    load_verified_match_result,
    load_verified_team_observation,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

if TYPE_CHECKING:
    from football_data_platform.storage.verification import VerificationSession


class TrainingArtifactConflict(ArchiveConflictError):
    """Raised when a training artifact conflicts with immutable storage."""


@dataclass(frozen=True, slots=True)
class _TrainingReferenceAvailability:
    reference: str
    available_at: datetime | None
    semantic_known_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _SessionSnapshotSourceValidator:
    archive: DerivedArchive
    verification_session: VerificationSession

    def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation:
        return self.archive.validate_snapshot_source(
            source_ref,
            verification_session=self.verification_session,
        )


@dataclass(frozen=True, slots=True)
class _AuditModelRunValidator:
    store: TrainingArtifactStore

    def load_model_run(self, model_run_id: str) -> ModelRunArtifact:
        return self.store.load_model_run_for_audit(model_run_id)


class TrainingArtifactStore:
    """Persist datasets and model runs as content-addressed derived records."""

    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout.ensure()
        self.derived = DerivedArchive(self.layout)

    def write_dataset(self, dataset: TrainingDatasetManifest) -> Path:
        """Persist a dataset; schema v1 remains audit-only compatibility data."""

        verify_training_dataset(dataset)
        if dataset.schema_version == TRAINING_DATASET_SCHEMA_VERSION:
            self._validate_formal_dataset(dataset)
        elif dataset.schema_version != LEGACY_TRAINING_DATASET_SCHEMA_VERSION:
            raise TrainingArtifactConflict("unsupported training dataset schema")
        self._validate_capture_evidence(dataset)
        self._validate_sample_references(dataset)
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

    def write_formal_dataset(self, dataset: TrainingDatasetManifest) -> Path:
        if dataset.schema_version != TRAINING_DATASET_SCHEMA_VERSION:
            raise TrainingArtifactConflict(
                "legacy training samples are audit-only and cannot enter formal training"
            )
        return self.write_dataset(dataset)

    def write_training_dataset(self, dataset: TrainingDatasetManifest) -> Path:
        """Formal writer; legacy schema v1 is persistable only as audit data."""

        return self.write_formal_dataset(dataset)

    def create_dataset(self, **kwargs: Any) -> TrainingDatasetManifest:
        dataset = TrainingDatasetManifest.create_formal(**kwargs)
        self.write_formal_dataset(dataset)
        return dataset

    def create_training_qualification(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        qualification: Qualification | str,
        ruleset_version: str,
        evaluated_at: datetime,
        snapshot_ref: str | None,
        result_ref: str | None,
    ) -> TrainingQualification:
        """Recompute and persist one task-specific qualification from storage."""

        artifact = self._recompute_training_qualification(
            match_id=match_id,
            match_version=match_version,
            qualification=Qualification(qualification),
            ruleset_version=ruleset_version,
            evaluated_at=evaluated_at,
            snapshot_ref=snapshot_ref,
            result_ref=result_ref,
        )
        payload = artifact.to_payload()
        self._write_json(self.qualification_path(artifact.qualification_id), payload)
        self.derived.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="training-qualification",
                schema_version=artifact.schema_version,
                payload=payload,
                generated_at=artifact.evaluated_at,
                transform_version=artifact.contract_version,
                code_version=DERIVED_CODE_VERSION,
                input_refs=artifact.input_refs,
                output_refs=(artifact.qualification_id,),
                status="succeeded",
                quality="passed" if artifact.passed else "failed",
            )
        )
        return artifact

    def load_training_qualification(self, qualification_id: str) -> TrainingQualification:
        manifest = self._load_fixed_code_manifest_for_output_ref(qualification_id)
        payload = self._verify_json_artifact(
            qualification_id,
            "training-qualification:",
            "training-qualifications",
            "training-qualification",
            manifest=manifest,
        )
        try:
            artifact = parse_training_qualification_payload(payload)
            if (
                manifest.schema_version != artifact.schema_version
                or manifest.transform_version != artifact.contract_version
                or manifest.code_version != DERIVED_CODE_VERSION
                or manifest.generated_at != artifact.evaluated_at
                or manifest.started_at != artifact.evaluated_at
                or manifest.ended_at != artifact.evaluated_at
                or manifest.input_refs != artifact.input_refs
                or manifest.output_refs != (artifact.qualification_id,)
                or manifest.error is not None
                or manifest.quality != ("passed" if artifact.passed else "failed")
            ):
                raise TrainingArtifactConflict(
                    "training qualification manifest does not match its contract"
                )
            replayed = self._recompute_training_qualification(
                match_id=MatchId(artifact.match_id),
                match_version=artifact.match_version,
                qualification=artifact.qualification,
                ruleset_version=artifact.ruleset_version,
                evaluated_at=artifact.evaluated_at,
                snapshot_ref=artifact.snapshot_ref,
                result_ref=artifact.result_ref,
            )
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise TrainingArtifactConflict(
                f"training qualification replay failed: {error}"
            ) from error
        if artifact != replayed:
            raise TrainingArtifactConflict(
                "training qualification does not match persisted canonical evidence"
            )
        return artifact

    def build_formal_score_sample(
        self,
        *,
        sample_id: str,
        qualification_ref: str,
        feature_version: str | None = None,
        split: str,
        as_of: datetime | None = None,
    ) -> TrainingSample:
        """Construct a score sample without accepting caller-provided values or readiness."""

        if (
            feature_version is not None
            and feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION
        ):
            raise TrainingArtifactConflict(
                "formal score samples require the current feature projection version"
            )

        artifact = self.load_training_qualification(qualification_ref)
        if artifact.qualification is not Qualification.SCORE_MODEL:
            raise TrainingArtifactConflict("score samples require score-model-ready qualification")
        snapshot = (
            None
            if artifact.snapshot_ref is None
            else self._load_qualification_snapshot(artifact.snapshot_ref)
        )
        if artifact.result_ref is None:
            raise TrainingArtifactConflict("score samples require a bound result-90 fact")
        result, _ = self._load_result_reference_availability(artifact.result_ref)
        sample_as_of = snapshot.as_of if snapshot is not None else as_of
        if sample_as_of is None:
            raise TrainingArtifactConflict("excluded score samples without snapshots require as_of")
        require_utc(sample_as_of, "sample as_of")
        feature_known_at = (
            max(feature.known_at for feature in snapshot.features)
            if snapshot is not None and snapshot.features
            else sample_as_of
        )
        feature_refs = tuple(
            sorted(
                {
                    qualification_ref,
                    *(() if artifact.snapshot_ref is None else (artifact.snapshot_ref,)),
                }
            )
        )
        return TrainingSample(
            sample_id=sample_id,
            as_of=sample_as_of,
            feature_known_at=feature_known_at,
            label_known_at=result.known_at,
            capture_mode=artifact.capture_mode or CaptureMode.RECONSTRUCTED,
            qualification=artifact.qualification.value,
            qualification_passed=artifact.passed,
            feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
            label_version="result-90/1",
            feature_refs=feature_refs,
            label_ref=artifact.result_ref,
            features=score_feature_payload(snapshot),
            label={"home_goals": result.home_goals, "away_goals": result.away_goals},
            exclusion_reasons=artifact.reason_codes,
            split=split,
            match_id=artifact.match_id,
            match_version=artifact.match_version,
            snapshot_ref=artifact.snapshot_ref,
            qualification_ref=artifact.qualification_id,
        )

    def load_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        path = self.dataset_path(dataset_id)
        payload = self._read_json(path)
        dataset = parse_training_dataset_payload(payload)
        if dataset.dataset_id != dataset_id:
            raise TrainingArtifactConflict("training dataset path and ID disagree")
        self._verify_dataset_manifest(dataset)
        if dataset.schema_version == TRAINING_DATASET_SCHEMA_VERSION:
            self._validate_formal_dataset(dataset)
        self._validate_capture_evidence(dataset)
        self._validate_sample_references(dataset)
        return dataset

    def load_formal_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        dataset = self.load_dataset(dataset_id)
        if dataset.schema_version != TRAINING_DATASET_SCHEMA_VERSION:
            raise TrainingArtifactConflict("legacy training dataset is audit-only")
        return dataset

    def load_training_dataset(self, dataset_id: str) -> TrainingDatasetManifest:
        return self.load_dataset(dataset_id)

    def write_model_run(self, artifact: ModelRunArtifact) -> Path:
        return self._write_model_run(artifact, require_formal=True)

    def write_model_run_for_audit(self, artifact: ModelRunArtifact) -> Path:
        """Persist a legacy-backed run that formal prediction consumers reject."""

        return self._write_model_run(artifact, require_formal=False)

    def _write_model_run(self, artifact: ModelRunArtifact, *, require_formal: bool) -> Path:
        if (
            require_formal
            and artifact.status is not ModelRunStatus.FAILED
            and artifact.feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION
        ):
            raise TrainingArtifactConflict(
                "formal model runs require the current score feature projection"
            )
        dataset = self._load_dataset_for_run(artifact, require_formal=require_formal)
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
        return self._load_model_run(model_run_id, require_formal=True)

    def load_model_run_for_audit(self, model_run_id: str) -> ModelRunArtifact:
        return self._load_model_run(model_run_id, require_formal=False)

    def _load_model_run(self, model_run_id: str, *, require_formal: bool) -> ModelRunArtifact:
        path = self.model_run_path(model_run_id)
        payload = self._read_json(path)
        artifact = parse_model_run_payload(payload)
        if artifact.model_run_id != model_run_id:
            raise TrainingArtifactConflict("model run path and ID disagree")
        if (
            require_formal
            and artifact.status is not ModelRunStatus.FAILED
            and artifact.feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION
        ):
            raise TrainingArtifactConflict(
                "formal model runs require the current score feature projection"
            )
        self._verify_model_run_manifest(artifact)
        dataset = self._load_dataset_for_run(artifact, require_formal=require_formal)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)
        return artifact

    def load_model_run_artifact(self, model_run_id: str) -> ModelRunArtifact:
        return self.load_model_run(model_run_id)

    def verify_model_run(self, artifact: ModelRunArtifact) -> None:
        if (
            artifact.status is not ModelRunStatus.FAILED
            and artifact.feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION
        ):
            raise TrainingArtifactConflict(
                "formal model runs require the current score feature projection"
            )
        dataset = self._load_dataset_for_run(artifact, require_formal=True)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)

    def verify_model_run_for_audit(self, artifact: ModelRunArtifact) -> None:
        dataset = self._load_dataset_for_run(artifact, require_formal=False)
        verify_model_run_artifact(artifact, dataset=dataset)
        self._validate_model_bytes(artifact)

    def load_verified_prediction(self, reference: str) -> ScorePrediction:
        """Load a prediction by replaying its snapshot, model, and composition contracts."""

        prediction, _ = self.load_verified_prediction_context(reference)
        return prediction

    def load_verified_prediction_context(
        self, reference: str
    ) -> tuple[ScorePrediction, PreMatchSnapshot]:
        """Replay once and return the formal prediction with its persisted snapshot."""

        return self._load_prediction_replay(reference, require_current=True)

    def load_prediction_for_audit(self, reference: str) -> ScorePrediction:
        """Replay a legacy prediction without admitting it to formal consumers."""

        prediction, _ = self._load_prediction_replay(reference, require_current=False)
        return prediction

    def _load_prediction_replay(
        self,
        reference: str,
        *,
        require_current: bool,
        verification_session: VerificationSession | None = None,
    ) -> tuple[ScorePrediction, PreMatchSnapshot]:
        if require_current and verification_session is None:
            from football_data_platform.storage.verification import (
                VerificationSession,
                active_verification_session,
                verification_session_scope,
            )

            try:
                active_session = active_verification_session(self.layout)
                if active_session is not None:
                    return self._load_prediction_replay(
                        reference,
                        require_current=True,
                        verification_session=active_session,
                    )
                with VerificationSession(self.layout) as session:
                    with verification_session_scope(session):
                        return self._load_prediction_replay(
                            reference,
                            require_current=True,
                            verification_session=session,
                        )
            except TrainingArtifactConflict:
                raise
            except ArchiveConflictError as error:
                raise TrainingArtifactConflict(
                    f"prediction domain replay is unavailable or invalid: {error}"
                ) from error
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise TrainingArtifactConflict(
                    f"prediction domain replay is unavailable or invalid: {error}"
                ) from error
        replay_archive = (
            self.derived
            if require_current
            else DerivedArchive(
                self.layout,
                model_run_validator=_AuditModelRunValidator(self),
            )
        )
        snapshot_validator = self.derived
        if require_current:
            assert verification_session is not None
            # This slice proves directly read prediction/snapshot/composition/source bytes.
            # Canonical contracts, model/dataset replay, and deeper match-context, official, and
            # result replay still use their existing verifiers without a shared session.
            snapshot_validator = _SessionSnapshotSourceValidator(
                self.derived,
                verification_session,
            )

        try:
            prediction_manifest = (
                self._load_fixed_code_manifest_for_output_ref(
                    reference, verification_session=verification_session
                )
                if require_current
                else replay_archive._load_artifact_manifest_for_output_ref(reference)
            )
            payload = self._verify_json_artifact(
                reference,
                "prediction:",
                "predictions",
                "prediction",
                manifest_archive=replay_archive,
                manifest=prediction_manifest,
                verification_session=verification_session,
            )
            snapshot_ref = payload.get("snapshot_id")
            if not isinstance(snapshot_ref, str):
                raise TrainingArtifactConflict("prediction lacks a snapshot reference")
            snapshot_manifest = (
                self._load_fixed_code_manifest_for_output_ref(
                    snapshot_ref, verification_session=verification_session
                )
                if require_current
                else replay_archive._load_artifact_manifest_for_output_ref(snapshot_ref)
            )
            snapshot_document = self._verify_json_artifact(
                snapshot_ref,
                "snapshot:",
                "snapshots",
                "prematch-snapshot",
                manifest_archive=replay_archive,
                manifest=snapshot_manifest,
                verification_session=verification_session,
            )
            snapshot = parse_snapshot_payload(
                snapshot_document,
                source_validator=snapshot_validator,
            )
            if require_current and snapshot.feature_spec_version != CURRENT_FEATURE_SPEC_VERSION:
                raise TrainingArtifactConflict(
                    "formal prediction requires prematch-features/3; legacy snapshot is audit-only"
                )
            if (
                snapshot_manifest.schema_version != snapshot.schema_version
                or snapshot_manifest.transform_version != snapshot.feature_spec_version
                or (require_current and snapshot_manifest.code_version != DERIVED_CODE_VERSION)
                or snapshot_manifest.generated_at != snapshot.observed_at
                or snapshot_manifest.started_at != snapshot.observed_at
                or snapshot_manifest.ended_at != snapshot.observed_at
                or snapshot_manifest.input_refs != snapshot.input_refs
                or snapshot_manifest.output_refs != (snapshot.id.value,)
                or snapshot_manifest.error is not None
                or snapshot_manifest.quality != snapshot.quality_status
            ):
                raise TrainingArtifactConflict(
                    "prediction snapshot manifest does not match persisted bytes"
                )
            prediction = parse_prediction_payload(
                payload,
                snapshot=snapshot,
                snapshot_validator=snapshot_validator,
                model_run_validator=(self if require_current else _AuditModelRunValidator(self)),
            )
            if require_current:
                verify_prediction_snapshot(
                    prediction,
                    snapshot=snapshot,
                    snapshot_validator=snapshot_validator,
                    model_run_validator=self,
                )
            expected_inputs = tuple(
                sorted({*prediction.input_refs, prediction.composition_artifact_ref})
            )
            if (
                prediction_manifest.schema_version != prediction.schema_version
                or prediction_manifest.transform_version != prediction.model_version
                or (require_current and prediction_manifest.code_version != DERIVED_CODE_VERSION)
                or prediction_manifest.generated_at != prediction.generated_at
                or prediction_manifest.started_at != prediction.generated_at
                or prediction_manifest.ended_at != prediction.generated_at
                or prediction_manifest.input_refs != expected_inputs
                or prediction_manifest.output_refs != (prediction.id.value,)
                or prediction_manifest.error is not None
                or prediction_manifest.quality != prediction.snapshot_quality_status
            ):
                raise TrainingArtifactConflict(
                    "prediction manifest does not match its domain contract"
                )
            replay_archive.verify_score_grid_composition(
                prediction,
                verification_session=verification_session,
            )
        except TrainingArtifactConflict as error:
            if _exception_chain_contains(error, "audit-only") or _exception_chain_contains(
                error, "current score feature projection"
            ):
                raise TrainingArtifactConflict("prediction model lineage is audit-only") from error
            raise
        except ArchiveConflictError as error:
            if _exception_chain_contains(error, "audit-only") or _exception_chain_contains(
                error, "current score feature projection"
            ):
                raise TrainingArtifactConflict("prediction model lineage is audit-only") from error
            raise TrainingArtifactConflict(
                f"prediction domain replay is unavailable or invalid: {error}"
            ) from error
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise TrainingArtifactConflict(
                f"prediction domain replay is unavailable or invalid: {error}"
            ) from error
        return prediction, snapshot

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

    def qualification_path(self, qualification_id: str) -> Path:
        digest = _digest_id(qualification_id, "training-qualification:")
        return self.layout.derived / "training-qualifications" / digest[:2] / f"{digest}.json"

    def model_run_path(self, model_run_id: str) -> Path:
        digest = _digest_id(model_run_id, "model-run:")
        return self.layout.derived / "model-runs" / digest[:2] / f"{digest}.json"

    def _load_dataset_for_run(
        self,
        artifact: ModelRunArtifact,
        *,
        require_formal: bool,
    ) -> TrainingDatasetManifest | None:
        if not self.dataset_path(artifact.dataset_id).exists():
            if artifact.status is ModelRunStatus.FAILED:
                return None
            raise TrainingArtifactConflict(
                "successful model runs require a persisted training dataset"
            )
        if require_formal and artifact.status is not ModelRunStatus.FAILED:
            return self.load_formal_dataset(artifact.dataset_id)
        return self.load_dataset(artifact.dataset_id)

    def _validate_formal_dataset(self, dataset: TrainingDatasetManifest) -> None:
        if dataset.task != "score-model" or dataset.label_version != "result-90/1":
            raise TrainingArtifactConflict(
                "training-dataset/2 currently admits only typed score-model samples"
            )
        if dataset.feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION:
            raise TrainingArtifactConflict(
                "formal training dataset requires the current score feature projection"
            )
        for sample in dataset.samples:
            if sample.feature_version != CURRENT_SCORE_FEATURE_PROJECTION_VERSION:
                raise TrainingArtifactConflict(
                    f"formal training sample {sample.sample_id} uses a legacy feature projection"
                )
            if sample.qualification_ref is None:
                raise TrainingArtifactConflict("formal sample lacks qualification_ref")
            expected = self.build_formal_score_sample(
                sample_id=sample.sample_id,
                qualification_ref=sample.qualification_ref,
                feature_version=sample.feature_version,
                split=sample.split,
                as_of=sample.as_of,
            )
            if sample != expected:
                raise TrainingArtifactConflict(
                    f"formal training sample {sample.sample_id} does not match replayed evidence"
                )
            qualification = self.load_training_qualification(sample.qualification_ref)
            if qualification.ruleset_version != dataset.qualification_ruleset_version:
                raise TrainingArtifactConflict(
                    f"formal training sample {sample.sample_id} ruleset mismatch"
                )
            if qualification.evaluated_at > dataset.generated_at:
                raise TrainingArtifactConflict(
                    f"formal training sample {sample.sample_id} predates qualification"
                )

    def _verify_dataset_manifest(self, dataset: TrainingDatasetManifest) -> None:
        manifest = self._load_unique_manifest_for_output_ref(dataset.dataset_id)
        expected_error = (
            dataset.error
            if dataset.status in {DatasetStatus.FAILED, DatasetStatus.PARTIAL}
            else None
        )
        if (
            manifest.artifact_type != "training-dataset"
            or manifest.schema_version != dataset.schema_version
            or manifest.payload != dataset.to_payload()
            or manifest.generated_at != dataset.generated_at
            or manifest.started_at != dataset.generated_at
            or manifest.ended_at != dataset.generated_at
            or manifest.transform_version != dataset.transform_version
            or manifest.code_version != (dataset.code_version or DERIVED_CODE_VERSION)
            or manifest.input_refs != dataset.input_refs
            or manifest.output_refs != (dataset.dataset_id,)
            or manifest.status != _manifest_status(dataset.status)
            or manifest.error != expected_error
            or manifest.quality != dataset.status.value
        ):
            raise TrainingArtifactConflict(
                "training dataset manifest does not match persisted dataset bytes"
            )

    def _verify_model_run_manifest(self, artifact: ModelRunArtifact) -> None:
        manifest = self._load_unique_manifest_for_output_ref(artifact.model_run_id)
        expected_inputs = tuple(
            sorted(
                {
                    artifact.dataset_id,
                    *artifact.model_artifact_refs,
                    *artifact.evaluation_cohort,
                }
            )
        )
        expected_outputs = tuple(sorted({artifact.model_run_id, *artifact.output_refs}))
        expected_error = (
            artifact.error
            if artifact.status in {ModelRunStatus.FAILED, ModelRunStatus.PARTIAL}
            else None
        )
        if (
            manifest.artifact_type != "model-run"
            or manifest.schema_version != artifact.schema_version
            or manifest.payload != artifact.to_payload()
            or manifest.generated_at != artifact.ended_at
            or manifest.started_at != artifact.ended_at
            or manifest.ended_at != artifact.ended_at
            or manifest.transform_version != artifact.model_version
            or manifest.code_version != (artifact.code_version or DERIVED_CODE_VERSION)
            or manifest.input_refs != expected_inputs
            or manifest.output_refs != expected_outputs
            or manifest.status != _manifest_status(artifact.status)
            or manifest.error != expected_error
            or manifest.quality != artifact.status.value
        ):
            raise TrainingArtifactConflict(
                "model run manifest does not match persisted model-run bytes"
            )

    def _load_unique_manifest_for_output_ref(
        self,
        output_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> DerivedArtifactManifest:
        # Build one request-local, fail-closed catalog.  The normal archive loader remains the
        # authority for typed manifest and lineage validation after the catalog selects a path.
        if verification_session is None:
            from football_data_platform.storage.verification import (
                VerificationSession,
                active_verification_session,
                verification_session_scope,
            )

            try:
                active_session = active_verification_session(self.layout)
                if active_session is not None:
                    return self._load_unique_manifest_for_output_ref(
                        output_ref,
                        verification_session=active_session,
                    )
                with VerificationSession(self.layout) as session:
                    with verification_session_scope(session):
                        return self._load_unique_manifest_for_output_ref(
                            output_ref,
                            verification_session=session,
                        )
            except TrainingArtifactConflict:
                raise
            except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
                if "requires exactly one derived artifact manifest" in str(error):
                    raise TrainingArtifactConflict(str(error)) from error
                raise TrainingArtifactConflict(
                    f"{output_ref} derived artifact manifest is unavailable or invalid"
                ) from error
        try:
            entry = verification_session.require_unique_manifest(output_ref)
            manifest = self.derived.load_artifact_manifest(
                entry.artifact_id,
                verification_session=verification_session,
            )
            entry.proof.verify()
            return manifest
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            if "requires exactly one derived artifact manifest" in str(error):
                raise TrainingArtifactConflict(str(error)) from error
            raise TrainingArtifactConflict(
                f"{output_ref} derived artifact manifest is unavailable or invalid"
            ) from error

    def _load_fixed_code_manifest_for_output_ref(
        self,
        output_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> DerivedArtifactManifest:
        manifest = self._load_unique_manifest_for_output_ref(
            output_ref, verification_session=verification_session
        )
        if manifest.code_version != DERIVED_CODE_VERSION:
            raise TrainingArtifactConflict(
                f"{output_ref} manifest does not use the fixed producer code_version"
            )
        return manifest

    def _recompute_training_qualification(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        qualification: Qualification,
        ruleset_version: str,
        evaluated_at: datetime,
        snapshot_ref: str | None,
        result_ref: str | None,
    ) -> TrainingQualification:
        if not isinstance(match_id, MatchId):
            raise TypeError("match_id must be a MatchId")
        if (
            not isinstance(match_version, int)
            or isinstance(match_version, bool)
            or match_version < 1
        ):
            raise ValueError("match_version must be a positive integer")
        require_utc(evaluated_at, "evaluated_at")
        snapshot = None if snapshot_ref is None else self._load_qualification_snapshot(snapshot_ref)
        if snapshot is not None and (
            snapshot.match_id != match_id or snapshot.match_version != match_version
        ):
            raise TrainingArtifactConflict(
                "qualification snapshot belongs to a different match or version"
            )
        if snapshot is not None and snapshot.capture_mode is CaptureMode.CAPTURED:
            raise TrainingArtifactConflict(
                "captured qualification requires an authoritative trusted capture-run validator"
            )

        canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
        fact_refs: tuple[str, ...] = ()
        if result_ref is not None:
            result, _ = self._load_result_reference_availability(
                result_ref,
                archive=RawArchive(self.layout),
                canonical=canonical,
            )
            with canonical.connect() as connection:
                row = connection.execute(
                    "SELECT match_id, match_version FROM match_results_90 WHERE record_id = ?",
                    (result_ref,),
                ).fetchone()
                latest = connection.execute(
                    "SELECT record_id FROM match_results_90 WHERE match_id = ? "
                    "AND match_version = ? AND known_at <= ? AND observed_at <= ? "
                    "ORDER BY observation_version DESC LIMIT 1",
                    (
                        match_id.value,
                        match_version,
                        _timestamp(evaluated_at),
                        _timestamp(evaluated_at),
                    ),
                ).fetchone()
            if (
                row is None
                or latest is None
                or latest["record_id"] != result_ref
                or row["match_id"] != match_id.value
                or int(row["match_version"]) != match_version
                or result.match_id != match_id
            ):
                raise TrainingArtifactConflict(
                    "qualification result belongs to a different match or version"
                )
            fact_refs = (result_ref,)

        fact_store = CanonicalFactStore(canonical)
        availability = fact_store.availability(
            match_id,
            snapshots=(() if snapshot is None else (SnapshotAvailability.from_snapshot(snapshot),)),
            as_of=evaluated_at,
        )
        if availability.match_version != match_version:
            raise TrainingArtifactConflict(
                "qualification match version is not authoritative at evaluated_at"
            )
        assessment = assess_lifecycle(
            availability,
            evaluated_at=evaluated_at,
            ruleset_version=ruleset_version,
            snapshot_validator=self.derived,
        )
        result = next(
            item for item in assessment.qualifications if item.qualification is qualification
        )
        if qualification is Qualification.TEAM_BASELINE:
            selected_team_refs = {
                availability.team_stat_refs[team_id]
                for team_id in availability.team_ids
                if team_id in availability.team_stat_refs
            }
            fact_refs = tuple(sorted({*fact_refs, *selected_team_refs}))
        reasons = list(result.reason_codes)
        if qualification in {Qualification.SCORE_MODEL, Qualification.TEAM_BASELINE} and (
            result_ref is None
        ):
            reasons.append("missing_bound_result_90")
        if qualification is Qualification.SCORE_MODEL and snapshot_ref is None:
            reasons.append("missing_bound_prematch_snapshot")
        if qualification is Qualification.PLAYER_PROFILE:
            reasons.append("typed_player_fact_replay_unavailable")
        normalized_reasons = tuple(sorted(set(reasons)))
        replayed_result = QualificationResult(
            qualification=qualification,
            ruleset_version=ruleset_version,
            passed=not normalized_reasons,
            reason_codes=normalized_reasons,
            evaluated_at=evaluated_at,
        )
        return TrainingQualification.create(
            match_id=match_id.value,
            match_version=match_version,
            result=replayed_result,
            snapshot_ref=snapshot_ref,
            result_ref=result_ref,
            fact_refs=fact_refs,
            capture_mode=None if snapshot is None else snapshot.capture_mode,
        )

    def _load_qualification_snapshot(self, snapshot_ref: str) -> Any:
        manifest = self._load_fixed_code_manifest_for_output_ref(snapshot_ref)
        payload = self._verify_json_artifact(
            snapshot_ref,
            "snapshot:",
            "snapshots",
            "prematch-snapshot",
            manifest=manifest,
        )
        raw_capture_mode = payload.get("capture_mode")
        if raw_capture_mode == CaptureMode.CAPTURED.value:
            raise TrainingArtifactConflict(
                "captured qualification requires an authoritative trusted capture-run validator"
            )
        snapshot = parse_snapshot_payload(payload, source_validator=self.derived)
        if snapshot.feature_spec_version != CURRENT_FEATURE_SPEC_VERSION:
            raise TrainingArtifactConflict(
                "formal qualification requires the current snapshot feature specification"
            )
        if (
            manifest.schema_version != snapshot.schema_version
            or manifest.transform_version != snapshot.feature_spec_version
            or manifest.code_version != DERIVED_CODE_VERSION
            or manifest.generated_at != snapshot.observed_at
            or manifest.started_at != snapshot.observed_at
            or manifest.ended_at != snapshot.observed_at
            or manifest.input_refs != snapshot.input_refs
            or manifest.output_refs != (snapshot.id.value,)
            or manifest.error is not None
            or manifest.quality != snapshot.quality_status
        ):
            raise TrainingArtifactConflict(
                "qualification snapshot manifest does not match persisted bytes"
            )
        return snapshot

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

        availabilities: dict[str, _TrainingReferenceAvailability] = {}
        for reference in dataset.input_refs:
            try:
                availabilities[reference] = self._verify_training_reference(reference)
            except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                label_sample = next(
                    (sample for sample in dataset.samples if sample.label_ref == reference),
                    None,
                )
                if (
                    label_sample is not None
                    and dataset.task == "score-model"
                    and dataset.label_version == "result-90/1"
                ):
                    raise TrainingArtifactConflict(
                        "training sample reference is unavailable or invalid: "
                        f"{reference}; score-model sample {label_sample.sample_id} "
                        "result lineage is invalid"
                    ) from error
                raise TrainingArtifactConflict(
                    f"training sample reference is unavailable or invalid: {reference}"
                ) from error
        if dataset.status in {DatasetStatus.SUCCEEDED, DatasetStatus.PARTIAL}:
            for reference, availability in availabilities.items():
                if availability.available_at is None:
                    raise TrainingArtifactConflict(
                        f"training reference lacks an authoritative availability time: {reference}"
                    )
                if availability.available_at > dataset.generated_at:
                    raise TrainingArtifactConflict(
                        f"training dataset predates input availability: {reference}"
                    )
            for sample in dataset.samples:
                for reference in sample.feature_refs:
                    availability = availabilities[reference]
                    if (
                        sample.capture_mode is CaptureMode.CAPTURED
                        and availability.available_at is not None
                        and availability.available_at > sample.as_of
                    ):
                        raise TrainingArtifactConflict(
                            f"captured training sample {sample.sample_id} feature input "
                            f"follows as_of: {reference}"
                        )
                    semantic_known_at = availability.semantic_known_at
                    if (
                        semantic_known_at is not None
                        and semantic_known_at > sample.feature_known_at
                    ):
                        raise TrainingArtifactConflict(
                            f"training sample {sample.sample_id} feature_known_at predates "
                            f"its source semantics: {reference}"
                        )
        self._validate_score_result_labels(dataset)

    def _validate_score_result_labels(self, dataset: TrainingDatasetManifest) -> None:
        if dataset.task != "score-model" or dataset.label_version != "result-90/1":
            return

        canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
        archive = RawArchive(self.layout)
        for sample in dataset.samples:
            try:
                result, _ = self._load_result_reference_availability(
                    sample.label_ref, archive=archive, canonical=canonical
                )
            except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
                raise TrainingArtifactConflict(
                    f"score-model sample {sample.sample_id} result lineage is invalid"
                ) from error
            if not isinstance(sample.label, dict):
                raise TrainingArtifactConflict(
                    f"score-model sample {sample.sample_id} label must be an object"
                )
            home_goals = sample.label.get("home_goals")
            away_goals = sample.label.get("away_goals")
            if (
                type(home_goals) is not int
                or type(away_goals) is not int
                or home_goals != result.home_goals
                or away_goals != result.away_goals
            ):
                raise TrainingArtifactConflict(
                    f"score-model sample {sample.sample_id} label goals do not match its result"
                )
            if sample.label_known_at != result.known_at:
                raise TrainingArtifactConflict(
                    f"score-model sample {sample.sample_id} label_known_at "
                    "does not match its result"
                )

    def _verify_training_reference(
        self,
        reference: str,
        *,
        _manifest_lineage: frozenset[str] = frozenset(),
        verification_session: VerificationSession | None = None,
    ) -> _TrainingReferenceAvailability:
        if verification_session is None:
            from football_data_platform.storage.verification import active_verification_session

            verification_session = active_verification_session(self.layout)
        if not isinstance(reference, str) or not reference or reference.strip() != reference:
            raise ValueError("reference must be non-empty text")
        if reference.startswith("raw-asset:"):
            raw = RawArchive(self.layout)
            asset = raw.load(RawAssetId(reference))
            raw.verify(asset)
            return _TrainingReferenceAvailability(reference, asset.observed_at)
        if reference.startswith("derived-source:"):
            _require_digest_reference(reference, "derived-source:")
            source = self.derived.validate_snapshot_source(
                reference,
                verification_session=verification_session,
            )
            return _TrainingReferenceAvailability(reference, source.observed_at, source.known_at)
        if reference.startswith("team-baseline:"):
            baseline = self.derived.load_team_baseline(reference)
            manifest = self._verify_manifest_output(reference, "team-baseline")
            return _TrainingReferenceAvailability(reference, manifest.generated_at, baseline.as_of)
        if reference.startswith("snapshot:"):
            payload = self._verify_json_artifact(
                reference, "snapshot:", "snapshots", "prematch-snapshot"
            )
            snapshot = parse_snapshot_payload(payload, source_validator=self.derived)
            manifest = self.derived._load_artifact_manifest_for_output_ref(reference)
            semantic_known_at = max(feature.known_at for feature in snapshot.features)
            return _TrainingReferenceAvailability(
                reference, manifest.generated_at, semantic_known_at
            )
        if reference.startswith("prediction:"):
            prediction = self.load_verified_prediction(reference)
            return _TrainingReferenceAvailability(
                reference, prediction.generated_at, prediction.generated_at
            )
        if reference.startswith("score-grid-composition:"):
            self.derived.load_score_grid_composition_payload(reference)
            manifest = self._verify_manifest_output(reference, "score-grid-composition")
            return _TrainingReferenceAvailability(reference, manifest.generated_at)
        if reference.startswith("derived-artifact:"):
            return self._verify_derived_artifact_reference(
                reference,
                lineage=_manifest_lineage,
                verification_session=verification_session,
            )
        if reference.startswith("training-qualification:"):
            qualification = self.load_training_qualification(reference)
            return _TrainingReferenceAvailability(reference, qualification.evaluated_at)
        if reference.startswith("training-dataset:"):
            dataset = self.load_dataset(reference)
            return _TrainingReferenceAvailability(reference, dataset.generated_at)
        if reference.startswith("model-run:"):
            model_run = self.load_model_run(reference)
            return _TrainingReferenceAvailability(reference, model_run.ended_at)
        if reference.startswith("model-artifact:"):
            self.verify_model_artifact(reference)
            return _TrainingReferenceAvailability(reference, None)
        if reference.startswith("model-output:"):
            self.verify_model_output(reference)
            return _TrainingReferenceAvailability(reference, None)
        if reference.startswith("fact:") or reference.startswith("canonical:"):
            if reference.startswith("fact:match_results_90:"):
                _, availability = self._load_result_reference_availability(reference)
                return availability
            if reference.startswith("fact:team_match_observations:"):
                observation = load_verified_team_observation(
                    reference,
                    archive=RawArchive(self.layout),
                    canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
                    verification_session=verification_session,
                )
                return _TrainingReferenceAvailability(
                    reference,
                    observation.observed_at,
                    observation.known_at,
                )
            self._verify_canonical_reference(reference)
            return _TrainingReferenceAvailability(reference, None)
        raise ValueError(f"unsupported training reference type: {reference}")

    def _verify_derived_artifact_reference(
        self,
        reference: str,
        *,
        lineage: frozenset[str],
        verification_session: VerificationSession | None = None,
    ) -> _TrainingReferenceAvailability:
        if reference in lineage:
            raise TrainingArtifactConflict(
                f"recursive derived artifact training lineage: {reference}"
            )
        manifest = self.derived.load_artifact_manifest(
            reference,
            verification_session=verification_session,
        )
        if manifest.status not in {"succeeded", "partial"}:
            raise TrainingArtifactConflict(
                f"training input derived artifact is not usable: {reference}"
            )

        nested_lineage = lineage | {reference}
        for input_reference in manifest.input_refs:
            availability = self._verify_training_reference(
                input_reference,
                _manifest_lineage=nested_lineage,
                verification_session=verification_session,
            )
            if availability.available_at is None:
                raise TrainingArtifactConflict(
                    "derived artifact input lacks an authoritative availability time: "
                    f"{input_reference}"
                )
            if availability.available_at > manifest.generated_at:
                raise TrainingArtifactConflict(
                    "derived artifact generated_at predates input availability: "
                    f"{reference} <- {input_reference}"
                )
        return _TrainingReferenceAvailability(reference, manifest.generated_at)

    def _verify_manifest_output(
        self, reference: str, artifact_type: str
    ) -> DerivedArtifactManifest:
        manifest = self.derived._load_artifact_manifest_for_output_ref(reference)
        if manifest.artifact_type != artifact_type or manifest.status != "succeeded":
            raise TrainingArtifactConflict(
                f"reference {reference} does not resolve to a succeeded {artifact_type} artifact"
            )
        return manifest

    def _load_result_reference_availability(
        self,
        reference: str,
        *,
        archive: RawArchive | None = None,
        canonical: CanonicalStore | None = None,
    ) -> tuple[Any, _TrainingReferenceAvailability]:
        raw = archive or RawArchive(self.layout)
        facts = canonical or CanonicalStore(self.layout.canonical / "platform.sqlite3")
        result = load_verified_match_result(reference, archive=raw, canonical=facts)
        with facts.connect() as connection:
            row = connection.execute(
                "SELECT raw_asset_id FROM match_results_90 WHERE record_id = ?",
                (reference,),
            ).fetchone()
        if row is None:
            raise TrainingArtifactConflict("canonical result availability is unavailable")
        asset = raw.load(RawAssetId(row["raw_asset_id"]))
        raw.verify(asset)
        return result, _TrainingReferenceAvailability(reference, asset.observed_at, result.known_at)

    def _verify_json_artifact(
        self,
        reference: str,
        prefix: str,
        directory: str,
        artifact_type: str,
        *,
        manifest_archive: DerivedArchive | None = None,
        manifest: DerivedArtifactManifest | None = None,
        verification_session: VerificationSession | None = None,
    ) -> dict[str, Any]:
        digest = _digest_id(reference, prefix)
        path = self.layout.derived / directory / digest[:2] / f"{digest}.json"
        if verification_session is not None:
            verification_session.file_proof(path)
        payload = self._read_json(path)
        if payload.get("id") != reference:
            raise TrainingArtifactConflict(f"{reference} ID does not match stored bytes")
        identity = dict(payload)
        identity.pop("id", None)
        if hashlib.sha256(_canonical_json(identity)).hexdigest() != digest:
            raise TrainingArtifactConflict(f"{reference} content hash does not match its ID")
        if manifest is None:
            manifest = (manifest_archive or self.derived)._load_artifact_manifest_for_output_ref(
                reference,
                verification_session=verification_session,
            )
        if (
            manifest.artifact_type != artifact_type
            or manifest.status != "succeeded"
            or manifest.payload != payload
        ):
            raise TrainingArtifactConflict(f"{reference} manifest does not match stored bytes")
        return payload

    def _verify_prediction_reference(self, reference: str) -> None:
        self.load_verified_prediction(reference)

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
        match_id=None if payload.get("match_id") is None else str(payload["match_id"]),
        match_version=(
            None
            if payload.get("match_version") is None
            else _strict_int(payload["match_version"], "match_version")
        ),
        snapshot_ref=(
            None if payload.get("snapshot_ref") is None else str(payload["snapshot_ref"])
        ),
        qualification_ref=(
            None if payload.get("qualification_ref") is None else str(payload["qualification_ref"])
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


def _timestamp(value: datetime) -> str:
    require_utc(value, "timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _exception_chain_contains(error: BaseException, text: str) -> bool:
    current: BaseException | None = error
    while current is not None:
        if text in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


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
