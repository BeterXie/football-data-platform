"""Immutable persistence and reference checks for model governance records.

Governance objects are derived artifacts, not mutable configuration rows.  This
adapter deliberately composes the existing raw, derived, and training stores;
it does not maintain a second model registry.  A record can therefore only be
written when every declared model/evaluation/cohort/sample/rollback reference
resolves to bytes or to a verified immutable manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import RawAssetId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.evaluation.governance import (
    ChallengerEvidence,
    PromotionDecision,
    PromotionPolicy,
    SubgroupDiagnostic,
    assess_promotion,
)
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import TrainingArtifactStore


class GovernanceArtifactConflict(ArchiveConflictError):
    """Raised when a governance artifact or one of its references is invalid."""


class GovernanceArtifactStore:
    """Persist and verify promotion policy, evidence, and decision artifacts."""

    def __init__(
        self,
        layout: DataLayout,
        *,
        reference_validator: Any | None = None,
        derived_archive: DerivedArchive | None = None,
        training_store: TrainingArtifactStore | None = None,
    ) -> None:
        self.layout = layout.ensure()
        self.derived = derived_archive or DerivedArchive(self.layout)
        self.training = training_store or TrainingArtifactStore(self.layout)
        self.raw = RawArchive(self.layout)
        self.reference_validator = reference_validator

    # ---- policy ---------------------------------------------------------
    def write_policy(self, policy: PromotionPolicy) -> Path:
        if not isinstance(policy, PromotionPolicy):
            raise TypeError("policy must be a PromotionPolicy")
        if policy.confidence_level is None:
            raise GovernanceArtifactConflict("persisted promotion policy requires confidence_level")
        self.verify_reference(policy.rollback_artifact_ref or policy.rollback_target or "")
        path = self._write_json(self.policy_path(policy.content_id), policy.to_payload())
        self._write_manifest(
            artifact_type="promotion-policy",
            artifact_id=policy.content_id,
            payload=policy.to_payload(),
            generated_at=policy.reviewed_at,
            input_refs=(policy.rollback_artifact_ref or policy.rollback_target or "",),
            quality="ready",
        )
        return path

    def write_promotion_policy(self, policy: PromotionPolicy) -> Path:
        return self.write_policy(policy)

    def load_policy(self, policy_id: str) -> PromotionPolicy:
        payload = self._read_json(self.policy_path(policy_id), "promotion policy")
        policy = parse_promotion_policy_payload(payload)
        if policy.content_id != policy_id:
            raise GovernanceArtifactConflict("promotion policy path and ID disagree")
        if policy.confidence_level is None:
            raise GovernanceArtifactConflict("persisted promotion policy requires confidence_level")
        self.verify_reference(policy.rollback_artifact_ref or policy.rollback_target or "")
        return policy

    def load_promotion_policy(self, policy_id: str) -> PromotionPolicy:
        return self.load_policy(policy_id)

    # ---- evidence -------------------------------------------------------
    def write_evidence(self, evidence: ChallengerEvidence) -> Path:
        if not isinstance(evidence, ChallengerEvidence):
            raise TypeError("evidence must be ChallengerEvidence")
        if evidence.evaluated_at is None:
            raise GovernanceArtifactConflict("persisted challenger evidence requires evaluated_at")
        self._verify_evidence_references(evidence)
        path = self._write_json(self.evidence_path(evidence.content_id), evidence.to_payload())
        self._write_manifest(
            artifact_type="challenger-evidence",
            artifact_id=evidence.content_id,
            payload=evidence.to_payload(),
            generated_at=evidence.evaluated_at,
            input_refs=_evidence_refs(evidence),
            quality="ready" if evidence.is_prospective_captured else "shadow",
        )
        return path

    def write_challenger_evidence(self, evidence: ChallengerEvidence) -> Path:
        return self.write_evidence(evidence)

    def load_evidence(self, evidence_id: str) -> ChallengerEvidence:
        payload = self._read_json(self.evidence_path(evidence_id), "challenger evidence")
        evidence = parse_challenger_evidence_payload(payload)
        if evidence.content_id != evidence_id:
            raise GovernanceArtifactConflict("challenger evidence path and ID disagree")
        self._verify_evidence_references(evidence)
        return evidence

    def load_challenger_evidence(self, evidence_id: str) -> ChallengerEvidence:
        return self.load_evidence(evidence_id)

    # ---- decisions ------------------------------------------------------
    def write_decision(self, decision: PromotionDecision) -> Path:
        if not isinstance(decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        if decision.decided_at is None:
            raise GovernanceArtifactConflict("persisted promotion decision requires decided_at")
        self._verify_decision_references(decision)
        path = self._write_json(self.decision_path(decision.content_id), decision.to_payload())
        self._write_manifest(
            artifact_type="promotion-decision",
            artifact_id=decision.content_id,
            payload=decision.to_payload(),
            generated_at=decision.decided_at,
            input_refs=_decision_refs(decision),
            quality="ready" if decision.promoted else "blocked",
        )
        return path

    def write_promotion_decision(self, decision: PromotionDecision) -> Path:
        return self.write_decision(decision)

    def load_decision(self, decision_id: str) -> PromotionDecision:
        payload = self._read_json(self.decision_path(decision_id), "promotion decision")
        decision = parse_promotion_decision_payload(payload)
        if decision.content_id != decision_id:
            raise GovernanceArtifactConflict("promotion decision path and ID disagree")
        self._verify_decision_references(decision)
        return decision

    def load_promotion_decision(self, decision_id: str) -> PromotionDecision:
        return self.load_decision(decision_id)

    # ---- registry protocol ---------------------------------------------
    def verify_reference(self, reference: str) -> None:
        """Verify one typed reference against existing immutable storage."""

        if not isinstance(reference, str) or not reference or reference.strip() != reference:
            raise GovernanceArtifactConflict("reference must be non-empty text")
        try:
            if reference.startswith("raw-asset:"):
                self.raw.verify(RawAssetId(reference))
            elif reference.startswith("training-dataset:"):
                self.training.load_dataset(reference)
            elif reference.startswith("model-run:"):
                self.training.load_model_run(reference)
            elif reference.startswith("model-artifact:"):
                self.training.verify_model_artifact(reference)
            elif reference.startswith("model-output:"):
                self.training.verify_model_output(reference)
            elif reference.startswith("derived-artifact:"):
                self.derived.load_artifact_manifest(reference)
            elif reference.startswith("promotion-policy:"):
                self.load_policy(reference)
            elif reference.startswith("challenger-evidence:"):
                self.load_evidence(reference)
            elif reference.startswith("promotion-decision:"):
                self.load_decision(reference)
            elif reference.startswith("prediction:"):
                self._verify_prediction_ref(reference)
            elif reference.startswith("evaluation:") or reference.startswith("cohort:"):
                self._verify_manifest_output_ref(reference)
            elif reference.startswith("sample:"):
                self._verify_sample_ref(reference)
            else:
                raise GovernanceArtifactConflict(
                    f"unsupported or unverified reference: {reference}"
                )
        except (OSError, ValueError, ArchiveConflictError, KeyError) as error:
            raise GovernanceArtifactConflict(
                f"reference is unavailable or invalid: {reference}"
            ) from error
        if self.reference_validator is not None:
            validator = self.reference_validator
            try:
                if hasattr(validator, "verify_reference"):
                    validator.verify_reference(reference)
                elif callable(validator):
                    result = validator(reference)
                    if result is False:
                        raise GovernanceArtifactConflict(f"reference is unavailable: {reference}")
                else:
                    raise TypeError(
                        "reference_validator must be callable or expose verify_reference"
                    )
            except GovernanceArtifactConflict:
                raise
            except (OSError, ValueError, ArchiveConflictError, KeyError) as error:
                raise GovernanceArtifactConflict(
                    f"reference failed additional validation: {reference}"
                ) from error

    # ---- paths and verification ----------------------------------------
    def policy_path(self, policy_id: str) -> Path:
        digest = _digest_id(policy_id, "promotion-policy:")
        return self.layout.derived / "governance" / "policies" / digest[:2] / f"{digest}.json"

    def evidence_path(self, evidence_id: str) -> Path:
        digest = _digest_id(evidence_id, "challenger-evidence:")
        return self.layout.derived / "governance" / "evidence" / digest[:2] / f"{digest}.json"

    def decision_path(self, decision_id: str) -> Path:
        digest = _digest_id(decision_id, "promotion-decision:")
        return self.layout.derived / "governance" / "decisions" / digest[:2] / f"{digest}.json"

    def _verify_evidence_references(self, evidence: ChallengerEvidence) -> None:
        refs = _evidence_refs(evidence)
        if not evidence.model_run_ref:
            raise GovernanceArtifactConflict("challenger evidence requires model_run_ref")
        if not evidence.evaluation_ref:
            raise GovernanceArtifactConflict("challenger evidence requires evaluation_ref")
        if not evidence.cohort_ref:
            raise GovernanceArtifactConflict("challenger evidence requires cohort_ref")
        if not evidence.sample_refs:
            raise GovernanceArtifactConflict("challenger evidence requires sample_refs")
        if evidence.evaluated_at is None:
            raise GovernanceArtifactConflict("challenger evidence requires evaluated_at")
        if evidence.confidence_method is None or evidence.confidence_level is None:
            raise GovernanceArtifactConflict(
                "challenger evidence requires confidence method and level"
            )
        if not evidence.confidence_interval:
            raise GovernanceArtifactConflict("challenger evidence requires confidence interval")
        if not evidence.subgroup_diagnostics:
            raise GovernanceArtifactConflict("challenger evidence requires subgroup diagnostics")
        for reference in refs:
            # sample IDs are checked against the cohort below; all other refs
            # are resolved directly by the existing registries.
            if reference.startswith("sample:"):
                continue
            self.verify_reference(reference)
        evaluation_manifest = self._load_evaluation_manifest(evidence.evaluation_ref)
        evaluation_payload = evaluation_manifest.payload
        if not isinstance(evaluation_payload, Mapping):
            raise GovernanceArtifactConflict("evaluation artifact payload must be an object")
        if evaluation_payload.get("model_run_ref") != evidence.model_run_ref:
            raise GovernanceArtifactConflict("evaluation model reference does not match evidence")
        if evaluation_payload.get("champion_model_ref") != evidence.champion_model_ref:
            raise GovernanceArtifactConflict(
                "evaluation champion reference does not match evidence"
            )
        if evaluation_payload.get("cohort_ref") != evidence.cohort_ref:
            raise GovernanceArtifactConflict("evaluation cohort reference does not match evidence")
        raw_evaluation_samples = evaluation_payload.get("sample_refs")
        if not isinstance(raw_evaluation_samples, (list, tuple)) or isinstance(
            raw_evaluation_samples, (str, bytes)
        ):
            raise GovernanceArtifactConflict("evaluation sample_refs must be a sequence")
        if any(not isinstance(item, str) for item in raw_evaluation_samples):
            raise GovernanceArtifactConflict("evaluation sample_refs must contain text")
        evaluation_samples = tuple(raw_evaluation_samples)
        if set(evaluation_samples) != set(evidence.sample_refs):
            raise GovernanceArtifactConflict("evaluation samples do not match evidence")
        if not evidence.cohort_ref.startswith("training-dataset:"):
            raise GovernanceArtifactConflict("cohort_ref must reference a training dataset")
        if evidence.cohort_ref.startswith("training-dataset:"):
            dataset = self.training.load_dataset(evidence.cohort_ref)
            by_id = {sample.sample_id: sample for sample in dataset.samples}
            missing = sorted(set(evidence.sample_refs) - set(by_id))
            if missing:
                raise GovernanceArtifactConflict(
                    f"challenger evidence references unknown samples: {', '.join(missing)}"
                )
            samples = [by_id[item] for item in evidence.sample_refs]
            challenger = self.training.load_model_run(evidence.model_run_ref)
            champion = self.training.load_model_run(evidence.champion_model_ref)
            for name, artifact in (("challenger", challenger), ("champion", champion)):
                if artifact.dataset_id != evidence.cohort_ref:
                    raise GovernanceArtifactConflict(
                        f"{name} model run dataset does not match cohort_ref"
                    )
                if set(artifact.evaluation_cohort) != set(evidence.sample_refs):
                    raise GovernanceArtifactConflict(
                        f"{name} model run evaluation cohort does not match sample_refs"
                    )
                if artifact.evaluation_capture_mode is not evidence.capture_mode:
                    raise GovernanceArtifactConflict(
                        f"{name} model run capture mode does not match evidence"
                    )
            if evidence.captured_samples != len(evidence.sample_refs):
                raise GovernanceArtifactConflict(
                    "captured_samples must equal the verified sample_refs count"
                )
            if evidence.capture_mode is CaptureMode.CAPTURED:
                if any(sample.capture_mode is not CaptureMode.CAPTURED for sample in samples):
                    raise GovernanceArtifactConflict(
                        "captured promotion evidence cannot include reconstructed samples"
                    )
                if any(sample.capture_evidence_ref is None for sample in samples):
                    raise GovernanceArtifactConflict(
                        "captured promotion evidence requires raw capture evidence"
                    )
                # TrainingArtifactStore already verifies raw bytes and timing
                # while loading the dataset.  Re-run the check here so a custom
                # store cannot silently downgrade captured evidence.
                self.training._validate_capture_evidence(dataset)
        if evidence.capture_mode is CaptureMode.CAPTURED and not evidence.is_prospective_captured:
            raise GovernanceArtifactConflict(
                "captured promotion evidence must be marked prospective"
            )

    def _verify_decision_references(self, decision: PromotionDecision) -> None:
        if not decision.policy_ref or not decision.evidence_ref:
            raise GovernanceArtifactConflict("promotion decision requires policy and evidence refs")
        policy = self.load_policy(decision.policy_ref)
        evidence = self.load_evidence(decision.evidence_ref)
        if decision.rollback_artifact_ref != policy.rollback_artifact_ref:
            raise GovernanceArtifactConflict(
                "promotion decision rollback reference does not match policy"
            )
        if decision.challenger_model_ref != evidence.model_run_ref:
            raise GovernanceArtifactConflict(
                "promotion decision challenger model does not match evidence"
            )
        if decision.champion_model_ref != evidence.champion_model_ref:
            raise GovernanceArtifactConflict(
                "promotion decision champion model does not match evidence"
            )
        for reference in _decision_refs(decision):
            self.verify_reference(reference)
        if decision.promoted and not decision.rollback_artifact_ref:
            raise GovernanceArtifactConflict("promoted decision requires rollback artifact ref")
        if decision.decided_at is None:
            raise GovernanceArtifactConflict("promotion decision requires decided_at")
        if decision.decided_at < policy.reviewed_at:
            raise GovernanceArtifactConflict("decision precedes policy review")
        if evidence.evaluated_at is not None and decision.decided_at < evidence.evaluated_at:
            raise GovernanceArtifactConflict("decision precedes evidence evaluation")
        expected = assess_promotion(
            evidence,
            policy=policy,
            reference_validator=self,
            decided_at=decision.decided_at,
        )
        if expected != decision:
            raise GovernanceArtifactConflict(
                "promotion decision does not match recomputed governance result"
            )

    def _verify_prediction_ref(self, reference: str) -> None:
        digest = _digest_id(reference, "prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        payload = self._read_json(path, "prediction")
        if payload.get("id") != reference:
            raise GovernanceArtifactConflict("prediction reference ID does not match stored bytes")
        if payload.get("schema_version") != 3:
            raise GovernanceArtifactConflict("unsupported prediction schema version")
        identity = dict(payload)
        identity.pop("id", None)
        calculated = hashlib.sha256(_canonical_json(identity)).hexdigest()
        if calculated != digest:
            raise GovernanceArtifactConflict("prediction content hash does not match its ID")

        composition_ref = payload.get("composition_artifact_ref")
        if not isinstance(composition_ref, str):
            raise GovernanceArtifactConflict("prediction lacks a composition artifact reference")
        try:
            composition = self.derived.load_score_grid_composition_payload(composition_ref)
            composition_manifest = self.derived._load_artifact_manifest_for_output_ref(
                composition_ref
            )
            prediction_manifest = self.derived._load_artifact_manifest_for_output_ref(reference)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise GovernanceArtifactConflict(
                "prediction composition or manifest is unavailable or invalid"
            ) from error

        expected_composition = _prediction_composition_payload(payload)
        if composition != expected_composition:
            raise GovernanceArtifactConflict(
                "prediction composition does not match prediction payload"
            )
        if (
            composition_manifest.artifact_type != "score-grid-composition"
            or composition_manifest.status != "succeeded"
            or composition_manifest.payload != composition
        ):
            raise GovernanceArtifactConflict("prediction composition manifest does not match bytes")
        if (
            prediction_manifest.artifact_type != "prediction"
            or prediction_manifest.status != "succeeded"
            or prediction_manifest.payload != payload
        ):
            raise GovernanceArtifactConflict("prediction manifest does not match bytes")

    def _load_evaluation_manifest(self, reference: str) -> DerivedArtifactManifest:
        if reference.startswith("derived-artifact:"):
            manifest = self.derived.load_artifact_manifest(reference)
        else:
            manifest = self._find_manifest_output_ref(reference)
        if manifest.artifact_type != "evaluation":
            raise GovernanceArtifactConflict("reference is not an evaluation artifact")
        return manifest

    def _verify_manifest_output_ref(self, reference: str) -> None:
        self._find_manifest_output_ref(reference)

    def _find_manifest_output_ref(self, reference: str) -> DerivedArtifactManifest:
        root = self.layout.derived / "manifests" / "artifacts"
        if not root.exists():
            raise GovernanceArtifactConflict(f"no derived manifests found for {reference}")
        for path in root.glob("**/*.json"):
            try:
                manifest = self.derived.load_artifact_manifest(_manifest_id_from_path(path))
            except (OSError, ValueError, ArchiveConflictError):
                continue
            if reference in manifest.output_refs:
                return manifest
        raise GovernanceArtifactConflict(
            f"no immutable manifest contains output reference: {reference}"
        )

    def _verify_sample_ref(self, reference: str) -> None:
        root = self.layout.derived / "training-datasets"
        if not root.exists():
            raise GovernanceArtifactConflict(f"sample reference is unavailable: {reference}")
        for path in root.glob("**/*.json"):
            try:
                dataset = self.training.load_dataset(_id_from_path(path, "training-dataset"))
            except (OSError, ValueError, ArchiveConflictError):
                continue
            if any(sample.sample_id == reference for sample in dataset.samples):
                return
        raise GovernanceArtifactConflict(f"sample reference is unavailable: {reference}")

    def _write_manifest(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        payload: Mapping[str, Any],
        generated_at: datetime,
        input_refs: tuple[str, ...],
        quality: str,
    ) -> None:
        self.derived.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type=artifact_type,
                payload=dict(payload),
                generated_at=generated_at,
                started_at=generated_at,
                ended_at=generated_at,
                transform_version="governance/1",
                code_version=DERIVED_CODE_VERSION,
                input_refs=input_refs,
                output_refs=(artifact_id,),
                status="succeeded",
                quality=quality,
            )
        )

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        encoded = (
            json.dumps(
                payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != encoded:
                raise GovernanceArtifactConflict(f"governance artifact conflicts at {path}")
            return path
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise GovernanceArtifactConflict(
                    f"governance artifact conflicts at {path}"
                ) from None
        return path

    @staticmethod
    def _read_json(path: Path, kind: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GovernanceArtifactConflict(f"cannot read {kind}: {path}") from error
        if not isinstance(payload, dict):
            raise GovernanceArtifactConflict(f"{kind} must be an object: {path}")
        return payload


def parse_promotion_policy_payload(payload: Mapping[str, Any]) -> PromotionPolicy:
    try:
        policy = PromotionPolicy(
            policy_version=_strict_text(payload["policy_version"], "policy_version"),
            reviewed_by=_strict_text(payload["reviewed_by"], "reviewed_by"),
            reviewed_at=_parse_datetime(payload["reviewed_at"], "reviewed_at"),
            confidence_method=_strict_text(payload["confidence_method"], "confidence_method"),
            minimum_captured_samples=_strict_int(
                payload["minimum_captured_samples"], "minimum_captured_samples"
            ),
            minimum_observation_days=_strict_int(
                payload["minimum_observation_days"], "minimum_observation_days"
            ),
            maximum_brier_delta=_strict_float(
                payload["maximum_brier_delta"], "maximum_brier_delta"
            ),
            maximum_log_loss_delta=_strict_float(
                payload["maximum_log_loss_delta"], "maximum_log_loss_delta"
            ),
            rollback_artifact_ref=_strict_text(
                payload.get("rollback_artifact_ref", payload.get("rollback_target", "")),
                "rollback_artifact_ref",
            ),
            confidence_level=(
                None
                if payload.get("confidence_level") is None
                else _strict_float(payload["confidence_level"], "confidence_level")
            ),
            required_subgroups=tuple(
                _strict_text(item, "required_subgroups")
                for item in payload.get("required_subgroups", ())
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise GovernanceArtifactConflict(f"invalid promotion policy: {error}") from error
    if payload != policy.to_payload():
        raise GovernanceArtifactConflict("promotion policy is not canonical")
    return policy


def parse_challenger_evidence_payload(payload: Mapping[str, Any]) -> ChallengerEvidence:
    try:
        interval = payload.get("confidence_interval")
        evidence = ChallengerEvidence(
            capture_mode=CaptureMode(_strict_text(payload["capture_mode"], "capture_mode")),
            captured_samples=_strict_int(payload["captured_samples"], "captured_samples"),
            observation_days=_strict_int(payload["observation_days"], "observation_days"),
            brier_delta_vs_champion=_strict_float(
                payload["brier_delta_vs_champion"], "brier_delta_vs_champion"
            ),
            log_loss_delta_vs_champion=_strict_float(
                payload["log_loss_delta_vs_champion"], "log_loss_delta_vs_champion"
            ),
            confidence_interval_passed=_strict_bool(
                payload["confidence_interval_passed"], "confidence_interval_passed"
            ),
            reliability_passed=_strict_bool(payload["reliability_passed"], "reliability_passed"),
            subgroup_diagnostics_passed=_strict_bool(
                payload["subgroup_diagnostics_passed"], "subgroup_diagnostics_passed"
            ),
            model_run_ref=_optional_text(payload.get("model_run_ref")),
            champion_model_ref=_optional_text(payload.get("champion_model_ref")),
            evaluation_ref=_optional_text(payload.get("evaluation_ref")),
            cohort_ref=_optional_text(payload.get("cohort_ref")),
            sample_refs=tuple(
                _strict_text(item, "sample_refs") for item in payload.get("sample_refs", ())
            ),
            confidence_level=(
                None
                if payload.get("confidence_level") is None
                else _strict_float(payload["confidence_level"], "confidence_level")
            ),
            confidence_interval=(
                None
                if interval is None
                else (
                    _strict_float(interval[0], "confidence_interval.lower"),
                    _strict_float(interval[1], "confidence_interval.upper"),
                )
            ),
            confidence_method=_optional_text(payload.get("confidence_method")),
            subgroup_diagnostics=tuple(
                SubgroupDiagnostic(
                    subgroup=_strict_text(item["subgroup"], "subgroup"),
                    sample_count=_strict_int(item["sample_count"], "sample_count"),
                    brier_delta=_strict_float(item["brier_delta"], "brier_delta"),
                    log_loss_delta=_strict_float(item["log_loss_delta"], "log_loss_delta"),
                    passed=_strict_bool(item["passed"], "passed"),
                )
                for item in payload.get("subgroup_diagnostics", ())
            ),
            prospective=(
                None
                if payload.get("prospective") is None
                else _strict_bool(payload["prospective"], "prospective")
            ),
            evaluated_at=(
                None
                if payload.get("evaluated_at") is None
                else _parse_datetime(payload["evaluated_at"], "evaluated_at")
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
        raise GovernanceArtifactConflict(f"invalid challenger evidence: {error}") from error
    if payload != evidence.to_payload():
        raise GovernanceArtifactConflict("challenger evidence is not canonical")
    return evidence


def parse_promotion_decision_payload(payload: Mapping[str, Any]) -> PromotionDecision:
    try:
        decision = PromotionDecision(
            promoted=_strict_bool(payload["promoted"], "promoted"),
            reason_codes=tuple(
                _strict_text(item, "reason_codes") for item in payload["reason_codes"]
            ),
            policy_ref=_optional_text(payload.get("policy_ref")),
            evidence_ref=_optional_text(payload.get("evidence_ref")),
            challenger_model_ref=_optional_text(payload.get("challenger_model_ref")),
            champion_model_ref=_optional_text(payload.get("champion_model_ref")),
            rollback_artifact_ref=_optional_text(payload.get("rollback_artifact_ref")),
            decided_at=(
                None
                if payload.get("decided_at") is None
                else _parse_datetime(payload["decided_at"], "decided_at")
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise GovernanceArtifactConflict(f"invalid promotion decision: {error}") from error
    if payload != decision.to_payload():
        raise GovernanceArtifactConflict("promotion decision is not canonical")
    return decision


def _evidence_refs(evidence: ChallengerEvidence) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ref
                for ref in (
                    evidence.model_run_ref,
                    evidence.champion_model_ref,
                    evidence.evaluation_ref,
                    evidence.cohort_ref,
                    *evidence.sample_refs,
                )
                if ref is not None
            }
        )
    )


def _decision_refs(decision: PromotionDecision) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ref
                for ref in (
                    decision.policy_ref,
                    decision.evidence_ref,
                    decision.challenger_model_ref,
                    decision.champion_model_ref,
                    decision.rollback_artifact_ref,
                )
                if ref is not None
            }
        )
    )


def _digest_id(value: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"invalid {prefix} ID")
    digest = value.removeprefix(prefix)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid {prefix} ID")
    return digest


def _id_from_path(path: Path, prefix: str) -> str:
    return f"{prefix}:{path.stem}"


def _manifest_id_from_path(path: Path) -> str:
    return f"derived-artifact:{path.stem}"


def _prediction_composition_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a persisted prediction onto its canonical grid composition payload."""

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
        raise GovernanceArtifactConflict(
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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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


def _strict_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _strict_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional reference must be text")
    return value
