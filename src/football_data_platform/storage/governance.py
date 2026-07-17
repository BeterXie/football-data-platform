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
import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    MatchResult90,
    ScorePrediction,
    exact_score_probability,
    result_probabilities,
)
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.domain.training import (
    ModelRunArtifact,
    ModelRunStatus,
    TrainingSample,
)
from football_data_platform.evaluation.governance import (
    EVALUATION_COMPARISON_SCHEMA_VERSION,
    EVALUATION_COMPARISON_TYPE,
    ChallengerEvidence,
    EvaluationComparison,
    EvaluationPairReference,
    EvaluationSubgroup,
    PairedEvaluation,
    PromotionDecision,
    PromotionPolicy,
    SubgroupDiagnostic,
    aggregate_challenger_evidence,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import (
    EvaluationRecord,
    evaluation_record_payload,
    parse_evaluation_record_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import TrainingArtifactStore


class GovernanceArtifactConflict(ArchiveConflictError):
    """Raised when a governance artifact or one of its references is invalid."""


_TRUSTED_CAPTURE_RUN_VALIDATION_AVAILABLE = False


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

    # ---- paired evaluation comparison ---------------------------------
    def write_comparison(self, comparison: EvaluationComparison) -> Path:
        if not isinstance(comparison, EvaluationComparison):
            raise TypeError("comparison must be an EvaluationComparison")
        self._aggregate_comparison(comparison)
        path = self._write_json(
            self.comparison_path(comparison.content_id), comparison.to_payload()
        )
        self._write_manifest(
            artifact_type="evaluation-comparison",
            artifact_id=comparison.content_id,
            payload=comparison.to_payload(),
            generated_at=comparison.generated_at,
            input_refs=_comparison_refs(comparison),
            quality="ready",
            schema_version=EVALUATION_COMPARISON_SCHEMA_VERSION,
        )
        return path

    def load_comparison(self, comparison_id: str) -> EvaluationComparison:
        comparison = self._load_comparison_document(comparison_id)
        self._aggregate_comparison(comparison)
        return comparison

    def _load_comparison_document(self, comparison_id: str) -> EvaluationComparison:
        payload = self._read_json(self.comparison_path(comparison_id), "evaluation comparison")
        comparison = parse_evaluation_comparison_payload(payload)
        if comparison.content_id != comparison_id:
            raise GovernanceArtifactConflict("evaluation comparison path and ID disagree")
        self._verify_comparison_manifest(comparison)
        return comparison

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
            elif reference.startswith("evaluation:"):
                manifest = self._find_manifest_output_ref(reference)
                if manifest.artifact_type == "evaluation-comparison":
                    self.load_comparison(reference)
                elif manifest.artifact_type == "evaluation":
                    self._load_evaluation_record(reference)
                else:
                    raise GovernanceArtifactConflict(
                        "evaluation reference has an unsupported artifact type"
                    )
            elif reference.startswith("cohort:"):
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

    def comparison_path(self, comparison_id: str) -> Path:
        digest = _digest_id(comparison_id, "evaluation:")
        return self.layout.derived / "governance" / "comparisons" / digest[:2] / f"{digest}.json"

    def _verify_evidence_references(self, evidence: ChallengerEvidence) -> None:
        if not evidence.evaluation_ref:
            raise GovernanceArtifactConflict("challenger evidence requires evaluation_ref")
        try:
            comparison = self._load_comparison_document(evidence.evaluation_ref)
            computed = self._aggregate_comparison(comparison)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise GovernanceArtifactConflict(
                "challenger evidence inputs are unavailable or invalid"
            ) from error
        if computed.to_payload() != evidence.to_payload():
            raise GovernanceArtifactConflict(
                "challenger evidence does not match recomputed evaluation records"
            )

    def _aggregate_comparison(self, comparison: EvaluationComparison) -> ChallengerEvidence:
        if not comparison.cohort_ref.startswith("training-dataset:"):
            raise GovernanceArtifactConflict("cohort_ref must reference a training dataset")
        try:
            dataset = self.training.load_formal_dataset(comparison.cohort_ref)
            challenger = self.training.load_model_run(comparison.model_run_ref)
            champion = self.training.load_model_run(comparison.champion_model_ref)
            challenger_dataset = self.training.load_formal_dataset(challenger.dataset_id)
            champion_dataset = self.training.load_formal_dataset(champion.dataset_id)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise GovernanceArtifactConflict(
                "evaluation comparison model cohort is unavailable or invalid"
            ) from error

        if dataset.generated_at > comparison.generated_at:
            raise GovernanceArtifactConflict(
                "evaluation comparison predates its formal evaluation cohort"
            )
        by_id = {sample.sample_id: sample for sample in dataset.samples}
        eligible_refs = {sample.sample_id for sample in dataset.included_samples}
        if set(comparison.sample_refs) != eligible_refs:
            raise GovernanceArtifactConflict(
                "evaluation comparison samples do not exactly match the formal cohort"
            )
        missing = sorted(set(comparison.sample_refs) - set(by_id))
        if missing:
            raise GovernanceArtifactConflict(
                f"evaluation comparison references unknown samples: {', '.join(missing)}"
            )
        samples = {sample_ref: by_id[sample_ref] for sample_ref in comparison.sample_refs}
        if any(not sample.eligible for sample in samples.values()):
            raise GovernanceArtifactConflict("evaluation comparison contains excluded samples")
        if any(sample.split == "train" for sample in samples.values()):
            raise GovernanceArtifactConflict(
                "formal evaluation cohorts cannot contain training samples"
            )
        comparison_samples = tuple(samples.values())
        _verify_independent_comparison_samples(
            comparison_samples,
            (*challenger_dataset.samples, *champion_dataset.samples),
        )
        _verify_comparison_model_timing(
            (("challenger", challenger), ("champion", champion)),
            comparison_samples,
        )

        pairs: list[PairedEvaluation] = []
        prospective_timing = True
        for pair_ref in comparison.pairs:
            challenger_record = self._load_evaluation_record(pair_ref.challenger_evaluation_ref)
            champion_record = self._load_evaluation_record(pair_ref.champion_evaluation_ref)
            sample = samples[pair_ref.sample_ref]
            challenger_generated_at = self._verify_evaluation_sources(
                challenger_record,
                sample=sample,
                expected_model_ref=comparison.model_run_ref,
            )
            champion_generated_at = self._verify_evaluation_sources(
                champion_record,
                sample=sample,
                expected_model_ref=comparison.champion_model_ref,
            )
            if any(
                generated_at >= sample.label_known_at
                for generated_at in (challenger_generated_at, champion_generated_at)
            ):
                raise GovernanceArtifactConflict(
                    "evaluation prediction must predate its typed result label"
                )
            prospective_timing = prospective_timing and all(
                generated_at < sample.label_known_at <= record.evaluated_at
                for generated_at, record in (
                    (challenger_generated_at, challenger_record),
                    (champion_generated_at, champion_record),
                )
            )
            pairs.append(PairedEvaluation(pair_ref.sample_ref, challenger_record, champion_record))
        if comparison.generated_at < max(
            max(pair.challenger.evaluated_at, pair.champion.evaluated_at) for pair in pairs
        ):
            raise GovernanceArtifactConflict(
                "evaluation comparison predates one of its evaluation records"
            )

        captured = all(
            sample.capture_mode is CaptureMode.CAPTURED
            and sample.capture_evidence_ref is not None
            and sample.capture_observed_at is not None
            and sample.capture_observed_at <= sample.as_of < sample.label_known_at
            for sample in samples.values()
        )
        if captured:
            self.training._validate_capture_evidence(dataset)
        # Raw sample evidence does not attest that the prediction snapshot itself came from a
        # trusted capture run. Keep captured comparisons in shadow governance until R01 has an
        # authoritative capture-run validator.
        prospective_captured = (
            captured and prospective_timing and _TRUSTED_CAPTURE_RUN_VALIDATION_AVAILABLE
        )
        try:
            return aggregate_challenger_evidence(
                comparison,
                pairs,
                sample_as_of={key: value.as_of for key, value in samples.items()},
                prospective_captured=prospective_captured,
            )
        except (TypeError, ValueError) as error:
            raise GovernanceArtifactConflict(
                "evaluation comparison cannot be aggregated"
            ) from error

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

    def _verify_prediction_ref(self, reference: str) -> ScorePrediction:
        try:
            return self.training.load_verified_prediction(reference)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise GovernanceArtifactConflict(
                f"prediction domain replay is unavailable or invalid: {error}"
            ) from error

    def _load_evaluation_record(self, reference: str) -> EvaluationRecord:
        manifest = self._find_manifest_output_ref(reference)
        if manifest.artifact_type != "evaluation" or manifest.status != "succeeded":
            raise GovernanceArtifactConflict("reference is not a successful evaluation record")
        if not isinstance(manifest.payload, Mapping):
            raise GovernanceArtifactConflict("evaluation record payload must be an object")
        try:
            record = parse_evaluation_record_payload(manifest.payload)
        except (TypeError, ValueError) as error:
            raise GovernanceArtifactConflict("evaluation record payload is invalid") from error
        canonical_payload = evaluation_record_payload(record)
        expected_ref = (
            "evaluation:" + hashlib.sha256(_canonical_json(canonical_payload)).hexdigest()
        )
        if reference != expected_ref:
            raise GovernanceArtifactConflict(
                "evaluation record reference does not match canonical payload"
            )
        if (
            manifest.schema_version != record.schema_version
            or manifest.transform_version != "evaluation/2"
            or manifest.payload != canonical_payload
            or manifest.input_refs != record.input_refs
            or manifest.output_refs != (reference,)
            or manifest.generated_at != record.evaluated_at
        ):
            raise GovernanceArtifactConflict(
                "evaluation manifest does not match its persisted record"
            )
        return record

    def _verify_comparison_manifest(self, comparison: EvaluationComparison) -> None:
        manifest = self._find_manifest_output_ref(comparison.content_id)
        if (
            manifest.artifact_type != "evaluation-comparison"
            or manifest.schema_version != EVALUATION_COMPARISON_SCHEMA_VERSION
            or manifest.status != "succeeded"
            or manifest.payload != comparison.to_payload()
            or manifest.input_refs != _comparison_refs(comparison)
            or manifest.output_refs != (comparison.content_id,)
            or manifest.generated_at != comparison.generated_at
        ):
            raise GovernanceArtifactConflict(
                "evaluation comparison manifest does not match persisted bytes"
            )

    def _verify_evaluation_sources(
        self,
        record: EvaluationRecord,
        *,
        sample: Any,
        expected_model_ref: str,
    ) -> datetime:
        if record.market_benchmark.available:
            raise GovernanceArtifactConflict(
                "governance evidence requires a persisted real market snapshot; "
                "available market benchmarks are unsupported by this store"
            )
        if record.sample_ref != sample.sample_id:
            raise GovernanceArtifactConflict("evaluation record sample does not match cohort")
        if record.model_run_ref != expected_model_ref:
            raise GovernanceArtifactConflict("evaluation record model run does not match pair")
        if record.capture_mode is not sample.capture_mode:
            raise GovernanceArtifactConflict("evaluation capture mode does not match sample")

        prediction = self._verify_prediction_ref(record.prediction_id)
        model_run = self.training.load_model_run(expected_model_ref)
        if (
            prediction.model_run_id.value != expected_model_ref
            or prediction.match_id.value != record.match_id
            or prediction.capture_mode is not record.capture_mode
            or expected_model_ref not in prediction.input_refs
        ):
            raise GovernanceArtifactConflict(
                "evaluation prediction does not match its model, match, or capture mode"
            )
        snapshot_as_of = prediction.snapshot_as_of
        generated_at = prediction.generated_at
        if snapshot_as_of != sample.as_of:
            raise GovernanceArtifactConflict(
                "evaluation prediction snapshot does not match sample as_of"
            )
        if generated_at > record.evaluated_at:
            raise GovernanceArtifactConflict("evaluation predates its prediction")
        if (
            model_run.status is not ModelRunStatus.SUCCEEDED
            or model_run.task != "score-model"
            or model_run.model_version != prediction.model_version
        ):
            raise GovernanceArtifactConflict(
                "evaluation prediction cites an unusable score model run"
            )
        if model_run.ended_at > generated_at:
            raise GovernanceArtifactConflict(
                "evaluation prediction predates its model run completion"
            )

        result = self._load_canonical_result(record)
        if sample.label_ref != result.source_ref:
            raise GovernanceArtifactConflict("evaluation result does not match sample label_ref")
        if sample.label_known_at != result.known_at:
            raise GovernanceArtifactConflict(
                "evaluation result known_at does not match sample label_known_at"
            )
        if not isinstance(sample.label, Mapping) or (
            sample.label.get("home_goals") != result.home_goals
            or sample.label.get("away_goals") != result.away_goals
        ):
            raise GovernanceArtifactConflict("evaluation result does not match sample label")
        if result.known_at > record.evaluated_at:
            raise GovernanceArtifactConflict("evaluation predates the canonical result")

        probabilities = tuple(result_probabilities(prediction).items())
        if not _probabilities_equal(record.result_probabilities, probabilities):
            raise GovernanceArtifactConflict(
                "evaluation probabilities do not match the persisted prediction"
            )
        home_goals, away_goals = (int(item) for item in record.actual_score.split(":"))
        score_probability = exact_score_probability(prediction, home_goals, away_goals)
        expected_score_log_loss = -math.log(max(score_probability, 1e-15))
        if not math.isclose(
            record.score_log_loss,
            expected_score_log_loss,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise GovernanceArtifactConflict(
                "evaluation score log loss does not match the persisted prediction"
            )
        return generated_at

    def _read_prediction_payload(self, reference: str) -> dict[str, Any]:
        digest = _digest_id(reference, "prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        return self._read_json(path, "prediction")

    def _load_canonical_result(self, record: EvaluationRecord) -> MatchResult90:
        result_refs = tuple(
            reference
            for reference in record.input_refs
            if reference.startswith("fact:match_results_90:")
        )
        if len(result_refs) != 1:
            raise GovernanceArtifactConflict(
                "evaluation record must cite exactly one canonical 90-minute result"
            )
        result_ref = result_refs[0]
        canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
        try:
            with canonical.connect() as connection:
                row = connection.execute(
                    "SELECT match_id, home_goals, away_goals, known_at, raw_asset_id "
                    "FROM match_results_90 WHERE record_id = ?",
                    (result_ref,),
                ).fetchone()
        except sqlite3.Error as error:
            raise GovernanceArtifactConflict("canonical result store is unavailable") from error
        if row is None:
            raise GovernanceArtifactConflict("evaluation canonical result is unavailable")
        result = MatchResult90(
            match_id=MatchId(str(row["match_id"])),
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
            known_at=_parse_datetime(row["known_at"], "canonical result known_at"),
            source_ref=result_ref,
        )
        try:
            self.raw.verify(RawAssetId(str(row["raw_asset_id"])))
            CanonicalFactStore(canonical).verify_match_result(result)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise GovernanceArtifactConflict(
                "evaluation canonical result evidence is unavailable or invalid"
            ) from error
        expected_score = f"{result.home_goals}:{result.away_goals}"
        expected_outcome = (
            "home"
            if result.home_goals > result.away_goals
            else "away"
            if result.home_goals < result.away_goals
            else "draw"
        )
        if (
            result.match_id.value != record.match_id
            or record.actual_score != expected_score
            or record.actual_outcome != expected_outcome
        ):
            raise GovernanceArtifactConflict(
                "evaluation outcome does not match the canonical result"
            )
        return result

    def _verify_manifest_output_ref(self, reference: str) -> None:
        self._find_manifest_output_ref(reference)

    def _find_manifest_output_ref(self, reference: str) -> DerivedArtifactManifest:
        try:
            return self.derived._load_artifact_manifest_for_output_ref(reference)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise GovernanceArtifactConflict(
                f"no immutable manifest contains output reference: {reference}"
            ) from error

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
        schema_version: int = 1,
    ) -> None:
        self.derived.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type=artifact_type,
                schema_version=schema_version,
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


def _verify_independent_comparison_samples(
    comparison_samples: Sequence[TrainingSample],
    model_samples: Sequence[TrainingSample],
) -> None:
    model_sample_ids = {sample.sample_id for sample in model_samples}
    model_matches = {
        (sample.match_id, sample.match_version)
        for sample in model_samples
        if sample.match_id is not None and sample.match_version is not None
    }
    model_snapshot_refs = {
        sample.snapshot_ref for sample in model_samples if sample.snapshot_ref is not None
    }
    model_label_refs = {sample.label_ref for sample in model_samples}
    model_qualification_refs = {
        sample.qualification_ref for sample in model_samples if sample.qualification_ref is not None
    }

    for sample in comparison_samples:
        reused: list[str] = []
        if sample.sample_id in model_sample_ids:
            reused.append("sample_id")
        if (sample.match_id, sample.match_version) in model_matches:
            reused.append("match_id+match_version")
        if sample.snapshot_ref is not None and sample.snapshot_ref in model_snapshot_refs:
            reused.append("snapshot_ref")
        if sample.label_ref in model_label_refs:
            reused.append("label_ref")
        if (
            sample.qualification_ref is not None
            and sample.qualification_ref in model_qualification_refs
        ):
            reused.append("qualification_ref")
        if reused:
            raise GovernanceArtifactConflict(
                "formal evaluation cohort reuses model dataset provenance: " + ", ".join(reused)
            )


def _verify_comparison_model_timing(
    model_runs: Sequence[tuple[str, ModelRunArtifact]],
    comparison_samples: Sequence[TrainingSample],
) -> None:
    for role, model_run in model_runs:
        if any(model_run.ended_at > sample.as_of for sample in comparison_samples):
            raise GovernanceArtifactConflict(
                f"{role} model run completed after formal evaluation sample as_of"
            )


def parse_evaluation_comparison_payload(
    payload: Mapping[str, Any],
) -> EvaluationComparison:
    if not isinstance(payload, Mapping):
        raise GovernanceArtifactConflict("evaluation comparison must be an object")
    try:
        if (
            _strict_int(payload["schema_version"], "schema_version")
            != EVALUATION_COMPARISON_SCHEMA_VERSION
            or _strict_text(payload["record_type"], "record_type") != EVALUATION_COMPARISON_TYPE
        ):
            raise ValueError("unsupported evaluation comparison schema")
        raw_pairs = payload["pairs"]
        raw_subgroups = payload["subgroups"]
        if not isinstance(raw_pairs, list) or not all(
            isinstance(item, Mapping) for item in raw_pairs
        ):
            raise ValueError("pairs must be a list of objects")
        if not isinstance(raw_subgroups, list) or not all(
            isinstance(item, Mapping) for item in raw_subgroups
        ):
            raise ValueError("subgroups must be a list of objects")
        comparison = EvaluationComparison(
            model_run_ref=_strict_text(payload["model_run_ref"], "model_run_ref"),
            champion_model_ref=_strict_text(payload["champion_model_ref"], "champion_model_ref"),
            cohort_ref=_strict_text(payload["cohort_ref"], "cohort_ref"),
            pairs=tuple(
                EvaluationPairReference(
                    sample_ref=_strict_text(item["sample_ref"], "sample_ref"),
                    challenger_evaluation_ref=_strict_text(
                        item["challenger_evaluation_ref"], "challenger_evaluation_ref"
                    ),
                    champion_evaluation_ref=_strict_text(
                        item["champion_evaluation_ref"], "champion_evaluation_ref"
                    ),
                )
                for item in raw_pairs
            ),
            subgroups=tuple(
                EvaluationSubgroup(
                    subgroup=_strict_text(item["subgroup"], "subgroup"),
                    sample_refs=tuple(
                        _strict_text(reference, "subgroup sample_ref")
                        for reference in item["sample_refs"]
                    ),
                )
                for item in raw_subgroups
            ),
            confidence_method=_strict_text(payload["confidence_method"], "confidence_method"),
            confidence_level=_strict_float(payload["confidence_level"], "confidence_level"),
            bootstrap_seed=_strict_int(payload["bootstrap_seed"], "bootstrap_seed"),
            bootstrap_resamples=_strict_int(payload["bootstrap_resamples"], "bootstrap_resamples"),
            generated_at=_parse_datetime(payload["generated_at"], "generated_at"),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise GovernanceArtifactConflict(f"invalid evaluation comparison: {error}") from error
    if dict(payload) != comparison.to_payload():
        raise GovernanceArtifactConflict("evaluation comparison is not canonical")
    return comparison


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


def _comparison_refs(comparison: EvaluationComparison) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                comparison.model_run_ref,
                comparison.champion_model_ref,
                comparison.cohort_ref,
                *comparison.sample_refs,
                *(
                    reference
                    for pair in comparison.pairs
                    for reference in (
                        pair.challenger_evaluation_ref,
                        pair.champion_evaluation_ref,
                    )
                ),
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


def _probabilities_equal(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
) -> bool:
    return len(left) == len(right) and all(
        left_outcome == right_outcome
        and math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)
        for (left_outcome, left_value), (right_outcome, right_value) in zip(
            left, right, strict=True
        )
    )


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
