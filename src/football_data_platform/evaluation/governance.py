"""Champion/challenger governance with immutable, reviewable evidence.

The small value objects in this module are deliberately independent of the
filesystem.  Their IDs are hashes of canonical JSON, so a persisted record can
be re-created and checked without trusting a caller-supplied identifier.  The
filesystem adapter lives in :mod:`football_data_platform.storage.governance`;
keeping the two layers separate also makes it possible to run governance
checks in an offline evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from football_data_platform.domain.ids import ModelRunId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode

GOVERNANCE_SCHEMA_VERSION = 1
_POLICY_PREFIX = "promotion-policy:"
_EVIDENCE_PREFIX = "challenger-evidence:"
_DECISION_PREFIX = "promotion-decision:"


class GovernanceReferenceValidator(Protocol):
    """Validate that a referenced immutable artifact really exists."""

    def verify_reference(self, reference: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SubgroupDiagnostic:
    """One pre-declared subgroup result used during challenger review."""

    subgroup: str
    sample_count: int
    brier_delta: float
    log_loss_delta: float
    passed: bool

    def __post_init__(self) -> None:
        _require_text(self.subgroup, "subgroup")
        if not _is_int(self.sample_count) or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        _require_finite(self.brier_delta, "brier_delta")
        _require_finite(self.log_loss_delta, "log_loss_delta")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")

    def to_payload(self) -> dict[str, Any]:
        return {
            "subgroup": self.subgroup,
            "sample_count": self.sample_count,
            "brier_delta": self.brier_delta,
            "log_loss_delta": self.log_loss_delta,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    policy_version: str
    reviewed_by: str
    reviewed_at: datetime
    confidence_method: str
    # Defaults keep the original positional constructor usable; validation in
    # ``__post_init__`` still rejects an incomplete policy.
    rollback_target: str | None = None
    minimum_captured_samples: int = 0
    minimum_observation_days: int = 0
    maximum_brier_delta: float = 0.0
    maximum_log_loss_delta: float = 0.0
    # ``rollback_artifact_ref`` is the descriptive name used by the design.
    # ``rollback_target`` remains accepted for compatibility with the initial
    # public API and is normalized to the same value.
    rollback_artifact_ref: str | None = None
    confidence_level: float | None = None
    required_subgroups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.policy_version, "policy_version")
        _require_text(self.reviewed_by, "reviewed_by")
        _require_text(self.confidence_method, "confidence_method")
        rollback_ref = self.rollback_artifact_ref or self.rollback_target
        if not rollback_ref:
            raise ValueError("rollback_artifact_ref is required")
        if self.rollback_target and self.rollback_artifact_ref:
            if self.rollback_target != self.rollback_artifact_ref:
                raise ValueError("rollback_target and rollback_artifact_ref must match")
        # Keep the old field populated for callers that still inspect it.
        object.__setattr__(self, "rollback_target", rollback_ref)
        object.__setattr__(self, "rollback_artifact_ref", rollback_ref)
        if rollback_ref.startswith("model-run:"):
            # A legacy human-readable model-run label is allowed in memory;
            # the persistence layer requires a content-addressed artifact.
            try:
                ModelRunId(rollback_ref)
            except ValueError:
                raise ValueError(
                    "rollback_artifact_ref must be a typed artifact reference"
                ) from None
        else:
            _require_ref(rollback_ref, "rollback_artifact_ref")
        require_utc(self.reviewed_at, "reviewed_at")
        if not _is_int(self.minimum_captured_samples) or self.minimum_captured_samples < 1:
            raise ValueError("minimum_captured_samples must be positive")
        if not _is_int(self.minimum_observation_days) or self.minimum_observation_days < 1:
            raise ValueError("minimum_observation_days must be positive")
        _require_finite(self.maximum_brier_delta, "maximum_brier_delta")
        _require_finite(self.maximum_log_loss_delta, "maximum_log_loss_delta")
        if self.confidence_level is not None:
            _require_finite(self.confidence_level, "confidence_level")
            if not 0 < self.confidence_level < 1:
                raise ValueError("confidence_level must be between zero and one")
        normalized_subgroups = _normalize_texts(self.required_subgroups, "required_subgroups")
        object.__setattr__(self, "required_subgroups", normalized_subgroups)

    @property
    def id(self) -> str:
        return self.content_id

    @property
    def content_id(self) -> str:
        return _content_id(_POLICY_PREFIX, self.identity_payload())

    @property
    def policy_id(self) -> str:
        return self.content_id

    @property
    def content_hash(self) -> str:
        return self.content_id.removeprefix(_POLICY_PREFIX)

    @property
    def artifact_id(self) -> str:
        return self.content_id

    @property
    def rollback_ref(self) -> str:
        return self.rollback_artifact_ref or self.rollback_target or ""

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "policy_version": self.policy_version,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": _timestamp(self.reviewed_at),
            "confidence_method": self.confidence_method,
            "rollback_artifact_ref": self.rollback_artifact_ref,
            "minimum_captured_samples": self.minimum_captured_samples,
            "minimum_observation_days": self.minimum_observation_days,
            "maximum_brier_delta": self.maximum_brier_delta,
            "maximum_log_loss_delta": self.maximum_log_loss_delta,
            "confidence_level": self.confidence_level,
            "required_subgroups": list(self.required_subgroups),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.content_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class ChallengerEvidence:
    capture_mode: CaptureMode
    captured_samples: int
    observation_days: int
    brier_delta_vs_champion: float
    log_loss_delta_vs_champion: float
    confidence_interval_passed: bool
    reliability_passed: bool
    subgroup_diagnostics_passed: bool
    model_run_ref: str | None = None
    champion_model_ref: str | None = None
    evaluation_ref: str | None = None
    cohort_ref: str | None = None
    sample_refs: tuple[str, ...] = ()
    confidence_level: float | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_method: str | None = None
    subgroup_diagnostics: tuple[SubgroupDiagnostic, ...] = ()
    prospective: bool | None = None
    evaluated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capture_mode, CaptureMode):
            raise TypeError("capture_mode must be a CaptureMode")
        if not _is_int(self.captured_samples) or self.captured_samples < 0:
            raise ValueError("captured_samples must be a non-negative integer")
        if not _is_int(self.observation_days) or self.observation_days < 0:
            raise ValueError("observation_days must be a non-negative integer")
        _require_finite(self.brier_delta_vs_champion, "brier_delta_vs_champion")
        _require_finite(self.log_loss_delta_vs_champion, "log_loss_delta_vs_champion")
        for field_name in (
            "confidence_interval_passed",
            "reliability_passed",
            "subgroup_diagnostics_passed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in (
            "model_run_ref",
            "champion_model_ref",
            "evaluation_ref",
            "cohort_ref",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_ref(value, field_name)
        normalized_samples = _normalize_refs(self.sample_refs, "sample_refs")
        object.__setattr__(self, "sample_refs", normalized_samples)
        if self.confidence_level is not None:
            _require_finite(self.confidence_level, "confidence_level")
            if not 0 < self.confidence_level < 1:
                raise ValueError("confidence_level must be between zero and one")
        if self.confidence_interval is not None:
            if len(self.confidence_interval) != 2:
                raise ValueError("confidence_interval must contain lower and upper bounds")
            lower, upper = self.confidence_interval
            _require_finite(lower, "confidence_interval.lower")
            _require_finite(upper, "confidence_interval.upper")
            if lower > upper:
                raise ValueError("confidence_interval lower bound exceeds upper bound")
            object.__setattr__(self, "confidence_interval", (float(lower), float(upper)))
        if self.confidence_method is not None:
            _require_text(self.confidence_method, "confidence_method")
        if self.prospective is not None and not isinstance(self.prospective, bool):
            raise TypeError("prospective must be a bool when present")
        if self.evaluated_at is not None:
            require_utc(self.evaluated_at, "evaluated_at")
        normalized_diagnostics = _normalize_diagnostics(self.subgroup_diagnostics)
        object.__setattr__(self, "subgroup_diagnostics", normalized_diagnostics)

    @property
    def is_prospective_captured(self) -> bool:
        """Whether this evidence can be used for a promotion decision."""

        # ``captured`` describes how bytes were obtained; it does not prove
        # that the cohort was collected prospectively.  The explicit marker is
        # therefore required before evidence can satisfy promotion gates.
        return self.capture_mode is CaptureMode.CAPTURED and self.prospective is True

    @property
    def id(self) -> str:
        return self.content_id

    @property
    def content_id(self) -> str:
        return _content_id(_EVIDENCE_PREFIX, self.identity_payload())

    @property
    def evidence_id(self) -> str:
        return self.content_id

    @property
    def content_hash(self) -> str:
        return self.content_id.removeprefix(_EVIDENCE_PREFIX)

    @property
    def artifact_id(self) -> str:
        return self.content_id

    @property
    def model_ref(self) -> str | None:
        return self.model_run_ref

    @property
    def evaluation_id(self) -> str | None:
        return self.evaluation_ref

    @property
    def cohort_id(self) -> str | None:
        return self.cohort_ref

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.sample_refs

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "capture_mode": self.capture_mode.value,
            "captured_samples": self.captured_samples,
            "observation_days": self.observation_days,
            "brier_delta_vs_champion": self.brier_delta_vs_champion,
            "log_loss_delta_vs_champion": self.log_loss_delta_vs_champion,
            "confidence_interval_passed": self.confidence_interval_passed,
            "reliability_passed": self.reliability_passed,
            "subgroup_diagnostics_passed": self.subgroup_diagnostics_passed,
            "model_run_ref": self.model_run_ref,
            "champion_model_ref": self.champion_model_ref,
            "evaluation_ref": self.evaluation_ref,
            "cohort_ref": self.cohort_ref,
            "sample_refs": list(self.sample_refs),
            "confidence_level": self.confidence_level,
            "confidence_interval": (
                None if self.confidence_interval is None else list(self.confidence_interval)
            ),
            "confidence_method": self.confidence_method,
            "subgroup_diagnostics": [item.to_payload() for item in self.subgroup_diagnostics],
            "prospective": self.prospective,
            "evaluated_at": None if self.evaluated_at is None else _timestamp(self.evaluated_at),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.content_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason_codes: tuple[str, ...]
    policy_ref: str | None = None
    evidence_ref: str | None = None
    challenger_model_ref: str | None = None
    champion_model_ref: str | None = None
    rollback_artifact_ref: str | None = None
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.promoted, bool):
            raise TypeError("promoted must be a bool")
        object.__setattr__(self, "reason_codes", _normalize_reason_codes(self.reason_codes))
        for field_name in (
            "policy_ref",
            "evidence_ref",
            "challenger_model_ref",
            "champion_model_ref",
            "rollback_artifact_ref",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_ref(value, field_name)
        if self.decided_at is not None:
            require_utc(self.decided_at, "decided_at")

    @property
    def id(self) -> str:
        return self.content_id

    @property
    def content_id(self) -> str:
        return _content_id(_DECISION_PREFIX, self.identity_payload())

    @property
    def decision_id(self) -> str:
        return self.content_id

    @property
    def content_hash(self) -> str:
        return self.content_id.removeprefix(_DECISION_PREFIX)

    @property
    def artifact_id(self) -> str:
        return self.content_id

    @property
    def model_run_ref(self) -> str | None:
        return self.challenger_model_ref

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "promoted": self.promoted,
            "reason_codes": list(self.reason_codes),
            "policy_ref": self.policy_ref,
            "evidence_ref": self.evidence_ref,
            "challenger_model_ref": self.challenger_model_ref,
            "champion_model_ref": self.champion_model_ref,
            "rollback_artifact_ref": self.rollback_artifact_ref,
            "decided_at": None if self.decided_at is None else _timestamp(self.decided_at),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.content_id, **self.identity_payload()}


def assess_promotion(
    evidence: ChallengerEvidence,
    *,
    policy: PromotionPolicy | None,
    reference_validator: GovernanceReferenceValidator | None = None,
    decided_at: datetime | None = None,
) -> PromotionDecision:
    """Refuse promotion until operators supply a reviewed numeric policy."""

    if policy is None:
        raise ValueError("an explicit reviewed promotion policy is required")
    if not isinstance(policy, PromotionPolicy):
        raise TypeError("policy must be a PromotionPolicy")
    if not isinstance(evidence, ChallengerEvidence):
        raise TypeError("evidence must be ChallengerEvidence")
    reasons: list[str] = []
    if not evidence.is_prospective_captured:
        reasons.append("prospective_captured_cohort_required")
    required_fields = {
        "challenger_model_ref": evidence.model_run_ref,
        "champion_model_ref": evidence.champion_model_ref,
        "evaluation_ref": evidence.evaluation_ref,
        "cohort_ref": evidence.cohort_ref,
        "sample_refs": evidence.sample_refs,
        "evaluated_at": evidence.evaluated_at,
        "confidence_method": evidence.confidence_method,
        "confidence_level": evidence.confidence_level,
        "confidence_interval": evidence.confidence_interval,
        "subgroup_diagnostics": evidence.subgroup_diagnostics,
    }
    for field_name, value in required_fields.items():
        if value is None or value == ():
            reasons.append(f"missing_{field_name}")
    if evidence.sample_refs and evidence.captured_samples != len(evidence.sample_refs):
        reasons.append("captured_sample_count_mismatch")
    if evidence.subgroup_diagnostics_passed and any(
        not diagnostic.passed for diagnostic in evidence.subgroup_diagnostics
    ):
        reasons.append("subgroup_diagnostic_conflict")
    if not evidence.subgroup_diagnostics_passed:
        reasons.append("subgroup_regression")
    if policy.confidence_level is None:
        reasons.append("missing_policy_confidence_level")
    elif evidence.confidence_level is not None and not math.isclose(
        evidence.confidence_level, policy.confidence_level, rel_tol=0.0, abs_tol=1e-12
    ):
        reasons.append("confidence_level_mismatch")
    if (
        evidence.confidence_method is not None
        and evidence.confidence_method != policy.confidence_method
    ):
        reasons.append("confidence_method_mismatch")
    if evidence.confidence_method is None:
        # Already reported by ``required_fields``; this branch keeps the
        # mismatch check explicit for callers inspecting reason ordering.
        pass
    effective_decided_at = decided_at
    if effective_decided_at is None:
        reasons.append("missing_decided_at")
    else:
        require_utc(effective_decided_at, "decided_at")
        if effective_decided_at < policy.reviewed_at:
            reasons.append("decision_before_policy_review")
        if evidence.evaluated_at is not None and effective_decided_at < evidence.evaluated_at:
            reasons.append("decision_before_evaluation")
    if reference_validator is None and _evidence_references(evidence):
        reasons.append("reference_validation_required")
    elif reference_validator is not None:
        references = (
            *_evidence_references(evidence),
            policy.rollback_artifact_ref or policy.rollback_target,
        )
        for reference in references:
            try:
                if hasattr(reference_validator, "verify_reference"):
                    reference_validator.verify_reference(reference)
                elif callable(reference_validator):
                    result = reference_validator(reference)
                    if result is False:
                        raise ValueError("reference validator rejected reference")
                else:
                    raise TypeError(
                        "reference_validator must be callable or expose verify_reference"
                    )
            except (AttributeError, OSError, ValueError, KeyError, RuntimeError, TypeError):
                reasons.append("reference_unavailable")
                break
    if evidence.captured_samples < policy.minimum_captured_samples:
        reasons.append("insufficient_captured_samples")
    if evidence.observation_days < policy.minimum_observation_days:
        reasons.append("insufficient_observation_period")
    if evidence.brier_delta_vs_champion > policy.maximum_brier_delta:
        reasons.append("brier_not_improved")
    if evidence.log_loss_delta_vs_champion > policy.maximum_log_loss_delta:
        reasons.append("log_loss_not_improved")
    if not evidence.confidence_interval_passed:
        reasons.append("confidence_interval_not_improved")
    if not evidence.reliability_passed:
        reasons.append("reliability_not_improved")
    if policy.required_subgroups:
        observed = {item.subgroup for item in evidence.subgroup_diagnostics}
        missing = set(policy.required_subgroups) - observed
        if missing:
            reasons.append("missing_required_subgroup_diagnostics")
    return PromotionDecision(
        not reasons,
        tuple(reasons),
        policy_ref=policy.content_id,
        evidence_ref=evidence.content_id,
        challenger_model_ref=evidence.model_run_ref,
        champion_model_ref=evidence.champion_model_ref,
        rollback_artifact_ref=policy.rollback_artifact_ref,
        decided_at=effective_decided_at,
    )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_ref(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value or ":" not in value:
        raise ValueError(f"{field_name} must be a typed non-empty reference")


def _normalize_refs(refs: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(refs, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of references")
    try:
        values = tuple(refs)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of references") from error
    for value in values:
        _require_ref(value, field_name)
    return tuple(sorted(set(values)))


def _normalize_reason_codes(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("reason_codes must be a sequence of text")
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("reason_codes must be a sequence of text") from error
    for value in values:
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError("reason_codes must contain non-empty text")
    # Preserve evaluation order: it is useful in operator diagnostics and is
    # part of the immutable decision identity.
    return tuple(dict.fromkeys(values))


def _normalize_texts(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of text")
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of text") from error
    for value in values:
        _require_text(value, field_name)
    return tuple(sorted(set(values)))


def _normalize_diagnostics(
    diagnostics: Sequence[SubgroupDiagnostic] | Mapping[str, Any],
) -> tuple[SubgroupDiagnostic, ...]:
    if isinstance(diagnostics, Mapping):
        values: list[SubgroupDiagnostic] = []
        for subgroup, value in diagnostics.items():
            values.append(_diagnostic_from_mapping(subgroup, value))
        diagnostics = values
    try:
        values = tuple(diagnostics)
    except TypeError as error:
        raise ValueError("subgroup_diagnostics must be a sequence") from error
    normalized: list[SubgroupDiagnostic] = []
    for item in values:
        if isinstance(item, SubgroupDiagnostic):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(_diagnostic_from_mapping(item.get("subgroup"), item))
        else:
            raise TypeError("subgroup_diagnostics must contain SubgroupDiagnostic values")
    return tuple(sorted(normalized, key=lambda item: item.subgroup))


def _diagnostic_from_mapping(subgroup: Any, value: Any) -> SubgroupDiagnostic:
    if not isinstance(subgroup, str):
        raise ValueError("subgroup must be text")
    if not isinstance(value, Mapping):
        raise ValueError("subgroup diagnostics mapping values must be objects")
    sample_count = value.get("sample_count")
    brier_delta = value.get("brier_delta", value.get("brier_delta_vs_champion"))
    log_loss_delta = value.get("log_loss_delta", value.get("log_loss_delta_vs_champion"))
    passed = value.get("passed")
    if not _is_int(sample_count) or sample_count < 0:
        raise ValueError("subgroup sample_count must be a non-negative integer")
    if brier_delta is None or log_loss_delta is None:
        raise ValueError("subgroup diagnostics require brier and log-loss deltas")
    if isinstance(brier_delta, bool) or not isinstance(brier_delta, (int, float)):
        raise ValueError("subgroup brier_delta must be numeric")
    if isinstance(log_loss_delta, bool) or not isinstance(log_loss_delta, (int, float)):
        raise ValueError("subgroup log_loss_delta must be numeric")
    if not isinstance(passed, bool):
        raise ValueError("subgroup passed must be a boolean")
    return SubgroupDiagnostic(
        subgroup=subgroup,
        sample_count=sample_count,
        brier_delta=brier_delta,
        log_loss_delta=log_loss_delta,
        passed=passed,
    )


def _evidence_references(evidence: ChallengerEvidence) -> tuple[str, ...]:
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


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return prefix + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timestamp(value: datetime) -> str:
    require_utc(value, "timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
