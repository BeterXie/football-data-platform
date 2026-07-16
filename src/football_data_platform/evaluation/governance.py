"""Champion/challenger governance with an explicit reviewed policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.ids import ModelRunId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    policy_version: str
    reviewed_by: str
    reviewed_at: datetime
    confidence_method: str
    rollback_target: str
    minimum_captured_samples: int
    minimum_observation_days: int
    maximum_brier_delta: float
    maximum_log_loss_delta: float

    def __post_init__(self) -> None:
        _require_text(self.policy_version, "policy_version")
        _require_text(self.reviewed_by, "reviewed_by")
        _require_text(self.confidence_method, "confidence_method")
        ModelRunId(self.rollback_target)
        require_utc(self.reviewed_at, "reviewed_at")
        if not _is_int(self.minimum_captured_samples) or self.minimum_captured_samples < 1:
            raise ValueError("minimum_captured_samples must be positive")
        if not _is_int(self.minimum_observation_days) or self.minimum_observation_days < 1:
            raise ValueError("minimum_observation_days must be positive")
        _require_finite(self.maximum_brier_delta, "maximum_brier_delta")
        _require_finite(self.maximum_log_loss_delta, "maximum_log_loss_delta")


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


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason_codes: tuple[str, ...]


def assess_promotion(
    evidence: ChallengerEvidence,
    *,
    policy: PromotionPolicy | None,
) -> PromotionDecision:
    """Refuse promotion until operators supply a reviewed numeric policy."""

    if policy is None:
        raise ValueError("an explicit reviewed promotion policy is required")
    reasons: list[str] = []
    if evidence.capture_mode is not CaptureMode.CAPTURED:
        reasons.append("prospective_captured_cohort_required")
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
    if not evidence.subgroup_diagnostics_passed:
        reasons.append("subgroup_regression")
    return PromotionDecision(not reasons, tuple(reasons))


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


def _require_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
