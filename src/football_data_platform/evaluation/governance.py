"""Champion/challenger governance with an explicit reviewed policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    minimum_captured_samples: int
    minimum_observation_days: int
    maximum_brier_delta: float
    maximum_log_loss_delta: float

    def __post_init__(self) -> None:
        if self.minimum_captured_samples < 1:
            raise ValueError("minimum_captured_samples must be positive")
        if self.minimum_observation_days < 1:
            raise ValueError("minimum_observation_days must be positive")


@dataclass(frozen=True, slots=True)
class ChallengerEvidence:
    captured_samples: int
    observation_days: int
    brier_delta_vs_champion: float
    log_loss_delta_vs_champion: float
    reliability_passed: bool
    subgroup_diagnostics_passed: bool


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
    if evidence.captured_samples < policy.minimum_captured_samples:
        reasons.append("insufficient_captured_samples")
    if evidence.observation_days < policy.minimum_observation_days:
        reasons.append("insufficient_observation_period")
    if evidence.brier_delta_vs_champion > policy.maximum_brier_delta:
        reasons.append("brier_not_improved")
    if evidence.log_loss_delta_vs_champion > policy.maximum_log_loss_delta:
        reasons.append("log_loss_not_improved")
    if not evidence.reliability_passed:
        reasons.append("reliability_not_improved")
    if not evidence.subgroup_diagnostics_passed:
        reasons.append("subgroup_regression")
    return PromotionDecision(not reasons, tuple(reasons))
