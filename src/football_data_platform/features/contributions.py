"""Compose baseline, lineup, and context effects without double counting."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedGoalsContribution:
    contribution_key: str
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    source_ref: str

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_home_multiplier", self.lambda_home_multiplier),
            ("lambda_away_multiplier", self.lambda_away_multiplier),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExpectedGoals:
    lambda_home: float
    lambda_away: float
    contribution_keys: tuple[str, ...]
    input_refs: tuple[str, ...]


def apply_expected_goals_contributions(
    lambda_home: float,
    lambda_away: float,
    contributions: tuple[ExpectedGoalsContribution, ...],
) -> ExpectedGoals:
    """Apply each auditable effect exactly once."""

    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("baseline expected goals must be positive")
    keys = [contribution.contribution_key for contribution in contributions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate expected-goals contribution key")
    for contribution in contributions:
        lambda_home *= contribution.lambda_home_multiplier
        lambda_away *= contribution.lambda_away_multiplier
    return ExpectedGoals(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        contribution_keys=tuple(keys),
        input_refs=tuple(sorted({item.source_ref for item in contributions})),
    )
