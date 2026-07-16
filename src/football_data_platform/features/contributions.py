"""Compose baseline, lineup, and context effects without double counting.

The football model has one expected-goals coordinate.  Derived features may
contribute to that coordinate only through an explicitly versioned multiplier
and a unique contribution key.  This module deliberately does not infer a
multiplier from market odds or silently turn missing values into neutral
values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

EXPECTED_GOALS_COMPOSITION_VERSION = "expected-goals-composition/1"
EXPECTED_GOALS_CONTRIBUTION_VERSION = "expected-goals-contribution/1"
LINEUP_CALIBRATION_VERSION = "lineup-delta-calibration/1"
CONTEXT_CALIBRATION_VERSION = "match-context-calibration/1"


@dataclass(frozen=True, slots=True)
class ExpectedGoalsContribution:
    contribution_key: str
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    source_ref: str
    version: str = EXPECTED_GOALS_CONTRIBUTION_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_home_multiplier", self.lambda_home_multiplier),
            ("lambda_away_multiplier", self.lambda_away_multiplier),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.contribution_key or self.contribution_key.strip() != self.contribution_key:
            raise ValueError("contribution_key must be non-empty text")
        if not self.source_ref or self.source_ref.strip() != self.source_ref:
            raise ValueError("source_ref must be non-empty text")
        if not self.version or self.version.strip() != self.version:
            raise ValueError("version must be non-empty text")


@dataclass(frozen=True, slots=True)
class ExpectedGoals:
    lambda_home: float
    lambda_away: float
    contribution_keys: tuple[str, ...]
    input_refs: tuple[str, ...]
    baseline_lambda_home: float = field(default=0.0, compare=True)
    baseline_lambda_away: float = field(default=0.0, compare=True)
    composition_version: str = EXPECTED_GOALS_COMPOSITION_VERSION
    contributions: tuple[ExpectedGoalsContribution, ...] = ()
    calibration_versions: tuple[str, ...] = ()


def apply_expected_goals_contributions(
    lambda_home: float,
    lambda_away: float,
    contributions: tuple[ExpectedGoalsContribution, ...],
) -> ExpectedGoals:
    """Apply each auditable effect exactly once.

    Contributions are sorted by key before application so equivalent feature
    sets produce the same artifact regardless of caller ordering.  Multipliers
    are applied in log-space conceptually (ordinary multiplication is used for
    the small bounded factors) and every applied key/source remains in the
    returned lineage.
    """

    if (
        not math.isfinite(lambda_home)
        or not math.isfinite(lambda_away)
        or lambda_home <= 0
        or lambda_away <= 0
    ):
        raise ValueError("baseline expected goals must be positive")
    ordered = tuple(sorted(contributions, key=lambda item: item.contribution_key))
    keys = [contribution.contribution_key for contribution in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate expected-goals contribution key")
    for contribution in ordered:
        lambda_home *= contribution.lambda_home_multiplier
        lambda_away *= contribution.lambda_away_multiplier
        if not math.isfinite(lambda_home) or not math.isfinite(lambda_away):
            raise ValueError("expected-goals composition overflowed finite range")
    return ExpectedGoals(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        contribution_keys=tuple(keys),
        input_refs=tuple(sorted({item.source_ref for item in ordered})),
        baseline_lambda_home=lambda_home
        / math.prod(item.lambda_home_multiplier for item in ordered),
        baseline_lambda_away=lambda_away
        / math.prod(item.lambda_away_multiplier for item in ordered),
        contributions=ordered,
    )


def compose_expected_goals(
    baseline_lambda_home: float,
    baseline_lambda_away: float,
    contributions: tuple[ExpectedGoalsContribution, ...] = (),
    *,
    calibration_versions: tuple[str, ...] = (),
) -> ExpectedGoals:
    """Return the single model coordinate after versioned contributions.

    This named entry point makes the baseline/contribution boundary explicit;
    ``apply_expected_goals_contributions`` remains as a compatibility alias.
    """

    result = apply_expected_goals_contributions(
        baseline_lambda_home, baseline_lambda_away, contributions
    )
    return ExpectedGoals(
        lambda_home=result.lambda_home,
        lambda_away=result.lambda_away,
        contribution_keys=result.contribution_keys,
        input_refs=result.input_refs,
        baseline_lambda_home=baseline_lambda_home,
        baseline_lambda_away=baseline_lambda_away,
        composition_version=EXPECTED_GOALS_COMPOSITION_VERSION,
        contributions=result.contributions,
        calibration_versions=tuple(sorted(set(calibration_versions))),
    )


def lineup_delta_contributions(
    deltas: Mapping[str, Mapping[str, Any]],
    *,
    home_team_id: str,
    away_team_id: str,
    source_ref: str,
    coefficient: float = 0.05,
    version: str = LINEUP_CALIBRATION_VERSION,
) -> tuple[ExpectedGoalsContribution, ...]:
    """Translate ready lineup dimensions into bounded lambda multipliers.

    Only ``quality_status == 'ready'`` entries contribute.  Preview or missing
    lineup evidence is intentionally omitted and must remain visible to the
    caller.  Attack-like dimensions affect the same team's scoring rate;
    defense-like dimensions affect the opponent's rate in the opposite
    direction.  Unknown dimensions fail closed instead of being guessed.
    """

    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and positive")
    if not home_team_id or not away_team_id or home_team_id == away_team_id:
        raise ValueError("lineup contribution requires distinct match teams")
    _require_version(version)
    _require_source(source_ref)
    contributions: list[ExpectedGoalsContribution] = []
    attack_dimensions = {"attack", "goals", "xg", "shots", "shots_on_target", "key_passes"}
    defense_dimensions = {
        "saves",
        "defensive_actions",
        "tackles",
        "interceptions",
        "clearances",
    }
    for team_id, payload in sorted(deltas.items()):
        if team_id not in {home_team_id, away_team_id}:
            raise ValueError(f"lineup delta contains unknown team {team_id!r}")
        if not isinstance(payload, Mapping):
            raise ValueError("lineup delta payload must be a mapping")
        if payload.get("quality_status") != "ready":
            continue
        dimensions = payload.get("dimension_deltas")
        if not isinstance(dimensions, Mapping) or not dimensions:
            raise ValueError("ready lineup delta must contain dimensions")
        for dimension, raw_delta in sorted(dimensions.items()):
            if not isinstance(dimension, str) or not dimension:
                raise ValueError("lineup dimension must be non-empty text")
            if not isinstance(raw_delta, (int, float)) or isinstance(raw_delta, bool):
                raise ValueError(f"lineup dimension {dimension!r} must be numeric")
            if not math.isfinite(float(raw_delta)):
                raise ValueError(f"lineup dimension {dimension!r} must be finite")
            if dimension in attack_dimensions:
                home_multiplier, away_multiplier = (1.0, 1.0)
                if team_id == home_team_id:
                    home_multiplier = _bounded_exp(coefficient * float(raw_delta))
                else:
                    away_multiplier = _bounded_exp(coefficient * float(raw_delta))
            elif dimension in defense_dimensions:
                home_multiplier, away_multiplier = (1.0, 1.0)
                if team_id == home_team_id:
                    away_multiplier = _bounded_exp(-coefficient * float(raw_delta))
                else:
                    home_multiplier = _bounded_exp(-coefficient * float(raw_delta))
            else:
                raise ValueError(f"unsupported lineup contribution dimension {dimension!r}")
            contributions.append(
                ExpectedGoalsContribution(
                    contribution_key=f"lineup:{team_id}:{dimension}:{version}",
                    lambda_home_multiplier=home_multiplier,
                    lambda_away_multiplier=away_multiplier,
                    source_ref=source_ref,
                    version=version,
                )
            )
    return tuple(contributions)


def context_contribution(
    context: Mapping[str, Any],
    *,
    source_ref: str,
    coefficient: float = 0.02,
    reference_rest_days: float = 5.0,
    version: str = CONTEXT_CALIBRATION_VERSION,
) -> ExpectedGoalsContribution:
    """Build the versioned match-context multiplier used by the score model.

    The first context contract exposes rest days only.  It is a bounded common
    pace adjustment, with the reference value yielding a neutral multiplier.
    Future context fields must receive their own contribution key and
    calibration rather than being silently folded into this one.
    """

    _require_version(version)
    _require_source(source_ref)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and positive")
    if not math.isfinite(reference_rest_days) or reference_rest_days < 0:
        raise ValueError("reference_rest_days must be finite and non-negative")
    raw_days = context.get("days_since_previous_match")
    if not isinstance(raw_days, (int, float)) or isinstance(raw_days, bool):
        raise ValueError("context requires numeric days_since_previous_match")
    if not math.isfinite(float(raw_days)) or float(raw_days) < 0:
        raise ValueError("days_since_previous_match must be finite and non-negative")
    multiplier = _bounded_exp(coefficient * (float(raw_days) - reference_rest_days))
    return ExpectedGoalsContribution(
        contribution_key=f"context:rest-days:{version}",
        lambda_home_multiplier=multiplier,
        lambda_away_multiplier=multiplier,
        source_ref=source_ref,
        version=version,
    )


def _bounded_exp(value: float) -> float:
    # Keep the first calibrated layer intentionally conservative and finite.
    return math.exp(max(-0.25, min(0.25, value)))


def _require_version(value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError("version must be non-empty text")


def _require_source(value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError("source_ref must be non-empty text")
