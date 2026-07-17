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
from typing import Any, Protocol

from football_data_platform.domain.contributions import (
    EXPECTED_GOALS_COMPOSITION_VERSION,
    LINEUP_ATTACK_DIMENSIONS,
    LINEUP_DEFENSE_DIMENSIONS,
    LINEUP_DELTA_CALIBRATION_VERSION,
    MATCH_CONTEXT_CALIBRATION_VERSION,
    ContributionCalibration,
    calibrated_multipliers,
    validate_contribution_calibration_policy,
    validate_generator_calibration_policy,
)
from football_data_platform.features.lineup import LINEUP_DELTA_INPUT_TRANSFORM_V3

EXPECTED_GOALS_CONTRIBUTION_VERSION = "expected-goals-contribution/2"
LINEUP_CALIBRATION_VERSION = LINEUP_DELTA_CALIBRATION_VERSION
CONTEXT_CALIBRATION_VERSION = MATCH_CONTEXT_CALIBRATION_VERSION


class LineupDeltaSourceValidator(Protocol):
    def validate_snapshot_source(self, source_ref: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ExpectedGoalsContribution:
    contribution_key: str
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    source_ref: str
    calibration: ContributionCalibration
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
        validate_contribution_calibration_policy(self)
        expected_home, expected_away = calibrated_multipliers(self.calibration)
        if not math.isclose(
            self.lambda_home_multiplier, expected_home, rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            self.lambda_away_multiplier, expected_away, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("contribution multipliers do not match calibration parameters")


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
        validate_contribution_calibration_policy(contribution)
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
    contribution_versions = tuple(sorted({item.version for item in result.contributions}))
    requested_versions = tuple(sorted(set(calibration_versions)))
    if requested_versions and requested_versions != contribution_versions:
        raise ValueError("calibration_versions must match contribution calibration versions")
    return ExpectedGoals(
        lambda_home=result.lambda_home,
        lambda_away=result.lambda_away,
        contribution_keys=result.contribution_keys,
        input_refs=result.input_refs,
        baseline_lambda_home=baseline_lambda_home,
        baseline_lambda_away=baseline_lambda_away,
        composition_version=EXPECTED_GOALS_COMPOSITION_VERSION,
        contributions=result.contributions,
        calibration_versions=contribution_versions,
    )


def lineup_delta_contributions(
    deltas: Mapping[str, Mapping[str, Any]],
    *,
    home_team_id: str,
    away_team_id: str,
    source_ref: str,
    source_validator: LineupDeltaSourceValidator | None = None,
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
    policy = validate_generator_calibration_policy(
        version,
        source_feature_name="lineup_delta",
        reference_value=0.0,
        coefficient=coefficient,
    )
    ready_payloads = tuple(
        payload
        for payload in deltas.values()
        if isinstance(payload, Mapping) and payload.get("quality_status") == "ready"
    )
    if ready_payloads:
        if source_validator is None:
            raise ValueError("ready lineup delta requires a persisted source validator")
        validation = source_validator.validate_snapshot_source(source_ref)
        if (
            validation.source_ref != source_ref
            or validation.source_kind != "derived"
            or validation.transform_version != LINEUP_DELTA_INPUT_TRANSFORM_V3
            or validation.value != dict(deltas)
        ):
            raise ValueError("ready lineup delta does not match its validated source")
    contributions: list[ExpectedGoalsContribution] = []
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
            if dimension in LINEUP_ATTACK_DIMENSIONS:
                home_coefficient, away_coefficient = (0.0, 0.0)
                if team_id == home_team_id:
                    home_coefficient = coefficient
                else:
                    away_coefficient = coefficient
            elif dimension in LINEUP_DEFENSE_DIMENSIONS:
                home_coefficient, away_coefficient = (0.0, 0.0)
                if team_id == home_team_id:
                    away_coefficient = -coefficient
                else:
                    home_coefficient = -coefficient
            else:
                raise ValueError(f"unsupported lineup contribution dimension {dimension!r}")
            calibration = ContributionCalibration(
                source_feature_name="lineup_delta",
                source_path=(team_id, "dimension_deltas", dimension),
                source_value=float(raw_delta),
                reference_value=0.0,
                lambda_home_coefficient=home_coefficient,
                lambda_away_coefficient=away_coefficient,
                minimum_log_multiplier=policy.minimum_log_multiplier,
                maximum_log_multiplier=policy.maximum_log_multiplier,
                formula_version=policy.formula_version,
            )
            home_multiplier, away_multiplier = calibrated_multipliers(calibration)
            contributions.append(
                ExpectedGoalsContribution(
                    contribution_key=f"lineup:{team_id}:{dimension}:{version}",
                    lambda_home_multiplier=home_multiplier,
                    lambda_away_multiplier=away_multiplier,
                    source_ref=source_ref,
                    calibration=calibration,
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
    policy = validate_generator_calibration_policy(
        version,
        source_feature_name="match_context",
        reference_value=reference_rest_days,
        coefficient=coefficient,
    )
    raw_days = context.get("days_since_previous_match")
    if not isinstance(raw_days, (int, float)) or isinstance(raw_days, bool):
        raise ValueError("context requires numeric days_since_previous_match")
    if not math.isfinite(float(raw_days)) or float(raw_days) < 0:
        raise ValueError("days_since_previous_match must be finite and non-negative")
    calibration = ContributionCalibration(
        source_feature_name="match_context",
        source_path=("days_since_previous_match",),
        source_value=float(raw_days),
        reference_value=reference_rest_days,
        lambda_home_coefficient=coefficient,
        lambda_away_coefficient=coefficient,
        minimum_log_multiplier=policy.minimum_log_multiplier,
        maximum_log_multiplier=policy.maximum_log_multiplier,
        formula_version=policy.formula_version,
    )
    home_multiplier, away_multiplier = calibrated_multipliers(calibration)
    return ExpectedGoalsContribution(
        contribution_key=f"context:rest-days:{version}",
        lambda_home_multiplier=home_multiplier,
        lambda_away_multiplier=away_multiplier,
        source_ref=source_ref,
        calibration=calibration,
        version=version,
    )


def _require_version(value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError("version must be non-empty text")


def _require_source(value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError("source_ref must be non-empty text")
