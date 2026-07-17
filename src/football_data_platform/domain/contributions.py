"""Self-describing calibration inputs for expected-goals contributions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

BOUNDED_EXP_LINEAR_VERSION = "bounded-exp-linear/1"
EXPECTED_GOALS_COMPOSITION_VERSION = "expected-goals-composition/2"
MATCH_CONTEXT_CALIBRATION_VERSION = "match-context-calibration/1"
MATCH_CONTEXT_CALIBRATION_VERSION_V2 = "match-context-calibration/2"
LINEUP_DELTA_CALIBRATION_VERSION = "lineup-delta-calibration/1"

LINEUP_ATTACK_DIMENSIONS = frozenset(
    {"attack", "goals", "xg", "shots", "shots_on_target", "key_passes"}
)
LINEUP_DEFENSE_DIMENSIONS = frozenset(
    {"saves", "defensive_actions", "tackles", "interceptions", "clearances"}
)


@dataclass(frozen=True, slots=True)
class ContributionCalibrationPolicy:
    version: str
    source_feature_name: str
    source_path_template: tuple[str, ...]
    contribution_key_template: str
    coefficient_rule: str
    reference_value: float
    coefficient: float
    minimum_log_multiplier: float = -0.25
    maximum_log_multiplier: float = 0.25
    formula_version: str = BOUNDED_EXP_LINEAR_VERSION


CONTRIBUTION_CALIBRATION_POLICIES: Mapping[str, ContributionCalibrationPolicy] = MappingProxyType(
    {
        MATCH_CONTEXT_CALIBRATION_VERSION: ContributionCalibrationPolicy(
            version=MATCH_CONTEXT_CALIBRATION_VERSION,
            source_feature_name="match_context",
            source_path_template=("days_since_previous_match",),
            contribution_key_template="context:rest-days:{version}",
            coefficient_rule="symmetric",
            reference_value=5.0,
            coefficient=0.02,
        ),
        MATCH_CONTEXT_CALIBRATION_VERSION_V2: ContributionCalibrationPolicy(
            version=MATCH_CONTEXT_CALIBRATION_VERSION_V2,
            source_feature_name="match_context",
            source_path_template=("teams", "{team_id}", "rest_days"),
            contribution_key_template="context:rest-days:{team_id}:{version}",
            coefficient_rule="team-side-rest",
            reference_value=5.0,
            coefficient=0.02,
        ),
        LINEUP_DELTA_CALIBRATION_VERSION: ContributionCalibrationPolicy(
            version=LINEUP_DELTA_CALIBRATION_VERSION,
            source_feature_name="lineup_delta",
            source_path_template=("{team_id}", "dimension_deltas", "{dimension}"),
            contribution_key_template="lineup:{team_id}:{dimension}:{version}",
            coefficient_rule="team-side-dimension",
            reference_value=0.0,
            coefficient=0.05,
        ),
    }
)


class PolicyBoundContribution(Protocol):
    contribution_key: str
    source_ref: str
    version: str
    calibration: ContributionCalibration


@dataclass(frozen=True, slots=True)
class ContributionCalibration:
    """The complete formula input needed to reproduce one multiplier pair."""

    source_feature_name: str
    source_path: tuple[str, ...]
    source_value: float
    reference_value: float
    lambda_home_coefficient: float
    lambda_away_coefficient: float
    minimum_log_multiplier: float = -0.25
    maximum_log_multiplier: float = 0.25
    formula_version: str = BOUNDED_EXP_LINEAR_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("source_feature_name", self.source_feature_name),
            ("formula_version", self.formula_version),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be non-empty text")
        if not self.source_path or any(
            not isinstance(item, str) or not item or item.strip() != item
            for item in self.source_path
        ):
            raise ValueError("source_path must contain non-empty text segments")
        for name, value in (
            ("source_value", self.source_value),
            ("reference_value", self.reference_value),
            ("lambda_home_coefficient", self.lambda_home_coefficient),
            ("lambda_away_coefficient", self.lambda_away_coefficient),
            ("minimum_log_multiplier", self.minimum_log_multiplier),
            ("maximum_log_multiplier", self.maximum_log_multiplier),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.formula_version != BOUNDED_EXP_LINEAR_VERSION:
            raise ValueError(f"unsupported contribution formula {self.formula_version!r}")
        if self.minimum_log_multiplier > self.maximum_log_multiplier:
            raise ValueError("minimum_log_multiplier cannot exceed maximum_log_multiplier")
        if self.lambda_home_coefficient == 0 and self.lambda_away_coefficient == 0:
            raise ValueError("a contribution must affect at least one lambda")


def validate_generator_calibration_policy(
    version: str,
    *,
    source_feature_name: str,
    reference_value: float,
    coefficient: float,
) -> ContributionCalibrationPolicy:
    """Require generator arguments to match the immutable versioned policy."""

    policy = _calibration_policy(version)
    if (
        source_feature_name != policy.source_feature_name
        or float(reference_value) != policy.reference_value
        or float(coefficient) != policy.coefficient
    ):
        raise ValueError(f"arguments do not match calibration policy {version!r}")
    return policy


def validate_contribution_calibration_policy(
    contribution: PolicyBoundContribution,
    *,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
) -> None:
    """Bind one contribution version to its complete formula and semantic role."""

    policy = _calibration_policy(contribution.version)
    calibration = contribution.calibration
    if not isinstance(calibration, ContributionCalibration):
        raise ValueError("contribution calibration policy requires structured parameters")
    if (
        calibration.formula_version != policy.formula_version
        or calibration.source_feature_name != policy.source_feature_name
        or float(calibration.reference_value) != policy.reference_value
        or float(calibration.minimum_log_multiplier) != policy.minimum_log_multiplier
        or float(calibration.maximum_log_multiplier) != policy.maximum_log_multiplier
    ):
        raise ValueError(
            f"contribution parameters do not match calibration policy {policy.version!r}"
        )

    if policy.coefficient_rule == "symmetric":
        expected_key = policy.contribution_key_template.format(version=policy.version)
        if (
            calibration.source_path != policy.source_path_template
            or contribution.contribution_key != expected_key
            or float(calibration.lambda_home_coefficient) != policy.coefficient
            or float(calibration.lambda_away_coefficient) != policy.coefficient
        ):
            raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
        return

    if policy.coefficient_rule == "team-side-rest":
        if (
            len(calibration.source_path) != 3
            or calibration.source_path[0] != "teams"
            or calibration.source_path[2] != "rest_days"
            or not calibration.source_path[1]
        ):
            raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
        team_id = calibration.source_path[1]
        expected_path = tuple(
            segment.format(team_id=team_id) for segment in policy.source_path_template
        )
        expected_key = policy.contribution_key_template.format(
            team_id=team_id,
            version=policy.version,
        )
        actual_pair = (
            float(calibration.lambda_home_coefficient),
            float(calibration.lambda_away_coefficient),
        )
        if (
            calibration.source_path != expected_path
            or contribution.contribution_key != expected_key
            or actual_pair not in {(policy.coefficient, 0.0), (0.0, policy.coefficient)}
        ):
            raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
        if home_team_id is None and away_team_id is None:
            return
        if (
            not home_team_id
            or not away_team_id
            or home_team_id == away_team_id
            or team_id not in {home_team_id, away_team_id}
        ):
            raise ValueError("context calibration policy requires the canonical match teams")
        expected_pair = (
            (policy.coefficient, 0.0) if team_id == home_team_id else (0.0, policy.coefficient)
        )
        if actual_pair != expected_pair:
            raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
        return

    team_id, dimension = _lineup_source_path(calibration.source_path)
    if policy.coefficient_rule != "team-side-dimension":
        raise ValueError(f"unsupported coefficient rule {policy.coefficient_rule!r}")
    expected_path = tuple(
        segment.format(team_id=team_id, dimension=dimension)
        for segment in policy.source_path_template
    )
    expected_key = policy.contribution_key_template.format(
        team_id=team_id,
        dimension=dimension,
        version=policy.version,
    )
    if calibration.source_path != expected_path:
        raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
    if contribution.contribution_key != expected_key:
        raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
    allowed_pairs = _lineup_coefficient_pairs(dimension, policy.coefficient)
    actual_pair = (
        float(calibration.lambda_home_coefficient),
        float(calibration.lambda_away_coefficient),
    )
    if actual_pair not in allowed_pairs:
        raise ValueError(f"contribution does not match calibration policy {policy.version!r}")
    if home_team_id is None and away_team_id is None:
        return
    if (
        not home_team_id
        or not away_team_id
        or home_team_id == away_team_id
        or team_id not in {home_team_id, away_team_id}
    ):
        raise ValueError("lineup calibration policy requires the canonical match teams")
    expected_pair = _lineup_coefficient_pair(
        team_id,
        dimension,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        coefficient=policy.coefficient,
    )
    if actual_pair != expected_pair:
        raise ValueError(f"contribution does not match calibration policy {policy.version!r}")


def validate_snapshot_contribution_policy(
    snapshot: Any,
    contributions: tuple[PolicyBoundContribution, ...],
) -> None:
    """Require the exact policy-defined contribution set for one verified snapshot."""

    home_team_id = snapshot.home_team_id.value
    away_team_id = snapshot.away_team_id.value
    actual_by_key: dict[str, PolicyBoundContribution] = {}
    for contribution in contributions:
        validate_contribution_calibration_policy(
            contribution,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        if contribution.contribution_key in actual_by_key:
            raise ValueError("formal prediction contribution set contains duplicate keys")
        actual_by_key[contribution.contribution_key] = contribution

    context_features = [feature for feature in snapshot.features if feature.name == "match_context"]
    if len(context_features) != 1:
        raise ValueError("formal prediction contribution set requires one match_context")
    context_feature = context_features[0]
    expected: dict[str, tuple[str, tuple[str, ...]]] = {}
    if snapshot.feature_spec_version == "prematch-features/3":
        context_policy = _calibration_policy(MATCH_CONTEXT_CALIBRATION_VERSION_V2)
        teams = (
            context_feature.value.get("teams")
            if isinstance(context_feature.value, Mapping)
            else None
        )
        if not isinstance(teams, Mapping):
            raise ValueError("formal context policy requires per-team context evidence")
        for team_id in (home_team_id, away_team_id):
            item = teams.get(team_id)
            if not isinstance(item, Mapping) or item.get("status") != "available":
                raise ValueError("formal context policy requires available rest for both teams")
            key = context_policy.contribution_key_template.format(
                team_id=team_id,
                version=context_policy.version,
            )
            path = tuple(
                segment.format(team_id=team_id) for segment in context_policy.source_path_template
            )
            expected[key] = (context_feature.source_ref, path)
    else:
        context_policy = _calibration_policy(MATCH_CONTEXT_CALIBRATION_VERSION)
        context_key = context_policy.contribution_key_template.format(
            version=context_policy.version
        )
        expected[context_key] = (
            context_feature.source_ref,
            context_policy.source_path_template,
        )

    lineup_features = [feature for feature in snapshot.features if feature.name == "lineup_delta"]
    if len(lineup_features) > 1:
        raise ValueError("formal prediction contribution set allows one lineup_delta")
    if lineup_features:
        lineup_feature = lineup_features[0]
        if not isinstance(lineup_feature.value, Mapping):
            raise ValueError("lineup calibration policy requires a team-keyed mapping")
        for team_id, item in lineup_feature.value.items():
            if not isinstance(item, Mapping) or item.get("quality_status") != "ready":
                continue
            if team_id not in {home_team_id, away_team_id}:
                raise ValueError("lineup calibration policy contains an unknown match team")
            dimensions = item.get("dimension_deltas")
            if not isinstance(dimensions, Mapping) or not dimensions:
                raise ValueError("ready lineup calibration policy requires dimensions")
            for dimension in dimensions:
                _lineup_dimension_kind(dimension)
                lineup_policy = _calibration_policy(LINEUP_DELTA_CALIBRATION_VERSION)
                key = lineup_policy.contribution_key_template.format(
                    team_id=team_id,
                    dimension=dimension,
                    version=lineup_policy.version,
                )
                source_path = tuple(
                    segment.format(team_id=team_id, dimension=dimension)
                    for segment in lineup_policy.source_path_template
                )
                expected[key] = (
                    lineup_feature.source_ref,
                    source_path,
                )

    if set(actual_by_key) != set(expected):
        raise ValueError("formal prediction contribution set does not match the calibration policy")
    features_by_ref = {feature.source_ref: feature for feature in snapshot.features}
    for key, (source_ref, source_path) in expected.items():
        contribution = actual_by_key[key]
        if contribution.source_ref != source_ref:
            raise ValueError(
                "expected-goals contribution source refs must cite verified "
                "snapshot feature sources"
            )
        if contribution.calibration.source_path != source_path:
            raise ValueError(
                "formal prediction contribution set does not match the calibration policy"
            )
        feature = features_by_ref.get(source_ref)
        if feature is None:
            raise ValueError("calibration policy source is not a snapshot feature")
        verify_calibration_source(
            contribution.calibration,
            feature_name=feature.name,
            feature_value=feature.value,
        )


def calibrated_multipliers(calibration: ContributionCalibration) -> tuple[float, float]:
    """Recompute the multiplier pair from the persisted formula inputs."""

    if not isinstance(calibration, ContributionCalibration):
        raise TypeError("calibration must be a ContributionCalibration")
    delta = calibration.source_value - calibration.reference_value
    return (
        _bounded_exp(calibration.lambda_home_coefficient * delta, calibration),
        _bounded_exp(calibration.lambda_away_coefficient * delta, calibration),
    )


def verify_calibration_source(
    calibration: ContributionCalibration,
    *,
    feature_name: str,
    feature_value: Any,
) -> None:
    """Bind persisted calibration input to the cited snapshot feature value."""

    if feature_name != calibration.source_feature_name:
        raise ValueError("contribution calibration cites the wrong snapshot feature type")
    value = feature_value
    for segment in calibration.source_path:
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError("contribution calibration source_path is unavailable")
        value = value[segment]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("contribution calibration source value must be numeric")
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), calibration.source_value, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("contribution calibration source value does not match the snapshot")


def contribution_calibration_payload(calibration: ContributionCalibration) -> dict[str, Any]:
    if not isinstance(calibration, ContributionCalibration):
        raise TypeError("calibration must be a ContributionCalibration")
    return {
        "formula_version": calibration.formula_version,
        "source_feature_name": calibration.source_feature_name,
        "source_path": list(calibration.source_path),
        "source_value": calibration.source_value,
        "reference_value": calibration.reference_value,
        "lambda_home_coefficient": calibration.lambda_home_coefficient,
        "lambda_away_coefficient": calibration.lambda_away_coefficient,
        "minimum_log_multiplier": calibration.minimum_log_multiplier,
        "maximum_log_multiplier": calibration.maximum_log_multiplier,
    }


def parse_contribution_calibration(payload: Any) -> ContributionCalibration:
    fields = {
        "formula_version",
        "source_feature_name",
        "source_path",
        "source_value",
        "reference_value",
        "lambda_home_coefficient",
        "lambda_away_coefficient",
        "minimum_log_multiplier",
        "maximum_log_multiplier",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("contribution calibration fields are invalid")
    source_path = payload["source_path"]
    if not isinstance(source_path, list) or any(not isinstance(item, str) for item in source_path):
        raise ValueError("contribution calibration source_path must be text")
    return ContributionCalibration(
        formula_version=payload["formula_version"],
        source_feature_name=payload["source_feature_name"],
        source_path=tuple(source_path),
        source_value=payload["source_value"],
        reference_value=payload["reference_value"],
        lambda_home_coefficient=payload["lambda_home_coefficient"],
        lambda_away_coefficient=payload["lambda_away_coefficient"],
        minimum_log_multiplier=payload["minimum_log_multiplier"],
        maximum_log_multiplier=payload["maximum_log_multiplier"],
    )


def _bounded_exp(value: float, calibration: ContributionCalibration) -> float:
    bounded = max(
        calibration.minimum_log_multiplier,
        min(calibration.maximum_log_multiplier, value),
    )
    return math.exp(bounded)


def _calibration_policy(version: str) -> ContributionCalibrationPolicy:
    policy = CONTRIBUTION_CALIBRATION_POLICIES.get(version)
    if policy is None:
        raise ValueError(f"unsupported contribution calibration policy {version!r}")
    return policy


def _lineup_source_path(source_path: tuple[str, ...]) -> tuple[str, str]:
    if len(source_path) != 3 or not source_path[0] or source_path[1] != "dimension_deltas":
        raise ValueError("lineup contribution source_path violates its calibration policy")
    team_id, _, dimension = source_path
    _lineup_dimension_kind(dimension)
    return team_id, dimension


def _lineup_dimension_kind(dimension: object) -> str:
    if dimension in LINEUP_ATTACK_DIMENSIONS:
        return "attack"
    if dimension in LINEUP_DEFENSE_DIMENSIONS:
        return "defense"
    raise ValueError(f"unsupported lineup calibration policy dimension {dimension!r}")


def _lineup_coefficient_pairs(
    dimension: str,
    coefficient: float,
) -> frozenset[tuple[float, float]]:
    if _lineup_dimension_kind(dimension) == "attack":
        return frozenset(((coefficient, 0.0), (0.0, coefficient)))
    return frozenset(((0.0, -coefficient), (-coefficient, 0.0)))


def _lineup_coefficient_pair(
    team_id: str,
    dimension: str,
    *,
    home_team_id: str,
    away_team_id: str,
    coefficient: float,
) -> tuple[float, float]:
    kind = _lineup_dimension_kind(dimension)
    if kind == "attack":
        return (coefficient, 0.0) if team_id == home_team_id else (0.0, coefficient)
    return (0.0, -coefficient) if team_id == home_team_id else (-coefficient, 0.0)
