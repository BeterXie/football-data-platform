from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from football_data_platform.domain.contributions import calibrated_multipliers
from football_data_platform.domain.ids import MatchId, ModelRunId, TeamId
from football_data_platform.domain.predictions import (
    build_score_prediction,
    parse_prediction_payload,
    prediction_payload,
)
from football_data_platform.domain.snapshots import (
    SnapshotFeature,
    SnapshotSourceValidation,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.features.contributions import (
    ExpectedGoalsContribution,
    compose_expected_goals,
    context_contribution,
    lineup_delta_contributions,
)

AS_OF = datetime(2025, 8, 22, 14, 0, tzinfo=UTC)
KICKOFF = AS_OF + timedelta(hours=24)
HOME = TeamId("team:calibration-home")
AWAY = TeamId("team:calibration-away")
BASELINE_REF = "derived-source:" + "a" * 64
CONTEXT_REF = "derived-source:" + "b" * 64
LINEUP_REF = "derived-source:" + "c" * 64
RAW_REF = "raw-asset:" + "d" * 64
HOME_LINEUP_REF = "derived-source:" + "e" * 64
AWAY_LINEUP_REF = "derived-source:" + "f" * 64
HOME_RAW_REF = "raw-asset:" + "1" * 64
AWAY_RAW_REF = "raw-asset:" + "2" * 64


class _SourceValidator:
    def __init__(self, values: dict[str, SnapshotSourceValidation]) -> None:
        self.values = values

    def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation:
        return self.values[source_ref]


def _source(
    source_ref: str,
    transform_version: str,
    value: object,
    *,
    input_refs: tuple[str, ...] = (RAW_REF,),
    source_context: dict[str, object] | None = None,
) -> SnapshotSourceValidation:
    return SnapshotSourceValidation(
        source_ref=source_ref,
        source_kind="derived",
        transform_version=transform_version,
        observed_at=AS_OF,
        value=value,
        input_refs=input_refs,
        known_at=AS_OF,
        source_context=source_context,
    )


def _snapshot():
    match_id = MatchId("match:calibration-policy")
    baseline = {
        "artifact_id": "team-baseline:" + "9" * 64,
        "lambda_home": 1.7,
        "lambda_away": 0.8,
    }
    context = {"days_since_previous_match": 6.0}
    home_players = [f"player:calibration-home-{index}" for index in range(11)]
    away_players = [f"player:calibration-away-{index}" for index in range(11)]
    lineup = {
        HOME.value: {
            "quality_status": "ready",
            "dimension_deltas": {"attack": 1.0, "saves": 2.0},
            "missing_fields": [],
            "input_refs": ["player-profile:" + "3" * 64],
            "not_applicable_fields": [],
            "blocking_reasons": [],
            "transform_version": "lineup-delta/2",
            "role_contract_versions": ["lineup-role-metrics/1"],
            "dimension_sample_sizes": {"attack": [1, 1], "saves": [1, 1]},
            "starter_ids": home_players,
            "lineup_input_refs": [HOME_RAW_REF],
        },
        AWAY.value: {
            "quality_status": "ready",
            "dimension_deltas": {"attack": -0.5},
            "missing_fields": [],
            "input_refs": ["player-profile:" + "4" * 64],
            "not_applicable_fields": [],
            "blocking_reasons": [],
            "transform_version": "lineup-delta/2",
            "role_contract_versions": ["lineup-role-metrics/1"],
            "dimension_sample_sizes": {"attack": [1, 1]},
            "starter_ids": away_players,
            "lineup_input_refs": [AWAY_RAW_REF],
        },
    }
    validator = _SourceValidator(
        {
            BASELINE_REF: _source(BASELINE_REF, "team-baseline-input/2", baseline),
            CONTEXT_REF: _source(CONTEXT_REF, "match-context-input/1", context),
            LINEUP_REF: _source(LINEUP_REF, "lineup-delta-input/3", lineup),
            HOME_LINEUP_REF: _source(
                HOME_LINEUP_REF,
                "official-lineup-input/2",
                home_players,
                input_refs=(HOME_RAW_REF,),
                source_context={
                    "match_id": match_id.value,
                    "match_version": 1,
                    "team_id": HOME.value,
                    "player_ids": home_players,
                },
            ),
            AWAY_LINEUP_REF: _source(
                AWAY_LINEUP_REF,
                "official-lineup-input/2",
                away_players,
                input_refs=(AWAY_RAW_REF,),
                source_context={
                    "match_id": match_id.value,
                    "match_version": 1,
                    "team_id": AWAY.value,
                    "player_ids": away_players,
                },
            ),
        }
    )
    snapshot = build_snapshot(
        match_id=match_id,
        match_version=1,
        snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
        as_of=AS_OF,
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="prematch-features/2",
        features=(
            SnapshotFeature("team_baseline", baseline, AS_OF, BASELINE_REF, "baseline"),
            SnapshotFeature("match_context", context, AS_OF, CONTEXT_REF, "context"),
            SnapshotFeature("lineup_delta", lineup, AS_OF, LINEUP_REF, "lineup"),
            SnapshotFeature(
                "official_lineup_confirmed",
                home_players,
                AS_OF,
                HOME_LINEUP_REF,
                "official-home",
                entity_id=HOME.value,
            ),
            SnapshotFeature(
                "official_lineup_confirmed",
                away_players,
                AS_OF,
                AWAY_LINEUP_REF,
                "official-away",
                entity_id=AWAY.value,
            ),
        ),
        home_team_id=HOME,
        away_team_id=AWAY,
        source_validator=validator,
    )
    return snapshot, validator


def _valid_contributions(snapshot, validator):
    context = next(feature for feature in snapshot.features if feature.name == "match_context")
    lineup = next(feature for feature in snapshot.features if feature.name == "lineup_delta")
    return (
        context_contribution(context.value, source_ref=context.source_ref),
        *lineup_delta_contributions(
            lineup.value,
            home_team_id=HOME.value,
            away_team_id=AWAY.value,
            source_ref=lineup.source_ref,
            source_validator=validator,
        ),
    )


def _build(snapshot, validator, contributions):
    baseline = next(feature for feature in snapshot.features if feature.name == "team_baseline")
    composition = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        tuple(contributions),
    )
    return build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=validator,
        model_run_id=ModelRunId("model-run:calibration-policy"),
        model_version="dixon-coles/calibration-policy",
        generated_at=AS_OF,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.1,
        max_goals=11,
        input_refs=(),
        expected_goals=composition,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lambda_home_coefficient", 0.03),
        ("reference_value", 4.0),
        ("maximum_log_multiplier", 0.1),
    ),
)
def test_same_policy_version_rejects_arbitrary_parameters(field: str, value: float) -> None:
    valid = context_contribution({"days_since_previous_match": 6.0}, source_ref=CONTEXT_REF)
    calibration = replace(valid.calibration, **{field: value})
    home_multiplier, away_multiplier = calibrated_multipliers(calibration)

    with pytest.raises(ValueError, match="calibration policy"):
        ExpectedGoalsContribution(
            contribution_key=valid.contribution_key,
            lambda_home_multiplier=home_multiplier,
            lambda_away_multiplier=away_multiplier,
            source_ref=valid.source_ref,
            calibration=calibration,
            version=valid.version,
        )


@pytest.mark.parametrize(
    "feature_name",
    ("team_baseline", "official_lineup_confirmed", "unsupported_feature"),
)
def test_policy_rejects_unsupported_feature_contributions(feature_name: str) -> None:
    valid = context_contribution({"days_since_previous_match": 6.0}, source_ref=CONTEXT_REF)
    calibration = replace(
        valid.calibration,
        source_feature_name=feature_name,
        source_path=("lambda_home",),
        source_value=1.7,
    )
    home_multiplier, away_multiplier = calibrated_multipliers(calibration)

    with pytest.raises(ValueError, match="calibration policy"):
        ExpectedGoalsContribution(
            contribution_key="baseline:double-count:forged/1",
            lambda_home_multiplier=home_multiplier,
            lambda_away_multiplier=away_multiplier,
            source_ref=BASELINE_REF,
            calibration=calibration,
            version=valid.version,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"coefficient": 0.03},
        {"reference_rest_days": 4.0},
        {"version": "match-context-calibration/2"},
    ),
)
def test_context_generator_rejects_unregistered_policy_parameters(
    overrides: dict[str, float | str],
) -> None:
    with pytest.raises(ValueError, match="calibration policy"):
        context_contribution(
            {"days_since_previous_match": 6.0},
            source_ref=CONTEXT_REF,
            **overrides,
        )


def test_formal_prediction_requires_match_context_exactly_once() -> None:
    snapshot, validator = _snapshot()
    contributions = _valid_contributions(snapshot, validator)

    with pytest.raises(ValueError, match="contribution set"):
        _build(snapshot, validator, contributions[1:])


def test_formal_prediction_rejects_missing_ready_lineup_dimension() -> None:
    snapshot, validator = _snapshot()
    contributions = _valid_contributions(snapshot, validator)

    with pytest.raises(ValueError, match="contribution set"):
        _build(snapshot, validator, contributions[:-1])


def test_policy_rejects_forged_lineup_contribution_key() -> None:
    snapshot, validator = _snapshot()
    contributions = _valid_contributions(snapshot, validator)
    valid_lineup = contributions[1]

    with pytest.raises(ValueError, match="calibration policy"):
        replace(valid_lineup, contribution_key="lineup:extra:forged/1")


def test_formal_prediction_rejects_policy_valid_extra_lineup_dimension() -> None:
    snapshot, validator = _snapshot()
    contributions = _valid_contributions(snapshot, validator)
    valid_lineup = contributions[1]
    calibration = replace(
        valid_lineup.calibration,
        source_path=(AWAY.value, "dimension_deltas", "shots"),
        source_value=0.25,
    )
    home_multiplier, away_multiplier = calibrated_multipliers(calibration)
    extra = ExpectedGoalsContribution(
        contribution_key=f"lineup:{AWAY.value}:shots:{valid_lineup.version}",
        lambda_home_multiplier=home_multiplier,
        lambda_away_multiplier=away_multiplier,
        source_ref=LINEUP_REF,
        calibration=calibration,
        version=valid_lineup.version,
    )

    with pytest.raises(ValueError, match="contribution set"):
        _build(snapshot, validator, (*contributions, extra))


def test_complete_policy_bound_composition_round_trips() -> None:
    snapshot, validator = _snapshot()
    prediction = _build(snapshot, validator, _valid_contributions(snapshot, validator))
    payload = prediction_payload(prediction)

    assert (
        parse_prediction_payload(
            payload,
            snapshot=snapshot,
            snapshot_validator=validator,
        )
        == prediction
    )
