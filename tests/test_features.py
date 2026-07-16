from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from football_data_platform.features.contributions import (
    ExpectedGoalsContribution,
    apply_expected_goals_contributions,
    compose_expected_goals,
    context_contribution,
    lineup_delta_contributions,
)
from football_data_platform.features.player_profiles import (
    PlayerAvailabilityObservation,
    PlayerMatchObservation,
    ProfileWindowSpec,
    RoleMetricContract,
    build_player_profiles,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
)

AS_OF = datetime(2026, 1, 1, tzinfo=UTC)


def _team_observations() -> tuple[TeamMatchProcess, ...]:
    return (
        TeamMatchProcess(
            "match-1",
            "team-a",
            "team-b",
            AS_OF - timedelta(days=30),
            AS_OF - timedelta(days=30),
            1.8,
            0.7,
            "canonical:team-stats-1",
        ),
        TeamMatchProcess(
            "match-2",
            "team-b",
            "team-a",
            AS_OF - timedelta(days=10),
            AS_OF - timedelta(days=10),
            1.1,
            1.3,
            "canonical:team-stats-2",
        ),
    )


def test_future_match_does_not_change_historical_team_baseline_hash() -> None:
    base = _team_observations()
    future = TeamMatchProcess(
        "future",
        "team-a",
        "team-b",
        AS_OF + timedelta(days=1),
        AS_OF + timedelta(days=1),
        9.0,
        0.0,
        "canonical:future",
    )

    first = build_team_baseline(base, as_of=AS_OF, half_life_days=90, iterations=4)
    second = build_team_baseline((*base, future), as_of=AS_OF, half_life_days=90, iterations=4)

    assert first.artifact.artifact_id == second.artifact.artifact_id
    assert second.excluded_input_refs == ("canonical:future",)
    home, away = expected_goals_from_baseline(
        first.artifact, home_team_id="team-a", away_team_id="team-b"
    )
    assert home > 0
    assert away > 0


def test_team_baseline_contract_has_no_odds_input() -> None:
    assert "odds" not in {field.name for field in fields(TeamMatchProcess)}


def test_player_profiles_preserve_missing_metrics_and_metric_sample_minutes() -> None:
    contract = RoleMetricContract(
        role="forward",
        version="forward-metrics/1",
        required_metrics=("shots",),
        optional_metrics=("key_passes",),
        not_applicable_metrics=("saves",),
        minimum_minutes=100,
        minimum_matches=2,
    )
    observations = (
        PlayerMatchObservation(
            "player-a",
            "team-a",
            "match-1",
            "forward",
            90,
            AS_OF - timedelta(days=2),
            {"shots": 3.0, "key_passes": None},
            "canonical:player-a-1",
            played_at=AS_OF - timedelta(days=30),
        ),
        PlayerMatchObservation(
            "player-a",
            "team-a",
            "match-2",
            "forward",
            45,
            AS_OF - timedelta(days=1),
            {"shots": 1.0},
            "canonical:player-a-2",
            played_at=AS_OF - timedelta(days=5),
        ),
        PlayerMatchObservation(
            "player-a",
            "team-a",
            "match-3",
            "forward",
            60,
            AS_OF - timedelta(days=1),
            {"shots": 2.0},
            "canonical:player-a-3",
            played_at=AS_OF - timedelta(days=1),
        ),
        PlayerMatchObservation(
            "player-b",
            "team-b",
            "match-1",
            "forward",
            90,
            AS_OF - timedelta(days=2),
            {"shots": 2.0, "key_passes": 1.0},
            "canonical:player-b-1",
            played_at=AS_OF - timedelta(days=5),
        ),
        PlayerMatchObservation(
            "player-b",
            "team-b",
            "match-2",
            "forward",
            90,
            AS_OF - timedelta(days=1),
            {"shots": 2.0, "key_passes": 1.0},
            "canonical:player-b-2",
            played_at=AS_OF - timedelta(days=1),
        ),
    )

    result = build_player_profiles(
        observations,
        as_of=AS_OF,
        minimum_minutes=0,
        role_contracts=(contract,),
        long_term_window=ProfileWindowSpec("long-term", "long-term/1", 30),
        recent_window=ProfileWindowSpec("recent", "recent/1", 7),
        load_window=ProfileWindowSpec("load", "load/1", 7),
        cohort_version="premier-league-forward/1",
        availability_observations=(
            PlayerAvailabilityObservation(
                "player-a",
                "team-a",
                "available",
                1.0,
                AS_OF - timedelta(hours=1),
                "event:player-a-available",
            ),
        ),
    )
    profile = next(item for item in result.profiles if item.player_id == "player-a")
    metrics = {metric.metric: metric for metric in profile.long_term_ability.metrics}

    assert profile.quality_status == "ready"
    assert profile.role_contract_version == "forward-metrics/1"
    assert profile.long_term_ability.total_minutes == 195
    assert profile.recent_form.total_minutes == 105
    assert metrics["shots"].sample_minutes == 195
    assert metrics["shots"].per90 == pytest.approx(6 / 195 * 90)
    assert metrics["shots"].cohort_version == "premier-league-forward/1"
    assert metrics["shots"].cohort_size == 2
    assert metrics["shots"].role_percentile == 75.0
    assert metrics["key_passes"].quality_status == "missing"
    assert metrics["saves"].quality_status == "not-applicable"
    assert profile.availability.status == "available"
    assert profile.load.quality_status == "ready"
    assert profile.load.minutes == 105


@pytest.mark.parametrize(
    ("metrics", "minimum_minutes", "expected_status"),
    (({}, 0, "missing-required-metrics"), ({"shots": 1.0}, 100, "insufficient-sample")),
)
def test_empty_metrics_and_insufficient_samples_are_not_profile_ready(
    metrics: dict[str, float], minimum_minutes: float, expected_status: str
) -> None:
    observation = PlayerMatchObservation(
        "player-a",
        "team-a",
        "match-1",
        "forward",
        90,
        AS_OF - timedelta(days=1),
        metrics,
        "canonical:empty",
        played_at=AS_OF - timedelta(days=1),
    )

    result = build_player_profiles((observation,), as_of=AS_OF, minimum_minutes=minimum_minutes)

    assert result.profiles[0].quality_status == expected_status


def test_missing_window_time_keeps_load_and_profile_not_ready() -> None:
    observation = PlayerMatchObservation(
        "player-a",
        "team-a",
        "match-1",
        "forward",
        90,
        AS_OF - timedelta(days=1),
        {"shots": 1.0},
        "canonical:missing-played-at",
    )

    result = build_player_profiles((observation,), as_of=AS_OF, minimum_minutes=0)

    assert result.profiles[0].quality_status == "missing-window-time"
    assert result.profiles[0].load.quality_status == "missing"
    assert result.profiles[0].availability.status == "missing"


def test_profiles_do_not_mix_transfers_or_roles_and_exclude_future_facts() -> None:
    observations = (
        PlayerMatchObservation(
            "player-a",
            "team-a",
            "match-1",
            "forward",
            90,
            AS_OF - timedelta(days=3),
            {"shots": 2.0},
            "canonical:team-a-forward",
            played_at=AS_OF - timedelta(days=3),
        ),
        PlayerMatchObservation(
            "player-a",
            "team-b",
            "match-2",
            "forward",
            90,
            AS_OF - timedelta(days=2),
            {"shots": 1.0},
            "canonical:team-b-forward",
            played_at=AS_OF - timedelta(days=2),
        ),
        PlayerMatchObservation(
            "player-a",
            "team-b",
            "match-3",
            "MF",
            90,
            AS_OF - timedelta(days=1),
            {"shots": 1.0},
            "canonical:team-b-midfield",
            played_at=AS_OF - timedelta(days=1),
        ),
        PlayerMatchObservation(
            "player-a",
            "team-b",
            "future",
            "MF",
            90,
            AS_OF + timedelta(days=1),
            {"shots": 99.0},
            "canonical:future",
            played_at=AS_OF + timedelta(days=1),
        ),
    )

    result = build_player_profiles(observations, as_of=AS_OF, minimum_minutes=0)

    assert {(profile.team_id, profile.role) for profile in result.profiles} == {
        ("team-a", "forward"),
        ("team-b", "forward"),
        ("team-b", "MF"),
    }
    assert len(result.partition_audits) == 1
    assert result.partition_audits[0].team_ids == ("team-a", "team-b")
    assert result.partition_audits[0].roles == ("MF", "forward")
    assert result.excluded_input_refs == ("canonical:future",)
    midfield = next(profile for profile in result.profiles if profile.role == "MF")
    assert {metric.metric: metric.total for metric in midfield.metrics}["shots"] == 1.0


def test_expected_goals_contributions_reject_double_counting() -> None:
    injury = ExpectedGoalsContribution("player-x-unavailable", 0.9, 1.0, "event:1")

    with pytest.raises(ValueError, match="duplicate"):
        apply_expected_goals_contributions(1.5, 1.0, (injury, injury))

    expected = apply_expected_goals_contributions(1.5, 1.0, (injury,))
    assert expected.lambda_home == pytest.approx(1.35)
    assert expected.lambda_away == 1.0


def test_team_baseline_uses_neutral_coordinate_and_applies_home_advantage_once() -> None:
    observations = (
        TeamMatchProcess(
            "match-home-a",
            "team-a",
            "team-b",
            AS_OF - timedelta(days=30),
            AS_OF - timedelta(days=30),
            1.5,
            1.0,
            "canonical:symmetric-1",
        ),
        TeamMatchProcess(
            "match-home-b",
            "team-b",
            "team-a",
            AS_OF - timedelta(days=10),
            AS_OF - timedelta(days=10),
            1.5,
            1.0,
            "canonical:symmetric-2",
        ),
    )

    result = build_team_baseline(observations, as_of=AS_OF, half_life_days=90, iterations=4)
    home, away = expected_goals_from_baseline(
        result.artifact, home_team_id="team-a", away_team_id="team-b"
    )

    assert result.artifact.transform_version == "team-baseline/2"
    assert result.artifact.league_xg_per_team == pytest.approx(1.0)
    assert result.artifact.home_advantage == pytest.approx(1.5)
    assert home == pytest.approx(1.5)
    assert away == pytest.approx(1.0)


def test_composition_consumes_context_and_ready_lineup_once() -> None:
    context = context_contribution(
        {"days_since_previous_match": 8.0}, source_ref="derived-source:context"
    )
    lineup = lineup_delta_contributions(
        {
            "team-home": {
                "quality_status": "ready",
                "dimension_deltas": {"attack": 2.0},
            },
            "team-away": {
                "quality_status": "preview",
                "dimension_deltas": {},
            },
        },
        home_team_id="team-home",
        away_team_id="team-away",
        source_ref="derived-source:lineup",
    )

    composed = compose_expected_goals(1.5, 1.0, (context, *lineup))

    assert composed.baseline_lambda_home == pytest.approx(1.5)
    assert composed.baseline_lambda_away == pytest.approx(1.0)
    assert composed.lambda_home > 1.5
    assert composed.lambda_away > 1.0
    assert composed.contribution_keys == tuple(sorted(composed.contribution_keys))
    assert composed.input_refs == (
        "derived-source:context",
        "derived-source:lineup",
    )


def test_preview_lineup_is_not_silently_converted_to_neutral_contribution() -> None:
    contributions = lineup_delta_contributions(
        {
            "team-home": {
                "quality_status": "preview",
                "dimension_deltas": {},
            },
            "team-away": {
                "quality_status": "preview",
                "dimension_deltas": {},
            },
        },
        home_team_id="team-home",
        away_team_id="team-away",
        source_ref="derived-source:lineup-preview",
    )

    composed = compose_expected_goals(1.5, 1.0, contributions)

    assert contributions == ()
    assert composed.lambda_home == pytest.approx(1.5)
    assert composed.lambda_away == pytest.approx(1.0)
    assert composed.contribution_keys == ()
