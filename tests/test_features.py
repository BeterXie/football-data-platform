from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from football_data_platform.features.contributions import (
    ExpectedGoalsContribution,
    apply_expected_goals_contributions,
)
from football_data_platform.features.player_profiles import (
    PlayerMatchObservation,
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
        ),
    )

    result = build_player_profiles(observations, as_of=AS_OF, minimum_minutes=100)
    profile = next(item for item in result.profiles if item.player_id == "player-a")
    metrics = {metric.metric: metric for metric in profile.metrics}

    assert profile.quality_status == "ready"
    assert metrics["shots"].sample_minutes == 135
    assert metrics["shots"].per90 == pytest.approx(4 / 135 * 90)
    assert "key_passes" not in metrics
    assert metrics["shots"].role_percentile == 75.0


def test_expected_goals_contributions_reject_double_counting() -> None:
    injury = ExpectedGoalsContribution("player-x-unavailable", 0.9, 1.0, "event:1")

    with pytest.raises(ValueError, match="duplicate"):
        apply_expected_goals_contributions(1.5, 1.0, (injury, injury))

    expected = apply_expected_goals_contributions(1.5, 1.0, (injury,))
    assert expected.lambda_home == pytest.approx(1.35)
    assert expected.lambda_away == 1.0
