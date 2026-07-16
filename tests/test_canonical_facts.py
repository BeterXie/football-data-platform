from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, SeasonId
from football_data_platform.domain.lifecycle import Qualification, assess_lifecycle
from football_data_platform.domain.models import MatchStatus
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

NOW = datetime(2026, 7, 16, 5, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]


@pytest.fixture
def fact_context(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    asset = archive.archive(
        b"match report",
        source="fbref",
        source_id="report-1",
        url="https://fbref.example/report-1",
        observed_at=NOW,
        target_event_time=NOW - timedelta(days=1),
        collector_version="test/1",
        media_type="text/html",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    canonical.register_registry(registry, registered_at=NOW)
    canonical.register_raw_asset(asset)
    teams = [
        canonical.resolve_or_create_team(
            source="fbref",
            source_id=f"team-{index}",
            canonical_name=f"Team {index}",
            competition_id=CompetitionId("competition:eng.1"),
            observed_at=NOW,
            raw_asset_id=asset.id,
        )
        for index in range(2)
    ]
    match, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="fixture-1",
        competition_id=CompetitionId("competition:eng.1"),
        season_id=SeasonId("season:eng.1.2025-26"),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW - timedelta(days=1),
        status=MatchStatus.FINISHED,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    return canonical, CanonicalFactStore(canonical), asset, teams, match, version


def test_result_and_team_observations_are_versioned_and_idempotent(fact_context) -> None:
    _, facts, asset, teams, match, version = fact_context
    arguments = {
        "match_id": match.id,
        "match_version": version.version,
        "home_goals": 2,
        "away_goals": 1,
        "known_at": NOW,
        "observed_at": NOW,
        "raw_asset_id": asset.id,
    }

    first = facts.append_result_90(**arguments)
    replay = facts.append_result_90(**arguments)
    correction = facts.append_result_90(**{**arguments, "home_goals": 3})
    facts.append_team_observation(
        match_id=match.id,
        match_version=version.version,
        team_id=teams[0].id,
        stats={"goals": 2, "xg": 1.7, "shots": 12, "shots_on_target": 5},
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    assert first == replay
    assert first.observation_version == 1
    assert correction.observation_version == 2


def test_missing_is_preserved_and_readiness_explains_it(fact_context) -> None:
    _, facts, asset, teams, match, version = fact_context
    facts.append_result_90(
        match_id=match.id,
        match_version=version.version,
        home_goals=2,
        away_goals=1,
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    for team in teams:
        facts.append_team_observation(
            match_id=match.id,
            match_version=version.version,
            team_id=team.id,
            stats={"goals": 1, "xg": None, "shots": 8, "shots_on_target": 3},
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )

    assessment = assess_lifecycle(facts.availability(match.id), evaluated_at=NOW)
    result = next(
        item
        for item in assessment.qualifications
        if item.qualification is Qualification.TEAM_BASELINE
    )

    assert not result.passed
    assert any(reason.endswith(":xg") for reason in result.reason_codes)


def test_unconfirmed_event_cannot_modify_features(fact_context) -> None:
    canonical, facts, asset, teams, match, _ = fact_context
    evidence = facts.add_news_evidence(
        source="club-site",
        url="https://club.example/news",
        title="Training update",
        published_at=NOW - timedelta(hours=2),
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    event = facts.add_prematch_event(
        match_id=match.id,
        team_id=teams[0].id,
        player_id=None,
        event_type="injury-rumour",
        occurred_at=None,
        known_at=NOW,
        confirmation_status="unconfirmed",
        evidence_refs=(evidence.record_id,),
    )

    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT can_modify_features FROM prematch_events WHERE record_id = ?",
            (event.record_id,),
        ).fetchone()
    assert row["can_modify_features"] == 0


def test_fact_observation_cannot_predate_when_it_became_known(fact_context) -> None:
    _, facts, asset, _, match, version = fact_context

    with pytest.raises(ValueError, match="known_at"):
        facts.append_result_90(
            match_id=match.id,
            match_version=version.version,
            home_goals=1,
            away_goals=0,
            known_at=NOW,
            observed_at=NOW - timedelta(minutes=1),
            raw_asset_id=asset.id,
        )
