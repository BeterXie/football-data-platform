from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, SeasonId
from football_data_platform.domain.lifecycle import Qualification, assess_lifecycle
from football_data_platform.domain.models import MatchStatus
from football_data_platform.domain.predictions import MatchResult90
from football_data_platform.sources.prematch import SourceDescriptor, SourceKind, SourceRegistry
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

NOW = datetime(2026, 7, 16, 5, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]
COMPETITION_ID = CompetitionId("competition:eng.1")
SEASON_ID = SeasonId("season:eng.1.2025-26")


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
    source_registry = SourceRegistry((SourceDescriptor("fbref", SourceKind.NEWS, "fbref"),))
    return (
        canonical,
        CanonicalFactStore(canonical, source_registry=source_registry),
        asset,
        teams,
        match,
        version,
    )


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


def test_canonical_result_validator_matches_source_ref_and_payload(fact_context) -> None:
    _, facts, asset, _, match, version = fact_context
    stored = facts.append_result_90(
        match_id=match.id,
        match_version=version.version,
        home_goals=2,
        away_goals=1,
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    result = MatchResult90(match.id, 2, 1, NOW, stored.record_id)

    facts.verify_match_result(result)
    with pytest.raises(ValueError, match="canonical fact"):
        facts.verify_match_result(replace(result, source_ref="canonical:forged"))
    with pytest.raises(ValueError, match="canonical fact"):
        facts.verify_match_result(replace(result, home_goals=3))


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
        source="fbref",
        url="https://fbref.example/report-1",
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


def test_availability_excludes_facts_known_after_as_of(fact_context) -> None:
    _, facts, asset, teams, match, version = fact_context
    future = NOW + timedelta(hours=1)
    facts.append_result_90(
        match_id=match.id,
        match_version=version.version,
        home_goals=1,
        away_goals=0,
        known_at=future,
        observed_at=future,
        raw_asset_id=asset.id,
    )
    facts.append_team_observation(
        match_id=match.id,
        match_version=version.version,
        team_id=teams[0].id,
        stats={"goals": 1, "xg": 1.2, "shots": 9, "shots_on_target": 4},
        known_at=future,
        observed_at=future,
        raw_asset_id=asset.id,
    )

    historical = facts.availability(match.id, as_of=NOW)
    latest = facts.availability(match.id)

    assert not historical.result_90_present
    assert teams[0].id.value not in historical.team_stat_fields
    assert historical.as_of == NOW
    assert latest.result_90_present
    assert teams[0].id.value in latest.team_stat_fields


def test_result_requires_a_finished_match_version(fact_context) -> None:
    canonical, facts, asset, teams, _, _ = fact_context
    scheduled, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="scheduled-fixture",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="finished match version"):
        facts.append_result_90(
            match_id=scheduled.id,
            match_version=version.version,
            home_goals=1,
            away_goals=0,
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )


def test_team_observation_rejects_a_non_participant(fact_context) -> None:
    canonical, facts, asset, teams, match, version = fact_context
    foreign = canonical.resolve_or_create_team(
        source="fbref",
        source_id="foreign-team",
        canonical_name="Foreign Team",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="does not belong to match"):
        facts.append_team_observation(
            match_id=match.id,
            match_version=version.version,
            team_id=foreign.id,
            stats={"goals": 1},
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )


def test_player_observation_and_lineup_reject_cross_team_assignment(fact_context) -> None:
    canonical, facts, asset, teams, match, version = fact_context
    player = canonical.resolve_or_create_player(
        source="fbref",
        source_id="cross-team-player",
        canonical_name="Cross Team Player",
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    facts.append_player_observation(
        match_id=match.id,
        match_version=version.version,
        team_id=teams[0].id,
        player_id=player.id,
        role="FW",
        minutes=90,
        metrics={"goals": 1},
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="already assigned to"):
        facts.append_player_observation(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[1].id,
            player_id=player.id,
            role="FW",
            minutes=90,
            metrics={"goals": 1},
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )

    with pytest.raises(ValueError, match="already assigned to"):
        facts.append_lineup_fact(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[1].id,
            player_id=player.id,
            lineup_role="starter",
            official=False,
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )


def test_lineup_rejects_a_non_participant_team(fact_context) -> None:
    canonical, facts, asset, teams, match, version = fact_context
    foreign = canonical.resolve_or_create_team(
        source="fbref",
        source_id="foreign-lineup-team",
        canonical_name="Foreign Lineup Team",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    player = canonical.resolve_or_create_player(
        source="fbref",
        source_id="foreign-lineup-player",
        canonical_name="Foreign Lineup Player",
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="does not belong to match"):
        facts.append_lineup_fact(
            match_id=match.id,
            match_version=version.version,
            team_id=foreign.id,
            player_id=player.id,
            lineup_role="starter",
            official=False,
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )


def test_lineup_fact_rejects_caller_claimed_official_provenance(fact_context) -> None:
    canonical, facts, asset, teams, match, version = fact_context
    player = canonical.resolve_or_create_player(
        source="fbref",
        source_id="forged-official-player",
        canonical_name="Forged Official Player",
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="append_official_lineup"):
        facts.append_lineup_fact(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[0].id,
            player_id=player.id,
            lineup_role="starter",
            official=True,
            known_at=NOW,
            observed_at=NOW,
            raw_asset_id=asset.id,
        )

    with canonical.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM lineup_facts WHERE player_id = ?",
                (player.id.value,),
            ).fetchone()[0]
            == 0
        )


def test_prematch_event_rejects_a_non_participant_team(fact_context) -> None:
    canonical, facts, asset, teams, match, _ = fact_context
    foreign = canonical.resolve_or_create_team(
        source="fbref",
        source_id="foreign-event-team",
        canonical_name="Foreign Event Team",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    evidence = facts.add_news_evidence(
        source="fbref",
        url="https://fbref.example/report-1",
        title="Foreign event",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="does not belong to match"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=foreign.id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
        )


def test_prematch_event_rejects_registered_player_without_match_assignment(
    fact_context,
) -> None:
    canonical, facts, asset, _, match, _ = fact_context
    player = canonical.resolve_or_create_player(
        source="fbref",
        source_id="unassigned-event-player",
        canonical_name="Unassigned Event Player",
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    evidence = facts.add_news_evidence(
        source="fbref",
        url="https://fbref.example/report-1",
        title="Unassigned player event",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="has no assignment for match"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=None,
            player_id=player.id,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
        )


def test_prematch_event_rejects_team_without_matching_player_assignment(fact_context) -> None:
    canonical, facts, asset, teams, match, _ = fact_context
    player = canonical.resolve_or_create_player(
        source="fbref",
        source_id="unassigned-team-event-player",
        canonical_name="Unassigned Team Event Player",
        observed_at=NOW,
        raw_asset_id=asset.id,
    )
    evidence = facts.add_news_evidence(
        source="fbref",
        url="https://fbref.example/report-1",
        title="Unassigned team player event",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=asset.id,
    )

    with pytest.raises(ValueError, match="has no assignment for match"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=player.id,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
        )
