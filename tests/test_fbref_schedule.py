from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.models import MatchStatus
from football_data_platform.pipelines.schedule import (
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.sources.fbref import (
    FBrefAccessBlockedError,
    ScheduleMatch,
    ScheduleParseResult,
    parse_schedule,
    schedule_url,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 3, 0, tzinfo=UTC)


def _registration():
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    competition = registry.competitions[0]
    return registry, competition, competition.seasons[0]


def test_registered_schedule_url_has_no_tournament_specific_branch() -> None:
    _, competition, season = _registration()

    url = schedule_url(competition, season)

    assert url == (
        "https://fbref.com/en/comps/9/2025-2026/schedule/"
        "2025-2026-Premier-League-Scores-and-Fixtures"
    )


def test_comment_wrapped_schedule_parses_all_valid_league_rows() -> None:
    _, competition, season = _registration()
    content = (ROOT / "tests" / "fixtures" / "fbref_premier_league_schedule.html").read_bytes()

    result = parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=schedule_url(competition, season),
    )

    assert result.rows_seen == 3
    assert len(result.matches) == 2
    assert result.matches[0].source_match_id == "aaaaaaaa"
    assert result.matches[0].home_goals == 2
    assert result.matches[0].away_goals == 1
    assert result.matches[0].kickoff_at == datetime(2025, 8, 15, 19, 0, tzinfo=UTC)
    assert result.matches[1].source_match_id is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_schedule_row"]


def test_access_control_page_is_not_interpreted_as_an_empty_schedule() -> None:
    _, competition, season = _registration()
    challenge = b"<html><title>Cloudflare</title><div id='cf-chl-widget'></div></html>"

    with pytest.raises(FBrefAccessBlockedError) as caught:
        parse_schedule(
            challenge,
            competition=competition,
            season=season,
            page_url=schedule_url(competition, season),
        )

    assert caught.value.diagnostic.code == "blocked_by_access_control"


def test_schedule_pipeline_archives_before_normalizing_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    registry, competition, season = _registration()
    content = (ROOT / "tests" / "fixtures" / "fbref_premier_league_schedule.html").read_bytes()
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    arguments = {
        "content": content,
        "page_url": schedule_url(competition, season),
        "competition": competition,
        "season": season,
        "observed_at": OBSERVED_AT,
        "archive": archive,
        "canonical": canonical,
    }

    first = ingest_fbref_schedule(**arguments)
    second = ingest_fbref_schedule(**arguments)

    assert first.raw_asset_id == second.raw_asset_id
    assert first.canonical_match_ids == second.canonical_match_ids
    assert canonical.counts()["teams"] == 2
    assert canonical.counts()["matches"] == 2
    assert not first.coverage.complete
    assert first.coverage.actual_matches == 2
    assert len(first.coverage.missing_collection_attempts) == 2


def test_premier_league_season_gate_requires_20_teams_380_matches_and_attempts() -> None:
    _, _, season = _registration()
    team_ids = [f"team-{index:02d}" for index in range(20)]
    matches = []
    match_number = 0
    for home_index, home in enumerate(team_ids):
        for away_index, away in enumerate(team_ids):
            if home_index == away_index:
                continue
            match_number += 1
            matches.append(
                ScheduleMatch(
                    source_fixture_id=f"fixture-{match_number:03d}",
                    source_match_id=f"report-{match_number:03d}",
                    round_name=f"Matchweek {(match_number - 1) // 10 + 1}",
                    kickoff_at=OBSERVED_AT,
                    home_source_id=home,
                    home_name=home,
                    away_source_id=away,
                    away_name=away,
                    status=MatchStatus.FINISHED,
                    home_goals=1,
                    away_goals=0,
                    report_url=f"https://fbref.example/matches/{match_number:03d}",
                )
            )
    parsed = ScheduleParseResult(tuple(matches), (), rows_seen=380)
    fixture_ids = {match.source_fixture_id for match in matches}

    complete = assess_season_coverage(
        parsed,
        season,
        attempted_fixture_ids=fixture_ids,
    )
    missing_attempt = assess_season_coverage(
        parsed,
        season,
        attempted_fixture_ids=fixture_ids - {"fixture-001"},
    )

    assert complete.schedule_complete
    assert complete.complete
    assert not missing_attempt.complete
    assert missing_attempt.missing_collection_attempts == ("fixture-001",)
