from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId
from football_data_platform.pipelines.match_report import ingest_fbref_match_report
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.sources.fbref_match_report import parse_match_report
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)


def test_match_report_parses_normal_and_comment_wrapped_summary_tables() -> None:
    content = (ROOT / "tests/fixtures/fbref_premier_league_match_report.html").read_bytes()

    parsed = parse_match_report(content)

    assert [team.source_team_id for team in parsed.teams] == ["18bb7c10", "cff3d9bb"]
    assert len(parsed.teams[0].players) == 2
    assert parsed.teams[0].aggregated_stats == pytest.approx(
        {"goals": 9.0, "xg": 1.7, "shots": 7.0, "shots_on_target": 4.0}
    )
    assert parsed.teams[1].players[0].starter is True


def test_match_report_ingest_creates_canonical_team_and_player_facts(tmp_path: Path) -> None:
    registry = load_competition_registry(ROOT / "config/competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    schedule = ingest_fbref_schedule(
        (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes(),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=OBSERVED_AT,
        archive=archive,
        canonical=canonical,
    )
    match_id = MatchId(schedule.canonical_match_ids[0])
    report_content = (ROOT / "tests/fixtures/fbref_premier_league_match_report.html").read_bytes()

    first = ingest_fbref_match_report(
        report_content,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=match_id,
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=OBSERVED_AT,
        archive=archive,
        canonical=canonical,
    )
    second = ingest_fbref_match_report(
        report_content,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=match_id,
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=OBSERVED_AT,
        archive=archive,
        canonical=canonical,
    )

    assert first == second
    assert len(first.team_fact_ids) == 2
    assert len(first.player_fact_ids) == 3
    assert canonical.counts()["players"] == 3
    home = canonical.mapped_team(source="fbref", source_id="18bb7c10")
    with canonical.connect() as connection:
        stats = connection.execute(
            "SELECT stats_json FROM team_match_observations WHERE team_id = ?",
            (home.id.value,),
        ).fetchone()["stats_json"]
    assert '"goals":2' in stats
