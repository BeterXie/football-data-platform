from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import CollectionAttemptOutcome
from football_data_platform.pipelines.match_report import (
    MatchReportIngestError,
    ingest_fbref_match_report,
)
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.sources.fbref_match_report import parse_match_report
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)


def _prepare_schedule(tmp_path: Path, content: bytes | None = None):
    registry = load_competition_registry(ROOT / "config/competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    schedule = ingest_fbref_schedule(
        content
        if content is not None
        else (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes(),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=OBSERVED_AT,
        archive=archive,
        canonical=canonical,
    )
    return archive, canonical, season, schedule


def _seed_first_result(canonical: CanonicalStore, schedule) -> str:
    fixture = schedule.parsed.matches[0]
    assert fixture.home_goals is not None
    assert fixture.away_goals is not None
    result = CanonicalFactStore(canonical).append_result_90(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=fixture.home_goals,
        away_goals=fixture.away_goals,
        known_at=OBSERVED_AT,
        observed_at=OBSERVED_AT,
        raw_asset_id=RawAssetId(schedule.raw_asset_id),
    )
    return result.record_id


def _report_content() -> bytes:
    return (ROOT / "tests/fixtures/fbref_premier_league_match_report.html").read_bytes()


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
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    match_id = MatchId(schedule.canonical_match_ids[0])
    result_fact_id = _seed_first_result(canonical, schedule)
    report_content = _report_content()

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
    assert first.result_fact_id == result_fact_id
    assert canonical.counts()["players"] == 3
    attempts = canonical.collection_attempts(season.id)
    report_attempts = [item for item in attempts if item.source == "fbref-match-report"]
    assert len(report_attempts) == 1
    assert report_attempts[0].outcome is CollectionAttemptOutcome.SUCCEEDED
    home = canonical.mapped_team(source="fbref", source_id="18bb7c10")
    with canonical.connect() as connection:
        stats = connection.execute(
            "SELECT stats_json FROM team_match_observations WHERE team_id = ?",
            (home.id.value,),
        ).fetchone()["stats_json"]
        result_count = connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0]
    assert '"goals":2' in stats
    assert result_count == 1


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"<html><body></body></html>", "match_report_parse_diagnostics"),
        (
            _report_content().replace(b'<a href="/en/players/playera1/A-One">A One</a>', b"A One"),
            "match_report_parse_diagnostics",
        ),
    ],
)
def test_parse_failures_archive_raw_and_record_failed_attempt(
    tmp_path: Path,
    content: bytes,
    expected_code: str,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            content,
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == expected_code
    archive.verify(RawAssetId(caught.value.raw_asset_id))
    attempts = [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ]
    assert len(attempts) == 1
    assert attempts[0].outcome is CollectionAttemptOutcome.FAILED
    assert attempts[0].diagnostic_code == expected_code
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0


def test_unknown_match_mapping_archives_raw_without_claiming_a_fixture_attempt(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content(),
            page_url="https://fbref.example/en/matches/unknown/report",
            source_match_id="unknown",
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "source_match_mapping_missing"
    archive.verify(RawAssetId(caught.value.raw_asset_id))
    assert [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ] == []
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0] == 0


def test_report_source_mapping_rejects_caller_supplied_wrong_match(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content(),
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=MatchId(schedule.canonical_match_ids[1]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "match_identity_mismatch"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.match_id == MatchId(schedule.canonical_match_ids[0])
    assert attempt.outcome is CollectionAttemptOutcome.FAILED
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0] == 0


def test_non_finished_match_version_cannot_ingest_report_facts(tmp_path: Path) -> None:
    schedule_content = (
        (ROOT / "tests/fixtures/fbref_premier_league_schedule.html")
        .read_bytes()
        .replace(b">2-1<", b"><", 1)
    )
    archive, canonical, season, schedule = _prepare_schedule(tmp_path, schedule_content)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content(),
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "match_version_not_finished"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.outcome is CollectionAttemptOutcome.FAILED
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM player_match_observations").fetchone()[0] == 0
        )


def test_report_requires_existing_canonical_result_and_rejects_claimed_score(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    match_id = MatchId(schedule.canonical_match_ids[0])

    with pytest.raises(MatchReportIngestError) as missing:
        ingest_fbref_match_report(
            _report_content(),
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
    assert missing.value.code == "canonical_result_missing"

    result_fact_id = _seed_first_result(canonical, schedule)
    retry_observed_at = OBSERVED_AT + timedelta(seconds=1)
    with pytest.raises(MatchReportIngestError) as mismatch:
        ingest_fbref_match_report(
            _report_content(),
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=match_id,
            match_version=1,
            home_goals=9,
            away_goals=9,
            known_at=OBSERVED_AT,
            observed_at=retry_observed_at,
            archive=archive,
            canonical=canonical,
        )

    assert mismatch.value.code == "score_conflicts_with_canonical"
    with canonical.connect() as connection:
        rows = connection.execute(
            "SELECT record_id, home_goals, away_goals FROM match_results_90"
        ).fetchall()
    assert [(row["record_id"], row["home_goals"], row["away_goals"]) for row in rows] == [
        (result_fact_id, 2, 1)
    ]
    attempts = [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ]
    assert attempts
    assert all(item.outcome is CollectionAttemptOutcome.FAILED for item in attempts)


def test_report_team_ids_must_map_to_both_canonical_match_teams(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    content = _report_content().replace(b"stats_18bb7c10_summary", b"stats_deadbeef_summary")

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            content,
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "report_team_mapping_missing"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.outcome is CollectionAttemptOutcome.FAILED
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0


def test_second_team_player_conflict_is_rejected_before_any_fact_write(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    content = _report_content().replace(
        b"/en/players/playerb1/B-One",
        b"/en/players/playera1/B-One",
    )

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            content,
            page_url="https://fbref.example/en/matches/aaaaaaaa/report",
            source_match_id="aaaaaaaa",
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "report_player_assignment_conflict"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.outcome is CollectionAttemptOutcome.FAILED
    with canonical.connect() as connection:
        for table in (
            "team_match_observations",
            "player_match_observations",
            "lineup_facts",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
