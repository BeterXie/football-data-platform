from __future__ import annotations

import io
import urllib.error
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.pipelines.schedule import (
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.sources.fbref import (
    FBrefAccessBlockedError,
    FBrefFetchError,
    ScheduleMatch,
    ScheduleParseResult,
    fetch_schedule,
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


@pytest.mark.parametrize(
    ("replacement", "diagnostic_code", "original"),
    [
        (
            b"",
            "kickoff_missing",
            b'<td data-stat="start_time">20:00</td>',
        ),
        (
            b'<td data-stat="start_time">not-a-time</td>',
            "kickoff_invalid",
            b'<td data-stat="start_time">15:00</td>',
        ),
    ],
)
def test_missing_or_invalid_start_time_keeps_fixture_unknown(
    replacement: bytes,
    diagnostic_code: str,
    original: bytes,
) -> None:
    _, competition, season = _registration()
    content = (ROOT / "tests" / "fixtures" / "fbref_premier_league_schedule.html").read_bytes()
    content = content.replace(original, replacement, 1)

    result = parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=schedule_url(competition, season),
    )

    assert len(result.matches) == 2
    assert result.matches[0 if diagnostic_code == "kickoff_missing" else 1].kickoff_at is None
    assert diagnostic_code in {item.code for item in result.diagnostics}
    assert all(
        match.kickoff_at != datetime(2025, 8, 15, 0, 0, tzinfo=UTC) for match in result.matches
    )


def test_fetch_schedule_preserves_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html><title>Cloudflare</title><div id='cf-chl-widget'></div></html>"
    url = "https://fbref.example/schedule"

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url,
            403,
            "forbidden",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)

    with pytest.raises(FBrefFetchError) as caught:
        fetch_schedule(url, observed_at=OBSERVED_AT)

    assert caught.value.diagnostic.body == body
    assert caught.value.diagnostic.response_body == body
    assert caught.value.diagnostic.as_dict()["body_present"] is True
    assert caught.value.diagnostic.as_dict()["body_size_bytes"] == len(body)


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

    canonical.record_collection_attempt(
        match_id=MatchId(first.canonical_match_ids[0]),
        source="football-data-results",
        target_url="https://football-data.example/results.csv",
        outcome=CollectionAttemptOutcome.BLOCKED,
        observed_at=OBSERVED_AT,
        collector_version="football-data-results/test",
        diagnostic_code="network_error",
    )
    wrong_source_attempt = assess_season_coverage(
        first.parsed,
        season,
        source="fbref",
        canonical=canonical,
    )
    assert len(wrong_source_attempt.missing_collection_attempts) == 2

    for fixture, match_id in zip(first.parsed.matches, first.canonical_match_ids, strict=True):
        canonical.record_collection_attempt(
            match_id=MatchId(match_id),
            source="fbref-match-report",
            target_url=fixture.report_url or f"https://fbref.example/{fixture.source_fixture_id}",
            outcome=CollectionAttemptOutcome.BLOCKED,
            observed_at=OBSERVED_AT,
            collector_version="fbref-match-report/test",
            diagnostic_code="blocked_by_access_control",
        )
    persisted_attempts = assess_season_coverage(
        first.parsed,
        season,
        source="fbref",
        canonical=canonical,
    )
    assert persisted_attempts.missing_collection_attempts == ()


def test_schedule_coverage_cross_checks_canonical_fixture_identity(tmp_path: Path) -> None:
    registry, competition, season = _registration()
    content = (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes()
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    ingested = ingest_fbref_schedule(
        content,
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=OBSERVED_AT,
        archive=archive,
        canonical=canonical,
    )

    first = ingested.parsed.matches[0]
    alternate_team = first.away_source_id
    tampered = replace(
        first,
        home_source_id=alternate_team,
        kickoff_at=first.kickoff_at.replace(minute=1) if first.kickoff_at else None,
    )
    tampered_parsed = ScheduleParseResult(
        (tampered, *ingested.parsed.matches[1:]),
        ingested.parsed.diagnostics,
        ingested.parsed.rows_seen,
    )

    coverage = assess_season_coverage(
        tampered_parsed,
        season,
        source="fbref",
        canonical=canonical,
    )

    assert not coverage.schedule_complete
    assert any(
        item.startswith("canonical_match_home_team_mismatch:")
        for item in coverage.structural_violations
    )
    assert any(
        item.startswith("canonical_match_kickoff_mismatch:")
        for item in coverage.structural_violations
    )


def test_round_change_keeps_match_identity_and_adds_version(tmp_path: Path) -> None:
    registry, competition, season = _registration()
    content = (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes()
    changed_round = content.replace(b"Matchweek 1", b"Matchweek 9")
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    arguments = {
        "page_url": schedule_url(competition, season),
        "competition": competition,
        "season": season,
        "archive": archive,
        "canonical": canonical,
    }

    first = ingest_fbref_schedule(content, observed_at=OBSERVED_AT, **arguments)
    second = ingest_fbref_schedule(
        changed_round,
        observed_at=OBSERVED_AT.replace(minute=5),
        **arguments,
    )

    assert second.canonical_match_ids == first.canonical_match_ids
    versions = canonical.match_versions(MatchId(first.canonical_match_ids[0]))
    assert [version.round_name for version in versions] == ["Matchweek 1", "Matchweek 9"]


def test_premier_league_season_gate_rejects_unpersisted_attempt_ids() -> None:
    _, _, season = _registration()
    team_ids = [team.source("fbref").source_id for team in season.teams]
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

    unverified_attempts = assess_season_coverage(
        parsed,
        season,
        attempted_fixture_ids=fixture_ids,
    )

    assert unverified_attempts.schedule_complete
    assert not unverified_attempts.complete
    assert len(unverified_attempts.missing_collection_attempts) == 380

    fake_ids = {team_id: f"unregistered-{index:02d}" for index, team_id in enumerate(team_ids)}
    unregistered = assess_season_coverage(
        ScheduleParseResult(
            tuple(
                replace(
                    match,
                    home_source_id=fake_ids[match.home_source_id],
                    away_source_id=fake_ids[match.away_source_id],
                )
                for match in matches
            ),
            (),
            rows_seen=380,
        ),
        season,
        source="fbref",
    )
    assert not unregistered.schedule_complete
    assert len(unregistered.unregistered_team_ids) == 20
    assert len(unregistered.missing_registered_teams) == 20


def test_premier_league_season_gate_rejects_380_rows_with_only_10_matchups() -> None:
    _, _, season = _registration()
    team_ids = [team.source("fbref").source_id for team in season.teams]
    matchups = [(team_ids[index], team_ids[(index + 1) % 20]) for index in range(10)]
    matches = tuple(
        ScheduleMatch(
            source_fixture_id=f"fixture-{index:03d}",
            source_match_id=f"report-{index:03d}",
            round_name=f"Matchweek {index // 10 + 1}",
            kickoff_at=OBSERVED_AT,
            home_source_id=matchups[index % len(matchups)][0],
            home_name=matchups[index % len(matchups)][0],
            away_source_id=matchups[index % len(matchups)][1],
            away_name=matchups[index % len(matchups)][1],
            status=MatchStatus.FINISHED,
            home_goals=1,
            away_goals=0,
            report_url=f"https://fbref.example/matches/{index:03d}",
        )
        for index in range(380)
    )

    coverage = assess_season_coverage(
        ScheduleParseResult(matches, (), rows_seen=380),
        season,
    )

    assert not coverage.schedule_complete
    assert "duplicate_directed_matchups" in coverage.structural_violations
