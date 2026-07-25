from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CollectionAttemptId, MatchId, RawAssetId
from football_data_platform.domain.lifecycle import Qualification, assess_lifecycle
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.pipelines.match_report import (
    PRODUCTION_COLLECTOR_VERSION,
    PRODUCTION_REQUIRED_TABLES,
    MatchReportIngestError,
)
from football_data_platform.pipelines.match_report import (
    ingest_fbref_match_report as _ingest_fbref_match_report,
)
from football_data_platform.pipelines.schedule import (
    SeasonCoverage,
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.sources.fbref_match_report import (
    REPORT_PARSER_VERSION,
    parse_match_report,
)
from football_data_platform.storage import match_report_contracts as contract_replay_module
from football_data_platform.storage.canonical import CanonicalConflictError, CanonicalStore
from football_data_platform.storage.facts import (
    CanonicalFactStore,
    load_verified_actual_lineup_fact,
    load_verified_match_report_player_batch,
    load_verified_match_result,
    load_verified_player_observation,
    load_verified_team_observation,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.match_report_contracts import (
    MatchReportContractReplayError,
    verify_match_report_contract,
)
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactConflict, TrainingArtifactStore
from football_data_platform.storage.verification import VerificationSession

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 6, 0, tzinfo=UTC)


def ingest_fbref_match_report(content, **kwargs):
    """Keep existing fixtures on the explicit summary-only preview contract."""

    kwargs.setdefault("required_tables", ("summary",))
    return _ingest_fbref_match_report(content, **kwargs)


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


def _complete_report_content() -> bytes:
    missing_tables = (
        b'<table id="stats_18bb7c10_keeper"><tbody>'
        b'<tr><th data-stat="player">Team Total</th></tr>'
        b"</tbody></table>"
        b'<table id="stats_cff3d9bb_passing"><tbody>'
        b'<tr><th data-stat="player">Team Total</th></tr>'
        b"</tbody></table>"
    )
    return _report_content().replace(b"</body>", missing_tables + b"</body>")


def _full_player_profile_report_content() -> bytes:
    def rows(prefix: str, team_name: str) -> str:
        return "".join(
            f"""
            <tr>
              <th data-stat="player">
                <a href="/en/players/{prefix}{index:02d}/p">{team_name} {index}</a>
              </th>
              <td data-stat="position">FW</td><td data-stat="games_starts">1</td>
              <td data-stat="minutes">90</td><td data-stat="goals">0</td>
              <td data-stat="xg">0.1</td><td data-stat="shots">1</td>
              <td data-stat="shots_on_target">1</td>
            </tr>
            """
            for index in range(11)
        )

    return f"""
    <!doctype html><html><body>
      <link rel="canonical" href="https://fbref.com/en/matches/aaaaaaaa/report">
      <div class="scorebox">
        <div class="scorebox_meta">
          <a href="/en/comps/9/2025-2026/Premier-League-Stats">Premier League</a>
          <span data-venue-date="2025-08-15"></span>
        </div>
        <div><strong><a href="/en/squads/18bb7c10/Arsenal-Stats">Arsenal</a></strong>
          <div class="score">2</div></div>
        <div><strong><a href="/en/squads/cff3d9bb/Chelsea-Stats">Chelsea</a></strong>
          <div class="score">1</div></div>
      </div>
      <table id="stats_18bb7c10_summary"><tbody>
        {rows("homep", "Home Player")}
        <tr><th data-stat="player">Team Total</th><td data-stat="goals">2</td>
          <td data-stat="xg">1.1</td><td data-stat="shots">11</td>
          <td data-stat="shots_on_target">11</td></tr>
      </tbody></table>
      <table id="stats_cff3d9bb_summary"><tbody>
        {rows("awayp", "Away Player")}
        <tr><th data-stat="player">Team Total</th><td data-stat="goals">1</td>
          <td data-stat="xg">1.1</td><td data-stat="shots">11</td>
          <td data-stat="shots_on_target">11</td></tr>
      </tbody></table>
    </body></html>
    """.encode()


def _seed_team_observation_replay(
    tmp_path: Path,
    *,
    content: bytes | None = None,
    known_at: datetime = OBSERVED_AT,
    observed_at: datetime = OBSERVED_AT,
):
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    result_ref = _seed_first_result(canonical, schedule)
    ingest = ingest_fbref_match_report(
        content or _report_content(),
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=known_at,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
    )
    return archive, canonical, season, schedule, result_ref, ingest


def test_match_report_parses_normal_and_comment_wrapped_summary_tables() -> None:
    content = (ROOT / "tests/fixtures/fbref_premier_league_match_report.html").read_bytes()

    parsed = parse_match_report(content)

    assert [team.source_team_id for team in parsed.teams] == ["18bb7c10", "cff3d9bb"]
    assert len(parsed.teams[0].players) == 2
    assert parsed.teams[0].aggregated_stats == pytest.approx(
        {"goals": 9.0, "xg": 1.7, "shots": 7.0, "shots_on_target": 4.0}
    )
    assert parsed.teams[1].players[0].starter is True


def test_match_report_identity_and_auxiliary_tables_are_retained() -> None:
    parsed = parse_match_report(_report_content())

    assert parsed.identity.source_match_id == "aaaaaaaa"
    assert parsed.identity.competition_source_id == "9"
    assert parsed.identity.played_on.isoformat() == "2025-08-15"
    assert parsed.identity.home_source_id == "18bb7c10"
    assert parsed.identity.away_source_id == "cff3d9bb"
    assert (parsed.identity.home_goals, parsed.identity.away_goals) == (2, 1)
    assert parsed.parser_version == "fbref-match-report/2"
    assert set(parsed.tables_present) == {
        "summary",
        "passing",
        "passing_types",
        "defense",
        "possession",
        "misc",
        "keeper",
    }

    home = parsed.teams[0]
    assert home.table_stats["passing"]["passes_completed"] == 31
    player = next(item for item in home.players if item.source_player_id == "playera1")
    assert player.metrics["passes_completed"] == 24
    assert player.metrics["passes_into_final_third"] == 2
    assert player.metrics["tackles_won"] == 1
    assert player.metrics["touches"] == 37
    assert player.metrics["fouls"] == 1
    away = parsed.teams[1]
    assert away.players[0].metrics["saves"] == 4


def test_match_report_required_auxiliary_table_missing_is_diagnostic() -> None:
    parsed = parse_match_report(_report_content(), required_tables=("summary", "keeper"))

    missing = [item for item in parsed.diagnostics if item.code == "required_report_table_missing"]
    assert {item.subject_id for item in missing} == {"18bb7c10"}


def test_match_report_parse_result_exposes_required_and_missing_tables() -> None:
    parsed = parse_match_report(_report_content(), required_tables=("summary", "keeper"))

    assert parsed.required_tables == ("summary", "keeper")
    assert parsed.missing_required_tables == ("keeper",)
    assert [item.source_team_id for item in parsed.teams if item.missing_tables] == ["18bb7c10"]


def test_match_report_required_table_gate_records_failure_before_fact_write(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)

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
            required_tables=("summary", "keeper"),
        )

    assert caught.value.code == "match_report_parse_diagnostics"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.diagnostic_code == "match_report_parse_diagnostics"
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0


def test_default_production_contract_blocks_incomplete_report_and_persists_contract(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)

    with pytest.raises(MatchReportIngestError) as caught:
        _ingest_fbref_match_report(
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

    assert caught.value.code == "match_report_parse_diagnostics"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.collector_version == PRODUCTION_COLLECTOR_VERSION
    coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    row = next(item for item in coverage.fixture_coverage if item.fixture_id == "aaaaaaaa")
    assert row.status == "failed"
    assert row.production_contract_satisfied is False
    assert row.report_contract_verified is False
    assert row.report_contract_diagnostics[0].startswith("report_contract_missing:")
    assert coverage.report_collection_complete is False


def test_complete_report_contract_is_persisted_and_can_pass_strong_fixture_gate(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    content = _complete_report_content()

    result = _ingest_fbref_match_report(
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
    with canonical.connect() as connection:
        report_lineups = connection.execute(
            "SELECT official FROM lineup_facts WHERE match_id = ?",
            (schedule.canonical_match_ids[0],),
        ).fetchall()
    assert report_lineups
    assert {row["official"] for row in report_lineups} == {0}
    contracts, diagnostics = canonical.match_report_contract_audit(season.id)
    assert diagnostics == ()
    assert [item.contract_id for item in contracts] == [result.contract_id]
    evidence = contracts[0]
    assert set(evidence.required_tables) == {
        "summary",
        "passing",
        "passing_types",
        "defense",
        "possession",
        "misc",
        "keeper",
    }

    coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    row = next(item for item in coverage.fixture_coverage if item.fixture_id == "aaaaaaaa")
    assert row.production_contract_satisfied is True
    without_archive = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
    )
    assert without_archive.report_contract_diagnostics == (
        f"match_report_contract_replay_unavailable:{result.contract_id}",
    )
    unverified_row = next(
        item for item in without_archive.fixture_coverage if item.fixture_id == "aaaaaaaa"
    )
    assert unverified_row.report_contract_verified is False
    assert unverified_row.production_contract_satisfied is False
    complete = SeasonCoverage(
        expected_matches=1,
        actual_matches=1,
        expected_teams=2,
        actual_teams=2,
        duplicate_fixture_ids=(),
        unregistered_team_ids=(),
        missing_registered_teams=(),
        structural_violations=(),
        missing_collection_attempts=(),
        blocking_diagnostics=(),
        fixture_coverage=(row,),
    )
    assert complete.report_collection_complete is True

    with canonical.connect() as connection:
        connection.execute(
            "UPDATE match_report_contracts SET required_tables_json = ? WHERE contract_id = ?",
            ('["summary"]', result.contract_id),
        )
    tampered_contracts, tamper_diagnostics = canonical.match_report_contract_audit(season.id)
    assert tampered_contracts == ()
    assert tamper_diagnostics == (f"match_report_contract_invalid:{result.contract_id}",)
    tampered_coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    assert tampered_coverage.report_contract_diagnostics == tamper_diagnostics
    tampered_row = next(
        item for item in tampered_coverage.fixture_coverage if item.fixture_id == "aaaaaaaa"
    )
    assert tampered_row.report_contract_verified is False
    assert tampered_row.production_contract_satisfied is False


def test_forged_report_contract_cannot_pass_without_matching_raw_parser_output(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    fixture = schedule.parsed.matches[0]
    match_id = MatchId(schedule.canonical_match_ids[0])
    observed_at = OBSERVED_AT + timedelta(seconds=1)
    report_url = "https://fbref.example/en/matches/aaaaaaaa/report"
    asset = archive.archive(
        _report_content(),
        source="fbref",
        source_id="aaaaaaaa",
        url=report_url,
        observed_at=observed_at,
        target_event_time=OBSERVED_AT,
        collector_version=PRODUCTION_COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    attempt = canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id="aaaaaaaa",
        target_url=report_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=observed_at,
        collector_version=PRODUCTION_COLLECTOR_VERSION,
        raw_asset_id=asset.id,
    )
    claimed_tables = {
        fixture.home_source_id: PRODUCTION_REQUIRED_TABLES,
        fixture.away_source_id: PRODUCTION_REQUIRED_TABLES,
    }
    contract = canonical.record_match_report_contract(
        collection_attempt_id=attempt.id,
        match_id=match_id,
        match_version=1,
        raw_asset_id=asset.id,
        source_match_id="aaaaaaaa",
        parser_version=REPORT_PARSER_VERSION,
        required_tables=PRODUCTION_REQUIRED_TABLES,
        team_tables=claimed_tables,
        observed_at=observed_at,
    )

    structurally_valid, structural_diagnostics = canonical.match_report_contract_audit(season.id)
    assert structurally_valid == (contract,)
    assert structural_diagnostics == ()

    coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    assert coverage.report_contract_diagnostics == (
        f"match_report_contract_replay_mismatch:{contract.contract_id}",
    )
    row = next(item for item in coverage.fixture_coverage if item.fixture_id == "aaaaaaaa")
    assert row.status == "succeeded"
    assert row.report_contract_verified is False
    assert row.production_contract_satisfied is False
    assert coverage.report_collection_complete is False


@pytest.mark.parametrize(
    ("replacements", "expected_code"),
    [
        (
            (
                (
                    b"/en/squads/18bb7c10/Arsenal-Stats",
                    b"/en/squads/cff3d9bb/Arsenal-Stats",
                ),
                (
                    b"/en/squads/cff3d9bb/Chelsea-Stats",
                    b"/en/squads/18bb7c10/Chelsea-Stats",
                ),
            ),
            "report_team_identity_mismatch",
        ),
        (
            ((b"/comps/9/2025-2026/", b"/comps/8/2025-2026/"),),
            "report_competition_identity_mismatch",
        ),
        (
            ((b"/comps/9/2025-2026/", b"/comps/9/2024-2025/"),),
            "report_season_identity_mismatch",
        ),
        (
            ((b'data-venue-date="2025-08-15"', b'data-venue-date="2025-08-16"'),),
            "report_date_identity_mismatch",
        ),
        (
            ((b'<div class="score">2</div>', b'<div class="score">9</div>'),),
            "report_score_identity_mismatch",
        ),
    ],
)
def test_self_consistent_raw_contract_must_match_canonical_fixture_identity(
    tmp_path: Path,
    replacements: tuple[tuple[bytes, bytes], ...],
    expected_code: str,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    content = _complete_report_content()
    for old, new in replacements:
        content = content.replace(old, new)
    parsed = parse_match_report(content, required_tables=PRODUCTION_REQUIRED_TABLES)
    assert parsed.missing_required_tables == ()

    match_id = MatchId(schedule.canonical_match_ids[0])
    observed_at = OBSERVED_AT + timedelta(seconds=1)
    report_url = "https://fbref.example/en/matches/aaaaaaaa/report"
    asset = archive.archive(
        content,
        source="fbref",
        source_id="aaaaaaaa",
        url=report_url,
        observed_at=observed_at,
        target_event_time=OBSERVED_AT,
        collector_version=PRODUCTION_COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    attempt = canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id="aaaaaaaa",
        target_url=report_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=observed_at,
        collector_version=PRODUCTION_COLLECTOR_VERSION,
        raw_asset_id=asset.id,
    )
    contract = canonical.record_match_report_contract(
        collection_attempt_id=attempt.id,
        match_id=match_id,
        match_version=1,
        raw_asset_id=asset.id,
        source_match_id="aaaaaaaa",
        parser_version=parsed.parser_version,
        required_tables=parsed.required_tables,
        team_tables={team.source_team_id: team.tables_present for team in parsed.teams},
        observed_at=observed_at,
    )
    structurally_valid, structural_diagnostics = canonical.match_report_contract_audit(season.id)
    assert structurally_valid == (contract,)
    assert structural_diagnostics == ()

    coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    assert coverage.report_contract_diagnostics == (
        f"match_report_contract_identity_mismatch:{contract.contract_id}:{expected_code}",
    )
    row = next(item for item in coverage.fixture_coverage if item.fixture_id == "aaaaaaaa")
    assert row.status == "succeeded"
    assert row.report_contract_verified is False
    assert row.production_contract_satisfied is False
    assert coverage.report_collection_complete is False


def test_match_report_identity_conflicts_are_visible() -> None:
    content = _report_content().replace(
        b'data-venue-date="2025-08-15"',
        b'data-venue-date="2025-08-15"><span data-venue-date="2025-08-16"></span>',
    )
    content = content.replace(
        b"</body>",
        b'<link rel="canonical" href="https://fbref.com/en/matches/bbbbbbbb/other"></body>',
    )
    parsed = parse_match_report(content)

    assert parsed.identity.played_on is None
    assert parsed.identity.source_match_id is None
    assert {item.code for item in parsed.diagnostics} >= {
        "match_date_conflict",
        "match_source_id_conflict",
    }


def test_match_report_unsupported_table_has_table_diagnostic() -> None:
    content = _report_content().replace(
        b"</body>",
        b'<table id="stats_18bb7c10_unknown"><tr><th data-stat="player">x</th></tr></table></body>',
    )
    parsed = parse_match_report(content)

    diagnostic = next(
        item for item in parsed.diagnostics if item.code == "unsupported_report_table"
    )
    assert diagnostic.table_id == "stats_18bb7c10_unknown"


@pytest.mark.parametrize(
    "extra_table",
    (
        b'<table id="stats_18bb7c10_passing"><tr><th data-stat="player">'
        b'<a href="/en/players/playera1/A-One">A One</a></th>'
        b'<td data-stat="passes_completed">999</td></tr></table>',
        b'<table id="stats_18bb7c10_unknown"><tr><th data-stat="player">x</th></tr></table>',
    ),
)
def test_match_report_table_conflicts_are_blocked_before_fact_write(
    tmp_path: Path,
    extra_table: bytes,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content().replace(b"</body>", extra_table + b"</body>"),
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

    assert caught.value.code == "match_report_parse_diagnostics"
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.diagnostic_code == "match_report_parse_diagnostics"
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM player_match_observations").fetchone()[0] == 0
        )


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
    assert report_attempts[0].collector_version.startswith(
        "fbref-match-report/3+custom-tables:summary"
    )
    coverage = assess_season_coverage(
        schedule.parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    fixture_coverage = next(
        item for item in coverage.fixture_coverage if item.fixture_id == "aaaaaaaa"
    )
    assert fixture_coverage.status == "succeeded"
    assert fixture_coverage.report_contract_verified is True
    assert fixture_coverage.report_contract_id == first.contract_id
    assert fixture_coverage.report_contract_required_tables == ("summary",)
    assert fixture_coverage.production_contract_satisfied is False
    assert coverage.report_collection_complete is False
    home = canonical.mapped_team(source="fbref", source_id="18bb7c10")
    with canonical.connect() as connection:
        stats = connection.execute(
            "SELECT stats_json FROM team_match_observations WHERE team_id = ?",
            (home.id.value,),
        ).fetchone()["stats_json"]
        result_count = connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0]
    assert '"goals":2' in stats
    assert result_count == 1


def test_verified_match_report_player_batch_replays_players_and_actual_lineup(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    match_id = MatchId(schedule.canonical_match_ids[0])

    batch = load_verified_match_report_player_batch(
        ingest.contract_id,
        archive=archive,
        canonical=canonical,
    )

    assert batch.contract_id == ingest.contract_id
    assert batch.match_id == match_id
    assert batch.match_version == 1
    assert batch.raw_asset_id.value == ingest.raw_asset_id
    assert batch.known_at == OBSERVED_AT == batch.observed_at
    assert {item.record_id for item in batch.player_observations} == set(ingest.player_fact_ids)
    assert len(batch.actual_lineup_facts) == 3
    assert sum(item.lineup_role == "starter" for item in batch.actual_lineup_facts) == 2
    assert all(not item.official for item in batch.actual_lineup_facts)
    assert {
        load_verified_player_observation(
            item.record_id,
            archive=archive,
            canonical=canonical,
        ).record_id
        for item in batch.player_observations
    } == {item.record_id for item in batch.player_observations}


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("role", "GK"),
        ("minutes", 1.0),
        ("metrics_json", '{"shots":999.0}'),
        ("known_at", "2026-07-16T05:59:00Z"),
        ("observed_at", "2026-07-16T05:59:00Z"),
    ],
)
def test_verified_player_observation_rejects_canonical_semantic_tampering(
    tmp_path: Path,
    column: str,
    replacement: object,
) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    reference = ingest.player_fact_ids[0]
    with canonical.connect() as connection:
        connection.execute(
            f"UPDATE player_match_observations SET {column} = ? WHERE record_id = ?",
            (replacement, reference),
        )

    with pytest.raises(ValueError, match="player observation|player fact batch"):
        load_verified_player_observation(reference, archive=archive, canonical=canonical)


def test_verified_player_batch_rejects_missing_actual_lineup_fact(tmp_path: Path) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    with canonical.connect() as connection:
        reference = connection.execute(
            "SELECT record_id FROM lineup_facts WHERE official = 0 ORDER BY record_id LIMIT 1"
        ).fetchone()["record_id"]
        connection.execute("DELETE FROM lineup_facts WHERE record_id = ?", (reference,))

    with pytest.raises(ValueError, match="actual lineup|player fact batch"):
        load_verified_match_report_player_batch(
            ingest.contract_id,
            archive=archive,
            canonical=canonical,
        )


def test_player_observation_version_is_immutable_and_history_is_replayed(tmp_path: Path) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    reference = ingest.player_fact_ids[0]

    with canonical.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="observation_version is immutable"):
            connection.execute(
                "UPDATE player_match_observations SET observation_version = 99 WHERE record_id = ?",
                (reference,),
            )

    assert (
        load_verified_player_observation(reference, archive=archive, canonical=canonical).record_id
        == reference
    )

    with canonical.connect() as connection:
        connection.execute("DROP TRIGGER player_match_observations_version_immutable")
        connection.execute(
            "UPDATE player_match_observations SET observation_version = 2 WHERE record_id = ?",
            (reference,),
        )

    with pytest.raises(ValueError, match="observation version history"):
        load_verified_player_observation(reference, archive=archive, canonical=canonical)
    with pytest.raises(ValueError, match="observation version history"):
        load_verified_match_report_player_batch(
            ingest.contract_id,
            archive=archive,
            canonical=canonical,
        )


def test_player_batch_accepts_reused_semantic_facts_with_new_contract_evidence(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    second = ingest_fbref_match_report(
        _report_content(),
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )

    assert second.player_fact_ids == first.player_fact_ids
    assert second.raw_asset_id != first.raw_asset_id
    batch = load_verified_match_report_player_batch(
        second.contract_id,
        archive=archive,
        canonical=canonical,
    )
    assert batch.raw_asset_id.value == second.raw_asset_id
    assert batch.observed_at == later
    assert {item.record_id for item in batch.player_observations} == set(first.player_fact_ids)
    with canonical.connect() as connection:
        assert {
            row["raw_asset_id"]
            for row in connection.execute(
                "SELECT raw_asset_id FROM fact_evidence WHERE record_id = ?",
                (first.player_fact_ids[0],),
            ).fetchall()
        } == {first.raw_asset_id, second.raw_asset_id}

    facts = CanonicalFactStore(canonical, raw_archive=archive)
    historical = facts.availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=OBSERVED_AT + timedelta(minutes=30),
    )
    latest = facts.availability(MatchId(schedule.canonical_match_ids[0]), as_of=later)
    assert historical.player_fact_contract_id == first.contract_id
    assert historical.player_fact_raw_asset_id == first.raw_asset_id
    assert latest.player_fact_contract_id == second.contract_id
    assert latest.player_fact_raw_asset_id == second.raw_asset_id
    assert latest.player_fact_refs == historical.player_fact_refs


def test_player_fact_cache_is_independent_of_contract_batch_load_order(tmp_path: Path) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    second = ingest_fbref_match_report(
        _report_content(),
        page_url="https://fbref.example/en/matches/aaaaaaaa/report?revision=2",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    player_ref = first.player_fact_ids[0]
    with canonical.connect() as connection:
        lineup_ref = connection.execute(
            "SELECT record_id FROM lineup_facts WHERE official = 0 ORDER BY record_id LIMIT 1"
        ).fetchone()["record_id"]

    with VerificationSession(DataLayout(tmp_path / "data"), canonical=canonical) as session:
        old_before = load_verified_player_observation(
            player_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        old_lineup_before = load_verified_actual_lineup_fact(
            lineup_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        new_batch = load_verified_match_report_player_batch(
            second.contract_id,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        old_after = load_verified_player_observation(
            player_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        old_lineup_after = load_verified_actual_lineup_fact(
            lineup_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
    assert old_before == old_after
    assert old_after.contract_id == first.contract_id
    assert old_after.observed_at == OBSERVED_AT
    assert old_lineup_before == old_lineup_after
    assert old_lineup_after.contract_id == first.contract_id
    assert new_batch.observed_at == later

    with VerificationSession(DataLayout(tmp_path / "data"), canonical=canonical) as session:
        new_first = load_verified_match_report_player_batch(
            second.contract_id,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        old = load_verified_player_observation(
            player_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        old_lineup = load_verified_actual_lineup_fact(
            lineup_ref,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
        new_again = load_verified_match_report_player_batch(
            second.contract_id,
            archive=archive,
            canonical=canonical,
            verification_session=session,
        )
    assert new_first == new_again
    assert new_again.observed_at == later
    assert old.contract_id == first.contract_id
    assert old.observed_at == OBSERVED_AT
    assert old_lineup.contract_id == first.contract_id
    assert old_lineup.observed_at == OBSERVED_AT


def test_player_batch_rejects_cross_match_fact_bound_to_same_raw_evidence(tmp_path: Path) -> None:
    archive, canonical, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    player = canonical.mapped_player(source="fbref", source_id="playera1")
    team = canonical.mapped_team(source="fbref", source_id="18bb7c10")
    CanonicalFactStore(canonical).append_player_observation(
        match_id=MatchId(schedule.canonical_match_ids[1]),
        match_version=1,
        team_id=team.id,
        player_id=player.id,
        role="FW",
        minutes=10,
        metrics={"shots": 1.0},
        known_at=OBSERVED_AT,
        observed_at=OBSERVED_AT,
        raw_asset_id=RawAssetId(ingest.raw_asset_id),
    )

    with pytest.raises(ValueError, match="player set"):
        load_verified_match_report_player_batch(
            ingest.contract_id,
            archive=archive,
            canonical=canonical,
        )


def test_player_profile_qualification_binds_exact_verified_report_batch_refs(
    tmp_path: Path,
) -> None:
    archive, _, _, schedule, result_ref, ingest = _seed_team_observation_replay(tmp_path)
    canonical = CanonicalStore(DataLayout(tmp_path / "data").canonical / "platform.sqlite3")
    batch = load_verified_match_report_player_batch(
        ingest.contract_id,
        archive=archive,
        canonical=canonical,
    )
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))

    qualification = store.create_training_qualification(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        qualification=Qualification.PLAYER_PROFILE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=None,
        result_ref=result_ref,
    )

    assert not qualification.passed
    assert "typed_player_fact_replay_unavailable" not in qualification.reason_codes
    assert any(reason.startswith("starter_count:") for reason in qualification.reason_codes)
    assert set(qualification.fact_refs) == {*batch.fact_refs, ingest.contract_id}
    assert store.load_training_qualification(qualification.qualification_id) == qualification


def test_complete_player_report_batch_passes_player_profile_and_training_qualification(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule = _prepare_schedule(tmp_path)
    result_ref = _seed_first_result(canonical, schedule)
    match_id = MatchId(schedule.canonical_match_ids[0])
    ingest = ingest_fbref_match_report(
        _full_player_profile_report_content(),
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
    batch = load_verified_match_report_player_batch(
        ingest.contract_id,
        archive=archive,
        canonical=canonical,
    )
    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        match_id,
        as_of=OBSERVED_AT,
    )
    assessment = assess_lifecycle(availability, evaluated_at=OBSERVED_AT)
    qualifications = {item.qualification: item for item in assessment.qualifications}

    assert len(batch.player_observations) == 22
    assert len(batch.actual_lineup_facts) == 22
    assert all(len(availability.starters[team_id]) == 11 for team_id in availability.team_ids)
    assert qualifications[Qualification.PLAYER_PROFILE].passed
    assert assessment.state.value == "training-ready"
    scheduled = assess_lifecycle(
        replace(availability, match_status=MatchStatus.SCHEDULED),
        evaluated_at=OBSERVED_AT,
    )
    assert scheduled.state.value == "discovered"

    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    qualification = store.create_training_qualification(
        match_id=match_id,
        match_version=1,
        qualification=Qualification.PLAYER_PROFILE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=None,
        result_ref=result_ref,
    )
    assert qualification.passed
    assert set(qualification.fact_refs) == {*batch.fact_refs, ingest.contract_id}
    assert store.load_training_qualification(qualification.qualification_id) == qualification


def test_player_availability_keeps_invalid_latest_batch_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    changed = _report_content().replace(
        b'<td data-stat="xg">1.4</td>',
        b'<td data-stat="xg">1.5</td>',
        1,
    )
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report?revision=2",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    changed_ref = next(iter(set(second.player_fact_ids) - set(first.player_fact_ids)))
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE player_match_observations SET metrics_json = ? WHERE record_id = ?",
            ('{"xg":999.0}', changed_ref),
        )

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=later,
    )

    assert availability.player_fact_contract_id is None
    assert availability.player_fact_contract_candidates == (second.contract_id,)
    assert availability.player_fact_diagnostic == "typed_player_fact_batch_replay_invalid"
    assert changed_ref in availability.player_fact_candidate_refs
    assert availability.player_fact_refs == ()
    assert availability.player_observation_ids == frozenset()
    qualifications = {
        item.qualification: item
        for item in assess_lifecycle(availability, evaluated_at=later).qualifications
    }
    assert (
        "typed_player_fact_batch_replay_invalid"
        in qualifications[Qualification.PLAYER_PROFILE].reason_codes
    )
    assert (
        "typed_player_fact_batch_replay_invalid"
        not in qualifications[Qualification.SCORE_MODEL].reason_codes
    )
    assert (
        "typed_player_fact_batch_replay_invalid"
        not in qualifications[Qualification.TEAM_BASELINE].reason_codes
    )


def test_player_availability_rejects_non_equivalent_contracts_at_same_observed_at(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    changed = _report_content().replace(
        b'<td data-stat="xg">1.4</td>',
        b'<td data-stat="xg">1.5</td>',
        1,
    )
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report?revision=2",
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

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=OBSERVED_AT,
    )

    assert availability.player_fact_contract_id is None
    assert availability.player_fact_contract_candidates == tuple(
        sorted((first.contract_id, second.contract_id))
    )
    assert availability.player_fact_diagnostic == "typed_player_fact_batch_ambiguous"
    assert availability.player_fact_refs == ()


def test_verified_team_observation_replays_raw_contract_and_team_mapping(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    match_id = MatchId(schedule.canonical_match_ids[0])
    match = canonical.match(match_id)

    observations = tuple(
        load_verified_team_observation(
            reference,
            archive=archive,
            canonical=canonical,
        )
        for reference in ingest.team_fact_ids
    )

    assert {item.team_id for item in observations} == {
        match.home_team_id,
        match.away_team_id,
    }
    assert {item.record_id for item in observations} == set(ingest.team_fact_ids)
    assert {item.raw_asset_id.value for item in observations} == {ingest.raw_asset_id}
    assert all(item.match_id == match_id and item.match_version == 1 for item in observations)
    assert all(item.known_at == OBSERVED_AT == item.observed_at for item in observations)
    home = next(item for item in observations if item.team_id == match.home_team_id)
    away = next(item for item in observations if item.team_id == match.away_team_id)
    assert home.stats == {"goals": 2, "shots": 7.0, "shots_on_target": 4.0, "xg": 1.7}
    assert away.stats == {"goals": 1, "shots": 4.0, "shots_on_target": 2.0, "xg": 0.8}
    with pytest.raises(TypeError):
        home.stats["xg"] = 99.0  # type: ignore[index]

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        match_id,
        as_of=OBSERVED_AT,
    )
    assert availability.team_stat_refs == {
        item.team_id.value: item.record_id for item in observations
    }
    assert availability.team_stat_pair_diagnostic is None
    assert set(availability.team_stat_contract_ids.values()) == {ingest.contract_id}
    assert set(availability.team_stat_raw_asset_ids.values()) == {ingest.raw_asset_id}
    assert set(availability.team_stats_observed_at.values()) == {OBSERVED_AT}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("content_id", "content ID"),
        ("stats_value", "content ID"),
        ("stats_type", "finite number or None"),
        ("stats_json", "not canonical"),
        ("evidence", "exact raw evidence"),
        ("raw_registration", "raw registration conflicts"),
        ("temporal", "known_at cannot be later"),
    ),
)
def test_verified_team_observation_rejects_canonical_fact_tampering(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    reference = ingest.team_fact_ids[0]
    candidate = reference
    with canonical.connect() as connection:
        if mutation == "content_id":
            candidate = "fact:team_match_observations:" + "0" * 64
            connection.execute(
                "UPDATE team_match_observations SET record_id = ? WHERE record_id = ?",
                (candidate, reference),
            )
            connection.execute(
                "UPDATE fact_evidence SET record_id = ? WHERE record_id = ?",
                (candidate, reference),
            )
        elif mutation in {"stats_value", "stats_type", "stats_json"}:
            stored = connection.execute(
                "SELECT stats_json FROM team_match_observations WHERE record_id = ?",
                (reference,),
            ).fetchone()["stats_json"]
            stats = json.loads(stored)
            if mutation == "stats_value":
                stats["xg"] = 99.0
                replacement = json.dumps(stats, separators=(",", ":"), sort_keys=True)
            elif mutation == "stats_type":
                stats["xg"] = True
                replacement = json.dumps(stats, separators=(",", ":"), sort_keys=True)
            else:
                replacement = stored + " "
            connection.execute(
                "UPDATE team_match_observations SET stats_json = ? WHERE record_id = ?",
                (replacement, reference),
            )
        elif mutation == "evidence":
            connection.execute("DELETE FROM fact_evidence WHERE record_id = ?", (reference,))
        elif mutation == "raw_registration":
            connection.execute(
                "UPDATE raw_assets SET url = ? WHERE raw_asset_id = ?",
                ("https://tampered.example/report", ingest.raw_asset_id),
            )
        else:
            earlier = (OBSERVED_AT - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
            connection.execute(
                "UPDATE team_match_observations SET observed_at = ? WHERE record_id = ?",
                (earlier, reference),
            )
            connection.execute(
                "UPDATE fact_evidence SET observed_at = ? WHERE record_id = ?",
                (earlier, reference),
            )

    with pytest.raises(ValueError, match=message):
        load_verified_team_observation(candidate, archive=archive, canonical=canonical)


def test_verified_team_observation_rejects_modified_raw_bytes(tmp_path: Path) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    asset = archive.load(RawAssetId(ingest.raw_asset_id))
    archive.layout.raw_object_path(asset.checksum).write_bytes(b"tampered report bytes")

    with pytest.raises(ValueError, match="checksum"):
        load_verified_team_observation(
            ingest.team_fact_ids[0],
            archive=archive,
            canonical=canonical,
        )


@pytest.mark.parametrize(
    ("tampered_field", "message"),
    (("stats_json", "stats do not match"), ("known_at", "known_at does not match")),
)
def test_verified_team_observation_replays_raw_after_recomputed_fact_identity(
    tmp_path: Path,
    tampered_field: str,
    message: str,
) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    reference = ingest.team_fact_ids[0]
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT match_id, match_version, team_id, known_at, observed_at, stats_json, "
            "raw_asset_id FROM team_match_observations WHERE record_id = ?",
            (reference,),
        ).fetchone()
        payload = dict(row)
        if tampered_field == "stats_json":
            stats = json.loads(payload["stats_json"])
            stats["xg"] = 99.0
            payload["stats_json"] = json.dumps(stats, separators=(",", ":"), sort_keys=True)
        else:
            payload["known_at"] = (
                (OBSERVED_AT - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
            )
        semantic = {
            key: value
            for key, value in payload.items()
            if key not in {"observed_at", "raw_asset_id"}
        }
        digest = hashlib.sha256(
            json.dumps(
                semantic,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        forged_ref = f"fact:team_match_observations:{digest}"
        connection.execute(
            f"UPDATE team_match_observations SET {tampered_field} = ?, record_id = ? "
            "WHERE record_id = ?",
            (payload[tampered_field], forged_ref, reference),
        )
        connection.execute(
            "UPDATE fact_evidence SET record_id = ? WHERE record_id = ?",
            (forged_ref, reference),
        )

    with pytest.raises(ValueError, match=message):
        load_verified_team_observation(forged_ref, archive=archive, canonical=canonical)


def test_verified_team_observation_allows_additional_semantic_evidence(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(minutes=1)
    second = ingest_fbref_match_report(
        _report_content(),
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )

    assert second.team_fact_ids == first.team_fact_ids
    with canonical.connect() as connection:
        evidence_counts = {
            reference: connection.execute(
                "SELECT COUNT(*) FROM fact_evidence WHERE record_id = ?", (reference,)
            ).fetchone()[0]
            for reference in first.team_fact_ids
        }
    assert set(evidence_counts.values()) == {2}
    for reference in first.team_fact_ids:
        assert (
            load_verified_team_observation(
                reference,
                archive=archive,
                canonical=canonical,
            ).record_id
            == reference
        )


@pytest.mark.parametrize("contract_state", ("missing", "legacy", "duplicate"))
def test_verified_team_observation_requires_one_current_replayable_contract(
    tmp_path: Path,
    contract_state: str,
) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    contract = canonical.match_report_contract(ingest.contract_id)
    if contract_state in {"missing", "legacy"}:
        with canonical.connect() as connection:
            connection.execute(
                "DELETE FROM match_report_contracts WHERE contract_id = ?",
                (contract.contract_id,),
            )
    if contract_state == "legacy":
        canonical.record_match_report_contract(
            collection_attempt_id=contract.collection_attempt_id,
            match_id=contract.match_id,
            match_version=contract.match_version,
            raw_asset_id=contract.raw_asset_id,
            source_match_id=contract.source_match_id,
            parser_version="fbref-match-report/legacy",
            required_tables=contract.required_tables,
            team_tables=dict(contract.team_tables),
            observed_at=contract.observed_at,
        )
    if contract_state == "duplicate":
        duplicate_attempt_id = CollectionAttemptId("collection-attempt:duplicate-report")
        with canonical.connect() as connection:
            connection.execute(
                "INSERT INTO collection_attempts("
                "collection_attempt_id, match_id, source, source_id, target_url, outcome, "
                "observed_at, collector_version, diagnostic_code, diagnostic_message, raw_asset_id"
                ") SELECT ?, match_id, source, source_id, target_url, outcome, observed_at, "
                "collector_version, diagnostic_code, diagnostic_message, raw_asset_id "
                "FROM collection_attempts WHERE collection_attempt_id = ?",
                (duplicate_attempt_id.value, contract.collection_attempt_id.value),
            )
        canonical.record_match_report_contract(
            collection_attempt_id=duplicate_attempt_id,
            match_id=contract.match_id,
            match_version=contract.match_version,
            raw_asset_id=contract.raw_asset_id,
            source_match_id=contract.source_match_id,
            parser_version=contract.parser_version,
            required_tables=contract.required_tables,
            team_tables=dict(contract.team_tables),
            observed_at=contract.observed_at,
        )

    message = "exactly one" if contract_state != "legacy" else "parser output"
    with pytest.raises(ValueError, match=message):
        load_verified_team_observation(
            ingest.team_fact_ids[0],
            archive=archive,
            canonical=canonical,
        )


def test_verified_team_observation_recomputes_parser_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, canonical, _, _, _, ingest = _seed_team_observation_replay(tmp_path)
    original = contract_replay_module.parse_match_report

    def altered_parser(*args, **kwargs):
        parsed = original(*args, **kwargs)
        home = parsed.teams[0]
        stats = {**home.aggregated_stats, "xg": 99.0}
        return replace(parsed, teams=(replace(home, aggregated_stats=stats), *parsed.teams[1:]))

    monkeypatch.setattr(contract_replay_module, "parse_match_report", altered_parser)
    with pytest.raises(ValueError, match="stats do not match replayed parser output"):
        load_verified_team_observation(
            ingest.team_fact_ids[0],
            archive=archive,
            canonical=canonical,
        )


def test_verified_team_observation_rejects_persisted_parser_output_change(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    changed = _report_content().replace(b'data-stat="xg">1.7', b'data-stat="xg">1.9')
    later = OBSERVED_AT + timedelta(minutes=1)
    original_asset = archive.load(RawAssetId(ingest.raw_asset_id))
    changed_asset = archive.archive(
        changed,
        source="fbref",
        source_id="aaaaaaaa",
        url=original_asset.url,
        observed_at=later,
        target_event_time=OBSERVED_AT,
        collector_version=original_asset.collector_version,
        media_type=original_asset.media_type,
    )
    canonical.register_raw_asset(changed_asset)
    attempt = canonical.record_collection_attempt(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        source="fbref-match-report",
        source_id="aaaaaaaa",
        target_url=changed_asset.url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=later,
        collector_version=changed_asset.collector_version,
        raw_asset_id=changed_asset.id,
    )
    parsed = parse_match_report(changed, required_tables=("summary",))
    canonical.record_match_report_contract(
        collection_attempt_id=attempt.id,
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        raw_asset_id=changed_asset.id,
        source_match_id="aaaaaaaa",
        parser_version=parsed.parser_version,
        required_tables=parsed.required_tables,
        team_tables={team.source_team_id: team.tables_present for team in parsed.teams},
        observed_at=later,
    )
    reference = ingest.team_fact_ids[0]
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE team_match_observations SET raw_asset_id = ?, observed_at = ? "
            "WHERE record_id = ?",
            (changed_asset.id.value, later.isoformat().replace("+00:00", "Z"), reference),
        )
        connection.execute(
            "INSERT INTO fact_evidence(record_id, raw_asset_id, observed_at) VALUES (?, ?, ?)",
            (
                reference,
                changed_asset.id.value,
                later.isoformat().replace("+00:00", "Z"),
            ),
        )

    with pytest.raises(ValueError, match="stats do not match replayed parser output"):
        load_verified_team_observation(reference, archive=archive, canonical=canonical)


def test_verified_team_observation_rejects_tampered_historical_team_mapping(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    match = canonical.match(MatchId(schedule.canonical_match_ids[0]))
    with canonical.connect() as connection:
        connection.execute("DROP TRIGGER source_mappings_close_only_update")
        connection.execute(
            "UPDATE source_mappings SET entity_id = ? WHERE source = 'fbref' "
            "AND entity_type = 'team' AND source_id = ?",
            (match.away_team_id.value, "18bb7c10"),
        )

    with pytest.raises(ValueError, match="evidence is unavailable"):
        load_verified_team_observation(
            ingest.team_fact_ids[0],
            archive=archive,
            canonical=canonical,
        )


@pytest.mark.parametrize("second_known_at", (OBSERVED_AT, OBSERVED_AT + timedelta(hours=1)))
def test_team_availability_applies_known_and_observed_as_of_boundaries(
    tmp_path: Path,
    second_known_at: datetime,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    changed = _report_content().replace(b'data-stat="xg">1.7', b'data-stat="xg">1.9')
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=second_known_at,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    facts = CanonicalFactStore(canonical, raw_archive=archive)
    match_id = MatchId(schedule.canonical_match_ids[0])

    historical = facts.availability(match_id, as_of=OBSERVED_AT + timedelta(minutes=30))
    latest = facts.availability(match_id)

    assert set(historical.team_stat_refs.values()) == set(first.team_fact_ids)
    assert set(latest.team_stat_refs.values()) == set(second.team_fact_ids)


def test_team_baseline_qualification_rejects_cross_contract_latest_pair(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, result_ref, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    changed = _report_content().replace(b'data-stat="xg">1.7', b'data-stat="xg">1.9')
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    match_id = MatchId(schedule.canonical_match_ids[0])
    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        match_id,
        as_of=later,
    )

    selected_refs = set(availability.team_stat_refs.values())
    assert selected_refs == set(second.team_fact_ids)
    assert selected_refs != set(first.team_fact_ids)
    assert len(set(availability.team_stat_contract_ids.values())) == 2
    assert availability.team_stat_pair_diagnostic == "typed_team_fact_pair_mismatch"
    lifecycle = {
        item.qualification: item
        for item in assess_lifecycle(
            replace(availability, team_stat_pair_diagnostic=None),
            evaluated_at=later,
        ).qualifications
    }
    assert not lifecycle[Qualification.TEAM_BASELINE].passed
    assert "typed_team_fact_pair_mismatch" in lifecycle[Qualification.TEAM_BASELINE].reason_codes

    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    artifact = store.create_training_qualification(
        match_id=match_id,
        match_version=1,
        qualification=Qualification.TEAM_BASELINE,
        ruleset_version="readiness/1",
        evaluated_at=later,
        snapshot_ref=None,
        result_ref=result_ref,
    )

    assert not artifact.passed
    assert "typed_team_fact_pair_mismatch" in artifact.reason_codes
    assert selected_refs <= set(artifact.fact_refs)
    assert store.load_training_qualification(artifact.qualification_id) == artifact


def test_team_availability_does_not_fallback_from_corrupted_latest_observation(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    changed = _report_content().replace(b'data-stat="xg">1.7', b'data-stat="xg">1.9')
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=later,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    newest = next(
        reference for reference in second.team_fact_ids if reference not in first.team_fact_ids
    )
    with canonical.connect() as connection:
        team_id = connection.execute(
            "SELECT team_id FROM team_match_observations WHERE record_id = ?", (newest,)
        ).fetchone()["team_id"]
        old_same_team = connection.execute(
            "SELECT record_id FROM team_match_observations WHERE team_id = ? "
            "AND record_id != ? ORDER BY observation_version DESC LIMIT 1",
            (team_id, newest),
        ).fetchone()["record_id"]
        connection.execute(
            "UPDATE team_match_observations SET stats_json = ? WHERE record_id = ?",
            ('{"goals":2,"shots":7.0,"shots_on_target":4.0,"xg":99.0}', newest),
        )

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0])
    )
    assert team_id not in availability.team_stat_fields
    assert team_id not in availability.team_stat_refs
    assert availability.team_stat_diagnostics == {team_id: "typed_team_fact_replay_invalid"}
    assert old_same_team not in availability.team_stat_refs.values()
    qualifications = {
        item.qualification: item
        for item in assess_lifecycle(availability, evaluated_at=later).qualifications
    }
    diagnostic = f"typed_team_fact_replay_invalid:{team_id}"
    assert diagnostic in qualifications[Qualification.TEAM_BASELINE].reason_codes
    assert diagnostic not in qualifications[Qualification.SCORE_MODEL].reason_codes


def test_team_observation_version_history_rejects_latest_hijack(tmp_path: Path) -> None:
    archive, canonical, _, schedule, _, first = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    changed = _report_content().replace(b'data-stat="xg">1.7', b'data-stat="xg">1.9')
    second = ingest_fbref_match_report(
        changed,
        page_url="https://fbref.example/en/matches/aaaaaaaa/report",
        source_match_id="aaaaaaaa",
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=later,
        observed_at=later,
        archive=archive,
        canonical=canonical,
    )
    newest = next(
        reference for reference in second.team_fact_ids if reference not in first.team_fact_ids
    )
    with canonical.connect() as connection:
        team_id = connection.execute(
            "SELECT team_id FROM team_match_observations WHERE record_id = ?", (newest,)
        ).fetchone()["team_id"]
        oldest = connection.execute(
            "SELECT record_id FROM team_match_observations WHERE team_id = ? "
            "ORDER BY observation_version LIMIT 1",
            (team_id,),
        ).fetchone()["record_id"]
        with pytest.raises(sqlite3.IntegrityError, match="observation_version is immutable"):
            connection.execute(
                "UPDATE team_match_observations SET observation_version = 3 WHERE record_id = ?",
                (oldest,),
            )
        connection.execute("DROP TRIGGER team_match_observations_version_immutable")
        connection.execute(
            "UPDATE team_match_observations SET observation_version = 3 WHERE record_id = ?",
            (oldest,),
        )

    with pytest.raises(ValueError, match="observation version history"):
        load_verified_team_observation(oldest, archive=archive, canonical=canonical)

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0])
    )
    assert team_id not in availability.team_stat_refs
    assert availability.team_stat_diagnostics[team_id] == "typed_team_fact_replay_invalid"


def test_availability_rejects_result_without_exact_evidence_but_keeps_team_facts(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, result_ref, ingest = _seed_team_observation_replay(tmp_path)
    with canonical.connect() as connection:
        connection.execute("DELETE FROM fact_evidence WHERE record_id = ?", (result_ref,))

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=OBSERVED_AT,
    )

    assert not availability.result_90_present
    assert availability.result_90_ref is None
    assert availability.result_90_diagnostic == "typed_result_fact_replay_invalid"
    assert set(availability.team_stat_refs.values()) == set(ingest.team_fact_ids)
    qualifications = {
        item.qualification: item
        for item in assess_lifecycle(availability, evaluated_at=OBSERVED_AT).qualifications
    }
    for qualification in (Qualification.SCORE_MODEL, Qualification.TEAM_BASELINE):
        assert "typed_result_fact_replay_invalid" in qualifications[qualification].reason_codes


def test_availability_does_not_fallback_from_latest_result_with_tampered_raw(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, old_result_ref, ingest = _seed_team_observation_replay(
        tmp_path
    )
    later = OBSERVED_AT + timedelta(hours=1)
    correction_raw = archive.archive(
        b"independent result correction",
        source="result-correction-test",
        source_id="aaaaaaaa-correction",
        url="https://result.example/aaaaaaaa-correction",
        observed_at=later,
        target_event_time=later,
        collector_version="result-correction/1",
        media_type="application/octet-stream",
    )
    canonical.register_raw_asset(correction_raw)
    correction = CanonicalFactStore(canonical).append_result_90(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=3,
        away_goals=1,
        known_at=later,
        observed_at=later,
        raw_asset_id=correction_raw.id,
    )
    archive.layout.raw_object_path(correction_raw.checksum).write_bytes(b"tampered correction")

    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=later,
    )

    assert not availability.result_90_present
    assert availability.result_90_ref is None
    assert availability.result_90_diagnostic == "typed_result_fact_replay_invalid"
    assert old_result_ref != correction.record_id
    assert old_result_ref != availability.result_90_ref
    assert set(availability.team_stat_refs.values()) == set(ingest.team_fact_ids)


def test_result_observation_version_history_rejects_latest_hijack(tmp_path: Path) -> None:
    archive, canonical, _, schedule, old_result_ref, _ = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    correction_raw = archive.archive(
        b"versioned result correction",
        source="result-correction-test",
        source_id="aaaaaaaa-versioned-correction",
        url="https://result.example/aaaaaaaa-versioned-correction",
        observed_at=later,
        target_event_time=later,
        collector_version="result-correction/1",
        media_type="application/octet-stream",
    )
    canonical.register_raw_asset(correction_raw)
    CanonicalFactStore(canonical).append_result_90(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=3,
        away_goals=1,
        known_at=later,
        observed_at=later,
        raw_asset_id=correction_raw.id,
    )
    with canonical.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="observation_version is immutable"):
            connection.execute(
                "UPDATE match_results_90 SET observation_version = 3 WHERE record_id = ?",
                (old_result_ref,),
            )
        connection.execute("DROP TRIGGER match_results_90_version_immutable")
        connection.execute(
            "UPDATE match_results_90 SET observation_version = 3 WHERE record_id = ?",
            (old_result_ref,),
        )

    with pytest.raises(ValueError, match="observation version history"):
        load_verified_match_result(old_result_ref, archive=archive, canonical=canonical)
    availability = CanonicalFactStore(canonical, raw_archive=archive).availability(
        MatchId(schedule.canonical_match_ids[0]),
        as_of=later,
    )
    assert not availability.result_90_present
    assert availability.result_90_ref is None
    assert availability.result_90_diagnostic == "typed_result_fact_replay_invalid"


@pytest.mark.parametrize("incomplete", (False, True))
def test_team_baseline_qualification_binds_verified_latest_team_fact_refs(
    tmp_path: Path,
    incomplete: bool,
) -> None:
    content = _report_content()
    if incomplete:
        content = content.replace(b'data-stat="xg">0.8', b'data-stat="xg">')
    _, _, _, schedule, result_ref, ingest = _seed_team_observation_replay(
        tmp_path,
        content=content,
    )
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    artifact = store.create_training_qualification(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        qualification=Qualification.TEAM_BASELINE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=None,
        result_ref=result_ref,
    )

    assert artifact.passed is not incomplete
    if incomplete:
        assert any(reason.endswith(":xg") for reason in artifact.reason_codes)
    expected_team_refs = set(ingest.team_fact_ids)
    assert expected_team_refs <= set(artifact.fact_refs)
    assert set(artifact.fact_refs) == {result_ref, *expected_team_refs}
    assert store.load_training_qualification(artifact.qualification_id) == artifact
    manifest = store.derived._load_artifact_manifest_for_output_ref(artifact.qualification_id)
    assert manifest.input_refs == artifact.input_refs
    assert expected_team_refs <= set(manifest.input_refs)


def test_team_baseline_qualification_requires_explicit_latest_result_ref(
    tmp_path: Path,
) -> None:
    _, _, _, schedule, _, ingest = _seed_team_observation_replay(tmp_path)
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))

    artifact = store.create_training_qualification(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        qualification=Qualification.TEAM_BASELINE,
        ruleset_version="readiness/1",
        evaluated_at=OBSERVED_AT,
        snapshot_ref=None,
        result_ref=None,
    )

    assert not artifact.passed
    assert "missing_bound_result_90" in artifact.reason_codes
    assert set(ingest.team_fact_ids) <= set(artifact.fact_refs)


def test_team_baseline_qualification_rejects_result_observed_after_evaluation(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule, _, _ = _seed_team_observation_replay(tmp_path)
    later = OBSERVED_AT + timedelta(hours=1)
    future_raw = archive.archive(
        b"future result correction",
        source="fbref",
        source_id="future-result-correction",
        url="https://fbref.example/future-result-correction",
        observed_at=later,
        target_event_time=OBSERVED_AT,
        collector_version="test-result/1",
        media_type="application/octet-stream",
    )
    canonical.register_raw_asset(future_raw)
    future_result = CanonicalFactStore(canonical).append_result_90(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=3,
        away_goals=1,
        known_at=OBSERVED_AT,
        observed_at=later,
        raw_asset_id=future_raw.id,
    )

    with pytest.raises(TrainingArtifactConflict, match="different match or version"):
        TrainingArtifactStore(DataLayout(tmp_path / "data")).create_training_qualification(
            match_id=MatchId(schedule.canonical_match_ids[0]),
            match_version=1,
            qualification=Qualification.TEAM_BASELINE,
            ruleset_version="readiness/1",
            evaluated_at=OBSERVED_AT + timedelta(minutes=30),
            snapshot_ref=None,
            result_ref=future_result.record_id,
        )


def test_historical_ingest_and_contract_replay_survive_later_mapping_revisions(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    match_id = MatchId(schedule.canonical_match_ids[0])
    arguments = {
        "content": _report_content(),
        "page_url": "https://fbref.example/en/matches/aaaaaaaa/report",
        "source_match_id": "aaaaaaaa",
        "match_id": match_id,
        "match_version": 1,
        "home_goals": 2,
        "away_goals": 1,
        "known_at": OBSERVED_AT,
        "observed_at": OBSERVED_AT,
        "archive": archive,
        "canonical": canonical,
    }
    first = ingest_fbref_match_report(**arguments)  # type: ignore[arg-type]
    effective_at = OBSERVED_AT + timedelta(hours=1)
    replacement_team = canonical.mapped_team(source="fbref", source_id="cff3d9bb")
    replacement_player = canonical.mapped_player(source="fbref", source_id="playerb1")
    replacements = (
        ("fbref-schedule", "match", "aaaaaaaa", MatchId(schedule.canonical_match_ids[1])),
        ("fbref", "team", "18bb7c10", replacement_team.id),
        ("fbref", "player", "playera1", replacement_player.id),
    )
    for source, entity_type, source_id, candidate_entity_id in replacements:
        current = canonical.resolve_source_mapping(
            source=source,
            entity_type=entity_type,
            source_id=source_id,
        )
        conflict = canonical.propose_source_mapping(
            source=source,
            entity_type=entity_type,
            source_id=source_id,
            candidate_entity_id=candidate_entity_id,
            proposed_at=effective_at - timedelta(minutes=1),
            actor="resolver:test",
            reason="later provider identity review",
            evidence_refs=(f"ticket:{entity_type}-revision",),
        )
        canonical.revise_source_mapping(
            source=source,
            entity_type=entity_type,
            source_id=source_id,
            conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
            expected_current_mapping_id=current.mapping_id or "",
            candidate_entity_id=candidate_entity_id,
            effective_at=effective_at,
            actor="operator:test",
            reason="accepted later provider identity review",
            evidence_refs=(f"ticket:{entity_type}-revision",),
        )

    replayed = ingest_fbref_match_report(**arguments)  # type: ignore[arg-type]

    assert replayed.contract_id == first.contract_id
    assert replayed.team_fact_ids == first.team_fact_ids
    assert replayed.player_fact_ids == first.player_fact_ids
    assert (
        verify_match_report_contract(
            first.contract_id,
            archive=archive,
            canonical=canonical,
        ).contract_id
        == first.contract_id
    )
    assert {
        load_verified_team_observation(
            reference,
            archive=archive,
            canonical=canonical,
        ).record_id
        for reference in first.team_fact_ids
    } == set(first.team_fact_ids)


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


def test_report_raw_source_identity_is_cross_checked_before_fact_write(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    content = _report_content().replace(
        b"/en/matches/aaaaaaaa/arsenal-chelsea-August-15-2025-Premier-League",
        b"/en/matches/bbbbbbbb/arsenal-chelsea-August-15-2025-Premier-League",
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

    assert caught.value.code == "report_source_match_identity_mismatch"
    attempts = [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ]
    assert len(attempts) == 1
    assert attempts[0].outcome is CollectionAttemptOutcome.FAILED


def test_report_page_url_mismatch_keeps_raw_and_records_safe_failed_attempt(
    tmp_path: Path,
) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)

    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content(),
            page_url="https://fbref.example/en/matches/bbbbbbbb/report",
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

    assert caught.value.code == "report_page_url_identity_mismatch"
    archive.verify(RawAssetId(caught.value.raw_asset_id))
    assert caught.value.attempt_raw_asset_id is not None
    archive.verify(RawAssetId(caught.value.attempt_raw_asset_id))
    attempt = next(
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    )
    assert attempt.outcome is CollectionAttemptOutcome.FAILED
    assert attempt.diagnostic_code == "report_page_url_identity_mismatch"
    assert attempt.target_url == "https://fbref.com/en/matches/aaaaaaaa/"
    assert attempt.raw_asset_id == RawAssetId(caught.value.attempt_raw_asset_id)


def test_report_raw_score_is_cross_checked_against_canonical_result(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    content = _report_content().replace(
        b'<div class="score">2</div>', b'<div class="score">9</div>'
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
            observed_at=OBSERVED_AT + timedelta(seconds=1),
            archive=archive,
            canonical=canonical,
        )

    assert caught.value.code == "report_score_identity_mismatch"
    attempts = [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ]
    assert attempts
    assert attempts[-1].outcome is CollectionAttemptOutcome.FAILED


def test_report_season_identity_is_cross_checked(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    content = _report_content().replace(b"/comps/9/2025-2026/", b"/comps/9/2024-2025/")

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

    assert caught.value.code == "report_season_identity_mismatch"
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0


def test_report_missing_season_identity_blocks_fact_write(tmp_path: Path) -> None:
    archive, canonical, season, schedule = _prepare_schedule(tmp_path)
    _seed_first_result(canonical, schedule)
    content = _report_content().replace(
        b'<a href="/en/comps/9/2025-2026/Premier-League-Stats">Premier League</a>',
        b"",
    )

    parsed = parse_match_report(content)
    assert "match_season_id_missing" in {item.code for item in parsed.diagnostics}

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

    assert caught.value.code == "match_report_parse_diagnostics"
    attempts = [
        item
        for item in canonical.collection_attempts(season.id)
        if item.source == "fbref-match-report"
    ]
    assert attempts[-1].diagnostic_code == "match_report_parse_diagnostics"
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM team_match_observations").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM player_match_observations").fetchone()[0] == 0
        )


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


def test_future_match_version_is_rejected_by_ingest_contract_write_and_replay(
    tmp_path: Path,
) -> None:
    archive, canonical, _, schedule = _prepare_schedule(tmp_path)
    match_id = MatchId(schedule.canonical_match_ids[0])
    match = canonical.match(match_id)
    version_one = canonical.match_versions(match_id)[0]
    future_at = OBSERVED_AT + timedelta(hours=1)
    future_asset = archive.archive(
        b"future schedule correction",
        source="fbref",
        source_id="future-schedule-correction",
        url="https://fbref.example/future-schedule-correction",
        observed_at=future_at,
        target_event_time=None,
        collector_version="test-schedule/2",
        media_type="text/html",
    )
    canonical.register_raw_asset(future_asset)
    _, future_version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="aaaaaaaa",
        competition_id=match.competition_id,
        season_id=match.season_id,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        kickoff_at=version_one.kickoff_at,
        status=MatchStatus.FINISHED,
        observed_at=future_at,
        raw_asset_id=future_asset.id,
        round_name="future correction",
    )
    assert future_version.version == 2

    report_url = "https://fbref.example/en/matches/aaaaaaaa/report"
    with pytest.raises(MatchReportIngestError) as caught:
        ingest_fbref_match_report(
            _report_content(),
            page_url=report_url,
            source_match_id="aaaaaaaa",
            match_id=match_id,
            match_version=future_version.version,
            home_goals=2,
            away_goals=1,
            known_at=OBSERVED_AT,
            observed_at=OBSERVED_AT,
            archive=archive,
            canonical=canonical,
        )
    assert caught.value.code == "match_version_not_visible"

    old_report_asset = archive.load(RawAssetId(caught.value.raw_asset_id))
    old_attempt = canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id="aaaaaaaa",
        target_url=report_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=OBSERVED_AT,
        collector_version=old_report_asset.collector_version,
        raw_asset_id=old_report_asset.id,
    )
    parsed = parse_match_report(_report_content(), required_tables=("summary",))
    team_tables = {team.source_team_id: team.tables_present for team in parsed.teams}
    with pytest.raises(CanonicalConflictError, match="observed after the contract"):
        canonical.record_match_report_contract(
            collection_attempt_id=old_attempt.id,
            match_id=match_id,
            match_version=future_version.version,
            raw_asset_id=old_report_asset.id,
            source_match_id="aaaaaaaa",
            parser_version=parsed.parser_version,
            required_tables=parsed.required_tables,
            team_tables=team_tables,
            observed_at=OBSERVED_AT,
        )

    replay_asset = archive.archive(
        _report_content(),
        source="fbref",
        source_id="aaaaaaaa",
        url=report_url,
        observed_at=future_at,
        target_event_time=OBSERVED_AT,
        collector_version=old_report_asset.collector_version,
        media_type="text/html",
    )
    canonical.register_raw_asset(replay_asset)
    replay_attempt = canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id="aaaaaaaa",
        target_url=report_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=future_at,
        collector_version=replay_asset.collector_version,
        raw_asset_id=replay_asset.id,
    )
    contract = canonical.record_match_report_contract(
        collection_attempt_id=replay_attempt.id,
        match_id=match_id,
        match_version=future_version.version,
        raw_asset_id=replay_asset.id,
        source_match_id="aaaaaaaa",
        parser_version=parsed.parser_version,
        required_tables=parsed.required_tables,
        team_tables=team_tables,
        observed_at=future_at,
    )
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE match_versions SET observed_at = ? WHERE match_id = ? AND version = ?",
            (
                (future_at + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                match_id.value,
                future_version.version,
            ),
        )

    with pytest.raises(MatchReportContractReplayError) as replay_error:
        verify_match_report_contract(
            contract.contract_id,
            archive=archive,
            canonical=canonical,
        )
    assert replay_error.value.category == "replay_failed"


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
