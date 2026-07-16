from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import CollectionAttemptOutcome
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
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

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
