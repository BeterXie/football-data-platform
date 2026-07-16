"""Local, repeatable command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from football_data_platform.config import DEFAULT_REGISTRY_PATH, load_competition_registry
from football_data_platform.pipelines.results_backfill import ingest_results_backfill
from football_data_platform.pipelines.schedule import assess_season_coverage
from football_data_platform.pipelines.vertical_slice import run_offline_vertical_slice
from football_data_platform.reporting.backfill import render_results_backfill_report
from football_data_platform.reporting.vertical_slice import write_static_report
from football_data_platform.sources.fbref import (
    FBrefFetchError,
    fetch_schedule,
    parse_schedule,
    schedule_url,
)
from football_data_platform.sources.football_data_csv import (
    FootballDataFetchError,
    fetch_results_csv,
    results_url,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdp", description="Football Data Platform")
    subcommands = parser.add_subparsers(dest="command", required=True)

    initialize = subcommands.add_parser("init", help="initialize local data layers")
    _common_paths(initialize)

    golden = subcommands.add_parser(
        "run-golden", help="replay the bundled two-match offline vertical slice"
    )
    _common_paths(golden)
    golden.add_argument("--fixtures-dir", type=Path, default=Path("examples/golden"))
    golden.add_argument("--observed-at", type=_utc_datetime, default=None)
    golden.add_argument(
        "--first-report-known-at",
        type=_utc_datetime,
        default=_utc_datetime("2025-08-15T22:00:00Z"),
    )
    golden.add_argument(
        "--second-result-known-at",
        type=_utc_datetime,
        default=_utc_datetime("2025-08-23T16:00:00Z"),
    )
    golden.add_argument("--profile-minimum-minutes", type=float, default=60.0)

    validate = subcommands.add_parser(
        "validate-schedule", help="validate a local FBref schedule against its registry gate"
    )
    _common_paths(validate)
    validate.add_argument("--file", type=Path, required=True)
    validate.add_argument("--attempted-fixtures", type=Path)
    validate.add_argument("--require-report-attempts", action="store_true")

    diagnose = subcommands.add_parser(
        "diagnose-fbref", help="probe the registered FBref schedule and persist diagnostics"
    )
    _common_paths(diagnose)
    diagnose.add_argument("--timeout-seconds", type=float, default=30.0)

    backfill = subcommands.add_parser(
        "backfill-results",
        help="backfill registered fixtures/results from the free backup CSV source",
    )
    _common_paths(backfill)
    backfill.add_argument("--file", type=Path)
    backfill.add_argument("--observed-at", type=_utc_datetime, default=None)
    backfill.add_argument("--timeout-seconds", type=float, default=30.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "init":
            return _initialize(arguments)
        if arguments.command == "run-golden":
            return _run_golden(arguments)
        if arguments.command == "validate-schedule":
            return _validate_schedule(arguments)
        if arguments.command == "diagnose-fbref":
            return _diagnose_fbref(arguments)
        if arguments.command == "backfill-results":
            return _backfill_results(arguments)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {arguments.command}")


def _initialize(arguments: argparse.Namespace) -> int:
    layout = DataLayout(arguments.data_root).ensure()
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(
        load_competition_registry(arguments.registry), registered_at=datetime.now(UTC)
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "data_root": str(layout.root.resolve()),
                "canonical": str(canonical.path.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_golden(arguments: argparse.Namespace) -> int:
    observed_at = arguments.observed_at or datetime.now(UTC)
    result = run_offline_vertical_slice(
        data_root=arguments.data_root,
        registry_path=arguments.registry,
        schedule_file=arguments.fixtures_dir / "fbref_premier_league_schedule.html",
        first_match_report_file=(arguments.fixtures_dir / "fbref_premier_league_match_report.html"),
        lineups_file=arguments.fixtures_dir / "premier_league_official_lineups.json",
        observed_at=observed_at,
        first_report_known_at=arguments.first_report_known_at,
        second_result_known_at=arguments.second_result_known_at,
        profile_minimum_minutes=arguments.profile_minimum_minutes,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": result.run_id,
                "summary": str(result.summary_path.resolve()),
                "report": str(result.report_path.resolve()),
                "prediction_id": result.prediction_id,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_schedule(arguments: argparse.Namespace) -> int:
    registry = load_competition_registry(arguments.registry)
    competition = registry.competitions[0]
    season = competition.seasons[0]
    parsed = parse_schedule(
        arguments.file.read_bytes(),
        competition=competition,
        season=season,
        page_url=schedule_url(competition, season),
    )
    attempted: set[str] | None = None
    if arguments.attempted_fixtures:
        payload = json.loads(arguments.attempted_fixtures.read_text(encoding="utf-8"))
        attempted = set(payload["attempted_fixture_ids"])
    coverage = assess_season_coverage(
        parsed,
        season,
        attempted_fixture_ids=attempted,
        source="fbref",
    )
    result = {
        "schedule_complete": coverage.schedule_complete,
        "complete": coverage.complete,
        "expected_matches": coverage.expected_matches,
        "actual_matches": coverage.actual_matches,
        "expected_teams": coverage.expected_teams,
        "actual_teams": coverage.actual_teams,
        "duplicate_fixture_ids": coverage.duplicate_fixture_ids,
        "unregistered_team_ids": coverage.unregistered_team_ids,
        "missing_registered_teams": coverage.missing_registered_teams,
        "structural_violations": coverage.structural_violations,
        "missing_collection_attempts": coverage.missing_collection_attempts,
        "blocking_diagnostics": coverage.blocking_diagnostics,
    }
    print(json.dumps(result, sort_keys=True))
    passed = coverage.complete if arguments.require_report_attempts else coverage.schedule_complete
    return 0 if passed else 2


def _diagnose_fbref(arguments: argparse.Namespace) -> int:
    layout = DataLayout(arguments.data_root).ensure()
    registry = load_competition_registry(arguments.registry)
    competition = registry.competitions[0]
    season = competition.seasons[0]
    url = schedule_url(competition, season)
    observed_at = datetime.now(UTC)
    diagnostics_path = layout.derived / "diagnostics" / "fbref-schedule-latest.json"
    try:
        content, observed_at = fetch_schedule(
            url,
            timeout_seconds=arguments.timeout_seconds,
            observed_at=observed_at,
        )
    except FBrefFetchError as error:
        payload = {"status": "failed", **error.diagnostic.as_dict()}
        _write_json(diagnostics_path, payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 3
    parsed = parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=url,
    )
    payload = {
        "status": "ok",
        "url": url,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "rows_seen": parsed.rows_seen,
        "parsed_matches": len(parsed.matches),
        "diagnostics": [item.code for item in parsed.diagnostics],
    }
    _write_json(diagnostics_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _backfill_results(arguments: argparse.Namespace) -> int:
    layout = DataLayout(arguments.data_root).ensure()
    registry = load_competition_registry(arguments.registry)
    competition = registry.competitions[0]
    season = competition.seasons[0]
    url = results_url(season)
    observed_at = arguments.observed_at or datetime.now(UTC)
    if arguments.file is not None:
        content = arguments.file.read_bytes()
    else:
        try:
            content, observed_at = fetch_results_csv(
                url,
                timeout_seconds=arguments.timeout_seconds,
                observed_at=observed_at,
            )
        except FootballDataFetchError as error:
            payload = {"status": "failed", **error.diagnostic.as_dict()}
            diagnostics_path = layout.derived / "diagnostics" / "football-data-results-latest.json"
            _write_json(diagnostics_path, payload)
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
            return 3
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=observed_at)
    result = ingest_results_backfill(
        content,
        page_url=url,
        competition=competition,
        season=season,
        observed_at=observed_at,
        archive=RawArchive(layout),
        canonical=canonical,
    )
    payload = {
        "status": "ok" if result.coverage.schedule_complete else "gate_failed",
        "source": "football-data",
        "raw_asset_id": result.raw_asset_id,
        "actual_matches": result.coverage.actual_matches,
        "expected_matches": result.coverage.expected_matches,
        "actual_teams": result.coverage.actual_teams,
        "expected_teams": result.coverage.expected_teams,
        "schedule_complete": result.coverage.schedule_complete,
        "full_collection_gate": result.coverage.complete,
        "result_facts": len(result.result_fact_ids),
        "diagnostic_codes": result.diagnostic_codes,
        "result_known_at_policy": "reconstructed:kickoff_plus_3h",
    }
    summary_path = layout.derived / "runs" / "football-data-results-latest.json"
    _write_json(summary_path, payload)
    report_path = layout.derived / "reports" / "football-data-results-latest.md"
    write_static_report(report_path, render_results_backfill_report(payload))
    print(
        json.dumps(
            {
                **payload,
                "summary": str(summary_path.resolve()),
                "report": str(report_path.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if result.coverage.schedule_complete else 2


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 datetime: {value}") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("datetime must be timezone-aware UTC")
    return parsed


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
