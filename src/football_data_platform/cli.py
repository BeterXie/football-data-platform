"""Local, repeatable command-line entry points.

Every command that mutates or validates platform data emits an immutable command artifact and
run manifest.  The ``*-latest.json`` files are intentionally only pointers to those immutable
records; they are convenience indexes, not facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.config import DEFAULT_REGISTRY_PATH, load_competition_registry
from football_data_platform.domain.ids import MatchId, SeasonId
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus, RawAsset
from football_data_platform.domain.predictions import MatchResult90
from football_data_platform.evaluation.governance import assess_promotion
from football_data_platform.pipelines.match_report import (
    COLLECTOR_VERSION as MATCH_REPORT_COLLECTOR_VERSION,
)
from football_data_platform.pipelines.match_report import (
    MatchReportIngestError,
    canonical_report_page_url,
    ingest_fbref_match_report,
    report_page_match_id,
)
from football_data_platform.pipelines.results_backfill import ingest_results_backfill
from football_data_platform.pipelines.schedule import (
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.pipelines.vertical_slice import run_offline_vertical_slice
from football_data_platform.reporting.backfill import render_results_backfill_report
from football_data_platform.sources.fbref import (
    COLLECTOR_VERSION as FBREF_SCHEDULE_COLLECTOR_VERSION,
)
from football_data_platform.sources.fbref import (
    FBrefFetchError,
    fetch_schedule,
    parse_schedule,
    schedule_url,
)
from football_data_platform.sources.fbref_match_report import parse_match_report
from football_data_platform.sources.football_data_csv import (
    COLLECTOR_VERSION as FOOTBALL_DATA_COLLECTOR_VERSION,
)
from football_data_platform.sources.football_data_csv import (
    FootballDataFetchError,
    fetch_results_csv,
    results_url,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
    RunManifest,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.governance import (
    GovernanceArtifactStore,
    parse_challenger_evidence_payload,
    parse_promotion_policy_payload,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.ledger import (
    PaperBetLedger,
    parse_paper_bet_entry_payload,
)
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactStore,
    parse_model_run_payload,
    parse_training_dataset_payload,
)


@dataclass(frozen=True, slots=True)
class _CommandResult:
    """A command outcome before immutable storage envelopes are written."""

    exit_code: int
    manifest_status: str
    payload: dict[str, object]
    quality: str
    checkpoint: str
    error: str | None = None
    report_content: str | None = None
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ArchivedReportFailure:
    """Original response plus the raw asset used for a canonical failed attempt."""

    raw_asset: RawAsset
    attempt_asset: RawAsset


_CommandOperation = Callable[[argparse.Namespace, DataLayout, datetime], _CommandResult]


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
    # Kept as a compatibility input, but never used to establish collection evidence.  The
    # validator always queries persisted canonical collection_attempts instead.
    validate.add_argument("--attempted-fixtures", type=Path)
    validate.add_argument("--require-report-attempts", action="store_true")

    diagnose = subcommands.add_parser(
        "diagnose-fbref", help="probe and persist the registered FBref schedule diagnostics"
    )
    _common_paths(diagnose)
    diagnose.add_argument("--timeout-seconds", type=float, default=30.0)
    diagnose.add_argument("--observed-at", type=_utc_datetime, default=None)

    backfill = subcommands.add_parser(
        "backfill-results",
        help="backfill registered fixtures/results from the free backup CSV source",
    )
    _common_paths(backfill)
    backfill.add_argument("--file", type=Path)
    backfill.add_argument("--observed-at", type=_utc_datetime, default=None)
    backfill.add_argument("--timeout-seconds", type=float, default=30.0)

    report = subcommands.add_parser(
        "ingest-match-report",
        help="archive and normalize one local FBref match report",
    )
    _common_paths(report)
    report.add_argument("--file", type=Path, required=True)
    report.add_argument("--source-match-id", required=True)
    report.add_argument("--page-url")
    report.add_argument("--match-version", type=int)
    report.add_argument("--known-at", type=_utc_datetime, required=True)
    report.add_argument("--observed-at", type=_utc_datetime, default=None)

    training = subcommands.add_parser(
        "register-training-dataset",
        help="validate and persist one versioned training dataset manifest",
    )
    _common_paths(training)
    training.add_argument("--file", type=Path, required=True)

    model_content = subcommands.add_parser(
        "register-model-artifact",
        help="persist content-addressed model or model-output bytes",
    )
    _common_paths(model_content)
    model_content.add_argument("--file", type=Path, required=True)
    model_content.add_argument("--artifact-kind", choices=("model", "output"), default="model")

    model_run = subcommands.add_parser(
        "register-model-run",
        help="validate and persist one model-run manifest and all referenced bytes",
    )
    _common_paths(model_run)
    model_run.add_argument("--file", type=Path, required=True)

    ledger_append = subcommands.add_parser(
        "paper-ledger-append",
        help="validate and append one immutable paper-ledger entry",
    )
    _common_paths(ledger_append)
    ledger_append.add_argument("--file", type=Path, required=True)

    ledger_settle = subcommands.add_parser(
        "paper-ledger-settle",
        help="settle one paper-ledger entry from a canonical 90-minute result",
    )
    _common_paths(ledger_settle)
    ledger_settle.add_argument("--entry-id", required=True)
    ledger_settle.add_argument("--result-file", type=Path, required=True)
    ledger_settle.add_argument("--settled-at", type=_utc_datetime, required=True)

    ledger_recompute = subcommands.add_parser(
        "paper-ledger-recompute",
        help="recompute balances and exposure from immutable latest revisions",
    )
    _common_paths(ledger_recompute)
    ledger_recompute.add_argument("--initial-bankroll", type=float, default=0.0)

    policy = subcommands.add_parser(
        "register-promotion-policy",
        help="validate and persist one reviewed champion/challenger policy",
    )
    _common_paths(policy)
    policy.add_argument("--file", type=Path, required=True)

    evidence = subcommands.add_parser(
        "register-challenger-evidence",
        help="validate and persist one challenger evaluation evidence record",
    )
    _common_paths(evidence)
    evidence.add_argument("--file", type=Path, required=True)

    assess = subcommands.add_parser(
        "assess-promotion",
        help="recompute and persist a promotion decision from policy and evidence",
    )
    _common_paths(assess)
    assess.add_argument("--policy-id", required=True)
    assess.add_argument("--evidence-id", required=True)
    assess.add_argument("--decided-at", type=_utc_datetime, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run-golden":
        try:
            return _run_golden(arguments)
        except Exception as error:  # pragma: no cover - final CLI boundary
            payload: dict[str, object] = {"status": "failed", "error": str(error)}
            failure = _latest_failed_golden_manifest(arguments.data_root)
            if failure is not None:
                run, manifest_path = failure
                payload.update(
                    {
                        "run_id": run.run_id,
                        "manifest": str(manifest_path.resolve()),
                        "checkpoint": run.checkpoint,
                    }
                )
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
            return 1

    operations: dict[str, _CommandOperation] = {
        "init": _initialize,
        "validate-schedule": _validate_schedule,
        "diagnose-fbref": _diagnose_fbref,
        "backfill-results": _backfill_results,
        "ingest-match-report": _ingest_match_report,
        "register-training-dataset": _register_training_dataset,
        "register-model-artifact": _register_model_artifact,
        "register-model-run": _register_model_run,
        "paper-ledger-append": _paper_ledger_append,
        "paper-ledger-settle": _paper_ledger_settle,
        "paper-ledger-recompute": _paper_ledger_recompute,
        "register-promotion-policy": _register_promotion_policy,
        "register-challenger-evidence": _register_challenger_evidence,
        "assess-promotion": _assess_promotion,
    }
    try:
        return _run_manifested(arguments, operations[arguments.command])
    except Exception as error:  # pragma: no cover - last-resort CLI boundary
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


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


def _run_manifested(
    arguments: argparse.Namespace,
    operation: _CommandOperation,
) -> int:
    command = str(arguments.command)
    started_at = datetime.now(UTC)
    layout = DataLayout(arguments.data_root).ensure()
    input_refs = _argument_input_refs(arguments)
    resume_from: tuple[str, ...] | None = None
    try:
        resume_from = _load_resume_manifest(arguments, layout, command)
        result = operation(arguments, layout, started_at)
    except Exception as error:
        result = _exception_result(error)
    try:
        return _finalize_command(
            command=command,
            arguments=arguments,
            layout=layout,
            started_at=started_at,
            input_refs=(*input_refs, *(resume_from or ())),
            result=result,
        )
    except Exception as error:
        # Do not allow a manifest-writing failure to hide the operation's diagnostic.  A best
        # effort failed run still gives operators a machine-readable checkpoint when possible.
        failure = _write_failure_manifest_best_effort(
            command=command,
            arguments=arguments,
            layout=layout,
            started_at=started_at,
            input_refs=input_refs,
            error=error,
        )
        payload = {"status": "failed", "error": str(error)}
        if failure is not None:
            manifest, manifest_path = failure
            payload.update(
                {
                    "run_id": manifest.run_id,
                    "manifest": str(manifest_path.resolve()),
                    "checkpoint": manifest.checkpoint,
                }
            )
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


def _initialize(
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
) -> _CommandResult:
    registry = load_competition_registry(arguments.registry)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=started_at)
    return _CommandResult(
        0,
        "succeeded",
        {
            "status": "ok",
            "data_root": str(layout.root.resolve()),
            "canonical": str(canonical.path.resolve()),
        },
        "ready",
        "registry-registered",
        output_refs=(_file_content_ref(canonical.path),),
    )


def _validate_schedule(
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
) -> _CommandResult:
    registry = load_competition_registry(arguments.registry)
    competition, season = _resolve_scope(registry, arguments)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=started_at)
    try:
        parsed = parse_schedule(
            arguments.file.read_bytes(),
            competition=competition,
            season=season,
            page_url=schedule_url(competition, season),
        )
    except FBrefFetchError as error:
        return _fetch_failure_result(
            error,
            checkpoint="schedule-parse",
            layout=layout,
            canonical=canonical,
            source="fbref",
            source_id=(
                f"{season.source('fbref').competition_id}:{season.source('fbref').season_id}"
            ),
            target_url=schedule_url(competition, season),
            observed_at=started_at,
            collector_version=FBREF_SCHEDULE_COLLECTOR_VERSION,
            media_type="text/html",
            season_id=season.id,
            attempt_source="fbref-schedule-fetch",
        )
    coverage = assess_season_coverage(
        parsed,
        season,
        source="fbref",
        canonical=canonical,
    )
    passed = coverage.complete if arguments.require_report_attempts else coverage.schedule_complete
    payload = {
        "status": "ok" if passed else "gate_failed",
        **_coverage_payload(coverage),
        "attempted_fixture_file_ignored": arguments.attempted_fixtures is not None,
    }
    return _CommandResult(
        0 if passed else 2,
        "succeeded" if passed else "partial",
        payload,
        "ready" if passed else "partial",
        "coverage-gate",
        error=None if passed else "schedule_coverage_gate_failed",
        output_refs=(_file_content_ref(canonical.path),),
    )


def _diagnose_fbref(
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
) -> _CommandResult:
    registry = load_competition_registry(arguments.registry)
    competition, season = _resolve_scope(registry, arguments)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    observed_at = arguments.observed_at or started_at
    canonical.register_registry(registry, registered_at=observed_at)
    url = schedule_url(competition, season)
    try:
        content, observed_at = fetch_schedule(
            url,
            timeout_seconds=arguments.timeout_seconds,
            observed_at=observed_at,
        )
    except FBrefFetchError as error:
        return _fetch_failure_result(
            error,
            checkpoint="schedule-fetch",
            layout=layout,
            canonical=canonical,
            source="fbref",
            source_id=(
                f"{season.source('fbref').competition_id}:{season.source('fbref').season_id}"
            ),
            target_url=url,
            observed_at=observed_at,
            collector_version=FBREF_SCHEDULE_COLLECTOR_VERSION,
            media_type="text/html",
            season_id=season.id,
            attempt_source="fbref-schedule-fetch",
        )
    try:
        ingested = ingest_fbref_schedule(
            content,
            page_url=url,
            competition=competition,
            season=season,
            observed_at=observed_at,
            archive=RawArchive(layout),
            canonical=canonical,
        )
    except FBrefFetchError as error:
        return _fetch_failure_result(
            error,
            checkpoint="schedule-parse",
            layout=layout,
            canonical=canonical,
            source="fbref",
            source_id=(
                f"{season.source('fbref').competition_id}:{season.source('fbref').season_id}"
            ),
            target_url=url,
            observed_at=observed_at,
            collector_version=FBREF_SCHEDULE_COLLECTOR_VERSION,
            media_type="text/html",
            season_id=season.id,
            attempt_source="fbref-schedule-fetch",
        )
    coverage = ingested.coverage
    passed = coverage.schedule_complete
    payload = {
        "status": "ok" if passed else "gate_failed",
        "url": url,
        "observed_at": _timestamp(observed_at),
        "raw_asset_id": ingested.raw_asset_id,
        "rows_seen": ingested.parsed.rows_seen,
        "parsed_matches": len(ingested.parsed.matches),
        "diagnostics": [asdict(item) for item in ingested.parsed.diagnostics],
        "coverage": _coverage_payload(coverage),
    }
    return _CommandResult(
        0 if passed else 2,
        "succeeded" if passed else "partial",
        payload,
        "ready" if passed else "partial",
        "schedule-ingested",
        error=None if passed else "schedule_coverage_gate_failed",
        output_refs=(ingested.raw_asset_id, _file_content_ref(canonical.path)),
    )


def _backfill_results(
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
) -> _CommandResult:
    registry = load_competition_registry(arguments.registry)
    competition, season = _resolve_scope(registry, arguments)
    url = results_url(season)
    observed_at = arguments.observed_at or started_at
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=observed_at)
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
            return _fetch_failure_result(
                error,
                checkpoint="results-fetch",
                layout=layout,
                canonical=canonical,
                source="football-data",
                source_id=(
                    f"{season.source('football-data').competition_id}:"
                    f"{season.source('football-data').season_id}"
                ),
                target_url=url,
                observed_at=observed_at,
                collector_version=FOOTBALL_DATA_COLLECTOR_VERSION,
                media_type="text/csv",
                season_id=season.id,
                attempt_source="football-data-schedule-fetch",
            )
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
        "coverage": _coverage_payload(result.coverage),
        "result_known_at_policy": "reconstructed:kickoff_plus_3h",
    }
    passed = result.coverage.schedule_complete
    return _CommandResult(
        0 if passed else 2,
        "succeeded" if passed else "partial",
        payload,
        "ready" if passed else "partial",
        "results-ingested",
        error=None if passed else "schedule_coverage_gate_failed",
        report_content=render_results_backfill_report(payload),
        output_refs=(result.raw_asset_id, _file_content_ref(canonical.path)),
    )


def _ingest_match_report(
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
) -> _CommandResult:
    registry = load_competition_registry(arguments.registry)
    competition, season = _resolve_scope(registry, arguments)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    observed_at = arguments.observed_at or started_at
    source_match_id = str(arguments.source_match_id)
    page_url = arguments.page_url or f"https://fbref.com/en/matches/{source_match_id}/"
    content = arguments.file.read_bytes()
    archive = RawArchive(layout)
    canonical.register_registry(registry, registered_at=observed_at)
    mapped = canonical.mapped_match_ids(source="fbref-schedule", source_ids=(source_match_id,)).get(
        source_match_id
    )
    if mapped is None:
        archived = _archive_report_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=None,
            diagnostic_code="source_match_mapping_missing",
            diagnostic_message=(
                f"no canonical match mapping exists for fbref-schedule:{source_match_id}"
            ),
        )
        return _CommandResult(
            2,
            "failed",
            {
                "status": "failed",
                "diagnostic_code": "source_match_mapping_missing",
                "diagnostic_message": "canonical match mapping is required before report ingest",
                "raw_asset_id": archived.raw_asset.id.value,
                "attempt_recorded": False,
            },
            "failed",
            "raw-archived-mapping-missing",
            error="source_match_mapping_missing",
            output_refs=(archived.raw_asset.id.value, _file_content_ref(canonical.path)),
        )

    if (
        report_page_match_id(page_url) is None
        or report_page_match_id(page_url).casefold() != source_match_id.casefold()
    ):
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="report_page_url_identity_mismatch",
            message=(
                f"report page URL {page_url!r} does not identify source match {source_match_id!r}"
            ),
        )

    canonical_match = canonical.match(mapped)
    if canonical_match.competition_id != competition.id or canonical_match.season_id != season.id:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="match_scope_mismatch",
            message=(
                f"mapped match {mapped} belongs to {canonical_match.competition_id}/"
                f"{canonical_match.season_id}, requested {competition.id}/{season.id}"
            ),
        )

    if arguments.known_at > observed_at:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="report_temporal_boundary_invalid",
            message="known_at cannot be later than observed_at",
        )

    versions = canonical.match_versions(mapped)
    if not versions:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="match_version_missing",
            message=f"canonical match {mapped} has no version",
        )
    try:
        report_identity = parse_match_report(content).identity
    except (TypeError, ValueError):
        report_identity = None
    version, version_error = _select_report_version(
        versions,
        requested_version=arguments.match_version,
        observed_at=observed_at,
        report_date=None if report_identity is None else report_identity.played_on,
    )
    if version is None:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code=version_error[0] if version_error is not None else "match_version_missing",
            message=(
                version_error[1]
                if version_error is not None
                else f"canonical match {mapped} has no version {arguments.match_version}"
            ),
        )
    if version.status is not MatchStatus.FINISHED:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="match_version_not_finished",
            message=f"canonical match version {mapped}:{version.version} is not finished",
        )
    result_row = _latest_result(
        canonical,
        mapped,
        version.version,
        known_at=arguments.known_at,
        observed_at=observed_at,
    )
    if result_row is None:
        return _report_precondition_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped,
            code="canonical_result_missing",
            message=f"canonical result missing for {mapped}:{version.version}",
        )
    result_fact_id, home_goals, away_goals = result_row
    try:
        ingested = ingest_fbref_match_report(
            content,
            page_url=page_url,
            source_match_id=source_match_id,
            match_id=mapped,
            match_version=version.version,
            home_goals=home_goals,
            away_goals=away_goals,
            known_at=arguments.known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
        )
    except MatchReportIngestError as error:
        return _CommandResult(
            2,
            "failed",
            {
                "status": "failed",
                "diagnostic_code": error.code,
                "diagnostic_message": str(error),
                "raw_asset_id": error.raw_asset_id,
                "attempt_raw_asset_id": error.attempt_raw_asset_id,
                "match_id": mapped.value,
                "match_version": version.version,
            },
            "failed",
            "report-attempt-recorded",
            error=error.code,
            output_refs=tuple(
                sorted(
                    {
                        error.raw_asset_id,
                        *(
                            (error.attempt_raw_asset_id,)
                            if error.attempt_raw_asset_id is not None
                            else ()
                        ),
                        _file_content_ref(canonical.path),
                    }
                )
            ),
        )
    payload = {
        "status": "ok",
        "raw_asset_id": ingested.raw_asset_id,
        "match_id": mapped.value,
        "match_version": version.version,
        "result_fact_id": ingested.result_fact_id,
        "canonical_result_fact_id": result_fact_id,
        "team_fact_ids": ingested.team_fact_ids,
        "player_fact_ids": ingested.player_fact_ids,
        "parser_version": getattr(ingested.parsed, "parser_version", None),
        "tables_present": getattr(ingested.parsed, "tables_present", ()),
        "diagnostics": [asdict(item) for item in ingested.parsed.diagnostics],
    }
    return _CommandResult(
        0,
        "succeeded",
        payload,
        "ready",
        "report-ingested",
        input_refs=(result_fact_id,),
        output_refs=(
            ingested.raw_asset_id,
            ingested.result_fact_id,
            _file_content_ref(canonical.path),
        ),
    )


def _register_training_dataset(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    dataset = parse_training_dataset_payload(_read_json_object(arguments.file))
    store = TrainingArtifactStore(layout)
    path = store.write_dataset(dataset)
    status_map = {
        "succeeded": (0, "succeeded", "ready"),
        "partial": (2, "partial", "partial"),
        "failed": (1, "failed", "failed"),
    }
    exit_code, manifest_status, quality = status_map[dataset.status.value]
    status_label = "ok" if dataset.status.value == "succeeded" else dataset.status.value
    return _CommandResult(
        exit_code,
        manifest_status,
        {
            "status": status_label,
            "dataset_id": dataset.dataset_id,
            "dataset_status": dataset.status.value,
            "eligible_samples": len(dataset.included_samples),
            "excluded_samples": len(dataset.excluded_samples),
            "path": str(path.resolve()),
        },
        quality,
        "training-dataset-registered",
        error=dataset.error,
        input_refs=dataset.input_refs,
        output_refs=(dataset.dataset_id,),
    )


def _register_model_artifact(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    store = TrainingArtifactStore(layout)
    content = arguments.file.read_bytes()
    if arguments.artifact_kind == "model":
        reference = store.write_model_artifact(content)
        path = store.model_artifact_path(reference)
    else:
        reference = store.write_model_output(content)
        path = store.model_output_path(reference)
    return _CommandResult(
        0,
        "succeeded",
        {
            "status": "ok",
            "artifact_kind": arguments.artifact_kind,
            "content_ref": reference,
            "content_hash": reference.rsplit(":", 1)[1],
            "path": str(path.resolve()),
        },
        "ready",
        "model-bytes-registered",
        output_refs=(reference,),
    )


def _register_model_run(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    artifact = parse_model_run_payload(_read_json_object(arguments.file))
    store = TrainingArtifactStore(layout)
    path = store.write_model_run(artifact)
    status_map = {
        "succeeded": (0, "succeeded", "ready"),
        "partial": (2, "partial", "partial"),
        "failed": (1, "failed", "failed"),
    }
    exit_code, manifest_status, quality = status_map[artifact.status.value]
    status_label = "ok" if artifact.status.value == "succeeded" else artifact.status.value
    return _CommandResult(
        exit_code,
        manifest_status,
        {
            "status": status_label,
            "model_run_id": artifact.model_run_id,
            "model_status": artifact.status.value,
            "dataset_id": artifact.dataset_id,
            "path": str(path.resolve()),
        },
        quality,
        "model-run-registered",
        error=artifact.error,
        input_refs=(
            artifact.dataset_id,
            *artifact.model_artifact_refs,
            *artifact.evaluation_cohort,
        ),
        output_refs=(artifact.model_run_id, *artifact.output_refs),
    )


def _paper_ledger_append(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    entry = parse_paper_bet_entry_payload(_read_json_object(arguments.file))
    ledger = PaperBetLedger(layout)
    path = ledger.append(entry)
    output_refs = [entry.entry_id]
    latest_path = layout.paper_ledger / "latest.json"
    if latest_path.is_file():
        latest = _read_json_object(latest_path)
        output_refs.extend(
            str(latest[name]) for name in ("artifact_id", "run_id") if latest.get(name)
        )
    rejected = entry.decision.value == "rejected"
    return _CommandResult(
        2 if rejected else 0,
        "partial" if rejected else "succeeded",
        {
            "status": "ok",
            "entry_id": entry.entry_id,
            "candidate_id": entry.candidate_id,
            "revision": entry.revision,
            "decision": entry.decision.value,
            "path": str(path.resolve()),
        },
        "blocked" if rejected else "ready",
        "paper-ledger-entry-appended",
        error=entry.decision_reason if rejected else None,
        input_refs=(entry.prediction_id.value, entry.model_run_id.value),
        output_refs=tuple(output_refs),
    )


def _paper_ledger_settle(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    result_payload = _read_json_object(arguments.result_file)
    result = MatchResult90(
        match_id=MatchId(str(result_payload["match_id"])),
        home_goals=_strict_integer(result_payload["home_goals"], "home_goals"),
        away_goals=_strict_integer(result_payload["away_goals"], "away_goals"),
        known_at=_utc_datetime(str(result_payload["known_at"])),
        source_ref=str(result_payload["source_ref"]),
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    ledger = PaperBetLedger(layout, result_validator=CanonicalFactStore(canonical))
    settled = ledger.settle(arguments.entry_id, result, settled_at=arguments.settled_at)
    return _CommandResult(
        0,
        "succeeded",
        {
            "status": "ok",
            "entry_id": settled.entry_id,
            "candidate_id": settled.candidate_id,
            "revision": settled.revision,
            "settlement_outcome": settled.settlement_outcome.value,
            "payout": settled.payout,
            "profit": settled.profit,
        },
        "ready",
        "paper-ledger-entry-settled",
        input_refs=(settled.prior_entry_id or arguments.entry_id, result.source_ref),
        output_refs=(settled.entry_id,),
    )


def _paper_ledger_recompute(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    ledger = PaperBetLedger(layout, result_validator=CanonicalFactStore(canonical))
    entries = ledger.entries(latest_only=True)
    if not entries:
        raise ValueError("cannot recompute an empty paper ledger")
    summary = asdict(ledger.recompute(initial_bankroll=arguments.initial_bankroll))
    logical_ref = (
        "paper-ledger-recompute:" + hashlib.sha256(_canonical_json(_json_safe(summary))).hexdigest()
    )
    return _CommandResult(
        0,
        "succeeded",
        {"status": "ok", "ledger_summary": summary},
        "ready",
        "paper-ledger-recomputed",
        input_refs=tuple(entry.entry_id for entry in entries),
        output_refs=(logical_ref,),
    )


def _register_promotion_policy(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    policy = parse_promotion_policy_payload(_read_json_object(arguments.file))
    store = GovernanceArtifactStore(layout)
    path = store.write_policy(policy)
    return _CommandResult(
        0,
        "succeeded",
        {
            "status": "ok",
            "policy_id": policy.content_id,
            "policy_version": policy.policy_version,
            "path": str(path.resolve()),
        },
        "ready",
        "promotion-policy-registered",
        input_refs=(policy.rollback_artifact_ref or policy.rollback_target or "",),
        output_refs=(policy.content_id,),
    )


def _register_challenger_evidence(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    evidence = parse_challenger_evidence_payload(_read_json_object(arguments.file))
    store = GovernanceArtifactStore(layout)
    path = store.write_evidence(evidence)
    quality = "ready" if evidence.is_prospective_captured else "shadow"
    return _CommandResult(
        0,
        "succeeded",
        {
            "status": "ok",
            "evidence_id": evidence.content_id,
            "capture_mode": evidence.capture_mode.value,
            "prospective": evidence.prospective,
            "path": str(path.resolve()),
        },
        quality,
        "challenger-evidence-registered",
        input_refs=tuple(
            ref
            for ref in (
                evidence.model_run_ref,
                evidence.champion_model_ref,
                evidence.evaluation_ref,
                evidence.cohort_ref,
                *evidence.sample_refs,
            )
            if ref is not None
        ),
        output_refs=(evidence.content_id,),
    )


def _assess_promotion(
    arguments: argparse.Namespace,
    layout: DataLayout,
    _started_at: datetime,
) -> _CommandResult:
    store = GovernanceArtifactStore(layout)
    policy = store.load_policy(arguments.policy_id)
    evidence = store.load_evidence(arguments.evidence_id)
    decision = assess_promotion(
        evidence,
        policy=policy,
        reference_validator=store,
        decided_at=arguments.decided_at,
    )
    path = store.write_decision(decision)
    if decision.promoted:
        return _CommandResult(
            0,
            "succeeded",
            {
                "status": "promoted",
                "decision_id": decision.content_id,
                "reason_codes": list(decision.reason_codes),
                "path": str(path.resolve()),
            },
            "ready",
            "promotion-decision-registered",
            input_refs=(policy.content_id, evidence.content_id),
            output_refs=(decision.content_id,),
        )
    return _CommandResult(
        2,
        "partial",
        {
            "status": "blocked",
            "decision_id": decision.content_id,
            "reason_codes": list(decision.reason_codes),
            "path": str(path.resolve()),
        },
        "blocked",
        "promotion-gate-failed",
        error=";".join(decision.reason_codes) or "promotion_not_approved",
        input_refs=(policy.content_id, evidence.content_id),
        output_refs=(decision.content_id,),
    )


def _report_precondition_failure(
    *,
    content: bytes,
    page_url: str,
    source_match_id: str,
    known_at: datetime,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    match_id: MatchId,
    code: str,
    message: str,
) -> _CommandResult:
    archived = _archive_report_failure(
        content=content,
        page_url=page_url,
        source_match_id=source_match_id,
        known_at=known_at,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
        match_id=match_id,
        diagnostic_code=code,
        diagnostic_message=message,
    )
    return _CommandResult(
        2,
        "failed",
        {
            "status": "failed",
            "diagnostic_code": code,
            "diagnostic_message": message,
            "raw_asset_id": archived.raw_asset.id.value,
            "attempt_raw_asset_id": (
                None
                if archived.attempt_asset.id == archived.raw_asset.id
                else archived.attempt_asset.id.value
            ),
            "match_id": match_id.value,
            "attempt_recorded": True,
        },
        "failed",
        "report-attempt-recorded",
        error=code,
        output_refs=tuple(
            sorted(
                {
                    archived.raw_asset.id.value,
                    archived.attempt_asset.id.value,
                    _file_content_ref(canonical.path),
                }
            )
        ),
    )


def _archive_report_failure(
    *,
    content: bytes,
    page_url: str,
    source_match_id: str,
    known_at: datetime,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    match_id: MatchId | None,
    diagnostic_code: str,
    diagnostic_message: str,
):
    asset = archive.archive(
        content,
        source="fbref",
        source_id=source_match_id,
        url=page_url,
        observed_at=observed_at,
        target_event_time=known_at,
        collector_version=MATCH_REPORT_COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    attempt_asset = asset
    if match_id is not None and (
        report_page_match_id(page_url) is None
        or report_page_match_id(page_url).casefold() != source_match_id.casefold()
    ):
        safe_url = canonical_report_page_url(source_match_id)
        attempt_asset = archive.archive(
            content,
            source="fbref",
            source_id=source_match_id,
            url=safe_url,
            observed_at=observed_at,
            target_event_time=known_at,
            collector_version=MATCH_REPORT_COLLECTOR_VERSION,
            media_type="text/html",
        )
        canonical.register_raw_asset(attempt_asset)
        diagnostic_message = (
            f"{diagnostic_message}; supplied_page_url={page_url!r}; "
            f"original_raw_asset_id={asset.id.value}"
        )
    if match_id is not None:
        canonical.record_collection_attempt(
            match_id=match_id,
            source="fbref-match-report",
            source_id=source_match_id,
            target_url=attempt_asset.url,
            outcome=CollectionAttemptOutcome.FAILED,
            observed_at=observed_at,
            collector_version=MATCH_REPORT_COLLECTOR_VERSION,
            diagnostic_code=diagnostic_code,
            diagnostic_message=diagnostic_message,
            raw_asset_id=attempt_asset.id,
        )
    return _ArchivedReportFailure(asset, attempt_asset)


def _select_report_version(
    versions,
    *,
    requested_version: int | None,
    observed_at: datetime,
    report_date,
):
    visible = [item for item in versions if item.observed_at <= observed_at]
    if requested_version is not None:
        selected = next((item for item in versions if item.version == requested_version), None)
        if selected is None:
            return None, (
                "match_version_missing",
                f"requested match version {requested_version} does not exist",
            )
        if selected.observed_at > observed_at:
            return None, (
                "match_version_not_visible",
                f"match version {requested_version} was observed after command as_of",
            )
        return selected, None

    if report_date is not None:
        dated = [
            item
            for item in visible
            if item.kickoff_at is not None and item.kickoff_at.date() == report_date
        ]
        if dated:
            visible = dated
        else:
            return None, (
                "match_version_date_mismatch",
                f"no visible match version has kickoff date {report_date.isoformat()}",
            )
    finished = [item for item in visible if item.status is MatchStatus.FINISHED]
    candidates = finished or visible
    if not candidates:
        return None, (
            "match_version_not_visible",
            "no match version was observed by the command boundary",
        )
    if len(candidates) != 1:
        return None, (
            "match_version_ambiguous",
            "multiple visible match versions remain after identity/date filtering; "
            "specify --match-version",
        )
    return candidates[0], None


def _latest_result(
    canonical: CanonicalStore,
    match_id: MatchId,
    match_version: int,
    *,
    known_at: datetime,
    observed_at: datetime,
) -> tuple[str, int, int] | None:
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT record_id, home_goals, away_goals FROM match_results_90 "
            "WHERE match_id = ? AND match_version = ? AND known_at <= ? AND observed_at <= ? "
            "ORDER BY observation_version DESC LIMIT 1",
            (
                match_id.value,
                match_version,
                _timestamp(known_at),
                _timestamp(observed_at),
            ),
        ).fetchone()
    if row is None:
        return None
    return str(row["record_id"]), int(row["home_goals"]), int(row["away_goals"])


def _finalize_command(
    *,
    command: str,
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
    input_refs: Sequence[str],
    result: _CommandResult,
) -> int:
    ended_at = datetime.now(UTC)
    payload = _json_safe(result.payload)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass contract
        raise TypeError("command payload must be a JSON object")
    summary_bytes = _canonical_json(payload)
    digest = hashlib.sha256(summary_bytes).hexdigest()
    summary_path = layout.derived / "runs" / f"{command}-{digest}.json"
    _write_immutable_bytes(summary_path, summary_bytes + b"\n")
    output_refs = list(result.output_refs)
    output_refs.append(_file_content_ref(summary_path))
    report_path: Path | None = None
    if result.report_content is not None:
        report_path = layout.derived / "reports" / f"{command}-{digest}.md"
        _write_immutable_bytes(report_path, result.report_content.encode("utf-8"))
        output_refs.append(_file_content_ref(report_path))
    normalized_inputs = tuple(sorted(set((*input_refs, *result.input_refs))))
    if not normalized_inputs:
        normalized_inputs = (f"command:{command}",)
    artifact = DerivedArtifactManifest.create(
        artifact_type=f"cli-{command}",
        payload=payload,
        generated_at=ended_at,
        started_at=started_at,
        ended_at=ended_at,
        transform_version="cli/2",
        code_version=DERIVED_CODE_VERSION,
        input_refs=normalized_inputs,
        output_refs=tuple(sorted(set(output_refs))),
        status=result.manifest_status,
        error=result.error,
        quality=result.quality,
    )
    derived = DerivedArchive(layout)
    derived.write_artifact_manifest(artifact)
    run_payload = {
        "command": command,
        "resume_mode": "replay" if arguments.resume_run_id is not None else "fresh",
        "resume_run_id": arguments.resume_run_id,
        "artifact_id": artifact.artifact_id,
        "summary": str(summary_path.resolve()),
        "report": str(report_path.resolve()) if report_path is not None else None,
        "result": payload,
    }
    run = RunManifest.create(
        run_type=f"cli-{command}",
        started_at=started_at,
        ended_at=ended_at,
        generated_at=ended_at,
        transform_version="cli/2",
        code_version=DERIVED_CODE_VERSION,
        input_refs=normalized_inputs,
        output_refs=(artifact.artifact_id, *tuple(sorted(set(output_refs)))),
        status=result.manifest_status,
        error=result.error,
        quality=result.quality,
        parameters=_command_parameters(arguments),
        checkpoint=result.checkpoint,
        payload=run_payload,
    )
    manifest_path = derived.write_run_manifest(run)
    latest_path = layout.derived / "runs" / f"{command}-latest.json"
    latest_payload = {
        "run_id": run.run_id,
        "artifact_id": artifact.artifact_id,
        "status": result.payload.get("status", result.manifest_status),
        "summary": str(summary_path.resolve()),
        "report": str(report_path.resolve()) if report_path is not None else None,
        "manifest": str(manifest_path.resolve()),
        "checkpoint": result.checkpoint,
    }
    _write_json(latest_path, latest_payload)
    response = {
        **payload,
        "run_id": run.run_id,
        "artifact_id": artifact.artifact_id,
        "manifest": str(manifest_path.resolve()),
        "summary": str(summary_path.resolve()),
        "latest": str(latest_path.resolve()),
    }
    if report_path is not None:
        response["report"] = str(report_path.resolve())
    stream = sys.stderr if result.manifest_status == "failed" else sys.stdout
    print(json.dumps(response, ensure_ascii=True, sort_keys=True), file=stream)
    return result.exit_code


def _exception_result(error: Exception) -> _CommandResult:
    diagnostic = _diagnostic_from_exception(error)
    return _CommandResult(
        1,
        "failed",
        {"status": "failed", **diagnostic},
        "failed",
        "operation-failed",
        error=f"{type(error).__name__}: {error}",
    )


def _fetch_failure_result(
    error: FBrefFetchError | FootballDataFetchError,
    *,
    checkpoint: str,
    layout: DataLayout | None = None,
    canonical: CanonicalStore | None = None,
    source: str | None = None,
    source_id: str | None = None,
    target_url: str | None = None,
    observed_at: datetime | None = None,
    collector_version: str | None = None,
    media_type: str = "application/octet-stream",
    season_id: SeasonId | None = None,
    attempt_source: str | None = None,
) -> _CommandResult:
    diagnostic = error.diagnostic
    payload: dict[str, object] = {"status": "failed", **diagnostic.as_dict()}
    output_refs: list[str] = []
    payload["raw_evidence_archived"] = False
    payload["collection_attempts_recorded"] = 0

    asset = None
    if (
        diagnostic.body is not None
        and layout is not None
        and canonical is not None
        and source is not None
        and collector_version is not None
    ):
        archive = RawArchive(layout)
        asset = archive.archive(
            diagnostic.body,
            source=source,
            source_id=source_id or "request-failure",
            url=target_url or diagnostic.url,
            observed_at=observed_at or diagnostic.observed_at,
            target_event_time=None,
            collector_version=collector_version,
            media_type=media_type,
        )
        canonical.register_raw_asset(asset)
        payload["raw_asset_id"] = asset.id.value
        payload["raw_evidence_archived"] = True
        output_refs.append(asset.id.value)

    if canonical is not None and season_id is not None and attempt_source is not None:
        outcome = (
            CollectionAttemptOutcome.BLOCKED
            if diagnostic.code == "blocked_by_access_control"
            else CollectionAttemptOutcome.FAILED
        )
        attempt_raw_asset_id = asset.id if asset is not None else None
        attempt_url = target_url or diagnostic.url
        attempt_observed_at = observed_at or diagnostic.observed_at
        attempt_collector_version = collector_version or f"{attempt_source}/1"
        existing_attempt_ids = {
            attempt.id.value for attempt in canonical.collection_attempts(season_id)
        }
        for match_id in canonical.match_ids_for_season(season_id):
            attempt = canonical.record_collection_attempt(
                match_id=match_id,
                source=attempt_source,
                target_url=attempt_url,
                outcome=outcome,
                observed_at=attempt_observed_at,
                collector_version=attempt_collector_version,
                diagnostic_code=diagnostic.code,
                diagnostic_message=diagnostic.message,
                raw_asset_id=attempt_raw_asset_id,
            )
            if attempt.id.value not in existing_attempt_ids:
                existing_attempt_ids.add(attempt.id.value)
                payload["collection_attempts_recorded"] = (
                    int(payload["collection_attempts_recorded"]) + 1
                )

    if canonical is not None:
        output_refs.append(_file_content_ref(canonical.path))
    return _CommandResult(
        3,
        "failed",
        payload,
        "failed",
        checkpoint,
        error=diagnostic.code,
        output_refs=tuple(output_refs),
    )


def _diagnostic_from_exception(error: Exception) -> dict[str, object]:
    diagnostic = getattr(error, "diagnostic", None)
    if diagnostic is not None and hasattr(diagnostic, "as_dict"):
        return dict(diagnostic.as_dict())
    return {"diagnostic_code": type(error).__name__, "diagnostic_message": str(error)}


def _write_failure_manifest_best_effort(
    *,
    command: str,
    arguments: argparse.Namespace,
    layout: DataLayout,
    started_at: datetime,
    input_refs: Sequence[str],
    error: Exception,
) -> tuple[RunManifest, Path] | None:
    try:
        ended_at = datetime.now(UTC)
        manifest = RunManifest.create(
            run_type=f"cli-{command}",
            started_at=started_at,
            ended_at=ended_at,
            generated_at=ended_at,
            transform_version="cli/2",
            code_version=DERIVED_CODE_VERSION,
            input_refs=tuple(input_refs) or (f"command:{command}",),
            output_refs=(),
            status="failed",
            error=f"{type(error).__name__}: {error}",
            quality="failed",
            parameters=_command_parameters(arguments),
            checkpoint="manifest-write-failed",
            payload=_diagnostic_from_exception(error),
        )
        manifest_path = DerivedArchive(layout).write_run_manifest(manifest)
        _write_json(
            layout.derived / "runs" / f"{command}-latest.json",
            {
                "run_id": manifest.run_id,
                "status": "failed",
                "manifest": str(manifest_path.resolve()),
                "checkpoint": "manifest-write-failed",
            },
        )
        return manifest, manifest_path
    except Exception:
        return None


def _resolve_scope(registry, arguments: argparse.Namespace):
    competitions = registry.competitions
    if not competitions:
        raise ValueError("competition registry is empty")
    competition = next(
        (
            item
            for item in competitions
            if arguments.competition_id is None or item.id.value == arguments.competition_id
        ),
        None,
    )
    if competition is None:
        raise KeyError(f"competition {arguments.competition_id!r} is not registered")
    season = next(
        (
            item
            for item in competition.seasons
            if arguments.season_id is None or item.id.value == arguments.season_id
        ),
        None,
    )
    if season is None:
        raise KeyError(
            f"season {arguments.season_id!r} is not registered for competition {competition.id}"
        )
    return competition, season


def _load_resume_manifest(
    arguments: argparse.Namespace,
    layout: DataLayout,
    command: str,
) -> tuple[str, ...] | None:
    run_id = getattr(arguments, "resume_run_id", None)
    if run_id is None:
        return None
    manifest = DerivedArchive(layout).load_run_manifest(run_id)
    if manifest.run_type != f"cli-{command}":
        raise ValueError(
            f"resume manifest {run_id} belongs to {manifest.run_type}, expected cli-{command}"
        )
    return (manifest.run_id,)


def _argument_input_refs(arguments: argparse.Namespace) -> tuple[str, ...]:
    refs = [_file_content_ref(Path(arguments.registry))]
    for name in ("file", "attempted_fixtures", "result_file"):
        value = getattr(arguments, name, None)
        if value is not None:
            refs.append(_file_content_ref(Path(value)))
    if getattr(arguments, "fixtures_dir", None) is not None:
        refs.extend(
            _file_content_ref(Path(arguments.fixtures_dir) / name)
            for name in (
                "fbref_premier_league_schedule.html",
                "fbref_premier_league_match_report.html",
                "premier_league_official_lineups.json",
            )
        )
    if getattr(arguments, "command", None) == "diagnose-fbref":
        refs.append("source-url:fbref-schedule")
    if getattr(arguments, "command", None) == "backfill-results":
        refs.append("source-url:football-data-results")
    return tuple(sorted(set(refs)))


def _command_parameters(arguments: argparse.Namespace) -> dict[str, object]:
    parameters = _json_safe(dict(vars(arguments)))
    if getattr(arguments, "resume_run_id", None) is not None:
        parameters["resume_mode"] = "replay"
    return parameters


def _latest_failed_golden_manifest(data_root: Path):
    """Find the most recent failed golden manifest without masking the original error."""

    layout = DataLayout(data_root)
    root = layout.derived / "manifests" / "runs"
    if not root.is_dir():
        return None
    candidates = []
    archive = DerivedArchive(layout)
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_id = payload.get("id")
            if not isinstance(run_id, str):
                continue
            run = archive.load_run_manifest(run_id)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if run.run_type == "offline-golden-replay" and run.status == "failed":
            candidates.append((run.generated_at, run, path))
    if not candidates:
        return None
    _, run, path = max(candidates, key=lambda item: (item[0], str(item[2])))
    return run, path


def _coverage_payload(coverage) -> dict[str, object]:
    return {
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


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--competition-id")
    parser.add_argument("--season-id")
    parser.add_argument("--resume-run-id")


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 datetime: {value}") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("datetime must be timezone-aware UTC")
    return parsed


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _file_content_ref(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return f"missing-file:{path.resolve()}"
    return f"file-sha256:{digest}"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _strict_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ArchiveConflictError(f"immutable command output conflicts at {path}")
        return
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ArchiveConflictError(f"immutable command output conflicts at {path}") from None


if __name__ == "__main__":
    raise SystemExit(main())
