"""Archive, parse, normalize, and validate a registered season schedule."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.domain.ids import MatchId
from football_data_platform.domain.models import require_utc
from football_data_platform.pipelines.match_report import PRODUCTION_REQUIRED_TABLES
from football_data_platform.sources.fbref import (
    COLLECTOR_VERSION,
    ScheduleParseResult,
    parse_schedule,
)
from football_data_platform.sources.fbref_match_report import REPORT_PARSER_VERSION
from football_data_platform.storage.canonical import CanonicalStore, MatchReportContractEvidence
from football_data_platform.storage.match_report_contracts import (
    MatchReportContractReplayError,
    verify_match_report_contract,
)
from football_data_platform.storage.raw import RawArchive


@dataclass(frozen=True, slots=True)
class SeasonCoverage:
    expected_matches: int
    actual_matches: int
    expected_teams: int
    actual_teams: int
    duplicate_fixture_ids: tuple[str, ...]
    unregistered_team_ids: tuple[str, ...]
    missing_registered_teams: tuple[str, ...]
    structural_violations: tuple[str, ...]
    missing_collection_attempts: tuple[str, ...]
    blocking_diagnostics: tuple[str, ...]
    fixture_coverage: tuple[FixtureCoverage, ...] = ()
    report_contract_diagnostics: tuple[str, ...] = ()

    @property
    def missing_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.fixture_coverage if item.status == "missing")

    @property
    def pending_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.fixture_coverage if item.status == "pending")

    @property
    def blocked_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.fixture_coverage if item.status == "blocked")

    @property
    def failed_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.fixture_coverage if item.status == "failed")

    @property
    def succeeded_fixture_ids(self) -> tuple[str, ...]:
        return tuple(
            item.fixture_id for item in self.fixture_coverage if item.status == "succeeded"
        )

    @property
    def attempted_fixture_ids(self) -> tuple[str, ...]:
        return tuple(item.fixture_id for item in self.fixture_coverage if item.attempt_count > 0)

    @property
    def attempt_identity_mismatch_fixture_ids(self) -> tuple[str, ...]:
        return tuple(
            item.fixture_id for item in self.fixture_coverage if item.identity_mismatch_count > 0
        )

    @property
    def fixture_status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in _FIXTURE_STATUSES}
        for item in self.fixture_coverage:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    @property
    def schedule_complete(self) -> bool:
        return (
            self.actual_matches == self.expected_matches
            and self.actual_teams == self.expected_teams
            and not self.duplicate_fixture_ids
            and not self.unregistered_team_ids
            and not self.missing_registered_teams
            and not self.structural_violations
            and not self.blocking_diagnostics
        )

    @property
    def complete(self) -> bool:
        """Return whether every catalog fixture has an identity-matched persisted attempt.

        This is the historical collection-attempt gate used by the CLI.  A blocked or failed
        identity-matched attempt satisfies the *attempt coverage* requirement but does not imply
        that the report facts were collected.  Observed identity-mismatched attempts remain in
        each fixture's diagnostics; callers that need successful report facts must use
        :attr:`report_collection_complete`.
        """

        return self.schedule_complete and not self.missing_collection_attempts

    @property
    def attempt_coverage_complete(self) -> bool:
        return self.complete

    @property
    def report_collection_complete(self) -> bool:
        """Whether every persisted fixture has a successful report attempt.

        Failed and access-blocked attempts remain visible in the per-fixture ledger and keep this
        gate open.  No synthetic or caller-supplied fixture IDs can satisfy it.
        """

        return (
            self.schedule_complete
            and bool(self.fixture_coverage)
            and all(
                item.status == "succeeded"
                and item.identity_mismatch_count == 0
                and item.production_contract_satisfied
                for item in self.fixture_coverage
            )
        )


@dataclass(frozen=True, slots=True)
class ScheduleIngestResult:
    parsed: ScheduleParseResult
    coverage: SeasonCoverage
    raw_asset_id: str
    canonical_match_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureCoverage:
    """Persisted collection state for one provider fixture identity.

    ``status`` is derived from the canonical mapping and the latest persisted collection attempt;
    it is never accepted from a caller-supplied attempt list.  ``missing`` means that the current
    provider fixture cannot be mapped to a canonical match.  ``pending`` means a canonical match
    exists but no attempt for the requested source has been recorded yet.
    """

    fixture_id: str
    match_id: str | None
    status: str
    in_schedule: bool
    attempt_count: int
    last_attempt_id: str | None = None
    last_attempt_at: datetime | None = None
    last_outcome: str | None = None
    last_diagnostic_code: str | None = None
    last_diagnostic_message: str | None = None
    last_raw_asset_id: str | None = None
    observed_attempt_count: int = 0
    last_observed_attempt_id: str | None = None
    last_observed_attempt_at: datetime | None = None
    identity_mismatch_count: int = 0
    identity_mismatch_diagnostics: tuple[str, ...] = ()
    last_collector_version: str | None = None
    production_contract_satisfied: bool = False
    report_contract_id: str | None = None
    report_contract_parser_version: str | None = None
    report_contract_required_tables: tuple[str, ...] = ()
    report_contract_team_tables: tuple[tuple[str, tuple[str, ...]], ...] = ()
    report_contract_verified: bool = False
    report_contract_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fixture_id or self.fixture_id.strip() != self.fixture_id:
            raise ValueError("fixture_id must be non-empty text without surrounding whitespace")
        if self.match_id is not None and (
            not self.match_id or self.match_id.strip() != self.match_id
        ):
            raise ValueError("match_id must be non-empty text without surrounding whitespace")
        if self.status not in _FIXTURE_STATUSES:
            raise ValueError(f"unsupported fixture coverage status: {self.status!r}")
        if not isinstance(self.in_schedule, bool):
            raise TypeError("in_schedule must be a bool")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int):
            raise TypeError("attempt_count must be a non-bool integer")
        if self.attempt_count < 0:
            raise ValueError("attempt_count must not be negative")
        if isinstance(self.observed_attempt_count, bool) or not isinstance(
            self.observed_attempt_count, int
        ):
            raise TypeError("observed_attempt_count must be a non-bool integer")
        if self.observed_attempt_count < self.attempt_count:
            raise ValueError("observed_attempt_count must include eligible attempts")
        if isinstance(self.identity_mismatch_count, bool) or not isinstance(
            self.identity_mismatch_count, int
        ):
            raise TypeError("identity_mismatch_count must be a non-bool integer")
        if (
            self.identity_mismatch_count < 0
            or self.identity_mismatch_count > self.observed_attempt_count
        ):
            raise ValueError("identity_mismatch_count must be within observed attempts")
        if self.status == "missing" and self.match_id is not None:
            raise ValueError("missing fixture must not have a canonical match ID")
        if self.status != "missing" and self.match_id is None:
            raise ValueError("mapped fixture status requires a canonical match ID")
        if self.attempt_count == 0:
            if any(
                value is not None
                for value in (
                    self.last_attempt_id,
                    self.last_attempt_at,
                    self.last_outcome,
                    self.last_diagnostic_code,
                    self.last_diagnostic_message,
                    self.last_raw_asset_id,
                    self.last_collector_version,
                )
            ):
                raise ValueError("attempt metadata requires attempt_count > 0")
            if self.status not in {"missing", "pending"}:
                raise ValueError("attempted fixture status requires attempt metadata")
        else:
            if self.last_attempt_id is None or self.last_attempt_at is None:
                raise ValueError("attempted fixture requires last attempt identity and time")
            if self.last_collector_version is None:
                raise ValueError("attempted fixture requires collector contract version")
            require_utc(self.last_attempt_at, "last_attempt_at")
            if self.last_outcome != self.status:
                raise ValueError("last_outcome must match fixture status")
            if self.status in {"blocked", "failed"} and not self.last_diagnostic_code:
                raise ValueError("blocked or failed fixture requires a diagnostic code")
            if self.status == "succeeded" and self.last_raw_asset_id is None:
                raise ValueError("succeeded fixture requires a raw asset reference")
        if self.observed_attempt_count == 0 and (
            self.last_observed_attempt_id is not None or self.last_observed_attempt_at is not None
        ):
            raise ValueError("last observed attempt metadata requires observed attempts")
        if self.observed_attempt_count > 0 and (
            self.last_observed_attempt_id is None or self.last_observed_attempt_at is None
        ):
            raise ValueError("observed attempts require last observed identity and time")
        if self.last_observed_attempt_at is not None:
            require_utc(self.last_observed_attempt_at, "last_observed_attempt_at")
        if self.identity_mismatch_count != self.observed_attempt_count - self.attempt_count:
            raise ValueError("identity mismatch count must equal ineligible observed attempts")
        if self.identity_mismatch_count != len(self.identity_mismatch_diagnostics):
            raise ValueError("identity mismatch diagnostics must match mismatch count")
        has_contract_payload = any(
            (
                self.report_contract_id is not None,
                self.report_contract_parser_version is not None,
                bool(self.report_contract_required_tables),
                bool(self.report_contract_team_tables),
            )
        )
        if self.report_contract_verified and not has_contract_payload:
            raise ValueError("verified report contract requires persisted contract evidence")
        production_tables = set(PRODUCTION_REQUIRED_TABLES)
        expected_contract = (
            self.report_contract_verified
            and self.report_contract_parser_version == REPORT_PARSER_VERSION
            and set(self.report_contract_required_tables) == production_tables
            and len(self.report_contract_team_tables) == 2
            and all(
                production_tables <= set(tables) for _, tables in self.report_contract_team_tables
            )
        )
        if self.production_contract_satisfied != expected_contract:
            raise ValueError("production contract flag conflicts with parser contract evidence")

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe diagnostic representation for CLI and static reports."""

        return {
            "fixture_id": self.fixture_id,
            "match_id": self.match_id,
            "status": self.status,
            "in_schedule": self.in_schedule,
            "attempt_count": self.attempt_count,
            "last_attempt_id": self.last_attempt_id,
            "last_attempt_at": (
                self.last_attempt_at.isoformat().replace("+00:00", "Z")
                if self.last_attempt_at is not None
                else None
            ),
            "last_outcome": self.last_outcome,
            "last_diagnostic_code": self.last_diagnostic_code,
            "last_diagnostic_message": self.last_diagnostic_message,
            "last_raw_asset_id": self.last_raw_asset_id,
            "observed_attempt_count": self.observed_attempt_count,
            "last_observed_attempt_id": self.last_observed_attempt_id,
            "last_observed_attempt_at": (
                self.last_observed_attempt_at.isoformat().replace("+00:00", "Z")
                if self.last_observed_attempt_at is not None
                else None
            ),
            "identity_mismatch_count": self.identity_mismatch_count,
            "identity_mismatch_diagnostics": self.identity_mismatch_diagnostics,
            "last_collector_version": self.last_collector_version,
            "production_contract_satisfied": self.production_contract_satisfied,
            "report_contract_id": self.report_contract_id,
            "report_contract_parser_version": self.report_contract_parser_version,
            "report_contract_required_tables": self.report_contract_required_tables,
            "report_contract_team_tables": dict(self.report_contract_team_tables),
            "report_contract_verified": self.report_contract_verified,
            "report_contract_diagnostics": self.report_contract_diagnostics,
        }


_FIXTURE_STATUSES = ("missing", "pending", "blocked", "succeeded", "failed")
_REPORT_MATCH_PATH = re.compile(r"/matches/([^/?#]+)(?:/|$)", re.IGNORECASE)


def ingest_fbref_schedule(
    content: bytes,
    *,
    page_url: str,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> ScheduleIngestResult:
    """Run the schedule slice without allowing the parser to bypass raw storage."""

    raw_asset = archive.archive(
        content,
        source="fbref",
        source_id=f"{season.source('fbref').competition_id}:{season.source('fbref').season_id}",
        url=page_url,
        observed_at=observed_at,
        target_event_time=None,
        collector_version=COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(raw_asset)
    parsed = parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=page_url,
    )
    match_ids: list[str] = []
    _require_round_robin_contract(season)
    for fixture in parsed.matches:
        home_definition = season.team("fbref", fixture.home_source_id)
        away_definition = season.team("fbref", fixture.away_source_id)
        home = canonical.resolve_registered_team(
            source="fbref",
            source_id=fixture.home_source_id,
            team_id=home_definition.id,
            canonical_name=home_definition.name,
            observed_name=fixture.home_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        away = canonical.resolve_registered_team(
            source="fbref",
            source_id=fixture.away_source_id,
            team_id=away_definition.id,
            canonical_name=away_definition.name,
            observed_name=fixture.away_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match, _ = canonical.resolve_or_create_round_robin_match(
            source="fbref-schedule",
            source_id=fixture.source_fixture_id,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home.id,
            away_team_id=away.id,
            round_name=fixture.round_name,
            kickoff_at=fixture.kickoff_at,
            status=fixture.status,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match_ids.append(match.id.value)
    coverage = assess_season_coverage(
        parsed,
        season,
        source="fbref",
        canonical=canonical,
        archive=archive,
    )
    return ScheduleIngestResult(parsed, coverage, raw_asset.id.value, tuple(match_ids))


def assess_season_coverage(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
    *,
    attempted_fixture_ids: set[str] | None = None,
    source: str | None = None,
    attempt_source: str = "fbref-match-report",
    match_mapping_source: str | None = None,
    canonical: CanonicalStore | None = None,
    archive: RawArchive | None = None,
) -> SeasonCoverage:
    fixture_ids = [match.source_fixture_id for match in parsed.matches]
    parsed_fixture_ids = set(fixture_ids)
    duplicate_diagnostics = {
        diagnostic.subject_id
        for diagnostic in parsed.diagnostics
        if diagnostic.code == "duplicate_fixture_id" and diagnostic.subject_id is not None
    }
    duplicates = tuple(
        sorted(
            {item for item in fixture_ids if fixture_ids.count(item) > 1} | duplicate_diagnostics
        )
    )
    team_ids = {
        source_id
        for match in parsed.matches
        for source_id in (match.home_source_id, match.away_source_id)
    }
    registered_team_ids, resolved_source = _registered_team_ids(season, team_ids, source)
    # Kept for CLI compatibility only. Unpersisted identifiers never establish an attempt.
    _ = attempted_fixture_ids
    match_mapping_source = match_mapping_source or (
        f"{resolved_source}-schedule" if resolved_source is not None else None
    )
    mapped_match_ids: dict[str, MatchId] = {}
    if canonical is not None and match_mapping_source is not None:
        # Start from the persisted catalog so a partial/blocked latest schedule cannot make
        # previously registered fixtures disappear from the coverage report.
        mapped_match_ids.update(
            canonical.mapped_match_source_ids(source=match_mapping_source, season_id=season.id)
        )

    attempts_by_match: dict[str, list[object]] = {}
    contracts_by_attempt: dict[str, object] = {}
    report_contract_diagnostics: tuple[str, ...] = ()
    if canonical is not None:
        for attempt in canonical.collection_attempts(season.id):
            if attempt.source == attempt_source:
                attempts_by_match.setdefault(attempt.match_id.value, []).append(attempt)
        if attempt_source == "fbref-match-report":
            contracts, report_contract_diagnostics = canonical.match_report_contract_audit(
                season.id
            )
            contracts, replay_diagnostics = _replay_match_report_contracts(
                contracts,
                archive=archive,
                canonical=canonical,
            )
            report_contract_diagnostics += replay_diagnostics
            contracts_by_attempt = {item.collection_attempt_id.value: item for item in contracts}

    parsed_match_ids = {
        mapped_match_ids[fixture_id]
        for fixture_id in parsed_fixture_ids
        if fixture_id in mapped_match_ids
    }
    persisted_fixture_ids = {
        fixture_id
        for fixture_id, match_id in mapped_match_ids.items()
        if fixture_id in parsed_fixture_ids or match_id not in parsed_match_ids
    }
    coverage_fixture_ids = set(parsed_fixture_ids) | persisted_fixture_ids
    expected_source_ids = {
        fixture.source_fixture_id: fixture.source_match_id for fixture in parsed.matches
    }
    expected_report_urls = {
        fixture.source_fixture_id: fixture.report_url for fixture in parsed.matches
    }
    all_attempts = {
        fixture_id: list(
            attempts_by_match.get(
                mapped_match_ids[fixture_id].value if fixture_id in mapped_match_ids else "",
                (),
            )
        )
        for fixture_id in coverage_fixture_ids
    }
    eligible_attempts = {
        fixture_id: [
            attempt
            for attempt in all_attempts[fixture_id]
            if _attempt_matches_fixture(
                attempt,
                expected_source_id=expected_source_ids.get(fixture_id),
                expected_report_url=expected_report_urls.get(fixture_id),
                allow_unknown_legacy=fixture_id in parsed_fixture_ids,
                bind_report_identity=attempt_source == "fbref-match-report",
            )
        ]
        for fixture_id in coverage_fixture_ids
    }
    missing_attempts = tuple(
        sorted(
            fixture_id for fixture_id in coverage_fixture_ids if not eligible_attempts[fixture_id]
        )
    )
    structural_violations = set(_round_robin_violations(parsed, season))
    if canonical is not None and resolved_source is not None:
        structural_violations.update(
            _canonical_identity_violations(
                parsed,
                season,
                canonical=canonical,
                team_source=resolved_source,
                mapped_match_ids=mapped_match_ids,
            )
        )
        structural_violations.update(
            _canonical_catalog_violations(
                parsed_fixture_ids=parsed_fixture_ids,
                mapped_match_ids=mapped_match_ids,
            )
        )
    return SeasonCoverage(
        expected_matches=season.expected_matches,
        actual_matches=len(parsed.matches),
        expected_teams=season.expected_teams,
        actual_teams=len(team_ids),
        duplicate_fixture_ids=duplicates,
        unregistered_team_ids=tuple(sorted(team_ids - registered_team_ids)),
        missing_registered_teams=tuple(sorted(registered_team_ids - team_ids)),
        structural_violations=tuple(sorted(structural_violations)),
        missing_collection_attempts=missing_attempts,
        blocking_diagnostics=tuple(sorted({item.code for item in parsed.diagnostics})),
        report_contract_diagnostics=report_contract_diagnostics,
        fixture_coverage=_fixture_coverage(
            coverage_fixture_ids=coverage_fixture_ids,
            parsed_fixture_ids=parsed_fixture_ids,
            mapped_match_ids=mapped_match_ids,
            attempts_by_fixture=eligible_attempts,
            all_attempts_by_fixture=all_attempts,
            contracts_by_attempt=contracts_by_attempt,
            report_contract_source_enabled=attempt_source == "fbref-match-report",
        ),
    )


def _fixture_coverage(
    *,
    coverage_fixture_ids: set[str],
    parsed_fixture_ids: set[str],
    mapped_match_ids: dict[str, MatchId],
    attempts_by_fixture: dict[str, list[object]],
    all_attempts_by_fixture: dict[str, list[object]],
    contracts_by_attempt: dict[str, object],
    report_contract_source_enabled: bool,
) -> tuple[FixtureCoverage, ...]:
    rows: list[FixtureCoverage] = []
    for fixture_id in sorted(coverage_fixture_ids):
        match_id = mapped_match_ids.get(fixture_id)
        attempts = sorted(
            attempts_by_fixture.get(fixture_id, ()),
            key=lambda item: (item.observed_at, item.id.value),
        )
        observed_attempts = sorted(
            all_attempts_by_fixture.get(fixture_id, ()),
            key=lambda item: (item.observed_at, item.id.value),
        )
        eligible_ids = {item.id.value for item in attempts}
        mismatches = [item for item in observed_attempts if item.id.value not in eligible_ids]
        latest = attempts[-1] if attempts else None
        latest_observed = observed_attempts[-1] if observed_attempts else None
        contract = (
            contracts_by_attempt.get(latest.id.value)
            if latest is not None and report_contract_source_enabled
            else None
        )
        contract_diagnostics: tuple[str, ...] = ()
        if latest is not None and contract is None:
            diagnostic = (
                "report_contract_missing"
                if report_contract_source_enabled
                else "report_contract_source_not_supported"
            )
            contract_diagnostics = (f"{diagnostic}:{latest.id.value}",)
        production_tables = set(PRODUCTION_REQUIRED_TABLES)
        production_contract_satisfied = bool(
            contract is not None
            and contract.parser_version == REPORT_PARSER_VERSION
            and set(contract.required_tables) == production_tables
            and len(contract.team_tables) == 2
            and all(production_tables <= set(tables) for _, tables in contract.team_tables)
        )
        if match_id is None:
            # Without a persisted mapping there is no evidence to call a fixture canonical.  It
            # remains explicitly missing even when the parser itself produced a valid row.
            status = "missing"
        elif latest is None:
            status = "pending"
        else:
            status = latest.outcome.value
        rows.append(
            FixtureCoverage(
                fixture_id=fixture_id,
                match_id=match_id.value if match_id is not None else None,
                status=status,
                in_schedule=fixture_id in parsed_fixture_ids,
                attempt_count=len(attempts),
                last_attempt_id=latest.id.value if latest is not None else None,
                last_attempt_at=latest.observed_at if latest is not None else None,
                last_outcome=latest.outcome.value if latest is not None else None,
                last_diagnostic_code=(latest.diagnostic_code if latest is not None else None),
                last_diagnostic_message=(latest.diagnostic_message if latest is not None else None),
                last_raw_asset_id=(
                    latest.raw_asset_id.value
                    if latest is not None and latest.raw_asset_id is not None
                    else None
                ),
                last_collector_version=(latest.collector_version if latest is not None else None),
                production_contract_satisfied=production_contract_satisfied,
                report_contract_id=(contract.contract_id if contract is not None else None),
                report_contract_parser_version=(
                    contract.parser_version if contract is not None else None
                ),
                report_contract_required_tables=(
                    contract.required_tables if contract is not None else ()
                ),
                report_contract_team_tables=(contract.team_tables if contract is not None else ()),
                report_contract_verified=contract is not None,
                report_contract_diagnostics=contract_diagnostics,
                observed_attempt_count=len(observed_attempts),
                last_observed_attempt_id=(
                    latest_observed.id.value if latest_observed is not None else None
                ),
                last_observed_attempt_at=(
                    latest_observed.observed_at if latest_observed is not None else None
                ),
                identity_mismatch_count=len(mismatches),
                identity_mismatch_diagnostics=tuple(
                    f"attempt_identity_mismatch:{item.id.value}" for item in mismatches
                ),
            )
        )
    return tuple(rows)


def _replay_match_report_contracts(
    contracts: tuple[MatchReportContractEvidence, ...],
    *,
    archive: RawArchive | None,
    canonical: CanonicalStore,
) -> tuple[tuple[MatchReportContractEvidence, ...], tuple[str, ...]]:
    verified: list[MatchReportContractEvidence] = []
    diagnostics: list[str] = []
    for contract in contracts:
        if archive is None:
            diagnostics.append(f"match_report_contract_replay_unavailable:{contract.contract_id}")
            continue
        try:
            verified_contract = verify_match_report_contract(
                contract.contract_id,
                archive=archive,
                canonical=canonical,
            )
        except MatchReportContractReplayError as error:
            diagnostics.append(error.diagnostic(contract.contract_id))
            continue
        if verified_contract != contract:
            diagnostics.append(f"match_report_contract_replay_mismatch:{contract.contract_id}")
            continue
        verified.append(verified_contract)
    return tuple(verified), tuple(diagnostics)


def _attempt_matches_fixture(
    attempt: object,
    *,
    expected_source_id: str | None,
    expected_report_url: str | None,
    allow_unknown_legacy: bool,
    bind_report_identity: bool,
) -> bool:
    if not bind_report_identity:
        return True
    # A report attempt is meaningful only when the report page identifies this fixture.  Legacy
    # rows without the newly persisted source_id remain eligible when their URL carries the right
    # provider ID; unknown-ID future fixtures stay pending rather than being guessed complete.
    if expected_source_id is None:
        return (
            allow_unknown_legacy
            and getattr(attempt, "source_id", None) is None
            and _report_match_id(getattr(attempt, "target_url", "")) is None
        )
    persisted_source_id = getattr(attempt, "source_id", None)
    target_match_id = _report_match_id(getattr(attempt, "target_url", ""))
    if (
        persisted_source_id is not None
        and persisted_source_id.casefold() != expected_source_id.casefold()
    ):
        return False
    if target_match_id is None:
        return (
            persisted_source_id is None
            and expected_report_url is not None
            and getattr(attempt, "target_url", "") == expected_report_url
        )
    if target_match_id.casefold() != expected_source_id.casefold():
        return False
    expected_match_id = _report_match_id(expected_report_url or "")
    return expected_match_id is None or expected_match_id.casefold() == target_match_id.casefold()


def _report_match_id(url: str) -> str | None:
    matched = _REPORT_MATCH_PATH.search(url)
    return None if matched is None else matched.group(1)


def _registered_team_ids(
    season: SeasonDefinition,
    observed_team_ids: set[str],
    source: str | None,
) -> tuple[set[str], str | None]:
    if not season.teams:
        return set(observed_team_ids), source
    registered_by_source: dict[str, set[str]] = {}
    for team in season.teams:
        for reference in team.sources:
            registered_by_source.setdefault(reference.source, set()).add(reference.source_id)
    if source is None:
        if not registered_by_source:
            return set(), None
        source = max(
            sorted(registered_by_source),
            key=lambda candidate: (
                len(observed_team_ids & registered_by_source[candidate]),
                candidate,
            ),
        )
    return registered_by_source.get(source, set()), source


def _canonical_identity_violations(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
    *,
    canonical: CanonicalStore,
    team_source: str,
    mapped_match_ids: dict[str, MatchId],
) -> set[str]:
    """Cross-check each parsed fixture against persisted canonical identity facts.

    Provider fixture IDs and persisted collection attempts are not sufficient evidence on their
    own: a stale or mis-mapped source ID can otherwise make coverage appear complete.  Keep the
    diagnostics fixture-scoped so an operator can identify the exact record requiring repair.
    """

    violations: set[str] = set()
    for fixture in parsed.matches:
        fixture_id = fixture.source_fixture_id
        match_id = mapped_match_ids.get(fixture_id)
        if match_id is None:
            violations.add(f"canonical_match_mapping_missing:{fixture_id}")
            continue
        try:
            match = canonical.match(match_id)
        except KeyError:
            violations.add(f"canonical_match_missing:{fixture_id}")
            continue

        if match.season_id != season.id:
            violations.add(f"canonical_match_season_mismatch:{fixture_id}")

        for side, source_id, canonical_team_id in (
            ("home", fixture.home_source_id, match.home_team_id),
            ("away", fixture.away_source_id, match.away_team_id),
        ):
            try:
                resolved_team = canonical.mapped_team(source=team_source, source_id=source_id)
            except KeyError:
                violations.add(f"canonical_{side}_team_mapping_missing:{fixture_id}")
            else:
                if resolved_team.id != canonical_team_id:
                    violations.add(f"canonical_match_{side}_team_mismatch:{fixture_id}")

        versions = canonical.match_versions(match_id)
        if not versions:
            violations.add(f"canonical_match_version_missing:{fixture_id}")
        elif versions[-1].kickoff_at != fixture.kickoff_at:
            violations.add(f"canonical_match_kickoff_mismatch:{fixture_id}")
    return violations


def _canonical_catalog_violations(
    *,
    parsed_fixture_ids: set[str],
    mapped_match_ids: dict[str, MatchId],
) -> set[str]:
    parsed_match_ids = {
        mapped_match_ids[fixture_id]
        for fixture_id in parsed_fixture_ids
        if fixture_id in mapped_match_ids
    }
    violations = {
        f"canonical_fixture_not_in_schedule:{fixture_id}"
        for fixture_id, match_id in mapped_match_ids.items()
        if fixture_id not in parsed_fixture_ids and match_id not in parsed_match_ids
    }
    match_to_fixtures: dict[str, list[str]] = {}
    for fixture_id in parsed_fixture_ids:
        match_id = mapped_match_ids.get(fixture_id)
        if match_id is not None:
            match_to_fixtures.setdefault(match_id.value, []).append(fixture_id)
    for match_id, fixture_ids in match_to_fixtures.items():
        if len(fixture_ids) > 1:
            violations.add(f"canonical_match_identity_reused:{match_id}")
    return violations


def _require_round_robin_contract(season: SeasonDefinition) -> None:
    teams = season.expected_teams
    if season.expected_matches not in {teams * (teams - 1), teams * (teams - 1) // 2}:
        raise ValueError(f"season {season.id} is not configured as a single/double round-robin")


def _round_robin_violations(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
) -> tuple[str, ...]:
    team_ids = {
        source_id
        for match in parsed.matches
        for source_id in (match.home_source_id, match.away_source_id)
    }
    violations: set[str] = set()
    if any(match.home_source_id == match.away_source_id for match in parsed.matches):
        violations.add("self_fixture")

    directed = Counter((match.home_source_id, match.away_source_id) for match in parsed.matches)
    double_round_robin_matches = season.expected_teams * (season.expected_teams - 1)
    single_round_robin_matches = double_round_robin_matches // 2
    if season.expected_matches == double_round_robin_matches:
        if any(count > 1 for count in directed.values()):
            violations.add("duplicate_directed_matchups")
        expected = {(home, away) for home in team_ids for away in team_ids if home != away}
        if set(directed) != expected:
            violations.add("missing_directed_matchups")
    elif season.expected_matches == single_round_robin_matches:
        unordered = Counter(frozenset(pair) for pair in directed)
        if any(count > 1 for count in unordered.values()):
            violations.add("duplicate_unordered_matchups")
        expected_unordered = {
            frozenset((first, second))
            for first in team_ids
            for second in team_ids
            if first < second
        }
        if set(unordered) != expected_unordered:
            violations.add("missing_unordered_matchups")
    else:
        violations.add("unsupported_round_robin_shape")
    return tuple(sorted(violations))
