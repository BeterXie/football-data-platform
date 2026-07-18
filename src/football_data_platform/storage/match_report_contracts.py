"""Shared replay validation for persisted FBref match-report contracts."""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from football_data_platform.domain.ids import MatchId
from football_data_platform.domain.models import Match, MatchStatus, MatchVersion, require_utc
from football_data_platform.sources.fbref import ParseDiagnostic
from football_data_platform.sources.fbref_match_report import (
    MatchReportParseResult,
    parse_match_report,
)
from football_data_platform.storage.canonical import (
    CanonicalStore,
    MatchReportContractEvidence,
    ResolvedTeam,
)
from football_data_platform.storage.raw import RawArchive

if TYPE_CHECKING:
    from football_data_platform.storage.verification import VerificationSession

MATCH_MAPPING_SOURCE = "fbref-schedule"
_BLOCKING_DIAGNOSTICS = frozenset(
    {
        "duplicate_player_row",
        "duplicate_report_table",
        "player_id_missing",
        "player_minutes_missing",
        "player_summary_tables_missing",
        "unsupported_report_table",
        "match_source_id_missing",
        "match_source_id_conflict",
        "match_competition_id_missing",
        "match_competition_id_conflict",
        "match_season_id_missing",
        "match_season_id_conflict",
        "match_team_identity_missing",
        "match_team_identity_conflict",
        "match_score_missing",
        "match_score_conflict",
        "match_date_missing",
        "match_date_invalid",
        "match_date_conflict",
    }
)


class MatchReportCanonicalValidationError(ValueError):
    """A raw report identity or result that conflicts with canonical state."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MatchReportContractReplayError(ValueError):
    """A persisted report contract that cannot be reproduced from its evidence."""

    def __init__(self, category: str, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.code = code

    def diagnostic(self, contract_id: str) -> str:
        suffix = f":{self.code}" if self.code is not None else ""
        return f"match_report_contract_{self.category}:{contract_id}{suffix}"


@dataclass(frozen=True, slots=True)
class MatchReportIdentityValidation:
    match: Match
    version: MatchVersion
    resolved_teams: dict[str, ResolvedTeam]


@dataclass(frozen=True, slots=True)
class MatchReportResultValidation:
    result_fact_id: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True, slots=True)
class MatchReportContractReplay:
    contract: MatchReportContractEvidence
    parsed: MatchReportParseResult
    identity: MatchReportIdentityValidation
    result: MatchReportResultValidation


def blocking_match_report_diagnostics(
    parsed: MatchReportParseResult,
) -> tuple[ParseDiagnostic, ...]:
    """Return parser diagnostics that make report facts unsafe to persist."""

    blocking_codes = _BLOCKING_DIAGNOSTICS | {"required_report_table_missing"}
    return tuple(
        diagnostic for diagnostic in parsed.diagnostics if diagnostic.code in blocking_codes
    )


def verify_match_report_contract(
    contract_id: str,
    *,
    archive: RawArchive,
    canonical: CanonicalStore,
    verification_session: VerificationSession | None = None,
) -> MatchReportContractEvidence:
    """Load one contract and replay its raw parser, fixture, and result boundaries."""

    return replay_match_report_contract(
        contract_id,
        archive=archive,
        canonical=canonical,
        verification_session=verification_session,
    ).contract


def replay_match_report_contract(
    contract_id: str,
    *,
    archive: RawArchive,
    canonical: CanonicalStore,
    verification_session: VerificationSession | None = None,
) -> MatchReportContractReplay:
    """Replay one contract and retain the parser and historical identity results."""

    connection = (
        None if verification_session is None else verification_session.canonical_connection()
    )
    if verification_session is not None:
        cached = verification_session.cached_match_report_replay(contract_id)
        if cached is not None:
            return cached
    try:
        contract = canonical.match_report_contract(contract_id, _connection=connection)
        if verification_session is not None:
            verification_session.file_proof(archive.layout.raw_manifest_path(contract.raw_asset_id))
        archive.verify(contract.raw_asset_id)
        raw_asset = archive.load(contract.raw_asset_id)
        if verification_session is not None:
            verification_session.file_proof(
                archive.layout.raw_object_path(raw_asset.checksum),
                expected_sha256=raw_asset.checksum,
            )
        content = archive.read(raw_asset)
        parsed = parse_match_report(content, required_tables=contract.required_tables)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise MatchReportContractReplayError(
            "replay_failed",
            "match report contract evidence is unavailable or invalid",
        ) from error

    replayed_team_tables = tuple(
        sorted((team.source_team_id, team.tables_present) for team in parsed.teams)
    )
    source_match_id = parsed.identity.source_match_id
    if (
        parsed.parser_version != contract.parser_version
        or raw_asset.source != "fbref"
        or raw_asset.source_id != contract.source_match_id
        or raw_asset.observed_at != contract.observed_at
        or source_match_id is None
        or source_match_id.casefold() != contract.source_match_id.casefold()
        or parsed.required_tables != contract.required_tables
        or replayed_team_tables != contract.team_tables
        or parsed.missing_required_tables
        or blocking_match_report_diagnostics(parsed)
    ):
        raise MatchReportContractReplayError(
            "replay_mismatch",
            "match report parser output does not match its persisted contract",
        )
    if raw_asset.target_event_time is None:
        raise MatchReportContractReplayError(
            "identity_mismatch",
            "match report raw evidence lacks its knowledge timestamp",
            code="report_known_at_missing",
        )
    try:
        identity = validate_match_report_identity(
            parsed,
            source_match_id=contract.source_match_id,
            match_id=contract.match_id,
            match_version=contract.match_version,
            canonical=canonical,
            mapping_as_of=contract.observed_at,
            _connection=connection,
        )
        result = validate_match_report_result(
            parsed,
            match_id=contract.match_id,
            match_version=contract.match_version,
            known_at=raw_asset.target_event_time,
            observed_at=contract.observed_at,
            canonical=canonical,
            _connection=connection,
        )
    except MatchReportCanonicalValidationError as error:
        raise MatchReportContractReplayError(
            "identity_mismatch",
            "match report parser output conflicts with canonical fixture state",
            code=error.code,
        ) from error
    replay = MatchReportContractReplay(contract, parsed, identity, result)
    if verification_session is not None:
        verification_session.remember_match_report_replay(contract_id, replay)
    return replay


def validate_match_report_identity(
    parsed: MatchReportParseResult,
    *,
    source_match_id: str,
    match_id: MatchId,
    match_version: int,
    canonical: CanonicalStore,
    mapping_as_of: datetime,
    _connection: sqlite3.Connection | None = None,
) -> MatchReportIdentityValidation:
    """Validate raw report identity against one canonical match version."""

    require_utc(mapping_as_of, "mapping_as_of")
    mapped_match_id = canonical.mapped_match_ids(
        source=MATCH_MAPPING_SOURCE,
        source_ids=(source_match_id,),
        as_of=mapping_as_of,
        _connection=_connection,
    ).get(source_match_id)
    if mapped_match_id is None:
        raise MatchReportCanonicalValidationError(
            "source_match_mapping_missing",
            f"no canonical match mapping exists for {MATCH_MAPPING_SOURCE}:{source_match_id}",
        )
    if mapped_match_id != match_id:
        raise MatchReportCanonicalValidationError(
            "match_identity_mismatch",
            f"mapped match is {mapped_match_id}, contract identifies {match_id}",
        )

    versions = {
        version.version: version
        for version in canonical.match_versions(match_id, _connection=_connection)
    }
    version = versions.get(match_version)
    if version is None:
        raise MatchReportCanonicalValidationError(
            "match_version_missing",
            f"canonical match version {match_id}:{match_version} does not exist",
        )
    if version.observed_at > mapping_as_of:
        raise MatchReportCanonicalValidationError(
            "match_version_not_visible",
            (
                f"canonical match version {match_id}:{match_version} was observed after "
                "the report boundary"
            ),
        )
    if version.status is not MatchStatus.FINISHED:
        raise MatchReportCanonicalValidationError(
            "match_version_not_finished",
            f"canonical match version {match_id}:{match_version} has status {version.status.value}",
        )

    source_team_ids = tuple(team.source_team_id for team in parsed.teams)
    if len(source_team_ids) != 2 or len(set(source_team_ids)) != 2:
        raise MatchReportCanonicalValidationError(
            "report_team_coverage_invalid",
            "match report must contain exactly two distinct team summary tables",
        )

    resolved_teams: dict[str, ResolvedTeam] = {}
    try:
        for source_team_id in source_team_ids:
            resolved_teams[source_team_id] = canonical.mapped_team(
                source="fbref",
                source_id=source_team_id,
                as_of=mapping_as_of,
                _connection=_connection,
            )
    except KeyError as error:
        raise MatchReportCanonicalValidationError(
            "report_team_mapping_missing",
            str(error),
        ) from error

    match = canonical.match(match_id, _connection=_connection)
    expected_team_ids = {match.home_team_id, match.away_team_id}
    report_team_ids = {team.id for team in resolved_teams.values()}
    if report_team_ids != expected_team_ids:
        raise MatchReportCanonicalValidationError(
            "report_teams_do_not_match_fixture",
            (
                f"report teams {sorted(str(item) for item in report_team_ids)} do not match "
                f"fixture teams {sorted(str(item) for item in expected_team_ids)}"
            ),
        )

    identity = parsed.identity
    if identity.source_match_id != source_match_id:
        raise MatchReportCanonicalValidationError(
            "report_source_match_identity_mismatch",
            f"raw report identifies {identity.source_match_id}, expected {source_match_id}",
        )

    expected_competition_id = _expected_competition_source_id(
        canonical,
        match,
        as_of=mapping_as_of,
        _connection=_connection,
    )
    if expected_competition_id is None:
        raise MatchReportCanonicalValidationError(
            "report_competition_identity_unverifiable",
            f"canonical competition source identity is missing for {match.season_id}",
        )
    if identity.competition_source_id != expected_competition_id:
        raise MatchReportCanonicalValidationError(
            "report_competition_identity_mismatch",
            (
                f"raw report identifies competition {identity.competition_source_id}, expected "
                f"{expected_competition_id}"
            ),
        )

    expected_season_id = _expected_season_source_id(
        canonical, match, as_of=mapping_as_of, _connection=_connection
    )
    if expected_season_id is None:
        raise MatchReportCanonicalValidationError(
            "report_season_identity_unverifiable",
            f"canonical season source identity is missing for {match.season_id}",
        )
    if identity.season_source_id != expected_season_id:
        raise MatchReportCanonicalValidationError(
            "report_season_identity_mismatch",
            (
                f"raw report identifies season {identity.season_source_id}, expected "
                f"{expected_season_id}"
            ),
        )

    if identity.home_source_id is None or identity.away_source_id is None:
        raise MatchReportCanonicalValidationError(
            "report_team_identity_mismatch",
            "raw report does not identify canonical home and away teams",
        )
    try:
        identity_home = canonical.mapped_team(
            source="fbref",
            source_id=identity.home_source_id,
            as_of=mapping_as_of,
            _connection=_connection,
        )
        identity_away = canonical.mapped_team(
            source="fbref",
            source_id=identity.away_source_id,
            as_of=mapping_as_of,
            _connection=_connection,
        )
    except KeyError as error:
        raise MatchReportCanonicalValidationError(
            "report_team_identity_mapping_missing",
            str(error),
        ) from error
    if identity_home.id != match.home_team_id or identity_away.id != match.away_team_id:
        raise MatchReportCanonicalValidationError(
            "report_team_identity_mismatch",
            (
                f"raw report teams {identity.home_source_id}/{identity.away_source_id} do not "
                "match canonical home/away teams"
            ),
        )

    if version.kickoff_at is None:
        raise MatchReportCanonicalValidationError(
            "report_date_identity_unverifiable",
            f"canonical match version {match_id}:{match_version} has no kickoff date",
        )
    if identity.played_on != version.kickoff_at.date():
        raw_date = None if identity.played_on is None else identity.played_on.isoformat()
        raise MatchReportCanonicalValidationError(
            "report_date_identity_mismatch",
            (
                f"raw report date {raw_date} does not match canonical kickoff date "
                f"{version.kickoff_at.date().isoformat()}"
            ),
        )

    return MatchReportIdentityValidation(match, version, resolved_teams)


def validate_match_report_result(
    parsed: MatchReportParseResult,
    *,
    match_id: MatchId,
    match_version: int,
    known_at: datetime,
    observed_at: datetime,
    canonical: CanonicalStore,
    claimed_score: tuple[int, int] | None = None,
    _connection: sqlite3.Connection | None = None,
) -> MatchReportResultValidation:
    """Validate the raw 90-minute score at the report knowledge boundary."""

    if known_at > observed_at:
        raise MatchReportCanonicalValidationError(
            "report_temporal_boundary_invalid",
            "known_at cannot be later than observed_at",
        )
    canonical_result = _result_as_of(
        canonical,
        match_id=match_id,
        match_version=match_version,
        known_at=known_at,
        observed_at=observed_at,
        _connection=_connection,
    )
    if canonical_result is None:
        raise MatchReportCanonicalValidationError(
            "canonical_result_missing",
            (
                f"no canonical 90-minute result exists for {match_id}:{match_version} "
                "at the report knowledge boundary"
            ),
        )
    result_fact_id, canonical_home_goals, canonical_away_goals = canonical_result
    if claimed_score is not None and claimed_score != (
        canonical_home_goals,
        canonical_away_goals,
    ):
        raise MatchReportCanonicalValidationError(
            "score_conflicts_with_canonical",
            (
                f"claimed score {claimed_score[0]}-{claimed_score[1]} conflicts with canonical "
                f"score {canonical_home_goals}-{canonical_away_goals}"
            ),
        )
    raw_score = (parsed.identity.home_goals, parsed.identity.away_goals)
    if raw_score != (canonical_home_goals, canonical_away_goals):
        raise MatchReportCanonicalValidationError(
            "report_score_identity_mismatch",
            (
                f"raw report score {raw_score[0]}-{raw_score[1]} conflicts with canonical score "
                f"{canonical_home_goals}-{canonical_away_goals}"
            ),
        )
    return MatchReportResultValidation(
        result_fact_id,
        canonical_home_goals,
        canonical_away_goals,
    )


def _expected_competition_source_id(
    canonical: CanonicalStore,
    match: Match,
    *,
    as_of: datetime,
    _connection: sqlite3.Connection | None = None,
) -> str | None:
    timestamp = as_of.isoformat(timespec="microseconds").replace("+00:00", "Z")
    with canonical.connect() if _connection is None else nullcontext(_connection) as connection:
        row = connection.execute(
            "SELECT source_id FROM source_mappings "
            "WHERE source = 'fbref' AND entity_type = 'season' AND entity_id = ? "
            "AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to) LIMIT 1",
            (match.season_id.value, timestamp, timestamp),
        ).fetchone()
    if row is None:
        return None
    return str(row["source_id"]).split(":", 1)[0]


def _expected_season_source_id(
    canonical: CanonicalStore,
    match: Match,
    *,
    as_of: datetime,
    _connection: sqlite3.Connection | None = None,
) -> str | None:
    timestamp = as_of.isoformat(timespec="microseconds").replace("+00:00", "Z")
    with canonical.connect() if _connection is None else nullcontext(_connection) as connection:
        row = connection.execute(
            "SELECT source_id FROM source_mappings "
            "WHERE source = 'fbref' AND entity_type = 'season' AND entity_id = ? "
            "AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to) LIMIT 1",
            (match.season_id.value, timestamp, timestamp),
        ).fetchone()
    if row is None:
        return None
    source_id = str(row["source_id"])
    return source_id.split(":", 1)[1] if ":" in source_id else None


def _result_as_of(
    canonical: CanonicalStore,
    *,
    match_id: MatchId,
    match_version: int,
    known_at: datetime,
    observed_at: datetime,
    _connection: sqlite3.Connection | None = None,
) -> tuple[str, int, int] | None:
    with canonical.connect() if _connection is None else nullcontext(_connection) as connection:
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


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
