"""Archive and normalize a single FBref match report."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import (
    CollectionAttemptOutcome,
    MatchStatus,
    RawAsset,
)
from football_data_platform.sources.fbref_match_report import (
    REPORT_TABLE_NAMES,
    MatchReportParseResult,
    parse_match_report,
)
from football_data_platform.storage.canonical import CanonicalStore, ResolvedTeam
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.match_report_contracts import (
    MATCH_MAPPING_SOURCE,
    MatchReportCanonicalValidationError,
    blocking_match_report_diagnostics,
    validate_match_report_identity,
    validate_match_report_result,
)
from football_data_platform.storage.match_report_contracts import (
    MatchReportIdentityValidation as MatchReportIdentityValidation,
)
from football_data_platform.storage.match_report_contracts import (
    MatchReportResultValidation as MatchReportResultValidation,
)
from football_data_platform.storage.raw import RawArchive

COLLECTOR_VERSION = "fbref-match-report/3"
PRODUCTION_REQUIRED_TABLES = REPORT_TABLE_NAMES
PRODUCTION_COMPLETENESS_CONTRACT = "complete-tables/1"
PRODUCTION_COLLECTOR_VERSION = f"{COLLECTOR_VERSION}+{PRODUCTION_COMPLETENESS_CONTRACT}"
_MATCH_PATH = re.compile(r"/matches/([^/?#]+)(?:/|$)", re.IGNORECASE)


def report_page_match_id(page_url: str) -> str | None:
    """Return the FBref match identifier encoded in a report URL."""

    matched = _MATCH_PATH.search(page_url)
    return None if matched is None else matched.group(1)


def canonical_report_page_url(source_match_id: str) -> str:
    """Build the stable URL used when recording a failed report attempt."""

    return f"https://fbref.com/en/matches/{source_match_id}/"


class MatchReportIngestError(ValueError):
    """A rejected report whose raw evidence remains archived."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        raw_asset_id: str,
        attempt_raw_asset_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.raw_asset_id = raw_asset_id
        self.attempt_raw_asset_id = attempt_raw_asset_id


@dataclass(frozen=True, slots=True)
class MatchReportIngestResult:
    raw_asset_id: str
    parsed: MatchReportParseResult
    team_fact_ids: tuple[str, ...]
    player_fact_ids: tuple[str, ...]
    result_fact_id: str
    contract_id: str


def ingest_fbref_match_report(
    content: bytes,
    *,
    page_url: str,
    source_match_id: str,
    match_id: MatchId,
    match_version: int,
    home_goals: int,
    away_goals: int,
    known_at: datetime,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    required_tables: tuple[str, ...] = PRODUCTION_REQUIRED_TABLES,
) -> MatchReportIngestResult:
    """Persist raw evidence, then emit facts only for a verified canonical match."""

    required_tables = tuple(dict.fromkeys(required_tables))
    collector_version = report_collector_version(required_tables)
    asset = archive.archive(
        content,
        source="fbref",
        source_id=source_match_id,
        url=page_url,
        observed_at=observed_at,
        target_event_time=known_at,
        collector_version=collector_version,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    mapped_match_id = canonical.mapped_match_ids(
        source=MATCH_MAPPING_SOURCE,
        source_ids=(source_match_id,),
    ).get(source_match_id)
    if mapped_match_id is None:
        raise MatchReportIngestError(
            "source_match_mapping_missing",
            f"no canonical match mapping exists for {MATCH_MAPPING_SOURCE}:{source_match_id}",
            raw_asset_id=asset.id.value,
        )
    page_match_id = report_page_match_id(page_url)
    if page_match_id is None or page_match_id.casefold() != source_match_id.casefold():
        raise _record_page_url_failure(
            content=content,
            page_url=page_url,
            source_match_id=source_match_id,
            known_at=known_at,
            observed_at=observed_at,
            archive=archive,
            canonical=canonical,
            match_id=mapped_match_id,
            raw_asset_id=asset,
            collector_version=collector_version,
        )
    parsed = parse_match_report(content, required_tables=required_tables)
    if mapped_match_id != match_id:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="match_identity_mismatch",
            message=f"mapped match is {mapped_match_id}, caller supplied {match_id}",
        )

    versions = {version.version: version for version in canonical.match_versions(mapped_match_id)}
    version = versions.get(match_version)
    if version is None:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="match_version_missing",
            message=f"canonical match version {mapped_match_id}:{match_version} does not exist",
        )
    if version.status is not MatchStatus.FINISHED:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="match_version_not_finished",
            message=(
                f"canonical match version {mapped_match_id}:{match_version} has status "
                f"{version.status.value}"
            ),
        )

    blocking_diagnostics = blocking_match_report_diagnostics(parsed)
    if blocking_diagnostics:
        codes = ", ".join(sorted({diagnostic.code for diagnostic in blocking_diagnostics}))
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="match_report_parse_diagnostics",
            message=f"blocking parser diagnostics: {codes}",
        )

    try:
        identity_validation = validate_match_report_identity(
            parsed,
            source_match_id=source_match_id,
            match_id=mapped_match_id,
            match_version=match_version,
            canonical=canonical,
            mapping_as_of=observed_at,
        )
    except MatchReportCanonicalValidationError as error:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code=error.code,
            message=str(error),
        ) from error
    match = identity_validation.match
    resolved_teams = identity_validation.resolved_teams

    player_validation_error = _validate_player_rows(
        parsed,
        resolved_teams,
        canonical,
        mapped_match_id,
    )
    if player_validation_error is not None:
        code, message = player_validation_error
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code=code,
            message=message,
        )
    try:
        result_validation = validate_match_report_result(
            parsed,
            match_id=mapped_match_id,
            match_version=match_version,
            known_at=known_at,
            observed_at=observed_at,
            canonical=canonical,
            claimed_score=(home_goals, away_goals),
        )
    except MatchReportCanonicalValidationError as error:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code=error.code,
            message=str(error),
        ) from error
    result_fact_id = result_validation.result_fact_id
    canonical_home_goals = result_validation.home_goals
    canonical_away_goals = result_validation.away_goals

    facts = CanonicalFactStore(canonical)
    team_fact_ids: list[str] = []
    player_fact_ids: list[str] = []
    for team_report in parsed.teams:
        team = resolved_teams[team_report.source_team_id]
        team_goals = canonical_home_goals if team.id == match.home_team_id else canonical_away_goals
        team_fact = facts.append_team_observation(
            match_id=mapped_match_id,
            match_version=match_version,
            team_id=team.id,
            stats={**team_report.aggregated_stats, "goals": team_goals},
            known_at=known_at,
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        team_fact_ids.append(team_fact.record_id)
        for player_row in team_report.players:
            player = canonical.resolve_or_create_player(
                source="fbref",
                source_id=player_row.source_player_id,
                canonical_name=player_row.player_name,
                observed_at=observed_at,
                raw_asset_id=asset.id,
            )
            player_fact = facts.append_player_observation(
                match_id=mapped_match_id,
                match_version=match_version,
                team_id=team.id,
                player_id=player.id,
                role=player_row.role,
                minutes=player_row.minutes,
                metrics=player_row.metrics,
                known_at=known_at,
                observed_at=observed_at,
                raw_asset_id=asset.id,
            )
            player_fact_ids.append(player_fact.record_id)
            if player_row.starter is not None:
                facts.append_lineup_fact(
                    match_id=mapped_match_id,
                    match_version=match_version,
                    team_id=team.id,
                    player_id=player.id,
                    lineup_role="starter" if player_row.starter else "bench",
                    official=False,
                    known_at=known_at,
                    observed_at=observed_at,
                    raw_asset_id=asset.id,
                )

    attempt = canonical.record_collection_attempt(
        match_id=mapped_match_id,
        source="fbref-match-report",
        source_id=source_match_id,
        target_url=page_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=observed_at,
        collector_version=collector_version,
        raw_asset_id=asset.id,
    )
    contract = canonical.record_match_report_contract(
        collection_attempt_id=attempt.id,
        match_id=mapped_match_id,
        match_version=match_version,
        raw_asset_id=asset.id,
        source_match_id=source_match_id,
        parser_version=parsed.parser_version,
        required_tables=parsed.required_tables,
        team_tables={team.source_team_id: team.tables_present for team in parsed.teams},
        observed_at=observed_at,
    )
    return MatchReportIngestResult(
        asset.id.value,
        parsed,
        tuple(team_fact_ids),
        tuple(player_fact_ids),
        result_fact_id,
        contract.contract_id,
    )


def _record_failure(
    *,
    canonical: CanonicalStore,
    match_id: MatchId,
    page_url: str,
    observed_at: datetime,
    raw_asset_id: RawAssetId,
    code: str,
    message: str,
) -> MatchReportIngestError:
    source_match_id = _raw_source_id(canonical, raw_asset_id)
    collector_version = _raw_collector_version(canonical, raw_asset_id)
    canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id=source_match_id,
        target_url=page_url,
        outcome=CollectionAttemptOutcome.FAILED,
        observed_at=observed_at,
        collector_version=collector_version,
        diagnostic_code=code,
        diagnostic_message=message,
        raw_asset_id=raw_asset_id,
    )
    return MatchReportIngestError(code, message, raw_asset_id=raw_asset_id.value)


def _record_page_url_failure(
    *,
    content: bytes,
    page_url: str,
    source_match_id: str,
    known_at: datetime,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    match_id: MatchId,
    raw_asset_id: RawAsset,
    collector_version: str,
) -> MatchReportIngestError:
    """Keep the operator-supplied URL as raw evidence while recording a safe attempt URL.

    A malformed/redirected URL cannot satisfy the canonical raw-lineage URL check.  The original
    response is therefore retained as-is, and a second immutable manifest for the same bytes is
    registered under the normalized FBref match URL solely for the failed attempt record.
    """

    safe_url = canonical_report_page_url(source_match_id)
    safe_asset = archive.archive(
        content,
        source="fbref",
        source_id=source_match_id,
        url=safe_url,
        observed_at=observed_at,
        target_event_time=known_at,
        collector_version=collector_version,
        media_type="text/html",
    )
    canonical.register_raw_asset(safe_asset)
    message = (
        f"report page URL {page_url!r} does not identify source match {source_match_id!r}; "
        f"original raw asset {raw_asset_id.id.value} retained"
    )
    canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        source_id=source_match_id,
        target_url=safe_url,
        outcome=CollectionAttemptOutcome.FAILED,
        observed_at=observed_at,
        collector_version=collector_version,
        diagnostic_code="report_page_url_identity_mismatch",
        diagnostic_message=message,
        raw_asset_id=safe_asset.id,
    )
    return MatchReportIngestError(
        "report_page_url_identity_mismatch",
        message,
        raw_asset_id=raw_asset_id.id.value,
        attempt_raw_asset_id=safe_asset.id.value,
    )


def _raw_source_id(canonical: CanonicalStore, raw_asset_id: RawAssetId) -> str | None:
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT source_id FROM raw_assets WHERE raw_asset_id = ?",
            (raw_asset_id.value,),
        ).fetchone()
    return None if row is None else str(row["source_id"])


def _raw_collector_version(canonical: CanonicalStore, raw_asset_id: RawAssetId) -> str:
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT collector_version FROM raw_assets WHERE raw_asset_id = ?",
            (raw_asset_id.value,),
        ).fetchone()
    if row is None:
        raise KeyError(f"raw asset {raw_asset_id} is not registered")
    return str(row["collector_version"])


def report_collector_version(required_tables: tuple[str, ...]) -> str:
    normalized = tuple(dict.fromkeys(required_tables))
    if len(normalized) == len(PRODUCTION_REQUIRED_TABLES) and set(normalized) == set(
        PRODUCTION_REQUIRED_TABLES
    ):
        return PRODUCTION_COLLECTOR_VERSION
    known = [name for name in PRODUCTION_REQUIRED_TABLES if name in normalized]
    unknown = sorted(set(normalized) - set(PRODUCTION_REQUIRED_TABLES))
    contract = ",".join((*known, *unknown)) or "none"
    return f"{COLLECTOR_VERSION}+custom-tables:{contract}"


def _validate_player_rows(
    parsed: MatchReportParseResult,
    resolved_teams: dict[str, ResolvedTeam],
    canonical: CanonicalStore,
    match_id: MatchId,
) -> tuple[str, str] | None:
    player_teams: dict[str, str] = {}
    source_player_ids: list[str] = []
    for team_report in parsed.teams:
        if not team_report.players:
            return (
                "report_player_rows_invalid",
                f"team {team_report.source_team_id} has no usable player rows",
            )
        for metric, value in team_report.aggregated_stats.items():
            if not metric or (value is not None and not math.isfinite(value)):
                return (
                    "report_player_rows_invalid",
                    f"team {team_report.source_team_id} has invalid aggregate {metric!r}",
                )
        for player in team_report.players:
            previous_team = player_teams.get(player.source_player_id)
            if previous_team is not None:
                return (
                    "report_player_assignment_conflict",
                    f"player {player.source_player_id} appears for both {previous_team} and "
                    f"{team_report.source_team_id}",
                )
            player_teams[player.source_player_id] = team_report.source_team_id
            source_player_ids.append(player.source_player_id)
            if (
                not player.source_player_id
                or not player.player_name
                or player.player_name.strip() != player.player_name
                or not player.role
                or player.role.strip() != player.role
                or not math.isfinite(player.minutes)
                or player.minutes < 0
                or (player.starter is not None and not isinstance(player.starter, bool))
                or any(
                    not metric or (value is not None and not math.isfinite(value))
                    for metric, value in player.metrics.items()
                )
            ):
                return (
                    "report_player_rows_invalid",
                    f"player {player.source_player_id!r} has invalid normalized fields",
                )

    placeholders = ", ".join("?" for _ in source_player_ids)
    with canonical.connect() as connection:
        mapped_players = connection.execute(
            "SELECT source_id, entity_id FROM source_mappings WHERE source = 'fbref' "
            "AND entity_type = 'player' AND valid_to IS NULL "
            f"AND source_id IN ({placeholders})",
            source_player_ids,
        ).fetchall()
        for mapped_player in mapped_players:
            source_team_id = player_teams[str(mapped_player["source_id"])]
            expected_team_id = resolved_teams[source_team_id].id.value
            for table in ("player_match_observations", "lineup_facts"):
                conflicting = connection.execute(
                    f"SELECT team_id FROM {table} WHERE match_id = ? AND player_id = ? "
                    "AND team_id <> ? LIMIT 1",
                    (match_id.value, mapped_player["entity_id"], expected_team_id),
                ).fetchone()
                if conflicting is not None:
                    return (
                        "report_player_assignment_conflict",
                        f"player {mapped_player['source_id']} is already assigned to "
                        f"{conflicting['team_id']} for {match_id}",
                    )
    return None
