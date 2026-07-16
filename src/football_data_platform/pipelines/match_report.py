"""Archive and normalize a single FBref match report."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.sources.fbref_match_report import (
    MatchReportParseResult,
    parse_match_report,
)
from football_data_platform.storage.canonical import CanonicalStore, ResolvedTeam
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.raw import RawArchive

COLLECTOR_VERSION = "fbref-match-report/1"
MATCH_MAPPING_SOURCE = "fbref-schedule"
_BLOCKING_DIAGNOSTICS = frozenset(
    {
        "player_id_missing",
        "player_minutes_missing",
        "player_summary_tables_missing",
    }
)


class MatchReportIngestError(ValueError):
    """A rejected report whose raw evidence remains archived."""

    def __init__(self, code: str, message: str, *, raw_asset_id: str) -> None:
        super().__init__(message)
        self.code = code
        self.raw_asset_id = raw_asset_id


@dataclass(frozen=True, slots=True)
class MatchReportIngestResult:
    raw_asset_id: str
    parsed: MatchReportParseResult
    team_fact_ids: tuple[str, ...]
    player_fact_ids: tuple[str, ...]
    result_fact_id: str


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
) -> MatchReportIngestResult:
    """Persist raw evidence, then emit facts only for a verified canonical match."""

    asset = archive.archive(
        content,
        source="fbref",
        source_id=source_match_id,
        url=page_url,
        observed_at=observed_at,
        target_event_time=known_at,
        collector_version=COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    parsed = parse_match_report(content)

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

    blocking_diagnostics = tuple(
        diagnostic for diagnostic in parsed.diagnostics if diagnostic.code in _BLOCKING_DIAGNOSTICS
    )
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

    source_team_ids = tuple(team.source_team_id for team in parsed.teams)
    if len(source_team_ids) != 2 or len(set(source_team_ids)) != 2:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="report_team_coverage_invalid",
            message="match report must contain exactly two distinct team summary tables",
        )

    resolved_teams = {}
    try:
        for source_team_id in source_team_ids:
            resolved_teams[source_team_id] = canonical.mapped_team(
                source="fbref", source_id=source_team_id
            )
    except KeyError as error:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="report_team_mapping_missing",
            message=str(error),
        ) from error

    match = canonical.match(mapped_match_id)
    expected_team_ids = {match.home_team_id, match.away_team_id}
    report_team_ids = {team.id for team in resolved_teams.values()}
    if report_team_ids != expected_team_ids:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="report_teams_do_not_match_fixture",
            message=(
                f"report teams {sorted(str(item) for item in report_team_ids)} do not match "
                f"fixture teams {sorted(str(item) for item in expected_team_ids)}"
            ),
        )

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
    if known_at > observed_at:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="report_temporal_boundary_invalid",
            message="known_at cannot be later than observed_at",
        )

    canonical_result = _result_as_of(
        canonical,
        match_id=mapped_match_id,
        match_version=match_version,
        known_at=known_at,
        observed_at=observed_at,
    )
    if canonical_result is None:
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="canonical_result_missing",
            message=(
                f"no canonical 90-minute result exists for {mapped_match_id}:{match_version} "
                "at the report knowledge boundary"
            ),
        )
    result_fact_id, canonical_home_goals, canonical_away_goals = canonical_result
    if (home_goals, away_goals) != (canonical_home_goals, canonical_away_goals):
        raise _record_failure(
            canonical=canonical,
            match_id=mapped_match_id,
            page_url=page_url,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            code="score_conflicts_with_canonical",
            message=(
                f"claimed score {home_goals}-{away_goals} conflicts with canonical score "
                f"{canonical_home_goals}-{canonical_away_goals}"
            ),
        )

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
                    official=True,
                    known_at=known_at,
                    observed_at=observed_at,
                    raw_asset_id=asset.id,
                )

    canonical.record_collection_attempt(
        match_id=mapped_match_id,
        source="fbref-match-report",
        source_id=source_match_id,
        target_url=page_url,
        outcome=CollectionAttemptOutcome.SUCCEEDED,
        observed_at=observed_at,
        collector_version=COLLECTOR_VERSION,
        raw_asset_id=asset.id,
    )
    return MatchReportIngestResult(
        asset.id.value,
        parsed,
        tuple(team_fact_ids),
        tuple(player_fact_ids),
        result_fact_id,
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
    canonical.record_collection_attempt(
        match_id=match_id,
        source="fbref-match-report",
        target_url=page_url,
        outcome=CollectionAttemptOutcome.FAILED,
        observed_at=observed_at,
        collector_version=COLLECTOR_VERSION,
        diagnostic_code=code,
        diagnostic_message=message,
        raw_asset_id=raw_asset_id,
    )
    return MatchReportIngestError(code, message, raw_asset_id=raw_asset_id.value)


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


def _result_as_of(
    canonical: CanonicalStore,
    *,
    match_id: MatchId,
    match_version: int,
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


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
