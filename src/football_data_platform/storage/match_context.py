"""Replayable match-context inputs derived from canonical schedules and results."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any

from football_data_platform.config import (
    CompetitionDefinition,
    SeasonDefinition,
    SourceSeasonReference,
)
from football_data_platform.domain.ids import CompetitionId, MatchId, RawAssetId, SeasonId, TeamId
from football_data_platform.domain.models import MatchStatus, MatchVersion, RawAsset, require_utc
from football_data_platform.sources.fbref import COLLECTOR_VERSION, parse_schedule
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import load_verified_match_result
from football_data_platform.storage.raw import RawArchive

MATCH_CONTEXT_INPUT_TRANSFORM_V2 = "match-context-input/2"
MATCH_CONTEXT_RULE_VERSION = "previous-completed-match-rest/1"
MATCH_CONTEXT_SCHEMA_VERSION = 2


class MatchContextReplayError(ValueError):
    """The formal context cannot be reproduced from its persisted evidence."""


class _ScheduleNotKnown(MatchContextReplayError):
    """An otherwise valid schedule version was not knowable at the requested cutoff."""


@dataclass(frozen=True, slots=True)
class MatchContextReplay:
    value: dict[str, Any]
    input_refs: tuple[str, ...]
    known_at: datetime
    observed_at: datetime
    source_context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ScheduleEvidence:
    match_id: MatchId
    match_version: int
    home_team_id: TeamId
    away_team_id: TeamId
    kickoff_at: datetime
    status: MatchStatus
    raw_asset_id: RawAssetId
    raw_observed_at: datetime
    schedule_known_at: datetime
    availability_basis: str
    season_schedule_coverage_complete: bool


@dataclass(frozen=True, slots=True)
class _ResultEvidence:
    source_ref: str
    known_at: datetime
    observed_at: datetime


def replay_match_context(
    *,
    match_id: MatchId,
    match_version: int,
    as_of: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> MatchContextReplay:
    """Recompute per-team rest from exact schedule versions and typed results."""

    if not isinstance(match_id, MatchId):
        raise TypeError("match_id must be a MatchId")
    if not isinstance(match_version, int) or isinstance(match_version, bool) or match_version < 1:
        raise ValueError("match_version must be a positive integer")
    require_utc(as_of, "as_of")

    with canonical.connect() as connection:
        connection.execute("PRAGMA query_only = ON")
        try:
            target = _replay_schedule_version(
                match_id=match_id,
                match_version=match_version,
                as_of=as_of,
                archive=archive,
                connection=connection,
            )
        except _ScheduleNotKnown as error:
            raise MatchContextReplayError("target match schedule was not known at as_of") from error
        if as_of >= target.kickoff_at:
            raise MatchContextReplayError("match context as_of must precede target kickoff")

        teams: dict[str, dict[str, Any]] = {}
        input_refs = {target.raw_asset_id.value}
        known_times = [target.schedule_known_at]
        observed_times = [target.raw_observed_at]
        if not target.season_schedule_coverage_complete:
            for team_id in (target.home_team_id, target.away_team_id):
                teams[team_id.value] = _missing_team(
                    team_id, "season_schedule_coverage_incomplete"
                )[0]
        else:
            for team_id in (target.home_team_id, target.away_team_id):
                item, item_refs, item_known, item_observed = _previous_match_context(
                    team_id=team_id,
                    target=target,
                    as_of=as_of,
                    archive=archive,
                    canonical=canonical,
                    connection=connection,
                )
                teams[team_id.value] = item
                input_refs.update(item_refs)
                known_times.extend(item_known)
                observed_times.extend(item_observed)

    quality_status = (
        "ready" if all(item["status"] == "available" for item in teams.values()) else "missing"
    )
    value: dict[str, Any] = {
        "schema_version": MATCH_CONTEXT_SCHEMA_VERSION,
        "rule_version": MATCH_CONTEXT_RULE_VERSION,
        "match_id": target.match_id.value,
        "match_version": target.match_version,
        "as_of": _timestamp(as_of),
        "scheduled_kickoff": _timestamp(target.kickoff_at),
        "home_team_id": target.home_team_id.value,
        "away_team_id": target.away_team_id.value,
        "quality_status": quality_status,
        "target_schedule": _schedule_payload(target),
        "teams": teams,
    }
    source_context = {
        "rule_version": MATCH_CONTEXT_RULE_VERSION,
        "match_id": target.match_id.value,
        "match_version": target.match_version,
        "as_of": _timestamp(as_of),
        "scheduled_kickoff": _timestamp(target.kickoff_at),
        "home_team_id": target.home_team_id.value,
        "away_team_id": target.away_team_id.value,
        "quality_status": quality_status,
    }
    return MatchContextReplay(
        value=value,
        input_refs=tuple(sorted(input_refs)),
        known_at=max(known_times),
        observed_at=max(observed_times),
        source_context=source_context,
    )


def _previous_match_context(
    *,
    team_id: TeamId,
    target: _ScheduleEvidence,
    as_of: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    connection: sqlite3.Connection,
) -> tuple[dict[str, Any], set[str], list[datetime], list[datetime]]:
    candidate_ids = _candidate_match_ids(
        connection,
        team_id=team_id,
        target_match_id=target.match_id,
    )
    candidates: list[_ScheduleEvidence] = []
    invalid_prior_kickoffs: list[datetime] = []
    for candidate_id in candidate_ids:
        versions = _match_versions(connection, candidate_id)
        eligible: list[_ScheduleEvidence] = []
        for version in versions:
            if version.kickoff_at is None or version.kickoff_at >= target.kickoff_at:
                continue
            try:
                evidence = _replay_schedule_version(
                    match_id=candidate_id,
                    match_version=version.version,
                    as_of=as_of,
                    archive=archive,
                    connection=connection,
                )
            except _ScheduleNotKnown:
                continue
            except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error):
                invalid_prior_kickoffs.append(version.kickoff_at)
                continue
            if not evidence.season_schedule_coverage_complete:
                invalid_prior_kickoffs.append(version.kickoff_at)
                continue
            eligible.append(evidence)
        if eligible:
            candidates.append(max(eligible, key=lambda item: item.match_version))

    candidates = [
        item
        for item in candidates
        if item.kickoff_at < target.kickoff_at
        and item.status not in {MatchStatus.CANCELLED, MatchStatus.POSTPONED}
    ]
    candidates.sort(key=lambda item: (item.kickoff_at, item.match_id.value), reverse=True)
    latest = candidates[0] if candidates else None
    if invalid_prior_kickoffs and (
        latest is None or max(invalid_prior_kickoffs) >= latest.kickoff_at
    ):
        return _missing_team(team_id, "prior_schedule_evidence_unavailable")
    if latest is None:
        return _missing_team(team_id, "previous_match_coverage_unproven")

    result = _result_as_of(
        latest,
        as_of=as_of,
        archive=archive,
        canonical=canonical,
        connection=connection,
    )
    if result is None:
        reason = (
            "latest_prior_fixture_completion_unverified"
            if latest.status is MatchStatus.SCHEDULED
            else "latest_prior_fixture_result_unavailable"
        )
        return _missing_team(team_id, reason)

    rest_days = (target.kickoff_at - latest.kickoff_at).total_seconds() / 86_400
    if not math.isfinite(rest_days) or rest_days <= 0:
        raise MatchContextReplayError("recomputed rest_days must be finite and positive")
    item = {
        "team_id": team_id.value,
        "status": "available",
        "previous_match_id": latest.match_id.value,
        "previous_match_version": latest.match_version,
        "previous_kickoff": _timestamp(latest.kickoff_at),
        "previous_result_ref": result.source_ref,
        "previous_result_known_at": _timestamp(result.known_at),
        "previous_schedule_raw_ref": latest.raw_asset_id.value,
        "previous_schedule_known_at": _timestamp(latest.schedule_known_at),
        "previous_schedule_availability_basis": latest.availability_basis,
        "rest_days": rest_days,
    }
    return (
        item,
        {latest.raw_asset_id.value, result.source_ref},
        [latest.schedule_known_at, result.known_at],
        [latest.raw_observed_at, result.observed_at],
    )


def _missing_team(
    team_id: TeamId,
    reason: str,
) -> tuple[dict[str, Any], set[str], list[datetime], list[datetime]]:
    return (
        {"team_id": team_id.value, "status": "missing", "reason": reason},
        set(),
        [],
        [],
    )


def _result_as_of(
    schedule: _ScheduleEvidence,
    *,
    as_of: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    connection: sqlite3.Connection,
) -> _ResultEvidence | None:
    rows = connection.execute(
        "SELECT record_id, known_at, observed_at FROM match_results_90 "
        "WHERE match_id = ? AND match_version = ? ORDER BY observation_version DESC",
        (schedule.match_id.value, schedule.match_version),
    ).fetchall()
    for row in rows:
        known_at = _parse_timestamp(row["known_at"], "result known_at")
        if known_at > as_of:
            continue
        result = load_verified_match_result(
            str(row["record_id"]),
            archive=archive,
            canonical=canonical,
            _connection=connection,
        )
        if result.match_id != schedule.match_id or result.known_at != known_at:
            raise MatchContextReplayError("typed result does not match its previous match")
        return _ResultEvidence(
            source_ref=result.source_ref,
            known_at=result.known_at,
            observed_at=_parse_timestamp(row["observed_at"], "result observed_at"),
        )
    return None


def _candidate_match_ids(
    connection: sqlite3.Connection,
    *,
    team_id: TeamId,
    target_match_id: MatchId,
) -> tuple[MatchId, ...]:
    rows = connection.execute(
        "SELECT match_id FROM matches WHERE match_id <> ? "
        "AND (home_team_id = ? OR away_team_id = ?) ORDER BY match_id",
        (target_match_id.value, team_id.value, team_id.value),
    ).fetchall()
    return tuple(MatchId(str(row["match_id"])) for row in rows)


def _match_versions(
    connection: sqlite3.Connection,
    match_id: MatchId,
) -> tuple[MatchVersion, ...]:
    rows = connection.execute(
        "SELECT match_id, version, round_name, kickoff_at, status, observed_at "
        "FROM match_versions WHERE match_id = ? ORDER BY version",
        (match_id.value,),
    ).fetchall()
    return tuple(
        MatchVersion(
            match_id=MatchId(str(row["match_id"])),
            version=int(row["version"]),
            round_name=row["round_name"],
            kickoff_at=(
                _parse_timestamp(row["kickoff_at"], "match kickoff")
                if row["kickoff_at"] is not None
                else None
            ),
            status=MatchStatus(str(row["status"])),
            observed_at=_parse_timestamp(row["observed_at"], "match observed_at"),
        )
        for row in rows
    )


def _replay_schedule_version(
    *,
    match_id: MatchId,
    match_version: int,
    as_of: datetime,
    archive: RawArchive,
    connection: sqlite3.Connection,
) -> _ScheduleEvidence:
    row = connection.execute(
        "SELECT m.competition_id, m.season_id, m.home_team_id, m.away_team_id, "
        "mv.round_name, mv.kickoff_at, mv.status, mv.observed_at, mv.raw_asset_id "
        "FROM matches AS m JOIN match_versions AS mv ON mv.match_id = m.match_id "
        "WHERE m.match_id = ? AND mv.version = ?",
        (match_id.value, match_version),
    ).fetchone()
    if row is None:
        raise MatchContextReplayError("canonical match version does not exist")
    source_rows = connection.execute(
        "SELECT source_id FROM source_mappings WHERE source = 'fbref-schedule' "
        "AND entity_type = 'match' AND entity_id = ? AND valid_to IS NULL",
        (match_id.value,),
    ).fetchall()
    team_sources = {
        str(team_row["entity_id"]): str(team_row["source_id"])
        for team_row in connection.execute(
            "SELECT entity_id, source_id FROM source_mappings WHERE source = 'fbref' "
            "AND entity_type = 'team' AND entity_id IN (?, ?) AND valid_to IS NULL",
            (row["home_team_id"], row["away_team_id"]),
        ).fetchall()
    }
    competition, season, registered_team_source_ids = _schedule_registration(
        connection,
        competition_id=str(row["competition_id"]),
        season_id=str(row["season_id"]),
    )
    raw_row = connection.execute(
        "SELECT source, source_id, url, observed_at, target_event_time, checksum, "
        "collector_version, media_type, size_bytes FROM raw_assets WHERE raw_asset_id = ?",
        (row["raw_asset_id"],),
    ).fetchone()

    if len(source_rows) != 1 or len(team_sources) != 2 or raw_row is None:
        raise MatchContextReplayError("schedule canonical lineage is incomplete")
    if row["kickoff_at"] is None:
        raise MatchContextReplayError("schedule match version has no kickoff")
    asset = _verified_registered_asset(
        RawAssetId(str(row["raw_asset_id"])),
        raw_row=raw_row,
        archive=archive,
    )
    if (
        asset.source != "fbref"
        or asset.collector_version != COLLECTOR_VERSION
        or asset.media_type != "text/html"
        or _parse_timestamp(row["observed_at"], "match version observed_at") != asset.observed_at
    ):
        raise MatchContextReplayError("match version is not backed by an FBref schedule asset")

    parsed = _parse_verified_schedule(
        archive.read(asset),
        competition,
        season,
        asset.url,
    )
    source_fixture_id = str(source_rows[0]["source_id"])
    fixtures = [item for item in parsed.matches if item.source_fixture_id == source_fixture_id]
    if len(fixtures) != 1:
        raise MatchContextReplayError(
            "schedule raw does not contain the mapped fixture exactly once"
        )
    fixture = fixtures[0]
    kickoff = _parse_timestamp(row["kickoff_at"], "match kickoff")
    if (
        fixture.kickoff_at != kickoff
        or fixture.status is not MatchStatus(str(row["status"]))
        or fixture.round_name != row["round_name"]
        or team_sources.get(str(row["home_team_id"])) != fixture.home_source_id
        or team_sources.get(str(row["away_team_id"])) != fixture.away_source_id
    ):
        raise MatchContextReplayError("canonical match version does not match schedule raw bytes")

    if asset.observed_at <= as_of:
        schedule_known_at = asset.observed_at
        basis = "raw-observed"
    elif fixture.fixture_known_at is not None and fixture.fixture_known_at <= as_of:
        schedule_known_at = fixture.fixture_known_at
        basis = "fixture-version-known-at-metadata"
    else:
        raise _ScheduleNotKnown("schedule version has no evidence available by as_of")
    coverage_complete, coverage_known_at = _season_schedule_coverage(
        parsed.matches,
        diagnostics=parsed.diagnostics,
        registered_team_source_ids=registered_team_source_ids,
        expected_matches=season.expected_matches,
        as_of=as_of,
        raw_observed_at=asset.observed_at,
    )
    if coverage_known_at is not None:
        schedule_known_at = max(schedule_known_at, coverage_known_at)
    return _ScheduleEvidence(
        match_id=match_id,
        match_version=match_version,
        home_team_id=TeamId(str(row["home_team_id"])),
        away_team_id=TeamId(str(row["away_team_id"])),
        kickoff_at=kickoff,
        status=MatchStatus(str(row["status"])),
        raw_asset_id=asset.id,
        raw_observed_at=asset.observed_at,
        schedule_known_at=schedule_known_at,
        availability_basis=basis,
        season_schedule_coverage_complete=coverage_complete,
    )


def _schedule_registration(
    connection: sqlite3.Connection,
    *,
    competition_id: str,
    season_id: str,
) -> tuple[CompetitionDefinition, SeasonDefinition, frozenset[str]]:
    competition_row = connection.execute(
        "SELECT name, country_code, kind, timezone FROM competitions WHERE competition_id = ?",
        (competition_id,),
    ).fetchone()
    season_row = connection.execute(
        "SELECT label, starts_on, ends_on FROM seasons WHERE season_id = ? AND competition_id = ?",
        (season_id, competition_id),
    ).fetchone()
    source_rows = connection.execute(
        "SELECT source_id FROM source_mappings WHERE source = 'fbref' "
        "AND entity_type = 'season' AND entity_id = ? AND valid_to IS NULL",
        (season_id,),
    ).fetchall()
    team_rows = connection.execute(
        "SELECT mapping.entity_id, mapping.source_id FROM competition_teams AS registered "
        "JOIN source_mappings AS mapping ON mapping.entity_id = registered.team_id "
        "WHERE registered.competition_id = ? AND mapping.source = 'fbref' "
        "AND mapping.entity_type = 'team' AND mapping.valid_to IS NULL",
        (competition_id,),
    ).fetchall()
    team_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM competition_teams WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()[0]
    )
    registered_team_source_ids = frozenset(str(item["source_id"]) for item in team_rows)
    if (
        competition_row is None
        or season_row is None
        or len(source_rows) != 1
        or team_count < 2
        or len(team_rows) != team_count
        or len(registered_team_source_ids) != team_count
    ):
        raise MatchContextReplayError("schedule registration is incomplete")
    source_competition, separator, source_season = str(source_rows[0]["source_id"]).partition(":")
    if not separator or not source_competition or not source_season:
        raise MatchContextReplayError("schedule season source mapping is malformed")
    season = SeasonDefinition(
        id=SeasonId(season_id),
        label=str(season_row["label"]),
        starts_on=date.fromisoformat(str(season_row["starts_on"])),
        ends_on=date.fromisoformat(str(season_row["ends_on"])),
        expected_teams=team_count,
        expected_matches=team_count * (team_count - 1),
        sources=(SourceSeasonReference("fbref", source_competition, source_season),),
    )
    competition = CompetitionDefinition(
        id=CompetitionId(competition_id),
        name=str(competition_row["name"]),
        country_code=str(competition_row["country_code"]),
        kind=str(competition_row["kind"]),
        timezone=str(competition_row["timezone"]),
        seasons=(season,),
    )
    return competition, season, registered_team_source_ids


def _season_schedule_coverage(
    matches: tuple[Any, ...],
    *,
    diagnostics: tuple[Any, ...],
    registered_team_source_ids: frozenset[str],
    expected_matches: int,
    as_of: datetime,
    raw_observed_at: datetime,
) -> tuple[bool, datetime | None]:
    if raw_observed_at <= as_of:
        coverage_known_at: datetime | None = raw_observed_at
    else:
        fixture_known_times = tuple(item.fixture_known_at for item in matches)
        if any(item is None or item > as_of for item in fixture_known_times):
            return False, None
        coverage_known_at = max(fixture_known_times, default=None)

    observed_team_ids = {
        source_id for match in matches for source_id in (match.home_source_id, match.away_source_id)
    }
    directed_matchups = {(match.home_source_id, match.away_source_id) for match in matches}
    expected_matchups = {
        (home, away)
        for home in registered_team_source_ids
        for away in registered_team_source_ids
        if home != away
    }
    complete = (
        not diagnostics
        and len(matches) == expected_matches
        and observed_team_ids == registered_team_source_ids
        and len(directed_matchups) == expected_matches
        and directed_matchups == expected_matchups
    )
    return complete, coverage_known_at


def _verified_registered_asset(
    asset_id: RawAssetId,
    *,
    raw_row: sqlite3.Row,
    archive: RawArchive,
) -> RawAsset:
    asset = archive.load(asset_id)
    persisted = (
        raw_row["source"],
        raw_row["source_id"],
        raw_row["url"],
        raw_row["observed_at"],
        raw_row["target_event_time"],
        raw_row["checksum"],
        raw_row["collector_version"],
        raw_row["media_type"],
        raw_row["size_bytes"],
    )
    archived = (
        asset.source,
        asset.source_id,
        asset.url,
        _timestamp(asset.observed_at),
        _timestamp(asset.target_event_time) if asset.target_event_time is not None else None,
        asset.checksum,
        asset.collector_version,
        asset.media_type,
        asset.size_bytes,
    )
    if persisted != archived:
        raise MatchContextReplayError("canonical raw registration conflicts with raw archive")
    return asset


@lru_cache(maxsize=128)
def _parse_verified_schedule(
    content: bytes,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    page_url: str,
):
    """Cache parsing only after RawArchive.read has reverified immutable bytes."""

    return parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=page_url,
    )


def _schedule_payload(evidence: _ScheduleEvidence) -> dict[str, Any]:
    return {
        "raw_ref": evidence.raw_asset_id.value,
        "known_at": _timestamp(evidence.schedule_known_at),
        "availability_basis": evidence.availability_basis,
        "parser_version": COLLECTOR_VERSION,
        "season_schedule_coverage_complete": evidence.season_schedule_coverage_complete,
    }


def _timestamp(value: datetime) -> str:
    require_utc(value, "timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise MatchContextReplayError(f"{field_name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MatchContextReplayError(f"{field_name} must be an ISO timestamp") from error
    require_utc(parsed, field_name)
    return parsed.astimezone(UTC)
