"""SQLite catalog for canonical identities, mappings, and match versions."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from football_data_platform.config import CompetitionRegistry
from football_data_platform.domain.ids import (
    CollectionAttemptId,
    CompetitionId,
    EntityId,
    MatchId,
    PlayerId,
    RawAssetId,
    SeasonId,
    TeamId,
)
from football_data_platform.domain.models import (
    CollectionAttempt,
    CollectionAttemptOutcome,
    MappingRule,
    Match,
    MatchStatus,
    MatchVersion,
    RawAsset,
    require_utc,
)

SCHEMA_VERSION = 2
_ID_NAMESPACE = uuid.UUID("c62a4fc0-2e72-4d9c-b4b3-113b31c31982")


class CanonicalConflictError(RuntimeError):
    """Raised when an input contradicts an existing canonical fact."""


class CanonicalRebuildRequiredError(CanonicalConflictError):
    """An old split identity cannot be merged without operator review."""

    code = "canonical_identity_rebuild_required"

    def __init__(self, message: str, *, subject: str) -> None:
        super().__init__(message)
        self.subject = subject

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "subject": self.subject, "message": str(self)}


@dataclass(frozen=True, slots=True)
class ResolvedTeam:
    id: TeamId
    canonical_name: str


@dataclass(frozen=True, slots=True)
class ResolvedPlayer:
    id: PlayerId
    canonical_name: str


class CanonicalStore:
    """Own the normalized identity catalog and versioned fixture facts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(singleton, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            elif row["version"] == 1:
                _migrate_v1_to_v2(connection)
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"canonical schema {row['version']} is not supported by "
                    f"this code (expected {SCHEMA_VERSION})"
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register_registry(
        self,
        registry: CompetitionRegistry,
        *,
        registered_at: datetime,
    ) -> None:
        require_utc(registered_at, "registered_at")
        timestamp = _timestamp(registered_at)
        with self.connect() as connection:
            for competition in registry.competitions:
                _insert_entity(connection, competition.id, "competition", timestamp)
                _insert_exact(
                    connection,
                    "competitions",
                    {
                        "competition_id": competition.id.value,
                        "name": competition.name,
                        "country_code": competition.country_code,
                        "kind": competition.kind,
                        "timezone": competition.timezone,
                    },
                    "competition_id",
                )
                for season in competition.seasons:
                    _insert_entity(connection, season.id, "season", timestamp)
                    _insert_exact(
                        connection,
                        "seasons",
                        {
                            "season_id": season.id.value,
                            "competition_id": competition.id.value,
                            "label": season.label,
                            "starts_on": season.starts_on.isoformat(),
                            "ends_on": season.ends_on.isoformat(),
                        },
                        "season_id",
                    )
                    for reference in season.sources:
                        self._add_mapping(
                            connection,
                            source=reference.source,
                            source_id=(f"{reference.competition_id}:{reference.season_id}"),
                            entity_id=season.id,
                            entity_type="season",
                            valid_from=datetime.combine(
                                season.starts_on,
                                datetime.min.time(),
                                tzinfo=UTC,
                            ),
                            match_rule=MappingRule.MANUAL_OVERRIDE,
                            confidence=1.0,
                            created_at=registered_at,
                            audit_note="competition registry",
                        )

    def register_raw_asset(self, asset: RawAsset) -> None:
        with self.connect() as connection:
            _insert_exact(
                connection,
                "raw_assets",
                {
                    "raw_asset_id": asset.id.value,
                    "source": asset.source,
                    "source_id": asset.source_id,
                    "url": asset.url,
                    "observed_at": _timestamp(asset.observed_at),
                    "target_event_time": (
                        _timestamp(asset.target_event_time)
                        if asset.target_event_time is not None
                        else None
                    ),
                    "checksum": asset.checksum,
                    "collector_version": asset.collector_version,
                    "media_type": asset.media_type,
                    "size_bytes": asset.size_bytes,
                },
                "raw_asset_id",
            )

    def resolve_or_create_team(
        self,
        *,
        source: str,
        source_id: str,
        canonical_name: str,
        competition_id: CompetitionId,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> ResolvedTeam:
        _require_text(canonical_name, "canonical_name")
        require_utc(observed_at, "observed_at")
        with self.connect() as connection:
            entity_id = _current_mapping(connection, source, "team", source_id)
            if entity_id is not None:
                _require_raw_asset(connection, raw_asset_id)
                _require_entity(connection, competition_id, "competition")
                row = connection.execute(
                    "SELECT canonical_name FROM teams WHERE team_id = ?",
                    (entity_id,),
                ).fetchone()
                if row is None:
                    raise CanonicalConflictError(f"mapped team does not exist: {entity_id}")
                _ensure_competition_team(
                    connection,
                    competition_id=competition_id,
                    team_id=TeamId(entity_id),
                    observed_at=observed_at,
                    raw_asset_id=raw_asset_id,
                )
                return ResolvedTeam(TeamId(entity_id), row["canonical_name"])

            _require_raw_asset(connection, raw_asset_id)
            _require_entity(connection, competition_id, "competition")
            team_id = TeamId(_stable_id("team", source, source_id))
            _insert_entity(connection, team_id, "team", _timestamp(observed_at))
            connection.execute(
                "INSERT INTO teams(team_id, canonical_name) VALUES (?, ?)",
                (team_id.value, canonical_name),
            )
            _ensure_competition_team(
                connection,
                competition_id=competition_id,
                team_id=team_id,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )
            self._add_mapping(
                connection,
                source=source,
                source_id=source_id,
                entity_id=team_id,
                entity_type="team",
                valid_from=observed_at,
                match_rule=MappingRule.SOURCE_ID,
                confidence=1.0,
                created_at=observed_at,
                audit_note=f"first observed as {canonical_name}",
            )
            return ResolvedTeam(team_id, canonical_name)

    def resolve_registered_team(
        self,
        *,
        source: str,
        source_id: str,
        team_id: TeamId,
        canonical_name: str,
        observed_name: str,
        competition_id: CompetitionId,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> ResolvedTeam:
        """Resolve a team only through an explicit operator-owned source mapping."""

        _require_text(source, "source")
        _require_text(source_id, "source_id")
        _require_text(canonical_name, "canonical_name")
        _require_text(observed_name, "observed_name")
        require_utc(observed_at, "observed_at")
        with self.connect() as connection:
            _require_raw_asset(connection, raw_asset_id)
            _require_entity(connection, competition_id, "competition")
            mapped = _current_mapping(connection, source, "team", source_id)
            if mapped is not None and mapped != team_id.value:
                raise CanonicalRebuildRequiredError(
                    f"registered team mapping {source}:{source_id} expects {team_id}, "
                    f"but canonical storage maps it to {mapped}; rebuild or explicitly "
                    "reconcile the old canonical catalog",
                    subject=f"{source}:team:{source_id}",
                )

            entity = connection.execute(
                "SELECT entity_type FROM entities WHERE entity_id = ?", (team_id.value,)
            ).fetchone()
            if entity is None:
                _insert_entity(connection, team_id, "team", _timestamp(observed_at))
                connection.execute(
                    "INSERT INTO teams(team_id, canonical_name) VALUES (?, ?)",
                    (team_id.value, canonical_name),
                )
            else:
                _require_entity(connection, team_id, "team")
                row = connection.execute(
                    "SELECT canonical_name FROM teams WHERE team_id = ?", (team_id.value,)
                ).fetchone()
                if row is None or row["canonical_name"] != canonical_name:
                    actual = None if row is None else row["canonical_name"]
                    raise CanonicalConflictError(
                        f"registered team {team_id} name conflicts: "
                        f"stored={actual!r}, configured={canonical_name!r}"
                    )

            self._add_mapping(
                connection,
                source=source,
                source_id=source_id,
                entity_id=team_id,
                entity_type="team",
                valid_from=observed_at,
                match_rule=MappingRule.MANUAL_OVERRIDE,
                confidence=1.0,
                created_at=observed_at,
                audit_note=f"season registry mapping; observed alias {observed_name!r}",
            )
            _ensure_competition_team(
                connection,
                competition_id=competition_id,
                team_id=team_id,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )
            return ResolvedTeam(team_id, canonical_name)

    def resolve_or_create_player(
        self,
        *,
        source: str,
        source_id: str,
        canonical_name: str,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> ResolvedPlayer:
        _require_text(canonical_name, "canonical_name")
        require_utc(observed_at, "observed_at")
        with self.connect() as connection:
            _require_raw_asset(connection, raw_asset_id)
            entity_id = _current_mapping(connection, source, "player", source_id)
            if entity_id is not None:
                row = connection.execute(
                    "SELECT canonical_name FROM players WHERE player_id = ?",
                    (entity_id,),
                ).fetchone()
                if row is None:
                    raise CanonicalConflictError(f"mapped player does not exist: {entity_id}")
                return ResolvedPlayer(PlayerId(entity_id), row["canonical_name"])

            player_id = PlayerId(_stable_id("player", source, source_id))
            _insert_entity(connection, player_id, "player", _timestamp(observed_at))
            connection.execute(
                "INSERT INTO players(player_id, canonical_name) VALUES (?, ?)",
                (player_id.value, canonical_name),
            )
            self._add_mapping(
                connection,
                source=source,
                source_id=source_id,
                entity_id=player_id,
                entity_type="player",
                valid_from=observed_at,
                match_rule=MappingRule.SOURCE_ID,
                confidence=1.0,
                created_at=observed_at,
                audit_note=f"first observed as {canonical_name}",
            )
            return ResolvedPlayer(player_id, canonical_name)

    def resolve_or_create_match(
        self,
        *,
        source: str,
        source_id: str,
        competition_id: CompetitionId,
        season_id: SeasonId,
        home_team_id: TeamId,
        away_team_id: TeamId,
        kickoff_at: datetime | None,
        status: MatchStatus,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
        round_name: str | None = None,
    ) -> tuple[Match, MatchVersion]:
        require_utc(observed_at, "observed_at")
        if kickoff_at is not None:
            require_utc(kickoff_at, "kickoff_at")
        if round_name is not None:
            _require_text(round_name, "round_name")
        with self.connect() as connection:
            _require_raw_asset(connection, raw_asset_id)
            _require_entity(connection, competition_id, "competition")
            _require_entity(connection, season_id, "season")
            _require_entity(connection, home_team_id, "team")
            _require_entity(connection, away_team_id, "team")
            mapped = _current_mapping(connection, source, "match", source_id)
            if mapped is None:
                match_id = MatchId(_stable_id("match", source, source_id))
                match = Match(
                    id=match_id,
                    competition_id=competition_id,
                    season_id=season_id,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                )
                _insert_entity(connection, match_id, "match", _timestamp(observed_at))
                connection.execute(
                    "INSERT INTO matches(match_id, competition_id, season_id, "
                    "home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?)",
                    (
                        match_id.value,
                        competition_id.value,
                        season_id.value,
                        home_team_id.value,
                        away_team_id.value,
                    ),
                )
                self._add_mapping(
                    connection,
                    source=source,
                    source_id=source_id,
                    entity_id=match_id,
                    entity_type="match",
                    valid_from=observed_at,
                    match_rule=MappingRule.SOURCE_ID,
                    confidence=1.0,
                    created_at=observed_at,
                    audit_note="source fixture identity",
                )
            else:
                match_id = MatchId(mapped)
                match = _load_match(connection, match_id)
                incoming_identity = (
                    competition_id,
                    season_id,
                    home_team_id,
                    away_team_id,
                )
                stored_identity = (
                    match.competition_id,
                    match.season_id,
                    match.home_team_id,
                    match.away_team_id,
                )
                if stored_identity != incoming_identity:
                    raise CanonicalConflictError(
                        f"source match {source}:{source_id} changed stable identity"
                    )

            version = _append_match_version(
                connection,
                match_id=match_id,
                round_name=round_name,
                kickoff_at=kickoff_at,
                status=status,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )
            return match, version

    def resolve_or_create_round_robin_match(
        self,
        *,
        source: str,
        source_id: str,
        competition_id: CompetitionId,
        season_id: SeasonId,
        home_team_id: TeamId,
        away_team_id: TeamId,
        round_name: str | None,
        kickoff_at: datetime | None,
        status: MatchStatus,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> tuple[Match, MatchVersion]:
        """Resolve a fixture in a registered single/double round-robin season.

        The season and directed platform-team pair are unique only under that
        competition contract. Cup and context competitions must use the generic
        source-identity resolver instead.
        """

        require_utc(observed_at, "observed_at")
        if kickoff_at is not None:
            require_utc(kickoff_at, "kickoff_at")
        if round_name is not None:
            _require_text(round_name, "round_name")
        identity_key = "|".join((season_id.value, home_team_id.value, away_team_id.value))
        match_id = MatchId(_stable_id("match", "round-robin", identity_key))
        incoming_identity = (
            competition_id,
            season_id,
            home_team_id,
            away_team_id,
        )
        with self.connect() as connection:
            _require_raw_asset(connection, raw_asset_id)
            _require_entity(connection, competition_id, "competition")
            _require_entity(connection, season_id, "season")
            _require_entity(connection, home_team_id, "team")
            _require_entity(connection, away_team_id, "team")
            mapped = _current_mapping(connection, source, "match", source_id)
            if mapped is not None and mapped != match_id.value:
                raise CanonicalRebuildRequiredError(
                    f"source match {source}:{source_id} predates the registered "
                    "round-robin identity; rebuild or explicitly reconcile the old mapping",
                    subject=f"{source}:match:{source_id}",
                )

            row = connection.execute(
                "SELECT 1 FROM matches WHERE match_id = ?", (match_id.value,)
            ).fetchone()
            if row is None:
                match = Match(
                    id=match_id,
                    competition_id=competition_id,
                    season_id=season_id,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                )
                _insert_entity(connection, match_id, "match", _timestamp(observed_at))
                connection.execute(
                    "INSERT INTO matches(match_id, competition_id, season_id, "
                    "home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?)",
                    (
                        match_id.value,
                        competition_id.value,
                        season_id.value,
                        home_team_id.value,
                        away_team_id.value,
                    ),
                )
            else:
                match = _load_match(connection, match_id)
                stored_identity = (
                    match.competition_id,
                    match.season_id,
                    match.home_team_id,
                    match.away_team_id,
                )
                if stored_identity != incoming_identity:
                    raise CanonicalConflictError(
                        f"round-robin match {match_id} changed stable identity"
                    )

            self._add_mapping(
                connection,
                source=source,
                source_id=source_id,
                entity_id=match_id,
                entity_type="match",
                valid_from=observed_at,
                match_rule=MappingRule.SOURCE_ID,
                confidence=1.0,
                created_at=observed_at,
                audit_note="registered round-robin fixture identity",
            )
            version = _append_match_version(
                connection,
                match_id=match_id,
                round_name=round_name,
                kickoff_at=kickoff_at,
                status=status,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )
            return match, version

    def match_versions(self, match_id: MatchId) -> tuple[MatchVersion, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT match_id, version, round_name, kickoff_at, status, observed_at "
                "FROM match_versions WHERE match_id = ? ORDER BY version",
                (match_id.value,),
            ).fetchall()
        return tuple(_match_version_from_row(row) for row in rows)

    def match(self, match_id: MatchId) -> Match:
        with self.connect() as connection:
            return _load_match(connection, match_id)

    def mapped_team(self, *, source: str, source_id: str) -> ResolvedTeam:
        with self.connect() as connection:
            entity_id = _current_mapping(connection, source, "team", source_id)
            if entity_id is None:
                raise KeyError(f"team mapping {source}:{source_id} does not exist")
            row = connection.execute(
                "SELECT canonical_name FROM teams WHERE team_id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                raise CanonicalConflictError(f"mapped team does not exist: {entity_id}")
        return ResolvedTeam(TeamId(entity_id), row["canonical_name"])

    def mapped_player(self, *, source: str, source_id: str) -> ResolvedPlayer:
        with self.connect() as connection:
            entity_id = _current_mapping(connection, source, "player", source_id)
            if entity_id is None:
                raise KeyError(f"player mapping {source}:{source_id} does not exist")
            row = connection.execute(
                "SELECT canonical_name FROM players WHERE player_id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                raise CanonicalConflictError(f"mapped player does not exist: {entity_id}")
        return ResolvedPlayer(PlayerId(entity_id), row["canonical_name"])

    def mapped_match_ids(
        self,
        *,
        source: str,
        source_ids: Sequence[str],
    ) -> dict[str, MatchId]:
        """Resolve persisted provider mappings for a set of match identifiers."""

        if not source_ids:
            return {}
        placeholders = ", ".join("?" for _ in source_ids)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT source_id, entity_id FROM source_mappings "
                "WHERE source = ? AND entity_type = 'match' AND valid_to IS NULL "
                f"AND source_id IN ({placeholders})",
                (source, *source_ids),
            ).fetchall()
        return {str(row["source_id"]): MatchId(row["entity_id"]) for row in rows}

    def record_collection_attempt(
        self,
        *,
        match_id: MatchId,
        source: str,
        source_id: str | None = None,
        target_url: str,
        outcome: CollectionAttemptOutcome,
        observed_at: datetime,
        collector_version: str,
        diagnostic_code: str | None = None,
        diagnostic_message: str | None = None,
        raw_asset_id: RawAssetId | None = None,
    ) -> CollectionAttempt:
        identity = "|".join(
            (
                match_id.value,
                source,
                source_id or "",
                target_url,
                _timestamp(observed_at),
                collector_version,
                outcome.value,
            )
        )
        attempt = CollectionAttempt(
            id=CollectionAttemptId(_stable_id("collection-attempt", source, identity)),
            match_id=match_id,
            source=source,
            target_url=target_url,
            outcome=outcome,
            observed_at=observed_at,
            collector_version=collector_version,
            diagnostic_code=diagnostic_code,
            diagnostic_message=diagnostic_message,
            raw_asset_id=raw_asset_id,
        )
        with self.connect() as connection:
            _require_entity(connection, match_id, "match")
            if raw_asset_id is not None:
                _require_attempt_raw_lineage(
                    connection,
                    raw_asset_id=raw_asset_id,
                    attempt_source=source,
                    attempt_source_id=source_id,
                    target_url=target_url,
                    observed_at=observed_at,
                    collector_version=collector_version,
                )
            _insert_exact(
                connection,
                "collection_attempts",
                {
                    "collection_attempt_id": attempt.id.value,
                    "match_id": match_id.value,
                    "source": source,
                    "target_url": target_url,
                    "outcome": outcome.value,
                    "observed_at": _timestamp(observed_at),
                    "collector_version": collector_version,
                    "diagnostic_code": diagnostic_code,
                    "diagnostic_message": diagnostic_message,
                    "raw_asset_id": raw_asset_id.value if raw_asset_id is not None else None,
                },
                "collection_attempt_id",
            )
        return attempt

    def collection_attempts(self, season_id: SeasonId) -> tuple[CollectionAttempt, ...]:
        with self.connect() as connection:
            _require_entity(connection, season_id, "season")
            rows = connection.execute(
                "SELECT a.* FROM collection_attempts AS a "
                "JOIN matches AS m ON m.match_id = a.match_id "
                "WHERE m.season_id = ? ORDER BY a.observed_at, a.collection_attempt_id",
                (season_id.value,),
            ).fetchall()
        return tuple(_collection_attempt_from_row(row) for row in rows)

    def counts(self) -> dict[str, int]:
        tables = ("competitions", "seasons", "teams", "players", "matches")
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def _add_mapping(
        self,
        connection: sqlite3.Connection,
        *,
        source: str,
        source_id: str,
        entity_id: EntityId,
        entity_type: str,
        valid_from: datetime,
        match_rule: MappingRule,
        confidence: float,
        created_at: datetime,
        audit_note: str,
    ) -> None:
        _require_text(source, "source")
        _require_text(source_id, "source_id")
        require_utc(valid_from, "valid_from")
        require_utc(created_at, "created_at")
        existing = _current_mapping(connection, source, entity_type, source_id)
        if existing is not None:
            if existing != entity_id.value:
                raise CanonicalConflictError(
                    f"{source}:{entity_type}:{source_id} is already mapped to {existing}"
                )
            return
        connection.execute(
            "INSERT INTO source_mappings(source, entity_type, source_id, entity_id, "
            "valid_from, valid_to, match_rule, confidence, created_at, audit_note) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                source,
                entity_type,
                source_id,
                entity_id.value,
                _timestamp(valid_from),
                match_rule.value,
                confidence,
                _timestamp(created_at),
                audit_note,
            ),
        )


def _insert_entity(
    connection: sqlite3.Connection,
    entity_id: EntityId,
    entity_type: str,
    created_at: str,
) -> None:
    row = connection.execute(
        "SELECT entity_type FROM entities WHERE entity_id = ?", (entity_id.value,)
    ).fetchone()
    if row is not None:
        if row["entity_type"] != entity_type:
            raise CanonicalConflictError(
                f"entity {entity_id} has type {row['entity_type']}, not {entity_type}"
            )
        return
    connection.execute(
        "INSERT INTO entities(entity_id, entity_type, created_at) VALUES (?, ?, ?)",
        (entity_id.value, entity_type, created_at),
    )


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(match_versions)").fetchall()
    }
    if "round_name" not in columns:
        connection.execute("ALTER TABLE match_versions ADD COLUMN round_name TEXT")
    connection.execute("UPDATE schema_meta SET version = ? WHERE singleton = 1", (SCHEMA_VERSION,))


def _ensure_competition_team(
    connection: sqlite3.Connection,
    *,
    competition_id: CompetitionId,
    team_id: TeamId,
    observed_at: datetime,
    raw_asset_id: RawAssetId,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO competition_teams(competition_id, team_id, first_seen_at, "
        "raw_asset_id) VALUES (?, ?, ?, ?)",
        (
            competition_id.value,
            team_id.value,
            _timestamp(observed_at),
            raw_asset_id.value,
        ),
    )


def _insert_exact(
    connection: sqlite3.Connection,
    table: str,
    values: dict[str, object],
    primary_key: str,
) -> None:
    row = connection.execute(
        f"SELECT * FROM {table} WHERE {primary_key} = ?", (values[primary_key],)
    ).fetchone()
    if row is not None:
        differences = {key: (row[key], value) for key, value in values.items() if row[key] != value}
        if differences:
            raise CanonicalConflictError(f"{table} {values[primary_key]} conflicts: {differences}")
        return
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO {table}({columns}) VALUES ({placeholders})", tuple(values.values())
    )


def _current_mapping(
    connection: sqlite3.Connection,
    source: str,
    entity_type: str,
    source_id: str,
) -> str | None:
    row = connection.execute(
        "SELECT entity_id FROM source_mappings WHERE source = ? AND entity_type = ? "
        "AND source_id = ? AND valid_to IS NULL",
        (source, entity_type, source_id),
    ).fetchone()
    return None if row is None else str(row["entity_id"])


def _append_match_version(
    connection: sqlite3.Connection,
    *,
    match_id: MatchId,
    round_name: str | None,
    kickoff_at: datetime | None,
    status: MatchStatus,
    observed_at: datetime,
    raw_asset_id: RawAssetId,
) -> MatchVersion:
    latest = connection.execute(
        "SELECT match_id, version, round_name, kickoff_at, status, observed_at, raw_asset_id "
        "FROM match_versions WHERE match_id = ? ORDER BY version DESC LIMIT 1",
        (match_id.value,),
    ).fetchone()
    kickoff_text = _timestamp(kickoff_at) if kickoff_at is not None else None
    effective_round_name = round_name
    if latest is not None:
        if effective_round_name is None:
            effective_round_name = latest["round_name"]
        if (
            latest["round_name"] == effective_round_name
            and latest["kickoff_at"] == kickoff_text
            and latest["status"] == status.value
        ):
            return _match_version_from_row(latest)
        if _parse_timestamp(latest["observed_at"]) > observed_at:
            raise CanonicalConflictError("cannot append an older observation as a new version")
        version_number = int(latest["version"]) + 1
    else:
        version_number = 1
    connection.execute(
        "INSERT INTO match_versions(match_id, version, round_name, kickoff_at, status, "
        "observed_at, raw_asset_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            match_id.value,
            version_number,
            effective_round_name,
            kickoff_text,
            status.value,
            _timestamp(observed_at),
            raw_asset_id.value,
        ),
    )
    return MatchVersion(
        match_id=match_id,
        version=version_number,
        round_name=effective_round_name,
        kickoff_at=kickoff_at,
        status=status,
        observed_at=observed_at,
    )


def _load_match(connection: sqlite3.Connection, match_id: MatchId) -> Match:
    row = connection.execute(
        "SELECT * FROM matches WHERE match_id = ?", (match_id.value,)
    ).fetchone()
    if row is None:
        raise KeyError(f"match {match_id} does not exist")
    return Match(
        id=match_id,
        competition_id=CompetitionId(row["competition_id"]),
        season_id=SeasonId(row["season_id"]),
        home_team_id=TeamId(row["home_team_id"]),
        away_team_id=TeamId(row["away_team_id"]),
    )


def _match_version_from_row(row: sqlite3.Row) -> MatchVersion:
    return MatchVersion(
        match_id=MatchId(row["match_id"]),
        version=int(row["version"]),
        round_name=row["round_name"],
        kickoff_at=(_parse_timestamp(row["kickoff_at"]) if row["kickoff_at"] is not None else None),
        status=MatchStatus(row["status"]),
        observed_at=_parse_timestamp(row["observed_at"]),
    )


def _collection_attempt_from_row(row: sqlite3.Row) -> CollectionAttempt:
    return CollectionAttempt(
        id=CollectionAttemptId(row["collection_attempt_id"]),
        match_id=MatchId(row["match_id"]),
        source=row["source"],
        target_url=row["target_url"],
        outcome=CollectionAttemptOutcome(row["outcome"]),
        observed_at=_parse_timestamp(row["observed_at"]),
        collector_version=row["collector_version"],
        diagnostic_code=row["diagnostic_code"],
        diagnostic_message=row["diagnostic_message"],
        raw_asset_id=(RawAssetId(row["raw_asset_id"]) if row["raw_asset_id"] is not None else None),
    )


def _require_raw_asset(connection: sqlite3.Connection, asset_id: RawAssetId) -> None:
    row = connection.execute(
        "SELECT 1 FROM raw_assets WHERE raw_asset_id = ?", (asset_id.value,)
    ).fetchone()
    if row is None:
        raise KeyError(f"raw asset {asset_id} is not registered")


def _require_attempt_raw_lineage(
    connection: sqlite3.Connection,
    *,
    raw_asset_id: RawAssetId,
    attempt_source: str,
    attempt_source_id: str | None,
    target_url: str,
    observed_at: datetime,
    collector_version: str,
) -> None:
    row = connection.execute(
        "SELECT source, source_id, url, observed_at, collector_version "
        "FROM raw_assets WHERE raw_asset_id = ?",
        (raw_asset_id.value,),
    ).fetchone()
    if row is None:
        raise KeyError(f"raw asset {raw_asset_id} is not registered")
    source_matches = row["source"] == attempt_source or attempt_source.startswith(
        f"{row['source']}-"
    )
    if not source_matches:
        raise CanonicalConflictError(
            f"collection attempt source conflicts with raw source: "
            f"raw={row['source']!r}, attempt={attempt_source!r}"
        )
    if attempt_source_id is not None and row["source_id"] != attempt_source_id:
        raise CanonicalConflictError(
            f"collection attempt source ID conflicts: "
            f"raw={row['source_id']!r}, attempt={attempt_source_id!r}"
        )
    if attempt_source == "fbref-match-report" and row["source"] == "fbref":
        if row["source_id"] not in target_url:
            raise CanonicalConflictError(
                "FBref match-report target URL does not identify the archived source match"
            )
    expected = (target_url, _timestamp(observed_at), collector_version)
    actual = (row["url"], row["observed_at"], row["collector_version"])
    if actual != expected:
        raise CanonicalConflictError(
            f"collection attempt raw lineage conflicts: stored={actual!r}, expected={expected!r}"
        )


def _require_entity(
    connection: sqlite3.Connection,
    entity_id: EntityId,
    entity_type: str,
) -> None:
    row = connection.execute(
        "SELECT entity_type FROM entities WHERE entity_id = ?", (entity_id.value,)
    ).fetchone()
    if row is None:
        raise KeyError(f"entity {entity_id} is not registered")
    if row["entity_type"] != entity_type:
        raise CanonicalConflictError(
            f"entity {entity_id} has type {row['entity_type']}, not {entity_type}"
        )


def _stable_id(entity_type: str, source: str, source_id: str) -> str:
    return f"{entity_type}:{uuid.uuid5(_ID_NAMESPACE, f'{entity_type}|{source}|{source_id}')}"


def _timestamp(value: datetime) -> str:
    require_utc(value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_utc(parsed)
    return parsed


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL CHECK (version > 0)
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (
        entity_type IN ('competition', 'season', 'team', 'player', 'match')
    ),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competitions (
    competition_id TEXT PRIMARY KEY REFERENCES entities(entity_id),
    name TEXT NOT NULL,
    country_code TEXT NOT NULL,
    kind TEXT NOT NULL,
    timezone TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seasons (
    season_id TEXT PRIMARY KEY REFERENCES entities(entity_id),
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    label TEXT NOT NULL,
    starts_on TEXT NOT NULL,
    ends_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY REFERENCES entities(entity_id),
    canonical_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY REFERENCES entities(entity_id),
    canonical_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competition_teams (
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    first_seen_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    PRIMARY KEY (competition_id, team_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY REFERENCES entities(entity_id),
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    season_id TEXT NOT NULL REFERENCES seasons(season_id),
    home_team_id TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id TEXT NOT NULL REFERENCES teams(team_id),
    CHECK (home_team_id <> away_team_id)
);

CREATE TABLE IF NOT EXISTS raw_assets (
    raw_asset_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    target_event_time TEXT,
    checksum TEXT NOT NULL,
    collector_version TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
);

CREATE TABLE IF NOT EXISTS source_mappings (
    source TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    match_rule TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL,
    audit_note TEXT NOT NULL,
    PRIMARY KEY (source, entity_type, source_id, valid_from)
);

CREATE UNIQUE INDEX IF NOT EXISTS source_mappings_current
ON source_mappings(source, entity_type, source_id)
WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS match_versions (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    version INTEGER NOT NULL CHECK (version > 0),
    round_name TEXT,
    kickoff_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('scheduled', 'postponed', 'cancelled', 'finished')
    ),
    observed_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    PRIMARY KEY (match_id, version)
);

CREATE TABLE IF NOT EXISTS collection_attempts (
    collection_attempt_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    source TEXT NOT NULL,
    target_url TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('succeeded', 'blocked', 'failed')),
    observed_at TEXT NOT NULL,
    collector_version TEXT NOT NULL,
    diagnostic_code TEXT,
    diagnostic_message TEXT,
    raw_asset_id TEXT REFERENCES raw_assets(raw_asset_id),
    CHECK (outcome <> 'succeeded' OR raw_asset_id IS NOT NULL),
    CHECK (outcome = 'succeeded' OR diagnostic_code IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS collection_attempts_match
ON collection_attempts(match_id, observed_at);

CREATE TABLE IF NOT EXISTS match_results_90 (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    observation_version INTEGER NOT NULL CHECK (observation_version > 0),
    home_goals INTEGER NOT NULL CHECK (home_goals >= 0),
    away_goals INTEGER NOT NULL CHECK (away_goals >= 0),
    known_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version),
    UNIQUE (match_id, match_version, observation_version)
);

CREATE TABLE IF NOT EXISTS team_match_observations (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    observation_version INTEGER NOT NULL CHECK (observation_version > 0),
    known_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version),
    UNIQUE (match_id, match_version, team_id, observation_version)
);

CREATE TABLE IF NOT EXISTS player_match_observations (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    observation_version INTEGER NOT NULL CHECK (observation_version > 0),
    role TEXT NOT NULL,
    minutes REAL NOT NULL CHECK (minutes >= 0),
    known_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version),
    UNIQUE (match_id, match_version, player_id, observation_version)
);

CREATE TABLE IF NOT EXISTS lineup_facts (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    lineup_role TEXT NOT NULL CHECK (lineup_role IN ('starter', 'bench')),
    official INTEGER NOT NULL CHECK (official IN (0, 1)),
    known_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version),
    UNIQUE (match_id, match_version, team_id, player_id, known_at, raw_asset_id)
);

CREATE TABLE IF NOT EXISTS news_evidence (
    record_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id)
);

CREATE TABLE IF NOT EXISTS prematch_events (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    team_id TEXT REFERENCES teams(team_id),
    player_id TEXT REFERENCES players(player_id),
    event_type TEXT NOT NULL,
    occurred_at TEXT,
    known_at TEXT NOT NULL,
    confirmation_status TEXT NOT NULL CHECK (
        confirmation_status IN ('official', 'corroborated', 'unconfirmed')
    ),
    evidence_refs_json TEXT NOT NULL,
    can_modify_features INTEGER NOT NULL CHECK (can_modify_features IN (0, 1))
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    record_id TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (record_id, raw_asset_id)
);
"""
