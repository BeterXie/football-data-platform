"""SQLite catalog for canonical identities, mappings, and match versions."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from football_data_platform.config import CompetitionRegistry
from football_data_platform.domain.ids import (
    CompetitionId,
    EntityId,
    MatchId,
    PlayerId,
    RawAssetId,
    SeasonId,
    TeamId,
)
from football_data_platform.domain.models import (
    MappingRule,
    Match,
    MatchStatus,
    MatchVersion,
    RawAsset,
    require_utc,
)

SCHEMA_VERSION = 1
_ID_NAMESPACE = uuid.UUID("c62a4fc0-2e72-4d9c-b4b3-113b31c31982")


class CanonicalConflictError(RuntimeError):
    """Raised when an input contradicts an existing canonical fact."""


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
            entity_id = _current_mapping(connection, source, "player", source_id)
            if entity_id is not None:
                row = connection.execute(
                    "SELECT canonical_name FROM players WHERE player_id = ?",
                    (entity_id,),
                ).fetchone()
                if row is None:
                    raise CanonicalConflictError(f"mapped player does not exist: {entity_id}")
                return ResolvedPlayer(PlayerId(entity_id), row["canonical_name"])

            _require_raw_asset(connection, raw_asset_id)
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
    ) -> tuple[Match, MatchVersion]:
        require_utc(observed_at, "observed_at")
        if kickoff_at is not None:
            require_utc(kickoff_at, "kickoff_at")
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
                kickoff_at=kickoff_at,
                status=status,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )
            return match, version

    def match_versions(self, match_id: MatchId) -> tuple[MatchVersion, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT match_id, version, kickoff_at, status, observed_at "
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
    kickoff_at: datetime | None,
    status: MatchStatus,
    observed_at: datetime,
    raw_asset_id: RawAssetId,
) -> MatchVersion:
    latest = connection.execute(
        "SELECT match_id, version, kickoff_at, status, observed_at, raw_asset_id "
        "FROM match_versions WHERE match_id = ? ORDER BY version DESC LIMIT 1",
        (match_id.value,),
    ).fetchone()
    kickoff_text = _timestamp(kickoff_at) if kickoff_at is not None else None
    if latest is not None:
        if latest["kickoff_at"] == kickoff_text and latest["status"] == status.value:
            return _match_version_from_row(latest)
        if _parse_timestamp(latest["observed_at"]) > observed_at:
            raise CanonicalConflictError("cannot append an older observation as a new version")
        version_number = int(latest["version"]) + 1
    else:
        version_number = 1
    connection.execute(
        "INSERT INTO match_versions(match_id, version, kickoff_at, status, observed_at, "
        "raw_asset_id) VALUES (?, ?, ?, ?, ?, ?)",
        (
            match_id.value,
            version_number,
            kickoff_text,
            status.value,
            _timestamp(observed_at),
            raw_asset_id.value,
        ),
    )
    return MatchVersion(
        match_id=match_id,
        version=version_number,
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
        kickoff_at=(_parse_timestamp(row["kickoff_at"]) if row["kickoff_at"] is not None else None),
        status=MatchStatus(row["status"]),
        observed_at=_parse_timestamp(row["observed_at"]),
    )


def _require_raw_asset(connection: sqlite3.Connection, asset_id: RawAssetId) -> None:
    row = connection.execute(
        "SELECT 1 FROM raw_assets WHERE raw_asset_id = ?", (asset_id.value,)
    ).fetchone()
    if row is None:
        raise KeyError(f"raw asset {asset_id} is not registered")


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
    kickoff_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('scheduled', 'postponed', 'cancelled', 'finished')
    ),
    observed_at TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    PRIMARY KEY (match_id, version)
);

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
