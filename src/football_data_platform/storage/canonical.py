"""SQLite catalog for canonical identities, mappings, and match versions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
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
    SourceMapping,
    SourceMappingConflict,
    SourceMappingDecision,
    SourceMappingEvidence,
    require_utc,
)

SCHEMA_VERSION = 7
OFFICIAL_LINEUP_CONTRACT_VERSION = 2
LEGACY_OFFICIAL_LINEUP_CONTRACT_VERSION = 1
_ID_NAMESPACE = uuid.UUID("c62a4fc0-2e72-4d9c-b4b3-113b31c31982")
_UUID_GLOB_SUFFIX = "-".join("[0-9a-f]" * length for length in (8, 4, 4, 4, 12))


class CanonicalConflictError(RuntimeError):
    """Raised when an input contradicts an existing canonical fact."""


class SourceMappingCASConflict(CanonicalConflictError):
    """Raised when a mapping changed after an operator loaded it for review."""


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


@dataclass(frozen=True, slots=True)
class MatchReportContractEvidence:
    contract_id: str
    collection_attempt_id: CollectionAttemptId
    match_id: MatchId
    match_version: int
    raw_asset_id: RawAssetId
    source_match_id: str
    parser_version: str
    required_tables: tuple[str, ...]
    team_tables: tuple[tuple[str, tuple[str, ...]], ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.match_version < 1:
            raise ValueError("match_version must be positive")
        for value, field_name in (
            (self.contract_id, "contract_id"),
            (self.source_match_id, "source_match_id"),
            (self.parser_version, "parser_version"),
        ):
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be non-empty text")
        if not self.required_tables:
            raise ValueError("required_tables must not be empty")
        if len(set(self.required_tables)) != len(self.required_tables):
            raise ValueError("required_tables must be unique")
        if len(self.team_tables) != 2:
            raise ValueError("team_tables must contain exactly two report teams")
        if len({team_id for team_id, _ in self.team_tables}) != len(self.team_tables):
            raise ValueError("team_tables team IDs must be unique")
        for team_id, tables in self.team_tables:
            if not team_id or not tables or len(set(tables)) != len(tables):
                raise ValueError("team table evidence must have a team ID and unique tables")
        require_utc(self.observed_at, "observed_at")


@dataclass(frozen=True, order=True, slots=True)
class OfficialLineupSourceBinding:
    source_team_id: str
    source_player_id: str
    team_id: TeamId
    player_id: PlayerId

    def __post_init__(self) -> None:
        _require_text(self.source_team_id, "source_team_id")
        _require_text(self.source_player_id, "source_player_id")
        if not isinstance(self.team_id, TeamId):
            raise TypeError("official lineup binding team_id must be a TeamId")
        if not isinstance(self.player_id, PlayerId):
            raise TypeError("official lineup binding player_id must be a PlayerId")

    def to_payload(self) -> dict[str, str]:
        return {
            "source_team_id": self.source_team_id,
            "source_player_id": self.source_player_id,
            "team_id": self.team_id.value,
            "player_id": self.player_id.value,
        }


@dataclass(frozen=True, slots=True)
class OfficialLineupContractEvidence:
    contract_version: int
    contract_id: str
    raw_asset_id: RawAssetId
    source: str
    source_match_id: str
    match_mapping_source: str
    team_mapping_source: str
    player_mapping_source: str
    match_id: MatchId
    match_version: int
    parser_version: str
    published_at: datetime
    observed_at: datetime
    team_lineups: tuple[tuple[TeamId, tuple[PlayerId, ...]], ...]
    source_bindings: tuple[OfficialLineupSourceBinding, ...]
    fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.contract_id, "contract_id"),
            (self.source, "source"),
            (self.source_match_id, "source_match_id"),
            (self.match_mapping_source, "match_mapping_source"),
            (self.team_mapping_source, "team_mapping_source"),
            (self.player_mapping_source, "player_mapping_source"),
            (self.parser_version, "parser_version"),
        ):
            _require_text(value, field_name)
        if isinstance(self.contract_version, bool) or self.contract_version not in {
            LEGACY_OFFICIAL_LINEUP_CONTRACT_VERSION,
            OFFICIAL_LINEUP_CONTRACT_VERSION,
        }:
            raise ValueError("unsupported official lineup contract version")
        if isinstance(self.match_version, bool) or self.match_version < 1:
            raise ValueError("match_version must be a positive integer")
        require_utc(self.published_at, "published_at")
        require_utc(self.observed_at, "observed_at")
        if self.published_at > self.observed_at:
            raise ValueError("official lineup publication cannot follow observation")
        if len(self.team_lineups) != 2:
            raise ValueError("official lineup contract requires exactly two teams")
        team_ids = [team_id for team_id, _ in self.team_lineups]
        if len(set(team_ids)) != 2:
            raise ValueError("official lineup contract team IDs must be unique")
        all_players: list[PlayerId] = []
        for team_id, player_ids in self.team_lineups:
            if not isinstance(team_id, TeamId):
                raise TypeError("official lineup contract teams must use TeamId")
            if (
                len(player_ids) != 11
                or len(set(player_ids)) != 11
                or any(not isinstance(player_id, PlayerId) for player_id in player_ids)
            ):
                raise ValueError("official lineup contract requires 11 unique players per team")
            all_players.extend(player_ids)
        if len(set(all_players)) != len(all_players):
            raise ValueError("official lineup players must be unique across both teams")
        if len(self.fact_ids) != 22 or len(set(self.fact_ids)) != 22:
            raise ValueError("official lineup contract requires 22 unique fact references")
        if any(not fact_id.startswith("fact:lineup:") for fact_id in self.fact_ids):
            raise ValueError("official lineup contract fact references are invalid")
        normalized_bindings = _normalize_official_lineup_source_bindings(self.source_bindings)
        if self.source_bindings != normalized_bindings:
            raise ValueError("official lineup source bindings must use canonical order")
        if self.contract_version == LEGACY_OFFICIAL_LINEUP_CONTRACT_VERSION:
            if self.source_bindings:
                raise ValueError("legacy official lineup contracts cannot contain source bindings")
        else:
            _validate_official_lineup_source_bindings(
                self.source_bindings,
                team_lineups=self.team_lineups,
            )
        expected_id = _official_lineup_contract_id(
            contract_version=self.contract_version,
            raw_asset_id=self.raw_asset_id,
            source=self.source,
            source_match_id=self.source_match_id,
            match_mapping_source=self.match_mapping_source,
            team_mapping_source=self.team_mapping_source,
            player_mapping_source=self.player_mapping_source,
            match_id=self.match_id,
            match_version=self.match_version,
            parser_version=self.parser_version,
            published_at=self.published_at,
            observed_at=self.observed_at,
            team_lineups=self.team_lineups,
            source_bindings=self.source_bindings,
            fact_ids=self.fact_ids,
        )
        if self.contract_id != expected_id:
            raise CanonicalConflictError(
                "official lineup contract content ID does not match payload"
            )


class CanonicalStore:
    """Own the normalized identity catalog and versioned fixture facts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _execute_sql_script(connection, _SCHEMA)
            row = connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(singleton, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            elif row["version"] not in range(1, SCHEMA_VERSION + 1):
                raise RuntimeError(
                    f"canonical schema {row['version']} is not supported by "
                    f"this code (expected {SCHEMA_VERSION})"
                )
            elif row["version"] < SCHEMA_VERSION:
                version = int(row["version"])
                if version <= 1:
                    _migrate_v1_to_v2(connection)
                if version <= 2:
                    _migrate_v2_to_v3(connection)
                if version <= 3:
                    _migrate_v3_to_v4(connection)
                if version <= 4:
                    _migrate_v4_to_v5(connection)
                if version <= 5:
                    _migrate_v5_to_v6(connection)
                _migrate_v6_to_v7(connection)
            _validate_existing_source_mappings_v7(connection)
            _create_source_mapping_v7_auxiliary(connection)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "source_mapping_id",
            4,
            _source_mapping_id,
            deterministic=True,
        )
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
                            match_rule=MappingRule.SOURCE_ID,
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
                match_rule=MappingRule.EXACT_ALIAS,
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
        with self.connect() as connection:
            return self._resolve_or_create_player(
                connection,
                source=source,
                source_id=source_id,
                canonical_name=canonical_name,
                observed_at=observed_at,
                raw_asset_id=raw_asset_id,
            )

    def _resolve_or_create_player(
        self,
        connection: sqlite3.Connection,
        *,
        source: str,
        source_id: str,
        canonical_name: str,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> ResolvedPlayer:
        """Resolve a player inside a caller-owned canonical transaction."""

        _require_text(canonical_name, "canonical_name")
        require_utc(observed_at, "observed_at")
        _require_raw_asset(connection, raw_asset_id)
        entity_id = _mapping_as_of(connection, source, "player", source_id, observed_at)
        if entity_id is not None:
            row = connection.execute(
                "SELECT canonical_name FROM players WHERE player_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise CanonicalConflictError(f"mapped player does not exist: {entity_id}")
            return ResolvedPlayer(PlayerId(entity_id), row["canonical_name"])

        later_mapping = connection.execute(
            "SELECT 1 FROM source_mappings WHERE source = ? AND entity_type = 'player' "
            "AND source_id = ? LIMIT 1",
            (source, source_id),
        ).fetchone()
        if later_mapping is not None:
            raise CanonicalConflictError(
                f"{source}:player:{source_id} has no mapping at {observed_at.isoformat()}"
            )

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

    def match_versions(
        self,
        match_id: MatchId,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> tuple[MatchVersion, ...]:
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            rows = connection.execute(
                "SELECT match_id, version, round_name, kickoff_at, status, observed_at "
                "FROM match_versions WHERE match_id = ? ORDER BY version",
                (match_id.value,),
            ).fetchall()
        return tuple(_match_version_from_row(row) for row in rows)

    def match(
        self,
        match_id: MatchId,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> Match:
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            return _load_match(connection, match_id)

    def mapped_team(
        self,
        *,
        source: str,
        source_id: str,
        as_of: datetime | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> ResolvedTeam:
        mapping = self.resolve_source_mapping(
            source=source,
            entity_type="team",
            source_id=source_id,
            as_of=as_of,
            _connection=_connection,
        )
        entity_id = mapping.entity_id.value
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            row = connection.execute(
                "SELECT canonical_name FROM teams WHERE team_id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                raise CanonicalConflictError(f"mapped team does not exist: {entity_id}")
        return ResolvedTeam(TeamId(entity_id), row["canonical_name"])

    def mapped_player(
        self,
        *,
        source: str,
        source_id: str,
        as_of: datetime | None = None,
    ) -> ResolvedPlayer:
        mapping = self.resolve_source_mapping(
            source=source,
            entity_type="player",
            source_id=source_id,
            as_of=as_of,
        )
        entity_id = mapping.entity_id.value
        with self.connect() as connection:
            row = connection.execute(
                "SELECT canonical_name FROM players WHERE player_id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                raise CanonicalConflictError(f"mapped player does not exist: {entity_id}")
        return ResolvedPlayer(PlayerId(entity_id), row["canonical_name"])

    def resolve_source_mapping(
        self,
        *,
        source: str,
        entity_type: str,
        source_id: str,
        as_of: datetime | None = None,
        version: int | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> SourceMapping:
        """Resolve the current, historical-time, or explicit version of one mapping key."""

        _validate_mapping_key(source, entity_type, source_id)
        if as_of is not None and version is not None:
            raise ValueError("source mapping resolution accepts either as_of or version")
        if as_of is not None:
            require_utc(as_of, "as_of")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            raise ValueError("source mapping version must be a positive integer")
        clauses = ["source = ?", "entity_type = ?", "source_id = ?"]
        parameters: list[object] = [source, entity_type, source_id]
        if version is not None:
            clauses.append("version = ?")
            parameters.append(version)
        elif as_of is not None:
            clauses.extend(("valid_from <= ?", "(valid_to IS NULL OR ? < valid_to)"))
            timestamp = _mapping_timestamp(as_of)
            parameters.extend((timestamp, timestamp))
        else:
            clauses.append("valid_to IS NULL")
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            row = connection.execute(
                "SELECT * FROM source_mappings WHERE " + " AND ".join(clauses),
                tuple(parameters),
            ).fetchone()
        if row is None:
            selector = f"version={version}" if version is not None else f"as_of={as_of}"
            raise KeyError(
                f"source mapping {source}:{entity_type}:{source_id} ({selector}) does not exist"
            )
        return _source_mapping_from_row(row)

    def mapping_history(
        self,
        *,
        source: str,
        entity_type: str,
        source_id: str,
    ) -> tuple[SourceMapping, ...]:
        _validate_mapping_key(source, entity_type, source_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM source_mappings WHERE source = ? AND entity_type = ? "
                "AND source_id = ? ORDER BY version",
                (source, entity_type, source_id),
            ).fetchall()
        return tuple(_source_mapping_from_row(row) for row in rows)

    def propose_source_mapping(
        self,
        *,
        source: str,
        entity_type: str,
        source_id: str,
        candidate_entity_id: EntityId,
        proposed_at: datetime,
        actor: str,
        reason: str,
        evidence_refs: Sequence[str],
    ) -> SourceMapping | SourceMappingConflict:
        """Persist evidence for the current target or queue a contradictory candidate."""

        _validate_mapping_key(source, entity_type, source_id)
        require_utc(proposed_at, "proposed_at")
        _require_text(actor, "actor")
        _require_text(reason, "reason")
        normalized_evidence = _normalize_mapping_evidence_refs(evidence_refs)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_entity(connection, candidate_entity_id, entity_type)
            current = connection.execute(
                "SELECT * FROM source_mappings WHERE source = ? AND entity_type = ? "
                "AND source_id = ? AND valid_to IS NULL",
                (source, entity_type, source_id),
            ).fetchone()
            if current is not None and current["entity_id"] == candidate_entity_id.value:
                for evidence_ref in normalized_evidence:
                    _insert_source_mapping_evidence(
                        connection,
                        mapping_id=str(current["mapping_id"]),
                        conflict_id=None,
                        evidence_ref=evidence_ref,
                        recorded_at=proposed_at,
                        recorded_by=actor,
                        reason=reason,
                    )
                return _source_mapping_from_row(current)

            conflict = connection.execute(
                "SELECT conflict.*, COALESCE(decision.decision_id, obsolete.decision_id) "
                "AS decision_id, current.entity_id AS current_entity_id "
                "FROM source_mapping_conflicts AS conflict "
                "LEFT JOIN source_mappings AS current "
                "ON current.mapping_id = conflict.current_mapping_id "
                "LEFT JOIN source_mapping_decisions AS decision "
                "ON decision.conflict_id = conflict.conflict_id "
                "LEFT JOIN source_mapping_conflict_obsoletions AS obsolete "
                "ON obsolete.conflict_id = conflict.conflict_id "
                "WHERE conflict.source = ? AND conflict.entity_type = ? "
                "AND conflict.source_id = ? AND conflict.candidate_entity_id = ? "
                "AND ((? IS NULL AND conflict.current_mapping_id IS NULL) "
                "OR conflict.current_mapping_id = ?) "
                "AND decision.decision_id IS NULL AND obsolete.obsoletion_id IS NULL",
                (
                    source,
                    entity_type,
                    source_id,
                    candidate_entity_id.value,
                    None if current is None else current["mapping_id"],
                    None if current is None else current["mapping_id"],
                ),
            ).fetchone()
            if conflict is None:
                conflict_id = _stable_id(
                    "source-mapping-conflict",
                    source,
                    "|".join(
                        (
                            entity_type,
                            source_id,
                            "unmapped" if current is None else str(current["mapping_id"]),
                            candidate_entity_id.value,
                        )
                    ),
                )
                connection.execute(
                    "INSERT INTO source_mapping_conflicts("
                    "conflict_id, source, entity_type, source_id, current_mapping_id, "
                    "candidate_entity_id, proposed_at, proposed_by, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        conflict_id,
                        source,
                        entity_type,
                        source_id,
                        None if current is None else current["mapping_id"],
                        candidate_entity_id.value,
                        _mapping_timestamp(proposed_at),
                        actor,
                        reason,
                    ),
                )
                conflict = connection.execute(
                    "SELECT conflict.*, NULL AS decision_id, "
                    "current.entity_id AS current_entity_id "
                    "FROM source_mapping_conflicts AS conflict "
                    "LEFT JOIN source_mappings AS current "
                    "ON current.mapping_id = conflict.current_mapping_id "
                    "WHERE conflict.conflict_id = ?",
                    (conflict_id,),
                ).fetchone()
                assert conflict is not None
            for evidence_ref in normalized_evidence:
                _insert_source_mapping_evidence(
                    connection,
                    mapping_id=None,
                    conflict_id=str(conflict["conflict_id"]),
                    evidence_ref=evidence_ref,
                    recorded_at=proposed_at,
                    recorded_by=actor,
                    reason=reason,
                )
            return _source_mapping_conflict_from_row(conflict)

    def revise_source_mapping(
        self,
        *,
        source: str,
        entity_type: str,
        source_id: str,
        conflict_id: str,
        expected_current_mapping_id: str | None,
        candidate_entity_id: EntityId,
        effective_at: datetime,
        actor: str,
        reason: str,
        evidence_refs: Sequence[str],
    ) -> tuple[SourceMapping, SourceMappingDecision]:
        """Accept one reviewed conflict using a compare-and-swap mapping revision."""

        _validate_mapping_key(source, entity_type, source_id)
        for value, field_name in (
            (conflict_id, "conflict_id"),
            (actor, "actor"),
            (reason, "reason"),
        ):
            _require_text(value, field_name)
        if expected_current_mapping_id is not None:
            _require_text(expected_current_mapping_id, "expected_current_mapping_id")
        require_utc(effective_at, "effective_at")
        normalized_evidence = _normalize_mapping_evidence_refs(evidence_refs)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_entity(connection, candidate_entity_id, entity_type)
            prior_decision = connection.execute(
                "SELECT * FROM source_mapping_decisions WHERE conflict_id = ?",
                (conflict_id,),
            ).fetchone()
            if prior_decision is not None:
                prior_mapping = connection.execute(
                    "SELECT * FROM source_mappings WHERE mapping_id = ?",
                    (prior_decision["new_mapping_id"],),
                ).fetchone()
                assert prior_mapping is not None
                expected_evidence_ids = {
                    _source_mapping_evidence_id(
                        mapping_id=str(prior_mapping["mapping_id"]),
                        conflict_id=None,
                        evidence_ref=evidence_ref,
                        recorded_at=effective_at,
                        recorded_by=actor,
                        reason=reason,
                    )
                    for evidence_ref in normalized_evidence
                }
                decided_evidence_ids = set(json.loads(str(prior_decision["evidence_ids_json"])))
                if (
                    prior_decision["previous_mapping_id"] == expected_current_mapping_id
                    and prior_mapping["source"] == source
                    and prior_mapping["entity_type"] == entity_type
                    and prior_mapping["source_id"] == source_id
                    and prior_mapping["entity_id"] == candidate_entity_id.value
                    and prior_mapping["valid_from"] == _mapping_timestamp(effective_at)
                    and prior_decision["decided_at"] == _mapping_timestamp(effective_at)
                    and prior_decision["decided_by"] == actor
                    and prior_decision["reason"] == reason
                    and expected_evidence_ids == decided_evidence_ids
                ):
                    return (
                        _source_mapping_from_row(prior_mapping),
                        _source_mapping_decision_from_row(prior_decision),
                    )
                raise SourceMappingCASConflict(
                    "source mapping conflict was already resolved by a different decision"
                )
            current = connection.execute(
                "SELECT * FROM source_mappings WHERE source = ? AND entity_type = ? "
                "AND source_id = ? AND valid_to IS NULL",
                (source, entity_type, source_id),
            ).fetchone()
            if (current is None) != (expected_current_mapping_id is None) or (
                current is not None and current["mapping_id"] != expected_current_mapping_id
            ):
                actual = None if current is None else str(current["mapping_id"])
                raise SourceMappingCASConflict(
                    f"source mapping changed during review: expected "
                    f"{expected_current_mapping_id!r}, current={actual!r}"
                )
            conflict = connection.execute(
                "SELECT conflict.*, decision.decision_id "
                "FROM source_mapping_conflicts AS conflict "
                "LEFT JOIN source_mapping_decisions AS decision "
                "ON decision.conflict_id = conflict.conflict_id "
                "WHERE conflict.conflict_id = ?",
                (conflict_id,),
            ).fetchone()
            if (
                conflict is None
                or conflict["decision_id"] is not None
                or conflict["source"] != source
                or conflict["entity_type"] != entity_type
                or conflict["source_id"] != source_id
                or conflict["current_mapping_id"] != expected_current_mapping_id
                or conflict["candidate_entity_id"] != candidate_entity_id.value
            ):
                raise CanonicalConflictError(
                    "manual override must match one open conflict and its reviewed candidate"
                )
            if effective_at < _parse_timestamp(str(conflict["proposed_at"])):
                raise CanonicalConflictError("mapping revision cannot predate its conflict")
            if current is not None and effective_at <= _parse_timestamp(str(current["valid_from"])):
                raise CanonicalConflictError(
                    "mapping revision must follow the current validity start"
                )
            new_version = 1 if current is None else int(current["version"]) + 1
            new_mapping_id = _source_mapping_id(source, entity_type, source_id, new_version)
            decision_id = _stable_id(
                "source-mapping-decision",
                source,
                f"{conflict_id}|{new_mapping_id}",
            )
            revision_event_id = _stable_id(
                "source-mapping-revision-event",
                source,
                f"{conflict_id}|{decision_id}",
            )
            evidence_payload = tuple(
                {
                    "evidence_id": _source_mapping_evidence_id(
                        mapping_id=new_mapping_id,
                        conflict_id=None,
                        evidence_ref=evidence_ref,
                        recorded_at=effective_at,
                        recorded_by=actor,
                        reason=reason,
                    ),
                    "evidence_ref": evidence_ref,
                }
                for evidence_ref in normalized_evidence
            )
            connection.execute(
                "INSERT INTO source_mapping_revision_events("
                "revision_event_id, decision_id, conflict_id, source, entity_type, source_id, "
                "expected_current_mapping_id, candidate_entity_id, new_mapping_id, "
                "new_version, effective_at, actor, reason, evidence_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    revision_event_id,
                    decision_id,
                    conflict_id,
                    source,
                    entity_type,
                    source_id,
                    expected_current_mapping_id,
                    candidate_entity_id.value,
                    new_mapping_id,
                    new_version,
                    _mapping_timestamp(effective_at),
                    actor,
                    reason,
                    json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            new_row = connection.execute(
                "SELECT * FROM source_mappings WHERE mapping_id = ?", (new_mapping_id,)
            ).fetchone()
            decision_row = connection.execute(
                "SELECT * FROM source_mapping_decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
            assert new_row is not None and decision_row is not None
            return (
                _source_mapping_from_row(new_row),
                _source_mapping_decision_from_row(decision_row),
            )

    def list_source_mapping_conflicts(
        self,
        *,
        source: str | None = None,
        entity_type: str | None = None,
        source_id: str | None = None,
        include_resolved: bool = False,
    ) -> tuple[SourceMappingConflict, ...]:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("conflict.source", source),
            ("conflict.entity_type", entity_type),
            ("conflict.source_id", source_id),
        ):
            if value is not None:
                _require_text(value, column.removeprefix("conflict."))
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if entity_type is not None and entity_type not in {
            "competition",
            "season",
            "team",
            "player",
            "match",
        }:
            raise ValueError(f"unsupported source mapping entity_type {entity_type!r}")
        if not include_resolved:
            clauses.extend(("decision.decision_id IS NULL", "obsolete.obsoletion_id IS NULL"))
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT conflict.*, COALESCE(decision.decision_id, obsolete.decision_id) "
                "AS decision_id, "
                "current.entity_id AS current_entity_id "
                "FROM source_mapping_conflicts AS conflict "
                "LEFT JOIN source_mappings AS current "
                "ON current.mapping_id = conflict.current_mapping_id "
                "LEFT JOIN source_mapping_decisions AS decision "
                "ON decision.conflict_id = conflict.conflict_id "
                "LEFT JOIN source_mapping_conflict_obsoletions AS obsolete "
                "ON obsolete.conflict_id = conflict.conflict_id "
                f"{where} ORDER BY conflict.proposed_at, conflict.conflict_id",
                tuple(parameters),
            ).fetchall()
        return tuple(_source_mapping_conflict_from_row(row) for row in rows)

    def source_mapping_evidence(
        self,
        *,
        mapping_id: str | None = None,
        conflict_id: str | None = None,
    ) -> tuple[SourceMappingEvidence, ...]:
        if (mapping_id is None) == (conflict_id is None):
            raise ValueError("select evidence by exactly one mapping_id or conflict_id")
        column, value = (
            ("mapping_id", mapping_id) if mapping_id is not None else ("conflict_id", conflict_id)
        )
        assert value is not None
        _require_text(value, column)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM source_mapping_evidence WHERE {column} = ? "
                "ORDER BY recorded_at, evidence_id",
                (value,),
            ).fetchall()
        return tuple(_source_mapping_evidence_from_row(row) for row in rows)

    def mapped_match_ids(
        self,
        *,
        source: str,
        source_ids: Sequence[str],
        as_of: datetime | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> dict[str, MatchId]:
        """Resolve persisted provider mappings for a set of match identifiers."""

        if not source_ids:
            return {}
        if as_of is not None:
            require_utc(as_of, "as_of")
        placeholders = ", ".join("?" for _ in source_ids)
        validity = "valid_to IS NULL"
        parameters: tuple[object, ...] = (source, *source_ids)
        if as_of is not None:
            validity = "valid_from <= ? AND (valid_to IS NULL OR ? < valid_to)"
            timestamp = _mapping_timestamp(as_of)
            parameters = (source, *source_ids, timestamp, timestamp)
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            rows = connection.execute(
                "SELECT source_id, entity_id FROM source_mappings "
                f"WHERE source = ? AND entity_type = 'match' AND source_id IN ({placeholders}) "
                f"AND {validity}",
                parameters,
            ).fetchall()
        return {str(row["source_id"]): MatchId(row["entity_id"]) for row in rows}

    def mapped_match_source_ids(
        self,
        *,
        source: str,
        season_id: SeasonId,
    ) -> dict[str, MatchId]:
        """Return current provider match mappings for one persisted season.

        Coverage validators must be able to inspect the canonical fixture catalog even when the
        latest source page is partial or unavailable.  This query deliberately returns only
        current, season-scoped mappings; it does not infer identities from display names or from
        collection-attempt URLs.
        """

        with self.connect() as connection:
            _require_entity(connection, season_id, "season")
            rows = connection.execute(
                "SELECT mapping.source_id, mapping.entity_id "
                "FROM source_mappings AS mapping "
                "JOIN matches AS match ON match.match_id = mapping.entity_id "
                "WHERE mapping.source = ? AND mapping.entity_type = 'match' "
                "AND mapping.valid_to IS NULL AND match.season_id = ? "
                "ORDER BY mapping.source_id",
                (source, season_id.value),
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
            source_id=source_id,
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
                    "source_id": source_id,
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

    def record_match_report_contract(
        self,
        *,
        collection_attempt_id: CollectionAttemptId,
        match_id: MatchId,
        match_version: int,
        raw_asset_id: RawAssetId,
        source_match_id: str,
        parser_version: str,
        required_tables: Sequence[str],
        team_tables: dict[str, Sequence[str]],
        observed_at: datetime,
    ) -> MatchReportContractEvidence:
        """Persist immutable parser-contract evidence for one successful report attempt."""

        normalized_required = tuple(dict.fromkeys(required_tables))
        normalized_team_tables = tuple(
            sorted(
                (
                    team_id,
                    tuple(dict.fromkeys(tables)),
                )
                for team_id, tables in team_tables.items()
            )
        )
        contract_id = _match_report_contract_id(
            collection_attempt_id=collection_attempt_id,
            match_id=match_id,
            match_version=match_version,
            raw_asset_id=raw_asset_id,
            source_match_id=source_match_id,
            parser_version=parser_version,
            required_tables=normalized_required,
            team_tables=normalized_team_tables,
            observed_at=observed_at,
        )
        evidence = MatchReportContractEvidence(
            contract_id=contract_id,
            collection_attempt_id=collection_attempt_id,
            match_id=match_id,
            match_version=match_version,
            raw_asset_id=raw_asset_id,
            source_match_id=source_match_id,
            parser_version=parser_version,
            required_tables=normalized_required,
            team_tables=normalized_team_tables,
            observed_at=observed_at,
        )
        with self.connect() as connection:
            _validate_match_report_contract_lineage(connection, evidence)
            _insert_exact(
                connection,
                "match_report_contracts",
                {
                    "contract_id": evidence.contract_id,
                    "collection_attempt_id": evidence.collection_attempt_id.value,
                    "match_id": evidence.match_id.value,
                    "match_version": evidence.match_version,
                    "raw_asset_id": evidence.raw_asset_id.value,
                    "source_match_id": evidence.source_match_id,
                    "parser_version": evidence.parser_version,
                    "required_tables_json": _json_text(evidence.required_tables),
                    "team_tables_json": _json_text(
                        {team_id: tables for team_id, tables in evidence.team_tables}
                    ),
                    "observed_at": _timestamp(evidence.observed_at),
                },
                "contract_id",
            )
        return evidence

    def match_report_contract_audit(
        self,
        season_id: SeasonId,
    ) -> tuple[tuple[MatchReportContractEvidence, ...], tuple[str, ...]]:
        """Load valid report contracts and retain tamper diagnostics per persisted row."""

        with self.connect() as connection:
            _require_entity(connection, season_id, "season")
            rows = connection.execute(
                "SELECT contract.* FROM match_report_contracts AS contract "
                "JOIN matches AS match ON match.match_id = contract.match_id "
                "WHERE match.season_id = ? ORDER BY contract.observed_at, contract.contract_id",
                (season_id.value,),
            ).fetchall()
            valid: list[MatchReportContractEvidence] = []
            diagnostics: list[str] = []
            for row in rows:
                contract_id = str(row["contract_id"])
                try:
                    evidence = _match_report_contract_from_row(row)
                    _validate_match_report_contract_lineage(connection, evidence)
                except (KeyError, TypeError, ValueError, CanonicalConflictError):
                    diagnostics.append(f"match_report_contract_invalid:{contract_id}")
                    continue
                valid.append(evidence)
        return tuple(valid), tuple(diagnostics)

    def match_report_contract(
        self,
        contract_id: str,
        *,
        _connection: sqlite3.Connection | None = None,
    ) -> MatchReportContractEvidence:
        """Load and lineage-verify one persisted match-report parser contract."""

        _require_text(contract_id, "contract_id")
        with self.connect() if _connection is None else nullcontext(_connection) as connection:
            row = connection.execute(
                "SELECT * FROM match_report_contracts WHERE contract_id = ?",
                (contract_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"match report contract {contract_id!r} does not exist")
            evidence = _match_report_contract_from_row(row)
            _validate_match_report_contract_lineage(connection, evidence)
        return evidence

    def official_lineup_contract(
        self,
        contract_id: str,
    ) -> OfficialLineupContractEvidence:
        """Load and content-verify one persisted official-lineup parser contract."""

        _require_text(contract_id, "contract_id")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM official_lineup_contracts WHERE contract_id = ?",
                (contract_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"official lineup contract {contract_id!r} does not exist")
        return _official_lineup_contract_from_row(row)

    def match_ids_for_season(self, season_id: SeasonId) -> tuple[MatchId, ...]:
        """Return persisted canonical fixture IDs for one registered season."""

        with self.connect() as connection:
            _require_entity(connection, season_id, "season")
            rows = connection.execute(
                "SELECT match_id FROM matches WHERE season_id = ? ORDER BY match_id",
                (season_id.value,),
            ).fetchall()
        return tuple(MatchId(row["match_id"]) for row in rows)

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
        created_by: str = "canonical-store",
    ) -> None:
        _require_text(source, "source")
        _require_text(source_id, "source_id")
        _require_text(created_by, "created_by")
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
            "INSERT INTO source_mappings(mapping_id, source, entity_type, source_id, "
            "entity_id, version, valid_from, valid_to, match_rule, confidence, created_at, "
            "created_by, audit_note, supersedes_mapping_id) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, NULL, ?, ?, ?, ?, ?, NULL)",
            (
                _source_mapping_id(source, entity_type, source_id, 1),
                source,
                entity_type,
                source_id,
                entity_id.value,
                _mapping_timestamp(valid_from),
                match_rule.value,
                confidence,
                _mapping_timestamp(created_at),
                created_by,
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


def _execute_sql_script(connection: sqlite3.Connection, script: str) -> None:
    """Execute schema DDL without the implicit commit performed by executescript()."""

    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                connection.execute(statement)
            pending = ""
    if pending.strip():
        raise RuntimeError("canonical schema contains an incomplete SQL statement")


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(match_versions)").fetchall()
    }
    if "round_name" not in columns:
        connection.execute("ALTER TABLE match_versions ADD COLUMN round_name TEXT")
    connection.execute("UPDATE schema_meta SET version = 2 WHERE singleton = 1")


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(collection_attempts)").fetchall()
    }
    if "source_id" not in columns:
        connection.execute("ALTER TABLE collection_attempts ADD COLUMN source_id TEXT")
    connection.execute(
        "UPDATE collection_attempts SET source_id = ("
        "SELECT raw.source_id FROM raw_assets AS raw "
        "WHERE raw.raw_asset_id = collection_attempts.raw_asset_id"
        ") WHERE source_id IS NULL AND raw_asset_id IS NOT NULL"
    )
    connection.execute("UPDATE schema_meta SET version = 3 WHERE singleton = 1")


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(prematch_events)").fetchall()
    }
    match_version_value = "match_version" if "match_version" in columns else "NULL"
    unbound_player = (
        "player_id IS NOT NULL AND match_version IS NULL"
        if "match_version" in columns
        else "player_id IS NOT NULL"
    )
    connection.execute("DROP TABLE IF EXISTS prematch_events_v4")
    connection.execute(_prematch_events_table_sql("prematch_events_v4", if_not_exists=False))
    connection.execute(
        "INSERT INTO prematch_events_v4(record_id, match_id, match_version, team_id, "
        "player_id, event_type, occurred_at, known_at, confirmation_status, "
        "evidence_refs_json, can_modify_features) "
        f"SELECT record_id, match_id, {match_version_value}, team_id, player_id, event_type, "
        "occurred_at, known_at, confirmation_status, evidence_refs_json, "
        f"CASE WHEN {unbound_player} THEN 0 ELSE can_modify_features END "
        "FROM prematch_events"
    )
    connection.execute("DROP TABLE prematch_events")
    connection.execute("ALTER TABLE prematch_events_v4 RENAME TO prematch_events")
    connection.execute("UPDATE schema_meta SET version = 4 WHERE singleton = 1")


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE schema_meta SET version = 5 WHERE singleton = 1")


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(official_lineup_contracts)").fetchall()
    }
    contract_version_value = "contract_version" if "contract_version" in columns else "1"
    source_bindings_value = "source_bindings_json" if "source_bindings_json" in columns else "NULL"
    connection.execute("DROP TABLE IF EXISTS official_lineup_contracts_v6")
    connection.execute(
        """
        CREATE TABLE official_lineup_contracts_v6 (
            contract_id TEXT PRIMARY KEY,
            contract_version INTEGER NOT NULL CHECK (contract_version IN (1, 2)),
            raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
            source TEXT NOT NULL,
            source_match_id TEXT NOT NULL,
            match_mapping_source TEXT NOT NULL,
            team_mapping_source TEXT NOT NULL,
            player_mapping_source TEXT NOT NULL,
            match_id TEXT NOT NULL,
            match_version INTEGER NOT NULL,
            parser_version TEXT NOT NULL,
            published_at TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            team_lineups_json TEXT NOT NULL,
            source_bindings_json TEXT,
            fact_ids_json TEXT NOT NULL,
            CHECK (
                (contract_version = 1 AND source_bindings_json IS NULL)
                OR (contract_version = 2 AND source_bindings_json IS NOT NULL)
            ),
            UNIQUE (raw_asset_id, parser_version, contract_version),
            FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version)
        )
        """
    )
    connection.execute(
        "INSERT INTO official_lineup_contracts_v6("
        "contract_id, contract_version, raw_asset_id, source, source_match_id, "
        "match_mapping_source, team_mapping_source, player_mapping_source, match_id, "
        "match_version, parser_version, published_at, observed_at, team_lineups_json, "
        "source_bindings_json, fact_ids_json) "
        f"SELECT contract_id, {contract_version_value}, raw_asset_id, source, source_match_id, "
        "match_mapping_source, team_mapping_source, player_mapping_source, match_id, "
        "match_version, parser_version, published_at, observed_at, team_lineups_json, "
        f"{source_bindings_value}, fact_ids_json FROM official_lineup_contracts"
    )
    connection.execute("DROP TABLE official_lineup_contracts")
    connection.execute(
        "ALTER TABLE official_lineup_contracts_v6 RENAME TO official_lineup_contracts"
    )
    connection.execute(
        "CREATE INDEX official_lineup_contracts_match "
        "ON official_lineup_contracts(match_id, match_version)"
    )
    connection.execute("UPDATE schema_meta SET version = 6 WHERE singleton = 1")


def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(source_mappings)").fetchall()
    }
    if "mapping_id" in columns:
        _validate_existing_source_mappings_v7(connection)
        connection.execute("UPDATE schema_meta SET version = 7 WHERE singleton = 1")
        return
    rows = connection.execute(
        "SELECT source, entity_type, source_id, entity_id, valid_from, valid_to, "
        "match_rule, confidence, created_at, audit_note FROM source_mappings "
        "ORDER BY source, entity_type, source_id, valid_from"
    ).fetchall()
    grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for row in rows:
        source = str(row["source"])
        entity_type = str(row["entity_type"])
        source_id = str(row["source_id"])
        audit_note = str(row["audit_note"])
        _require_text(source, "source")
        _require_text(source_id, "source_id")
        _require_text(audit_note, "audit_note")
        if entity_type not in {"competition", "season", "team", "player", "match"}:
            raise CanonicalConflictError(
                f"cannot migrate source mapping with invalid entity type {entity_type!r}"
            )
        entity = connection.execute(
            "SELECT entity_type FROM entities WHERE entity_id = ?", (row["entity_id"],)
        ).fetchone()
        if entity is None or entity["entity_type"] != entity_type:
            raise CanonicalConflictError(
                f"cannot migrate {source}:{entity_type}:{source_id}: entity type mismatch"
            )
        if str(row["match_rule"]) not in {rule.value for rule in MappingRule}:
            raise CanonicalConflictError(
                f"cannot migrate {source}:{entity_type}:{source_id}: invalid match rule"
            )
        confidence = float(row["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise CanonicalConflictError(
                f"cannot migrate {source}:{entity_type}:{source_id}: invalid confidence"
            )
        valid_from = _parse_timestamp(str(row["valid_from"]))
        _parse_timestamp(str(row["created_at"]))
        if row["valid_to"] is not None:
            valid_to = _parse_timestamp(str(row["valid_to"]))
            if valid_to <= valid_from:
                raise CanonicalConflictError(
                    f"cannot migrate {source}:{entity_type}:{source_id}: invalid validity range"
                )
        grouped.setdefault((source, entity_type, source_id), []).append(row)

    connection.execute("DROP TABLE IF EXISTS source_mappings_v7")
    connection.execute(_source_mappings_v7_table_sql("source_mappings_v7"))
    for (source, entity_type, source_id), history in grouped.items():
        history.sort(key=lambda item: _parse_timestamp(str(item["valid_from"])))
        previous_valid_to: datetime | None = None
        previous_mapping_id: str | None = None
        for index, row in enumerate(history, start=1):
            valid_from = _parse_timestamp(str(row["valid_from"]))
            valid_to = None if row["valid_to"] is None else _parse_timestamp(str(row["valid_to"]))
            if index > 1 and previous_valid_to != valid_from:
                raise CanonicalConflictError(
                    f"cannot migrate {source}:{entity_type}:{source_id}: "
                    "validity intervals are not contiguous"
                )
            mapping_id = _source_mapping_id(source, entity_type, source_id, index)
            connection.execute(
                "INSERT INTO source_mappings_v7("
                "mapping_id, source, entity_type, source_id, entity_id, version, valid_from, "
                "valid_to, match_rule, confidence, created_at, created_by, audit_note, "
                "supersedes_mapping_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mapping_id,
                    source,
                    entity_type,
                    source_id,
                    row["entity_id"],
                    index,
                    _mapping_timestamp(valid_from),
                    None if valid_to is None else _mapping_timestamp(valid_to),
                    row["match_rule"],
                    float(row["confidence"]),
                    _mapping_timestamp(_parse_timestamp(str(row["created_at"]))),
                    "legacy-v6-migration",
                    row["audit_note"],
                    previous_mapping_id,
                ),
            )
            previous_valid_to = valid_to
            previous_mapping_id = mapping_id

    connection.execute("DROP TABLE source_mappings")
    connection.execute("ALTER TABLE source_mappings_v7 RENAME TO source_mappings")
    connection.execute(
        "CREATE UNIQUE INDEX source_mappings_current "
        "ON source_mappings(source, entity_type, source_id) WHERE valid_to IS NULL"
    )
    _create_source_mapping_v7_auxiliary(connection)
    connection.execute("UPDATE schema_meta SET version = 7 WHERE singleton = 1")


def _validate_existing_source_mappings_v7(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT * FROM source_mappings ORDER BY source, entity_type, source_id, version"
    ).fetchall()
    previous_by_key: dict[tuple[str, str, str], SourceMapping] = {}
    for row in rows:
        mapping = _source_mapping_from_row(row)
        assert mapping.mapping_id is not None
        assert mapping.entity_type is not None
        key = (mapping.source, mapping.entity_type, mapping.source_id)
        previous = previous_by_key.get(key)
        expected_version = 1 if previous is None else previous.version + 1
        expected_supersedes = None if previous is None else previous.mapping_id
        if (
            mapping.version != expected_version
            or mapping.mapping_id != _source_mapping_id(*key, mapping.version)
            or mapping.supersedes_mapping_id != expected_supersedes
            or (previous is not None and previous.valid_to != mapping.valid_from)
        ):
            raise CanonicalConflictError(
                f"cannot accept malformed v7 source mapping history for {':'.join(key)}"
            )
        entity = connection.execute(
            "SELECT entity_type FROM entities WHERE entity_id = ?",
            (mapping.entity_id.value,),
        ).fetchone()
        if entity is None or entity["entity_type"] != mapping.entity_type:
            raise CanonicalConflictError(f"cannot accept {':'.join(key)}: entity type mismatch")
        previous_by_key[key] = mapping


def _source_mappings_v7_table_sql(table_name: str) -> str:
    if table_name not in {"source_mappings", "source_mappings_v7"}:
        raise ValueError("unsupported source mappings table name")
    return f"""
        CREATE TABLE {table_name} (
            mapping_id TEXT PRIMARY KEY CHECK (
                {_canonical_uuid_id_sql("mapping_id", "source-mapping")}
            ),
            source TEXT NOT NULL CHECK (
                length(trim(source)) > 0 AND source = trim(source)
            ),
            entity_type TEXT NOT NULL CHECK (
                entity_type IN ('competition', 'season', 'team', 'player', 'match')
            ),
            source_id TEXT NOT NULL CHECK (
                length(trim(source_id)) > 0 AND source_id = trim(source_id)
            ),
            entity_id TEXT NOT NULL REFERENCES entities(entity_id),
            version INTEGER NOT NULL CHECK (version > 0),
            valid_from TEXT NOT NULL CHECK (
                length(valid_from) = 27
                AND substr(valid_from, 11, 1) = 'T'
                AND substr(valid_from, 20, 1) = '.'
                AND substr(valid_from, 27, 1) = 'Z'
                AND julianday(valid_from) IS NOT NULL
            ),
            valid_to TEXT,
            match_rule TEXT NOT NULL CHECK (
                match_rule IN ('source_id', 'exact_alias', 'fuzzy_alias', 'manual_override')
            ),
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            created_at TEXT NOT NULL CHECK (
                length(created_at) = 27
                AND substr(created_at, 11, 1) = 'T'
                AND substr(created_at, 20, 1) = '.'
                AND substr(created_at, 27, 1) = 'Z'
                AND julianday(created_at) IS NOT NULL
            ),
            created_by TEXT NOT NULL CHECK (
                length(trim(created_by)) > 0 AND created_by = trim(created_by)
            ),
            audit_note TEXT NOT NULL CHECK (
                length(trim(audit_note)) > 0 AND audit_note = trim(audit_note)
            ),
            supersedes_mapping_id TEXT REFERENCES {table_name}(mapping_id),
            CHECK (
                valid_to IS NULL OR (
                    length(valid_to) = 27
                    AND substr(valid_to, 11, 1) = 'T'
                    AND substr(valid_to, 20, 1) = '.'
                    AND substr(valid_to, 27, 1) = 'Z'
                    AND julianday(valid_to) IS NOT NULL
                    AND valid_to > valid_from
                )
            ),
            CHECK (
                (version = 1 AND supersedes_mapping_id IS NULL)
                OR (version > 1 AND supersedes_mapping_id IS NOT NULL)
            ),
            UNIQUE (source, entity_type, source_id, version)
        )
    """


def _create_source_mapping_v7_auxiliary(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS source_mappings_current "
        "ON source_mappings(source, entity_type, source_id) WHERE valid_to IS NULL"
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS source_mapping_conflicts (
            conflict_id TEXT PRIMARY KEY CHECK (
                {_canonical_uuid_id_sql("conflict_id", "source-mapping-conflict")}
            ),
            source TEXT NOT NULL CHECK (length(trim(source)) > 0 AND source = trim(source)),
            entity_type TEXT NOT NULL CHECK (
                entity_type IN ('competition', 'season', 'team', 'player', 'match')
            ),
            source_id TEXT NOT NULL CHECK (
                length(trim(source_id)) > 0 AND source_id = trim(source_id)
            ),
            current_mapping_id TEXT REFERENCES source_mappings(mapping_id),
            candidate_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
            proposed_at TEXT NOT NULL CHECK (
                length(proposed_at) = 27 AND substr(proposed_at, 27, 1) = 'Z'
                AND julianday(proposed_at) IS NOT NULL
            ),
            proposed_by TEXT NOT NULL CHECK (
                length(trim(proposed_by)) > 0 AND proposed_by = trim(proposed_by)
            ),
            reason TEXT NOT NULL CHECK (length(trim(reason)) > 0 AND reason = trim(reason)),
            UNIQUE (current_mapping_id, candidate_entity_id)
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS source_mapping_conflicts_unmapped_candidate "
        "ON source_mapping_conflicts(source, entity_type, source_id, candidate_entity_id) "
        "WHERE current_mapping_id IS NULL"
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS source_mapping_revision_events (
            revision_event_id TEXT PRIMARY KEY CHECK (
                {_canonical_uuid_id_sql("revision_event_id", "source-mapping-revision-event")}
            ),
            decision_id TEXT NOT NULL UNIQUE CHECK (
                {_canonical_uuid_id_sql("decision_id", "source-mapping-decision")}
            ),
            conflict_id TEXT NOT NULL UNIQUE REFERENCES source_mapping_conflicts(conflict_id),
            source TEXT NOT NULL CHECK (length(trim(source)) > 0 AND source = trim(source)),
            entity_type TEXT NOT NULL CHECK (
                entity_type IN ('competition', 'season', 'team', 'player', 'match')
            ),
            source_id TEXT NOT NULL CHECK (
                length(trim(source_id)) > 0 AND source_id = trim(source_id)
            ),
            expected_current_mapping_id TEXT REFERENCES source_mappings(mapping_id),
            candidate_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
            new_mapping_id TEXT NOT NULL UNIQUE CHECK (
                {_canonical_uuid_id_sql("new_mapping_id", "source-mapping")}
            ),
            new_version INTEGER NOT NULL CHECK (new_version > 0),
            effective_at TEXT NOT NULL CHECK (
                length(effective_at) = 27 AND substr(effective_at, 27, 1) = 'Z'
                AND julianday(effective_at) IS NOT NULL
            ),
            actor TEXT NOT NULL CHECK (length(trim(actor)) > 0 AND actor = trim(actor)),
            reason TEXT NOT NULL CHECK (length(trim(reason)) > 0 AND reason = trim(reason)),
            evidence_json TEXT NOT NULL CHECK (
                json_valid(evidence_json) AND json_type(evidence_json) = 'array'
            )
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS source_mapping_evidence (
            evidence_id TEXT PRIMARY KEY CHECK (
                {_canonical_uuid_id_sql("evidence_id", "source-mapping-evidence")}
            ),
            mapping_id TEXT REFERENCES source_mappings(mapping_id),
            conflict_id TEXT REFERENCES source_mapping_conflicts(conflict_id),
            evidence_ref TEXT NOT NULL CHECK (
                length(trim(evidence_ref)) > 0 AND evidence_ref = trim(evidence_ref)
            ),
            recorded_at TEXT NOT NULL CHECK (
                length(recorded_at) = 27 AND substr(recorded_at, 27, 1) = 'Z'
                AND julianday(recorded_at) IS NOT NULL
            ),
            recorded_by TEXT NOT NULL CHECK (
                length(trim(recorded_by)) > 0 AND recorded_by = trim(recorded_by)
            ),
            reason TEXT NOT NULL CHECK (length(trim(reason)) > 0 AND reason = trim(reason)),
            CHECK ((mapping_id IS NULL) <> (conflict_id IS NULL))
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS source_mapping_decisions (
            decision_id TEXT PRIMARY KEY CHECK (
                {_canonical_uuid_id_sql("decision_id", "source-mapping-decision")}
            ),
            revision_event_id TEXT NOT NULL UNIQUE
                REFERENCES source_mapping_revision_events(revision_event_id),
            conflict_id TEXT NOT NULL UNIQUE
                REFERENCES source_mapping_conflicts(conflict_id),
            previous_mapping_id TEXT REFERENCES source_mappings(mapping_id),
            new_mapping_id TEXT NOT NULL UNIQUE REFERENCES source_mappings(mapping_id),
            decided_at TEXT NOT NULL CHECK (
                length(decided_at) = 27 AND substr(decided_at, 27, 1) = 'Z'
                AND julianday(decided_at) IS NOT NULL
            ),
            decided_by TEXT NOT NULL CHECK (
                length(trim(decided_by)) > 0 AND decided_by = trim(decided_by)
            ),
            reason TEXT NOT NULL CHECK (length(trim(reason)) > 0 AND reason = trim(reason)),
            evidence_ids_json TEXT NOT NULL CHECK (
                json_valid(evidence_ids_json) AND json_type(evidence_ids_json) = 'array'
                AND json_array_length(evidence_ids_json) > 0
            )
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS source_mapping_conflict_obsoletions (
            obsoletion_id TEXT PRIMARY KEY,
            conflict_id TEXT NOT NULL UNIQUE REFERENCES source_mapping_conflicts(conflict_id),
            revision_event_id TEXT NOT NULL REFERENCES source_mapping_revision_events(
                revision_event_id
            ),
            decision_id TEXT NOT NULL REFERENCES source_mapping_decisions(decision_id),
            obsoleted_at TEXT NOT NULL,
            reason TEXT NOT NULL CHECK (length(trim(reason)) > 0 AND reason = trim(reason))
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS source_mapping_conflicts_key "
        "ON source_mapping_conflicts(source, entity_type, source_id, proposed_at)"
    )
    _create_source_mapping_v7_triggers(connection)


def _create_source_mapping_v7_triggers(connection: sqlite3.Connection) -> None:
    trigger_names = (
        "source_mappings_id_insert",
        "source_mappings_entity_type_insert",
        "source_mappings_revision_insert",
        "source_mappings_history_insert",
        "source_mappings_overlap_insert",
        "source_mappings_close_only_update",
        "source_mappings_no_delete",
        "source_mapping_conflicts_id_insert",
        "source_mapping_conflicts_validate_insert",
        "source_mapping_revision_events_id_insert",
        "source_mapping_revision_events_require_evidence",
        "source_mapping_revision_events_validate_evidence",
        "source_mapping_revision_events_validate_insert",
        "source_mapping_revision_events_apply_insert",
        "source_mapping_evidence_id_insert",
        "source_mapping_decisions_id_insert",
        "source_mapping_decisions_validate_insert",
        "source_mapping_conflict_obsoletions_validate_insert",
    )
    append_only_tables = (
        "source_mapping_evidence",
        "source_mapping_conflicts",
        "source_mapping_revision_events",
        "source_mapping_decisions",
        "source_mapping_conflict_obsoletions",
    )
    for trigger_name in trigger_names:
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    for table in append_only_tables:
        connection.execute(f"DROP TRIGGER IF EXISTS {table}_no_update")
        connection.execute(f"DROP TRIGGER IF EXISTS {table}_no_delete")

    statements = (
        f"""
        CREATE TRIGGER source_mappings_id_insert
        BEFORE INSERT ON source_mappings
        WHEN NOT ({_canonical_uuid_id_sql("NEW.mapping_id", "source-mapping")})
            OR NEW.mapping_id <> source_mapping_id(
                NEW.source, NEW.entity_type, NEW.source_id, NEW.version
            )
            OR (
                NEW.supersedes_mapping_id IS NOT NULL
                AND NOT ({_canonical_uuid_id_sql("NEW.supersedes_mapping_id", "source-mapping")})
            )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping ID is not its canonical deterministic identity');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mappings_entity_type_insert
        BEFORE INSERT ON source_mappings
        WHEN NOT EXISTS (
            SELECT 1 FROM entities
            WHERE entity_id = NEW.entity_id AND entity_type = NEW.entity_type
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping entity type mismatch');
        END
        """,
        """
        CREATE TRIGGER source_mappings_revision_insert
        BEFORE INSERT ON source_mappings
        WHEN (NEW.version > 1 OR NEW.match_rule = 'manual_override') AND NOT EXISTS (
            SELECT 1 FROM source_mapping_revision_events AS event
            WHERE event.new_mapping_id = NEW.mapping_id
                AND event.source = NEW.source
                AND event.entity_type = NEW.entity_type
                AND event.source_id = NEW.source_id
                AND event.expected_current_mapping_id IS NEW.supersedes_mapping_id
                AND event.candidate_entity_id = NEW.entity_id
                AND event.new_version = NEW.version
                AND event.effective_at = NEW.valid_from
                AND event.actor = NEW.created_by
                AND event.reason = NEW.audit_note
                AND NEW.valid_to IS NULL
                AND NEW.match_rule = 'manual_override'
                AND NEW.confidence = 1.0
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping revision requires a matching revision event');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mappings_history_insert
        BEFORE INSERT ON source_mappings
        WHEN (
            NEW.version = 1 AND EXISTS (
                SELECT 1 FROM source_mappings
                WHERE source = NEW.source AND entity_type = NEW.entity_type
                    AND source_id = NEW.source_id
            )
        ) OR (
            NEW.version > 1 AND NOT EXISTS (
                SELECT 1 FROM source_mappings AS previous
                WHERE previous.mapping_id = NEW.supersedes_mapping_id
                    AND previous.source = NEW.source
                    AND previous.entity_type = NEW.entity_type
                    AND previous.source_id = NEW.source_id
                    AND previous.version = NEW.version - 1
                    AND previous.valid_to = NEW.valid_from
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping version history is not continuous');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mappings_overlap_insert
        BEFORE INSERT ON source_mappings
        WHEN EXISTS (
            SELECT 1 FROM source_mappings AS existing
            WHERE existing.source = NEW.source
                AND existing.entity_type = NEW.entity_type
                AND existing.source_id = NEW.source_id
                AND existing.valid_from < COALESCE(NEW.valid_to, '9999-12-31T23:59:59Z')
                AND NEW.valid_from < COALESCE(existing.valid_to, '9999-12-31T23:59:59Z')
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping validity intervals overlap');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mappings_close_only_update
        BEFORE UPDATE ON source_mappings
        WHEN NOT (
            OLD.valid_to IS NULL AND NEW.valid_to IS NOT NULL
            AND OLD.mapping_id IS NEW.mapping_id
            AND OLD.source IS NEW.source
            AND OLD.entity_type IS NEW.entity_type
            AND OLD.source_id IS NEW.source_id
            AND OLD.entity_id IS NEW.entity_id
            AND OLD.version IS NEW.version
            AND OLD.valid_from IS NEW.valid_from
            AND OLD.match_rule IS NEW.match_rule
            AND OLD.confidence IS NEW.confidence
            AND OLD.created_at IS NEW.created_at
            AND OLD.created_by IS NEW.created_by
            AND OLD.audit_note IS NEW.audit_note
            AND OLD.supersedes_mapping_id IS NEW.supersedes_mapping_id
            AND EXISTS (
                SELECT 1 FROM source_mapping_revision_events AS event
                WHERE event.expected_current_mapping_id = OLD.mapping_id
                    AND event.source = OLD.source
                    AND event.entity_type = OLD.entity_type
                    AND event.source_id = OLD.source_id
                    AND event.effective_at = NEW.valid_to
            )
        )
        BEGIN
            SELECT RAISE(
                ABORT,
                'source mappings may only close through a matching revision event'
            );
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mappings_no_delete
        BEFORE DELETE ON source_mappings
        BEGIN
            SELECT RAISE(ABORT, 'source mappings are append-only');
        END
        """,
        f"""
        CREATE TRIGGER source_mapping_conflicts_id_insert
        BEFORE INSERT ON source_mapping_conflicts
        WHEN NOT (
                {_canonical_uuid_id_sql("NEW.conflict_id", "source-mapping-conflict")}
            ) OR (
                NEW.current_mapping_id IS NOT NULL
                AND NOT ({_canonical_uuid_id_sql("NEW.current_mapping_id", "source-mapping")})
            )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping conflict ID format is not canonical');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mapping_conflicts_validate_insert
        BEFORE INSERT ON source_mapping_conflicts
        WHEN NOT EXISTS (
            SELECT 1 FROM entities AS candidate
            WHERE candidate.entity_id = NEW.candidate_entity_id
                AND candidate.entity_type = NEW.entity_type
                AND (
                    (
                        NEW.current_mapping_id IS NULL
                        AND NOT EXISTS (
                            SELECT 1 FROM source_mappings AS current
                            WHERE current.source = NEW.source
                                AND current.entity_type = NEW.entity_type
                                AND current.source_id = NEW.source_id
                                AND current.valid_to IS NULL
                        )
                    ) OR EXISTS (
                        SELECT 1 FROM source_mappings AS current
                        WHERE current.mapping_id = NEW.current_mapping_id
                            AND current.source = NEW.source
                            AND current.entity_type = NEW.entity_type
                            AND current.source_id = NEW.source_id
                            AND current.valid_to IS NULL
                            AND current.entity_id <> NEW.candidate_entity_id
                    )
                )
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping conflict candidate is invalid');
        END
        """,
        """
        CREATE TRIGGER source_mapping_revision_events_require_evidence
        BEFORE INSERT ON source_mapping_revision_events
        WHEN json_array_length(NEW.evidence_json) = 0
        BEGIN
            SELECT RAISE(ABORT, 'source mapping revision requires at least one evidence');
        END
        """,
        f"""
        CREATE TRIGGER source_mapping_revision_events_id_insert
        BEFORE INSERT ON source_mapping_revision_events
        WHEN NOT (
                {_canonical_uuid_id_sql("NEW.revision_event_id", "source-mapping-revision-event")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.decision_id", "source-mapping-decision")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.conflict_id", "source-mapping-conflict")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.new_mapping_id", "source-mapping")}
            ) OR (
                NEW.expected_current_mapping_id IS NOT NULL
                AND NOT (
                    {_canonical_uuid_id_sql("NEW.expected_current_mapping_id", "source-mapping")}
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping revision ID format is not canonical');
        END
        """,
        """
        CREATE TRIGGER source_mapping_revision_events_validate_evidence
        BEFORE INSERT ON source_mapping_revision_events
        WHEN json_array_length(NEW.evidence_json) > 0 AND (
            EXISTS (
                SELECT 1 FROM json_each(NEW.evidence_json) AS item
                WHERE json_type(item.value) <> 'object'
                    OR json_type(item.value, '$.evidence_id') <> 'text'
                    OR length(trim(json_extract(item.value, '$.evidence_id'))) = 0
                    OR json_extract(item.value, '$.evidence_id') <>
                        trim(json_extract(item.value, '$.evidence_id'))
                    OR json_type(item.value, '$.evidence_ref') <> 'text'
                    OR length(trim(json_extract(item.value, '$.evidence_ref'))) = 0
                    OR json_extract(item.value, '$.evidence_ref') <>
                        trim(json_extract(item.value, '$.evidence_ref'))
            )
            OR (SELECT COUNT(*) FROM json_each(NEW.evidence_json)) <>
                (
                    SELECT COUNT(DISTINCT json_extract(value, '$.evidence_id'))
                    FROM json_each(NEW.evidence_json)
                )
            OR (SELECT COUNT(*) FROM json_each(NEW.evidence_json)) <>
                (
                    SELECT COUNT(DISTINCT json_extract(value, '$.evidence_ref'))
                    FROM json_each(NEW.evidence_json)
                )
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping revision evidence payload is invalid');
        END
        """,
        """
        CREATE TRIGGER source_mapping_revision_events_validate_insert
        BEFORE INSERT ON source_mapping_revision_events
        WHEN NOT EXISTS (
            SELECT 1 FROM source_mapping_conflicts AS conflict
            JOIN entities AS candidate
                ON candidate.entity_id = NEW.candidate_entity_id
                AND candidate.entity_type = NEW.entity_type
            LEFT JOIN source_mappings AS current
                ON current.mapping_id = conflict.current_mapping_id
            LEFT JOIN source_mapping_decisions AS decision
                ON decision.conflict_id = conflict.conflict_id
            LEFT JOIN source_mapping_conflict_obsoletions AS obsolete
                ON obsolete.conflict_id = conflict.conflict_id
            WHERE conflict.conflict_id = NEW.conflict_id
                AND conflict.source = NEW.source
                AND conflict.entity_type = NEW.entity_type
                AND conflict.source_id = NEW.source_id
                AND conflict.current_mapping_id IS NEW.expected_current_mapping_id
                AND conflict.candidate_entity_id = NEW.candidate_entity_id
                AND decision.decision_id IS NULL
                AND obsolete.obsoletion_id IS NULL
                AND NEW.effective_at >= conflict.proposed_at
                AND (
                    (
                        NEW.expected_current_mapping_id IS NULL
                        AND current.mapping_id IS NULL
                        AND NEW.new_version = 1
                        AND NOT EXISTS (
                            SELECT 1 FROM source_mappings AS active
                            WHERE active.source = NEW.source
                                AND active.entity_type = NEW.entity_type
                                AND active.source_id = NEW.source_id
                                AND active.valid_to IS NULL
                        )
                    ) OR (
                        NEW.expected_current_mapping_id IS NOT NULL
                        AND current.mapping_id = NEW.expected_current_mapping_id
                        AND current.valid_to IS NULL
                        AND NEW.new_version = current.version + 1
                        AND NEW.effective_at > current.valid_from
                    )
                )
        )
        BEGIN
            SELECT RAISE(
                ABORT,
                'source mapping revision event failed CAS or reviewed conflict validation'
            );
        END
        """,
        """
        CREATE TRIGGER source_mapping_revision_events_apply_insert
        AFTER INSERT ON source_mapping_revision_events
        BEGIN
            UPDATE source_mappings
            SET valid_to = NEW.effective_at
            WHERE mapping_id = NEW.expected_current_mapping_id AND valid_to IS NULL;

            INSERT INTO source_mappings(
                mapping_id, source, entity_type, source_id, entity_id, version, valid_from,
                valid_to, match_rule, confidence, created_at, created_by, audit_note,
                supersedes_mapping_id
            ) VALUES (
                NEW.new_mapping_id, NEW.source, NEW.entity_type, NEW.source_id,
                NEW.candidate_entity_id, NEW.new_version, NEW.effective_at, NULL,
                'manual_override', 1.0, NEW.effective_at, NEW.actor, NEW.reason,
                NEW.expected_current_mapping_id
            );

            INSERT INTO source_mapping_evidence(
                evidence_id, mapping_id, conflict_id, evidence_ref, recorded_at,
                recorded_by, reason
            )
            SELECT
                json_extract(item.value, '$.evidence_id'), NEW.new_mapping_id, NULL,
                json_extract(item.value, '$.evidence_ref'), NEW.effective_at, NEW.actor,
                NEW.reason
            FROM json_each(NEW.evidence_json) AS item;

            INSERT INTO source_mapping_decisions(
                decision_id, revision_event_id, conflict_id, previous_mapping_id,
                new_mapping_id, decided_at, decided_by, reason, evidence_ids_json
            ) VALUES (
                NEW.decision_id, NEW.revision_event_id, NEW.conflict_id,
                NEW.expected_current_mapping_id, NEW.new_mapping_id, NEW.effective_at,
                NEW.actor, NEW.reason,
                (
                    SELECT json_group_array(json_extract(item.value, '$.evidence_id'))
                    FROM json_each(NEW.evidence_json) AS item
                )
            );

            INSERT INTO source_mapping_conflict_obsoletions(
                obsoletion_id, conflict_id, revision_event_id, decision_id,
                obsoleted_at, reason
            )
            SELECT
                'source-mapping-conflict-obsoletion:' || conflict.conflict_id || ':' ||
                    NEW.revision_event_id,
                conflict.conflict_id, NEW.revision_event_id, NEW.decision_id,
                NEW.effective_at, 'superseded by accepted sibling candidate'
            FROM source_mapping_conflicts AS conflict
            LEFT JOIN source_mapping_decisions AS decision
                ON decision.conflict_id = conflict.conflict_id
            LEFT JOIN source_mapping_conflict_obsoletions AS obsolete
                ON obsolete.conflict_id = conflict.conflict_id
            WHERE conflict.conflict_id <> NEW.conflict_id
                AND conflict.current_mapping_id IS NEW.expected_current_mapping_id
                AND conflict.source = NEW.source
                AND conflict.entity_type = NEW.entity_type
                AND conflict.source_id = NEW.source_id
                AND decision.decision_id IS NULL
                AND obsolete.obsoletion_id IS NULL;
        END
        """,
        f"""
        CREATE TRIGGER source_mapping_evidence_id_insert
        BEFORE INSERT ON source_mapping_evidence
        WHEN NOT (
                {_canonical_uuid_id_sql("NEW.evidence_id", "source-mapping-evidence")}
            ) OR (
                NEW.mapping_id IS NOT NULL
                AND NOT ({_canonical_uuid_id_sql("NEW.mapping_id", "source-mapping")})
            ) OR (
                NEW.conflict_id IS NOT NULL
                AND NOT (
                    {_canonical_uuid_id_sql("NEW.conflict_id", "source-mapping-conflict")}
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping evidence ID format is not canonical');
        END
        """,
        f"""
        CREATE TRIGGER source_mapping_decisions_id_insert
        BEFORE INSERT ON source_mapping_decisions
        WHEN NOT (
                {_canonical_uuid_id_sql("NEW.decision_id", "source-mapping-decision")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.revision_event_id", "source-mapping-revision-event")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.conflict_id", "source-mapping-conflict")}
            ) OR NOT (
                {_canonical_uuid_id_sql("NEW.new_mapping_id", "source-mapping")}
            ) OR (
                NEW.previous_mapping_id IS NOT NULL
                AND NOT (
                    {_canonical_uuid_id_sql("NEW.previous_mapping_id", "source-mapping")}
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping decision ID format is not canonical');
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS source_mapping_decisions_validate_insert
        BEFORE INSERT ON source_mapping_decisions
        WHEN NOT EXISTS (
            SELECT 1 FROM source_mapping_revision_events AS event
            JOIN source_mapping_conflicts AS conflict
                ON conflict.conflict_id = event.conflict_id
            LEFT JOIN source_mappings AS previous
                ON previous.mapping_id = event.expected_current_mapping_id
            JOIN source_mappings AS replacement
                ON replacement.mapping_id = event.new_mapping_id
            WHERE event.revision_event_id = NEW.revision_event_id
                AND event.decision_id = NEW.decision_id
                AND event.conflict_id = NEW.conflict_id
                AND event.expected_current_mapping_id IS NEW.previous_mapping_id
                AND event.new_mapping_id = NEW.new_mapping_id
                AND event.effective_at = NEW.decided_at
                AND event.actor = NEW.decided_by
                AND event.reason = NEW.reason
                AND conflict.current_mapping_id IS NEW.previous_mapping_id
                AND replacement.supersedes_mapping_id IS NEW.previous_mapping_id
                AND replacement.source = event.source
                AND replacement.entity_type = event.entity_type
                AND replacement.source_id = event.source_id
                AND replacement.entity_id = event.candidate_entity_id
                AND replacement.valid_from = event.effective_at
                AND (previous.mapping_id IS NULL OR previous.valid_to = replacement.valid_from)
                AND NEW.evidence_ids_json = (
                    SELECT json_group_array(json_extract(item.value, '$.evidence_id'))
                    FROM json_each(event.evidence_json) AS item
                )
                AND (SELECT COUNT(*) FROM source_mapping_evidence AS evidence
                    WHERE evidence.mapping_id = NEW.new_mapping_id) =
                    json_array_length(event.evidence_json)
                AND NOT EXISTS (
                    SELECT 1 FROM json_each(event.evidence_json) AS item
                    LEFT JOIN source_mapping_evidence AS evidence
                        ON evidence.evidence_id = json_extract(item.value, '$.evidence_id')
                    WHERE evidence.mapping_id <> NEW.new_mapping_id
                        OR evidence.evidence_ref <>
                            json_extract(item.value, '$.evidence_ref')
                        OR evidence.recorded_at <> event.effective_at
                        OR evidence.recorded_by <> event.actor
                        OR evidence.reason <> event.reason
                )
        )
        BEGIN
            SELECT RAISE(
                ABORT,
                'source mapping decision requires its exact revision event evidence'
            );
        END
        """,
        """
        CREATE TRIGGER source_mapping_conflict_obsoletions_validate_insert
        BEFORE INSERT ON source_mapping_conflict_obsoletions
        WHEN NOT EXISTS (
            SELECT 1 FROM source_mapping_revision_events AS event
            JOIN source_mapping_decisions AS decision
                ON decision.revision_event_id = event.revision_event_id
            JOIN source_mapping_conflicts AS obsolete
                ON obsolete.conflict_id = NEW.conflict_id
            WHERE event.revision_event_id = NEW.revision_event_id
                AND decision.decision_id = NEW.decision_id
                AND obsolete.conflict_id <> event.conflict_id
                AND obsolete.current_mapping_id IS event.expected_current_mapping_id
                AND obsolete.source = event.source
                AND obsolete.entity_type = event.entity_type
                AND obsolete.source_id = event.source_id
                AND NEW.obsoleted_at = event.effective_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'source mapping conflict obsoletion is invalid');
        END
        """,
    )
    for statement in statements:
        connection.execute(statement)
    for table in append_only_tables:
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{table} is append-only');
            END
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{table} is append-only');
            END
            """
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


def _mapping_as_of(
    connection: sqlite3.Connection,
    source: str,
    entity_type: str,
    source_id: str,
    as_of: datetime,
) -> str | None:
    timestamp = _mapping_timestamp(as_of)
    row = connection.execute(
        "SELECT entity_id FROM source_mappings WHERE source = ? AND entity_type = ? "
        "AND source_id = ? AND valid_from <= ? AND (valid_to IS NULL OR ? < valid_to)",
        (source, entity_type, source_id, timestamp, timestamp),
    ).fetchone()
    return None if row is None else str(row["entity_id"])


def _validate_mapping_key(source: str, entity_type: str, source_id: str) -> None:
    _require_text(source, "source")
    _require_text(source_id, "source_id")
    if entity_type not in {"competition", "season", "team", "player", "match"}:
        raise ValueError(f"unsupported source mapping entity_type {entity_type!r}")


def _entity_id_from_value(entity_type: str, value: str) -> EntityId:
    constructors = {
        "competition": CompetitionId,
        "season": SeasonId,
        "team": TeamId,
        "player": PlayerId,
        "match": MatchId,
    }
    try:
        constructor = constructors[entity_type]
    except KeyError as error:
        raise ValueError(f"unsupported source mapping entity_type {entity_type!r}") from error
    return constructor(value)


def _source_mapping_from_row(row: sqlite3.Row) -> SourceMapping:
    entity_type = str(row["entity_type"])
    return SourceMapping(
        source=str(row["source"]),
        source_id=str(row["source_id"]),
        entity_id=_entity_id_from_value(entity_type, str(row["entity_id"])),
        version=int(row["version"]),
        valid_from=_parse_timestamp(str(row["valid_from"])),
        valid_to=(None if row["valid_to"] is None else _parse_timestamp(str(row["valid_to"]))),
        match_rule=MappingRule(str(row["match_rule"])),
        confidence=float(row["confidence"]),
        created_at=_parse_timestamp(str(row["created_at"])),
        created_by=str(row["created_by"]),
        audit_note=str(row["audit_note"]),
        mapping_id=str(row["mapping_id"]),
        entity_type=entity_type,
        supersedes_mapping_id=(
            None if row["supersedes_mapping_id"] is None else str(row["supersedes_mapping_id"])
        ),
    )


def _source_mapping_conflict_from_row(row: sqlite3.Row) -> SourceMappingConflict:
    entity_type = str(row["entity_type"])
    return SourceMappingConflict(
        conflict_id=str(row["conflict_id"]),
        source=str(row["source"]),
        entity_type=entity_type,
        source_id=str(row["source_id"]),
        current_mapping_id=(
            None if row["current_mapping_id"] is None else str(row["current_mapping_id"])
        ),
        current_entity_id=(
            None
            if row["current_entity_id"] is None
            else _entity_id_from_value(entity_type, str(row["current_entity_id"]))
        ),
        candidate_entity_id=_entity_id_from_value(entity_type, str(row["candidate_entity_id"])),
        proposed_at=_parse_timestamp(str(row["proposed_at"])),
        proposed_by=str(row["proposed_by"]),
        reason=str(row["reason"]),
        decision_id=None if row["decision_id"] is None else str(row["decision_id"]),
    )


def _source_mapping_decision_from_row(row: sqlite3.Row) -> SourceMappingDecision:
    return SourceMappingDecision(
        decision_id=str(row["decision_id"]),
        conflict_id=str(row["conflict_id"]),
        previous_mapping_id=(
            None if row["previous_mapping_id"] is None else str(row["previous_mapping_id"])
        ),
        new_mapping_id=str(row["new_mapping_id"]),
        decided_at=_parse_timestamp(str(row["decided_at"])),
        decided_by=str(row["decided_by"]),
        reason=str(row["reason"]),
        revision_event_id=str(row["revision_event_id"]),
        evidence_ids=tuple(str(value) for value in json.loads(str(row["evidence_ids_json"]))),
    )


def _source_mapping_evidence_from_row(row: sqlite3.Row) -> SourceMappingEvidence:
    return SourceMappingEvidence(
        evidence_id=str(row["evidence_id"]),
        mapping_id=None if row["mapping_id"] is None else str(row["mapping_id"]),
        conflict_id=None if row["conflict_id"] is None else str(row["conflict_id"]),
        evidence_ref=str(row["evidence_ref"]),
        recorded_at=_parse_timestamp(str(row["recorded_at"])),
        recorded_by=str(row["recorded_by"]),
        reason=str(row["reason"]),
    )


def _normalize_mapping_evidence_refs(evidence_refs: Sequence[str]) -> tuple[str, ...]:
    if isinstance(evidence_refs, (str, bytes)):
        raise TypeError("evidence_refs must be a sequence of evidence identifiers")
    normalized: list[str] = []
    for evidence_ref in evidence_refs:
        if not isinstance(evidence_ref, str):
            raise TypeError("mapping evidence references must be strings")
        _require_text(evidence_ref, "evidence_ref")
        normalized.append(evidence_ref)
    if not normalized:
        raise ValueError("source mapping review requires at least one evidence reference")
    return tuple(sorted(set(normalized)))


def _insert_source_mapping_evidence(
    connection: sqlite3.Connection,
    *,
    mapping_id: str | None,
    conflict_id: str | None,
    evidence_ref: str,
    recorded_at: datetime,
    recorded_by: str,
    reason: str,
) -> None:
    evidence_id = _source_mapping_evidence_id(
        mapping_id=mapping_id,
        conflict_id=conflict_id,
        evidence_ref=evidence_ref,
        recorded_at=recorded_at,
        recorded_by=recorded_by,
        reason=reason,
    )
    _insert_exact(
        connection,
        "source_mapping_evidence",
        {
            "evidence_id": evidence_id,
            "mapping_id": mapping_id,
            "conflict_id": conflict_id,
            "evidence_ref": evidence_ref,
            "recorded_at": _mapping_timestamp(recorded_at),
            "recorded_by": recorded_by,
            "reason": reason,
        },
        "evidence_id",
    )


def _source_mapping_evidence_id(
    *,
    mapping_id: str | None,
    conflict_id: str | None,
    evidence_ref: str,
    recorded_at: datetime,
    recorded_by: str,
    reason: str,
) -> str:
    target = mapping_id if mapping_id is not None else conflict_id
    assert target is not None
    return _stable_id(
        "source-mapping-evidence",
        "canonical",
        "|".join((target, evidence_ref, _mapping_timestamp(recorded_at), recorded_by, reason)),
    )


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
        source_id=row["source_id"],
    )


def _match_report_contract_from_row(row: sqlite3.Row) -> MatchReportContractEvidence:
    required_value = json.loads(row["required_tables_json"])
    team_value = json.loads(row["team_tables_json"])
    if not isinstance(required_value, list) or not all(
        isinstance(item, str) for item in required_value
    ):
        raise ValueError("required report tables must be a string list")
    if not isinstance(team_value, dict):
        raise ValueError("team report tables must be an object")
    team_tables: list[tuple[str, tuple[str, ...]]] = []
    for team_id, tables in team_value.items():
        if (
            not isinstance(team_id, str)
            or not isinstance(tables, list)
            or not all(isinstance(item, str) for item in tables)
        ):
            raise ValueError("team report tables must map strings to string lists")
        team_tables.append((team_id, tuple(tables)))
    evidence = MatchReportContractEvidence(
        contract_id=str(row["contract_id"]),
        collection_attempt_id=CollectionAttemptId(row["collection_attempt_id"]),
        match_id=MatchId(row["match_id"]),
        match_version=int(row["match_version"]),
        raw_asset_id=RawAssetId(row["raw_asset_id"]),
        source_match_id=str(row["source_match_id"]),
        parser_version=str(row["parser_version"]),
        required_tables=tuple(required_value),
        team_tables=tuple(sorted(team_tables)),
        observed_at=_parse_timestamp(row["observed_at"]),
    )
    expected_id = _match_report_contract_id(
        collection_attempt_id=evidence.collection_attempt_id,
        match_id=evidence.match_id,
        match_version=evidence.match_version,
        raw_asset_id=evidence.raw_asset_id,
        source_match_id=evidence.source_match_id,
        parser_version=evidence.parser_version,
        required_tables=evidence.required_tables,
        team_tables=evidence.team_tables,
        observed_at=evidence.observed_at,
    )
    if evidence.contract_id != expected_id:
        raise CanonicalConflictError("match report contract content ID does not match payload")
    return evidence


def _validate_match_report_contract_lineage(
    connection: sqlite3.Connection,
    evidence: MatchReportContractEvidence,
) -> None:
    attempt = connection.execute(
        "SELECT match_id, source, source_id, outcome, observed_at, raw_asset_id "
        "FROM collection_attempts WHERE collection_attempt_id = ?",
        (evidence.collection_attempt_id.value,),
    ).fetchone()
    if attempt is None:
        raise KeyError(f"collection attempt {evidence.collection_attempt_id} does not exist")
    expected_attempt = (
        evidence.match_id.value,
        "fbref-match-report",
        evidence.source_match_id,
        CollectionAttemptOutcome.SUCCEEDED.value,
        _timestamp(evidence.observed_at),
        evidence.raw_asset_id.value,
    )
    actual_attempt = (
        attempt["match_id"],
        attempt["source"],
        attempt["source_id"],
        attempt["outcome"],
        attempt["observed_at"],
        attempt["raw_asset_id"],
    )
    if actual_attempt != expected_attempt:
        raise CanonicalConflictError("match report contract conflicts with collection attempt")

    version = connection.execute(
        "SELECT observed_at FROM match_versions WHERE match_id = ? AND version = ?",
        (evidence.match_id.value, evidence.match_version),
    ).fetchone()
    if version is None:
        raise KeyError(f"match version {evidence.match_id}:{evidence.match_version} does not exist")
    if _parse_timestamp(str(version["observed_at"])) > evidence.observed_at:
        raise CanonicalConflictError(
            "match report contract references a match version observed after the contract"
        )
    raw = connection.execute(
        "SELECT source, source_id, observed_at FROM raw_assets WHERE raw_asset_id = ?",
        (evidence.raw_asset_id.value,),
    ).fetchone()
    if raw is None:
        raise KeyError(f"raw asset {evidence.raw_asset_id} does not exist")
    if (raw["source"], raw["source_id"], raw["observed_at"]) != (
        "fbref",
        evidence.source_match_id,
        _timestamp(evidence.observed_at),
    ):
        raise CanonicalConflictError("match report contract conflicts with raw evidence")

    match = _load_match(connection, evidence.match_id)
    resolved_team_ids = set()
    for source_team_id, _ in evidence.team_tables:
        mapped = _mapping_as_of(
            connection,
            "fbref",
            "team",
            source_team_id,
            evidence.observed_at,
        )
        if mapped is None:
            raise CanonicalConflictError("match report contract team mapping is missing")
        resolved_team_ids.add(mapped)
    if resolved_team_ids != {match.home_team_id.value, match.away_team_id.value}:
        raise CanonicalConflictError("match report contract teams do not match fixture")


def _match_report_contract_id(
    *,
    collection_attempt_id: CollectionAttemptId,
    match_id: MatchId,
    match_version: int,
    raw_asset_id: RawAssetId,
    source_match_id: str,
    parser_version: str,
    required_tables: tuple[str, ...],
    team_tables: tuple[tuple[str, tuple[str, ...]], ...],
    observed_at: datetime,
) -> str:
    payload = {
        "collection_attempt_id": collection_attempt_id.value,
        "match_id": match_id.value,
        "match_version": match_version,
        "raw_asset_id": raw_asset_id.value,
        "source_match_id": source_match_id,
        "parser_version": parser_version,
        "required_tables": required_tables,
        "team_tables": {team_id: tables for team_id, tables in team_tables},
        "observed_at": _timestamp(observed_at),
    }
    digest = hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()
    return f"match-report-contract:{digest}"


def build_official_lineup_contract(
    *,
    raw_asset_id: RawAssetId,
    source: str,
    source_match_id: str,
    match_mapping_source: str,
    team_mapping_source: str,
    player_mapping_source: str,
    match_id: MatchId,
    match_version: int,
    parser_version: str,
    published_at: datetime,
    observed_at: datetime,
    team_lineups: Sequence[tuple[TeamId, Sequence[PlayerId]]],
    source_bindings: Sequence[OfficialLineupSourceBinding],
    fact_ids: Sequence[str],
) -> OfficialLineupContractEvidence:
    """Normalize and identify one verified paired-XI parser contract."""

    normalized_lineups = tuple(
        sorted(
            (
                team_id,
                tuple(sorted(player_ids, key=lambda player_id: player_id.value)),
            )
            for team_id, player_ids in team_lineups
        )
    )
    normalized_bindings = _normalize_official_lineup_source_bindings(source_bindings)
    normalized_fact_ids = tuple(sorted(fact_ids))
    contract_id = _official_lineup_contract_id(
        contract_version=OFFICIAL_LINEUP_CONTRACT_VERSION,
        raw_asset_id=raw_asset_id,
        source=source,
        source_match_id=source_match_id,
        match_mapping_source=match_mapping_source,
        team_mapping_source=team_mapping_source,
        player_mapping_source=player_mapping_source,
        match_id=match_id,
        match_version=match_version,
        parser_version=parser_version,
        published_at=published_at,
        observed_at=observed_at,
        team_lineups=normalized_lineups,
        source_bindings=normalized_bindings,
        fact_ids=normalized_fact_ids,
    )
    return OfficialLineupContractEvidence(
        contract_version=OFFICIAL_LINEUP_CONTRACT_VERSION,
        contract_id=contract_id,
        raw_asset_id=raw_asset_id,
        source=source,
        source_match_id=source_match_id,
        match_mapping_source=match_mapping_source,
        team_mapping_source=team_mapping_source,
        player_mapping_source=player_mapping_source,
        match_id=match_id,
        match_version=match_version,
        parser_version=parser_version,
        published_at=published_at,
        observed_at=observed_at,
        team_lineups=normalized_lineups,
        source_bindings=normalized_bindings,
        fact_ids=normalized_fact_ids,
    )


def _official_lineup_contract_id(
    *,
    contract_version: int,
    raw_asset_id: RawAssetId,
    source: str,
    source_match_id: str,
    match_mapping_source: str,
    team_mapping_source: str,
    player_mapping_source: str,
    match_id: MatchId,
    match_version: int,
    parser_version: str,
    published_at: datetime,
    observed_at: datetime,
    team_lineups: tuple[tuple[TeamId, tuple[PlayerId, ...]], ...],
    source_bindings: tuple[OfficialLineupSourceBinding, ...],
    fact_ids: tuple[str, ...],
) -> str:
    payload = {
        "raw_asset_id": raw_asset_id.value,
        "source": source,
        "source_match_id": source_match_id,
        "match_mapping_source": match_mapping_source,
        "team_mapping_source": team_mapping_source,
        "player_mapping_source": player_mapping_source,
        "match_id": match_id.value,
        "match_version": match_version,
        "parser_version": parser_version,
        "published_at": _timestamp(published_at),
        "observed_at": _timestamp(observed_at),
        "team_lineups": {
            team_id.value: [player_id.value for player_id in player_ids]
            for team_id, player_ids in team_lineups
        },
        "fact_ids": fact_ids,
    }
    if contract_version == OFFICIAL_LINEUP_CONTRACT_VERSION:
        payload = {
            "contract_version": contract_version,
            **payload,
            "source_bindings": [binding.to_payload() for binding in source_bindings],
        }
    elif contract_version != LEGACY_OFFICIAL_LINEUP_CONTRACT_VERSION:
        raise ValueError("unsupported official lineup contract version")
    digest = hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()
    return f"official-lineup-contract:{digest}"


def _normalize_official_lineup_source_bindings(
    bindings: Sequence[OfficialLineupSourceBinding],
) -> tuple[OfficialLineupSourceBinding, ...]:
    return tuple(
        sorted(
            bindings,
            key=lambda binding: (
                binding.source_team_id,
                binding.source_player_id,
                binding.team_id.value,
                binding.player_id.value,
            ),
        )
    )


def _validate_official_lineup_source_bindings(
    bindings: tuple[OfficialLineupSourceBinding, ...],
    *,
    team_lineups: tuple[tuple[TeamId, tuple[PlayerId, ...]], ...],
) -> None:
    if len(bindings) != 22:
        raise ValueError("official lineup contract requires 22 source-player bindings")
    if len({binding.source_player_id for binding in bindings}) != len(bindings):
        raise ValueError("official lineup source player IDs must be unique")
    source_teams: dict[str, TeamId] = {}
    source_team_counts: dict[str, int] = {}
    for binding in bindings:
        mapped_team = source_teams.setdefault(binding.source_team_id, binding.team_id)
        if mapped_team != binding.team_id:
            raise ValueError("one official source team cannot bind multiple platform teams")
        source_team_counts[binding.source_team_id] = (
            source_team_counts.get(binding.source_team_id, 0) + 1
        )
    lineup_assignments = {
        (team_id, player_id) for team_id, player_ids in team_lineups for player_id in player_ids
    }
    if (
        len(source_teams) != 2
        or set(source_teams.values()) != {team_id for team_id, _ in team_lineups}
        or set(source_team_counts.values()) != {11}
        or {(binding.team_id, binding.player_id) for binding in bindings} != lineup_assignments
    ):
        raise ValueError("official lineup source bindings do not match paired platform XIs")


def _official_lineup_contract_from_row(
    row: sqlite3.Row,
) -> OfficialLineupContractEvidence:
    lineup_value = json.loads(row["team_lineups_json"])
    fact_value = json.loads(row["fact_ids_json"])
    binding_text = row["source_bindings_json"]
    binding_value = None if binding_text is None else json.loads(binding_text)
    if not isinstance(lineup_value, dict) or not isinstance(fact_value, list):
        raise ValueError("official lineup contract JSON fields are invalid")
    lineups: list[tuple[TeamId, tuple[PlayerId, ...]]] = []
    for team_id, players in lineup_value.items():
        if (
            not isinstance(team_id, str)
            or not isinstance(players, list)
            or not all(isinstance(player_id, str) for player_id in players)
        ):
            raise ValueError("official lineup contract team lineups are invalid")
        lineups.append((TeamId(team_id), tuple(PlayerId(player_id) for player_id in players)))
    if not all(isinstance(fact_id, str) for fact_id in fact_value):
        raise ValueError("official lineup contract fact references are invalid")
    bindings: list[OfficialLineupSourceBinding] = []
    if binding_value is not None:
        if not isinstance(binding_value, list):
            raise ValueError("official lineup contract source bindings are invalid")
        expected_fields = {
            "source_team_id",
            "source_player_id",
            "team_id",
            "player_id",
        }
        for item in binding_value:
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise ValueError("official lineup contract source binding is invalid")
            if any(not isinstance(item[field], str) for field in expected_fields):
                raise ValueError("official lineup contract source binding values are invalid")
            bindings.append(
                OfficialLineupSourceBinding(
                    source_team_id=item["source_team_id"],
                    source_player_id=item["source_player_id"],
                    team_id=TeamId(item["team_id"]),
                    player_id=PlayerId(item["player_id"]),
                )
            )
    return OfficialLineupContractEvidence(
        contract_version=int(row["contract_version"]),
        contract_id=str(row["contract_id"]),
        raw_asset_id=RawAssetId(row["raw_asset_id"]),
        source=str(row["source"]),
        source_match_id=str(row["source_match_id"]),
        match_mapping_source=str(row["match_mapping_source"]),
        team_mapping_source=str(row["team_mapping_source"]),
        player_mapping_source=str(row["player_mapping_source"]),
        match_id=MatchId(row["match_id"]),
        match_version=int(row["match_version"]),
        parser_version=str(row["parser_version"]),
        published_at=_parse_timestamp(row["published_at"]),
        observed_at=_parse_timestamp(row["observed_at"]),
        team_lineups=tuple(sorted(lineups)),
        source_bindings=tuple(bindings),
        fact_ids=tuple(fact_value),
    )


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


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


def _source_mapping_id(source: str, entity_type: str, source_id: str, version: int) -> str:
    identity = f"{entity_type}|{source_id}|{version}"
    return _stable_id("source-mapping", source, identity)


def _timestamp(value: datetime) -> str:
    require_utc(value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _mapping_timestamp(value: datetime) -> str:
    """Use fixed-width UTC text so SQLite range constraints preserve microsecond order."""

    require_utc(value)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_uuid_id_sql(column: str, prefix: str) -> str:
    expected_length = len(prefix) + 1 + 36
    pattern = f"{prefix}:{_UUID_GLOB_SUFFIX}"
    return (
        f"{column} IS NOT NULL AND length({column}) = {expected_length} "
        f"AND {column} GLOB '{pattern}'"
    )


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_utc(parsed)
    return parsed


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


def _prematch_events_table_sql(table_name: str, *, if_not_exists: bool) -> str:
    if table_name not in {"prematch_events", "prematch_events_v4"}:
        raise ValueError("unsupported prematch events table name")
    qualifier = " IF NOT EXISTS" if if_not_exists else ""
    return f"""
CREATE TABLE{qualifier} {table_name} (
    record_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    match_version INTEGER,
    team_id TEXT REFERENCES teams(team_id),
    player_id TEXT REFERENCES players(player_id),
    event_type TEXT NOT NULL,
    occurred_at TEXT,
    known_at TEXT NOT NULL,
    confirmation_status TEXT NOT NULL CHECK (
        confirmation_status IN ('official', 'corroborated', 'unconfirmed')
    ),
    evidence_refs_json TEXT NOT NULL,
    can_modify_features INTEGER NOT NULL CHECK (can_modify_features IN (0, 1)),
    CHECK (player_id IS NULL OR match_version IS NOT NULL OR can_modify_features = 0),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version)
);
"""


_SCHEMA = (
    f"""
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
    mapping_id TEXT PRIMARY KEY CHECK (
        {_canonical_uuid_id_sql("mapping_id", "source-mapping")}
    ),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0 AND source = trim(source)),
    entity_type TEXT NOT NULL CHECK (
        entity_type IN ('competition', 'season', 'team', 'player', 'match')
    ),
    source_id TEXT NOT NULL CHECK (
        length(trim(source_id)) > 0 AND source_id = trim(source_id)
    ),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    version INTEGER NOT NULL CHECK (version > 0),
    valid_from TEXT NOT NULL CHECK (
        length(valid_from) = 27
        AND substr(valid_from, 11, 1) = 'T'
        AND substr(valid_from, 20, 1) = '.'
        AND substr(valid_from, 27, 1) = 'Z'
        AND julianday(valid_from) IS NOT NULL
    ),
    valid_to TEXT,
    match_rule TEXT NOT NULL CHECK (
        match_rule IN ('source_id', 'exact_alias', 'fuzzy_alias', 'manual_override')
    ),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL CHECK (
        length(created_at) = 27
        AND substr(created_at, 11, 1) = 'T'
        AND substr(created_at, 20, 1) = '.'
        AND substr(created_at, 27, 1) = 'Z'
        AND julianday(created_at) IS NOT NULL
    ),
    created_by TEXT NOT NULL CHECK (
        length(trim(created_by)) > 0 AND created_by = trim(created_by)
    ),
    audit_note TEXT NOT NULL CHECK (
        length(trim(audit_note)) > 0 AND audit_note = trim(audit_note)
    ),
    supersedes_mapping_id TEXT REFERENCES source_mappings(mapping_id),
    CHECK (
        valid_to IS NULL OR (
            length(valid_to) = 27
            AND substr(valid_to, 11, 1) = 'T'
            AND substr(valid_to, 20, 1) = '.'
            AND substr(valid_to, 27, 1) = 'Z'
            AND julianday(valid_to) IS NOT NULL
            AND valid_to > valid_from
        )
    ),
    CHECK (
        (version = 1 AND supersedes_mapping_id IS NULL)
        OR (version > 1 AND supersedes_mapping_id IS NOT NULL)
    ),
    UNIQUE (source, entity_type, source_id, version)
);

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
    source_id TEXT,
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

CREATE TABLE IF NOT EXISTS match_report_contracts (
    contract_id TEXT PRIMARY KEY,
    collection_attempt_id TEXT NOT NULL UNIQUE
        REFERENCES collection_attempts(collection_attempt_id),
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    source_match_id TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    required_tables_json TEXT NOT NULL,
    team_tables_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version)
);

CREATE INDEX IF NOT EXISTS match_report_contracts_match
ON match_report_contracts(match_id, match_version);

CREATE TABLE IF NOT EXISTS official_lineup_contracts (
    contract_id TEXT PRIMARY KEY,
    contract_version INTEGER NOT NULL CHECK (contract_version IN (1, 2)),
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    source TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    match_mapping_source TEXT NOT NULL,
    team_mapping_source TEXT NOT NULL,
    player_mapping_source TEXT NOT NULL,
    match_id TEXT NOT NULL,
    match_version INTEGER NOT NULL,
    parser_version TEXT NOT NULL,
    published_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    team_lineups_json TEXT NOT NULL,
    source_bindings_json TEXT,
    fact_ids_json TEXT NOT NULL,
    CHECK (
        (contract_version = 1 AND source_bindings_json IS NULL)
        OR (contract_version = 2 AND source_bindings_json IS NOT NULL)
    ),
    UNIQUE (raw_asset_id, parser_version, contract_version),
    FOREIGN KEY (match_id, match_version) REFERENCES match_versions(match_id, version)
);

CREATE INDEX IF NOT EXISTS official_lineup_contracts_match
ON official_lineup_contracts(match_id, match_version);

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

CREATE TRIGGER IF NOT EXISTS match_results_90_version_immutable
BEFORE UPDATE OF observation_version ON match_results_90
WHEN OLD.observation_version <> NEW.observation_version
BEGIN
    SELECT RAISE(ABORT, 'match result observation_version is immutable');
END;

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

CREATE TRIGGER IF NOT EXISTS team_match_observations_version_immutable
BEFORE UPDATE OF observation_version ON team_match_observations
WHEN OLD.observation_version <> NEW.observation_version
BEGIN
    SELECT RAISE(ABORT, 'team observation observation_version is immutable');
END;

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
"""
    + _prematch_events_table_sql("prematch_events", if_not_exists=True)
    + """
CREATE TABLE IF NOT EXISTS fact_evidence (
    record_id TEXT NOT NULL,
    raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (record_id, raw_asset_id)
);
"""
)
