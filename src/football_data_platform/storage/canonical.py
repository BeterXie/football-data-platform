"""SQLite catalog for canonical identities, mappings, and match versions."""

from __future__ import annotations

import hashlib
import json
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

SCHEMA_VERSION = 6
OFFICIAL_LINEUP_CONTRACT_VERSION = 2
LEGACY_OFFICIAL_LINEUP_CONTRACT_VERSION = 1
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
                _migrate_v2_to_v3(connection)
                _migrate_v3_to_v4(connection)
                _migrate_v4_to_v5(connection)
                _migrate_v5_to_v6(connection)
            elif row["version"] == 2:
                _migrate_v2_to_v3(connection)
                _migrate_v3_to_v4(connection)
                _migrate_v4_to_v5(connection)
                _migrate_v5_to_v6(connection)
            elif row["version"] == 3:
                _migrate_v3_to_v4(connection)
                _migrate_v4_to_v5(connection)
                _migrate_v5_to_v6(connection)
            elif row["version"] == 4:
                _migrate_v4_to_v5(connection)
                _migrate_v5_to_v6(connection)
            elif row["version"] == 5:
                _migrate_v5_to_v6(connection)
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

    def match_report_contract(self, contract_id: str) -> MatchReportContractEvidence:
        """Load and lineage-verify one persisted match-report parser contract."""

        _require_text(contract_id, "contract_id")
        with self.connect() as connection:
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
        "SELECT 1 FROM match_versions WHERE match_id = ? AND version = ?",
        (evidence.match_id.value, evidence.match_version),
    ).fetchone()
    if version is None:
        raise KeyError(f"match version {evidence.match_id}:{evidence.match_version} does not exist")
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
        mapped = _current_mapping(connection, "fbref", "team", source_team_id)
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
    """
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
