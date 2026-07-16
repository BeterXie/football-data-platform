"""Versioned canonical match facts with mandatory raw lineage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.ids import MatchId, PlayerId, RawAssetId, TeamId
from football_data_platform.domain.lifecycle import (
    MatchAvailability,
    SnapshotAvailability,
)
from football_data_platform.domain.models import MatchStatus, require_utc
from football_data_platform.storage.canonical import CanonicalStore


@dataclass(frozen=True, slots=True)
class StoredFact:
    record_id: str
    observation_version: int | None


class CanonicalFactStore:
    def __init__(self, canonical: CanonicalStore) -> None:
        self.canonical = canonical

    def append_result_90(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        home_goals: int,
        away_goals: int,
        known_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> StoredFact:
        if home_goals < 0 or away_goals < 0:
            raise ValueError("goals must be non-negative")
        payload = {
            "match_id": match_id.value,
            "match_version": match_version,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
            "raw_asset_id": raw_asset_id.value,
        }
        return self._append_versioned(
            "match_results_90",
            payload,
            scope={"match_id": match_id.value, "match_version": match_version},
        )

    def append_team_observation(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        team_id: TeamId,
        stats: dict[str, float | int | None],
        known_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> StoredFact:
        _validate_metrics(stats, "stats")
        payload = {
            "match_id": match_id.value,
            "match_version": match_version,
            "team_id": team_id.value,
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
            "stats_json": _json_text(stats),
            "raw_asset_id": raw_asset_id.value,
        }
        return self._append_versioned(
            "team_match_observations",
            payload,
            scope={
                "match_id": match_id.value,
                "match_version": match_version,
                "team_id": team_id.value,
            },
        )

    def append_player_observation(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        team_id: TeamId,
        player_id: PlayerId,
        role: str,
        minutes: float,
        metrics: dict[str, float | int | None],
        known_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> StoredFact:
        if not math.isfinite(minutes) or minutes < 0:
            raise ValueError("minutes must be finite and non-negative")
        _validate_metrics(metrics, "metrics")
        payload = {
            "match_id": match_id.value,
            "match_version": match_version,
            "team_id": team_id.value,
            "player_id": player_id.value,
            "role": role,
            "minutes": minutes,
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
            "metrics_json": _json_text(metrics),
            "raw_asset_id": raw_asset_id.value,
        }
        return self._append_versioned(
            "player_match_observations",
            payload,
            scope={
                "match_id": match_id.value,
                "match_version": match_version,
                "player_id": player_id.value,
            },
        )

    def append_lineup_fact(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        team_id: TeamId,
        player_id: PlayerId,
        lineup_role: str,
        official: bool,
        known_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> StoredFact:
        if lineup_role not in {"starter", "bench"}:
            raise ValueError("lineup_role must be starter or bench")
        payload = {
            "match_id": match_id.value,
            "match_version": match_version,
            "team_id": team_id.value,
            "player_id": player_id.value,
            "lineup_role": lineup_role,
            "official": int(official),
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
            "raw_asset_id": raw_asset_id.value,
        }
        record_id = _record_id("lineup", _semantic_payload(payload))
        with self.canonical.connect() as connection:
            _require_temporal_order(known_at, observed_at)
            existing = connection.execute(
                "SELECT record_id FROM lineup_facts WHERE record_id = ?", (record_id,)
            ).fetchone()
            if existing is None:
                columns = ", ".join(("record_id", *payload.keys()))
                placeholders = ", ".join("?" for _ in range(len(payload) + 1))
                connection.execute(
                    f"INSERT INTO lineup_facts({columns}) VALUES ({placeholders})",
                    (record_id, *payload.values()),
                )
            _link_evidence(
                connection,
                record_id=record_id,
                raw_asset_id=raw_asset_id.value,
                observed_at=payload["observed_at"],
            )
        return StoredFact(record_id, None)

    def add_news_evidence(
        self,
        *,
        source: str,
        url: str,
        title: str,
        published_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> StoredFact:
        _require_temporal_order(published_at, observed_at)
        payload = {
            "source": source,
            "url": url,
            "title": title,
            "published_at": _timestamp(published_at),
            "observed_at": _timestamp(observed_at),
            "raw_asset_id": raw_asset_id.value,
        }
        record_id = _record_id("news", _semantic_payload(payload))
        self._insert_unversioned("news_evidence", record_id, payload)
        return StoredFact(record_id, None)

    def add_prematch_event(
        self,
        *,
        match_id: MatchId,
        team_id: TeamId | None,
        player_id: PlayerId | None,
        event_type: str,
        occurred_at: datetime | None,
        known_at: datetime,
        confirmation_status: str,
        evidence_refs: tuple[str, ...],
    ) -> StoredFact:
        if confirmation_status not in {"official", "corroborated", "unconfirmed"}:
            raise ValueError("invalid confirmation_status")
        require_utc(known_at, "known_at")
        if occurred_at is not None:
            require_utc(occurred_at, "occurred_at")
        if not evidence_refs:
            raise ValueError("pre-match events require evidence")
        can_modify = confirmation_status in {"official", "corroborated"}
        payload = {
            "match_id": match_id.value,
            "team_id": team_id.value if team_id is not None else None,
            "player_id": player_id.value if player_id is not None else None,
            "event_type": event_type,
            "occurred_at": _timestamp(occurred_at) if occurred_at is not None else None,
            "known_at": _timestamp(known_at),
            "confirmation_status": confirmation_status,
            "evidence_refs_json": _json_text(sorted(set(evidence_refs))),
            "can_modify_features": int(can_modify),
        }
        record_id = _record_id("prematch-event", payload)
        with self.canonical.connect() as connection:
            found = connection.execute(
                "SELECT COUNT(*) FROM news_evidence WHERE record_id IN "
                f"({','.join('?' for _ in evidence_refs)})",
                evidence_refs,
            ).fetchone()[0]
            if found != len(set(evidence_refs)):
                raise KeyError("pre-match event references unknown news evidence")
        self._insert_unversioned("prematch_events", record_id, payload)
        return StoredFact(record_id, None)

    def availability(
        self,
        match_id: MatchId,
        *,
        snapshots: tuple[SnapshotAvailability, ...] = (),
    ) -> MatchAvailability:
        with self.canonical.connect() as connection:
            match = connection.execute(
                "SELECT home_team_id, away_team_id FROM matches WHERE match_id = ?",
                (match_id.value,),
            ).fetchone()
            if match is None:
                raise KeyError(f"match {match_id} does not exist")
            version = connection.execute(
                "SELECT version, status FROM match_versions WHERE match_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (match_id.value,),
            ).fetchone()
            if version is None:
                raise KeyError(f"match {match_id} has no versions")
            result_present = (
                connection.execute(
                    "SELECT 1 FROM match_results_90 WHERE match_id = ? AND match_version = ? "
                    "LIMIT 1",
                    (match_id.value, version["version"]),
                ).fetchone()
                is not None
            )
            team_stats_rows = connection.execute(
                "SELECT team_id, stats_json FROM team_match_observations WHERE match_id = ? "
                "AND match_version = ? ORDER BY observation_version DESC",
                (match_id.value, version["version"]),
            ).fetchall()
            team_fields: dict[str, frozenset[str]] = {}
            for row in team_stats_rows:
                if row["team_id"] not in team_fields:
                    values = json.loads(row["stats_json"])
                    team_fields[row["team_id"]] = frozenset(
                        key for key, value in values.items() if value is not None
                    )
            lineup_rows = connection.execute(
                "SELECT team_id, player_id FROM lineup_facts WHERE match_id = ? AND "
                "match_version = ? AND lineup_role = 'starter' AND official = 1",
                (match_id.value, version["version"]),
            ).fetchall()
            starters: dict[str, set[str]] = {}
            for row in lineup_rows:
                starters.setdefault(row["team_id"], set()).add(row["player_id"])
            player_rows = connection.execute(
                "SELECT DISTINCT player_id FROM player_match_observations WHERE match_id = ? "
                "AND match_version = ?",
                (match_id.value, version["version"]),
            ).fetchall()
        return MatchAvailability(
            match_status=MatchStatus(version["status"]),
            team_ids=(match["home_team_id"], match["away_team_id"]),
            snapshots=snapshots,
            result_90_present=result_present,
            team_stat_fields=team_fields,
            starters={team_id: frozenset(players) for team_id, players in starters.items()},
            player_observation_ids=frozenset(row["player_id"] for row in player_rows),
        )

    def _append_versioned(
        self,
        table: str,
        payload: dict[str, Any],
        *,
        scope: dict[str, Any],
    ) -> StoredFact:
        known_at = _parse_timestamp(payload["known_at"])
        observed_at = _parse_timestamp(payload["observed_at"])
        _require_temporal_order(known_at, observed_at)
        record_id = _record_id(table, _semantic_payload(payload))
        with self.canonical.connect() as connection:
            existing = connection.execute(
                f"SELECT observation_version FROM {table} WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if existing is not None:
                _link_evidence(
                    connection,
                    record_id=record_id,
                    raw_asset_id=payload["raw_asset_id"],
                    observed_at=payload["observed_at"],
                )
                return StoredFact(record_id, int(existing["observation_version"]))
            where = " AND ".join(f"{key} = ?" for key in scope)
            row = connection.execute(
                f"SELECT COALESCE(MAX(observation_version), 0) FROM {table} WHERE {where}",
                tuple(scope.values()),
            ).fetchone()
            observation_version = int(row[0]) + 1
            values = {
                "record_id": record_id,
                **payload,
                "observation_version": observation_version,
            }
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO {table}({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            _link_evidence(
                connection,
                record_id=record_id,
                raw_asset_id=payload["raw_asset_id"],
                observed_at=payload["observed_at"],
            )
        return StoredFact(record_id, observation_version)

    def _insert_unversioned(self, table: str, record_id: str, payload: dict[str, Any]) -> None:
        with self.canonical.connect() as connection:
            exists = (
                connection.execute(
                    f"SELECT 1 FROM {table} WHERE record_id = ?", (record_id,)
                ).fetchone()
                is not None
            )
            if not exists:
                values = {"record_id": record_id, **payload}
                columns = ", ".join(values)
                placeholders = ", ".join("?" for _ in values)
                connection.execute(
                    f"INSERT INTO {table}({columns}) VALUES ({placeholders})",
                    tuple(values.values()),
                )
            if "raw_asset_id" in payload and "observed_at" in payload:
                _link_evidence(
                    connection,
                    record_id=record_id,
                    raw_asset_id=payload["raw_asset_id"],
                    observed_at=payload["observed_at"],
                )


def _validate_metrics(values: dict[str, float | int | None], name: str) -> None:
    for key, value in values.items():
        if not key:
            raise ValueError(f"{name} contains an empty key")
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} value {key!r} must be finite or None")
    _json_text(values)


def _record_id(kind: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()
    return f"fact:{kind}:{digest}"


def _semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if key not in {"observed_at", "raw_asset_id"}
    }


def _link_evidence(
    connection,
    *,
    record_id: str,
    raw_asset_id: str,
    observed_at: str,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO fact_evidence(record_id, raw_asset_id, observed_at) "
        "VALUES (?, ?, ?)",
        (record_id, raw_asset_id, observed_at),
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _timestamp(value: datetime) -> str:
    require_utc(value)
    return value.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_temporal_order(known_at: datetime, observed_at: datetime) -> None:
    require_utc(known_at, "known_at")
    require_utc(observed_at, "observed_at")
    if known_at > observed_at:
        raise ValueError("known_at cannot be later than observed_at")
