"""Versioned canonical match facts with mandatory raw lineage."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.ids import MatchId, PlayerId, RawAssetId, TeamId
from football_data_platform.domain.lifecycle import (
    MatchAvailability,
    PlayerObservationAvailability,
    SnapshotAvailability,
)
from football_data_platform.domain.models import (
    CollectionAttempt,
    CollectionAttemptOutcome,
    MatchStatus,
    require_utc,
)
from football_data_platform.domain.predictions import MatchResult90
from football_data_platform.sources.official_lineup import parse_official_lineup_json
from football_data_platform.sources.prematch import (
    NewsEvidenceDTO,
    OfficialLineupDTO,
    PrematchEventDTO,
    SourceKind,
    SourceRegistry,
    classify_confirmation,
)
from football_data_platform.storage.canonical import (
    OFFICIAL_LINEUP_CONTRACT_VERSION,
    CanonicalStore,
    OfficialLineupContractEvidence,
    OfficialLineupSourceBinding,
    build_official_lineup_contract,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive


@dataclass(frozen=True, slots=True)
class StoredFact:
    record_id: str
    observation_version: int | None


class OfficialLineupRawParseError(ValueError):
    """The archived official-lineup payload could not be parsed."""


class CanonicalFactStore:
    def __init__(
        self,
        canonical: CanonicalStore,
        *,
        source_registry: SourceRegistry | None = None,
        raw_archive: RawArchive | None = None,
    ) -> None:
        self.canonical = canonical
        self.source_registry = source_registry or SourceRegistry()
        self.raw_archive = raw_archive

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
        with self.canonical.connect() as connection:
            _require_match_version(
                connection,
                match_id=match_id,
                match_version=match_version,
                require_finished=True,
            )
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

    def verify_match_result(self, result: MatchResult90) -> None:
        """Verify a settlement result against one immutable canonical fact."""

        if not isinstance(result, MatchResult90):
            raise TypeError("result must be a MatchResult90")
        archive = self.raw_archive or RawArchive(DataLayout(self.canonical.path.parent.parent))
        verified = load_verified_match_result(
            result.source_ref,
            archive=archive,
            canonical=self.canonical,
        )
        if verified != result:
            raise ValueError("result does not match its canonical fact")

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
        with self.canonical.connect() as connection:
            match_row = _require_match_version(
                connection,
                match_id=match_id,
                match_version=match_version,
                require_finished=True,
            )
            _require_match_team(match_row, team_id)
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
        with self.canonical.connect() as connection:
            match_row = _require_match_version(
                connection,
                match_id=match_id,
                match_version=match_version,
                require_finished=True,
            )
            _require_match_team(match_row, team_id)
            _require_registered_player(connection, player_id)
            _require_player_assignment(
                connection,
                match_id=match_id,
                match_version=match_version,
                player_id=player_id,
                team_id=team_id,
            )
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
        if not isinstance(official, bool) or official:
            raise ValueError("official lineup facts must be written through append_official_lineup")
        if lineup_role not in {"starter", "bench"}:
            raise ValueError("lineup_role must be starter or bench")
        _require_temporal_order(known_at, observed_at)
        payload = {
            "match_id": match_id.value,
            "match_version": match_version,
            "team_id": team_id.value,
            "player_id": player_id.value,
            "lineup_role": lineup_role,
            "official": 0,
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
            "raw_asset_id": raw_asset_id.value,
        }
        with self.canonical.connect() as connection:
            match_row = _require_match_version(
                connection,
                match_id=match_id,
                match_version=match_version,
            )
            _require_match_team(match_row, team_id)
            _require_registered_player(connection, player_id)
            _require_player_assignment(
                connection,
                match_id=match_id,
                match_version=match_version,
                player_id=player_id,
                team_id=team_id,
            )
            return _insert_lineup_payload(connection, payload)

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
        _require_text(source, "source")
        _require_text(url, "url")
        _require_text(title, "title")
        _require_temporal_order(published_at, observed_at)
        self.source_registry.validate_url(source, url)
        with self.canonical.connect() as connection:
            raw = connection.execute(
                "SELECT source, url, observed_at FROM raw_assets WHERE raw_asset_id = ?",
                (raw_asset_id.value,),
            ).fetchone()
            if raw is None:
                raise KeyError(f"raw asset {raw_asset_id} is not registered")
            if raw["observed_at"] != _timestamp(observed_at):
                raise ValueError("news observed_at must match the raw archive observation")
            if raw["source"] != source:
                message = (
                    f"raw asset source {raw['source']!r} does not match news source {source!r}"
                )
                raise ValueError(message)
            if raw["url"] != url:
                raise ValueError("news URL does not match the raw archive")
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

    def add_news_evidence_dto(self, evidence: NewsEvidenceDTO) -> StoredFact:
        """Persist a parser-produced news DTO while retaining its raw asset reference."""

        return self.add_news_evidence(
            source=evidence.source,
            url=evidence.url,
            title=evidence.title,
            published_at=evidence.published_at,
            observed_at=evidence.observed_at,
            raw_asset_id=evidence.raw_asset_id,
        )

    def add_prematch_event(
        self,
        *,
        match_id: MatchId,
        team_id: TeamId | None,
        player_id: PlayerId | None,
        event_type: str,
        occurred_at: datetime | None,
        known_at: datetime,
        confirmation_status: str | None = None,
        evidence_refs: tuple[str, ...] = (),
        as_of: datetime | None = None,
    ) -> StoredFact:
        _require_text(event_type, "event_type")
        require_utc(known_at, "known_at")
        if as_of is not None:
            require_utc(as_of, "as_of")
            if known_at > as_of:
                raise ValueError("known_at cannot be later than as_of")
        if occurred_at is not None:
            require_utc(occurred_at, "occurred_at")
            if occurred_at > known_at:
                raise ValueError("occurred_at cannot be later than known_at")
        if not evidence_refs:
            raise ValueError("pre-match events require evidence")
        unique_refs = tuple(dict.fromkeys(evidence_refs))
        if any(not isinstance(ref, str) or not ref for ref in unique_refs):
            raise ValueError("pre-match event evidence references must be non-empty text")
        effective_as_of = as_of if as_of is not None else known_at
        match_version: int | None = None
        with self.canonical.connect() as connection:
            placeholders = ",".join("?" for _ in unique_refs)
            evidence_rows = connection.execute(
                "SELECT n.record_id, n.source, n.url, n.published_at, n.observed_at, "
                "n.raw_asset_id, r.source AS raw_source, r.url AS raw_url, "
                "r.observed_at AS raw_observed_at "
                "FROM news_evidence AS n JOIN raw_assets AS r "
                "ON r.raw_asset_id = n.raw_asset_id "
                f"WHERE n.record_id IN ({placeholders})",
                unique_refs,
            ).fetchall()
            if len(evidence_rows) != len(unique_refs):
                raise KeyError("pre-match event references unknown news evidence")
            for row in evidence_rows:
                if row["raw_source"] != row["source"]:
                    raise ValueError(
                        "news evidence source does not match its registered raw archive"
                    )
                if row["raw_url"] != row["url"]:
                    raise ValueError("news evidence URL does not match its raw archive")
                if row["raw_observed_at"] != row["observed_at"]:
                    raise ValueError("news evidence observed_at does not match its raw archive")
                if self.source_registry.get(row["source"]) is None:
                    raise ValueError(f"event evidence source {row['source']!r} is not registered")
            latest_published_at = max(
                _parse_timestamp(row["published_at"]) for row in evidence_rows
            )
            latest_observed_at = max(_parse_timestamp(row["observed_at"]) for row in evidence_rows)
            if known_at < latest_published_at:
                raise ValueError("known_at cannot precede supporting evidence publication")
            if known_at > latest_observed_at:
                raise ValueError("known_at cannot follow supporting evidence observation")
            confirmation_status = classify_confirmation(
                tuple(row["source"] for row in evidence_rows),
                self.source_registry,
                requested=confirmation_status,
            )
            match_row = _require_match(connection, match_id)
            if team_id is not None:
                _require_match_team(match_row, team_id)
            if player_id is not None:
                _require_registered_player(connection, player_id)
                match_version = _visible_match_version(
                    connection,
                    match_id=match_id,
                    visible_at=effective_as_of,
                )
                _require_existing_player_assignment(
                    connection,
                    match_row=match_row,
                    match_id=match_id,
                    match_version=match_version,
                    player_id=player_id,
                    team_id=team_id,
                    known_at_cutoff=known_at,
                    observed_at_cutoff=effective_as_of,
                )
        payload = {
            "match_id": match_id.value,
            "team_id": team_id.value if team_id is not None else None,
            "player_id": player_id.value if player_id is not None else None,
            "event_type": event_type,
            "occurred_at": _timestamp(occurred_at) if occurred_at is not None else None,
            "known_at": _timestamp(known_at),
            "confirmation_status": confirmation_status,
            "evidence_refs_json": _json_text(sorted(unique_refs)),
            "can_modify_features": int(confirmation_status in {"official", "corroborated"}),
        }
        if match_version is not None:
            payload["match_version"] = match_version
        record_id = _record_id("prematch-event", payload)
        self._insert_unversioned("prematch_events", record_id, payload)
        return StoredFact(record_id, None)

    def record_prematch_collection_attempt(
        self,
        *,
        match_id: MatchId,
        source: str,
        kind: SourceKind,
        target_url: str,
        observed_at: datetime,
        collector_version: str,
        outcome: CollectionAttemptOutcome,
        source_id: str | None = None,
        diagnostic_code: str | None = None,
        diagnostic_message: str | None = None,
        raw_asset_id: RawAssetId | None = None,
    ) -> CollectionAttempt:
        """Persist a source-registered prematch attempt with optional raw response evidence.

        The canonical attempt store supplies idempotent identity and raw lineage checks.  This
        wrapper adds the R09 source registry gate so an unregistered or misclassified adapter
        cannot silently create a successful (or diagnostic) prematch attempt.
        """

        descriptor = self.source_registry.get(source)
        if descriptor is None:
            raise ValueError(f"prematch source {source!r} is not registered")
        requested_kind = SourceKind(kind)
        if descriptor.kind is not requested_kind:
            raise ValueError(
                f"prematch source {source!r} is registered as {descriptor.kind.value!r}, "
                f"not {requested_kind.value!r}"
            )
        self.source_registry.validate_url(source, target_url)
        return self.canonical.record_collection_attempt(
            match_id=match_id,
            source=source,
            source_id=source_id,
            target_url=target_url,
            outcome=CollectionAttemptOutcome(outcome),
            observed_at=observed_at,
            collector_version=collector_version,
            diagnostic_code=diagnostic_code,
            diagnostic_message=diagnostic_message,
            raw_asset_id=raw_asset_id,
        )

    def add_prematch_event_dto(self, event: PrematchEventDTO) -> StoredFact:
        """Persist an event after recalculating confirmation from canonical evidence."""

        return self.add_prematch_event(
            match_id=event.match_id,
            team_id=event.team_id,
            player_id=event.player_id,
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            known_at=event.known_at,
            confirmation_status=event.requested_confirmation_status or "unconfirmed",
            evidence_refs=event.evidence_refs,
            as_of=event.as_of,
        )

    def append_official_lineup(self, lineup: OfficialLineupDTO) -> tuple[StoredFact, ...]:
        """Reject legacy DTO writes that cannot prove the archived raw payload."""

        if not isinstance(lineup, OfficialLineupDTO):
            raise TypeError("lineup must be an OfficialLineupDTO")
        raise ValueError("official lineup facts require a verified raw replay contract")

    def append_official_lineup_asset(
        self,
        *,
        archive: RawArchive,
        raw_asset_id: RawAssetId,
    ) -> tuple[OfficialLineupContractEvidence, tuple[StoredFact, ...]]:
        """Parse archived bytes and atomically persist their paired official XIs."""

        asset = archive.load(raw_asset_id)
        content = archive.read(asset)
        try:
            parsed = parse_official_lineup_json(content)
        except Exception as error:
            raise OfficialLineupRawParseError(str(error)) from error
        source = asset.source
        source_match_id = asset.source_id
        match_mapping_source = parsed.match_mapping_source
        team_mapping_source = parsed.team_mapping_source
        player_mapping_source = parsed.player_mapping_source
        published_at = parsed.published_at
        observed_at = asset.observed_at
        url = asset.url
        parser_version = parsed.parser_version
        if (
            parsed.source != source
            or parsed.source_match_id != source_match_id
            or parsed.published_at > observed_at
            or (
                asset.target_event_time is not None
                and asset.target_event_time != parsed.published_at
            )
            or asset.collector_version != parser_version
            or asset.media_type != "application/json"
        ):
            raise ValueError("official lineup raw metadata conflicts with parsed content")

        mapped_match_id = self.canonical.mapped_match_ids(
            source=match_mapping_source,
            source_ids=(source_match_id,),
        ).get(source_match_id)
        if mapped_match_id is None:
            raise ValueError("official lineup source match mapping is missing")
        visible_versions = tuple(
            version
            for version in self.canonical.match_versions(mapped_match_id)
            if version.observed_at <= observed_at
        )
        if not visible_versions:
            raise ValueError("official lineup canonical match version is not visible")
        match_version = visible_versions[-1].version
        match = self.canonical.match(mapped_match_id)
        team_source_lineups: list[tuple[str, TeamId, tuple[tuple[str, str], ...]]] = []
        for parsed_team in parsed.teams:
            team = self.canonical.mapped_team(
                source=team_mapping_source,
                source_id=parsed_team.source_team_id,
            )
            source_players = tuple(
                (player.source_player_id, player.name) for player in parsed_team.starters
            )
            team_source_lineups.append((parsed_team.source_team_id, team.id, source_players))
        if {team_id for _, team_id, _ in team_source_lineups} != {
            match.home_team_id,
            match.away_team_id,
        }:
            raise ValueError("official lineup teams do not match the canonical fixture")
        match_id = mapped_match_id
        normalized_team_source_lineups = tuple(team_source_lineups)

        descriptor = self.source_registry.get(source)
        if descriptor is None or not descriptor.official:
            raise ValueError(f"lineup source {source!r} is not a verified official source")
        if descriptor.kind is not SourceKind.OFFICIAL_LINEUP:
            raise ValueError(f"source {source!r} is not an official-lineup adapter")
        self.source_registry.validate_url(source, url)
        _require_temporal_order(published_at, observed_at)
        if len(normalized_team_source_lineups) != 2:
            raise ValueError("official lineup contract requires exactly two teams")
        if len({source_team_id for source_team_id, _, _ in normalized_team_source_lineups}) != 2:
            raise ValueError("official lineup contract source teams must be unique")
        if len({team_id for _, team_id, _ in normalized_team_source_lineups}) != 2:
            raise ValueError("official lineup contract teams must be unique")
        all_source_player_ids = [
            source_player_id
            for _, _, players in normalized_team_source_lineups
            for source_player_id, _ in players
        ]
        if any(
            len(players) != 11 or len({source_player_id for source_player_id, _ in players}) != 11
            for _, _, players in normalized_team_source_lineups
        ):
            raise ValueError("official lineup contract requires 11 unique players per team")
        if len(set(all_source_player_ids)) != len(all_source_player_ids):
            raise ValueError("official lineup players must be unique across both teams")
        with self.canonical.connect() as connection:
            raw = connection.execute(
                "SELECT source, source_id, url, observed_at, target_event_time, "
                "collector_version FROM raw_assets WHERE raw_asset_id = ?",
                (raw_asset_id.value,),
            ).fetchone()
            if raw is None:
                raise KeyError(f"raw asset {raw_asset_id} is not registered")
            expected_raw = (
                source,
                source_match_id,
                url,
                _timestamp(observed_at),
                (
                    _timestamp(asset.target_event_time)
                    if asset.target_event_time is not None
                    else None
                ),
                parser_version,
            )
            actual_raw = tuple(raw)
            if actual_raw != expected_raw:
                raise ValueError("official lineup contract conflicts with raw archive metadata")
            mapping = connection.execute(
                "SELECT entity_id FROM source_mappings WHERE source = ? "
                "AND entity_type = 'match' AND source_id = ? AND valid_to IS NULL",
                (match_mapping_source, source_match_id),
            ).fetchone()
            if mapping is None or mapping["entity_id"] != match_id.value:
                raise ValueError("official lineup source match does not map to the canonical match")
            match_row = _require_match_version(
                connection,
                match_id=match_id,
                match_version=match_version,
            )
            if {team_id.value for _, team_id, _ in normalized_team_source_lineups} != {
                match_row["home_team_id"],
                match_row["away_team_id"],
            }:
                raise ValueError("official lineup teams do not match the canonical fixture")
            for source_team_id, team_id, _ in normalized_team_source_lineups:
                team_mapping = connection.execute(
                    "SELECT entity_id FROM source_mappings WHERE source = ? "
                    "AND entity_type = 'team' AND source_id = ? AND valid_to IS NULL",
                    (team_mapping_source, source_team_id),
                ).fetchone()
                if team_mapping is None or team_mapping["entity_id"] != team_id.value:
                    raise ValueError(
                        "official lineup source team does not map to the canonical fixture team"
                    )

            team_lineup_items: list[tuple[TeamId, tuple[PlayerId, ...]]] = []
            source_bindings: list[OfficialLineupSourceBinding] = []
            for source_team_id, team_id, players in normalized_team_source_lineups:
                mapped_players: list[PlayerId] = []
                for source_player_id, canonical_name in players:
                    player_id = self.canonical._resolve_or_create_player(
                        connection,
                        source=player_mapping_source,
                        source_id=source_player_id,
                        canonical_name=canonical_name,
                        observed_at=observed_at,
                        raw_asset_id=raw_asset_id,
                    ).id
                    mapped_players.append(player_id)
                    source_bindings.append(
                        OfficialLineupSourceBinding(
                            source_team_id=source_team_id,
                            source_player_id=source_player_id,
                            team_id=team_id,
                            player_id=player_id,
                        )
                    )
                team_lineup_items.append((team_id, tuple(mapped_players)))
            team_lineups = tuple(team_lineup_items)
            all_player_ids = [
                player_id for _, player_ids in team_lineups for player_id in player_ids
            ]
            if len(set(all_player_ids)) != len(all_player_ids):
                raise ValueError(
                    "official lineup source players must map to 22 distinct canonical players"
                )
            for team_id, player_ids in team_lineups:
                for player_id in player_ids:
                    _require_registered_player(connection, player_id)
                    _require_player_assignment(
                        connection,
                        match_id=match_id,
                        match_version=match_version,
                        player_id=player_id,
                        team_id=team_id,
                    )

            payloads = [
                {
                    "match_id": match_id.value,
                    "match_version": match_version,
                    "team_id": team_id.value,
                    "player_id": player_id.value,
                    "lineup_role": "starter",
                    "official": 1,
                    "known_at": _timestamp(published_at),
                    "observed_at": _timestamp(observed_at),
                    "raw_asset_id": raw_asset_id.value,
                }
                for team_id, players in team_lineups
                for player_id in players
            ]
            fact_ids = tuple(
                _record_id("lineup", _semantic_payload(payload)) for payload in payloads
            )
            contract = build_official_lineup_contract(
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
                team_lineups=team_lineups,
                source_bindings=source_bindings,
                fact_ids=fact_ids,
            )
            stored = tuple(_insert_lineup_payload(connection, payload) for payload in payloads)
            contract_payload = {
                "contract_id": contract.contract_id,
                "contract_version": contract.contract_version,
                "raw_asset_id": contract.raw_asset_id.value,
                "source": contract.source,
                "source_match_id": contract.source_match_id,
                "match_mapping_source": contract.match_mapping_source,
                "team_mapping_source": contract.team_mapping_source,
                "player_mapping_source": contract.player_mapping_source,
                "match_id": contract.match_id.value,
                "match_version": contract.match_version,
                "parser_version": contract.parser_version,
                "published_at": _timestamp(contract.published_at),
                "observed_at": _timestamp(contract.observed_at),
                "team_lineups_json": _json_text(
                    {
                        team_id.value: [player_id.value for player_id in player_ids]
                        for team_id, player_ids in contract.team_lineups
                    }
                ),
                "source_bindings_json": _json_text(
                    [binding.to_payload() for binding in contract.source_bindings]
                ),
                "fact_ids_json": _json_text(contract.fact_ids),
            }
            _insert_exact_payload(
                connection,
                table="official_lineup_contracts",
                primary_key="contract_id",
                payload=contract_payload,
            )
        return contract, stored

    def availability(
        self,
        match_id: MatchId,
        *,
        snapshots: tuple[SnapshotAvailability, ...] = (),
        as_of: datetime | None = None,
    ) -> MatchAvailability:
        if as_of is not None:
            require_utc(as_of, "as_of")
        as_of_text = _timestamp(as_of) if as_of is not None else None
        with self.canonical.connect() as connection:
            match = connection.execute(
                "SELECT home_team_id, away_team_id FROM matches WHERE match_id = ?",
                (match_id.value,),
            ).fetchone()
            if match is None:
                raise KeyError(f"match {match_id} does not exist")
            version_query = "SELECT version, status FROM match_versions WHERE match_id = ?"
            version_parameters: list[object] = [match_id.value]
            if as_of_text is not None:
                version_query += " AND observed_at <= ?"
                version_parameters.append(as_of_text)
            version_query += " ORDER BY version DESC LIMIT 1"
            version = connection.execute(version_query, tuple(version_parameters)).fetchone()
            if version is None:
                suffix = " by as_of" if as_of_text is not None else ""
                raise KeyError(f"match {match_id} has no versions{suffix}")
            result_query = (
                "SELECT known_at FROM match_results_90 WHERE match_id = ? AND match_version = ?"
            )
            result_parameters: list[object] = [match_id.value, version["version"]]
            if as_of_text is not None:
                result_query += " AND known_at <= ?"
                result_parameters.append(as_of_text)
            result_query += " ORDER BY observation_version DESC LIMIT 1"
            result_row = connection.execute(result_query, tuple(result_parameters)).fetchone()
            result_present = result_row is not None
            result_known_at = _parse_timestamp(result_row["known_at"]) if result_row else None
            team_query = (
                "SELECT team_id, stats_json, known_at FROM team_match_observations "
                "WHERE match_id = ? AND match_version = ?"
            )
            team_parameters: list[object] = [match_id.value, version["version"]]
            if as_of_text is not None:
                team_query += " AND known_at <= ?"
                team_parameters.append(as_of_text)
            team_query += " ORDER BY observation_version DESC"
            team_stats_rows = connection.execute(team_query, tuple(team_parameters)).fetchall()
            team_fields: dict[str, frozenset[str]] = {}
            team_stats_known_at: dict[str, datetime] = {}
            for row in team_stats_rows:
                if row["team_id"] not in team_fields:
                    values = json.loads(row["stats_json"])
                    team_fields[row["team_id"]] = frozenset(
                        key for key, value in values.items() if value is not None
                    )
                    team_stats_known_at[row["team_id"]] = _parse_timestamp(row["known_at"])
            lineup_query = (
                "SELECT team_id, player_id FROM lineup_facts WHERE match_id = ? AND "
                "match_version = ? AND lineup_role = 'starter' AND official = 1"
            )
            lineup_parameters: list[object] = [match_id.value, version["version"]]
            if as_of_text is not None:
                lineup_query += " AND known_at <= ?"
                lineup_parameters.append(as_of_text)
            lineup_rows = connection.execute(lineup_query, tuple(lineup_parameters)).fetchall()
            starters: dict[str, set[str]] = {}
            for row in lineup_rows:
                starters.setdefault(row["team_id"], set()).add(row["player_id"])
            player_query = (
                "SELECT player_id, team_id, role, minutes, metrics_json, known_at "
                "FROM player_match_observations WHERE match_id = ? AND match_version = ?"
            )
            player_parameters: list[object] = [match_id.value, version["version"]]
            if as_of_text is not None:
                player_query += " AND known_at <= ?"
                player_parameters.append(as_of_text)
            player_query += " ORDER BY observation_version DESC"
            player_rows = connection.execute(player_query, tuple(player_parameters)).fetchall()
            player_observations: dict[str, PlayerObservationAvailability] = {}
            for row in player_rows:
                if row["player_id"] in player_observations:
                    continue
                metrics = json.loads(row["metrics_json"])
                player_observations[row["player_id"]] = PlayerObservationAvailability(
                    team_id=row["team_id"],
                    role=row["role"],
                    minutes=float(row["minutes"]),
                    metric_fields=frozenset(
                        key for key, value in metrics.items() if value is not None
                    ),
                    known_at=_parse_timestamp(row["known_at"]),
                )
        return MatchAvailability(
            match_status=MatchStatus(version["status"]),
            team_ids=(match["home_team_id"], match["away_team_id"]),
            snapshots=snapshots,
            result_90_present=result_present,
            team_stat_fields=team_fields,
            starters={team_id: frozenset(players) for team_id, players in starters.items()},
            player_observation_ids=frozenset(player_observations),
            match_id=match_id.value,
            match_version=int(version["version"]),
            result_90_known_at=result_known_at,
            team_stats_known_at=team_stats_known_at,
            player_observations=player_observations,
            as_of=as_of,
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


def load_verified_match_result(
    source_ref: str,
    *,
    archive: RawArchive,
    canonical: CanonicalStore,
    _connection: sqlite3.Connection | None = None,
) -> MatchResult90:
    """Verify result identity, exact evidence binding, and registered raw bytes/timestamp."""

    _require_text(source_ref, "source_ref")
    if _connection is None:
        with canonical.connect() as connection:
            row, raw_row, evidence_row = _match_result_rows(source_ref, connection)
    else:
        row, raw_row, evidence_row = _match_result_rows(source_ref, _connection)

    if any(
        not isinstance(row[field], str)
        for field in ("record_id", "match_id", "known_at", "observed_at", "raw_asset_id")
    ) or any(
        type(row[field]) is not int for field in ("match_version", "home_goals", "away_goals")
    ):
        raise ValueError("canonical result fact contains invalid persisted field types")
    payload = {
        "match_id": row["match_id"],
        "match_version": row["match_version"],
        "home_goals": row["home_goals"],
        "away_goals": row["away_goals"],
        "known_at": row["known_at"],
        "observed_at": row["observed_at"],
        "raw_asset_id": row["raw_asset_id"],
    }
    expected_ref = _record_id("match_results_90", _semantic_payload(payload))
    if expected_ref != source_ref:
        raise ValueError("canonical result fact content ID does not match persisted semantics")
    if evidence_row is None:
        raise ValueError("canonical result fact lacks its exact raw evidence binding")
    if raw_row is None:
        raise ValueError("canonical result fact raw asset is not registered")
    if (
        any(
            not isinstance(raw_row[field], str)
            for field in (
                "source",
                "source_id",
                "url",
                "observed_at",
                "checksum",
                "collector_version",
                "media_type",
            )
        )
        or (
            raw_row["target_event_time"] is not None
            and not isinstance(raw_row["target_event_time"], str)
        )
        or type(raw_row["size_bytes"]) is not int
    ):
        raise ValueError("canonical result raw registration contains invalid field types")

    asset_id = RawAssetId(payload["raw_asset_id"])
    asset = archive.load(asset_id)
    archive.verify(asset)
    persisted_raw = (
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
    archived_raw = (
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
    if persisted_raw != archived_raw:
        raise ValueError("canonical result raw registration conflicts with the raw archive")
    if payload["observed_at"] != _timestamp(asset.observed_at):
        raise ValueError("canonical result observed_at does not match its raw evidence")

    return MatchResult90(
        match_id=MatchId(payload["match_id"]),
        home_goals=payload["home_goals"],
        away_goals=payload["away_goals"],
        known_at=_parse_timestamp(payload["known_at"]),
        source_ref=source_ref,
    )


def _match_result_rows(
    source_ref: str,
    connection: sqlite3.Connection,
) -> tuple[sqlite3.Row, sqlite3.Row | None, sqlite3.Row | None]:
    row = connection.execute(
        "SELECT record_id, match_id, match_version, home_goals, away_goals, "
        "known_at, observed_at, raw_asset_id FROM match_results_90 "
        "WHERE record_id = ?",
        (source_ref,),
    ).fetchone()
    if row is None:
        raise ValueError("result source_ref does not identify a canonical fact")
    raw_row = connection.execute(
        "SELECT source, source_id, url, observed_at, target_event_time, checksum, "
        "collector_version, media_type, size_bytes FROM raw_assets "
        "WHERE raw_asset_id = ?",
        (row["raw_asset_id"],),
    ).fetchone()
    evidence_row = connection.execute(
        "SELECT 1 FROM fact_evidence WHERE record_id = ? AND raw_asset_id = ? AND observed_at = ?",
        (source_ref, row["raw_asset_id"], row["observed_at"]),
    ).fetchone()
    return row, raw_row, evidence_row


def verify_official_lineup_contract(
    contract_id: str,
    *,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> OfficialLineupContractEvidence:
    """Replay raw evidence against a content-addressed contract and canonical facts."""

    contract = canonical.official_lineup_contract(contract_id)
    if contract.contract_version != OFFICIAL_LINEUP_CONTRACT_VERSION:
        raise ValueError("legacy official lineup contracts cannot prove source-player bindings")
    asset = archive.load(contract.raw_asset_id)
    content = archive.read(asset)
    parsed = parse_official_lineup_json(content)
    if (
        parsed.parser_version != contract.parser_version
        or asset.source != contract.source
        or asset.source_id != contract.source_match_id
        or asset.observed_at != contract.observed_at
        or (
            asset.target_event_time is not None and asset.target_event_time != contract.published_at
        )
        or asset.collector_version != contract.parser_version
        or asset.media_type != "application/json"
        or parsed.source != contract.source
        or parsed.source_match_id != contract.source_match_id
        or parsed.match_mapping_source != contract.match_mapping_source
        or parsed.team_mapping_source != contract.team_mapping_source
        or parsed.player_mapping_source != contract.player_mapping_source
        or parsed.published_at != contract.published_at
    ):
        raise ValueError("official lineup raw replay conflicts with its persisted contract")

    mapped_match_id = canonical.mapped_match_ids(
        source=parsed.match_mapping_source,
        source_ids=(parsed.source_match_id,),
    ).get(parsed.source_match_id)
    if mapped_match_id != contract.match_id:
        raise ValueError("official lineup replay match mapping conflicts with its contract")
    versions = {version.version for version in canonical.match_versions(contract.match_id)}
    if contract.match_version not in versions:
        raise ValueError("official lineup replay match version no longer exists")

    replayed_lineups: list[tuple[TeamId, tuple[PlayerId, ...]]] = []
    replayed_bindings: list[OfficialLineupSourceBinding] = []
    for parsed_team in parsed.teams:
        team = canonical.mapped_team(
            source=parsed.team_mapping_source,
            source_id=parsed_team.source_team_id,
        )
        players: list[PlayerId] = []
        for player in parsed_team.starters:
            player_id = canonical.mapped_player(
                source=parsed.player_mapping_source,
                source_id=player.source_player_id,
            ).id
            players.append(player_id)
            replayed_bindings.append(
                OfficialLineupSourceBinding(
                    source_team_id=parsed_team.source_team_id,
                    source_player_id=player.source_player_id,
                    team_id=team.id,
                    player_id=player_id,
                )
            )
        replayed_lineups.append(
            (team.id, tuple(sorted(players, key=lambda player_id: player_id.value)))
        )
    normalized_bindings = tuple(
        sorted(
            replayed_bindings,
            key=lambda binding: (
                binding.source_team_id,
                binding.source_player_id,
                binding.team_id.value,
                binding.player_id.value,
            ),
        )
    )
    if (
        tuple(sorted(replayed_lineups)) != contract.team_lineups
        or normalized_bindings != contract.source_bindings
    ):
        raise ValueError("official lineup raw replay player mappings conflict with its contract")

    placeholders = ", ".join("?" for _ in contract.fact_ids)
    with canonical.connect() as connection:
        rows = connection.execute(
            "SELECT record_id, match_id, match_version, team_id, player_id, lineup_role, "
            "official, known_at, observed_at, raw_asset_id FROM lineup_facts "
            f"WHERE record_id IN ({placeholders})",
            contract.fact_ids,
        ).fetchall()
        evidence_rows = connection.execute(
            "SELECT record_id, raw_asset_id, observed_at FROM fact_evidence "
            f"WHERE record_id IN ({placeholders})",
            contract.fact_ids,
        ).fetchall()
    expected_assignments = {
        (team_id.value, player_id.value)
        for team_id, player_ids in contract.team_lineups
        for player_id in player_ids
    }
    evidence = {
        (row["record_id"], row["raw_asset_id"], row["observed_at"]) for row in evidence_rows
    }
    contract_evidence = {
        (
            fact_id,
            contract.raw_asset_id.value,
            _timestamp(contract.observed_at),
        )
        for fact_id in contract.fact_ids
    }
    fact_provenance = {(row["record_id"], row["raw_asset_id"], row["observed_at"]) for row in rows}
    fact_identity_mismatches = tuple(
        row["record_id"]
        for row in rows
        if row["record_id"]
        != _record_id(
            "lineup",
            _semantic_payload(
                {
                    "match_id": row["match_id"],
                    "match_version": row["match_version"],
                    "team_id": row["team_id"],
                    "player_id": row["player_id"],
                    "lineup_role": row["lineup_role"],
                    "official": row["official"],
                    "known_at": row["known_at"],
                    "observed_at": row["observed_at"],
                    "raw_asset_id": row["raw_asset_id"],
                }
            ),
        )
    )
    if (
        len(rows) != 22
        or {row["record_id"] for row in rows} != set(contract.fact_ids)
        or fact_identity_mismatches
        or {row["match_id"] for row in rows} != {contract.match_id.value}
        or {row["match_version"] for row in rows} != {contract.match_version}
        or {(row["team_id"], row["player_id"]) for row in rows} != expected_assignments
        or {row["lineup_role"] for row in rows} != {"starter"}
        or {row["official"] for row in rows} != {1}
        or {row["known_at"] for row in rows} != {_timestamp(contract.published_at)}
        or not contract_evidence <= evidence
        or not fact_provenance <= evidence
    ):
        raise ValueError("official lineup contract does not match canonical official lineup facts")
    return contract


def _validate_metrics(values: dict[str, float | int | None], name: str) -> None:
    for key, value in values.items():
        if not key:
            raise ValueError(f"{name} contains an empty key")
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} value {key!r} must be finite or None")
    _json_text(values)


def _require_match(connection, match_id: MatchId):
    row = connection.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches WHERE match_id = ?",
        (match_id.value,),
    ).fetchone()
    if row is None:
        raise KeyError(f"match {match_id} does not exist")
    return row


def _require_match_version(
    connection,
    *,
    match_id: MatchId,
    match_version: int,
    require_finished: bool = False,
):
    if isinstance(match_version, bool) or not isinstance(match_version, int) or match_version < 1:
        raise ValueError("match_version must be a positive integer")
    row = connection.execute(
        "SELECT m.match_id, m.home_team_id, m.away_team_id, v.status, v.kickoff_at "
        "FROM matches AS m JOIN match_versions AS v ON v.match_id = m.match_id "
        "WHERE m.match_id = ? AND v.version = ?",
        (match_id.value, match_version),
    ).fetchone()
    if row is None:
        raise KeyError(f"match version {match_id}:{match_version} does not exist")
    if require_finished and row["status"] != MatchStatus.FINISHED.value:
        raise ValueError("canonical fact requires a finished match version")
    return row


def _require_match_team(match_row, team_id: TeamId) -> None:
    if team_id.value not in {match_row["home_team_id"], match_row["away_team_id"]}:
        raise ValueError(f"team {team_id} does not belong to match {match_row['match_id']}")


def _require_registered_player(connection, player_id: PlayerId) -> None:
    row = connection.execute(
        "SELECT entity_type FROM entities WHERE entity_id = ?", (player_id.value,)
    ).fetchone()
    if row is None:
        raise KeyError(f"player {player_id} is not registered")
    if row["entity_type"] != "player":
        raise ValueError(f"entity {player_id} is not a player")


def _require_player_assignment(
    connection,
    *,
    match_id: MatchId,
    match_version: int | None,
    player_id: PlayerId,
    team_id: TeamId,
) -> None:
    predicates = ["match_id = ?", "player_id = ?", "team_id <> ?"]
    parameters: list[object] = [match_id.value, player_id.value, team_id.value]
    if match_version is not None:
        predicates.append("match_version = ?")
        parameters.append(match_version)
    where = " AND ".join(predicates)
    tables = ("player_match_observations", "lineup_facts")
    for table in tables:
        row = connection.execute(
            f"SELECT team_id FROM {table} WHERE {where} LIMIT 1", tuple(parameters)
        ).fetchone()
        if row is not None:
            raise ValueError(
                f"player {player_id} is already assigned to {row['team_id']} for match {match_id}"
            )


def _require_existing_player_assignment(
    connection,
    *,
    match_row,
    match_id: MatchId,
    match_version: int,
    player_id: PlayerId,
    team_id: TeamId | None,
    known_at_cutoff: datetime,
    observed_at_cutoff: datetime,
) -> None:
    known_at_text = _timestamp(known_at_cutoff)
    observed_at_text = _timestamp(observed_at_cutoff)
    rows = connection.execute(
        "SELECT team_id FROM player_match_observations "
        "WHERE match_id = ? AND match_version = ? AND player_id = ? "
        "AND known_at <= ? AND observed_at <= ? "
        "UNION SELECT team_id FROM lineup_facts "
        "WHERE match_id = ? AND match_version = ? AND player_id = ? "
        "AND known_at <= ? AND observed_at <= ?",
        (
            match_id.value,
            match_version,
            player_id.value,
            known_at_text,
            observed_at_text,
            match_id.value,
            match_version,
            player_id.value,
            known_at_text,
            observed_at_text,
        ),
    ).fetchall()
    assigned_team_ids = {row["team_id"] for row in rows}
    if not assigned_team_ids:
        raise ValueError(
            f"player {player_id} has no assignment for match {match_id} version "
            f"{match_version} visible at {observed_at_text}"
        )

    participant_team_ids = {match_row["home_team_id"], match_row["away_team_id"]}
    if not assigned_team_ids <= participant_team_ids:
        raise ValueError(f"player {player_id} has an assignment outside match {match_id}")
    if len(assigned_team_ids) != 1:
        raise ValueError(f"player {player_id} has conflicting assignments for match {match_id}")

    assigned_team_id = next(iter(assigned_team_ids))
    if team_id is not None and assigned_team_id != team_id.value:
        raise ValueError(
            f"player {player_id} is assigned to {assigned_team_id} for match {match_id}, "
            f"not {team_id}"
        )


def _visible_match_version(
    connection,
    *,
    match_id: MatchId,
    visible_at: datetime,
) -> int:
    visible_at_text = _timestamp(visible_at)
    row = connection.execute(
        "SELECT version FROM match_versions WHERE match_id = ? AND observed_at <= ? "
        "ORDER BY version DESC LIMIT 1",
        (match_id.value, visible_at_text),
    ).fetchone()
    if row is None:
        raise ValueError(f"match {match_id} has no version visible at {visible_at_text}")
    return int(row["version"])


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


def _insert_lineup_payload(connection, payload: dict[str, Any]) -> StoredFact:
    record_id = _record_id("lineup", _semantic_payload(payload))
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
        raw_asset_id=payload["raw_asset_id"],
        observed_at=payload["observed_at"],
    )
    return StoredFact(record_id, None)


def _insert_exact_payload(
    connection,
    *,
    table: str,
    primary_key: str,
    payload: dict[str, Any],
) -> None:
    row = connection.execute(
        f"SELECT * FROM {table} WHERE {primary_key} = ?",
        (payload[primary_key],),
    ).fetchone()
    if row is not None:
        differences = {
            key: (row[key], value) for key, value in payload.items() if row[key] != value
        }
        if differences:
            raise ValueError(f"{table} {payload[primary_key]} conflicts: {differences}")
        return
    columns = ", ".join(payload)
    placeholders = ", ".join("?" for _ in payload)
    connection.execute(
        f"INSERT INTO {table}({columns}) VALUES ({placeholders})",
        tuple(payload.values()),
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


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
