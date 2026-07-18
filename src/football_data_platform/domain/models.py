"""Immutable core domain contracts shared across adapters and storage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MatchStatus(StrEnum):
    SCHEDULED = "scheduled"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class MappingRule(StrEnum):
    SOURCE_ID = "source_id"
    EXACT_ALIAS = "exact_alias"
    FUZZY_ALIAS = "fuzzy_alias"
    MANUAL_OVERRIDE = "manual_override"


class CollectionAttemptOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Match:
    id: MatchId
    competition_id: CompetitionId
    season_id: SeasonId
    home_team_id: TeamId
    away_team_id: TeamId

    def __post_init__(self) -> None:
        if self.home_team_id == self.away_team_id:
            raise ValueError("home and away teams must differ")


@dataclass(frozen=True, slots=True)
class MatchVersion:
    """A versioned view of mutable scheduling facts for a stable match identity."""

    match_id: MatchId
    version: int
    kickoff_at: datetime | None
    status: MatchStatus
    observed_at: datetime
    round_name: str | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("version must be positive")
        if self.kickoff_at is not None:
            require_utc(self.kickoff_at, "kickoff_at")
        require_utc(self.observed_at, "observed_at")
        if self.round_name is not None:
            _require_text(self.round_name, "round_name")


@dataclass(frozen=True, slots=True)
class CollectionAttempt:
    """An auditable source request targeting one canonical match."""

    id: CollectionAttemptId
    match_id: MatchId
    source: str
    target_url: str
    outcome: CollectionAttemptOutcome
    observed_at: datetime
    collector_version: str
    diagnostic_code: str | None
    diagnostic_message: str | None
    raw_asset_id: RawAssetId | None
    source_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        if self.source_id is not None:
            _require_text(self.source_id, "source_id")
        _require_text(self.target_url, "target_url")
        _require_text(self.collector_version, "collector_version")
        require_utc(self.observed_at, "observed_at")
        if self.diagnostic_code is not None:
            _require_text(self.diagnostic_code, "diagnostic_code")
        if self.diagnostic_message is not None:
            _require_text(self.diagnostic_message, "diagnostic_message")
        if self.outcome is CollectionAttemptOutcome.SUCCEEDED:
            if self.raw_asset_id is None:
                raise ValueError("successful collection attempts require raw_asset_id")
        elif self.diagnostic_code is None:
            raise ValueError("unsuccessful collection attempts require diagnostic_code")


@dataclass(frozen=True, slots=True)
class SourceMapping:
    """A versioned association from a provider identifier to a platform entity."""

    source: str
    source_id: str
    entity_id: EntityId
    version: int
    valid_from: datetime
    valid_to: datetime | None
    match_rule: MappingRule
    confidence: float
    created_at: datetime
    created_by: str
    audit_note: str
    mapping_id: str | None = None
    entity_type: str | None = None
    supersedes_mapping_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.source_id, "source_id")
        _require_text(self.created_by, "created_by")
        _require_text(self.audit_note, "audit_note")
        if self.mapping_id is not None:
            _require_text(self.mapping_id, "mapping_id")
        if self.entity_type is not None:
            _require_text(self.entity_type, "entity_type")
            if self.entity_type != _entity_type(self.entity_id):
                raise ValueError("source mapping entity_type does not match entity_id")
        if self.supersedes_mapping_id is not None:
            _require_text(self.supersedes_mapping_id, "supersedes_mapping_id")
            if self.mapping_id is None:
                raise ValueError("supersedes_mapping_id requires mapping_id")
        if self.version < 1:
            raise ValueError("version must be positive")
        require_utc(self.valid_from, "valid_from")
        require_utc(self.created_at, "created_at")
        if self.valid_to is not None:
            require_utc(self.valid_to, "valid_to")
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must follow valid_from")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")


@dataclass(frozen=True, slots=True)
class SourceMappingEvidence:
    """Immutable evidence attached to a mapping or unresolved candidate conflict."""

    evidence_id: str
    evidence_ref: str
    recorded_at: datetime
    recorded_by: str
    reason: str
    mapping_id: str | None = None
    conflict_id: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.evidence_id, "evidence_id"),
            (self.evidence_ref, "evidence_ref"),
            (self.recorded_by, "recorded_by"),
            (self.reason, "reason"),
        ):
            _require_text(value, field_name)
        require_utc(self.recorded_at, "recorded_at")
        if (self.mapping_id is None) == (self.conflict_id is None):
            raise ValueError("mapping evidence must reference exactly one mapping or conflict")
        if self.mapping_id is not None:
            _require_text(self.mapping_id, "mapping_id")
        if self.conflict_id is not None:
            _require_text(self.conflict_id, "conflict_id")


@dataclass(frozen=True, slots=True)
class SourceMappingConflict:
    """A persisted candidate that contradicts the current source mapping."""

    conflict_id: str
    source: str
    entity_type: str
    source_id: str
    current_mapping_id: str | None
    current_entity_id: EntityId | None
    candidate_entity_id: EntityId
    proposed_at: datetime
    proposed_by: str
    reason: str
    decision_id: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.conflict_id, "conflict_id"),
            (self.source, "source"),
            (self.entity_type, "entity_type"),
            (self.source_id, "source_id"),
            (self.proposed_by, "proposed_by"),
            (self.reason, "reason"),
        ):
            _require_text(value, field_name)
        require_utc(self.proposed_at, "proposed_at")
        if self.current_mapping_id is not None:
            _require_text(self.current_mapping_id, "current_mapping_id")
        if (self.current_mapping_id is None) != (self.current_entity_id is None):
            raise ValueError("mapping conflict current mapping and entity must both be present")
        if self.entity_type != _entity_type(self.candidate_entity_id) or (
            self.current_entity_id is not None
            and self.entity_type != _entity_type(self.current_entity_id)
        ):
            raise ValueError("mapping conflict entity_type does not match its candidates")
        if (
            self.current_entity_id is not None
            and self.current_entity_id == self.candidate_entity_id
        ):
            raise ValueError("mapping conflict candidate must differ from current entity")
        if self.decision_id is not None:
            _require_text(self.decision_id, "decision_id")


@dataclass(frozen=True, slots=True)
class SourceMappingDecision:
    """Append-only audit record for one accepted source mapping revision."""

    decision_id: str
    conflict_id: str
    previous_mapping_id: str | None
    new_mapping_id: str
    decided_at: datetime
    decided_by: str
    reason: str
    revision_event_id: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.decision_id, "decision_id"),
            (self.conflict_id, "conflict_id"),
            (self.new_mapping_id, "new_mapping_id"),
            (self.decided_by, "decided_by"),
            (self.reason, "reason"),
        ):
            _require_text(value, field_name)
        if self.previous_mapping_id is not None:
            _require_text(self.previous_mapping_id, "previous_mapping_id")
        if self.revision_event_id is not None:
            _require_text(self.revision_event_id, "revision_event_id")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("mapping decision evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            _require_text(evidence_id, "evidence_id")
        require_utc(self.decided_at, "decided_at")


@dataclass(frozen=True, slots=True)
class RawAsset:
    """Manifest contract for one immutable observation of source bytes."""

    id: RawAssetId
    source: str
    source_id: str
    url: str
    observed_at: datetime
    target_event_time: datetime | None
    checksum: str
    collector_version: str
    media_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.source_id, "source_id")
        _require_text(self.url, "url")
        _require_text(self.collector_version, "collector_version")
        _require_text(self.media_type, "media_type")
        require_utc(self.observed_at, "observed_at")
        if self.target_event_time is not None:
            require_utc(self.target_event_time, "target_event_time")
        if not _SHA256.fullmatch(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 hex digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")


def require_utc(value: datetime, field_name: str = "datetime") -> None:
    """Reject naive or non-UTC datetimes at a domain boundary."""

    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


def _entity_type(entity_id: EntityId) -> str:
    types = (
        (CompetitionId, "competition"),
        (SeasonId, "season"),
        (TeamId, "team"),
        (PlayerId, "player"),
        (MatchId, "match"),
    )
    for id_type, entity_type in types:
        if isinstance(entity_id, id_type):
            return entity_type
    raise TypeError("source mapping entity_id must be a canonical EntityId")


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
