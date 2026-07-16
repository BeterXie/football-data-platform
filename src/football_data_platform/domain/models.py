"""Immutable core domain contracts shared across adapters and storage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from football_data_platform.domain.ids import (
    CompetitionId,
    EntityId,
    MatchId,
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

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("version must be positive")
        if self.kickoff_at is not None:
            require_utc(self.kickoff_at, "kickoff_at")
        require_utc(self.observed_at, "observed_at")


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

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.source_id, "source_id")
        _require_text(self.created_by, "created_by")
        _require_text(self.audit_note, "audit_note")
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


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
