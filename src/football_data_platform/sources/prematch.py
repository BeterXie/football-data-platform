"""Small, source-agnostic contracts for pre-match evidence adapters.

The adapters in this module deliberately accept already collected structured payloads.  Network
collection, authentication, and access-control handling belong to a source-specific collector;
these contracts only validate timestamps, source identity, and raw lineage references before a
payload is handed to the canonical fact store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from football_data_platform.domain.ids import MatchId, PlayerId, RawAssetId, TeamId
from football_data_platform.domain.models import require_utc


class SourceKind(StrEnum):
    """Supported pre-match source families."""

    NEWS = "news"
    INJURY = "injury"
    SUSPENSION = "suspension"
    OFFICIAL_LINEUP = "official-lineup"


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """Operator-owned identity and trust metadata for one source.

    ``independence_group`` is intentionally explicit: two URLs from the same wire service or
    syndicated publisher must not become two corroborating sources merely because their URLs
    differ.  An official flag is a registry fact, never a caller-provided event attribute.
    """

    name: str
    kind: SourceKind
    independence_group: str
    official: bool = False
    allowed_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.name, "source name")
        object.__setattr__(self, "kind", SourceKind(self.kind))
        _require_text(self.independence_group, "independence_group")
        if not isinstance(self.official, bool):
            raise TypeError("official must be a boolean")
        for host in self.allowed_hosts:
            _require_text(host, "allowed host")


class SourceRegistry:
    """Immutable-by-default registry used when calculating evidence strength.

    The registry is intentionally lightweight and in-memory.  A deployment can load it from
    configuration, while tests and import jobs can construct one directly without a schema
    migration.  Unknown sources are never official and do not contribute to corroboration;
    operators must register production sources to collapse syndicated aliases and enforce URL
    host checks.
    """

    def __init__(self, descriptors: Sequence[SourceDescriptor] = ()) -> None:
        self._descriptors: dict[str, SourceDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: SourceDescriptor) -> None:
        if not isinstance(descriptor, SourceDescriptor):
            raise TypeError("source registry entries must be SourceDescriptor values")
        previous = self._descriptors.get(descriptor.name)
        if previous is not None and previous != descriptor:
            raise ValueError(f"source {descriptor.name!r} is already registered differently")
        self._descriptors[descriptor.name] = descriptor

    def get(self, source: str) -> SourceDescriptor | None:
        return self._descriptors.get(source)

    def require(self, source: str) -> SourceDescriptor:
        descriptor = self.get(source)
        if descriptor is None:
            raise KeyError(f"source {source!r} is not registered")
        return descriptor

    def is_official(self, source: str) -> bool:
        descriptor = self.get(source)
        return descriptor is not None and descriptor.official

    def independence_group(self, source: str) -> str:
        descriptor = self.get(source)
        return descriptor.independence_group if descriptor is not None else source

    def validate_url(self, source: str, url: str) -> None:
        descriptor = self.get(source)
        if descriptor is None or not descriptor.allowed_hosts:
            return
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allowed = {item.lower().lstrip(".") for item in descriptor.allowed_hosts}
        if not host or not any(host == item or host.endswith("." + item) for item in allowed):
            raise ValueError(f"URL host for source {source!r} is not registered: {host!r}")

    def descriptors(self) -> tuple[SourceDescriptor, ...]:
        return tuple(self._descriptors.values())


@dataclass(frozen=True, slots=True)
class NewsEvidenceDTO:
    """Structured news evidence extracted from one archived raw observation."""

    source: str
    url: str
    title: str
    published_at: datetime
    observed_at: datetime
    raw_asset_id: RawAssetId
    language: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.url, "url")
        _require_text(self.title, "title")
        require_utc(self.published_at, "published_at")
        require_utc(self.observed_at, "observed_at")
        if self.published_at > self.observed_at:
            raise ValueError("published_at cannot be later than observed_at")
        if self.language is not None:
            _require_text(self.language, "language")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source: str,
        raw_asset_id: RawAssetId,
        observed_at: datetime,
    ) -> NewsEvidenceDTO:
        """Adapt a parser-owned mapping without accepting generated summaries as evidence."""

        if "title" not in payload or "published_at" not in payload:
            raise ValueError("news payload requires title and published_at")
        if any(payload.get(key) for key in ("generated_summary", "is_generated", "auto_summary")):
            raise ValueError("generated summaries are not source evidence")
        return cls(
            source=source,
            url=str(payload.get("url", "")),
            title=str(payload["title"]),
            published_at=_coerce_datetime(payload["published_at"], "published_at"),
            observed_at=observed_at,
            raw_asset_id=raw_asset_id,
            language=(str(payload["language"]) if payload.get("language") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class AdapterDiagnostic:
    """Structured failure information that a collector can persist alongside an attempt."""

    source: str
    kind: SourceKind
    code: str
    message: str
    observed_at: datetime
    raw_asset_id: RawAssetId | None = None

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        object.__setattr__(self, "kind", SourceKind(self.kind))
        _require_text(self.code, "diagnostic code")
        _require_text(self.message, "diagnostic message")
        require_utc(self.observed_at, "observed_at")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "kind": self.kind.value,
            "code": self.code,
            "message": self.message,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
            "raw_asset_id": self.raw_asset_id.value if self.raw_asset_id is not None else None,
        }


@dataclass(frozen=True, slots=True)
class PrematchEventDTO:
    """A candidate injury, suspension, rotation, or other pre-match event."""

    match_id: MatchId
    team_id: TeamId | None
    player_id: PlayerId | None
    event_type: str
    occurred_at: datetime | None
    known_at: datetime
    evidence_refs: tuple[str, ...]
    requested_confirmation_status: str | None = None
    as_of: datetime | None = None

    @property
    def confirmation_status(self) -> str | None:
        """Backward-compatible name for a caller's requested, never trusted, status."""

        return self.requested_confirmation_status

    def __post_init__(self) -> None:
        _require_text(self.event_type, "event_type")
        require_utc(self.known_at, "known_at")
        if self.occurred_at is not None:
            require_utc(self.occurred_at, "occurred_at")
            if self.occurred_at > self.known_at:
                raise ValueError("occurred_at cannot be later than known_at")
        if self.as_of is not None:
            require_utc(self.as_of, "as_of")
            if self.known_at > self.as_of:
                raise ValueError("known_at cannot be later than as_of")
        refs = tuple(dict.fromkeys(self.evidence_refs))
        if not refs or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("pre-match events require non-empty evidence references")
        object.__setattr__(self, "evidence_refs", refs)
        if (
            self.requested_confirmation_status is not None
            and self.requested_confirmation_status
            not in {"official", "corroborated", "unconfirmed"}
        ):
            raise ValueError("invalid requested_confirmation_status")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        match_id: MatchId,
        team_id: TeamId | None,
        player_id: PlayerId | None,
        event_type: str,
        known_at: datetime,
        evidence_refs: Sequence[str],
        as_of: datetime | None = None,
        requested_confirmation_status: str | None = None,
    ) -> PrematchEventDTO:
        occurred = payload.get("occurred_at")
        return cls(
            match_id=match_id,
            team_id=team_id,
            player_id=player_id,
            event_type=event_type,
            occurred_at=(
                _coerce_datetime(occurred, "occurred_at") if occurred is not None else None
            ),
            known_at=known_at,
            evidence_refs=tuple(evidence_refs),
            requested_confirmation_status=requested_confirmation_status,
            as_of=as_of,
        )


@dataclass(frozen=True, slots=True)
class OfficialLineupDTO:
    """Official starting XI with its publication and raw archive references."""

    match_id: MatchId
    team_id: TeamId
    player_ids: tuple[PlayerId, ...]
    source: str
    published_at: datetime
    observed_at: datetime
    raw_asset_id: RawAssetId
    match_version: int = 1
    url: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        require_utc(self.published_at, "published_at")
        require_utc(self.observed_at, "observed_at")
        if self.published_at > self.observed_at:
            raise ValueError("published_at cannot be later than observed_at")
        if isinstance(self.match_version, bool) or self.match_version < 1:
            raise ValueError("match_version must be a positive integer")
        if self.url is not None:
            _require_text(self.url, "url")
        if len(self.player_ids) != 11 or len(set(self.player_ids)) != 11:
            raise ValueError("official lineup requires 11 unique player IDs")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        match_id: MatchId,
        team_id: TeamId,
        source: str,
        raw_asset_id: RawAssetId,
        match_version: int = 1,
        url: str | None = None,
        observed_at: datetime,
    ) -> OfficialLineupDTO:
        players = payload.get("player_ids")
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
            raise ValueError("official lineup payload requires player_ids")
        published = payload.get("published_at")
        if published is None:
            raise ValueError("official lineup payload requires published_at")
        return cls(
            match_id=match_id,
            team_id=team_id,
            player_ids=tuple(
                item if isinstance(item, PlayerId) else PlayerId(str(item)) for item in players
            ),
            source=source,
            published_at=_coerce_datetime(published, "published_at"),
            observed_at=observed_at,
            raw_asset_id=raw_asset_id,
            match_version=match_version,
            url=url,
        )


def validate_official_lineups(
    home: OfficialLineupDTO,
    away: OfficialLineupDTO,
    registry: SourceRegistry,
) -> None:
    """Validate the paired official lineups before either is written to canonical storage."""

    if home.match_id != away.match_id:
        raise ValueError("official lineups must reference the same match")
    if home.team_id == away.team_id:
        raise ValueError("official lineups must reference different teams")
    if home.match_version != away.match_version:
        raise ValueError("official lineups must reference the same match version")
    for lineup in (home, away):
        descriptor = registry.get(lineup.source)
        if descriptor is None or not descriptor.official:
            raise ValueError(f"lineup source {lineup.source!r} is not a verified official source")
        if descriptor.kind is not SourceKind.OFFICIAL_LINEUP:
            raise ValueError(f"source {lineup.source!r} is not an official-lineup adapter")
        if lineup.url is not None:
            registry.validate_url(lineup.source, lineup.url)
    if home.published_at > home.observed_at or away.published_at > away.observed_at:
        raise ValueError("official lineup publication cannot follow observation")


def classify_confirmation(
    sources: Sequence[str],
    registry: SourceRegistry | None = None,
    *,
    requested: str | None = None,
) -> str:
    """Compute confirmation strength from registered source facts.

    The strongest registered official source wins.  Otherwise, two distinct independence groups
    are required for ``corroborated``.  A requested status can only ask for a weaker result; a
    stronger unproven claim raises instead of silently persisting a false status.
    """

    registry = registry or SourceRegistry()
    source_names = tuple(dict.fromkeys(sources))
    if not source_names or any(
        not isinstance(source, str) or not source for source in source_names
    ):
        raise ValueError("confirmation requires at least one source")
    known_sources = tuple(source for source in source_names if registry.get(source) is not None)
    computed = (
        "official"
        if any(registry.is_official(source) for source in known_sources)
        else (
            "corroborated"
            if len({registry.independence_group(source) for source in known_sources}) >= 2
            else "unconfirmed"
        )
    )
    if requested is not None:
        if requested not in {"official", "corroborated", "unconfirmed"}:
            raise ValueError("invalid confirmation_status")
        rank = {"unconfirmed": 0, "corroborated": 1, "official": 2}
        if rank[requested] > rank[computed]:
            message = (
                f"confirmation_status {requested!r} is unsupported by evidence; "
                f"computed {computed!r}"
            )
            raise ValueError(message)
    return computed


def adapt_news_evidence(
    payload: Mapping[str, Any],
    *,
    source: str,
    raw_asset_id: RawAssetId,
    observed_at: datetime,
) -> NewsEvidenceDTO:
    """Adapt a parsed news object after its original response was archived."""

    return NewsEvidenceDTO.from_mapping(
        payload,
        source=source,
        raw_asset_id=raw_asset_id,
        observed_at=observed_at,
    )


def adapt_injury_event(
    payload: Mapping[str, Any],
    *,
    match_id: MatchId,
    team_id: TeamId | None,
    player_id: PlayerId | None,
    known_at: datetime,
    evidence_refs: Sequence[str],
    as_of: datetime | None = None,
) -> PrematchEventDTO:
    return PrematchEventDTO.from_mapping(
        payload,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        event_type="injury",
        known_at=known_at,
        evidence_refs=evidence_refs,
        as_of=as_of,
    )


def adapt_suspension_event(
    payload: Mapping[str, Any],
    *,
    match_id: MatchId,
    team_id: TeamId | None,
    player_id: PlayerId | None,
    known_at: datetime,
    evidence_refs: Sequence[str],
    as_of: datetime | None = None,
) -> PrematchEventDTO:
    return PrematchEventDTO.from_mapping(
        payload,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        event_type="suspension",
        known_at=known_at,
        evidence_refs=evidence_refs,
        as_of=as_of,
    )


def adapt_official_lineup(
    payload: Mapping[str, Any],
    *,
    match_id: MatchId,
    team_id: TeamId,
    source: str,
    raw_asset_id: RawAssetId,
    observed_at: datetime,
    match_version: int = 1,
    url: str | None = None,
) -> OfficialLineupDTO:
    return OfficialLineupDTO.from_mapping(
        payload,
        match_id=match_id,
        team_id=team_id,
        source=source,
        raw_asset_id=raw_asset_id,
        observed_at=observed_at,
        match_version=match_version,
        url=url,
    )


def _coerce_datetime(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        require_utc(value, name)
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp") from error
    require_utc(parsed, name)
    return parsed.astimezone(UTC)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
