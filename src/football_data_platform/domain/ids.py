"""Typed, platform-owned identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

_ID_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, order=True, slots=True)
class PlatformId:
    """Base value object for IDs that do not depend on provider names or display labels."""

    value: str
    prefix: ClassVar[str] = "platform"

    def __post_init__(self) -> None:
        expected_prefix = f"{self.prefix}:"
        if not self.value.startswith(expected_prefix):
            raise ValueError(f"{type(self).__name__} must start with {expected_prefix!r}")
        suffix = self.value.removeprefix(expected_prefix)
        if not _ID_SUFFIX.fullmatch(suffix):
            raise ValueError(f"invalid {type(self).__name__} value: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True, slots=True)
class CompetitionId(PlatformId):
    prefix: ClassVar[str] = "competition"


@dataclass(frozen=True, order=True, slots=True)
class SeasonId(PlatformId):
    prefix: ClassVar[str] = "season"


@dataclass(frozen=True, order=True, slots=True)
class TeamId(PlatformId):
    prefix: ClassVar[str] = "team"


@dataclass(frozen=True, order=True, slots=True)
class PlayerId(PlatformId):
    prefix: ClassVar[str] = "player"


@dataclass(frozen=True, order=True, slots=True)
class MatchId(PlatformId):
    prefix: ClassVar[str] = "match"


@dataclass(frozen=True, order=True, slots=True)
class RawAssetId(PlatformId):
    prefix: ClassVar[str] = "raw-asset"


@dataclass(frozen=True, order=True, slots=True)
class SnapshotId(PlatformId):
    prefix: ClassVar[str] = "snapshot"


@dataclass(frozen=True, order=True, slots=True)
class PredictionId(PlatformId):
    prefix: ClassVar[str] = "prediction"


@dataclass(frozen=True, order=True, slots=True)
class ModelRunId(PlatformId):
    prefix: ClassVar[str] = "model-run"


@dataclass(frozen=True, order=True, slots=True)
class MarketSnapshotId(PlatformId):
    prefix: ClassVar[str] = "market-snapshot"


EntityId: TypeAlias = CompetitionId | SeasonId | TeamId | PlayerId | MatchId
