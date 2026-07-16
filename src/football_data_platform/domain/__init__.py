"""Stable domain identifiers and immutable platform contracts."""

from football_data_platform.domain.ids import (
    CompetitionId,
    MatchId,
    PlatformId,
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
    SourceMapping,
)

__all__ = [
    "CompetitionId",
    "MappingRule",
    "Match",
    "MatchId",
    "MatchStatus",
    "MatchVersion",
    "PlatformId",
    "PlayerId",
    "RawAsset",
    "RawAssetId",
    "SeasonId",
    "SourceMapping",
    "TeamId",
]
