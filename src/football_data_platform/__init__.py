"""Core contracts for the football data platform."""

from football_data_platform.config import (
    CompetitionDefinition,
    CompetitionRegistry,
    SeasonDefinition,
    SourceSeasonReference,
    load_competition_registry,
)

__all__ = [
    "CompetitionDefinition",
    "CompetitionRegistry",
    "SeasonDefinition",
    "SourceSeasonReference",
    "load_competition_registry",
]
