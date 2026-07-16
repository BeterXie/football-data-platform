"""Competition and season registry loaded from operator-owned TOML configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType

from football_data_platform.domain.ids import CompetitionId, SeasonId

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "resources" / "competitions.toml"


@dataclass(frozen=True, slots=True)
class SourceSeasonReference:
    """Provider-owned identifiers for one registered competition season."""

    source: str
    competition_id: str
    season_id: str

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.competition_id, "competition_id")
        _require_text(self.season_id, "season_id")


@dataclass(frozen=True, slots=True)
class SeasonDefinition:
    id: SeasonId
    label: str
    starts_on: date
    ends_on: date
    expected_teams: int
    expected_matches: int
    sources: tuple[SourceSeasonReference, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.label, "label")
        if self.ends_on < self.starts_on:
            raise ValueError("ends_on must not precede starts_on")
        if self.expected_teams < 2:
            raise ValueError("expected_teams must be at least 2")
        if self.expected_matches < 1:
            raise ValueError("expected_matches must be positive")
        source_names = [reference.source for reference in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError(f"duplicate source in season {self.id}")

    def source(self, name: str) -> SourceSeasonReference:
        """Return this season's reference for a provider."""

        for reference in self.sources:
            if reference.source == name:
                return reference
        raise KeyError(f"source {name!r} is not registered for season {self.id}")


@dataclass(frozen=True, slots=True)
class CompetitionDefinition:
    id: CompetitionId
    name: str
    country_code: str
    kind: str
    timezone: str
    seasons: tuple[SeasonDefinition, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.kind, "kind")
        _require_text(self.timezone, "timezone")
        if len(self.country_code) != 3 or not self.country_code.isalpha():
            raise ValueError("country_code must be a three-letter code")
        season_ids = [season.id for season in self.seasons]
        if len(season_ids) != len(set(season_ids)):
            raise ValueError(f"duplicate season in competition {self.id}")


@dataclass(frozen=True, slots=True)
class CompetitionRegistry:
    schema_version: int
    competitions: tuple[CompetitionDefinition, ...]
    _competitions_by_id: Mapping[CompetitionId, CompetitionDefinition] = field(
        init=False, repr=False
    )
    _seasons_by_id: Mapping[SeasonId, SeasonDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")

        competitions_by_id: dict[CompetitionId, CompetitionDefinition] = {}
        seasons_by_id: dict[SeasonId, SeasonDefinition] = {}
        for competition in self.competitions:
            if competition.id in competitions_by_id:
                raise ValueError(f"duplicate competition {competition.id}")
            competitions_by_id[competition.id] = competition
            for season in competition.seasons:
                if season.id in seasons_by_id:
                    raise ValueError(f"duplicate season {season.id}")
                seasons_by_id[season.id] = season

        object.__setattr__(self, "_competitions_by_id", MappingProxyType(competitions_by_id))
        object.__setattr__(self, "_seasons_by_id", MappingProxyType(seasons_by_id))

    def competition(self, competition_id: CompetitionId) -> CompetitionDefinition:
        try:
            return self._competitions_by_id[competition_id]
        except KeyError:
            raise KeyError(f"competition {competition_id} is not registered") from None

    def season(self, season_id: SeasonId) -> SeasonDefinition:
        try:
            return self._seasons_by_id[season_id]
        except KeyError:
            raise KeyError(f"season {season_id} is not registered") from None


def load_competition_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> CompetitionRegistry:
    """Load and validate a competition registry from TOML."""

    registry_path = Path(path)
    with registry_path.open("rb") as registry_file:
        payload = tomllib.load(registry_file)

    competitions_payload = payload.get("competitions", [])
    if not isinstance(competitions_payload, list):
        raise ValueError("competitions must be a list of TOML tables")
    competitions = tuple(_parse_competition(_require_table(item)) for item in competitions_payload)
    return CompetitionRegistry(
        schema_version=int(payload.get("schema_version", 0)),
        competitions=competitions,
    )


def _parse_competition(payload: Mapping[str, object]) -> CompetitionDefinition:
    seasons_payload = payload.get("seasons", [])
    if not isinstance(seasons_payload, list):
        raise ValueError("competition seasons must be a list")
    return CompetitionDefinition(
        id=CompetitionId(str(payload["id"])),
        name=str(payload["name"]),
        country_code=str(payload["country_code"]).upper(),
        kind=str(payload["kind"]),
        timezone=str(payload["timezone"]),
        seasons=tuple(_parse_season(_require_table(item)) for item in seasons_payload),
    )


def _parse_season(payload: Mapping[str, object]) -> SeasonDefinition:
    sources_payload = payload.get("sources", [])
    if not isinstance(sources_payload, list):
        raise ValueError("season sources must be a list")
    starts_on = payload["starts_on"]
    ends_on = payload["ends_on"]
    if type(starts_on) is not date or type(ends_on) is not date:
        raise ValueError("season dates must be TOML local dates")
    return SeasonDefinition(
        id=SeasonId(str(payload["id"])),
        label=str(payload["label"]),
        starts_on=starts_on,
        ends_on=ends_on,
        expected_teams=int(payload["expected_teams"]),
        expected_matches=int(payload["expected_matches"]),
        sources=tuple(_parse_source(_require_table(item)) for item in sources_payload),
    )


def _parse_source(payload: Mapping[str, object]) -> SourceSeasonReference:
    return SourceSeasonReference(
        source=str(payload["source"]),
        competition_id=str(payload["competition_id"]),
        season_id=str(payload["season_id"]),
    )


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")


def _require_table(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("registry entries must be TOML tables")
    return value
