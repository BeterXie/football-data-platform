"""Archive, parse, normalize, and validate a registered season schedule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.sources.fbref import (
    COLLECTOR_VERSION,
    ScheduleParseResult,
    parse_schedule,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.raw import RawArchive


@dataclass(frozen=True, slots=True)
class SeasonCoverage:
    expected_matches: int
    actual_matches: int
    expected_teams: int
    actual_teams: int
    duplicate_fixture_ids: tuple[str, ...]
    missing_collection_attempts: tuple[str, ...]
    blocking_diagnostics: tuple[str, ...]

    @property
    def schedule_complete(self) -> bool:
        return (
            self.actual_matches == self.expected_matches
            and self.actual_teams == self.expected_teams
            and not self.duplicate_fixture_ids
            and not self.blocking_diagnostics
        )

    @property
    def complete(self) -> bool:
        return self.schedule_complete and not self.missing_collection_attempts


@dataclass(frozen=True, slots=True)
class ScheduleIngestResult:
    parsed: ScheduleParseResult
    coverage: SeasonCoverage
    raw_asset_id: str
    canonical_match_ids: tuple[str, ...]


def ingest_fbref_schedule(
    content: bytes,
    *,
    page_url: str,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> ScheduleIngestResult:
    """Run the schedule slice without allowing the parser to bypass raw storage."""

    raw_asset = archive.archive(
        content,
        source="fbref",
        source_id=f"{season.source('fbref').competition_id}:{season.source('fbref').season_id}",
        url=page_url,
        observed_at=observed_at,
        target_event_time=None,
        collector_version=COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(raw_asset)
    parsed = parse_schedule(
        content,
        competition=competition,
        season=season,
        page_url=page_url,
    )
    match_ids: list[str] = []
    for fixture in parsed.matches:
        home = canonical.resolve_or_create_team(
            source="fbref",
            source_id=fixture.home_source_id,
            canonical_name=fixture.home_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        away = canonical.resolve_or_create_team(
            source="fbref",
            source_id=fixture.away_source_id,
            canonical_name=fixture.away_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match, _ = canonical.resolve_or_create_match(
            source="fbref-schedule",
            source_id=fixture.source_fixture_id,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff_at=fixture.kickoff_at,
            status=fixture.status,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match_ids.append(match.id.value)
    coverage = assess_season_coverage(parsed, season)
    return ScheduleIngestResult(parsed, coverage, raw_asset.id.value, tuple(match_ids))


def assess_season_coverage(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
    *,
    attempted_fixture_ids: set[str] | None = None,
) -> SeasonCoverage:
    fixture_ids = [match.source_fixture_id for match in parsed.matches]
    duplicate_diagnostics = {
        diagnostic.subject_id
        for diagnostic in parsed.diagnostics
        if diagnostic.code == "duplicate_fixture_id" and diagnostic.subject_id is not None
    }
    duplicates = tuple(
        sorted(
            {item for item in fixture_ids if fixture_ids.count(item) > 1} | duplicate_diagnostics
        )
    )
    team_ids = {
        source_id
        for match in parsed.matches
        for source_id in (match.home_source_id, match.away_source_id)
    }
    attempted = set() if attempted_fixture_ids is None else attempted_fixture_ids
    return SeasonCoverage(
        expected_matches=season.expected_matches,
        actual_matches=len(parsed.matches),
        expected_teams=season.expected_teams,
        actual_teams=len(team_ids),
        duplicate_fixture_ids=duplicates,
        missing_collection_attempts=tuple(sorted(set(fixture_ids) - attempted)),
        blocking_diagnostics=tuple(sorted({item.code for item in parsed.diagnostics})),
    )
