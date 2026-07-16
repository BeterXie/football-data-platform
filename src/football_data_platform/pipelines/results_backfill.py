"""Backfill registered fixtures and 90-minute results from a free backup source."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.pipelines.schedule import (
    SeasonCoverage,
    assess_season_coverage,
)
from football_data_platform.sources.football_data_csv import (
    COLLECTOR_VERSION,
    parse_results_csv,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.raw import RawArchive


@dataclass(frozen=True, slots=True)
class ResultsBackfillResult:
    raw_asset_id: str
    canonical_match_ids: tuple[str, ...]
    result_fact_ids: tuple[str, ...]
    coverage: SeasonCoverage
    diagnostic_codes: tuple[str, ...]


def ingest_results_backfill(
    content: bytes,
    *,
    page_url: str,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> ResultsBackfillResult:
    asset = archive.archive(
        content,
        source="football-data",
        source_id=(
            f"{season.source('football-data').competition_id}:"
            f"{season.source('football-data').season_id}"
        ),
        url=page_url,
        observed_at=observed_at,
        target_event_time=None,
        collector_version=COLLECTOR_VERSION,
        media_type="text/csv",
    )
    canonical.register_raw_asset(asset)
    parsed = parse_results_csv(
        content,
        competition=competition,
        season=season,
    )
    facts = CanonicalFactStore(canonical)
    match_ids: list[str] = []
    result_ids: list[str] = []
    for fixture in parsed.matches:
        home = canonical.resolve_or_create_team(
            source="football-data",
            source_id=fixture.home_source_id,
            canonical_name=fixture.home_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        away = canonical.resolve_or_create_team(
            source="football-data",
            source_id=fixture.away_source_id,
            canonical_name=fixture.away_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        match, version = canonical.resolve_or_create_match(
            source="football-data-schedule",
            source_id=fixture.source_fixture_id,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff_at=fixture.kickoff_at,
            status=fixture.status,
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        match_ids.append(match.id.value)
        if fixture.home_goals is not None and fixture.away_goals is not None:
            if fixture.kickoff_at is None:
                continue
            result = facts.append_result_90(
                match_id=match.id,
                match_version=version.version,
                home_goals=fixture.home_goals,
                away_goals=fixture.away_goals,
                known_at=fixture.kickoff_at + timedelta(hours=3),
                observed_at=observed_at,
                raw_asset_id=asset.id,
            )
            result_ids.append(result.record_id)
    coverage = assess_season_coverage(
        parsed,
        season,
        attempted_fixture_ids=set(),
    )
    return ResultsBackfillResult(
        asset.id.value,
        tuple(match_ids),
        tuple(result_ids),
        coverage,
        tuple(sorted({diagnostic.code for diagnostic in parsed.diagnostics})),
    )
