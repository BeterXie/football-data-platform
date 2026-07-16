"""Archive and normalize a single FBref match report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.ids import MatchId
from football_data_platform.sources.fbref_match_report import (
    MatchReportParseResult,
    parse_match_report,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.raw import RawArchive

COLLECTOR_VERSION = "fbref-match-report/1"


@dataclass(frozen=True, slots=True)
class MatchReportIngestResult:
    raw_asset_id: str
    parsed: MatchReportParseResult
    team_fact_ids: tuple[str, ...]
    player_fact_ids: tuple[str, ...]
    result_fact_id: str


def ingest_fbref_match_report(
    content: bytes,
    *,
    page_url: str,
    source_match_id: str,
    match_id: MatchId,
    match_version: int,
    home_goals: int,
    away_goals: int,
    known_at: datetime,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> MatchReportIngestResult:
    """Persist raw evidence before emitting canonical result and player facts."""

    asset = archive.archive(
        content,
        source="fbref",
        source_id=source_match_id,
        url=page_url,
        observed_at=observed_at,
        target_event_time=known_at,
        collector_version=COLLECTOR_VERSION,
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    parsed = parse_match_report(content)
    facts = CanonicalFactStore(canonical)
    result = facts.append_result_90(
        match_id=match_id,
        match_version=match_version,
        home_goals=home_goals,
        away_goals=away_goals,
        known_at=known_at,
        observed_at=observed_at,
        raw_asset_id=asset.id,
    )
    team_fact_ids: list[str] = []
    player_fact_ids: list[str] = []
    match = canonical.match(match_id)
    for team_report in parsed.teams:
        team = canonical.mapped_team(source="fbref", source_id=team_report.source_team_id)
        if team.id == match.home_team_id:
            team_goals = home_goals
        elif team.id == match.away_team_id:
            team_goals = away_goals
        else:
            raise ValueError(f"report team {team.id} does not belong to match {match_id}")
        team_fact = facts.append_team_observation(
            match_id=match_id,
            match_version=match_version,
            team_id=team.id,
            stats={**team_report.aggregated_stats, "goals": team_goals},
            known_at=known_at,
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        team_fact_ids.append(team_fact.record_id)
        for player_row in team_report.players:
            player = canonical.resolve_or_create_player(
                source="fbref",
                source_id=player_row.source_player_id,
                canonical_name=player_row.player_name,
                observed_at=observed_at,
                raw_asset_id=asset.id,
            )
            player_fact = facts.append_player_observation(
                match_id=match_id,
                match_version=match_version,
                team_id=team.id,
                player_id=player.id,
                role=player_row.role,
                minutes=player_row.minutes,
                metrics=player_row.metrics,
                known_at=known_at,
                observed_at=observed_at,
                raw_asset_id=asset.id,
            )
            player_fact_ids.append(player_fact.record_id)
            if player_row.starter is not None:
                facts.append_lineup_fact(
                    match_id=match_id,
                    match_version=match_version,
                    team_id=team.id,
                    player_id=player.id,
                    lineup_role="starter" if player_row.starter else "bench",
                    official=True,
                    known_at=known_at,
                    observed_at=observed_at,
                    raw_asset_id=asset.id,
                )
    return MatchReportIngestResult(
        asset.id.value,
        parsed,
        tuple(team_fact_ids),
        tuple(player_fact_ids),
        result.record_id,
    )
