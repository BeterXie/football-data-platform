"""Archive, parse, normalize, and validate a registered season schedule."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.domain.ids import MatchId
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
    unregistered_team_ids: tuple[str, ...]
    missing_registered_teams: tuple[str, ...]
    structural_violations: tuple[str, ...]
    missing_collection_attempts: tuple[str, ...]
    blocking_diagnostics: tuple[str, ...]

    @property
    def schedule_complete(self) -> bool:
        return (
            self.actual_matches == self.expected_matches
            and self.actual_teams == self.expected_teams
            and not self.duplicate_fixture_ids
            and not self.unregistered_team_ids
            and not self.missing_registered_teams
            and not self.structural_violations
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
    _require_round_robin_contract(season)
    for fixture in parsed.matches:
        home_definition = season.team("fbref", fixture.home_source_id)
        away_definition = season.team("fbref", fixture.away_source_id)
        home = canonical.resolve_registered_team(
            source="fbref",
            source_id=fixture.home_source_id,
            team_id=home_definition.id,
            canonical_name=home_definition.name,
            observed_name=fixture.home_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        away = canonical.resolve_registered_team(
            source="fbref",
            source_id=fixture.away_source_id,
            team_id=away_definition.id,
            canonical_name=away_definition.name,
            observed_name=fixture.away_name,
            competition_id=competition.id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match, _ = canonical.resolve_or_create_round_robin_match(
            source="fbref-schedule",
            source_id=fixture.source_fixture_id,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home.id,
            away_team_id=away.id,
            round_name=fixture.round_name,
            kickoff_at=fixture.kickoff_at,
            status=fixture.status,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        match_ids.append(match.id.value)
    coverage = assess_season_coverage(
        parsed,
        season,
        source="fbref",
        canonical=canonical,
    )
    return ScheduleIngestResult(parsed, coverage, raw_asset.id.value, tuple(match_ids))


def assess_season_coverage(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
    *,
    attempted_fixture_ids: set[str] | None = None,
    source: str | None = None,
    attempt_source: str = "fbref-match-report",
    match_mapping_source: str | None = None,
    canonical: CanonicalStore | None = None,
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
    registered_team_ids, resolved_source = _registered_team_ids(season, team_ids, source)
    # Kept for CLI compatibility only. Unpersisted identifiers never establish an attempt.
    _ = attempted_fixture_ids
    attempted_match_ids = (
        {
            attempt.match_id
            for attempt in canonical.collection_attempts(season.id)
            if attempt.source == attempt_source
        }
        if canonical is not None
        else set()
    )
    match_mapping_source = match_mapping_source or (
        f"{resolved_source}-schedule" if resolved_source is not None else None
    )
    mapped_match_ids = (
        canonical.mapped_match_ids(source=match_mapping_source, source_ids=fixture_ids)
        if canonical is not None and match_mapping_source is not None
        else {}
    )
    missing_attempts = tuple(
        sorted(
            fixture_id
            for fixture_id in fixture_ids
            if mapped_match_ids.get(fixture_id) not in attempted_match_ids
        )
    )
    structural_violations = set(_round_robin_violations(parsed, season))
    if canonical is not None and resolved_source is not None:
        structural_violations.update(
            _canonical_identity_violations(
                parsed,
                season,
                canonical=canonical,
                team_source=resolved_source,
                mapped_match_ids=mapped_match_ids,
            )
        )
    return SeasonCoverage(
        expected_matches=season.expected_matches,
        actual_matches=len(parsed.matches),
        expected_teams=season.expected_teams,
        actual_teams=len(team_ids),
        duplicate_fixture_ids=duplicates,
        unregistered_team_ids=tuple(sorted(team_ids - registered_team_ids)),
        missing_registered_teams=tuple(sorted(registered_team_ids - team_ids)),
        structural_violations=tuple(sorted(structural_violations)),
        missing_collection_attempts=missing_attempts,
        blocking_diagnostics=tuple(sorted({item.code for item in parsed.diagnostics})),
    )


def _registered_team_ids(
    season: SeasonDefinition,
    observed_team_ids: set[str],
    source: str | None,
) -> tuple[set[str], str | None]:
    if not season.teams:
        return set(observed_team_ids), source
    registered_by_source: dict[str, set[str]] = {}
    for team in season.teams:
        for reference in team.sources:
            registered_by_source.setdefault(reference.source, set()).add(reference.source_id)
    if source is None:
        if not registered_by_source:
            return set(), None
        source = max(
            sorted(registered_by_source),
            key=lambda candidate: (
                len(observed_team_ids & registered_by_source[candidate]),
                candidate,
            ),
        )
    return registered_by_source.get(source, set()), source


def _canonical_identity_violations(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
    *,
    canonical: CanonicalStore,
    team_source: str,
    mapped_match_ids: dict[str, MatchId],
) -> set[str]:
    """Cross-check each parsed fixture against persisted canonical identity facts.

    Provider fixture IDs and persisted collection attempts are not sufficient evidence on their
    own: a stale or mis-mapped source ID can otherwise make coverage appear complete.  Keep the
    diagnostics fixture-scoped so an operator can identify the exact record requiring repair.
    """

    violations: set[str] = set()
    for fixture in parsed.matches:
        fixture_id = fixture.source_fixture_id
        match_id = mapped_match_ids.get(fixture_id)
        if match_id is None:
            violations.add(f"canonical_match_mapping_missing:{fixture_id}")
            continue
        try:
            match = canonical.match(match_id)
        except KeyError:
            violations.add(f"canonical_match_missing:{fixture_id}")
            continue

        if match.season_id != season.id:
            violations.add(f"canonical_match_season_mismatch:{fixture_id}")

        for side, source_id, canonical_team_id in (
            ("home", fixture.home_source_id, match.home_team_id),
            ("away", fixture.away_source_id, match.away_team_id),
        ):
            try:
                resolved_team = canonical.mapped_team(source=team_source, source_id=source_id)
            except KeyError:
                violations.add(f"canonical_{side}_team_mapping_missing:{fixture_id}")
            else:
                if resolved_team.id != canonical_team_id:
                    violations.add(f"canonical_match_{side}_team_mismatch:{fixture_id}")

        versions = canonical.match_versions(match_id)
        if not versions:
            violations.add(f"canonical_match_version_missing:{fixture_id}")
        elif versions[-1].kickoff_at != fixture.kickoff_at:
            violations.add(f"canonical_match_kickoff_mismatch:{fixture_id}")
    return violations


def _require_round_robin_contract(season: SeasonDefinition) -> None:
    teams = season.expected_teams
    if season.expected_matches not in {teams * (teams - 1), teams * (teams - 1) // 2}:
        raise ValueError(f"season {season.id} is not configured as a single/double round-robin")


def _round_robin_violations(
    parsed: ScheduleParseResult,
    season: SeasonDefinition,
) -> tuple[str, ...]:
    team_ids = {
        source_id
        for match in parsed.matches
        for source_id in (match.home_source_id, match.away_source_id)
    }
    violations: set[str] = set()
    if any(match.home_source_id == match.away_source_id for match in parsed.matches):
        violations.add("self_fixture")

    directed = Counter((match.home_source_id, match.away_source_id) for match in parsed.matches)
    double_round_robin_matches = season.expected_teams * (season.expected_teams - 1)
    single_round_robin_matches = double_round_robin_matches // 2
    if season.expected_matches == double_round_robin_matches:
        if any(count > 1 for count in directed.values()):
            violations.add("duplicate_directed_matchups")
        expected = {(home, away) for home in team_ids for away in team_ids if home != away}
        if set(directed) != expected:
            violations.add("missing_directed_matchups")
    elif season.expected_matches == single_round_robin_matches:
        unordered = Counter(frozenset(pair) for pair in directed)
        if any(count > 1 for count in unordered.values()):
            violations.add("duplicate_unordered_matchups")
        expected_unordered = {
            frozenset((first, second))
            for first in team_ids
            for second in team_ids
            if first < second
        }
        if set(unordered) != expected_unordered:
            violations.add("missing_unordered_matchups")
    else:
        violations.add("unsupported_round_robin_shape")
    return tuple(sorted(violations))
