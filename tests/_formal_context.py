from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import (
    CompetitionId,
    MatchId,
    RawAssetId,
    SeasonId,
    TeamId,
)
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

ROOT = Path(__file__).parents[1]
_ID_NAMESPACE = uuid.UUID("c62a4fc0-2e72-4d9c-b4b3-113b31c31982")


@dataclass(frozen=True, slots=True)
class FormalContextFixture:
    canonical: CanonicalStore
    raw: RawArchive
    derived: DerivedArchive
    match_id: MatchId
    match_version: int
    home_team_id: TeamId
    away_team_id: TeamId
    context_ref: str
    context_value: dict[str, object]
    context_known_at: datetime
    schedule_raw_id: RawAssetId


def registered_target_identity() -> tuple[MatchId, TeamId, TeamId]:
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    season = registry.competitions[0].seasons[0]
    home = season.teams[0].id
    away = season.teams[2].id
    synthetic_season_id = _synthetic_season_id(_fixture_token("prediction-evaluation"))
    source_id = "|".join((synthetic_season_id.value, home.value, away.value))
    match_id = MatchId("match:" + str(uuid.uuid5(_ID_NAMESPACE, f"match|round-robin|{source_id}")))
    return match_id, home, away


def seed_formal_context(
    layout: DataLayout,
    *,
    as_of: datetime,
    kickoff: datetime,
    key: str,
    observed_at: datetime | None = None,
    team_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> FormalContextFixture:
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    observed_at = observed_at or kickoff + timedelta(days=1)
    token = _fixture_token(key)
    competition, season, registry = _synthetic_registry(
        registry,
        token=token,
        team_indices=team_indices,
    )
    raw = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=observed_at)
    source_ids = (f"{token}homeprev", f"{token}awayprev", f"{token}target")
    local_target = kickoff.astimezone(ZoneInfo(competition.timezone))
    local_dates = (
        local_target - timedelta(days=9),
        local_target - timedelta(days=6),
        local_target,
    )
    home_index, home_opponent_index, away_index, away_opponent_index = range(4)
    team_pairs = (
        (home_index, home_opponent_index),
        (away_index, away_opponent_index),
        (home_index, away_index),
    )
    rows: list[str] = []
    fixture_rows = list(zip(source_ids, local_dates, team_pairs, strict=True))
    existing_pairs = set(team_pairs)
    fixture_rows.extend(
        (
            f"{token}filler{home}{away}",
            local_target + timedelta(days=index),
            (home, away),
        )
        for index, (home, away) in enumerate(
            (
                (home, away)
                for home in range(4)
                for away in range(4)
                if home != away and (home, away) not in existing_pairs
            ),
            start=1,
        )
    )
    for index, (source_id, local, pair) in enumerate(fixture_rows, start=1):
        home = season.teams[pair[0]].source("fbref")
        away = season.teams[pair[1]].source("fbref")
        known_at = as_of.isoformat().replace("+00:00", "Z")
        score = "1-0" if source_id in source_ids else ""
        rows.append(
            f'<tr data-fdp-fixture-known-at="{known_at}">'
            f'<th data-stat="round">Matchweek {index}</th>'
            f'<td data-stat="date">{local:%Y-%m-%d}</td>'
            f'<td data-stat="start_time">{local:%H:%M}</td>'
            '<td data-stat="home_team">'
            f'<a href="/en/squads/{home.source_id}/x">{home.alias}</a></td>'
            f'<td data-stat="score">{score}</td>'
            '<td data-stat="away_team">'
            f'<a href="/en/squads/{away.source_id}/x">{away.alias}</a></td>'
            '<td data-stat="match_report">'
            f'<a href="/en/matches/{source_id}/report">Report</a></td></tr>'
        )
    known_at = as_of.isoformat().replace("+00:00", "Z")
    content = (
        "<!doctype html><html><body>"
        f'<table id="sched_formal" data-fdp-fixture-known-at="{known_at}"><tbody>'
        + "".join(rows)
        + "</tbody></table></body></html>"
    ).encode()
    ingested = ingest_fbref_schedule(
        content,
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=observed_at,
        archive=raw,
        canonical=canonical,
    )
    match_ids = {
        fixture.source_fixture_id: MatchId(match_id)
        for fixture, match_id in zip(
            ingested.parsed.matches,
            ingested.canonical_match_ids,
            strict=True,
        )
    }
    facts = CanonicalFactStore(canonical)
    for source_id in source_ids[:2]:
        facts.append_result_90(
            match_id=match_ids[source_id],
            match_version=1,
            home_goals=1,
            away_goals=0,
            known_at=as_of,
            observed_at=observed_at,
            raw_asset_id=RawAssetId(ingested.raw_asset_id),
        )
    derived = DerivedArchive(layout)
    target_id = match_ids[source_ids[2]]
    context_ref = derived.write_match_context_source(
        match_id=target_id,
        match_version=1,
        as_of=as_of,
    )
    validation = derived.validate_snapshot_source(context_ref)
    return FormalContextFixture(
        canonical=canonical,
        raw=raw,
        derived=derived,
        match_id=target_id,
        match_version=1,
        home_team_id=season.teams[home_index].id,
        away_team_id=season.teams[away_index].id,
        context_ref=context_ref,
        context_value=validation.value,
        context_known_at=validation.known_at,
        schedule_raw_id=RawAssetId(ingested.raw_asset_id),
    )


def _synthetic_season_id(token: str) -> SeasonId:
    return SeasonId(f"season:formal-{token}")


def _fixture_token(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold()) or "formalcontext"


def _synthetic_registry(registry, *, token: str, team_indices: tuple[int, int, int, int]):
    if len(set(team_indices)) != 4:
        raise ValueError("formal context fixtures require four distinct team indices")
    base_competition = registry.competitions[0]
    base_season = base_competition.seasons[0]
    teams = tuple(base_season.teams[index] for index in team_indices)
    source = base_season.source("fbref")
    season = type(base_season)(
        id=_synthetic_season_id(token),
        label=f"Formal context {token}",
        starts_on=base_season.starts_on,
        ends_on=base_season.ends_on,
        expected_teams=4,
        expected_matches=12,
        sources=(
            type(source)(
                source="fbref",
                competition_id=f"formal-{token}",
                season_id=f"formal-{token}",
            ),
        ),
        teams=teams,
    )
    competition = type(base_competition)(
        id=CompetitionId(f"competition:formal-{token}"),
        name=f"Formal context {token}",
        country_code=base_competition.country_code,
        kind=base_competition.kind,
        timezone=base_competition.timezone,
        seasons=(season,),
    )
    synthetic = type(registry)(
        schema_version=registry.schema_version,
        competitions=(competition,),
    )
    return competition, season, synthetic


def seed_legacy_snapshot_source(
    derived: DerivedArchive,
    *,
    value: object,
    input_refs: tuple[RawAssetId | str, ...],
    transform_version: str,
    generated_at: datetime,
    known_at: datetime | None = None,
) -> str:
    """Materialize historical bytes for audit tests without reopening a writer API."""

    normalized_refs = sorted(
        {
            reference.value if isinstance(reference, RawAssetId) else reference
            for reference in input_refs
        }
    )
    identity: dict[str, object] = {
        "schema_version": 1,
        "value": value,
        "input_refs": normalized_refs,
        "transform_version": transform_version,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
    }
    if known_at is not None:
        identity["known_at"] = known_at.isoformat().replace("+00:00", "Z")
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    source_ref = f"derived-source:{digest}"
    derived._write_json(
        derived._snapshot_source_path(source_ref),
        {"id": source_ref, **identity},
    )
    return source_ref
