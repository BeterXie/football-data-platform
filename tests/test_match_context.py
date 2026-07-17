from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, RawAssetId, TeamId
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.features.contributions import context_contributions
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.match_context import (
    MatchContextReplayError,
    replay_match_context,
)
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
SCHEDULE_KNOWN_AT = datetime(2025, 6, 18, 8, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _FixtureRow:
    source_id: str
    home_index: int
    away_index: int
    date: str
    status: str = "finished"


@dataclass(frozen=True, slots=True)
class _World:
    layout: DataLayout
    raw: RawArchive
    canonical: CanonicalStore
    derived: DerivedArchive
    rows: tuple[_FixtureRow, ...]
    match_ids: dict[str, MatchId]
    team_ids: dict[int, TeamId]
    kickoffs: dict[str, datetime]
    result_refs: dict[str, str]
    schedule_raw_ref: str


def _build_world(
    tmp_path: Path,
    rows: tuple[_FixtureRow, ...],
    *,
    result_known_at: dict[str, datetime] | None = None,
    schedule_known_at: datetime | None = SCHEDULE_KNOWN_AT,
    observed_at: datetime = OBSERVED_AT,
    row_metadata: bool = True,
    complete_schedule: bool = True,
) -> _World:
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    competition = registry.competitions[0]
    season = replace(
        competition.seasons[0],
        expected_teams=4,
        expected_matches=12,
        teams=competition.seasons[0].teams[:4],
    )
    competition = replace(competition, seasons=(season,))
    registry = replace(registry, competitions=(competition,))
    if complete_schedule:
        rows = _complete_round_robin(rows, team_count=season.expected_teams)
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=observed_at)
    content = _schedule_html(
        rows,
        season.teams,
        schedule_known_at=schedule_known_at,
        row_metadata=row_metadata,
    )
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
    kickoffs = {
        fixture.source_fixture_id: fixture.kickoff_at
        for fixture in ingested.parsed.matches
        if fixture.kickoff_at is not None
    }
    team_ids = {
        index: canonical.mapped_team(
            source="fbref",
            source_id=season.teams[index].source("fbref").source_id,
        ).id
        for index in {item for row in rows for item in (row.home_index, row.away_index)}
    }
    result_refs: dict[str, str] = {}
    facts = CanonicalFactStore(canonical)
    for source_id, known_at in (result_known_at or {}).items():
        stored = facts.append_result_90(
            match_id=match_ids[source_id],
            match_version=1,
            home_goals=1,
            away_goals=0,
            known_at=known_at,
            observed_at=observed_at,
            raw_asset_id=RawAssetId(ingested.raw_asset_id),
        )
        result_refs[source_id] = stored.record_id
    return _World(
        layout=layout,
        raw=raw,
        canonical=canonical,
        derived=DerivedArchive(layout),
        rows=rows,
        match_ids=match_ids,
        team_ids=team_ids,
        kickoffs=kickoffs,
        result_refs=result_refs,
        schedule_raw_ref=ingested.raw_asset_id,
    )


def _complete_round_robin(
    rows: tuple[_FixtureRow, ...],
    *,
    team_count: int,
) -> tuple[_FixtureRow, ...]:
    existing = {(row.home_index, row.away_index) for row in rows}
    fillers = tuple(
        _FixtureRow(
            f"contextfiller{home}{away}",
            home,
            away,
            f"2026-04-{index:02d}",
            status="scheduled",
        )
        for index, (home, away) in enumerate(
            (
                (home, away)
                for home in range(team_count)
                for away in range(team_count)
                if home != away and (home, away) not in existing
            ),
            start=1,
        )
    )
    return (*rows, *fillers)


def _schedule_html(
    rows,
    teams,
    *,
    schedule_known_at: datetime | None,
    row_metadata: bool = True,
) -> bytes:
    known_attribute = (
        ""
        if schedule_known_at is None
        else ' data-fdp-fixture-known-at="'
        + schedule_known_at.isoformat().replace("+00:00", "Z")
        + '"'
    )
    row_known_attribute = known_attribute if row_metadata else ""
    rendered_rows = []
    for index, row in enumerate(rows, start=1):
        home = teams[row.home_index].source("fbref")
        away = teams[row.away_index].source("fbref")
        score = {
            "cancelled": "Cancelled",
            "finished": "1-0",
            "postponed": "Postponed",
            "scheduled": "",
        }[row.status]
        rendered_rows.append(
            f'<tr{row_known_attribute}><th data-stat="round">Matchweek {index}</th>'
            f'<td data-stat="date">{row.date}</td>'
            '<td data-stat="start_time">15:00</td>'
            '<td data-stat="home_team">'
            f'<a href="/en/squads/{home.source_id}/x">{home.alias}</a></td>'
            f'<td data-stat="score">{score}</td>'
            '<td data-stat="away_team">'
            f'<a href="/en/squads/{away.source_id}/x">{away.alias}</a></td>'
            '<td data-stat="match_report">'
            f'<a href="/en/matches/{row.source_id}/report">Report</a></td></tr>'
        )
    return (
        "<!doctype html><html><body>"
        f'<table id="sched_context"{known_attribute}><tbody>'
        + "".join(rendered_rows)
        + "</tbody></table></body></html>"
    ).encode()


def _standard_world(
    tmp_path: Path,
    *,
    home_result_known_at: datetime | None = None,
    away_result_known_at: datetime | None = None,
    schedule_known_at: datetime | None = SCHEDULE_KNOWN_AT,
    observed_at: datetime = OBSERVED_AT,
) -> _World:
    rows = (
        _FixtureRow("contexthomeprev", 0, 1, "2025-08-01"),
        _FixtureRow("contextawayprev", 2, 3, "2025-08-04"),
        _FixtureRow("contexttarget", 0, 2, "2025-08-10"),
    )
    target_kickoff = datetime(2025, 8, 10, 14, 0, tzinfo=UTC)
    default_known_at = target_kickoff - timedelta(days=2)
    return _build_world(
        tmp_path,
        rows,
        result_known_at={
            "contexthomeprev": home_result_known_at or default_known_at,
            "contextawayprev": away_result_known_at or default_known_at,
        },
        schedule_known_at=schedule_known_at,
        observed_at=observed_at,
    )


def _write_standard_context(world: _World) -> tuple[str, object]:
    as_of = world.kickoffs["contexttarget"] - timedelta(hours=24)
    source_ref = world.derived.write_match_context_source(
        match_id=world.match_ids["contexttarget"],
        match_version=1,
        as_of=as_of,
    )
    return source_ref, world.derived.validate_snapshot_source(source_ref)


def test_single_replay_reuses_one_read_only_canonical_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _standard_world(tmp_path)
    original_connect = world.canonical.connect
    connect_calls = 0
    query_only_states: list[int] = []

    @contextmanager
    def counting_connect():
        nonlocal connect_calls
        connect_calls += 1
        with original_connect() as connection:
            yield connection
            query_only_states.append(int(connection.execute("PRAGMA query_only").fetchone()[0]))

    monkeypatch.setattr(world.canonical, "connect", counting_connect)

    replay = replay_match_context(
        match_id=world.match_ids["contexttarget"],
        match_version=1,
        as_of=world.kickoffs["contexttarget"] - timedelta(hours=24),
        archive=world.raw,
        canonical=world.canonical,
    )

    assert replay.value["quality_status"] == "ready"
    assert connect_calls == 1
    assert query_only_states == [1]


def test_formal_context_recomputes_different_per_team_rest_and_own_side_effect(
    tmp_path: Path,
) -> None:
    world = _standard_world(tmp_path)

    source_ref, validation = _write_standard_context(world)

    value = validation.value
    home_id = world.team_ids[0].value
    away_id = world.team_ids[2].value
    assert value["quality_status"] == "ready"
    assert value["teams"][home_id]["rest_days"] == 9.0
    assert value["teams"][away_id]["rest_days"] == 6.0
    assert value["teams"][home_id]["previous_result_ref"] == world.result_refs["contexthomeprev"]
    assert set(validation.input_refs) == {
        world.schedule_raw_ref,
        world.result_refs["contexthomeprev"],
        world.result_refs["contextawayprev"],
    }
    home_effect, away_effect = context_contributions(
        value,
        home_team_id=home_id,
        away_team_id=away_id,
        source_ref=source_ref,
        source_validator=world.derived,
    )
    assert home_effect.lambda_home_multiplier > 1
    assert home_effect.lambda_away_multiplier == 1
    assert away_effect.lambda_home_multiplier == 1
    assert away_effect.lambda_away_multiplier > 1


def test_generic_writer_and_rehashed_caller_rest_cannot_bypass_replay(tmp_path: Path) -> None:
    world = _standard_world(tmp_path)
    source_ref, _ = _write_standard_context(world)

    with pytest.raises(ValueError, match="write_match_context_source"):
        world.derived.write_snapshot_source(
            value={"caller_rest_days": 99},
            input_refs=(RawAssetId(world.schedule_raw_ref),),
            transform_version="match-context-input/2",
            generated_at=OBSERVED_AT,
        )
    with pytest.raises(ValueError, match="audit-only"):
        world.derived.write_snapshot_source(
            value={"days_since_previous_match": 99.0},
            input_refs=(RawAssetId(world.schedule_raw_ref),),
            transform_version="match-context-input/1",
            generated_at=OBSERVED_AT,
        )

    payload = json.loads(
        world.derived._snapshot_source_path(source_ref).read_text(encoding="utf-8")
    )
    identity = copy.deepcopy(payload)
    identity.pop("id")
    home_id = world.team_ids[0].value
    identity["value"]["teams"][home_id]["rest_days"] = 99.0
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    forged_ref = f"derived-source:{digest}"
    forged_path = world.derived._snapshot_source_path(forged_ref)
    forged_path.parent.mkdir(parents=True, exist_ok=True)
    forged_path.write_text(
        json.dumps({"id": forged_ref, **identity}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArchiveConflictError, match="recomputed canonical context"):
        world.derived.validate_snapshot_source(forged_ref)


@pytest.mark.parametrize(
    "mutation",
    ("match", "version", "teams", "kickoff"),
)
def test_snapshot_v3_binds_exact_canonical_fixture(tmp_path: Path, mutation: str) -> None:
    world = _standard_world(tmp_path)
    source_ref, validation = _write_standard_context(world)
    target_id = world.match_ids["contexttarget"]
    kickoff = world.kickoffs["contexttarget"]
    as_of = kickoff - timedelta(hours=24)
    home = world.team_ids[0]
    away = world.team_ids[2]
    feature = SnapshotFeature(
        name="match_context",
        value=validation.value,
        known_at=validation.known_at,
        source_ref=source_ref,
        contribution_key="match-context",
    )
    match_id = target_id
    match_version = 1
    if mutation == "match":
        match_id = MatchId("match:unknown-context-target")
    elif mutation == "version":
        match_version = 2
    elif mutation == "teams":
        home, away = away, home
    elif mutation == "kickoff":
        kickoff += timedelta(days=1)
        as_of = kickoff - timedelta(hours=24)

    with pytest.raises(ValueError, match="match context|match_context"):
        build_snapshot(
            match_id=match_id,
            match_version=match_version,
            snapshot_type=SnapshotType.T24H,
            as_of=as_of,
            scheduled_kickoff_used=kickoff,
            feature_spec_version="prematch-features/3",
            features=(feature,),
            home_team_id=home,
            away_team_id=away,
            source_validator=world.derived,
        )


def test_no_previous_match_is_explicit_missing_not_neutral_or_na(tmp_path: Path) -> None:
    rows = (_FixtureRow("contextfirst", 0, 2, "2025-08-10"),)
    world = _build_world(tmp_path, rows)
    as_of = world.kickoffs["contextfirst"] - timedelta(hours=24)

    source_ref = world.derived.write_match_context_source(
        match_id=world.match_ids["contextfirst"],
        match_version=1,
        as_of=as_of,
    )
    validation = world.derived.validate_snapshot_source(source_ref)

    assert validation.value["quality_status"] == "missing"
    assert {item["reason"] for item in validation.value["teams"].values()} == {
        "previous_match_coverage_unproven"
    }
    assert all("rest_days" not in item for item in validation.value["teams"].values())
    feature = SnapshotFeature(
        name="match_context",
        value=validation.value,
        known_at=validation.known_at,
        source_ref=source_ref,
        contribution_key="match-context",
    )
    snapshot = build_snapshot(
        match_id=world.match_ids["contextfirst"],
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        as_of=as_of,
        scheduled_kickoff_used=world.kickoffs["contextfirst"],
        feature_spec_version="prematch-features/3",
        features=(feature,),
        home_team_id=world.team_ids[0],
        away_team_id=world.team_ids[2],
        source_validator=world.derived,
    )
    assert snapshot.capture_mode is CaptureMode.RECONSTRUCTED
    assert snapshot.quality_status == "preview"
    context_missing = {
        item for item in snapshot.missing_fields if item.startswith("feature:match_context:")
    }
    assert context_missing == {
        f"feature:match_context:{world.team_ids[0].value}:previous_match",
        f"feature:match_context:{world.team_ids[2].value}:previous_match",
    }
    with pytest.raises(ValueError, match="unavailable"):
        context_contributions(
            validation.value,
            home_team_id=world.team_ids[0].value,
            away_team_id=world.team_ids[2].value,
            source_ref=source_ref,
            source_validator=world.derived,
        )


def test_partial_schedule_cannot_hide_a_more_recent_fixture_and_fallback(
    tmp_path: Path,
) -> None:
    rows = (
        _FixtureRow("contextolder", 0, 1, "2025-08-01"),
        _FixtureRow("contextawayprev", 2, 3, "2025-08-04"),
        _FixtureRow("contexttarget", 0, 2, "2025-08-10"),
    )
    world = _build_world(
        tmp_path,
        rows,
        result_known_at={
            "contextolder": datetime(2025, 8, 2, tzinfo=UTC),
            "contextawayprev": datetime(2025, 8, 5, tzinfo=UTC),
        },
        complete_schedule=False,
    )

    source_ref = world.derived.write_match_context_source(
        match_id=world.match_ids["contexttarget"],
        match_version=1,
        as_of=world.kickoffs["contexttarget"] - timedelta(hours=24),
    )
    value = world.derived.validate_snapshot_source(source_ref).value

    assert value["quality_status"] == "missing"
    assert value["target_schedule"]["season_schedule_coverage_complete"] is False
    assert {item["reason"] for item in value["teams"].values()} == {
        "season_schedule_coverage_incomplete"
    }
    assert all("rest_days" not in item for item in value["teams"].values())
    with pytest.raises(ValueError, match="unavailable"):
        context_contributions(
            value,
            home_team_id=world.team_ids[0].value,
            away_team_id=world.team_ids[2].value,
            source_ref=source_ref,
            source_validator=world.derived,
        )


def test_latest_scheduled_fixture_without_result_blocks_fallback_to_older_result(
    tmp_path: Path,
) -> None:
    rows = (
        _FixtureRow("contextolder", 0, 1, "2025-08-01"),
        _FixtureRow("contextunverified", 0, 3, "2025-08-06", status="scheduled"),
        _FixtureRow("contexttarget", 0, 2, "2025-08-10"),
        _FixtureRow("contextawayprev", 2, 3, "2025-08-04"),
    )
    world = _build_world(
        tmp_path,
        rows,
        result_known_at={
            "contextolder": datetime(2025, 8, 2, tzinfo=UTC),
            "contextawayprev": datetime(2025, 8, 5, tzinfo=UTC),
        },
    )

    _, validation = _write_standard_context(world)

    assert validation.value["teams"][world.team_ids[0].value] == {
        "team_id": world.team_ids[0].value,
        "status": "missing",
        "reason": "latest_prior_fixture_completion_unverified",
    }


@pytest.mark.parametrize("status", ("postponed", "cancelled"))
def test_unplayed_fixture_is_skipped_for_last_completed_match(
    tmp_path: Path,
    status: str,
) -> None:
    rows = (
        _FixtureRow("contextolder", 0, 1, "2025-08-01"),
        _FixtureRow("contextunplayed", 0, 3, "2025-08-06", status=status),
        _FixtureRow("contexttarget", 0, 2, "2025-08-10"),
        _FixtureRow("contextawayprev", 2, 3, "2025-08-04"),
    )
    world = _build_world(
        tmp_path,
        rows,
        result_known_at={
            "contextolder": datetime(2025, 8, 2, tzinfo=UTC),
            "contextawayprev": datetime(2025, 8, 5, tzinfo=UTC),
        },
    )

    _, validation = _write_standard_context(world)

    home_context = validation.value["teams"][world.team_ids[0].value]
    assert home_context["status"] == "available"
    assert home_context["previous_match_id"] == world.match_ids["contextolder"].value


def test_future_result_is_missing_but_equal_as_of_boundary_is_accepted(tmp_path: Path) -> None:
    target_kickoff = datetime(2025, 8, 10, 14, 0, tzinfo=UTC)
    as_of = target_kickoff - timedelta(hours=24)
    world = _standard_world(
        tmp_path,
        home_result_known_at=as_of + timedelta(seconds=1),
        away_result_known_at=as_of,
        schedule_known_at=as_of,
    )

    _, validation = _write_standard_context(world)

    assert validation.value["teams"][world.team_ids[0].value]["reason"] == (
        "latest_prior_fixture_result_unavailable"
    )
    assert validation.value["teams"][world.team_ids[2].value]["status"] == "available"
    assert validation.known_at == as_of


def test_schedule_availability_requires_raw_observation_or_replayed_metadata(
    tmp_path: Path,
) -> None:
    target = _FixtureRow("contextfirst", 0, 2, "2025-08-10", status="scheduled")
    future_metadata = datetime(2025, 8, 10, 13, 0, tzinfo=UTC)
    world = _build_world(tmp_path, (target,), schedule_known_at=future_metadata)
    as_of = world.kickoffs["contextfirst"] - timedelta(hours=24)
    with pytest.raises(MatchContextReplayError, match="not known"):
        world.derived.write_match_context_source(
            match_id=world.match_ids["contextfirst"],
            match_version=1,
            as_of=as_of,
        )

    prospective = _build_world(
        tmp_path / "prospective",
        (target,),
        schedule_known_at=None,
        observed_at=as_of,
    )
    source_ref = prospective.derived.write_match_context_source(
        match_id=prospective.match_ids["contextfirst"],
        match_version=1,
        as_of=as_of,
    )
    validation = prospective.derived.validate_snapshot_source(source_ref)
    assert validation.value["target_schedule"]["availability_basis"] == "raw-observed"


def test_table_level_known_at_cannot_certify_an_exact_fixture_version(tmp_path: Path) -> None:
    target = _FixtureRow("contextfirst", 0, 2, "2025-08-10", status="scheduled")
    as_of = datetime(2025, 8, 9, 14, 0, tzinfo=UTC)
    world = _build_world(
        tmp_path,
        (target,),
        schedule_known_at=as_of,
        row_metadata=False,
    )

    with pytest.raises(MatchContextReplayError, match="not known"):
        world.derived.write_match_context_source(
            match_id=world.match_ids["contextfirst"],
            match_version=1,
            as_of=as_of,
        )


def test_reschedule_keeps_old_version_and_rejects_new_version_before_known_at(
    tmp_path: Path,
) -> None:
    world = _standard_world(tmp_path)
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    changed_rows = tuple(
        _FixtureRow(row.source_id, row.home_index, row.away_index, "2025-08-12", row.status)
        if row.source_id == "contexttarget"
        else row
        for row in world.rows
    )
    changed_known_at = datetime(2025, 8, 11, 8, 0, tzinfo=UTC)
    ingest_fbref_schedule(
        _schedule_html(changed_rows, season.teams, schedule_known_at=changed_known_at),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=OBSERVED_AT + timedelta(minutes=1),
        archive=world.raw,
        canonical=world.canonical,
    )
    target_id = world.match_ids["contexttarget"]
    versions = world.canonical.match_versions(target_id)
    assert len(versions) == 2
    old_as_of = world.kickoffs["contexttarget"] - timedelta(hours=24)
    world.derived.write_match_context_source(
        match_id=target_id,
        match_version=1,
        as_of=old_as_of,
    )
    with pytest.raises(MatchContextReplayError, match="not known"):
        world.derived.write_match_context_source(
            match_id=target_id,
            match_version=2,
            as_of=old_as_of,
        )
    new_as_of = changed_known_at
    new_ref = world.derived.write_match_context_source(
        match_id=target_id,
        match_version=2,
        as_of=new_as_of,
    )
    assert world.derived.validate_snapshot_source(new_ref).value["match_version"] == 2


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
