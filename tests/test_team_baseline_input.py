from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId
from football_data_platform.domain.snapshots import SnapshotFeature, SnapshotType, build_snapshot
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
SCHEDULE_KNOWN_AT = datetime(2025, 6, 18, 8, 0, tzinfo=UTC)


def _schedule_html(teams) -> bytes:
    rows = (
        ("baseline-prev-home", 0, 1, "2025-08-01", "1-0"),
        ("baseline-prev-away", 2, 3, "2025-08-04", "1-0"),
        ("baseline-target", 0, 2, "2025-08-10", ""),
    )
    rendered = []
    for index, (source_id, home_index, away_index, date, score) in enumerate(rows, start=1):
        home = teams[home_index].source("fbref")
        away = teams[away_index].source("fbref")
        known_at = SCHEDULE_KNOWN_AT.isoformat().replace("+00:00", "Z")
        rendered.append(
            f'<tr data-fdp-fixture-known-at="{known_at}">'
            f'<th data-stat="round">Matchweek {index}</th><td data-stat="date">{date}</td>'
            '<td data-stat="start_time">15:00</td><td data-stat="home_team">'
            f'<a href="/en/squads/{home.source_id}/x">{home.alias}</a></td>'
            f'<td data-stat="score">{score}</td><td data-stat="away_team">'
            f'<a href="/en/squads/{away.source_id}/x">{away.alias}</a></td>'
            f'<td data-stat="match_report"><a href="/en/matches/{source_id}/report">Report</a></td>'
            "</tr>"
        )
    return (
        '<!doctype html><html><body><table id="sched_baseline"><tbody>'
        + "".join(rendered)
        + "</tbody></table></body></html>"
    ).encode()


def _world(tmp_path: Path):
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    schedule = ingest_fbref_schedule(
        _schedule_html(season.teams),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=OBSERVED_AT,
        archive=raw,
        canonical=canonical,
    )
    parsed_match_ids = {
        fixture.source_fixture_id: MatchId(match_id)
        for fixture, match_id in zip(
            schedule.parsed.matches, schedule.canonical_match_ids, strict=True
        )
    }
    parsed_kickoffs = {
        fixture.source_fixture_id: fixture.kickoff_at
        for fixture in schedule.parsed.matches
        if fixture.kickoff_at is not None
    }
    aliases = ("baseline-prev-home", "baseline-prev-away", "baseline-target")
    source_ids = tuple(parsed_match_ids)
    match_ids = {
        alias: parsed_match_ids[source_id]
        for alias, source_id in zip(aliases, source_ids, strict=True)
    }
    kickoffs = {
        alias: parsed_kickoffs[source_id]
        for alias, source_id in zip(aliases, source_ids, strict=True)
    }
    teams = {
        index: canonical.mapped_team(
            source="fbref", source_id=season.teams[index].source("fbref").source_id
        ).id.value
        for index in range(4)
    }
    return layout, raw, canonical, DerivedArchive(layout), match_ids, kickoffs, teams


def _baseline(
    world,
    *,
    as_of: datetime,
    input_asset,
    baseline_as_of: datetime | None = None,
    code_version: str | None = None,
):
    _, raw, canonical, derived, match_ids, kickoffs, teams = world
    baseline_as_of = as_of if baseline_as_of is None else baseline_as_of
    artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "baseline-observation",
                teams[0],
                teams[2],
                kickoffs["baseline-prev-home"],
                baseline_as_of - timedelta(days=1),
                1.4,
                0.9,
                input_asset.id.value,
            ),
        ),
        as_of=baseline_as_of,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    write_options = {} if code_version is None else {"code_version": code_version}
    derived.write_team_baseline(
        artifact,
        generated_at=raw.load(input_asset.id).observed_at,
        **write_options,
    )
    return artifact


def _semantic_asset(world, *, target_event_time: datetime | None):
    _, raw, canonical, _, _, _, _ = world
    asset = raw.archive(
        b"baseline report evidence",
        source="fbref",
        source_id=f"baseline-report-{target_event_time}",
        url="fixture://baseline-report",
        observed_at=OBSERVED_AT,
        target_event_time=target_event_time,
        collector_version="fbref-match-report/1",
        media_type="text/html",
    )
    canonical.register_raw_asset(asset)
    return asset


def _valid_source(world, *, code_version: str | None = None):
    _, _, canonical, derived, match_ids, kickoffs, _ = world
    target = match_ids["baseline-target"]
    as_of = kickoffs["baseline-target"] - timedelta(hours=24)
    asset = _semantic_asset(world, target_event_time=as_of - timedelta(days=1))
    baseline = _baseline(world, as_of=as_of, input_asset=asset, code_version=code_version)
    source_ref = derived.write_team_baseline_source(
        match_id=target,
        match_version=1,
        as_of=as_of,
        baseline_artifact_id=baseline.artifact_id,
    )
    return target, as_of, asset, baseline, source_ref


def _rehash_source(derived: DerivedArchive, source_ref: str, mutate) -> str:
    payload = json.loads(derived._snapshot_source_path(source_ref).read_text(encoding="utf-8"))
    payload.pop("id")
    mutate(payload)
    identity = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    digest = hashlib.sha256(identity).hexdigest()
    forged = f"derived-source:{digest}"
    derived._write_json(derived._snapshot_source_path(forged), {"id": forged, **payload})
    return forged


def _rehash_baseline_manifest(derived, baseline, mutate, *, replace: bool) -> str:
    manifest = derived._load_artifact_manifest_for_output_ref(baseline.artifact_id)
    original_path = derived.artifact_manifest_path(manifest.artifact_id)
    identity = manifest.to_payload()
    identity.pop("id")
    mutate(identity)
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    forged = f"derived-artifact:{digest}"
    if replace:
        original_path.unlink()
    derived._write_json(derived.artifact_manifest_path(forged), {"id": forged, **identity})
    return forged


def test_team_baseline_source_replays_fixture_and_lambdas(tmp_path: Path) -> None:
    world = _world(tmp_path)
    target, as_of, _, baseline, source_ref = _valid_source(world)
    _, _, canonical, derived, _, _, _ = world
    validation = derived.validate_snapshot_source(source_ref)
    match = canonical.match(target)
    expected = expected_goals_from_baseline(
        baseline,
        home_team_id=match.home_team_id.value,
        away_team_id=match.away_team_id.value,
    )
    assert validation.transform_version == "team-baseline-input/3"
    assert validation.known_at == baseline.as_of == as_of
    assert (validation.value["lambda_home"], validation.value["lambda_away"]) == expected
    assert validation.source_context["match_id"] == target.value
    assert validation.source_context["baseline_artifact_id"] == baseline.artifact_id


def test_team_baseline_manifest_accepts_writer_selected_code_version(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, _, source_ref = _valid_source(world, code_version="test-build/42")
    _, _, _, derived, _, _, _ = world

    assert derived.validate_snapshot_source(source_ref).transform_version == "team-baseline-input/3"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("artifact_type", "forged-baseline"),
        lambda payload: payload.__setitem__("schema_version", 3),
        lambda payload: payload["payload"].__setitem__("half_life_days", 91.0),
        lambda payload: payload.__setitem__("transform_version", "baseline-forged/1"),
        lambda payload: payload.__setitem__("input_refs", [f"raw-asset:{'0' * 64}"]),
        lambda payload: payload.__setitem__(
            "output_refs", sorted((*payload["output_refs"], f"team-baseline:{'0' * 64}"))
        ),
        lambda payload: payload.__setitem__("quality", "forged"),
        lambda payload: payload.__setitem__("status", "partial"),
        lambda payload: payload.update({"status": "partial", "error": "forged"}),
        lambda payload: payload.__setitem__("started_at", "2026-07-16T07:59:00Z"),
        lambda payload: payload.__setitem__("ended_at", "2026-07-16T08:01:00Z"),
        lambda payload: payload.update(
            {
                "started_at": "2025-01-01T00:00:00Z",
                "generated_at": "2025-01-01T00:00:00Z",
                "ended_at": "2025-01-01T00:00:00Z",
            }
        ),
    ),
    ids=(
        "artifact-type",
        "schema-version",
        "payload",
        "transform-version",
        "input-refs",
        "output-refs",
        "quality",
        "status",
        "error",
        "started-at",
        "ended-at",
        "generation-before-as-of",
    ),
)
def test_rehashed_team_baseline_manifest_tampering_is_rejected(tmp_path: Path, mutate) -> None:
    world = _world(tmp_path)
    _, _, _, baseline, source_ref = _valid_source(world)
    _, _, _, derived, _, _, _ = world
    _rehash_baseline_manifest(derived, baseline, mutate, replace=True)

    with pytest.raises(ArchiveConflictError):
        derived.validate_snapshot_source(source_ref)


def test_duplicate_team_baseline_manifest_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, baseline, source_ref = _valid_source(world)
    _, _, _, derived, _, _, _ = world
    _rehash_baseline_manifest(
        derived,
        baseline,
        lambda payload: payload.__setitem__("code_version", "duplicate-build/1"),
        replace=False,
    )

    with pytest.raises(ArchiveConflictError):
        derived.validate_snapshot_source(source_ref)


def test_missing_team_baseline_manifest_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, baseline, source_ref = _valid_source(world)
    _, _, _, derived, _, _, _ = world
    manifest = derived._load_artifact_manifest_for_output_ref(baseline.artifact_id)
    derived.artifact_manifest_path(manifest.artifact_id).unlink()

    assert derived.load_team_baseline(baseline.artifact_id) == baseline
    with pytest.raises(ArchiveConflictError):
        derived.validate_snapshot_source(source_ref)


def test_team_baseline_writer_rejects_generation_before_cutoff(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, baseline, _ = _valid_source(world)
    _, _, _, derived, _, _, _ = world

    with pytest.raises(ValueError, match="generated_at cannot precede as_of"):
        derived.write_team_baseline(
            baseline,
            generated_at=baseline.as_of - timedelta(microseconds=1),
        )


def test_team_baseline_writer_rejects_a_second_manifest(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, baseline, _ = _valid_source(world)
    _, _, _, derived, _, _, _ = world

    with pytest.raises(ArchiveConflictError, match="already has another manifest"):
        derived.write_team_baseline(
            baseline,
            generated_at=OBSERVED_AT,
            code_version="different-build/1",
        )


@pytest.mark.parametrize(
    "transform", ("team-baseline-input/1", "team-baseline-input/2", "team-baseline-input/3")
)
def test_generic_baseline_source_writer_is_closed(tmp_path: Path, transform: str) -> None:
    world = _world(tmp_path)
    _, raw, _, derived, _, _, _ = world
    asset = _semantic_asset(world, target_event_time=OBSERVED_AT - timedelta(days=1))
    with pytest.raises(ValueError, match="write_team_baseline_source"):
        derived.write_snapshot_source(
            value={"arbitrary": True},
            input_refs=(asset.id,),
            transform_version=transform,
            generated_at=raw.load(asset.id).observed_at,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["value"].__setitem__("lambda_home", 999.0),
        lambda payload: payload["source_context"].__setitem__("match_id", "match:forged"),
        lambda payload: payload["source_context"].__setitem__("match_version", 2),
        lambda payload: payload["source_context"].__setitem__("home_team_id", "team:forged"),
        lambda payload: payload["source_context"].__setitem__(
            "scheduled_kickoff", "2025-08-11T14:00:00Z"
        ),
        lambda payload: payload["source_context"].__setitem__("as_of", "2025-08-09T13:00:00Z"),
    ),
)
def test_rehashed_baseline_source_tampering_is_rejected(tmp_path: Path, mutate) -> None:
    world = _world(tmp_path)
    _, _, _, derived, _, _, _ = world
    _, _, _, _, source_ref = _valid_source(world)
    forged = _rehash_source(derived, source_ref, mutate)
    with pytest.raises((ArchiveConflictError, ValueError, KeyError)):
        derived.validate_snapshot_source(forged)


def test_baseline_missing_team_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, derived, match_ids, kickoffs, teams = world
    asset = _semantic_asset(
        world, target_event_time=kickoffs["baseline-target"] - timedelta(days=2)
    )
    as_of = kickoffs["baseline-target"] - timedelta(hours=24)
    one_team = build_team_baseline(
        (
            TeamMatchProcess(
                "one-team",
                teams[0],
                teams[1],
                kickoffs["baseline-prev-home"],
                as_of - timedelta(days=1),
                1.0,
                0.8,
                asset.id.value,
            ),
        ),
        as_of=as_of,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(one_team)
    with pytest.raises((ArchiveConflictError, KeyError, ValueError), match="absent|baseline"):
        derived.write_team_baseline_source(
            match_id=match_ids["baseline-target"],
            match_version=1,
            as_of=as_of,
            baseline_artifact_id=one_team.artifact_id,
        )


def test_future_baseline_cutoff_and_unstamped_raw_are_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, _, derived, match_ids, kickoffs, _ = world
    as_of = kickoffs["baseline-target"] - timedelta(hours=24)
    future_asset = _semantic_asset(world, target_event_time=as_of + timedelta(minutes=1))
    future_baseline = _baseline(
        world,
        as_of=as_of,
        input_asset=future_asset,
        baseline_as_of=as_of,
    )
    with pytest.raises(ArchiveConflictError, match="cutoff"):
        derived.write_team_baseline_source(
            match_id=match_ids["baseline-target"],
            match_version=1,
            as_of=as_of,
            baseline_artifact_id=future_baseline.artifact_id,
        )

    unstamped_asset = _semantic_asset(world, target_event_time=None)
    unstamped_baseline = _baseline(world, as_of=as_of, input_asset=unstamped_asset)
    with pytest.raises(ArchiveConflictError, match="semantic known_at"):
        derived.write_team_baseline_source(
            match_id=match_ids["baseline-target"],
            match_version=1,
            as_of=as_of,
            baseline_artifact_id=unstamped_baseline.artifact_id,
        )

    known_asset = _semantic_asset(world, target_event_time=as_of - timedelta(days=1))
    future_cutoff = _baseline(
        world,
        as_of=as_of,
        input_asset=known_asset,
        baseline_as_of=as_of + timedelta(minutes=1),
    )
    with pytest.raises(ArchiveConflictError, match="baseline as_of"):
        derived.write_team_baseline_source(
            match_id=match_ids["baseline-target"],
            match_version=1,
            as_of=as_of,
            baseline_artifact_id=future_cutoff.artifact_id,
        )


def test_later_valid_source_cutoff_cannot_impersonate_snapshot_cutoff(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, canonical, derived, match_ids, kickoffs, _ = world
    target, as_of, _, _, source_ref = _valid_source(world)
    later = as_of + timedelta(hours=1)
    forged = _rehash_source(
        derived,
        source_ref,
        lambda payload: payload["source_context"].__setitem__(
            "as_of", later.isoformat().replace("+00:00", "Z")
        ),
    )
    baseline_validation = derived.validate_snapshot_source(forged)
    context_ref = derived.write_match_context_source(
        match_id=target,
        match_version=1,
        as_of=as_of,
    )
    context = derived.validate_snapshot_source(context_ref)
    match = canonical.match(target)
    with pytest.raises(ValueError, match="snapshot identity"):
        build_snapshot(
            match_id=target,
            match_version=1,
            snapshot_type=SnapshotType.T24H,
            as_of=as_of,
            scheduled_kickoff_used=kickoffs["baseline-target"],
            feature_spec_version="prematch-features/3",
            features=(
                SnapshotFeature(
                    "team_baseline",
                    baseline_validation.value,
                    baseline_validation.known_at,
                    forged,
                    "team-baseline",
                ),
                SnapshotFeature(
                    "match_context",
                    context.value,
                    context.known_at,
                    context_ref,
                    "match-context",
                ),
            ),
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            source_validator=derived,
        )


def test_formal_snapshot_rejects_legacy_baseline_transform(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _, _, canonical, derived, match_ids, kickoffs, _ = world
    target = match_ids["baseline-target"]
    as_of = kickoffs["baseline-target"] - timedelta(hours=24)
    asset = _semantic_asset(world, target_event_time=as_of - timedelta(days=1))
    baseline = _baseline(world, as_of=as_of, input_asset=asset)
    # Construct an audit-only /2 source directly; the public writer is closed.
    value = {
        "artifact_id": baseline.artifact_id,
        "artifact": team_baseline_payload(baseline),
        "lambda_home": 999.0,
        "lambda_away": 999.0,
    }
    identity = {
        "schema_version": 1,
        "value": value,
        "input_refs": [asset.id.value],
        "transform_version": "team-baseline-input/2",
        "generated_at": OBSERVED_AT.isoformat().replace("+00:00", "Z"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    legacy_ref = f"derived-source:{digest}"
    derived._write_json(derived._snapshot_source_path(legacy_ref), {"id": legacy_ref, **identity})
    context_ref = derived.write_match_context_source(
        match_id=target,
        match_version=1,
        as_of=as_of,
    )
    context = derived.validate_snapshot_source(context_ref)
    match = canonical.match(target)
    with pytest.raises(ValueError, match="baseline source"):
        build_snapshot(
            match_id=target,
            match_version=1,
            snapshot_type=SnapshotType.T24H,
            as_of=as_of,
            scheduled_kickoff_used=kickoffs["baseline-target"],
            feature_spec_version="prematch-features/3",
            features=(
                SnapshotFeature("team_baseline", value, as_of, legacy_ref, "team-baseline"),
                SnapshotFeature(
                    "match_context", context.value, context.known_at, context_ref, "match-context"
                ),
            ),
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            source_validator=derived,
        )
