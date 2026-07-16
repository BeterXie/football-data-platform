from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, SeasonId
from football_data_platform.domain.models import MatchStatus
from football_data_platform.features.contributions import lineup_delta_contributions
from football_data_platform.features.lineup import (
    LINEUP_DELTA_INPUT_TRANSFORM_V3,
    lineup_delta_input_payload,
)
from football_data_platform.features.player_profiles import (
    PlayerMatchObservation,
    RoleMetricContract,
    build_player_profiles,
)
from football_data_platform.sources.prematch import (
    OfficialLineupDTO,
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    ArchiveConflictError,
    DerivedArchive,
    DerivedArtifactManifest,
    RunManifest,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

GENERATED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
RAW_REF = "raw-asset:" + "a" * 64
ROOT = Path(__file__).parents[1]


def test_derived_artifact_manifest_requires_lineage_and_is_content_addressed(
    tmp_path: Path,
) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    with pytest.raises(ValueError, match="input_refs"):
        DerivedArtifactManifest.create(
            artifact_type="team-baseline",
            payload={"value": 1},
            generated_at=GENERATED_AT,
            transform_version="baseline/1",
            code_version="git:test",
            input_refs=(),
            output_refs=("team-baseline:test",),
            quality="ready",
        )

    manifest = DerivedArtifactManifest.create(
        artifact_type="team-baseline",
        payload={"value": 1},
        generated_at=GENERATED_AT,
        transform_version="baseline/1",
        code_version="git:test",
        input_refs=(RAW_REF,),
        output_refs=("team-baseline:test",),
        quality="ready",
    )
    path = archive.write_artifact_manifest(manifest)
    assert path == archive.artifact_manifest_path(manifest.artifact_id)
    assert archive.write_artifact_manifest(manifest) == path
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest


def test_derived_artifact_manifest_rejects_tampering(tmp_path: Path) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    manifest = DerivedArtifactManifest.create(
        artifact_type="prediction",
        payload={"prediction_id": "prediction:test"},
        generated_at=GENERATED_AT,
        transform_version="model/1",
        code_version="git:test",
        input_refs=("snapshot:test",),
        output_refs=("prediction:test",),
        quality="ready",
    )
    path = archive.write_artifact_manifest(manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["quality"] = "preview"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchiveConflictError, match="identity"):
        archive.load_artifact_manifest(manifest.artifact_id)


def test_failed_run_manifest_is_persisted_and_idempotent(tmp_path: Path) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    run = RunManifest.create(
        run_type="offline-golden",
        started_at=GENERATED_AT,
        ended_at=GENERATED_AT + timedelta(seconds=2),
        transform_version="vertical-slice/1",
        code_version="git:test",
        input_refs=(),
        output_refs=(),
        status="failed",
        error="blocked_by_access_control",
        quality="failed",
    )
    path = archive.write_run_manifest(run)
    assert archive.write_run_manifest(run) == path
    loaded = archive.load_run_manifest(run.run_id)
    assert loaded.status == "failed"
    assert loaded.error == "blocked_by_access_control"
    assert loaded.ended_at == GENERATED_AT + timedelta(seconds=2)


def test_failed_run_without_error_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="error"):
        RunManifest.create(
            run_type="offline-golden",
            started_at=GENERATED_AT,
            ended_at=GENERATED_AT,
            transform_version="vertical-slice/1",
            code_version="git:test",
            input_refs=(),
            output_refs=(),
            status="failed",
            error=None,
            quality="failed",
        )


def test_player_profile_manifest_resolves_raw_lineage_and_rejects_unknown_ref(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"player-profile-evidence",
        source="test-source",
        source_id="player-profile-evidence",
        url="fixture://player-profile-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    observation = PlayerMatchObservation(
        "player:manifest",
        "team:manifest",
        "match:manifest",
        "forward",
        90,
        GENERATED_AT,
        {"shots": 2.0},
        asset.id.value,
        played_at=GENERATED_AT - timedelta(days=1),
    )
    result = build_player_profiles((observation,), as_of=GENERATED_AT, minimum_minutes=0)
    archive = DerivedArchive(layout)
    manifest_path = archive.write_player_profiles(result, generated_at=GENERATED_AT)
    assert manifest_path.exists()

    unknown = PlayerMatchObservation(
        "player:unknown",
        "team:manifest",
        "match:manifest",
        "forward",
        90,
        GENERATED_AT,
        {"shots": 1.0},
        "canonical:missing-player-observation",
        played_at=GENERATED_AT - timedelta(days=1),
    )
    unknown_result = build_player_profiles((unknown,), as_of=GENERATED_AT, minimum_minutes=0)
    with pytest.raises(ArchiveConflictError, match="input reference"):
        archive.write_player_profiles(unknown_result, generated_at=GENERATED_AT)


def test_player_profile_manifest_rejects_mixed_transform_versions(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"mixed-profile-evidence",
        source="test-source",
        source_id="mixed-profile-evidence",
        url="fixture://mixed-profile-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    observations = tuple(
        PlayerMatchObservation(
            f"player:mixed-{index}",
            "team:manifest",
            f"match:mixed-{index}",
            "forward",
            90,
            GENERATED_AT,
            {"shots": 1.0},
            asset.id.value,
            played_at=GENERATED_AT - timedelta(days=1),
        )
        for index in range(2)
    )
    first = build_player_profiles(
        (observations[0],),
        as_of=GENERATED_AT,
        minimum_minutes=0,
        transform_version="player-profile/a",
    ).profiles[0]
    second = build_player_profiles(
        (observations[1],),
        as_of=GENERATED_AT,
        minimum_minutes=0,
        transform_version="player-profile/b",
    ).profiles[0]

    with pytest.raises(ValueError, match="one transform_version"):
        DerivedArchive(layout).write_player_profiles((first, second), generated_at=GENERATED_AT)


def test_player_profile_manifest_rejects_invalid_excluded_refs_on_write(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"excluded-player-profile-evidence",
        source="test-source",
        source_id="excluded-player-profile-evidence",
        url="fixture://excluded-player-profile-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    observation = PlayerMatchObservation(
        "player:excluded",
        "team:excluded",
        "match:excluded",
        "forward",
        90,
        GENERATED_AT,
        {"shots": 1.0},
        asset.id.value,
        played_at=GENERATED_AT - timedelta(days=1),
    )
    archive = DerivedArchive(layout)
    future_time = GENERATED_AT + timedelta(hours=1)
    future_ref = archive.write_snapshot_source(
        value={"future": True},
        input_refs=(asset.id,),
        transform_version="future-profile-input/1",
        generated_at=future_time,
        known_at=future_time,
    )
    result = build_player_profiles(
        (
            observation,
            replace(
                observation,
                match_id="match:excluded-future",
                known_at=future_time,
                source_ref=future_ref,
            ),
        ),
        as_of=GENERATED_AT,
        minimum_minutes=0,
    )
    assert result.excluded_input_refs == (future_ref,)
    archive.write_player_profiles(result, generated_at=future_time)
    assert archive.load_player_profile(result.profiles[0].artifact_id) == result.profiles[0]

    for excluded_ref in (
        "canonical:missing-excluded-observation",
        "unsupported:excluded-observation",
    ):
        invalid = replace(result, excluded_input_refs=(excluded_ref,))
        with pytest.raises(ArchiveConflictError, match="input reference"):
            archive.write_player_profiles(invalid, generated_at=future_time)


def test_player_profile_manifest_rejects_invalid_excluded_refs_on_load(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"persisted-excluded-player-profile-evidence",
        source="test-source",
        source_id="persisted-excluded-player-profile-evidence",
        url="fixture://persisted-excluded-player-profile-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    profile = build_player_profiles(
        (
            PlayerMatchObservation(
                "player:persisted-excluded",
                "team:persisted-excluded",
                "match:persisted-excluded",
                "forward",
                90,
                GENERATED_AT,
                {"shots": 1.0},
                asset.id.value,
                played_at=GENERATED_AT - timedelta(days=1),
            ),
        ),
        as_of=GENERATED_AT,
        minimum_minutes=0,
    ).profiles[0]
    manifest = DerivedArtifactManifest.create(
        artifact_type="player-profiles",
        payload={
            "profiles": [profile],
            "excluded_input_refs": ["canonical:missing-persisted-excluded"],
            "partition_audits": [],
        },
        generated_at=GENERATED_AT,
        transform_version=profile.transform_version,
        code_version="git:test",
        input_refs=(asset.id.value, "canonical:missing-persisted-excluded"),
        output_refs=(profile.artifact_id,),
        quality="ready",
    )
    archive = DerivedArchive(layout)
    archive.write_artifact_manifest(manifest)

    with pytest.raises(ArchiveConflictError, match="input reference"):
        archive.load_player_profile(profile.artifact_id)


def test_lineup_delta_source_recomputes_persisted_player_profiles(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"lineup-and-player-profile-evidence",
        source="test-source",
        source_id="lineup-and-player-profile-evidence",
        url="fixture://lineup-and-player-profile-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    profile_as_of = GENERATED_AT - timedelta(hours=2)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(
        load_competition_registry(ROOT / "config" / "competitions.toml"),
        registered_at=GENERATED_AT,
    )
    canonical.register_raw_asset(asset)
    teams = tuple(
        canonical.resolve_or_create_team(
            source="fbref",
            source_id=f"lineup-manifest-team-{index}",
            canonical_name=f"Lineup Manifest Team {index}",
            competition_id=CompetitionId("competition:eng.1"),
            observed_at=GENERATED_AT,
            raw_asset_id=asset.id,
        )
        for index in range(2)
    )
    reference_match, reference_version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="lineup-manifest-reference",
        competition_id=CompetitionId("competition:eng.1"),
        season_id=SeasonId("season:eng.1.2025-26"),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=GENERATED_AT + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
        observed_at=GENERATED_AT,
        raw_asset_id=asset.id,
    )
    players = tuple(
        canonical.resolve_or_create_player(
            source="lineup-manifest-test",
            source_id=f"lineup-player-{index}",
            canonical_name=f"Lineup Player {index}",
            observed_at=GENERATED_AT,
            raw_asset_id=asset.id,
        ).id
        for index in range(22)
    )
    starters = tuple(player_id.value for player_id in players[:11])
    reference = tuple(player_id.value for player_id in players[11:])
    facts = CanonicalFactStore(
        canonical,
        source_registry=SourceRegistry(
            (
                SourceDescriptor(
                    "test-source",
                    SourceKind.OFFICIAL_LINEUP,
                    "test-source",
                    official=True,
                ),
            )
        ),
    )
    facts.append_official_lineup(
        OfficialLineupDTO(
            match_id=reference_match.id,
            match_version=reference_version.version,
            team_id=teams[0].id,
            player_ids=players[11:],
            source="test-source",
            published_at=profile_as_of,
            observed_at=GENERATED_AT,
            raw_asset_id=asset.id,
            url="fixture://lineup-and-player-profile-evidence",
        )
    )
    contract = RoleMetricContract(
        role="generic",
        version="lineup-role-metrics/1",
        required_metrics=("attack",),
        optional_metrics=(),
        not_applicable_metrics=(),
        minimum_minutes=0,
        minimum_matches=1,
    )
    observations = tuple(
        PlayerMatchObservation(
            player_id,
            teams[0].id.value,
            f"match:{player_id}",
            "generic",
            90,
            profile_as_of,
            {"attack": 2.0 if player_id in starters else 1.0},
            asset.id.value,
            played_at=profile_as_of - timedelta(days=1),
        )
        for player_id in (*starters, *reference)
    )
    profiles = build_player_profiles(
        observations,
        as_of=profile_as_of,
        minimum_minutes=0,
        role_contracts=(contract,),
    ).profiles
    archive = DerivedArchive(layout)
    archive.write_player_profiles(profiles, generated_at=GENERATED_AT)
    reference_source_ref = archive.write_official_lineup_source(
        match_id=reference_match.id,
        match_version=reference_version.version,
        team_id=teams[0].id,
        player_ids=players[11:],
        known_at=profile_as_of,
        observed_at=GENERATED_AT,
        raw_asset_id=asset.id,
    )
    value = {
        teams[0].id.value: lineup_delta_input_payload(
            starter_ids=starters,
            reference_starter_ids=reference,
            profiles=profiles,
            lineup_input_refs=(asset.id.value,),
            reference_lineup_ref=reference_source_ref,
        )
    }
    input_refs = (
        asset.id,
        reference_source_ref,
        *(profile.artifact_id for profile in profiles),
    )

    source_ref = archive.write_snapshot_source(
        value=value,
        input_refs=input_refs,
        transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
        generated_at=GENERATED_AT,
        known_at=profile_as_of,
    )

    validation = archive.validate_snapshot_source(source_ref)
    assert validation.value == value
    assert validation.known_at == profile_as_of
    normalized_input_refs = tuple(
        sorted(
            reference.value if hasattr(reference, "value") else reference
            for reference in input_refs
        )
    )

    missing_reference = copy.deepcopy(value)
    missing_reference[teams[0].id.value]["reference_lineup_ref"] = None
    with pytest.raises(ArchiveConflictError, match="player evidence"):
        archive._validate_lineup_delta_source(
            missing_reference,
            tuple(
                reference
                for reference in normalized_input_refs
                if reference != reference_source_ref
            ),
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )

    tampered_reference = copy.deepcopy(value)
    tampered_reference[teams[0].id.value]["reference_starter_ids"][0] = starters[0]
    with pytest.raises(ArchiveConflictError, match="IDs do not match"):
        archive._validate_lineup_delta_source(
            tampered_reference,
            normalized_input_refs,
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )

    fake_reference = copy.deepcopy(value)
    fake_reference_ref = "derived-source:" + "f" * 64
    fake_reference[teams[0].id.value]["reference_lineup_ref"] = fake_reference_ref
    with pytest.raises(ArchiveConflictError, match="input reference"):
        archive.write_snapshot_source(
            value=fake_reference,
            input_refs=tuple(
                fake_reference_ref if reference == reference_source_ref else reference
                for reference in input_refs
            ),
            transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )

    contributions = lineup_delta_contributions(
        value,
        home_team_id=teams[0].id.value,
        away_team_id="team:other",
        source_ref=source_ref,
        source_validator=archive,
    )
    assert len(contributions) == 1
    assert contributions[0].lambda_home_multiplier > 1

    tampered = copy.deepcopy(value)
    tampered[teams[0].id.value]["dimension_deltas"]["attack"] = 99.0
    with pytest.raises(ArchiveConflictError, match="recomputed profiles"):
        archive.write_snapshot_source(
            value=tampered,
            input_refs=input_refs,
            transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )

    with pytest.raises(ArchiveConflictError, match="nested refs"):
        archive.write_snapshot_source(
            value=value,
            input_refs=input_refs[:-1],
            transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )

    fake = copy.deepcopy(value)
    first_player = starters[0]
    original_profile_ref = fake[teams[0].id.value]["player_profile_refs"][first_player]
    fake[teams[0].id.value]["player_profile_refs"][first_player] = "player-profile:" + "f" * 64
    fake[teams[0].id.value]["input_refs"].remove(original_profile_ref)
    fake[teams[0].id.value]["input_refs"].append("player-profile:" + "f" * 64)
    fake_input_refs = tuple(
        reference for reference in input_refs if reference != original_profile_ref
    ) + ("player-profile:" + "f" * 64,)
    with pytest.raises(ArchiveConflictError, match="input reference"):
        archive.write_snapshot_source(
            value=fake,
            input_refs=fake_input_refs,
            transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
            generated_at=GENERATED_AT,
            known_at=profile_as_of,
        )
