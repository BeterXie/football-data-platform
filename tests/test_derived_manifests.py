from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import football_data_platform.storage.derived as derived_module
from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, MatchId, RawAssetId, SeasonId
from football_data_platform.domain.models import MatchStatus
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.evaluation.metrics import BenchmarkScore, EvaluationRecord
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
from football_data_platform.pipelines.match_report import (
    PRODUCTION_REQUIRED_TABLES,
    ingest_fbref_match_report,
)
from football_data_platform.pipelines.official_lineup import (
    ingest_official_lineup_json,
    replay_official_lineup_contract,
)
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.sources.prematch import (
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


def _archive_evidence(layout: DataLayout, content: bytes = b"manifest-evidence"):
    return RawArchive(layout).archive(
        content,
        source="test-source",
        source_id="manifest-evidence",
        url="fixture://manifest-evidence",
        observed_at=GENERATED_AT,
        target_event_time=GENERATED_AT - timedelta(hours=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )


def _schedule_contract_context(tmp_path: Path):
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=GENERATED_AT)
    schedule = ingest_fbref_schedule(
        (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes(),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=GENERATED_AT,
        archive=raw,
        canonical=canonical,
    )
    return layout, raw, canonical, schedule


def _contract_manifest(contract_id: str) -> DerivedArtifactManifest:
    return DerivedArtifactManifest.create(
        artifact_type="contract-replay-test",
        payload={"contract_id": contract_id},
        generated_at=GENERATED_AT,
        transform_version="contract-replay-test/1",
        code_version="git:test",
        input_refs=(contract_id,),
        output_refs=("file-sha256:" + "c" * 64,),
        quality="ready",
    )


def _persisted_match_report_manifest(tmp_path: Path):
    layout, raw, canonical, schedule = _schedule_contract_context(tmp_path)
    fixture = schedule.parsed.matches[0]
    match_id = MatchId(schedule.canonical_match_ids[0])
    assert fixture.home_goals is not None
    assert fixture.away_goals is not None
    CanonicalFactStore(canonical).append_result_90(
        match_id=match_id,
        match_version=1,
        home_goals=fixture.home_goals,
        away_goals=fixture.away_goals,
        known_at=GENERATED_AT,
        observed_at=GENERATED_AT,
        raw_asset_id=RawAssetId(schedule.raw_asset_id),
    )
    missing_tables = (
        b'<table id="stats_18bb7c10_keeper"><tbody>'
        b'<tr><th data-stat="player">Team Total</th></tr>'
        b"</tbody></table>"
        b'<table id="stats_cff3d9bb_passing"><tbody>'
        b'<tr><th data-stat="player">Team Total</th></tr>'
        b"</tbody></table>"
    )
    content = (
        (ROOT / "tests/fixtures/fbref_premier_league_match_report.html")
        .read_bytes()
        .replace(b"</body>", missing_tables + b"</body>")
    )
    result = ingest_fbref_match_report(
        content,
        page_url=f"https://fbref.example/en/matches/{fixture.source_match_id}/report",
        source_match_id=fixture.source_match_id,
        match_id=match_id,
        match_version=1,
        home_goals=fixture.home_goals,
        away_goals=fixture.away_goals,
        known_at=GENERATED_AT,
        observed_at=GENERATED_AT,
        archive=raw,
        canonical=canonical,
        required_tables=PRODUCTION_REQUIRED_TABLES,
    )
    archive = DerivedArchive(layout)
    manifest = _contract_manifest(result.contract_id)
    archive.write_artifact_manifest(manifest)
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest
    contract = canonical.match_report_contract(result.contract_id)
    return layout, raw, canonical, archive, manifest, contract


def _persisted_official_lineup_manifest(tmp_path: Path):
    layout, raw, canonical, schedule = _schedule_contract_context(tmp_path)
    fixture = schedule.parsed.matches[0]
    source = "manifest-official"
    source_registry = SourceRegistry(
        (SourceDescriptor(source, SourceKind.OFFICIAL_LINEUP, source, official=True),)
    )
    content = json.dumps(
        {
            "schema_version": 2,
            "source": source,
            "source_match_id": fixture.source_match_id,
            "match_mapping_source": "fbref-schedule",
            "team_mapping_source": "fbref",
            "player_mapping_source": source,
            "published_at": (GENERATED_AT - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "teams": [
                {
                    "source_team_id": fixture.home_source_id,
                    "starters": [
                        {
                            "source_player_id": f"manifest-official-player-{index}",
                            "name": f"Manifest Official Player {index}",
                        }
                        for index in range(11)
                    ],
                },
                {
                    "source_team_id": fixture.away_source_id,
                    "starters": [
                        {
                            "source_player_id": f"manifest-official-player-{index}",
                            "name": f"Manifest Official Player {index}",
                        }
                        for index in range(11, 22)
                    ],
                },
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    result = ingest_official_lineup_json(
        content,
        source=source,
        source_match_id=fixture.source_match_id,
        page_url="fixture://manifest-official-lineup",
        observed_at=GENERATED_AT,
        archive=raw,
        canonical=canonical,
        source_registry=source_registry,
    )
    archive = DerivedArchive(layout)
    manifest = _contract_manifest(result.contract_id)
    archive.write_artifact_manifest(manifest)
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest
    contract = canonical.official_lineup_contract(result.contract_id)
    return canonical, archive, manifest, contract, source


def _persisted_result_fact_manifest(tmp_path: Path):
    layout, raw, canonical, schedule = _schedule_contract_context(tmp_path)
    fixture = schedule.parsed.matches[0]
    assert fixture.home_goals is not None
    assert fixture.away_goals is not None
    result = CanonicalFactStore(canonical).append_result_90(
        match_id=MatchId(schedule.canonical_match_ids[0]),
        match_version=1,
        home_goals=fixture.home_goals,
        away_goals=fixture.away_goals,
        known_at=GENERATED_AT,
        observed_at=GENERATED_AT,
        raw_asset_id=RawAssetId(schedule.raw_asset_id),
    )
    archive = DerivedArchive(layout)
    manifest = DerivedArtifactManifest.create(
        artifact_type="result-fact-replay-test",
        payload={"result_ref": result.record_id},
        generated_at=GENERATED_AT,
        transform_version="result-fact-replay-test/1",
        code_version="git:test",
        input_refs=(result.record_id,),
        output_refs=("file-sha256:" + "d" * 64,),
        quality="ready",
    )
    archive.write_artifact_manifest(manifest)
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest
    return layout, raw, canonical, archive, manifest, result


def test_output_reference_lookup_deep_parses_only_matching_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    evidence = _archive_evidence(layout)
    manifests = tuple(
        DerivedArtifactManifest.create(
            artifact_type="lookup-test",
            payload={"index": index, "large": list(range(500))},
            generated_at=GENERATED_AT,
            transform_version="lookup-test/1",
            code_version="git:test",
            input_refs=(evidence.id.value,),
            output_refs=(f"team-baseline:{index:064x}",),
            quality="ready",
        )
        for index in range(1, 6)
    )
    for manifest in manifests:
        archive.write_artifact_manifest(manifest)

    parsed_output_refs: list[tuple[str, ...]] = []
    original = derived_module._parse_artifact_manifest

    def tracked(payload):
        parsed_output_refs.append(tuple(payload.get("output_refs", ())))
        return original(payload)

    monkeypatch.setattr(derived_module, "_parse_artifact_manifest", tracked)
    target = manifests[-1].output_refs[0]

    assert archive._load_artifact_manifest_for_output_ref(target) == manifests[-1]
    assert parsed_output_refs == [(target,), (target,)]


def test_output_reference_lookup_rejects_code_version_difference(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    evidence = _archive_evidence(layout)
    output_ref = "team-baseline:" + "f" * 64
    common = {
        "artifact_type": "duplicate-output-test",
        "payload": {"value": 1, "parameters": {"alpha": 0.5}},
        "generated_at": GENERATED_AT,
        "started_at": GENERATED_AT - timedelta(seconds=1),
        "ended_at": GENERATED_AT + timedelta(seconds=1),
        "transform_version": "duplicate-output/1",
        "input_refs": (evidence.id.value,),
        "output_refs": (output_ref,),
        "quality": "ready",
    }
    first = DerivedArtifactManifest.create(
        **common,
        code_version="git:first",
    )
    second = DerivedArtifactManifest.create(
        **common,
        code_version="git:second",
    )
    archive.write_artifact_manifest(first)
    archive.write_artifact_manifest(second)

    with pytest.raises(ArchiveConflictError, match="conflicting manifests"):
        archive._load_artifact_manifest_for_output_ref(output_ref)


def test_evaluation_output_lookup_rejects_replay_from_different_code_version(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    evidence = _archive_evidence(layout)
    evaluation = EvaluationRecord(
        schema_version=2,
        record_type="evaluation-record",
        prediction_id=evidence.id.value,
        model_run_ref="model-run:" + "a" * 64,
        sample_ref=None,
        match_id="match:test",
        capture_mode=CaptureMode.RECONSTRUCTED,
        actual_outcome="home",
        actual_score="1:0",
        result_probabilities=(("home", 0.5), ("draw", 0.25), ("away", 0.25)),
        result_brier=0.375,
        result_log_loss=-math.log(0.5),
        score_log_loss=1.0,
        market_benchmark=BenchmarkScore(False, None, None, "market_snapshot_missing", (), None),
        evaluated_at=GENERATED_AT,
        input_refs=(evidence.id.value,),
    )

    first_path = archive.write_evaluation(evaluation, code_version="git:first")
    second_path = archive.write_evaluation(evaluation, code_version="git:replay")
    manifest_ids = tuple(
        sorted(
            (
                f"derived-artifact:{first_path.stem}",
                f"derived-artifact:{second_path.stem}",
            )
        )
    )
    output_ref = archive.load_artifact_manifest(manifest_ids[0]).output_refs[0]

    assert first_path != second_path
    with pytest.raises(ArchiveConflictError, match="conflicting manifests"):
        archive._load_artifact_manifest_for_output_ref(output_ref)


@pytest.mark.parametrize(
    "difference",
    (
        "payload",
        "parameters",
        "lineage",
        "quality",
        "generated_at",
        "started_at",
        "ended_at",
    ),
)
def test_output_reference_lookup_rejects_non_code_version_differences(
    tmp_path: Path,
    difference: str,
) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    first_evidence = _archive_evidence(layout, b"first-output-evidence")
    second_evidence = _archive_evidence(layout, b"second-output-evidence")
    output_ref = "evaluation:" + "e" * 64
    payload = {"score": 1, "parameters": {"alpha": 0.5}}
    first = DerivedArtifactManifest.create(
        artifact_type="evaluation",
        payload=payload,
        generated_at=GENERATED_AT,
        transform_version="evaluation/2",
        code_version="git:first",
        input_refs=(first_evidence.id.value,),
        output_refs=(output_ref,),
        quality="ready",
    )
    generated_at = (
        GENERATED_AT + timedelta(seconds=1) if difference == "generated_at" else GENERATED_AT
    )
    second_payload = copy.deepcopy(payload)
    if difference == "payload":
        second_payload["score"] = 2
    elif difference == "parameters":
        second_payload["parameters"]["alpha"] = 0.75
    second = DerivedArtifactManifest.create(
        artifact_type="evaluation",
        payload=second_payload,
        generated_at=generated_at,
        started_at=(
            generated_at - timedelta(seconds=1) if difference == "started_at" else generated_at
        ),
        ended_at=(
            generated_at + timedelta(seconds=1) if difference == "ended_at" else generated_at
        ),
        transform_version="evaluation/2",
        code_version=first.code_version,
        input_refs=((second_evidence.id.value,) if difference == "lineage" else first.input_refs),
        output_refs=(output_ref,),
        quality="preview" if difference == "quality" else "ready",
    )
    archive.write_artifact_manifest(first)
    archive.write_artifact_manifest(second)

    with pytest.raises(ArchiveConflictError, match="conflicting manifests"):
        archive._load_artifact_manifest_for_output_ref(output_ref)


def test_derived_artifact_manifest_requires_lineage_and_is_content_addressed(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    evidence = _archive_evidence(layout)
    with pytest.raises(ValueError, match="input_refs"):
        DerivedArtifactManifest.create(
            artifact_type="team-baseline",
            payload={"value": 1},
            generated_at=GENERATED_AT,
            transform_version="baseline/1",
            code_version="git:test",
            input_refs=(),
            output_refs=("team-baseline:" + "b" * 64,),
            quality="ready",
        )

    manifest = DerivedArtifactManifest.create(
        artifact_type="team-baseline",
        payload={"value": 1},
        generated_at=GENERATED_AT,
        transform_version="baseline/1",
        code_version="git:test",
        input_refs=(evidence.id.value,),
        output_refs=("team-baseline:" + "b" * 64,),
        quality="ready",
    )
    path = archive.write_artifact_manifest(manifest)
    assert path == archive.artifact_manifest_path(manifest.artifact_id)
    assert archive.write_artifact_manifest(manifest) == path
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest


def test_derived_artifact_manifest_rejects_tampering(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    archive = DerivedArchive(layout)
    evidence = _archive_evidence(layout)
    manifest = DerivedArtifactManifest.create(
        artifact_type="prediction",
        payload={"prediction_id": "prediction:" + "c" * 64},
        generated_at=GENERATED_AT,
        transform_version="model/1",
        code_version="git:test",
        input_refs=(evidence.id.value,),
        output_refs=("prediction:" + "c" * 64,),
        quality="ready",
    )
    path = archive.write_artifact_manifest(manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["quality"] = "preview"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchiveConflictError, match="identity"):
        archive.load_artifact_manifest(manifest.artifact_id)


@pytest.mark.parametrize(
    "reference",
    (
        "fake:anything",
        "raw-asset:not-a-digest",
        "raw-asset:" + "f" * 64,
        "market-snapshot:" + "e" * 64,
    ),
)
def test_successful_manifest_rejects_unknown_malformed_or_unresolved_input(
    tmp_path: Path,
    reference: str,
) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    manifest = DerivedArtifactManifest.create(
        artifact_type="evaluation",
        payload={"status": "ready"},
        generated_at=GENERATED_AT,
        transform_version="evaluation/1",
        code_version="git:test",
        input_refs=(reference,),
        output_refs=("evaluation:" + "d" * 64,),
        quality="ready",
    )

    with pytest.raises(ArchiveConflictError, match="reference"):
        archive.write_artifact_manifest(manifest)


@pytest.mark.parametrize("status", ("succeeded", "partial"))
def test_completed_run_requires_resolvable_lineage_beyond_locator_qualifiers(
    tmp_path: Path,
    status: str,
) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))

    qualifier_only = RunManifest.create(
        run_type="source-probe",
        started_at=GENERATED_AT,
        ended_at=GENERATED_AT,
        transform_version="probe/1",
        code_version="git:test",
        input_refs=("source-url:fbref-schedule",),
        output_refs=("file-sha256:" + "a" * 64,),
        status=status,
        error=None,
        quality="ready" if status == "succeeded" else "partial",
    )
    with pytest.raises(ArchiveConflictError, match="resolvable input lineage"):
        archive.write_run_manifest(qualifier_only)

    content_addressed = RunManifest.create(
        run_type="source-probe",
        started_at=GENERATED_AT,
        ended_at=GENERATED_AT,
        transform_version="probe/1",
        code_version="git:test",
        input_refs=("file-sha256:" + "b" * 64, "source-url:fbref-schedule"),
        output_refs=("file-sha256:" + "c" * 64,),
        status=status,
        error=None,
        quality="ready" if status == "succeeded" else "partial",
    )
    archive.write_run_manifest(content_addressed)
    assert archive.load_run_manifest(content_addressed.run_id) == content_addressed


@pytest.mark.parametrize("reference", ("missing-file:fixture.html", "command:run-golden"))
def test_successful_manifest_rejects_failure_only_diagnostics(
    tmp_path: Path,
    reference: str,
) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    manifest = DerivedArtifactManifest.create(
        artifact_type="diagnostic-test",
        payload={"status": "ready"},
        generated_at=GENERATED_AT,
        transform_version="diagnostic/1",
        code_version="git:test",
        input_refs=(reference,),
        output_refs=("file-sha256:" + "d" * 64,),
        quality="ready",
    )

    with pytest.raises(ArchiveConflictError, match="only allowed in failed manifests"):
        archive.write_artifact_manifest(manifest)


def test_successful_manifest_revalidates_upstream_bytes_when_loaded(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    evidence = _archive_evidence(layout, b"load-time-evidence")
    archive = DerivedArchive(layout)
    manifest = DerivedArtifactManifest.create(
        artifact_type="evaluation",
        payload={"status": "ready"},
        generated_at=GENERATED_AT,
        transform_version="evaluation/1",
        code_version="git:test",
        input_refs=(evidence.id.value,),
        output_refs=("evaluation:" + "e" * 64,),
        quality="ready",
    )
    archive.write_artifact_manifest(manifest)
    layout.raw_object_path(evidence.checksum).write_bytes(b"tampered")

    with pytest.raises(ArchiveConflictError, match="unavailable or invalid"):
        archive.load_artifact_manifest(manifest.artifact_id)


@pytest.mark.parametrize(
    "tamper",
    ("parser_version", "raw_bytes", "fixture_identity", "canonical_result"),
)
def test_successful_manifest_replays_match_report_contract_on_load(
    tmp_path: Path,
    tamper: str,
) -> None:
    layout, raw, canonical, archive, manifest, contract = _persisted_match_report_manifest(tmp_path)

    if tamper == "parser_version":
        with canonical.connect() as connection:
            connection.execute(
                "UPDATE match_report_contracts SET parser_version = ? WHERE contract_id = ?",
                ("forged-parser/1", contract.contract_id),
            )
    elif tamper == "raw_bytes":
        asset = raw.load(contract.raw_asset_id)
        layout.raw_object_path(asset.checksum).write_bytes(b"tampered report bytes")
    elif tamper == "fixture_identity":
        with canonical.connect() as connection:
            connection.execute(
                "UPDATE match_versions SET kickoff_at = ? WHERE match_id = ? AND version = ?",
                ("2025-08-16T19:00:00Z", contract.match_id.value, contract.match_version),
            )
    else:
        with canonical.connect() as connection:
            connection.execute(
                "UPDATE match_results_90 SET home_goals = 9 "
                "WHERE match_id = ? AND match_version = ?",
                (contract.match_id.value, contract.match_version),
            )

    with pytest.raises(ArchiveConflictError, match="input reference is unavailable or invalid"):
        archive.load_artifact_manifest(manifest.artifact_id)


def test_match_report_manifest_replays_mapping_visible_at_contract_observation(
    tmp_path: Path,
) -> None:
    _, _, canonical, archive, manifest, contract = _persisted_match_report_manifest(tmp_path)
    current = canonical.resolve_source_mapping(
        source="fbref-schedule",
        entity_type="match",
        source_id=contract.source_match_id,
    )
    candidate = next(
        match_id
        for match_id in canonical.match_ids_for_season(SeasonId("season:eng.1.2025-26"))
        if match_id != contract.match_id
    )
    proposed_at = contract.observed_at + timedelta(minutes=1)
    conflict = canonical.propose_source_mapping(
        source="fbref-schedule",
        entity_type="match",
        source_id=contract.source_match_id,
        candidate_entity_id=candidate,
        proposed_at=proposed_at,
        actor="resolver:test",
        reason="later match identity candidate",
        evidence_refs=("raw-asset:later-match-candidate",),
    )
    canonical.revise_source_mapping(
        source="fbref-schedule",
        entity_type="match",
        source_id=contract.source_match_id,
        conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
        expected_current_mapping_id=current.mapping_id,
        candidate_entity_id=candidate,
        effective_at=proposed_at + timedelta(minutes=1),
        actor="operator:test",
        reason="accepted later match identity",
        evidence_refs=("ticket:later-match-review",),
    )

    assert (
        canonical.resolve_source_mapping(
            source="fbref-schedule",
            entity_type="match",
            source_id=contract.source_match_id,
        ).entity_id
        == candidate
    )
    assert (
        canonical.resolve_source_mapping(
            source="fbref-schedule",
            entity_type="match",
            source_id=contract.source_match_id,
            as_of=contract.observed_at,
        ).entity_id
        == contract.match_id
    )
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest


@pytest.mark.parametrize("tamper", ("current_mapping", "canonical_fact"))
def test_successful_manifest_replays_official_lineup_contract_on_load(
    tmp_path: Path,
    tamper: str,
) -> None:
    canonical, archive, manifest, contract, source = _persisted_official_lineup_manifest(tmp_path)

    with canonical.connect() as connection:
        if tamper == "current_mapping":
            connection.execute("DROP TRIGGER source_mappings_no_delete")
            connection.execute(
                "DELETE FROM source_mappings WHERE source = ? AND entity_type = 'player' "
                "AND source_id = ? AND valid_to IS NULL",
                (source, "manifest-official-player-0"),
            )
        else:
            connection.execute(
                "DELETE FROM lineup_facts WHERE record_id = ?",
                (contract.fact_ids[0],),
            )

    with pytest.raises(ArchiveConflictError, match="input reference is unavailable or invalid"):
        archive.load_artifact_manifest(manifest.artifact_id)


def test_official_lineup_manifest_replays_mapping_visible_at_contract_observation(
    tmp_path: Path,
) -> None:
    canonical, archive, manifest, contract, source = _persisted_official_lineup_manifest(tmp_path)
    binding = contract.source_bindings[0]
    candidate = canonical.resolve_or_create_player(
        source="review-candidate",
        source_id="later-player-identity",
        canonical_name="Later Player Identity",
        observed_at=contract.observed_at + timedelta(seconds=1),
        raw_asset_id=contract.raw_asset_id,
    )
    current = canonical.resolve_source_mapping(
        source=source,
        entity_type="player",
        source_id=binding.source_player_id,
    )
    proposed_at = contract.observed_at + timedelta(minutes=1)
    conflict = canonical.propose_source_mapping(
        source=source,
        entity_type="player",
        source_id=binding.source_player_id,
        candidate_entity_id=candidate.id,
        proposed_at=proposed_at,
        actor="resolver:test",
        reason="later player identity candidate",
        evidence_refs=("raw-asset:later-player-candidate",),
    )
    canonical.revise_source_mapping(
        source=source,
        entity_type="player",
        source_id=binding.source_player_id,
        conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
        expected_current_mapping_id=current.mapping_id,
        candidate_entity_id=candidate.id,
        effective_at=proposed_at + timedelta(minutes=1),
        actor="operator:test",
        reason="accepted later player identity",
        evidence_refs=("ticket:later-player-review",),
    )

    assert (
        canonical.mapped_player(
            source=source,
            source_id=binding.source_player_id,
        ).id
        == candidate.id
    )
    assert (
        canonical.mapped_player(
            source=source,
            source_id=binding.source_player_id,
            as_of=contract.observed_at,
        ).id
        == binding.player_id
    )
    assert archive.load_artifact_manifest(manifest.artifact_id) == manifest


@pytest.mark.parametrize("tamper", ("goals", "evidence", "raw_bytes"))
def test_successful_manifest_replays_result_fact_lineage_on_load(
    tmp_path: Path,
    tamper: str,
) -> None:
    layout, raw, canonical, archive, manifest, result = _persisted_result_fact_manifest(tmp_path)

    if tamper == "goals":
        with canonical.connect() as connection:
            connection.execute(
                "UPDATE match_results_90 SET home_goals = 9 WHERE record_id = ?",
                (result.record_id,),
            )
    elif tamper == "evidence":
        replacement = raw.archive(
            b"unrelated-result-evidence",
            source="test-source",
            source_id="unrelated-result-evidence",
            url="fixture://unrelated-result-evidence",
            observed_at=GENERATED_AT,
            target_event_time=None,
            collector_version="test/1",
            media_type="text/plain",
        )
        canonical.register_raw_asset(replacement)
        with canonical.connect() as connection:
            connection.execute(
                "UPDATE fact_evidence SET raw_asset_id = ? WHERE record_id = ?",
                (replacement.id.value, result.record_id),
            )
    else:
        with canonical.connect() as connection:
            raw_asset_id = connection.execute(
                "SELECT raw_asset_id FROM match_results_90 WHERE record_id = ?",
                (result.record_id,),
            ).fetchone()[0]
        asset = raw.load(RawAssetId(raw_asset_id))
        layout.raw_object_path(asset.checksum).write_bytes(b"tampered result bytes")

    with pytest.raises(ArchiveConflictError, match="input reference is unavailable or invalid"):
        archive.load_artifact_manifest(manifest.artifact_id)


def test_successful_manifest_rejects_unknown_output_namespace(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    evidence = _archive_evidence(layout)
    archive = DerivedArchive(layout)
    manifest = DerivedArtifactManifest.create(
        artifact_type="evaluation",
        payload={"status": "ready"},
        generated_at=GENERATED_AT,
        transform_version="evaluation/1",
        code_version="git:test",
        input_refs=(evidence.id.value,),
        output_refs=("fake:anything",),
        quality="ready",
    )

    with pytest.raises(ArchiveConflictError, match="unsupported manifest reference namespace"):
        archive.write_artifact_manifest(manifest)


def test_failed_run_manifest_is_persisted_and_idempotent(tmp_path: Path) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    run = RunManifest.create(
        run_type="offline-golden",
        started_at=GENERATED_AT,
        ended_at=GENERATED_AT + timedelta(seconds=2),
        transform_version="vertical-slice/1",
        code_version="git:test",
        input_refs=(
            "command:offline-golden",
            "missing-file:blocked-response.html",
            "source-url:fbref-schedule",
        ),
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
    assert "missing-file:blocked-response.html" in loaded.input_refs


def test_failed_run_rejects_unknown_reference_namespace(tmp_path: Path) -> None:
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    run = RunManifest.create(
        run_type="offline-golden",
        started_at=GENERATED_AT,
        ended_at=GENERATED_AT,
        transform_version="vertical-slice/1",
        code_version="git:test",
        input_refs=("fake:anything",),
        output_refs=(),
        status="failed",
        error="blocked_by_access_control",
        quality="failed",
    )

    with pytest.raises(ArchiveConflictError, match="unsupported manifest reference namespace"):
        archive.write_run_manifest(run)


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
    path = archive.artifact_manifest_path(manifest.artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_payload(), sort_keys=True), encoding="utf-8")

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
    lineup_sources = SourceRegistry(
        (
            SourceDescriptor(
                "test-source",
                SourceKind.OFFICIAL_LINEUP,
                "test-source",
                official=True,
            ),
        )
    )
    lineup_content = json.dumps(
        {
            "schema_version": 2,
            "source": "test-source",
            "source_match_id": "lineup-manifest-reference",
            "match_mapping_source": "fbref-schedule",
            "team_mapping_source": "fbref",
            "player_mapping_source": "test-source",
            "published_at": profile_as_of.isoformat().replace("+00:00", "Z"),
            "teams": [
                {
                    "source_team_id": "lineup-manifest-team-0",
                    "starters": [
                        {
                            "source_player_id": f"lineup-player-{index}",
                            "name": f"Lineup Player {index}",
                        }
                        for index in range(11, 22)
                    ],
                },
                {
                    "source_team_id": "lineup-manifest-team-1",
                    "starters": [
                        {
                            "source_player_id": f"lineup-player-{index}",
                            "name": f"Lineup Player {index}",
                        }
                        for index in range(11)
                    ],
                },
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    lineup_ingest = ingest_official_lineup_json(
        lineup_content,
        source="test-source",
        source_match_id="lineup-manifest-reference",
        page_url="fixture://lineup-and-player-profile-evidence",
        observed_at=GENERATED_AT,
        archive=raw,
        canonical=canonical,
        source_registry=lineup_sources,
    )
    lineup_contract = replay_official_lineup_contract(
        lineup_ingest.contract_id,
        archive=raw,
        canonical=canonical,
    )
    players = tuple(
        canonical.mapped_player(source="test-source", source_id=f"lineup-player-{index}").id
        for index in range(22)
    )
    starters = tuple(player_id.value for player_id in players[:11])
    reference = tuple(player_id.value for player_id in players[11:])
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
        contract_id=lineup_contract.contract_id,
        match_id=reference_match.id,
        match_version=reference_version.version,
        team_id=teams[0].id,
        player_ids=players[11:],
        known_at=profile_as_of,
        observed_at=GENERATED_AT,
        raw_asset_id=lineup_contract.raw_asset_id,
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
