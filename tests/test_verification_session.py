from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import football_data_platform.storage.derived as derived_module
import football_data_platform.storage.verification as verification_module
from football_data_platform.features.player_profiles import (
    PlayerMatchObservation,
    build_player_profiles,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)
from football_data_platform.storage.verification import (
    IdentityState,
    VerificationSession,
)

OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


def _layout(tmp_path: Path) -> DataLayout:
    layout = DataLayout(tmp_path / "data")
    layout.ensure()
    return layout


def _evidence(layout: DataLayout):
    return RawArchive(layout).archive(
        b"verification-session-evidence",
        source="verification-test",
        source_id="verification-test",
        url="fixture://verification-test",
        observed_at=OBSERVED_AT,
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )


def _manifest(
    layout: DataLayout,
    *,
    output_ref: str,
    value: int,
    code_version: str = "git:test",
) -> DerivedArtifactManifest:
    evidence = _evidence(layout)
    return DerivedArtifactManifest.create(
        artifact_type="verification-test",
        payload={"value": value},
        generated_at=OBSERVED_AT,
        transform_version="verification-test/1",
        code_version=code_version,
        input_refs=(evidence.id.value,),
        output_refs=(output_ref,),
        quality="ready",
    )


def _persist_json_artifact(
    store: TrainingArtifactStore,
    *,
    prefix: str,
    directory: str,
    artifact_type: str,
) -> tuple[str, Path, DerivedArtifactManifest]:
    identity = {"probe": artifact_type}
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    reference = prefix + digest
    payload = {"id": reference, **identity}
    path = store.layout.derived / directory / digest[:2] / f"{digest}.json"
    store._write_json(path, payload)
    manifest = DerivedArtifactManifest.create(
        artifact_type=artifact_type,
        payload=payload,
        generated_at=OBSERVED_AT,
        transform_version="verification-json/1",
        code_version="football-data-platform/0.1.0",
        input_refs=(_evidence(store.layout).id.value,),
        output_refs=(reference,),
        quality="ready",
    )
    return reference, path, manifest


def test_canonical_connection_is_one_read_only_snapshot_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()

    calls = 0
    original_connect = verification_module.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(verification_module.sqlite3, "connect", tracked_connect)
    with VerificationSession(layout) as session:
        first = session.canonical_connection()
        second = session.canonical_connection()
        assert first is second
        assert session.lookup_identity("team:missing", namespace="team") is IdentityState.ABSENT
        with pytest.raises(sqlite3.OperationalError):
            first.execute("CREATE TABLE should_not_write(id INTEGER)")
        with pytest.raises(ValueError, match="SELECT"):
            session.canonical_query("PRAGMA query_only")
    assert calls == 1


def test_canonical_snapshot_is_fixed_before_the_first_identity_lookup(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    database = layout.canonical / "platform.sqlite3"
    CanonicalStore(database).initialize()
    anchor = sqlite3.connect(database)
    anchor.execute("PRAGMA journal_mode = WAL")
    anchor.execute("SELECT name FROM sqlite_schema ORDER BY name LIMIT 1").fetchone()
    anchor.commit()
    session = VerificationSession(layout)
    session.__enter__()
    try:
        session.canonical_connection()
        with sqlite3.connect(database) as writer:
            writer.execute(
                "INSERT INTO entities(entity_id, entity_type, created_at) VALUES (?, ?, ?)",
                ("team:writer-after-open", "team", "2026-07-16T08:00:00Z"),
            )
        assert (
            session.lookup_identity("team:writer-after-open", namespace="team")
            is IdentityState.ABSENT
        )
        assert (
            anchor.execute(
                "SELECT 1 FROM entities WHERE entity_id = ?",
                ("team:writer-after-open",),
            ).fetchone()
            is not None
        )
    finally:
        # The writer changed immutable bytes during this request; the snapshot result above is
        # still valid, while the stability gate must fail closed at completion.
        with pytest.raises(ArchiveConflictError):
            session.close()
        anchor.close()


def test_identity_lookup_rejects_non_entity_namespaces_without_opening_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    CanonicalStore(layout.canonical / "platform.sqlite3").initialize()

    def forbidden_connect(*args, **kwargs):
        raise AssertionError("unsupported namespace must not query SQLite")

    monkeypatch.setattr(verification_module.sqlite3, "connect", forbidden_connect)
    with VerificationSession(layout) as session:
        assert (
            session.lookup_identity("fact:match_results_90:any", namespace="fact")
            is IdentityState.UNKNOWN
        )


def test_identity_unknown_is_not_cached_and_missing_is_cached_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    connect_calls = 0
    original_connect = verification_module.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(verification_module.sqlite3, "connect", tracked_connect)
    session = VerificationSession(layout)
    with pytest.raises(ArchiveConflictError, match="must be entered"):
        session.lookup_identity("team:missing", namespace="team")
    session.close()

    with VerificationSession(layout) as session:
        assert session.lookup_identity("team:missing", namespace="team") is IdentityState.UNKNOWN
        assert session.lookup_identity("team:missing", namespace="team") is IdentityState.UNKNOWN
    assert connect_calls == 0

    monkeypatch.setattr(verification_module.sqlite3, "connect", original_connect)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    monkeypatch.setattr(verification_module.sqlite3, "connect", tracked_connect)
    with VerificationSession(layout) as session:
        assert session.lookup_identity("team:missing", namespace="team") is IdentityState.ABSENT
        assert session.lookup_identity("team:missing", namespace="team") is IdentityState.ABSENT
    assert connect_calls == 1


def test_manifest_catalog_is_session_local_and_does_not_reuse_tampered_bytes(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    output_ref = "file-sha256:" + "a" * 64
    archive = DerivedArchive(layout)
    manifest = _manifest(layout, output_ref=output_ref, value=1)
    archive.write_artifact_manifest(manifest)

    with VerificationSession(layout) as first_session:
        entries = first_session.manifests_for_output_ref(output_ref)
        assert len(entries) == 1
        manifest_path = entries[0].path

    manifest_path.write_text('{"id": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ArchiveConflictError):
        with VerificationSession(layout) as second_session:
            second_session.manifests_for_output_ref(output_ref)


def test_duplicate_manifest_output_is_fail_closed(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    output_ref = "file-sha256:" + "b" * 64
    archive = DerivedArchive(layout)
    archive.write_artifact_manifest(_manifest(layout, output_ref=output_ref, value=1))
    archive.write_artifact_manifest(_manifest(layout, output_ref=output_ref, value=2))

    with VerificationSession(layout) as session:
        assert len(session.manifests_for_output_ref(output_ref)) == 2
        with pytest.raises(ArchiveConflictError, match="exactly one"):
            session.require_unique_manifest(output_ref)


def test_public_prediction_replay_reuses_one_catalog_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    archive = DerivedArchive(layout)
    store = TrainingArtifactStore(layout)
    references = ("prediction:" + "1" * 64, "snapshot:" + "2" * 64)
    for index, reference in enumerate(references):
        archive.write_artifact_manifest(
            _manifest(
                layout,
                output_ref=reference,
                value=index,
                code_version="football-data-platform/0.1.0",
            )
        )

    calls = 0
    original = verification_module._catalog_files

    def tracked(root: Path):
        nonlocal calls
        calls += 1
        return original(root)

    monkeypatch.setattr(verification_module, "_catalog_files", tracked)
    seen_sessions: list[VerificationSession | None] = []
    original_lookup = store._load_fixed_code_manifest_for_output_ref

    def tracked_lookup(
        output_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ):
        seen_sessions.append(verification_session)
        return original_lookup(output_ref, verification_session=verification_session)

    def stop_after_snapshot_lookup(reference: str, *args, **kwargs):
        if reference == references[0]:
            return {"snapshot_id": references[1]}
        raise RuntimeError("controlled prediction replay stop")

    monkeypatch.setattr(store, "_load_fixed_code_manifest_for_output_ref", tracked_lookup)
    monkeypatch.setattr(store, "_verify_json_artifact", stop_after_snapshot_lookup)

    for expected_builds in (1, 2):
        with pytest.raises(TrainingArtifactConflict, match="controlled prediction replay stop"):
            store.load_verified_prediction_context(references[0])
        assert calls == expected_builds

    assert seen_sessions[0] is seen_sessions[1]
    assert seen_sessions[0] is not None
    assert seen_sessions[2] is seen_sessions[3]
    assert seen_sessions[2] is not None
    assert seen_sessions[0] is not seen_sessions[2]


@pytest.mark.parametrize("artifact_kind", ("prediction", "snapshot", "composition", "source"))
def test_public_prediction_loader_rejects_direct_file_mutation_at_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    layout = _layout(tmp_path)
    store = TrainingArtifactStore(layout)
    archive = store.derived
    if artifact_kind in {"prediction", "snapshot"}:
        prefix, directory, artifact_type = (
            ("prediction:", "predictions", "prediction")
            if artifact_kind == "prediction"
            else ("snapshot:", "snapshots", "prematch-snapshot")
        )
        reference, proof_path, manifest = _persist_json_artifact(
            store,
            prefix=prefix,
            directory=directory,
            artifact_type=artifact_type,
        )

        def exercise(session: VerificationSession) -> None:
            store._verify_json_artifact(
                reference,
                prefix,
                directory,
                artifact_type,
                manifest=manifest,
                verification_session=session,
            )

    elif artifact_kind == "source":
        evidence = _evidence(layout)
        reference = archive.write_snapshot_source(
            value={"probe": "source"},
            input_refs=(evidence.id,),
            transform_version="verification-source/1",
            generated_at=OBSERVED_AT,
        )
        proof_path = archive._snapshot_source_path(reference)

        def exercise(session: VerificationSession) -> None:
            archive.validate_snapshot_source(reference, verification_session=session)

    else:
        evidence = _evidence(layout)
        reference = "score-grid-composition:" + "7" * 64
        composition_payload = {"schema_version": 1, "probe": "composition"}
        proof_path = archive.score_grid_composition_path(reference)
        archive._write_json(proof_path, {"id": reference, **composition_payload})
        prediction = SimpleNamespace(
            composition_artifact_ref=reference,
            snapshot_quality_status="ready",
            generated_at=OBSERVED_AT,
            input_refs=(evidence.id.value,),
            composition_version="verification-composition/1",
            snapshot_as_of=OBSERVED_AT,
        )
        archive.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="score-grid-composition",
                schema_version=1,
                payload=composition_payload,
                generated_at=OBSERVED_AT,
                transform_version=prediction.composition_version,
                code_version="football-data-platform/0.1.0",
                input_refs=prediction.input_refs,
                output_refs=(reference,),
                quality=prediction.snapshot_quality_status,
            )
        )
        monkeypatch.setattr(derived_module, "verify_score_prediction", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            derived_module,
            "score_grid_composition_payload",
            lambda value: composition_payload,
        )
        monkeypatch.setattr(
            derived_module,
            "score_grid_composition_artifact_id",
            lambda value: reference,
        )

        def exercise(session: VerificationSession) -> None:
            archive.verify_score_grid_composition(
                prediction,  # type: ignore[arg-type]
                verification_session=session,
            )

    original_replay = store._load_prediction_replay

    def controlled_replay(
        reference: str,
        *,
        require_current: bool,
        verification_session: VerificationSession | None = None,
    ):
        if verification_session is None:
            return original_replay(
                reference,
                require_current=require_current,
                verification_session=None,
            )
        exercise(verification_session)
        proof_path.write_bytes(proof_path.read_bytes() + b" ")
        return object(), object()

    monkeypatch.setattr(store, "_load_prediction_replay", controlled_replay)
    with pytest.raises(TrainingArtifactConflict, match="prediction domain replay") as error:
        store.load_verified_prediction_context("prediction:" + "a" * 64)
    assert isinstance(error.value.__cause__, ArchiveConflictError)


def test_unique_manifest_lookup_translates_session_close_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    store = TrainingArtifactStore(layout)
    output_ref = "file-sha256:" + "3" * 64
    manifest = _manifest(
        layout,
        output_ref=output_ref,
        value=3,
        code_version="football-data-platform/0.1.0",
    )
    store.derived.write_artifact_manifest(manifest)
    path = store.derived.artifact_manifest_path(manifest.artifact_id)
    original_load = store.derived.load_artifact_manifest

    def mutate_after_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return loaded

    monkeypatch.setattr(store.derived, "load_artifact_manifest", mutate_after_load)
    with pytest.raises(TrainingArtifactConflict, match="unavailable or invalid") as error:
        store._load_fixed_code_manifest_for_output_ref(output_ref)
    assert isinstance(error.value.__cause__, ArchiveConflictError)


def test_audit_prediction_replay_does_not_create_verification_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TrainingArtifactStore(_layout(tmp_path))

    class ForbiddenSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("audit replay must not create VerificationSession")

    monkeypatch.setattr(verification_module, "VerificationSession", ForbiddenSession)
    with pytest.raises(TrainingArtifactConflict, match="prediction domain replay"):
        store.load_prediction_for_audit("prediction:" + "4" * 64)


def test_nested_derived_manifest_resolution_reuses_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    archive = DerivedArchive(layout)
    leaf = _manifest(layout, output_ref="file-sha256:" + "5" * 64, value=5)
    archive.write_artifact_manifest(leaf)
    parent_output = "file-sha256:" + "6" * 64
    parent = DerivedArtifactManifest.create(
        artifact_type="verification-parent",
        payload={"leaf": leaf.artifact_id},
        generated_at=OBSERVED_AT,
        transform_version="verification-parent/1",
        code_version="git:test",
        input_refs=(leaf.artifact_id,),
        output_refs=(parent_output,),
        quality="ready",
    )
    archive.write_artifact_manifest(parent)
    calls = 0
    original_catalog = verification_module._catalog_files

    def tracked_catalog(root: Path):
        nonlocal calls
        calls += 1
        return original_catalog(root)

    monkeypatch.setattr(verification_module, "_catalog_files", tracked_catalog)
    with VerificationSession(layout) as session:
        assert (
            archive._load_artifact_manifest_for_output_ref(
                parent_output,
                verification_session=session,
            )
            == parent
        )
    assert calls == 1


def test_nested_player_profile_resolution_reuses_explicit_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    archive = DerivedArchive(layout)
    evidence = _evidence(layout)
    profile = build_player_profiles(
        (
            PlayerMatchObservation(
                player_id="player:session-profile",
                team_id="team:session-profile",
                match_id="match:session-profile",
                role="forward",
                minutes=90,
                known_at=OBSERVED_AT,
                metrics={"shots": 1.0},
                source_ref=evidence.id.value,
                played_at=OBSERVED_AT,
            ),
        ),
        as_of=OBSERVED_AT,
        minimum_minutes=0,
    ).profiles[0]
    archive.write_player_profiles((profile,), generated_at=OBSERVED_AT)
    parent = DerivedArtifactManifest.create(
        artifact_type="verification-profile-parent",
        payload={"profile": profile.artifact_id},
        generated_at=OBSERVED_AT,
        transform_version="verification-profile-parent/1",
        code_version="git:test",
        input_refs=(profile.artifact_id,),
        output_refs=("file-sha256:" + "7" * 64,),
        quality="ready",
    )
    archive.write_artifact_manifest(parent)
    sessions: list[VerificationSession | None] = []
    original_load = archive.load_player_profile

    def tracked_load(
        artifact_id: str,
        *,
        verification_session: VerificationSession | None = None,
    ):
        sessions.append(verification_session)
        return original_load(
            artifact_id,
            verification_session=verification_session,
        )

    monkeypatch.setattr(archive, "load_player_profile", tracked_load)
    with VerificationSession(layout) as session:
        assert (
            archive.load_artifact_manifest(
                parent.artifact_id,
                verification_session=session,
            )
            == parent
        )
        assert sessions == [session]


def test_manifest_catalog_detects_file_added_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    archive = DerivedArchive(layout)
    first_ref = "file-sha256:" + "d" * 64
    second_ref = "file-sha256:" + "e" * 64
    archive.write_artifact_manifest(_manifest(layout, output_ref=first_ref, value=1))
    late = _manifest(layout, output_ref=second_ref, value=2)
    late_path = archive.artifact_manifest_path(late.artifact_id)
    original = verification_module._load_json
    injected = False

    def add_manifest(path: Path, description: str):
        nonlocal injected
        payload = original(path, description)
        if not injected:
            injected = True
            late_path.parent.mkdir(parents=True, exist_ok=True)
            late_path.write_text(json.dumps(late.to_payload()), encoding="utf-8")
        return payload

    monkeypatch.setattr(verification_module, "_load_json", add_manifest)
    with pytest.raises(ArchiveConflictError, match="changed while being scanned"):
        with VerificationSession(layout) as session:
            session.manifests_for_output_ref(first_ref)


def test_sample_catalog_detects_file_added_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    root = layout.derived / "training-datasets"

    def dataset_payload(digest: str, sample_id: str) -> dict[str, object]:
        return {
            "id": "training-dataset:" + digest,
            "samples": [{"sample_id": sample_id}],
        }

    first_digest = "1" * 64
    second_digest = "2" * 64
    first_path = root / first_digest[:2] / f"{first_digest}.json"
    second_path = root / second_digest[:2] / f"{second_digest}.json"
    first_path.parent.mkdir(parents=True)
    first_path.write_text(
        json.dumps(dataset_payload(first_digest, "sample:first")), encoding="utf-8"
    )
    original = verification_module._load_json
    injected = False

    def add_dataset(path: Path, description: str):
        nonlocal injected
        payload = original(path, description)
        if not injected:
            injected = True
            second_path.parent.mkdir(parents=True)
            second_path.write_text(
                json.dumps(dataset_payload(second_digest, "sample:second")), encoding="utf-8"
            )
        return payload

    monkeypatch.setattr(verification_module, "_load_json", add_dataset)
    with pytest.raises(ArchiveConflictError, match="changed while being scanned"):
        with VerificationSession(layout) as session:
            session.sample_entries("sample:first")


def test_file_proof_detects_same_size_tampering_at_stability_check(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = layout.derived / "proof.bin"
    path.write_bytes(b"abcd")
    expected = hashlib.sha256(b"abcd").hexdigest()

    with pytest.raises(ArchiveConflictError, match="changed"):
        with VerificationSession(layout) as session:
            proof = session.file_proof(path, expected_sha256=expected)
            assert proof.sha256 == expected
            path.write_bytes(b"wxyz")


def test_file_proof_rejects_mutation_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    path = layout.derived / "capture-race.bin"
    path.write_bytes(b"before")
    original = verification_module._hash_file

    def mutate_after_hash(candidate: Path) -> str:
        digest = original(candidate)
        candidate.write_bytes(b"after-and-longer")
        return digest

    monkeypatch.setattr(verification_module, "_hash_file", mutate_after_hash)
    with pytest.raises(ArchiveConflictError, match="changed while being proved"):
        verification_module.FileProof.capture(path)


def test_canonical_session_detects_new_sqlite_sidecar(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    database = layout.canonical / "platform.sqlite3"
    CanonicalStore(database).initialize()

    with pytest.raises(ArchiveConflictError, match="sidecars changed"):
        with VerificationSession(layout) as session:
            session.canonical_connection()
            Path(str(database) + "-unexpected").write_bytes(b"sidecar")


def test_manifest_catalog_rejects_duplicate_output_refs_in_one_file(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    archive = DerivedArchive(layout)
    output_ref = "file-sha256:" + "f" * 64
    manifest = _manifest(layout, output_ref=output_ref, value=1)
    path = archive.artifact_manifest_path(manifest.artifact_id)
    payload = manifest.to_payload()
    payload["output_refs"] = [output_ref, output_ref]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArchiveConflictError, match="duplicate"):
        with VerificationSession(layout) as session:
            session.manifests_for_output_ref(output_ref)


def test_file_proof_rejects_non_text_digest(tmp_path: Path) -> None:
    path = _layout(tmp_path).derived / "digest.bin"
    path.write_bytes(b"digest")

    with pytest.raises(ValueError, match="SHA-256"):
        verification_module.FileProof.capture(path, expected_sha256=123)  # type: ignore[arg-type]


def test_sample_catalog_rejects_malformed_file_instead_of_skipping_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    root = layout.derived / "training-datasets" / "aa"
    root.mkdir(parents=True)
    (root / ("c" * 64 + ".json")).write_text("not-json", encoding="utf-8")

    with pytest.raises(ArchiveConflictError, match="training dataset"):
        with VerificationSession(layout) as session:
            session.sample_entries("sample:missing")
