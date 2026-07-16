from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.storage.derived import (
    ArchiveConflictError,
    DerivedArchive,
    DerivedArtifactManifest,
    RunManifest,
)
from football_data_platform.storage.layout import DataLayout

GENERATED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
RAW_REF = "raw-asset:" + "a" * 64


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
