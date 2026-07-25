from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArtifactManifest,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)

GENERATED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
INPUT_REF = "file-sha256:" + "1" * 64


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_payload(prefix: str, identity: dict[str, object]) -> tuple[str, dict[str, object]]:
    reference = prefix + hashlib.sha256(_canonical_json(identity)).hexdigest()
    return reference, {"id": reference, **identity}


def _manifest(
    *,
    output_ref: str,
    payload: dict[str, object],
    artifact_type: str,
    code_version: str = DERIVED_CODE_VERSION,
) -> DerivedArtifactManifest:
    return DerivedArtifactManifest.create(
        artifact_type=artifact_type,
        schema_version=1,
        payload=payload,
        generated_at=GENERATED_AT,
        transform_version="formal-manifest-gate-test/1",
        code_version=code_version,
        input_refs=(INPUT_REF,),
        output_refs=(output_ref,),
        status="succeeded",
        quality="ready",
    )


def _write_content(
    store: TrainingArtifactStore,
    *,
    directory: str,
    prefix: str,
    reference: str,
    payload: dict[str, object],
) -> None:
    digest = reference.removeprefix(prefix)
    store._write_json(
        store.layout.derived / directory / digest[:2] / f"{digest}.json",
        payload,
    )


@pytest.mark.parametrize(
    ("prefix", "artifact_type"),
    (
        ("training-qualification:", "training-qualification"),
        ("snapshot:", "prematch-snapshot"),
        ("prediction:", "prediction"),
    ),
)
@pytest.mark.parametrize("attack", ("duplicate", "missing", "replace-code", "malformed"))
def test_fixed_code_manifest_gate_rejects_non_exact_candidates(
    tmp_path: Path,
    prefix: str,
    artifact_type: str,
    attack: str,
) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    output_ref, payload = _content_payload(prefix, {"probe": artifact_type})
    original = _manifest(
        output_ref=output_ref,
        payload=payload,
        artifact_type=artifact_type,
    )
    replacement = _manifest(
        output_ref=output_ref,
        payload=payload,
        artifact_type=artifact_type,
        code_version="attacker-replay/1",
    )

    if attack == "duplicate":
        store.derived.write_artifact_manifest(original)
        store.derived.write_artifact_manifest(replacement)
    elif attack == "replace-code":
        store.derived.write_artifact_manifest(replacement)
    elif attack == "malformed":
        store._write_json(
            store.layout.derived / "manifests" / "artifacts" / "malformed.json",
            {"output_refs": [output_ref]},
        )

    with pytest.raises(
        TrainingArtifactConflict,
        match="exactly one derived artifact manifest|unavailable or invalid|fixed producer",
    ):
        store._load_fixed_code_manifest_for_output_ref(output_ref)


@pytest.mark.parametrize("artifact_role", ("qualification", "prediction", "snapshot"))
def test_formal_loaders_reject_replace_only_manifest_field_attack(
    tmp_path: Path,
    artifact_role: str,
) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))

    if artifact_role == "qualification":
        reference, payload = _content_payload("training-qualification:", {"probe": "qualification"})
        _write_content(
            store,
            directory="training-qualifications",
            prefix="training-qualification:",
            reference=reference,
            payload=payload,
        )
        store.derived.write_artifact_manifest(
            _manifest(
                output_ref=reference,
                payload=payload,
                artifact_type="tampered-training-qualification",
            )
        )
        loader = store.load_training_qualification
    elif artifact_role == "prediction":
        reference, payload = _content_payload(
            "prediction:", {"snapshot_id": "snapshot:" + "2" * 64}
        )
        _write_content(
            store,
            directory="predictions",
            prefix="prediction:",
            reference=reference,
            payload=payload,
        )
        store.derived.write_artifact_manifest(
            _manifest(
                output_ref=reference,
                payload=payload,
                artifact_type="tampered-prediction",
            )
        )
        loader = store.load_verified_prediction
    else:
        snapshot_ref, snapshot_payload = _content_payload("snapshot:", {"probe": "snapshot"})
        _write_content(
            store,
            directory="snapshots",
            prefix="snapshot:",
            reference=snapshot_ref,
            payload=snapshot_payload,
        )
        store.derived.write_artifact_manifest(
            _manifest(
                output_ref=snapshot_ref,
                payload=snapshot_payload,
                artifact_type="tampered-prematch-snapshot",
            )
        )
        reference, payload = _content_payload("prediction:", {"snapshot_id": snapshot_ref})
        _write_content(
            store,
            directory="predictions",
            prefix="prediction:",
            reference=reference,
            payload=payload,
        )
        store.derived.write_artifact_manifest(
            _manifest(
                output_ref=reference,
                payload=payload,
                artifact_type="prediction",
            )
        )
        loader = store.load_verified_prediction

    with pytest.raises(TrainingArtifactConflict, match="manifest does not match stored bytes"):
        loader(reference)
