from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _training_refs import seed_training_references

from football_data_platform.domain.ids import RawAssetId
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.domain.training import (
    MODEL_ARTIFACT_REF_PREFIX,
    MODEL_OUTPUT_REF_PREFIX,
    DatasetStatus,
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import load_verified_match_result
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
    parse_training_dataset_payload,
)

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
FEATURE_REF = "derived-source:" + "a" * 64
LABEL_REF = "canonical:result:match:premier-league-2025-26-001"
MODEL_BYTES = b'{"algorithm":"dixon-coles","parameters":{"rho":-0.1}}\n'
OUTPUT_BYTES = b'{"evaluation":"out-of-time","status":"ready"}\n'
MODEL_REF = MODEL_ARTIFACT_REF_PREFIX + hashlib.sha256(MODEL_BYTES).hexdigest()
OUTPUT_HASH = hashlib.sha256(OUTPUT_BYTES).hexdigest()
OUTPUT_REF = MODEL_OUTPUT_REF_PREFIX + OUTPUT_HASH


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _rehash_dataset_payload(payload: dict[str, object]) -> str:
    identity = dict(payload)
    identity.pop("id")
    dataset_id = "training-dataset:" + _json_digest(identity)
    payload["id"] = dataset_id
    return dataset_id


def _sample(
    sample_id: str,
    *,
    capture_mode: CaptureMode = CaptureMode.RECONSTRUCTED,
    qualification_passed: bool = True,
    exclusion_reasons: tuple[str, ...] = (),
    feature_known_at: datetime | None = None,
    label_known_at: datetime | None = None,
    capture_evidence_ref: str | None = None,
    capture_observed_at: datetime | None = None,
    split: str = "train",
    sample_as_of: datetime = AS_OF,
) -> TrainingSample:
    return TrainingSample(
        sample_id=sample_id,
        as_of=sample_as_of,
        feature_known_at=feature_known_at or sample_as_of - timedelta(hours=1),
        label_known_at=label_known_at or sample_as_of + timedelta(hours=4),
        capture_mode=capture_mode,
        qualification="score-model-ready",
        qualification_passed=qualification_passed,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(FEATURE_REF,),
        label_ref=LABEL_REF + ":" + sample_id,
        features={"home_strength": 1.2, "away_strength": 0.9},
        label={"home_goals": 2, "away_goals": 1},
        exclusion_reasons=exclusion_reasons,
        split=split,
        capture_evidence_ref=capture_evidence_ref,
        capture_observed_at=capture_observed_at,
    )


def _dataset(
    samples: tuple[TrainingSample, ...],
    *,
    layout: DataLayout | None = None,
    split_strategy: str = "forward-chaining/1",
    generated_at: datetime | None = None,
    status: DatasetStatus | None = None,
    error: str | None = None,
) -> TrainingDatasetManifest:
    if layout is not None:
        seeded_samples: list[TrainingSample] = []
        for sample in samples:
            assert isinstance(sample.label, dict)
            feature_ref, label_ref = seed_training_references(
                layout,
                label_known_at=sample.label_known_at,
                home_goals=sample.label["home_goals"],
                away_goals=sample.label["away_goals"],
                reference_key=sample.sample_id,
            )
            if sample.capture_mode is CaptureMode.CAPTURED:
                assert sample.capture_evidence_ref is not None
                feature_ref = sample.capture_evidence_ref
            seeded_samples.append(replace(sample, feature_refs=(feature_ref,), label_ref=label_ref))
        samples = tuple(seeded_samples)
    return TrainingDatasetManifest.create(
        dataset_version="score-dataset/1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=AS_OF,
        split_strategy=split_strategy,
        samples=samples,
        generated_at=generated_at or AS_OF + timedelta(days=1),
        transform_version="training-dataset/1",
        code_version="git:test",
        status=status,
        error=error,
    )


def _model_run(
    dataset: TrainingDatasetManifest,
    *,
    cohort: tuple[str, ...],
    capture_mode: CaptureMode,
    run_role: str = "challenger",
    started_at: datetime | None = None,
    status: ModelRunStatus = ModelRunStatus.SUCCEEDED,
    error: str | None = None,
) -> ModelRunArtifact:
    outputs = () if status is ModelRunStatus.FAILED else (OUTPUT_REF,)
    normalized_started_at = started_at or AS_OF + timedelta(days=1)
    return ModelRunArtifact.create(
        model_version="dixon-coles/1",
        run_role=run_role,
        task="score-model",
        dataset_id=dataset.dataset_id,
        feature_version="score-features/1",
        label_version="result-90/1",
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:test",
        environment_version="python-3.11:test-lock",
        started_at=normalized_started_at,
        ended_at=normalized_started_at + timedelta(seconds=2),
        random_seed=17,
        model_artifact_refs=() if status is ModelRunStatus.FAILED else (MODEL_REF,),
        evaluation_cohort=cohort,
        evaluation_capture_mode=capture_mode,
        output_refs=outputs,
        output_hashes=() if not outputs else (OUTPUT_HASH,),
        status=status,
        error=error,
    )


def _write_model_bytes(store: TrainingArtifactStore) -> None:
    assert store.write_model_artifact(MODEL_BYTES) == MODEL_REF
    assert store.write_model_output(OUTPUT_BYTES) == OUTPUT_REF


def _capture_asset(layout: DataLayout, *, observed_at: datetime) -> str:
    asset = RawArchive(layout).archive(
        b'{"snapshot":"captured"}',
        source="prospective-capture",
        source_id="match-001:t24h",
        url="https://example.invalid/capture/match-001",
        observed_at=observed_at,
        target_event_time=AS_OF + timedelta(hours=24),
        collector_version="capture/1",
        media_type="application/json",
    )
    return asset.id.value


def test_dataset_manifest_is_deterministic_and_keeps_exclusions() -> None:
    included = _sample("sample:included")
    excluded = _sample(
        "sample:future",
        qualification_passed=False,
        exclusion_reasons=("feature_after_as_of",),
        feature_known_at=AS_OF + timedelta(minutes=1),
    )

    first = _dataset((excluded, included))
    second = _dataset((included, excluded))

    assert first.dataset_id == second.dataset_id
    assert first.status is DatasetStatus.PARTIAL
    assert [sample.sample_id for sample in first.samples] == [
        "sample:future",
        "sample:included",
    ]
    assert first.included_samples == (included,)
    assert first.excluded_samples == (excluded,)
    assert first.to_payload()["samples"][0]["exclusion_reasons"] == ["feature_after_as_of"]


def test_dataset_rejects_manual_success_and_time_leakage() -> None:
    excluded = _sample(
        "sample:excluded",
        qualification_passed=False,
        exclusion_reasons=("qualification_failed",),
    )
    with pytest.raises(ValueError, match="status must match"):
        _dataset((excluded,), status=DatasetStatus.SUCCEEDED)

    with pytest.raises(ValueError, match="feature is known after"):
        _sample("sample:future", feature_known_at=AS_OF + timedelta(seconds=1))

    with pytest.raises(ValueError, match="label is known at or before"):
        _sample("sample:label-leak", label_known_at=AS_OF)


def test_dataset_create_rejects_generation_before_sample_knowledge() -> None:
    later_label = _sample(
        "sample:later-label",
        label_known_at=AS_OF + timedelta(hours=6),
    )
    with pytest.raises(ValueError, match="label_known_at"):
        _dataset((later_label,), generated_at=AS_OF + timedelta(hours=5))

    later_feature = _sample(
        "sample:later-feature",
        qualification_passed=False,
        exclusion_reasons=("feature_after_as_of",),
        feature_known_at=AS_OF + timedelta(hours=6),
    )
    with pytest.raises(ValueError, match="feature_known_at"):
        _dataset((later_feature,), generated_at=AS_OF + timedelta(hours=5))


def test_dataset_generation_accepts_sample_knowledge_boundary() -> None:
    boundary = AS_OF + timedelta(hours=4)
    sample = _sample("sample:knowledge-boundary", label_known_at=boundary)

    dataset = _dataset((sample,), generated_at=boundary)

    assert dataset.generated_at == sample.label_known_at


@pytest.mark.parametrize(
    "label",
    (
        {"home_goals": 2, "away_goals": 1, "extra": "forged"},
        {"home_goals": 2},
        {"home_goals": -1, "away_goals": 1},
        {"home_goals": True, "away_goals": 1},
    ),
)
def test_result_90_training_label_schema_is_exact(label: object) -> None:
    with pytest.raises(ValueError, match="result-90/1 label"):
        replace(_sample("sample:invalid-label"), label=label)


def test_unknown_training_label_version_keeps_generic_json_contract() -> None:
    sample = replace(
        _sample("sample:future-label"),
        label_version="future-result/1",
        label={"winner": "home", "confidence": 0.75},
    )

    assert sample.label == {"winner": "home", "confidence": 0.75}


def test_failed_dataset_and_model_run_are_persisted(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    excluded = _sample(
        "sample:excluded",
        qualification_passed=False,
        exclusion_reasons=("qualification_failed",),
    )
    dataset = _dataset((excluded,), layout=store.layout)
    assert dataset.status is DatasetStatus.FAILED
    assert dataset.error == "no_eligible_training_samples"
    store.write_dataset(dataset)
    assert store.load_dataset(dataset.dataset_id) == dataset

    failed = _model_run(
        dataset,
        cohort=(),
        capture_mode=CaptureMode.RECONSTRUCTED,
        status=ModelRunStatus.FAILED,
        error="training_process_failed",
    )
    store.write_model_run(failed)
    assert store.load_model_run(failed.model_run_id) == failed


def test_captured_sample_requires_real_raw_evidence(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    evidence_time = AS_OF - timedelta(minutes=5)
    evidence_ref = _capture_asset(layout, observed_at=evidence_time)
    captured = _sample(
        "sample:captured",
        capture_mode=CaptureMode.CAPTURED,
        capture_evidence_ref=evidence_ref,
        capture_observed_at=evidence_time,
    )
    dataset = _dataset((captured,), layout=layout)
    assert evidence_ref in dataset.input_refs
    store = TrainingArtifactStore(layout)
    store.write_dataset(dataset)
    assert store.load_dataset(dataset.dataset_id) == dataset

    forged = replace(captured, capture_evidence_ref="raw-asset:" + "f" * 64)
    forged_dataset = _dataset((forged,), layout=layout)
    with pytest.raises(TrainingArtifactConflict, match="lacks verifiable raw evidence"):
        store.write_dataset(forged_dataset)


def test_captured_sample_rejects_feature_input_after_as_of(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    evidence_time = AS_OF - timedelta(minutes=5)
    evidence_ref = _capture_asset(layout, observed_at=evidence_time)
    captured = _sample(
        "sample:captured-future-feature",
        capture_mode=CaptureMode.CAPTURED,
        capture_evidence_ref=evidence_ref,
        capture_observed_at=evidence_time,
    )
    seeded = _dataset((captured,), layout=layout)
    future_feature = RawArchive(layout).archive(
        b"captured future feature",
        source="captured-feature",
        source_id="future",
        url="https://example.invalid/captured-feature/future",
        observed_at=AS_OF + timedelta(seconds=1),
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    sample = replace(seeded.samples[0], feature_refs=(future_feature.id.value,))
    dataset = _dataset((sample,))

    with pytest.raises(TrainingArtifactConflict, match="feature input follows as_of"):
        store.write_dataset(dataset)


def test_captured_sample_accepts_feature_input_at_as_of(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    evidence_time = AS_OF - timedelta(minutes=5)
    evidence_ref = _capture_asset(layout, observed_at=evidence_time)
    captured = _sample(
        "sample:captured-feature-boundary",
        capture_mode=CaptureMode.CAPTURED,
        capture_evidence_ref=evidence_ref,
        capture_observed_at=evidence_time,
    )
    seeded = _dataset((captured,), layout=layout)
    boundary_feature = RawArchive(layout).archive(
        b"captured boundary feature",
        source="captured-feature",
        source_id="boundary",
        url="https://example.invalid/captured-feature/boundary",
        observed_at=AS_OF,
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    sample = replace(seeded.samples[0], feature_refs=(boundary_feature.id.value,))
    dataset = _dataset((sample,))

    store.write_dataset(dataset)

    assert store.load_dataset(dataset.dataset_id) == dataset


def test_training_store_rejects_backdated_derived_artifact_input(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample("sample:backdated-derived"),), layout=layout)
    future_input = RawArchive(layout).archive(
        b"future derived input",
        source="training-derived-input",
        source_id="future",
        url="https://example.invalid/training-derived-input/future",
        observed_at=AS_OF + timedelta(days=2),
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    manifest = store.derived.write_derived_artifact(
        artifact_type="training-feature",
        schema_version=1,
        payload={"value": "backdated"},
        generated_at=AS_OF + timedelta(days=1),
        transform_version="training-feature/1",
        code_version="git:test",
        input_refs=(future_input.id.value,),
        output_refs=("cohort:" + "b" * 64,),
        status="succeeded",
        quality="ready",
    )
    sample = replace(seeded.samples[0], feature_refs=(manifest.artifact_id,))
    dataset = _dataset((sample,))

    with pytest.raises(TrainingArtifactConflict) as write_error:
        store.write_dataset(dataset)
    assert write_error.value.__cause__ is not None
    assert "generated_at predates input availability" in str(write_error.value.__cause__)

    store._write_json(store.dataset_path(dataset.dataset_id), dataset.to_payload())
    with pytest.raises(TrainingArtifactConflict) as load_error:
        store.load_dataset(dataset.dataset_id)
    assert load_error.value.__cause__ is not None
    assert "generated_at predates input availability" in str(load_error.value.__cause__)


def test_training_store_accepts_derived_artifact_availability_boundary(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample("sample:derived-boundary"),), layout=layout)
    boundary = AS_OF + timedelta(hours=1)
    boundary_input = RawArchive(layout).archive(
        b"boundary derived input",
        source="training-derived-input",
        source_id="boundary",
        url="https://example.invalid/training-derived-input/boundary",
        observed_at=boundary,
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    manifest = store.derived.write_derived_artifact(
        artifact_type="training-feature",
        schema_version=1,
        payload={"value": "boundary"},
        generated_at=boundary,
        transform_version="training-feature/1",
        code_version="git:test",
        input_refs=(boundary_input.id.value,),
        output_refs=("cohort:" + "c" * 64,),
        status="succeeded",
        quality="ready",
    )
    sample = replace(seeded.samples[0], feature_refs=(manifest.artifact_id,))
    dataset = _dataset((sample,))

    store.write_dataset(dataset)

    assert store.load_dataset(dataset.dataset_id) == dataset


@pytest.mark.parametrize("field", ("feature_refs", "label_ref"))
def test_training_dataset_rejects_unavailable_sample_references(tmp_path: Path, field: str) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    valid = _dataset((_sample("sample:valid"),), layout=layout)
    sample = valid.samples[0]
    forged = replace(
        sample,
        **(
            {"feature_refs": ("derived-source:" + "f" * 64,)}
            if field == "feature_refs"
            else {"label_ref": "fact:match_results_90:" + "f" * 64}
        ),
    )
    forged_dataset = _dataset((forged,))

    with pytest.raises(TrainingArtifactConflict, match="reference is unavailable or invalid"):
        store.write_dataset(forged_dataset)


def test_model_run_requires_persisted_dataset_and_matching_capture_mode(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    training = _sample("sample:training", sample_as_of=AS_OF - timedelta(days=1))
    reconstructed = _sample("sample:reconstructed", split="holdout")
    dataset = _dataset((training, reconstructed), layout=store.layout)
    run = _model_run(
        dataset,
        cohort=(reconstructed.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )
    with pytest.raises(TrainingArtifactConflict, match="persisted training dataset"):
        store.write_model_run(run)

    store.write_dataset(dataset)
    claimed_captured = _model_run(
        dataset,
        cohort=(reconstructed.sample_id,),
        capture_mode=CaptureMode.CAPTURED,
    )
    with pytest.raises(ValueError, match="capture mode"):
        store.write_model_run(claimed_captured)

    with pytest.raises(ValueError, match="captured evaluation cohort"):
        _model_run(
            dataset,
            cohort=(reconstructed.sample_id,),
            capture_mode=CaptureMode.RECONSTRUCTED,
            run_role="formal",
        )


def test_failed_model_run_rejects_corrupt_existing_dataset(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    dataset = _dataset((_sample("sample:corrupt"),), layout=layout)
    store.write_dataset(dataset)
    dataset_path = store.dataset_path(dataset.dataset_id)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload["task"] = "tampered-task"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    failed = _model_run(
        dataset,
        cohort=(),
        capture_mode=CaptureMode.RECONSTRUCTED,
        status=ModelRunStatus.FAILED,
        error="training_process_failed",
    )
    with pytest.raises(TrainingArtifactConflict, match="canonical|identity"):
        store.write_model_run(failed)


def test_partial_manifest_preserves_diagnostic_error(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    training = _sample("sample:training", sample_as_of=AS_OF - timedelta(days=1))
    included = _sample("sample:included", split="holdout")
    excluded = _sample(
        "sample:excluded",
        qualification_passed=False,
        exclusion_reasons=("insufficient-sample",),
    )
    dataset = _dataset(
        (training, included, excluded), layout=store.layout, error="one sample excluded"
    )
    store.write_dataset(dataset)
    manifests = list((store.layout.derived / "manifests" / "artifacts").rglob("*.json"))
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    dataset_manifest = next(
        item for item in manifest_payloads if item.get("artifact_type") == "training-dataset"
    )
    assert dataset_manifest["error"] == "one sample excluded"

    run = _model_run(
        dataset,
        cohort=(included.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
        status=ModelRunStatus.PARTIAL,
        error="evaluation report incomplete",
    )
    _write_model_bytes(store)
    store.write_model_run(run)
    manifests = list((store.layout.derived / "manifests" / "artifacts").rglob("*.json"))
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    run_manifest = next(
        item
        for item in manifest_payloads
        if item.get("artifact_type") == "model-run"
        and run.model_run_id in item.get("output_refs", [])
    )
    assert run_manifest["error"] == "evaluation report incomplete"


def test_training_parser_rejects_integer_disguised_as_bool(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    dataset = _dataset((_sample("sample:typed"),), layout=store.layout)
    store.write_dataset(dataset)
    path = store.dataset_path(dataset.dataset_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["qualification_passed"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingArtifactConflict, match="boolean|canonical"):
        store.load_dataset(dataset.dataset_id)


def test_training_store_rejects_rehashed_result_label_extra_key(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    dataset = _dataset((_sample("sample:extra-label-key"),), layout=store.layout)
    store.write_dataset(dataset)
    payload = dataset.to_payload()
    sample_payload = payload["samples"][0]
    sample_payload["label"]["extra"] = "forged"
    sample_payload["label_hash"] = _json_digest(sample_payload["label"])
    forged_id = _rehash_dataset_payload(payload)
    store._write_json(store.dataset_path(forged_id), payload)

    with pytest.raises(TrainingArtifactConflict, match="must contain exactly"):
        store.load_dataset(forged_id)


def test_training_parse_and_load_reject_rehashed_future_sample_knowledge(
    tmp_path: Path,
) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    dataset = _dataset((_sample("sample:future-persisted-label"),), layout=store.layout)
    store.write_dataset(dataset)
    payload = dataset.to_payload()
    payload["generated_at"] = (AS_OF + timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    forged_id = _rehash_dataset_payload(payload)

    with pytest.raises(TrainingArtifactConflict, match="label_known_at"):
        parse_training_dataset_payload(payload)

    store._write_json(store.dataset_path(forged_id), payload)
    with pytest.raises(TrainingArtifactConflict, match="label_known_at"):
        store.load_dataset(forged_id)


def test_training_write_rejects_dataset_generated_before_sample_knowledge(
    tmp_path: Path,
) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    dataset = _dataset((_sample("sample:future-domain-label"),), layout=store.layout)
    forged = replace(dataset, generated_at=AS_OF + timedelta(hours=3))

    with pytest.raises(ValueError, match="label_known_at"):
        store.write_dataset(forged)


@pytest.mark.parametrize("reference_kind", ("raw", "derived"))
def test_training_store_rejects_input_available_after_dataset_generation(
    tmp_path: Path,
    reference_kind: str,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample(f"sample:future-{reference_kind}"),), layout=layout)
    asset = RawArchive(layout).archive(
        b"future training input",
        source="future-training-input",
        source_id=reference_kind,
        url=f"https://example.invalid/future-{reference_kind}",
        observed_at=AS_OF + timedelta(days=2),
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    reference = asset.id.value
    if reference_kind == "derived":
        reference = store.derived.write_snapshot_source(
            value={"future": True},
            input_refs=(asset.id,),
            transform_version="future-training-input/1",
            generated_at=asset.observed_at,
        )
    sample = replace(seeded.samples[0], feature_refs=(reference,))
    dataset = _dataset((sample,))

    with pytest.raises(TrainingArtifactConflict, match="predates input availability"):
        store.write_dataset(dataset)


def test_training_store_rejects_feature_known_before_source_semantics(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample("sample:false-feature-known"),), layout=layout)
    asset = RawArchive(layout).archive(
        b"semantic training input",
        source="semantic-training-input",
        source_id="known-at",
        url="https://example.invalid/semantic-known-at",
        observed_at=AS_OF,
        target_event_time=None,
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    feature_ref = store.derived.write_snapshot_source(
        value={"semantic": True},
        input_refs=(asset.id,),
        transform_version="semantic-training-input/1",
        generated_at=AS_OF,
        known_at=AS_OF,
    )
    sample = replace(
        seeded.samples[0],
        feature_refs=(feature_ref,),
        feature_known_at=AS_OF - timedelta(minutes=1),
    )
    dataset = _dataset((sample,))

    with pytest.raises(TrainingArtifactConflict, match="predates its source semantics"):
        store.write_dataset(dataset)


def test_training_store_uses_raw_observation_for_result_availability(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    dataset = _dataset((_sample("sample:result-observed-at"),), layout=layout)
    store.write_dataset(dataset)
    sample = dataset.samples[0]
    forged_observed_at = (
        (sample.label_known_at - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE match_results_90 SET observed_at = ? WHERE record_id = ?",
            (forged_observed_at, sample.label_ref),
        )
        connection.execute(
            "UPDATE fact_evidence SET observed_at = ? WHERE record_id = ?",
            (forged_observed_at, sample.label_ref),
        )

    with pytest.raises(TrainingArtifactConflict, match="reference is unavailable or invalid"):
        store.load_dataset(dataset.dataset_id)


def test_model_run_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    observed_at = AS_OF - timedelta(minutes=5)
    evidence_ref = _capture_asset(layout, observed_at=observed_at)
    captured = _sample(
        "sample:captured",
        capture_mode=CaptureMode.CAPTURED,
        capture_evidence_ref=evidence_ref,
        capture_observed_at=observed_at,
        split="test",
    )
    training = _sample("sample:training", sample_as_of=AS_OF - timedelta(days=1))
    dataset = _dataset((training, captured), layout=layout)
    store = TrainingArtifactStore(layout)
    store.write_dataset(dataset)
    _write_model_bytes(store)
    run = _model_run(
        dataset,
        cohort=(captured.sample_id,),
        capture_mode=CaptureMode.CAPTURED,
        run_role="formal",
    )
    assert run.started_at == dataset.generated_at
    path = store.write_model_run(run)
    assert store.write_model_run(run) == path
    assert store.load_model_run(run.model_run_id) == run

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parameters"]["rho"] = -0.2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingArtifactConflict, match="identity"):
        store.load_model_run(run.model_run_id)


def test_model_run_rejects_start_before_dataset_generation(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    training = _sample("sample:early-model-train", sample_as_of=AS_OF - timedelta(days=1))
    holdout = _sample("sample:early-model-holdout", split="holdout")
    dataset = _dataset((training, holdout), layout=store.layout)
    store.write_dataset(dataset)
    _write_model_bytes(store)
    run = _model_run(
        dataset,
        cohort=(holdout.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
        started_at=dataset.generated_at - timedelta(seconds=2),
    )

    with pytest.raises(ValueError, match="cannot start before"):
        store.write_model_run(run)


def test_model_run_rejects_train_evaluation_split_and_missing_or_tampered_bytes(
    tmp_path: Path,
) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    train_sample = _sample("sample:train", split="train")
    dataset = _dataset((train_sample,), layout=store.layout)
    store.write_dataset(dataset)
    run = _model_run(
        dataset,
        cohort=(train_sample.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )
    with pytest.raises(ValueError, match="out-of-time split"):
        store.write_model_run(run)

    training = _sample("sample:training", sample_as_of=AS_OF - timedelta(days=1))
    holdout = _sample("sample:holdout", split="holdout")
    dataset = _dataset((training, holdout), layout=store.layout)
    store.write_dataset(dataset)
    run = _model_run(
        dataset,
        cohort=(holdout.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )
    with pytest.raises(TrainingArtifactConflict, match="unavailable"):
        store.write_model_run(run)

    _write_model_bytes(store)
    store.write_model_run(run)
    store.model_artifact_path(MODEL_REF).write_bytes(b"tampered")
    with pytest.raises(TrainingArtifactConflict, match="hash mismatch"):
        store.load_model_run(run.model_run_id)


def test_model_run_rejects_non_temporal_or_mislabeled_holdout(tmp_path: Path) -> None:
    store = TrainingArtifactStore(DataLayout(tmp_path / "data"))
    training = _sample("sample:training", sample_as_of=AS_OF)
    earlier_holdout = _sample(
        "sample:earlier-holdout",
        split="holdout",
        sample_as_of=AS_OF - timedelta(days=1),
    )
    dataset = _dataset((training, earlier_holdout), layout=store.layout)
    store.write_dataset(dataset)
    _write_model_bytes(store)
    run = _model_run(
        dataset,
        cohort=(earlier_holdout.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )
    with pytest.raises(ValueError, match="strictly later"):
        store.write_model_run(run)

    later_holdout = _sample("sample:later", split="holdout", sample_as_of=AS_OF)
    temporal_base = _sample("sample:base", sample_as_of=AS_OF - timedelta(days=1))
    random_dataset = _dataset(
        (temporal_base, later_holdout),
        layout=store.layout,
        split_strategy="random-shuffle/1",
    )
    store.write_dataset(random_dataset)
    random_run = _model_run(
        random_dataset,
        cohort=(later_holdout.sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )
    with pytest.raises(ValueError, match="forward or temporal"):
        store.write_model_run(random_run)


def test_score_dataset_loads_typed_canonical_result(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    dataset = _dataset((_sample("sample:typed-result"),), layout=layout)
    store.write_dataset(dataset)

    assert store.load_dataset(dataset.dataset_id) == dataset
    sample = dataset.samples[0]
    result = load_verified_match_result(
        sample.label_ref,
        archive=RawArchive(layout),
        canonical=CanonicalStore(layout.canonical / "platform.sqlite3"),
    )
    assert (result.home_goals, result.away_goals, result.known_at) == (
        sample.label["home_goals"],
        sample.label["away_goals"],
        sample.label_known_at,
    )


@pytest.mark.parametrize("tampered_goal", (9, 2.5))
def test_score_dataset_rejects_result_goal_tamper_with_preserved_record_id(
    tmp_path: Path,
    tampered_goal: float,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    dataset = _dataset((_sample("sample:goal-tamper"),), layout=layout)
    store.write_dataset(dataset)
    with CanonicalStore(layout.canonical / "platform.sqlite3").connect() as connection:
        connection.execute(
            "UPDATE match_results_90 SET home_goals = ? WHERE record_id = ?",
            (tampered_goal, dataset.samples[0].label_ref),
        )

    with pytest.raises(TrainingArtifactConflict, match="result lineage is invalid"):
        store.load_dataset(dataset.dataset_id)


def test_score_dataset_rejects_result_evidence_swap(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    dataset = _dataset((_sample("sample:evidence-swap"),), layout=layout)
    store.write_dataset(dataset)
    archive = RawArchive(layout)
    replacement = archive.archive(
        b"unrelated-result-evidence",
        source="fbref",
        source_id="unrelated-result-evidence",
        url="https://fbref.example/unrelated-result-evidence",
        observed_at=AS_OF + timedelta(hours=4),
        target_event_time=AS_OF + timedelta(hours=1),
        collector_version="test-training/1",
        media_type="text/plain",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.register_raw_asset(replacement)
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE fact_evidence SET raw_asset_id = ?, observed_at = ? WHERE record_id = ?",
            (
                replacement.id.value,
                replacement.observed_at.isoformat().replace("+00:00", "Z"),
                dataset.samples[0].label_ref,
            ),
        )

    with pytest.raises(TrainingArtifactConflict, match="result lineage is invalid"):
        store.load_dataset(dataset.dataset_id)


def test_score_dataset_rejects_result_raw_byte_tamper(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample("sample:raw-tamper"),), layout=layout)
    archive = RawArchive(layout)
    feature_asset = archive.archive(
        b"independent-training-feature",
        source="test-feature",
        source_id="independent-training-feature",
        url="https://example.invalid/independent-training-feature",
        observed_at=AS_OF,
        target_event_time=None,
        collector_version="test-feature/1",
        media_type="text/plain",
    )
    feature_ref = store.derived.write_snapshot_source(
        value={"independent": True},
        input_refs=(feature_asset.id,),
        transform_version="test-feature/1",
        generated_at=AS_OF,
    )
    sample = replace(seeded.samples[0], feature_refs=(feature_ref,))
    dataset = _dataset((sample,))
    store.write_dataset(dataset)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    with canonical.connect() as connection:
        raw_asset_id = connection.execute(
            "SELECT raw_asset_id FROM match_results_90 WHERE record_id = ?",
            (sample.label_ref,),
        ).fetchone()[0]
    asset = archive.load(RawAssetId(raw_asset_id))
    layout.raw_object_path(asset.checksum).write_bytes(b"tampered-result-bytes")

    with pytest.raises(TrainingArtifactConflict, match="result lineage is invalid"):
        store.load_dataset(dataset.dataset_id)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    (
        ("label", "label goals do not match"),
        ("label_known_at", "label_known_at does not match"),
    ),
)
def test_score_dataset_load_rejects_label_contract_mismatch(
    tmp_path: Path,
    field: str,
    expected_message: str,
) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    seeded = _dataset((_sample(f"sample:{field}-mismatch"),), layout=layout)
    sample = seeded.samples[0]
    forged = replace(
        sample,
        **(
            {"label": {"home_goals": 7, "away_goals": 1}}
            if field == "label"
            else {"label_known_at": sample.label_known_at + timedelta(minutes=1)}
        ),
    )
    dataset = _dataset((forged,))
    store._write_json(store.dataset_path(dataset.dataset_id), dataset.to_payload())

    with pytest.raises(TrainingArtifactConflict, match=expected_message):
        store.load_dataset(dataset.dataset_id)


def test_model_run_rejects_dataset_with_tampered_canonical_result(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = TrainingArtifactStore(layout)
    training = _sample("sample:result-train", sample_as_of=AS_OF - timedelta(days=1))
    holdout = _sample("sample:result-holdout", split="holdout")
    dataset = _dataset((training, holdout), layout=layout)
    store.write_dataset(dataset)
    with CanonicalStore(layout.canonical / "platform.sqlite3").connect() as connection:
        connection.execute(
            "UPDATE match_results_90 SET away_goals = 8 WHERE record_id = ?",
            (dataset.samples[1].label_ref,),
        )
    run = _model_run(
        dataset,
        cohort=(dataset.samples[1].sample_id,),
        capture_mode=CaptureMode.RECONSTRUCTED,
    )

    with pytest.raises(TrainingArtifactConflict, match="result lineage is invalid"):
        store.write_model_run(run)
