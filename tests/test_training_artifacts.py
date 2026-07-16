from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _training_refs import seed_training_references

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
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
FEATURE_REF = "derived-source:" + "a" * 64
LABEL_REF = "canonical:result:match:premier-league-2025-26-001"
MODEL_BYTES = b'{"algorithm":"dixon-coles","parameters":{"rho":-0.1}}\n'
OUTPUT_BYTES = b'{"evaluation":"out-of-time","status":"ready"}\n'
MODEL_REF = MODEL_ARTIFACT_REF_PREFIX + hashlib.sha256(MODEL_BYTES).hexdigest()
OUTPUT_HASH = hashlib.sha256(OUTPUT_BYTES).hexdigest()
OUTPUT_REF = MODEL_OUTPUT_REF_PREFIX + OUTPUT_HASH


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
    status: DatasetStatus | None = None,
    error: str | None = None,
) -> TrainingDatasetManifest:
    if layout is not None:
        feature_ref, label_ref = seed_training_references(layout)
        samples = tuple(
            replace(sample, feature_refs=(feature_ref,), label_ref=label_ref) for sample in samples
        )
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
        generated_at=AS_OF + timedelta(days=1),
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
    status: ModelRunStatus = ModelRunStatus.SUCCEEDED,
    error: str | None = None,
) -> ModelRunArtifact:
    outputs = () if status is ModelRunStatus.FAILED else (OUTPUT_REF,)
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
        started_at=AS_OF + timedelta(days=1),
        ended_at=AS_OF + timedelta(days=1, seconds=2),
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
    path = store.write_model_run(run)
    assert store.write_model_run(run) == path
    assert store.load_model_run(run.model_run_id) == run

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parameters"]["rho"] = -0.2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingArtifactConflict, match="identity"):
        store.load_model_run(run.model_run_id)


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
