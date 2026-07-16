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
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.evaluation.governance import (
    ChallengerEvidence,
    PromotionPolicy,
    SubgroupDiagnostic,
    assess_promotion,
)
from football_data_platform.storage.derived import DerivedArtifactManifest
from football_data_platform.storage.governance import (
    GovernanceArtifactConflict,
    GovernanceArtifactStore,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactStore

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


def _artifacts(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"prospective snapshot",
        source="prospective-capture",
        source_id="match-001:t24h",
        url="https://example.invalid/match-001",
        observed_at=AS_OF - timedelta(minutes=5),
        target_event_time=AS_OF + timedelta(hours=24),
        collector_version="capture/1",
        media_type="application/json",
    )
    sample = TrainingSample(
        sample_id="sample:captured-001",
        as_of=AS_OF,
        feature_known_at=AS_OF - timedelta(hours=1),
        label_known_at=AS_OF + timedelta(hours=4),
        capture_mode=CaptureMode.CAPTURED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=("derived-source:" + "a" * 64,),
        label_ref="canonical:result:match-001",
        features={"home_strength": 1.2},
        label={"home_goals": 2, "away_goals": 1},
        split="test",
        capture_evidence_ref=asset.id.value,
        capture_observed_at=asset.observed_at,
    )
    training_sample = TrainingSample(
        sample_id="sample:train-001",
        as_of=AS_OF - timedelta(days=2),
        feature_known_at=AS_OF - timedelta(days=3),
        label_known_at=AS_OF - timedelta(days=1),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=("derived-source:" + "b" * 64,),
        label_ref="canonical:result:match-train-001",
        features={"home_strength": 1.1},
        label={"home_goals": 1, "away_goals": 0},
        split="train",
    )
    feature_ref, label_ref = seed_training_references(layout)
    sample = replace(sample, feature_refs=(feature_ref,), label_ref=label_ref)
    training_sample = replace(training_sample, feature_refs=(feature_ref,), label_ref=label_ref)
    dataset = TrainingDatasetManifest.create(
        dataset_version="score-dataset/1",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=AS_OF,
        split_strategy="forward-chaining/1",
        samples=(training_sample, sample),
        generated_at=AS_OF + timedelta(days=1),
        input_refs=(asset.id.value, *sample.feature_refs, *training_sample.feature_refs),
        transform_version="training-dataset/1",
        code_version="git:test",
    )
    training = TrainingArtifactStore(layout)
    training.write_dataset(dataset)
    model_ref = training.write_model_artifact(b"challenger-model")
    champion_model_ref = training.write_model_artifact(b"champion-model")
    output_ref = training.write_model_output(b"evaluation-output")

    def model_run(model_artifact_ref: str, role: str) -> ModelRunArtifact:
        return ModelRunArtifact.create(
            model_version="dixon-coles/1",
            run_role=role,
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
            model_artifact_refs=(model_artifact_ref,),
            evaluation_cohort=(sample.sample_id,),
            evaluation_capture_mode=CaptureMode.CAPTURED,
            output_refs=(output_ref,),
            output_hashes=(output_ref.removeprefix("model-output:"),),
            status=ModelRunStatus.SUCCEEDED,
        )

    challenger = model_run(model_ref, "challenger")
    champion = model_run(champion_model_ref, "champion")
    training.write_model_run(challenger)
    training.write_model_run(champion)

    evaluation_ref = "evaluation:" + "e" * 64
    from football_data_platform.storage.derived import DerivedArchive

    DerivedArchive(layout).write_artifact_manifest(
        DerivedArtifactManifest.create(
            artifact_type="evaluation",
            payload={
                "evaluation_ref": evaluation_ref,
                "sample_count": 1,
                "model_run_ref": challenger.model_run_id,
                "champion_model_ref": champion.model_run_id,
                "cohort_ref": dataset.dataset_id,
                "sample_refs": [sample.sample_id],
            },
            generated_at=AS_OF + timedelta(days=2),
            transform_version="evaluation/1",
            code_version="git:test",
            input_refs=(challenger.model_run_id, champion.model_run_id, sample.sample_id),
            output_refs=(evaluation_ref,),
            status="succeeded",
            quality="ready",
        )
    )
    return layout, dataset, sample, challenger, champion, evaluation_ref


def _policy(champion_ref: str) -> PromotionPolicy:
    return PromotionPolicy(
        policy_version="promotion-policy/1",
        reviewed_by="model-risk-reviewer",
        reviewed_at=AS_OF,
        confidence_method="paired-bootstrap/1",
        rollback_artifact_ref=champion_ref,
        minimum_captured_samples=1,
        minimum_observation_days=1,
        maximum_brier_delta=-0.001,
        maximum_log_loss_delta=-0.001,
        confidence_level=0.95,
    )


def _evidence(dataset_id: str, challenger_ref: str, champion_ref: str, evaluation_ref: str):
    return ChallengerEvidence(
        capture_mode=CaptureMode.CAPTURED,
        captured_samples=1,
        observation_days=30,
        brier_delta_vs_champion=-0.01,
        log_loss_delta_vs_champion=-0.02,
        confidence_interval_passed=True,
        reliability_passed=True,
        subgroup_diagnostics_passed=True,
        model_run_ref=challenger_ref,
        champion_model_ref=champion_ref,
        evaluation_ref=evaluation_ref,
        cohort_ref=dataset_id,
        sample_refs=("sample:captured-001",),
        confidence_level=0.95,
        confidence_interval=(-0.03, -0.001),
        confidence_method="paired-bootstrap/1",
        subgroup_diagnostics=(SubgroupDiagnostic("all", 1, -0.01, -0.02, True),),
        prospective=True,
        evaluated_at=AS_OF + timedelta(days=2),
    )


def test_governance_artifacts_are_content_addressed_and_replayable(tmp_path: Path) -> None:
    layout, dataset, _, challenger, champion, evaluation_ref = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)
    evidence = _evidence(
        dataset.dataset_id, challenger.model_run_id, champion.model_run_id, evaluation_ref
    )

    assert store.write_policy(policy) == store.policy_path(policy.content_id)
    assert store.write_evidence(evidence) == store.evidence_path(evidence.content_id)
    decision = assess_promotion(
        evidence,
        policy=policy,
        reference_validator=store,
        decided_at=AS_OF + timedelta(days=3),
    )
    assert decision.promoted
    store.write_decision(decision)

    assert store.load_policy(policy.content_id) == policy
    assert store.load_evidence(evidence.content_id) == evidence
    assert store.load_decision(decision.content_id) == decision

    # Re-writing the same bytes is idempotent; changing a field changes the ID.
    assert store.write_evidence(evidence) == store.evidence_path(evidence.content_id)
    assert (
        evidence.content_id
        != _evidence(
            dataset.dataset_id,
            challenger.model_run_id,
            champion.model_run_id,
            "evaluation:" + "f" * 64,
        ).content_id
    )


def test_governance_rejects_missing_or_tampered_references(tmp_path: Path) -> None:
    layout, dataset, _, challenger, champion, evaluation_ref = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)
    evidence = _evidence(
        dataset.dataset_id, challenger.model_run_id, champion.model_run_id, evaluation_ref
    )
    store.write_policy(policy)
    store.write_evidence(evidence)

    model_path = store.training.model_artifact_path(next(iter(challenger.model_artifact_refs)))
    model_path.write_bytes(b"tampered")
    with pytest.raises(GovernanceArtifactConflict, match="unavailable|invalid|hash"):
        store.load_evidence(evidence.content_id)


def test_prediction_reference_rejects_bytes_tampered_after_id_assignment(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    store = GovernanceArtifactStore(layout)
    identity = {
        "schema_version": 3,
        "composition_artifact_ref": "score-grid-composition:" + "a" * 64,
    }
    reference = (
        "prediction:"
        + hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    path = layout.derived / "predictions" / reference.removeprefix("prediction:")[:2]
    path.mkdir(parents=True)
    prediction_path = path / f"{reference.removeprefix('prediction:')}.json"
    prediction_path.write_text(
        json.dumps({"id": reference, **identity}, sort_keys=True), encoding="utf-8"
    )
    tampered = {"id": reference, **identity, "extra": "tampered"}
    prediction_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    with pytest.raises(GovernanceArtifactConflict, match="hash|invalid"):
        store.verify_reference(reference)


def test_reconstructed_or_unmarked_captured_evidence_never_promotes(tmp_path: Path) -> None:
    layout, dataset, _, challenger, champion, evaluation_ref = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)
    evidence = _evidence(
        dataset.dataset_id, challenger.model_run_id, champion.model_run_id, evaluation_ref
    )

    unmarked = replace(evidence, prospective=None)
    decision = assess_promotion(
        unmarked,
        policy=policy,
        reference_validator=store,
        decided_at=AS_OF + timedelta(days=3),
    )
    assert not decision.promoted
    assert "prospective_captured_cohort_required" in decision.reason_codes
