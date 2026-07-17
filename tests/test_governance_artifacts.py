from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _formal_training import seed_formal_score_sample
from _prediction_forgery import forge_persisted_prediction

import football_data_platform.storage.governance as governance_storage
from football_data_platform.domain.ids import ModelRunId
from football_data_platform.domain.predictions import MatchResult90, build_score_prediction
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.domain.training import (
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
)
from football_data_platform.evaluation.governance import (
    PAIRED_BOOTSTRAP_RESAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    EvaluationComparison,
    EvaluationPairReference,
    EvaluationSubgroup,
    PromotionPolicy,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import (
    BenchmarkScore,
    categorical_log_loss,
    evaluate_prediction,
    evaluation_record_payload,
    multiclass_brier,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contributions,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.governance import (
    GovernanceArtifactConflict,
    GovernanceArtifactStore,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.training import TrainingArtifactStore

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
_ARTIFACT_TEMPLATE = None


def _evaluation_ref(record) -> str:
    payload = evaluation_record_payload(record)
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return "evaluation:" + digest


def _build_artifact_template(template_path: Path):
    layout = DataLayout(template_path / "data")
    training = TrainingArtifactStore(layout)
    historical_evaluated_at = AS_OF - timedelta(days=2)
    training_sample, _ = seed_formal_score_sample(
        layout,
        key="governance-training",
        kickoff=AS_OF - timedelta(days=20),
        observed_at=historical_evaluated_at,
        evaluated_at=historical_evaluated_at,
        team_indices=(4, 5, 6, 7),
        goals=(1, 0),
        split="train",
    )
    holdout_sample, _ = seed_formal_score_sample(
        layout,
        key="governance-holdout",
        kickoff=AS_OF - timedelta(days=10),
        observed_at=historical_evaluated_at,
        evaluated_at=historical_evaluated_at,
        team_indices=(8, 9, 10, 11),
        goals=(2, 1),
        split="holdout",
    )
    dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/governance-training-2",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        as_of=holdout_sample.as_of,
        split_strategy="forward-chaining/1",
        samples=(training_sample, holdout_sample),
        generated_at=historical_evaluated_at,
        code_version="git:test",
    )
    training.write_formal_dataset(dataset)
    model_ref = training.write_model_artifact(b"challenger-model")
    champion_model_ref = training.write_model_artifact(b"champion-model")
    output_ref = training.write_model_output(b"evaluation-output")

    def model_run(model_artifact_ref: str, role: str) -> ModelRunArtifact:
        return ModelRunArtifact.create(
            model_version="dixon-coles/1",
            run_role=role,
            task="score-model",
            dataset_id=dataset.dataset_id,
            feature_version=dataset.feature_version,
            label_version=dataset.label_version,
            algorithm="dixon-coles",
            parameters={"rho": -0.1, "max_goals": 11},
            code_version="git:test",
            environment_version="python-3.11:test-lock",
            started_at=AS_OF - timedelta(days=1),
            ended_at=AS_OF - timedelta(days=1) + timedelta(seconds=2),
            random_seed=17,
            model_artifact_refs=(model_artifact_ref,),
            evaluation_cohort=(holdout_sample.sample_id,),
            evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
            output_refs=(output_ref,),
            output_hashes=(output_ref.removeprefix("model-output:"),),
            status=ModelRunStatus.SUCCEEDED,
        )

    challenger = model_run(model_ref, "challenger")
    champion = model_run(champion_model_ref, "incumbent-shadow")
    training.write_model_run(challenger)
    training.write_model_run(champion)

    target_kickoff = AS_OF + timedelta(days=1)
    target_evaluated_at = target_kickoff + timedelta(hours=3)
    sample, snapshot = seed_formal_score_sample(
        layout,
        key="governance-evaluation-target",
        kickoff=target_kickoff,
        observed_at=target_evaluated_at,
        evaluated_at=target_evaluated_at,
        team_indices=(0, 1, 2, 3),
        goals=(2, 1),
        split="test",
    )
    evaluation_dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/governance-evaluation-2",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        as_of=sample.as_of,
        split_strategy="prospective-evaluation/1",
        samples=(sample,),
        generated_at=target_evaluated_at,
        code_version="git:test",
    )
    training.write_formal_dataset(evaluation_dataset)

    baseline = next(feature for feature in snapshot.features if feature.name == "team_baseline")
    context = next(feature for feature in snapshot.features if feature.name == "match_context")
    composition = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        context_contributions(
            context.value,
            home_team_id=snapshot.home_team_id.value,
            away_team_id=snapshot.away_team_id.value,
            source_ref=context.source_ref,
            source_validator=DerivedArchive(layout),
        ),
    )

    def prediction(model_run: ModelRunArtifact, rho: float):
        value = build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=DerivedArchive(layout),
            model_run_id=ModelRunId(model_run.model_run_id),
            model_version=model_run.model_version,
            generated_at=AS_OF + timedelta(hours=1),
            lambda_home=composition.lambda_home,
            lambda_away=composition.lambda_away,
            rho=rho,
            max_goals=11,
            input_refs=(),
            model_run_validator=training,
            expected_goals=composition,
        )
        DerivedArchive(layout).write_prediction(value, snapshot=snapshot)
        return value

    challenger_prediction = prediction(challenger, -0.12)
    champion_prediction = prediction(champion, -0.04)
    result = MatchResult90(
        snapshot.match_id,
        sample.label["home_goals"],
        sample.label["away_goals"],
        sample.label_known_at,
        sample.label_ref,
    )

    def evaluation(prediction_value):
        value = evaluate_prediction(
            prediction_value,
            result,
            evaluated_at=target_evaluated_at + timedelta(hours=1),
            result_validator=CanonicalFactStore(
                CanonicalStore(layout.canonical / "platform.sqlite3")
            ),
            sample_ref=sample.sample_id,
        )
        DerivedArchive(layout).write_evaluation(value)
        return _evaluation_ref(value)

    challenger_evaluation_ref = evaluation(challenger_prediction)
    champion_evaluation_ref = evaluation(champion_prediction)
    comparison = EvaluationComparison(
        model_run_ref=challenger.model_run_id,
        champion_model_ref=champion.model_run_id,
        cohort_ref=evaluation_dataset.dataset_id,
        pairs=(
            EvaluationPairReference(
                sample.sample_id,
                challenger_evaluation_ref,
                champion_evaluation_ref,
            ),
        ),
        subgroups=(EvaluationSubgroup("all", (sample.sample_id,)),),
        confidence_level=0.95,
        bootstrap_seed=PAIRED_BOOTSTRAP_SEED,
        bootstrap_resamples=PAIRED_BOOTSTRAP_RESAMPLES,
        generated_at=target_evaluated_at + timedelta(hours=2),
    )
    store = GovernanceArtifactStore(layout)
    store.write_comparison(comparison)
    evidence = store._aggregate_comparison(comparison)
    return layout, dataset, sample, challenger, champion, comparison, evidence, snapshot


def _artifacts(tmp_path: Path):
    global _ARTIFACT_TEMPLATE
    if _ARTIFACT_TEMPLATE is None:
        built = _build_artifact_template(tmp_path.parent / "governance-artifact-template")
        _ARTIFACT_TEMPLATE = (built[0].root, built[1:])
    template_root, artifacts = _ARTIFACT_TEMPLATE
    layout = DataLayout(tmp_path / "data")
    shutil.copytree(template_root, layout.root)
    return (layout, *artifacts)


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


def _write_formal_comparison_cohort(
    training: TrainingArtifactStore,
    source: TrainingDatasetManifest,
    sample: TrainingSample,
    *,
    dataset_version: str,
) -> TrainingDatasetManifest:
    cohort = TrainingDatasetManifest.create_formal(
        dataset_version=dataset_version,
        task=source.task,
        qualification=source.qualification,
        qualification_ruleset_version=source.qualification_ruleset_version,
        feature_version=source.feature_version,
        label_version=source.label_version,
        as_of=sample.as_of,
        split_strategy="prospective-evaluation/1",
        samples=(sample,),
        generated_at=source.generated_at,
        code_version=source.code_version,
    )
    training.write_formal_dataset(cohort)
    return cohort


def _model_run_with_times(
    source: ModelRunArtifact,
    *,
    role: str,
    started_at: datetime,
    ended_at: datetime,
) -> ModelRunArtifact:
    return ModelRunArtifact.create(
        model_version=source.model_version,
        run_role=role,
        task=source.task,
        dataset_id=source.dataset_id,
        feature_version=source.feature_version,
        label_version=source.label_version,
        algorithm=source.algorithm,
        parameters=source.parameters,
        code_version=source.code_version,
        environment_version=source.environment_version,
        started_at=started_at,
        ended_at=ended_at,
        random_seed=source.random_seed,
        model_artifact_refs=source.model_artifact_refs,
        evaluation_cohort=source.evaluation_cohort,
        evaluation_capture_mode=source.evaluation_capture_mode,
        output_refs=source.output_refs,
        output_hashes=source.output_hashes,
        status=source.status,
        error=source.error,
    )


def _semantic_training_sample(key: str) -> TrainingSample:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return TrainingSample(
        sample_id=f"sample:{key}",
        as_of=AS_OF,
        feature_known_at=AS_OF - timedelta(hours=1),
        label_known_at=AS_OF + timedelta(hours=2),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        feature_refs=(),
        label_ref=f"fact:match_results_90:{digest}",
        features={"lambda_home": 1.2, "lambda_away": 0.9},
        label={"home_goals": 1, "away_goals": 0},
        split="test",
        match_id=f"match:{digest}",
        match_version=1,
        snapshot_ref=f"snapshot:{digest}",
        qualification_ref=f"training-qualification:{digest}",
    )


def test_governance_artifacts_are_content_addressed_and_replayable(tmp_path: Path) -> None:
    layout, dataset, sample, challenger, champion, comparison, evidence, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)
    model_samples = dataset.samples

    assert comparison.cohort_ref != dataset.dataset_id
    assert set(comparison.sample_refs).isdisjoint(challenger.evaluation_cohort)
    assert sample.sample_id not in {item.sample_id for item in model_samples}
    assert (sample.match_id, sample.match_version) not in {
        (item.match_id, item.match_version) for item in model_samples
    }
    assert sample.snapshot_ref not in {item.snapshot_ref for item in model_samples}
    assert sample.label_ref not in {item.label_ref for item in model_samples}
    assert sample.qualification_ref not in {item.qualification_ref for item in model_samples}
    assert evidence.capture_mode is CaptureMode.RECONSTRUCTED
    assert evidence.captured_samples == 0
    assert evidence.prospective is False
    assert evidence.evaluated_at == comparison.generated_at
    assert store.write_policy(policy) == store.policy_path(policy.content_id)
    assert store.load_comparison(comparison.content_id) == comparison
    assert store.write_evidence(evidence) == store.evidence_path(evidence.content_id)
    decision = assess_promotion(
        evidence,
        policy=policy,
        reference_validator=store,
        decided_at=AS_OF + timedelta(days=3),
    )
    assert not decision.promoted
    assert "prospective_captured_cohort_required" in decision.reason_codes
    store.write_decision(decision)

    assert store.load_policy(policy.content_id) == policy
    assert store.load_evidence(evidence.content_id) == evidence
    assert store.load_decision(decision.content_id) == decision

    # Re-writing the same bytes is idempotent; changing a field changes the ID.
    assert store.write_evidence(evidence) == store.evidence_path(evidence.content_id)
    assert (
        evidence.content_id
        != replace(evidence, reliability_passed=not evidence.reliability_passed).content_id
    )


@pytest.mark.parametrize(
    "reused_field",
    (
        "sample_id",
        "match_id+match_version",
        "snapshot_ref",
        "label_ref",
        "qualification_ref",
    ),
)
def test_comparison_cohort_rejects_each_reused_model_provenance_key(
    reused_field: str,
) -> None:
    sample = _semantic_training_sample("comparison")
    model_sample = _semantic_training_sample("model")
    replacements = {
        "sample_id": {"sample_id": model_sample.sample_id},
        "match_id+match_version": {
            "match_id": model_sample.match_id,
            "match_version": model_sample.match_version,
        },
        "snapshot_ref": {"snapshot_ref": model_sample.snapshot_ref},
        "label_ref": {"label_ref": model_sample.label_ref},
        "qualification_ref": {"qualification_ref": model_sample.qualification_ref},
    }
    forged = replace(sample, **replacements[reused_field])

    with pytest.raises(GovernanceArtifactConflict, match=reused_field.replace("+", r"\+")):
        governance_storage._verify_independent_comparison_samples(
            (forged,),
            (model_sample,),
        )


def test_comparison_store_rejects_same_id_and_renamed_model_sample(tmp_path: Path) -> None:
    layout, dataset, sample, _, _, comparison, _, _ = _artifacts(tmp_path)
    training = TrainingArtifactStore(layout)
    comparison_dataset = training.load_formal_dataset(comparison.cohort_ref)
    model_sample = next(item for item in dataset.samples if item.split == "holdout")
    attacks = (
        (
            "same-id",
            replace(sample, sample_id=model_sample.sample_id),
            "sample_id",
        ),
        (
            "renamed-model-sample",
            replace(
                model_sample,
                sample_id="sample:renamed-governance-evaluation",
                split="test",
            ),
            r"match_id\+match_version",
        ),
    )

    for name, forged_sample, expected_error in attacks:
        cohort = _write_formal_comparison_cohort(
            training,
            comparison_dataset,
            forged_sample,
            dataset_version=f"score-dataset/governance-{name}-attack-2",
        )
        forged_pair = replace(comparison.pairs[0], sample_ref=forged_sample.sample_id)
        forged = replace(
            comparison,
            cohort_ref=cohort.dataset_id,
            pairs=(forged_pair,),
            subgroups=(EvaluationSubgroup("all", (forged_sample.sample_id,)),),
        )
        with pytest.raises(GovernanceArtifactConflict, match=expected_error):
            GovernanceArtifactStore(layout).write_comparison(forged)


def test_comparison_rejects_swapped_training_cohort_sample_run_and_time(
    tmp_path: Path,
) -> None:
    layout, dataset, _, challenger, champion, comparison, _, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    holdout_ref = challenger.evaluation_cohort[0]
    swapped_pair = replace(comparison.pairs[0], sample_ref=holdout_ref)
    attacks = (
        (
            replace(comparison, cohort_ref=dataset.dataset_id),
            "formal cohort|training samples",
        ),
        (
            replace(
                comparison,
                pairs=(swapped_pair,),
                subgroups=(EvaluationSubgroup("all", (holdout_ref,)),),
            ),
            "formal cohort|unknown samples",
        ),
        (
            replace(
                comparison,
                model_run_ref=champion.model_run_id,
                champion_model_ref=challenger.model_run_id,
            ),
            "model run",
        ),
        (
            replace(comparison, model_run_ref=champion.model_run_id),
            "model run",
        ),
        (
            replace(comparison, generated_at=AS_OF),
            "predates its formal evaluation cohort",
        ),
    )

    for forged, expected_error in attacks:
        with pytest.raises(GovernanceArtifactConflict, match=expected_error):
            store.write_comparison(forged)


def test_comparison_rejects_model_run_completed_after_sample_as_of(tmp_path: Path) -> None:
    layout, _, sample, challenger, _, comparison, _, _ = _artifacts(tmp_path)
    training = TrainingArtifactStore(layout)
    late_run = _model_run_with_times(
        challenger,
        role="late-comparison-challenger",
        started_at=sample.as_of,
        ended_at=sample.as_of + timedelta(seconds=1),
    )
    training.write_model_run(late_run)

    with pytest.raises(
        GovernanceArtifactConflict,
        match="challenger model run completed after formal evaluation sample as_of",
    ):
        GovernanceArtifactStore(layout).write_comparison(
            replace(comparison, model_run_ref=late_run.model_run_id)
        )


def test_comparison_allows_model_run_completed_at_sample_as_of(tmp_path: Path) -> None:
    _, _, sample, challenger, _, _, _, _ = _artifacts(tmp_path)
    boundary_run = _model_run_with_times(
        challenger,
        role="cutoff-comparison-challenger",
        started_at=sample.as_of - timedelta(seconds=1),
        ended_at=sample.as_of,
    )

    governance_storage._verify_comparison_model_timing(
        (("challenger", boundary_run),),
        (sample,),
    )


def test_governance_rejects_missing_or_tampered_references(tmp_path: Path) -> None:
    layout, _, _, challenger, champion, _, evidence, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)
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


def test_governance_rejects_rehashed_semantic_prediction_forgery(tmp_path: Path) -> None:
    layout, _, _, _, _, comparison, _, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    derived = DerivedArchive(layout)
    original = store._load_evaluation_record(comparison.pairs[0].challenger_evaluation_ref)
    forged_ref = forge_persisted_prediction(
        layout,
        original.prediction_id,
        lambda payload: payload.__setitem__(
            "baseline_lambda_home", payload["baseline_lambda_home"] * 2
        ),
    )
    forged_record = replace(
        original,
        prediction_id=forged_ref,
        input_refs=tuple(
            sorted(
                forged_ref if reference == original.prediction_id else reference
                for reference in original.input_refs
            )
        ),
    )
    derived.write_evaluation(forged_record)
    forged_pair = replace(
        comparison.pairs[0],
        challenger_evaluation_ref=_evaluation_ref(forged_record),
    )

    with pytest.raises(GovernanceArtifactConflict, match="baseline.*snapshot"):
        store.write_comparison(replace(comparison, pairs=(forged_pair,)))


def test_reconstructed_or_unmarked_captured_evidence_never_promotes(tmp_path: Path) -> None:
    layout, _, _, _, champion, _, evidence, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)

    decision = assess_promotion(
        evidence,
        policy=policy,
        reference_validator=store,
        decided_at=AS_OF + timedelta(days=3),
    )
    assert not decision.promoted
    assert "prospective_captured_cohort_required" in decision.reason_codes


def test_governance_rejects_self_reported_aggregate_metrics(tmp_path: Path) -> None:
    layout, _, _, _, _, _, evidence, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    tampers = (
        lambda evidence: replace(
            evidence, brier_delta_vs_champion=evidence.brier_delta_vs_champion + 0.1
        ),
        lambda evidence: replace(
            evidence, log_loss_delta_vs_champion=evidence.log_loss_delta_vs_champion + 0.1
        ),
        lambda evidence: replace(evidence, confidence_interval=(-0.9, -0.8)),
        lambda evidence: replace(
            evidence, confidence_interval_passed=not evidence.confidence_interval_passed
        ),
        lambda evidence: replace(evidence, reliability_passed=not evidence.reliability_passed),
        lambda evidence: replace(
            evidence,
            subgroup_diagnostics=(replace(evidence.subgroup_diagnostics[0], brier_delta=0.5),),
        ),
    )

    for tamper in tampers:
        with pytest.raises(GovernanceArtifactConflict, match="recomputed evaluation records"):
            store.write_evidence(tamper(evidence))


def test_comparison_cross_checks_each_evaluation_against_persisted_sources(
    tmp_path: Path,
) -> None:
    layout, _, _, _, _, comparison, _, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    derived = DerivedArchive(layout)
    original = store._load_evaluation_record(comparison.pairs[0].challenger_evaluation_ref)
    probabilities = dict(original.result_probabilities)
    swapped = {
        "home": probabilities["away"],
        "draw": probabilities["draw"],
        "away": probabilities["home"],
    }
    market_probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    forged_records = (
        (
            replace(
                original,
                result_probabilities=tuple(swapped.items()),
                result_brier=multiclass_brier(swapped, original.actual_outcome),
                result_log_loss=categorical_log_loss(swapped, original.actual_outcome),
            ),
            "probabilities",
        ),
        (
            replace(original, model_run_ref=comparison.champion_model_ref),
            "model run",
        ),
        (
            replace(
                original,
                actual_outcome="draw",
                actual_score="1:1",
                result_brier=multiclass_brier(probabilities, "draw"),
                result_log_loss=categorical_log_loss(probabilities, "draw"),
                score_log_loss=1.0,
            ),
            "canonical result|outcome",
        ),
    )

    for forged, expected_error in forged_records:
        derived.write_evaluation(forged)
        forged_pair = replace(
            comparison.pairs[0],
            challenger_evaluation_ref=_evaluation_ref(forged),
        )
        forged_comparison = replace(comparison, pairs=(forged_pair,))
        with pytest.raises(GovernanceArtifactConflict, match=expected_error):
            store.write_comparison(forged_comparison)

    forged_market = replace(
        original,
        market_benchmark=BenchmarkScore(
            available=True,
            brier=multiclass_brier(market_probabilities, original.actual_outcome),
            log_loss=categorical_log_loss(market_probabilities, original.actual_outcome),
            reason=None,
            probabilities=tuple(market_probabilities.items()),
            data_kind="real",
        ),
    )
    with pytest.raises(ValueError, match="market snapshot and raw evidence refs"):
        derived.write_evaluation(forged_market)

    prediction_payload = store._read_prediction_payload(original.prediction_id)
    tampered_prediction = copy.deepcopy(prediction_payload)
    home_probability = tampered_prediction["markets"][0]["outcomes"][0]["probability"]
    away_probability = tampered_prediction["markets"][0]["outcomes"][2]["probability"]
    tampered_prediction["markets"][0]["outcomes"][0]["probability"] = away_probability
    tampered_prediction["markets"][0]["outcomes"][2]["probability"] = home_probability
    identity = dict(tampered_prediction)
    identity.pop("id")
    prediction_ref = (
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
    tampered_prediction["id"] = prediction_ref
    digest = prediction_ref.removeprefix("prediction:")
    prediction_path = layout.derived / "predictions" / digest[:2] / f"{digest}.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps(tampered_prediction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_manifest = derived._load_artifact_manifest_for_output_ref(original.prediction_id)
    derived.write_artifact_manifest(
        DerivedArtifactManifest.create(
            artifact_type="prediction",
            schema_version=original_manifest.schema_version,
            payload=tampered_prediction,
            generated_at=original_manifest.generated_at,
            transform_version=original_manifest.transform_version,
            code_version=original_manifest.code_version,
            input_refs=original_manifest.input_refs,
            output_refs=(prediction_ref,),
            status="succeeded",
            quality=original_manifest.quality,
        )
    )
    forged_prediction_record = replace(
        original,
        prediction_id=prediction_ref,
        input_refs=tuple(
            sorted(
                prediction_ref if reference == original.prediction_id else reference
                for reference in original.input_refs
            )
        ),
    )
    derived.write_evaluation(forged_prediction_record)
    forged_pair = replace(
        comparison.pairs[0],
        challenger_evaluation_ref=_evaluation_ref(forged_prediction_record),
    )
    with pytest.raises(
        GovernanceArtifactConflict,
        match="canonical score grid|market views.*Dixon-Coles grid",
    ):
        store.write_comparison(replace(comparison, pairs=(forged_pair,)))


def test_formal_prediction_rejects_model_run_completed_after_generation(
    tmp_path: Path,
) -> None:
    (
        layout,
        dataset,
        _,
        challenger,
        _,
        _,
        _,
        snapshot,
    ) = _artifacts(tmp_path)
    training = TrainingArtifactStore(layout)
    future_run = ModelRunArtifact.create(
        model_version=challenger.model_version,
        run_role="future-challenger",
        task=challenger.task,
        dataset_id=dataset.dataset_id,
        feature_version=challenger.feature_version,
        label_version=challenger.label_version,
        algorithm=challenger.algorithm,
        parameters=challenger.parameters,
        code_version=challenger.code_version,
        environment_version=challenger.environment_version,
        started_at=AS_OF,
        ended_at=AS_OF + timedelta(hours=2),
        random_seed=challenger.random_seed,
        model_artifact_refs=challenger.model_artifact_refs,
        evaluation_cohort=challenger.evaluation_cohort,
        evaluation_capture_mode=challenger.evaluation_capture_mode,
        output_refs=challenger.output_refs,
        output_hashes=challenger.output_hashes,
        status=ModelRunStatus.SUCCEEDED,
    )
    training.write_model_run(future_run)
    baseline = next(feature for feature in snapshot.features if feature.name == "team_baseline")
    context = next(feature for feature in snapshot.features if feature.name == "match_context")
    derived = DerivedArchive(layout)
    composition = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        context_contributions(
            context.value,
            home_team_id=snapshot.home_team_id.value,
            away_team_id=snapshot.away_team_id.value,
            source_ref=context.source_ref,
            source_validator=derived,
        ),
    )

    with pytest.raises(ValueError, match="model run completion"):
        build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=derived,
            model_run_id=ModelRunId(future_run.model_run_id),
            model_version=future_run.model_version,
            generated_at=AS_OF + timedelta(hours=1),
            lambda_home=composition.lambda_home,
            lambda_away=composition.lambda_away,
            rho=-0.12,
            max_goals=11,
            input_refs=(),
            model_run_validator=training,
            expected_goals=composition,
        )


def test_evaluation_comparison_rejects_cherry_picked_subgroups_and_bootstrap_seed() -> None:
    pair = EvaluationPairReference(
        "sample:test",
        "evaluation:" + "a" * 64,
        "evaluation:" + "b" * 64,
    )
    arguments = {
        "model_run_ref": "model-run:" + "a" * 64,
        "champion_model_ref": "model-run:" + "b" * 64,
        "cohort_ref": "training-dataset:" + "c" * 64,
        "pairs": (pair,),
        "subgroups": (EvaluationSubgroup("all", (pair.sample_ref,)),),
        "confidence_level": 0.95,
        "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "bootstrap_resamples": PAIRED_BOOTSTRAP_RESAMPLES,
        "generated_at": AS_OF,
    }

    with pytest.raises(ValueError, match="only supports the all subgroup"):
        EvaluationComparison(
            **{
                **arguments,
                "subgroups": (
                    *arguments["subgroups"],
                    EvaluationSubgroup("favourable", (pair.sample_ref,)),
                ),
            }
        )
    with pytest.raises(ValueError, match="bootstrap seed"):
        EvaluationComparison(**{**arguments, "bootstrap_seed": PAIRED_BOOTSTRAP_SEED + 1})


def test_captured_governance_is_fail_closed_without_capture_run_validator() -> None:
    assert governance_storage._TRUSTED_CAPTURE_RUN_VALIDATION_AVAILABLE is False
