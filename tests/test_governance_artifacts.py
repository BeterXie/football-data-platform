from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _prediction_forgery import forge_persisted_prediction
from _training_refs import seed_training_references

import football_data_platform.storage.governance as governance_storage
from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, ModelRunId, SeasonId
from football_data_platform.domain.models import MatchStatus
from football_data_platform.domain.predictions import MatchResult90, build_score_prediction
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
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
    context_contribution,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.governance import (
    GovernanceArtifactConflict,
    GovernanceArtifactStore,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactStore

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]


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


def _artifacts(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"reconstructed governance sample",
        source="test-source",
        source_id="governance-match",
        url="https://example.invalid/governance-match",
        observed_at=AS_OF + timedelta(days=1, hours=4),
        target_event_time=AS_OF + timedelta(hours=24),
        collector_version="test-governance/1",
        media_type="application/json",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(
        load_competition_registry(ROOT / "config" / "competitions.toml"),
        registered_at=AS_OF,
    )
    canonical.register_raw_asset(asset)
    teams = tuple(
        canonical.resolve_or_create_team(
            source="test-source",
            source_id=f"governance-team-{index}",
            canonical_name=f"Governance Team {index}",
            competition_id=CompetitionId("competition:eng.1"),
            observed_at=asset.observed_at,
            raw_asset_id=asset.id,
        )
        for index in range(2)
    )
    match, match_version = canonical.resolve_or_create_match(
        source="test-source",
        source_id="governance-match",
        competition_id=CompetitionId("competition:eng.1"),
        season_id=SeasonId("season:eng.1.2025-26"),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=AS_OF + timedelta(hours=24),
        status=MatchStatus.FINISHED,
        observed_at=asset.observed_at,
        raw_asset_id=asset.id,
    )
    result_fact = CanonicalFactStore(canonical).append_result_90(
        match_id=match.id,
        match_version=match_version.version,
        home_goals=2,
        away_goals=1,
        known_at=AS_OF + timedelta(days=1, hours=2),
        observed_at=asset.observed_at,
        raw_asset_id=asset.id,
    )

    derived = DerivedArchive(layout)
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "governance-baseline",
                teams[0].id.value,
                teams[1].id.value,
                AS_OF - timedelta(days=30),
                AS_OF - timedelta(days=1),
                1.7,
                0.8,
                asset.id.value,
            ),
        ),
        as_of=AS_OF,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=teams[0].id.value,
        away_team_id=teams[1].id.value,
    )
    baseline = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    baseline_source = derived.write_snapshot_source(
        value=baseline,
        input_refs=(asset.id,),
        transform_version="team-baseline-input/2",
        generated_at=asset.observed_at,
    )
    context = {"days_since_previous_match": 6.0}
    context_source = derived.write_snapshot_source(
        value=context,
        input_refs=(asset.id,),
        transform_version="match-context-input/1",
        generated_at=asset.observed_at,
    )
    snapshot = build_snapshot(
        match_id=match.id,
        match_version=match_version.version,
        snapshot_type=SnapshotType.T24H,
        as_of=AS_OF,
        scheduled_kickoff_used=AS_OF + timedelta(hours=24),
        feature_spec_version="prematch-features/1",
        features=(
            SnapshotFeature(
                name="team_baseline",
                value=baseline,
                known_at=AS_OF - timedelta(hours=1),
                source_ref=baseline_source,
                contribution_key="team-baseline",
            ),
            SnapshotFeature(
                name="match_context",
                value=context,
                known_at=AS_OF - timedelta(hours=1),
                source_ref=context_source,
                contribution_key="context:rest-days",
            ),
        ),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        source_validator=derived,
    )
    derived.write_snapshot(snapshot)

    sample = TrainingSample(
        sample_id="sample:reconstructed-001",
        as_of=snapshot.as_of,
        feature_known_at=AS_OF - timedelta(hours=1),
        label_known_at=AS_OF + timedelta(days=1, hours=2),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(snapshot.id.value,),
        label_ref=result_fact.record_id,
        features={"match_id": match.id.value, "home_strength": 1.2},
        label={"home_goals": 2, "away_goals": 1},
        split="test",
    )
    _, training_label_ref = seed_training_references(
        layout,
        label_known_at=AS_OF - timedelta(days=1),
        home_goals=1,
        away_goals=0,
        reference_key="governance-training",
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
        feature_refs=(baseline_source,),
        label_ref=training_label_ref,
        features={"home_strength": 1.1},
        label={"home_goals": 1, "away_goals": 0},
        split="train",
    )
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
        generated_at=AS_OF + timedelta(days=1, hours=4),
        input_refs=(asset.id.value,),
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
            started_at=AS_OF + timedelta(days=1, hours=4),
            ended_at=AS_OF + timedelta(days=1, hours=4, seconds=2),
            random_seed=17,
            model_artifact_refs=(model_artifact_ref,),
            evaluation_cohort=(sample.sample_id,),
            evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
            output_refs=(output_ref,),
            output_hashes=(output_ref.removeprefix("model-output:"),),
            status=ModelRunStatus.SUCCEEDED,
        )

    challenger = model_run(model_ref, "challenger")
    champion = model_run(champion_model_ref, "incumbent-shadow")
    training.write_model_run(challenger)
    training.write_model_run(champion)

    composition = compose_expected_goals(
        lambda_home,
        lambda_away,
        (context_contribution(context, source_ref=context_source),),
    )

    def prediction(model_run: ModelRunArtifact, rho: float):
        value = build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=derived,
            model_run_id=ModelRunId(model_run.model_run_id),
            model_version=model_run.model_version,
            generated_at=AS_OF + timedelta(days=1, hours=6),
            lambda_home=composition.lambda_home,
            lambda_away=composition.lambda_away,
            rho=rho,
            max_goals=11,
            input_refs=(),
            model_run_validator=training,
            expected_goals=composition,
        )
        derived.write_prediction(value, snapshot=snapshot)
        return value

    challenger_prediction = prediction(challenger, -0.12)
    champion_prediction = prediction(champion, -0.04)
    result = MatchResult90(
        match.id,
        2,
        1,
        AS_OF + timedelta(days=1, hours=2),
        result_fact.record_id,
    )

    def evaluation(prediction_value):
        value = evaluate_prediction(
            prediction_value,
            result,
            evaluated_at=AS_OF + timedelta(days=1, hours=7),
            result_validator=CanonicalFactStore(canonical),
            sample_ref=sample.sample_id,
        )
        derived.write_evaluation(value)
        return _evaluation_ref(value)

    challenger_evaluation_ref = evaluation(challenger_prediction)
    champion_evaluation_ref = evaluation(champion_prediction)
    comparison = EvaluationComparison(
        model_run_ref=challenger.model_run_id,
        champion_model_ref=champion.model_run_id,
        cohort_ref=dataset.dataset_id,
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
        generated_at=AS_OF + timedelta(days=1, hours=8),
    )
    store = GovernanceArtifactStore(layout)
    store.write_comparison(comparison)
    evidence = store._aggregate_comparison(comparison)
    return layout, dataset, sample, challenger, champion, comparison, evidence, snapshot


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


def test_governance_artifacts_are_content_addressed_and_replayable(tmp_path: Path) -> None:
    layout, _, _, _, champion, comparison, evidence, _ = _artifacts(tmp_path)
    store = GovernanceArtifactStore(layout)
    policy = _policy(champion.model_run_id)

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


def test_comparison_rejects_prediction_created_before_model_run_completed(
    tmp_path: Path,
) -> None:
    (
        layout,
        dataset,
        sample,
        challenger,
        _,
        comparison,
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
        started_at=AS_OF + timedelta(days=2),
        ended_at=AS_OF + timedelta(days=2, seconds=2),
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
    composition = compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        (context_contribution(context.value, source_ref=context.source_ref),),
    )
    derived = DerivedArchive(layout)
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=derived,
        model_run_id=ModelRunId(future_run.model_run_id),
        model_version=future_run.model_version,
        generated_at=AS_OF + timedelta(days=1, hours=6),
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=(),
        expected_goals=composition,
    )
    derived.write_prediction(prediction, snapshot=snapshot)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    evaluation = evaluate_prediction(
        prediction,
        MatchResult90(
            snapshot.match_id,
            sample.label["home_goals"],
            sample.label["away_goals"],
            sample.label_known_at,
            sample.label_ref,
        ),
        evaluated_at=AS_OF + timedelta(days=1, hours=7),
        result_validator=CanonicalFactStore(canonical),
        sample_ref=sample.sample_id,
    )
    derived.write_evaluation(evaluation)
    pair = replace(
        comparison.pairs[0],
        challenger_evaluation_ref=_evaluation_ref(evaluation),
    )
    future_comparison = replace(
        comparison,
        model_run_ref=future_run.model_run_id,
        pairs=(pair,),
    )

    with pytest.raises(GovernanceArtifactConflict, match="model run completion"):
        GovernanceArtifactStore(layout).write_comparison(future_comparison)


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
