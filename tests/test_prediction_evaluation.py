from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.domain.ids import (
    MatchId,
    ModelRunId,
    RawAssetId,
    SnapshotId,
    TeamId,
)
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketStatus,
    MatchResult90,
    build_market_snapshot,
    build_score_prediction,
    parse_prediction_payload,
    prediction_payload,
    result_probabilities,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    MODEL_ARTIFACT_REF_PREFIX,
    MODEL_OUTPUT_REF_PREFIX,
    ModelRunArtifact,
    ModelRunStatus,
)
from football_data_platform.evaluation.governance import (
    ChallengerEvidence,
    PromotionPolicy,
    SubgroupDiagnostic,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import (
    evaluate_prediction,
    multiclass_brier,
)
from football_data_platform.features.contributions import (
    ExpectedGoalsContribution,
    compose_expected_goals,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

MATCH = MatchId("match:prediction-test")
HOME = TeamId("team:prediction-home")
AWAY = TeamId("team:prediction-away")
GENERATED_AT = datetime(2025, 8, 16, 12, 0, tzinfo=UTC)
KICKOFF = GENERATED_AT + timedelta(hours=24)
DEFAULT_MODEL_RUN_ID = ModelRunId("model-run:dc-v1")
MODEL_DIGEST = hashlib.sha256(b"prediction-evaluation-model").hexdigest()
OUTPUT_DIGEST = hashlib.sha256(b"prediction-evaluation-output").hexdigest()


def _snapshot(tmp_path: Path, *, ready: bool = True):
    layout = DataLayout(tmp_path / "snapshot-data")
    archive = RawArchive(layout)
    derived = DerivedArchive(layout)
    asset = archive.archive(
        b"prediction snapshot evidence",
        source="test-source",
        source_id="prediction-fixture",
        url="fixture://prediction-fixture",
        observed_at=KICKOFF + timedelta(days=1),
        target_event_time=None,
        collector_version="test-collector/1",
        media_type="application/octet-stream",
    )
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "prediction-baseline",
                HOME.value,
                AWAY.value,
                GENERATED_AT - timedelta(days=30),
                GENERATED_AT - timedelta(days=1),
                1.7,
                0.8,
                asset.id.value,
            ),
        ),
        as_of=GENERATED_AT,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=HOME.value,
        away_team_id=AWAY.value,
    )
    baseline = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    features = (
        SnapshotFeature(
            name="team_baseline",
            value=baseline,
            known_at=GENERATED_AT - timedelta(days=1),
            source_ref=derived.write_snapshot_source(
                value=baseline,
                input_refs=(asset.id,),
                transform_version="team-baseline-input/2",
                generated_at=asset.observed_at,
            ),
            contribution_key="team-baseline",
        ),
    )
    if ready:
        context = {"days_since_previous_match": 6.0}
        features += (
            SnapshotFeature(
                name="match_context",
                value=context,
                known_at=GENERATED_AT - timedelta(days=1),
                source_ref=derived.write_snapshot_source(
                    value=context,
                    input_refs=(asset.id,),
                    transform_version="match-context-input/1",
                    generated_at=asset.observed_at,
                ),
                contribution_key="context:rest-days",
            ),
        )
    return (
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.T24H,
            as_of=GENERATED_AT,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/1",
            features=features,
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        ),
        derived,
    )


def _prediction(
    snapshot_with_validator,
    *,
    generated_at: datetime = GENERATED_AT,
    model_run_id: ModelRunId = DEFAULT_MODEL_RUN_ID,
    model_version: str = "dixon-coles/1",
    model_run_validator=None,
):
    snapshot, validator = snapshot_with_validator
    return build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=validator,
        model_run_id=model_run_id,
        model_version=model_version,
        generated_at=generated_at,
        lambda_home=1.7,
        lambda_away=0.8,
        rho=-0.12,
        max_goals=11,
        input_refs=("snapshot:" + "a" * 64, "baseline:v1"),
        model_run_validator=model_run_validator,
    )


def _model_run(
    *,
    model_version: str = "dixon-coles/1",
    task: str = "score-model",
    ended_at: datetime = GENERATED_AT - timedelta(seconds=1),
    status: ModelRunStatus = ModelRunStatus.SUCCEEDED,
) -> ModelRunArtifact:
    output_refs = (
        () if status is ModelRunStatus.FAILED else (MODEL_OUTPUT_REF_PREFIX + OUTPUT_DIGEST,)
    )
    return ModelRunArtifact.create(
        model_version=model_version,
        run_role="challenger",
        task=task,
        dataset_id="training-dataset:" + "a" * 64,
        feature_version="score-features/1",
        label_version="result-90/1",
        algorithm="dixon-coles",
        parameters={"rho": -0.12, "max_goals": 11},
        code_version="git:test",
        environment_version="python-3.11:test-lock",
        started_at=ended_at - timedelta(seconds=1),
        ended_at=ended_at,
        random_seed=17,
        model_artifact_refs=(
            () if status is ModelRunStatus.FAILED else (MODEL_ARTIFACT_REF_PREFIX + MODEL_DIGEST,)
        ),
        evaluation_cohort=() if status is ModelRunStatus.FAILED else ("sample:test",),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=output_refs,
        output_hashes=() if not output_refs else (OUTPUT_DIGEST,),
        status=status,
        error="training failed" if status is ModelRunStatus.FAILED else None,
    )


class _ModelRunRegistry:
    def __init__(self, artifact: ModelRunArtifact | None) -> None:
        self.artifact = artifact

    def load_model_run(self, model_run_id: str) -> ModelRunArtifact:
        if self.artifact is None:
            raise FileNotFoundError(model_run_id)
        return self.artifact


class _ResultRegistry:
    def __init__(self, expected: MatchResult90) -> None:
        self.expected = expected

    def verify_match_result(self, result: MatchResult90) -> None:
        if result != self.expected:
            raise ValueError("unknown canonical result")


class _GovernanceReferenceRegistry:
    def __init__(self, *references: str) -> None:
        self.references = set(references)

    def verify_reference(self, reference: str) -> None:
        if reference not in self.references:
            raise ValueError(f"unknown governance reference: {reference}")


def _market(
    tmp_path: Path,
    *,
    observed_at: datetime = GENERATED_AT,
    quotes: tuple[MarketQuote, ...] = (
        MarketQuote("home", 2.0),
        MarketQuote("draw", 3.5),
        MarketQuote("away", 4.0),
    ),
    data_kind: MarketDataKind = MarketDataKind.REAL,
    status: MarketStatus = MarketStatus.OPEN,
):
    archive = RawArchive(DataLayout(tmp_path / "market-data"))
    asset = archive.archive(
        b'{"market":"result_90"}',
        source="bookmaker-api",
        source_id="prediction-fixture-result-90",
        url="https://bookmaker.example/prediction-fixture",
        observed_at=observed_at,
        target_event_time=KICKOFF,
        collector_version="market-collector/1",
        media_type="application/json",
    )
    return (
        build_market_snapshot(
            match_id=MATCH,
            market_type="result_90",
            status=status,
            data_kind=data_kind,
            quotes=quotes,
            raw_asset=asset,
        ),
        archive,
        asset,
    )


def test_prediction_schema_has_one_coordinate_explicit_normalized_grid(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    payload = prediction_payload(prediction)

    assert payload["schema_version"] == 3
    assert len(payload["score_cells"]) == 144
    assert len({(cell["home_goals"], cell["away_goals"]) for cell in payload["score_cells"]}) == 144
    assert sum(cell["probability"] for cell in payload["score_cells"]) == pytest.approx(1.0)
    assert sum(result_probabilities(prediction).values()) == pytest.approx(1.0)


def test_prediction_persists_expected_goals_provenance_and_grid_manifest(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    composition = compose_expected_goals(
        1.7,
        0.8,
        (
            ExpectedGoalsContribution(
                contribution_key="context:rest-days:calibration/1",
                lambda_home_multiplier=1.02,
                lambda_away_multiplier=0.98,
                source_ref="derived-source:context",
                version="calibration/1",
            ),
        ),
    )
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=validator,
        model_run_id=DEFAULT_MODEL_RUN_ID,
        model_version="dixon-coles/1",
        generated_at=GENERATED_AT,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=("derived-source:baseline",),
        expected_goals=composition,
        calibration_versions=("calibration/1",),
    )

    payload = prediction_payload(prediction)
    assert payload["baseline_lambda_home"] == pytest.approx(1.7)
    assert payload["baseline_lambda_away"] == pytest.approx(0.8)
    assert payload["contribution_keys"] == ["context:rest-days:calibration/1"]
    assert payload["contribution_multipliers"][0]["lambda_home_multiplier"] == pytest.approx(1.02)
    assert payload["composition_version"] == composition.composition_version
    assert payload["calibration_versions"] == ["calibration/1"]
    assert prediction.composition_artifact_ref.startswith("score-grid-composition:")
    assert DEFAULT_MODEL_RUN_ID.value in prediction.input_refs

    archive = DerivedArchive(DataLayout(tmp_path / "prediction-data"))
    composition_path = archive.write_score_grid_composition(prediction)
    assert composition_path == archive.score_grid_composition_path(
        prediction.composition_artifact_ref
    )
    archive.write_prediction(prediction)
    assert archive.verify_score_grid_composition(prediction).artifact_type == (
        "score-grid-composition"
    )
    assert archive.load_score_grid_composition_payload(prediction.composition_artifact_ref)[
        "rho"
    ] == pytest.approx(-0.12)


def test_prediction_composition_bytes_are_tamper_evident(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    archive = DerivedArchive(DataLayout(tmp_path / "prediction-data"))
    archive.write_prediction(prediction)
    path = archive.score_grid_composition_path(prediction.composition_artifact_ref)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["lambda_home"] = 9.0
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ArchiveConflictError, match="composition"):
        archive.load_prediction_payload(prediction)


def test_prediction_and_market_are_separate_evaluation_inputs(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result")
    without_market = evaluate_prediction(
        prediction,
        result,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    market, market_archive, _ = _market(tmp_path)
    with_market = evaluate_prediction(
        prediction,
        result,
        market_snapshot=market,
        market_source_validator=market_archive,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )

    assert not without_market.market_benchmark.available
    assert without_market.market_benchmark.reason == "market_snapshot_missing"
    assert with_market.market_benchmark.available
    assert with_market.market_benchmark.brier is not None
    assert with_market.capture_mode is CaptureMode.RECONSTRUCTED


def test_evaluation_rejects_a_result_unknown_at_evaluation_time(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(
        MATCH,
        2,
        1,
        GENERATED_AT + timedelta(days=2),
        "canonical:result",
    )

    with pytest.raises(ValueError, match="not known"):
        evaluate_prediction(
            prediction,
            result,
            evaluated_at=GENERATED_AT + timedelta(days=1),
        )


def test_evaluation_can_require_canonical_result_evidence(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(MATCH, 2, 1, GENERATED_AT, "fact:match-results:known")
    validator = _ResultRegistry(result)

    evaluated = evaluate_prediction(
        prediction,
        result,
        evaluated_at=GENERATED_AT + timedelta(days=1),
        result_validator=validator,
    )
    assert "fact:match-results:known" in evaluated.input_refs

    with pytest.raises(ValueError, match="result source evidence"):
        evaluate_prediction(
            prediction,
            replace(result, source_ref="canonical:forged"),
            evaluated_at=GENERATED_AT + timedelta(days=1),
            result_validator=validator,
        )


@pytest.mark.parametrize(
    ("market_options", "expected_error"),
    (
        ({"observed_at": GENERATED_AT + timedelta(seconds=1)}, "after prediction"),
        (
            {"quotes": (MarketQuote("home", 2.0), MarketQuote("away", 4.0))},
            "exactly home, draw, and away",
        ),
        ({"data_kind": MarketDataKind.SYNTHETIC}, "synthetic"),
        ({"status": MarketStatus.SUSPENDED}, "open market"),
    ),
)
def test_market_benchmark_rejects_future_incomplete_synthetic_or_closed_quotes(
    tmp_path: Path,
    market_options: dict[str, object],
    expected_error: str,
) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, _ = _market(tmp_path, **market_options)

    with pytest.raises(ValueError, match=expected_error):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_benchmark_requires_verified_raw_lineage(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, asset = _market(tmp_path)
    unknown_asset = replace(asset, id=RawAssetId("raw-asset:" + "f" * 64))
    unknown_market = build_market_snapshot(
        match_id=MATCH,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=market.quotes,
        raw_asset=unknown_asset,
    )

    with pytest.raises(ValueError, match="raw evidence"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=unknown_market,
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="source metadata"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=replace(market, observed_at=GENERATED_AT - timedelta(minutes=1)),
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_time_is_checked_against_snapshot_as_of_not_generation_time(
    tmp_path: Path,
) -> None:
    prediction = _prediction(
        _snapshot(tmp_path),
        generated_at=KICKOFF + timedelta(hours=1),
    )
    market, archive, _ = _market(tmp_path, observed_at=KICKOFF + timedelta(minutes=1))

    with pytest.raises(ValueError, match="snapshot as_of"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=KICKOFF + timedelta(days=1),
        )


def test_market_benchmark_requires_a_validator_when_present(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, _, _ = _market(tmp_path)

    with pytest.raises(ValueError, match="raw evidence validator"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_benchmark_rejects_quote_unknown_at_evaluation_time(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, _ = _market(tmp_path, observed_at=GENERATED_AT - timedelta(minutes=1))

    with pytest.raises(ValueError, match="after evaluation"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=GENERATED_AT - timedelta(minutes=2),
        )


@pytest.mark.parametrize("odds", (math.nan, math.inf, -1.0, 1.0))
def test_market_quote_rejects_non_finite_or_non_positive_price(odds: float) -> None:
    with pytest.raises(ValueError, match="decimal_odds"):
        MarketQuote("home", odds)


def test_brier_and_log_loss_use_known_toy_values(tmp_path: Path) -> None:
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    expected_brier = (0.5 - 1) ** 2 + 0.3**2 + 0.2**2
    prediction = _prediction(_snapshot(tmp_path))

    assert multiclass_brier(probabilities, "home") == pytest.approx(expected_brier)
    evaluation = evaluate_prediction(
        prediction,
        MatchResult90(MATCH, 1, 0, GENERATED_AT, "canonical:result"),
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    assert evaluation.result_log_loss == pytest.approx(
        -math.log(result_probabilities(prediction)["home"])
    )


def test_prediction_archive_never_mixes_batches_or_market_objects(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    archive = DerivedArchive(DataLayout(tmp_path / "data"))

    first = archive.write_prediction(prediction)
    second = archive.write_prediction(prediction)

    assert first == second
    assert archive.load_prediction_payload(prediction)["id"] == prediction.id.value
    assert first.parent.name == prediction.id.value.removeprefix("prediction:")[:2]


def test_prediction_archive_optionally_requires_a_valid_model_run(tmp_path: Path) -> None:
    artifact = _model_run()
    registry = _ModelRunRegistry(artifact)
    prediction = _prediction(
        _snapshot(tmp_path),
        model_run_id=ModelRunId(artifact.model_run_id),
        model_run_validator=registry,
    )
    archive = DerivedArchive(
        DataLayout(tmp_path / "data"),
        model_run_validator=registry,
    )

    archive.write_prediction(prediction)
    assert archive.load_prediction_payload(prediction)["model_run_id"] == artifact.model_run_id

    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        DerivedArchive(
            DataLayout(tmp_path / "missing-model-data"),
            model_run_validator=_ModelRunRegistry(None),
        ).write_prediction(prediction)


@pytest.mark.parametrize(
    ("artifact_options", "prediction_options", "expected_error"),
    (
        (
            {"model_version": "dixon-coles/2"},
            {"model_version": "dixon-coles/1"},
            "model_version does not match",
        ),
        (
            {"task": "player-profile"},
            {},
            "score-model task",
        ),
        (
            {"ended_at": GENERATED_AT + timedelta(seconds=1)},
            {},
            "model run completion",
        ),
        (
            {"status": ModelRunStatus.PARTIAL},
            {},
            "successful model run",
        ),
    ),
)
def test_prediction_model_run_contract_rejects_mismatched_or_unusable_artifacts(
    tmp_path: Path,
    artifact_options: dict[str, object],
    prediction_options: dict[str, object],
    expected_error: str,
) -> None:
    artifact = _model_run(**artifact_options)
    with pytest.raises(ValueError, match=expected_error):
        _prediction(
            _snapshot(tmp_path),
            model_run_id=ModelRunId(artifact.model_run_id),
            model_run_validator=_ModelRunRegistry(artifact),
            **prediction_options,
        )


def test_evaluation_and_archive_reject_tampered_prediction_content(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    archive = DerivedArchive(DataLayout(tmp_path / "data"))
    shifted_as_of = replace(
        prediction,
        snapshot_as_of=prediction.snapshot_as_of - timedelta(minutes=1),
    )
    tampered_cells = replace(
        prediction,
        score_cells=(
            replace(prediction.score_cells[0], probability=0.99),
            *prediction.score_cells[1:],
        ),
    )

    for tampered in (shifted_as_of, tampered_cells):
        with pytest.raises(ValueError, match="prediction (identity|score cells)"):
            evaluate_prediction(
                tampered,
                MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
                evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
            )
        with pytest.raises(ValueError, match="prediction (identity|score cells)"):
            archive.write_prediction(tampered)


def test_challenger_promotion_requires_an_explicit_policy() -> None:
    challenger_ref = "model-run:" + "a" * 64
    champion_ref = "model-run:" + "b" * 64
    evaluation_ref = "evaluation:" + "c" * 64
    cohort_ref = "training-dataset:" + "d" * 64
    sample_ref = "sample:captured-001"
    rollback_ref = "model-run:" + "e" * 64
    evidence = ChallengerEvidence(
        capture_mode=CaptureMode.CAPTURED,
        captured_samples=1,
        observation_days=120,
        brier_delta_vs_champion=-0.01,
        log_loss_delta_vs_champion=-0.02,
        confidence_interval_passed=True,
        reliability_passed=True,
        subgroup_diagnostics_passed=True,
        model_run_ref=challenger_ref,
        champion_model_ref=champion_ref,
        evaluation_ref=evaluation_ref,
        cohort_ref=cohort_ref,
        sample_refs=(sample_ref,),
        confidence_level=0.95,
        confidence_interval=(-0.02, -0.005),
        confidence_method="paired-bootstrap/1",
        subgroup_diagnostics=(SubgroupDiagnostic("all", 1, -0.01, -0.02, True),),
        prospective=True,
        evaluated_at=GENERATED_AT,
    )

    with pytest.raises(ValueError, match="explicit reviewed"):
        assess_promotion(evidence, policy=None)

    decision = assess_promotion(
        evidence,
        policy=PromotionPolicy(
            policy_version="promotion-policy/1",
            reviewed_by="model-risk-reviewer",
            reviewed_at=GENERATED_AT,
            confidence_method="paired-bootstrap/1",
            rollback_target=rollback_ref,
            minimum_captured_samples=1,
            minimum_observation_days=90,
            maximum_brier_delta=-0.005,
            maximum_log_loss_delta=-0.01,
            confidence_level=0.95,
        ),
        reference_validator=_GovernanceReferenceRegistry(
            challenger_ref,
            champion_ref,
            evaluation_ref,
            cohort_ref,
            sample_ref,
            rollback_ref,
        ),
        decided_at=GENERATED_AT + timedelta(days=1),
    )
    assert decision.promoted


def test_challenger_promotion_rejects_invalid_or_non_captured_evidence() -> None:
    policy = PromotionPolicy(
        policy_version="promotion-policy/1",
        reviewed_by="model-risk-reviewer",
        reviewed_at=GENERATED_AT,
        confidence_method="paired-bootstrap/1",
        rollback_target="model-run:" + "e" * 64,
        minimum_captured_samples=400,
        minimum_observation_days=90,
        maximum_brier_delta=-0.005,
        maximum_log_loss_delta=-0.01,
    )
    common = {
        "capture_mode": CaptureMode.CAPTURED,
        "captured_samples": 500,
        "observation_days": 120,
        "brier_delta_vs_champion": -0.01,
        "log_loss_delta_vs_champion": -0.02,
        "confidence_interval_passed": True,
        "reliability_passed": True,
        "subgroup_diagnostics_passed": True,
    }

    with pytest.raises(ValueError, match="finite"):
        ChallengerEvidence(**{**common, "brier_delta_vs_champion": math.nan})
    with pytest.raises(ValueError, match="non-negative"):
        ChallengerEvidence(**{**common, "captured_samples": -1})

    decision = assess_promotion(
        ChallengerEvidence(**{**common, "capture_mode": CaptureMode.RECONSTRUCTED}),
        policy=policy,
    )
    assert not decision.promoted
    assert "prospective_captured_cohort_required" in decision.reason_codes


def test_promotion_policy_requires_review_and_rollback_metadata() -> None:
    with pytest.raises(ValueError, match="reviewed_by"):
        PromotionPolicy(
            policy_version="promotion-policy/1",
            reviewed_by="",
            reviewed_at=GENERATED_AT,
            confidence_method="paired-bootstrap/1",
            rollback_target="model-run:champion-v1",
            minimum_captured_samples=400,
            minimum_observation_days=90,
            maximum_brier_delta=-0.005,
            maximum_log_loss_delta=-0.01,
        )


def test_formal_prediction_rejects_preview_snapshot(tmp_path: Path) -> None:
    preview, validator = _snapshot(tmp_path, ready=False)

    with pytest.raises(ValueError, match="ready snapshot"):
        build_score_prediction(
            snapshot=preview,
            snapshot_validator=validator,
            model_run_id=ModelRunId("model-run:dc-v1"),
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.2,
            lambda_away=1.0,
            rho=-0.1,
            max_goals=11,
            input_refs=(),
        )


def test_formal_prediction_rejects_tampered_snapshot_identity(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    tampered = replace(snapshot, id=SnapshotId("snapshot:" + "d" * 64))

    with pytest.raises(ValueError, match="snapshot identity"):
        build_score_prediction(
            snapshot=tampered,
            snapshot_validator=validator,
            model_run_id=ModelRunId("model-run:dc-v1"),
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.2,
            lambda_away=1.0,
            rho=-0.1,
            max_goals=11,
            input_refs=(),
        )


def test_prediction_loader_rejects_unknown_or_tampered_schema(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    prediction = _prediction((snapshot, validator))
    payload = prediction_payload(prediction)

    assert (
        parse_prediction_payload(
            payload,
            snapshot=snapshot,
            snapshot_validator=validator,
        )
        == prediction
    )
    for unsupported_version in (1, 99):
        with pytest.raises(ValueError, match="unsupported"):
            parse_prediction_payload(
                {**payload, "schema_version": unsupported_version},
                snapshot=snapshot,
                snapshot_validator=validator,
            )
    tampered = {**payload, "score_cells": [*payload["score_cells"]]}
    tampered["score_cells"][0] = {
        **tampered["score_cells"][0],
        "probability": 0.99,
    }
    with pytest.raises(ValueError, match="canonical score grid"):
        parse_prediction_payload(
            tampered,
            snapshot=snapshot,
            snapshot_validator=validator,
        )
