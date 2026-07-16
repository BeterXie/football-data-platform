from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.domain.ids import (
    MarketSnapshotId,
    MatchId,
    ModelRunId,
    SnapshotId,
)
from football_data_platform.domain.predictions import (
    MarketQuote,
    MarketSnapshot,
    MatchResult90,
    build_score_prediction,
    parse_prediction_payload,
    prediction_payload,
    result_probabilities,
)
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.evaluation.governance import (
    ChallengerEvidence,
    PromotionPolicy,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import (
    evaluate_prediction,
    multiclass_brier,
)
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout

MATCH = MatchId("match:prediction-test")
GENERATED_AT = datetime(2025, 8, 16, 12, 0, tzinfo=UTC)


def _prediction():
    return build_score_prediction(
        match_id=MATCH,
        snapshot_id=SnapshotId("snapshot:" + "a" * 64),
        capture_mode=CaptureMode.RECONSTRUCTED,
        snapshot_quality_status="ready",
        model_run_id=ModelRunId("model-run:dc-v1"),
        model_version="dixon-coles/1",
        generated_at=GENERATED_AT,
        lambda_home=1.7,
        lambda_away=0.8,
        rho=-0.12,
        max_goals=11,
        input_refs=("snapshot:" + "a" * 64, "baseline:v1"),
    )


def test_prediction_schema_has_one_coordinate_explicit_normalized_grid() -> None:
    prediction = _prediction()
    payload = prediction_payload(prediction)

    assert payload["schema_version"] == 1
    assert len(payload["score_cells"]) == 144
    assert len({(cell["home_goals"], cell["away_goals"]) for cell in payload["score_cells"]}) == 144
    assert sum(cell["probability"] for cell in payload["score_cells"]) == pytest.approx(1.0)
    assert sum(result_probabilities(prediction).values()) == pytest.approx(1.0)


def test_prediction_and_market_are_separate_evaluation_inputs() -> None:
    prediction = _prediction()
    result = MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result")
    without_market = evaluate_prediction(
        prediction,
        result,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    market = MarketSnapshot(
        id=MarketSnapshotId("market-snapshot:" + "b" * 64),
        schema_version=1,
        match_id=MATCH,
        source="real-bookmaker-snapshot",
        market_type="result_90",
        observed_at=GENERATED_AT,
        quotes=(
            MarketQuote("home", 2.0),
            MarketQuote("draw", 3.5),
            MarketQuote("away", 4.0),
        ),
        raw_asset_ref="raw-asset:" + "c" * 64,
    )
    with_market = evaluate_prediction(
        prediction,
        result,
        market_snapshot=market,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )

    assert not without_market.market_benchmark.available
    assert without_market.market_benchmark.reason == "market_snapshot_missing"
    assert with_market.market_benchmark.available
    assert with_market.market_benchmark.brier is not None
    assert with_market.capture_mode is CaptureMode.RECONSTRUCTED


def test_brier_and_log_loss_use_known_toy_values() -> None:
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    expected_brier = (0.5 - 1) ** 2 + 0.3**2 + 0.2**2

    assert multiclass_brier(probabilities, "home") == pytest.approx(expected_brier)
    evaluation = evaluate_prediction(
        _prediction(),
        MatchResult90(MATCH, 1, 0, GENERATED_AT, "canonical:result"),
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    assert evaluation.result_log_loss == pytest.approx(
        -math.log(result_probabilities(_prediction())["home"])
    )


def test_prediction_archive_never_mixes_batches_or_market_objects(tmp_path: Path) -> None:
    prediction = _prediction()
    archive = DerivedArchive(DataLayout(tmp_path / "data"))

    first = archive.write_prediction(prediction)
    second = archive.write_prediction(prediction)

    assert first == second
    assert archive.load_prediction_payload(prediction)["id"] == prediction.id.value
    assert first.parent.name == prediction.id.value.removeprefix("prediction:")[:2]


def test_challenger_promotion_requires_an_explicit_policy() -> None:
    evidence = ChallengerEvidence(
        captured_samples=500,
        observation_days=120,
        brier_delta_vs_champion=-0.01,
        log_loss_delta_vs_champion=-0.02,
        reliability_passed=True,
        subgroup_diagnostics_passed=True,
    )

    with pytest.raises(ValueError, match="explicit reviewed"):
        assess_promotion(evidence, policy=None)

    decision = assess_promotion(
        evidence,
        policy=PromotionPolicy(
            minimum_captured_samples=400,
            minimum_observation_days=90,
            maximum_brier_delta=-0.005,
            maximum_log_loss_delta=-0.01,
        ),
    )
    assert decision.promoted


def test_formal_prediction_rejects_preview_snapshot() -> None:
    with pytest.raises(ValueError, match="ready snapshot"):
        build_score_prediction(
            match_id=MATCH,
            snapshot_id=SnapshotId("snapshot:" + "d" * 64),
            capture_mode=CaptureMode.RECONSTRUCTED,
            snapshot_quality_status="preview",
            model_run_id=ModelRunId("model-run:dc-v1"),
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.2,
            lambda_away=1.0,
            rho=-0.1,
            max_goals=11,
            input_refs=(),
        )


def test_prediction_loader_rejects_unknown_or_tampered_schema() -> None:
    payload = prediction_payload(_prediction())

    assert parse_prediction_payload(payload) == _prediction()
    with pytest.raises(ValueError, match="unsupported"):
        parse_prediction_payload({**payload, "schema_version": 99})
    tampered = {**payload, "score_cells": [*payload["score_cells"]]}
    tampered["score_cells"][0] = {
        **tampered["score_cells"][0],
        "probability": 0.99,
    }
    with pytest.raises(ValueError, match="canonical score grid"):
        parse_prediction_payload(tampered)
