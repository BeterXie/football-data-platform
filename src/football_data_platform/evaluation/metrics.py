"""Evaluate predictions against real outcomes and optional real market quotes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketSnapshot,
    MarketSourceValidator,
    MarketStatus,
    MatchResult90,
    MatchResultValidator,
    ScorePrediction,
    exact_score_probability,
    result_probabilities,
    verify_market_snapshot,
    verify_match_result_source,
    verify_score_prediction,
)
from football_data_platform.domain.snapshots import CaptureMode


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    available: bool
    brier: float | None
    log_loss: float | None
    reason: str | None
    probabilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    schema_version: int
    prediction_id: str
    match_id: str
    capture_mode: CaptureMode
    actual_outcome: str
    actual_score: str
    result_brier: float
    result_log_loss: float
    score_log_loss: float
    market_benchmark: BenchmarkScore
    evaluated_at: datetime
    input_refs: tuple[str, ...]


def evaluate_prediction(
    prediction: ScorePrediction,
    result: MatchResult90,
    *,
    evaluated_at: datetime,
    market_snapshot: MarketSnapshot | None = None,
    market_source_validator: MarketSourceValidator | None = None,
    result_validator: MatchResultValidator | None = None,
) -> EvaluationRecord:
    """Build one evaluation sample without dropping missing market data."""

    require_utc(evaluated_at, "evaluated_at")
    verify_score_prediction(prediction)
    if result_validator is not None:
        verify_match_result_source(result, result_validator)
    if prediction.match_id != result.match_id:
        raise ValueError("prediction and result refer to different matches")
    actual_outcome = _outcome(result.home_goals, result.away_goals)
    model_probabilities = result_probabilities(prediction)
    model_brier = multiclass_brier(model_probabilities, actual_outcome)
    model_log_loss = categorical_log_loss(model_probabilities, actual_outcome)
    score_probability = exact_score_probability(prediction, result.home_goals, result.away_goals)
    score_log_loss = -math.log(max(score_probability, 1e-15))

    if market_snapshot is None:
        benchmark = BenchmarkScore(False, None, None, "market_snapshot_missing", ())
        market_ref: tuple[str, ...] = ()
    else:
        if market_snapshot.match_id != prediction.match_id:
            raise ValueError("market snapshot and prediction refer to different matches")
        if market_snapshot.market_type != "result_90":
            raise ValueError("market benchmark must use result_90 quotes")
        if market_source_validator is None:
            raise ValueError("market benchmark requires a raw evidence validator")
        verify_market_snapshot(market_snapshot, source_validator=market_source_validator)
        if market_snapshot.data_kind is not MarketDataKind.REAL:
            raise ValueError("synthetic market quotes cannot establish a benchmark")
        if market_snapshot.status is not MarketStatus.OPEN:
            raise ValueError("market benchmark requires an open market snapshot")
        if market_snapshot.observed_at > prediction.snapshot_as_of:
            raise ValueError("market snapshot was observed after prediction snapshot as_of")
        if market_snapshot.observed_at > evaluated_at:
            raise ValueError("market snapshot was observed after evaluation time")
        _validate_result_market(market_snapshot)
        probabilities = de_vig_probabilities(market_snapshot)
        benchmark = BenchmarkScore(
            True,
            multiclass_brier(probabilities, actual_outcome),
            categorical_log_loss(probabilities, actual_outcome),
            None,
            tuple(probabilities.items()),
        )
        market_ref = (market_snapshot.id.value, market_snapshot.raw_asset_ref)
    if result.known_at > evaluated_at:
        raise ValueError("match result was not known at evaluation time")
    return EvaluationRecord(
        schema_version=1,
        prediction_id=prediction.id.value,
        match_id=prediction.match_id.value,
        capture_mode=prediction.capture_mode,
        actual_outcome=actual_outcome,
        actual_score=f"{result.home_goals}:{result.away_goals}",
        result_brier=model_brier,
        result_log_loss=model_log_loss,
        score_log_loss=score_log_loss,
        market_benchmark=benchmark,
        evaluated_at=evaluated_at,
        input_refs=(prediction.id.value, result.source_ref, *market_ref),
    )


def multiclass_brier(probabilities: dict[str, float], actual: str) -> float:
    _validate_distribution(probabilities)
    if actual not in probabilities:
        raise ValueError(f"actual outcome {actual!r} is absent from probabilities")
    return math.fsum(
        (probability - (1.0 if outcome == actual else 0.0)) ** 2
        for outcome, probability in probabilities.items()
    )


def categorical_log_loss(probabilities: dict[str, float], actual: str) -> float:
    _validate_distribution(probabilities)
    if actual not in probabilities:
        raise ValueError(f"actual outcome {actual!r} is absent from probabilities")
    return -math.log(max(probabilities[actual], 1e-15))


def de_vig_probabilities(snapshot: MarketSnapshot) -> dict[str, float]:
    _validate_result_market(snapshot)
    inverse = {quote.outcome: 1.0 / quote.decimal_odds for quote in snapshot.quotes}
    total = math.fsum(inverse.values())
    if total <= 0:
        raise ValueError("market implied probability sum is not positive")
    probabilities = {outcome: value / total for outcome, value in inverse.items()}
    _validate_distribution(probabilities)
    return probabilities


def _validate_result_market(snapshot: MarketSnapshot) -> None:
    outcomes = {quote.outcome for quote in snapshot.quotes}
    if outcomes != {"home", "draw", "away"} or len(snapshot.quotes) != 3:
        raise ValueError("result_90 market requires exactly home, draw, and away quotes")


def _validate_distribution(probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probability distribution is empty")
    if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
        raise ValueError("probability distribution contains invalid values")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"
