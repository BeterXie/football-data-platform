"""Evaluate predictions against real outcomes and optional real market quotes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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

EVALUATION_RECORD_SCHEMA_VERSION = 2
EVALUATION_RECORD_TYPE = "evaluation-record"
_RESULT_OUTCOMES = ("home", "draw", "away")


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    available: bool
    brier: float | None
    log_loss: float | None
    reason: str | None
    probabilities: tuple[tuple[str, float], ...]
    data_kind: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    schema_version: int
    record_type: str
    prediction_id: str
    model_run_ref: str
    sample_ref: str | None
    match_id: str
    capture_mode: CaptureMode
    actual_outcome: str
    actual_score: str
    result_probabilities: tuple[tuple[str, float], ...]
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
    sample_ref: str | None = None,
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
        benchmark = BenchmarkScore(False, None, None, "market_snapshot_missing", (), None)
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
            market_snapshot.data_kind.value,
        )
        market_ref = (market_snapshot.id.value, market_snapshot.raw_asset_ref)
    if result.known_at > evaluated_at:
        raise ValueError("match result was not known at evaluation time")
    return EvaluationRecord(
        schema_version=EVALUATION_RECORD_SCHEMA_VERSION,
        record_type=EVALUATION_RECORD_TYPE,
        prediction_id=prediction.id.value,
        model_run_ref=prediction.model_run_id.value,
        sample_ref=sample_ref,
        match_id=prediction.match_id.value,
        capture_mode=prediction.capture_mode,
        actual_outcome=actual_outcome,
        actual_score=f"{result.home_goals}:{result.away_goals}",
        result_probabilities=tuple(model_probabilities.items()),
        result_brier=model_brier,
        result_log_loss=model_log_loss,
        score_log_loss=score_log_loss,
        market_benchmark=benchmark,
        evaluated_at=evaluated_at,
        input_refs=tuple(sorted({prediction.id.value, result.source_ref, *market_ref})),
    )


def verify_evaluation_record(record: EvaluationRecord) -> None:
    """Recompute every score carried by one persisted evaluation record."""

    if not isinstance(record, EvaluationRecord):
        raise TypeError("record must be an EvaluationRecord")
    if (
        record.schema_version != EVALUATION_RECORD_SCHEMA_VERSION
        or record.record_type != EVALUATION_RECORD_TYPE
    ):
        raise ValueError("unsupported evaluation record schema")
    for name, value in (
        ("prediction_id", record.prediction_id),
        ("model_run_ref", record.model_run_ref),
        ("match_id", record.match_id),
        ("actual_outcome", record.actual_outcome),
        ("actual_score", record.actual_score),
    ):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"evaluation {name} must be non-empty text")
    if record.sample_ref is not None and (
        not isinstance(record.sample_ref, str)
        or not record.sample_ref
        or record.sample_ref.strip() != record.sample_ref
    ):
        raise ValueError("evaluation sample_ref must be non-empty text when present")
    if not isinstance(record.capture_mode, CaptureMode):
        raise TypeError("evaluation capture_mode must be a CaptureMode")
    require_utc(record.evaluated_at, "evaluated_at")
    probabilities = _probability_mapping(record.result_probabilities, "result_probabilities")
    if tuple(probabilities) != _RESULT_OUTCOMES:
        raise ValueError("evaluation result probabilities must use home, draw, away order")
    home_goals, away_goals = _parse_actual_score(record.actual_score)
    if record.actual_outcome != _outcome(home_goals, away_goals):
        raise ValueError("evaluation actual outcome does not match actual score")
    expected_brier = multiclass_brier(probabilities, record.actual_outcome)
    expected_log_loss = categorical_log_loss(probabilities, record.actual_outcome)
    if not math.isclose(record.result_brier, expected_brier, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("evaluation result_brier is not reproducible")
    if not math.isclose(record.result_log_loss, expected_log_loss, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("evaluation result_log_loss is not reproducible")
    if not math.isfinite(record.score_log_loss) or record.score_log_loss < 0:
        raise ValueError("evaluation score_log_loss must be finite and non-negative")
    if (
        record.input_refs != tuple(sorted(set(record.input_refs)))
        or record.prediction_id not in record.input_refs
        or any(not isinstance(item, str) or not item for item in record.input_refs)
    ):
        raise ValueError("evaluation input_refs must be unique, sorted, and cite the prediction")
    _verify_benchmark(record.market_benchmark, actual_outcome=record.actual_outcome)
    market_refs = tuple(
        reference for reference in record.input_refs if reference.startswith("market-snapshot:")
    )
    if record.market_benchmark.available:
        raw_refs = tuple(
            reference for reference in record.input_refs if reference.startswith("raw-asset:")
        )
        if (
            len(market_refs) != 1
            or not _is_digest_ref(market_refs[0], "market-snapshot:")
            or not raw_refs
            or any(not _is_digest_ref(reference, "raw-asset:") for reference in raw_refs)
        ):
            raise ValueError(
                "available market benchmark requires one market snapshot and raw evidence refs"
            )
    elif market_refs:
        raise ValueError("unavailable market benchmark cannot cite a market snapshot")


def evaluation_record_payload(record: EvaluationRecord) -> dict[str, Any]:
    verify_evaluation_record(record)
    return {
        "schema_version": record.schema_version,
        "record_type": record.record_type,
        "prediction_id": record.prediction_id,
        "model_run_ref": record.model_run_ref,
        "sample_ref": record.sample_ref,
        "match_id": record.match_id,
        "capture_mode": record.capture_mode.value,
        "actual_outcome": record.actual_outcome,
        "actual_score": record.actual_score,
        "result_probabilities": [list(item) for item in record.result_probabilities],
        "result_brier": record.result_brier,
        "result_log_loss": record.result_log_loss,
        "score_log_loss": record.score_log_loss,
        "market_benchmark": _benchmark_payload(record.market_benchmark),
        "evaluated_at": record.evaluated_at.isoformat().replace("+00:00", "Z"),
        "input_refs": list(record.input_refs),
    }


def parse_evaluation_record_payload(payload: Mapping[str, Any]) -> EvaluationRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation record payload must be an object")
    try:
        benchmark_payload = payload["market_benchmark"]
        if not isinstance(benchmark_payload, Mapping):
            raise ValueError("market_benchmark must be an object")
        evaluated_at = datetime.fromisoformat(str(payload["evaluated_at"]).replace("Z", "+00:00"))
        record = EvaluationRecord(
            schema_version=int(payload["schema_version"]),
            record_type=str(payload["record_type"]),
            prediction_id=str(payload["prediction_id"]),
            model_run_ref=str(payload["model_run_ref"]),
            sample_ref=None if payload.get("sample_ref") is None else str(payload["sample_ref"]),
            match_id=str(payload["match_id"]),
            capture_mode=CaptureMode(str(payload["capture_mode"])),
            actual_outcome=str(payload["actual_outcome"]),
            actual_score=str(payload["actual_score"]),
            result_probabilities=_probability_pairs(payload["result_probabilities"]),
            result_brier=float(payload["result_brier"]),
            result_log_loss=float(payload["result_log_loss"]),
            score_log_loss=float(payload["score_log_loss"]),
            market_benchmark=BenchmarkScore(
                available=_strict_bool(benchmark_payload["available"], "market available"),
                brier=_optional_float(benchmark_payload.get("brier")),
                log_loss=_optional_float(benchmark_payload.get("log_loss")),
                reason=(
                    None
                    if benchmark_payload.get("reason") is None
                    else str(benchmark_payload["reason"])
                ),
                probabilities=_probability_pairs(benchmark_payload["probabilities"]),
                data_kind=(
                    None
                    if benchmark_payload.get("data_kind") is None
                    else str(benchmark_payload["data_kind"])
                ),
            ),
            evaluated_at=evaluated_at,
            input_refs=tuple(str(item) for item in payload["input_refs"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid evaluation record payload: {error}") from error
    verify_evaluation_record(record)
    if dict(payload) != evaluation_record_payload(record):
        raise ValueError("evaluation record payload is not canonical")
    return record


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


def _verify_benchmark(benchmark: BenchmarkScore, *, actual_outcome: str) -> None:
    if not isinstance(benchmark, BenchmarkScore):
        raise TypeError("market_benchmark must be a BenchmarkScore")
    if benchmark.available:
        if benchmark.data_kind != MarketDataKind.REAL.value:
            raise ValueError("available market benchmark must be real")
        if benchmark.reason is not None:
            raise ValueError("available market benchmark cannot contain a reason")
        probabilities = _probability_mapping(benchmark.probabilities, "market probabilities")
        if tuple(probabilities) != _RESULT_OUTCOMES:
            raise ValueError("market probabilities must use home, draw, away order")
        expected_brier = multiclass_brier(probabilities, actual_outcome)
        expected_log_loss = categorical_log_loss(probabilities, actual_outcome)
        if benchmark.brier is None or not math.isclose(
            benchmark.brier, expected_brier, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("market benchmark brier is not reproducible")
        if benchmark.log_loss is None or not math.isclose(
            benchmark.log_loss, expected_log_loss, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("market benchmark log_loss is not reproducible")
        return
    if (
        benchmark.brier is not None
        or benchmark.log_loss is not None
        or benchmark.probabilities
        or benchmark.data_kind is not None
        or not benchmark.reason
    ):
        raise ValueError("unavailable market benchmark contains usable values")


def _benchmark_payload(benchmark: BenchmarkScore) -> dict[str, Any]:
    return {
        "available": benchmark.available,
        "brier": benchmark.brier,
        "log_loss": benchmark.log_loss,
        "reason": benchmark.reason,
        "probabilities": [list(item) for item in benchmark.probabilities],
        "data_kind": benchmark.data_kind,
    }


def _probability_mapping(value: Any, field_name: str) -> dict[str, float]:
    pairs = _probability_pairs(value)
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError(f"{field_name} outcomes must be unique")
    return result


def _probability_pairs(value: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError("probabilities must be a sequence")
    pairs: list[tuple[str, float]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], (int, float))
            or isinstance(item[1], bool)
        ):
            raise ValueError("probability entries must be outcome/value pairs")
        pairs.append((item[0], float(item[1])))
    return tuple(pairs)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("optional score must be numeric")
    return float(value)


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _parse_actual_score(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2 or any(not item.isdigit() for item in parts):
        raise ValueError("evaluation actual_score must use non-negative home:away goals")
    home_goals, away_goals = (int(item) for item in parts)
    if value != f"{home_goals}:{away_goals}":
        raise ValueError("evaluation actual_score is not canonical")
    return home_goals, away_goals


def _is_digest_ref(value: str, prefix: str) -> bool:
    digest = value.removeprefix(prefix)
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
