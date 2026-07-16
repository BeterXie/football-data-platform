"""Versioned football prediction and independent market contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.ids import (
    MarketSnapshotId,
    MatchId,
    ModelRunId,
    PredictionId,
    SnapshotId,
)
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.models.score_grid import DixonColesGrid


@dataclass(frozen=True, slots=True)
class OutcomeProbability:
    outcome: str
    probability: float

    def __post_init__(self) -> None:
        _require_text(self.outcome, "outcome")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class MarketView:
    market_type: str
    parameter: str | None
    outcomes: tuple[OutcomeProbability, ...]


@dataclass(frozen=True, slots=True)
class ScoreCell:
    home_goals: int
    away_goals: int
    probability: float


@dataclass(frozen=True, slots=True)
class ScorePrediction:
    id: PredictionId
    schema_version: int
    match_id: MatchId
    snapshot_id: SnapshotId
    capture_mode: CaptureMode
    snapshot_quality_status: str
    model_run_id: ModelRunId
    model_version: str
    generated_at: datetime
    lambda_home: float
    lambda_away: float
    rho: float
    max_goals: int
    score_cells: tuple[ScoreCell, ...]
    markets: tuple[MarketView, ...]
    normalization_residual: float
    input_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchResult90:
    match_id: MatchId
    home_goals: int
    away_goals: int
    known_at: datetime
    source_ref: str

    def __post_init__(self) -> None:
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("goals must not be negative")
        require_utc(self.known_at, "known_at")
        _require_text(self.source_ref, "source_ref")


@dataclass(frozen=True, slots=True)
class MarketQuote:
    outcome: str
    decimal_odds: float

    def __post_init__(self) -> None:
        _require_text(self.outcome, "outcome")
        if not math.isfinite(self.decimal_odds) or self.decimal_odds <= 1:
            raise ValueError("decimal_odds must be finite and greater than 1")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    id: MarketSnapshotId
    schema_version: int
    match_id: MatchId
    source: str
    market_type: str
    observed_at: datetime
    quotes: tuple[MarketQuote, ...]
    raw_asset_ref: str

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        _require_text(self.market_type, "market_type")
        _require_text(self.raw_asset_ref, "raw_asset_ref")
        require_utc(self.observed_at, "observed_at")
        outcomes = [quote.outcome for quote in self.quotes]
        if len(outcomes) != len(set(outcomes)):
            raise ValueError("market snapshot contains duplicate outcomes")


def build_score_prediction(
    *,
    match_id: MatchId,
    snapshot_id: SnapshotId,
    capture_mode: CaptureMode,
    snapshot_quality_status: str,
    model_run_id: ModelRunId,
    model_version: str,
    generated_at: datetime,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int,
    input_refs: tuple[str, ...],
) -> ScorePrediction:
    """Build the one score distribution consumed by every football market view."""

    require_utc(generated_at, "generated_at")
    _require_text(model_version, "model_version")
    if snapshot_quality_status != "ready":
        raise ValueError("formal score predictions require a ready snapshot")
    grid = DixonColesGrid(lambda_home, lambda_away, rho=rho, max_goals=max_goals)
    result = grid.result_probabilities()
    handicap = grid.handicap_probabilities(-1)
    totals = grid.total_goals_probabilities()
    markets = (
        _market_view("result_90", None, result),
        _market_view("home_handicap_3way", "-1", handicap),
        _market_view("total_goals", "0-6,7+", totals),
    )
    cells = tuple(ScoreCell(**cell) for cell in grid.score_cells())
    payload = {
        "schema_version": 1,
        "match_id": match_id.value,
        "snapshot_id": snapshot_id.value,
        "capture_mode": capture_mode.value,
        "snapshot_quality_status": snapshot_quality_status,
        "model_run_id": model_run_id.value,
        "model_version": model_version,
        "generated_at": _timestamp(generated_at),
        "lambda_home": grid.lambda_home,
        "lambda_away": grid.lambda_away,
        "rho": grid.rho,
        "max_goals": grid.max_goals,
        "score_cells": [_score_cell_payload(cell) for cell in cells],
        "markets": [_market_payload(market) for market in markets],
        "normalization_residual": grid.normalization_residual,
        "input_refs": sorted(set(input_refs)),
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return ScorePrediction(
        id=PredictionId(f"prediction:{digest}"),
        schema_version=1,
        match_id=match_id,
        snapshot_id=snapshot_id,
        capture_mode=capture_mode,
        snapshot_quality_status=snapshot_quality_status,
        model_run_id=model_run_id,
        model_version=model_version,
        generated_at=generated_at,
        lambda_home=grid.lambda_home,
        lambda_away=grid.lambda_away,
        rho=grid.rho,
        max_goals=grid.max_goals,
        score_cells=cells,
        markets=markets,
        normalization_residual=grid.normalization_residual,
        input_refs=tuple(sorted(set(input_refs))),
    )


def prediction_payload(prediction: ScorePrediction) -> dict[str, Any]:
    return {
        "id": prediction.id.value,
        "schema_version": prediction.schema_version,
        "match_id": prediction.match_id.value,
        "snapshot_id": prediction.snapshot_id.value,
        "capture_mode": prediction.capture_mode.value,
        "snapshot_quality_status": prediction.snapshot_quality_status,
        "model_run_id": prediction.model_run_id.value,
        "model_version": prediction.model_version,
        "generated_at": _timestamp(prediction.generated_at),
        "lambda_home": prediction.lambda_home,
        "lambda_away": prediction.lambda_away,
        "rho": prediction.rho,
        "max_goals": prediction.max_goals,
        "score_cells": [_score_cell_payload(cell) for cell in prediction.score_cells],
        "markets": [_market_payload(market) for market in prediction.markets],
        "normalization_residual": prediction.normalization_residual,
        "input_refs": list(prediction.input_refs),
    }


def parse_prediction_payload(payload: dict[str, Any]) -> ScorePrediction:
    """Validate one prediction document; mixed or unknown schemas fail explicitly."""

    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported prediction schema_version {payload.get('schema_version')!r}")
    try:
        prediction = build_score_prediction(
            match_id=MatchId(str(payload["match_id"])),
            snapshot_id=SnapshotId(str(payload["snapshot_id"])),
            capture_mode=CaptureMode(str(payload["capture_mode"])),
            snapshot_quality_status=str(payload["snapshot_quality_status"]),
            model_run_id=ModelRunId(str(payload["model_run_id"])),
            model_version=str(payload["model_version"]),
            generated_at=datetime.fromisoformat(
                str(payload["generated_at"]).replace("Z", "+00:00")
            ),
            lambda_home=float(payload["lambda_home"]),
            lambda_away=float(payload["lambda_away"]),
            rho=float(payload["rho"]),
            max_goals=int(payload["max_goals"]),
            input_refs=tuple(str(item) for item in payload["input_refs"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prediction schema: {error}") from error
    if prediction_payload(prediction) != payload:
        raise ValueError("prediction payload does not match its canonical score grid")
    return prediction


def result_probabilities(prediction: ScorePrediction) -> dict[str, float]:
    for market in prediction.markets:
        if market.market_type == "result_90":
            return {item.outcome: item.probability for item in market.outcomes}
    raise ValueError("prediction has no result_90 market view")


def exact_score_probability(prediction: ScorePrediction, home_goals: int, away_goals: int) -> float:
    for cell in prediction.score_cells:
        if cell.home_goals == home_goals and cell.away_goals == away_goals:
            return cell.probability
    return 0.0


def _market_view(
    market_type: str,
    parameter: str | None,
    probabilities: dict[str, float],
) -> MarketView:
    return MarketView(
        market_type=market_type,
        parameter=parameter,
        outcomes=tuple(
            OutcomeProbability(outcome, probability)
            for outcome, probability in probabilities.items()
        ),
    )


def _market_payload(market: MarketView) -> dict[str, Any]:
    return {
        "market_type": market.market_type,
        "parameter": market.parameter,
        "outcomes": [
            {"outcome": item.outcome, "probability": item.probability} for item in market.outcomes
        ],
    }


def _score_cell_payload(cell: ScoreCell) -> dict[str, int | float]:
    return {
        "home_goals": cell.home_goals,
        "away_goals": cell.away_goals,
        "probability": cell.probability,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
