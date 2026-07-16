"""Versioned football prediction and independent market contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from football_data_platform.domain.ids import (
    MarketSnapshotId,
    MatchId,
    ModelRunId,
    PredictionId,
    RawAssetId,
    SnapshotId,
)
from football_data_platform.domain.models import RawAsset, require_utc
from football_data_platform.domain.snapshots import (
    CaptureMode,
    PreMatchSnapshot,
    SnapshotSourceValidator,
    verify_snapshot,
)
from football_data_platform.domain.training import (
    ModelRunArtifact,
    ModelRunStatus,
    verify_model_run_artifact,
)
from football_data_platform.models.score_grid import DixonColesGrid

PREDICTION_SCHEMA_VERSION = 2


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
    snapshot_as_of: datetime
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


class MarketDataKind(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class MarketStatus(StrEnum):
    OPEN = "open"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class MarketSourceValidator(Protocol):
    def load(self, asset_id: RawAssetId) -> RawAsset: ...

    def verify(self, asset: RawAsset | RawAssetId) -> None: ...


class ModelRunValidator(Protocol):
    def load_model_run(self, model_run_id: str) -> ModelRunArtifact: ...


class MatchResultValidator(Protocol):
    def verify_match_result(self, result: MatchResult90) -> None: ...


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
    status: MarketStatus
    data_kind: MarketDataKind
    observed_at: datetime
    quotes: tuple[MarketQuote, ...]
    raw_asset_ref: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported market snapshot schema_version {self.schema_version!r}")
        _require_text(self.source, "source")
        _require_text(self.market_type, "market_type")
        if not isinstance(self.status, MarketStatus):
            raise TypeError("status must be a MarketStatus")
        if not isinstance(self.data_kind, MarketDataKind):
            raise TypeError("data_kind must be a MarketDataKind")
        RawAssetId(self.raw_asset_ref)
        require_utc(self.observed_at, "observed_at")
        if not self.quotes or any(not isinstance(quote, MarketQuote) for quote in self.quotes):
            raise ValueError("market snapshot requires market quotes")
        outcomes = [quote.outcome for quote in self.quotes]
        if len(outcomes) != len(set(outcomes)):
            raise ValueError("market snapshot contains duplicate outcomes")


def build_market_snapshot(
    *,
    match_id: MatchId,
    market_type: str,
    status: MarketStatus,
    data_kind: MarketDataKind,
    quotes: tuple[MarketQuote, ...],
    raw_asset: RawAsset,
) -> MarketSnapshot:
    """Build an immutable market document whose metadata cites one raw observation."""

    if not isinstance(raw_asset, RawAsset):
        raise TypeError("raw_asset must be a RawAsset")
    fields = {
        "match_id": match_id,
        "source": raw_asset.source,
        "market_type": market_type,
        "status": status,
        "data_kind": data_kind,
        "observed_at": raw_asset.observed_at,
        "quotes": quotes,
        "raw_asset_ref": raw_asset.id.value,
    }
    identity = _market_snapshot_identity(**fields)
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return MarketSnapshot(
        id=MarketSnapshotId(f"market-snapshot:{digest}"),
        schema_version=1,
        **fields,
    )


def verify_market_snapshot(
    snapshot: MarketSnapshot,
    *,
    source_validator: MarketSourceValidator,
) -> None:
    """Verify immutable raw lineage and the content-derived market identity."""

    raw_asset_id = RawAssetId(snapshot.raw_asset_ref)
    try:
        source_validator.verify(raw_asset_id)
        raw_asset = source_validator.load(raw_asset_id)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("market snapshot raw evidence is unavailable or invalid") from error
    if (
        raw_asset.id != raw_asset_id
        or raw_asset.source != snapshot.source
        or raw_asset.observed_at != snapshot.observed_at
    ):
        raise ValueError("market snapshot source metadata does not match raw evidence")
    identity = _market_snapshot_identity(
        match_id=snapshot.match_id,
        source=snapshot.source,
        market_type=snapshot.market_type,
        status=snapshot.status,
        data_kind=snapshot.data_kind,
        observed_at=snapshot.observed_at,
        quotes=snapshot.quotes,
        raw_asset_ref=snapshot.raw_asset_ref,
    )
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if snapshot.id != MarketSnapshotId(f"market-snapshot:{digest}"):
        raise ValueError("market snapshot identity does not match its canonical content")


def build_score_prediction(
    *,
    snapshot: PreMatchSnapshot,
    snapshot_validator: SnapshotSourceValidator,
    model_run_id: ModelRunId,
    model_version: str,
    generated_at: datetime,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int,
    input_refs: tuple[str, ...],
    model_run_validator: ModelRunValidator | None = None,
) -> ScorePrediction:
    """Build the one score distribution consumed by every football market view."""

    verify_snapshot(snapshot, source_validator=snapshot_validator)
    require_utc(generated_at, "generated_at")
    _require_text(model_version, "model_version")
    if snapshot.quality_status != "ready":
        raise ValueError("formal score predictions require a ready snapshot")
    if generated_at < snapshot.as_of:
        raise ValueError("prediction generated_at cannot precede snapshot as_of")
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
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "match_id": snapshot.match_id.value,
        "snapshot_id": snapshot.id.value,
        "capture_mode": snapshot.capture_mode.value,
        "snapshot_quality_status": snapshot.quality_status,
        "model_run_id": model_run_id.value,
        "model_version": model_version,
        "generated_at": _timestamp(generated_at),
        "snapshot_as_of": _timestamp(snapshot.as_of),
        "lambda_home": grid.lambda_home,
        "lambda_away": grid.lambda_away,
        "rho": grid.rho,
        "max_goals": grid.max_goals,
        "score_cells": [_score_cell_payload(cell) for cell in cells],
        "markets": [_market_payload(market) for market in markets],
        "normalization_residual": grid.normalization_residual,
        "input_refs": sorted({*input_refs, snapshot.id.value}),
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    prediction = ScorePrediction(
        id=PredictionId(f"prediction:{digest}"),
        schema_version=PREDICTION_SCHEMA_VERSION,
        match_id=snapshot.match_id,
        snapshot_id=snapshot.id,
        capture_mode=snapshot.capture_mode,
        snapshot_quality_status=snapshot.quality_status,
        model_run_id=model_run_id,
        model_version=model_version,
        generated_at=generated_at,
        snapshot_as_of=snapshot.as_of,
        lambda_home=grid.lambda_home,
        lambda_away=grid.lambda_away,
        rho=grid.rho,
        max_goals=grid.max_goals,
        score_cells=cells,
        markets=markets,
        normalization_residual=grid.normalization_residual,
        input_refs=tuple(sorted({*input_refs, snapshot.id.value})),
    )
    verify_score_prediction(prediction, model_run_validator=model_run_validator)
    return prediction


def verify_score_prediction(
    prediction: ScorePrediction,
    *,
    model_run_validator: ModelRunValidator | None = None,
) -> None:
    """Rebuild the canonical grid and verify the prediction content identity."""

    if not isinstance(prediction, ScorePrediction):
        raise TypeError("prediction must be a ScorePrediction")
    if prediction.schema_version != PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction schema_version {prediction.schema_version!r}")
    if not isinstance(prediction.id, PredictionId):
        raise TypeError("prediction id must be a PredictionId")
    if not isinstance(prediction.match_id, MatchId):
        raise TypeError("prediction match_id must be a MatchId")
    if not isinstance(prediction.snapshot_id, SnapshotId):
        raise TypeError("prediction snapshot_id must be a SnapshotId")
    if not isinstance(prediction.model_run_id, ModelRunId):
        raise TypeError("prediction model_run_id must be a ModelRunId")
    if not isinstance(prediction.capture_mode, CaptureMode):
        raise TypeError("prediction capture_mode must be a CaptureMode")
    require_utc(prediction.generated_at, "prediction generated_at")
    require_utc(prediction.snapshot_as_of, "prediction snapshot_as_of")
    if prediction.generated_at < prediction.snapshot_as_of:
        raise ValueError("prediction generated_at cannot precede snapshot_as_of")
    if prediction.snapshot_quality_status != "ready":
        raise ValueError("formal score prediction must cite a ready snapshot")
    _require_text(prediction.model_version, "model_version")
    if (
        prediction.input_refs != tuple(sorted(set(prediction.input_refs)))
        or prediction.snapshot_id.value not in prediction.input_refs
        or any(not isinstance(item, str) or not item for item in prediction.input_refs)
    ):
        raise ValueError("prediction input_refs must be unique, sorted, and cite the snapshot")

    grid = DixonColesGrid(
        prediction.lambda_home,
        prediction.lambda_away,
        rho=prediction.rho,
        max_goals=prediction.max_goals,
    )
    expected_cells = tuple(ScoreCell(**cell) for cell in grid.score_cells())
    expected_markets = (
        _market_view("result_90", None, grid.result_probabilities()),
        _market_view("home_handicap_3way", "-1", grid.handicap_probabilities(-1)),
        _market_view("total_goals", "0-6,7+", grid.total_goals_probabilities()),
    )
    if prediction.score_cells != expected_cells:
        raise ValueError("prediction score cells do not match its Dixon-Coles grid")
    if prediction.markets != expected_markets:
        raise ValueError("prediction market views do not match its Dixon-Coles grid")
    if prediction.normalization_residual != grid.normalization_residual:
        raise ValueError("prediction normalization residual does not match its grid")

    payload = prediction_payload(prediction)
    payload.pop("id")
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if prediction.id != PredictionId(f"prediction:{digest}"):
        raise ValueError("prediction identity does not match its canonical content")
    if model_run_validator is not None:
        _verify_prediction_model_run(prediction, model_run_validator)


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
        "snapshot_as_of": _timestamp(prediction.snapshot_as_of),
        "lambda_home": prediction.lambda_home,
        "lambda_away": prediction.lambda_away,
        "rho": prediction.rho,
        "max_goals": prediction.max_goals,
        "score_cells": [_score_cell_payload(cell) for cell in prediction.score_cells],
        "markets": [_market_payload(market) for market in prediction.markets],
        "normalization_residual": prediction.normalization_residual,
        "input_refs": list(prediction.input_refs),
    }


def parse_prediction_payload(
    payload: dict[str, Any],
    *,
    snapshot: PreMatchSnapshot,
    snapshot_validator: SnapshotSourceValidator,
    model_run_validator: ModelRunValidator | None = None,
) -> ScorePrediction:
    """Validate one prediction document; mixed or unknown schemas fail explicitly."""

    if payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction schema_version {payload.get('schema_version')!r}")
    try:
        prediction = build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=snapshot_validator,
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
            model_run_validator=model_run_validator,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prediction schema: {error}") from error
    if prediction_payload(prediction) != payload:
        raise ValueError("prediction payload does not match its canonical score grid")
    return prediction


def _verify_prediction_model_run(
    prediction: ScorePrediction,
    validator: ModelRunValidator,
) -> None:
    try:
        artifact = validator.load_model_run(prediction.model_run_id.value)
        verify_model_run_artifact(artifact)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("prediction model run is unavailable or invalid") from error
    if artifact.model_run_id != prediction.model_run_id.value:
        raise ValueError("prediction model run validator returned a different model run")
    if artifact.status is not ModelRunStatus.SUCCEEDED:
        raise ValueError("formal score prediction requires a successful model run")
    if artifact.task != "score-model":
        raise ValueError("prediction model run must target the score-model task")
    if artifact.model_version != prediction.model_version:
        raise ValueError("prediction model_version does not match its model run")
    if artifact.ended_at > prediction.generated_at:
        raise ValueError("prediction cannot precede its model run completion")


def verify_match_result_source(
    result: MatchResult90,
    validator: MatchResultValidator,
) -> None:
    """Verify one result against an immutable canonical source when configured."""

    try:
        validator.verify_match_result(result)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("result source evidence is unavailable or invalid") from error


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


def _market_snapshot_identity(
    *,
    match_id: MatchId,
    source: str,
    market_type: str,
    status: MarketStatus,
    data_kind: MarketDataKind,
    observed_at: datetime,
    quotes: tuple[MarketQuote, ...],
    raw_asset_ref: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "match_id": match_id.value,
        "source": source,
        "market_type": market_type,
        "status": status.value,
        "data_kind": data_kind.value,
        "observed_at": _timestamp(observed_at),
        "quotes": [
            {"outcome": quote.outcome, "decimal_odds": quote.decimal_odds} for quote in quotes
        ],
        "raw_asset_ref": raw_asset_ref,
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
