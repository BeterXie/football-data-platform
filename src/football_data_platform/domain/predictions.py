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

PREDICTION_SCHEMA_VERSION = 3
SCORE_GRID_COMPOSITION_SCHEMA_VERSION = 1
SCORE_GRID_COMPOSITION_ARTIFACT_TYPE = "score-grid-composition"
SCORE_GRID_COMPOSITION_CODE_VERSION = "football-data-platform/0.1.0"
LEGACY_COMPOSITION_VERSION = "legacy-inline/1"


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
class PredictionContribution:
    """The immutable contribution detail retained by a score prediction.

    ``ExpectedGoalsContribution`` lives in the feature layer.  Predictions
    intentionally carry this small domain copy so the persisted prediction
    does not depend on importing or reconstructing feature objects later.
    """

    contribution_key: str
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    source_ref: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.contribution_key, "contribution_key")
        _require_text(self.source_ref, "source_ref")
        _require_text(self.version, "version")
        for name, value in (
            ("lambda_home_multiplier", self.lambda_home_multiplier),
            ("lambda_away_multiplier", self.lambda_away_multiplier),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


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
    baseline_lambda_home: float | None = None
    baseline_lambda_away: float | None = None
    contribution_keys: tuple[str, ...] = ()
    contribution_multipliers: tuple[PredictionContribution, ...] = ()
    composition_version: str = LEGACY_COMPOSITION_VERSION
    calibration_versions: tuple[str, ...] = ()
    composition_artifact_ref: str | None = None

    @property
    def contributions(self) -> tuple[PredictionContribution, ...]:
        """Compatibility alias for callers that use the feature terminology."""

        return self.contribution_multipliers

    @property
    def composition_ref(self) -> str | None:
        """Short alias used by report and registry consumers."""

        return self.composition_artifact_ref

    @property
    def grid_artifact_ref(self) -> str | None:
        """The composition artifact contains the canonical Dixon-Coles grid."""

        return self.composition_artifact_ref


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
    expected_goals: Any | None = None,
    composition: Any | None = None,
    composition_artifact_ref: str | None = None,
    calibration_versions: tuple[str, ...] | None = None,
) -> ScorePrediction:
    """Build the one score distribution consumed by every football market view.

    ``expected_goals`` (or its ``composition`` alias) is the preferred input
    because it carries the baseline and auditable feature contributions.  A
    legacy low-level caller may omit it; that path is explicitly represented
    as ``legacy-inline/1`` and still receives a content-addressed composition
    reference so persistence cannot silently lose provenance.
    """

    verify_snapshot(snapshot, source_validator=snapshot_validator)
    require_utc(generated_at, "generated_at")
    _require_text(model_version, "model_version")
    if snapshot.quality_status != "ready":
        raise ValueError("formal score predictions require a ready snapshot")
    if generated_at < snapshot.as_of:
        raise ValueError("prediction generated_at cannot precede snapshot as_of")
    if expected_goals is not None and composition is not None:
        raise ValueError("pass only one of expected_goals or composition")
    expected_goals = expected_goals if expected_goals is not None else composition
    (
        baseline_lambda_home,
        baseline_lambda_away,
        contribution_keys,
        contribution_multipliers,
        composition_version,
        resolved_calibration_versions,
        composition_input_refs,
    ) = _prediction_composition_metadata(
        expected_goals,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        calibration_versions=calibration_versions,
    )
    _verify_prediction_contribution_sources(snapshot, contribution_multipliers)
    normalized_input_refs = tuple(
        sorted({*input_refs, snapshot.id.value, model_run_id.value, *composition_input_refs})
    )
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
    composition_payload = _score_grid_composition_payload_from_values(
        match_id=snapshot.match_id,
        snapshot_id=snapshot.id,
        model_run_id=model_run_id,
        model_version=model_version,
        generated_at=generated_at,
        snapshot_as_of=snapshot.as_of,
        baseline_lambda_home=baseline_lambda_home,
        baseline_lambda_away=baseline_lambda_away,
        contribution_keys=contribution_keys,
        contribution_multipliers=contribution_multipliers,
        composition_version=composition_version,
        calibration_versions=resolved_calibration_versions,
        lambda_home=grid.lambda_home,
        lambda_away=grid.lambda_away,
        rho=grid.rho,
        max_goals=grid.max_goals,
        normalization_residual=grid.normalization_residual,
        score_cells=cells,
        input_refs=normalized_input_refs,
    )
    expected_composition_ref = score_grid_composition_artifact_id(composition_payload)
    if (
        composition_artifact_ref is not None
        and composition_artifact_ref != expected_composition_ref
    ):
        raise ValueError("composition_artifact_ref does not match its content")
    composition_artifact_ref = expected_composition_ref
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
        "input_refs": list(normalized_input_refs),
        "baseline_lambda_home": baseline_lambda_home,
        "baseline_lambda_away": baseline_lambda_away,
        "contribution_keys": list(contribution_keys),
        "contribution_multipliers": [
            _prediction_contribution_payload(item) for item in contribution_multipliers
        ],
        "composition_version": composition_version,
        "calibration_versions": list(resolved_calibration_versions),
        "composition_artifact_ref": composition_artifact_ref,
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
        input_refs=normalized_input_refs,
        baseline_lambda_home=baseline_lambda_home,
        baseline_lambda_away=baseline_lambda_away,
        contribution_keys=contribution_keys,
        contribution_multipliers=contribution_multipliers,
        composition_version=composition_version,
        calibration_versions=resolved_calibration_versions,
        composition_artifact_ref=composition_artifact_ref,
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
    _verify_prediction_composition_metadata(prediction)

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
    composition_payload = score_grid_composition_payload(prediction)
    expected_composition_ref = score_grid_composition_artifact_id(composition_payload)
    if prediction.composition_artifact_ref != expected_composition_ref:
        raise ValueError("prediction composition artifact reference does not match its content")
    if model_run_validator is not None:
        _verify_prediction_model_run(prediction, model_run_validator)


def verify_prediction_snapshot(
    prediction: ScorePrediction,
    *,
    snapshot: PreMatchSnapshot,
    snapshot_validator: SnapshotSourceValidator,
    model_run_validator: ModelRunValidator | None = None,
) -> None:
    """Verify a formal prediction against its validated prematch snapshot."""

    verify_snapshot(snapshot, source_validator=snapshot_validator)
    verify_score_prediction(prediction, model_run_validator=model_run_validator)
    if prediction.snapshot_id != snapshot.id:
        raise ValueError("prediction snapshot_id does not match the validated snapshot")
    if prediction.match_id != snapshot.match_id:
        raise ValueError("prediction match_id does not match the validated snapshot")
    if prediction.snapshot_as_of != snapshot.as_of:
        raise ValueError("prediction snapshot_as_of does not match the validated snapshot")
    if prediction.capture_mode != snapshot.capture_mode:
        raise ValueError("prediction capture_mode does not match the validated snapshot")
    if prediction.snapshot_quality_status != snapshot.quality_status:
        raise ValueError("prediction quality does not match the validated snapshot")
    _verify_prediction_contribution_sources(snapshot, prediction.contribution_multipliers)


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
        "baseline_lambda_home": prediction.baseline_lambda_home,
        "baseline_lambda_away": prediction.baseline_lambda_away,
        "contribution_keys": list(prediction.contribution_keys),
        "contribution_multipliers": [
            _prediction_contribution_payload(item) for item in prediction.contribution_multipliers
        ],
        "composition_version": prediction.composition_version,
        "calibration_versions": list(prediction.calibration_versions),
        "composition_artifact_ref": prediction.composition_artifact_ref,
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
        contribution_payloads = payload.get("contribution_multipliers", ())
        if not isinstance(contribution_payloads, (list, tuple)):
            raise ValueError("contribution_multipliers must be a sequence")
        from football_data_platform.features.contributions import (
            ExpectedGoals,
            ExpectedGoalsContribution,
        )

        contributions = tuple(
            ExpectedGoalsContribution(
                contribution_key=str(item["contribution_key"]),
                lambda_home_multiplier=float(item["lambda_home_multiplier"]),
                lambda_away_multiplier=float(item["lambda_away_multiplier"]),
                source_ref=str(item["source_ref"]),
                version=str(item["version"]),
            )
            for item in contribution_payloads
        )
        expected_goals = ExpectedGoals(
            lambda_home=float(payload["lambda_home"]),
            lambda_away=float(payload["lambda_away"]),
            contribution_keys=tuple(str(item) for item in payload["contribution_keys"]),
            input_refs=tuple(
                str(item) for contribution in contributions for item in (contribution.source_ref,)
            ),
            baseline_lambda_home=float(payload["baseline_lambda_home"]),
            baseline_lambda_away=float(payload["baseline_lambda_away"]),
            composition_version=str(payload["composition_version"]),
            contributions=contributions,
        )
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
            expected_goals=expected_goals,
            composition_artifact_ref=str(payload["composition_artifact_ref"]),
            calibration_versions=tuple(str(item) for item in payload["calibration_versions"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prediction schema: {error}") from error
    if prediction_payload(prediction) != payload:
        raise ValueError("prediction payload does not match its canonical score grid")
    return prediction


def _verify_prediction_contribution_sources(
    snapshot: PreMatchSnapshot,
    contributions: tuple[PredictionContribution, ...],
) -> None:
    feature_source_refs = frozenset(feature.source_ref for feature in snapshot.features)
    if any(item.source_ref not in feature_source_refs for item in contributions):
        raise ValueError(
            "expected-goals contribution source refs must cite verified snapshot feature sources"
        )


def _prediction_composition_metadata(
    expected_goals: Any | None,
    *,
    lambda_home: float,
    lambda_away: float,
    calibration_versions: tuple[str, ...] | None,
) -> tuple[
    float,
    float,
    tuple[str, ...],
    tuple[PredictionContribution, ...],
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    if expected_goals is None:
        if (
            not math.isfinite(lambda_home)
            or not math.isfinite(lambda_away)
            or lambda_home <= 0
            or lambda_away <= 0
        ):
            raise ValueError("prediction lambdas must be finite and positive")
        return (
            float(lambda_home),
            float(lambda_away),
            (),
            (),
            LEGACY_COMPOSITION_VERSION,
            tuple(sorted(set(calibration_versions or ()))),
            (),
        )

    try:
        baseline_home = float(expected_goals.baseline_lambda_home)
        baseline_away = float(expected_goals.baseline_lambda_away)
        raw_keys = tuple(str(item) for item in expected_goals.contribution_keys)
        raw_contributions = tuple(expected_goals.contributions)
        composition_version = str(expected_goals.composition_version)
        raw_input_refs = tuple(str(item) for item in expected_goals.input_refs)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("expected_goals does not satisfy the composition contract") from error
    if (
        not math.isfinite(baseline_home)
        or not math.isfinite(baseline_away)
        or baseline_home <= 0
        or baseline_away <= 0
    ):
        raise ValueError("expected-goals baseline lambdas must be finite and positive")
    _require_text(composition_version, "composition_version")
    contributions = tuple(_prediction_contribution(item) for item in raw_contributions)
    keys = tuple(item.contribution_key for item in contributions)
    if raw_keys != keys:
        raise ValueError("expected-goals contribution keys do not match contribution details")
    if keys != tuple(sorted(set(keys))):
        raise ValueError("expected-goals contribution keys must be unique and sorted")
    source_refs = tuple(item.source_ref for item in contributions)
    input_refs = tuple(sorted(set((*raw_input_refs, *source_refs))))
    if any(not item for item in input_refs):
        raise ValueError("expected-goals input refs must be non-empty")
    composed_home = baseline_home * math.prod(item.lambda_home_multiplier for item in contributions)
    composed_away = baseline_away * math.prod(item.lambda_away_multiplier for item in contributions)
    if not math.isclose(composed_home, lambda_home, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("expected-goals home lambda does not match prediction")
    if not math.isclose(composed_away, lambda_away, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("expected-goals away lambda does not match prediction")
    explicit_calibrations = (
        calibration_versions
        if calibration_versions is not None
        else tuple(getattr(expected_goals, "calibration_versions", ()))
    )
    resolved_calibration_versions = tuple(sorted(set(explicit_calibrations)))
    return (
        baseline_home,
        baseline_away,
        keys,
        contributions,
        composition_version,
        resolved_calibration_versions,
        input_refs,
    )


def _prediction_contribution(value: Any) -> PredictionContribution:
    try:
        return PredictionContribution(
            contribution_key=str(value.contribution_key),
            lambda_home_multiplier=float(value.lambda_home_multiplier),
            lambda_away_multiplier=float(value.lambda_away_multiplier),
            source_ref=str(value.source_ref),
            version=str(value.version),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid expected-goals contribution") from error


def _verify_prediction_composition_metadata(prediction: ScorePrediction) -> None:
    for name, value in (
        ("baseline_lambda_home", prediction.baseline_lambda_home),
        ("baseline_lambda_away", prediction.baseline_lambda_away),
    ):
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"prediction {name} must be finite and positive")
    _require_text(prediction.composition_version, "composition_version")
    if prediction.contribution_keys != tuple(sorted(set(prediction.contribution_keys))):
        raise ValueError("prediction contribution keys must be unique and sorted")
    if any(not isinstance(item, str) or not item for item in prediction.contribution_keys):
        raise ValueError("prediction contribution keys must be non-empty text")
    if any(
        not isinstance(item, PredictionContribution) for item in prediction.contribution_multipliers
    ):
        raise TypeError("prediction contribution multipliers must be PredictionContribution values")
    contribution_keys = tuple(item.contribution_key for item in prediction.contribution_multipliers)
    if contribution_keys != prediction.contribution_keys:
        raise ValueError("prediction contribution keys do not match multiplier details")
    if prediction.calibration_versions != tuple(sorted(set(prediction.calibration_versions))):
        raise ValueError("prediction calibration_versions must be unique and sorted")
    if any(not isinstance(item, str) or not item for item in prediction.calibration_versions):
        raise ValueError("prediction calibration_versions must be non-empty text")
    for item in prediction.contribution_multipliers:
        if item.source_ref not in prediction.input_refs:
            raise ValueError("prediction contribution source is missing from input_refs")
    calculated_home = prediction.baseline_lambda_home * math.prod(
        item.lambda_home_multiplier for item in prediction.contribution_multipliers
    )
    calculated_away = prediction.baseline_lambda_away * math.prod(
        item.lambda_away_multiplier for item in prediction.contribution_multipliers
    )
    if not math.isclose(calculated_home, prediction.lambda_home, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("prediction home lambda is not reproducible from its composition")
    if not math.isclose(calculated_away, prediction.lambda_away, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("prediction away lambda is not reproducible from its composition")
    if prediction.composition_artifact_ref is None:
        raise ValueError("prediction requires a composition artifact reference")
    _require_content_ref(prediction.composition_artifact_ref, "composition_artifact_ref")


def score_grid_composition_payload(prediction: ScorePrediction) -> dict[str, Any]:
    """Return the canonical derived payload referenced by a prediction."""

    return _score_grid_composition_payload_from_values(
        match_id=prediction.match_id,
        snapshot_id=prediction.snapshot_id,
        model_run_id=prediction.model_run_id,
        model_version=prediction.model_version,
        generated_at=prediction.generated_at,
        snapshot_as_of=prediction.snapshot_as_of,
        baseline_lambda_home=prediction.baseline_lambda_home,
        baseline_lambda_away=prediction.baseline_lambda_away,
        contribution_keys=prediction.contribution_keys,
        contribution_multipliers=prediction.contribution_multipliers,
        composition_version=prediction.composition_version,
        calibration_versions=prediction.calibration_versions,
        lambda_home=prediction.lambda_home,
        lambda_away=prediction.lambda_away,
        rho=prediction.rho,
        max_goals=prediction.max_goals,
        normalization_residual=prediction.normalization_residual,
        score_cells=prediction.score_cells,
        input_refs=prediction.input_refs,
    )


def _score_grid_composition_payload_from_values(
    *,
    match_id: MatchId,
    snapshot_id: SnapshotId,
    model_run_id: ModelRunId,
    model_version: str,
    generated_at: datetime,
    snapshot_as_of: datetime,
    baseline_lambda_home: float | None,
    baseline_lambda_away: float | None,
    contribution_keys: tuple[str, ...],
    contribution_multipliers: tuple[PredictionContribution, ...],
    composition_version: str,
    calibration_versions: tuple[str, ...],
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int,
    normalization_residual: float,
    score_cells: tuple[ScoreCell, ...],
    input_refs: tuple[str, ...],
) -> dict[str, Any]:
    contributions = [_prediction_contribution_payload(item) for item in contribution_multipliers]
    grid = {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "rho": rho,
        "max_goals": max_goals,
        "normalization_residual": normalization_residual,
        "score_cells": [_score_cell_payload(item) for item in score_cells],
    }
    return {
        "schema_version": SCORE_GRID_COMPOSITION_SCHEMA_VERSION,
        "artifact_type": SCORE_GRID_COMPOSITION_ARTIFACT_TYPE,
        "match_id": match_id.value,
        "snapshot_id": snapshot_id.value,
        "model_run_id": model_run_id.value,
        "model_version": model_version,
        "generated_at": _timestamp(generated_at),
        "snapshot_as_of": _timestamp(snapshot_as_of),
        "baseline_lambda_home": baseline_lambda_home,
        "baseline_lambda_away": baseline_lambda_away,
        "contribution_keys": list(contribution_keys),
        "contribution_multipliers": contributions,
        "composition_version": composition_version,
        "calibration_versions": list(calibration_versions),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "rho": rho,
        "max_goals": max_goals,
        "normalization_residual": normalization_residual,
        "score_cells": grid["score_cells"],
        "input_refs": list(input_refs),
        "grid": grid,
    }


def score_grid_composition_output_ref(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"score-grid-composition:{digest}"


def score_grid_composition_artifact_id(payload: dict[str, Any]) -> str:
    """Return the immutable content reference for a composition payload.

    The reference is deliberately derived from the payload only.  Runtime
    metadata belongs to the derived manifest and must not change the logical
    composition identity or force the prediction builder to duplicate storage
    envelope hashing rules.
    """

    return score_grid_composition_output_ref(payload)


def _prediction_contribution_payload(item: PredictionContribution) -> dict[str, Any]:
    return {
        "contribution_key": item.contribution_key,
        "lambda_home_multiplier": item.lambda_home_multiplier,
        "lambda_away_multiplier": item.lambda_away_multiplier,
        "source_ref": item.source_ref,
        "version": item.version,
    }


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


def _require_content_ref(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith("score-grid-composition:"):
        raise ValueError(f"{field_name} must be a score-grid-composition reference")
    digest = value.removeprefix("score-grid-composition:")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field_name} must contain a 64-character content digest")
