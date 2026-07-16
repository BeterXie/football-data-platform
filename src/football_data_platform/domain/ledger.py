"""Immutable paper-betting contracts and deterministic settlement rules.

The ledger is deliberately downstream of the football model.  A paper entry
stores references to a validated prediction and market snapshot; it never
feeds market odds back into model inputs and it never accepts a caller-provided
settlement amount.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from football_data_platform.domain.ids import MatchId, ModelRunId, PredictionId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketSnapshot,
    MarketSourceValidator,
    MarketStatus,
    MatchResult90,
    MatchResultValidator,
    ModelRunValidator,
    ScorePrediction,
    verify_market_snapshot,
    verify_match_result_source,
    verify_score_prediction,
)
from football_data_platform.domain.training import ModelRunStatus, verify_model_run_artifact

PAPER_LEDGER_SCHEMA_VERSION = 1
_ENTRY_PREFIX = "paper-bet-entry:"
_CANDIDATE_PREFIX = "paper-bet-candidate:"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_ENTRY_ID = _ENTRY_PREFIX + "0" * 64


class CandidateDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SettlementOutcome(StrEnum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    HALF_WIN = "half_win"
    HALF_LOSS = "half_loss"
    VOID = "void"


# Discoverable aliases used by downstream reporting code.
BetDecision = CandidateDecision
SettlementStatus = SettlementOutcome


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Versioned limits used to approve a paper stake."""

    version: str = "risk/1"
    bankroll: float = 1000.0
    max_stake: float = 100.0
    max_match_exposure: float = 100.0
    max_daily_exposure: float = 500.0
    kelly_fraction: float | None = None
    currency: str = "unit"

    def __post_init__(self) -> None:
        _require_text(self.version, "risk config version")
        _require_text(self.currency, "risk config currency")
        for name in (
            "bankroll",
            "max_stake",
            "max_match_exposure",
            "max_daily_exposure",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.kelly_fraction is not None and (
            not math.isfinite(self.kelly_fraction) or not 0 <= self.kelly_fraction <= 1
        ):
            raise ValueError("kelly_fraction must be finite and between 0 and 1")

    def validate_exposure(self, stake: float, match_exposure: float, daily_exposure: float) -> None:
        for name, value in (
            ("stake", stake),
            ("match_exposure", match_exposure),
            ("daily_exposure", daily_exposure),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if stake <= 0:
            raise ValueError("accepted stake must be positive")
        if stake > self.max_stake:
            raise ValueError("stake exceeds risk config max_stake")
        if stake > self.bankroll:
            raise ValueError("stake exceeds risk config bankroll")
        if match_exposure < stake:
            raise ValueError("match_exposure must include stake")
        if daily_exposure < stake:
            raise ValueError("daily_exposure must include stake")
        if match_exposure > self.max_match_exposure:
            raise ValueError("match exposure exceeds risk config max_match_exposure")
        if daily_exposure > self.max_daily_exposure:
            raise ValueError("daily exposure exceeds risk config max_daily_exposure")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bankroll": float(self.bankroll),
            "max_stake": float(self.max_stake),
            "max_match_exposure": float(self.max_match_exposure),
            "max_daily_exposure": float(self.max_daily_exposure),
            "kelly_fraction": (None if self.kelly_fraction is None else float(self.kelly_fraction)),
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class SettlementRules:
    """Versioned 90-minute settlement definition for one selection.

    ``line`` is a selected-side handicap for ``asian_handicap`` and the total
    goals line for ``total_goals``. Quarter lines are split into their two
    adjacent half-lines, yielding half-win or half-loss deterministically.
    """

    version: str = "settlement/1"
    market_type: str = "result_90"
    selection: str = "home"
    line: float | None = None
    void_if_no_result: bool = False

    def __post_init__(self) -> None:
        _require_text(self.version, "settlement rules version")
        _require_text(self.market_type, "settlement market_type")
        _require_text(self.selection, "settlement selection")
        if self.line is not None and (not math.isfinite(self.line) or abs(self.line) > 1000):
            raise ValueError("settlement line must be finite")

    def settle(self, result: MatchResult90 | None) -> SettlementOutcome:
        """Compute the outcome from a 90-minute result, never from caller status."""

        market_type = self.market_type.lower()
        if market_type in {"void", "cancelled", "postponed"}:
            return SettlementOutcome.VOID
        if result is None:
            if self.void_if_no_result:
                return SettlementOutcome.VOID
            raise ValueError("a 90-minute result is required for settlement")
        if market_type in {"result", "result_90", "1x2"}:
            if self.selection not in {"home", "draw", "away"}:
                raise ValueError("result_90 selection must be home, draw, or away")
            actual = _result_outcome(result)
            return SettlementOutcome.WIN if self.selection == actual else SettlementOutcome.LOSS
        if market_type in {"total_goals", "totals"}:
            if self.selection not in {"over", "under"} or self.line is None:
                raise ValueError("total_goals requires over/under selection and line")
            total = result.home_goals + result.away_goals
            if self.selection == "over":
                return _settle_adjusted_value(float(total), -self.line)
            return _settle_adjusted_value(float(-total), self.line)
        if market_type in {"asian_handicap", "handicap", "home_handicap_3way"}:
            if self.selection not in {"home", "away"} or self.line is None:
                raise ValueError("asian_handicap requires home/away selection and line")
            margin = result.home_goals - result.away_goals
            selected_margin = margin if self.selection == "home" else -margin
            return _settle_adjusted_value(float(selected_margin), self.line)
        raise ValueError(f"unsupported settlement market_type {self.market_type!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "market_type": self.market_type,
            "selection": self.selection,
            "line": self.line,
            "void_if_no_result": self.void_if_no_result,
        }


SettlementRule = SettlementRules


@dataclass(frozen=True, slots=True)
class SettlementComputation:
    outcome: SettlementOutcome
    payout: float
    profit: float


def compute_settlement(
    rules: SettlementRules,
    *,
    result: MatchResult90 | None,
    stake: float,
    decimal_odds: float,
) -> SettlementComputation:
    """Return deterministic payout/profit for one decimal-odds paper bet."""

    if not math.isfinite(stake) or stake <= 0:
        raise ValueError("stake must be finite and positive")
    if not math.isfinite(decimal_odds) or decimal_odds <= 1:
        raise ValueError("decimal_odds must be finite and greater than 1")
    outcome = rules.settle(result)
    payout = _payout(outcome, stake, decimal_odds)
    return SettlementComputation(outcome, payout, payout - stake)


@dataclass(frozen=True, slots=True)
class PaperBetEntry:
    """One immutable candidate/placement/settlement revision."""

    entry_id: str
    candidate_id: str
    revision: int
    decision: CandidateDecision
    decision_reason: str
    match_id: MatchId
    prediction_id: PredictionId
    model_run_id: ModelRunId
    prediction_generated_at: datetime
    prediction_snapshot_as_of: datetime
    market_snapshot: MarketSnapshot | None
    market_type: str | None
    selection: str | None
    decimal_odds: float | None
    risk_config: RiskConfig
    stake: float
    match_exposure: float
    daily_exposure: float
    placed_at: datetime
    settlement_rules: SettlementRules | None = None
    result: MatchResult90 | None = None
    settlement_outcome: SettlementOutcome = SettlementOutcome.PENDING
    payout: float | None = None
    profit: float | None = None
    settled_at: datetime | None = None
    prior_entry_id: str | None = None
    schema_version: int = PAPER_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAPER_LEDGER_SCHEMA_VERSION:
            raise ValueError("unsupported paper ledger schema version")
        _validate_digest_id(self.entry_id, _ENTRY_PREFIX, "entry_id")
        _validate_digest_id(self.candidate_id, _CANDIDATE_PREFIX, "candidate_id")
        if self.revision < 1:
            raise ValueError("revision must be positive")
        if self.revision == 1 and self.prior_entry_id is not None:
            raise ValueError("first revision cannot have prior_entry_id")
        if self.revision > 1 and self.prior_entry_id is None:
            raise ValueError("later revisions require prior_entry_id")
        _require_text(self.decision_reason, "decision_reason")
        if not isinstance(self.decision, CandidateDecision):
            raise TypeError("decision must be a CandidateDecision")
        if not isinstance(self.match_id, MatchId):
            raise TypeError("match_id must be a MatchId")
        if not isinstance(self.prediction_id, PredictionId):
            raise TypeError("prediction_id must be a PredictionId")
        if not isinstance(self.model_run_id, ModelRunId):
            raise TypeError("model_run_id must be a ModelRunId")
        require_utc(self.placed_at, "placed_at")
        require_utc(self.prediction_generated_at, "prediction_generated_at")
        require_utc(self.prediction_snapshot_as_of, "prediction_snapshot_as_of")
        if self.prediction_generated_at > self.placed_at:
            raise ValueError("prediction generated_at cannot follow placed_at")
        if self.prediction_snapshot_as_of > self.placed_at:
            raise ValueError("prediction snapshot_as_of cannot follow placed_at")
        for name in ("stake", "match_exposure", "daily_exposure"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.market_snapshot is not None:
            if self.market_snapshot.match_id != self.match_id:
                raise ValueError("market snapshot and entry refer to different matches")
            if self.decision is CandidateDecision.ACCEPTED:
                if self.market_type != self.market_snapshot.market_type:
                    raise ValueError("entry market_type does not match market snapshot")
                if self.selection is None:
                    raise ValueError("accepted entry requires a selection")
                _find_quote(self.market_snapshot.quotes, self.selection)
        if self.result is not None and self.result.match_id != self.match_id:
            raise ValueError("result and entry refer to different matches")
        if self.result is not None and self.result.known_at < self.placed_at:
            raise ValueError("result known_at cannot precede placed_at")
        if self.settled_at is not None:
            require_utc(self.settled_at, "settled_at")
            if self.result is not None and self.settled_at < self.result.known_at:
                raise ValueError("settled_at cannot precede result known_at")
        if self.decision is CandidateDecision.REJECTED:
            if any(value != 0 for value in (self.stake, self.match_exposure, self.daily_exposure)):
                raise ValueError("rejected candidates must have zero stake and exposure")
            if self.settlement_rules is not None or self.result is not None:
                raise ValueError("rejected candidates cannot be settled")
            if self.settlement_outcome is not SettlementOutcome.PENDING:
                raise ValueError("rejected candidates remain pending")
            if self.payout is not None or self.profit is not None or self.settled_at is not None:
                raise ValueError("rejected candidates cannot have payout")
        else:
            if self.market_snapshot is None:
                raise ValueError("accepted entries require a market snapshot")
            if self.market_type is None or self.selection is None:
                raise ValueError("accepted entries require market_type and selection")
            if (
                self.decimal_odds is None
                or not math.isfinite(self.decimal_odds)
                or self.decimal_odds <= 1
            ):
                raise ValueError("accepted entries require valid decimal odds")
            if self.stake <= 0:
                raise ValueError("accepted entries require positive stake")
            if self.settlement_rules is None:
                raise ValueError("accepted entries require settlement rules")
            self.risk_config.validate_exposure(self.stake, self.match_exposure, self.daily_exposure)
            if self.settlement_outcome is SettlementOutcome.PENDING:
                if self.result is not None or self.payout is not None or self.profit is not None:
                    raise ValueError("pending entries cannot have settlement values")
                if self.settled_at is not None:
                    raise ValueError("pending entries cannot have settled_at")
            else:
                if self.result is None and self.settlement_outcome is not SettlementOutcome.VOID:
                    raise ValueError("non-void settled entries require a result")
                if self.payout is None or self.profit is None:
                    raise ValueError("settled entries require computed payout")
                if self.settled_at is None:
                    raise ValueError("settled entries require settled_at")
                expected = compute_settlement(
                    self.settlement_rules,
                    result=self.result,
                    stake=self.stake,
                    decimal_odds=self.decimal_odds,
                )
                if self.settlement_outcome is not expected.outcome:
                    raise ValueError("settlement outcome does not match settlement rules")
                if not math.isclose(self.payout, expected.payout, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("payout does not match settlement rules")
                if not math.isclose(self.profit, expected.profit, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("profit does not match settlement rules")

    @classmethod
    def accepted(
        cls,
        *,
        prediction: ScorePrediction,
        market_snapshot: MarketSnapshot,
        market_source_validator: MarketSourceValidator,
        selection: str,
        risk_config: RiskConfig,
        stake: float,
        match_exposure: float,
        daily_exposure: float,
        placed_at: datetime,
        settlement_rules: SettlementRules,
        model_run_validator: ModelRunValidator | None = None,
        candidate_id: str | None = None,
        revision: int = 1,
        prior_entry_id: str | None = None,
        decision_reason: str = "accepted",
    ) -> PaperBetEntry:
        """Build an accepted entry after all prediction/market gates pass."""

        verify_score_prediction(prediction, model_run_validator=model_run_validator)
        verify_market_snapshot(market_snapshot, source_validator=market_source_validator)
        _validate_market_gate(
            prediction=prediction,
            market_snapshot=market_snapshot,
            selection=selection,
            settlement_rules=settlement_rules,
            placed_at=placed_at,
        )
        risk_config.validate_exposure(stake, match_exposure, daily_exposure)
        quote = _find_quote(market_snapshot.quotes, selection)
        normalized_candidate_id = _normalize_candidate_id(
            candidate_id,
            prediction=prediction,
            market_snapshot=market_snapshot,
            selection=selection,
            placed_at=placed_at,
        )
        entry = cls(
            entry_id=_PLACEHOLDER_ENTRY_ID,
            candidate_id=normalized_candidate_id,
            revision=revision,
            decision=CandidateDecision.ACCEPTED,
            decision_reason=decision_reason,
            match_id=prediction.match_id,
            prediction_id=prediction.id,
            model_run_id=prediction.model_run_id,
            prediction_generated_at=prediction.generated_at,
            prediction_snapshot_as_of=prediction.snapshot_as_of,
            market_snapshot=market_snapshot,
            market_type=market_snapshot.market_type,
            selection=selection,
            decimal_odds=quote.decimal_odds,
            risk_config=risk_config,
            stake=stake,
            match_exposure=match_exposure,
            daily_exposure=daily_exposure,
            placed_at=placed_at,
            settlement_rules=settlement_rules,
            prior_entry_id=prior_entry_id,
        )
        return _with_entry_id(entry)

    @classmethod
    def rejected(
        cls,
        *,
        prediction: ScorePrediction,
        reason: str,
        placed_at: datetime,
        risk_config: RiskConfig | None = None,
        market_snapshot: MarketSnapshot | None = None,
        selection: str | None = None,
        candidate_id: str | None = None,
        revision: int = 1,
        prior_entry_id: str | None = None,
        model_run_validator: ModelRunValidator | None = None,
    ) -> PaperBetEntry:
        """Record a rejected candidate without creating a position."""

        verify_score_prediction(prediction, model_run_validator=model_run_validator)
        require_utc(placed_at, "placed_at")
        normalized_candidate_id = _normalize_candidate_id(
            candidate_id,
            prediction=prediction,
            market_snapshot=market_snapshot,
            selection=selection,
            placed_at=placed_at,
        )
        entry = cls(
            entry_id=_PLACEHOLDER_ENTRY_ID,
            candidate_id=normalized_candidate_id,
            revision=revision,
            decision=CandidateDecision.REJECTED,
            decision_reason=reason,
            match_id=prediction.match_id,
            prediction_id=prediction.id,
            model_run_id=prediction.model_run_id,
            prediction_generated_at=prediction.generated_at,
            prediction_snapshot_as_of=prediction.snapshot_as_of,
            market_snapshot=market_snapshot,
            market_type=market_snapshot.market_type if market_snapshot else None,
            selection=selection,
            decimal_odds=None,
            risk_config=risk_config or RiskConfig(),
            stake=0.0,
            match_exposure=0.0,
            daily_exposure=0.0,
            placed_at=placed_at,
            prior_entry_id=prior_entry_id,
        )
        return _with_entry_id(entry)

    def settle(
        self,
        result: MatchResult90 | None,
        *,
        settled_at: datetime,
        result_validator: MatchResultValidator | None = None,
    ) -> PaperBetEntry:
        """Append a computed settlement revision for this accepted entry."""

        if self.decision is not CandidateDecision.ACCEPTED:
            raise ValueError("rejected candidates cannot be settled")
        if self.settlement_outcome is not SettlementOutcome.PENDING:
            raise ValueError("entry is already settled")
        require_utc(settled_at, "settled_at")
        if settled_at < self.placed_at:
            raise ValueError("settled_at cannot precede placed_at")
        if result is not None and result_validator is not None:
            verify_match_result_source(result, result_validator)
        if result is not None and result.known_at < self.placed_at:
            raise ValueError("result known_at cannot precede placed_at")
        if self.settlement_rules is None or self.decimal_odds is None:
            raise ValueError("entry has no settlement contract")
        computation = compute_settlement(
            self.settlement_rules,
            result=result,
            stake=self.stake,
            decimal_odds=self.decimal_odds,
        )
        revised = replace(
            self,
            entry_id=_PLACEHOLDER_ENTRY_ID,
            revision=self.revision + 1,
            prior_entry_id=self.entry_id,
            result=result,
            settlement_outcome=computation.outcome,
            payout=computation.payout,
            profit=computation.profit,
            settled_at=settled_at,
        )
        return _with_entry_id(revised)

    def revised(self, **changes: Any) -> PaperBetEntry:
        """Create an append-only revision; the prior entry remains untouched."""

        forbidden = {"entry_id", "candidate_id", "revision", "prior_entry_id", "schema_version"}
        if forbidden.intersection(changes):
            raise ValueError("revision identity fields are managed by the ledger")
        revised = replace(
            self,
            **changes,
            entry_id=_PLACEHOLDER_ENTRY_ID,
            revision=self.revision + 1,
            prior_entry_id=self.entry_id,
        )
        return _with_entry_id(revised)

    @property
    def is_accepted(self) -> bool:
        return self.decision is CandidateDecision.ACCEPTED

    @property
    def status(self) -> CandidateDecision:
        return self.decision

    @property
    def settlement(self) -> SettlementOutcome:
        return self.settlement_outcome

    @property
    def market_snapshot_id(self) -> str | None:
        return self.market_snapshot.id.value if self.market_snapshot is not None else None

    @property
    def market_raw_asset_ref(self) -> str | None:
        return self.market_snapshot.raw_asset_ref if self.market_snapshot is not None else None

    @property
    def risk_config_version(self) -> str:
        return self.risk_config.version

    @property
    def settlement_rules_version(self) -> str | None:
        return self.settlement_rules.version if self.settlement_rules is not None else None


# Compatibility names for callers that use "record" terminology.
PaperBetRecord = PaperBetEntry
PaperBetLedgerEntry = PaperBetEntry
PaperBetCandidate = PaperBetEntry


def verify_paper_bet_entry(
    entry: PaperBetEntry,
    *,
    market_source_validator: MarketSourceValidator | None = None,
    result_validator: MatchResultValidator | None = None,
    model_run_validator: ModelRunValidator | None = None,
) -> None:
    """Verify content addressing and optional market/result source lineage."""

    if not isinstance(entry, PaperBetEntry):
        raise TypeError("entry must be a PaperBetEntry")
    _verify_prediction_model_reference(entry, model_run_validator)
    if entry.decision is CandidateDecision.ACCEPTED:
        if market_source_validator is None:
            raise ValueError("accepted entry requires a market raw evidence validator")
        assert entry.market_snapshot is not None
        verify_market_snapshot(entry.market_snapshot, source_validator=market_source_validator)
        _validate_market_gate(
            prediction=None,
            market_snapshot=entry.market_snapshot,
            selection=entry.selection or "",
            settlement_rules=entry.settlement_rules,
            placed_at=entry.placed_at,
            prediction_snapshot_as_of=entry.prediction_snapshot_as_of,
        )
    if entry.result is not None and result_validator is not None:
        verify_match_result_source(entry.result, result_validator)
    expected = _entry_digest(entry)
    if entry.entry_id != expected:
        raise ValueError("paper bet entry identity does not match its canonical content")


def paper_bet_entry_payload(entry: PaperBetEntry, *, include_id: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": entry.schema_version,
        "candidate_id": entry.candidate_id,
        "revision": entry.revision,
        "decision": entry.decision.value,
        "decision_reason": entry.decision_reason,
        "match_id": entry.match_id.value,
        "prediction_id": entry.prediction_id.value,
        "model_run_id": entry.model_run_id.value,
        "market_snapshot_id": entry.market_snapshot_id,
        "market_raw_asset_ref": entry.market_raw_asset_ref,
        "prediction_generated_at": _timestamp(entry.prediction_generated_at),
        "prediction_snapshot_as_of": _timestamp(entry.prediction_snapshot_as_of),
        "market_snapshot": _market_snapshot_payload(entry.market_snapshot),
        "market_type": entry.market_type,
        "selection": entry.selection,
        "decimal_odds": None if entry.decimal_odds is None else float(entry.decimal_odds),
        "risk_config": entry.risk_config.to_payload(),
        "risk_config_version": entry.risk_config_version,
        "stake": float(entry.stake),
        "match_exposure": float(entry.match_exposure),
        "daily_exposure": float(entry.daily_exposure),
        "placed_at": _timestamp(entry.placed_at),
        "settlement_rules": (
            entry.settlement_rules.to_payload() if entry.settlement_rules is not None else None
        ),
        "settlement_rules_version": entry.settlement_rules_version,
        "result": _result_payload(entry.result),
        "settlement_outcome": entry.settlement_outcome.value,
        "payout": None if entry.payout is None else float(entry.payout),
        "profit": None if entry.profit is None else float(entry.profit),
        "settled_at": _timestamp(entry.settled_at) if entry.settled_at else None,
        "prior_entry_id": entry.prior_entry_id,
    }
    if include_id:
        payload = {"id": entry.entry_id, **payload}
    return payload


def settle_paper_bet(
    entry: PaperBetEntry,
    result: MatchResult90 | None,
    *,
    settled_at: datetime,
    result_validator: MatchResultValidator | None = None,
) -> PaperBetEntry:
    return entry.settle(
        result,
        settled_at=settled_at,
        result_validator=result_validator,
    )


def _verify_prediction_model_reference(
    entry: PaperBetEntry,
    validator: ModelRunValidator | None,
) -> None:
    if validator is None:
        return
    try:
        artifact = validator.load_model_run(entry.model_run_id.value)
        verify_model_run_artifact(artifact)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("prediction model run is unavailable or invalid") from error
    if artifact.model_run_id != entry.model_run_id.value:
        raise ValueError("ledger model run validator returned a different model run")
    if artifact.status is not ModelRunStatus.SUCCEEDED:
        raise ValueError("ledger entries require a successful model run")
    if artifact.task != "score-model":
        raise ValueError("ledger entries require a score-model run")
    if artifact.ended_at > entry.prediction_generated_at:
        raise ValueError("ledger prediction precedes its model run completion")


def _validate_market_gate(
    *,
    prediction: ScorePrediction | None,
    market_snapshot: MarketSnapshot,
    selection: str,
    settlement_rules: SettlementRules | None,
    placed_at: datetime,
    prediction_snapshot_as_of: datetime | None = None,
) -> None:
    if market_snapshot.data_kind is not MarketDataKind.REAL:
        raise ValueError("paper betting requires a real market snapshot")
    if market_snapshot.status is not MarketStatus.OPEN:
        raise ValueError("paper betting requires an open market snapshot")
    require_utc(placed_at, "placed_at")
    if market_snapshot.observed_at > placed_at:
        raise ValueError("market snapshot was observed after placed_at")
    if prediction is not None and prediction.generated_at > placed_at:
        raise ValueError("prediction was generated after placed_at")
    as_of = prediction.snapshot_as_of if prediction is not None else prediction_snapshot_as_of
    if as_of is not None and market_snapshot.observed_at > as_of:
        raise ValueError("market snapshot was observed after prediction snapshot as_of")
    if not market_snapshot.quotes:
        raise ValueError("paper betting requires a complete market snapshot")
    outcomes = {quote.outcome for quote in market_snapshot.quotes}
    expected_outcomes: set[str] | None = None
    if market_snapshot.market_type in {"result_90", "result", "1x2"}:
        expected_outcomes = {"home", "draw", "away"}
    elif market_snapshot.market_type in {"total_goals", "totals"}:
        expected_outcomes = {"over", "under"}
    elif market_snapshot.market_type in {"asian_handicap", "handicap"}:
        expected_outcomes = {"home", "away"}
    elif market_snapshot.market_type == "home_handicap_3way":
        expected_outcomes = {"home", "draw", "away"}
    else:
        raise ValueError(f"unsupported paper betting market_type {market_snapshot.market_type!r}")
    if expected_outcomes is not None and outcomes != expected_outcomes:
        raise ValueError(f"paper betting requires a complete {market_snapshot.market_type} market")
    _find_quote(market_snapshot.quotes, selection)
    if settlement_rules is not None and settlement_rules.selection != selection:
        raise ValueError("settlement rules selection does not match market selection")
    if (
        settlement_rules is not None
        and settlement_rules.market_type
        not in _compatible_market_types(market_snapshot.market_type)
    ):
        raise ValueError("settlement rules market_type does not match market snapshot")


def _find_quote(quotes: tuple[MarketQuote, ...], selection: str) -> MarketQuote:
    if not selection or selection.strip() != selection:
        raise ValueError("selection must be non-empty text")
    for quote in quotes:
        if quote.outcome == selection:
            return quote
    raise ValueError(f"market snapshot has no quote for selection {selection!r}")


def _compatible_market_types(market_type: str) -> set[str]:
    if market_type in {"result_90", "result", "1x2"}:
        return {"result_90", "result", "1x2"}
    if market_type in {"total_goals", "totals"}:
        return {"total_goals", "totals"}
    if market_type in {"asian_handicap", "handicap"}:
        return {"asian_handicap", "handicap"}
    if market_type == "home_handicap_3way":
        return {"home_handicap_3way"}
    return {market_type}


def _derive_candidate_id(
    *,
    prediction: ScorePrediction,
    market_snapshot: MarketSnapshot | None,
    selection: str | None,
    placed_at: datetime,
) -> str:
    payload = {
        "prediction_id": prediction.id.value,
        "model_run_id": prediction.model_run_id.value,
        "match_id": prediction.match_id.value,
        "market_snapshot_id": market_snapshot.id.value if market_snapshot else None,
        "selection": selection,
        "placed_at": _timestamp(placed_at),
    }
    return _CANDIDATE_PREFIX + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _normalize_candidate_id(
    candidate_id: str | None,
    *,
    prediction: ScorePrediction,
    market_snapshot: MarketSnapshot | None,
    selection: str | None,
    placed_at: datetime,
) -> str:
    if candidate_id is None:
        return _derive_candidate_id(
            prediction=prediction,
            market_snapshot=market_snapshot,
            selection=selection,
            placed_at=placed_at,
        )
    _require_text(candidate_id, "candidate_id")
    if candidate_id.startswith(_CANDIDATE_PREFIX) and _DIGEST.fullmatch(
        candidate_id.removeprefix(_CANDIDATE_PREFIX)
    ):
        return candidate_id
    return (
        _CANDIDATE_PREFIX
        + hashlib.sha256(_canonical_json({"idempotency_key": candidate_id})).hexdigest()
    )


def _with_entry_id(entry: PaperBetEntry) -> PaperBetEntry:
    return replace(entry, entry_id=_entry_digest(entry))


def _entry_digest(entry: PaperBetEntry) -> str:
    payload = paper_bet_entry_payload(entry, include_id=False)
    return _ENTRY_PREFIX + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _market_snapshot_payload(snapshot: MarketSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "id": snapshot.id.value,
        "schema_version": snapshot.schema_version,
        "match_id": snapshot.match_id.value,
        "source": snapshot.source,
        "market_type": snapshot.market_type,
        "status": snapshot.status.value,
        "data_kind": snapshot.data_kind.value,
        "observed_at": _timestamp(snapshot.observed_at),
        "quotes": [
            {"outcome": quote.outcome, "decimal_odds": float(quote.decimal_odds)}
            for quote in snapshot.quotes
        ],
        "raw_asset_ref": snapshot.raw_asset_ref,
    }


def _result_payload(result: MatchResult90 | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "match_id": result.match_id.value,
        "home_goals": result.home_goals,
        "away_goals": result.away_goals,
        "known_at": _timestamp(result.known_at),
        "source_ref": result.source_ref,
    }


def _result_outcome(result: MatchResult90) -> str:
    if result.home_goals > result.away_goals:
        return "home"
    if result.home_goals < result.away_goals:
        return "away"
    return "draw"


def _settle_adjusted_value(base_value: float, adjustment: float) -> SettlementOutcome:
    if abs(adjustment * 2 - round(adjustment * 2)) < 1e-9:
        return _sign_outcome(base_value + adjustment)
    lower_adjustment = math.floor(adjustment * 2) / 2
    upper_adjustment = lower_adjustment + 0.5
    first = _sign_outcome(base_value + lower_adjustment)
    second = _sign_outcome(base_value + upper_adjustment)
    return _combine_half_outcomes(first, second)


def _sign_outcome(value: float) -> SettlementOutcome:
    if abs(value) < 1e-9:
        return SettlementOutcome.PUSH
    return SettlementOutcome.WIN if value > 0 else SettlementOutcome.LOSS


def _combine_half_outcomes(
    first: SettlementOutcome, second: SettlementOutcome
) -> SettlementOutcome:
    if first is second:
        return first
    pair = {first, second}
    if pair == {SettlementOutcome.WIN, SettlementOutcome.PUSH}:
        return SettlementOutcome.HALF_WIN
    if pair == {SettlementOutcome.LOSS, SettlementOutcome.PUSH}:
        return SettlementOutcome.HALF_LOSS
    raise ValueError("invalid quarter-line settlement combination")


def _payout(outcome: SettlementOutcome, stake: float, odds: float) -> float:
    if outcome is SettlementOutcome.WIN:
        return stake * odds
    if outcome is SettlementOutcome.LOSS:
        return 0.0
    if outcome is SettlementOutcome.PUSH or outcome is SettlementOutcome.VOID:
        return stake
    if outcome is SettlementOutcome.HALF_WIN:
        return stake * (odds + 1.0) / 2.0
    if outcome is SettlementOutcome.HALF_LOSS:
        return stake / 2.0
    raise ValueError("pending outcome cannot be paid out")


def _validate_digest_id(value: str, prefix: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{field_name} must start with {prefix!r}")
    if not _DIGEST.fullmatch(value.removeprefix(prefix)):
        raise ValueError(f"{field_name} must contain a SHA-256 digest")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _timestamp(value: datetime) -> str:
    require_utc(value, "timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
