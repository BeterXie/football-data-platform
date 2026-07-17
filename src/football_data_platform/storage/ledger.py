"""Content-addressed, append-only storage for paper betting entries."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import (
    MarketSnapshotId,
    MatchId,
    ModelRunId,
    PredictionId,
)
from football_data_platform.domain.ledger import (
    CandidateDecision,
    MatchResultValidator,
    ModelRunValidator,
    PaperBetEntry,
    RiskConfig,
    SettlementOutcome,
    SettlementRules,
    paper_bet_entry_payload,
    verify_paper_bet_entry,
)
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketSnapshot,
    MarketSourceValidator,
    MarketStatus,
    MatchResult90,
    ScorePrediction,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    DerivedArtifactManifest,
    RunManifest,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import TrainingArtifactStore


class LedgerConflictError(ArchiveConflictError):
    """Raised when an immutable entry or revision conflicts with existing data."""


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    """Accounting totals reconstructed only from the latest entry revisions."""

    initial_bankroll: float
    total_entries: int
    accepted_entries: int
    rejected_entries: int
    settled_entries: int
    total_staked: float
    total_payout: float
    net_profit: float
    balance: float
    open_stake: float
    match_exposure: tuple[tuple[str, float], ...]
    daily_exposure: tuple[tuple[str, float], ...]

    def match_exposure_map(self) -> dict[str, float]:
        return dict(self.match_exposure)

    def daily_exposure_map(self) -> dict[str, float]:
        return dict(self.daily_exposure)


class PaperBetLedger:
    """Append-only paper ledger with deterministic idempotency and revisions."""

    def __init__(
        self,
        layout: DataLayout,
        *,
        market_source_validator: MarketSourceValidator | None = None,
        result_validator: MatchResultValidator | None = None,
        model_run_validator: ModelRunValidator | None = None,
        derived_archive: DerivedArchive | None = None,
    ) -> None:
        self.layout = layout.ensure()
        self.market_source_validator = market_source_validator or RawArchive(self.layout)
        if result_validator is None:
            canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
            canonical.initialize()
            result_validator = CanonicalFactStore(
                canonical,
                raw_archive=RawArchive(self.layout),
            )
        self.result_validator = result_validator
        self.persisted_model_runs = TrainingArtifactStore(self.layout)
        self.model_run_validator = model_run_validator or self.persisted_model_runs
        self.derived = derived_archive or DerivedArchive(self.layout)
        self.root = self.layout.paper_ledger / "entries"
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, entry: PaperBetEntry) -> Path:
        """Persist one entry; an identical retry returns the same path."""

        self._verify_persisted_prediction_entry(entry)
        verify_paper_bet_entry(
            entry,
            market_source_validator=self.market_source_validator,
            result_validator=self.result_validator,
            model_run_validator=self.model_run_validator,
        )
        path = self.entry_path(entry.entry_id)
        encoded = _encode(paper_bet_entry_payload(entry))
        if path.exists():
            if path.read_bytes() != encoded:
                raise LedgerConflictError(f"paper ledger entry conflicts at {path}")
            # A previous process may have persisted the entry but failed before registering the
            # derived summary.  Re-running the exact append repairs that checkpoint idempotently.
            self._write_derived_summary()
            return path
        if entry.revision > 1:
            self._validate_revision_parent(entry)
        elif any(item.candidate_id == entry.candidate_id for item in self.entries()):
            raise LedgerConflictError("candidate already has an immutable first revision")
        self._validate_declared_exposure(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise LedgerConflictError(f"paper ledger entry conflicts at {path}") from None
        self._write_derived_summary()
        return path

    def append_candidate(
        self,
        *,
        prediction: ScorePrediction,
        market_snapshot: MarketSnapshot | None,
        selection: str | None,
        risk_config: RiskConfig,
        stake: float,
        match_exposure: float,
        daily_exposure: float,
        placed_at: datetime,
        settlement_rules: SettlementRules | None,
        candidate_id: str | None = None,
    ) -> PaperBetEntry:
        """Evaluate and persist an accepted or rejected candidate.

        Validation failures become an auditable rejected candidate with zero
        stake. This keeps market/risk failures visible without manufacturing a
        position.
        """

        try:
            self._verify_persisted_prediction(prediction)
            if market_snapshot is None or settlement_rules is None or selection is None:
                raise ValueError("market snapshot, selection, and settlement rules are required")
            entry = PaperBetEntry.accepted(
                prediction=prediction,
                market_snapshot=market_snapshot,
                market_source_validator=self._require_market_validator(),
                model_run_validator=self.model_run_validator,
                selection=selection,
                risk_config=risk_config,
                stake=stake,
                match_exposure=match_exposure,
                daily_exposure=daily_exposure,
                placed_at=placed_at,
                settlement_rules=settlement_rules,
                candidate_id=candidate_id,
            )
            self._validate_declared_exposure(entry)
        except (TypeError, ValueError, OSError, RuntimeError) as error:
            entry = PaperBetEntry.rejected(
                prediction=prediction,
                reason=_reason(error),
                placed_at=placed_at,
                risk_config=risk_config,
                market_snapshot=market_snapshot,
                selection=selection,
                candidate_id=candidate_id,
                model_run_validator=self.model_run_validator,
            )
        self.append(entry)
        return entry

    def load(self, entry_id: str) -> PaperBetEntry:
        path = self.entry_path(entry_id)
        payload = _read_object(path)
        entry = parse_paper_bet_entry_payload(payload)
        self._verify_persisted_prediction_entry(entry)
        verify_paper_bet_entry(
            entry,
            market_source_validator=self.market_source_validator,
            result_validator=self.result_validator,
            model_run_validator=self.model_run_validator,
        )
        return entry

    def entries(self, *, latest_only: bool = False) -> tuple[PaperBetEntry, ...]:
        parsed: list[PaperBetEntry] = []
        for path in sorted(self.root.glob("*/*.json")):
            entry = parse_paper_bet_entry_payload(_read_object(path))
            self._verify_persisted_prediction_entry(entry)
            verify_paper_bet_entry(
                entry,
                market_source_validator=self.market_source_validator,
                result_validator=self.result_validator,
                model_run_validator=self.model_run_validator,
            )
            parsed.append(entry)
        parsed.sort(key=lambda item: (item.candidate_id, item.revision, item.entry_id))
        if not latest_only:
            return tuple(parsed)
        latest: dict[str, PaperBetEntry] = {}
        for entry in parsed:
            current = latest.get(entry.candidate_id)
            if current is None or entry.revision > current.revision:
                latest[entry.candidate_id] = entry
        return tuple(sorted(latest.values(), key=lambda item: (item.placed_at, item.entry_id)))

    def latest(self, candidate_id: str) -> PaperBetEntry:
        matches = [entry for entry in self.entries() if entry.candidate_id == candidate_id]
        if not matches:
            raise FileNotFoundError(f"paper betting candidate not found: {candidate_id}")
        return max(matches, key=lambda item: item.revision)

    def append_revision(self, previous: PaperBetEntry, **changes: Any) -> PaperBetEntry:
        """Create and persist a new revision while retaining ``previous``."""

        current = self.load(previous.entry_id)
        if current != previous:
            raise LedgerConflictError("revision parent is not the persisted entry")
        revised = previous.revised(**changes)
        self.append(revised)
        return revised

    def settle(
        self,
        entry_or_id: PaperBetEntry | str,
        result: MatchResult90 | None,
        *,
        settled_at: datetime,
    ) -> PaperBetEntry:
        """Compute and append settlement; payout/status cannot be supplied by caller."""

        entry = entry_or_id if isinstance(entry_or_id, PaperBetEntry) else self.load(entry_or_id)
        persisted = self.load(entry.entry_id)
        if persisted != entry:
            raise LedgerConflictError("settlement parent is not the persisted entry")
        entry = persisted
        revised = entry.settle(
            result,
            settled_at=settled_at,
            result_validator=self.result_validator,
        )
        self.append(revised)
        return revised

    def recompute(self, *, initial_bankroll: float = 0.0) -> LedgerSummary:
        """Rebuild balance and open exposure from immutable latest revisions."""

        if initial_bankroll < 0:
            raise ValueError("initial_bankroll must be non-negative")
        latest = self.entries(latest_only=True)
        accepted = [entry for entry in latest if entry.decision is CandidateDecision.ACCEPTED]
        rejected = [entry for entry in latest if entry.decision is CandidateDecision.REJECTED]
        total_staked = sum(entry.stake for entry in accepted)
        total_payout = sum(entry.payout or 0.0 for entry in accepted)
        net_profit = sum(entry.profit or 0.0 for entry in accepted)
        open_entries = [
            entry for entry in accepted if entry.settlement_outcome is SettlementOutcome.PENDING
        ]
        match_exposure: defaultdict[str, float] = defaultdict(float)
        daily_exposure: defaultdict[str, float] = defaultdict(float)
        for entry in open_entries:
            match_exposure[entry.match_id.value] += entry.stake
            daily_exposure[entry.placed_at.date().isoformat()] += entry.stake
        return LedgerSummary(
            initial_bankroll=initial_bankroll,
            total_entries=len(latest),
            accepted_entries=len(accepted),
            rejected_entries=len(rejected),
            settled_entries=sum(
                entry.settlement_outcome is not SettlementOutcome.PENDING for entry in accepted
            ),
            total_staked=total_staked,
            total_payout=total_payout,
            net_profit=net_profit,
            balance=initial_bankroll - total_staked + total_payout,
            open_stake=sum(entry.stake for entry in open_entries),
            match_exposure=tuple(sorted(match_exposure.items())),
            daily_exposure=tuple(sorted(daily_exposure.items())),
        )

    def recompute_balance(self, *, initial_bankroll: float = 0.0) -> float:
        return self.recompute(initial_bankroll=initial_bankroll).balance

    def recompute_exposure(self) -> dict[str, dict[str, float]]:
        summary = self.recompute()
        return {
            "match": summary.match_exposure_map(),
            "day": summary.daily_exposure_map(),
        }

    def _write_derived_summary(self) -> tuple[str, str, str]:
        """Register the current ledger state as immutable derived evidence."""

        entries = self.entries(latest_only=True)
        if not entries:
            raise LedgerConflictError("cannot register an empty paper ledger")
        summary = asdict(self.recompute())
        summary["match_exposure"] = dict(summary["match_exposure"])
        summary["daily_exposure"] = dict(summary["daily_exposure"])
        entry_refs = tuple(sorted(entry.entry_id for entry in entries))
        semantic_digest = hashlib.sha256(_canonical_json(summary)).hexdigest()
        logical_ref = f"paper-ledger-summary:{semantic_digest}"
        summary_path = self.layout.paper_ledger / "summaries" / f"{semantic_digest}.json"
        encoded = _encode(summary)
        _write_immutable(summary_path, encoded)
        file_ref = f"file-sha256:{hashlib.sha256(encoded).hexdigest()}"
        generated_at = max(
            (item.settled_at or item.placed_at for item in entries),
            default=entries[-1].placed_at,
        )
        artifact = DerivedArtifactManifest.create(
            artifact_type="paper-ledger-summary",
            payload=summary,
            generated_at=generated_at,
            started_at=generated_at,
            ended_at=generated_at,
            transform_version="paper-ledger/1",
            code_version=DERIVED_CODE_VERSION,
            input_refs=entry_refs,
            output_refs=(logical_ref, file_ref),
            quality="ready",
        )
        self.derived.write_artifact_manifest(artifact)
        run = RunManifest.create(
            run_type="paper-ledger-recompute",
            started_at=generated_at,
            ended_at=generated_at,
            generated_at=generated_at,
            transform_version="paper-ledger/1",
            code_version=DERIVED_CODE_VERSION,
            input_refs=(*entry_refs, logical_ref),
            output_refs=(artifact.artifact_id, logical_ref, file_ref),
            status="succeeded",
            error=None,
            quality="ready",
            parameters={"entry_count": len(entries)},
            checkpoint="summary-registered",
            payload={
                "artifact_id": artifact.artifact_id,
                "summary_path": str(summary_path.resolve()),
                "entry_ids": list(entry_refs),
            },
        )
        run_path = self.derived.write_run_manifest(run)
        _write_json(
            self.layout.paper_ledger / "latest.json",
            {
                "artifact_id": artifact.artifact_id,
                "run_id": run.run_id,
                "summary": str(summary_path.resolve()),
                "manifest": str(run_path.resolve()),
            },
        )
        return artifact.artifact_id, run.run_id, str(summary_path)

    def entry_path(self, entry_id: str) -> Path:
        if not isinstance(entry_id, str) or not entry_id.startswith("paper-bet-entry:"):
            raise ValueError("invalid paper betting entry ID")
        digest = entry_id.removeprefix("paper-bet-entry:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid paper betting entry ID")
        return self.root / digest[:2] / f"{digest}.json"

    def _require_market_validator(self) -> MarketSourceValidator:
        return self.market_source_validator

    def _verify_persisted_prediction(self, prediction: ScorePrediction) -> None:
        try:
            stored = self.persisted_model_runs.load_verified_prediction(prediction.id.value)
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise ValueError(
                "paper ledger prediction artifact is unavailable or invalid"
            ) from error
        if stored != prediction:
            raise ValueError("paper ledger prediction does not match its persisted artifact")

    def _verify_persisted_prediction_entry(self, entry: PaperBetEntry) -> None:
        try:
            prediction = self.persisted_model_runs.load_verified_prediction(
                entry.prediction_id.value
            )
        except (OSError, RuntimeError, TypeError, ValueError, ArchiveConflictError) as error:
            raise ValueError(
                "paper ledger prediction artifact is unavailable or invalid"
            ) from error
        if (
            prediction.match_id != entry.match_id
            or prediction.model_run_id != entry.model_run_id
            or prediction.generated_at != entry.prediction_generated_at
            or prediction.snapshot_as_of != entry.prediction_snapshot_as_of
        ):
            raise ValueError("paper ledger entry does not match its prediction artifact")

    def _validate_revision_parent(self, entry: PaperBetEntry) -> None:
        assert entry.prior_entry_id is not None
        parent = self.load(entry.prior_entry_id)
        if parent.candidate_id != entry.candidate_id:
            raise LedgerConflictError("revision parent belongs to another candidate")
        if parent.revision + 1 != entry.revision:
            raise LedgerConflictError("revision number does not follow its parent")
        latest = self.latest(entry.candidate_id)
        if latest.entry_id != parent.entry_id:
            raise LedgerConflictError("revision parent is not the latest candidate revision")

    def _validate_declared_exposure(self, entry: PaperBetEntry) -> None:
        if (
            entry.decision is not CandidateDecision.ACCEPTED
            or entry.settlement_outcome is not SettlementOutcome.PENDING
        ):
            return
        expected_match, expected_day = self._expected_exposure(
            match_id=entry.match_id,
            placed_at=entry.placed_at,
            stake=entry.stake,
            exclude_candidate_id=entry.candidate_id,
        )
        if not math.isclose(entry.match_exposure, expected_match, rel_tol=0, abs_tol=1e-12):
            raise ValueError("match_exposure does not match ledger-computed exposure")
        if not math.isclose(entry.daily_exposure, expected_day, rel_tol=0, abs_tol=1e-12):
            raise ValueError("daily_exposure does not match ledger-computed exposure")

    def _expected_exposure(
        self,
        *,
        match_id: MatchId,
        placed_at: datetime,
        stake: float,
        exclude_candidate_id: str,
    ) -> tuple[float, float]:
        match_exposure = stake
        daily_exposure = stake
        placed_day = placed_at.date()
        for existing in self.entries(latest_only=True):
            if existing.candidate_id == exclude_candidate_id:
                continue
            if (
                existing.decision is not CandidateDecision.ACCEPTED
                or existing.settlement_outcome is not SettlementOutcome.PENDING
            ):
                continue
            if existing.match_id == match_id:
                match_exposure += existing.stake
            if existing.placed_at.date() == placed_day:
                daily_exposure += existing.stake
        return match_exposure, daily_exposure


# Discoverable storage alias.
PaperBetLedgerStore = PaperBetLedger


def parse_paper_bet_entry_payload(payload: dict[str, Any]) -> PaperBetEntry:
    """Parse one canonical JSON entry and reject tampered representations."""

    if not isinstance(payload, dict):
        raise LedgerConflictError("paper betting entry must be an object")
    try:
        market = _parse_market_snapshot(payload.get("market_snapshot"))
        risk_payload = payload["risk_config"]
        risk_config = RiskConfig(
            version=str(risk_payload["version"]),
            bankroll=float(risk_payload["bankroll"]),
            max_stake=float(risk_payload["max_stake"]),
            max_match_exposure=float(risk_payload["max_match_exposure"]),
            max_daily_exposure=float(risk_payload["max_daily_exposure"]),
            kelly_fraction=(
                None
                if risk_payload.get("kelly_fraction") is None
                else float(risk_payload["kelly_fraction"])
            ),
            currency=str(risk_payload["currency"]),
        )
        rules_payload = payload.get("settlement_rules")
        rules = (
            None
            if rules_payload is None
            else SettlementRules(
                version=str(rules_payload["version"]),
                market_type=str(rules_payload["market_type"]),
                selection=str(rules_payload["selection"]),
                line=(None if rules_payload.get("line") is None else float(rules_payload["line"])),
                void_if_no_result=_strict_bool(
                    rules_payload.get("void_if_no_result", False), "void_if_no_result"
                ),
            )
        )
        result = _parse_result(payload.get("result"))
        entry = PaperBetEntry(
            entry_id=str(payload["id"]),
            candidate_id=str(payload["candidate_id"]),
            revision=_strict_int(payload["revision"], "revision"),
            decision=CandidateDecision(str(payload["decision"])),
            decision_reason=str(payload["decision_reason"]),
            match_id=MatchId(str(payload["match_id"])),
            prediction_id=PredictionId(str(payload["prediction_id"])),
            model_run_id=ModelRunId(str(payload["model_run_id"])),
            prediction_generated_at=_parse_datetime(
                payload["prediction_generated_at"], "prediction_generated_at"
            ),
            prediction_snapshot_as_of=_parse_datetime(
                payload["prediction_snapshot_as_of"], "prediction_snapshot_as_of"
            ),
            market_snapshot=market,
            market_type=None if payload.get("market_type") is None else str(payload["market_type"]),
            selection=None if payload.get("selection") is None else str(payload["selection"]),
            decimal_odds=(
                None if payload.get("decimal_odds") is None else float(payload["decimal_odds"])
            ),
            risk_config=risk_config,
            stake=float(payload["stake"]),
            match_exposure=float(payload["match_exposure"]),
            daily_exposure=float(payload["daily_exposure"]),
            placed_at=_parse_datetime(payload["placed_at"], "placed_at"),
            settlement_rules=rules,
            result=result,
            settlement_outcome=SettlementOutcome(str(payload["settlement_outcome"])),
            payout=None if payload.get("payout") is None else float(payload["payout"]),
            profit=None if payload.get("profit") is None else float(payload["profit"]),
            settled_at=(
                None
                if payload.get("settled_at") is None
                else _parse_datetime(payload["settled_at"], "settled_at")
            ),
            prior_entry_id=(
                None if payload.get("prior_entry_id") is None else str(payload["prior_entry_id"])
            ),
            schema_version=_strict_int(payload["schema_version"], "schema_version"),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise LedgerConflictError(f"invalid paper betting entry: {error}") from error
    if payload != paper_bet_entry_payload(entry):
        raise LedgerConflictError("paper betting entry is not canonical")
    return entry


def _parse_market_snapshot(payload: Any) -> MarketSnapshot | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("market_snapshot must be an object")
    return MarketSnapshot(
        id=MarketSnapshotId(str(payload["id"])),
        schema_version=_strict_int(payload["schema_version"], "schema_version"),
        match_id=MatchId(str(payload["match_id"])),
        source=str(payload["source"]),
        market_type=str(payload["market_type"]),
        status=MarketStatus(str(payload["status"])),
        data_kind=MarketDataKind(str(payload["data_kind"])),
        observed_at=_parse_datetime(payload["observed_at"], "market observed_at"),
        quotes=tuple(
            MarketQuote(str(item["outcome"]), float(item["decimal_odds"]))
            for item in payload["quotes"]
        ),
        raw_asset_ref=str(payload["raw_asset_ref"]),
    )


def _parse_result(payload: Any) -> MatchResult90 | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("result must be an object")
    return MatchResult90(
        match_id=MatchId(str(payload["match_id"])),
        home_goals=_strict_int(payload["home_goals"], "home_goals"),
        away_goals=_strict_int(payload["away_goals"], "away_goals"),
        known_at=_parse_datetime(payload["known_at"], "result known_at"),
        source_ref=str(payload["source_ref"]),
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LedgerConflictError(f"cannot read paper betting entry {path}") from error
    if not isinstance(payload, dict):
        raise LedgerConflictError(f"paper betting entry is not an object: {path}")
    return payload


def _encode(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise LedgerConflictError(f"paper ledger summary conflicts at {path}")
        return
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise LedgerConflictError(f"paper ledger summary conflicts at {path}") from None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strict_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require_utc(parsed, field_name)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}") from error


def _reason(error: BaseException) -> str:
    message = str(error).strip() or type(error).__name__
    return f"rejected:{message}"
