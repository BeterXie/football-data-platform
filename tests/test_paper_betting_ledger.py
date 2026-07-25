from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from _formal_context import seed_formal_context
from _prediction_forgery import forge_persisted_prediction

from football_data_platform.domain.ids import (
    MarketSnapshotId,
    MatchId,
    ModelRunId,
    PredictionId,
    RawAssetId,
)
from football_data_platform.domain.ledger import (
    CandidateDecision,
    PaperBetEntry,
    RiskConfig,
    SettlementOutcome,
    SettlementRules,
    _validate_market_gate,
    compute_settlement,
    paper_bet_entry_payload,
)
from football_data_platform.domain.lifecycle import Qualification
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    MatchResult90,
    ScorePrediction,
    build_market_snapshot,
    build_score_prediction,
    verify_market_snapshot,
    verify_prediction_snapshot,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    PreMatchSnapshot,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
    verify_current_snapshot,
)
from football_data_platform.domain.training import (
    ModelRunArtifact,
    TrainingDatasetManifest,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contributions,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore, load_verified_match_result
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.ledger import (
    LedgerConflictError,
    PaperBetLedger,
    parse_paper_bet_entry_payload,
)
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactStore

MATCH = MatchId("match:paper-ledger")
KICKOFF = datetime(2025, 8, 17, 12, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(hours=24)

_FIXTURE_TEMPLATE = None
_FIXTURE_TEMPLATE_OWNER = None


class _MissingModelRunRegistry:
    def load_model_run(self, model_run_id: str):
        raise FileNotFoundError(model_run_id)


class _InMemoryMarketSnapshotRegistry:
    """Test authority that replays raw lineage before returning exact snapshots."""

    def __init__(self, raw: RawArchive, *snapshots: MarketSnapshot) -> None:
        self.raw = raw
        self.snapshots = {snapshot.id.value: snapshot for snapshot in snapshots}

    def load_verified_market_snapshot(self, snapshot_id: str) -> MarketSnapshot:
        try:
            snapshot = self.snapshots[snapshot_id]
        except KeyError as error:
            raise FileNotFoundError(snapshot_id) from error
        verify_market_snapshot(snapshot, source_validator=self.raw)
        return snapshot


class _InMemoryPredictionContextRegistry:
    """Test authority that validates one exact prediction/snapshot registration."""

    def __init__(
        self,
        prediction: ScorePrediction,
        snapshot: PreMatchSnapshot,
        derived: DerivedArchive,
    ) -> None:
        verify_current_snapshot(snapshot, source_validator=derived)
        verify_prediction_snapshot(
            prediction,
            snapshot=snapshot,
            snapshot_validator=derived,
        )
        self.prediction = prediction
        self.snapshot = snapshot

    def load_verified_prediction_context(
        self,
        reference: str,
    ) -> tuple[ScorePrediction, PreMatchSnapshot]:
        if reference != self.prediction.id.value:
            raise FileNotFoundError(reference)
        return self.prediction, self.snapshot


def _market_registry(
    raw: RawArchive, *snapshots: MarketSnapshot
) -> _InMemoryMarketSnapshotRegistry:
    return _InMemoryMarketSnapshotRegistry(raw, *snapshots)


def _legacy_market_snapshot(snapshot: MarketSnapshot) -> MarketSnapshot:
    identity = {
        "schema_version": 1,
        "match_id": snapshot.match_id.value,
        "source": snapshot.source,
        "market_type": snapshot.market_type,
        "status": snapshot.status.value,
        "data_kind": snapshot.data_kind.value,
        "observed_at": snapshot.observed_at.isoformat().replace("+00:00", "Z"),
        "quotes": [
            {"outcome": quote.outcome, "decimal_odds": quote.decimal_odds}
            for quote in snapshot.quotes
        ],
        "raw_asset_ref": snapshot.raw_asset_ref,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return MarketSnapshot(
        id=MarketSnapshotId(f"market-snapshot:{digest}"),
        schema_version=1,
        match_id=snapshot.match_id,
        source=snapshot.source,
        market_type=snapshot.market_type,
        status=snapshot.status,
        data_kind=snapshot.data_kind,
        observed_at=snapshot.observed_at,
        quotes=snapshot.quotes,
        raw_asset_ref=snapshot.raw_asset_ref,
    )


def _rehash_entry_payload(payload: dict[str, object]) -> str:
    identity = dict(payload)
    identity.pop("id", None)
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"paper-bet-entry:{digest}"


def _replace_entry_file(
    ledger: PaperBetLedger,
    original_entry_id: str,
    payload: dict[str, object],
) -> str:
    forged_entry_id = _rehash_entry_payload(payload)
    payload["id"] = forged_entry_id
    original_path = ledger.entry_path(original_entry_id)
    forged_path = ledger.entry_path(forged_entry_id)
    forged_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.replace(forged_path)
    forged_path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return forged_entry_id


def _fixture(tmp_path: Path):
    global _FIXTURE_TEMPLATE, _FIXTURE_TEMPLATE_OWNER
    if _FIXTURE_TEMPLATE is None:
        _FIXTURE_TEMPLATE_OWNER = tempfile.TemporaryDirectory(prefix="paper-ledger-fixture-")
        template_root = Path(_FIXTURE_TEMPLATE_OWNER.name)
        layout, _raw, prediction, market, result, prediction_context = _build_fixture(template_root)
        _FIXTURE_TEMPLATE = (layout, prediction, market, result, prediction_context)

    template_layout, prediction, market, result, prediction_context = _FIXTURE_TEMPLATE
    layout = DataLayout(tmp_path / "data")
    shutil.copytree(template_layout.root, layout.root)
    return layout, RawArchive(layout), prediction, market, result, prediction_context


def _build_fixture(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    formal = seed_formal_context(
        layout,
        as_of=AS_OF,
        kickoff=KICKOFF,
        key="paper-ledger",
        observed_at=KICKOFF + timedelta(days=1),
    )
    raw = formal.raw
    derived = formal.derived
    canonical = formal.canonical
    settlement_asset = raw.archive(
        b'{"home_goals":2,"away_goals":1}',
        source="test-source",
        source_id="paper-ledger-settlement",
        url="fixture://paper-ledger-settlement",
        observed_at=KICKOFF + timedelta(days=1),
        target_event_time=KICKOFF,
        collector_version="test/1",
        media_type="application/json",
    )
    canonical.register_raw_asset(settlement_asset)
    settlement_fact = CanonicalFactStore(canonical).append_result_90(
        match_id=formal.match_id,
        match_version=formal.match_version,
        home_goals=2,
        away_goals=1,
        known_at=KICKOFF,
        observed_at=settlement_asset.observed_at,
        raw_asset_id=settlement_asset.id,
    )
    settlement_result = load_verified_match_result(
        settlement_fact.record_id,
        archive=raw,
        canonical=canonical,
    )
    canonical_match = canonical.match(formal.match_id)
    evidence = raw.archive(
        b"paper ledger baseline evidence",
        source="test-source",
        source_id="paper-ledger-baseline",
        url="fixture://paper-ledger-baseline",
        observed_at=AS_OF - timedelta(days=1),
        target_event_time=AS_OF - timedelta(days=1),
        collector_version="test/1",
        media_type="application/octet-stream",
    )
    baseline = build_team_baseline(
        (
            TeamMatchProcess(
                "paper-ledger-match",
                canonical_match.home_team_id.value,
                canonical_match.away_team_id.value,
                AS_OF - timedelta(days=30),
                AS_OF - timedelta(days=1),
                1.7,
                0.8,
                evidence.id.value,
            ),
        ),
        as_of=AS_OF,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline)
    baseline_ref = derived.write_team_baseline_source(
        match_id=formal.match_id,
        match_version=formal.match_version,
        as_of=AS_OF,
        baseline_artifact_id=baseline.artifact_id,
    )
    baseline_validation = derived.validate_snapshot_source(baseline_ref)
    baseline_value = baseline_validation.value
    context_value = formal.context_value
    context_ref = formal.context_ref
    snapshot = build_snapshot(
        match_id=formal.match_id,
        match_version=formal.match_version,
        snapshot_type=SnapshotType.T24H,
        as_of=AS_OF,
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="prematch-features/3",
        features=(
            SnapshotFeature(
                "team_baseline",
                baseline_value,
                baseline_validation.known_at,
                baseline_ref,
                "team-baseline",
            ),
            SnapshotFeature(
                "match_context",
                context_value,
                formal.context_known_at,
                context_ref,
                "match-context",
            ),
        ),
        home_team_id=formal.home_team_id,
        away_team_id=formal.away_team_id,
        source_validator=derived,
    )
    derived.write_snapshot(snapshot)

    training = TrainingArtifactStore(layout)
    qualification_observed_at = AS_OF - timedelta(hours=2)
    qualification_evaluated_at = AS_OF - timedelta(hours=1)

    def formal_sample(
        *,
        key: str,
        kickoff: datetime,
        team_indices: tuple[int, int, int, int],
        goals: tuple[int, int],
        split: str,
    ):
        sample_as_of = kickoff - timedelta(hours=24)
        sample_formal = seed_formal_context(
            layout,
            as_of=sample_as_of,
            kickoff=kickoff,
            key=key,
            observed_at=qualification_observed_at,
            team_indices=team_indices,
        )
        sample_evidence = raw.archive(
            f"{key} baseline evidence".encode(),
            source="test-source",
            source_id=f"{key}-baseline",
            url=f"fixture://{key}-baseline",
            observed_at=qualification_observed_at,
            target_event_time=sample_as_of - timedelta(days=1),
            collector_version="test/1",
            media_type="application/octet-stream",
        )
        sample_baseline = build_team_baseline(
            (
                TeamMatchProcess(
                    f"{key}-baseline-match",
                    sample_formal.home_team_id.value,
                    sample_formal.away_team_id.value,
                    sample_as_of - timedelta(days=30),
                    sample_as_of - timedelta(days=1),
                    1.4,
                    0.9,
                    sample_evidence.id.value,
                ),
            ),
            as_of=sample_as_of,
            half_life_days=90.0,
            iterations=2,
        ).artifact
        derived.write_team_baseline(
            sample_baseline,
            generated_at=qualification_observed_at,
        )
        sample_baseline_ref = derived.write_team_baseline_source(
            match_id=sample_formal.match_id,
            match_version=sample_formal.match_version,
            as_of=sample_as_of,
            baseline_artifact_id=sample_baseline.artifact_id,
        )
        sample_baseline_validation = derived.validate_snapshot_source(sample_baseline_ref)
        sample_snapshot = build_snapshot(
            match_id=sample_formal.match_id,
            match_version=sample_formal.match_version,
            snapshot_type=SnapshotType.T24H,
            as_of=sample_as_of,
            scheduled_kickoff_used=kickoff,
            feature_spec_version="prematch-features/3",
            features=(
                SnapshotFeature(
                    "team_baseline",
                    sample_baseline_validation.value,
                    sample_baseline_validation.known_at,
                    sample_baseline_ref,
                    "team-baseline",
                ),
                SnapshotFeature(
                    "match_context",
                    sample_formal.context_value,
                    sample_formal.context_known_at,
                    sample_formal.context_ref,
                    "match-context",
                ),
            ),
            home_team_id=sample_formal.home_team_id,
            away_team_id=sample_formal.away_team_id,
            source_validator=derived,
        )
        derived.write_snapshot(sample_snapshot)
        result_asset = raw.archive(
            json.dumps(
                {"home_goals": goals[0], "away_goals": goals[1]},
                separators=(",", ":"),
            ).encode(),
            source="test-source",
            source_id=f"{key}-result",
            url=f"fixture://{key}-result",
            observed_at=qualification_observed_at,
            target_event_time=kickoff,
            collector_version="test/1",
            media_type="application/json",
        )
        canonical.register_raw_asset(result_asset)
        result_fact = CanonicalFactStore(canonical).append_result_90(
            match_id=sample_formal.match_id,
            match_version=sample_formal.match_version,
            home_goals=goals[0],
            away_goals=goals[1],
            known_at=kickoff,
            observed_at=qualification_observed_at,
            raw_asset_id=result_asset.id,
        )
        qualification = training.create_training_qualification(
            match_id=sample_formal.match_id,
            match_version=sample_formal.match_version,
            qualification=Qualification.SCORE_MODEL,
            ruleset_version="readiness/1",
            evaluated_at=qualification_evaluated_at,
            snapshot_ref=sample_snapshot.id.value,
            result_ref=result_fact.record_id,
        )
        return training.build_formal_score_sample(
            sample_id=f"sample:{key}",
            qualification_ref=qualification.qualification_id,
            feature_version="snapshot-score-features/1",
            split=split,
        )

    train_sample = formal_sample(
        key="paper-ledger-training",
        kickoff=AS_OF - timedelta(days=20),
        team_indices=(4, 5, 6, 7),
        goals=(1, 0),
        split="train",
    )
    holdout_sample = formal_sample(
        key="paper-ledger-holdout",
        kickoff=AS_OF - timedelta(days=10),
        team_indices=(8, 9, 10, 11),
        goals=(2, 1),
        split="holdout",
    )
    dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/paper-ledger",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="snapshot-score-features/1",
        label_version="result-90/1",
        as_of=holdout_sample.as_of,
        split_strategy="forward-chaining/1",
        samples=(train_sample, holdout_sample),
        generated_at=qualification_evaluated_at,
        code_version="git:test",
    )
    training.write_formal_dataset(dataset)
    model_ref = training.write_model_artifact(b"paper-ledger-model-v1")
    output_ref = training.write_model_output(b"paper-ledger-output-v1")
    model_run = ModelRunArtifact.create(
        model_version="dixon-coles/paper-ledger",
        run_role="research",
        task="score-model",
        dataset_id=dataset.dataset_id,
        feature_version=dataset.feature_version,
        label_version=dataset.label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:test",
        environment_version="python:test-lock",
        started_at=AS_OF,
        ended_at=AS_OF,
        random_seed=17,
        model_artifact_refs=(model_ref,),
        evaluation_cohort=(holdout_sample.sample_id,),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_ref,),
        output_hashes=(output_ref.removeprefix("model-output:"),),
    )
    training.write_model_run(model_run)
    derived.model_run_validator = training
    context_feature = next(
        feature for feature in snapshot.features if feature.name == "match_context"
    )
    composition = compose_expected_goals(
        float(baseline_value["lambda_home"]),
        float(baseline_value["lambda_away"]),
        context_contributions(
            context_feature.value,
            home_team_id=snapshot.home_team_id.value,
            away_team_id=snapshot.away_team_id.value,
            source_ref=context_feature.source_ref,
            source_validator=derived,
        ),
    )
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=derived,
        model_run_id=ModelRunId(model_run.model_run_id),
        model_version="dixon-coles/paper-ledger",
        generated_at=AS_OF,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.1,
        max_goals=11,
        input_refs=(),
        expected_goals=composition,
    )
    derived.write_prediction(prediction, snapshot=snapshot)
    market_asset = raw.archive(
        b'{"market":"result_90"}',
        source="bookmaker",
        source_id="paper-ledger-market",
        url="https://bookmaker.example/paper-ledger",
        observed_at=AS_OF - timedelta(minutes=1),
        target_event_time=KICKOFF,
        collector_version="market/1",
        media_type="application/json",
    )
    market = build_market_snapshot(
        match_id=settlement_result.match_id,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=(
            MarketQuote("home", 2.0),
            MarketQuote("draw", 3.5),
            MarketQuote("away", 4.0),
        ),
        raw_asset=market_asset,
    )
    prediction_context = _InMemoryPredictionContextRegistry(
        prediction,
        snapshot,
        derived,
    )
    return layout, raw, prediction, market, settlement_result, prediction_context


def test_settlement_rules_compute_all_supported_outcomes() -> None:
    def result(home: int, away: int) -> MatchResult90:
        return MatchResult90(MATCH, home, away, AS_OF + timedelta(days=1), "canonical:result")

    win = compute_settlement(
        SettlementRules(selection="home"),
        result=result(2, 1),
        stake=10,
        decimal_odds=2.5,
    )
    assert win.outcome is SettlementOutcome.WIN
    assert win.payout == pytest.approx(25.0)
    assert win.profit == pytest.approx(15.0)
    loss = compute_settlement(
        SettlementRules(selection="home"),
        result=result(0, 1),
        stake=10,
        decimal_odds=2.5,
    )
    assert loss.outcome is SettlementOutcome.LOSS
    assert loss.payout == 0
    assert (
        compute_settlement(
            SettlementRules(selection="home"),
            result=result(1, 1),
            stake=10,
            decimal_odds=2.5,
        ).outcome
        is SettlementOutcome.LOSS
    )
    push = compute_settlement(
        SettlementRules(market_type="asian_handicap", selection="home", line=0.0),
        result=result(1, 1),
        stake=10,
        decimal_odds=2.0,
    )
    assert push.outcome is SettlementOutcome.PUSH
    assert push.payout == 10
    half_loss = compute_settlement(
        SettlementRules(market_type="total_goals", selection="over", line=2.25),
        result=result(1, 1),
        stake=10,
        decimal_odds=2.0,
    )
    assert half_loss.outcome is SettlementOutcome.HALF_LOSS
    assert half_loss.payout == 5
    half_win = compute_settlement(
        SettlementRules(market_type="total_goals", selection="under", line=2.25),
        result=result(1, 1),
        stake=10,
        decimal_odds=2.0,
    )
    assert half_win.outcome is SettlementOutcome.HALF_WIN
    assert half_win.payout == 15
    assert (
        compute_settlement(
            SettlementRules(market_type="void", selection="home"),
            result=None,
            stake=10,
            decimal_odds=2.0,
        ).payout
        == 10
    )


def test_default_ledger_records_self_reported_real_market_as_rejected(
    tmp_path: Path,
) -> None:
    layout, _raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        prediction_context_validator=prediction_context,
    )

    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=market,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )

    assert entry.decision is CandidateDecision.REJECTED
    assert "authoritative market snapshot validator" in entry.decision_reason
    assert ledger.entries() == (entry,)
    assert ledger.recompute(initial_bankroll=100).balance == 100


def test_injected_derived_archive_must_share_the_market_authority(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path / "data")
    authority = _market_registry(RawArchive(layout))
    prediction_context = TrainingArtifactStore(layout)

    with pytest.raises(ValueError, match="same market snapshot validator"):
        PaperBetLedger(
            layout,
            market_snapshot_validator=authority,
            prediction_context_validator=prediction_context,
            derived_archive=DerivedArchive(
                layout,
                prediction_context_validator=prediction_context,
            ),
        )

    matching_archive = DerivedArchive(
        layout,
        market_snapshot_validator=authority,
        prediction_context_validator=prediction_context,
    )
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=authority,
        prediction_context_validator=prediction_context,
        derived_archive=matching_archive,
    )
    assert ledger.derived is matching_archive

    with pytest.raises(ValueError, match="same prediction context validator"):
        PaperBetLedger(
            layout,
            market_snapshot_validator=authority,
            prediction_context_validator=TrainingArtifactStore(layout),
            derived_archive=matching_archive,
        )


def test_raw_archive_cannot_stand_in_for_market_snapshot_authority(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=raw,  # type: ignore[arg-type]
        prediction_context_validator=prediction_context,
    )

    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=market,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )

    assert entry.decision is CandidateDecision.REJECTED
    assert "authoritative semantic lookup" in entry.decision_reason
    assert ledger.recompute().accepted_entries == 0


def test_authority_requires_exact_market_snapshot_semantics(tmp_path: Path) -> None:
    _layout, raw, prediction, market, _result, _prediction_context = _fixture(tmp_path)
    registry = _market_registry(raw, market)
    other_asset = raw.archive(
        b'{"market":"other"}',
        source="bookmaker",
        source_id="paper-ledger-other-market",
        url="https://bookmaker.example/paper-ledger-other",
        observed_at=market.observed_at,
        target_event_time=KICKOFF,
        collector_version="market/1",
        media_type="application/json",
    )
    for forged in (
        replace(market, match_id=MatchId("match:other")),
        replace(market, market_type="1x2"),
        replace(market, status=MarketStatus.SUSPENDED),
        replace(market, data_kind=MarketDataKind.SYNTHETIC),
        replace(
            market,
            quotes=(MarketQuote("home", 9.0), *market.quotes[1:]),
        ),
        replace(market, raw_asset_ref=other_asset.id.value),
    ):
        with pytest.raises(ValueError, match="authoritative record"):
            PaperBetEntry.accepted(
                prediction=prediction,
                market_snapshot=forged,
                market_snapshot_validator=registry,
                selection="home",
                risk_config=RiskConfig(),
                stake=10,
                match_exposure=10,
                daily_exposure=10,
                placed_at=AS_OF,
                settlement_rules=SettlementRules(selection="home"),
            )


def test_market_snapshot_v2_line_contract_and_v1_audit_readability(tmp_path: Path) -> None:
    raw = RawArchive(DataLayout(tmp_path / "data"))
    asset = raw.archive(
        b'{"market":"result_90"}',
        source="bookmaker",
        source_id="market-v2-contract",
        url="https://bookmaker.example/market-v2-contract",
        observed_at=AS_OF - timedelta(minutes=1),
        target_event_time=KICKOFF,
        collector_version="market/1",
        media_type="application/json",
    )
    market = build_market_snapshot(
        match_id=MATCH,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=(
            MarketQuote("home", 2.0),
            MarketQuote("draw", 3.5),
            MarketQuote("away", 4.0),
        ),
        raw_asset=asset,
    )

    assert market.schema_version == 2
    assert market.period == "90m"
    assert market.line is None
    with pytest.raises(ValueError, match="cannot contain a line"):
        replace(market, line=0.5)
    with pytest.raises(ValueError, match="result settlement rules cannot contain a line"):
        SettlementRules(selection="home", line=0.5)
    with pytest.raises(ValueError, match="period='90m'"):
        build_market_snapshot(
            match_id=MATCH,
            market_type="result_90",
            status=MarketStatus.OPEN,
            data_kind=MarketDataKind.REAL,
            quotes=market.quotes,
            raw_asset=asset,
            period="full-time",
        )
    with pytest.raises(ValueError, match="finite, reasonable line"):
        build_market_snapshot(
            match_id=MATCH,
            market_type="total_goals",
            status=MarketStatus.OPEN,
            data_kind=MarketDataKind.REAL,
            quotes=(MarketQuote("over", 2.0), MarketQuote("under", 2.0)),
            raw_asset=asset,
        )
    for market_type, quotes, selection, line in (
        (
            "total_goals",
            (MarketQuote("over", 2.0), MarketQuote("under", 2.0)),
            "over",
            2.5,
        ),
        (
            "asian_handicap",
            (MarketQuote("home", 1.95), MarketQuote("away", 1.95)),
            "home",
            -0.5,
        ),
    ):
        line_market = build_market_snapshot(
            match_id=MATCH,
            market_type=market_type,
            status=MarketStatus.OPEN,
            data_kind=MarketDataKind.REAL,
            quotes=quotes,
            raw_asset=asset,
            line=line,
        )
        _validate_market_gate(
            prediction=None,
            market_snapshot=line_market,
            selection=selection,
            settlement_rules=SettlementRules(
                market_type=market_type,
                selection=selection,
                line=line,
            ),
            placed_at=AS_OF,
            prediction_snapshot_as_of=AS_OF,
        )
        with pytest.raises(ValueError, match="line does not match"):
            _validate_market_gate(
                prediction=None,
                market_snapshot=line_market,
                selection=selection,
                settlement_rules=SettlementRules(
                    market_type=market_type,
                    selection=selection,
                    line=line + 0.5,
                ),
                placed_at=AS_OF,
                prediction_snapshot_as_of=AS_OF,
            )

    legacy = _legacy_market_snapshot(market)
    verify_market_snapshot(legacy, source_validator=raw)
    with pytest.raises(ValueError, match="current market snapshot"):
        _validate_market_gate(
            prediction=None,
            market_snapshot=legacy,
            selection="home",
            placed_at=AS_OF,
            settlement_rules=SettlementRules(selection="home"),
            prediction_snapshot_as_of=AS_OF,
        )


def test_invalid_market_is_recorded_as_zero_stake_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    asset = raw.load(RawAssetId(market.raw_asset_ref))
    synthetic = build_market_snapshot(
        match_id=prediction.match_id,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.SYNTHETIC,
        quotes=market.quotes,
        raw_asset=asset,
    )
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, synthetic),
        prediction_context_validator=prediction_context,
    )
    arguments = {
        "prediction": prediction,
        "market_snapshot": synthetic,
        "selection": "home",
        "risk_config": RiskConfig(),
        "stake": 10.0,
        "match_exposure": 10.0,
        "daily_exposure": 10.0,
        "placed_at": AS_OF,
        "settlement_rules": SettlementRules(selection="home"),
    }

    first = ledger.append_candidate(**arguments)
    second = ledger.append_candidate(**arguments)

    assert first.decision is CandidateDecision.REJECTED
    assert first.stake == 0
    assert "real market" in first.decision_reason
    assert second.entry_id == first.entry_id
    assert len(ledger.entries()) == 1
    assert ledger.recompute().open_stake == 0


@pytest.mark.parametrize("invalid_kind", ("future", "suspended", "incomplete"))
def test_future_non_open_or_incomplete_market_never_creates_a_position(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    asset = raw.load(RawAssetId(market.raw_asset_ref))
    status = MarketStatus.OPEN
    quotes = market.quotes
    if invalid_kind == "future":
        asset = raw.archive(
            b'{"market":"future"}',
            source="bookmaker",
            source_id="paper-ledger-future-market",
            url="https://bookmaker.example/paper-ledger-future",
            observed_at=AS_OF + timedelta(minutes=1),
            target_event_time=KICKOFF,
            collector_version="market/1",
            media_type="application/json",
        )
    elif invalid_kind == "suspended":
        status = MarketStatus.SUSPENDED
    else:
        quotes = market.quotes[:2]
    invalid_market = build_market_snapshot(
        match_id=prediction.match_id,
        market_type="result_90",
        status=status,
        data_kind=MarketDataKind.REAL,
        quotes=quotes,
        raw_asset=asset,
    )
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, invalid_market),
        prediction_context_validator=prediction_context,
    )

    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=invalid_market,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
        candidate_id=f"{invalid_kind}-candidate",
    )

    assert entry.decision is CandidateDecision.REJECTED
    assert entry.stake == entry.match_exposure == entry.daily_exposure == 0
    assert ledger.recompute(initial_bankroll=100).balance == 100


@pytest.mark.parametrize(
    ("market_type", "quotes", "selection", "line"),
    (
        (
            "total_goals",
            (MarketQuote("over", 2.0), MarketQuote("under", 2.0)),
            "over",
            2.5,
        ),
        (
            "asian_handicap",
            (MarketQuote("home", 1.95), MarketQuote("away", 1.95)),
            "home",
            -0.5,
        ),
    ),
)
def test_line_markets_require_the_exact_snapshot_settlement_line(
    tmp_path: Path,
    market_type: str,
    quotes: tuple[MarketQuote, ...],
    selection: str,
    line: float,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    line_market = build_market_snapshot(
        match_id=prediction.match_id,
        market_type=market_type,
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=quotes,
        raw_asset=raw.load(RawAssetId(market.raw_asset_ref)),
        line=line,
    )
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, line_market),
        prediction_context_validator=prediction_context,
    )
    accepted = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=line_market,
        selection=selection,
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(
            market_type=market_type,
            selection=selection,
            line=line,
        ),
        candidate_id=f"{market_type}-exact-line",
    )
    mismatched = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=line_market,
        selection=selection,
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=20,
        daily_exposure=20,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(
            market_type=market_type,
            selection=selection,
            line=line + 0.5,
        ),
        candidate_id=f"{market_type}-changed-line",
    )

    assert accepted.decision is CandidateDecision.ACCEPTED
    assert mismatched.decision is CandidateDecision.REJECTED
    assert "line does not match" in mismatched.decision_reason


def test_rehashed_market_line_tamper_is_rejected_on_load_and_recompute(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    asset = raw.load(RawAssetId(market.raw_asset_ref))
    total_market = build_market_snapshot(
        match_id=prediction.match_id,
        market_type="total_goals",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=(MarketQuote("over", 2.0), MarketQuote("under", 2.0)),
        raw_asset=asset,
        line=2.5,
    )
    market_registry = _market_registry(raw, total_market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=total_market,
        selection="over",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(
            market_type="total_goals",
            selection="over",
            line=2.5,
        ),
    )
    assert entry.decision is CandidateDecision.ACCEPTED

    forged_market = build_market_snapshot(
        match_id=prediction.match_id,
        market_type="total_goals",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=total_market.quotes,
        raw_asset=asset,
        line=3.0,
    )
    market_registry.snapshots[forged_market.id.value] = forged_market
    payload = json.loads(ledger.entry_path(entry.entry_id).read_text(encoding="utf-8"))
    payload["market_snapshot"]["id"] = forged_market.id.value
    payload["market_snapshot"]["line"] = forged_market.line
    payload["market_snapshot_id"] = forged_market.id.value
    forged_entry_id = _replace_entry_file(ledger, entry.entry_id, payload)
    forged_entry = parse_paper_bet_entry_payload(payload)

    with pytest.raises(ValueError, match="line does not match"):
        ledger.append(forged_entry)
    with pytest.raises(ValueError, match="line does not match"):
        ledger.load(forged_entry_id)
    with pytest.raises(ValueError, match="line does not match"):
        ledger.entries()
    with pytest.raises(ValueError, match="line does not match"):
        ledger.recompute()


def test_exposure_is_recomputed_and_cannot_be_underreported(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, market),
        prediction_context_validator=prediction_context,
    )
    common = {
        "prediction": prediction,
        "market_snapshot": market,
        "selection": "home",
        "risk_config": RiskConfig(max_match_exposure=25, max_daily_exposure=25),
        "stake": 10,
        "placed_at": AS_OF,
        "settlement_rules": SettlementRules(selection="home"),
    }
    first = ledger.append_candidate(
        **common,
        match_exposure=10,
        daily_exposure=10,
        candidate_id="first-candidate",
    )
    assert first.decision is CandidateDecision.ACCEPTED
    underreported = ledger.append_candidate(
        **common,
        match_exposure=10,
        daily_exposure=10,
        candidate_id="second-candidate",
    )
    assert underreported.decision is CandidateDecision.REJECTED
    assert "ledger-computed exposure" in underreported.decision_reason
    correct = ledger.append_candidate(
        **common,
        match_exposure=20,
        daily_exposure=20,
        candidate_id="third-candidate",
    )
    assert correct.decision is CandidateDecision.ACCEPTED
    assert ledger.recompute().match_exposure_map() == {prediction.match_id.value: 20}


@pytest.mark.parametrize("placed_at", (KICKOFF, KICKOFF + timedelta(seconds=1)))
def test_candidate_at_or_after_kickoff_is_recorded_only_as_rejected(
    tmp_path: Path,
    placed_at: datetime,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, market),
        prediction_context_validator=prediction_context,
    )

    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=market,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=placed_at,
        settlement_rules=SettlementRules(selection="home"),
        candidate_id=f"kickoff-{placed_at.isoformat()}",
    )

    assert entry.decision is CandidateDecision.REJECTED
    assert "placed before kickoff" in entry.decision_reason
    assert ledger.recompute().accepted_entries == 0


def test_rehashed_kickoff_placement_is_rejected_on_load_and_recompute(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=_market_registry(raw, market),
        prediction_context_validator=prediction_context,
    )
    entry = ledger.append_candidate(
        prediction=prediction,
        market_snapshot=market,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    assert entry.decision is CandidateDecision.ACCEPTED

    payload = json.loads(ledger.entry_path(entry.entry_id).read_text(encoding="utf-8"))
    payload["placed_at"] = KICKOFF.isoformat().replace("+00:00", "Z")
    forged_entry_id = _replace_entry_file(ledger, entry.entry_id, payload)
    forged_entry = parse_paper_bet_entry_payload(payload)

    with pytest.raises(ValueError, match="placed before kickoff"):
        ledger.append(forged_entry)
    with pytest.raises(ValueError, match="placed before kickoff"):
        ledger.load(forged_entry_id)
    with pytest.raises(ValueError, match="placed before kickoff"):
        ledger.entries()
    with pytest.raises(ValueError, match="placed before kickoff"):
        ledger.recompute()


def test_rehashed_kickoff_prediction_is_rejected_by_all_accepted_entry_paths(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result, _prediction_context = _fixture(tmp_path)
    forged_ref = forge_persisted_prediction(
        layout,
        prediction.id.value,
        lambda payload: payload.__setitem__(
            "generated_at",
            KICKOFF.isoformat().replace("+00:00", "Z"),
        ),
    )
    market_registry = _market_registry(raw, market)
    valid_entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    forged_payload = paper_bet_entry_payload(
        replace(
            valid_entry,
            prediction_id=PredictionId(forged_ref),
            prediction_generated_at=KICKOFF,
            placed_at=KICKOFF + timedelta(seconds=1),
        )
    )
    forged_payload["id"] = _rehash_entry_payload(forged_payload)
    forged_entry = parse_paper_bet_entry_payload(forged_payload)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
    )

    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.append(forged_entry)
    forged_path = ledger.entry_path(forged_entry.entry_id)
    forged_path.parent.mkdir(parents=True, exist_ok=True)
    forged_path.write_text(
        json.dumps(
            forged_payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.load(forged_entry.entry_id)
    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.entries()
    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.recompute()


def test_accepted_entry_settlement_is_an_append_only_revision_and_recomputable(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    path = ledger.append(entry)
    assert ledger.append(entry) == path
    assert ledger.load(entry.entry_id) == entry
    open_summary = ledger.recompute(initial_bankroll=100)
    assert open_summary.balance == 90
    assert open_summary.open_stake == 10
    assert open_summary.match_exposure_map() == {prediction.match_id.value: 10}

    settled = ledger.settle(
        entry,
        result,
        settled_at=AS_OF + timedelta(days=1, hours=1),
    )
    assert settled.revision == 2
    assert settled.prior_entry_id == entry.entry_id
    assert ledger.load(entry.entry_id) == entry
    summary = ledger.recompute(initial_bankroll=100)
    assert summary.total_entries == 1
    assert summary.settled_entries == 1
    assert summary.total_staked == 10
    assert summary.total_payout == 20
    assert summary.balance == 110
    assert summary.open_stake == 0
    latest = json.loads((layout.paper_ledger / "latest.json").read_text(encoding="utf-8"))
    derived = DerivedArchive(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    artifact = derived.load_artifact_manifest(latest["artifact_id"])
    run = derived.load_run_manifest(latest["run_id"])
    assert artifact.artifact_type == "paper-ledger-summary"
    assert artifact.artifact_id in run.output_refs
    assert any(ref.startswith("file-sha256:") for ref in artifact.output_refs)

    with pytest.raises(ValueError, match="payout"):
        replace(settled, payout=999.0)

    payload_path = ledger.entry_path(settled.entry_id)
    payload_path.write_text(
        payload_path.read_text(encoding="utf-8").replace('"payout": 20.0', '"payout": 999.0')
    )
    with pytest.raises((LedgerConflictError, ValueError)):
        ledger.load(settled.entry_id)


def test_accepted_entry_operations_fail_when_market_raw_lineage_is_unverifiable(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    ledger.append(entry)
    default_ledger = PaperBetLedger(
        layout,
        prediction_context_validator=prediction_context,
    )
    for operation in (
        lambda: default_ledger.load(entry.entry_id),
        default_ledger.entries,
        default_ledger.recompute,
    ):
        with pytest.raises(
            ValueError,
            match="authoritative market snapshot validator",
        ):
            operation()
    raw_only_ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=raw,  # type: ignore[arg-type]
        prediction_context_validator=prediction_context,
    )
    with pytest.raises(ValueError, match="authoritative semantic lookup"):
        raw_only_ledger.load(entry.entry_id)

    asset = raw.load(RawAssetId(market.raw_asset_ref))
    layout.raw_object_path(asset.checksum).unlink()

    for operation in (
        lambda: ledger.append(entry),
        lambda: ledger.load(entry.entry_id),
        ledger.entries,
        ledger.recompute,
    ):
        with pytest.raises(ValueError, match="authoritative market snapshot"):
            operation()


def test_retry_repairs_entry_when_derived_registration_failed_after_append(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    original_write_summary = PaperBetLedger._write_derived_summary
    calls = 0

    def flaky_write_summary(current: PaperBetLedger):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LedgerConflictError("transient registry failure")
        return original_write_summary(current)

    with patch.object(PaperBetLedger, "_write_derived_summary", flaky_write_summary):
        with pytest.raises(LedgerConflictError, match="transient registry failure"):
            ledger.append(entry)
        assert ledger.entry_path(entry.entry_id).is_file()
        ledger.append(entry)
    assert (layout.paper_ledger / "latest.json").is_file()


def test_default_ledger_requires_canonical_result_evidence_for_settlement_and_load(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
        result_validator=None,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    ledger.append(entry)

    tampers = (
        replace(result, source_ref="canonical:forged"),
        replace(result, home_goals=result.home_goals + 1),
        replace(result, known_at=result.known_at + timedelta(seconds=1)),
    )
    for tampered in tampers:
        with pytest.raises(ValueError, match="result source evidence"):
            ledger.settle(
                entry,
                tampered,
                settled_at=AS_OF + timedelta(days=1, hours=1),
            )

    settled = ledger.settle(
        entry,
        result,
        settled_at=AS_OF + timedelta(days=1, hours=1),
    )
    assert ledger.load(settled.entry_id) == settled

    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT raw_asset_id FROM match_results_90 WHERE record_id = ?",
            (result.source_ref,),
        ).fetchone()
    assert row is not None
    result_asset = raw.load(RawAssetId(row["raw_asset_id"]))
    layout.raw_object_path(result_asset.checksum).unlink()
    with pytest.raises(ValueError, match="result source evidence"):
        PaperBetLedger(
            layout,
            market_snapshot_validator=market_registry,
            prediction_context_validator=prediction_context,
        ).load(settled.entry_id)


def test_void_settlement_waives_result_but_not_market_authority(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home", void_if_no_result=True),
    )
    ledger.append(entry)

    settled = ledger.settle(
        entry,
        None,
        settled_at=AS_OF + timedelta(days=1, hours=1),
    )

    assert settled.result is None
    assert settled.settlement_outcome is SettlementOutcome.VOID
    assert ledger.load(settled.entry_id) == settled
    with pytest.raises(
        ValueError,
        match="authoritative market snapshot validator",
    ):
        PaperBetLedger(
            layout,
            prediction_context_validator=prediction_context,
        ).load(settled.entry_id)


def test_ledger_rejects_entries_that_reference_an_unavailable_model_run(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result, prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    accepted_arguments = {
        "prediction": prediction,
        "market_snapshot": market,
        "market_snapshot_validator": market_registry,
        "selection": "home",
        "risk_config": RiskConfig(),
        "stake": 10,
        "match_exposure": 10,
        "daily_exposure": 10,
        "placed_at": AS_OF,
        "settlement_rules": SettlementRules(selection="home"),
    }
    entry = PaperBetEntry.accepted(**accepted_arguments)
    missing = _MissingModelRunRegistry()
    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        PaperBetEntry.accepted(
            **accepted_arguments,
            model_run_validator=missing,
        )
    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        PaperBetEntry.rejected(
            prediction=prediction,
            reason="no market",
            placed_at=AS_OF,
            model_run_validator=missing,
        )
    ledger = PaperBetLedger(
        layout,
        market_snapshot_validator=market_registry,
        prediction_context_validator=prediction_context,
        model_run_validator=missing,
    )

    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        ledger.append(entry)


def test_default_ledger_rejects_missing_prediction_or_model_bytes(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, _prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    ledger = PaperBetLedger(layout, market_snapshot_validator=market_registry)
    prediction_digest = prediction.id.value.removeprefix("prediction:")
    prediction_path = (
        layout.derived / "predictions" / prediction_digest[:2] / f"{prediction_digest}.json"
    )
    prediction_path.unlink()
    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.append(entry)

    layout, raw, prediction, market, _result, _prediction_context = _fixture(tmp_path / "model")
    market_registry = _market_registry(raw, market)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    model_store = TrainingArtifactStore(layout)
    run = model_store.load_model_run(prediction.model_run_id.value)
    model_store.model_artifact_path(run.model_artifact_refs[0]).unlink()
    with pytest.raises(ValueError, match="prediction artifact"):
        PaperBetLedger(
            layout,
            market_snapshot_validator=market_registry,
        ).append(entry)


def test_ledger_rejects_rehashed_semantic_prediction_forgery(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result, _prediction_context = _fixture(tmp_path)
    market_registry = _market_registry(raw, market)
    forged_ref = forge_persisted_prediction(
        layout,
        prediction.id.value,
        lambda payload: payload.__setitem__(
            "baseline_lambda_home", payload["baseline_lambda_home"] * 2
        ),
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_snapshot_validator=market_registry,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    forged_entry = replace(entry, prediction_id=PredictionId(forged_ref))
    identity = paper_bet_entry_payload(forged_entry, include_id=False)
    entry_id = (
        "paper-bet-entry:"
        + hashlib.sha256(
            json.dumps(
                identity,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    forged_entry = replace(forged_entry, entry_id=entry_id)

    with pytest.raises(ValueError, match="prediction artifact"):
        PaperBetLedger(
            layout,
            market_snapshot_validator=market_registry,
        ).append(forged_entry)


def test_ledger_parser_rejects_bool_schema_version(tmp_path: Path) -> None:
    layout, raw, prediction, _market, _result, prediction_context = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        prediction_context_validator=prediction_context,
    )
    entry = PaperBetEntry.rejected(
        prediction=prediction,
        reason="market unavailable",
        placed_at=AS_OF,
    )
    path = ledger.append(entry)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LedgerConflictError, match="integer|canonical"):
        ledger.load(entry.entry_id)
