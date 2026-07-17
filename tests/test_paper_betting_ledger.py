from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from _prediction_forgery import forge_persisted_prediction
from _training_refs import seed_training_references

from football_data_platform.domain.ids import (
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
    compute_settlement,
    paper_bet_entry_payload,
)
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketStatus,
    MatchResult90,
    build_market_snapshot,
    build_score_prediction,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    ModelRunArtifact,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contribution,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    team_baseline_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import load_verified_match_result
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.ledger import LedgerConflictError, PaperBetLedger
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactStore

MATCH = MatchId("match:paper-ledger")
KICKOFF = datetime(2025, 8, 17, 12, tzinfo=UTC)
AS_OF = KICKOFF - timedelta(hours=24)


class _MissingModelRunRegistry:
    def load_model_run(self, model_run_id: str):
        raise FileNotFoundError(model_run_id)


def _fixture(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    derived = DerivedArchive(layout)
    _, settlement_result_ref = seed_training_references(
        layout,
        label_known_at=AS_OF + timedelta(days=1),
        home_goals=2,
        away_goals=1,
        reference_key="paper-ledger-settlement",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    settlement_result = load_verified_match_result(
        settlement_result_ref,
        archive=raw,
        canonical=canonical,
    )
    canonical_match = canonical.match(settlement_result.match_id)
    evidence = raw.archive(
        b"paper ledger baseline evidence",
        source="test-source",
        source_id="paper-ledger-baseline",
        url="fixture://paper-ledger-baseline",
        observed_at=AS_OF - timedelta(days=1),
        target_event_time=KICKOFF,
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
    baseline_value = {
        "artifact_id": baseline.artifact_id,
        "artifact": team_baseline_payload(baseline),
        "lambda_home": 1.7,
        "lambda_away": 0.8,
    }
    baseline_ref = derived.write_snapshot_source(
        value=baseline_value,
        input_refs=(evidence.id,),
        transform_version="team-baseline-input/2",
        generated_at=AS_OF,
    )
    context_value = {"days_since_previous_match": 6.0}
    context_ref = derived.write_snapshot_source(
        value=context_value,
        input_refs=(evidence.id,),
        transform_version="match-context-input/1",
        generated_at=AS_OF,
    )
    snapshot = build_snapshot(
        match_id=settlement_result.match_id,
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        as_of=AS_OF,
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="prematch-features/1",
        features=(
            SnapshotFeature(
                "team_baseline",
                baseline_value,
                AS_OF - timedelta(days=1),
                baseline_ref,
                "team-baseline",
            ),
            SnapshotFeature(
                "match_context",
                context_value,
                AS_OF - timedelta(days=1),
                context_ref,
                "context:rest-days",
            ),
        ),
        home_team_id=canonical_match.home_team_id,
        away_team_id=canonical_match.away_team_id,
        source_validator=derived,
    )
    derived.write_snapshot(snapshot)
    training = TrainingArtifactStore(layout)
    _, training_label_ref = seed_training_references(
        layout,
        label_known_at=AS_OF - timedelta(days=1),
        observed_at=AS_OF,
        home_goals=1,
        away_goals=0,
        reference_key="paper-ledger-training",
    )
    _, holdout_label_ref = seed_training_references(
        layout,
        label_known_at=AS_OF - timedelta(hours=1),
        observed_at=AS_OF,
        home_goals=2,
        away_goals=1,
        reference_key="paper-ledger-holdout",
    )
    train_sample = TrainingSample(
        sample_id="sample:paper-ledger-train",
        as_of=AS_OF - timedelta(days=2),
        feature_known_at=AS_OF - timedelta(days=3),
        label_known_at=AS_OF - timedelta(days=1),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(evidence.id.value,),
        label_ref=training_label_ref,
        features={"home_strength": 1.7, "away_strength": 0.8},
        label={"home_goals": 1, "away_goals": 0},
        split="train",
    )
    holdout_sample = TrainingSample(
        sample_id="sample:paper-ledger-holdout",
        as_of=AS_OF - timedelta(hours=2),
        feature_known_at=AS_OF - timedelta(hours=3),
        label_known_at=AS_OF - timedelta(hours=1),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(evidence.id.value,),
        label_ref=holdout_label_ref,
        features={"home_strength": 1.7, "away_strength": 0.8},
        label={"home_goals": 2, "away_goals": 1},
        split="holdout",
    )
    dataset = TrainingDatasetManifest.create(
        dataset_version="score-dataset/paper-ledger",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=AS_OF,
        split_strategy="forward-chaining/1",
        samples=(train_sample, holdout_sample),
        generated_at=AS_OF,
        code_version="git:test",
    )
    training.write_dataset(dataset)
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
        1.7,
        0.8,
        (context_contribution(context_feature.value, source_ref=context_feature.source_ref),),
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
    return layout, raw, prediction, market, settlement_result


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


def test_invalid_market_is_recorded_as_zero_stake_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    asset = raw.load(RawAssetId(market.raw_asset_ref))
    synthetic = build_market_snapshot(
        match_id=prediction.match_id,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.SYNTHETIC,
        quotes=market.quotes,
        raw_asset=asset,
    )
    ledger = PaperBetLedger(layout, market_source_validator=raw)
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
    layout, raw, prediction, market, _result = _fixture(tmp_path)
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
    ledger = PaperBetLedger(layout, market_source_validator=raw)

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


def test_exposure_is_recomputed_and_cannot_be_underreported(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    ledger = PaperBetLedger(layout, market_source_validator=raw)
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


def test_accepted_entry_settlement_is_an_append_only_revision_and_recomputable(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, result = _fixture(tmp_path)
    ledger = PaperBetLedger(layout, market_source_validator=raw)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
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
    assert PaperBetLedger(layout).load(entry.entry_id) == entry
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
    derived = DerivedArchive(layout)
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


def test_retry_repairs_entry_when_derived_registration_failed_after_append(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    ledger = PaperBetLedger(layout, market_source_validator=raw)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
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
    layout, raw, prediction, market, result = _fixture(tmp_path)
    ledger = PaperBetLedger(
        layout,
        market_source_validator=raw,
        result_validator=None,
    )
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
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
        PaperBetLedger(layout, market_source_validator=raw).load(settled.entry_id)


def test_default_ledger_allows_void_settlement_without_a_result(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    ledger = PaperBetLedger(layout, market_source_validator=raw)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
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
    assert PaperBetLedger(layout).load(settled.entry_id) == settled


def test_ledger_rejects_entries_that_reference_an_unavailable_model_run(
    tmp_path: Path,
) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    accepted_arguments = {
        "prediction": prediction,
        "market_snapshot": market,
        "market_source_validator": raw,
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
        market_source_validator=raw,
        model_run_validator=missing,
    )

    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        ledger.append(entry)


def test_default_ledger_rejects_missing_prediction_or_model_bytes(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
        selection="home",
        risk_config=RiskConfig(),
        stake=10,
        match_exposure=10,
        daily_exposure=10,
        placed_at=AS_OF,
        settlement_rules=SettlementRules(selection="home"),
    )
    ledger = PaperBetLedger(layout, market_source_validator=raw)
    prediction_digest = prediction.id.value.removeprefix("prediction:")
    prediction_path = (
        layout.derived / "predictions" / prediction_digest[:2] / f"{prediction_digest}.json"
    )
    prediction_path.unlink()
    with pytest.raises(ValueError, match="prediction artifact"):
        ledger.append(entry)

    layout, raw, prediction, market, _result = _fixture(tmp_path / "model")
    entry = PaperBetEntry.accepted(
        prediction=prediction,
        market_snapshot=market,
        market_source_validator=raw,
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
        PaperBetLedger(layout, market_source_validator=raw).append(entry)


def test_ledger_rejects_rehashed_semantic_prediction_forgery(tmp_path: Path) -> None:
    layout, raw, prediction, market, _result = _fixture(tmp_path)
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
        market_source_validator=raw,
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
        PaperBetLedger(layout, market_source_validator=raw).append(forged_entry)


def test_ledger_parser_rejects_bool_schema_version(tmp_path: Path) -> None:
    layout, raw, prediction, _market, _result = _fixture(tmp_path)
    ledger = PaperBetLedger(layout, market_source_validator=raw)
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
