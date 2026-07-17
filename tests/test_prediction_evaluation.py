from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _formal_context import registered_target_identity, seed_formal_context
from _prediction_forgery import forge_persisted_prediction
from _training_refs import seed_training_references
from test_snapshots_lifecycle import _strict_official_lineup_context

import football_data_platform.domain.snapshots as snapshot_contract
from football_data_platform.domain.ids import (
    ModelRunId,
    PlayerId,
    PredictionId,
    RawAssetId,
    SnapshotId,
)
from football_data_platform.domain.lifecycle import Qualification
from football_data_platform.domain.predictions import (
    MarketDataKind,
    MarketQuote,
    MarketStatus,
    MatchResult90,
    build_market_snapshot,
    build_score_prediction,
    parse_prediction_payload,
    prediction_payload,
    result_probabilities,
    score_grid_composition_artifact_id,
    score_grid_composition_payload,
    verify_score_prediction,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    PreMatchSnapshot,
    SnapshotFeature,
    SnapshotSourceValidation,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    MODEL_ARTIFACT_REF_PREFIX,
    MODEL_OUTPUT_REF_PREFIX,
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
)
from football_data_platform.evaluation.governance import (
    ChallengerEvidence,
    PromotionPolicy,
    SubgroupDiagnostic,
    assess_promotion,
)
from football_data_platform.evaluation.metrics import (
    BenchmarkScore,
    categorical_log_loss,
    evaluate_prediction,
    evaluation_record_payload,
    multiclass_brier,
    parse_evaluation_record_payload,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contribution,
    context_contributions,
    lineup_delta_contributions,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive
from football_data_platform.storage.training import (
    TrainingArtifactConflict,
    TrainingArtifactStore,
)

MATCH, HOME, AWAY = registered_target_identity()
GENERATED_AT = datetime(2025, 8, 16, 12, 0, tzinfo=UTC)
KICKOFF = GENERATED_AT + timedelta(hours=24)
DEFAULT_MODEL_RUN_ID = ModelRunId("model-run:dc-v1")
MODEL_DIGEST = hashlib.sha256(b"prediction-evaluation-model").hexdigest()
OUTPUT_DIGEST = hashlib.sha256(b"prediction-evaluation-output").hexdigest()


def _snapshot(
    tmp_path: Path,
    *,
    ready: bool = True,
    as_of: datetime = GENERATED_AT,
    fixture_key: str = "prediction-evaluation",
    team_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
):
    layout = DataLayout(tmp_path / "snapshot-data")
    kickoff = as_of + timedelta(hours=24)
    formal = seed_formal_context(
        layout,
        as_of=as_of,
        kickoff=kickoff,
        key=fixture_key,
        observed_at=as_of,
        team_indices=team_indices,
    )
    archive = formal.raw
    derived = formal.derived
    asset = archive.archive(
        f"prediction snapshot evidence:{fixture_key}".encode(),
        source="test-source",
        source_id=f"prediction-fixture:{fixture_key}",
        url=f"fixture://prediction-fixture/{fixture_key}",
        observed_at=as_of,
        target_event_time=as_of - timedelta(days=1),
        collector_version="test-collector/1",
        media_type="application/octet-stream",
    )
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                f"prediction-baseline:{fixture_key}",
                formal.home_team_id.value,
                formal.away_team_id.value,
                as_of - timedelta(days=30),
                as_of - timedelta(days=1),
                1.7,
                0.8,
                asset.id.value,
            ),
        ),
        as_of=as_of,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=formal.home_team_id.value,
        away_team_id=formal.away_team_id.value,
    )
    baseline = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    features = (
        SnapshotFeature(
            name="team_baseline",
            value=baseline,
            known_at=as_of,
            source_ref=derived.write_team_baseline_source(
                match_id=formal.match_id,
                match_version=formal.match_version,
                as_of=as_of,
                baseline_artifact_id=baseline_artifact.artifact_id,
            ),
            contribution_key="team-baseline",
        ),
    )
    if ready:
        features += (
            SnapshotFeature(
                name="match_context",
                value=formal.context_value,
                known_at=formal.context_known_at,
                source_ref=formal.context_ref,
                contribution_key="match-context",
            ),
        )
    return (
        build_snapshot(
            match_id=formal.match_id,
            match_version=formal.match_version,
            snapshot_type=SnapshotType.T24H,
            as_of=as_of,
            scheduled_kickoff_used=kickoff,
            feature_spec_version="prematch-features/3",
            features=features,
            home_team_id=formal.home_team_id,
            away_team_id=formal.away_team_id,
            source_validator=derived,
        ),
        derived,
    )


def _prediction(
    snapshot_with_validator,
    *,
    generated_at: datetime = GENERATED_AT,
    model_run_id: ModelRunId = DEFAULT_MODEL_RUN_ID,
    model_version: str = "dixon-coles/1",
    model_run_validator=None,
):
    snapshot, validator = snapshot_with_validator
    composition = _composition(snapshot, validator)
    return build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=validator,
        model_run_id=model_run_id,
        model_version=model_version,
        generated_at=generated_at,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=(),
        model_run_validator=model_run_validator,
        expected_goals=composition,
    )


def _composition(
    snapshot,
    validator,
    *,
    context_source_ref: str | None = None,
    coefficient: float = 0.02,
):
    baseline = next(feature for feature in snapshot.features if feature.name == "team_baseline")
    context = next(feature for feature in snapshot.features if feature.name == "match_context")
    effects = context_contributions(
        context.value,
        home_team_id=snapshot.home_team_id.value,
        away_team_id=snapshot.away_team_id.value,
        source_ref=context_source_ref or context.source_ref,
        source_validator=validator,
        coefficient=coefficient,
    )
    return compose_expected_goals(
        baseline.value["lambda_home"],
        baseline.value["lambda_away"],
        effects,
    )


def _reidentify_prediction(prediction):
    composition_ref = score_grid_composition_artifact_id(score_grid_composition_payload(prediction))
    prediction = replace(prediction, composition_artifact_ref=composition_ref)
    payload = prediction_payload(prediction)
    payload.pop("id")
    digest = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return replace(prediction, id=PredictionId(f"prediction:{digest}"))


def _model_run(
    *,
    model_version: str = "dixon-coles/1",
    task: str = "score-model",
    ended_at: datetime = GENERATED_AT - timedelta(seconds=1),
    status: ModelRunStatus = ModelRunStatus.SUCCEEDED,
) -> ModelRunArtifact:
    output_refs = (
        () if status is ModelRunStatus.FAILED else (MODEL_OUTPUT_REF_PREFIX + OUTPUT_DIGEST,)
    )
    return ModelRunArtifact.create(
        model_version=model_version,
        run_role="challenger",
        task=task,
        dataset_id="training-dataset:" + "a" * 64,
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        algorithm="dixon-coles",
        parameters={"rho": -0.12, "max_goals": 11},
        code_version="git:test",
        environment_version="python-3.11:test-lock",
        started_at=ended_at - timedelta(seconds=1),
        ended_at=ended_at,
        random_seed=17,
        model_artifact_refs=(
            () if status is ModelRunStatus.FAILED else (MODEL_ARTIFACT_REF_PREFIX + MODEL_DIGEST,)
        ),
        evaluation_cohort=() if status is ModelRunStatus.FAILED else ("sample:test",),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=output_refs,
        output_hashes=() if not output_refs else (OUTPUT_DIGEST,),
        status=status,
        error="training failed" if status is ModelRunStatus.FAILED else None,
    )


def _persisted_prediction_fixture(tmp_path: Path):
    training_snapshot, archive = _snapshot(
        tmp_path,
        as_of=GENERATED_AT - timedelta(days=10),
        fixture_key="prediction-evaluation-train",
        team_indices=(4, 5, 6, 7),
    )
    archive.write_snapshot(training_snapshot)
    holdout_snapshot, archive = _snapshot(
        tmp_path,
        as_of=GENERATED_AT - timedelta(days=5),
        fixture_key="prediction-evaluation-holdout",
        team_indices=(8, 9, 10, 11),
    )
    archive.write_snapshot(holdout_snapshot)
    snapshot, archive = _snapshot(tmp_path)
    archive.write_snapshot(snapshot)
    training = TrainingArtifactStore(archive.layout)
    canonical = CanonicalStore(archive.layout.canonical / "platform.sqlite3")
    raw = RawArchive(archive.layout)
    facts = CanonicalFactStore(canonical, raw_archive=raw)

    def formal_sample(
        source_snapshot: PreMatchSnapshot,
        *,
        sample_id: str,
        split: str,
        key: str,
    ) -> tuple[TrainingSample, datetime]:
        result_known_at = source_snapshot.scheduled_kickoff_used + timedelta(hours=2)
        result_asset = raw.archive(
            f"persisted prediction result:{key}".encode(),
            source="test-result",
            source_id=f"persisted-prediction-result:{key}",
            url=f"fixture://persisted-prediction-result/{key}",
            observed_at=result_known_at,
            target_event_time=source_snapshot.scheduled_kickoff_used,
            collector_version="test-result/1",
            media_type="application/octet-stream",
        )
        canonical.register_raw_asset(result_asset)
        result = facts.append_result_90(
            match_id=source_snapshot.match_id,
            match_version=source_snapshot.match_version,
            home_goals=1,
            away_goals=0,
            known_at=result_known_at,
            observed_at=result_known_at,
            raw_asset_id=result_asset.id,
        )
        evaluated_at = result_known_at + timedelta(hours=1)
        qualification = training.create_training_qualification(
            match_id=source_snapshot.match_id,
            match_version=source_snapshot.match_version,
            qualification=Qualification.SCORE_MODEL,
            ruleset_version="readiness/1",
            evaluated_at=evaluated_at,
            snapshot_ref=source_snapshot.id.value,
            result_ref=result.record_id,
        )
        sample = training.build_formal_score_sample(
            sample_id=sample_id,
            qualification_ref=qualification.qualification_id,
            feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
            split=split,
        )
        return sample, evaluated_at

    training_sample, training_evaluated_at = formal_sample(
        training_snapshot,
        sample_id="sample:prediction-train",
        split="train",
        key="train",
    )
    holdout_sample, holdout_evaluated_at = formal_sample(
        holdout_snapshot,
        sample_id="sample:prediction-holdout",
        split="test",
        key="holdout",
    )
    dataset_generated_at = max(training_evaluated_at, holdout_evaluated_at)
    dataset = TrainingDatasetManifest.create_formal(
        dataset_version="prediction-dataset/2",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        as_of=dataset_generated_at,
        split_strategy="forward-chaining/1",
        samples=(training_sample, holdout_sample),
        generated_at=dataset_generated_at,
        transform_version="training-dataset/2",
        code_version="git:test",
    )
    training.write_dataset(dataset)
    model_ref = training.write_model_artifact(b"persisted prediction model")
    output_ref = training.write_model_output(b"persisted prediction output")
    model_run = ModelRunArtifact.create(
        model_version="dixon-coles/1",
        run_role="challenger",
        task="score-model",
        dataset_id=dataset.dataset_id,
        feature_version=dataset.feature_version,
        label_version=dataset.label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.12, "max_goals": 11},
        code_version="git:test",
        environment_version="python-3.11:test-lock",
        started_at=dataset_generated_at,
        ended_at=dataset_generated_at + timedelta(seconds=1),
        random_seed=17,
        model_artifact_refs=(model_ref,),
        evaluation_cohort=(holdout_sample.sample_id,),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_ref,),
        output_hashes=(output_ref.removeprefix(MODEL_OUTPUT_REF_PREFIX),),
    )
    training.write_model_run(model_run)
    archive.model_run_validator = training
    prediction = _prediction(
        (snapshot, archive),
        generated_at=GENERATED_AT,
        model_run_id=ModelRunId(model_run.model_run_id),
        model_run_validator=training,
    )
    return snapshot, archive, prediction, model_run


def _persisted_ready_lineup_prediction_fixture(tmp_path: Path):
    canonical, _, archive, lineup_asset, teams, match, version, home_players, contract = (
        _strict_official_lineup_context(tmp_path)
    )
    with canonical.connect() as connection:
        away_players = tuple(
            PlayerId(row["player_id"])
            for row in connection.execute(
                "SELECT player_id FROM lineup_facts WHERE match_id = ? AND team_id = ? "
                "AND lineup_role = 'starter' AND official = 1 ORDER BY player_id",
                (match.id.value, teams[1].id.value),
            ).fetchall()
        )
    player_objects_by_team = {
        teams[0].id.value: home_players,
        teams[1].id.value: away_players,
    }
    player_ids_by_team = {
        team_id: tuple(player_id.value for player_id in player_ids)
        for team_id, player_ids in player_objects_by_team.items()
    }
    observed_at = lineup_asset.observed_at
    official_sources = {
        team.id.value: archive.write_official_lineup_source(
            contract_id=contract.contract_id,
            match_id=match.id,
            match_version=version.version,
            team_id=team.id,
            player_ids=player_objects_by_team[team.id.value],
            known_at=observed_at,
            observed_at=observed_at,
            raw_asset_id=lineup_asset.id,
        )
        for team in teams
    }
    delta_value = {
        team_id: {
            "quality_status": "ready",
            "dimension_deltas": {"attack": 0.1 if index == 0 else -0.1},
            "missing_fields": [],
        }
        for index, team_id in enumerate(player_ids_by_team)
    }
    delta_ref = archive.write_snapshot_source(
        value=delta_value,
        input_refs=(lineup_asset.id,),
        transform_version="lineup-delta-input/1",
        generated_at=observed_at,
    )

    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "ready-lineup-baseline",
                teams[0].id.value,
                teams[1].id.value,
                observed_at - timedelta(days=30),
                observed_at - timedelta(days=1),
                1.7,
                0.8,
                lineup_asset.id.value,
            ),
        ),
        as_of=observed_at,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    archive.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=teams[0].id.value,
        away_team_id=teams[1].id.value,
    )
    baseline_value = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    baseline_ref = archive.write_snapshot_source(
        value=baseline_value,
        input_refs=(lineup_asset.id,),
        transform_version="team-baseline-input/2",
        generated_at=observed_at,
    )
    context_value = {"days_since_previous_match": 6.0}
    context_ref = archive.write_snapshot_source(
        value=context_value,
        input_refs=(lineup_asset.id,),
        transform_version="match-context-input/1",
        generated_at=observed_at,
    )
    raw_features = (
        SnapshotFeature("team_baseline", baseline_value, observed_at, baseline_ref, "baseline"),
        SnapshotFeature("match_context", context_value, observed_at, context_ref, "context"),
        SnapshotFeature("lineup_delta", delta_value, observed_at, delta_ref, "lineup"),
        *(
            SnapshotFeature(
                "official_lineup_confirmed",
                list(player_ids),
                observed_at,
                official_sources[team_id],
                f"official-lineup:{team_id}",
                team_id,
            )
            for team_id, player_ids in player_ids_by_team.items()
        ),
    )
    features, source_observed_at, input_refs, missing_fields = snapshot_contract._validated_state(
        match.id,
        version.version,
        SnapshotType.LINEUPS_CONFIRMED,
        observed_at,
        version.kickoff_at,
        "prematch-features/1",
        raw_features,
        teams[0].id,
        teams[1].id,
        archive,
    )
    fields = {
        "match_id": match.id,
        "match_version": version.version,
        "home_team_id": teams[0].id,
        "away_team_id": teams[1].id,
        "snapshot_type": SnapshotType.LINEUPS_CONFIRMED,
        "capture_mode": CaptureMode.RECONSTRUCTED,
        "as_of": observed_at,
        "observed_at": source_observed_at,
        "scheduled_kickoff_used": version.kickoff_at,
        "feature_spec_version": "prematch-features/1",
        "features": features,
        "input_refs": input_refs,
        "quality_status": "ready",
        "missing_fields": missing_fields,
    }
    identity = snapshot_contract._snapshot_identity(**fields)
    snapshot = PreMatchSnapshot(
        id=SnapshotId(
            "snapshot:"
            + hashlib.sha256(
                json.dumps(
                    identity,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        ),
        schema_version=snapshot_contract.SNAPSHOT_SCHEMA_VERSION,
        **fields,
    )
    archive.write_snapshot(snapshot)

    label_known_at = observed_at - timedelta(minutes=5)
    feature_ref, label_ref = seed_training_references(
        archive.layout,
        label_known_at=label_known_at,
        observed_at=observed_at,
        home_goals=1,
        away_goals=0,
        reference_key="ready-lineup-prediction",
    )
    sample = TrainingSample(
        sample_id="sample:ready-lineup",
        as_of=observed_at - timedelta(hours=1),
        feature_known_at=observed_at - timedelta(hours=2),
        label_known_at=label_known_at,
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=True,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(feature_ref,),
        label_ref=label_ref,
        features={"strength": 1.0},
        label={"home_goals": 1, "away_goals": 0},
        split="train",
    )
    holdout = replace(
        sample,
        sample_id="sample:ready-lineup-holdout",
        as_of=observed_at - timedelta(minutes=15),
        feature_known_at=observed_at - timedelta(minutes=15),
        split="test",
    )
    dataset = TrainingDatasetManifest.create(
        dataset_version="prediction-dataset/ready-lineup",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=observed_at,
        split_strategy="forward-chaining/1",
        samples=(sample, holdout),
        generated_at=observed_at,
        code_version="git:test",
    )
    training = TrainingArtifactStore(archive.layout)
    training.write_dataset(dataset)
    model_ref = training.write_model_artifact(b"ready lineup prediction model")
    output_ref = training.write_model_output(b"ready lineup prediction output")
    model_run = ModelRunArtifact.create(
        model_version="dixon-coles/ready-lineup",
        run_role="challenger",
        task="score-model",
        dataset_id=dataset.dataset_id,
        feature_version=dataset.feature_version,
        label_version=dataset.label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.12, "max_goals": 11},
        code_version="git:test",
        environment_version="python:test-lock",
        started_at=observed_at,
        ended_at=observed_at,
        random_seed=17,
        model_artifact_refs=(model_ref,),
        evaluation_cohort=(holdout.sample_id,),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_ref,),
        output_hashes=(output_ref.removeprefix(MODEL_OUTPUT_REF_PREFIX),),
    )
    training.write_model_run(model_run)
    context = context_contribution(context_value, source_ref=context_ref)

    class _ContributionSourceValidator:
        def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation:
            return SnapshotSourceValidation(
                source_ref=source_ref,
                source_kind="derived",
                transform_version="lineup-delta-input/3",
                observed_at=observed_at,
                value=delta_value,
                input_refs=(lineup_asset.id.value,),
            )

    lineup = lineup_delta_contributions(
        delta_value,
        home_team_id=teams[0].id.value,
        away_team_id=teams[1].id.value,
        source_ref=delta_ref,
        source_validator=_ContributionSourceValidator(),
    )
    composition = compose_expected_goals(
        lambda_home,
        lambda_away,
        (context, *lineup),
    )
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=archive,
        model_run_id=ModelRunId(model_run.model_run_id),
        model_version=model_run.model_version,
        generated_at=observed_at,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=(),
        model_run_validator=training,
        expected_goals=composition,
    )
    archive.model_run_validator = training
    archive.write_prediction(prediction, snapshot=snapshot)
    return snapshot, archive, prediction


class _ModelRunRegistry:
    def __init__(self, artifact: ModelRunArtifact | None) -> None:
        self.artifact = artifact

    def load_model_run(self, model_run_id: str) -> ModelRunArtifact:
        if self.artifact is None:
            raise FileNotFoundError(model_run_id)
        return self.artifact


class _ResultRegistry:
    def __init__(self, expected: MatchResult90) -> None:
        self.expected = expected

    def verify_match_result(self, result: MatchResult90) -> None:
        if result != self.expected:
            raise ValueError("unknown canonical result")


class _GovernanceReferenceRegistry:
    def __init__(self, *references: str) -> None:
        self.references = set(references)

    def verify_reference(self, reference: str) -> None:
        if reference not in self.references:
            raise ValueError(f"unknown governance reference: {reference}")


def _market(
    tmp_path: Path,
    *,
    observed_at: datetime = GENERATED_AT,
    quotes: tuple[MarketQuote, ...] = (
        MarketQuote("home", 2.0),
        MarketQuote("draw", 3.5),
        MarketQuote("away", 4.0),
    ),
    data_kind: MarketDataKind = MarketDataKind.REAL,
    status: MarketStatus = MarketStatus.OPEN,
):
    archive = RawArchive(DataLayout(tmp_path / "market-data"))
    asset = archive.archive(
        b'{"market":"result_90"}',
        source="bookmaker-api",
        source_id="prediction-fixture-result-90",
        url="https://bookmaker.example/prediction-fixture",
        observed_at=observed_at,
        target_event_time=KICKOFF,
        collector_version="market-collector/1",
        media_type="application/json",
    )
    return (
        build_market_snapshot(
            match_id=MATCH,
            market_type="result_90",
            status=status,
            data_kind=data_kind,
            quotes=quotes,
            raw_asset=asset,
        ),
        archive,
        asset,
    )


def test_prediction_schema_has_one_coordinate_explicit_normalized_grid(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    payload = prediction_payload(prediction)

    assert payload["schema_version"] == 4
    assert len(payload["score_cells"]) == 144
    assert len({(cell["home_goals"], cell["away_goals"]) for cell in payload["score_cells"]}) == 144
    assert sum(cell["probability"] for cell in payload["score_cells"]) == pytest.approx(1.0)
    assert sum(result_probabilities(prediction).values()) == pytest.approx(1.0)


def test_prediction_persists_expected_goals_provenance_and_grid_manifest(tmp_path: Path) -> None:
    snapshot, archive, prediction, model_run = _persisted_prediction_fixture(tmp_path)
    context_source_ref = next(
        feature.source_ref for feature in snapshot.features if feature.name == "match_context"
    )

    payload = prediction_payload(prediction)
    assert payload["baseline_lambda_home"] == pytest.approx(1.7)
    assert payload["baseline_lambda_away"] == pytest.approx(0.8)
    assert set(payload["contribution_keys"]) == {
        f"context:rest-days:{HOME.value}:match-context-calibration/2",
        f"context:rest-days:{AWAY.value}:match-context-calibration/2",
    }
    multipliers = {item["contribution_key"]: item for item in payload["contribution_multipliers"]}
    assert multipliers[f"context:rest-days:{HOME.value}:match-context-calibration/2"][
        "lambda_home_multiplier"
    ] == pytest.approx(math.exp(0.08))
    assert multipliers[f"context:rest-days:{AWAY.value}:match-context-calibration/2"][
        "lambda_away_multiplier"
    ] == pytest.approx(math.exp(0.02))
    assert payload["composition_version"] == prediction.composition_version
    assert payload["calibration_versions"] == ["match-context-calibration/2"]
    assert prediction.composition_artifact_ref.startswith("score-grid-composition:")
    assert model_run.model_run_id in prediction.input_refs
    assert context_source_ref in snapshot.input_refs

    composition_path = archive.write_score_grid_composition(prediction)
    assert composition_path == archive.score_grid_composition_path(
        prediction.composition_artifact_ref
    )
    archive.write_prediction(prediction, snapshot=snapshot)
    assert archive.verify_score_grid_composition(prediction).artifact_type == (
        "score-grid-composition"
    )
    assert archive.load_score_grid_composition_payload(prediction.composition_artifact_ref)[
        "rho"
    ] == pytest.approx(-0.12)


def test_verified_prediction_loader_rejects_rehashed_baseline_forgery(
    tmp_path: Path,
) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    archive.write_prediction(prediction, snapshot=snapshot)

    forged_ref = forge_persisted_prediction(
        archive.layout,
        prediction.id.value,
        lambda payload: payload.__setitem__(
            "baseline_lambda_home", payload["baseline_lambda_home"] * 2
        ),
    )

    with pytest.raises(TrainingArtifactConflict, match="baseline.*snapshot"):
        TrainingArtifactStore(archive.layout).load_verified_prediction(forged_ref)


def test_verified_prediction_loader_rejects_rehashed_calibration_forgery(
    tmp_path: Path,
) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    archive.write_prediction(prediction, snapshot=snapshot)

    def mutate(payload: dict) -> None:
        payload["contribution_multipliers"][0]["calibration"]["lambda_home_coefficient"] = 0.03

    forged_ref = forge_persisted_prediction(
        archive.layout,
        prediction.id.value,
        mutate,
    )

    with pytest.raises(TrainingArtifactConflict, match="calibration policy"):
        TrainingArtifactStore(archive.layout).load_verified_prediction(forged_ref)


def test_verified_prediction_loader_rejects_rehashed_context_omission(
    tmp_path: Path,
) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    archive.write_prediction(prediction, snapshot=snapshot)
    omitted_key = prediction.contribution_keys[0]

    def mutate(payload: dict) -> None:
        payload["contribution_multipliers"] = [
            item
            for item in payload["contribution_multipliers"]
            if item["contribution_key"] != omitted_key
        ]

    forged_ref = forge_persisted_prediction(
        archive.layout,
        prediction.id.value,
        mutate,
    )

    with pytest.raises(TrainingArtifactConflict, match="contribution set"):
        TrainingArtifactStore(archive.layout).load_verified_prediction(forged_ref)


def test_verified_prediction_loader_rejects_rehashed_post_kickoff_prediction(
    tmp_path: Path,
) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    archive.write_prediction(prediction, snapshot=snapshot)
    post_kickoff = snapshot.scheduled_kickoff_used + timedelta(seconds=1)

    forged_ref = forge_persisted_prediction(
        archive.layout,
        prediction.id.value,
        lambda payload: payload.__setitem__(
            "generated_at", post_kickoff.isoformat().replace("+00:00", "Z")
        ),
    )
    store = TrainingArtifactStore(archive.layout)

    assert store.load_prediction_for_audit(forged_ref).generated_at == post_kickoff
    with pytest.raises(TrainingArtifactConflict, match="scheduled_kickoff_used"):
        store.load_verified_prediction(forged_ref)


def test_formal_prediction_rejects_legacy_inline_lambdas(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="legacy-inline/1 is read-only"):
        build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=validator,
            model_run_id=DEFAULT_MODEL_RUN_ID,
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.7,
            lambda_away=0.8,
            rho=-0.12,
            max_goals=11,
            input_refs=(),
        )


def test_calibration_version_rejects_unreviewed_parameter_change(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="coefficient|calibration"):
        _composition(snapshot, validator, coefficient=0.03)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lambda_home_coefficient", 0.03),
        ("reference_value", 4.0),
        ("maximum_log_multiplier", 0.10),
    ),
)
def test_prediction_loader_rejects_tampered_calibration_parameters(
    tmp_path: Path, field: str, value: float
) -> None:
    snapshot, validator = _snapshot(tmp_path)
    payload = prediction_payload(_prediction((snapshot, validator)))
    payload["contribution_multipliers"][0]["calibration"][field] = value

    with pytest.raises(ValueError, match="calibration|composition_artifact_ref|canonical"):
        parse_prediction_payload(
            payload,
            snapshot=snapshot,
            snapshot_validator=validator,
        )


def test_prediction_rejects_contribution_source_outside_verified_snapshot(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    feature_source_refs = {feature.source_ref for feature in snapshot.features}
    nested_raw_ref = next(ref for ref in snapshot.input_refs if ref not in feature_source_refs)
    for invalid_source_ref in (nested_raw_ref, "derived-source:" + "f" * 64):
        with pytest.raises((OSError, ValueError)):
            _composition(
                snapshot=snapshot,
                validator=validator,
                context_source_ref=invalid_source_ref,
            )


def test_prediction_archive_rejects_forged_contribution_lineage(tmp_path: Path) -> None:
    snapshot, archive = _snapshot(tmp_path)
    composition = _composition(snapshot, archive)
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=archive,
        model_run_id=DEFAULT_MODEL_RUN_ID,
        model_version="dixon-coles/1",
        generated_at=GENERATED_AT,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=("baseline:v1",),
        expected_goals=composition,
    )
    feature_source_refs = {feature.source_ref for feature in snapshot.features}
    nested_raw_ref = next(ref for ref in snapshot.input_refs if ref not in feature_source_refs)
    forged = replace(
        prediction,
        input_refs=tuple(sorted({*prediction.input_refs, nested_raw_ref})),
        contribution_multipliers=tuple(
            replace(item, source_ref=nested_raw_ref) for item in prediction.contribution_multipliers
        ),
    )
    forged = _reidentify_prediction(forged)
    verify_score_prediction(forged)

    with pytest.raises(ValueError, match="verified snapshot feature sources"):
        archive.write_prediction(forged, snapshot=snapshot)


def test_prediction_composition_bytes_are_tamper_evident(tmp_path: Path) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    archive.write_prediction(prediction, snapshot=snapshot)
    path = archive.score_grid_composition_path(prediction.composition_artifact_ref)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["lambda_home"] = 9.0
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ArchiveConflictError, match="composition"):
        archive.load_prediction_payload(prediction)


def test_prediction_and_market_are_separate_evaluation_inputs(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result")
    without_market = evaluate_prediction(
        prediction,
        result,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    market, market_archive, _ = _market(tmp_path)
    with_market = evaluate_prediction(
        prediction,
        result,
        market_snapshot=market,
        market_source_validator=market_archive,
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )

    assert not without_market.market_benchmark.available
    assert without_market.market_benchmark.reason == "market_snapshot_missing"
    assert with_market.market_benchmark.available
    assert with_market.market_benchmark.brier is not None
    assert with_market.capture_mode is CaptureMode.RECONSTRUCTED


def test_evaluation_rejects_a_result_unknown_at_evaluation_time(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(
        MATCH,
        2,
        1,
        GENERATED_AT + timedelta(days=2),
        "canonical:result",
    )

    with pytest.raises(ValueError, match="not known"):
        evaluate_prediction(
            prediction,
            result,
            evaluated_at=GENERATED_AT + timedelta(days=1),
        )


def test_evaluation_can_require_canonical_result_evidence(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    result = MatchResult90(MATCH, 2, 1, GENERATED_AT, "fact:match-results:known")
    validator = _ResultRegistry(result)

    evaluated = evaluate_prediction(
        prediction,
        result,
        evaluated_at=GENERATED_AT + timedelta(days=1),
        result_validator=validator,
    )
    assert "fact:match-results:known" in evaluated.input_refs

    with pytest.raises(ValueError, match="result source evidence"):
        evaluate_prediction(
            prediction,
            replace(result, source_ref="canonical:forged"),
            evaluated_at=GENERATED_AT + timedelta(days=1),
            result_validator=validator,
        )


@pytest.mark.parametrize(
    ("market_options", "expected_error"),
    (
        ({"observed_at": GENERATED_AT + timedelta(seconds=1)}, "after prediction"),
        (
            {"quotes": (MarketQuote("home", 2.0), MarketQuote("away", 4.0))},
            "exactly home, draw, and away",
        ),
        ({"data_kind": MarketDataKind.SYNTHETIC}, "synthetic"),
        ({"status": MarketStatus.SUSPENDED}, "open market"),
    ),
)
def test_market_benchmark_rejects_future_incomplete_synthetic_or_closed_quotes(
    tmp_path: Path,
    market_options: dict[str, object],
    expected_error: str,
) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, _ = _market(tmp_path, **market_options)

    with pytest.raises(ValueError, match=expected_error):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_benchmark_requires_verified_raw_lineage(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, asset = _market(tmp_path)
    unknown_asset = replace(asset, id=RawAssetId("raw-asset:" + "f" * 64))
    unknown_market = build_market_snapshot(
        match_id=MATCH,
        market_type="result_90",
        status=MarketStatus.OPEN,
        data_kind=MarketDataKind.REAL,
        quotes=market.quotes,
        raw_asset=unknown_asset,
    )

    with pytest.raises(ValueError, match="raw evidence"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=unknown_market,
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="source metadata"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=replace(market, observed_at=GENERATED_AT - timedelta(minutes=1)),
            market_source_validator=archive,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_time_is_checked_against_snapshot_as_of_not_generation_time(
    tmp_path: Path,
) -> None:
    prediction = _prediction(
        _snapshot(tmp_path),
        generated_at=KICKOFF + timedelta(hours=1),
    )
    market, archive, _ = _market(tmp_path, observed_at=KICKOFF + timedelta(minutes=1))

    with pytest.raises(ValueError, match="snapshot as_of"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=KICKOFF + timedelta(days=1),
        )


def test_market_benchmark_requires_a_validator_when_present(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, _, _ = _market(tmp_path)

    with pytest.raises(ValueError, match="raw evidence validator"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        )


def test_market_benchmark_rejects_quote_unknown_at_evaluation_time(tmp_path: Path) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    market, archive, _ = _market(tmp_path, observed_at=GENERATED_AT - timedelta(minutes=1))

    with pytest.raises(ValueError, match="after evaluation"):
        evaluate_prediction(
            prediction,
            MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
            market_snapshot=market,
            market_source_validator=archive,
            evaluated_at=GENERATED_AT - timedelta(minutes=2),
        )


@pytest.mark.parametrize("odds", (math.nan, math.inf, -1.0, 1.0))
def test_market_quote_rejects_non_finite_or_non_positive_price(odds: float) -> None:
    with pytest.raises(ValueError, match="decimal_odds"):
        MarketQuote("home", odds)


def test_brier_and_log_loss_use_known_toy_values(tmp_path: Path) -> None:
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    expected_brier = (0.5 - 1) ** 2 + 0.3**2 + 0.2**2
    prediction = _prediction(_snapshot(tmp_path))

    assert multiclass_brier(probabilities, "home") == pytest.approx(expected_brier)
    evaluation = evaluate_prediction(
        prediction,
        MatchResult90(MATCH, 1, 0, GENERATED_AT, "canonical:result"),
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
    )
    assert evaluation.result_log_loss == pytest.approx(
        -math.log(result_probabilities(prediction)["home"])
    )


def test_evaluation_record_v2_is_replayable_and_rejects_synthetic_benchmark(
    tmp_path: Path,
) -> None:
    prediction = _prediction(_snapshot(tmp_path))
    evaluation = evaluate_prediction(
        prediction,
        MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        sample_ref="sample:test",
    )
    payload = evaluation_record_payload(evaluation)

    assert payload["schema_version"] == 2
    assert payload["record_type"] == "evaluation-record"
    assert parse_evaluation_record_payload(payload) == evaluation

    tampered_score = copy.deepcopy(payload)
    tampered_score["result_brier"] += 0.1
    with pytest.raises(ValueError, match="result_brier is not reproducible"):
        parse_evaluation_record_payload(tampered_score)

    synthetic = copy.deepcopy(payload)
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    synthetic["market_benchmark"] = {
        "available": True,
        "brier": multiclass_brier(probabilities, "home"),
        "log_loss": categorical_log_loss(probabilities, "home"),
        "reason": None,
        "probabilities": [list(item) for item in probabilities.items()],
        "data_kind": "synthetic",
    }
    with pytest.raises(ValueError, match="benchmark must be real"):
        parse_evaluation_record_payload(synthetic)

    real_without_lineage = replace(
        evaluation,
        market_benchmark=BenchmarkScore(
            available=True,
            brier=multiclass_brier(probabilities, "home"),
            log_loss=categorical_log_loss(probabilities, "home"),
            reason=None,
            probabilities=tuple(probabilities.items()),
            data_kind="real",
        ),
    )
    with pytest.raises(ValueError, match="market snapshot and raw evidence refs"):
        evaluation_record_payload(real_without_lineage)

    archive = DerivedArchive(DataLayout(tmp_path / "invalid-evaluation-write"))
    with pytest.raises(TypeError, match="EvaluationRecord"):
        archive.write_evaluation(payload)
    with pytest.raises(ValueError, match="result_brier is not reproducible"):
        archive.write_evaluation(replace(evaluation, result_brier=evaluation.result_brier + 0.1))
    with pytest.raises(ValueError, match="generated_at must equal evaluated_at"):
        archive.write_evaluation(
            evaluation,
            generated_at=evaluation.evaluated_at + timedelta(seconds=1),
        )


def test_prediction_archive_never_mixes_batches_or_market_objects(tmp_path: Path) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)

    first = archive.write_prediction(prediction, snapshot=snapshot)
    second = archive.write_prediction(prediction, snapshot=snapshot)

    assert first == second
    assert archive.load_prediction_payload(prediction)["id"] == prediction.id.value
    assert first.parent.name == prediction.id.value.removeprefix("prediction:")[:2]


def test_prediction_archive_optionally_requires_a_valid_model_run(tmp_path: Path) -> None:
    snapshot, archive, prediction, artifact = _persisted_prediction_fixture(tmp_path)

    archive.write_prediction(prediction, snapshot=snapshot)
    assert archive.load_prediction_payload(prediction)["model_run_id"] == artifact.model_run_id

    with pytest.raises(ValueError, match="model run is unavailable or invalid"):
        DerivedArchive(
            archive.layout,
            model_run_validator=_ModelRunRegistry(None),
        ).write_prediction(prediction, snapshot=snapshot)


@pytest.mark.parametrize(
    ("artifact_options", "prediction_options", "expected_error"),
    (
        (
            {"model_version": "dixon-coles/2"},
            {"model_version": "dixon-coles/1"},
            "model_version does not match",
        ),
        (
            {"task": "player-profile"},
            {},
            "score-model task",
        ),
        (
            {"ended_at": GENERATED_AT + timedelta(seconds=1)},
            {},
            "model run completion",
        ),
        (
            {"status": ModelRunStatus.PARTIAL},
            {},
            "successful model run",
        ),
    ),
)
def test_prediction_model_run_contract_rejects_mismatched_or_unusable_artifacts(
    tmp_path: Path,
    artifact_options: dict[str, object],
    prediction_options: dict[str, object],
    expected_error: str,
) -> None:
    artifact = _model_run(**artifact_options)
    with pytest.raises(ValueError, match=expected_error):
        _prediction(
            _snapshot(tmp_path),
            model_run_id=ModelRunId(artifact.model_run_id),
            model_run_validator=_ModelRunRegistry(artifact),
            **prediction_options,
        )


def test_evaluation_and_archive_reject_tampered_prediction_content(tmp_path: Path) -> None:
    snapshot, archive = _snapshot(tmp_path)
    prediction = _prediction((snapshot, archive))
    shifted_as_of = replace(
        prediction,
        snapshot_as_of=prediction.snapshot_as_of - timedelta(minutes=1),
    )
    tampered_cells = replace(
        prediction,
        score_cells=(
            replace(prediction.score_cells[0], probability=0.99),
            *prediction.score_cells[1:],
        ),
    )

    for tampered in (shifted_as_of, tampered_cells):
        with pytest.raises(ValueError, match="prediction (identity|score cells)"):
            evaluate_prediction(
                tampered,
                MatchResult90(MATCH, 2, 1, GENERATED_AT, "canonical:result"),
                evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
            )
        with pytest.raises(ValueError, match="prediction (identity|score cells)"):
            archive.write_prediction(tampered, snapshot=snapshot)


def test_challenger_promotion_requires_an_explicit_policy() -> None:
    challenger_ref = "model-run:" + "a" * 64
    champion_ref = "model-run:" + "b" * 64
    evaluation_ref = "evaluation:" + "c" * 64
    cohort_ref = "training-dataset:" + "d" * 64
    sample_ref = "sample:captured-001"
    rollback_ref = "model-run:" + "e" * 64
    evidence = ChallengerEvidence(
        capture_mode=CaptureMode.CAPTURED,
        captured_samples=1,
        observation_days=120,
        brier_delta_vs_champion=-0.01,
        log_loss_delta_vs_champion=-0.02,
        confidence_interval_passed=True,
        reliability_passed=True,
        subgroup_diagnostics_passed=True,
        model_run_ref=challenger_ref,
        champion_model_ref=champion_ref,
        evaluation_ref=evaluation_ref,
        cohort_ref=cohort_ref,
        sample_refs=(sample_ref,),
        confidence_level=0.95,
        confidence_interval=(-0.02, -0.005),
        confidence_method="paired-bootstrap/1",
        subgroup_diagnostics=(SubgroupDiagnostic("all", 1, -0.01, -0.02, True),),
        prospective=True,
        evaluated_at=GENERATED_AT,
    )

    with pytest.raises(ValueError, match="explicit reviewed"):
        assess_promotion(evidence, policy=None)

    decision = assess_promotion(
        evidence,
        policy=PromotionPolicy(
            policy_version="promotion-policy/1",
            reviewed_by="model-risk-reviewer",
            reviewed_at=GENERATED_AT,
            confidence_method="paired-bootstrap/1",
            rollback_target=rollback_ref,
            minimum_captured_samples=1,
            minimum_observation_days=90,
            maximum_brier_delta=-0.005,
            maximum_log_loss_delta=-0.01,
            confidence_level=0.95,
        ),
        reference_validator=_GovernanceReferenceRegistry(
            challenger_ref,
            champion_ref,
            evaluation_ref,
            cohort_ref,
            sample_ref,
            rollback_ref,
        ),
        decided_at=GENERATED_AT + timedelta(days=1),
    )
    assert decision.promoted


def test_challenger_promotion_rejects_invalid_or_non_captured_evidence() -> None:
    policy = PromotionPolicy(
        policy_version="promotion-policy/1",
        reviewed_by="model-risk-reviewer",
        reviewed_at=GENERATED_AT,
        confidence_method="paired-bootstrap/1",
        rollback_target="model-run:" + "e" * 64,
        minimum_captured_samples=400,
        minimum_observation_days=90,
        maximum_brier_delta=-0.005,
        maximum_log_loss_delta=-0.01,
    )
    common = {
        "capture_mode": CaptureMode.CAPTURED,
        "captured_samples": 500,
        "observation_days": 120,
        "brier_delta_vs_champion": -0.01,
        "log_loss_delta_vs_champion": -0.02,
        "confidence_interval_passed": True,
        "reliability_passed": True,
        "subgroup_diagnostics_passed": True,
    }

    with pytest.raises(ValueError, match="finite"):
        ChallengerEvidence(**{**common, "brier_delta_vs_champion": math.nan})
    with pytest.raises(ValueError, match="non-negative"):
        ChallengerEvidence(**{**common, "captured_samples": -1})

    decision = assess_promotion(
        ChallengerEvidence(**{**common, "capture_mode": CaptureMode.RECONSTRUCTED}),
        policy=policy,
    )
    assert not decision.promoted
    assert "prospective_captured_cohort_required" in decision.reason_codes


def test_promotion_policy_requires_review_and_rollback_metadata() -> None:
    with pytest.raises(ValueError, match="reviewed_by"):
        PromotionPolicy(
            policy_version="promotion-policy/1",
            reviewed_by="",
            reviewed_at=GENERATED_AT,
            confidence_method="paired-bootstrap/1",
            rollback_target="model-run:champion-v1",
            minimum_captured_samples=400,
            minimum_observation_days=90,
            maximum_brier_delta=-0.005,
            maximum_log_loss_delta=-0.01,
        )


def test_formal_prediction_rejects_preview_snapshot(tmp_path: Path) -> None:
    preview, validator = _snapshot(tmp_path, ready=False)

    with pytest.raises(ValueError, match="ready snapshot"):
        build_score_prediction(
            snapshot=preview,
            snapshot_validator=validator,
            model_run_id=ModelRunId("model-run:dc-v1"),
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.2,
            lambda_away=1.0,
            rho=-0.1,
            max_goals=11,
            input_refs=(),
        )


def test_formal_prediction_must_be_generated_before_scheduled_kickoff(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    composition = _composition(snapshot, validator)

    with pytest.raises(ValueError, match="scheduled_kickoff_used"):
        build_score_prediction(
            snapshot=snapshot,
            snapshot_validator=validator,
            model_run_id=DEFAULT_MODEL_RUN_ID,
            model_version="dixon-coles/1",
            generated_at=snapshot.scheduled_kickoff_used,
            lambda_home=composition.lambda_home,
            lambda_away=composition.lambda_away,
            rho=-0.12,
            max_goals=11,
            input_refs=(),
            expected_goals=composition,
        )


def test_prediction_archive_rejects_post_kickoff_domain_object(tmp_path: Path) -> None:
    snapshot, archive, prediction, _ = _persisted_prediction_fixture(tmp_path)
    post_kickoff = _reidentify_prediction(
        replace(prediction, generated_at=snapshot.scheduled_kickoff_used)
    )

    with pytest.raises(ValueError, match="scheduled_kickoff_used"):
        archive.write_prediction(post_kickoff, snapshot=snapshot)


def test_formal_prediction_rejects_tampered_snapshot_identity(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    tampered = replace(snapshot, id=SnapshotId("snapshot:" + "d" * 64))

    with pytest.raises(ValueError, match="snapshot identity"):
        build_score_prediction(
            snapshot=tampered,
            snapshot_validator=validator,
            model_run_id=ModelRunId("model-run:dc-v1"),
            model_version="dixon-coles/1",
            generated_at=GENERATED_AT,
            lambda_home=1.2,
            lambda_away=1.0,
            rho=-0.1,
            max_goals=11,
            input_refs=(),
        )


def test_prediction_loader_rejects_unknown_or_tampered_schema(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    prediction = _prediction((snapshot, validator))
    payload = prediction_payload(prediction)

    assert (
        parse_prediction_payload(
            payload,
            snapshot=snapshot,
            snapshot_validator=validator,
        )
        == prediction
    )
    for unsupported_version in (1, 99):
        with pytest.raises(ValueError, match="unsupported"):
            parse_prediction_payload(
                {**payload, "schema_version": unsupported_version},
                snapshot=snapshot,
                snapshot_validator=validator,
            )
    tampered = {**payload, "score_cells": [*payload["score_cells"]]}
    tampered["score_cells"][0] = {
        **tampered["score_cells"][0],
        "probability": 0.99,
    }
    with pytest.raises(ValueError, match="canonical score grid"):
        parse_prediction_payload(
            tampered,
            snapshot=snapshot,
            snapshot_validator=validator,
        )


def test_prediction_loader_rejects_contribution_source_outside_snapshot(tmp_path: Path) -> None:
    snapshot, validator = _snapshot(tmp_path)
    composition = _composition(snapshot, validator)
    prediction = build_score_prediction(
        snapshot=snapshot,
        snapshot_validator=validator,
        model_run_id=DEFAULT_MODEL_RUN_ID,
        model_version="dixon-coles/1",
        generated_at=GENERATED_AT,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.12,
        max_goals=11,
        input_refs=("baseline:v1",),
        expected_goals=composition,
        calibration_versions=composition.calibration_versions,
    )
    payload = prediction_payload(prediction)
    unrelated_source_ref = "derived-source:" + "f" * 64
    tampered = {
        **payload,
        "contribution_multipliers": [
            {
                **item,
                "source_ref": unrelated_source_ref,
            }
            for item in payload["contribution_multipliers"]
        ],
        "input_refs": sorted({*payload["input_refs"], unrelated_source_ref}),
    }

    with pytest.raises(ValueError, match="verified snapshot feature sources"):
        parse_prediction_payload(
            tampered,
            snapshot=snapshot,
            snapshot_validator=validator,
        )
