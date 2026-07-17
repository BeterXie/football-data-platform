"""Offline-replayable Premier League 2025-26 vertical slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.lifecycle import (
    Qualification,
    SnapshotAvailability,
    assess_lifecycle,
)
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import (
    ModelRunArtifact,
    ModelRunStatus,
    TrainingDatasetManifest,
)
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contributions,
    lineup_delta_contributions,
)
from football_data_platform.features.lineup import (
    LINEUP_DELTA_INPUT_TRANSFORM_V3,
    lineup_delta_input_payload,
)
from football_data_platform.features.player_profiles import (
    PlayerMatchObservation,
    build_player_profiles,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
)
from football_data_platform.pipelines.match_report import ingest_fbref_match_report
from football_data_platform.pipelines.official_lineup import (
    ingest_official_lineup_json,
    replay_official_lineup_contract,
)
from football_data_platform.pipelines.schedule import (
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.reporting.vertical_slice import (
    render_vertical_slice_report,
    write_static_report,
)
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.sources.prematch import (
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
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


@dataclass(frozen=True, slots=True)
class VerticalSliceResult:
    run_id: str
    summary_path: Path
    report_path: Path
    t24_snapshot_id: str
    lineups_snapshot_id: str
    prediction_id: str | None
    canonical_counts: dict[str, int]


def run_offline_vertical_slice(
    *,
    data_root: Path,
    registry_path: Path,
    schedule_file: Path,
    first_match_report_file: Path,
    lineups_file: Path,
    observed_at: datetime,
    first_report_known_at: datetime,
    second_result_known_at: datetime,
    profile_minimum_minutes: float,
) -> VerticalSliceResult:
    """Replay the golden slice and retain a failed run manifest on errors."""

    try:
        return _run_offline_vertical_slice(
            data_root=data_root,
            registry_path=registry_path,
            schedule_file=schedule_file,
            first_match_report_file=first_match_report_file,
            lineups_file=lineups_file,
            observed_at=observed_at,
            first_report_known_at=first_report_known_at,
            second_result_known_at=second_result_known_at,
            profile_minimum_minutes=profile_minimum_minutes,
        )
    except Exception as error:
        _write_failed_run_manifest(
            data_root=data_root,
            input_files=(
                registry_path,
                schedule_file,
                first_match_report_file,
                lineups_file,
            ),
            observed_at=observed_at,
            profile_minimum_minutes=profile_minimum_minutes,
            error=error,
        )
        raise


def _run_offline_vertical_slice(
    *,
    data_root: Path,
    registry_path: Path,
    schedule_file: Path,
    first_match_report_file: Path,
    lineups_file: Path,
    observed_at: datetime,
    first_report_known_at: datetime,
    second_result_known_at: datetime,
    profile_minimum_minutes: float,
) -> VerticalSliceResult:
    """Replay a two-match golden slice through every architecture boundary."""

    layout = DataLayout(data_root).ensure()
    archive = RawArchive(layout)
    derived = DerivedArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    registry = load_competition_registry(registry_path)
    canonical.register_registry(registry, registered_at=observed_at)
    competition = registry.competitions[0]
    season = competition.seasons[0]

    schedule_ingest = ingest_fbref_schedule(
        schedule_file.read_bytes(),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
    )
    schedule_known_at = schedule_ingest.parsed.fixture_known_at
    if schedule_known_at is None:
        raise ValueError("golden schedule fixture requires explicit known-at metadata")
    if len(schedule_ingest.parsed.matches) < 2:
        raise ValueError("golden vertical slice requires at least two parsed matches")
    first_fixture, second_fixture = schedule_ingest.parsed.matches[:2]
    first_match_id = MatchId(schedule_ingest.canonical_match_ids[0])
    second_match_id = MatchId(schedule_ingest.canonical_match_ids[1])
    if first_fixture.kickoff_at is None or second_fixture.kickoff_at is None:
        raise ValueError("golden fixtures require known kickoffs")
    if second_fixture.home_goals is None or second_fixture.away_goals is None:
        raise ValueError("second golden fixture requires a real 90-minute result")
    if first_fixture.home_goals is None or first_fixture.away_goals is None:
        raise ValueError("first golden fixture requires a real 90-minute result")

    facts = CanonicalFactStore(canonical)
    first_result_fact = facts.append_result_90(
        match_id=first_match_id,
        match_version=1,
        home_goals=first_fixture.home_goals,
        away_goals=first_fixture.away_goals,
        known_at=first_report_known_at,
        observed_at=observed_at,
        raw_asset_id=RawAssetId(schedule_ingest.raw_asset_id),
    )
    report_ingest = ingest_fbref_match_report(
        first_match_report_file.read_bytes(),
        page_url=first_fixture.report_url or "https://fbref.invalid/missing-report-url",
        source_match_id=first_fixture.source_match_id or first_fixture.source_fixture_id,
        match_id=first_match_id,
        match_version=1,
        home_goals=first_fixture.home_goals,
        away_goals=first_fixture.away_goals,
        known_at=first_report_known_at,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
        required_tables=("summary",),
    )
    result_fact = facts.append_result_90(
        match_id=second_match_id,
        match_version=1,
        home_goals=second_fixture.home_goals,
        away_goals=second_fixture.away_goals,
        known_at=second_result_known_at,
        observed_at=observed_at,
        raw_asset_id=RawAssetId(schedule_ingest.raw_asset_id),
    )

    lineup_content = lineups_file.read_bytes()
    lineup_source = "golden-official-lineup-fixture"
    lineup_url = "fixture://official-lineups"
    lineup_sources = SourceRegistry(
        (
            SourceDescriptor(
                lineup_source,
                SourceKind.OFFICIAL_LINEUP,
                lineup_source,
                official=True,
            ),
        )
    )
    lineup_ingest = ingest_official_lineup_json(
        lineup_content,
        source=lineup_source,
        source_match_id=second_fixture.source_fixture_id,
        page_url=lineup_url,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
        source_registry=lineup_sources,
    )
    lineup_contract = replay_official_lineup_contract(
        lineup_ingest.contract_id,
        archive=archive,
        canonical=canonical,
    )
    if lineup_contract.match_id != second_match_id or lineup_contract.match_version != 1:
        raise ValueError("official lineup replay does not match the golden target fixture")
    lineup_known_at = lineup_contract.published_at
    lineup_asset_id = lineup_contract.raw_asset_id
    lineup_features: list[SnapshotFeature] = []
    lineup_player_ids_by_team: dict[str, tuple[str, ...]] = {}
    for team_id, contract_player_ids in lineup_contract.team_lineups:
        player_ids = tuple(player_id.value for player_id in contract_player_ids)
        lineup_player_ids_by_team[team_id.value] = player_ids
        lineup_source_ref = derived.write_official_lineup_source(
            contract_id=lineup_contract.contract_id,
            match_id=second_match_id,
            match_version=1,
            team_id=team_id,
            player_ids=contract_player_ids,
            known_at=lineup_known_at,
            observed_at=observed_at,
            raw_asset_id=lineup_asset_id,
        )
        lineup_features.append(
            SnapshotFeature(
                name="official_lineup_confirmed",
                value=list(player_ids),
                known_at=lineup_known_at,
                source_ref=lineup_source_ref,
                contribution_key=f"official-lineup:{team_id.value}",
                entity_id=team_id.value,
            )
        )

    first_teams = {team.source_team_id: team for team in report_ingest.parsed.teams}
    first_home = canonical.mapped_team(source="fbref", source_id=first_fixture.home_source_id)
    first_away = canonical.mapped_team(source="fbref", source_id=first_fixture.away_source_id)
    baseline_result = build_team_baseline(
        (
            TeamMatchProcess(
                match_id=first_match_id.value,
                home_team_id=first_home.id.value,
                away_team_id=first_away.id.value,
                kickoff_at=first_fixture.kickoff_at,
                known_at=first_report_known_at,
                home_xg=_required_stat(
                    first_teams[first_fixture.home_source_id].aggregated_stats, "xg"
                ),
                away_xg=_required_stat(
                    first_teams[first_fixture.away_source_id].aggregated_stats, "xg"
                ),
                source_ref=report_ingest.raw_asset_id,
            ),
        ),
        as_of=second_fixture.kickoff_at - timedelta(hours=24),
        half_life_days=90.0,
        iterations=4,
    )
    baseline = baseline_result.artifact

    player_observations: list[PlayerMatchObservation] = []
    for team_report in report_ingest.parsed.teams:
        team = canonical.mapped_team(source="fbref", source_id=team_report.source_team_id)
        for player_row in team_report.players:
            player = canonical.mapped_player(source="fbref", source_id=player_row.source_player_id)
            player_observations.append(
                PlayerMatchObservation(
                    player_id=player.id.value,
                    team_id=team.id.value,
                    match_id=first_match_id.value,
                    role=player_row.role,
                    minutes=player_row.minutes,
                    known_at=first_report_known_at,
                    metrics=player_row.metrics,
                    source_ref=report_ingest.raw_asset_id,
                )
            )
    profiles = build_player_profiles(
        tuple(player_observations),
        as_of=second_fixture.kickoff_at - timedelta(hours=24),
        minimum_minutes=profile_minimum_minutes,
    )

    second_home = canonical.mapped_team(source="fbref", source_id=second_fixture.home_source_id)
    second_away = canonical.mapped_team(source="fbref", source_id=second_fixture.away_source_id)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline,
        home_team_id=second_home.id.value,
        away_team_id=second_away.id.value,
    )
    t24_as_of = second_fixture.kickoff_at - timedelta(hours=24)
    derived.write_team_baseline(baseline, generated_at=observed_at)
    profile_manifest_path = derived.write_player_profiles(profiles, generated_at=observed_at)
    baseline_source_ref = derived.write_team_baseline_source(
        match_id=second_match_id,
        match_version=1,
        as_of=t24_as_of,
        baseline_artifact_id=baseline.artifact_id,
    )
    baseline_validation = derived.validate_snapshot_source(baseline_source_ref)
    baseline_value = baseline_validation.value
    baseline_feature = SnapshotFeature(
        name="team_baseline",
        value=baseline_value,
        known_at=baseline_validation.known_at,
        source_ref=baseline_source_ref,
        contribution_key="team-baseline",
    )
    context_source_ref = derived.write_match_context_source(
        match_id=second_match_id,
        match_version=1,
        as_of=t24_as_of,
    )
    context_validation = derived.validate_snapshot_source(context_source_ref)
    context_value = context_validation.value
    context_feature = SnapshotFeature(
        name="match_context",
        value=context_value,
        known_at=context_validation.known_at,
        source_ref=context_source_ref,
        contribution_key="match-context",
    )
    t24_snapshot = build_snapshot(
        match_id=second_match_id,
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        as_of=t24_as_of,
        scheduled_kickoff_used=second_fixture.kickoff_at,
        feature_spec_version="prematch-features/3",
        features=(baseline_feature, context_feature),
        home_team_id=second_home.id,
        away_team_id=second_away.id,
        source_validator=derived,
    )
    lineup_delta_value = {
        team_id: lineup_delta_input_payload(
            starter_ids=player_ids,
            reference_starter_ids=None,
            profiles=(),
            lineup_input_refs=(lineup_asset_id.value,),
        )
        for team_id, player_ids in lineup_player_ids_by_team.items()
    }
    lineup_delta_source_ref = derived.write_snapshot_source(
        value=lineup_delta_value,
        input_refs=(lineup_asset_id,),
        transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
        generated_at=observed_at,
        known_at=lineup_known_at,
    )
    lineup_delta_feature = SnapshotFeature(
        name="lineup_delta",
        value=lineup_delta_value,
        known_at=lineup_known_at,
        source_ref=lineup_delta_source_ref,
        contribution_key="lineup-delta",
    )
    lineup_context_source_ref = derived.write_match_context_source(
        match_id=second_match_id,
        match_version=1,
        as_of=lineup_known_at,
    )
    lineup_context_validation = derived.validate_snapshot_source(lineup_context_source_ref)
    lineup_context_feature = SnapshotFeature(
        name="match_context",
        value=lineup_context_validation.value,
        known_at=lineup_context_validation.known_at,
        source_ref=lineup_context_source_ref,
        contribution_key="match-context",
    )
    lineup_baseline_source_ref = derived.write_team_baseline_source(
        match_id=second_match_id,
        match_version=1,
        as_of=lineup_known_at,
        baseline_artifact_id=baseline.artifact_id,
    )
    lineup_baseline_validation = derived.validate_snapshot_source(lineup_baseline_source_ref)
    lineup_baseline_feature = SnapshotFeature(
        name="team_baseline",
        value=lineup_baseline_validation.value,
        known_at=lineup_baseline_validation.known_at,
        source_ref=lineup_baseline_source_ref,
        contribution_key="team-baseline",
    )
    lineups_snapshot = build_snapshot(
        match_id=second_match_id,
        match_version=1,
        snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
        as_of=lineup_known_at,
        scheduled_kickoff_used=second_fixture.kickoff_at,
        feature_spec_version="prematch-features/3",
        features=(
            lineup_baseline_feature,
            lineup_context_feature,
            lineup_delta_feature,
            *lineup_features,
        ),
        home_team_id=second_home.id,
        away_team_id=second_away.id,
        source_validator=derived,
    )
    derived.write_snapshot(t24_snapshot)
    derived.write_snapshot(lineups_snapshot)

    t24_composition = None
    if t24_snapshot.quality_status == "ready":
        context_effects = context_contributions(
            context_value,
            home_team_id=second_home.id.value,
            away_team_id=second_away.id.value,
            source_ref=context_source_ref,
            source_validator=derived,
        )
        t24_composition = compose_expected_goals(
            lambda_home,
            lambda_away,
            context_effects,
        )
    lineups_composition = None
    if lineups_snapshot.quality_status == "ready":
        lineup_effects = lineup_delta_contributions(
            lineup_delta_value,
            home_team_id=second_home.id.value,
            away_team_id=second_away.id.value,
            source_ref=lineup_delta_source_ref,
            source_validator=derived,
        )
        lineups_composition = compose_expected_goals(
            lambda_home,
            lambda_away,
            (
                *context_contributions(
                    lineup_context_validation.value,
                    home_team_id=second_home.id.value,
                    away_team_id=second_away.id.value,
                    source_ref=lineup_context_source_ref,
                    source_validator=derived,
                ),
                *lineup_effects,
            ),
        )
    prediction_snapshot = (
        lineups_snapshot if lineups_snapshot.quality_status == "ready" else t24_snapshot
    )
    composition = (
        lineups_composition if prediction_snapshot is lineups_snapshot else t24_composition
    )
    composition_calibration_versions = (
        ()
        if composition is None
        else tuple(sorted({item.version for item in composition.contributions}))
    )
    model_store = TrainingArtifactStore(layout)
    first_qualification = model_store.create_training_qualification(
        match_id=first_match_id,
        match_version=1,
        qualification=Qualification.SCORE_MODEL,
        ruleset_version="readiness/1",
        evaluated_at=observed_at,
        snapshot_ref=None,
        result_ref=first_result_fact.record_id,
    )
    second_qualification = model_store.create_training_qualification(
        match_id=second_match_id,
        match_version=1,
        qualification=Qualification.SCORE_MODEL,
        ruleset_version="readiness/1",
        evaluated_at=observed_at,
        snapshot_ref=prediction_snapshot.id.value,
        result_ref=result_fact.record_id,
    )
    training_sample = model_store.build_formal_score_sample(
        sample_id=f"sample:{first_match_id.value}:score-model",
        qualification_ref=first_qualification.qualification_id,
        feature_version="snapshot-score-features/1",
        split="train",
        as_of=first_fixture.kickoff_at - timedelta(hours=24),
    )
    holdout_sample = model_store.build_formal_score_sample(
        sample_id=f"sample:{second_match_id.value}:score-model",
        qualification_ref=second_qualification.qualification_id,
        feature_version="snapshot-score-features/1",
        split="holdout",
    )
    training_dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/vertical-slice-2",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="snapshot-score-features/1",
        label_version="result-90/1",
        as_of=prediction_snapshot.as_of,
        split_strategy="forward-chaining/1",
        samples=(training_sample, holdout_sample),
        generated_at=observed_at,
        transform_version="training-dataset/2",
        code_version=DERIVED_CODE_VERSION,
    )
    model_store.write_formal_dataset(training_dataset)
    model_run = ModelRunArtifact.create(
        model_version="dixon-coles/unavailable-no-eligible-train/1",
        run_role="shadow",
        task="score-model",
        dataset_id=training_dataset.dataset_id,
        feature_version=training_dataset.feature_version,
        label_version=training_dataset.label_version,
        algorithm="dixon-coles",
        parameters={"gate": "no_eligible_train_split"},
        code_version=DERIVED_CODE_VERSION,
        environment_version="python-runtime:locked",
        started_at=observed_at,
        ended_at=observed_at,
        random_seed=None,
        model_artifact_refs=(),
        evaluation_cohort=(),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(),
        output_hashes=(),
        status=ModelRunStatus.FAILED,
        error="no_eligible_train_split",
    )
    model_store.write_model_run(model_run)
    baseline_manifest_id = _manifest_id_for_output_ref(derived, baseline.artifact_id)
    profile_manifest_id = _manifest_id_from_path(profile_manifest_path)

    coverage = assess_season_coverage(
        schedule_ingest.parsed,
        season,
        canonical=canonical,
        source="fbref",
        archive=archive,
    )
    first_lifecycle = assess_lifecycle(
        facts.availability(first_match_id, as_of=observed_at),
        evaluated_at=observed_at,
    )
    second_lifecycle = assess_lifecycle(
        facts.availability(
            second_match_id,
            as_of=observed_at,
            snapshots=(
                SnapshotAvailability.from_snapshot(t24_snapshot),
                SnapshotAvailability.from_snapshot(lineups_snapshot),
            ),
        ),
        evaluated_at=observed_at,
        snapshot_validator=derived,
    )
    run_identity = {
        "schema_version": 3,
        "mode": "offline-golden-replay",
        "observed_at": _timestamp(observed_at),
        "snapshot_ids": [t24_snapshot.id.value, lineups_snapshot.id.value],
        "prediction_id": None,
        "training_dataset_id": training_dataset.dataset_id,
        "model_run_id": model_run.model_run_id,
        "model_gate": "failed:no_eligible_train_split",
        "result_fact_id": result_fact.record_id,
    }
    semantic_digest = hashlib.sha256(_canonical_json(run_identity)).hexdigest()
    summary_logical_ref = f"vertical-slice-summary:{semantic_digest}"
    report_logical_ref = f"vertical-slice-report:{semantic_digest}"
    run_manifest = RunManifest.create(
        run_type="offline-golden-replay",
        started_at=observed_at,
        ended_at=observed_at,
        generated_at=observed_at,
        transform_version="vertical-slice/4",
        code_version=DERIVED_CODE_VERSION,
        input_refs=(
            schedule_ingest.raw_asset_id,
            report_ingest.raw_asset_id,
            lineup_asset_id.value,
            training_dataset.dataset_id,
        ),
        output_refs=(
            baseline_manifest_id,
            profile_manifest_id,
            t24_snapshot.id.value,
            lineups_snapshot.id.value,
            first_qualification.qualification_id,
            second_qualification.qualification_id,
            training_dataset.dataset_id,
            model_run.model_run_id,
            summary_logical_ref,
            report_logical_ref,
        ),
        status="succeeded",
        error=None,
        quality="partial",
        parameters={
            "mode": "offline-golden-replay",
            "observed_at": _timestamp(observed_at),
            "profile_minimum_minutes": profile_minimum_minutes,
        },
        checkpoint="static-report-written",
        payload=run_identity,
    )
    run_id = run_manifest.run_id
    coverage_payload = asdict(coverage)
    coverage_payload["fixture_coverage"] = [item.to_payload() for item in coverage.fixture_coverage]
    coverage_payload["schedule_complete"] = coverage.schedule_complete
    coverage_payload["complete"] = coverage.complete
    coverage_payload["attempt_coverage_complete"] = coverage.attempt_coverage_complete
    coverage_payload["report_collection_complete"] = coverage.report_collection_complete
    coverage_payload["fixture_status_counts"] = coverage.fixture_status_counts
    coverage_payload["missing_fixture_ids"] = coverage.missing_fixture_ids
    coverage_payload["pending_fixture_ids"] = coverage.pending_fixture_ids
    coverage_payload["blocked_fixture_ids"] = coverage.blocked_fixture_ids
    coverage_payload["failed_fixture_ids"] = coverage.failed_fixture_ids
    coverage_payload["succeeded_fixture_ids"] = coverage.succeeded_fixture_ids
    coverage_payload["attempted_fixture_ids"] = coverage.attempted_fixture_ids
    coverage_payload["attempt_identity_mismatch_fixture_ids"] = (
        coverage.attempt_identity_mismatch_fixture_ids
    )
    coverage_payload["report_contract_diagnostics"] = coverage.report_contract_diagnostics
    if composition is None:
        lambda_composition_summary = {
            "available": False,
            "reason": "prediction_snapshot_not_ready",
            "baseline_lambda_home": lambda_home,
            "baseline_lambda_away": lambda_away,
            "composition_artifact_ref": None,
            "prediction_snapshot": prediction_snapshot.snapshot_type.value,
            "missing_fields": list(prediction_snapshot.missing_fields),
            "status": "unavailable",
        }
        prediction_unavailable_reason = "snapshot_not_ready:expected_goals_unavailable"
    else:
        lambda_composition_summary = {
            "available": True,
            "version": composition.composition_version,
            "baseline_lambda_home": composition.baseline_lambda_home,
            "baseline_lambda_away": composition.baseline_lambda_away,
            "lambda_home": composition.lambda_home,
            "lambda_away": composition.lambda_away,
            "contribution_keys": list(composition.contribution_keys),
            "input_refs": list(composition.input_refs),
            "calibration_versions": list(composition_calibration_versions),
            "contribution_multipliers": [
                {
                    "contribution_key": item.contribution_key,
                    "lambda_home_multiplier": item.lambda_home_multiplier,
                    "lambda_away_multiplier": item.lambda_away_multiplier,
                    "source_ref": item.source_ref,
                    "version": item.version,
                }
                for item in composition.contributions
            ],
            "composition_artifact_ref": None,
            "prediction_snapshot": prediction_snapshot.snapshot_type.value,
            "status": "candidate-features-only",
        }
        prediction_unavailable_reason = "model_run_failed:no_eligible_train_split"

    summary = {
        **run_identity,
        "run_id": run_id,
        "canonical_counts": canonical.counts(),
        "coverage": coverage_payload,
        "snapshots": [
            _snapshot_summary(t24_snapshot),
            _snapshot_summary(lineups_snapshot),
        ],
        "team_baseline": {
            "artifact_id": baseline.artifact_id,
            "input_refs": list(baseline.input_refs),
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
        },
        "lambda_composition": lambda_composition_summary,
        "player_profiles": {
            "count": len(profiles.profiles),
            "ready": sum(profile.quality_status == "ready" for profile in profiles.profiles),
        },
        "derived_artifacts": {
            "team_baseline": baseline_manifest_id,
            "player_profiles": profile_manifest_id,
            "evaluation": None,
            "training_dataset": training_dataset.dataset_id,
            "model_run": model_run.model_run_id,
            "model_artifact": None,
            "model_output": None,
            "training_qualifications": [
                first_qualification.qualification_id,
                second_qualification.qualification_id,
            ],
        },
        "training": {
            "schema_version": training_dataset.schema_version,
            "status": training_dataset.status.value,
            "eligible_train_samples": [
                sample.sample_id
                for sample in training_dataset.included_samples
                if sample.split == "train"
            ],
            "eligible_holdout_samples": [
                sample.sample_id
                for sample in training_dataset.included_samples
                if sample.split == "holdout"
            ],
            "excluded_samples": [
                {
                    "sample_id": sample.sample_id,
                    "reason_codes": list(sample.exclusion_reasons),
                }
                for sample in training_dataset.excluded_samples
            ],
            "model_gate": {
                "passed": False,
                "reason": model_run.error,
                "model_run_id": model_run.model_run_id,
            },
        },
        "prediction": {
            "available": False,
            "reason": prediction_unavailable_reason,
        },
        "evaluation": {
            "available": False,
            "reason": "prediction_unavailable",
            "capture_mode": CaptureMode.RECONSTRUCTED.value,
            "market_benchmark": {
                "available": False,
                "reason": "no real timestamped market snapshot",
            },
        },
        "lifecycle": {
            first_match_id.value: _lifecycle_summary(first_lifecycle),
            second_match_id.value: _lifecycle_summary(second_lifecycle),
        },
        "diagnostics": [
            "historical_snapshots_are_reconstructed",
            "lineups_snapshot_is_preview_when_player_profiles_are_missing",
            "market_benchmark_unavailable_without_real_market_snapshot",
            "formal_model_unavailable_without_an_eligible_training_split",
        ],
    }
    digest = run_id.removeprefix("run:")
    summary_path = layout.derived / "runs" / f"{digest}.json"
    _write_json_exact(summary_path, summary)
    report_path = layout.derived / "reports" / f"{digest}.md"
    report_content = render_vertical_slice_report(summary)
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report_content:
        raise ArchiveConflictError(f"derived report conflicts at {report_path}")
    write_static_report(report_path, report_content)
    derived.write_run_manifest(run_manifest)
    _register_static_outputs(
        derived=derived,
        run_manifest=run_manifest,
        summary=summary,
        summary_path=summary_path,
        report_content=report_content,
        report_path=report_path,
        summary_logical_ref=summary_logical_ref,
        report_logical_ref=report_logical_ref,
        generated_at=observed_at,
    )
    return VerticalSliceResult(
        run_id,
        summary_path,
        report_path,
        t24_snapshot.id.value,
        lineups_snapshot.id.value,
        None,
        canonical.counts(),
    )


def _register_static_outputs(
    *,
    derived: DerivedArchive,
    run_manifest: RunManifest,
    summary: dict[str, Any],
    summary_path: Path,
    report_content: str,
    report_path: Path,
    summary_logical_ref: str,
    report_logical_ref: str,
    generated_at: datetime,
) -> tuple[str, str, str]:
    """Register immutable report payloads without making the parent run ID self-referential."""

    summary_ref = _file_content_ref(summary_path)
    report_ref = _file_content_ref(report_path)
    summary_artifact = DerivedArtifactManifest.create(
        artifact_type="vertical-slice-summary",
        payload=summary,
        generated_at=generated_at,
        started_at=generated_at,
        ended_at=generated_at,
        transform_version="vertical-slice-report/1",
        code_version=DERIVED_CODE_VERSION,
        input_refs=(run_manifest.run_id, *run_manifest.input_refs),
        output_refs=(summary_logical_ref, summary_ref),
        quality=run_manifest.quality,
    )
    report_artifact = DerivedArtifactManifest.create(
        artifact_type="vertical-slice-report",
        payload={
            "run_id": run_manifest.run_id,
            "content_sha256": report_ref.removeprefix("file-sha256:"),
            "content": report_content,
        },
        generated_at=generated_at,
        started_at=generated_at,
        ended_at=generated_at,
        transform_version="vertical-slice-report/1",
        code_version=DERIVED_CODE_VERSION,
        input_refs=(run_manifest.run_id, summary_artifact.artifact_id),
        output_refs=(report_logical_ref, report_ref),
        quality=run_manifest.quality,
    )
    derived.write_artifact_manifest(summary_artifact)
    derived.write_artifact_manifest(report_artifact)
    registration = RunManifest.create(
        run_type="offline-golden-output-registration",
        started_at=generated_at,
        ended_at=generated_at,
        generated_at=generated_at,
        transform_version="vertical-slice-report/1",
        code_version=DERIVED_CODE_VERSION,
        input_refs=(run_manifest.run_id, summary_logical_ref, report_logical_ref),
        output_refs=(summary_artifact.artifact_id, report_artifact.artifact_id),
        status="succeeded",
        error=None,
        quality=run_manifest.quality,
        parameters={"parent_run_id": run_manifest.run_id},
        checkpoint="static-artifacts-registered",
        payload={
            "parent_run_id": run_manifest.run_id,
            "summary_artifact_id": summary_artifact.artifact_id,
            "report_artifact_id": report_artifact.artifact_id,
        },
    )
    derived.write_run_manifest(registration)
    return summary_artifact.artifact_id, report_artifact.artifact_id, registration.run_id


def _snapshot_summary(snapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id.value,
        "type": snapshot.snapshot_type.value,
        "capture_mode": snapshot.capture_mode.value,
        "quality_status": snapshot.quality_status,
        "missing_fields": list(snapshot.missing_fields),
    }


def _write_failed_run_manifest(
    *,
    data_root: Path,
    input_files: tuple[Path, ...],
    observed_at: datetime,
    profile_minimum_minutes: float,
    error: Exception,
) -> None:
    """Best-effort failure evidence that never masks the original exception."""

    try:
        input_refs = tuple(_file_content_ref(path) for path in input_files)
        manifest = RunManifest.create(
            run_type="offline-golden-replay",
            started_at=observed_at,
            ended_at=observed_at,
            generated_at=observed_at,
            transform_version="vertical-slice/4",
            code_version=DERIVED_CODE_VERSION,
            input_refs=input_refs,
            output_refs=(),
            status="failed",
            error=f"{type(error).__name__}: {error}",
            quality="failed",
            parameters={
                "mode": "offline-golden-replay",
                "observed_at": _timestamp(observed_at),
                "profile_minimum_minutes": profile_minimum_minutes,
            },
            checkpoint="before-completion",
        )
        DerivedArchive(DataLayout(data_root)).write_run_manifest(manifest)
    except Exception:
        return


def _file_content_ref(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return f"missing-file:{path.name}"
    return f"file-sha256:{digest}"


def _manifest_id_from_path(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_id = payload.get("id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError(f"derived manifest at {path} has no immutable ID")
    return manifest_id


def _manifest_id_for_output_ref(derived: DerivedArchive, output_ref: str) -> str:
    root = derived.layout.derived / "manifests" / "artifacts"
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if output_ref in payload.get("output_refs", ()):
            return _manifest_id_from_path(path)
    raise ValueError(f"no derived manifest cites output reference {output_ref!r}")


def _lifecycle_summary(assessment) -> dict[str, Any]:
    return {
        "state": assessment.state.value,
        "reason_codes": list(assessment.reason_codes),
        "qualifications": [
            {
                "qualification": item.qualification.value,
                "passed": item.passed,
                "reason_codes": list(item.reason_codes),
            }
            for item in assessment.qualifications
        ],
    }


def _required_stat(stats: dict[str, float | None], name: str) -> float:
    value = stats.get(name)
    if value is None:
        raise ValueError(f"golden report is missing required team stat {name}")
    return value


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be timezone-aware UTC")
    return parsed


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_json_exact(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ArchiveConflictError(f"derived run summary conflicts at {path}")
        return
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ArchiveConflictError(f"derived run summary conflicts at {path}") from None
