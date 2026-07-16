"""Offline-replayable Premier League 2025-26 vertical slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, ModelRunId, RawAssetId
from football_data_platform.domain.lifecycle import (
    SnapshotAvailability,
    assess_lifecycle,
)
from football_data_platform.domain.predictions import (
    MatchResult90,
    build_score_prediction,
    prediction_payload,
)
from football_data_platform.domain.snapshots import (
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.evaluation.metrics import evaluate_prediction
from football_data_platform.features.contributions import (
    compose_expected_goals,
    context_contribution,
    lineup_delta_contributions,
)
from football_data_platform.features.lineup import build_lineup_delta
from football_data_platform.features.player_profiles import (
    PlayerMatchObservation,
    build_player_profiles,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.pipelines.match_report import ingest_fbref_match_report
from football_data_platform.pipelines.schedule import (
    assess_season_coverage,
    ingest_fbref_schedule,
)
from football_data_platform.reporting.vertical_slice import (
    render_vertical_slice_report,
    write_static_report,
)
from football_data_platform.sources.fbref import schedule_url
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import (
    DERIVED_CODE_VERSION,
    DerivedArchive,
    RunManifest,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive


@dataclass(frozen=True, slots=True)
class VerticalSliceResult:
    run_id: str
    summary_path: Path
    report_path: Path
    t24_snapshot_id: str
    lineups_snapshot_id: str
    prediction_id: str
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
    facts.append_result_90(
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

    lineup_payload = json.loads(lineups_file.read_text(encoding="utf-8"))
    if lineup_payload.get("schema_version") != 1:
        raise ValueError("unsupported lineup fixture schema")
    lineup_known_at = _parse_timestamp(lineup_payload["published_at"])
    schedule_known_at = _parse_timestamp(lineup_payload["schedule_published_at"])
    lineup_asset = archive.archive(
        lineups_file.read_bytes(),
        source=str(lineup_payload["source"]),
        source_id=second_fixture.source_fixture_id,
        url="fixture://official-lineups",
        observed_at=observed_at,
        target_event_time=lineup_known_at,
        collector_version="manual-lineup-import/1",
        media_type="application/json",
    )
    canonical.register_raw_asset(lineup_asset)
    lineup_features: list[SnapshotFeature] = []
    lineup_player_ids_by_team: dict[str, tuple[str, ...]] = {}
    for team_payload in lineup_payload["teams"]:
        team = canonical.mapped_team(source="fbref", source_id=str(team_payload["source_team_id"]))
        player_ids: list[str] = []
        for player_payload in team_payload["starters"]:
            player = canonical.resolve_or_create_player(
                source=str(lineup_payload["source"]),
                source_id=str(player_payload["source_player_id"]),
                canonical_name=str(player_payload["name"]),
                observed_at=observed_at,
                raw_asset_id=lineup_asset.id,
            )
            player_ids.append(player.id.value)
            facts.append_lineup_fact(
                match_id=second_match_id,
                match_version=1,
                team_id=team.id,
                player_id=player.id,
                lineup_role="starter",
                official=True,
                known_at=lineup_known_at,
                observed_at=observed_at,
                raw_asset_id=lineup_asset.id,
            )
        if len(player_ids) != 11:
            raise ValueError(f"official lineup for {team.id} must contain 11 starters")
        lineup_player_ids_by_team[team.id.value] = tuple(player_ids)
        lineup_source_ref = derived.write_snapshot_source(
            value=player_ids,
            input_refs=(lineup_asset.id,),
            transform_version="official-lineup-input/1",
            generated_at=observed_at,
        )
        lineup_features.append(
            SnapshotFeature(
                name="official_lineup_confirmed",
                value=player_ids,
                known_at=lineup_known_at,
                source_ref=lineup_source_ref,
                contribution_key=f"official-lineup:{team.id.value}",
                entity_id=team.id.value,
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
    baseline_value = {
        "artifact_id": baseline.artifact_id,
        "artifact": team_baseline_payload(baseline),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    baseline_feature = SnapshotFeature(
        name="team_baseline",
        value=baseline_value,
        known_at=first_report_known_at,
        source_ref=derived.write_snapshot_source(
            value=baseline_value,
            input_refs=(RawAssetId(report_ingest.raw_asset_id),),
            transform_version="team-baseline-input/2",
            generated_at=observed_at,
        ),
        contribution_key="team-baseline",
    )
    context_value = {
        "days_since_previous_match": (
            second_fixture.kickoff_at - first_fixture.kickoff_at
        ).total_seconds()
        / 86_400
    }
    context_source_ref = derived.write_snapshot_source(
        value=context_value,
        input_refs=(RawAssetId(schedule_ingest.raw_asset_id),),
        transform_version="match-context-input/1",
        generated_at=observed_at,
    )
    context_feature = SnapshotFeature(
        name="match_context",
        value=context_value,
        known_at=max(schedule_known_at, first_report_known_at),
        source_ref=context_source_ref,
        contribution_key="context:rest-days",
    )
    t24_snapshot = build_snapshot(
        match_id=second_match_id,
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        as_of=t24_as_of,
        scheduled_kickoff_used=second_fixture.kickoff_at,
        feature_spec_version="prematch-features/1",
        features=(baseline_feature, context_feature),
        home_team_id=second_home.id,
        away_team_id=second_away.id,
        source_validator=derived,
    )
    lineup_deltas = {
        team_id: build_lineup_delta(
            starter_ids=player_ids,
            reference_starter_ids=None,
            player_values=(),
        )
        for team_id, player_ids in lineup_player_ids_by_team.items()
    }
    lineup_delta_value = {
        team_id: {
            "quality_status": delta.quality_status,
            "dimension_deltas": delta.dimension_deltas,
            "missing_fields": list(delta.missing_fields),
        }
        for team_id, delta in lineup_deltas.items()
    }
    lineup_delta_source_ref = derived.write_snapshot_source(
        value=lineup_delta_value,
        input_refs=(lineup_asset.id, RawAssetId(report_ingest.raw_asset_id)),
        transform_version="lineup-delta-input/1",
        generated_at=observed_at,
    )
    lineup_delta_feature = SnapshotFeature(
        name="lineup_delta",
        value=lineup_delta_value,
        known_at=lineup_known_at,
        source_ref=lineup_delta_source_ref,
        contribution_key="lineup-delta",
    )
    lineups_snapshot = build_snapshot(
        match_id=second_match_id,
        match_version=1,
        snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
        as_of=lineup_known_at,
        scheduled_kickoff_used=second_fixture.kickoff_at,
        feature_spec_version="prematch-features/1",
        features=(
            baseline_feature,
            context_feature,
            lineup_delta_feature,
            *lineup_features,
        ),
        home_team_id=second_home.id,
        away_team_id=second_away.id,
        source_validator=derived,
    )
    derived.write_snapshot(t24_snapshot)
    derived.write_snapshot(lineups_snapshot)

    context_effect = context_contribution(context_value, source_ref=context_source_ref)
    t24_composition = compose_expected_goals(
        lambda_home,
        lambda_away,
        (context_effect,),
    )
    lineup_effects = lineup_delta_contributions(
        lineup_delta_value,
        home_team_id=second_home.id.value,
        away_team_id=second_away.id.value,
        source_ref=lineup_delta_source_ref,
    )
    lineups_composition = compose_expected_goals(
        lambda_home,
        lambda_away,
        (context_effect, *lineup_effects),
    )
    prediction_snapshot = (
        lineups_snapshot if lineups_snapshot.quality_status == "ready" else t24_snapshot
    )
    composition = (
        lineups_composition if prediction_snapshot is lineups_snapshot else t24_composition
    )
    prediction = build_score_prediction(
        snapshot=prediction_snapshot,
        snapshot_validator=derived,
        model_run_id=ModelRunId("model-run:dixon-coles-baseline-v1"),
        model_version="dixon-coles-composed/1",
        generated_at=observed_at,
        lambda_home=composition.lambda_home,
        lambda_away=composition.lambda_away,
        rho=-0.1,
        max_goals=11,
        input_refs=(
            baseline.artifact_id,
            *composition.input_refs,
        ),
    )
    derived.write_prediction(prediction)
    evaluation = evaluate_prediction(
        prediction,
        MatchResult90(
            second_match_id,
            second_fixture.home_goals,
            second_fixture.away_goals,
            second_result_known_at,
            result_fact.record_id,
        ),
        evaluated_at=observed_at,
        result_validator=facts,
    )
    evaluation_manifest_path = derived.write_evaluation(evaluation, generated_at=observed_at)
    baseline_manifest_id = _manifest_id_for_output_ref(derived, baseline.artifact_id)
    profile_manifest_id = _manifest_id_from_path(profile_manifest_path)
    evaluation_manifest_id = _manifest_id_from_path(evaluation_manifest_path)

    coverage = assess_season_coverage(
        schedule_ingest.parsed,
        season,
        canonical=canonical,
        source="fbref",
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
        "schema_version": 1,
        "mode": "offline-golden-replay",
        "observed_at": _timestamp(observed_at),
        "snapshot_ids": [t24_snapshot.id.value, lineups_snapshot.id.value],
        "prediction_id": prediction.id.value,
        "result_fact_id": result_fact.record_id,
    }
    run_manifest = RunManifest.create(
        run_type="offline-golden-replay",
        started_at=observed_at,
        ended_at=observed_at,
        generated_at=observed_at,
        transform_version="vertical-slice/2",
        code_version=DERIVED_CODE_VERSION,
        input_refs=(
            schedule_ingest.raw_asset_id,
            report_ingest.raw_asset_id,
            lineup_asset.id.value,
        ),
        output_refs=(
            baseline_manifest_id,
            profile_manifest_id,
            evaluation_manifest_id,
            t24_snapshot.id.value,
            lineups_snapshot.id.value,
            prediction.id.value,
        ),
        status="succeeded",
        error=None,
        quality="partial" if lineups_snapshot.quality_status != "ready" else "ready",
        parameters={
            "mode": "offline-golden-replay",
            "observed_at": _timestamp(observed_at),
            "profile_minimum_minutes": profile_minimum_minutes,
        },
        checkpoint="static-report-written",
        payload=run_identity,
    )
    run_id = run_manifest.run_id
    summary = {
        **run_identity,
        "run_id": run_id,
        "canonical_counts": canonical.counts(),
        "coverage": {
            **asdict(coverage),
            "schedule_complete": coverage.schedule_complete,
            "complete": coverage.complete,
        },
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
        "lambda_composition": {
            "version": composition.composition_version,
            "baseline_lambda_home": composition.baseline_lambda_home,
            "baseline_lambda_away": composition.baseline_lambda_away,
            "lambda_home": composition.lambda_home,
            "lambda_away": composition.lambda_away,
            "contribution_keys": list(composition.contribution_keys),
            "input_refs": list(composition.input_refs),
            "prediction_snapshot": prediction_snapshot.snapshot_type.value,
        },
        "player_profiles": {
            "count": len(profiles.profiles),
            "ready": sum(profile.quality_status == "ready" for profile in profiles.profiles),
        },
        "derived_artifacts": {
            "team_baseline": baseline_manifest_id,
            "player_profiles": profile_manifest_id,
            "evaluation": evaluation_manifest_id,
        },
        "prediction": prediction_payload(prediction),
        "evaluation": _evaluation_summary(evaluation),
        "lifecycle": {
            first_match_id.value: _lifecycle_summary(first_lifecycle),
            second_match_id.value: _lifecycle_summary(second_lifecycle),
        },
        "diagnostics": [
            "historical_snapshots_are_reconstructed",
            "lineups_snapshot_is_preview_when_player_profiles_are_missing",
            "market_benchmark_unavailable_without_real_market_snapshot",
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
    return VerticalSliceResult(
        run_id,
        summary_path,
        report_path,
        t24_snapshot.id.value,
        lineups_snapshot.id.value,
        prediction.id.value,
        canonical.counts(),
    )


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
            transform_version="vertical-slice/2",
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


def _evaluation_summary(evaluation) -> dict[str, Any]:
    return {
        "prediction_id": evaluation.prediction_id,
        "capture_mode": evaluation.capture_mode.value,
        "actual_outcome": evaluation.actual_outcome,
        "actual_score": evaluation.actual_score,
        "result_brier": evaluation.result_brier,
        "result_log_loss": evaluation.result_log_loss,
        "score_log_loss": evaluation.score_log_loss,
        "market_benchmark": {
            "available": evaluation.market_benchmark.available,
            "reason": evaluation.market_benchmark.reason,
        },
        "input_refs": list(evaluation.input_refs),
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
