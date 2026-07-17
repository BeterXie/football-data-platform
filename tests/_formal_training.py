from __future__ import annotations

import json
from datetime import datetime, timedelta

from _formal_context import seed_formal_context

from football_data_platform.domain.lifecycle import Qualification
from football_data_platform.domain.snapshots import (
    CURRENT_FEATURE_SPEC_VERSION,
    PreMatchSnapshot,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
)
from football_data_platform.domain.training import TrainingSample
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
)
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.training import TrainingArtifactStore


def seed_formal_score_sample(
    layout: DataLayout,
    *,
    key: str,
    kickoff: datetime,
    observed_at: datetime,
    evaluated_at: datetime,
    team_indices: tuple[int, int, int, int],
    goals: tuple[int, int],
    split: str,
) -> tuple[TrainingSample, PreMatchSnapshot]:
    """Persist and replay the complete formal evidence chain for one score sample."""

    as_of = kickoff - timedelta(hours=24)
    result_known_at = kickoff + timedelta(hours=2)
    if observed_at < result_known_at or evaluated_at < observed_at:
        raise ValueError("formal sample observation/evaluation must follow its typed result")

    formal = seed_formal_context(
        layout,
        as_of=as_of,
        kickoff=kickoff,
        key=key,
        observed_at=observed_at,
        team_indices=team_indices,
    )
    evidence = formal.raw.archive(
        f"{key} baseline evidence".encode(),
        source="test-source",
        source_id=f"{key}-baseline",
        url=f"fixture://{key}-baseline",
        observed_at=observed_at,
        target_event_time=as_of - timedelta(days=1),
        collector_version="test-formal-training/1",
        media_type="application/octet-stream",
    )
    formal.canonical.register_raw_asset(evidence)
    baseline = build_team_baseline(
        (
            TeamMatchProcess(
                f"{key}-baseline-match",
                formal.home_team_id.value,
                formal.away_team_id.value,
                as_of - timedelta(days=30),
                as_of - timedelta(days=1),
                1.4,
                0.9,
                evidence.id.value,
            ),
        ),
        as_of=as_of,
        half_life_days=90.0,
        iterations=2,
    ).artifact
    formal.derived.write_team_baseline(baseline, generated_at=observed_at)
    baseline_ref = formal.derived.write_team_baseline_source(
        match_id=formal.match_id,
        match_version=formal.match_version,
        as_of=as_of,
        baseline_artifact_id=baseline.artifact_id,
    )
    baseline_validation = formal.derived.validate_snapshot_source(baseline_ref)
    snapshot = build_snapshot(
        match_id=formal.match_id,
        match_version=formal.match_version,
        snapshot_type=SnapshotType.T24H,
        as_of=as_of,
        scheduled_kickoff_used=kickoff,
        feature_spec_version=CURRENT_FEATURE_SPEC_VERSION,
        features=(
            SnapshotFeature(
                name="team_baseline",
                value=baseline_validation.value,
                known_at=baseline_validation.known_at,
                source_ref=baseline_ref,
                contribution_key="team-baseline",
            ),
            SnapshotFeature(
                name="match_context",
                value=formal.context_value,
                known_at=formal.context_known_at,
                source_ref=formal.context_ref,
                contribution_key="match-context",
            ),
        ),
        home_team_id=formal.home_team_id,
        away_team_id=formal.away_team_id,
        source_validator=formal.derived,
    )
    formal.derived.write_snapshot(snapshot)

    result_asset = formal.raw.archive(
        json.dumps(
            {"home_goals": goals[0], "away_goals": goals[1]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        source="test-source",
        source_id=f"{key}-result",
        url=f"fixture://{key}-result",
        observed_at=observed_at,
        target_event_time=kickoff,
        collector_version="test-formal-training/1",
        media_type="application/json",
    )
    formal.canonical.register_raw_asset(result_asset)
    result = CanonicalFactStore(formal.canonical).append_result_90(
        match_id=formal.match_id,
        match_version=formal.match_version,
        home_goals=goals[0],
        away_goals=goals[1],
        known_at=result_known_at,
        observed_at=observed_at,
        raw_asset_id=result_asset.id,
    )
    store = TrainingArtifactStore(layout)
    qualification = store.create_training_qualification(
        match_id=formal.match_id,
        match_version=formal.match_version,
        qualification=Qualification.SCORE_MODEL,
        ruleset_version="readiness/1",
        evaluated_at=evaluated_at,
        snapshot_ref=snapshot.id.value,
        result_ref=result.record_id,
    )
    sample = store.build_formal_score_sample(
        sample_id=f"sample:{key}",
        qualification_ref=qualification.qualification_id,
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        split=split,
    )
    return sample, snapshot
