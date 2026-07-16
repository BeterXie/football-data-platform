from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.domain.ids import MatchId, SnapshotId, TeamId
from football_data_platform.domain.lifecycle import (
    LifecycleState,
    MatchAvailability,
    PlayerObservationAvailability,
    Qualification,
    SnapshotAvailability,
    assess_lifecycle,
)
from football_data_platform.domain.models import MatchStatus
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotType,
    build_snapshot,
    verify_snapshot,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ChecksumMismatchError, RawArchive

KICKOFF = datetime(2025, 8, 16, 14, 0, tzinfo=UTC)
OBSERVED_AT = KICKOFF + timedelta(days=1)
HOME = TeamId("team:home")
AWAY = TeamId("team:away")
MATCH = MatchId("match:test")


def _stores(tmp_path: Path, *, observed_at: datetime = OBSERVED_AT):
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"fixture evidence",
        source="test-source",
        source_id="fixture-evidence",
        url="fixture://evidence",
        observed_at=observed_at,
        target_event_time=None,
        collector_version="test-collector/1",
        media_type="application/octet-stream",
    )
    return raw, DerivedArchive(layout), asset


def _source(derived: DerivedArchive, asset, value, transform: str) -> str:
    return derived.write_snapshot_source(
        value=value,
        input_refs=(asset.id,),
        transform_version=transform,
        generated_at=asset.observed_at,
    )


def _t24_features(
    derived: DerivedArchive,
    asset,
    *,
    known_at: datetime | None = None,
) -> tuple[SnapshotFeature, SnapshotFeature]:
    feature_known_at = known_at or KICKOFF - timedelta(days=2)
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "match-snapshot-baseline",
                HOME.value,
                AWAY.value,
                KICKOFF - timedelta(days=30),
                KICKOFF - timedelta(days=3),
                1.5,
                1.0,
                asset.id.value,
            ),
        ),
        as_of=KICKOFF - timedelta(days=2),
        half_life_days=90,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=HOME.value,
        away_team_id=AWAY.value,
    )
    baseline = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    context = {"days_since_previous_match": 6.0}
    return (
        SnapshotFeature(
            "team_baseline",
            baseline,
            feature_known_at,
            _source(derived, asset, baseline, "team-baseline-input/2"),
            "team-baseline",
        ),
        SnapshotFeature(
            "match_context",
            context,
            feature_known_at,
            _source(derived, asset, context, "match-context-input/1"),
            "context:rest-days",
        ),
    )


def _snapshot_arguments(derived: DerivedArchive) -> dict:
    return {
        "match_id": MATCH,
        "match_version": 1,
        "snapshot_type": SnapshotType.T24H,
        "as_of": KICKOFF - timedelta(hours=24),
        "scheduled_kickoff_used": KICKOFF,
        "feature_spec_version": "prematch-features/1",
        "home_team_id": HOME,
        "away_team_id": AWAY,
        "source_validator": derived,
    }


def test_t24_snapshot_rejects_future_information(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=24)

    with pytest.raises(ValueError, match="known after"):
        build_snapshot(
            features=_t24_features(derived, asset, known_at=as_of + timedelta(minutes=1)),
            **_snapshot_arguments(derived),
        )


def test_team_baseline_source_rejects_fake_artifact_identity(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline = build_team_baseline(
        (
            TeamMatchProcess(
                "match-a",
                "team:home",
                "team:away",
                KICKOFF - timedelta(days=30),
                KICKOFF - timedelta(days=30),
                1.5,
                1.0,
                asset.id.value,
            ),
        ),
        as_of=KICKOFF - timedelta(days=1),
        half_life_days=90,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline)
    payload = {
        "artifact_id": "team-baseline:" + "f" * 64,
        "artifact": team_baseline_payload(baseline),
        "lambda_home": 1.5,
        "lambda_away": 1.0,
    }
    source_ref = derived.write_snapshot_source(
        value=payload,
        input_refs=(asset.id,),
        transform_version="team-baseline-input/2",
        generated_at=asset.observed_at,
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        derived.validate_snapshot_source(source_ref)


def test_raw_observation_time_cannot_self_authorize_captured_mode(tmp_path: Path) -> None:
    as_of = KICKOFF - timedelta(hours=24)
    _, derived, asset = _stores(tmp_path, observed_at=as_of)

    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )

    assert snapshot.capture_mode is CaptureMode.RECONSTRUCTED


def test_source_validator_rejects_modified_raw_content(tmp_path: Path) -> None:
    raw, derived, asset = _stores(tmp_path)
    features = _t24_features(derived, asset)
    raw.layout.raw_object_path(asset.checksum).write_bytes(b"modified")

    with pytest.raises(ChecksumMismatchError, match="checksum"):
        build_snapshot(features=features, **_snapshot_arguments(derived))


def test_source_validator_rejects_missing_or_mismatched_derived_source(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, context = _t24_features(derived, asset)
    mismatched = replace(
        baseline,
        value={**baseline.value, "lambda_home": 9.0},
    )

    with pytest.raises(ValueError, match="derived source value"):
        build_snapshot(features=(mismatched, context), **_snapshot_arguments(derived))
    wrong_transform = replace(
        context,
        source_ref=_source(derived, asset, context.value, "unexpected-transform/1"),
    )
    with pytest.raises(ValueError, match="source transform"):
        build_snapshot(
            features=(baseline, wrong_transform),
            **_snapshot_arguments(derived),
        )
    with pytest.raises(FileNotFoundError):
        build_snapshot(
            features=(replace(baseline, source_ref="derived-source:" + "f" * 64), context),
            **_snapshot_arguments(derived),
        )


def test_feature_spec_computes_preview_instead_of_trusting_missing_fields(
    tmp_path: Path,
) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, _ = _t24_features(derived, asset)

    snapshot = build_snapshot(features=(baseline,), **_snapshot_arguments(derived))

    assert snapshot.quality_status == "preview"
    assert snapshot.missing_fields == ("feature:match_context",)


def _lineup_feature(
    derived: DerivedArchive,
    asset,
    *,
    team_id: TeamId,
    value,
    known_at: datetime,
) -> SnapshotFeature:
    return SnapshotFeature(
        "official_lineup_confirmed",
        value,
        known_at,
        _source(derived, asset, value, "official-lineup-input/1"),
        f"official-lineup:{team_id.value}",
        team_id.value,
    )


def test_lineups_snapshot_requires_both_official_lineups(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home = _lineup_feature(
        derived,
        asset,
        team_id=HOME,
        value=[f"player:home-{index}" for index in range(11)],
        known_at=as_of,
    )

    with pytest.raises(ValueError, match="both teams"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/1",
            features=(*_t24_features(derived, asset), home),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_lineups_snapshot_rejects_null_or_invalid_player_ids(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    away = _lineup_feature(
        derived,
        asset,
        team_id=AWAY,
        value=[f"player:away-{index}" for index in range(11)],
        known_at=as_of,
    )
    arguments = {
        "match_id": MATCH,
        "match_version": 1,
        "snapshot_type": SnapshotType.LINEUPS_CONFIRMED,
        "as_of": as_of,
        "scheduled_kickoff_used": KICKOFF,
        "feature_spec_version": "prematch-features/1",
        "home_team_id": HOME,
        "away_team_id": AWAY,
        "source_validator": derived,
    }

    for invalid in (None, ["not-a-player"] * 11):
        home = _lineup_feature(
            derived,
            asset,
            team_id=HOME,
            value=invalid,
            known_at=as_of,
        )
        with pytest.raises(ValueError, match="11 unique platform player IDs"):
            build_snapshot(
                features=(*_t24_features(derived, asset), home, away),
                **arguments,
            )


def test_snapshot_is_content_identified_and_idempotently_archived(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )

    assert derived.write_snapshot(snapshot) == derived.write_snapshot(snapshot)
    assert derived.load_snapshot_payload(snapshot)["capture_mode"] == "reconstructed"


def test_snapshot_identity_is_independent_of_feature_input_order(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    first, second = _t24_features(derived, asset)

    ordered = build_snapshot(features=(first, second), **_snapshot_arguments(derived))
    reversed_order = build_snapshot(features=(second, first), **_snapshot_arguments(derived))

    assert ordered.id == reversed_order.id


def test_snapshot_verifier_rejects_tampered_identity_and_quality(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, context = _t24_features(derived, asset)
    ready = build_snapshot(features=(baseline, context), **_snapshot_arguments(derived))
    preview = build_snapshot(features=(baseline,), **_snapshot_arguments(derived))

    with pytest.raises(ValueError, match="snapshot identity"):
        verify_snapshot(
            replace(ready, id=SnapshotId("snapshot:" + "f" * 64)),
            source_validator=derived,
        )
    with pytest.raises(ValueError, match="snapshot quality"):
        verify_snapshot(
            replace(preview, quality_status="ready", missing_fields=()),
            source_validator=derived,
        )


def test_qualifications_are_independent_but_cannot_skip_incomplete_archive() -> None:
    eleven_home = frozenset(f"home-player-{index}" for index in range(11))
    eleven_away = frozenset(f"away-player-{index}" for index in range(11))
    availability = MatchAvailability(
        match_status=MatchStatus.FINISHED,
        team_ids=(HOME.value, AWAY.value),
        snapshots=(
            SnapshotAvailability(
                SnapshotType.T24H,
                CaptureMode.RECONSTRUCTED,
                "ready",
            ),
        ),
        result_90_present=True,
        team_stat_fields={
            HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            AWAY.value: frozenset({"goals", "shots"}),
        },
        starters={HOME.value: eleven_home, AWAY.value: eleven_away},
        player_observation_ids=eleven_home | eleven_away,
    )

    assessment = assess_lifecycle(
        availability,
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    by_name = {result.qualification: result for result in assessment.qualifications}

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)
    assert not by_name[Qualification.SCORE_MODEL].passed
    assert "unverified_snapshot_evidence" in by_name[Qualification.SCORE_MODEL].reason_codes
    assert not by_name[Qualification.TEAM_BASELINE].passed
    assert by_name[Qualification.PLAYER_PROFILE].passed
    assert "missing_team_stat:team:away:xg" in by_name[Qualification.TEAM_BASELINE].reason_codes


def test_finished_but_incomplete_archive_is_pending() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={},
            starters={},
            player_observation_ids=frozenset(),
        ),
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)


def test_empty_team_stats_and_starters_cannot_archive_as_complete() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset(),
                AWAY.value: frozenset(),
            },
            starters={
                HOME.value: frozenset(),
                AWAY.value: frozenset(),
            },
            player_observation_ids=frozenset(),
        ),
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)


def test_unverified_snapshot_summary_cannot_advance_lifecycle() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.SCHEDULED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(
                SnapshotAvailability(
                    SnapshotType.LINEUPS_CONFIRMED,
                    CaptureMode.CAPTURED,
                    "ready",
                ),
            ),
            result_90_present=False,
            team_stat_fields={},
            starters={},
            player_observation_ids=frozenset(),
        ),
        evaluated_at=OBSERVED_AT,
    )
    score = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.SCORE_MODEL
    )

    assert assessment.state is LifecycleState.DISCOVERED
    assert "unverified_snapshot_evidence" in score.reason_codes


def test_persisted_verified_snapshot_advances_only_after_it_was_observed(
    tmp_path: Path,
) -> None:
    snapshot_time = KICKOFF - timedelta(hours=24)
    _, derived, asset = _stores(tmp_path, observed_at=snapshot_time)
    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )
    derived.write_snapshot(snapshot)
    availability = MatchAvailability(
        match_status=MatchStatus.SCHEDULED,
        team_ids=(HOME.value, AWAY.value),
        snapshots=(SnapshotAvailability.from_snapshot(snapshot),),
        result_90_present=False,
        team_stat_fields={},
        starters={},
        player_observation_ids=frozenset(),
        match_id=MATCH.value,
        match_version=1,
    )

    before = assess_lifecycle(
        availability,
        evaluated_at=snapshot_time - timedelta(seconds=1),
        snapshot_validator=derived,
    )
    available = assess_lifecycle(
        availability,
        evaluated_at=snapshot_time,
        snapshot_validator=derived,
    )

    assert before.state is LifecycleState.DISCOVERED
    assert available.state is LifecycleState.T24_READY


def test_empty_player_observations_do_not_pass_player_profile_readiness() -> None:
    starters = {
        HOME.value: frozenset(f"player:home-{index}" for index in range(11)),
        AWAY.value: frozenset(f"player:away-{index}" for index in range(11)),
    }
    details = {
        player_id: PlayerObservationAvailability(
            team_id=team_id,
            role="",
            minutes=0,
            metric_fields=frozenset(),
            known_at=OBSERVED_AT,
        )
        for team_id, player_ids in starters.items()
        for player_id in player_ids
    }
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters=starters,
            player_observation_ids=frozenset(details),
            player_observations=details,
        ),
        evaluated_at=OBSERVED_AT,
    )
    player_profile = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.PLAYER_PROFILE
    )

    assert not player_profile.passed
    assert any(
        code.startswith("missing_player_role_or_minutes:") for code in player_profile.reason_codes
    )
    assert any(code.startswith("missing_player_metrics:") for code in player_profile.reason_codes)


def test_unknown_readiness_ruleset_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported readiness ruleset"):
        assess_lifecycle(
            MatchAvailability(
                match_status=MatchStatus.SCHEDULED,
                team_ids=(HOME.value, AWAY.value),
                snapshots=(),
                result_90_present=False,
                team_stat_fields={},
                starters={},
                player_observation_ids=frozenset(),
            ),
            evaluated_at=OBSERVED_AT,
            ruleset_version="readiness/unknown",
        )


def test_future_availability_cutoff_cannot_be_used_for_an_earlier_assessment() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters={},
            player_observation_ids=frozenset(),
            as_of=OBSERVED_AT + timedelta(hours=1),
        ),
        evaluated_at=OBSERVED_AT,
    )
    team_baseline = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.TEAM_BASELINE
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert "availability_as_of_after_evaluation" in team_baseline.reason_codes
