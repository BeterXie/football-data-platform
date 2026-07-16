from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.domain.ids import MatchId, TeamId
from football_data_platform.domain.lifecycle import (
    LifecycleState,
    MatchAvailability,
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
)
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout

KICKOFF = datetime(2025, 8, 16, 14, 0, tzinfo=UTC)
HOME = TeamId("team:home")
AWAY = TeamId("team:away")
MATCH = MatchId("match:test")


def _feature(
    name: str = "rest_days",
    *,
    known_at: datetime | None = None,
    contribution_key: str = "home:rest",
    entity_id: str | None = None,
) -> SnapshotFeature:
    return SnapshotFeature(
        name=name,
        value=6,
        known_at=known_at or KICKOFF - timedelta(days=2),
        source_ref="canonical:fact-1",
        contribution_key=contribution_key,
        entity_id=entity_id,
    )


def test_t24_snapshot_rejects_future_information() -> None:
    as_of = KICKOFF - timedelta(hours=24)

    with pytest.raises(ValueError, match="known after"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.T24H,
            capture_mode=CaptureMode.RECONSTRUCTED,
            as_of=as_of,
            observed_at=KICKOFF + timedelta(days=1),
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="features/1",
            features=(_feature(known_at=as_of + timedelta(minutes=1)),),
            home_team_id=HOME,
            away_team_id=AWAY,
        )


def test_historical_snapshot_cannot_masquerade_as_captured() -> None:
    as_of = KICKOFF - timedelta(hours=24)

    with pytest.raises(ValueError, match="observed_at == as_of"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.T24H,
            capture_mode=CaptureMode.CAPTURED,
            as_of=as_of,
            observed_at=KICKOFF + timedelta(days=1),
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="features/1",
            features=(_feature(),),
            home_team_id=HOME,
            away_team_id=AWAY,
        )


def test_lineups_snapshot_requires_both_official_lineups() -> None:
    as_of = KICKOFF - timedelta(hours=1)
    home_lineup = _feature(
        "official_lineup_confirmed",
        known_at=as_of,
        contribution_key="home:lineup",
        entity_id=HOME.value,
    )

    with pytest.raises(ValueError, match="both teams"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            capture_mode=CaptureMode.RECONSTRUCTED,
            as_of=as_of,
            observed_at=KICKOFF + timedelta(days=1),
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="features/1",
            features=(home_lineup,),
            home_team_id=HOME,
            away_team_id=AWAY,
        )


def test_snapshot_is_content_identified_and_idempotently_archived(tmp_path: Path) -> None:
    as_of = KICKOFF - timedelta(hours=24)
    snapshot = build_snapshot(
        match_id=MATCH,
        match_version=1,
        snapshot_type=SnapshotType.T24H,
        capture_mode=CaptureMode.RECONSTRUCTED,
        as_of=as_of,
        observed_at=KICKOFF + timedelta(days=1),
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="features/1",
        features=(_feature(),),
        home_team_id=HOME,
        away_team_id=AWAY,
    )
    archive = DerivedArchive(DataLayout(tmp_path / "data"))

    first = archive.write_snapshot(snapshot)
    second = archive.write_snapshot(snapshot)

    assert first == second
    assert archive.load_snapshot_payload(snapshot)["capture_mode"] == "reconstructed"


def test_snapshot_identity_is_independent_of_feature_input_order() -> None:
    as_of = KICKOFF - timedelta(hours=24)
    first = _feature(contribution_key="a")
    second = _feature(name="travel", contribution_key="b")
    arguments = {
        "match_id": MATCH,
        "match_version": 1,
        "snapshot_type": SnapshotType.T24H,
        "capture_mode": CaptureMode.RECONSTRUCTED,
        "as_of": as_of,
        "observed_at": KICKOFF + timedelta(days=1),
        "scheduled_kickoff_used": KICKOFF,
        "feature_spec_version": "features/1",
        "home_team_id": HOME,
        "away_team_id": AWAY,
    }

    ordered = build_snapshot(features=(first, second), **arguments)
    reversed_order = build_snapshot(features=(second, first), **arguments)

    assert ordered.id == reversed_order.id


def test_qualifications_are_independent_and_explain_missing_data() -> None:
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

    assert assessment.state is LifecycleState.TRAINING_READY
    assert by_name[Qualification.SCORE_MODEL].passed
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
