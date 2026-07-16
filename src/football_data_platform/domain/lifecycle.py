"""Pure lifecycle and task-specific training qualification validators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from football_data_platform.domain.models import MatchStatus, require_utc
from football_data_platform.domain.snapshots import CaptureMode, SnapshotType


class LifecycleState(StrEnum):
    DISCOVERED = "discovered"
    T24_READY = "t24-ready"
    LINEUPS_READY = "lineups-ready"
    FINISHED_STATS_PENDING = "finished-stats-pending"
    ARCHIVED_COMPLETE = "archived-complete"
    TRAINING_READY = "training-ready"


class Qualification(StrEnum):
    SCORE_MODEL = "score-model-ready"
    TEAM_BASELINE = "team-baseline-ready"
    PLAYER_PROFILE = "player-profile-ready"


@dataclass(frozen=True, slots=True)
class SnapshotAvailability:
    snapshot_type: SnapshotType
    capture_mode: CaptureMode
    quality_status: str


@dataclass(frozen=True, slots=True)
class MatchAvailability:
    match_status: MatchStatus
    team_ids: tuple[str, str]
    snapshots: tuple[SnapshotAvailability, ...]
    result_90_present: bool
    team_stat_fields: dict[str, frozenset[str]]
    starters: dict[str, frozenset[str]]
    player_observation_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class QualificationResult:
    qualification: Qualification
    ruleset_version: str
    passed: bool
    reason_codes: tuple[str, ...]
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class LifecycleAssessment:
    state: LifecycleState
    qualifications: tuple[QualificationResult, ...]
    reason_codes: tuple[str, ...]
    evaluated_at: datetime


def assess_lifecycle(
    availability: MatchAvailability,
    *,
    evaluated_at: datetime,
    ruleset_version: str = "readiness/1",
) -> LifecycleAssessment:
    """Compute lifecycle and three independent training qualifications."""

    require_utc(evaluated_at, "evaluated_at")
    qualifications = (
        _score_model_qualification(availability, evaluated_at, ruleset_version),
        _team_baseline_qualification(availability, evaluated_at, ruleset_version),
        _player_profile_qualification(availability, evaluated_at, ruleset_version),
    )
    ready_snapshots = {
        snapshot.snapshot_type
        for snapshot in availability.snapshots
        if snapshot.quality_status == "ready"
    }
    archive_complete = _archive_complete(availability)
    reasons: list[str] = []
    if availability.match_status is MatchStatus.FINISHED:
        if not archive_complete:
            state = LifecycleState.FINISHED_STATS_PENDING
            reasons.append("post_match_archive_incomplete")
        elif any(result.passed for result in qualifications):
            state = LifecycleState.TRAINING_READY
        else:
            state = LifecycleState.ARCHIVED_COMPLETE
            reasons.append("no_training_qualification_passed")
    elif SnapshotType.LINEUPS_CONFIRMED in ready_snapshots:
        state = LifecycleState.LINEUPS_READY
    elif SnapshotType.T24H in ready_snapshots:
        state = LifecycleState.T24_READY
    else:
        state = LifecycleState.DISCOVERED
    return LifecycleAssessment(
        state=state,
        qualifications=qualifications,
        reason_codes=tuple(reasons),
        evaluated_at=evaluated_at,
    )


def _score_model_qualification(
    availability: MatchAvailability,
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    reasons: list[str] = []
    if not availability.result_90_present:
        reasons.append("missing_result_90")
    if not any(snapshot.quality_status == "ready" for snapshot in availability.snapshots):
        reasons.append("missing_ready_prematch_snapshot")
    return _qualification(Qualification.SCORE_MODEL, reasons, evaluated_at, ruleset_version)


def _team_baseline_qualification(
    availability: MatchAvailability,
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    reasons: list[str] = []
    if not availability.result_90_present:
        reasons.append("missing_result_90")
    required = frozenset({"goals", "xg", "shots", "shots_on_target"})
    for team_id in availability.team_ids:
        missing = sorted(required - availability.team_stat_fields.get(team_id, frozenset()))
        reasons.extend(f"missing_team_stat:{team_id}:{field}" for field in missing)
    return _qualification(Qualification.TEAM_BASELINE, reasons, evaluated_at, ruleset_version)


def _player_profile_qualification(
    availability: MatchAvailability,
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    reasons: list[str] = []
    for team_id in availability.team_ids:
        starters = availability.starters.get(team_id, frozenset())
        if len(starters) != 11:
            reasons.append(f"starter_count:{team_id}:{len(starters)}")
        missing_observations = sorted(starters - availability.player_observation_ids)
        reasons.extend(
            f"missing_player_observation:{player_id}" for player_id in missing_observations
        )
    return _qualification(Qualification.PLAYER_PROFILE, reasons, evaluated_at, ruleset_version)


def _qualification(
    qualification: Qualification,
    reasons: list[str],
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    return QualificationResult(
        qualification=qualification,
        ruleset_version=ruleset_version,
        passed=not reasons,
        reason_codes=tuple(reasons),
        evaluated_at=evaluated_at,
    )


def _archive_complete(availability: MatchAvailability) -> bool:
    if not availability.result_90_present:
        return False
    if any(team_id not in availability.team_stat_fields for team_id in availability.team_ids):
        return False
    return all(team_id in availability.starters for team_id in availability.team_ids)
