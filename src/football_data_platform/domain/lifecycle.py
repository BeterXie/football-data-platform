"""Pure lifecycle and task-specific training qualification validators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from football_data_platform.domain.models import MatchStatus, require_utc
from football_data_platform.domain.snapshots import (
    CaptureMode,
    PreMatchSnapshot,
    SnapshotSourceValidator,
    SnapshotType,
    verify_snapshot,
)


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
    evidence: PreMatchSnapshot | None = None

    @classmethod
    def from_snapshot(cls, snapshot: PreMatchSnapshot) -> SnapshotAvailability:
        """Create a summary that can be trusted only after store verification."""

        return cls(
            snapshot_type=snapshot.snapshot_type,
            capture_mode=snapshot.capture_mode,
            quality_status=snapshot.quality_status,
            evidence=snapshot,
        )


@dataclass(frozen=True, slots=True)
class PlayerObservationAvailability:
    team_id: str
    role: str
    minutes: float
    metric_fields: frozenset[str]
    known_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.team_id, str) or not self.team_id:
            raise ValueError("player observation team_id must be non-empty text")
        if not isinstance(self.role, str):
            raise TypeError("player observation role must be text")
        if not math.isfinite(float(self.minutes)) or self.minutes < 0:
            raise ValueError("player observation minutes must be finite and non-negative")
        if self.known_at is not None:
            require_utc(self.known_at, "player observation known_at")


@dataclass(frozen=True, slots=True)
class MatchAvailability:
    match_status: MatchStatus
    team_ids: tuple[str, str]
    snapshots: tuple[SnapshotAvailability, ...]
    result_90_present: bool
    team_stat_fields: dict[str, frozenset[str]]
    starters: dict[str, frozenset[str]]
    player_observation_ids: frozenset[str]
    match_id: str | None = None
    match_version: int | None = None
    result_90_known_at: datetime | None = None
    team_stats_known_at: dict[str, datetime] = field(default_factory=dict)
    player_observations: dict[str, PlayerObservationAvailability] | None = None
    as_of: datetime | None = None


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


_REQUIRED_TEAM_STAT_FIELDS = frozenset({"goals", "xg", "shots", "shots_on_target"})
_SUPPORTED_RULESET_VERSIONS = frozenset({"readiness/1"})
_PLAYER_REQUIRED_METRICS_BY_RULESET = {
    "readiness/1": {
        "forward": frozenset({"shots"}),
        "FW": frozenset({"shots"}),
        "MF": frozenset({"shots"}),
        "DF": frozenset({"shots"}),
        "GK": frozenset({"saves"}),
    }
}


def assess_lifecycle(
    availability: MatchAvailability,
    *,
    evaluated_at: datetime,
    ruleset_version: str = "readiness/1",
    snapshot_validator: SnapshotSourceValidator | None = None,
) -> LifecycleAssessment:
    """Compute lifecycle and three independent training qualifications."""

    require_utc(evaluated_at, "evaluated_at")
    if ruleset_version not in _SUPPORTED_RULESET_VERSIONS:
        raise ValueError(f"unsupported readiness ruleset {ruleset_version!r}")
    ready_snapshots, has_unverified_ready_snapshot = _verified_ready_snapshot_types(
        availability,
        evaluated_at=evaluated_at,
        snapshot_validator=snapshot_validator,
    )
    qualifications = (
        _score_model_qualification(
            availability,
            evaluated_at,
            ruleset_version,
            ready_snapshots=ready_snapshots,
            has_unverified_ready_snapshot=has_unverified_ready_snapshot,
        ),
        _team_baseline_qualification(availability, evaluated_at, ruleset_version),
        _player_profile_qualification(availability, evaluated_at, ruleset_version),
    )
    archive_complete = _archive_complete(availability, evaluated_at=evaluated_at)
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
    *,
    ready_snapshots: set[SnapshotType],
    has_unverified_ready_snapshot: bool,
) -> QualificationResult:
    reasons: list[str] = []
    if availability.match_status is not MatchStatus.FINISHED:
        reasons.append("match_not_finished")
    if _known_after(availability.as_of, evaluated_at):
        reasons.append("availability_as_of_after_evaluation")
    if not availability.result_90_present:
        reasons.append("missing_result_90")
    if not ready_snapshots:
        reasons.append("missing_ready_prematch_snapshot")
        if has_unverified_ready_snapshot:
            reasons.append("unverified_snapshot_evidence")
    if _known_after(availability.result_90_known_at, evaluated_at):
        reasons.append("result_known_after_evaluation")
    return _qualification(Qualification.SCORE_MODEL, reasons, evaluated_at, ruleset_version)


def _team_baseline_qualification(
    availability: MatchAvailability,
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    reasons: list[str] = []
    if availability.match_status is not MatchStatus.FINISHED:
        reasons.append("match_not_finished")
    if _known_after(availability.as_of, evaluated_at):
        reasons.append("availability_as_of_after_evaluation")
    if not availability.result_90_present:
        reasons.append("missing_result_90")
    if _known_after(availability.result_90_known_at, evaluated_at):
        reasons.append("result_known_after_evaluation")
    for team_id in availability.team_ids:
        missing = sorted(
            _REQUIRED_TEAM_STAT_FIELDS - availability.team_stat_fields.get(team_id, frozenset())
        )
        reasons.extend(f"missing_team_stat:{team_id}:{field}" for field in missing)
        if _known_after(availability.team_stats_known_at.get(team_id), evaluated_at):
            reasons.append(f"team_stats_known_after_evaluation:{team_id}")
    return _qualification(Qualification.TEAM_BASELINE, reasons, evaluated_at, ruleset_version)


def _player_profile_qualification(
    availability: MatchAvailability,
    evaluated_at: datetime,
    ruleset_version: str,
) -> QualificationResult:
    reasons: list[str] = []
    required_metrics_by_role = _PLAYER_REQUIRED_METRICS_BY_RULESET[ruleset_version]
    if availability.match_status is not MatchStatus.FINISHED:
        reasons.append("match_not_finished")
    if _known_after(availability.as_of, evaluated_at):
        reasons.append("availability_as_of_after_evaluation")
    for team_id in availability.team_ids:
        starters = availability.starters.get(team_id, frozenset())
        if len(starters) != 11:
            reasons.append(f"starter_count:{team_id}:{len(starters)}")
        missing_observations = sorted(starters - availability.player_observation_ids)
        reasons.extend(
            f"missing_player_observation:{player_id}" for player_id in missing_observations
        )
        if availability.player_observations is not None:
            for player_id in sorted(starters & availability.player_observation_ids):
                observation = availability.player_observations.get(player_id)
                if observation is None:
                    reasons.append(f"missing_player_observation_detail:{player_id}")
                    continue
                if observation.team_id != team_id:
                    reasons.append(f"player_observation_team_mismatch:{player_id}")
                role = observation.role.strip()
                if not role:
                    reasons.append(f"missing_player_role:{player_id}")
                if observation.minutes <= 0:
                    reasons.append(f"missing_player_minutes:{player_id}")
                if not role or observation.minutes <= 0:
                    reasons.append(f"missing_player_role_or_minutes:{player_id}")
                if not observation.metric_fields:
                    reasons.append(f"missing_player_metrics:{player_id}")
                else:
                    required_metrics = required_metrics_by_role.get(role)
                    if required_metrics is None:
                        reasons.append(
                            f"missing_player_role_contract:{player_id}:{role or 'missing'}"
                        )
                        required_metrics = frozenset()
                    reasons.extend(
                        f"missing_player_metric:{player_id}:{metric}"
                        for metric in sorted(required_metrics - observation.metric_fields)
                    )
                if observation.known_at is None:
                    reasons.append(f"missing_player_known_at:{player_id}")
                if _known_after(observation.known_at, evaluated_at):
                    reasons.append(f"player_observation_known_after_evaluation:{player_id}")
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


def _archive_complete(availability: MatchAvailability, *, evaluated_at: datetime) -> bool:
    if availability.match_status is not MatchStatus.FINISHED or not availability.result_90_present:
        return False
    if _known_after(availability.as_of, evaluated_at):
        return False
    if _known_after(availability.result_90_known_at, evaluated_at):
        return False
    if set(availability.team_stat_fields) - set(availability.team_ids):
        return False
    if set(availability.starters) - set(availability.team_ids):
        return False
    for team_id in availability.team_ids:
        fields = availability.team_stat_fields.get(team_id, frozenset())
        if not _REQUIRED_TEAM_STAT_FIELDS <= fields:
            return False
        if _known_after(availability.team_stats_known_at.get(team_id), evaluated_at):
            return False
        starters = availability.starters.get(team_id, frozenset())
        if len(starters) != 11:
            return False
        if not starters <= availability.player_observation_ids:
            return False
    return True


def _verified_ready_snapshot_types(
    availability: MatchAvailability,
    *,
    evaluated_at: datetime,
    snapshot_validator: SnapshotSourceValidator | None,
) -> tuple[set[SnapshotType], bool]:
    ready_types: set[SnapshotType] = set()
    has_unverified = False
    for summary in availability.snapshots:
        if summary.quality_status != "ready":
            continue
        if not _snapshot_is_verified(
            summary,
            availability=availability,
            evaluated_at=evaluated_at,
            snapshot_validator=snapshot_validator,
        ):
            has_unverified = True
            continue
        ready_types.add(summary.snapshot_type)
    return ready_types, has_unverified


def _snapshot_is_verified(
    summary: SnapshotAvailability,
    *,
    availability: MatchAvailability,
    evaluated_at: datetime,
    snapshot_validator: SnapshotSourceValidator | None,
) -> bool:
    evidence = summary.evidence
    if evidence is None or snapshot_validator is None:
        return False
    if (
        summary.snapshot_type is not evidence.snapshot_type
        or summary.capture_mode is not evidence.capture_mode
        or summary.quality_status != evidence.quality_status
    ):
        return False
    if availability.match_id is not None and evidence.match_id.value != availability.match_id:
        return False
    if (
        availability.match_version is not None
        and evidence.match_version != availability.match_version
    ):
        return False
    if evidence.observed_at > evaluated_at:
        return False
    try:
        verify_snapshot(evidence, source_validator=snapshot_validator)
        loader = getattr(snapshot_validator, "load_snapshot_payload", None)
        if loader is None:
            return False
        payload = loader(evidence)
        if payload.get("id") != evidence.id.value:
            return False
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return evidence.quality_status == "ready"


def _known_after(known_at: datetime | None, evaluated_at: datetime) -> bool:
    if known_at is None:
        return False
    require_utc(known_at, "known_at")
    return known_at > evaluated_at
