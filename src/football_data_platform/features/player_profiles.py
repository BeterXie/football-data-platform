"""Versioned role-based player profiles with explicit missingness."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from football_data_platform.domain.models import require_utc


@dataclass(frozen=True, slots=True)
class RoleMetricContract:
    role: str
    version: str
    required_metrics: tuple[str, ...]
    optional_metrics: tuple[str, ...]
    not_applicable_metrics: tuple[str, ...]
    minimum_minutes: float
    minimum_matches: int

    def __post_init__(self) -> None:
        if not self.role or not self.version:
            raise ValueError("role and version must be non-empty")
        groups = (
            self.required_metrics,
            self.optional_metrics,
            self.not_applicable_metrics,
        )
        if any(not metric for group in groups for metric in group):
            raise ValueError("metric names must be non-empty")
        flattened = tuple(metric for group in groups for metric in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("required, optional, and not-applicable metrics must be disjoint")
        if not math.isfinite(self.minimum_minutes) or self.minimum_minutes < 0:
            raise ValueError("minimum_minutes must be finite and non-negative")
        if self.minimum_matches < 1:
            raise ValueError("minimum_matches must be positive")

    @property
    def applicable_metrics(self) -> tuple[str, ...]:
        return self.required_metrics + self.optional_metrics


@dataclass(frozen=True, slots=True)
class ProfileWindowSpec:
    name: str
    version: str
    lookback_days: float

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("window name and version must be non-empty")
        if not math.isfinite(self.lookback_days) or self.lookback_days <= 0:
            raise ValueError("lookback_days must be finite and positive")

    def starts_at(self, as_of: datetime) -> datetime:
        return as_of - timedelta(days=self.lookback_days)


@dataclass(frozen=True, slots=True)
class PlayerMatchObservation:
    player_id: str
    team_id: str
    match_id: str
    role: str
    minutes: float
    known_at: datetime
    metrics: dict[str, float | None]
    source_ref: str
    played_at: datetime | None = None

    def __post_init__(self) -> None:
        require_utc(self.known_at, "known_at")
        if self.played_at is not None:
            require_utc(self.played_at, "played_at")
            if self.played_at > self.known_at:
                raise ValueError("played_at must not be later than known_at")
        if not math.isfinite(self.minutes) or self.minutes < 0:
            raise ValueError("minutes must be finite and non-negative")
        for metric, value in self.metrics.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"metric {metric!r} must be finite or None")


@dataclass(frozen=True, slots=True)
class PlayerAvailabilityObservation:
    player_id: str
    team_id: str
    status: str
    probability: float | None
    known_at: datetime
    source_ref: str
    definition_version: str = "availability/1"

    def __post_init__(self) -> None:
        require_utc(self.known_at, "known_at")
        if not self.status or self.status == "missing" or not self.definition_version:
            raise ValueError("availability status must describe observed evidence")
        if self.probability is not None and (
            not math.isfinite(self.probability) or not 0 <= self.probability <= 1
        ):
            raise ValueError("availability probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    metric: str
    total: float | None
    per90: float | None
    sample_minutes: float
    sample_matches: int
    role_percentile: float | None
    cohort_version: str
    cohort_size: int
    quality_status: str


@dataclass(frozen=True, slots=True)
class ProfileWindow:
    name: str
    version: str
    as_of: datetime
    starts_at: datetime
    total_minutes: float
    sample_matches: int
    metrics: tuple[ProfileMetric, ...]
    quality_status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileAvailability:
    status: str
    probability: float | None
    definition_version: str
    as_of: datetime
    known_at: datetime | None
    source_ref: str | None


@dataclass(frozen=True, slots=True)
class ProfileLoad:
    window_name: str
    window_version: str
    as_of: datetime
    starts_at: datetime
    minutes: float | None
    sample_matches: int
    quality_status: str
    input_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfilePartitionAudit:
    player_id: str
    team_ids: tuple[str, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    artifact_id: str
    schema_version: int
    player_id: str
    team_id: str
    role: str
    as_of: datetime
    transform_version: str
    metric_definition_version: str
    role_contract_version: str | None
    cohort_version: str
    long_term_ability: ProfileWindow
    recent_form: ProfileWindow
    availability: ProfileAvailability
    load: ProfileLoad
    input_refs: tuple[str, ...]
    quality_status: str
    quality_reasons: tuple[str, ...]

    @property
    def total_minutes(self) -> float:
        """Compatibility view of the long-term sample."""

        return self.long_term_ability.total_minutes

    @property
    def sample_matches(self) -> int:
        """Compatibility view of the long-term sample."""

        return self.long_term_ability.sample_matches

    @property
    def metrics(self) -> tuple[ProfileMetric, ...]:
        """Compatibility view of long-term metrics."""

        return self.long_term_ability.metrics


@dataclass(frozen=True, slots=True)
class ProfileBuildResult:
    profiles: tuple[PlayerProfile, ...]
    excluded_input_refs: tuple[str, ...]
    partition_audits: tuple[ProfilePartitionAudit, ...]


_OUTFIELD_OPTIONAL = ("goals", "xg", "shots_on_target", "key_passes")
DEFAULT_ROLE_CONTRACTS = (
    RoleMetricContract(
        "forward", "role-metrics/1", ("shots",), _OUTFIELD_OPTIONAL, ("saves",), 0, 1
    ),
    RoleMetricContract("FW", "role-metrics/1", ("shots",), _OUTFIELD_OPTIONAL, ("saves",), 0, 1),
    RoleMetricContract("MF", "role-metrics/1", ("shots",), _OUTFIELD_OPTIONAL, ("saves",), 0, 1),
    RoleMetricContract("DF", "role-metrics/1", ("shots",), _OUTFIELD_OPTIONAL, ("saves",), 0, 1),
    RoleMetricContract(
        "GK",
        "role-metrics/1",
        ("saves",),
        (),
        ("goals", "xg", "shots", "shots_on_target", "key_passes"),
        0,
        1,
    ),
)
DEFAULT_LONG_TERM_WINDOW = ProfileWindowSpec("long-term", "long-term/1", 365)
DEFAULT_RECENT_WINDOW = ProfileWindowSpec("recent", "recent/1", 30)
DEFAULT_LOAD_WINDOW = ProfileWindowSpec("load", "load/1", 14)


def build_player_profiles(
    observations: tuple[PlayerMatchObservation, ...],
    *,
    as_of: datetime,
    minimum_minutes: float,
    transform_version: str = "player-profile/2",
    metric_definition_version: str = "player-metrics/1",
    cohort_version: str = "role-cohort/1",
    role_contracts: tuple[RoleMetricContract, ...] = DEFAULT_ROLE_CONTRACTS,
    long_term_window: ProfileWindowSpec = DEFAULT_LONG_TERM_WINDOW,
    recent_window: ProfileWindowSpec = DEFAULT_RECENT_WINDOW,
    load_window: ProfileWindowSpec = DEFAULT_LOAD_WINDOW,
    availability_observations: tuple[PlayerAvailabilityObservation, ...] = (),
) -> ProfileBuildResult:
    """Build independent ability, form, availability, and load components."""

    require_utc(as_of, "as_of")
    if not math.isfinite(minimum_minutes) or minimum_minutes < 0:
        raise ValueError("minimum_minutes must be finite and non-negative")
    if recent_window.lookback_days > long_term_window.lookback_days:
        raise ValueError("recent window must not be longer than long-term window")
    if load_window.lookback_days > long_term_window.lookback_days:
        raise ValueError("load window must not be longer than long-term window")
    if not transform_version or not metric_definition_version or not cohort_version:
        raise ValueError("profile and cohort versions must be non-empty")
    contracts = _contracts_by_role(role_contracts)

    known_observations = tuple(
        observation for observation in observations if observation.known_at <= as_of
    )
    long_term_starts_at = long_term_window.starts_at(as_of)
    eligible = tuple(
        observation
        for observation in known_observations
        if observation.played_at is None or observation.played_at >= long_term_starts_at
    )
    eligible_availability = tuple(
        item for item in availability_observations if item.known_at <= as_of
    )
    excluded = tuple(
        sorted(
            {
                item.source_ref
                for item in (*observations, *availability_observations)
                if item.known_at > as_of
            }
        )
    )
    by_partition: dict[tuple[str, str, str], list[PlayerMatchObservation]] = {}
    by_player_team: dict[tuple[str, str], list[PlayerMatchObservation]] = {}
    for observation in eligible:
        by_partition.setdefault(
            (observation.player_id, observation.team_id, observation.role), []
        ).append(observation)
        by_player_team.setdefault((observation.player_id, observation.team_id), []).append(
            observation
        )

    availability_by_player_team = _latest_availability(eligible_availability)
    profiles = [
        _build_profile(
            player_id=key[0],
            team_id=key[1],
            role=key[2],
            rows=tuple(rows),
            load_rows=tuple(by_player_team[(key[0], key[1])]),
            availability=availability_by_player_team.get((key[0], key[1])),
            as_of=as_of,
            minimum_minutes=minimum_minutes,
            transform_version=transform_version,
            metric_definition_version=metric_definition_version,
            cohort_version=cohort_version,
            contract=contracts.get(key[2]),
            long_term_window=long_term_window,
            recent_window=recent_window,
            load_window=load_window,
        )
        for key, rows in sorted(by_partition.items())
    ]
    profiles = _with_role_percentiles(profiles)
    profiles = [replace(profile, artifact_id=_artifact_id(profile)) for profile in profiles]
    return ProfileBuildResult(
        profiles=tuple(profiles),
        excluded_input_refs=excluded,
        partition_audits=_partition_audits(known_observations),
    )


def _contracts_by_role(
    contracts: tuple[RoleMetricContract, ...],
) -> dict[str, RoleMetricContract]:
    by_role: dict[str, RoleMetricContract] = {}
    for contract in contracts:
        if contract.role in by_role:
            raise ValueError(f"duplicate role contract: {contract.role}")
        by_role[contract.role] = contract
    return by_role


def _latest_availability(
    observations: tuple[PlayerAvailabilityObservation, ...],
) -> dict[tuple[str, str], PlayerAvailabilityObservation]:
    result: dict[tuple[str, str], PlayerAvailabilityObservation] = {}
    for observation in observations:
        key = (observation.player_id, observation.team_id)
        current = result.get(key)
        if current is None or (observation.known_at, observation.source_ref) > (
            current.known_at,
            current.source_ref,
        ):
            result[key] = observation
    return result


def _build_profile(
    *,
    player_id: str,
    team_id: str,
    role: str,
    rows: tuple[PlayerMatchObservation, ...],
    load_rows: tuple[PlayerMatchObservation, ...],
    availability: PlayerAvailabilityObservation | None,
    as_of: datetime,
    minimum_minutes: float,
    transform_version: str,
    metric_definition_version: str,
    cohort_version: str,
    contract: RoleMetricContract | None,
    long_term_window: ProfileWindowSpec,
    recent_window: ProfileWindowSpec,
    load_window: ProfileWindowSpec,
) -> PlayerProfile:
    missing_window_time = any(row.played_at is None for row in rows)
    effective_minimum = max(minimum_minutes, contract.minimum_minutes if contract else 0)
    minimum_matches = contract.minimum_matches if contract else 1
    long_term = _aggregate_window(
        rows,
        spec=long_term_window,
        as_of=as_of,
        contract=contract,
        minimum_minutes=effective_minimum,
        minimum_matches=minimum_matches,
        cohort_version=cohort_version,
    )
    recent = _aggregate_window(
        rows,
        spec=recent_window,
        as_of=as_of,
        contract=contract,
        minimum_minutes=effective_minimum,
        minimum_matches=minimum_matches,
        cohort_version=cohort_version,
    )
    load = _build_load(load_rows, spec=load_window, as_of=as_of)
    availability_state = _availability_state(availability, as_of=as_of)
    if missing_window_time:
        quality_status = "missing-window-time"
        reasons = ("played_at:missing",)
    elif contract is None:
        quality_status = "missing-role-contract"
        reasons = (f"role_contract:{role}:missing",)
    elif long_term.quality_status != "ready":
        quality_status = long_term.quality_status
        reasons = tuple(f"long-term:{reason}" for reason in long_term.reasons)
    elif recent.quality_status != "ready":
        quality_status = recent.quality_status
        reasons = tuple(f"recent:{reason}" for reason in recent.reasons)
    else:
        quality_status = "ready"
        reasons = ()
    input_refs = {row.source_ref for row in rows}
    input_refs.update(load.input_refs)
    if availability_state.source_ref is not None:
        input_refs.add(availability_state.source_ref)
    return PlayerProfile(
        artifact_id="",
        schema_version=2,
        player_id=player_id,
        team_id=team_id,
        role=role,
        as_of=as_of,
        transform_version=transform_version,
        metric_definition_version=metric_definition_version,
        role_contract_version=contract.version if contract else None,
        cohort_version=cohort_version,
        long_term_ability=long_term,
        recent_form=recent,
        availability=availability_state,
        load=load,
        input_refs=tuple(sorted(input_refs)),
        quality_status=quality_status,
        quality_reasons=reasons,
    )


def _aggregate_window(
    rows: tuple[PlayerMatchObservation, ...],
    *,
    spec: ProfileWindowSpec,
    as_of: datetime,
    contract: RoleMetricContract | None,
    minimum_minutes: float,
    minimum_matches: int,
    cohort_version: str,
) -> ProfileWindow:
    starts_at = spec.starts_at(as_of)
    window_rows = tuple(
        row for row in rows if row.played_at is not None and starts_at <= row.played_at <= as_of
    )
    total_minutes = math.fsum(row.minutes for row in window_rows)
    sample_matches = len({row.match_id for row in window_rows})
    if contract is None:
        return ProfileWindow(
            spec.name,
            spec.version,
            as_of,
            starts_at,
            total_minutes,
            sample_matches,
            (),
            "missing-role-contract",
            ("role_contract:missing",),
        )
    enough_sample = total_minutes >= minimum_minutes and sample_matches >= minimum_matches
    metrics = tuple(
        _aggregate_metric(
            metric,
            window_rows,
            minimum_minutes=minimum_minutes,
            minimum_matches=minimum_matches,
            cohort_version=cohort_version,
            not_applicable=False,
        )
        for metric in sorted(contract.applicable_metrics)
    ) + tuple(
        _aggregate_metric(
            metric,
            window_rows,
            minimum_minutes=minimum_minutes,
            minimum_matches=minimum_matches,
            cohort_version=cohort_version,
            not_applicable=True,
        )
        for metric in sorted(contract.not_applicable_metrics)
    )
    by_metric = {metric.metric: metric for metric in metrics}
    reasons: list[str] = []
    if not contract.applicable_metrics:
        reasons.append("applicable_metrics:empty")
        status = "no-applicable-metrics"
    elif not enough_sample:
        if total_minutes < minimum_minutes:
            reasons.append(f"minimum_minutes:{minimum_minutes:g}")
        if sample_matches < minimum_matches:
            reasons.append(f"minimum_matches:{minimum_matches}")
        status = "insufficient-sample"
    else:
        invalid_required = tuple(
            (metric, by_metric[metric].quality_status)
            for metric in contract.required_metrics
            if by_metric[metric].quality_status != "ready"
        )
        if invalid_required:
            reasons.extend(
                f"required_metric:{metric}:{metric_status}"
                for metric, metric_status in invalid_required
            )
            if any(metric_status == "missing" for _, metric_status in invalid_required):
                status = "missing-required-metrics"
            else:
                status = "insufficient-sample"
        else:
            status = "ready"
    return ProfileWindow(
        spec.name,
        spec.version,
        as_of,
        starts_at,
        total_minutes,
        sample_matches,
        tuple(sorted(metrics, key=lambda item: item.metric)),
        status,
        tuple(reasons),
    )


def _aggregate_metric(
    metric: str,
    rows: tuple[PlayerMatchObservation, ...],
    *,
    minimum_minutes: float,
    minimum_matches: int,
    cohort_version: str,
    not_applicable: bool,
) -> ProfileMetric:
    if not_applicable:
        return ProfileMetric(metric, None, None, 0.0, 0, None, cohort_version, 0, "not-applicable")
    available = tuple(row for row in rows if row.metrics.get(metric) is not None)
    if not available:
        return ProfileMetric(metric, None, None, 0.0, 0, None, cohort_version, 0, "missing")
    total = math.fsum(float(row.metrics[metric]) for row in available)
    sample_minutes = math.fsum(row.minutes for row in available)
    sample_matches = len({row.match_id for row in available})
    per90 = total / sample_minutes * 90.0 if sample_minutes > 0 else None
    status = (
        "ready"
        if sample_minutes >= minimum_minutes
        and sample_matches >= minimum_matches
        and per90 is not None
        else "insufficient-sample"
    )
    return ProfileMetric(
        metric,
        total,
        per90,
        sample_minutes,
        sample_matches,
        None,
        cohort_version,
        0,
        status,
    )


def _availability_state(
    observation: PlayerAvailabilityObservation | None, *, as_of: datetime
) -> ProfileAvailability:
    if observation is None:
        return ProfileAvailability("missing", None, "availability/1", as_of, None, None)
    return ProfileAvailability(
        observation.status,
        observation.probability,
        observation.definition_version,
        as_of,
        observation.known_at,
        observation.source_ref,
    )


def _build_load(
    rows: tuple[PlayerMatchObservation, ...], *, spec: ProfileWindowSpec, as_of: datetime
) -> ProfileLoad:
    starts_at = spec.starts_at(as_of)
    if any(row.played_at is None for row in rows):
        return ProfileLoad(spec.name, spec.version, as_of, starts_at, None, 0, "missing", ())
    window_rows = tuple(
        row for row in rows if row.played_at is not None and starts_at <= row.played_at <= as_of
    )
    if not window_rows:
        return ProfileLoad(spec.name, spec.version, as_of, starts_at, None, 0, "missing", ())
    return ProfileLoad(
        spec.name,
        spec.version,
        as_of,
        starts_at,
        math.fsum(row.minutes for row in window_rows),
        len({row.match_id for row in window_rows}),
        "ready",
        tuple(sorted({row.source_ref for row in window_rows})),
    )


def _with_role_percentiles(profiles: list[PlayerProfile]) -> list[PlayerProfile]:
    updated: list[PlayerProfile] = []
    for profile in profiles:
        long_term = _window_with_percentiles(profile, profile.long_term_ability, profiles)
        recent = _window_with_percentiles(profile, profile.recent_form, profiles)
        updated.append(replace(profile, long_term_ability=long_term, recent_form=recent))
    return updated


def _window_with_percentiles(
    profile: PlayerProfile,
    window: ProfileWindow,
    profiles: list[PlayerProfile],
) -> ProfileWindow:
    metrics: list[ProfileMetric] = []
    for metric in window.metrics:
        peers = sorted(
            peer_metric.per90
            for peer in profiles
            if peer.role == profile.role
            for peer_window in (
                peer.long_term_ability
                if window.name == profile.long_term_ability.name
                else peer.recent_form,
            )
            for peer_metric in peer_window.metrics
            if peer_metric.metric == metric.metric
            and peer_metric.quality_status == "ready"
            and peer_metric.per90 is not None
        )
        percentile = None
        if metric.quality_status == "ready" and metric.per90 is not None and peers:
            below = sum(peer < metric.per90 for peer in peers)
            equal = sum(peer == metric.per90 for peer in peers)
            percentile = (below + 0.5 * equal) / len(peers) * 100.0
        metrics.append(replace(metric, role_percentile=percentile, cohort_size=len(peers)))
    return replace(window, metrics=tuple(metrics))


def _partition_audits(
    observations: tuple[PlayerMatchObservation, ...],
) -> tuple[ProfilePartitionAudit, ...]:
    by_player: dict[str, list[PlayerMatchObservation]] = {}
    for observation in observations:
        by_player.setdefault(observation.player_id, []).append(observation)
    return tuple(
        ProfilePartitionAudit(
            player_id,
            tuple(sorted({row.team_id for row in rows})),
            tuple(sorted({row.role for row in rows})),
        )
        for player_id, rows in sorted(by_player.items())
        if len({row.team_id for row in rows}) > 1 or len({row.role for row in rows}) > 1
    )


def _artifact_id(profile: PlayerProfile) -> str:
    identity = asdict(replace(profile, artifact_id=""))
    identity = _timestamps(identity)
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return f"player-profile:{digest}"


def _timestamps(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _timestamps(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_timestamps(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
