"""Versioned role-based player profiles with explicit missingness."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
        for name, value in (
            ("player_id", self.player_id),
            ("team_id", self.team_id),
            ("match_id", self.match_id),
            ("role", self.role),
            ("source_ref", self.source_ref),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be non-empty text")
        require_utc(self.known_at, "known_at")
        if self.played_at is not None:
            require_utc(self.played_at, "played_at")
            if self.played_at > self.known_at:
                raise ValueError("played_at must not be later than known_at")
        if not math.isfinite(self.minutes) or self.minutes < 0:
            raise ValueError("minutes must be finite and non-negative")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a mapping")
        for metric, value in self.metrics.items():
            if not isinstance(metric, str) or not metric or metric.strip() != metric:
                raise ValueError("metric names must be non-empty text")
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
        if (
            not isinstance(self.player_id, str)
            or not self.player_id
            or self.player_id.strip() != self.player_id
            or not isinstance(self.team_id, str)
            or not self.team_id
            or self.team_id.strip() != self.team_id
            or not isinstance(self.source_ref, str)
            or not self.source_ref
            or self.source_ref.strip() != self.source_ref
        ):
            raise ValueError("availability identity and source_ref must be non-empty text")
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
    role_contract: RoleMetricContract | None
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

_PROFILE_STATUSES = frozenset(
    {
        "ready",
        "missing-window-time",
        "missing-role-contract",
        "no-applicable-metrics",
        "missing-applicable-metrics",
        "missing-required-metrics",
        "insufficient-sample",
    }
)
_WINDOW_STATUSES = _PROFILE_STATUSES - {"missing-window-time"}
_METRIC_STATUSES = frozenset({"ready", "missing", "insufficient-sample", "not-applicable"})


def build_player_profiles(
    observations: tuple[PlayerMatchObservation, ...],
    *,
    as_of: datetime,
    minimum_minutes: float,
    transform_version: str = "player-profile/3",
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


def validate_player_profile(profile: PlayerProfile) -> None:
    """Validate one profile before it is exposed as a derived artifact.

    The builder computes these invariants, but the storage boundary must also
    reject a hand-constructed or tampered ``PlayerProfile``.  In particular,
    a ``ready`` profile must contain a usable role dimension in both windows;
    missing availability or load remain independent quality states.
    """

    if not isinstance(profile, PlayerProfile):
        raise TypeError("profile must be a PlayerProfile")
    if profile.schema_version != 3:
        raise ValueError("unsupported player profile schema version")
    if not re.fullmatch(r"player-profile:[0-9a-f]{64}", profile.artifact_id):
        raise ValueError("invalid player profile artifact ID")
    if _artifact_id(profile) != profile.artifact_id:
        raise ValueError("player profile artifact identity failed")
    require_utc(profile.as_of, "profile as_of")
    for name, value in (
        ("player_id", profile.player_id),
        ("team_id", profile.team_id),
        ("role", profile.role),
        ("transform_version", profile.transform_version),
        ("metric_definition_version", profile.metric_definition_version),
        ("cohort_version", profile.cohort_version),
    ):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"profile {name} must be non-empty text")
    if profile.role_contract_version is not None and (
        not profile.role_contract_version
        or profile.role_contract_version.strip() != profile.role_contract_version
    ):
        raise ValueError("profile role_contract_version must be non-empty text when present")
    if profile.role_contract is None and profile.role_contract_version is not None:
        raise ValueError("profile role_contract is missing for its version")
    if profile.role_contract is not None:
        if profile.role_contract.version != profile.role_contract_version:
            raise ValueError("profile role contract version does not match contract")
        if profile.role_contract.role != profile.role:
            raise ValueError("profile role contract does not match role")
    if profile.quality_status not in _PROFILE_STATUSES:
        raise ValueError(f"unsupported player profile quality status {profile.quality_status!r}")
    if any(not isinstance(ref, str) or not ref or ref.strip() != ref for ref in profile.input_refs):
        raise ValueError("profile input_refs must contain non-empty references")
    _validate_profile_window(
        profile.long_term_ability, profile.as_of, "long-term", profile.role_contract
    )
    _validate_profile_window(profile.recent_form, profile.as_of, "recent", profile.role_contract)
    _validate_profile_load(profile.load, profile.as_of)
    if profile.availability.as_of != profile.as_of:
        raise ValueError("profile availability as_of must match profile")
    if profile.availability.known_at is not None:
        require_utc(profile.availability.known_at, "profile availability known_at")
        if profile.availability.known_at > profile.as_of:
            raise ValueError("profile availability known_at is after as_of")
    if profile.availability.source_ref is not None and (
        not profile.availability.source_ref
        or profile.availability.source_ref.strip() != profile.availability.source_ref
    ):
        raise ValueError("profile availability source_ref must be non-empty text")
    if profile.quality_status == "ready":
        if profile.role_contract_version is None or profile.role_contract is None:
            raise ValueError("ready profile requires a role contract version")
        if profile.role_contract.version != profile.role_contract_version:
            raise ValueError("profile role contract version does not match contract")
        if profile.role_contract.role != profile.role:
            raise ValueError("profile role contract does not match role")
        for window_name, window in (
            ("long-term", profile.long_term_ability),
            ("recent", profile.recent_form),
        ):
            if window.quality_status != "ready":
                raise ValueError(f"ready profile has non-ready {window_name} window")
            by_metric = {metric.metric: metric for metric in window.metrics}
            if any(
                by_metric[metric].quality_status != "ready"
                for metric in profile.role_contract.required_metrics
            ):
                raise ValueError(f"ready profile has a non-ready required {window_name} metric")
        if not profile.input_refs:
            raise ValueError("ready profile requires input_refs lineage")


def parse_player_profile_payload(payload: Any) -> PlayerProfile:
    """Parse and revalidate one persisted player-profile payload."""

    try:
        value = _payload_object(
            payload,
            "player profile",
            {
                "artifact_id",
                "schema_version",
                "player_id",
                "team_id",
                "role",
                "as_of",
                "transform_version",
                "metric_definition_version",
                "role_contract_version",
                "role_contract",
                "cohort_version",
                "long_term_ability",
                "recent_form",
                "availability",
                "load",
                "input_refs",
                "quality_status",
                "quality_reasons",
            },
        )
        raw_contract = value["role_contract"]
        contract = None if raw_contract is None else _parse_role_contract(raw_contract)
        profile = PlayerProfile(
            artifact_id=value["artifact_id"],
            schema_version=value["schema_version"],
            player_id=value["player_id"],
            team_id=value["team_id"],
            role=value["role"],
            as_of=_payload_timestamp(value["as_of"], "player profile as_of"),
            transform_version=value["transform_version"],
            metric_definition_version=value["metric_definition_version"],
            role_contract_version=value["role_contract_version"],
            role_contract=contract,
            cohort_version=value["cohort_version"],
            long_term_ability=_parse_profile_window(value["long_term_ability"]),
            recent_form=_parse_profile_window(value["recent_form"]),
            availability=_parse_profile_availability(value["availability"]),
            load=_parse_profile_load(value["load"]),
            input_refs=_payload_strings(value["input_refs"], "player profile input_refs"),
            quality_status=value["quality_status"],
            quality_reasons=_payload_strings(
                value["quality_reasons"], "player profile quality_reasons"
            ),
        )
        validate_player_profile(profile)
        return profile
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid persisted player profile payload") from error


def _validate_profile_window(
    window: ProfileWindow,
    as_of: datetime,
    name: str,
    contract: RoleMetricContract | None,
) -> None:
    if not isinstance(window, ProfileWindow):
        raise TypeError(f"{name} window must be a ProfileWindow")
    if not window.name or not window.version:
        raise ValueError(f"{name} window requires name and version")
    require_utc(window.as_of, f"{name} window as_of")
    require_utc(window.starts_at, f"{name} window starts_at")
    if window.as_of != as_of or window.starts_at > window.as_of:
        raise ValueError(f"{name} window time bounds are invalid")
    if not math.isfinite(window.total_minutes) or window.total_minutes < 0:
        raise ValueError(f"{name} window total_minutes is invalid")
    if not isinstance(window.sample_matches, int) or window.sample_matches < 0:
        raise ValueError(f"{name} window sample_matches is invalid")
    if window.quality_status not in _WINDOW_STATUSES:
        raise ValueError(f"unsupported {name} window quality status {window.quality_status!r}")
    if any(not isinstance(reason, str) or not reason for reason in window.reasons):
        raise ValueError(f"{name} window reasons must be non-empty text")
    if window.quality_status != "ready" and not window.reasons:
        raise ValueError(f"non-ready {name} window requires reasons")
    names = [metric.metric for metric in window.metrics]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} window metrics must be unique")
    if contract is None and names:
        raise ValueError(f"{name} window cannot contain metrics without a role contract")
    if contract is not None:
        expected = set(contract.applicable_metrics) | set(contract.not_applicable_metrics)
        if set(names) != expected:
            raise ValueError(f"{name} window metrics do not match the role contract")
    for metric in window.metrics:
        _validate_profile_metric(metric, name)
        if contract is not None and metric.metric in contract.applicable_metrics:
            if metric.quality_status == "ready" and (
                metric.sample_minutes < contract.minimum_minutes
                or metric.sample_matches < contract.minimum_matches
            ):
                raise ValueError(f"ready {name} metric does not meet role sample thresholds")
    if window.quality_status == "ready" and (
        contract is None
        or not contract.applicable_metrics
        or not any(
            metric.quality_status == "ready"
            for metric in window.metrics
            if metric.metric in contract.applicable_metrics
        )
    ):
        raise ValueError(f"ready {name} window requires a ready applicable metric")


def _validate_profile_metric(metric: ProfileMetric, window_name: str) -> None:
    if (
        not isinstance(metric.metric, str)
        or not metric.metric
        or metric.metric.strip() != metric.metric
    ):
        raise ValueError(f"{window_name} metric name must be non-empty text")
    if metric.quality_status not in _METRIC_STATUSES:
        raise ValueError(f"unsupported profile metric quality status {metric.quality_status!r}")
    for field_name, value in (
        ("sample_minutes", metric.sample_minutes),
        ("role_percentile", metric.role_percentile),
    ):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"profile metric {field_name} is invalid")
    if metric.role_percentile is not None and metric.role_percentile > 100:
        raise ValueError("profile metric role_percentile must not exceed 100")
    if not isinstance(metric.sample_matches, int) or metric.sample_matches < 0:
        raise ValueError("profile metric sample_matches is invalid")
    if not isinstance(metric.cohort_version, str) or not metric.cohort_version:
        raise ValueError("profile metric cohort_version must be non-empty")
    if not isinstance(metric.cohort_size, int) or metric.cohort_size < 0:
        raise ValueError("profile metric cohort_size is invalid")
    for field_name, value in (("total", metric.total), ("per90", metric.per90)):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"profile metric {field_name} is not finite")
    if metric.quality_status in {"missing", "not-applicable"} and (
        metric.total is not None or metric.per90 is not None or metric.sample_minutes != 0
    ):
        raise ValueError(f"{metric.quality_status} metric cannot contain values")
    if metric.quality_status == "not-applicable" and (
        metric.sample_matches != 0 or metric.cohort_size != 0
    ):
        raise ValueError("not-applicable metric cannot contain sample counts")
    if metric.quality_status == "ready" and (
        metric.total is None
        or metric.per90 is None
        or metric.sample_minutes <= 0
        or metric.sample_matches < 1
    ):
        raise ValueError("ready metric requires a positive sample and value")


def _validate_profile_load(load: ProfileLoad, as_of: datetime) -> None:
    if not isinstance(load, ProfileLoad):
        raise TypeError("profile load must be a ProfileLoad")
    if not load.window_name or not load.window_version:
        raise ValueError("profile load requires a window name and version")
    require_utc(load.as_of, "profile load as_of")
    require_utc(load.starts_at, "profile load starts_at")
    if load.as_of != as_of or load.starts_at > load.as_of:
        raise ValueError("profile load time bounds are invalid")
    if load.minutes is not None and (not math.isfinite(load.minutes) or load.minutes < 0):
        raise ValueError("profile load minutes is invalid")
    if not isinstance(load.sample_matches, int) or load.sample_matches < 0:
        raise ValueError("profile load sample_matches is invalid")
    if load.quality_status not in {"ready", "missing"}:
        raise ValueError(f"unsupported profile load quality status {load.quality_status!r}")
    if load.quality_status == "ready" and load.minutes is None:
        raise ValueError("ready profile load requires minutes")
    if any(not isinstance(ref, str) or not ref for ref in load.input_refs):
        raise ValueError("profile load input_refs must contain non-empty references")


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
        schema_version=3,
        player_id=player_id,
        team_id=team_id,
        role=role,
        as_of=as_of,
        transform_version=transform_version,
        metric_definition_version=metric_definition_version,
        role_contract_version=contract.version if contract else None,
        role_contract=contract,
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
        elif not any(
            metric.quality_status == "ready"
            for metric in metrics
            if metric.metric in contract.applicable_metrics
        ):
            # A contract with only optional dimensions still needs at least
            # one observed, sufficiently sampled dimension to be usable.
            if all(
                metric.quality_status == "missing"
                for metric in metrics
                if metric.metric in contract.applicable_metrics
            ):
                reasons.append("applicable_metrics:all-missing")
                status = "missing-applicable-metrics"
            else:
                reasons.append("applicable_metrics:insufficient-sample")
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


def _parse_role_contract(value: Any) -> RoleMetricContract:
    payload = _payload_object(
        value,
        "role contract",
        {
            "role",
            "version",
            "required_metrics",
            "optional_metrics",
            "not_applicable_metrics",
            "minimum_minutes",
            "minimum_matches",
        },
    )
    return RoleMetricContract(
        role=payload["role"],
        version=payload["version"],
        required_metrics=_payload_strings(
            payload["required_metrics"], "role contract required_metrics"
        ),
        optional_metrics=_payload_strings(
            payload["optional_metrics"], "role contract optional_metrics"
        ),
        not_applicable_metrics=_payload_strings(
            payload["not_applicable_metrics"], "role contract not_applicable_metrics"
        ),
        minimum_minutes=payload["minimum_minutes"],
        minimum_matches=payload["minimum_matches"],
    )


def _parse_profile_window(value: Any) -> ProfileWindow:
    payload = _payload_object(
        value,
        "profile window",
        {
            "name",
            "version",
            "as_of",
            "starts_at",
            "total_minutes",
            "sample_matches",
            "metrics",
            "quality_status",
            "reasons",
        },
    )
    raw_metrics = payload["metrics"]
    if not isinstance(raw_metrics, list):
        raise ValueError("profile window metrics must be a list")
    return ProfileWindow(
        name=payload["name"],
        version=payload["version"],
        as_of=_payload_timestamp(payload["as_of"], "profile window as_of"),
        starts_at=_payload_timestamp(payload["starts_at"], "profile window starts_at"),
        total_minutes=payload["total_minutes"],
        sample_matches=payload["sample_matches"],
        metrics=tuple(_parse_profile_metric(item) for item in raw_metrics),
        quality_status=payload["quality_status"],
        reasons=_payload_strings(payload["reasons"], "profile window reasons"),
    )


def _parse_profile_metric(value: Any) -> ProfileMetric:
    payload = _payload_object(
        value,
        "profile metric",
        {
            "metric",
            "total",
            "per90",
            "sample_minutes",
            "sample_matches",
            "role_percentile",
            "cohort_version",
            "cohort_size",
            "quality_status",
        },
    )
    return ProfileMetric(**payload)


def _parse_profile_availability(value: Any) -> ProfileAvailability:
    payload = _payload_object(
        value,
        "profile availability",
        {"status", "probability", "definition_version", "as_of", "known_at", "source_ref"},
    )
    known_at = payload["known_at"]
    return ProfileAvailability(
        status=payload["status"],
        probability=payload["probability"],
        definition_version=payload["definition_version"],
        as_of=_payload_timestamp(payload["as_of"], "profile availability as_of"),
        known_at=(
            None
            if known_at is None
            else _payload_timestamp(known_at, "profile availability known_at")
        ),
        source_ref=payload["source_ref"],
    )


def _parse_profile_load(value: Any) -> ProfileLoad:
    payload = _payload_object(
        value,
        "profile load",
        {
            "window_name",
            "window_version",
            "as_of",
            "starts_at",
            "minutes",
            "sample_matches",
            "quality_status",
            "input_refs",
        },
    )
    return ProfileLoad(
        window_name=payload["window_name"],
        window_version=payload["window_version"],
        as_of=_payload_timestamp(payload["as_of"], "profile load as_of"),
        starts_at=_payload_timestamp(payload["starts_at"], "profile load starts_at"),
        minutes=payload["minutes"],
        sample_matches=payload["sample_matches"],
        quality_status=payload["quality_status"],
        input_refs=_payload_strings(payload["input_refs"], "profile load input_refs"),
    )


def _payload_object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _payload_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of text values")
    return tuple(value)


def _payload_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_utc(parsed, name)
    return parsed


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
