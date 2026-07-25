"""Role-aware lineup deltas that preserve missing and N/A evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from football_data_platform.features.player_profiles import (
    PlayerProfile,
    RoleMetricContract,
)

LINEUP_DELTA_INPUT_TRANSFORM_V3 = "lineup-delta-input/3"
_PROFILE_WINDOWS = frozenset({"long_term_ability", "recent_form"})


@dataclass(frozen=True, slots=True)
class PlayerLineupValue:
    player_id: str
    dimensions: dict[str, float | None]
    profile_ref: str
    role: str = "generic"
    profile_quality_status: str = "ready"

    def __post_init__(self) -> None:
        for name, value in (
            ("player_id", self.player_id),
            ("profile_ref", self.profile_ref),
            ("role", self.role),
            ("profile_quality_status", self.profile_quality_status),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.dimensions, dict):
            raise TypeError("dimensions must be a mapping")
        for name, value in self.dimensions.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("lineup dimension names must be non-empty text")
            if value is not None and not math.isfinite(value):
                raise ValueError(f"lineup dimension {name!r} must be finite or None")


@dataclass(frozen=True, slots=True)
class LineupDelta:
    quality_status: str
    dimension_deltas: dict[str, float]
    missing_fields: tuple[str, ...]
    input_refs: tuple[str, ...]
    not_applicable_fields: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    transform_version: str = "lineup-delta/2"
    role_contract_versions: tuple[str, ...] = ()
    dimension_sample_sizes: dict[str, tuple[int, int]] | None = None


def validate_lineup_delta(delta: LineupDelta) -> None:
    """Validate the persisted contract for one lineup delta."""

    if not isinstance(delta, LineupDelta):
        raise TypeError("delta must be a LineupDelta")
    if delta.quality_status not in {"ready", "preview"}:
        raise ValueError("lineup delta quality_status must be ready or preview")
    if not isinstance(delta.transform_version, str) or not delta.transform_version:
        raise ValueError("lineup delta requires a transform version")
    if any(
        not isinstance(item, str) or not item or item.strip() != item
        for item in (*delta.input_refs, *delta.role_contract_versions)
    ):
        raise ValueError("lineup delta references and versions must be non-empty text")
    if tuple(sorted(set(delta.input_refs))) != delta.input_refs:
        raise ValueError("lineup delta input_refs must be sorted and unique")
    if tuple(sorted(set(delta.role_contract_versions))) != delta.role_contract_versions:
        raise ValueError("lineup delta role_contract_versions must be sorted and unique")
    for field_name, values in (
        ("missing_fields", delta.missing_fields),
        ("not_applicable_fields", delta.not_applicable_fields),
        ("blocking_reasons", delta.blocking_reasons),
    ):
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"lineup delta {field_name} must contain non-empty text")
        if tuple(sorted(set(values))) != values:
            raise ValueError(f"lineup delta {field_name} must be sorted and unique")
    if not isinstance(delta.dimension_deltas, dict):
        raise ValueError("lineup delta dimension_deltas must be a mapping")
    for dimension, value in delta.dimension_deltas.items():
        if not isinstance(dimension, str) or not dimension or dimension.strip() != dimension:
            raise ValueError("lineup delta dimensions must have non-empty names")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"lineup delta dimension {dimension!r} must be finite numeric")
    sample_sizes = delta.dimension_sample_sizes
    if sample_sizes is None:
        sample_sizes = {}
    if set(sample_sizes) != set(delta.dimension_deltas):
        raise ValueError("lineup delta sample sizes must match dimensions")
    for dimension, pair in sample_sizes.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in pair
            )
        ):
            raise ValueError(f"lineup delta sample size for {dimension!r} is invalid")
    if delta.quality_status == "ready":
        if not delta.dimension_deltas:
            raise ValueError("ready lineup delta requires non-empty dimensions")
        if delta.missing_fields or delta.blocking_reasons:
            raise ValueError("ready lineup delta cannot contain missing or blocking reasons")
        if not delta.input_refs or not delta.role_contract_versions:
            raise ValueError("ready lineup delta requires profile lineage and contract versions")
    else:
        if delta.dimension_deltas:
            raise ValueError("preview lineup delta cannot expose usable dimensions")
        if not delta.missing_fields and not delta.blocking_reasons:
            raise ValueError("preview lineup delta requires missing or blocking reasons")


def lineup_delta_payload(delta: LineupDelta) -> dict[str, Any]:
    """Return the complete JSON payload used in snapshots and manifests."""

    validate_lineup_delta(delta)
    return {
        "quality_status": delta.quality_status,
        "dimension_deltas": dict(sorted(delta.dimension_deltas.items())),
        "missing_fields": list(delta.missing_fields),
        "input_refs": list(delta.input_refs),
        "not_applicable_fields": list(delta.not_applicable_fields),
        "blocking_reasons": list(delta.blocking_reasons),
        "transform_version": delta.transform_version,
        "role_contract_versions": list(delta.role_contract_versions),
        "dimension_sample_sizes": {
            dimension: list(pair)
            for dimension, pair in sorted((delta.dimension_sample_sizes or {}).items())
        },
    }


def parse_lineup_delta_payload(payload: Any) -> LineupDelta:
    """Parse the aggregate portion of one persisted lineup-delta item."""

    required = {
        "quality_status",
        "dimension_deltas",
        "missing_fields",
        "input_refs",
        "not_applicable_fields",
        "blocking_reasons",
        "transform_version",
        "role_contract_versions",
        "dimension_sample_sizes",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("lineup delta payload fields are invalid")
    sample_sizes = payload["dimension_sample_sizes"]
    if not isinstance(sample_sizes, dict):
        raise ValueError("lineup delta sample sizes must be a mapping")
    try:
        delta = LineupDelta(
            quality_status=payload["quality_status"],
            dimension_deltas=payload["dimension_deltas"],
            missing_fields=tuple(payload["missing_fields"]),
            input_refs=tuple(payload["input_refs"]),
            not_applicable_fields=tuple(payload["not_applicable_fields"]),
            blocking_reasons=tuple(payload["blocking_reasons"]),
            transform_version=payload["transform_version"],
            role_contract_versions=tuple(payload["role_contract_versions"]),
            dimension_sample_sizes={
                dimension: tuple(pair) for dimension, pair in sample_sizes.items()
            },
        )
    except TypeError as error:
        raise ValueError("lineup delta payload collections are invalid") from error
    validate_lineup_delta(delta)
    return delta


def lineup_delta_input_payload(
    *,
    starter_ids: tuple[str, ...],
    reference_starter_ids: tuple[str, ...] | None,
    profiles: tuple[PlayerProfile, ...],
    profile_window: str = "recent_form",
    profile_window_version: str = "recent/1",
    lineup_input_refs: tuple[str, ...] = (),
    reference_lineup_ref: str | None = None,
) -> dict[str, Any]:
    """Build the complete v3 source item from versioned player profiles."""

    if (reference_starter_ids is None) != (reference_lineup_ref is None):
        raise ValueError("reference lineup IDs and source ref must be provided together")
    if reference_lineup_ref is not None and (
        not isinstance(reference_lineup_ref, str)
        or not reference_lineup_ref.startswith("derived-source:")
    ):
        raise ValueError("reference_lineup_ref must be a derived source reference")
    delta = build_lineup_delta_from_profiles(
        starter_ids=starter_ids,
        reference_starter_ids=reference_starter_ids,
        profiles=profiles,
        profile_window=profile_window,
        profile_window_version=profile_window_version,
    )
    payload = lineup_delta_payload(delta)
    payload.update(
        {
            "starter_ids": list(starter_ids),
            "reference_starter_ids": (
                None if reference_starter_ids is None else list(reference_starter_ids)
            ),
            "player_profile_refs": {
                profile.player_id: profile.artifact_id
                for profile in sorted(profiles, key=lambda item: item.player_id)
            },
            "profile_window": profile_window,
            "profile_window_version": profile_window_version,
            "lineup_input_refs": list(tuple(sorted(set(lineup_input_refs)))),
            "reference_lineup_ref": reference_lineup_ref,
        }
    )
    return payload


def build_lineup_delta_from_profiles(
    *,
    starter_ids: tuple[str, ...],
    reference_starter_ids: tuple[str, ...] | None,
    profiles: tuple[PlayerProfile, ...],
    profile_window: str,
    profile_window_version: str,
) -> LineupDelta:
    """Recompute a lineup delta from complete player-profile artifacts."""

    if profile_window not in _PROFILE_WINDOWS:
        raise ValueError(f"unsupported lineup profile window {profile_window!r}")
    if not profile_window_version or profile_window_version.strip() != profile_window_version:
        raise ValueError("lineup profile window version must be non-empty text")
    if len({profile.player_id for profile in profiles}) != len(profiles):
        raise ValueError("lineup profiles must contain unique player IDs")
    required_players = set(starter_ids) | set(reference_starter_ids or ())
    if set(profile.player_id for profile in profiles) - required_players:
        raise ValueError("lineup profiles contain players outside the current and reference XI")

    contracts: dict[str, RoleMetricContract] = {}
    values: list[PlayerLineupValue] = []
    for profile in profiles:
        contract = profile.role_contract
        if contract is None or contract.version != profile.role_contract_version:
            raise ValueError(f"player profile {profile.artifact_id} lacks its role contract")
        existing = contracts.get(profile.role)
        if existing is not None and existing != contract:
            raise ValueError(f"player role {profile.role!r} has conflicting contracts")
        contracts[profile.role] = contract
        window = getattr(profile, profile_window)
        if window.version != profile_window_version:
            raise ValueError(
                f"player profile {profile.artifact_id} has the wrong lineup window version"
            )
        values.append(
            PlayerLineupValue(
                player_id=profile.player_id,
                dimensions={
                    metric.metric: (metric.per90 if metric.quality_status == "ready" else None)
                    for metric in window.metrics
                },
                profile_ref=profile.artifact_id,
                role=profile.role,
                profile_quality_status=profile.quality_status,
            )
        )
    return build_lineup_delta(
        starter_ids=starter_ids,
        reference_starter_ids=reference_starter_ids,
        player_values=tuple(values),
        role_contracts=tuple(contracts[role] for role in sorted(contracts)),
    )


DEFAULT_LINEUP_ROLE_CONTRACTS = (
    RoleMetricContract("generic", "lineup-role-metrics/1", ("attack",), (), (), 0, 1),
)


def build_lineup_delta(
    *,
    starter_ids: tuple[str, ...],
    reference_starter_ids: tuple[str, ...] | None,
    player_values: tuple[PlayerLineupValue, ...],
    role_contracts: tuple[RoleMetricContract, ...] = DEFAULT_LINEUP_ROLE_CONTRACTS,
    transform_version: str = "lineup-delta/2",
) -> LineupDelta:
    """Compare role-applicable dimensions for a current and reference XI."""

    _validate_xi(starter_ids, "starter_ids")
    if reference_starter_ids is not None:
        _validate_xi(reference_starter_ids, "reference_starter_ids")
    if not transform_version:
        raise ValueError("transform_version must be non-empty")
    contracts = _contracts_by_role(role_contracts)
    if len({value.player_id for value in player_values}) != len(player_values):
        raise ValueError("player_values must contain unique player IDs")

    by_player = {value.player_id: value for value in player_values}
    required_players = set(starter_ids) | set(reference_starter_ids or ())
    present_players = required_players & set(by_player)
    missing: list[str] = []
    blockers: set[str] = set()
    if reference_starter_ids is None:
        missing.append("reference_lineup")
        blockers.add("reference_lineup_missing")
    absent_players = required_players - set(by_player)
    missing.extend(f"player_profile:{player_id}" for player_id in sorted(absent_players))
    if absent_players:
        blockers.add("missing_player_profiles")

    used_contracts: dict[str, RoleMetricContract] = {}
    for player_id in sorted(present_players):
        value = by_player[player_id]
        contract = contracts.get(value.role)
        if contract is None:
            missing.append(f"role_contract:{player_id}:{value.role}")
            blockers.add("missing_role_contracts")
            continue
        used_contracts[player_id] = contract
        if value.profile_quality_status != "ready":
            missing.append(f"player_profile_status:{player_id}:{value.profile_quality_status}")
            blockers.add("player_profiles_not_ready")
        contracted = set(contract.applicable_metrics) | set(contract.not_applicable_metrics)
        if set(value.dimensions) - contracted:
            missing.extend(
                f"uncontracted_dimension:{player_id}:{dimension}"
                for dimension in sorted(set(value.dimensions) - contracted)
            )
            blockers.add("uncontracted_dimensions")

    dimensions = sorted(
        {
            dimension
            for contract in used_contracts.values()
            for dimension in (*contract.applicable_metrics, *contract.not_applicable_metrics)
        }
    )
    not_applicable: list[str] = []
    missing_required = False
    for player_id, contract in sorted(used_contracts.items()):
        value = by_player[player_id]
        applicable = set(contract.applicable_metrics)
        declared_na = set(contract.not_applicable_metrics)
        for dimension in dimensions:
            if dimension in declared_na:
                not_applicable.append(f"player_dimension:{player_id}:{dimension}")
                if value.dimensions.get(dimension) is not None:
                    blockers.add("values_supplied_for_not_applicable_dimensions")
                continue
            if dimension not in applicable:
                missing.append(f"unclassified_dimension:{player_id}:{dimension}")
                blockers.add("unclassified_role_dimensions")
                continue
            if value.dimensions.get(dimension) is None:
                missing.append(f"player_dimension:{player_id}:{dimension}")
                if dimension in contract.required_metrics:
                    missing_required = True
    if missing_required:
        blockers.add("missing_required_dimensions")

    deltas: dict[str, float] = {}
    sample_sizes: dict[str, tuple[int, int]] = {}
    if reference_starter_ids is not None:
        for dimension in dimensions:
            starter_values = _applicable_values(
                starter_ids, dimension=dimension, by_player=by_player, contracts=used_contracts
            )
            reference_values = _applicable_values(
                reference_starter_ids,
                dimension=dimension,
                by_player=by_player,
                contracts=used_contracts,
            )
            if starter_values is None or reference_values is None:
                continue
            deltas[dimension] = _mean(starter_values) - _mean(reference_values)
            sample_sizes[dimension] = (len(starter_values), len(reference_values))
    if not deltas:
        blockers.add("no_comparable_dimensions")

    input_refs = tuple(sorted({by_player[player_id].profile_ref for player_id in present_players}))
    quality_status = "ready" if not blockers and not missing else "preview"
    if quality_status != "ready":
        deltas = {}
        sample_sizes = {}
    result = LineupDelta(
        quality_status=quality_status,
        dimension_deltas=deltas,
        missing_fields=tuple(sorted(missing)),
        input_refs=input_refs,
        not_applicable_fields=tuple(sorted(not_applicable)),
        blocking_reasons=tuple(sorted(blockers)),
        transform_version=transform_version,
        role_contract_versions=tuple(
            sorted({contract.version for contract in used_contracts.values()})
        ),
        dimension_sample_sizes=sample_sizes,
    )
    validate_lineup_delta(result)
    return result


def _validate_xi(player_ids: tuple[str, ...], name: str) -> None:
    if len(player_ids) != 11 or len(set(player_ids)) != 11:
        raise ValueError(f"{name} must contain 11 unique players")


def _contracts_by_role(
    contracts: tuple[RoleMetricContract, ...],
) -> dict[str, RoleMetricContract]:
    by_role: dict[str, RoleMetricContract] = {}
    for contract in contracts:
        if contract.role in by_role:
            raise ValueError(f"duplicate role contract: {contract.role}")
        by_role[contract.role] = contract
    return by_role


def _applicable_values(
    player_ids: tuple[str, ...],
    *,
    dimension: str,
    by_player: dict[str, PlayerLineupValue],
    contracts: dict[str, RoleMetricContract],
) -> tuple[float, ...] | None:
    applicable_player_ids = tuple(
        player_id
        for player_id in player_ids
        if player_id in contracts and dimension in contracts[player_id].applicable_metrics
    )
    if not applicable_player_ids:
        return None
    values = tuple(
        by_player[player_id].dimensions.get(dimension) for player_id in applicable_player_ids
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values if value is not None)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)
