"""Role-aware lineup deltas that preserve missing and N/A evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from football_data_platform.features.player_profiles import RoleMetricContract


@dataclass(frozen=True, slots=True)
class PlayerLineupValue:
    player_id: str
    dimensions: dict[str, float | None]
    profile_ref: str
    role: str = "generic"
    profile_quality_status: str = "ready"

    def __post_init__(self) -> None:
        for name, value in self.dimensions.items():
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
    quality_status = "ready" if not blockers else "preview"
    if quality_status != "ready":
        deltas = {}
        sample_sizes = {}
    return LineupDelta(
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
