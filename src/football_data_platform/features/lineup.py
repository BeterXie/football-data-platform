"""Lineup deltas that preserve missing player evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerLineupValue:
    player_id: str
    dimensions: dict[str, float | None]
    profile_ref: str

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


def build_lineup_delta(
    *,
    starter_ids: tuple[str, ...],
    reference_starter_ids: tuple[str, ...] | None,
    player_values: tuple[PlayerLineupValue, ...],
) -> LineupDelta:
    """Compare two XI vectors or return an explicit preview when evidence is missing."""

    if len(starter_ids) != 11 or len(set(starter_ids)) != 11:
        raise ValueError("starter_ids must contain 11 unique players")
    if reference_starter_ids is not None and (
        len(reference_starter_ids) != 11 or len(set(reference_starter_ids)) != 11
    ):
        raise ValueError("reference_starter_ids must contain 11 unique players")
    by_player = {value.player_id: value for value in player_values}
    required_players = set(starter_ids) | set(reference_starter_ids or ())
    missing: list[str] = [] if reference_starter_ids is not None else ["reference_lineup"]
    missing.extend(
        f"player_profile:{player_id}" for player_id in sorted(required_players - set(by_player))
    )
    dimensions = sorted(
        {
            name
            for player_id in required_players & set(by_player)
            for name in by_player[player_id].dimensions
        }
    )
    for player_id in sorted(required_players & set(by_player)):
        for dimension in dimensions:
            if by_player[player_id].dimensions.get(dimension) is None:
                missing.append(f"player_dimension:{player_id}:{dimension}")
    input_refs = tuple(
        sorted(
            {by_player[player_id].profile_ref for player_id in required_players & set(by_player)}
        )
    )
    if missing:
        return LineupDelta("preview", {}, tuple(missing), input_refs)

    assert reference_starter_ids is not None

    deltas = {
        dimension: _mean(
            float(by_player[player_id].dimensions[dimension]) for player_id in starter_ids
        )
        - _mean(
            float(by_player[player_id].dimensions[dimension]) for player_id in reference_starter_ids
        )
        for dimension in dimensions
    }
    return LineupDelta("ready", deltas, (), input_refs)


def _mean(values) -> float:
    materialized = tuple(values)
    return math.fsum(materialized) / len(materialized)
