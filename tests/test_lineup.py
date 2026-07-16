from __future__ import annotations

import pytest

from football_data_platform.features.lineup import (
    PlayerLineupValue,
    build_lineup_delta,
)


def test_lineup_delta_is_preview_instead_of_neutral_when_profiles_are_missing() -> None:
    starters = tuple(f"starter-{index}" for index in range(11))
    reference = tuple(f"reference-{index}" for index in range(11))

    delta = build_lineup_delta(
        starter_ids=starters,
        reference_starter_ids=reference,
        player_values=(),
    )

    assert delta.quality_status == "preview"
    assert delta.dimension_deltas == {}
    assert len(delta.missing_fields) == 22


def test_complete_lineup_delta_compares_versioned_dimensions() -> None:
    starters = tuple(f"starter-{index}" for index in range(11))
    reference = tuple(f"reference-{index}" for index in range(11))
    values = tuple(
        PlayerLineupValue(player_id, {"attack": 2.0}, f"profile:{player_id}")
        for player_id in starters
    ) + tuple(
        PlayerLineupValue(player_id, {"attack": 1.5}, f"profile:{player_id}")
        for player_id in reference
    )

    delta = build_lineup_delta(
        starter_ids=starters,
        reference_starter_ids=reference,
        player_values=values,
    )

    assert delta.quality_status == "ready"
    assert delta.dimension_deltas["attack"] == pytest.approx(0.5)
    assert not delta.missing_fields
