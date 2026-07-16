from __future__ import annotations

import pytest

from football_data_platform.features.lineup import (
    PlayerLineupValue,
    build_lineup_delta,
)
from football_data_platform.features.player_profiles import RoleMetricContract


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


def test_lineup_delta_is_preview_when_dimensions_are_empty() -> None:
    starters = tuple(f"starter-{index}" for index in range(11))
    reference = tuple(f"reference-{index}" for index in range(11))
    values = tuple(
        PlayerLineupValue(player_id, {}, f"profile:{player_id}")
        for player_id in (*starters, *reference)
    )

    delta = build_lineup_delta(
        starter_ids=starters,
        reference_starter_ids=reference,
        player_values=values,
    )

    assert delta.quality_status == "preview"
    assert delta.dimension_deltas == {}
    assert "no_comparable_dimensions" in delta.blocking_reasons


def test_lineup_delta_uses_role_applicability_and_separates_na_from_missing() -> None:
    starters = tuple(f"starter-{index}" for index in range(11))
    reference = tuple(f"reference-{index}" for index in range(11))
    outfield = RoleMetricContract(
        role="outfield",
        version="lineup-roles/1",
        required_metrics=("attack",),
        optional_metrics=(),
        not_applicable_metrics=("saves",),
        minimum_minutes=1,
        minimum_matches=1,
    )
    goalkeeper = RoleMetricContract(
        role="goalkeeper",
        version="lineup-roles/1",
        required_metrics=("saves",),
        optional_metrics=(),
        not_applicable_metrics=("attack",),
        minimum_minutes=1,
        minimum_matches=1,
    )
    values = tuple(
        PlayerLineupValue(
            player_id,
            {"attack": 2.0},
            f"profile:{player_id}",
            role="outfield",
        )
        for player_id in (*starters[:10], *reference[:10])
    ) + (
        PlayerLineupValue(starters[10], {"saves": 3.0}, "profile:starter-gk", "goalkeeper"),
        PlayerLineupValue(reference[10], {"saves": 2.0}, "profile:reference-gk", "goalkeeper"),
    )

    delta = build_lineup_delta(
        starter_ids=starters,
        reference_starter_ids=reference,
        player_values=values,
        role_contracts=(outfield, goalkeeper),
    )

    assert delta.quality_status == "ready"
    assert delta.dimension_deltas == {"attack": pytest.approx(0.0), "saves": pytest.approx(1.0)}
    assert not delta.missing_fields
    assert len(delta.not_applicable_fields) == 22


def test_missing_required_role_dimension_blocks_lineup_delta() -> None:
    starters = tuple(f"starter-{index}" for index in range(11))
    reference = tuple(f"reference-{index}" for index in range(11))
    values = tuple(
        PlayerLineupValue(player_id, {"attack": 2.0}, f"profile:{player_id}")
        for player_id in (*starters, *reference)
    )
    values = (
        *values[:-1],
        PlayerLineupValue(
            reference[-1],
            {"attack": None},
            "profile:missing",
            profile_quality_status="insufficient-sample",
        ),
    )

    delta = build_lineup_delta(
        starter_ids=starters,
        reference_starter_ids=reference,
        player_values=values,
    )

    assert delta.quality_status == "preview"
    assert f"player_dimension:{reference[-1]}:attack" in delta.missing_fields
    assert "missing_required_dimensions" in delta.blocking_reasons
    assert "player_profiles_not_ready" in delta.blocking_reasons
