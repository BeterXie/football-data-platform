import math

import pytest

from football_data_platform.models.score_grid import (
    DixonColesGrid,
    dixon_coles_tau,
)


def test_dixon_coles_tau_uses_formal_lambda_orientation() -> None:
    lambda_home = 1.7
    lambda_away = 0.8
    rho = -0.12

    assert dixon_coles_tau(0, 0, lambda_home, lambda_away, rho) == pytest.approx(
        1 - lambda_home * lambda_away * rho
    )
    assert dixon_coles_tau(0, 1, lambda_home, lambda_away, rho) == pytest.approx(
        1 + lambda_home * rho
    )
    assert dixon_coles_tau(1, 0, lambda_home, lambda_away, rho) == pytest.approx(
        1 + lambda_away * rho
    )
    assert dixon_coles_tau(1, 1, lambda_home, lambda_away, rho) == pytest.approx(1 - rho)
    assert dixon_coles_tau(2, 0, lambda_home, lambda_away, rho) == 1.0


def test_dixon_coles_tau_matches_regression_values_from_legacy_bug() -> None:
    assert dixon_coles_tau(0, 0, 1.4, 0.8, -0.1) == pytest.approx(1.112)
    assert dixon_coles_tau(0, 1, 1.4, 0.8, -0.1) == pytest.approx(0.86)
    assert dixon_coles_tau(1, 0, 1.4, 0.8, -0.1) == pytest.approx(0.92)
    assert dixon_coles_tau(1, 1, 1.4, 0.8, -0.1) == pytest.approx(1.1)


def test_grid_cell_ratios_use_the_same_tau_definition() -> None:
    grid = DixonColesGrid(1.7, 0.8, rho=-0.12, max_goals=11)

    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        independent = (
            math.exp(-grid.lambda_home)
            * grid.lambda_home**home_goals
            / math.factorial(home_goals)
            * math.exp(-grid.lambda_away)
            * grid.lambda_away**away_goals
            / math.factorial(away_goals)
        )
        raw_adjusted = grid.probability(home_goals, away_goals) * grid.unnormalized_mass
        assert raw_adjusted / independent == pytest.approx(
            dixon_coles_tau(
                home_goals,
                away_goals,
                grid.lambda_home,
                grid.lambda_away,
                grid.rho,
            )
        )


def test_all_market_views_are_normalized_aggregations_of_one_grid() -> None:
    grid = DixonColesGrid(1.45, 1.05, rho=-0.1, max_goals=11)

    assert grid.normalization_residual < 1e-12
    assert sum(grid.result_probabilities().values()) == pytest.approx(1.0)
    assert sum(grid.handicap_probabilities(-1).values()) == pytest.approx(1.0)
    assert sum(grid.total_goals_probabilities().values()) == pytest.approx(1.0)
    assert sum(cell["probability"] for cell in grid.score_cells()) == pytest.approx(1.0)
    assert len(grid.score_cells()) == 12 * 12
    assert len({(cell["home_goals"], cell["away_goals"]) for cell in grid.score_cells()}) == len(
        grid.score_cells()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_home": 0, "lambda_away": 1}, "lambda_home"),
        ({"lambda_home": 1, "lambda_away": float("nan")}, "lambda_away"),
        ({"lambda_home": 1, "lambda_away": 1, "rho": float("inf")}, "rho"),
        ({"lambda_home": 1, "lambda_away": 1, "max_goals": 0}, "max_goals"),
        ({"lambda_home": 1, "lambda_away": 1, "rho": 2}, "negative"),
    ],
)
def test_invalid_grid_parameters_fail_loudly(kwargs: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DixonColesGrid(**kwargs)
