"""The single probability surface used by football prediction consumers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

ScorePredicate = Callable[[int, int], bool]


def _require_finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    """Return the Dixon-Coles low-score correction factor.

    ``rho`` is a low-score dependence correction. It is not a general
    correlation coefficient.
    """

    if (home_goals, away_goals) == (0, 0):
        return 1.0 - lambda_home * lambda_away * rho
    if (home_goals, away_goals) == (0, 1):
        return 1.0 + lambda_home * rho
    if (home_goals, away_goals) == (1, 0):
        return 1.0 + lambda_away * rho
    if (home_goals, away_goals) == (1, 1):
        return 1.0 - rho
    return 1.0


def _poisson_probability(goals: int, expectation: float) -> float:
    return math.exp(-expectation) * expectation**goals / math.factorial(goals)


@dataclass(frozen=True, slots=True)
class DixonColesGrid:
    """A normalized, truncated joint score distribution.

    All result and market views must aggregate this object. Consumers must not
    reconstruct independent Poisson marginals or apply another truncation.
    """

    lambda_home: float
    lambda_away: float
    rho: float = -0.1
    max_goals: int = 11
    _probabilities: tuple[tuple[float, ...], ...] = field(init=False, repr=False)
    unnormalized_mass: float = field(init=False)

    def __post_init__(self) -> None:
        lambda_home = _require_finite_positive(self.lambda_home, "lambda_home")
        lambda_away = _require_finite_positive(self.lambda_away, "lambda_away")
        rho = float(self.rho)
        if not math.isfinite(rho):
            raise ValueError("rho must be finite")
        if isinstance(self.max_goals, bool) or not isinstance(self.max_goals, int):
            raise TypeError("max_goals must be an integer")
        if self.max_goals < 1:
            raise ValueError("max_goals must be at least 1")

        raw_rows: list[tuple[float, ...]] = []
        for home_goals in range(self.max_goals + 1):
            row: list[float] = []
            for away_goals in range(self.max_goals + 1):
                tau = dixon_coles_tau(
                    home_goals,
                    away_goals,
                    lambda_home,
                    lambda_away,
                    rho,
                )
                value = (
                    tau
                    * _poisson_probability(home_goals, lambda_home)
                    * _poisson_probability(away_goals, lambda_away)
                )
                if value < 0:
                    raise ValueError(
                        "rho produces a negative Dixon-Coles probability at "
                        f"({home_goals}, {away_goals})"
                    )
                row.append(value)
            raw_rows.append(tuple(row))

        mass = math.fsum(value for row in raw_rows for value in row)
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("score grid has non-positive probability mass")

        object.__setattr__(self, "lambda_home", lambda_home)
        object.__setattr__(self, "lambda_away", lambda_away)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "unnormalized_mass", mass)
        object.__setattr__(
            self,
            "_probabilities",
            tuple(tuple(value / mass for value in row) for row in raw_rows),
        )

    @property
    def normalization_residual(self) -> float:
        return abs(1.0 - math.fsum(value for row in self._probabilities for value in row))

    def probability(self, home_goals: int, away_goals: int) -> float:
        """Return a normalized exact-score probability within the grid."""

        if not (0 <= home_goals <= self.max_goals):
            return 0.0
        if not (0 <= away_goals <= self.max_goals):
            return 0.0
        return self._probabilities[home_goals][away_goals]

    def sum_where(self, predicate: ScorePredicate) -> float:
        """Aggregate normalized cells matching ``predicate``."""

        return math.fsum(
            self._probabilities[home_goals][away_goals]
            for home_goals in range(self.max_goals + 1)
            for away_goals in range(self.max_goals + 1)
            if predicate(home_goals, away_goals)
        )

    def result_probabilities(self) -> dict[str, float]:
        """Return 90-minute home/draw/away probabilities."""

        return {
            "home": self.sum_where(lambda home, away: home > away),
            "draw": self.sum_where(lambda home, away: home == away),
            "away": self.sum_where(lambda home, away: home < away),
        }

    def handicap_probabilities(self, home_handicap: int) -> dict[str, float]:
        """Return three-way probabilities after an integer home handicap."""

        if isinstance(home_handicap, bool) or not isinstance(home_handicap, int):
            raise TypeError("home_handicap must be an integer")
        return {
            "home": self.sum_where(lambda home, away: home + home_handicap > away),
            "draw": self.sum_where(lambda home, away: home + home_handicap == away),
            "away": self.sum_where(lambda home, away: home + home_handicap < away),
        }

    def total_goals_probabilities(self, exact_through: int = 6) -> dict[str, float]:
        """Return exact total-goal buckets followed by one upper-tail bucket."""

        if isinstance(exact_through, bool) or not isinstance(exact_through, int):
            raise TypeError("exact_through must be an integer")
        if exact_through < 0 or exact_through >= self.max_goals * 2:
            raise ValueError("exact_through must leave a non-empty upper-tail bucket")
        probabilities = {
            str(total): self.sum_where(lambda home, away, expected=total: home + away == expected)
            for total in range(exact_through + 1)
        }
        probabilities[f"{exact_through + 1}+"] = self.sum_where(
            lambda home, away: home + away > exact_through
        )
        return probabilities

    def score_cells(self) -> tuple[dict[str, int | float], ...]:
        """Return the canonical, coordinate-explicit score-cell representation."""

        return tuple(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability": self._probabilities[home_goals][away_goals],
            }
            for home_goals in range(self.max_goals + 1)
            for away_goals in range(self.max_goals + 1)
        )
