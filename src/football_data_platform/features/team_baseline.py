"""Opponent-adjusted, time-decayed team process baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.models import require_utc


@dataclass(frozen=True, slots=True)
class TeamMatchProcess:
    match_id: str
    home_team_id: str
    away_team_id: str
    kickoff_at: datetime
    known_at: datetime
    home_xg: float
    away_xg: float
    source_ref: str

    def __post_init__(self) -> None:
        require_utc(self.kickoff_at, "kickoff_at")
        require_utc(self.known_at, "known_at")
        for name, value in (("home_xg", self.home_xg), ("away_xg", self.away_xg)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TeamStrength:
    team_id: str
    attack: float
    defense: float
    sample_matches: int


@dataclass(frozen=True, slots=True)
class TeamBaselineArtifact:
    artifact_id: str
    schema_version: int
    as_of: datetime
    transform_version: str
    half_life_days: float
    iterations: int
    league_xg_per_team: float
    home_advantage: float
    teams: tuple[TeamStrength, ...]
    input_refs: tuple[str, ...]
    quality_status: str


@dataclass(frozen=True, slots=True)
class BaselineBuildResult:
    artifact: TeamBaselineArtifact
    excluded_input_refs: tuple[str, ...]


def build_team_baseline(
    observations: tuple[TeamMatchProcess, ...],
    *,
    as_of: datetime,
    half_life_days: float,
    iterations: int,
    transform_version: str = "team-baseline/1",
) -> BaselineBuildResult:
    """Build a baseline using only observations knowable at ``as_of``."""

    require_utc(as_of, "as_of")
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    eligible = tuple(
        observation
        for observation in observations
        if observation.known_at <= as_of and observation.kickoff_at < as_of
    )
    excluded = tuple(
        sorted(
            observation.source_ref for observation in observations if observation not in eligible
        )
    )
    if not eligible:
        raise ValueError("no time-eligible team match observations")

    weighted_matches = tuple(
        (observation, _time_weight(observation.kickoff_at, as_of, half_life_days))
        for observation in eligible
    )
    total_weight = math.fsum(weight for _, weight in weighted_matches)
    league_xg = math.fsum(
        weight * (observation.home_xg + observation.away_xg)
        for observation, weight in weighted_matches
    ) / (2.0 * total_weight)
    if league_xg <= 0:
        raise ValueError("league xG mean must be positive")
    away_xg = math.fsum(weight * observation.away_xg for observation, weight in weighted_matches)
    home_xg = math.fsum(weight * observation.home_xg for observation, weight in weighted_matches)
    home_advantage = home_xg / away_xg if away_xg > 0 else 1.0

    team_observations: dict[str, list[tuple[float, float, str, float]]] = {}
    for observation, weight in weighted_matches:
        team_observations.setdefault(observation.home_team_id, []).append(
            (observation.home_xg, observation.away_xg, observation.away_team_id, weight)
        )
        team_observations.setdefault(observation.away_team_id, []).append(
            (observation.away_xg, observation.home_xg, observation.home_team_id, weight)
        )
    attack = dict.fromkeys(team_observations, 1.0)
    defense = dict.fromkeys(team_observations, 1.0)
    for _ in range(iterations):
        next_attack: dict[str, float] = {}
        next_defense: dict[str, float] = {}
        for team_id, rows in team_observations.items():
            weight_sum = math.fsum(row[3] for row in rows)
            next_attack[team_id] = (
                math.fsum(
                    weight * (xg_for / league_xg) / max(defense[opponent], 1e-9)
                    for xg_for, _, opponent, weight in rows
                )
                / weight_sum
            )
            next_defense[team_id] = (
                math.fsum(
                    weight * (xg_against / league_xg) / max(attack[opponent], 1e-9)
                    for _, xg_against, opponent, weight in rows
                )
                / weight_sum
            )
        attack = _normalize_strengths(next_attack)
        defense = _normalize_strengths(next_defense)

    teams = tuple(
        TeamStrength(
            team_id=team_id,
            attack=attack[team_id],
            defense=defense[team_id],
            sample_matches=len(team_observations[team_id]),
        )
        for team_id in sorted(team_observations)
    )
    input_refs = tuple(sorted({observation.source_ref for observation in eligible}))
    identity = {
        "schema_version": 1,
        "as_of": _timestamp(as_of),
        "transform_version": transform_version,
        "half_life_days": half_life_days,
        "iterations": iterations,
        "league_xg_per_team": league_xg,
        "home_advantage": home_advantage,
        "teams": [asdict(team) for team in teams],
        "input_refs": input_refs,
        "quality_status": "ready",
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    artifact = TeamBaselineArtifact(
        artifact_id=f"team-baseline:{digest}",
        schema_version=1,
        as_of=as_of,
        transform_version=transform_version,
        half_life_days=half_life_days,
        iterations=iterations,
        league_xg_per_team=league_xg,
        home_advantage=home_advantage,
        teams=teams,
        input_refs=input_refs,
        quality_status="ready",
    )
    return BaselineBuildResult(artifact, excluded)


def expected_goals_from_baseline(
    baseline: TeamBaselineArtifact,
    *,
    home_team_id: str,
    away_team_id: str,
) -> tuple[float, float]:
    strengths = {team.team_id: team for team in baseline.teams}
    try:
        home = strengths[home_team_id]
        away = strengths[away_team_id]
    except KeyError as error:
        raise KeyError(f"team {error.args[0]!r} is absent from baseline") from None
    return (
        baseline.league_xg_per_team * home.attack * away.defense * baseline.home_advantage,
        baseline.league_xg_per_team * away.attack * home.defense,
    )


def _normalize_strengths(values: dict[str, float]) -> dict[str, float]:
    mean = math.fsum(max(value, 1e-9) for value in values.values()) / len(values)
    return {key: max(value, 1e-9) / mean for key, value in values.items()}


def _time_weight(kickoff_at: datetime, as_of: datetime, half_life_days: float) -> float:
    age_days = (as_of - kickoff_at).total_seconds() / 86_400
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
