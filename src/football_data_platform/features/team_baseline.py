"""Opponent-adjusted, time-decayed team process baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.models import require_utc

TEAM_BASELINE_SCHEMA_VERSION = 2
TEAM_BASELINE_COORDINATE_VERSION = "away-mean-neutral/1"
TEAM_BASELINE_INPUT_TRANSFORM_V3 = "team-baseline-input/3"


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
    coordinate_version: str = TEAM_BASELINE_COORDINATE_VERSION


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
    transform_version: str = "team-baseline/2",
) -> BaselineBuildResult:
    """Build a baseline using only observations knowable at ``as_of``."""

    require_utc(as_of, "as_of")
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be finite and positive")
    half_life_days = float(half_life_days)
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
    away_xg = math.fsum(weight * observation.away_xg for observation, weight in weighted_matches)
    home_xg = math.fsum(weight * observation.home_xg for observation, weight in weighted_matches)
    # Use the away scoring mean as the neutral coordinate.  Home observations
    # are venue-adjusted before fitting strengths, and the advantage is applied
    # exactly once when producing the home lambda.  The previous overall mean
    # coordinate multiplied by the home ratio a second time.
    league_xg = away_xg / total_weight
    if league_xg <= 0:
        raise ValueError("away league xG mean must be positive")
    home_advantage = home_xg / away_xg if home_xg > 0 and away_xg > 0 else 1.0
    if not math.isfinite(home_advantage) or home_advantage <= 0:
        raise ValueError("home advantage must be finite and positive")

    team_observations: dict[str, list[tuple[float, float, str, float]]] = {}
    for observation, weight in weighted_matches:
        team_observations.setdefault(observation.home_team_id, []).append(
            (
                observation.home_xg / home_advantage,
                observation.away_xg,
                observation.away_team_id,
                weight,
            )
        )
        team_observations.setdefault(observation.away_team_id, []).append(
            (
                observation.away_xg,
                observation.home_xg / home_advantage,
                observation.home_team_id,
                weight,
            )
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
        "schema_version": TEAM_BASELINE_SCHEMA_VERSION,
        "coordinate_version": TEAM_BASELINE_COORDINATE_VERSION,
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
        schema_version=TEAM_BASELINE_SCHEMA_VERSION,
        as_of=as_of,
        transform_version=transform_version,
        half_life_days=half_life_days,
        iterations=iterations,
        league_xg_per_team=league_xg,
        home_advantage=home_advantage,
        teams=teams,
        input_refs=input_refs,
        quality_status="ready",
        coordinate_version=TEAM_BASELINE_COORDINATE_VERSION,
    )
    verify_team_baseline_artifact(artifact)
    return BaselineBuildResult(artifact, excluded)


def expected_goals_from_baseline(
    baseline: TeamBaselineArtifact,
    *,
    home_team_id: str,
    away_team_id: str,
) -> tuple[float, float]:
    verify_team_baseline_artifact(baseline)
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


def team_baseline_identity(artifact: TeamBaselineArtifact) -> dict[str, Any]:
    """Return the content that determines a baseline artifact ID."""

    return {
        "schema_version": artifact.schema_version,
        "coordinate_version": artifact.coordinate_version,
        "as_of": _timestamp(artifact.as_of),
        "transform_version": artifact.transform_version,
        "half_life_days": artifact.half_life_days,
        "iterations": artifact.iterations,
        "league_xg_per_team": artifact.league_xg_per_team,
        "home_advantage": artifact.home_advantage,
        "teams": [asdict(team) for team in artifact.teams],
        "input_refs": list(artifact.input_refs),
        "quality_status": artifact.quality_status,
    }


def team_baseline_payload(artifact: TeamBaselineArtifact) -> dict[str, Any]:
    """Return a stable, complete JSON payload for derived storage."""

    verify_team_baseline_artifact(artifact)
    return {"artifact_id": artifact.artifact_id, **team_baseline_identity(artifact)}


def parse_team_baseline_payload(payload: Any) -> TeamBaselineArtifact:
    """Parse and verify one persisted baseline payload."""

    if not isinstance(payload, dict):
        raise ValueError("team baseline payload must be an object")
    try:
        as_of = datetime.fromisoformat(str(payload["as_of"]).replace("Z", "+00:00"))
        teams = tuple(TeamStrength(**item) for item in payload["teams"])
        artifact = TeamBaselineArtifact(
            artifact_id=str(payload["artifact_id"]),
            schema_version=int(payload["schema_version"]),
            as_of=as_of,
            transform_version=str(payload["transform_version"]),
            half_life_days=float(payload["half_life_days"]),
            iterations=int(payload["iterations"]),
            league_xg_per_team=float(payload["league_xg_per_team"]),
            home_advantage=float(payload["home_advantage"]),
            teams=teams,
            input_refs=tuple(str(item) for item in payload["input_refs"]),
            quality_status=str(payload["quality_status"]),
            coordinate_version=str(
                payload.get("coordinate_version", TEAM_BASELINE_COORDINATE_VERSION)
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid team baseline payload: {error}") from error
    verify_team_baseline_artifact(artifact)
    if team_baseline_payload(artifact) != payload:
        raise ValueError("team baseline payload is not canonical")
    return artifact


def verify_team_baseline_artifact(artifact: TeamBaselineArtifact) -> None:
    """Verify the immutable identity and quality contract of a baseline."""

    if not isinstance(artifact, TeamBaselineArtifact):
        raise TypeError("baseline must be a TeamBaselineArtifact")
    if artifact.schema_version != TEAM_BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported team baseline schema version")
    if artifact.coordinate_version != TEAM_BASELINE_COORDINATE_VERSION:
        raise ValueError("unsupported team baseline coordinate version")
    require_utc(artifact.as_of, "baseline as_of")
    if (
        not artifact.transform_version
        or artifact.transform_version.strip() != artifact.transform_version
    ):
        raise ValueError("baseline transform_version must be non-empty text")
    if (
        not math.isfinite(artifact.half_life_days)
        or artifact.half_life_days <= 0
        or artifact.iterations < 1
        or not math.isfinite(artifact.league_xg_per_team)
        or artifact.league_xg_per_team <= 0
        or not math.isfinite(artifact.home_advantage)
        or artifact.home_advantage <= 0
    ):
        raise ValueError("baseline calibration values must be finite and positive")
    if artifact.quality_status != "ready" or not artifact.teams:
        raise ValueError("team baseline artifact is not ready")
    team_ids = [team.team_id for team in artifact.teams]
    if len(team_ids) != len(set(team_ids)):
        raise ValueError("team baseline contains duplicate teams")
    for team in artifact.teams:
        if (
            not team.team_id
            or not math.isfinite(team.attack)
            or team.attack <= 0
            or not math.isfinite(team.defense)
            or team.defense <= 0
            or team.sample_matches < 1
        ):
            raise ValueError("team baseline contains invalid team strength")
    if not artifact.input_refs or any(not ref or ref.strip() != ref for ref in artifact.input_refs):
        raise ValueError("team baseline requires input references")
    digest = hashlib.sha256(_canonical_json(team_baseline_identity(artifact))).hexdigest()
    if artifact.artifact_id != f"team-baseline:{digest}":
        raise ValueError("team baseline artifact identity does not match its content")
