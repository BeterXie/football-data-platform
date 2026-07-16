"""Role-based player profiles with explicit samples and missingness."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from football_data_platform.domain.models import require_utc


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

    def __post_init__(self) -> None:
        require_utc(self.known_at, "known_at")
        if not math.isfinite(self.minutes) or self.minutes < 0:
            raise ValueError("minutes must be finite and non-negative")
        for metric, value in self.metrics.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"metric {metric!r} must be finite or None")


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    metric: str
    total: float
    per90: float | None
    sample_minutes: float
    role_percentile: float | None


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    artifact_id: str
    schema_version: int
    player_id: str
    team_id: str
    role: str
    as_of: datetime
    transform_version: str
    total_minutes: float
    sample_matches: int
    metrics: tuple[ProfileMetric, ...]
    input_refs: tuple[str, ...]
    quality_status: str


@dataclass(frozen=True, slots=True)
class ProfileBuildResult:
    profiles: tuple[PlayerProfile, ...]
    excluded_input_refs: tuple[str, ...]


def build_player_profiles(
    observations: tuple[PlayerMatchObservation, ...],
    *,
    as_of: datetime,
    minimum_minutes: float,
    transform_version: str = "player-profile/1",
) -> ProfileBuildResult:
    """Aggregate per-90 metrics without converting missing values to zero."""

    require_utc(as_of, "as_of")
    if not math.isfinite(minimum_minutes) or minimum_minutes < 0:
        raise ValueError("minimum_minutes must be finite and non-negative")
    eligible = tuple(observation for observation in observations if observation.known_at <= as_of)
    excluded = tuple(
        sorted(
            observation.source_ref for observation in observations if observation.known_at > as_of
        )
    )
    by_player: dict[str, list[PlayerMatchObservation]] = {}
    for observation in eligible:
        by_player.setdefault(observation.player_id, []).append(observation)

    raw_profiles: dict[str, dict[str, Any]] = {}
    for player_id, rows in by_player.items():
        role_minutes: Counter[str] = Counter()
        for row in rows:
            role_minutes[row.role] += row.minutes
        role = sorted(role_minutes, key=lambda item: (-role_minutes[item], item))[0]
        metric_names = sorted({name for row in rows for name in row.metrics})
        metric_values: dict[str, tuple[float, float, float | None]] = {}
        for metric in metric_names:
            available = [row for row in rows if row.metrics.get(metric) is not None]
            if not available:
                continue
            total = math.fsum(float(row.metrics[metric]) for row in available)
            sample_minutes = math.fsum(row.minutes for row in available)
            per90 = total / sample_minutes * 90.0 if sample_minutes > 0 else None
            metric_values[metric] = (total, sample_minutes, per90)
        total_minutes = math.fsum(row.minutes for row in rows)
        team_minutes: Counter[str] = Counter()
        for row in rows:
            team_minutes[row.team_id] += row.minutes
        team_id = sorted(team_minutes, key=lambda item: (-team_minutes[item], item))[0]
        raw_profiles[player_id] = {
            "team_id": team_id,
            "role": role,
            "total_minutes": total_minutes,
            "sample_matches": len({row.match_id for row in rows}),
            "metric_values": metric_values,
            "input_refs": tuple(sorted({row.source_ref for row in rows})),
        }

    profiles: list[PlayerProfile] = []
    for player_id, raw in sorted(raw_profiles.items()):
        metrics = tuple(
            ProfileMetric(
                metric=metric,
                total=values[0],
                sample_minutes=values[1],
                per90=values[2],
                role_percentile=_role_percentile(
                    raw_profiles,
                    role=raw["role"],
                    metric=metric,
                    value=values[2],
                ),
            )
            for metric, values in sorted(raw["metric_values"].items())
        )
        quality_status = (
            "ready" if raw["total_minutes"] >= minimum_minutes else "insufficient-minutes"
        )
        identity = {
            "schema_version": 1,
            "player_id": player_id,
            "team_id": raw["team_id"],
            "role": raw["role"],
            "as_of": _timestamp(as_of),
            "transform_version": transform_version,
            "total_minutes": raw["total_minutes"],
            "sample_matches": raw["sample_matches"],
            "metrics": [
                {
                    "metric": metric.metric,
                    "total": metric.total,
                    "per90": metric.per90,
                    "sample_minutes": metric.sample_minutes,
                    "role_percentile": metric.role_percentile,
                }
                for metric in metrics
            ],
            "input_refs": raw["input_refs"],
            "quality_status": quality_status,
        }
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        profiles.append(
            PlayerProfile(
                artifact_id=f"player-profile:{digest}",
                schema_version=1,
                player_id=player_id,
                team_id=raw["team_id"],
                role=raw["role"],
                as_of=as_of,
                transform_version=transform_version,
                total_minutes=raw["total_minutes"],
                sample_matches=raw["sample_matches"],
                metrics=metrics,
                input_refs=raw["input_refs"],
                quality_status=quality_status,
            )
        )
    return ProfileBuildResult(tuple(profiles), excluded)


def _role_percentile(
    profiles: dict[str, dict[str, Any]],
    *,
    role: str,
    metric: str,
    value: float | None,
) -> float | None:
    if value is None:
        return None
    peers = sorted(
        metric_values[metric][2]
        for profile in profiles.values()
        if profile["role"] == role
        for metric_values in (profile["metric_values"],)
        if metric in metric_values and metric_values[metric][2] is not None
    )
    if not peers:
        return None
    below = sum(peer < value for peer in peers)
    equal = sum(peer == value for peer in peers)
    return (below + 0.5 * equal) / len(peers) * 100.0


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
