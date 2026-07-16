"""Time-honest, immutable pre-match snapshot contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from football_data_platform.domain.ids import MatchId, PlayerId, SnapshotId, TeamId
from football_data_platform.domain.models import require_utc

SNAPSHOT_SCHEMA_VERSION = 2
_DERIVED_REF = re.compile(r"^derived-source:[0-9a-f]{64}$")


class SnapshotType(StrEnum):
    T24H = "t24h"
    LINEUPS_CONFIRMED = "lineups-confirmed"


class CaptureMode(StrEnum):
    CAPTURED = "captured"
    RECONSTRUCTED = "reconstructed"


@dataclass(frozen=True, slots=True)
class SnapshotSourceValidation:
    source_ref: str
    source_kind: str
    transform_version: str
    observed_at: datetime
    value: Any | None
    input_refs: tuple[str, ...]


class SnapshotSourceValidator(Protocol):
    def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation: ...


@dataclass(frozen=True, slots=True)
class SnapshotFeature:
    name: str
    value: Any
    known_at: datetime
    source_ref: str
    contribution_key: str
    entity_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.source_ref, "source_ref")
        if not (
            self.source_ref.startswith("canonical:") or _DERIVED_REF.fullmatch(self.source_ref)
        ):
            raise ValueError("source_ref must be a typed canonical or derived reference")
        _require_text(self.contribution_key, "contribution_key")
        if self.entity_id is not None:
            _require_text(self.entity_id, "entity_id")
        require_utc(self.known_at, "known_at")
        try:
            json.dumps(self.value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("snapshot feature value must be finite JSON data") from error


@dataclass(frozen=True, slots=True)
class PreMatchSnapshot:
    id: SnapshotId
    schema_version: int
    match_id: MatchId
    match_version: int
    home_team_id: TeamId
    away_team_id: TeamId
    snapshot_type: SnapshotType
    capture_mode: CaptureMode
    as_of: datetime
    observed_at: datetime
    scheduled_kickoff_used: datetime
    feature_spec_version: str
    features: tuple[SnapshotFeature, ...]
    input_refs: tuple[str, ...]
    quality_status: str
    missing_fields: tuple[str, ...]


_FEATURE_SPEC_V1 = {
    SnapshotType.T24H: frozenset({"team_baseline", "match_context"}),
    SnapshotType.LINEUPS_CONFIRMED: frozenset(
        {"team_baseline", "match_context", "lineup_delta", "official_lineup_confirmed"}
    ),
}
_FEATURE_SPECS = {"prematch-features/1": _FEATURE_SPEC_V1}
_FEATURE_TRANSFORMS = {
    "team_baseline": frozenset({"team-baseline-input/1", "team-baseline-input/2"}),
    "match_context": frozenset({"match-context-input/1"}),
    "lineup_delta": frozenset({"lineup-delta-input/1"}),
    "official_lineup_confirmed": frozenset({"official-lineup-input/1"}),
}


def build_snapshot(
    *,
    match_id: MatchId,
    match_version: int,
    snapshot_type: SnapshotType,
    as_of: datetime,
    scheduled_kickoff_used: datetime,
    feature_spec_version: str,
    features: tuple[SnapshotFeature, ...],
    home_team_id: TeamId,
    away_team_id: TeamId,
    source_validator: SnapshotSourceValidator,
) -> PreMatchSnapshot:
    """Build a snapshot from store-verified, versioned feature sources."""

    features, observed_at, input_refs, missing_fields = _validated_state(
        match_id,
        match_version,
        snapshot_type,
        as_of,
        scheduled_kickoff_used,
        feature_spec_version,
        features,
        home_team_id,
        away_team_id,
        source_validator,
    )
    fields = {
        "match_id": match_id,
        "match_version": match_version,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "snapshot_type": snapshot_type,
        "capture_mode": CaptureMode.RECONSTRUCTED,
        "as_of": as_of,
        "observed_at": observed_at,
        "scheduled_kickoff_used": scheduled_kickoff_used,
        "feature_spec_version": feature_spec_version,
        "features": features,
        "input_refs": input_refs,
        "quality_status": "ready" if not missing_fields else "preview",
        "missing_fields": missing_fields,
    }
    digest = hashlib.sha256(_canonical_json(_snapshot_identity(**fields))).hexdigest()
    snapshot = PreMatchSnapshot(
        id=SnapshotId(f"snapshot:{digest}"), schema_version=SNAPSHOT_SCHEMA_VERSION, **fields
    )
    verify_snapshot(snapshot, source_validator=source_validator)
    return snapshot


def verify_snapshot(
    snapshot: PreMatchSnapshot, *, source_validator: SnapshotSourceValidator
) -> None:
    """Recompute source existence, readiness, and content identity."""

    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema_version {snapshot.schema_version!r}")
    features, observed_at, input_refs, missing_fields = _validated_state(
        snapshot.match_id,
        snapshot.match_version,
        snapshot.snapshot_type,
        snapshot.as_of,
        snapshot.scheduled_kickoff_used,
        snapshot.feature_spec_version,
        snapshot.features,
        snapshot.home_team_id,
        snapshot.away_team_id,
        source_validator,
    )
    if features != snapshot.features:
        raise ValueError("snapshot features are not in canonical contribution order")
    expected_quality = "ready" if not missing_fields else "preview"
    if (snapshot.quality_status, snapshot.missing_fields) != (expected_quality, missing_fields):
        raise ValueError("snapshot quality does not match its versioned feature spec")
    if snapshot.capture_mode is not CaptureMode.RECONSTRUCTED:
        raise ValueError("captured snapshots require a platform capture-run validator")
    if (snapshot.observed_at, snapshot.input_refs) != (observed_at, input_refs):
        raise ValueError("snapshot source metadata does not match validated storage")
    digest = hashlib.sha256(_canonical_json(_snapshot_identity_from(snapshot))).hexdigest()
    if snapshot.id != SnapshotId(f"snapshot:{digest}"):
        raise ValueError("snapshot identity does not match its canonical content")


def snapshot_payload(
    snapshot: PreMatchSnapshot, *, source_validator: SnapshotSourceValidator
) -> dict[str, Any]:
    verify_snapshot(snapshot, source_validator=source_validator)
    return {"id": snapshot.id.value, **_snapshot_identity_from(snapshot)}


def _validated_state(
    match_id: MatchId,
    match_version: int,
    snapshot_type: SnapshotType,
    as_of: datetime,
    kickoff: datetime,
    spec_version: str,
    features: tuple[SnapshotFeature, ...],
    home_team_id: TeamId,
    away_team_id: TeamId,
    source_validator: SnapshotSourceValidator,
) -> tuple[tuple[SnapshotFeature, ...], datetime, tuple[str, ...], tuple[str, ...]]:
    features = _validate_inputs(
        match_id,
        match_version,
        snapshot_type,
        as_of,
        kickoff,
        spec_version,
        features,
        home_team_id,
        away_team_id,
    )
    missing = _assess_readiness(spec_version, snapshot_type, features, home_team_id, away_team_id)
    observed_at, input_refs = _validate_sources(features, source_validator)
    return features, observed_at, input_refs, missing


def _validate_inputs(
    match_id: MatchId,
    match_version: int,
    snapshot_type: SnapshotType,
    as_of: datetime,
    kickoff: datetime,
    spec_version: str,
    features: tuple[SnapshotFeature, ...],
    home_team_id: TeamId,
    away_team_id: TeamId,
) -> tuple[SnapshotFeature, ...]:
    if not isinstance(match_id, MatchId):
        raise TypeError("match_id must be a MatchId")
    if not isinstance(home_team_id, TeamId) or not isinstance(away_team_id, TeamId):
        raise TypeError("snapshot team IDs must be TeamId values")
    if home_team_id == away_team_id or match_version < 1:
        raise ValueError("snapshot requires distinct teams and a positive match_version")
    require_utc(as_of, "as_of")
    require_utc(kickoff, "scheduled_kickoff_used")
    if spec_version not in _FEATURE_SPECS:
        raise ValueError(f"unsupported snapshot feature spec {spec_version!r}")
    if as_of >= kickoff:
        raise ValueError("snapshot as_of must precede scheduled kickoff")
    if snapshot_type is SnapshotType.T24H and as_of != kickoff - timedelta(hours=24):
        raise ValueError("t24h as_of must equal scheduled kickoff minus 24 hours")
    keys: set[str] = set()
    for feature in features:
        if not isinstance(feature, SnapshotFeature):
            raise TypeError("features must contain SnapshotFeature values")
        if feature.known_at > as_of:
            raise ValueError(f"feature {feature.name!r} was known after snapshot as_of")
        if feature.contribution_key in keys:
            raise ValueError(f"duplicate feature contribution {feature.contribution_key!r}")
        keys.add(feature.contribution_key)
    return tuple(sorted(features, key=lambda item: item.contribution_key))


def _validate_sources(
    features: tuple[SnapshotFeature, ...], validator: SnapshotSourceValidator
) -> tuple[datetime, tuple[str, ...]]:
    if not features:
        raise ValueError("snapshot requires at least one versioned feature source")
    validations: dict[str, SnapshotSourceValidation] = {}
    for feature in features:
        if feature.source_ref not in validations:
            validations[feature.source_ref] = validator.validate_snapshot_source(feature.source_ref)
        result = validations[feature.source_ref]
        if result.source_ref != feature.source_ref:
            raise ValueError("snapshot source validator returned a mismatched reference")
        if result.source_kind not in {"canonical", "derived"}:
            raise ValueError("snapshot source validator returned an invalid source kind")
        if result.transform_version not in _FEATURE_TRANSFORMS[feature.name]:
            raise ValueError(f"snapshot source transform does not match feature {feature.name!r}")
        require_utc(result.observed_at, "source observed_at")
        if _canonical_json(result.value) != _canonical_json(feature.value):
            raise ValueError(f"derived source value does not match feature {feature.name!r}")
    observed_at = max(item.observed_at for item in validations.values())
    refs = set(validations)
    refs.update(ref for item in validations.values() for ref in item.input_refs)
    return observed_at, tuple(sorted(refs))


def _assess_readiness(
    spec_version: str,
    snapshot_type: SnapshotType,
    features: tuple[SnapshotFeature, ...],
    home_team_id: TeamId,
    away_team_id: TeamId,
) -> tuple[str, ...]:
    required = _FEATURE_SPECS[spec_version][snapshot_type]
    by_name: dict[str, list[SnapshotFeature]] = {}
    for feature in features:
        if feature.name not in required:
            raise ValueError(f"feature {feature.name!r} is not defined by {spec_version}")
        by_name.setdefault(feature.name, []).append(feature)
    missing = {f"feature:{name}" for name in required - set(by_name)}
    for name in ("team_baseline", "match_context", "lineup_delta"):
        if len(by_name.get(name, ())) > 1:
            raise ValueError(f"feature spec allows only one {name!r} feature")
        if name in by_name and by_name[name][0].value is None:
            missing.add(f"feature:{name}:value")
    _validate_core_values(by_name)
    if snapshot_type is SnapshotType.LINEUPS_CONFIRMED:
        _validate_lineups(by_name.get("official_lineup_confirmed", []), home_team_id, away_team_id)
        if by_name.get("lineup_delta") and by_name["lineup_delta"][0].value is not None:
            missing.update(
                _lineup_delta_missing(by_name["lineup_delta"][0].value, home_team_id, away_team_id)
            )
    return tuple(sorted(missing))


def _validate_core_values(by_name: dict[str, list[SnapshotFeature]]) -> None:
    if by_name.get("team_baseline") and (value := by_name["team_baseline"][0].value) is not None:
        numbers = (
            (value.get("lambda_home"), value.get("lambda_away")) if isinstance(value, dict) else ()
        )
        if (
            not isinstance(value, dict)
            or not re.fullmatch(r"team-baseline:[0-9a-f]{64}", str(value.get("artifact_id")))
            or len(numbers) != 2
            or any(
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or number <= 0
                for number in numbers
            )
        ):
            raise ValueError("team_baseline does not satisfy prematch-features/1")
    if by_name.get("match_context") and (value := by_name["match_context"][0].value) is not None:
        days = value.get("days_since_previous_match") if isinstance(value, dict) else None
        if not isinstance(days, (int, float)) or isinstance(days, bool) or days < 0:
            raise ValueError("match_context does not satisfy prematch-features/1")


def _validate_lineups(
    features: list[SnapshotFeature], home_team_id: TeamId, away_team_id: TeamId
) -> None:
    expected = {home_team_id.value, away_team_id.value}
    by_team: dict[str, set[str]] = {}
    for feature in features:
        value = feature.value
        try:
            players = (
                {PlayerId(item).value for item in value}
                if isinstance(value, (list, tuple))
                else set()
            )
        except (TypeError, ValueError):
            players = set()
        if feature.entity_id not in expected or feature.entity_id in by_team or len(players) != 11:
            raise ValueError("official lineup must contain 11 unique platform player IDs per team")
        by_team[feature.entity_id] = players
    missing = sorted(expected - set(by_team))
    if missing:
        raise ValueError(
            "lineups-confirmed snapshot requires official lineups for both teams: "
            + ", ".join(missing)
        )
    if by_team[home_team_id.value] & by_team[away_team_id.value]:
        raise ValueError("official home and away lineups must not overlap")


def _lineup_delta_missing(value: Any, home: TeamId, away: TeamId) -> set[str]:
    if not isinstance(value, dict) or set(value) - {home.value, away.value}:
        raise ValueError("lineup_delta must be keyed by the match teams")
    missing: set[str] = set()
    for team_id in (home.value, away.value):
        item = value.get(team_id)
        if item is None:
            missing.add(f"lineup_delta:{team_id}")
            continue
        fields = item.get("missing_fields") if isinstance(item, dict) else None
        dimensions = item.get("dimension_deltas") if isinstance(item, dict) else None
        status = item.get("quality_status") if isinstance(item, dict) else None
        if (
            status not in {"ready", "preview"}
            or not isinstance(fields, list)
            or len(fields) != len(set(fields))
            or not isinstance(dimensions, dict)
            or (status == "ready" and fields)
            or (status == "preview" and not fields)
        ):
            raise ValueError(f"lineup_delta for {team_id} is invalid")
        missing.update(f"{team_id}:{name}" for name in fields)
    return missing


def _snapshot_identity_from(snapshot: PreMatchSnapshot) -> dict[str, Any]:
    return _snapshot_identity(
        **{
            name: getattr(snapshot, name)
            for name in (
                "match_id",
                "match_version",
                "home_team_id",
                "away_team_id",
                "snapshot_type",
                "capture_mode",
                "as_of",
                "observed_at",
                "scheduled_kickoff_used",
                "feature_spec_version",
                "features",
                "input_refs",
                "quality_status",
                "missing_fields",
            )
        }
    )


def _snapshot_identity(**fields: Any) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "match_id": fields["match_id"].value,
        "match_version": fields["match_version"],
        "home_team_id": fields["home_team_id"].value,
        "away_team_id": fields["away_team_id"].value,
        "snapshot_type": fields["snapshot_type"].value,
        "capture_mode": fields["capture_mode"].value,
        "as_of": _timestamp(fields["as_of"]),
        "observed_at": _timestamp(fields["observed_at"]),
        "scheduled_kickoff_used": _timestamp(fields["scheduled_kickoff_used"]),
        "feature_spec_version": fields["feature_spec_version"],
        "features": [_feature_payload(item) for item in fields["features"]],
        "input_refs": list(fields["input_refs"]),
        "quality_status": fields["quality_status"],
        "missing_fields": list(fields["missing_fields"]),
    }


def _feature_payload(feature: SnapshotFeature) -> dict[str, Any]:
    return {
        "name": feature.name,
        "value": feature.value,
        "known_at": _timestamp(feature.known_at),
        "source_ref": feature.source_ref,
        "contribution_key": feature.contribution_key,
        "entity_id": feature.entity_id,
    }


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
