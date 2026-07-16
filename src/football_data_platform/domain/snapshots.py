"""Time-honest, immutable pre-match snapshot contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from football_data_platform.domain.ids import MatchId, SnapshotId, TeamId
from football_data_platform.domain.models import require_utc


class SnapshotType(StrEnum):
    T24H = "t24h"
    LINEUPS_CONFIRMED = "lineups-confirmed"


class CaptureMode(StrEnum):
    CAPTURED = "captured"
    RECONSTRUCTED = "reconstructed"


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
        _require_text(self.contribution_key, "contribution_key")
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


def build_snapshot(
    *,
    match_id: MatchId,
    match_version: int,
    snapshot_type: SnapshotType,
    capture_mode: CaptureMode,
    as_of: datetime,
    observed_at: datetime,
    scheduled_kickoff_used: datetime,
    feature_spec_version: str,
    features: tuple[SnapshotFeature, ...],
    home_team_id: TeamId,
    away_team_id: TeamId,
    missing_fields: tuple[str, ...] = (),
) -> PreMatchSnapshot:
    """Validate and build one content-identified pre-match snapshot."""

    for name, value in (
        ("as_of", as_of),
        ("observed_at", observed_at),
        ("scheduled_kickoff_used", scheduled_kickoff_used),
    ):
        require_utc(value, name)
    if match_version < 1:
        raise ValueError("match_version must be positive")
    _require_text(feature_spec_version, "feature_spec_version")
    if as_of >= scheduled_kickoff_used:
        raise ValueError("snapshot as_of must precede scheduled kickoff")
    if capture_mode is CaptureMode.CAPTURED and observed_at != as_of:
        raise ValueError("captured snapshots require observed_at == as_of")
    if capture_mode is CaptureMode.RECONSTRUCTED and observed_at < as_of:
        raise ValueError("reconstructed snapshots cannot be observed before as_of")
    if snapshot_type is SnapshotType.T24H:
        expected_as_of = scheduled_kickoff_used - timedelta(hours=24)
        if as_of != expected_as_of:
            raise ValueError("t24h as_of must equal scheduled kickoff minus 24 hours")

    contribution_keys: set[str] = set()
    for feature in features:
        if feature.known_at > as_of:
            raise ValueError(f"feature {feature.name!r} was known after snapshot as_of")
        if feature.contribution_key in contribution_keys:
            raise ValueError(f"duplicate feature contribution {feature.contribution_key!r}")
        contribution_keys.add(feature.contribution_key)
    features = tuple(sorted(features, key=lambda item: item.contribution_key))

    if snapshot_type is SnapshotType.LINEUPS_CONFIRMED:
        official_lineup_teams = {
            feature.entity_id for feature in features if feature.name == "official_lineup_confirmed"
        }
        required = {home_team_id.value, away_team_id.value}
        missing = sorted(required - official_lineup_teams)
        if missing:
            raise ValueError(
                "lineups-confirmed snapshot requires official lineups for both teams: "
                + ", ".join(missing)
            )

    input_refs = tuple(sorted({feature.source_ref for feature in features}))
    identity = {
        "schema_version": 1,
        "match_id": match_id.value,
        "match_version": match_version,
        "snapshot_type": snapshot_type.value,
        "capture_mode": capture_mode.value,
        "as_of": _timestamp(as_of),
        "observed_at": _timestamp(observed_at),
        "scheduled_kickoff_used": _timestamp(scheduled_kickoff_used),
        "feature_spec_version": feature_spec_version,
        "features": [_feature_payload(feature) for feature in features],
        "input_refs": input_refs,
        "quality_status": "ready" if not missing_fields else "preview",
        "missing_fields": missing_fields,
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return PreMatchSnapshot(
        id=SnapshotId(f"snapshot:{digest}"),
        schema_version=1,
        match_id=match_id,
        match_version=match_version,
        snapshot_type=snapshot_type,
        capture_mode=capture_mode,
        as_of=as_of,
        observed_at=observed_at,
        scheduled_kickoff_used=scheduled_kickoff_used,
        feature_spec_version=feature_spec_version,
        features=features,
        input_refs=input_refs,
        quality_status="ready" if not missing_fields else "preview",
        missing_fields=missing_fields,
    )


def snapshot_payload(snapshot: PreMatchSnapshot) -> dict[str, Any]:
    """Return the stable JSON representation written to the derived layer."""

    return {
        "id": snapshot.id.value,
        "schema_version": snapshot.schema_version,
        "match_id": snapshot.match_id.value,
        "match_version": snapshot.match_version,
        "snapshot_type": snapshot.snapshot_type.value,
        "capture_mode": snapshot.capture_mode.value,
        "as_of": _timestamp(snapshot.as_of),
        "observed_at": _timestamp(snapshot.observed_at),
        "scheduled_kickoff_used": _timestamp(snapshot.scheduled_kickoff_used),
        "feature_spec_version": snapshot.feature_spec_version,
        "features": [_feature_payload(feature) for feature in snapshot.features],
        "input_refs": list(snapshot.input_refs),
        "quality_status": snapshot.quality_status,
        "missing_fields": list(snapshot.missing_fields),
    }


def _feature_payload(feature: SnapshotFeature) -> dict[str, Any]:
    payload = asdict(feature)
    payload["known_at"] = _timestamp(feature.known_at)
    return payload


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


def _require_text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty text without surrounding whitespace")
