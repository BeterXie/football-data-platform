"""Immutable storage for rebuildable derived JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import RawAssetId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    ModelRunValidator,
    ScorePrediction,
    prediction_payload,
    verify_score_prediction,
)
from football_data_platform.domain.snapshots import (
    PreMatchSnapshot,
    SnapshotSourceValidation,
    snapshot_payload,
    verify_snapshot,
)
from football_data_platform.features.team_baseline import (
    TeamBaselineArtifact,
    parse_team_baseline_payload,
    team_baseline_payload,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

DERIVED_MANIFEST_VERSION = 1
DERIVED_CODE_VERSION = "football-data-platform/0.1.0"
_SUCCESS_STATUSES = frozenset({"succeeded", "partial"})
_KNOWN_STATUSES = frozenset({"running", "succeeded", "partial", "failed"})


@dataclass(frozen=True, slots=True)
class DerivedArtifactManifest:
    """Immutable metadata and payload envelope for one derived artifact.

    The manifest ID is derived from every field except ``artifact_id``.  The
    payload is intentionally JSON data, so a report can be rebuilt without
    parsing a Markdown or HTML rendering.
    """

    artifact_id: str
    manifest_version: int
    schema_version: int
    artifact_type: str
    generated_at: datetime
    started_at: datetime
    ended_at: datetime
    transform_version: str
    code_version: str
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    status: str
    error: str | None
    quality: str
    payload: Any

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        payload: Any,
        generated_at: datetime,
        transform_version: str,
        code_version: str,
        input_refs: Sequence[str],
        output_refs: Sequence[str],
        quality: str,
        schema_version: int = 1,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        status: str = "succeeded",
        error: str | None = None,
    ) -> DerivedArtifactManifest:
        started_at = generated_at if started_at is None else started_at
        ended_at = generated_at if ended_at is None else ended_at
        normalized = _artifact_fields(
            schema_version=schema_version,
            artifact_type=artifact_type,
            generated_at=generated_at,
            started_at=started_at,
            ended_at=ended_at,
            transform_version=transform_version,
            code_version=code_version,
            input_refs=input_refs,
            output_refs=output_refs,
            status=status,
            error=error,
            quality=quality,
            payload=payload,
        )
        digest = hashlib.sha256(
            _canonical_json({"manifest_version": DERIVED_MANIFEST_VERSION, **normalized})
        ).hexdigest()
        return cls(
            artifact_id=f"derived-artifact:{digest}",
            manifest_version=DERIVED_MANIFEST_VERSION,
            **normalized,
        )

    @property
    def id(self) -> str:
        return self.artifact_id

    @property
    def manifest_id(self) -> str:
        return self.artifact_id

    @property
    def type(self) -> str:
        return self.artifact_type

    @property
    def quality_status(self) -> str:
        return self.quality

    def identity_payload(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            **_serialize_fields(
                _artifact_fields(
                    schema_version=self.schema_version,
                    artifact_type=self.artifact_type,
                    generated_at=self.generated_at,
                    started_at=self.started_at,
                    ended_at=self.ended_at,
                    transform_version=self.transform_version,
                    code_version=self.code_version,
                    input_refs=self.input_refs,
                    output_refs=self.output_refs,
                    status=self.status,
                    error=self.error,
                    quality=self.quality,
                    payload=self.payload,
                )
            ),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.artifact_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable execution record, including failed and partial runs."""

    run_id: str
    manifest_version: int
    schema_version: int
    run_type: str
    generated_at: datetime
    started_at: datetime
    ended_at: datetime
    transform_version: str
    code_version: str
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    status: str
    error: str | None
    quality: str
    parameters: Any
    checkpoint: str | None
    payload: Any

    @classmethod
    def create(
        cls,
        *,
        run_type: str,
        started_at: datetime,
        ended_at: datetime | None,
        transform_version: str,
        code_version: str,
        input_refs: Sequence[str],
        output_refs: Sequence[str],
        status: str,
        error: str | None,
        quality: str,
        parameters: Any = None,
        checkpoint: str | None = None,
        payload: Any = None,
        generated_at: datetime | None = None,
        schema_version: int = 1,
    ) -> RunManifest:
        ended_at = started_at if ended_at is None else ended_at
        generated_at = ended_at if generated_at is None else generated_at
        normalized = _run_fields(
            schema_version=schema_version,
            run_type=run_type,
            generated_at=generated_at,
            started_at=started_at,
            ended_at=ended_at,
            transform_version=transform_version,
            code_version=code_version,
            input_refs=input_refs,
            output_refs=output_refs,
            status=status,
            error=error,
            quality=quality,
            parameters=parameters,
            checkpoint=checkpoint,
            payload=payload,
        )
        digest = hashlib.sha256(
            _canonical_json({"manifest_version": DERIVED_MANIFEST_VERSION, **normalized})
        ).hexdigest()
        return cls(
            run_id=f"run:{digest}",
            manifest_version=DERIVED_MANIFEST_VERSION,
            **normalized,
        )

    @property
    def id(self) -> str:
        return self.run_id

    @property
    def manifest_id(self) -> str:
        return self.run_id

    @property
    def quality_status(self) -> str:
        return self.quality

    def identity_payload(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            **_serialize_fields(
                _run_fields(
                    schema_version=self.schema_version,
                    run_type=self.run_type,
                    generated_at=self.generated_at,
                    started_at=self.started_at,
                    ended_at=self.ended_at,
                    transform_version=self.transform_version,
                    code_version=self.code_version,
                    input_refs=self.input_refs,
                    output_refs=self.output_refs,
                    status=self.status,
                    error=self.error,
                    quality=self.quality,
                    parameters=self.parameters,
                    checkpoint=self.checkpoint,
                    payload=self.payload,
                )
            ),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.run_id, **self.identity_payload()}


# Compatibility aliases make the contract discoverable without introducing a
# second representation for the same immutable record.
DerivedManifest = DerivedArtifactManifest
RunArtifactManifest = RunManifest


def _artifact_fields(
    *,
    schema_version: int,
    artifact_type: str,
    generated_at: datetime,
    started_at: datetime,
    ended_at: datetime,
    transform_version: str,
    code_version: str,
    input_refs: Sequence[str],
    output_refs: Sequence[str],
    status: str,
    error: str | None,
    quality: str,
    payload: Any,
) -> dict[str, Any]:
    _validate_common_fields(
        schema_version=schema_version,
        generated_at=generated_at,
        started_at=started_at,
        ended_at=ended_at,
        transform_version=transform_version,
        code_version=code_version,
        input_refs=input_refs,
        output_refs=output_refs,
        status=status,
        error=error,
        quality=quality,
    )
    normalized_input_refs = _normalize_refs(input_refs, "input_refs")
    normalized_output_refs = _normalize_refs(output_refs, "output_refs")
    if status in _SUCCESS_STATUSES and not normalized_input_refs:
        raise ValueError("successful derived artifacts require input_refs lineage")
    if status in _SUCCESS_STATUSES and not normalized_output_refs:
        raise ValueError("successful derived artifacts require output_refs lineage")
    normalized_payload = _json_safe(payload)
    if status in _SUCCESS_STATUSES and normalized_payload is None:
        raise ValueError("successful derived artifacts require payload")
    return {
        "schema_version": int(schema_version),
        "artifact_type": artifact_type,
        "generated_at": generated_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "transform_version": transform_version,
        "code_version": code_version,
        "input_refs": normalized_input_refs,
        "output_refs": normalized_output_refs,
        "status": status,
        "error": error,
        "quality": quality,
        "payload": normalized_payload,
    }


def _run_fields(
    *,
    schema_version: int,
    run_type: str,
    generated_at: datetime,
    started_at: datetime,
    ended_at: datetime,
    transform_version: str,
    code_version: str,
    input_refs: Sequence[str],
    output_refs: Sequence[str],
    status: str,
    error: str | None,
    quality: str,
    parameters: Any,
    checkpoint: str | None,
    payload: Any,
) -> dict[str, Any]:
    _validate_common_fields(
        schema_version=schema_version,
        generated_at=generated_at,
        started_at=started_at,
        ended_at=ended_at,
        transform_version=transform_version,
        code_version=code_version,
        input_refs=input_refs,
        output_refs=output_refs,
        status=status,
        error=error,
        quality=quality,
    )
    if not isinstance(run_type, str) or not run_type or run_type.strip() != run_type:
        raise ValueError("run_type must be non-empty text")
    return {
        "schema_version": int(schema_version),
        "run_type": run_type,
        "generated_at": generated_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "transform_version": transform_version,
        "code_version": code_version,
        "input_refs": _normalize_refs(input_refs, "input_refs"),
        "output_refs": _normalize_refs(output_refs, "output_refs"),
        "status": status,
        "error": error,
        "quality": quality,
        "parameters": _json_safe(parameters),
        "checkpoint": checkpoint,
        "payload": _json_safe(payload),
    }


def _serialize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Convert typed manifest fields into their canonical JSON representation."""

    return {key: _json_safe(value) for key, value in fields.items()}


def _validate_common_fields(
    *,
    schema_version: int,
    generated_at: datetime,
    started_at: datetime,
    ended_at: datetime,
    transform_version: str,
    code_version: str,
    input_refs: Sequence[str],
    output_refs: Sequence[str],
    status: str,
    error: str | None,
    quality: str,
) -> None:
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        raise ValueError("schema_version must be a positive integer")
    for name, value in (
        ("generated_at", generated_at),
        ("started_at", started_at),
        ("ended_at", ended_at),
    ):
        if not isinstance(value, datetime):
            raise ValueError(f"{name} must be a datetime")
        require_utc(value, name)
    if ended_at < started_at:
        raise ValueError("ended_at cannot precede started_at")
    if generated_at < started_at or generated_at > ended_at:
        raise ValueError("generated_at must fall between started_at and ended_at")
    for name, value in (("transform_version", transform_version), ("code_version", code_version)):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"{name} must be non-empty text")
    if status not in _KNOWN_STATUSES:
        raise ValueError(f"unsupported manifest status {status!r}")
    if not isinstance(quality, str) or not quality or quality.strip() != quality:
        raise ValueError("quality must be non-empty text")
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ValueError("error must be non-empty text when present")
    if status == "failed" and error is None:
        raise ValueError("failed manifests require an error")
    if status == "succeeded" and error is not None:
        raise ValueError("succeeded manifests cannot contain an error")
    _normalize_refs(input_refs, "input_refs")
    _normalize_refs(output_refs, "output_refs")


def _normalize_refs(refs: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(refs, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of references")
    try:
        values = tuple(refs)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of references") from error
    if any(not isinstance(item, str) or not item or item.strip() != item for item in values):
        raise ValueError(f"{field_name} must contain non-empty references")
    return tuple(sorted(set(values)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        require_utc(value, "manifest payload datetime")
        return _timestamp(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest payload contains a non-finite number")
    if isinstance(value, Path):
        return str(value)
    return value


class DerivedArchive:
    def __init__(
        self,
        layout: DataLayout,
        *,
        model_run_validator: ModelRunValidator | None = None,
    ) -> None:
        self.layout = layout.ensure()
        self.model_run_validator = model_run_validator

    def write_artifact_manifest(self, manifest: DerivedArtifactManifest) -> Path:
        """Write one immutable derived artifact manifest and recheck its identity."""

        _verify_artifact_manifest(manifest)
        return self._write_json(
            self.artifact_manifest_path(manifest.artifact_id), manifest.to_payload()
        )

    def load_artifact_manifest(self, artifact_id: str) -> DerivedArtifactManifest:
        path = self.artifact_manifest_path(artifact_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArchiveConflictError(f"cannot load derived artifact manifest: {path}") from error
        manifest = _parse_artifact_manifest(payload)
        if manifest.artifact_id != artifact_id:
            raise ArchiveConflictError("derived artifact manifest ID does not match path")
        _verify_artifact_manifest(manifest)
        return manifest

    def write_derived_artifact(self, **kwargs: Any) -> DerivedArtifactManifest:
        """Create and persist a generic derived artifact manifest."""

        manifest = DerivedArtifactManifest.create(**kwargs)
        self.write_artifact_manifest(manifest)
        return manifest

    # Short aliases keep callers independent from the storage filename choice.
    write_artifact = write_derived_artifact
    load_artifact = load_artifact_manifest

    def write_run_manifest(self, manifest: RunManifest) -> Path:
        """Persist a successful, partial, running, or failed run immutably."""

        _verify_run_manifest(manifest)
        return self._write_json(self.run_manifest_path(manifest.run_id), manifest.to_payload())

    def load_run_manifest(self, run_id: str) -> RunManifest:
        path = self.run_manifest_path(run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArchiveConflictError(f"cannot load run manifest: {path}") from error
        manifest = _parse_run_manifest(payload)
        if manifest.run_id != run_id:
            raise ArchiveConflictError("run manifest ID does not match path")
        _verify_run_manifest(manifest)
        return manifest

    def write_run(self, **kwargs: Any) -> RunManifest:
        manifest = RunManifest.create(**kwargs)
        self.write_run_manifest(manifest)
        return manifest

    load_run = load_run_manifest

    def write_snapshot(self, snapshot: PreMatchSnapshot) -> Path:
        verify_snapshot(snapshot, source_validator=self)
        path = self.snapshot_path(snapshot)
        payload = (
            json.dumps(
                snapshot_payload(snapshot, source_validator=self),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ArchiveConflictError(f"derived snapshot conflicts at {path}")
        else:
            try:
                with path.open("xb") as destination:
                    destination.write(payload)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ArchiveConflictError(f"derived snapshot conflicts at {path}") from None
        self.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="prematch-snapshot",
                schema_version=snapshot.schema_version,
                payload=snapshot_payload(snapshot, source_validator=self),
                generated_at=snapshot.observed_at,
                started_at=snapshot.observed_at,
                ended_at=snapshot.observed_at,
                transform_version=snapshot.feature_spec_version,
                code_version=DERIVED_CODE_VERSION,
                input_refs=snapshot.input_refs,
                output_refs=(snapshot.id.value,),
                status="succeeded",
                quality=snapshot.quality_status,
            )
        )
        return path

    def load_snapshot_payload(self, snapshot: PreMatchSnapshot) -> dict[str, Any]:
        verify_snapshot(snapshot, source_validator=self)
        return json.loads(self.snapshot_path(snapshot).read_text(encoding="utf-8"))

    def write_team_baseline(
        self,
        artifact: TeamBaselineArtifact,
        *,
        generated_at: datetime | None = None,
        code_version: str = DERIVED_CODE_VERSION,
    ) -> Path:
        """Persist a complete, content-addressed team baseline artifact."""

        payload = team_baseline_payload(artifact)
        path = self._write_json(self.team_baseline_path(artifact.artifact_id), payload)
        status = "succeeded" if artifact.quality_status == "ready" else "partial"
        self.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="team-baseline",
                schema_version=artifact.schema_version,
                payload=payload,
                generated_at=generated_at or artifact.as_of,
                transform_version=artifact.transform_version,
                code_version=code_version,
                input_refs=artifact.input_refs,
                output_refs=(artifact.artifact_id,),
                status=status,
                quality=artifact.quality_status,
            )
        )
        return path

    def load_team_baseline(self, artifact_id: str) -> TeamBaselineArtifact:
        """Load and re-verify a persisted team baseline by its stable ID."""

        payload = json.loads(self.team_baseline_path(artifact_id).read_text(encoding="utf-8"))
        artifact = parse_team_baseline_payload(payload)
        if artifact.artifact_id != artifact_id:
            raise ArchiveConflictError("team baseline path and artifact ID disagree")
        return artifact

    def write_snapshot_source(
        self,
        *,
        value: Any,
        input_refs: tuple[RawAssetId, ...],
        transform_version: str,
        generated_at: datetime,
    ) -> str:
        """Archive one content-addressed feature value and its verified raw lineage."""

        require_utc(generated_at, "generated_at")
        if not transform_version or transform_version.strip() != transform_version:
            raise ValueError("transform_version must be non-empty text")
        if not input_refs:
            raise ValueError("derived snapshot sources require raw input_refs")
        raw = RawArchive(self.layout)
        input_observed_at = []
        for input_ref in input_refs:
            raw.verify(input_ref)
            input_observed_at.append(raw.load(input_ref).observed_at)
        if generated_at < max(input_observed_at):
            raise ValueError("generated_at cannot precede a raw input observation")
        identity = {
            "schema_version": 1,
            "value": value,
            "input_refs": sorted({item.value for item in input_refs}),
            "transform_version": transform_version,
            "generated_at": _timestamp(generated_at),
        }
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        source_ref = f"derived-source:{digest}"
        self._write_json(self._snapshot_source_path(source_ref), {"id": source_ref, **identity})
        return source_ref

    def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation:
        """Verify a raw or derived snapshot source against immutable storage."""

        raw = RawArchive(self.layout)
        if not source_ref.startswith("derived-source:"):
            raise ValueError(f"snapshot source is not available in derived storage: {source_ref}")
        path = self._snapshot_source_path(source_ref)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_id = payload.pop("id", None)
        calculated = f"derived-source:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        if stored_id != source_ref or calculated != source_ref:
            raise ArchiveConflictError(f"derived snapshot source identity failed at {path}")
        if payload.get("schema_version") != 1:
            raise ArchiveConflictError("unsupported derived snapshot source schema")
        input_refs = tuple(str(item) for item in payload.get("input_refs", ()))
        input_observed_at = []
        for input_ref in input_refs:
            asset_id = RawAssetId(input_ref)
            raw.verify(asset_id)
            input_observed_at.append(raw.load(asset_id).observed_at)
        generated_at = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
        require_utc(generated_at, "generated_at")
        if not input_observed_at or generated_at < max(input_observed_at):
            raise ArchiveConflictError("derived snapshot source has invalid observation lineage")
        transform_version = str(payload["transform_version"])
        value = payload["value"]
        if transform_version in {"team-baseline-input/1", "team-baseline-input/2"}:
            self._validate_team_baseline_source(value, input_refs)
        return SnapshotSourceValidation(
            source_ref,
            "derived",
            transform_version,
            generated_at,
            value,
            input_refs,
        )

    def write_prediction(self, prediction: ScorePrediction) -> Path:
        verify_score_prediction(
            prediction,
            model_run_validator=self.model_run_validator,
        )
        digest = prediction.id.value.removeprefix("prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        payload = prediction_payload(prediction)
        result = self._write_json(path, payload)
        self.write_artifact_manifest(
            DerivedArtifactManifest.create(
                artifact_type="prediction",
                schema_version=prediction.schema_version,
                payload=payload,
                generated_at=prediction.generated_at,
                transform_version=prediction.model_version,
                code_version=DERIVED_CODE_VERSION,
                input_refs=prediction.input_refs,
                output_refs=(prediction.id.value,),
                status="succeeded",
                quality=prediction.snapshot_quality_status,
            )
        )
        return result

    def load_prediction_payload(self, prediction: ScorePrediction) -> dict[str, Any]:
        verify_score_prediction(
            prediction,
            model_run_validator=self.model_run_validator,
        )
        digest = prediction.id.value.removeprefix("prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload != prediction_payload(prediction):
            raise ArchiveConflictError(f"derived prediction identity failed at {path}")
        return payload

    def write_player_profiles(
        self,
        profiles: Any,
        *,
        generated_at: datetime | None = None,
        code_version: str = DERIVED_CODE_VERSION,
    ) -> Path:
        """Archive a profile build result as one lineage-addressed artifact."""

        profile_items = tuple(getattr(profiles, "profiles", profiles))
        profile_payload = {
            "profiles": [_json_safe(item) for item in profile_items],
            "excluded_input_refs": list(getattr(profiles, "excluded_input_refs", ())),
            "partition_audits": [
                _json_safe(item) for item in getattr(profiles, "partition_audits", ())
            ],
        }
        input_refs = tuple(
            sorted(
                {ref for item in profile_items for ref in getattr(item, "input_refs", ())}
                | set(profile_payload["excluded_input_refs"])
            )
        )
        output_refs = tuple(
            sorted(
                str(item.artifact_id)
                for item in profile_items
                if getattr(item, "artifact_id", None)
            )
        )
        profile_times = [getattr(item, "as_of", None) for item in profile_items]
        profile_times = [item for item in profile_times if isinstance(item, datetime)]
        generated = generated_at or (max(profile_times) if profile_times else None)
        if generated is None:
            raise ValueError("player profiles require generated_at when no profiles are present")
        ready = bool(profile_items) and all(
            getattr(item, "quality_status", None) == "ready" for item in profile_items
        )
        status = "succeeded" if ready else ("partial" if profile_items else "failed")
        error = None if profile_items else "no_player_profiles"
        quality = "ready" if ready else ("preview" if profile_items else "failed")
        transform_versions = {
            str(item.transform_version)
            for item in profile_items
            if getattr(item, "transform_version", None)
        }
        transform_version = (
            sorted(transform_versions)[0] if transform_versions else "player-profile/unknown"
        )
        manifest = DerivedArtifactManifest.create(
            artifact_type="player-profiles",
            payload=profile_payload,
            generated_at=generated,
            transform_version=transform_version,
            code_version=code_version,
            input_refs=input_refs,
            output_refs=output_refs,
            status=status,
            error=error,
            quality=quality,
        )
        return self.write_artifact_manifest(manifest)

    def write_evaluation(
        self,
        evaluation: Any,
        *,
        generated_at: datetime | None = None,
        code_version: str = DERIVED_CODE_VERSION,
    ) -> Path:
        """Archive one evaluation record without making market data implicit."""

        payload = _json_safe(evaluation)
        if not isinstance(payload, dict):
            raise ValueError("evaluation must serialize to an object")
        input_refs = tuple(str(item) for item in getattr(evaluation, "input_refs", ()))
        evaluated_at = generated_at or getattr(evaluation, "evaluated_at", None)
        if not isinstance(evaluated_at, datetime):
            raise ValueError("evaluation requires evaluated_at or generated_at")
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        manifest = DerivedArtifactManifest.create(
            artifact_type="evaluation",
            schema_version=int(payload.get("schema_version", 1)),
            payload=payload,
            generated_at=evaluated_at,
            transform_version="evaluation/1",
            code_version=code_version,
            input_refs=input_refs,
            output_refs=(f"evaluation:{digest}",),
            status="succeeded",
            quality="ready",
        )
        return self.write_artifact_manifest(manifest)

    def artifact_manifest_path(self, artifact_id: str) -> Path:
        digest = _manifest_digest(artifact_id, "derived-artifact")
        return self.layout.derived / "manifests" / "artifacts" / digest[:2] / f"{digest}.json"

    def run_manifest_path(self, run_id: str) -> Path:
        digest = _manifest_digest(run_id, "run")
        return self.layout.derived / "manifests" / "runs" / digest[:2] / f"{digest}.json"

    def snapshot_path(self, snapshot: PreMatchSnapshot) -> Path:
        digest = snapshot.id.value.removeprefix("snapshot:")
        return self.layout.derived / "snapshots" / digest[:2] / f"{digest}.json"

    def team_baseline_path(self, artifact_id: str) -> Path:
        if not artifact_id.startswith("team-baseline:"):
            raise ValueError("invalid team baseline artifact ID")
        digest = artifact_id.removeprefix("team-baseline:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid team baseline artifact ID")
        return self.layout.derived / "team-baselines" / digest[:2] / f"{digest}.json"

    def _snapshot_source_path(self, source_ref: str) -> Path:
        digest = source_ref.removeprefix("derived-source:")
        return self.layout.derived / "snapshot-sources" / digest[:2] / f"{digest}.json"

    def _validate_team_baseline_source(
        self, value: Any, input_refs: tuple[str, ...]
    ) -> TeamBaselineArtifact:
        if not isinstance(value, dict) or not isinstance(value.get("artifact"), dict):
            raise ArchiveConflictError(
                "team baseline source must contain a complete artifact payload"
            )
        artifact = self.load_team_baseline(str(value.get("artifact_id", "")))
        nested = parse_team_baseline_payload(value["artifact"])
        if nested != artifact or value.get("artifact_id") != artifact.artifact_id:
            raise ArchiveConflictError("team baseline source does not match persisted artifact")
        if set(artifact.input_refs) != set(input_refs):
            raise ArchiveConflictError("team baseline source input refs do not match artifact")
        for name in ("lambda_home", "lambda_away"):
            number = value.get(name)
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or number <= 0
            ):
                raise ArchiveConflictError(f"team baseline source has invalid {name}")
        return artifact

    def _write_json(self, path: Path, value: dict[str, Any]) -> Path:
        payload = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ArchiveConflictError(f"derived artifact conflicts at {path}")
            return path
        try:
            with path.open("xb") as destination:
                destination.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ArchiveConflictError(f"derived artifact conflicts at {path}") from None
        return path


def _verify_artifact_manifest(manifest: DerivedArtifactManifest) -> None:
    if not isinstance(manifest, DerivedArtifactManifest):
        raise TypeError("manifest must be a DerivedArtifactManifest")
    if manifest.manifest_version != DERIVED_MANIFEST_VERSION:
        raise ValueError("unsupported derived manifest version")
    fields = _artifact_fields(
        schema_version=manifest.schema_version,
        artifact_type=manifest.artifact_type,
        generated_at=manifest.generated_at,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        transform_version=manifest.transform_version,
        code_version=manifest.code_version,
        input_refs=manifest.input_refs,
        output_refs=manifest.output_refs,
        status=manifest.status,
        error=manifest.error,
        quality=manifest.quality,
        payload=manifest.payload,
    )
    if fields != {
        "schema_version": manifest.schema_version,
        "artifact_type": manifest.artifact_type,
        "generated_at": manifest.generated_at,
        "started_at": manifest.started_at,
        "ended_at": manifest.ended_at,
        "transform_version": manifest.transform_version,
        "code_version": manifest.code_version,
        "input_refs": manifest.input_refs,
        "output_refs": manifest.output_refs,
        "status": manifest.status,
        "error": manifest.error,
        "quality": manifest.quality,
        "payload": manifest.payload,
    }:
        raise ArchiveConflictError("derived artifact manifest content is not canonical")
    expected = (
        "derived-artifact:"
        + hashlib.sha256(
            _canonical_json({"manifest_version": manifest.manifest_version, **fields})
        ).hexdigest()
    )
    if manifest.artifact_id != expected:
        raise ArchiveConflictError("derived artifact manifest identity failed")


def _verify_run_manifest(manifest: RunManifest) -> None:
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    if manifest.manifest_version != DERIVED_MANIFEST_VERSION:
        raise ValueError("unsupported run manifest version")
    fields = _run_fields(
        schema_version=manifest.schema_version,
        run_type=manifest.run_type,
        generated_at=manifest.generated_at,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        transform_version=manifest.transform_version,
        code_version=manifest.code_version,
        input_refs=manifest.input_refs,
        output_refs=manifest.output_refs,
        status=manifest.status,
        error=manifest.error,
        quality=manifest.quality,
        parameters=manifest.parameters,
        checkpoint=manifest.checkpoint,
        payload=manifest.payload,
    )
    expected_fields = {
        "schema_version": manifest.schema_version,
        "run_type": manifest.run_type,
        "generated_at": manifest.generated_at,
        "started_at": manifest.started_at,
        "ended_at": manifest.ended_at,
        "transform_version": manifest.transform_version,
        "code_version": manifest.code_version,
        "input_refs": manifest.input_refs,
        "output_refs": manifest.output_refs,
        "status": manifest.status,
        "error": manifest.error,
        "quality": manifest.quality,
        "parameters": manifest.parameters,
        "checkpoint": manifest.checkpoint,
        "payload": manifest.payload,
    }
    if fields != expected_fields:
        raise ArchiveConflictError("run manifest content is not canonical")
    expected = (
        "run:"
        + hashlib.sha256(
            _canonical_json({"manifest_version": manifest.manifest_version, **fields})
        ).hexdigest()
    )
    if manifest.run_id != expected:
        raise ArchiveConflictError("run manifest identity failed")


def _parse_artifact_manifest(payload: Any) -> DerivedArtifactManifest:
    if not isinstance(payload, dict):
        raise ArchiveConflictError("derived artifact manifest must be an object")
    try:
        artifact_id = str(payload.get("id", payload.get("artifact_id", "")))
        manifest = DerivedArtifactManifest(
            artifact_id=artifact_id,
            manifest_version=int(payload["manifest_version"]),
            schema_version=int(payload["schema_version"]),
            artifact_type=str(payload["artifact_type"]),
            generated_at=_parse_manifest_datetime(payload["generated_at"], "generated_at"),
            started_at=_parse_manifest_datetime(payload["started_at"], "started_at"),
            ended_at=_parse_manifest_datetime(payload["ended_at"], "ended_at"),
            transform_version=str(payload["transform_version"]),
            code_version=str(payload["code_version"]),
            input_refs=tuple(str(item) for item in payload["input_refs"]),
            output_refs=tuple(str(item) for item in payload["output_refs"]),
            status=str(payload["status"]),
            error=None if payload.get("error") is None else str(payload["error"]),
            quality=str(payload["quality"]),
            payload=payload.get("payload"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArchiveConflictError(f"invalid derived artifact manifest: {error}") from error
    _verify_artifact_manifest(manifest)
    if payload != manifest.to_payload():
        raise ArchiveConflictError("derived artifact manifest is not canonical")
    return manifest


def _parse_run_manifest(payload: Any) -> RunManifest:
    if not isinstance(payload, dict):
        raise ArchiveConflictError("run manifest must be an object")
    try:
        run_id = str(payload.get("id", payload.get("run_id", "")))
        manifest = RunManifest(
            run_id=run_id,
            manifest_version=int(payload["manifest_version"]),
            schema_version=int(payload["schema_version"]),
            run_type=str(payload["run_type"]),
            generated_at=_parse_manifest_datetime(payload["generated_at"], "generated_at"),
            started_at=_parse_manifest_datetime(payload["started_at"], "started_at"),
            ended_at=_parse_manifest_datetime(payload["ended_at"], "ended_at"),
            transform_version=str(payload["transform_version"]),
            code_version=str(payload["code_version"]),
            input_refs=tuple(str(item) for item in payload["input_refs"]),
            output_refs=tuple(str(item) for item in payload["output_refs"]),
            status=str(payload["status"]),
            error=None if payload.get("error") is None else str(payload["error"]),
            quality=str(payload["quality"]),
            parameters=payload.get("parameters"),
            checkpoint=None if payload.get("checkpoint") is None else str(payload["checkpoint"]),
            payload=payload.get("payload"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArchiveConflictError(f"invalid run manifest: {error}") from error
    _verify_run_manifest(manifest)
    if payload != manifest.to_payload():
        raise ArchiveConflictError("run manifest is not canonical")
    return manifest


def _parse_manifest_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ArchiveConflictError(f"manifest {field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require_utc(parsed, field_name)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError) as error:
        raise ArchiveConflictError(f"invalid manifest {field_name}") from error


def _manifest_digest(value: str, prefix: str) -> str:
    marker = f"{prefix}:"
    if not isinstance(value, str) or not value.startswith(marker):
        raise ValueError(f"invalid {prefix} manifest ID")
    digest = value.removeprefix(marker)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid {prefix} manifest ID")
    return digest


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _timestamp(value: datetime) -> str:
    require_utc(value, "manifest timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
