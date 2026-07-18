"""Immutable storage for rebuildable derived JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from football_data_platform.domain.ids import MatchId, PlayerId, RawAssetId, TeamId
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.predictions import (
    SCORE_GRID_COMPOSITION_SCHEMA_VERSION,
    MarketSnapshotValidator,
    ModelRunValidator,
    PersistedPredictionContextValidator,
    ScorePrediction,
    prediction_payload,
    score_grid_composition_artifact_id,
    score_grid_composition_payload,
    verify_prediction_snapshot,
    verify_score_prediction,
)
from football_data_platform.domain.snapshots import (
    PreMatchSnapshot,
    SnapshotSourceValidation,
    snapshot_payload,
    verify_current_snapshot,
    verify_snapshot,
)
from football_data_platform.features.lineup import (
    LINEUP_DELTA_INPUT_TRANSFORM_V3,
    lineup_delta_input_payload,
    parse_lineup_delta_payload,
)
from football_data_platform.features.player_profiles import (
    PlayerProfile,
    parse_player_profile_payload,
    validate_player_profile,
)
from football_data_platform.features.team_baseline import (
    TEAM_BASELINE_INPUT_TRANSFORM_V3,
    TeamBaselineArtifact,
    expected_goals_from_baseline,
    parse_team_baseline_payload,
    team_baseline_payload,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.facts import (
    load_verified_match_result,
    load_verified_team_observation,
    verify_official_lineup_contract,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.match_context import (
    MATCH_CONTEXT_INPUT_TRANSFORM_V2,
    replay_match_context,
)
from football_data_platform.storage.match_report_contracts import verify_match_report_contract
from football_data_platform.storage.raw import ArchiveConflictError, RawArchive

if TYPE_CHECKING:
    from football_data_platform.storage.verification import VerificationSession

DERIVED_MANIFEST_VERSION = 1
DERIVED_CODE_VERSION = "football-data-platform/0.1.0"
_SUCCESS_STATUSES = frozenset({"succeeded", "partial"})
_KNOWN_STATUSES = frozenset({"running", "succeeded", "partial", "failed"})
_REFERENCE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REFERENCE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_DIGEST_REFERENCE_NAMESPACES = frozenset(
    {
        "challenger-evidence",
        "cohort",
        "derived-artifact",
        "derived-source",
        "evaluation",
        "file-sha256",
        "market-snapshot",
        "match-report-contract",
        "model-artifact",
        "model-output",
        "model-run",
        "official-lineup-contract",
        "paper-bet-entry",
        "paper-ledger-recompute",
        "paper-ledger-summary",
        "player-profile",
        "prediction",
        "promotion-decision",
        "promotion-policy",
        "raw-asset",
        "run",
        "score-grid-composition",
        "snapshot",
        "team-baseline",
        "training-dataset",
        "training-qualification",
        "vertical-slice-report",
        "vertical-slice-summary",
    }
)
_CANONICAL_RECORD_NAMESPACES = frozenset({"canonical", "event", "fact", "lineup"})
_ENTITY_REFERENCE_NAMESPACES = frozenset({"competition", "match", "player", "season", "team"})
_MANIFEST_OUTPUT_REFERENCE_NAMESPACES = frozenset(
    {
        "cohort",
        "evaluation",
        "paper-ledger-recompute",
        "paper-ledger-summary",
        "vertical-slice-report",
        "vertical-slice-summary",
    }
)
_TRAINING_REFERENCE_NAMESPACES = frozenset(
    {
        "derived-source",
        "model-artifact",
        "model-output",
        "model-run",
        "score-grid-composition",
        "snapshot",
        "team-baseline",
        "training-dataset",
        "training-qualification",
    }
)


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
        market_snapshot_validator: MarketSnapshotValidator | None = None,
        prediction_context_validator: PersistedPredictionContextValidator | None = None,
    ) -> None:
        self.layout = layout.ensure()
        self.model_run_validator = model_run_validator
        self.market_snapshot_validator = market_snapshot_validator
        self.prediction_context_validator = prediction_context_validator

    def write_artifact_manifest(self, manifest: DerivedArtifactManifest) -> Path:
        """Write one immutable derived artifact manifest and recheck its identity."""

        _verify_artifact_manifest(manifest)
        from football_data_platform.storage.verification import (
            VerificationSession,
            active_verification_session,
            verification_session_scope,
        )

        session = active_verification_session(self.layout)
        if session is not None:
            _ManifestReferenceResolver(self, session).verify(
                input_refs=manifest.input_refs,
                output_refs=manifest.output_refs,
                status=manifest.status,
            )
        else:
            with VerificationSession(self.layout) as owned_session:
                with verification_session_scope(owned_session):
                    _ManifestReferenceResolver(self, owned_session).verify(
                        input_refs=manifest.input_refs,
                        output_refs=manifest.output_refs,
                        status=manifest.status,
                    )
        return self._write_json(
            self.artifact_manifest_path(manifest.artifact_id), manifest.to_payload()
        )

    def load_artifact_manifest(
        self,
        artifact_id: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> DerivedArtifactManifest:
        from football_data_platform.storage.verification import (
            VerificationSession,
            active_verification_session,
            verification_session_scope,
        )

        if verification_session is None:
            verification_session = active_verification_session(self.layout)
        if verification_session is None:
            with VerificationSession(self.layout) as session:
                with verification_session_scope(session):
                    return self.load_artifact_manifest(
                        artifact_id,
                        verification_session=session,
                    )
        path = self.artifact_manifest_path(artifact_id)
        verification_session.file_proof(path)
        manifest = verification_session.cached_artifact_manifest(artifact_id)
        if manifest is None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ArchiveConflictError(
                    f"cannot load derived artifact manifest: {path}"
                ) from error
            manifest = _parse_artifact_manifest(payload)
            if manifest.artifact_id != artifact_id:
                raise ArchiveConflictError("derived artifact manifest ID does not match path")
            _verify_artifact_manifest(manifest)
            verification_session.remember_artifact_manifest(artifact_id, manifest)
        with verification_session.resolving_artifact_manifest(artifact_id):
            _ManifestReferenceResolver(self, verification_session).verify(
                input_refs=manifest.input_refs,
                output_refs=manifest.output_refs,
                status=manifest.status,
            )
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
        from football_data_platform.storage.verification import (
            VerificationSession,
            active_verification_session,
            verification_session_scope,
        )

        session = active_verification_session(self.layout)
        if session is not None:
            _ManifestReferenceResolver(self, session).verify(
                input_refs=manifest.input_refs,
                output_refs=manifest.output_refs,
                status=manifest.status,
            )
        else:
            with VerificationSession(self.layout) as owned_session:
                with verification_session_scope(owned_session):
                    _ManifestReferenceResolver(self, owned_session).verify(
                        input_refs=manifest.input_refs,
                        output_refs=manifest.output_refs,
                        status=manifest.status,
                    )
        return self._write_json(self.run_manifest_path(manifest.run_id), manifest.to_payload())

    def load_run_manifest(
        self,
        run_id: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> RunManifest:
        from football_data_platform.storage.verification import (
            VerificationSession,
            active_verification_session,
            verification_session_scope,
        )

        if verification_session is None:
            verification_session = active_verification_session(self.layout)
        if verification_session is None:
            with VerificationSession(self.layout) as session:
                with verification_session_scope(session):
                    return self.load_run_manifest(run_id, verification_session=session)
        path = self.run_manifest_path(run_id)
        verification_session.file_proof(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArchiveConflictError(f"cannot load run manifest: {path}") from error
        manifest = _parse_run_manifest(payload)
        if manifest.run_id != run_id:
            raise ArchiveConflictError("run manifest ID does not match path")
        _verify_run_manifest(manifest)
        _ManifestReferenceResolver(self, verification_session).verify(
            input_refs=manifest.input_refs,
            output_refs=manifest.output_refs,
            status=manifest.status,
        )
        return manifest

    def write_run(self, **kwargs: Any) -> RunManifest:
        manifest = RunManifest.create(**kwargs)
        self.write_run_manifest(manifest)
        return manifest

    load_run = load_run_manifest

    def write_snapshot(self, snapshot: PreMatchSnapshot) -> Path:
        verify_current_snapshot(snapshot, source_validator=self)
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
        manifest_generated_at = artifact.as_of if generated_at is None else generated_at
        require_utc(manifest_generated_at, "team baseline generated_at")
        if manifest_generated_at < artifact.as_of:
            raise ValueError("team baseline generated_at cannot precede as_of")
        status = "succeeded" if artifact.quality_status == "ready" else "partial"
        manifest = DerivedArtifactManifest.create(
            artifact_type="team-baseline",
            schema_version=artifact.schema_version,
            payload=payload,
            generated_at=manifest_generated_at,
            transform_version=artifact.transform_version,
            code_version=code_version,
            input_refs=artifact.input_refs,
            output_refs=(artifact.artifact_id,),
            status=status,
            quality=artifact.quality_status,
        )
        existing = self._load_artifact_manifests_for_output_ref(artifact.artifact_id)
        if existing and (len(existing) != 1 or existing[0].artifact_id != manifest.artifact_id):
            raise ArchiveConflictError("team baseline artifact already has another manifest")
        path = self._write_json(self.team_baseline_path(artifact.artifact_id), payload)
        self.write_artifact_manifest(manifest)
        return path

    def load_team_baseline(self, artifact_id: str) -> TeamBaselineArtifact:
        """Load and re-verify a persisted team baseline by its stable ID."""

        payload = json.loads(self.team_baseline_path(artifact_id).read_text(encoding="utf-8"))
        artifact = parse_team_baseline_payload(payload)
        if artifact.artifact_id != artifact_id:
            raise ArchiveConflictError("team baseline path and artifact ID disagree")
        return artifact

    def write_team_baseline_source(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        as_of: datetime,
        baseline_artifact_id: str,
    ) -> str:
        """Persist a canonical team-baseline contribution for one fixture.

        The caller supplies only the fixture identity, cutoff, and the ID of an
        already persisted baseline.  Fixture direction, kickoff, baseline
        lambdas, lineage, and source timestamps are all reconstructed here.
        This keeps a caller from reusing a valid baseline with an arbitrary
        match or lambda pair.
        """

        if not isinstance(match_id, MatchId):
            raise TypeError("match_id must be a MatchId")
        if (
            not isinstance(match_version, int)
            or isinstance(match_version, bool)
            or match_version < 1
        ):
            raise ValueError("match_version must be a positive integer")
        require_utc(as_of, "as_of")
        if not isinstance(baseline_artifact_id, str) or not baseline_artifact_id.startswith(
            "team-baseline:"
        ):
            raise ValueError("baseline_artifact_id must be a team-baseline reference")

        baseline = self.load_team_baseline(baseline_artifact_id)
        self._load_verified_team_baseline_manifest(baseline)
        match, version = self._load_exact_match_version(match_id, match_version)
        kickoff = version.kickoff_at
        if kickoff is None:
            raise ValueError("team-baseline-input/3 requires a scheduled kickoff")
        if as_of >= kickoff:
            raise ValueError("team-baseline-input/3 as_of must precede kickoff")
        self._validate_team_baseline_temporal_inputs(baseline, as_of)
        lambda_home, lambda_away = expected_goals_from_baseline(
            baseline,
            home_team_id=match.home_team_id.value,
            away_team_id=match.away_team_id.value,
        )
        value = {
            "artifact_id": baseline.artifact_id,
            "artifact": team_baseline_payload(baseline),
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
        }
        source_context = self._team_baseline_source_context(
            match_id=match.id,
            match_version=version.version,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            kickoff=kickoff,
            as_of=as_of,
            baseline=baseline,
        )
        input_refs = tuple(sorted(set(baseline.input_refs)))
        input_observed_at = [
            self._validate_snapshot_source_input_ref(reference) for reference in input_refs
        ]
        generated_at = max((version.observed_at, *input_observed_at))
        known_at = baseline.as_of
        if known_at > generated_at:
            raise ValueError("team-baseline-input/3 baseline as_of follows generated_at")
        return self._write_snapshot_source(
            value=value,
            input_refs=input_refs,
            transform_version=TEAM_BASELINE_INPUT_TRANSFORM_V3,
            generated_at=generated_at,
            known_at=known_at,
            source_context=source_context,
        )

    def write_snapshot_source(
        self,
        *,
        value: Any,
        input_refs: tuple[RawAssetId | str, ...],
        transform_version: str,
        generated_at: datetime,
        known_at: datetime | None = None,
    ) -> str:
        """Archive one content-addressed feature value and its verified lineage."""

        if transform_version == "match-context-input/1":
            raise ValueError("match-context-input/1 is audit-only and cannot be newly written")
        if transform_version == MATCH_CONTEXT_INPUT_TRANSFORM_V2:
            raise ValueError("use write_match_context_source for match-context-input/2")
        if transform_version in {
            "team-baseline-input/1",
            "team-baseline-input/2",
            TEAM_BASELINE_INPUT_TRANSFORM_V3,
        }:
            raise ValueError("use write_team_baseline_source for team baseline inputs")
        if transform_version == "official-lineup-input/2":
            raise ValueError("use write_official_lineup_source for official-lineup-input/2")
        return self._write_snapshot_source(
            value=value,
            input_refs=input_refs,
            transform_version=transform_version,
            generated_at=generated_at,
            known_at=known_at,
            source_context=None,
        )

    def write_match_context_source(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        as_of: datetime,
    ) -> str:
        """Persist per-team rest only after replaying canonical schedule and results."""

        replay = replay_match_context(
            match_id=match_id,
            match_version=match_version,
            as_of=as_of,
            archive=RawArchive(self.layout),
            canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
        )
        return self._write_snapshot_source(
            value=replay.value,
            input_refs=replay.input_refs,
            transform_version=MATCH_CONTEXT_INPUT_TRANSFORM_V2,
            generated_at=replay.observed_at,
            known_at=replay.known_at,
            source_context=replay.source_context,
        )

    def write_official_lineup_source(
        self,
        *,
        contract_id: str,
        match_id: MatchId,
        match_version: int,
        team_id: TeamId,
        player_ids: tuple[PlayerId, ...],
        known_at: datetime,
        observed_at: datetime,
        raw_asset_id: RawAssetId,
    ) -> str:
        """Persist an official XI only when exact canonical facts already exist."""

        if not isinstance(match_id, MatchId) or not isinstance(team_id, TeamId):
            raise TypeError("official lineup match_id and team_id must be platform IDs")
        if (
            not isinstance(contract_id, str)
            or not contract_id.startswith("official-lineup-contract:")
            or not _REFERENCE_DIGEST.fullmatch(
                contract_id.removeprefix("official-lineup-contract:")
            )
        ):
            raise ValueError("official lineup contract_id is invalid")
        if (
            not isinstance(match_version, int)
            or isinstance(match_version, bool)
            or match_version < 1
        ):
            raise ValueError("official lineup match_version must be a positive integer")
        if (
            len(player_ids) != 11
            or len(set(player_ids)) != 11
            or any(not isinstance(player_id, PlayerId) for player_id in player_ids)
        ):
            raise ValueError("official lineup source requires 11 unique PlayerId values")
        if not isinstance(raw_asset_id, RawAssetId):
            raise TypeError("official lineup raw_asset_id must be a RawAssetId")
        require_utc(known_at, "known_at")
        require_utc(observed_at, "observed_at")
        source_context = {
            "contract_id": contract_id,
            "match_id": match_id.value,
            "match_version": match_version,
            "team_id": team_id.value,
            "player_ids": [player_id.value for player_id in player_ids],
            "known_at": _timestamp(known_at),
            "observed_at": _timestamp(observed_at),
        }
        return self._write_snapshot_source(
            value=[player_id.value for player_id in player_ids],
            input_refs=(raw_asset_id, contract_id),
            transform_version="official-lineup-input/2",
            generated_at=observed_at,
            known_at=known_at,
            source_context=source_context,
        )

    def _write_snapshot_source(
        self,
        *,
        value: Any,
        input_refs: tuple[RawAssetId | str, ...],
        transform_version: str,
        generated_at: datetime,
        known_at: datetime | None,
        source_context: dict[str, Any] | None,
    ) -> str:

        require_utc(generated_at, "generated_at")
        if not transform_version or transform_version.strip() != transform_version:
            raise ValueError("transform_version must be non-empty text")
        if not input_refs:
            raise ValueError("derived snapshot sources require input_refs")
        normalized_refs = tuple(
            sorted({_snapshot_source_ref_value(reference) for reference in input_refs})
        )
        input_observed_at = [
            self._validate_snapshot_source_input_ref(reference) for reference in normalized_refs
        ]
        if generated_at < max(input_observed_at):
            raise ValueError("generated_at cannot precede an input observation")
        if known_at is not None:
            require_utc(known_at, "known_at")
            if known_at > generated_at:
                raise ValueError("known_at cannot follow generated_at")
        if transform_version == LINEUP_DELTA_INPUT_TRANSFORM_V3:
            if known_at is None:
                raise ValueError("lineup-delta-input/3 requires known_at")
            self._validate_lineup_delta_source(
                value,
                normalized_refs,
                generated_at=generated_at,
                known_at=known_at,
            )
        if transform_version == MATCH_CONTEXT_INPUT_TRANSFORM_V2:
            if known_at is None or source_context is None:
                raise ValueError("match-context-input/2 requires canonical source context")
            self._validate_match_context_source(
                value,
                normalized_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        if transform_version == TEAM_BASELINE_INPUT_TRANSFORM_V3:
            if known_at is None or source_context is None:
                raise ValueError("team-baseline-input/3 requires canonical source context")
            self._validate_team_baseline_source_v3(
                value,
                normalized_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        if transform_version == "official-lineup-input/2":
            if known_at is None or source_context is None:
                raise ValueError("official-lineup-input/2 requires canonical source context")
            self._validate_official_lineup_source(
                value,
                normalized_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        identity = {
            "schema_version": 1,
            "value": value,
            "input_refs": list(normalized_refs),
            "transform_version": transform_version,
            "generated_at": _timestamp(generated_at),
        }
        if known_at is not None:
            identity["known_at"] = _timestamp(known_at)
        if source_context is not None:
            identity["source_context"] = source_context
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        source_ref = f"derived-source:{digest}"
        self._write_json(self._snapshot_source_path(source_ref), {"id": source_ref, **identity})
        return source_ref

    def validate_snapshot_source(
        self,
        source_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> SnapshotSourceValidation:
        """Verify a raw or derived snapshot source against immutable storage."""

        if verification_session is None:
            from football_data_platform.storage.verification import active_verification_session

            verification_session = active_verification_session(self.layout)
        if not source_ref.startswith("derived-source:"):
            raise ValueError(f"snapshot source is not available in derived storage: {source_ref}")
        path = self._snapshot_source_path(source_ref)
        if verification_session is not None:
            verification_session.file_proof(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_id = payload.pop("id", None)
        calculated = f"derived-source:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        if stored_id != source_ref or calculated != source_ref:
            raise ArchiveConflictError(f"derived snapshot source identity failed at {path}")
        if payload.get("schema_version") != 1:
            raise ArchiveConflictError("unsupported derived snapshot source schema")
        input_refs = tuple(str(item) for item in payload.get("input_refs", ()))
        if tuple(sorted(set(input_refs))) != input_refs:
            raise ArchiveConflictError("derived snapshot source input_refs are not canonical")
        input_observed_at = [
            self._validate_snapshot_source_input_ref(
                reference,
                verification_session=verification_session,
            )
            for reference in input_refs
        ]
        generated_at = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
        require_utc(generated_at, "generated_at")
        if not input_observed_at or generated_at < max(input_observed_at):
            raise ArchiveConflictError("derived snapshot source has invalid observation lineage")
        raw_known_at = payload.get("known_at")
        known_at = None
        if raw_known_at is not None:
            known_at = datetime.fromisoformat(str(raw_known_at).replace("Z", "+00:00"))
            require_utc(known_at, "known_at")
            if known_at > generated_at:
                raise ArchiveConflictError("derived snapshot source has invalid known_at")
        source_context = payload.get("source_context")
        if source_context is not None and not isinstance(source_context, dict):
            raise ArchiveConflictError("derived snapshot source context must be an object")
        transform_version = str(payload["transform_version"])
        value = payload["value"]
        if transform_version in {"team-baseline-input/1", "team-baseline-input/2"}:
            self._validate_team_baseline_source(value, input_refs)
        if transform_version == TEAM_BASELINE_INPUT_TRANSFORM_V3:
            if known_at is None or source_context is None:
                raise ArchiveConflictError(
                    "team-baseline-input/3 source is missing canonical context"
                )
            self._validate_team_baseline_source_v3(
                value,
                input_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        if transform_version == LINEUP_DELTA_INPUT_TRANSFORM_V3:
            if known_at is None:
                raise ArchiveConflictError("lineup-delta-input/3 source is missing known_at")
            self._validate_lineup_delta_source(
                value,
                input_refs,
                generated_at=generated_at,
                known_at=known_at,
            )
        if transform_version == MATCH_CONTEXT_INPUT_TRANSFORM_V2:
            if known_at is None or source_context is None:
                raise ArchiveConflictError(
                    "match-context-input/2 source is missing canonical context"
                )
            self._validate_match_context_source(
                value,
                input_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        if transform_version == "official-lineup-input/2":
            if known_at is None or source_context is None:
                raise ArchiveConflictError(
                    "official-lineup-input/2 source is missing canonical context"
                )
            self._validate_official_lineup_source(
                value,
                input_refs,
                generated_at=generated_at,
                known_at=known_at,
                source_context=source_context,
            )
        return SnapshotSourceValidation(
            source_ref,
            "derived",
            transform_version,
            generated_at,
            value,
            input_refs,
            known_at,
            source_context,
        )

    def write_prediction(
        self,
        prediction: ScorePrediction,
        *,
        snapshot: PreMatchSnapshot,
    ) -> Path:
        """Persist a prediction together with its immutable score-grid provenance."""

        verify_current_snapshot(snapshot, source_validator=self)
        verify_prediction_snapshot(
            prediction,
            snapshot=snapshot,
            snapshot_validator=self,
            model_run_validator=self.model_run_validator,
        )
        self.write_score_grid_composition(prediction)
        self.verify_score_grid_composition(prediction)
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
                input_refs=(*prediction.input_refs, prediction.composition_artifact_ref),
                output_refs=(prediction.id.value,),
                status="succeeded",
                quality=prediction.snapshot_quality_status,
            )
        )
        return result

    def write_score_grid_composition(self, prediction: ScorePrediction) -> Path:
        """Write the content-addressed composition/grid manifest for a prediction."""

        verify_score_prediction(
            prediction,
            model_run_validator=self.model_run_validator,
        )
        payload = score_grid_composition_payload(prediction)
        output_ref = score_grid_composition_artifact_id(payload)
        if prediction.composition_artifact_ref != output_ref:
            raise ArchiveConflictError("prediction composition artifact reference is not canonical")
        composition_path = self.score_grid_composition_path(output_ref)
        self._write_json(composition_path, {"id": output_ref, **payload})
        manifest = DerivedArtifactManifest.create(
            artifact_type="score-grid-composition",
            schema_version=SCORE_GRID_COMPOSITION_SCHEMA_VERSION,
            payload=payload,
            generated_at=prediction.generated_at,
            started_at=prediction.generated_at,
            ended_at=prediction.generated_at,
            transform_version=prediction.composition_version,
            code_version=DERIVED_CODE_VERSION,
            input_refs=prediction.input_refs,
            output_refs=(output_ref,),
            status="succeeded",
            quality=prediction.snapshot_quality_status,
        )
        self.write_artifact_manifest(manifest)
        return composition_path

    def verify_score_grid_composition(
        self,
        prediction: ScorePrediction,
        *,
        verification_session: VerificationSession | None = None,
    ) -> DerivedArtifactManifest:
        """Verify the referenced composition bytes and manifest against a prediction."""

        verify_score_prediction(
            prediction,
            model_run_validator=self.model_run_validator,
        )
        artifact_ref = prediction.composition_artifact_ref
        if artifact_ref is None:
            raise ArchiveConflictError("prediction has no composition artifact reference")
        expected_payload = score_grid_composition_payload(prediction)
        expected_output_ref = score_grid_composition_artifact_id(expected_payload)
        if artifact_ref != expected_output_ref:
            raise ArchiveConflictError("prediction composition artifact reference is not canonical")
        composition_path = self.score_grid_composition_path(artifact_ref)
        if verification_session is not None:
            verification_session.file_proof(composition_path)
        try:
            composition_payload = json.loads(composition_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ArchiveConflictError(
                "prediction composition artifact bytes are unavailable or invalid"
            ) from error
        if composition_payload.pop("id", None) != artifact_ref:
            raise ArchiveConflictError("prediction composition artifact ID does not match bytes")
        if composition_payload != expected_payload:
            raise ArchiveConflictError("prediction composition bytes do not match prediction")
        try:
            manifest = self._load_artifact_manifest_for_output_ref(
                artifact_ref,
                verification_session=verification_session,
            )
        except (OSError, ValueError, ArchiveConflictError) as error:
            raise ArchiveConflictError(
                "prediction composition artifact is unavailable or invalid"
            ) from error
        if (
            manifest.artifact_type != "score-grid-composition"
            or manifest.status != "succeeded"
            or manifest.quality != prediction.snapshot_quality_status
            or manifest.generated_at != prediction.generated_at
            or manifest.input_refs != prediction.input_refs
            or manifest.transform_version != prediction.composition_version
            or expected_output_ref not in manifest.output_refs
            or manifest.payload != expected_payload
        ):
            raise ArchiveConflictError("prediction composition artifact does not match prediction")
        if manifest.generated_at < prediction.snapshot_as_of:
            raise ArchiveConflictError("composition artifact predates prediction snapshot as_of")
        return manifest

    def load_score_grid_composition_payload(
        self,
        artifact_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> dict[str, Any]:
        """Load and verify one standalone composition payload by content ID."""

        _validate_score_grid_composition_ref(artifact_ref)
        path = self.score_grid_composition_path(artifact_ref)
        if verification_session is not None:
            verification_session.file_proof(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArchiveConflictError("cannot load score-grid-composition bytes") from error
        stored_id = payload.pop("id", None)
        if stored_id != artifact_ref or score_grid_composition_artifact_id(payload) != artifact_ref:
            raise ArchiveConflictError("score-grid-composition content identity failed")
        return payload

    def load_prediction_payload(self, prediction: ScorePrediction) -> dict[str, Any]:
        verify_score_prediction(
            prediction,
            model_run_validator=self.model_run_validator,
        )
        self.verify_score_grid_composition(prediction)
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
        for profile in profile_items:
            if not isinstance(profile, PlayerProfile):
                raise TypeError("player profile artifacts must be PlayerProfile values")
            validate_player_profile(profile)
        transform_versions = {
            str(item.transform_version)
            for item in profile_items
            if getattr(item, "transform_version", None)
        }
        if len(transform_versions) > 1:
            raise ValueError("player profiles must use one transform_version per manifest")
        for reference in sorted(
            {ref for item in profile_items for ref in getattr(item, "input_refs", ())}
        ):
            self._verify_player_profile_input_ref(reference)
        profile_payload = {
            "profiles": [_json_safe(item) for item in profile_items],
            "excluded_input_refs": list(getattr(profiles, "excluded_input_refs", ())),
            "partition_audits": [
                _json_safe(item) for item in getattr(profiles, "partition_audits", ())
            ],
        }
        for reference in sorted(set(profile_payload["excluded_input_refs"])):
            self._verify_player_profile_input_ref(reference)
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

    def load_player_profile(self, artifact_id: str) -> PlayerProfile:
        """Load one profile only after validating its enclosing manifest."""

        profile, _ = self._load_player_profile_with_manifest(artifact_id)
        return profile

    def _load_player_profile_with_manifest(
        self, artifact_id: str
    ) -> tuple[PlayerProfile, DerivedArtifactManifest]:
        if not isinstance(artifact_id, str) or not artifact_id.startswith("player-profile:"):
            raise ArchiveConflictError("invalid player profile artifact reference")
        manifest = self._load_artifact_manifest_for_output_ref(artifact_id)
        if manifest.artifact_type != "player-profiles":
            raise ArchiveConflictError(
                "player profile reference resolves to the wrong artifact type"
            )
        payload = manifest.payload
        if not isinstance(payload, dict) or set(payload) != {
            "profiles",
            "excluded_input_refs",
            "partition_audits",
        }:
            raise ArchiveConflictError("player profile manifest payload is invalid")
        raw_profiles = payload["profiles"]
        if not isinstance(raw_profiles, list):
            raise ArchiveConflictError("player profile manifest profiles are invalid")
        try:
            profiles = tuple(parse_player_profile_payload(item) for item in raw_profiles)
        except ValueError as error:
            raise ArchiveConflictError(
                "player profile manifest contains an invalid profile"
            ) from error
        by_id = {profile.artifact_id: profile for profile in profiles}
        if len(by_id) != len(profiles) or set(by_id) != set(manifest.output_refs):
            raise ArchiveConflictError("player profile manifest outputs do not match profiles")
        if artifact_id not in by_id:
            raise ArchiveConflictError("player profile artifact is not exposed by its manifest")
        excluded = payload["excluded_input_refs"]
        if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
            raise ArchiveConflictError("player profile manifest excluded refs are invalid")
        for reference in sorted(set(excluded)):
            self._verify_player_profile_input_ref(reference)
        expected_inputs = tuple(
            sorted({ref for profile in profiles for ref in profile.input_refs} | set(excluded))
        )
        if manifest.input_refs != expected_inputs:
            raise ArchiveConflictError("player profile manifest inputs do not match profiles")
        transforms = {profile.transform_version for profile in profiles}
        expected_quality = (
            "ready"
            if profiles and all(profile.quality_status == "ready" for profile in profiles)
            else ("preview" if profiles else "failed")
        )
        expected_status = (
            "succeeded" if expected_quality == "ready" else ("partial" if profiles else "failed")
        )
        expected_error = None if profiles else "no_player_profiles"
        expected_transform = sorted(transforms)[0] if transforms else "player-profile/unknown"
        if (
            len(transforms) > 1
            or manifest.transform_version != expected_transform
            or manifest.quality != expected_quality
            or manifest.status != expected_status
            or manifest.error != expected_error
            or any(manifest.generated_at < profile.as_of for profile in profiles)
        ):
            raise ArchiveConflictError("player profile manifest metadata does not match profiles")
        for profile in profiles:
            for reference in profile.input_refs:
                self._verify_player_profile_input_ref(reference)
        return by_id[artifact_id], manifest

    def _verify_player_profile_input_ref(self, reference: str) -> None:
        """Resolve a profile input through an immutable raw, derived, or canonical store."""

        if not isinstance(reference, str) or not reference or reference.strip() != reference:
            raise ArchiveConflictError("player profile input reference must be non-empty text")
        try:
            if reference.startswith("raw-asset:"):
                RawArchive(self.layout).verify(RawAssetId(reference))
                return
            if reference.startswith("derived-source:"):
                self.validate_snapshot_source(reference)
                return
            if reference.startswith("derived-artifact:"):
                self.load_artifact_manifest(reference)
                return
            if reference.startswith("team-baseline:"):
                self.load_team_baseline(reference)
                manifest = self._load_artifact_manifest_for_output_ref(reference)
                if manifest.artifact_type != "team-baseline":
                    raise ArchiveConflictError(
                        "profile input team baseline manifest has wrong type"
                    )
                return
            if reference.startswith("player-profile:"):
                self._load_player_profile_with_manifest(reference)
                return
            if reference.startswith(("canonical:", "fact:", "event:", "lineup:")):
                self._verify_canonical_player_profile_ref(reference)
                return
        except (OSError, KeyError, ValueError, sqlite3.Error, ArchiveConflictError) as error:
            raise ArchiveConflictError(
                f"player profile input reference is unavailable or invalid: {reference}"
            ) from error
        raise ArchiveConflictError(f"unsupported player profile input reference: {reference}")

    def _verify_canonical_player_profile_ref(self, reference: str) -> None:
        path = self.layout.canonical / "platform.sqlite3"
        if not path.exists():
            raise ArchiveConflictError("canonical store is unavailable for player profile input")
        with sqlite3.connect(path) as connection:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            for (table_name,) in table_rows:
                if not isinstance(table_name, str) or '"' in table_name:
                    continue
                columns = {
                    row[1] for row in connection.execute(f'PRAGMA table_info("{table_name}")')
                }
                if "record_id" not in columns:
                    continue
                query = f'SELECT 1 FROM "{table_name}" WHERE record_id = ? LIMIT 1'
                if connection.execute(query, (reference,)).fetchone() is not None:
                    return
        raise ArchiveConflictError(f"canonical player profile input does not exist: {reference}")

    def write_evaluation(
        self,
        evaluation: Any,
        *,
        generated_at: datetime | None = None,
        code_version: str = DERIVED_CODE_VERSION,
    ) -> Path:
        """Archive one evaluation record without making market data implicit."""

        from football_data_platform.evaluation.metrics import (
            EvaluationRecord,
            evaluation_record_payload,
            verify_evaluation_record,
        )

        if not isinstance(evaluation, EvaluationRecord):
            raise TypeError("evaluation must be an EvaluationRecord")
        verify_evaluation_record(evaluation)
        if generated_at is not None and generated_at != evaluation.evaluated_at:
            raise ValueError("evaluation generated_at must equal evaluated_at")
        payload = evaluation_record_payload(evaluation)
        input_refs = evaluation.input_refs
        evaluated_at = evaluation.evaluated_at
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        manifest = DerivedArtifactManifest.create(
            artifact_type="evaluation",
            schema_version=evaluation.schema_version,
            payload=payload,
            generated_at=evaluated_at,
            transform_version="evaluation/2",
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

    def score_grid_composition_path(self, artifact_ref: str) -> Path:
        _validate_score_grid_composition_ref(artifact_ref)
        digest = artifact_ref.removeprefix("score-grid-composition:")
        return self.layout.derived / "compositions" / digest[:2] / f"{digest}.json"

    def _load_artifact_manifest_for_output_ref(
        self,
        output_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> DerivedArtifactManifest:
        """Resolve a logical output ref through immutable manifests.

        Manifest IDs include execution metadata, whereas composition IDs are
        payload content IDs.  Resolving by ``output_refs`` keeps those concerns
        separate and lets the manifest remain fully content-addressed.
        """

        matches = self._load_artifact_manifests_for_output_ref(
            output_ref,
            verification_session=verification_session,
        )
        if not matches:
            raise ArchiveConflictError(f"no derived manifest exposes output ref {output_ref}")
        if len(matches) > 1:
            semantic_identities = {_artifact_output_semantic_identity(item) for item in matches}
            if len(semantic_identities) > 1:
                raise ArchiveConflictError(
                    f"output ref {output_ref} resolves to conflicting manifests"
                )
        return min(matches, key=lambda item: item.artifact_id)

    def _load_artifact_manifests_for_output_ref(
        self,
        output_ref: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> list[DerivedArtifactManifest]:
        """Load every valid manifest file that claims one logical output ref."""

        if verification_session is None:
            from football_data_platform.storage.verification import active_verification_session

            verification_session = active_verification_session(self.layout)
        if verification_session is not None:
            return [
                self.load_artifact_manifest(
                    entry.artifact_id,
                    verification_session=verification_session,
                )
                for entry in verification_session.manifests_for_output_ref(output_ref)
            ]
        matches: list[DerivedArtifactManifest] = []
        for path in (self.layout.derived / "manifests" / "artifacts").rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            raw_output_refs = payload.get("output_refs")
            if (
                not isinstance(raw_output_refs, (list, tuple))
                or any(not isinstance(item, str) for item in raw_output_refs)
                or output_ref not in raw_output_refs
            ):
                continue
            manifest = _parse_artifact_manifest(payload)
            matches.append(self.load_artifact_manifest(manifest.artifact_id))
        return matches

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

    def _validate_snapshot_source_input_ref(
        self,
        reference: str,
        *,
        verification_session: VerificationSession | None = None,
    ) -> datetime:
        if reference.startswith("raw-asset:"):
            asset_id = RawAssetId(reference)
            raw = RawArchive(self.layout)
            if verification_session is not None:
                verification_session.file_proof(self.layout.raw_manifest_path(asset_id))
            asset = raw.load(asset_id)
            if verification_session is not None:
                verification_session.file_proof(self.layout.raw_object_path(asset.checksum))
            # FileProof extends byte stability only. RawArchive remains the domain authority for
            # manifest identity, object checksum, and size.
            raw.verify(asset_id)
            return asset.observed_at
        if reference.startswith("official-lineup-contract:"):
            return self._replay_official_lineup_contract(reference).observed_at
        if reference.startswith("fact:team_match_observations:"):
            return load_verified_team_observation(
                reference,
                archive=RawArchive(self.layout),
                canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
                verification_session=verification_session,
            ).observed_at
        if reference.startswith("fact:match_results_90:"):
            canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
            load_verified_match_result(
                reference,
                archive=RawArchive(self.layout),
                canonical=canonical,
                verification_session=verification_session,
            )
            with canonical.connect() as connection:
                row = connection.execute(
                    "SELECT observed_at FROM match_results_90 WHERE record_id = ?",
                    (reference,),
                ).fetchone()
            if row is None:
                raise ArchiveConflictError("canonical result reference is unavailable")
            return _parse_manifest_datetime(row["observed_at"], "result observed_at")
        if reference.startswith("team-baseline:"):
            baseline = self.load_team_baseline(reference)
            self._load_verified_team_baseline_manifest(baseline)
            return baseline.as_of
        try:
            if reference.startswith("player-profile:"):
                _, manifest = self._load_player_profile_with_manifest(reference)
                return manifest.generated_at
            if reference.startswith("derived-source:"):
                return self.validate_snapshot_source(
                    reference,
                    verification_session=verification_session,
                ).observed_at
        except (OSError, KeyError, TypeError, ValueError, ArchiveConflictError) as error:
            raise ArchiveConflictError(
                f"snapshot source input reference is unavailable or invalid: {reference}"
            ) from error
        raise ArchiveConflictError(f"unsupported snapshot source input reference: {reference}")

    def _validate_match_context_source(
        self,
        value: Any,
        input_refs: tuple[str, ...],
        *,
        generated_at: datetime,
        known_at: datetime,
        source_context: dict[str, Any],
    ) -> None:
        expected_fields = {
            "rule_version",
            "match_id",
            "match_version",
            "as_of",
            "scheduled_kickoff",
            "home_team_id",
            "away_team_id",
            "quality_status",
        }
        if set(source_context) != expected_fields:
            raise ArchiveConflictError("match-context-input/2 source context is invalid")
        try:
            replay = replay_match_context(
                match_id=MatchId(str(source_context["match_id"])),
                match_version=source_context["match_version"],
                as_of=_parse_manifest_datetime(source_context["as_of"], "context as_of"),
                archive=RawArchive(self.layout),
                canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise ArchiveConflictError(
                "match-context-input/2 canonical replay is unavailable or invalid"
            ) from error
        if (
            _canonical_json(value) != _canonical_json(replay.value)
            or input_refs != replay.input_refs
            or generated_at != replay.observed_at
            or known_at != replay.known_at
            or source_context != replay.source_context
        ):
            raise ArchiveConflictError(
                "match-context-input/2 does not match recomputed canonical context"
            )

    def _validate_official_lineup_source(
        self,
        value: Any,
        input_refs: tuple[str, ...],
        *,
        generated_at: datetime,
        known_at: datetime,
        source_context: dict[str, Any],
    ) -> None:
        expected_fields = {
            "contract_id",
            "match_id",
            "match_version",
            "team_id",
            "player_ids",
            "known_at",
            "observed_at",
        }
        if (
            set(source_context) != expected_fields
            or not isinstance(value, list)
            or len(value) != 11
            or any(not isinstance(player_id, str) for player_id in value)
            or len(set(value)) != 11
            or source_context.get("player_ids") != value
            or source_context.get("known_at") != _timestamp(known_at)
            or source_context.get("observed_at") != _timestamp(generated_at)
        ):
            raise ArchiveConflictError("official-lineup-input/2 source context is invalid")
        raw_refs = tuple(
            reference for reference in input_refs if reference.startswith("raw-asset:")
        )
        contract_refs = tuple(
            reference
            for reference in input_refs
            if reference.startswith("official-lineup-contract:")
        )
        if (
            len(raw_refs) != 1
            or len(contract_refs) != 1
            or len(input_refs) != 2
            or source_context.get("contract_id") != contract_refs[0]
        ):
            raise ArchiveConflictError("official-lineup-input/2 lineage is invalid")
        try:
            MatchId(source_context["match_id"])
            TeamId(source_context["team_id"])
            players = {PlayerId(player_id).value for player_id in value}
        except (KeyError, TypeError, ValueError) as error:
            raise ArchiveConflictError(
                "official-lineup-input/2 platform IDs are invalid"
            ) from error
        match_version = source_context["match_version"]
        if (
            not isinstance(match_version, int)
            or isinstance(match_version, bool)
            or match_version < 1
        ):
            raise ArchiveConflictError("official-lineup-input/2 match_version is invalid")
        contract = self._replay_official_lineup_contract(contract_refs[0])
        contract_lineup = dict(contract.team_lineups).get(TeamId(source_context["team_id"]))
        if (
            contract.raw_asset_id.value != raw_refs[0]
            or contract.match_id.value != source_context["match_id"]
            or contract.match_version != match_version
            or contract.published_at != known_at
            or contract.observed_at != generated_at
            or contract_lineup is None
            or {player_id.value for player_id in contract_lineup} != players
        ):
            raise ArchiveConflictError(
                "official-lineup-input/2 does not match its verified contract"
            )

    def _replay_official_lineup_contract(self, contract_id: str):
        try:
            return verify_official_lineup_contract(
                contract_id,
                archive=RawArchive(self.layout),
                canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise ArchiveConflictError(
                "official lineup contract replay is unavailable or invalid"
            ) from error

    def _validate_lineup_delta_source(
        self,
        value: Any,
        input_refs: tuple[str, ...],
        *,
        generated_at: datetime,
        known_at: datetime,
    ) -> None:
        if not isinstance(value, dict) or not value:
            raise ArchiveConflictError("lineup-delta-input/3 value must be keyed by team")
        if not any(reference.startswith("raw-asset:") for reference in input_refs):
            raise ArchiveConflictError("lineup-delta-input/3 requires raw lineup evidence")
        aggregate_fields = {
            "quality_status",
            "dimension_deltas",
            "missing_fields",
            "input_refs",
            "not_applicable_fields",
            "blocking_reasons",
            "transform_version",
            "role_contract_versions",
            "dimension_sample_sizes",
        }
        evidence_fields = {
            "starter_ids",
            "reference_starter_ids",
            "player_profile_refs",
            "profile_window",
            "profile_window_version",
            "lineup_input_refs",
            "reference_lineup_ref",
        }
        nested_profile_refs: set[str] = set()
        nested_lineup_refs: set[str] = set()
        nested_reference_refs: set[str] = set()
        for team_id, item in sorted(value.items()):
            if (
                not isinstance(team_id, str)
                or not team_id
                or team_id.strip() != team_id
                or not isinstance(item, dict)
                or set(item) != aggregate_fields | evidence_fields
            ):
                raise ArchiveConflictError("lineup-delta-input/3 team evidence is invalid")
            raw_starters = item["starter_ids"]
            raw_reference = item["reference_starter_ids"]
            raw_profile_refs = item["player_profile_refs"]
            raw_lineup_refs = item["lineup_input_refs"]
            raw_reference_ref = item["reference_lineup_ref"]
            if (
                not isinstance(raw_starters, list)
                or any(not isinstance(player_id, str) for player_id in raw_starters)
                or (
                    raw_reference is not None
                    and (
                        not isinstance(raw_reference, list)
                        or any(not isinstance(player_id, str) for player_id in raw_reference)
                    )
                )
                or not isinstance(raw_profile_refs, dict)
                or any(
                    not isinstance(player_id, str)
                    or not isinstance(reference, str)
                    or not reference.startswith("player-profile:")
                    for player_id, reference in raw_profile_refs.items()
                )
                or not isinstance(raw_lineup_refs, list)
                or len(raw_lineup_refs) != len(set(raw_lineup_refs))
                or any(
                    not isinstance(reference, str) or not reference.startswith("raw-asset:")
                    for reference in raw_lineup_refs
                )
                or (
                    raw_reference_ref is not None
                    and (
                        not isinstance(raw_reference_ref, str)
                        or not raw_reference_ref.startswith("derived-source:")
                    )
                )
                or ((raw_reference is None) != (raw_reference_ref is None))
            ):
                raise ArchiveConflictError("lineup-delta-input/3 player evidence is invalid")
            starters = tuple(raw_starters)
            reference = None if raw_reference is None else tuple(raw_reference)
            profile_refs = dict(raw_profile_refs)
            lineup_refs = tuple(raw_lineup_refs)
            nested_lineup_refs.update(lineup_refs)
            reference_source_ids: tuple[str, ...] | None = None
            if raw_reference_ref is not None:
                reference_validation = self.validate_snapshot_source(raw_reference_ref)
                context = reference_validation.source_context
                if (
                    reference_validation.transform_version != "official-lineup-input/2"
                    or not isinstance(reference_validation.value, list)
                    or not isinstance(context, dict)
                    or context.get("team_id") != team_id
                    or reference_validation.known_at is None
                    or reference_validation.known_at > known_at
                    or reference_validation.observed_at > generated_at
                ):
                    raise ArchiveConflictError(
                        "reference lineup source does not match lineup delta context"
                    )
                reference_source_ids = tuple(reference_validation.value)
                if tuple(raw_reference or ()) != reference_source_ids:
                    raise ArchiveConflictError(
                        "reference lineup IDs do not match referenced lineup source"
                    )
                nested_reference_refs.add(raw_reference_ref)
            required_players = set(starters) | set(reference or ())
            if set(profile_refs) - required_players:
                raise ArchiveConflictError("lineup profile refs contain players outside the XI")
            profiles: list[PlayerProfile] = []
            for player_id, profile_ref in sorted(profile_refs.items()):
                profile, manifest = self._load_player_profile_with_manifest(profile_ref)
                if (
                    profile.player_id != player_id
                    or profile.team_id != team_id
                    or profile.as_of > known_at
                    or manifest.generated_at > generated_at
                ):
                    raise ArchiveConflictError(
                        "lineup player profile identity or time does not match evidence"
                    )
                profiles.append(profile)
                nested_profile_refs.add(profile_ref)
            try:
                expected = lineup_delta_input_payload(
                    starter_ids=starters,
                    reference_starter_ids=reference,
                    profiles=tuple(profiles),
                    profile_window=item["profile_window"],
                    profile_window_version=item["profile_window_version"],
                    lineup_input_refs=lineup_refs,
                    reference_lineup_ref=raw_reference_ref,
                )
                delta = parse_lineup_delta_payload(item)
            except (TypeError, ValueError) as error:
                raise ArchiveConflictError("lineup-delta-input/3 cannot be recomputed") from error
            if _canonical_json(expected) != _canonical_json(item):
                raise ArchiveConflictError(
                    "lineup-delta-input/3 does not match recomputed profiles"
                )
            if delta.quality_status == "ready" and any(
                profile.quality_status != "ready" for profile in profiles
            ):
                raise ArchiveConflictError("ready lineup delta requires ready player profiles")
        if nested_profile_refs | nested_lineup_refs | nested_reference_refs != set(input_refs):
            raise ArchiveConflictError("lineup delta nested refs do not match source input_refs")

    def _load_exact_match_version(self, match_id: MatchId, match_version: int):
        """Load one immutable canonical match/version pair."""

        try:
            match = CanonicalStore(self.layout.canonical / "platform.sqlite3").match(match_id)
            versions = CanonicalStore(self.layout.canonical / "platform.sqlite3").match_versions(
                match_id
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise ArchiveConflictError("canonical match version is unavailable") from error
        for version in versions:
            if version.version == match_version:
                return match, version
        raise ArchiveConflictError("canonical match version does not exist")

    def _load_verified_team_baseline_manifest(
        self, baseline: TeamBaselineArtifact
    ) -> DerivedArtifactManifest:
        """Require exactly one manifest matching the baseline writer contract."""

        try:
            matches = self._load_artifact_manifests_for_output_ref(baseline.artifact_id)
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise ArchiveConflictError("team baseline artifact manifest is unavailable") from error
        if len(matches) != 1:
            raise ArchiveConflictError("team baseline artifact requires exactly one manifest")
        manifest = matches[0]
        expected_status = "succeeded" if baseline.quality_status == "ready" else "partial"
        if (
            manifest.artifact_type != "team-baseline"
            or manifest.schema_version != baseline.schema_version
            or manifest.transform_version != baseline.transform_version
            or manifest.input_refs != tuple(sorted(set(baseline.input_refs)))
            or manifest.output_refs != (baseline.artifact_id,)
            or manifest.status != expected_status
            or manifest.quality != baseline.quality_status
            or manifest.error is not None
            or manifest.payload != team_baseline_payload(baseline)
        ):
            raise ArchiveConflictError("team baseline artifact manifest does not match payload")
        if not (
            manifest.started_at == manifest.generated_at == manifest.ended_at
            and manifest.generated_at >= baseline.as_of
        ):
            raise ArchiveConflictError("team baseline artifact manifest times are invalid")
        return manifest

    def _validate_team_baseline_temporal_inputs(
        self, baseline: TeamBaselineArtifact, as_of: datetime
    ) -> None:
        """Ensure baseline semantics and known input evidence do not follow the cutoff.

        Historical reconstructed evidence may have been collected after the
        historical cutoff.  In that case a raw asset's explicit target event
        time (or a typed fact/source ``known_at``) remains the semantic clock;
        an unknown target event time is not silently interpreted as a future
        fact.
        """

        if baseline.as_of > as_of:
            raise ArchiveConflictError("team baseline as_of follows feature cutoff")
        for reference in baseline.input_refs:
            self._validate_team_baseline_input_available(reference, as_of)

    def _validate_team_baseline_input_available(self, reference: str, as_of: datetime) -> None:
        if reference.startswith("raw-asset:"):
            raw = RawArchive(self.layout)
            asset = raw.load(RawAssetId(reference))
            raw.verify(asset)
            if asset.target_event_time is None:
                raise ArchiveConflictError(
                    "team baseline raw input lacks a semantic known_at timestamp"
                )
            if asset.target_event_time > as_of:
                raise ArchiveConflictError(
                    f"team baseline input follows feature cutoff: {reference}"
                )
            return
        if reference.startswith("fact:team_match_observations:"):
            observation = load_verified_team_observation(
                reference,
                archive=RawArchive(self.layout),
                canonical=CanonicalStore(self.layout.canonical / "platform.sqlite3"),
            )
            if observation.known_at > as_of:
                raise ArchiveConflictError(
                    f"team baseline input known after feature cutoff: {reference}"
                )
            return
        if reference.startswith("fact:match_results_90:"):
            canonical = CanonicalStore(self.layout.canonical / "platform.sqlite3")
            result = load_verified_match_result(
                reference,
                archive=RawArchive(self.layout),
                canonical=canonical,
            )
            if result.known_at > as_of:
                raise ArchiveConflictError(
                    f"team baseline input known after feature cutoff: {reference}"
                )
            return
        if reference.startswith("derived-source:"):
            validation = self.validate_snapshot_source(reference)
            semantic_time = validation.known_at or validation.observed_at
            if semantic_time > as_of:
                raise ArchiveConflictError(
                    f"team baseline input known after feature cutoff: {reference}"
                )
            return
        if reference.startswith("team-baseline:"):
            nested = self.load_team_baseline(reference)
            if nested.as_of > as_of:
                raise ArchiveConflictError(
                    f"nested team baseline follows feature cutoff: {reference}"
                )
            return
        # Let the normal source resolver provide the typed-reference error.
        self._validate_snapshot_source_input_ref(reference)

    def _team_baseline_source_context(
        self,
        *,
        match_id: MatchId,
        match_version: int,
        home_team_id: TeamId,
        away_team_id: TeamId,
        kickoff: datetime,
        as_of: datetime,
        baseline: TeamBaselineArtifact,
    ) -> dict[str, Any]:
        return {
            "rule_version": "expected-goals-from-team-baseline/1",
            "transform_version": TEAM_BASELINE_INPUT_TRANSFORM_V3,
            "match_id": match_id.value,
            "match_version": match_version,
            "home_team_id": home_team_id.value,
            "away_team_id": away_team_id.value,
            "scheduled_kickoff": _timestamp(kickoff),
            "as_of": _timestamp(as_of),
            "baseline_artifact_id": baseline.artifact_id,
            "baseline_as_of": _timestamp(baseline.as_of),
            "baseline_transform_version": baseline.transform_version,
        }

    def _validate_team_baseline_source_v3(
        self,
        value: Any,
        input_refs: tuple[str, ...],
        *,
        generated_at: datetime,
        known_at: datetime,
        source_context: dict[str, Any],
    ) -> TeamBaselineArtifact:
        """Replay and validate the formal baseline contribution contract."""

        expected_context_fields = {
            "rule_version",
            "transform_version",
            "match_id",
            "match_version",
            "home_team_id",
            "away_team_id",
            "scheduled_kickoff",
            "as_of",
            "baseline_artifact_id",
            "baseline_as_of",
            "baseline_transform_version",
        }
        if set(source_context) != expected_context_fields:
            raise ArchiveConflictError("team-baseline-input/3 source context is invalid")
        if source_context.get("rule_version") != "expected-goals-from-team-baseline/1":
            raise ArchiveConflictError("team-baseline-input/3 rule version is invalid")
        if source_context.get("transform_version") != TEAM_BASELINE_INPUT_TRANSFORM_V3:
            raise ArchiveConflictError("team-baseline-input/3 transform context is invalid")
        if not isinstance(value, dict) or set(value) != {
            "artifact_id",
            "artifact",
            "lambda_home",
            "lambda_away",
        }:
            raise ArchiveConflictError("team-baseline-input/3 value is not canonical")
        try:
            match_id = MatchId(str(source_context["match_id"]))
            match_version = source_context["match_version"]
            home_team_id = TeamId(str(source_context["home_team_id"]))
            away_team_id = TeamId(str(source_context["away_team_id"]))
            as_of = _parse_manifest_datetime(source_context["as_of"], "baseline as_of")
            kickoff = _parse_manifest_datetime(
                source_context["scheduled_kickoff"], "scheduled kickoff"
            )
            baseline_as_of = _parse_manifest_datetime(
                source_context["baseline_as_of"], "baseline artifact as_of"
            )
        except (TypeError, ValueError, ArchiveConflictError) as error:
            raise ArchiveConflictError(
                "team-baseline-input/3 source context has invalid IDs"
            ) from error
        if (
            not isinstance(match_version, int)
            or isinstance(match_version, bool)
            or match_version < 1
            or home_team_id == away_team_id
        ):
            raise ArchiveConflictError("team-baseline-input/3 match identity is invalid")
        if known_at != baseline_as_of or as_of >= kickoff or baseline_as_of > as_of:
            raise ArchiveConflictError("team-baseline-input/3 temporal context is invalid")
        try:
            baseline = self.load_team_baseline(str(source_context["baseline_artifact_id"]))
            self._load_verified_team_baseline_manifest(baseline)
            nested = parse_team_baseline_payload(value["artifact"])
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
            raise ArchiveConflictError(
                "team-baseline-input/3 baseline artifact is unavailable"
            ) from error
        if (
            value["artifact_id"] != baseline.artifact_id
            or source_context["baseline_artifact_id"] != baseline.artifact_id
            or nested != baseline
            or source_context["baseline_transform_version"] != baseline.transform_version
            or source_context["baseline_as_of"] != _timestamp(baseline.as_of)
        ):
            raise ArchiveConflictError("team-baseline-input/3 baseline identity is inconsistent")
        expected_refs = tuple(sorted(set(baseline.input_refs)))
        if input_refs != expected_refs:
            raise ArchiveConflictError("team-baseline-input/3 input_refs do not match baseline")
        self._validate_team_baseline_temporal_inputs(baseline, as_of)
        match, version = self._load_exact_match_version(match_id, match_version)
        if (
            match.home_team_id != home_team_id
            or match.away_team_id != away_team_id
            or version.kickoff_at != kickoff
        ):
            raise ArchiveConflictError(
                "team-baseline-input/3 match context does not match canonical"
            )
        expected_context = self._team_baseline_source_context(
            match_id=match.id,
            match_version=version.version,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            kickoff=kickoff,
            as_of=as_of,
            baseline=baseline,
        )
        if source_context != expected_context:
            raise ArchiveConflictError("team-baseline-input/3 source context is not canonical")
        expected_home, expected_away = expected_goals_from_baseline(
            baseline,
            home_team_id=home_team_id.value,
            away_team_id=away_team_id.value,
        )
        if (
            type(value["lambda_home"]) not in {int, float}
            or type(value["lambda_away"]) not in {int, float}
            or not math.isfinite(float(value["lambda_home"]))
            or not math.isfinite(float(value["lambda_away"]))
            or value["lambda_home"] <= 0
            or value["lambda_away"] <= 0
            or value["lambda_home"] != expected_home
            or value["lambda_away"] != expected_away
        ):
            raise ArchiveConflictError("team-baseline-input/3 lambdas do not match baseline replay")
        expected_generated_at = max(
            (
                version.observed_at,
                *(self._validate_snapshot_source_input_ref(reference) for reference in input_refs),
            )
        )
        if generated_at != expected_generated_at:
            raise ArchiveConflictError("team-baseline-input/3 generated_at is not canonical")
        return baseline

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


class _ManifestReferenceResolver:
    """Validate typed manifest refs and resolve successful input lineage."""

    def __init__(
        self,
        archive: DerivedArchive,
        verification_session: VerificationSession | None = None,
    ) -> None:
        self.archive = archive
        self.verification_session = verification_session

    def verify(
        self,
        *,
        input_refs: tuple[str, ...],
        output_refs: tuple[str, ...],
        status: str,
    ) -> None:
        for reference in output_refs:
            namespace = self._validate_syntax(reference)
            if namespace in {"command", "missing-file", "source-url"}:
                raise ArchiveConflictError(
                    f"manifest output reference cannot be a diagnostic or locator: {reference}"
                )
            if namespace in {"market-snapshot", "sample"}:
                raise ArchiveConflictError(
                    f"manifest output reference has no authoritative standalone store: {reference}"
                )

        resolved_inputs = 0
        for reference in input_refs:
            namespace = self._validate_syntax(reference)
            if namespace in {"command", "missing-file"}:
                if status != "failed":
                    raise ArchiveConflictError(
                        f"{namespace}: references are only allowed in failed manifests"
                    )
                continue
            if namespace == "source-url":
                continue
            if status not in _SUCCESS_STATUSES:
                continue
            try:
                self._resolve_input(reference, namespace)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, sqlite3.Error) as error:
                raise ArchiveConflictError(
                    f"manifest input reference is unavailable or invalid: {reference}"
                ) from error
            resolved_inputs += 1

        if status in _SUCCESS_STATUSES and resolved_inputs == 0:
            raise ArchiveConflictError(
                "successful and partial manifests require resolvable input lineage"
            )

    @staticmethod
    def _validate_syntax(reference: str) -> str:
        if not isinstance(reference, str) or not reference or reference.strip() != reference:
            raise ArchiveConflictError("manifest reference must be non-empty canonical text")
        namespace, separator, suffix = reference.partition(":")
        if not separator or not suffix:
            raise ArchiveConflictError(f"malformed manifest reference: {reference}")
        if namespace in _DIGEST_REFERENCE_NAMESPACES:
            if not _REFERENCE_DIGEST.fullmatch(suffix):
                raise ArchiveConflictError(f"malformed {namespace} manifest reference: {reference}")
            return namespace
        if namespace == "collection-attempt":
            if not _REFERENCE_UUID.fullmatch(suffix):
                raise ArchiveConflictError(f"malformed collection-attempt reference: {reference}")
            return namespace
        if namespace == "fact":
            kind, kind_separator, digest = suffix.partition(":")
            if (
                not kind_separator
                or not _REFERENCE_TOKEN.fullmatch(kind)
                or not _REFERENCE_DIGEST.fullmatch(digest)
            ):
                raise ArchiveConflictError(f"malformed canonical fact reference: {reference}")
            return namespace
        if namespace in _ENTITY_REFERENCE_NAMESPACES or namespace in {"command", "source-url"}:
            if not _REFERENCE_TOKEN.fullmatch(suffix):
                raise ArchiveConflictError(f"malformed {namespace} manifest reference: {reference}")
            return namespace
        if namespace in _CANONICAL_RECORD_NAMESPACES | {"sample"}:
            if any(character.isspace() for character in suffix):
                raise ArchiveConflictError(f"malformed {namespace} manifest reference: {reference}")
            return namespace
        if namespace == "missing-file":
            return namespace
        raise ArchiveConflictError(
            f"unsupported manifest reference namespace {namespace!r}: {reference}"
        )

    def _resolve_input(self, reference: str, namespace: str) -> None:
        if namespace == "file-sha256":
            return
        if namespace == "raw-asset":
            RawArchive(self.archive.layout).verify(RawAssetId(reference))
            return
        if namespace == "prediction":
            self._resolve_prediction_reference(reference)
            return
        if namespace in _TRAINING_REFERENCE_NAMESPACES:
            self._resolve_training_reference(reference)
            return
        if namespace == "derived-artifact":
            self.archive.load_artifact_manifest(
                reference,
                verification_session=self.verification_session,
            )
            return
        if namespace == "run":
            self.archive.load_run_manifest(
                reference,
                verification_session=self.verification_session,
            )
            return
        if namespace == "player-profile":
            self.archive.load_player_profile(reference)
            return
        if namespace in _MANIFEST_OUTPUT_REFERENCE_NAMESPACES:
            self.archive._load_artifact_manifest_for_output_ref(
                reference,
                verification_session=self.verification_session,
            )
            return
        if (
            namespace in _CANONICAL_RECORD_NAMESPACES | _ENTITY_REFERENCE_NAMESPACES
            or namespace
            in {
                "collection-attempt",
                "match-report-contract",
                "official-lineup-contract",
            }
        ):
            self._resolve_canonical_reference(reference, namespace)
            return
        if namespace == "sample":
            self._resolve_sample_reference(reference)
            return
        if namespace == "paper-bet-entry":
            from football_data_platform.storage.ledger import PaperBetLedger

            PaperBetLedger(
                self.archive.layout,
                market_snapshot_validator=self.archive.market_snapshot_validator,
                prediction_context_validator=self.archive.prediction_context_validator,
            ).load(reference)
            return
        if namespace in {"challenger-evidence", "promotion-decision", "promotion-policy"}:
            from football_data_platform.storage.governance import GovernanceArtifactStore

            GovernanceArtifactStore(self.archive.layout).verify_reference(reference)
            return
        if namespace == "market-snapshot":
            raise ArchiveConflictError(
                "market-snapshot references require an authoritative persisted snapshot store"
            )
        raise ArchiveConflictError(f"manifest reference has no resolver: {reference}")

    def _resolve_training_reference(self, reference: str) -> None:
        if reference.startswith("model-run:") and self.archive.model_run_validator is not None:
            self.archive.model_run_validator.load_model_run(reference)
            return
        if reference.startswith("score-grid-composition:"):
            manifest = self.archive._load_artifact_manifest_for_output_ref(
                reference,
                verification_session=self.verification_session,
            )
            if manifest.artifact_type != "score-grid-composition":
                raise ArchiveConflictError(
                    "score-grid-composition reference resolves to the wrong artifact type"
                )
            self.archive.load_score_grid_composition_payload(
                reference,
                verification_session=self.verification_session,
            )
            return
        from football_data_platform.storage.training import TrainingArtifactStore

        store = TrainingArtifactStore(self.archive.layout)
        store._verify_training_reference(
            reference,
            verification_session=self.verification_session,
        )

    def _resolve_prediction_reference(self, reference: str) -> None:
        digest = reference.removeprefix("prediction:")
        path = self.archive.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ArchiveConflictError("prediction artifact must be an object")
        identity = dict(payload)
        stored_id = identity.pop("id", None)
        if (
            stored_id != reference
            or hashlib.sha256(_canonical_json(identity)).hexdigest() != digest
        ):
            raise ArchiveConflictError("prediction artifact identity failed")
        manifest = self.archive._load_artifact_manifest_for_output_ref(
            reference,
            verification_session=self.verification_session,
        )
        composition_ref = payload.get("composition_artifact_ref")
        if (
            manifest.artifact_type != "prediction"
            or manifest.status != "succeeded"
            or manifest.payload != payload
            or not isinstance(composition_ref, str)
            or composition_ref not in manifest.input_refs
        ):
            raise ArchiveConflictError("prediction artifact does not match its manifest")
        self.archive.load_score_grid_composition_payload(
            composition_ref,
            verification_session=self.verification_session,
        )

    def _resolve_sample_reference(self, reference: str) -> None:
        from football_data_platform.storage.training import TrainingArtifactStore

        store = TrainingArtifactStore(self.archive.layout)
        root = self.archive.layout.derived / "training-datasets"
        for path in root.rglob("*.json"):
            if not _REFERENCE_DIGEST.fullmatch(path.stem):
                continue
            try:
                dataset = store.load_dataset(f"training-dataset:{path.stem}")
            except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
                continue
            if any(sample.sample_id == reference for sample in dataset.samples):
                return
        raise ArchiveConflictError(f"training sample reference does not exist: {reference}")

    def _resolve_canonical_reference(self, reference: str, namespace: str) -> None:
        if namespace == "official-lineup-contract":
            self.archive._replay_official_lineup_contract(reference)
            return
        if namespace == "match-report-contract":
            verify_match_report_contract(
                reference,
                archive=RawArchive(self.archive.layout),
                canonical=CanonicalStore(self.archive.layout.canonical / "platform.sqlite3"),
            )
            return
        if namespace == "fact" and reference.startswith("fact:match_results_90:"):
            load_verified_match_result(
                reference,
                archive=RawArchive(self.archive.layout),
                canonical=CanonicalStore(self.archive.layout.canonical / "platform.sqlite3"),
                verification_session=self.verification_session,
            )
            return
        if namespace == "fact" and reference.startswith("fact:team_match_observations:"):
            load_verified_team_observation(
                reference,
                archive=RawArchive(self.archive.layout),
                canonical=CanonicalStore(self.archive.layout.canonical / "platform.sqlite3"),
                verification_session=self.verification_session,
            )
            return
        path = self.archive.layout.canonical / "platform.sqlite3"
        if not path.is_file():
            raise ArchiveConflictError("canonical store is unavailable")
        with sqlite3.connect(path) as connection:
            if namespace in _ENTITY_REFERENCE_NAMESPACES:
                row = connection.execute(
                    "SELECT 1 FROM entities WHERE entity_id = ? AND entity_type = ? LIMIT 1",
                    (reference, namespace),
                ).fetchone()
            elif namespace == "collection-attempt":
                row = connection.execute(
                    "SELECT 1 FROM collection_attempts WHERE collection_attempt_id = ? LIMIT 1",
                    (reference,),
                ).fetchone()
            else:
                row = self._canonical_record_reference(connection, reference)
        if row is None:
            raise ArchiveConflictError(f"canonical reference does not exist: {reference}")

    @staticmethod
    def _canonical_record_reference(
        connection: sqlite3.Connection, reference: str
    ) -> tuple[Any, ...] | None:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for (table_name,) in tables:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
            if "record_id" not in columns:
                continue
            row = connection.execute(
                f"SELECT 1 FROM {table_name} WHERE record_id = ? LIMIT 1", (reference,)
            ).fetchone()
            if row is not None:
                return row
        return None


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


def _snapshot_source_ref_value(reference: RawAssetId | str) -> str:
    value = reference.value if isinstance(reference, RawAssetId) else reference
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("snapshot source input_refs must contain typed references")
    return value


def _manifest_digest(value: str, prefix: str) -> str:
    marker = f"{prefix}:"
    if not isinstance(value, str) or not value.startswith(marker):
        raise ValueError(f"invalid {prefix} manifest ID")
    digest = value.removeprefix(marker)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid {prefix} manifest ID")
    return digest


def _validate_score_grid_composition_ref(value: str) -> None:
    marker = "score-grid-composition:"
    if not isinstance(value, str) or not value.startswith(marker):
        raise ValueError("invalid score-grid-composition reference")
    digest = value.removeprefix(marker)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid score-grid-composition reference digest")


def _artifact_output_semantic_identity(manifest: DerivedArtifactManifest) -> bytes:
    return _canonical_json(
        {
            "manifest_version": manifest.manifest_version,
            "schema_version": manifest.schema_version,
            "artifact_type": manifest.artifact_type,
            "generated_at": _timestamp(manifest.generated_at),
            "started_at": _timestamp(manifest.started_at),
            "ended_at": _timestamp(manifest.ended_at),
            "transform_version": manifest.transform_version,
            "code_version": manifest.code_version,
            "input_refs": list(manifest.input_refs),
            "output_refs": list(manifest.output_refs),
            "status": manifest.status,
            "error": manifest.error,
            "quality": manifest.quality,
            "parameters": (
                manifest.payload.get("parameters")
                if isinstance(manifest.payload, Mapping)
                else None
            ),
            "payload": manifest.payload,
        }
    )


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
