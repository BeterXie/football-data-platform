"""Content-addressed training qualification contracts.

The contract records a validator result, but does not trust it.  Storage
replays the bound snapshot and canonical facts before admitting the artifact
to a formal training dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from football_data_platform.domain.lifecycle import Qualification, QualificationResult
from football_data_platform.domain.models import require_utc
from football_data_platform.domain.snapshots import CaptureMode, PreMatchSnapshot

TRAINING_QUALIFICATION_SCHEMA_VERSION = 1
TRAINING_QUALIFICATION_CONTRACT_VERSION = "training-qualification/1"
TRAINING_QUALIFICATION_PREFIX = "training-qualification:"
CURRENT_SCORE_FEATURE_PROJECTION_VERSION = "snapshot-score-features/1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TrainingQualification:
    qualification_id: str
    schema_version: int
    contract_version: str
    match_id: str
    match_version: int
    qualification: Qualification
    ruleset_version: str
    evaluated_at: datetime
    passed: bool
    reason_codes: tuple[str, ...]
    snapshot_ref: str | None
    result_ref: str | None
    fact_refs: tuple[str, ...]
    capture_mode: CaptureMode | None

    @classmethod
    def create(
        cls,
        *,
        match_id: str,
        match_version: int,
        result: QualificationResult,
        snapshot_ref: str | None,
        result_ref: str | None,
        fact_refs: tuple[str, ...],
        capture_mode: CaptureMode | None,
    ) -> TrainingQualification:
        fields = _qualification_fields(
            match_id=match_id,
            match_version=match_version,
            qualification=result.qualification,
            ruleset_version=result.ruleset_version,
            evaluated_at=result.evaluated_at,
            passed=result.passed,
            reason_codes=result.reason_codes,
            snapshot_ref=snapshot_ref,
            result_ref=result_ref,
            fact_refs=fact_refs,
            capture_mode=capture_mode,
        )
        identity = {
            "schema_version": TRAINING_QUALIFICATION_SCHEMA_VERSION,
            "contract_version": TRAINING_QUALIFICATION_CONTRACT_VERSION,
            **_serialized_fields(fields),
        }
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        return cls(
            qualification_id=TRAINING_QUALIFICATION_PREFIX + digest,
            schema_version=TRAINING_QUALIFICATION_SCHEMA_VERSION,
            contract_version=TRAINING_QUALIFICATION_CONTRACT_VERSION,
            **fields,
        )

    @property
    def input_refs(self) -> tuple[str, ...]:
        refs = set(self.fact_refs)
        if self.snapshot_ref is not None:
            refs.add(self.snapshot_ref)
        if self.result_ref is not None:
            refs.add(self.result_ref)
        return tuple(sorted(refs))

    def to_payload(self) -> dict[str, Any]:
        verify_training_qualification(self)
        return {
            "id": self.qualification_id,
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            **_serialized_fields(_fields_from(self)),
        }


def verify_training_qualification(artifact: TrainingQualification) -> None:
    if not isinstance(artifact, TrainingQualification):
        raise TypeError("artifact must be a TrainingQualification")
    if artifact.schema_version != TRAINING_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported training qualification schema_version")
    if artifact.contract_version != TRAINING_QUALIFICATION_CONTRACT_VERSION:
        raise ValueError("unsupported training qualification contract_version")
    fields = _fields_from(artifact)
    identity = {
        "schema_version": artifact.schema_version,
        "contract_version": artifact.contract_version,
        **_serialized_fields(fields),
    }
    expected = TRAINING_QUALIFICATION_PREFIX + hashlib.sha256(_canonical_json(identity)).hexdigest()
    if artifact.qualification_id != expected:
        raise ValueError("training qualification identity does not match canonical content")


def parse_training_qualification_payload(payload: Any) -> TrainingQualification:
    if not isinstance(payload, dict):
        raise ValueError("training qualification must be an object")
    try:
        raw_reasons = payload["reason_codes"]
        raw_facts = payload["fact_refs"]
        if not isinstance(raw_reasons, list) or not isinstance(raw_facts, list):
            raise ValueError("training qualification refs and reasons must be lists")
        artifact = TrainingQualification(
            qualification_id=_text(payload["id"], "id"),
            schema_version=_strict_int(payload["schema_version"], "schema_version"),
            contract_version=_text(payload["contract_version"], "contract_version"),
            match_id=_text(payload["match_id"], "match_id"),
            match_version=_strict_int(payload["match_version"], "match_version"),
            qualification=Qualification(_text(payload["qualification"], "qualification")),
            ruleset_version=_text(payload["ruleset_version"], "ruleset_version"),
            evaluated_at=_datetime(payload["evaluated_at"], "evaluated_at"),
            passed=_strict_bool(payload["passed"], "passed"),
            reason_codes=tuple(_text(item, "reason_code") for item in raw_reasons),
            snapshot_ref=_optional_text(payload.get("snapshot_ref"), "snapshot_ref"),
            result_ref=_optional_text(payload.get("result_ref"), "result_ref"),
            fact_refs=tuple(_text(item, "fact_ref") for item in raw_facts),
            capture_mode=(
                None
                if payload.get("capture_mode") is None
                else CaptureMode(_text(payload["capture_mode"], "capture_mode"))
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid training qualification: {error}") from error
    verify_training_qualification(artifact)
    if artifact.to_payload() != payload:
        raise ValueError("training qualification payload is not canonical")
    return artifact


def score_feature_payload(snapshot: PreMatchSnapshot | None) -> dict[str, Any]:
    """Return the deterministic score-feature projection of a verified snapshot."""

    if snapshot is None:
        return {
            "projection_version": CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
            "snapshot_ref": None,
            "features": [],
        }
    return {
        "projection_version": CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        "snapshot_ref": snapshot.id.value,
        "match_id": snapshot.match_id.value,
        "match_version": snapshot.match_version,
        "as_of": _timestamp(snapshot.as_of),
        "feature_spec_version": snapshot.feature_spec_version,
        "features": [
            {
                "name": feature.name,
                "entity_id": feature.entity_id,
                "contribution_key": feature.contribution_key,
                "known_at": _timestamp(feature.known_at),
                "source_ref": feature.source_ref,
                "value": feature.value,
            }
            for feature in snapshot.features
        ],
    }


def _fields_from(artifact: TrainingQualification) -> dict[str, Any]:
    return _qualification_fields(
        match_id=artifact.match_id,
        match_version=artifact.match_version,
        qualification=artifact.qualification,
        ruleset_version=artifact.ruleset_version,
        evaluated_at=artifact.evaluated_at,
        passed=artifact.passed,
        reason_codes=artifact.reason_codes,
        snapshot_ref=artifact.snapshot_ref,
        result_ref=artifact.result_ref,
        fact_refs=artifact.fact_refs,
        capture_mode=artifact.capture_mode,
    )


def _qualification_fields(**values: Any) -> dict[str, Any]:
    match_id = _text(values["match_id"], "match_id")
    match_version = _strict_int(values["match_version"], "match_version")
    if match_version < 1:
        raise ValueError("match_version must be positive")
    qualification = values["qualification"]
    if not isinstance(qualification, Qualification):
        raise TypeError("qualification must be a Qualification")
    ruleset_version = _text(values["ruleset_version"], "ruleset_version")
    evaluated_at = values["evaluated_at"]
    require_utc(evaluated_at, "evaluated_at")
    passed = _strict_bool(values["passed"], "passed")
    reason_codes = tuple(sorted(set(values["reason_codes"])))
    if any(
        not isinstance(reason, str) or not reason or reason.strip() != reason
        for reason in reason_codes
    ):
        raise ValueError("reason_codes must contain canonical non-empty text")
    if passed == bool(reason_codes):
        raise ValueError("passed must be true exactly when reason_codes is empty")
    snapshot_ref = values["snapshot_ref"]
    if snapshot_ref is not None:
        _digest_ref(snapshot_ref, "snapshot:", "snapshot_ref")
    result_ref = values["result_ref"]
    if result_ref is not None and not result_ref.startswith("fact:match_results_90:"):
        raise ValueError("result_ref must identify a typed 90-minute result")
    fact_refs = tuple(sorted(set(values["fact_refs"])))
    if any(not isinstance(ref, str) or not ref or ref.strip() != ref for ref in fact_refs):
        raise ValueError("fact_refs must contain canonical non-empty refs")
    capture_mode = values["capture_mode"]
    if capture_mode is not None and not isinstance(capture_mode, CaptureMode):
        raise TypeError("capture_mode must be a CaptureMode when present")
    return {
        "match_id": match_id,
        "match_version": match_version,
        "qualification": qualification,
        "ruleset_version": ruleset_version,
        "evaluated_at": evaluated_at,
        "passed": passed,
        "reason_codes": reason_codes,
        "snapshot_ref": snapshot_ref,
        "result_ref": result_ref,
        "fact_refs": fact_refs,
        "capture_mode": capture_mode,
    }


def _serialized_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": fields["match_id"],
        "match_version": fields["match_version"],
        "qualification": fields["qualification"].value,
        "ruleset_version": fields["ruleset_version"],
        "evaluated_at": _timestamp(fields["evaluated_at"]),
        "passed": fields["passed"],
        "reason_codes": list(fields["reason_codes"]),
        "snapshot_ref": fields["snapshot_ref"],
        "result_ref": fields["result_ref"],
        "fact_refs": list(fields["fact_refs"]),
        "capture_mode": (None if fields["capture_mode"] is None else fields["capture_mode"].value),
    }


def _digest_ref(value: str, prefix: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{field_name} must start with {prefix!r}")
    if not _DIGEST.fullmatch(value.removeprefix(prefix)):
        raise ValueError(f"{field_name} must contain a SHA-256 digest")


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be non-empty canonical text")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    return None if value is None else _text(value, field_name)


def _strict_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_utc(parsed, field_name)
    return parsed.astimezone(UTC)


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
