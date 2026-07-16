"""Content-addressed, immutable archive for raw source evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_data_platform.domain.ids import RawAssetId
from football_data_platform.domain.models import RawAsset
from football_data_platform.storage.layout import DataLayout

MANIFEST_VERSION = 1


class ChecksumMismatchError(ValueError):
    """Raised when bytes do not match their expected SHA-256 checksum."""


class ArchiveConflictError(RuntimeError):
    """Raised when an immutable archive path contains unexpected data."""


class RawArchive:
    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout.ensure()

    def archive(
        self,
        content: bytes,
        *,
        source: str,
        source_id: str,
        url: str,
        observed_at: datetime,
        target_event_time: datetime | None,
        collector_version: str,
        media_type: str,
        expected_checksum: str | None = None,
    ) -> RawAsset:
        """Archive bytes and return their immutable observation manifest."""

        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        checksum = hashlib.sha256(content).hexdigest()
        if expected_checksum is not None and checksum != expected_checksum:
            raise ChecksumMismatchError(
                f"expected checksum {expected_checksum}, calculated {checksum}"
            )

        identity = {
            "manifest_version": MANIFEST_VERSION,
            "source": source,
            "source_id": source_id,
            "url": url,
            "observed_at": _format_datetime(observed_at),
            "target_event_time": _format_datetime(target_event_time),
            "checksum": checksum,
            "collector_version": collector_version,
            "media_type": media_type,
            "size_bytes": len(content),
        }
        asset_digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        asset = RawAsset(
            id=RawAssetId(f"raw-asset:{asset_digest}"),
            source=source,
            source_id=source_id,
            url=url,
            observed_at=observed_at,
            target_event_time=target_event_time,
            checksum=checksum,
            collector_version=collector_version,
            media_type=media_type,
            size_bytes=len(content),
        )

        self._write_content(asset, content)
        self._write_manifest(asset, identity)
        return asset

    def load(self, asset_id: RawAssetId) -> RawAsset:
        """Load a manifest and validate its content-derived identity."""

        path = self.layout.raw_manifest_path(asset_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ArchiveConflictError(f"manifest is not an object: {path}")
        stored_id = payload.pop("id", None)
        if stored_id != asset_id.value:
            raise ArchiveConflictError(f"manifest ID does not match path: {path}")
        calculated_id = RawAssetId(
            f"raw-asset:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        )
        if calculated_id != asset_id:
            raise ArchiveConflictError(f"manifest identity checksum failed: {path}")
        return _asset_from_manifest(asset_id, payload)

    def read(self, asset: RawAsset | RawAssetId) -> bytes:
        """Read raw bytes, rejecting missing, truncated, or modified content."""

        asset_id = asset if isinstance(asset, RawAssetId) else asset.id
        manifest = self.load(asset_id)
        if isinstance(asset, RawAsset) and manifest != asset:
            raise ArchiveConflictError(f"asset does not match stored manifest: {asset.id}")
        content_path = self.layout.raw_object_path(manifest.checksum)
        content = content_path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != manifest.checksum:
            raise ChecksumMismatchError(f"content checksum mismatch for {manifest.id}: {checksum}")
        if len(content) != manifest.size_bytes:
            raise ChecksumMismatchError(f"content size mismatch for {manifest.id}")
        return content

    def verify(self, asset: RawAsset | RawAssetId) -> None:
        """Validate both the manifest identity and referenced content checksum."""

        manifest = self.load(asset) if isinstance(asset, RawAssetId) else self.load(asset.id)
        self.read(manifest)

    def _write_content(self, asset: RawAsset, content: bytes) -> None:
        path = self.layout.raw_object_path(asset.checksum)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing_checksum != asset.checksum:
                raise ArchiveConflictError(f"unexpected content at immutable path: {path}")
            return
        _write_exclusive(path, content)

    def _write_manifest(self, asset: RawAsset, identity: Mapping[str, Any]) -> None:
        path = self.layout.raw_manifest_path(asset.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"id": asset.id.value, **identity}
        encoded = (
            json.dumps(
                manifest,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if path.exists():
            if path.read_bytes() != encoded:
                raise ArchiveConflictError(f"unexpected manifest at immutable path: {path}")
            return
        _write_exclusive(path, encoded)


def _write_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ArchiveConflictError(
                f"concurrent write conflict at immutable path: {path}"
            ) from None


def _asset_from_manifest(asset_id: RawAssetId, payload: Mapping[str, Any]) -> RawAsset:
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ArchiveConflictError("unsupported raw manifest version")
    return RawAsset(
        id=asset_id,
        source=str(payload["source"]),
        source_id=str(payload["source_id"]),
        url=str(payload["url"]),
        observed_at=_parse_datetime(payload["observed_at"], "observed_at"),
        target_event_time=_parse_datetime(
            payload["target_event_time"], "target_event_time", optional=True
        ),
        checksum=str(payload["checksum"]),
        collector_version=str(payload["collector_version"]),
        media_type=str(payload["media_type"]),
        size_bytes=int(payload["size_bytes"]),
    )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("manifest datetimes must be timezone-aware UTC")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("manifest datetimes must be timezone-aware UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(
    value: Any,
    field_name: str,
    *,
    optional: bool = False,
) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ArchiveConflictError(f"manifest {field_name} must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArchiveConflictError(f"invalid manifest {field_name}") from error
