"""Local directory layout for the platform's three data layers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from football_data_platform.domain.ids import RawAssetId

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DataLayout:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def canonical(self) -> Path:
        return self.root / "canonical"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def paper_ledger(self) -> Path:
        """Derived append-only paper betting ledger root."""

        return self.derived / "paper-ledger"

    @property
    def paper_betting_ledger(self) -> Path:
        """Compatibility alias for the paper ledger root."""

        return self.paper_ledger

    def ensure(self) -> DataLayout:
        for layer in (self.raw, self.canonical, self.derived):
            layer.mkdir(parents=True, exist_ok=True)
        return self

    def raw_object_path(self, checksum: str) -> Path:
        _validate_digest(checksum, "checksum")
        return self.raw / "objects" / "sha256" / checksum[:2] / checksum

    def raw_manifest_path(self, asset_id: RawAssetId) -> Path:
        digest = asset_id.value.removeprefix("raw-asset:")
        _validate_digest(digest, "raw asset digest")
        return self.raw / "manifests" / "sha256" / digest[:2] / f"{digest}.json"


def _validate_digest(value: str, name: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
