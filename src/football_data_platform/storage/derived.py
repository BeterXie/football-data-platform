"""Immutable storage for rebuildable derived JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_data_platform.domain.predictions import ScorePrediction, prediction_payload
from football_data_platform.domain.snapshots import PreMatchSnapshot, snapshot_payload
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError


class DerivedArchive:
    def __init__(self, layout: DataLayout) -> None:
        self.layout = layout.ensure()

    def write_snapshot(self, snapshot: PreMatchSnapshot) -> Path:
        path = self.snapshot_path(snapshot)
        payload = (
            json.dumps(
                snapshot_payload(snapshot),
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
            return path
        try:
            with path.open("xb") as destination:
                destination.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ArchiveConflictError(f"derived snapshot conflicts at {path}") from None
        return path

    def load_snapshot_payload(self, snapshot: PreMatchSnapshot) -> dict[str, Any]:
        return json.loads(self.snapshot_path(snapshot).read_text(encoding="utf-8"))

    def write_prediction(self, prediction: ScorePrediction) -> Path:
        digest = prediction.id.value.removeprefix("prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        return self._write_json(path, prediction_payload(prediction))

    def load_prediction_payload(self, prediction: ScorePrediction) -> dict[str, Any]:
        digest = prediction.id.value.removeprefix("prediction:")
        path = self.layout.derived / "predictions" / digest[:2] / f"{digest}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot_path(self, snapshot: PreMatchSnapshot) -> Path:
        digest = snapshot.id.value.removeprefix("snapshot:")
        return self.layout.derived / "snapshots" / digest[:2] / f"{digest}.json"

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
