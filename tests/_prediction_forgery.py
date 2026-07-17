from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any

from football_data_platform.models.score_grid import DixonColesGrid
from football_data_platform.storage.derived import DerivedArchive, DerivedArtifactManifest
from football_data_platform.storage.layout import DataLayout


def forge_persisted_prediction(
    layout: DataLayout,
    original_ref: str,
    mutate: Callable[[dict[str, Any]], None],
) -> str:
    """Rewrite one semantic forgery with internally consistent IDs and manifests."""

    derived = DerivedArchive(layout)
    original_digest = original_ref.removeprefix("prediction:")
    original_path = layout.derived / "predictions" / original_digest[:2] / f"{original_digest}.json"
    payload = deepcopy(json.loads(original_path.read_text(encoding="utf-8")))
    original_composition_ref = payload["composition_artifact_ref"]
    prediction_manifest = derived._load_artifact_manifest_for_output_ref(original_ref)
    composition_manifest = derived._load_artifact_manifest_for_output_ref(original_composition_ref)

    mutate(payload)
    _rebuild_prediction_grid(payload)
    forged_generated_at = datetime.fromisoformat(
        str(payload["generated_at"]).replace("Z", "+00:00")
    )
    composition_payload = _composition_payload(payload)
    composition_ref = (
        "score-grid-composition:" + hashlib.sha256(_canonical_json(composition_payload)).hexdigest()
    )
    payload["composition_artifact_ref"] = composition_ref
    identity = dict(payload)
    identity.pop("id", None)
    prediction_ref = "prediction:" + hashlib.sha256(_canonical_json(identity)).hexdigest()
    payload["id"] = prediction_ref

    derived._write_json(
        derived.score_grid_composition_path(composition_ref),
        {"id": composition_ref, **composition_payload},
    )
    derived.write_artifact_manifest(
        DerivedArtifactManifest.create(
            artifact_type="score-grid-composition",
            schema_version=composition_manifest.schema_version,
            payload=composition_payload,
            generated_at=forged_generated_at,
            started_at=forged_generated_at,
            ended_at=forged_generated_at,
            transform_version=composition_manifest.transform_version,
            code_version=composition_manifest.code_version,
            input_refs=tuple(payload["input_refs"]),
            output_refs=(composition_ref,),
            status="succeeded",
            quality=composition_manifest.quality,
        )
    )

    prediction_digest = prediction_ref.removeprefix("prediction:")
    derived._write_json(
        layout.derived / "predictions" / prediction_digest[:2] / f"{prediction_digest}.json",
        payload,
    )
    derived.write_artifact_manifest(
        DerivedArtifactManifest.create(
            artifact_type="prediction",
            schema_version=prediction_manifest.schema_version,
            payload=payload,
            generated_at=forged_generated_at,
            started_at=forged_generated_at,
            ended_at=forged_generated_at,
            transform_version=prediction_manifest.transform_version,
            code_version=prediction_manifest.code_version,
            input_refs=tuple(
                composition_ref if item == original_composition_ref else item
                for item in prediction_manifest.input_refs
            ),
            output_refs=(prediction_ref,),
            status="succeeded",
            quality=prediction_manifest.quality,
        )
    )
    return prediction_ref


def _rebuild_prediction_grid(payload: dict[str, Any]) -> None:
    contributions = sorted(
        payload["contribution_multipliers"], key=lambda item: item["contribution_key"]
    )
    for contribution in contributions:
        calibration = contribution["calibration"]
        delta = calibration["source_value"] - calibration["reference_value"]
        minimum = calibration["minimum_log_multiplier"]
        maximum = calibration["maximum_log_multiplier"]
        home_log = max(
            minimum,
            min(maximum, calibration["lambda_home_coefficient"] * delta),
        )
        away_log = max(
            minimum,
            min(maximum, calibration["lambda_away_coefficient"] * delta),
        )
        contribution["lambda_home_multiplier"] = math.exp(home_log)
        contribution["lambda_away_multiplier"] = math.exp(away_log)

    payload["contribution_multipliers"] = contributions
    payload["contribution_keys"] = [item["contribution_key"] for item in contributions]
    payload["calibration_versions"] = sorted({item["version"] for item in contributions})
    lambda_home = payload["baseline_lambda_home"] * math.prod(
        item["lambda_home_multiplier"] for item in contributions
    )
    lambda_away = payload["baseline_lambda_away"] * math.prod(
        item["lambda_away_multiplier"] for item in contributions
    )
    grid = DixonColesGrid(
        lambda_home,
        lambda_away,
        rho=payload["rho"],
        max_goals=payload["max_goals"],
    )
    payload["lambda_home"] = grid.lambda_home
    payload["lambda_away"] = grid.lambda_away
    payload["score_cells"] = grid.score_cells()
    payload["normalization_residual"] = grid.normalization_residual
    payload["markets"] = [
        _market("result_90", None, grid.result_probabilities()),
        _market("home_handicap_3way", "-1", grid.handicap_probabilities(-1)),
        _market("total_goals", "0-6,7+", grid.total_goals_probabilities()),
    ]


def _market(
    market_type: str,
    parameter: str | None,
    probabilities: dict[str, float],
) -> dict[str, Any]:
    return {
        "market_type": market_type,
        "parameter": parameter,
        "outcomes": [
            {"outcome": outcome, "probability": probability}
            for outcome, probability in probabilities.items()
        ],
    }


def _composition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "match_id",
        "snapshot_id",
        "model_run_id",
        "model_version",
        "generated_at",
        "snapshot_as_of",
        "baseline_lambda_home",
        "baseline_lambda_away",
        "contribution_keys",
        "contribution_multipliers",
        "composition_version",
        "calibration_versions",
        "lambda_home",
        "lambda_away",
        "rho",
        "max_goals",
        "normalization_residual",
        "score_cells",
        "input_refs",
    )
    result = {name: deepcopy(payload[name]) for name in fields}
    result.update(
        {
            "schema_version": 2,
            "artifact_type": "score-grid-composition",
            "grid": {
                "lambda_home": result["lambda_home"],
                "lambda_away": result["lambda_away"],
                "rho": result["rho"],
                "max_goals": result["max_goals"],
                "normalization_residual": result["normalization_residual"],
                "score_cells": deepcopy(result["score_cells"]),
            },
        }
    )
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
