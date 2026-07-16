"""Render a compact, lineage-oriented vertical-slice report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def render_vertical_slice_report(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    prediction = summary["prediction"]
    evaluation = summary["evaluation"]
    snapshots = summary["snapshots"]
    lines = [
        "# Premier League 2025-26 Vertical Slice",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Observed at: `{summary['observed_at']}`",
        f"- Canonical matches in golden slice: `{summary['canonical_counts']['matches']}`",
        "",
        "## Season Gate",
        "",
        f"- Expected season: `{coverage['expected_teams']}` teams / "
        f"`{coverage['expected_matches']}` matches",
        f"- Current evidence: `{coverage['actual_teams']}` teams / "
        f"`{coverage['actual_matches']}` matches",
        f"- Full-season gate: `{'pass' if coverage['complete'] else 'not-yet-pass'}`",
        f"- Missing match-report attempts: `{len(coverage['missing_collection_attempts'])}`",
        f"- Blocking diagnostics: `{', '.join(coverage['blocking_diagnostics']) or 'none'}`",
        "",
        "## Prematch Snapshots",
        "",
    ]
    for snapshot in snapshots:
        lines.append(
            f"- `{snapshot['type']}` / `{snapshot['capture_mode']}` / "
            f"`{snapshot['quality_status']}`: `{snapshot['id']}`"
        )
        if snapshot["missing_fields"]:
            lines.append(f"  Missing fields: `{', '.join(snapshot['missing_fields'])}`")
    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- Prediction: `{prediction['id']}`",
            f"- Model: `{prediction['model_version']}`",
            f"- Expected goals: `{prediction['lambda_home']:.4f}` / "
            f"`{prediction['lambda_away']:.4f}`",
            f"- Dixon-Coles rho: `{prediction['rho']:.4f}`",
            f"- Score-grid residual: `{prediction['normalization_residual']:.3e}`",
            "",
            "## Evaluation",
            "",
            f"- Cohort: `{evaluation['capture_mode']}`",
            f"- Actual result: `{evaluation['actual_score']}`",
            f"- Result Brier: `{evaluation['result_brier']:.6f}`",
            f"- Result LogLoss: `{evaluation['result_log_loss']:.6f}`",
            f"- Exact-score LogLoss: `{evaluation['score_log_loss']:.6f}`",
            "- Market benchmark: `unavailable (no real timestamped market snapshot)`",
            "",
            "## Training Readiness",
            "",
        ]
    )
    for match_id, assessment in summary["lifecycle"].items():
        lines.append(f"- `{match_id}`: `{assessment['state']}`")
        for qualification in assessment["qualifications"]:
            status = "pass" if qualification["passed"] else "fail"
            reasons = ", ".join(qualification["reason_codes"]) or "none"
            lines.append(f"  `{qualification['qualification']}`: `{status}`; reasons: `{reasons}`")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- All historical prematch artifacts are reconstructed, not captured.",
            "- The golden slice proves contracts and replay; it is not full-season coverage.",
            "- No synthetic odds, in-play model, automatic bet, or real-money action is present.",
            "",
        ]
    )
    return "\n".join(lines)


def write_static_report(path: Path, content: str) -> Path:
    payload = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return path
    path.write_bytes(payload)
    return path
