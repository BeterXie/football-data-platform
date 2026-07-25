"""Static reporting for real fixture/result backfill coverage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_results_backfill_report(summary: Mapping[str, Any]) -> str:
    coverage = summary.get("coverage")
    if not isinstance(coverage, Mapping):
        coverage = summary
    status_counts = coverage.get("fixture_status_counts", {})
    if not isinstance(status_counts, Mapping):
        status_counts = {}
    fixture_rows = coverage.get("fixture_coverage", ())
    if not isinstance(fixture_rows, (tuple, list)):
        fixture_rows = ()
    fixture_lines = []
    for item in fixture_rows:
        if not isinstance(item, Mapping):
            continue
        fixture_lines.append(
            f"- `{item.get('fixture_id', '')}`: `{item.get('status', 'unknown')}`; "
            f"attempts=`{item.get('attempt_count', 0)}`; "
            f"last=`{item.get('last_attempt_at') or 'none'}`; "
            f"diagnostic=`{item.get('last_diagnostic_code') or 'none'}`; "
            f"production-contract=`{bool(item.get('production_contract_satisfied', False))}`; "
            f"contract=`{item.get('report_contract_id') or 'none'}`"
        )
    if not fixture_lines:
        fixture_lines.append("- none")
    return "\n".join(
        (
            "# Premier League 2025-26 Results Backfill",
            "",
            f"- Source: `{summary['source']}`",
            f"- Raw asset: `{summary['raw_asset_id']}`",
            f"- Teams: `{summary['actual_teams']}` / `{summary['expected_teams']}`",
            f"- Matches: `{summary['actual_matches']}` / `{summary['expected_matches']}`",
            f"- 90-minute result facts: `{summary['result_facts']}`",
            f"- Schedule gate: `{'pass' if summary['schedule_complete'] else 'fail'}`",
            "- Full FBref statistics gate: "
            f"`{'pass' if summary['full_collection_gate'] else 'open'}`",
            "- Fixture attempt status counts: "
            + ", ".join(
                f"`{name}={int(status_counts.get(name, 0))}`"
                for name in ("missing", "pending", "blocked", "failed", "succeeded")
            ),
            "- Attempt identity mismatches: "
            f"`{len(coverage.get('attempt_identity_mismatch_fixture_ids', ()))}`",
            f"- Invalid report contracts: `{len(coverage.get('report_contract_diagnostics', ()))}`",
            "",
            "## Fixture Collection Attempts",
            "",
            *fixture_lines,
            "",
            "## Temporal Status",
            "",
            f"- Result known-at policy: `{summary['result_known_at_policy']}`",
            "- These are reconstructed historical results, not captured prematch snapshots.",
            "",
            "## Source Boundary",
            "",
            "- Football-Data is used only for fixtures and 90-minute results.",
            "- CSV odds columns are ignored and are not football-model inputs or market "
            "benchmarks.",
            "- FBref remains the primary team/player process-statistics source.",
            "",
        )
    )
