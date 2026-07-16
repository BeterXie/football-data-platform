"""Static reporting for real fixture/result backfill coverage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_results_backfill_report(summary: Mapping[str, Any]) -> str:
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
