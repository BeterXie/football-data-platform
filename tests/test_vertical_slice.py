from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from football_data_platform.pipelines.vertical_slice import run_offline_vertical_slice

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


def test_bundled_golden_files_match_parser_regression_fixtures() -> None:
    for name in (
        "fbref_premier_league_schedule.html",
        "fbref_premier_league_match_report.html",
        "premier_league_official_lineups.json",
    ):
        assert (ROOT / "examples/golden" / name).read_bytes() == (
            ROOT / "tests/fixtures" / name
        ).read_bytes()


def test_offline_vertical_slice_replays_idempotently_end_to_end(tmp_path: Path) -> None:
    arguments = {
        "data_root": tmp_path / "data",
        "registry_path": ROOT / "config/competitions.toml",
        "schedule_file": ROOT / "tests/fixtures/fbref_premier_league_schedule.html",
        "first_match_report_file": ROOT / "tests/fixtures/fbref_premier_league_match_report.html",
        "lineups_file": ROOT / "tests/fixtures/premier_league_official_lineups.json",
        "observed_at": OBSERVED_AT,
        "first_report_known_at": datetime(2025, 8, 15, 22, 0, tzinfo=UTC),
        "second_result_known_at": datetime(2025, 8, 23, 16, 0, tzinfo=UTC),
        "profile_minimum_minutes": 60,
    }

    first = run_offline_vertical_slice(**arguments)
    second = run_offline_vertical_slice(**arguments)

    assert first == second
    assert first.canonical_counts == {
        "competitions": 1,
        "seasons": 1,
        "teams": 2,
        "players": 25,
        "matches": 2,
    }
    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert summary["snapshots"][0]["quality_status"] == "ready"
    assert summary["snapshots"][0]["capture_mode"] == "reconstructed"
    assert summary["snapshots"][1]["quality_status"] == "preview"
    assert not summary["evaluation"]["market_benchmark"]["available"]
    assert summary["evaluation"]["capture_mode"] == "reconstructed"
    assert first.report_path.read_text(encoding="utf-8").startswith(
        "# Premier League 2025-26 Vertical Slice"
    )
