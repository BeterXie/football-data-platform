from __future__ import annotations

import json
from pathlib import Path

from football_data_platform.cli import main

ROOT = Path(__file__).parents[1]


def test_cli_runs_golden_slice_and_emits_artifact_pointers(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run-golden",
            "--data-root",
            str(tmp_path / "data"),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--fixtures-dir",
            str(ROOT / "examples/golden"),
            "--observed-at",
            "2026-07-16T08:00:00Z",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert Path(payload["summary"]).is_file()
    assert Path(payload["report"]).is_file()


def test_cli_schedule_gate_returns_nonzero_for_partial_golden_fixture(
    capsys,
) -> None:
    exit_code = main(
        [
            "validate-schedule",
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(ROOT / "examples/golden/fbref_premier_league_schedule.html"),
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["actual_matches"] == 2
    assert payload["expected_matches"] == 380


def test_cli_results_backfill_reports_partial_schedule_gate(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "backfill-results",
            "--data-root",
            str(tmp_path / "data"),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(ROOT / "tests/fixtures/football_data_e0_sample.csv"),
            "--observed-at",
            "2026-07-16T10:00:00Z",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "gate_failed"
    assert payload["result_facts"] == 2
    assert not payload["schedule_complete"]
    assert Path(payload["report"]).is_file()
