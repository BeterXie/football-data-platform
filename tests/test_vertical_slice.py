from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.pipelines.vertical_slice import run_offline_vertical_slice
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError

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
    run_manifest = DerivedArchive(DataLayout(arguments["data_root"])).load_run_manifest(
        first.run_id
    )
    assert run_manifest.status == "succeeded"
    assert run_manifest.checkpoint == "static-report-written"
    assert summary["run_id"] == run_manifest.run_id
    assert summary["snapshots"][0]["quality_status"] == "ready"
    assert summary["snapshots"][0]["capture_mode"] == "reconstructed"
    assert summary["snapshots"][1]["quality_status"] == "preview"
    assert not summary["evaluation"]["market_benchmark"]["available"]
    assert summary["evaluation"]["capture_mode"] == "reconstructed"
    assert first.report_path.read_text(encoding="utf-8").startswith(
        "# Premier League 2025-26 Vertical Slice"
    )

    first.summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArchiveConflictError, match="summary conflicts"):
        run_offline_vertical_slice(**arguments)


def test_offline_vertical_slice_persists_a_failed_run_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with pytest.raises(FileNotFoundError):
        run_offline_vertical_slice(
            data_root=data_root,
            registry_path=ROOT / "config/competitions.toml",
            schedule_file=tmp_path / "missing-schedule.html",
            first_match_report_file=ROOT / "tests/fixtures/fbref_premier_league_match_report.html",
            lineups_file=ROOT / "tests/fixtures/premier_league_official_lineups.json",
            observed_at=OBSERVED_AT,
            first_report_known_at=datetime(2025, 8, 15, 22, 0, tzinfo=UTC),
            second_result_known_at=datetime(2025, 8, 23, 16, 0, tzinfo=UTC),
            profile_minimum_minutes=60,
        )

    manifest_paths = list((data_root / "derived/manifests/runs").rglob("*.json"))
    assert len(manifest_paths) == 1
    payload = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["id"])
    assert manifest.status == "failed"
    assert manifest.checkpoint == "before-completion"
    assert manifest.error and manifest.error.startswith("FileNotFoundError:")
