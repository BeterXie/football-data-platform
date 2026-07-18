from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.pipelines.vertical_slice import run_offline_vertical_slice
from football_data_platform.storage import (
    derived as derived_module,
)
from football_data_platform.storage import (
    facts as facts_module,
)
from football_data_platform.storage import (
    match_report_contracts as contract_module,
)

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


def _arguments(tmp_path: Path) -> dict[str, object]:
    return {
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


def test_run_manifest_request_replays_each_team_fact_contract_and_manifest_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_run_type: str | None = None
    calls: Counter[tuple[str, str]] = Counter()
    contract_calls: Counter[str] = Counter()
    manifest_calls: Counter[tuple[str, str]] = Counter()
    original_write_run = derived_module.DerivedArchive.write_run_manifest
    original_team_rows = facts_module._team_observation_rows
    original_parse_manifest = derived_module._parse_artifact_manifest
    original_parse_report = contract_module.parse_match_report

    def counted_team_rows(source_ref: str, connection: object):
        if active_run_type is not None:
            key = (active_run_type, source_ref)
            calls[key] += 1
        return original_team_rows(source_ref, connection)  # type: ignore[arg-type]

    def counted_parse_report(*args: object, **kwargs: object):
        if active_run_type is not None:
            contract_calls[active_run_type] += 1
        return original_parse_report(*args, **kwargs)  # type: ignore[arg-type]

    def counted_write_run(self: derived_module.DerivedArchive, manifest: object):
        nonlocal active_run_type
        previous = active_run_type
        active_run_type = manifest.run_type  # type: ignore[attr-defined]
        try:
            return original_write_run(self, manifest)  # type: ignore[arg-type]
        finally:
            active_run_type = previous

    def counted_parse_manifest(payload: object):
        if active_run_type is not None and isinstance(payload, dict):
            artifact_id = payload.get("id")
            if isinstance(artifact_id, str):
                manifest_calls[(active_run_type, artifact_id)] += 1
        return original_parse_manifest(payload)

    monkeypatch.setattr(facts_module, "_team_observation_rows", counted_team_rows)
    monkeypatch.setattr(contract_module, "parse_match_report", counted_parse_report)
    monkeypatch.setattr(derived_module, "_parse_artifact_manifest", counted_parse_manifest)
    monkeypatch.setattr(derived_module.DerivedArchive, "write_run_manifest", counted_write_run)

    result = run_offline_vertical_slice(**_arguments(tmp_path))  # type: ignore[arg-type]

    team_replays = {
        reference: count
        for (run_type, reference), count in calls.items()
        if run_type == "offline-golden-replay"
    }
    manifest_parses = {
        reference: count
        for (run_type, reference), count in manifest_calls.items()
        if run_type == "offline-golden-replay"
    }

    assert result.run_id.startswith("run:")
    assert len(team_replays) == 2
    assert set(team_replays.values()) == {1}
    assert contract_calls["offline-golden-replay"] == 1
    assert manifest_parses
    assert set(manifest_parses.values()) == {1}
