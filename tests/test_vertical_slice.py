from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from football_data_platform.domain.ids import MatchId, RawAssetId, TeamId
from football_data_platform.pipelines.vertical_slice import (
    _paired_baseline_team_facts,
    run_offline_vertical_slice,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import TeamMatchObservation
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import ArchiveConflictError
from football_data_platform.storage.training import TrainingArtifactStore

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


def test_baseline_team_fact_pair_rejects_cross_contract_mix() -> None:
    match_id = MatchId("match:paired-facts")
    home_team_id = TeamId("team:home")
    away_team_id = TeamId("team:away")
    common = {
        "match_id": match_id,
        "match_version": 1,
        "observation_version": 1,
        "stats": {"xg": 1.0},
        "known_at": OBSERVED_AT,
        "observed_at": OBSERVED_AT,
        "raw_asset_id": RawAssetId("raw-asset:" + "0" * 64),
        "contract_id": "match-report-contract:" + "1" * 64,
    }
    home = TeamMatchObservation(
        record_id="fact:team_match_observations:" + "2" * 64,
        team_id=home_team_id,
        **common,
    )
    away = TeamMatchObservation(
        record_id="fact:team_match_observations:" + "3" * 64,
        team_id=away_team_id,
        **common,
    )

    assert _paired_baseline_team_facts(
        (home, away),
        match_id=match_id,
        match_version=1,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    ) == (home, away)
    with pytest.raises(ValueError, match="share one canonical match observation"):
        _paired_baseline_team_facts(
            (home, replace(away, contract_id="match-report-contract:" + "4" * 64)),
            match_id=match_id,
            match_version=1,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )


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
    assert any(ref.startswith("vertical-slice-summary:") for ref in run_manifest.output_refs)
    assert any(ref.startswith("vertical-slice-report:") for ref in run_manifest.output_refs)
    assert summary["run_id"] == run_manifest.run_id
    assert summary["snapshots"][0]["quality_status"] == "preview"
    assert summary["snapshots"][0]["capture_mode"] == "reconstructed"
    assert summary["snapshots"][1]["quality_status"] == "preview"
    assert all(
        any(field.startswith("feature:match_context:") for field in snapshot["missing_fields"])
        for snapshot in summary["snapshots"]
    )
    assert first.prediction_id is None
    assert summary["prediction"] == {
        "available": False,
        "reason": "snapshot_not_ready:expected_goals_unavailable",
    }
    assert not summary["evaluation"]["available"]
    assert not summary["evaluation"]["market_benchmark"]["available"]
    assert summary["evaluation"]["capture_mode"] == "reconstructed"
    assert first.report_path.read_text(encoding="utf-8").startswith(
        "# Premier League 2025-26 Vertical Slice"
    )
    canonical = CanonicalStore(DataLayout(arguments["data_root"]).canonical / "platform.sqlite3")
    with canonical.connect() as connection:
        lineup_sources = connection.execute(
            "SELECT lineup.official, raw.source, COUNT(*) AS count "
            "FROM lineup_facts AS lineup JOIN raw_assets AS raw "
            "ON raw.raw_asset_id = lineup.raw_asset_id "
            "GROUP BY lineup.official, raw.source"
        ).fetchall()
    assert {(row["official"], row["source"]): row["count"] for row in lineup_sources} == {
        (0, "fbref"): 3,
        (1, "golden-official-lineup-fixture"): 22,
    }
    derived = DerivedArchive(DataLayout(arguments["data_root"]))
    baseline = derived.load_team_baseline(summary["team_baseline"]["artifact_id"])
    assert len(baseline.input_refs) == 2
    assert set(baseline.input_refs) == set(summary["team_baseline"]["input_refs"])
    assert all(
        reference.startswith("fact:team_match_observations:") for reference in baseline.input_refs
    )
    artifacts = [
        derived.load_artifact_manifest(json.loads(path.read_text(encoding="utf-8"))["id"])
        for path in (arguments["data_root"] / "derived/manifests/artifacts").rglob("*.json")
    ]
    summary_artifact = next(
        item for item in artifacts if item.artifact_type == "vertical-slice-summary"
    )
    report_artifact = next(
        item for item in artifacts if item.artifact_type == "vertical-slice-report"
    )
    qualifications = [item for item in artifacts if item.artifact_type == "training-qualification"]
    assert summary_artifact.input_refs == (run_manifest.run_id,)
    assert report_artifact.input_refs == (summary_artifact.artifact_id,)
    assert sorted(item.quality for item in qualifications) == ["failed", "failed"]
    assert not any(item.artifact_type == "score-grid-composition" for item in artifacts)
    assert not any(item.artifact_type == "prediction" for item in artifacts)
    assert not any(item.artifact_type == "evaluation" for item in artifacts)
    training = TrainingArtifactStore(DataLayout(arguments["data_root"]))
    dataset = training.load_formal_dataset(summary["training_dataset_id"])
    model_run = training.load_model_run(summary["model_run_id"])
    assert dataset.schema_version == 2
    assert dataset.status.value == "failed"
    train_sample = next(sample for sample in dataset.samples if sample.split == "train")
    holdout_sample = next(sample for sample in dataset.samples if sample.split == "holdout")
    assert not train_sample.qualification_passed
    assert "missing_bound_prematch_snapshot" in train_sample.exclusion_reasons
    assert not holdout_sample.qualification_passed
    assert "missing_ready_prematch_snapshot" in holdout_sample.exclusion_reasons
    assert model_run.status.value == "failed"
    assert model_run.error == "no_eligible_train_split"
    assert summary["training"]["model_gate"]["passed"] is False
    assert summary["training"]["eligible_train_samples"] == []
    assert summary["training"]["eligible_holdout_samples"] == []
    assert len(summary["training"]["excluded_samples"]) == 2
    assert summary["lambda_composition"]["available"] is False
    assert summary["lambda_composition"]["reason"] == "prediction_snapshot_not_ready"
    assert "lambda_home" not in summary["lambda_composition"]
    assert "lambda_away" not in summary["lambda_composition"]
    assert summary["lambda_composition"]["composition_artifact_ref"] is None
    assert any(ref.startswith("file-sha256:") for ref in summary_artifact.output_refs)
    assert any(ref.startswith("file-sha256:") for ref in report_artifact.output_refs)
    registrations = [
        derived.load_run_manifest(json.loads(path.read_text(encoding="utf-8"))["id"])
        for path in (arguments["data_root"] / "derived/manifests/runs").rglob("*.json")
    ]
    registration = next(
        item for item in registrations if item.run_type == "offline-golden-output-registration"
    )
    assert registration.input_refs == (run_manifest.run_id,)
    assert summary_artifact.artifact_id in registration.output_refs
    assert report_artifact.artifact_id in registration.output_refs

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
