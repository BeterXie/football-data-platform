from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _formal_training import seed_formal_score_sample
from _training_refs import seed_training_references

from football_data_platform.cli import _latest_result, _select_report_version, main
from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import MatchStatus, MatchVersion
from football_data_platform.domain.snapshots import CaptureMode
from football_data_platform.domain.training import (
    TRAINING_DATASET_SCHEMA_VERSION,
    ModelRunArtifact,
    TrainingDatasetManifest,
    TrainingSample,
)
from football_data_platform.domain.training_qualification import (
    CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
)
from football_data_platform.pipelines.schedule import ingest_fbref_schedule
from football_data_platform.sources.fbref import FBrefFetchError, FetchDiagnostic, schedule_url
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive
from football_data_platform.storage.training import TrainingArtifactStore

ROOT = Path(__file__).parents[1]


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_manifest_payloads(layout: DataLayout) -> tuple[dict[str, object], ...]:
    root = layout.derived / "manifests" / "artifacts"
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("*.json"))


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


def test_cli_result_lookup_respects_known_and_observed_boundaries(tmp_path: Path) -> None:
    registry = load_competition_registry(ROOT / "config/competitions.toml")
    competition = registry.competitions[0]
    season = competition.seasons[0]
    observed_at = datetime(2026, 7, 16, 6, tzinfo=UTC)
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=observed_at)
    schedule = ingest_fbref_schedule(
        (ROOT / "tests/fixtures/fbref_premier_league_schedule.html").read_bytes(),
        page_url=schedule_url(competition, season),
        competition=competition,
        season=season,
        observed_at=observed_at,
        archive=archive,
        canonical=canonical,
    )
    match_id = MatchId(schedule.canonical_match_ids[0])
    raw_asset_id = RawAssetId(schedule.raw_asset_id)
    facts = CanonicalFactStore(canonical)
    first = facts.append_result_90(
        match_id=match_id,
        match_version=1,
        home_goals=2,
        away_goals=1,
        known_at=observed_at,
        observed_at=observed_at,
        raw_asset_id=raw_asset_id,
    )
    facts.append_result_90(
        match_id=match_id,
        match_version=1,
        home_goals=9,
        away_goals=9,
        known_at=observed_at,
        observed_at=observed_at + timedelta(hours=1),
        raw_asset_id=raw_asset_id,
    )

    assert _latest_result(
        canonical,
        match_id,
        1,
        known_at=observed_at + timedelta(minutes=30),
        observed_at=observed_at + timedelta(minutes=30),
    ) == (first.record_id, 2, 1)


def test_cli_golden_failure_reports_failed_run_manifest(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    code = main(
        [
            "run-golden",
            "--data-root",
            str(data_root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--fixtures-dir",
            str(tmp_path / "missing-fixtures"),
            "--observed-at",
            "2026-07-16T08:00:00Z",
        ]
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["run_id"].startswith("run:")
    assert Path(payload["manifest"]).is_file()
    assert payload["checkpoint"] == "before-completion"


def test_cli_missing_registry_still_persists_failed_run_manifest(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    missing_registry = tmp_path / "missing-registry.toml"

    assert (
        main(
            [
                "init",
                "--data-root",
                str(data_root),
                "--registry",
                str(missing_registry),
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["run_id"])
    assert manifest.status == "failed"
    assert manifest.checkpoint == "operation-failed"
    assert f"missing-file:{missing_registry.resolve()}" in manifest.input_refs


def test_cli_fetch_failure_archives_empty_body_and_counts_only_new_attempts(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data"
    observed_at = datetime(2026, 7, 16, 9, tzinfo=UTC)
    assert (
        main(
            [
                "run-golden",
                "--data-root",
                str(data_root),
                "--registry",
                str(ROOT / "config/competitions.toml"),
                "--fixtures-dir",
                str(ROOT / "examples/golden"),
                "--observed-at",
                "2026-07-16T08:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()

    def fail_fetch(url: str, **_kwargs):
        raise FBrefFetchError(
            FetchDiagnostic(
                code="http_error",
                url=url,
                http_status=503,
                message="FBref request failed with HTTP 503",
                observed_at=observed_at,
                body=b"",
            )
        )

    monkeypatch.setattr("football_data_platform.cli.fetch_schedule", fail_fetch)
    arguments = [
        "diagnose-fbref",
        "--data-root",
        str(data_root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
        "--observed-at",
        "2026-07-16T09:00:00Z",
    ]

    assert main(arguments) == 3
    first = json.loads(capsys.readouterr().err)
    assert first["body_present"] is True
    assert first["body_size_bytes"] == 0
    assert first["raw_evidence_archived"] is True
    assert first["collection_attempts_recorded"] == 2
    assert RawArchive(DataLayout(data_root)).read(RawAssetId(first["raw_asset_id"])) == b""

    assert main(arguments) == 3
    second = json.loads(capsys.readouterr().err)
    assert second["raw_asset_id"] == first["raw_asset_id"]
    assert second["collection_attempts_recorded"] == 0
    canonical = CanonicalStore(DataLayout(data_root).canonical / "platform.sqlite3")
    season_id = (
        load_competition_registry(ROOT / "config/competitions.toml").competitions[0].seasons[0].id
    )
    attempts = [
        attempt
        for attempt in canonical.collection_attempts(season_id)
        if attempt.source == "fbref-schedule-fetch"
    ]
    assert len(attempts) == 2


def test_report_version_selection_rejects_ambiguous_visible_versions() -> None:
    match_id = MatchId("match:version-selection")
    observed_at = datetime(2026, 7, 16, 6, tzinfo=UTC)
    versions = (
        MatchVersion(
            match_id=match_id,
            version=1,
            kickoff_at=datetime(2025, 8, 15, 19, tzinfo=UTC),
            status=MatchStatus.FINISHED,
            observed_at=observed_at,
        ),
        MatchVersion(
            match_id=match_id,
            version=2,
            kickoff_at=datetime(2025, 8, 15, 20, tzinfo=UTC),
            status=MatchStatus.FINISHED,
            observed_at=observed_at + timedelta(hours=1),
        ),
    )

    selected, error = _select_report_version(
        versions,
        requested_version=None,
        observed_at=observed_at + timedelta(hours=2),
        report_date=datetime(2025, 8, 15, tzinfo=UTC).date(),
    )
    assert selected is None
    assert error and error[0] == "match_version_ambiguous"

    selected, error = _select_report_version(
        versions,
        requested_version=1,
        observed_at=observed_at + timedelta(minutes=30),
        report_date=None,
    )
    assert selected is versions[0]
    assert error is None


def test_cli_schedule_gate_returns_nonzero_for_partial_golden_fixture(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    exit_code = main(
        [
            "validate-schedule",
            "--data-root",
            str(data_root),
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
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["run_id"])
    assert manifest.status == "partial"
    assert manifest.checkpoint == "coverage-gate"


def test_cli_schedule_gate_ignores_unpersisted_attempt_file(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    attempted = tmp_path / "attempted.json"
    attempted.write_text(
        json.dumps({"attempted_fixture_ids": ["aaaaaaaa", "2025-2026:cff3d9bb:18bb7c10"]}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "validate-schedule",
            "--data-root",
            str(data_root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(ROOT / "examples/golden/fbref_premier_league_schedule.html"),
            "--attempted-fixtures",
            str(attempted),
            "--require-report-attempts",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["attempted_fixture_file_ignored"]
    assert len(payload["missing_collection_attempts"]) == 2
    latest = json.loads(Path(payload["latest"]).read_text(encoding="utf-8"))
    assert latest["run_id"] == payload["run_id"]
    assert "actual_matches" not in latest


def test_cli_results_backfill_reports_partial_schedule_gate(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    exit_code = main(
        [
            "backfill-results",
            "--data-root",
            str(data_root),
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
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["run_id"])
    assert manifest.status == "partial"
    assert payload["artifact_id"] in manifest.output_refs
    assert any(reference.startswith("file-sha256:") for reference in manifest.output_refs)


def test_cli_backfill_outputs_are_content_addressed_and_tamper_evident(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    arguments = [
        "backfill-results",
        "--data-root",
        str(data_root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
        "--file",
        str(ROOT / "tests/fixtures/football_data_e0_sample.csv"),
        "--observed-at",
        "2026-07-16T10:00:00Z",
    ]

    assert main(arguments) == 2
    first = json.loads(capsys.readouterr().out)
    assert main(arguments) == 2
    second = json.loads(capsys.readouterr().out)
    assert first["summary"] == second["summary"]
    assert first["report"] == second["report"]
    latest = json.loads(Path(second["latest"]).read_text(encoding="utf-8"))
    assert set(latest) == {
        "artifact_id",
        "checkpoint",
        "manifest",
        "report",
        "run_id",
        "status",
        "summary",
    }

    Path(second["report"]).write_text("tampered\n", encoding="utf-8")
    assert main(arguments) == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["checkpoint"] == "manifest-write-failed"
    failed_manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(failure["run_id"])
    assert failed_manifest.status == "failed"
    assert "immutable command output conflicts" in failed_manifest.error


def test_cli_ingest_match_report_persists_success_and_identity_failure(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    golden = [
        "run-golden",
        "--data-root",
        str(data_root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
        "--fixtures-dir",
        str(ROOT / "examples/golden"),
        "--observed-at",
        "2026-07-16T08:00:00Z",
    ]
    assert main(golden) == 0
    capsys.readouterr()
    report_arguments = [
        "ingest-match-report",
        "--data-root",
        str(data_root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
        "--file",
        str(ROOT / "examples/golden/fbref_premier_league_match_report.html"),
        "--source-match-id",
        "aaaaaaaa",
        "--page-url",
        "https://fbref.com/en/matches/aaaaaaaa/report",
        "--required-report-table",
        "summary",
        "--known-at",
        "2025-08-15T22:00:00Z",
        "--observed-at",
        "2026-07-16T08:00:00Z",
    ]
    assert main(report_arguments) == 0
    success = json.loads(capsys.readouterr().out)
    assert success["parser_version"] == "fbref-match-report/2"
    assert set(success["tables_present"]) == {
        "summary",
        "passing",
        "passing_types",
        "defense",
        "possession",
        "misc",
        "keeper",
    }
    successful_manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(success["run_id"])
    assert successful_manifest.status == "succeeded"

    conflicting = tmp_path / "conflicting-report.html"
    conflicting.write_bytes(
        (ROOT / "examples/golden/fbref_premier_league_match_report.html")
        .read_bytes()
        .replace(
            b"/en/matches/aaaaaaaa/arsenal-chelsea-August-15-2025-Premier-League",
            b"/en/matches/bbbbbbbb/arsenal-chelsea-August-15-2025-Premier-League",
        )
    )
    fixture_path = str(ROOT / "examples/golden/fbref_premier_league_match_report.html")
    report_arguments[report_arguments.index(fixture_path)] = str(conflicting)
    report_arguments[-1] = "2026-07-16T08:00:01Z"
    assert main(report_arguments) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["diagnostic_code"] == "report_source_match_identity_mismatch"
    failed_manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(failure["run_id"])
    assert failed_manifest.status == "failed"
    assert failed_manifest.checkpoint == "report-attempt-recorded"


def test_cli_ingest_match_report_url_failure_keeps_both_raw_refs(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    golden = [
        "run-golden",
        "--data-root",
        str(data_root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
        "--fixtures-dir",
        str(ROOT / "examples/golden"),
        "--observed-at",
        "2026-07-16T08:00:00Z",
    ]
    assert main(golden) == 0
    capsys.readouterr()
    result = main(
        [
            "ingest-match-report",
            "--data-root",
            str(data_root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(ROOT / "examples/golden/fbref_premier_league_match_report.html"),
            "--source-match-id",
            "aaaaaaaa",
            "--page-url",
            "https://fbref.com/en/matches/bbbbbbbb/report",
            "--known-at",
            "2025-08-15T22:00:00Z",
            "--observed-at",
            "2026-07-16T08:01:00Z",
        ]
    )
    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["diagnostic_code"] == "report_page_url_identity_mismatch"
    assert payload["attempt_raw_asset_id"]
    assert payload["raw_asset_id"] != payload["attempt_raw_asset_id"]
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["run_id"])
    assert payload["raw_asset_id"] in manifest.output_refs
    assert payload["attempt_raw_asset_id"] in manifest.output_refs


def test_cli_registers_temporal_training_and_content_addressed_model_run(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    layout = DataLayout(data_root)
    train_as_of = datetime(2025, 8, 1, 12, tzinfo=UTC)
    holdout_as_of = train_as_of + timedelta(days=7)
    evaluated_at = holdout_as_of + timedelta(days=1, hours=3)
    training, _ = seed_formal_score_sample(
        layout,
        key="cli-training",
        kickoff=train_as_of + timedelta(hours=24),
        observed_at=evaluated_at,
        evaluated_at=evaluated_at,
        team_indices=(0, 1, 2, 3),
        goals=(2, 1),
        split="train",
    )
    holdout, _ = seed_formal_score_sample(
        layout,
        key="cli-holdout",
        kickoff=holdout_as_of + timedelta(hours=24),
        observed_at=evaluated_at,
        evaluated_at=evaluated_at,
        team_indices=(4, 5, 6, 7),
        goals=(1, 0),
        split="holdout",
    )
    dataset = TrainingDatasetManifest.create_formal(
        dataset_version="score-dataset/cli-2",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version=CURRENT_SCORE_FEATURE_PROJECTION_VERSION,
        label_version="result-90/1",
        as_of=holdout.as_of,
        split_strategy="forward-chaining/1",
        samples=(training, holdout),
        generated_at=evaluated_at,
        code_version="git:test",
    )
    assert dataset.schema_version == TRAINING_DATASET_SCHEMA_VERSION
    dataset_file = tmp_path / "dataset.json"
    _write_payload(dataset_file, dataset.to_payload())
    common = ["--data-root", str(data_root), "--registry", str(ROOT / "config/competitions.toml")]
    assert main(["register-training-dataset", *common, "--file", str(dataset_file)]) == 0
    dataset_result = json.loads(capsys.readouterr().out)
    assert dataset_result["dataset_id"] == dataset.dataset_id
    manifests = _artifact_manifest_payloads(layout)
    dataset_owners = [
        payload for payload in manifests if dataset.dataset_id in payload.get("output_refs", [])
    ]
    assert len(dataset_owners) == 1
    assert dataset_owners[0]["artifact_type"] == "training-dataset"
    dataset_command = next(
        payload
        for payload in manifests
        if payload["artifact_type"] == "cli-register-training-dataset"
    )
    assert dataset.dataset_id not in dataset_command["output_refs"]
    assert any(reference.startswith("file-sha256:") for reference in dataset_command["output_refs"])

    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"deterministic dixon-coles parameters\n")
    assert main(["register-model-artifact", *common, "--file", str(model_file)]) == 0
    model_result = json.loads(capsys.readouterr().out)
    output_file = tmp_path / "evaluation.json"
    output_file.write_bytes(b'{"cohort":"holdout","status":"ready"}\n')
    assert (
        main(
            [
                "register-model-artifact",
                *common,
                "--file",
                str(output_file),
                "--artifact-kind",
                "output",
            ]
        )
        == 0
    )
    output_result = json.loads(capsys.readouterr().out)
    artifact = ModelRunArtifact.create(
        model_version="dixon-coles/cli-2",
        run_role="research",
        task="score-model",
        dataset_id=dataset.dataset_id,
        feature_version=dataset.feature_version,
        label_version=dataset.label_version,
        algorithm="dixon-coles",
        parameters={"rho": -0.1, "max_goals": 11},
        code_version="git:test",
        environment_version="python:test-lock",
        started_at=evaluated_at,
        ended_at=evaluated_at + timedelta(seconds=1),
        random_seed=17,
        model_artifact_refs=(model_result["content_ref"],),
        evaluation_cohort=(holdout.sample_id,),
        evaluation_capture_mode=CaptureMode.RECONSTRUCTED,
        output_refs=(output_result["content_ref"],),
        output_hashes=(output_result["content_hash"],),
    )
    run_file = tmp_path / "model-run.json"
    _write_payload(run_file, artifact.to_payload())
    assert main(["register-model-run", *common, "--file", str(run_file)]) == 0
    run_result = json.loads(capsys.readouterr().out)
    assert run_result["model_run_id"] == artifact.model_run_id
    manifests = _artifact_manifest_payloads(layout)
    model_run_owners = [
        payload for payload in manifests if artifact.model_run_id in payload.get("output_refs", [])
    ]
    assert len(model_run_owners) == 1
    assert model_run_owners[0]["artifact_type"] == "model-run"
    model_run_command = next(
        payload for payload in manifests if payload["artifact_type"] == "cli-register-model-run"
    )
    assert artifact.model_run_id not in model_run_command["output_refs"]
    assert any(
        reference.startswith("file-sha256:") for reference in model_run_command["output_refs"]
    )
    assert (
        TrainingArtifactStore(DataLayout(data_root)).load_model_run(artifact.model_run_id)
        == artifact
    )


def test_cli_resume_lineage_survives_operation_failure_and_rejects_wrong_command(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"resume-test-model\n")
    common = ["--data-root", str(data_root), "--registry", str(ROOT / "config/competitions.toml")]
    arguments = ["register-model-artifact", *common, "--file", str(model_file)]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)

    assert main([*arguments, "--resume-run-id", first["run_id"]]) == 0
    resumed = json.loads(capsys.readouterr().out)
    archive = DerivedArchive(DataLayout(data_root))
    resumed_manifest = archive.load_run_manifest(resumed["run_id"])
    assert first["run_id"] in resumed_manifest.input_refs
    assert resumed_manifest.parameters["resume_mode"] == "replay"
    assert resumed_manifest.payload["resume_mode"] == "replay"

    missing = tmp_path / "missing.bin"
    failed_arguments = [
        "register-model-artifact",
        *common,
        "--file",
        str(missing),
        "--resume-run-id",
        resumed["run_id"],
    ]
    assert main(failed_arguments) == 1
    failed = json.loads(capsys.readouterr().err)
    failed_manifest = archive.load_run_manifest(failed["run_id"])
    assert failed_manifest.status == "failed"
    assert resumed["run_id"] in failed_manifest.input_refs

    assert main(["init", *common, "--resume-run-id", first["run_id"]]) == 1
    wrong_command = json.loads(capsys.readouterr().err)
    assert "belongs to cli-register-model-artifact" in wrong_command["diagnostic_message"]


def test_cli_training_gate_returns_nonzero_for_failed_dataset(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    as_of = datetime(2025, 8, 1, 12, tzinfo=UTC)
    feature_ref, label_ref = seed_training_references(
        DataLayout(data_root),
        label_known_at=as_of + timedelta(hours=2),
        observed_at=as_of + timedelta(hours=2),
        home_goals=0,
        away_goals=0,
        reference_key="cli-excluded",
    )
    excluded = TrainingSample(
        sample_id="sample:excluded",
        as_of=as_of,
        feature_known_at=as_of + timedelta(minutes=1),
        label_known_at=as_of + timedelta(hours=2),
        capture_mode=CaptureMode.RECONSTRUCTED,
        qualification="score-model-ready",
        qualification_passed=False,
        feature_version="score-features/1",
        label_version="result-90/1",
        feature_refs=(feature_ref,),
        label_ref=label_ref,
        features={"home": 1.0},
        label={"home_goals": 0, "away_goals": 0},
        exclusion_reasons=("feature_after_as_of",),
        split="train",
    )
    dataset = TrainingDatasetManifest.create(
        dataset_version="score-dataset/failed-cli",
        task="score-model",
        qualification="score-model-ready",
        qualification_ruleset_version="readiness/1",
        feature_version="score-features/1",
        label_version="result-90/1",
        as_of=as_of,
        split_strategy="forward-chaining/1",
        samples=(excluded,),
        generated_at=as_of + timedelta(hours=2),
        code_version="git:test",
    )
    manifest = tmp_path / "failed-dataset.json"
    _write_payload(manifest, dataset.to_payload())
    code = main(
        [
            "register-training-dataset",
            "--data-root",
            str(data_root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(manifest),
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["dataset_status"] == "failed"


def test_cli_assess_promotion_missing_artifacts_is_a_failed_gate(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "data"
    code = main(
        [
            "assess-promotion",
            "--data-root",
            str(data_root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--policy-id",
            "promotion-policy:" + "a" * 64,
            "--evidence-id",
            "challenger-evidence:" + "b" * 64,
            "--decided-at",
            "2026-07-16T08:00:00Z",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    manifest = DerivedArchive(DataLayout(data_root)).load_run_manifest(payload["run_id"])
    assert manifest.status == "failed"


def test_cli_registers_governance_and_persists_blocked_decision(
    tmp_path: Path,
    capsys,
) -> None:
    from test_governance_artifacts import _artifacts, _policy

    layout, _, _, _, champion, _, evidence, _ = _artifacts(tmp_path)
    policy = _policy(champion.model_run_id)
    policy_file = tmp_path / "policy.json"
    evidence_file = tmp_path / "evidence.json"
    _write_payload(policy_file, policy.to_payload())
    _write_payload(evidence_file, evidence.to_payload())
    common = [
        "--data-root",
        str(layout.root),
        "--registry",
        str(ROOT / "config/competitions.toml"),
    ]
    assert main(["register-promotion-policy", *common, "--file", str(policy_file)]) == 0
    policy_result = json.loads(capsys.readouterr().out)
    assert policy_result["policy_id"] == policy.content_id
    assert main(["register-challenger-evidence", *common, "--file", str(evidence_file)]) == 0
    evidence_result = json.loads(capsys.readouterr().out)
    assert evidence_result["evidence_id"] == evidence.content_id

    code = main(
        [
            "assess-promotion",
            *common,
            "--policy-id",
            policy.content_id,
            "--evidence-id",
            evidence.content_id,
            "--decided-at",
            "2026-07-19T08:00:00Z",
        ]
    )
    assert code == 2
    decision_result = json.loads(capsys.readouterr().out)
    assert decision_result["status"] == "blocked"
    decision_manifest = DerivedArchive(layout).load_run_manifest(decision_result["run_id"])
    assert decision_manifest.status == "partial"
    assert decision_result["decision_id"] in decision_manifest.output_refs
    latest = json.loads(Path(decision_result["latest"]).read_text(encoding="utf-8"))
    assert latest["run_id"] == decision_result["run_id"]


def test_cli_rejects_self_reported_challenger_metrics(tmp_path: Path, capsys) -> None:
    from dataclasses import replace

    from test_governance_artifacts import _artifacts

    layout, _, _, _, _, _, evidence, _ = _artifacts(tmp_path)
    tampered = replace(
        evidence,
        brier_delta_vs_champion=evidence.brier_delta_vs_champion + 0.1,
    )
    evidence_file = tmp_path / "tampered-evidence.json"
    _write_payload(evidence_file, tampered.to_payload())

    code = main(
        [
            "register-challenger-evidence",
            "--data-root",
            str(layout.root),
            "--registry",
            str(ROOT / "config/competitions.toml"),
            "--file",
            str(evidence_file),
        ]
    )

    assert code == 1
    result = json.loads(capsys.readouterr().err)
    assert result["status"] == "failed"
    assert "recomputed evaluation records" in result["diagnostic_message"]
