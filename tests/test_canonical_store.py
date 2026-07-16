from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, RawAssetId, SeasonId
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.storage.canonical import (
    CanonicalConflictError,
    CanonicalStore,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

NOW = datetime(2026, 7, 16, 2, 0, tzinfo=UTC)
COMPETITION_ID = CompetitionId("competition:eng.1")
SEASON_ID = SeasonId("season:eng.1.2025-26")


@pytest.fixture
def prepared_store(tmp_path: Path) -> tuple[CanonicalStore, object]:
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    asset = archive.archive(
        b"fixture source",
        source="fbref",
        source_id="schedule",
        url="https://fbref.example/schedule",
        observed_at=NOW,
        target_event_time=None,
        collector_version="test/1",
        media_type="text/html",
    )
    store = CanonicalStore(layout.canonical / "platform.sqlite3")
    store.initialize()
    store.register_registry(
        load_competition_registry(Path(__file__).parents[1] / "config" / "competitions.toml"),
        registered_at=NOW,
    )
    store.register_raw_asset(asset)
    return store, asset


def test_registry_and_raw_asset_registration_are_idempotent(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    registry = load_competition_registry(Path(__file__).parents[1] / "config" / "competitions.toml")

    store.register_registry(registry, registered_at=NOW)
    store.register_raw_asset(asset)  # type: ignore[arg-type]

    assert store.counts() == {
        "competitions": 1,
        "seasons": 1,
        "teams": 0,
        "players": 0,
        "matches": 0,
    }


def test_display_names_are_not_identity_keys(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    first = store.resolve_or_create_team(
        source="fbref",
        source_id="team-a",
        canonical_name="United",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )
    second = store.resolve_or_create_team(
        source="fbref",
        source_id="team-b",
        canonical_name="United",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )

    assert first.id != second.id
    assert first.canonical_name == second.canonical_name


def test_fixture_replay_is_idempotent_and_reschedule_adds_a_version(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    home = store.resolve_or_create_team(
        source="fbref",
        source_id="arsenal",
        canonical_name="Arsenal",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )
    away = store.resolve_or_create_team(
        source="fbref",
        source_id="chelsea",
        canonical_name="Chelsea",
        competition_id=COMPETITION_ID,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )
    arguments = {
        "source": "fbref",
        "source_id": "match-123",
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "home_team_id": home.id,
        "away_team_id": away.id,
        "kickoff_at": NOW + timedelta(days=1),
        "status": MatchStatus.SCHEDULED,
        "observed_at": NOW,
        "raw_asset_id": asset.id,  # type: ignore[attr-defined]
    }

    match, initial = store.resolve_or_create_match(**arguments)  # type: ignore[arg-type]
    replayed_match, replayed = store.resolve_or_create_match(**arguments)  # type: ignore[arg-type]
    _, rescheduled = store.resolve_or_create_match(
        **{
            **arguments,
            "kickoff_at": NOW + timedelta(days=2),
            "observed_at": NOW + timedelta(hours=1),
        }
    )  # type: ignore[arg-type]

    assert match == replayed_match
    assert initial == replayed
    assert rescheduled.version == 2
    assert [version.version for version in store.match_versions(match.id)] == [1, 2]
    assert store.counts()["matches"] == 1


def test_source_match_cannot_silently_change_teams(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    teams = [
        store.resolve_or_create_team(
            source="fbref",
            source_id=f"team-{index}",
            canonical_name=f"Team {index}",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=asset.id,  # type: ignore[attr-defined]
        )
        for index in range(3)
    ]
    base = {
        "source": "fbref",
        "source_id": "match-conflict",
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "home_team_id": teams[0].id,
        "away_team_id": teams[1].id,
        "kickoff_at": NOW,
        "status": MatchStatus.SCHEDULED,
        "observed_at": NOW,
        "raw_asset_id": asset.id,  # type: ignore[attr-defined]
    }
    store.resolve_or_create_match(**base)  # type: ignore[arg-type]

    with pytest.raises(CanonicalConflictError, match="changed stable identity"):
        store.resolve_or_create_match(**{**base, "away_team_id": teams[2].id})  # type: ignore[arg-type]


def test_canonical_facts_require_registered_raw_lineage(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, _ = prepared_store

    with pytest.raises(KeyError, match="raw asset"):
        store.resolve_or_create_team(
            source="fbref",
            source_id="orphan",
            canonical_name="Orphan FC",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=RawAssetId("raw-asset:" + "a" * 64),
        )


def test_existing_player_resolution_still_requires_registered_raw_lineage(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    store.resolve_or_create_player(
        source="fbref",
        source_id="player-1",
        canonical_name="Player One",
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )

    with pytest.raises(KeyError, match="raw asset"):
        store.resolve_or_create_player(
            source="fbref",
            source_id="player-1",
            canonical_name="Player One",
            observed_at=NOW,
            raw_asset_id=RawAssetId("raw-asset:" + "b" * 64),
        )


def test_collection_attempts_are_persisted_and_blocked_attempts_need_no_raw(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    teams = [
        store.resolve_or_create_team(
            source="fbref",
            source_id=f"attempt-team-{index}",
            canonical_name=f"Attempt Team {index}",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=asset.id,  # type: ignore[attr-defined]
        )
        for index in range(2)
    ]
    match, _ = store.resolve_or_create_match(
        source="fbref",
        source_id="attempt-match",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )

    blocked = store.record_collection_attempt(
        match_id=match.id,
        source="fbref-match-report",
        target_url="https://fbref.example/matches/attempt-match",
        outcome=CollectionAttemptOutcome.BLOCKED,
        observed_at=NOW + timedelta(minutes=1),
        collector_version="fbref-match-report/test",
        diagnostic_code="blocked_by_access_control",
        diagnostic_message="HTTP 403",
    )

    assert blocked.raw_asset_id is None
    assert store.collection_attempts(SEASON_ID) == (blocked,)
    with pytest.raises(ValueError, match="successful collection attempts require raw_asset_id"):
        store.record_collection_attempt(
            match_id=match.id,
            source="fbref-match-report",
            target_url="https://fbref.example/matches/attempt-match",
            outcome=CollectionAttemptOutcome.SUCCEEDED,
            observed_at=NOW + timedelta(minutes=2),
            collector_version="fbref-match-report/test",
        )


def test_initialize_migrates_v1_match_versions_and_collection_attempts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT INTO schema_meta(singleton, version) VALUES (1, 1);
            CREATE TABLE match_versions (
                match_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                kickoff_at TEXT,
                status TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                raw_asset_id TEXT NOT NULL,
                PRIMARY KEY (match_id, version)
            );
            """
        )

    store = CanonicalStore(path)
    store.initialize()

    with store.connect() as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == 2
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(match_versions)")}
        assert "round_name" in columns
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'collection_attempts'"
        ).fetchone()
