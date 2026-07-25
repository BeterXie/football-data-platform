from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, RawAssetId, SeasonId
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.storage.canonical import (
    SCHEMA_VERSION,
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
        source_id="attempt-match",
        target_url="https://fbref.example/matches/attempt-match",
        outcome=CollectionAttemptOutcome.BLOCKED,
        observed_at=NOW + timedelta(minutes=1),
        collector_version="fbref-match-report/test",
        diagnostic_code="blocked_by_access_control",
        diagnostic_message="HTTP 403",
    )

    assert blocked.raw_asset_id is None
    assert blocked.source_id == "attempt-match"
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
            == SCHEMA_VERSION
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(match_versions)")}
        assert "round_name" in columns
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'collection_attempts'"
        ).fetchone()
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(collection_attempts)")
        }
        assert "source_id" in columns
        prematch_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(prematch_events)")
        }
        assert "match_version" in prematch_columns
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'match_report_contracts'"
        ).fetchone()
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'official_lineup_contracts'"
        ).fetchone()


def test_initialize_migrates_v2_attempt_source_id_from_raw_asset(tmp_path: Path) -> None:
    path = tmp_path / "canonical-v2.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT INTO schema_meta(singleton, version) VALUES (1, 2);
            CREATE TABLE raw_assets (
                raw_asset_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                url TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                collector_version TEXT NOT NULL
            );
            CREATE TABLE collection_attempts (
                collection_attempt_id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                source TEXT NOT NULL,
                target_url TEXT NOT NULL,
                outcome TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                collector_version TEXT NOT NULL,
                diagnostic_code TEXT,
                diagnostic_message TEXT,
                raw_asset_id TEXT
            );
            INSERT INTO raw_assets VALUES (
                'raw-asset:legacy', 'fbref', 'aaaaaaaa',
                'https://fbref.example/en/matches/aaaaaaaa/report',
                '2026-07-16T03:00:00Z', 'fbref-match-report/2'
            );
            INSERT INTO collection_attempts VALUES (
                'collection-attempt:legacy', 'match:legacy', 'fbref-match-report',
                'https://fbref.example/en/matches/aaaaaaaa/report', 'succeeded',
                '2026-07-16T03:00:00Z', 'fbref-match-report/2', NULL, NULL,
                'raw-asset:legacy'
            );
            """
        )

    store = CanonicalStore(path)
    store.initialize()

    with store.connect() as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == SCHEMA_VERSION
        )
        row = connection.execute(
            "SELECT source_id FROM collection_attempts "
            "WHERE collection_attempt_id = 'collection-attempt:legacy'"
        ).fetchone()
    assert row["source_id"] == "aaaaaaaa"


def test_initialize_migrates_v4_to_official_lineup_contract_schema(tmp_path: Path) -> None:
    path = tmp_path / "canonical-v4.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT INTO schema_meta(singleton, version) VALUES (1, 4);
            """
        )

    store = CanonicalStore(path)
    store.initialize()

    with store.connect() as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == SCHEMA_VERSION
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(official_lineup_contracts)")
        }
    assert {
        "contract_id",
        "raw_asset_id",
        "match_id",
        "match_version",
        "parser_version",
        "team_lineups_json",
        "contract_version",
        "source_bindings_json",
        "fact_ids_json",
    } <= columns


def test_initialize_rebuilds_real_v5_official_contract_table_without_losing_legacy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical-v5.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT INTO schema_meta(singleton, version) VALUES (1, 5);
            CREATE TABLE raw_assets (
                raw_asset_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                url TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                target_event_time TEXT,
                checksum TEXT NOT NULL,
                collector_version TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
            );
            INSERT INTO raw_assets VALUES (
                'raw-asset:legacy', 'official', 'source-match',
                'https://official.example/source-match', '2026-07-16T02:00:00Z',
                '2026-07-16T01:00:00Z', 'legacy-checksum',
                'official-lineup/legacy', 'application/json', 2
            );
            CREATE TABLE match_versions (
                match_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                PRIMARY KEY (match_id, version)
            );
            INSERT INTO match_versions VALUES ('match:legacy', 1);
            CREATE TABLE official_lineup_contracts (
                contract_id TEXT PRIMARY KEY,
                raw_asset_id TEXT NOT NULL REFERENCES raw_assets(raw_asset_id),
                source TEXT NOT NULL,
                source_match_id TEXT NOT NULL,
                match_mapping_source TEXT NOT NULL,
                team_mapping_source TEXT NOT NULL,
                player_mapping_source TEXT NOT NULL,
                match_id TEXT NOT NULL,
                match_version INTEGER NOT NULL,
                parser_version TEXT NOT NULL,
                published_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                team_lineups_json TEXT NOT NULL,
                fact_ids_json TEXT NOT NULL,
                UNIQUE (raw_asset_id, parser_version),
                FOREIGN KEY (match_id, match_version)
                    REFERENCES match_versions(match_id, version)
            );
            INSERT INTO official_lineup_contracts(
                contract_id, raw_asset_id, source, source_match_id,
                match_mapping_source, team_mapping_source, player_mapping_source,
                match_id, match_version, parser_version, published_at, observed_at,
                team_lineups_json, fact_ids_json
            ) VALUES (
                'official-lineup-contract:legacy', 'raw-asset:legacy', 'official',
                'source-match', 'schedule', 'teams', 'players', 'match:legacy', 1,
                'official-lineup/legacy', '2026-07-16T01:00:00Z',
                '2026-07-16T02:00:00Z', '{}', '[]'
            );
            """
        )

    store = CanonicalStore(path)
    store.initialize()
    store.initialize()

    with store.connect() as connection:
        version = connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT contract_id, raw_asset_id, parser_version, contract_version, "
            "source_bindings_json "
            "FROM official_lineup_contracts WHERE contract_id = ?",
            ("official-lineup-contract:legacy",),
        ).fetchone()
        unique_indexes = {
            tuple(
                column["name"]
                for column in connection.execute(f"PRAGMA index_info({index['name']})").fetchall()
            )
            for index in connection.execute(
                "PRAGMA index_list(official_lineup_contracts)"
            ).fetchall()
            if index["unique"]
        }
        foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        connection.execute(
            "INSERT INTO official_lineup_contracts("
            "contract_id, contract_version, raw_asset_id, source, source_match_id, "
            "match_mapping_source, team_mapping_source, player_mapping_source, match_id, "
            "match_version, parser_version, published_at, observed_at, team_lineups_json, "
            "source_bindings_json, fact_ids_json) VALUES ("
            "'official-lineup-contract:v2', 2, 'raw-asset:legacy', 'official', "
            "'source-match', 'schedule', 'teams', 'players', 'match:legacy', 1, "
            "'official-lineup/legacy', '2026-07-16T01:00:00Z', "
            "'2026-07-16T02:00:00Z', '{}', '[]', '[]')"
        )
        contract_count = connection.execute(
            "SELECT COUNT(*) FROM official_lineup_contracts"
        ).fetchone()[0]
    assert version == SCHEMA_VERSION
    assert tuple(row) == (
        "official-lineup-contract:legacy",
        "raw-asset:legacy",
        "official-lineup/legacy",
        1,
        None,
    )
    assert ("raw_asset_id", "parser_version", "contract_version") in unique_indexes
    assert ("raw_asset_id", "parser_version") not in unique_indexes
    assert foreign_key_violations == []
    assert contract_count == 2


def test_initialize_rebuilds_v3_prematch_events_with_v4_constraints(
    prepared_store: tuple[CanonicalStore, object],
) -> None:
    store, asset = prepared_store
    teams = [
        store.resolve_or_create_team(
            source="fbref",
            source_id=f"migration-team-{index}",
            canonical_name=f"Migration Team {index}",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=asset.id,  # type: ignore[attr-defined]
        )
        for index in range(2)
    ]
    player = store.resolve_or_create_player(
        source="fbref",
        source_id="migration-player",
        canonical_name="Migration Player",
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )
    match, version = store.resolve_or_create_match(
        source="fbref",
        source_id="migration-match",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW,
        raw_asset_id=asset.id,  # type: ignore[attr-defined]
    )
    with store.connect() as connection:
        connection.executescript(
            """
            DROP TABLE prematch_events;
            CREATE TABLE prematch_events (
                record_id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL REFERENCES matches(match_id),
                team_id TEXT REFERENCES teams(team_id),
                player_id TEXT REFERENCES players(player_id),
                event_type TEXT NOT NULL,
                occurred_at TEXT,
                known_at TEXT NOT NULL,
                confirmation_status TEXT NOT NULL CHECK (
                    confirmation_status IN ('official', 'corroborated', 'unconfirmed')
                ),
                evidence_refs_json TEXT NOT NULL,
                can_modify_features INTEGER NOT NULL CHECK (can_modify_features IN (0, 1))
            );
            UPDATE schema_meta SET version = 3 WHERE singleton = 1;
            """
        )
        connection.executemany(
            "INSERT INTO prematch_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    "fact:legacy-player-event",
                    match.id.value,
                    teams[0].id.value,
                    player.id.value,
                    "injury",
                    None,
                    "2026-07-16T02:00:00Z",
                    "official",
                    '["news:legacy-player"]',
                    1,
                ),
                (
                    "fact:legacy-team-event",
                    match.id.value,
                    teams[0].id.value,
                    None,
                    "travel-disruption",
                    None,
                    "2026-07-16T02:00:00Z",
                    "official",
                    '["news:legacy-team"]',
                    1,
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO fact_evidence(record_id, raw_asset_id, observed_at) VALUES (?, ?, ?)",
            (
                (
                    "fact:legacy-player-event",
                    asset.id.value,  # type: ignore[attr-defined]
                    "2026-07-16T02:00:00Z",
                ),
                (
                    "fact:legacy-team-event",
                    asset.id.value,  # type: ignore[attr-defined]
                    "2026-07-16T02:00:00Z",
                ),
            ),
        )

    store.initialize()

    with store.connect() as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == SCHEMA_VERSION
        )
        rows = connection.execute(
            "SELECT record_id, match_id, match_version, team_id, player_id, event_type, "
            "evidence_refs_json, can_modify_features FROM prematch_events ORDER BY record_id"
        ).fetchall()
        migrated = {row["record_id"]: row for row in rows}
        assert set(migrated) == {"fact:legacy-player-event", "fact:legacy-team-event"}
        assert migrated["fact:legacy-player-event"]["match_id"] == match.id.value
        assert migrated["fact:legacy-player-event"]["match_version"] is None
        assert migrated["fact:legacy-player-event"]["player_id"] == player.id.value
        assert migrated["fact:legacy-player-event"]["event_type"] == "injury"
        assert migrated["fact:legacy-player-event"]["evidence_refs_json"] == (
            '["news:legacy-player"]'
        )
        assert migrated["fact:legacy-player-event"]["can_modify_features"] == 0
        assert migrated["fact:legacy-team-event"]["match_version"] is None
        assert migrated["fact:legacy-team-event"]["player_id"] is None
        assert migrated["fact:legacy-team-event"]["can_modify_features"] == 1
        evidence_ids = {
            row["record_id"]
            for row in connection.execute(
                "SELECT record_id FROM fact_evidence WHERE record_id LIKE 'fact:legacy-%'"
            )
        }
        assert evidence_ids == {"fact:legacy-player-event", "fact:legacy-team-event"}
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'match_report_contracts'"
        ).fetchone()

    insert_sql = (
        "INSERT INTO prematch_events(record_id, match_id, match_version, team_id, player_id, "
        "event_type, occurred_at, known_at, confirmation_status, evidence_refs_json, "
        "can_modify_features) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                insert_sql,
                (
                    "fact:invalid-match-version",
                    match.id.value,
                    version.version + 99,
                    teams[0].id.value,
                    player.id.value,
                    "injury",
                    None,
                    "2026-07-16T02:00:00Z",
                    "official",
                    "[]",
                    1,
                ),
            )
    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                insert_sql,
                (
                    "fact:unbound-modifying-player",
                    match.id.value,
                    None,
                    teams[0].id.value,
                    player.id.value,
                    "injury",
                    None,
                    "2026-07-16T02:00:00Z",
                    "official",
                    "[]",
                    1,
                ),
            )
