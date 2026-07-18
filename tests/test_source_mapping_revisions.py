from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, TeamId
from football_data_platform.storage.canonical import (
    SCHEMA_VERSION,
    CanonicalConflictError,
    CanonicalStore,
    SourceMappingCASConflict,
)
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

NOW = datetime(2026, 7, 16, 2, 0, tzinfo=UTC)
COMPETITION_ID = CompetitionId("competition:eng.1")


@pytest.fixture
def mapping_store(tmp_path: Path) -> tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]]:
    layout = DataLayout(tmp_path / "data")
    asset = RawArchive(layout).archive(
        b"mapping revision fixture",
        source="fbref",
        source_id="mapping-revision",
        url="https://fbref.example/mapping-revision",
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
    teams = tuple(
        store.resolve_or_create_team(
            source="fbref",
            source_id=f"revision-team-{index}",
            canonical_name=f"Revision Team {index}",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=asset.id,
        ).id
        for index in range(3)
    )
    return store, teams  # type: ignore[return-value]


def test_same_target_proposal_is_idempotent_evidence_not_a_conflict(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    arguments = {
        "source": "fbref",
        "entity_type": "team",
        "source_id": "revision-team-0",
        "candidate_entity_id": teams[0],
        "proposed_at": NOW + timedelta(minutes=10),
        "actor": "resolver:test",
        "reason": "provider URL confirms the existing target",
        "evidence_refs": ("raw-asset:evidence-same-target",),
    }

    first = store.propose_source_mapping(**arguments)  # type: ignore[arg-type]
    replay = store.propose_source_mapping(**arguments)  # type: ignore[arg-type]

    assert first == replay
    assert first.entity_id == teams[0]
    assert first.mapping_id is not None
    assert len(store.source_mapping_evidence(mapping_id=first.mapping_id)) == 1
    assert store.list_source_mapping_conflicts() == ()


def test_conflict_revision_preserves_half_open_history_and_override_evidence(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    proposed_at = NOW + timedelta(minutes=30)
    effective_at = NOW + timedelta(hours=1)
    conflict = store.propose_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        candidate_entity_id=teams[1],
        proposed_at=proposed_at,
        actor="resolver:test",
        reason="fallback provider identifies the registered team",
        evidence_refs=("raw-asset:fallback-candidate",),
    )
    current = store.resolve_source_mapping(
        source="fbref", entity_type="team", source_id="revision-team-0"
    )
    assert conflict.current_mapping_id == current.mapping_id  # type: ignore[union-attr]
    assert store.list_source_mapping_conflicts() == (conflict,)

    replacement, decision = store.revise_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
        expected_current_mapping_id=current.mapping_id or "",
        candidate_entity_id=teams[1],
        effective_at=effective_at,
        actor="operator:alice",
        reason="manual review accepted the fallback identity",
        evidence_refs=("raw-asset:manual-review", "ticket:identity-42"),
    )

    history = store.mapping_history(source="fbref", entity_type="team", source_id="revision-team-0")
    assert [mapping.version for mapping in history] == [1, 2]
    assert history[0].valid_to == effective_at
    assert replacement.supersedes_mapping_id == history[0].mapping_id
    assert replacement.match_rule.value == "manual_override"
    assert replacement.created_by == "operator:alice"
    assert decision.previous_mapping_id == history[0].mapping_id
    assert decision.new_mapping_id == replacement.mapping_id
    assert (
        store.resolve_source_mapping(
            source="fbref",
            entity_type="team",
            source_id="revision-team-0",
            as_of=effective_at - timedelta(microseconds=1),
        ).entity_id
        == teams[0]
    )
    assert (
        store.resolve_source_mapping(
            source="fbref",
            entity_type="team",
            source_id="revision-team-0",
            as_of=effective_at,
        ).entity_id
        == teams[1]
    )
    assert (
        store.resolve_source_mapping(
            source="fbref", entity_type="team", source_id="revision-team-0", version=1
        )
        == history[0]
    )
    assert store.list_source_mapping_conflicts() == ()
    resolved = store.list_source_mapping_conflicts(include_resolved=True)
    assert resolved[0].decision_id == decision.decision_id
    assert {
        item.evidence_ref
        for item in store.source_mapping_evidence(mapping_id=replacement.mapping_id)
    } == {"raw-asset:manual-review", "ticket:identity-42"}
    assert set(decision.evidence_ids) == {
        item.evidence_id
        for item in store.source_mapping_evidence(mapping_id=replacement.mapping_id)
    }
    assert decision.revision_event_id is not None
    for table, primary_key, value in (
        ("source_mapping_conflicts", "conflict_id", conflict.conflict_id),  # type: ignore[union-attr]
        ("source_mapping_decisions", "decision_id", decision.decision_id),
        (
            "source_mapping_evidence",
            "evidence_id",
            store.source_mapping_evidence(mapping_id=replacement.mapping_id)[0].evidence_id,
        ),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.connect() as connection:
                connection.execute(
                    f"DELETE FROM {table} WHERE {primary_key} = ?",
                    (value,),
                )


def test_revision_rejects_stale_cas_and_unreviewed_candidate(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    conflict = store.propose_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        candidate_entity_id=teams[1],
        proposed_at=NOW + timedelta(minutes=1),
        actor="resolver:test",
        reason="candidate differs",
        evidence_refs=("raw-asset:candidate",),
    )
    stale_conflict = store.propose_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        candidate_entity_id=teams[2],
        proposed_at=NOW + timedelta(minutes=2),
        actor="resolver:test",
        reason="second candidate differs",
        evidence_refs=("raw-asset:second-candidate",),
    )
    current = store.resolve_source_mapping(
        source="fbref", entity_type="team", source_id="revision-team-0"
    )
    common = {
        "source": "fbref",
        "entity_type": "team",
        "source_id": "revision-team-0",
        "conflict_id": conflict.conflict_id,  # type: ignore[union-attr]
        "expected_current_mapping_id": current.mapping_id or "",
        "effective_at": NOW + timedelta(hours=1),
        "actor": "operator:alice",
        "reason": "reviewed override",
        "evidence_refs": ("ticket:review",),
    }

    with pytest.raises(CanonicalConflictError, match="reviewed candidate"):
        store.revise_source_mapping(candidate_entity_id=teams[2], **common)  # type: ignore[arg-type]

    accepted = store.revise_source_mapping(candidate_entity_id=teams[1], **common)  # type: ignore[arg-type]
    assert (
        store.revise_source_mapping(candidate_entity_id=teams[1], **common)  # type: ignore[arg-type]
        == accepted
    )
    assert store.list_source_mapping_conflicts() == ()
    assert len(store.list_source_mapping_conflicts(include_resolved=True)) == 2
    with pytest.raises(SourceMappingCASConflict, match="changed during review"):
        store.revise_source_mapping(
            **{
                **common,
                "conflict_id": stale_conflict.conflict_id,  # type: ignore[union-attr]
                "candidate_entity_id": teams[2],
            }
        )  # type: ignore[arg-type]


def test_unmapped_candidate_is_queued_and_can_be_accepted_as_version_one(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    proposed_at = NOW + timedelta(minutes=5)
    conflict = store.propose_source_mapping(
        source="scout",
        entity_type="team",
        source_id="unmapped-low-confidence",
        candidate_entity_id=teams[1],
        proposed_at=proposed_at,
        actor="resolver:test",
        reason="low-confidence alias candidate requires review",
        evidence_refs=("raw-asset:low-confidence-candidate",),
    )

    assert conflict.current_mapping_id is None  # type: ignore[union-attr]
    assert conflict.current_entity_id is None  # type: ignore[union-attr]
    assert store.list_source_mapping_conflicts() == (conflict,)

    accepted_at = proposed_at + timedelta(minutes=5)
    mapping, decision = store.revise_source_mapping(
        source="scout",
        entity_type="team",
        source_id="unmapped-low-confidence",
        conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
        expected_current_mapping_id=None,
        candidate_entity_id=teams[1],
        effective_at=accepted_at,
        actor="operator:alice",
        reason="review accepted the new provider identity",
        evidence_refs=("ticket:identity-new",),
    )

    assert mapping.version == 1
    assert mapping.valid_from == accepted_at
    assert mapping.supersedes_mapping_id is None
    assert decision.previous_mapping_id is None
    assert decision.evidence_ids == tuple(
        item.evidence_id for item in store.source_mapping_evidence(mapping_id=mapping.mapping_id)
    )
    assert (
        store.resolve_source_mapping(
            source="scout", entity_type="team", source_id="unmapped-low-confidence"
        )
        == mapping
    )
    assert store.list_source_mapping_conflicts() == ()


def test_manual_override_requires_evidence_before_any_mapping_is_closed(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    conflict = store.propose_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        candidate_entity_id=teams[1],
        proposed_at=NOW + timedelta(minutes=1),
        actor="resolver:test",
        reason="candidate differs",
        evidence_refs=("raw-asset:candidate",),
    )
    current = store.resolve_source_mapping(
        source="fbref", entity_type="team", source_id="revision-team-0"
    )

    with pytest.raises(ValueError, match="at least one evidence"):
        store.revise_source_mapping(
            source="fbref",
            entity_type="team",
            source_id="revision-team-0",
            conflict_id=conflict.conflict_id,  # type: ignore[union-attr]
            expected_current_mapping_id=current.mapping_id or "",
            candidate_entity_id=teams[1],
            effective_at=NOW + timedelta(hours=1),
            actor="operator:alice",
            reason="reviewed override",
            evidence_refs=(),
        )

    assert (
        store.resolve_source_mapping(
            source="fbref", entity_type="team", source_id="revision-team-0"
        )
        == current
    )
    assert store.list_source_mapping_conflicts() == (conflict,)


def test_sql_constraints_reject_type_time_overlap_and_mutation(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, _ = mapping_store
    current = store.resolve_source_mapping(
        source="fbref", entity_type="team", source_id="revision-team-0"
    )
    values = (
        "source-mapping:sql-attack",
        "attack-source",
        "player",
        "attack-id",
        current.entity_id.value,
        1,
        "2026-07-16T02:00:00.000000Z",
        None,
        "source_id",
        1.0,
        "2026-07-16T02:00:00.000000Z",
        "attacker",
        "invalid mapping",
        None,
    )
    insert_sql = (
        "INSERT INTO source_mappings(mapping_id, source, entity_type, source_id, entity_id, "
        "version, valid_from, valid_to, match_rule, confidence, created_at, created_by, "
        "audit_note, supersedes_mapping_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError, match="entity type mismatch"):
        with store.connect() as connection:
            connection.execute(insert_sql, values)

    with pytest.raises(sqlite3.IntegrityError, match="revision event|CHECK constraint"):
        with store.connect() as connection:
            connection.execute(
                insert_sql,
                (
                    "source-mapping:time-attack",
                    "attack-source",
                    "team",
                    "invalid-time",
                    current.entity_id.value,
                    1,
                    "not-a-utc-timestamp",
                    None,
                    "source_id",
                    1.0,
                    "2026-07-16T02:00:00.000000Z",
                    "attacker",
                    "invalid timestamp",
                    None,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="revision event|CHECK constraint"):
        with store.connect() as connection:
            connection.execute(
                "UPDATE source_mappings SET valid_to = valid_from WHERE mapping_id = ?",
                (current.mapping_id,),
            )

    with pytest.raises(sqlite3.IntegrityError, match="revision event|overlap|continuous"):
        with store.connect() as connection:
            connection.execute(
                insert_sql,
                (
                    "source-mapping:overlap-attack",
                    "fbref",
                    "team",
                    "revision-team-0",
                    current.entity_id.value,
                    2,
                    "2026-07-16T02:30:00.000000Z",
                    None,
                    "source_id",
                    1.0,
                    "2026-07-16T02:30:00.000000Z",
                    "attacker",
                    "overlapping mapping",
                    current.mapping_id,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="may only close"):
        with store.connect() as connection:
            connection.execute(
                "UPDATE source_mappings SET entity_id = ? WHERE mapping_id = ?",
                (values[3], current.mapping_id),
            )


def test_sql_layer_requires_one_evidenced_revision_event(
    mapping_store: tuple[CanonicalStore, tuple[TeamId, TeamId, TeamId]],
) -> None:
    store, teams = mapping_store
    conflict = store.propose_source_mapping(
        source="fbref",
        entity_type="team",
        source_id="revision-team-0",
        candidate_entity_id=teams[1],
        proposed_at=NOW + timedelta(minutes=1),
        actor="resolver:test",
        reason="candidate differs",
        evidence_refs=("raw-asset:candidate",),
    )
    current = store.resolve_source_mapping(
        source="fbref", entity_type="team", source_id="revision-team-0"
    )
    effective_at = "2026-07-16T03:00:00.000000Z"

    with pytest.raises(sqlite3.IntegrityError, match="revision event"):
        with store.connect() as connection:
            connection.execute(
                "UPDATE source_mappings SET valid_to = ? WHERE mapping_id = ?",
                (effective_at, current.mapping_id),
            )

    with pytest.raises(sqlite3.IntegrityError, match="revision event"):
        with store.connect() as connection:
            connection.execute(
                "INSERT INTO source_mappings("
                "mapping_id, source, entity_type, source_id, entity_id, version, valid_from, "
                "valid_to, match_rule, confidence, created_at, created_by, audit_note, "
                "supersedes_mapping_id) VALUES (?, ?, ?, ?, ?, 1, ?, NULL, "
                "'manual_override', 1.0, ?, ?, ?, NULL)",
                (
                    "source-mapping:manual-without-event",
                    "attacker",
                    "team",
                    "manual-without-event",
                    teams[1].value,
                    effective_at,
                    effective_at,
                    "attacker",
                    "manual mapping without evidence",
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="revision event"):
        with store.connect() as connection:
            connection.execute(
                "INSERT INTO source_mapping_decisions("
                "decision_id, revision_event_id, conflict_id, previous_mapping_id, "
                "new_mapping_id, decided_at, decided_by, reason, evidence_ids_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "source-mapping-decision:attack",
                    "source-mapping-revision-event:missing",
                    conflict.conflict_id,  # type: ignore[union-attr]
                    current.mapping_id,
                    current.mapping_id,
                    effective_at,
                    "attacker",
                    "decision without evidence",
                    "[]",
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="at least one evidence"):
        with store.connect() as connection:
            connection.execute(
                "INSERT INTO source_mapping_revision_events("
                "revision_event_id, decision_id, conflict_id, source, entity_type, source_id, "
                "expected_current_mapping_id, candidate_entity_id, new_mapping_id, new_version, "
                "effective_at, actor, reason, evidence_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "source-mapping-revision-event:no-evidence",
                    "source-mapping-decision:no-evidence",
                    conflict.conflict_id,  # type: ignore[union-attr]
                    "fbref",
                    "team",
                    "revision-team-0",
                    current.mapping_id,
                    teams[1].value,
                    "source-mapping:no-evidence",
                    2,
                    effective_at,
                    "attacker",
                    "manual override without evidence",
                    "[]",
                ),
            )

    assert (
        store.resolve_source_mapping(
            source="fbref", entity_type="team", source_id="revision-team-0"
        )
        == current
    )
    assert store.list_source_mapping_conflicts() == (conflict,)


def test_v6_migration_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "canonical-v6.sqlite3"
    _create_v6_mapping_database(path, overlap=False, gap=False, type_mismatch=False)
    store = CanonicalStore(path)

    store.initialize()
    first = store.mapping_history(source="legacy", entity_type="team", source_id="legacy-team")
    store.initialize()
    replay = store.mapping_history(source="legacy", entity_type="team", source_id="legacy-team")

    assert first == replay
    assert [item.version for item in first] == [1, 2]
    assert first[1].supersedes_mapping_id == first[0].mapping_id
    assert {item.created_by for item in first} == {"legacy-v6-migration"}
    with store.connect() as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == SCHEMA_VERSION
        )
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'source_mapping_decisions'"
        ).fetchone()


@pytest.mark.parametrize(
    ("overlap", "gap", "type_mismatch", "message"),
    (
        (True, False, False, "contiguous"),
        (False, True, False, "contiguous"),
        (False, False, True, "entity type mismatch"),
    ),
)
def test_invalid_v6_migration_fails_atomically(
    tmp_path: Path,
    overlap: bool,
    gap: bool,
    type_mismatch: bool,
    message: str,
) -> None:
    path = tmp_path / f"invalid-v6-{overlap}-{gap}-{type_mismatch}.sqlite3"
    _create_v6_mapping_database(
        path,
        overlap=overlap,
        gap=gap,
        type_mismatch=type_mismatch,
    )

    with pytest.raises(CanonicalConflictError, match=message):
        CanonicalStore(path).initialize()

    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == 6
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(source_mappings)")}
        assert "mapping_id" not in columns
        assert connection.execute("SELECT COUNT(*) FROM source_mappings").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'teams'"
            ).fetchone()
            is None
        )


def test_existing_v7_validation_rejects_gap_atomically(tmp_path: Path) -> None:
    path = tmp_path / "invalid-existing-v7-gap.sqlite3"
    _create_v6_mapping_database(path, overlap=False, gap=False, type_mismatch=False)
    store = CanonicalStore(path)
    store.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER source_mappings_close_only_update")
        connection.execute(
            "UPDATE source_mappings SET valid_to = '2026-07-16T02:30:00.000000Z' WHERE version = 1"
        )
        connection.execute("UPDATE schema_meta SET version = 6 WHERE singleton = 1")

    with pytest.raises(CanonicalConflictError, match="malformed v7.*history"):
        store.initialize()

    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT version FROM schema_meta WHERE singleton = 1").fetchone()[0]
            == 6
        )
        assert (
            connection.execute("SELECT valid_to FROM source_mappings WHERE version = 1").fetchone()[
                0
            ]
            == "2026-07-16T02:30:00.000000Z"
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                "AND name = 'source_mappings_close_only_update'"
            ).fetchone()
            is None
        )


def _create_v6_mapping_database(
    path: Path,
    *,
    overlap: bool,
    gap: bool,
    type_mismatch: bool,
) -> None:
    if overlap:
        first_valid_to = "2026-07-16T04:00:00Z"
    elif gap:
        first_valid_to = "2026-07-16T02:30:00Z"
    else:
        first_valid_to = "2026-07-16T03:00:00Z"
    entity_type = "player" if type_mismatch else "team"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            );
            INSERT INTO schema_meta(singleton, version) VALUES (1, 6);
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO entities VALUES (
                'team:legacy', 'team', '2026-07-16T02:00:00Z'
            );
            CREATE TABLE source_mappings (
                source TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                entity_id TEXT NOT NULL REFERENCES entities(entity_id),
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                match_rule TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                created_at TEXT NOT NULL,
                audit_note TEXT NOT NULL,
                PRIMARY KEY (source, entity_type, source_id, valid_from)
            );
            """
        )
        connection.executemany(
            "INSERT INTO source_mappings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    "legacy",
                    entity_type,
                    "legacy-team",
                    "team:legacy",
                    "2026-07-16T02:00:00Z",
                    first_valid_to,
                    "source_id",
                    1.0,
                    "2026-07-16T02:00:00Z",
                    "legacy first mapping",
                ),
                (
                    "legacy",
                    entity_type,
                    "legacy-team",
                    "team:legacy",
                    "2026-07-16T03:00:00Z",
                    None,
                    "manual_override",
                    1.0,
                    "2026-07-16T03:00:00Z",
                    "legacy current mapping",
                ),
            ),
        )
