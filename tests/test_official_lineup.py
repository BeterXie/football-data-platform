from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import football_data_platform.storage.canonical as canonical_module
import football_data_platform.storage.facts as facts_module
from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, PlayerId, SeasonId
from football_data_platform.domain.models import MatchStatus
from football_data_platform.pipelines.official_lineup import (
    OfficialLineupIngestError,
    OfficialLineupIngestResult,
    ingest_official_lineup_json,
    replay_official_lineup_contract,
)
from football_data_platform.sources.official_lineup import (
    OFFICIAL_LINEUP_PARSER_VERSION,
    parse_official_lineup_json,
)
from football_data_platform.sources.prematch import (
    OfficialLineupDTO,
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import (
    ArchiveConflictError,
    ChecksumMismatchError,
    RawArchive,
)

ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
KICKOFF = NOW + timedelta(hours=1)
SOURCE = "club-official"
SOURCE_MATCH_ID = "official-fixture-1"
PAGE_URL = "https://club.example/matches/official-fixture-1/lineup"


def _payload(
    *,
    source: str = SOURCE,
    source_match_id: str = SOURCE_MATCH_ID,
    home_team_id: str = "official-home",
    away_team_id: str = "official-away",
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source": source,
        "source_match_id": source_match_id,
        "match_mapping_source": "fbref-schedule",
        "team_mapping_source": "fbref",
        "player_mapping_source": source,
        "published_at": "2026-07-16T07:55:00Z",
        "teams": [
            {
                "source_team_id": home_team_id,
                "starters": [
                    {"source_player_id": f"home-{index:02d}", "name": f"Home {index:02d}"}
                    for index in range(1, 12)
                ],
            },
            {
                "source_team_id": away_team_id,
                "starters": [
                    {"source_player_id": f"away-{index:02d}", "name": f"Away {index:02d}"}
                    for index in range(1, 12)
                ],
            },
        ],
    }


def _content(**overrides: object) -> bytes:
    return json.dumps({**_payload(), **overrides}, sort_keys=True).encode("utf-8")


def _context(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    canonical.register_registry(registry, registered_at=NOW)
    schedule_asset = archive.archive(
        b"registered schedule evidence",
        source="fbref",
        source_id="schedule",
        url="https://fbref.example/schedule",
        observed_at=NOW - timedelta(minutes=30),
        target_event_time=None,
        collector_version="test-schedule/1",
        media_type="text/html",
    )
    canonical.register_raw_asset(schedule_asset)
    teams = tuple(
        canonical.resolve_or_create_team(
            source="fbref",
            source_id=source_team_id,
            canonical_name=name,
            competition_id=CompetitionId("competition:eng.1"),
            observed_at=NOW - timedelta(minutes=30),
            raw_asset_id=schedule_asset.id,
        )
        for source_team_id, name in (
            ("official-home", "Official Home"),
            ("official-away", "Official Away"),
        )
    )
    match, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id=SOURCE_MATCH_ID,
        competition_id=CompetitionId("competition:eng.1"),
        season_id=SeasonId("season:eng.1.2025-26"),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=KICKOFF,
        status=MatchStatus.SCHEDULED,
        observed_at=NOW - timedelta(minutes=30),
        raw_asset_id=schedule_asset.id,
    )
    sources = SourceRegistry(
        (
            SourceDescriptor(
                SOURCE,
                SourceKind.OFFICIAL_LINEUP,
                SOURCE,
                official=True,
                allowed_hosts=("club.example",),
            ),
        )
    )
    return layout, archive, canonical, sources, teams, match, version


def _ingest(tmp_path: Path, *, content: bytes | None = None):
    layout, archive, canonical, sources, teams, match, version = _context(tmp_path)
    result = ingest_official_lineup_json(
        content or _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW,
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )
    return layout, archive, canonical, sources, teams, match, version, result


def test_official_lineup_parser_has_strict_versioned_source_contract() -> None:
    parsed = parse_official_lineup_json(_content())

    assert parsed.parser_version == OFFICIAL_LINEUP_PARSER_VERSION
    assert parsed.source == SOURCE
    assert parsed.source_match_id == SOURCE_MATCH_ID
    assert len(parsed.teams) == 2
    assert all(len(team.starters) == 11 for team in parsed.teams)

    duplicate = _payload()
    duplicate["teams"][1]["starters"][0]["source_player_id"] = "home-01"  # type: ignore[index]
    with pytest.raises(ValueError, match="unique across both teams"):
        parse_official_lineup_json(json.dumps(duplicate).encode())

    unsupported = _payload()
    unsupported["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        parse_official_lineup_json(json.dumps(unsupported).encode())


def test_ingest_writes_paired_xi_and_content_addressed_contract_atomically(
    tmp_path: Path,
) -> None:
    _, archive, canonical, sources, teams, match, version, first = _ingest(tmp_path)
    replayed = ingest_official_lineup_json(
        _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW,
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )

    assert replayed == first
    assert first.contract_id.startswith("official-lineup-contract:")
    assert len(first.fact_ids) == 22
    assert first.match_id == match.id
    assert first.match_version == version.version
    contract = canonical.official_lineup_contract(first.contract_id)
    assert contract.contract_version == 2
    assert len(contract.source_bindings) == 22
    assert {
        (binding.source_team_id, binding.source_player_id) for binding in contract.source_bindings
    } == {
        (source_team_id, f"{side}-{index:02d}")
        for source_team_id, side in (
            ("official-home", "home"),
            ("official-away", "away"),
        )
        for index in range(1, 12)
    }
    with canonical.connect() as connection:
        rows = connection.execute(
            "SELECT team_id, player_id, official, raw_asset_id FROM lineup_facts "
            "WHERE match_id = ? ORDER BY team_id, player_id",
            (match.id.value,),
        ).fetchall()
        contract_count = connection.execute(
            "SELECT COUNT(*) FROM official_lineup_contracts"
        ).fetchone()[0]
    assert len(rows) == 22
    assert {row["team_id"] for row in rows} == {team.id.value for team in teams}
    assert {row["official"] for row in rows} == {1}
    assert {row["raw_asset_id"] for row in rows} == {first.raw_asset_id}
    assert contract_count == 1

    verified = replay_official_lineup_contract(
        first.contract_id,
        archive=archive,
        canonical=canonical,
    )
    assert verified.contract_id == first.contract_id


def test_repeated_observation_replays_stable_facts_with_exact_evidence(
    tmp_path: Path,
) -> None:
    _, archive, canonical, sources, _, _, _ = _context(tmp_path)
    first = ingest_official_lineup_json(
        _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW,
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )
    second = ingest_official_lineup_json(
        _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW + timedelta(minutes=5),
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )

    assert first.contract_id != second.contract_id
    assert first.fact_ids == second.fact_ids
    assert (
        replay_official_lineup_contract(
            second.contract_id,
            archive=archive,
            canonical=canonical,
        ).contract_id
        == second.contract_id
    )
    with canonical.connect() as connection:
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM fact_evidence WHERE raw_asset_id = ? AND observed_at = ?",
            (
                second.raw_asset_id,
                (NOW + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            ),
        ).fetchone()[0]
    assert evidence_count == 22


def test_verified_fact_store_has_no_caller_lineup_bypass(tmp_path: Path) -> None:
    _, _, canonical, sources, _, _, _ = _context(tmp_path)
    facts = CanonicalFactStore(canonical, source_registry=sources)

    assert not hasattr(facts, "_append_verified_official_lineups")


def test_parser_failure_is_archived_before_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, archive, canonical, sources, _, _, _ = _context(tmp_path)

    def fail_parser(_content: bytes) -> None:
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(
        facts_module,
        "parse_official_lineup_json",
        fail_parser,
        raising=False,
    )
    with pytest.raises(OfficialLineupIngestError) as caught:
        ingest_official_lineup_json(
            _content(),
            source=SOURCE,
            source_match_id=SOURCE_MATCH_ID,
            page_url=PAGE_URL,
            observed_at=NOW,
            archive=archive,
            canonical=canonical,
            source_registry=sources,
        )

    archive.verify(caught.value.raw_asset_id)


@pytest.mark.parametrize(
    "tamper",
    (
        "contract-delete",
        "contract-update",
        "raw",
        "fact",
        "fact-swap",
        "fact-provenance",
        "evidence",
        "source-player-mapping-swap",
    ),
)
def test_derived_official_source_write_rejects_tampered_contract_chain(
    tmp_path: Path,
    tamper: str,
) -> None:
    layout, archive, canonical, _, _, _, _, result = _ingest(tmp_path)
    contract = canonical.official_lineup_contract(result.contract_id)
    _tamper_contract_chain(
        tamper,
        layout=layout,
        archive=archive,
        canonical=canonical,
        result=result,
    )
    team_id, player_ids = contract.team_lineups[0]

    with pytest.raises(ArchiveConflictError):
        DerivedArchive(layout).write_official_lineup_source(
            contract_id=result.contract_id,
            match_id=contract.match_id,
            match_version=contract.match_version,
            team_id=team_id,
            player_ids=player_ids,
            known_at=contract.published_at,
            observed_at=contract.observed_at,
            raw_asset_id=contract.raw_asset_id,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "contract-delete",
        "contract-update",
        "raw",
        "fact",
        "fact-swap",
        "fact-provenance",
        "evidence",
        "source-player-mapping-swap",
    ),
)
def test_derived_official_source_load_revalidates_contract_chain(
    tmp_path: Path,
    tamper: str,
) -> None:
    layout, archive, canonical, _, _, _, _, result = _ingest(tmp_path)
    contract = canonical.official_lineup_contract(result.contract_id)
    team_id, player_ids = contract.team_lineups[0]
    derived = DerivedArchive(layout)
    source_ref = derived.write_official_lineup_source(
        contract_id=result.contract_id,
        match_id=contract.match_id,
        match_version=contract.match_version,
        team_id=team_id,
        player_ids=player_ids,
        known_at=contract.published_at,
        observed_at=contract.observed_at,
        raw_asset_id=contract.raw_asset_id,
    )
    validation = derived.validate_snapshot_source(source_ref)
    assert validation.source_context is not None
    assert validation.source_context["contract_id"] == result.contract_id
    _tamper_contract_chain(
        tamper,
        layout=layout,
        archive=archive,
        canonical=canonical,
        result=result,
    )

    with pytest.raises(ArchiveConflictError):
        derived.validate_snapshot_source(source_ref)


def _tamper_contract_chain(
    tamper: str,
    *,
    layout: DataLayout,
    archive: RawArchive,
    canonical: CanonicalStore,
    result: OfficialLineupIngestResult,
) -> None:
    if tamper == "raw":
        asset = archive.load(archive_id_from(result.raw_asset_id))
        layout.raw_object_path(asset.checksum).write_bytes(b"tampered")
        return
    with canonical.connect() as connection:
        if tamper == "contract-delete":
            connection.execute(
                "DELETE FROM official_lineup_contracts WHERE contract_id = ?",
                (result.contract_id,),
            )
        elif tamper == "contract-update":
            connection.execute(
                "UPDATE official_lineup_contracts SET parser_version = 'forged' "
                "WHERE contract_id = ?",
                (result.contract_id,),
            )
        elif tamper == "fact":
            connection.execute(
                "DELETE FROM lineup_facts WHERE record_id = ?",
                (result.fact_ids[0],),
            )
        elif tamper == "fact-swap":
            rows = connection.execute(
                "SELECT record_id, player_id FROM lineup_facts ORDER BY team_id, player_id LIMIT 2"
            ).fetchall()
            temporary = canonical._resolve_or_create_player(
                connection,
                source=SOURCE,
                source_id="tamper-temporary-player",
                canonical_name="Tamper Temporary Player",
                observed_at=NOW,
                raw_asset_id=archive_id_from(result.raw_asset_id),
            )
            connection.execute(
                "UPDATE lineup_facts SET player_id = ? WHERE record_id = ?",
                (temporary.id.value, rows[0]["record_id"]),
            )
            connection.execute(
                "UPDATE lineup_facts SET player_id = ? WHERE record_id = ?",
                (rows[0]["player_id"], rows[1]["record_id"]),
            )
            connection.execute(
                "UPDATE lineup_facts SET player_id = ? WHERE record_id = ?",
                (rows[1]["player_id"], rows[0]["record_id"]),
            )
        elif tamper == "fact-provenance":
            connection.execute(
                "UPDATE lineup_facts SET observed_at = ? WHERE record_id = ?",
                (
                    (NOW + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                    result.fact_ids[0],
                ),
            )
        elif tamper == "evidence":
            connection.execute(
                "DELETE FROM fact_evidence WHERE record_id = ? AND raw_asset_id = ?",
                (result.fact_ids[0], result.raw_asset_id),
            )
        elif tamper == "source-player-mapping-swap":
            rows = connection.execute(
                "SELECT source_id, entity_id FROM source_mappings "
                "WHERE source = ? AND entity_type = 'player' "
                "AND source_id IN ('home-01', 'home-02') AND valid_to IS NULL "
                "ORDER BY source_id",
                (SOURCE,),
            ).fetchall()
            assert len(rows) == 2
            connection.execute(
                "UPDATE source_mappings SET entity_id = ? "
                "WHERE source = ? AND entity_type = 'player' AND source_id = ? "
                "AND valid_to IS NULL",
                (rows[1]["entity_id"], SOURCE, rows[0]["source_id"]),
            )
            connection.execute(
                "UPDATE source_mappings SET entity_id = ? "
                "WHERE source = ? AND entity_type = 'player' AND source_id = ? "
                "AND valid_to IS NULL",
                (rows[0]["entity_id"], SOURCE, rows[1]["source_id"]),
            )
        else:
            raise AssertionError(f"unsupported tamper: {tamper}")


def test_old_dto_cannot_write_official_facts_without_raw_contract(tmp_path: Path) -> None:
    _, _, canonical, sources, teams, match, version = _context(tmp_path)
    facts = CanonicalFactStore(canonical, source_registry=sources)
    players = tuple(PlayerId(f"player:forged-{index}") for index in range(11))

    with pytest.raises(ValueError, match="raw replay contract"):
        facts.append_official_lineup(
            OfficialLineupDTO(
                match_id=match.id,
                match_version=version.version,
                team_id=teams[0].id,
                player_ids=players,
                source=SOURCE,
                published_at=NOW - timedelta(minutes=5),
                observed_at=NOW,
                raw_asset_id=archive_id(),
                url=PAGE_URL,
            )
        )


def archive_id():
    from football_data_platform.domain.ids import RawAssetId

    return RawAssetId("raw-asset:" + "f" * 64)


def test_second_team_failure_rolls_back_both_official_lineups(tmp_path: Path) -> None:
    _, archive, canonical, sources, teams, match, version = _context(tmp_path)
    conflict_asset = archive.archive(
        b"conflicting assignment",
        source="test",
        source_id="conflict",
        url="fixture://conflict",
        observed_at=NOW - timedelta(minutes=1),
        target_event_time=None,
        collector_version="test/1",
        media_type="text/plain",
    )
    canonical.register_raw_asset(conflict_asset)
    player = canonical.resolve_or_create_player(
        source=SOURCE,
        source_id="away-11",
        canonical_name="Away 11",
        observed_at=NOW - timedelta(minutes=1),
        raw_asset_id=conflict_asset.id,
    )
    CanonicalFactStore(canonical).append_lineup_fact(
        match_id=match.id,
        match_version=version.version,
        team_id=teams[0].id,
        player_id=player.id,
        lineup_role="bench",
        official=False,
        known_at=NOW - timedelta(minutes=1),
        observed_at=NOW - timedelta(minutes=1),
        raw_asset_id=conflict_asset.id,
    )

    with pytest.raises(ValueError, match="already assigned"):
        ingest_official_lineup_json(
            _content(),
            source=SOURCE,
            source_match_id=SOURCE_MATCH_ID,
            page_url=PAGE_URL,
            observed_at=NOW,
            archive=archive,
            canonical=canonical,
            source_registry=sources,
        )

    with canonical.connect() as connection:
        official_count = connection.execute(
            "SELECT COUNT(*) FROM lineup_facts WHERE official = 1"
        ).fetchone()[0]
        contract_count = connection.execute(
            "SELECT COUNT(*) FROM official_lineup_contracts"
        ).fetchone()[0]
        player_count = connection.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        mapping_count = connection.execute(
            "SELECT COUNT(*) FROM source_mappings WHERE source = ? AND entity_type = 'player'",
            (SOURCE,),
        ).fetchone()[0]
    assert official_count == 0
    assert contract_count == 0
    assert player_count == 1
    assert mapping_count == 1


@pytest.mark.parametrize("mismatch", ("source", "match", "team"))
def test_raw_identity_mismatch_is_archived_but_writes_no_canonical_xi(
    tmp_path: Path,
    mismatch: str,
) -> None:
    _, archive, canonical, sources, _, _, _ = _context(tmp_path)
    payload = _payload()
    if mismatch == "source":
        payload["source"] = "other-official"
    elif mismatch == "match":
        payload["source_match_id"] = "other-fixture"
    else:
        payload["teams"][1]["source_team_id"] = "unknown-team"  # type: ignore[index]

    with pytest.raises(OfficialLineupIngestError) as caught:
        ingest_official_lineup_json(
            json.dumps(payload).encode(),
            source=SOURCE,
            source_match_id=SOURCE_MATCH_ID,
            page_url=PAGE_URL,
            observed_at=NOW,
            archive=archive,
            canonical=canonical,
            source_registry=sources,
        )

    archive.verify(caught.value.raw_asset_id)
    with canonical.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM lineup_facts WHERE official = 1").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM official_lineup_contracts").fetchone()[0] == 0
        )
        if mismatch == "team":
            assert connection.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 0
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM source_mappings WHERE source = ? "
                    "AND entity_type = 'player'",
                    (SOURCE,),
                ).fetchone()[0]
                == 0
            )


def test_replay_rejects_raw_or_canonical_xi_tampering(tmp_path: Path) -> None:
    layout, archive, canonical, _, _, _, _, result = _ingest(tmp_path)
    asset = archive.load(archive_id_from(result.raw_asset_id))
    object_path = layout.raw_object_path(asset.checksum)
    original = object_path.read_bytes()
    object_path.write_bytes(b"tampered")
    with pytest.raises(ChecksumMismatchError):
        replay_official_lineup_contract(
            result.contract_id,
            archive=archive,
            canonical=canonical,
        )
    object_path.write_bytes(original)

    with canonical.connect() as connection:
        connection.execute(
            "DELETE FROM lineup_facts WHERE record_id = ?",
            (result.fact_ids[0],),
        )
    with pytest.raises(ValueError, match="canonical official lineup facts"):
        replay_official_lineup_contract(
            result.contract_id,
            archive=archive,
            canonical=canonical,
        )


def test_replay_rejects_swapped_fact_identities(tmp_path: Path) -> None:
    layout, archive, canonical, _, _, _, _, result = _ingest(tmp_path)
    _tamper_contract_chain(
        "fact-swap",
        layout=layout,
        archive=archive,
        canonical=canonical,
        result=result,
    )

    with pytest.raises(ValueError, match="canonical official lineup facts"):
        replay_official_lineup_contract(
            result.contract_id,
            archive=archive,
            canonical=canonical,
        )


def test_write_load_and_replay_reject_swapped_source_player_mappings(
    tmp_path: Path,
) -> None:
    layout, archive, canonical, _, _, _, _, result = _ingest(tmp_path)
    contract = canonical.official_lineup_contract(result.contract_id)
    team_id, player_ids = contract.team_lineups[0]
    derived = DerivedArchive(layout)
    source_ref = derived.write_official_lineup_source(
        contract_id=result.contract_id,
        match_id=contract.match_id,
        match_version=contract.match_version,
        team_id=team_id,
        player_ids=player_ids,
        known_at=contract.published_at,
        observed_at=contract.observed_at,
        raw_asset_id=contract.raw_asset_id,
    )
    _tamper_contract_chain(
        "source-player-mapping-swap",
        layout=layout,
        archive=archive,
        canonical=canonical,
        result=result,
    )

    with pytest.raises(ValueError, match="player mappings"):
        replay_official_lineup_contract(
            result.contract_id,
            archive=archive,
            canonical=canonical,
        )
    with pytest.raises(ArchiveConflictError):
        derived.write_official_lineup_source(
            contract_id=result.contract_id,
            match_id=contract.match_id,
            match_version=contract.match_version,
            team_id=team_id,
            player_ids=player_ids,
            known_at=contract.published_at,
            observed_at=contract.observed_at,
            raw_asset_id=contract.raw_asset_id,
        )
    with pytest.raises(ArchiveConflictError):
        derived.validate_snapshot_source(source_ref)


def test_contract_content_id_rejects_swapped_source_player_bindings(
    tmp_path: Path,
) -> None:
    _, _, canonical, _, _, _, _, result = _ingest(tmp_path)
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT source_bindings_json FROM official_lineup_contracts WHERE contract_id = ?",
            (result.contract_id,),
        ).fetchone()
        bindings = json.loads(row["source_bindings_json"])
        home_bindings = [
            binding for binding in bindings if binding["source_team_id"] == "official-home"
        ]
        home_bindings[0]["player_id"], home_bindings[1]["player_id"] = (
            home_bindings[1]["player_id"],
            home_bindings[0]["player_id"],
        )
        connection.execute(
            "UPDATE official_lineup_contracts SET source_bindings_json = ? WHERE contract_id = ?",
            (canonical_module._json_text(bindings), result.contract_id),
        )

    with pytest.raises(canonical_module.CanonicalConflictError, match="content ID"):
        canonical.official_lineup_contract(result.contract_id)


def _restore_v5_official_contract_schema(
    canonical: CanonicalStore,
    *,
    current_contract_id: str,
    legacy_contract_id: str,
) -> None:
    with canonical.connect() as connection:
        connection.execute(
            "UPDATE official_lineup_contracts SET contract_id = ?, contract_version = 1, "
            "source_bindings_json = NULL WHERE contract_id = ?",
            (legacy_contract_id, current_contract_id),
        )
        connection.execute(
            """
            CREATE TABLE official_lineup_contracts_v5 (
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
            )
            """
        )
        connection.execute(
            "INSERT INTO official_lineup_contracts_v5("
            "contract_id, raw_asset_id, source, source_match_id, match_mapping_source, "
            "team_mapping_source, player_mapping_source, match_id, match_version, "
            "parser_version, published_at, observed_at, team_lineups_json, fact_ids_json) "
            "SELECT contract_id, raw_asset_id, source, source_match_id, "
            "match_mapping_source, team_mapping_source, player_mapping_source, match_id, "
            "match_version, parser_version, published_at, observed_at, team_lineups_json, "
            "fact_ids_json FROM official_lineup_contracts"
        )
        connection.execute("DROP TABLE official_lineup_contracts")
        connection.execute(
            "ALTER TABLE official_lineup_contracts_v5 RENAME TO official_lineup_contracts"
        )
        connection.execute(
            "CREATE INDEX official_lineup_contracts_match "
            "ON official_lineup_contracts(match_id, match_version)"
        )
        connection.execute("UPDATE schema_meta SET version = 5 WHERE singleton = 1")


def test_v5_legacy_contract_migrates_and_coexists_with_replayable_v2(
    tmp_path: Path,
) -> None:
    _, archive, canonical, sources, _, _, _, result = _ingest(tmp_path)
    contract = canonical.official_lineup_contract(result.contract_id)
    legacy_id = canonical_module._official_lineup_contract_id(
        contract_version=1,
        raw_asset_id=contract.raw_asset_id,
        source=contract.source,
        source_match_id=contract.source_match_id,
        match_mapping_source=contract.match_mapping_source,
        team_mapping_source=contract.team_mapping_source,
        player_mapping_source=contract.player_mapping_source,
        match_id=contract.match_id,
        match_version=contract.match_version,
        parser_version=contract.parser_version,
        published_at=contract.published_at,
        observed_at=contract.observed_at,
        team_lineups=contract.team_lineups,
        source_bindings=(),
        fact_ids=contract.fact_ids,
    )
    _restore_v5_official_contract_schema(
        canonical,
        current_contract_id=result.contract_id,
        legacy_contract_id=legacy_id,
    )
    canonical.initialize()
    canonical.initialize()

    legacy = canonical.official_lineup_contract(legacy_id)
    assert legacy.contract_version == 1
    assert legacy.source_bindings == ()
    with pytest.raises(ValueError, match="legacy.*source-player bindings"):
        replay_official_lineup_contract(
            legacy_id,
            archive=archive,
            canonical=canonical,
        )

    v2 = ingest_official_lineup_json(
        _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW,
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )
    assert v2.contract_id == result.contract_id
    assert (
        replay_official_lineup_contract(
            v2.contract_id,
            archive=archive,
            canonical=canonical,
        ).contract_id
        == v2.contract_id
    )

    repeated = ingest_official_lineup_json(
        _content(),
        source=SOURCE,
        source_match_id=SOURCE_MATCH_ID,
        page_url=PAGE_URL,
        observed_at=NOW,
        archive=archive,
        canonical=canonical,
        source_registry=sources,
    )
    with canonical.connect() as connection:
        rows = connection.execute(
            "SELECT contract_id, contract_version FROM official_lineup_contracts "
            "ORDER BY contract_version"
        ).fetchall()
        foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert repeated.contract_id == v2.contract_id
    assert [tuple(row) for row in rows] == [(legacy_id, 1), (v2.contract_id, 2)]
    assert foreign_key_violations == []


def archive_id_from(value: str):
    from football_data_platform.domain.ids import RawAssetId

    return RawAssetId(value)
