from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import (
    CompetitionId,
    MatchId,
    PlayerId,
    RawAssetId,
    SeasonId,
    TeamId,
)
from football_data_platform.domain.models import CollectionAttemptOutcome, MatchStatus
from football_data_platform.sources.prematch import (
    NewsEvidenceDTO,
    OfficialLineupDTO,
    PrematchEventDTO,
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
    classify_confirmation,
)
from football_data_platform.storage.canonical import CanonicalConflictError, CanonicalStore
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

NOW = datetime(2026, 7, 16, 5, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]
COMPETITION_ID = CompetitionId("competition:eng.1")
SEASON_ID = SeasonId("season:eng.1.2025-26")


def _context(tmp_path: Path):
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    canonical.register_registry(registry, registered_at=NOW)
    assets = {}
    for name, source in (
        ("a", "wire-a"),
        ("b", "wire-b"),
        ("official", "club-official"),
        ("unknown", "unknown-blog"),
    ):
        asset = archive.archive(
            f"{name} evidence".encode(),
            source=source,
            source_id=f"{name}-1",
            url=f"https://{source}.example/{name}",
            observed_at=NOW,
            target_event_time=NOW - timedelta(days=1),
            collector_version="test/1",
            media_type="text/html",
        )
        canonical.register_raw_asset(asset)
        assets[name] = asset
    teams = [
        canonical.resolve_or_create_team(
            source="test",
            source_id=f"team-{index}",
            canonical_name=f"Team {index}",
            competition_id=COMPETITION_ID,
            observed_at=NOW,
            raw_asset_id=assets["a"].id,
        )
        for index in range(2)
    ]
    match, _ = canonical.resolve_or_create_match(
        source="test",
        source_id="fixture-1",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    source_registry = SourceRegistry(
        (
            SourceDescriptor("wire-a", SourceKind.NEWS, "wire-a"),
            SourceDescriptor("wire-b", SourceKind.NEWS, "wire-b"),
            SourceDescriptor(
                "club-official",
                SourceKind.OFFICIAL_LINEUP,
                "club",
                official=True,
            ),
        )
    )
    facts = CanonicalFactStore(canonical, source_registry=source_registry)
    return canonical, facts, assets, teams, match


def _player_event_subject(canonical, facts, assets, *, suffix: str):
    player = canonical.resolve_or_create_player(
        source="test",
        source_id=f"event-player-{suffix}",
        canonical_name=f"Event Player {suffix}",
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    evidence = facts.add_news_evidence(
        source="wire-a",
        url="https://wire-a.example/a",
        title=f"Player event {suffix}",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    return player, evidence


def test_single_source_cannot_claim_corroborated() -> None:
    registry = SourceRegistry((SourceDescriptor("wire", SourceKind.NEWS, "wire"),))
    assert classify_confirmation(("wire",), registry) == "unconfirmed"
    with pytest.raises(ValueError, match="corroborated"):
        classify_confirmation(("wire", "wire"), registry, requested="corroborated")


def test_same_source_duplicate_does_not_count_as_independent() -> None:
    registry = SourceRegistry(
        (
            SourceDescriptor("wire-a", SourceKind.NEWS, "wire"),
            SourceDescriptor("wire-a-mirror", SourceKind.NEWS, "wire"),
        )
    )
    assert classify_confirmation(("wire-a", "wire-a-mirror"), registry) == "unconfirmed"


def test_event_known_at_is_bounded_by_publication_and_observation(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    evidence = facts.add_news_evidence(
        source="wire-a",
        url="https://wire-a.example/a",
        title="Injury",
        published_at=NOW - timedelta(hours=2),
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    stored = facts.add_prematch_event(
        match_id=match.id,
        team_id=teams[0].id,
        player_id=None,
        event_type="injury",
        occurred_at=None,
        known_at=NOW - timedelta(minutes=1),
        confirmation_status="unconfirmed",
        evidence_refs=(evidence.record_id,),
    )
    assert stored.record_id.startswith("fact:prematch-event:")
    with pytest.raises(ValueError, match="publication"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW - timedelta(hours=3),
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
        )
    with pytest.raises(ValueError, match="observation"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW + timedelta(minutes=1),
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
        )


def test_unknown_raw_and_forged_official_status_are_rejected(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    with pytest.raises(KeyError, match="raw asset"):
        facts.add_news_evidence(
            source="club-official",
            url="https://club-official.example/news",
            title="Official",
            published_at=NOW - timedelta(hours=1),
            observed_at=NOW,
            raw_asset_id=RawAssetId("raw-asset:" + "0" * 64),
        )


def test_news_lineage_must_match_source_url_and_observation(tmp_path: Path) -> None:
    _, facts, assets, _, _ = _context(tmp_path)
    with pytest.raises(ValueError, match="raw asset source"):
        facts.add_news_evidence(
            source="wire-a",
            url="https://wire-a.example/a",
            title="Forged source",
            published_at=NOW - timedelta(hours=1),
            observed_at=NOW,
            raw_asset_id=assets["b"].id,
        )
    with pytest.raises(ValueError, match="news URL"):
        facts.add_news_evidence(
            source="wire-a",
            url="https://wire-a.example/other",
            title="Forged URL",
            published_at=NOW - timedelta(hours=1),
            observed_at=NOW,
            raw_asset_id=assets["a"].id,
        )


def test_prematch_collection_attempt_is_registered_idempotent_and_keeps_failure_raw(
    tmp_path: Path,
) -> None:
    canonical, facts, assets, _, match = _context(tmp_path)
    first = facts.record_prematch_collection_attempt(
        match_id=match.id,
        source="wire-a",
        kind=SourceKind.NEWS,
        target_url="https://wire-a.example/a",
        observed_at=NOW,
        collector_version="test/1",
        outcome=CollectionAttemptOutcome.BLOCKED,
        diagnostic_code="blocked_by_access_control",
        diagnostic_message="challenge body retained",
        raw_asset_id=assets["a"].id,
    )
    replay = facts.record_prematch_collection_attempt(
        match_id=match.id,
        source="wire-a",
        kind=SourceKind.NEWS,
        target_url="https://wire-a.example/a",
        observed_at=NOW,
        collector_version="test/1",
        outcome=CollectionAttemptOutcome.BLOCKED,
        diagnostic_code="blocked_by_access_control",
        diagnostic_message="challenge body retained",
        raw_asset_id=assets["a"].id,
    )
    assert replay == first
    attempts = canonical.collection_attempts(SEASON_ID)
    assert len(attempts) == 1
    assert attempts[0].raw_asset_id == assets["a"].id
    assert attempts[0].diagnostic_code == "blocked_by_access_control"

    with pytest.raises(ValueError, match="not registered"):
        facts.record_prematch_collection_attempt(
            match_id=match.id,
            source="unknown-blog",
            kind=SourceKind.NEWS,
            target_url="https://unknown.example/post",
            observed_at=NOW,
            collector_version="test/1",
            outcome=CollectionAttemptOutcome.FAILED,
            diagnostic_code="network_error",
        )


def test_unregistered_source_cannot_raise_confirmation_strength(tmp_path: Path) -> None:
    _, facts, assets, teams, match = _context(tmp_path)
    evidence = facts.add_news_evidence(
        source="unknown-blog",
        url="https://unknown-blog.example/unknown",
        title="Claim",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=assets["unknown"].id,
    )
    with pytest.raises(ValueError, match="not registered"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="corroborated",
            evidence_refs=(evidence.record_id,),
        )
    evidence = facts.add_news_evidence(
        source="wire-a",
        url="https://wire-a.example/a",
        title="Rumour",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    with pytest.raises(ValueError, match="official"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="official",
            evidence_refs=(evidence.record_id,),
        )


def test_two_independent_sources_are_corroborated(tmp_path: Path) -> None:
    _, facts, assets, teams, match = _context(tmp_path)
    refs = tuple(
        facts.add_news_evidence(
            source=source,
            url=f"https://{source}.example/{name}",
            title="Injury",
            published_at=NOW - timedelta(hours=1),
            observed_at=NOW,
            raw_asset_id=assets[name].id,
        ).record_id
        for source, name in (("wire-a", "a"), ("wire-b", "b"))
    )
    event = facts.add_prematch_event(
        match_id=match.id,
        team_id=teams[0].id,
        player_id=None,
        event_type="injury",
        occurred_at=None,
        known_at=NOW,
        confirmation_status="corroborated",
        evidence_refs=refs,
    )
    assert event.record_id.startswith("fact:prematch-event:")


def test_news_dto_preserves_raw_reference() -> None:
    dto = NewsEvidenceDTO(
        source="wire-a",
        url="https://wire-a.example/news",
        title="Update",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=RawAssetId("raw-asset:" + "1" * 64),
    )
    assert dto.raw_asset_id.value.startswith("raw-asset:")
    with pytest.raises(ValueError, match="generated summaries"):
        NewsEvidenceDTO.from_mapping(
            {
                "url": "https://wire-a.example/news",
                "title": "Generated",
                "published_at": NOW - timedelta(hours=1),
                "generated_summary": True,
            },
            source="wire-a",
            raw_asset_id=dto.raw_asset_id,
            observed_at=NOW,
        )


def test_event_dto_enforces_snapshot_cutoff_and_store_recomputes_status(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    evidence = facts.add_news_evidence(
        source="club-official",
        url="https://club-official.example/official",
        title="Official injury update",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
    )
    dto = PrematchEventDTO(
        match_id=match.id,
        team_id=teams[0].id,
        player_id=None,
        event_type="injury",
        occurred_at=None,
        known_at=NOW,
        evidence_refs=(evidence.record_id,),
        as_of=NOW,
    )
    stored = facts.add_prematch_event_dto(dto)
    with canonical.connect() as connection:
        row = connection.execute(
            "SELECT confirmation_status, can_modify_features FROM prematch_events "
            "WHERE record_id = ?",
            (stored.record_id,),
        ).fetchone()
    assert tuple(row) == ("official", 1)

    with pytest.raises(ValueError, match="as_of"):
        PrematchEventDTO(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            evidence_refs=(evidence.record_id,),
            as_of=NOW - timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    ("assignment_known_at", "assignment_observed_at", "event_as_of"),
    (
        (
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=2),
        ),
        (NOW - timedelta(minutes=1), NOW + timedelta(minutes=1), NOW),
    ),
    ids=("future-known-at", "future-observed-at"),
)
def test_player_event_rejects_assignment_not_visible_at_boundary(
    tmp_path: Path,
    assignment_known_at: datetime,
    assignment_observed_at: datetime,
    event_as_of: datetime,
) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    player, evidence = _player_event_subject(
        canonical,
        facts,
        assets,
        suffix="future-assignment",
    )
    facts.append_lineup_fact(
        match_id=match.id,
        match_version=1,
        team_id=teams[0].id,
        player_id=player.id,
        lineup_role="bench",
        official=False,
        known_at=assignment_known_at,
        observed_at=assignment_observed_at,
        raw_asset_id=assets["official"].id,
    )

    with pytest.raises(ValueError, match="has no assignment for match .* version 1 visible"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=player.id,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
            as_of=event_as_of,
        )


def test_player_event_rejects_assignment_from_later_match_version(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    player, evidence = _player_event_subject(
        canonical,
        facts,
        assets,
        suffix="later-version",
    )
    _, later_version = canonical.resolve_or_create_match(
        source="test",
        source_id="fixture-1",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=2),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW + timedelta(hours=1),
        raw_asset_id=assets["a"].id,
    )
    facts.append_lineup_fact(
        match_id=match.id,
        match_version=later_version.version,
        team_id=teams[0].id,
        player_id=player.id,
        lineup_role="bench",
        official=False,
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
    )

    with pytest.raises(ValueError, match="has no assignment for match .* version 1 visible"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=None,
            player_id=player.id,
            event_type="suspension",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
            as_of=NOW,
        )


def test_player_event_rejects_assignment_from_noncurrent_visible_version(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    player, evidence = _player_event_subject(
        canonical,
        facts,
        assets,
        suffix="wrong-version",
    )
    facts.append_lineup_fact(
        match_id=match.id,
        match_version=1,
        team_id=teams[0].id,
        player_id=player.id,
        lineup_role="bench",
        official=False,
        known_at=NOW,
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
    )
    _, current_version = canonical.resolve_or_create_match(
        source="test",
        source_id="fixture-1",
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=NOW + timedelta(days=2),
        status=MatchStatus.SCHEDULED,
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    assert current_version.version == 2

    with pytest.raises(ValueError, match="has no assignment for match .* version 2 visible"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=player.id,
            event_type="injury",
            occurred_at=None,
            known_at=NOW,
            confirmation_status="unconfirmed",
            evidence_refs=(evidence.record_id,),
            as_of=NOW,
        )


def test_player_event_accepts_visible_assignment_from_selected_match_version(
    tmp_path: Path,
) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    player, evidence = _player_event_subject(
        canonical,
        facts,
        assets,
        suffix="visible-assignment",
    )
    facts.append_lineup_fact(
        match_id=match.id,
        match_version=1,
        team_id=teams[0].id,
        player_id=player.id,
        lineup_role="bench",
        official=False,
        known_at=NOW,
        observed_at=NOW + timedelta(minutes=1),
        raw_asset_id=assets["official"].id,
    )

    stored = facts.add_prematch_event(
        match_id=match.id,
        team_id=None,
        player_id=player.id,
        event_type="injury",
        occurred_at=None,
        known_at=NOW,
        confirmation_status="unconfirmed",
        evidence_refs=(evidence.record_id,),
        as_of=NOW + timedelta(minutes=1),
    )

    assert stored.record_id.startswith("fact:prematch-event:")
    with canonical.connect() as connection:
        persisted_version = connection.execute(
            "SELECT match_version FROM prematch_events WHERE record_id = ?",
            (stored.record_id,),
        ).fetchone()[0]
    assert persisted_version == 1


def test_official_lineup_dto_rejects_future_publication() -> None:
    with pytest.raises(ValueError, match="published_at"):
        OfficialLineupDTO(
            match_id=MatchId("match:fixture"),
            team_id=TeamId("team:home"),
            player_ids=tuple(PlayerId(f"player:p-{index}") for index in range(11)),
            source="club-official",
            published_at=NOW + timedelta(seconds=1),
            observed_at=NOW,
            raw_asset_id=RawAssetId("raw-asset:" + "2" * 64),
        )


def test_official_lineup_adapter_is_the_verified_official_write_path(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    players = tuple(
        canonical.resolve_or_create_player(
            source="club-official",
            source_id=f"verified-player-{index}",
            canonical_name=f"Verified Player {index}",
            observed_at=NOW,
            raw_asset_id=assets["official"].id,
        ).id
        for index in range(11)
    )
    lineup = OfficialLineupDTO(
        match_id=match.id,
        team_id=teams[0].id,
        player_ids=players,
        source="club-official",
        published_at=NOW,
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
        url="https://club-official.example/official",
    )

    stored = facts.append_official_lineup(lineup)

    assert len(stored) == 11
    with canonical.connect() as connection:
        rows = connection.execute(
            "SELECT official, raw_asset_id FROM lineup_facts WHERE match_id = ? AND team_id = ?",
            (match.id.value, teams[0].id.value),
        ).fetchall()
    assert len(rows) == 11
    assert {(row["official"], row["raw_asset_id"]) for row in rows} == {
        (1, assets["official"].id.value)
    }


def test_official_lineup_validation_failure_leaves_no_partial_xi(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    valid = canonical.resolve_or_create_player(
        source="test",
        source_id="valid-player",
        canonical_name="Valid Player",
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
    )
    lineup = OfficialLineupDTO(
        match_id=match.id,
        team_id=teams[0].id,
        player_ids=(valid.id, *(PlayerId(f"player:missing-{index}") for index in range(10))),
        source="club-official",
        published_at=NOW,
        observed_at=NOW,
        raw_asset_id=assets["official"].id,
        url="https://club-official.example/official",
    )

    with pytest.raises(KeyError, match="not registered"):
        facts.append_official_lineup(lineup)
    with canonical.connect() as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM lineup_facts WHERE match_id = ? AND team_id = ?",
            (match.id.value, teams[0].id.value),
        ).fetchone()[0]
    assert stored == 0


def test_collection_attempt_rejects_unrelated_raw_source(tmp_path: Path) -> None:
    canonical, _, assets, _, match = _context(tmp_path)
    with pytest.raises(CanonicalConflictError, match="raw source"):
        # A matching URL/timestamp alone must not let an unrelated raw asset satisfy a
        # successful match-report attempt.
        canonical.record_collection_attempt(
            match_id=match.id,
            source="fbref-match-report",
            target_url="https://wire-a.example/a",
            outcome=CollectionAttemptOutcome.SUCCEEDED,
            observed_at=NOW,
            collector_version="test/1",
            raw_asset_id=assets["a"].id,
        )
