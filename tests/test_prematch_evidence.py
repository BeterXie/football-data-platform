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
    for name, source in (("a", "wire-a"), ("b", "wire-b"), ("official", "club-official")):
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


def test_event_known_at_cannot_predate_evidence_observation(tmp_path: Path) -> None:
    canonical, facts, assets, teams, match = _context(tmp_path)
    evidence = facts.add_news_evidence(
        source="wire-a",
        url="https://wire-a.example/a",
        title="Injury",
        published_at=NOW - timedelta(hours=2),
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    with pytest.raises(ValueError, match="known_at"):
        facts.add_prematch_event(
            match_id=match.id,
            team_id=teams[0].id,
            player_id=None,
            event_type="injury",
            occurred_at=None,
            known_at=NOW - timedelta(minutes=1),
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


def test_unregistered_source_cannot_raise_confirmation_strength(tmp_path: Path) -> None:
    _, facts, assets, teams, match = _context(tmp_path)
    evidence = facts.add_news_evidence(
        source="unknown-blog",
        url="https://unknown.example/post",
        title="Claim",
        published_at=NOW - timedelta(hours=1),
        observed_at=NOW,
        raw_asset_id=assets["a"].id,
    )
    with pytest.raises(ValueError, match="corroborated"):
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
