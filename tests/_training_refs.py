from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from football_data_platform.config import load_competition_registry
from football_data_platform.domain.ids import CompetitionId, SeasonId
from football_data_platform.domain.models import MatchStatus
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

AS_OF = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]


def seed_training_references(layout: DataLayout) -> tuple[str, str]:
    """Create one real derived feature ref and one canonical result ref for tests."""

    archive = RawArchive(layout)
    raw_asset = archive.archive(
        b"training-reference-fixture",
        source="fbref",
        source_id="training-reference",
        url="https://fbref.example/training-reference",
        observed_at=AS_OF,
        target_event_time=AS_OF - timedelta(hours=1),
        collector_version="test-training/1",
        media_type="text/plain",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    registry = load_competition_registry(ROOT / "config" / "competitions.toml")
    canonical.register_registry(registry, registered_at=AS_OF)
    canonical.register_raw_asset(raw_asset)
    competition_id = CompetitionId("competition:eng.1")
    season_id = SeasonId("season:eng.1.2025-26")
    teams = tuple(
        canonical.resolve_or_create_team(
            source="fbref",
            source_id=f"training-reference-team-{index}",
            canonical_name=f"Training Reference Team {index}",
            competition_id=competition_id,
            observed_at=AS_OF,
            raw_asset_id=raw_asset.id,
        )
        for index in range(2)
    )
    match, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="training-reference-match",
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=AS_OF - timedelta(days=1),
        status=MatchStatus.FINISHED,
        observed_at=AS_OF,
        raw_asset_id=raw_asset.id,
    )
    result = CanonicalFactStore(canonical).append_result_90(
        match_id=match.id,
        match_version=version.version,
        home_goals=2,
        away_goals=1,
        known_at=AS_OF,
        observed_at=AS_OF,
        raw_asset_id=raw_asset.id,
    )
    feature_ref = DerivedArchive(layout).write_snapshot_source(
        value={"training_reference": True},
        input_refs=(raw_asset.id,),
        transform_version="test-training-feature/1",
        generated_at=AS_OF,
    )
    return feature_ref, result.record_id
