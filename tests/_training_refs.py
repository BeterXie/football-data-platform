from __future__ import annotations

import hashlib
import json
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


def seed_training_references(
    layout: DataLayout,
    *,
    label_known_at: datetime = AS_OF,
    observed_at: datetime | None = None,
    home_goals: int = 2,
    away_goals: int = 1,
    reference_key: str = "default",
) -> tuple[str, str]:
    """Create one real derived feature ref and one canonical result ref for tests."""

    identity = hashlib.sha256(
        (f"{reference_key}|{label_known_at.isoformat()}|{home_goals}|{away_goals}").encode()
    ).hexdigest()[:16]
    observed_at = max(AS_OF if observed_at is None else observed_at, label_known_at)
    kickoff_at = label_known_at - timedelta(hours=3)
    content = json.dumps(
        {
            "away_goals": away_goals,
            "home_goals": home_goals,
            "known_at": label_known_at.isoformat(),
            "reference_key": reference_key,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    archive = RawArchive(layout)
    raw_asset = archive.archive(
        content,
        source="fbref",
        source_id=f"training-reference-{identity}",
        url=f"https://fbref.example/training-reference-{identity}",
        observed_at=observed_at,
        target_event_time=kickoff_at,
        collector_version="test-training/1",
        media_type="application/json",
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
            source_id=f"training-reference-{identity}-team-{index}",
            canonical_name=f"Training Reference Team {index}",
            competition_id=competition_id,
            observed_at=observed_at,
            raw_asset_id=raw_asset.id,
        )
        for index in range(2)
    )
    match, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id=f"training-reference-match-{identity}",
        competition_id=competition_id,
        season_id=season_id,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=kickoff_at,
        status=MatchStatus.FINISHED,
        observed_at=observed_at,
        raw_asset_id=raw_asset.id,
    )
    result = CanonicalFactStore(canonical).append_result_90(
        match_id=match.id,
        match_version=version.version,
        home_goals=home_goals,
        away_goals=away_goals,
        known_at=label_known_at,
        observed_at=observed_at,
        raw_asset_id=raw_asset.id,
    )
    feature_ref = DerivedArchive(layout).write_snapshot_source(
        value={"training_reference": True},
        input_refs=(raw_asset.id,),
        transform_version="test-training-feature/1",
        generated_at=observed_at,
    )
    return feature_ref, result.record_id
