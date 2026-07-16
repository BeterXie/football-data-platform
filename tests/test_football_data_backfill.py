from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from football_data_platform.config import load_competition_registry
from football_data_platform.pipelines.results_backfill import ingest_results_backfill
from football_data_platform.sources.football_data_csv import (
    parse_results_csv,
    results_url,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import RawArchive

ROOT = Path(__file__).parents[1]
OBSERVED_AT = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)


def _registration():
    registry = load_competition_registry(ROOT / "config/competitions.toml")
    competition = registry.competitions[0]
    return registry, competition, competition.seasons[0]


def test_results_parser_ignores_odds_columns_and_keeps_90_minute_results() -> None:
    _, competition, season = _registration()
    content = (ROOT / "tests/fixtures/football_data_e0_sample.csv").read_bytes()

    parsed = parse_results_csv(content, competition=competition, season=season)

    assert len(parsed.matches) == 2
    assert parsed.matches[0].home_name == "Liverpool"
    assert parsed.matches[0].home_goals == 4
    assert parsed.matches[0].away_goals == 2
    assert not hasattr(parsed.matches[0], "odds")
    assert results_url(season).endswith("/2526/E0.csv")


def test_results_backfill_is_raw_first_and_idempotent(tmp_path: Path) -> None:
    registry, competition, season = _registration()
    content = (ROOT / "tests/fixtures/football_data_e0_sample.csv").read_bytes()
    layout = DataLayout(tmp_path / "data")
    archive = RawArchive(layout)
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(registry, registered_at=OBSERVED_AT)
    arguments = {
        "content": content,
        "page_url": results_url(season),
        "competition": competition,
        "season": season,
        "observed_at": OBSERVED_AT,
        "archive": archive,
        "canonical": canonical,
    }

    first = ingest_results_backfill(**arguments)
    second = ingest_results_backfill(**{**arguments, "observed_at": OBSERVED_AT.replace(minute=5)})

    assert first.raw_asset_id != second.raw_asset_id
    assert first.canonical_match_ids == second.canonical_match_ids
    assert first.result_fact_ids == second.result_fact_ids
    assert canonical.counts()["teams"] == 4
    assert canonical.counts()["matches"] == 2
    assert len(first.result_fact_ids) == 2
    assert first.coverage.schedule_complete is False
    with canonical.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM match_results_90").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM fact_evidence").fetchone()[0] == 4
