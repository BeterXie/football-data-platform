from __future__ import annotations

import unittest
from pathlib import Path

from football_data_platform.config import DEFAULT_REGISTRY_PATH, load_competition_registry
from football_data_platform.domain.ids import CompetitionId, SeasonId


class CompetitionRegistryTests(unittest.TestCase):
    def test_packaged_and_operator_registry_are_identical(self) -> None:
        operator_registry = Path(__file__).parents[1] / "config" / "competitions.toml"
        self.assertEqual(
            DEFAULT_REGISTRY_PATH.read_bytes(),
            operator_registry.read_bytes(),
        )

    def test_loads_premier_league_vertical_slice(self) -> None:
        registry = load_competition_registry()

        competition = registry.competition(CompetitionId("competition:eng.1"))
        season = registry.season(SeasonId("season:eng.1.2025-26"))

        self.assertEqual(registry.schema_version, 2)
        self.assertEqual(competition.name, "Premier League")
        self.assertEqual(season.label, "2025-26")
        self.assertEqual(season.expected_teams, 20)
        self.assertEqual(season.expected_matches, 380)
        self.assertEqual(season.source("fbref").competition_id, "9")
        self.assertEqual(season.source("fbref").season_id, "2025-2026")
        self.assertEqual(len(season.teams), 20)
        self.assertEqual(
            season.team("fbref", "18bb7c10").id,
            season.team("football-data", "Arsenal").id,
        )

    def test_unknown_source_is_explicit(self) -> None:
        season = load_competition_registry().season(SeasonId("season:eng.1.2025-26"))

        with self.assertRaises(KeyError):
            season.source("unknown")


if __name__ == "__main__":
    unittest.main()
