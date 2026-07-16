from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta, timezone

from football_data_platform.domain.ids import (
    CompetitionId,
    MatchId,
    SeasonId,
    TeamId,
)
from football_data_platform.domain.models import (
    MappingRule,
    Match,
    MatchStatus,
    MatchVersion,
    SourceMapping,
)

UTC_TIME = datetime(2026, 1, 2, 12, 30, tzinfo=UTC)


class PlatformIdTests(unittest.TestCase):
    def test_ids_are_typed_and_require_their_prefix(self) -> None:
        self.assertEqual(str(TeamId("team:01.test")), "team:01.test")
        self.assertNotEqual(TeamId("team:01.test"), MatchId("match:01.test"))

        with self.assertRaises(ValueError):
            TeamId("player:01.test")
        with self.assertRaises(ValueError):
            TeamId("team:Display Name")


class DomainContractTests(unittest.TestCase):
    def test_match_rejects_same_home_and_away_team(self) -> None:
        team = TeamId("team:01")

        with self.assertRaises(ValueError):
            Match(
                id=MatchId("match:01"),
                competition_id=CompetitionId("competition:eng.1"),
                season_id=SeasonId("season:eng.1.2025-26"),
                home_team_id=team,
                away_team_id=team,
            )

    def test_match_version_requires_utc_aware_times(self) -> None:
        with self.assertRaises(ValueError):
            MatchVersion(
                match_id=MatchId("match:01"),
                version=1,
                kickoff_at=datetime(2026, 1, 2, 15, 0),
                status=MatchStatus.SCHEDULED,
                observed_at=UTC_TIME,
            )

        with self.assertRaises(ValueError):
            MatchVersion(
                match_id=MatchId("match:01"),
                version=1,
                kickoff_at=UTC_TIME,
                status=MatchStatus.SCHEDULED,
                observed_at=UTC_TIME.astimezone(timezone(timedelta(hours=8))),
            )

    def test_source_mapping_enforces_versioned_validity_and_confidence(self) -> None:
        mapping = SourceMapping(
            source="fbref",
            source_id="team-123",
            entity_id=TeamId("team:01"),
            version=1,
            valid_from=UTC_TIME,
            valid_to=None,
            match_rule=MappingRule.SOURCE_ID,
            confidence=1.0,
            created_at=UTC_TIME,
            created_by="collector",
            audit_note="Provider identifier observed in the team URL.",
        )

        self.assertEqual(mapping.entity_id, TeamId("team:01"))

        with self.assertRaises(ValueError):
            SourceMapping(
                source="fbref",
                source_id="team-123",
                entity_id=TeamId("team:01"),
                version=1,
                valid_from=UTC_TIME,
                valid_to=UTC_TIME - timedelta(seconds=1),
                match_rule=MappingRule.SOURCE_ID,
                confidence=1.1,
                created_at=UTC_TIME,
                created_by="collector",
                audit_note="invalid",
            )


if __name__ == "__main__":
    unittest.main()
