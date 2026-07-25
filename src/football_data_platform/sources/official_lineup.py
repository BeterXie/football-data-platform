"""Versioned parser for archived official-lineup JSON evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from football_data_platform.domain.models import require_utc

OFFICIAL_LINEUP_SCHEMA_VERSION = 2
OFFICIAL_LINEUP_PARSER_VERSION = "official-lineup-json/1"
OFFICIAL_LINEUP_MATCH_MAPPING_SOURCE = "fbref-schedule"
OFFICIAL_LINEUP_TEAM_MAPPING_SOURCE = "fbref"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "source",
    "source_match_id",
    "match_mapping_source",
    "team_mapping_source",
    "player_mapping_source",
    "published_at",
    "teams",
}
_TEAM_FIELDS = {"source_team_id", "starters"}
_PLAYER_FIELDS = {"source_player_id", "name"}


@dataclass(frozen=True, slots=True)
class OfficialLineupPlayer:
    source_player_id: str
    name: str

    def __post_init__(self) -> None:
        _require_text(self.source_player_id, "source_player_id")
        _require_text(self.name, "player name")


@dataclass(frozen=True, slots=True)
class OfficialLineupTeam:
    source_team_id: str
    starters: tuple[OfficialLineupPlayer, ...]

    def __post_init__(self) -> None:
        _require_text(self.source_team_id, "source_team_id")
        player_ids = tuple(player.source_player_id for player in self.starters)
        if len(player_ids) != 11 or len(set(player_ids)) != 11:
            raise ValueError("official lineup team requires 11 unique starters")


@dataclass(frozen=True, slots=True)
class OfficialLineupParseResult:
    source: str
    source_match_id: str
    match_mapping_source: str
    team_mapping_source: str
    player_mapping_source: str
    published_at: datetime
    teams: tuple[OfficialLineupTeam, ...]
    parser_version: str = OFFICIAL_LINEUP_PARSER_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.source, "source"),
            (self.source_match_id, "source_match_id"),
            (self.match_mapping_source, "match_mapping_source"),
            (self.team_mapping_source, "team_mapping_source"),
            (self.player_mapping_source, "player_mapping_source"),
            (self.parser_version, "parser_version"),
        ):
            _require_text(value, name)
        require_utc(self.published_at, "published_at")
        if len(self.teams) != 2 or len({team.source_team_id for team in self.teams}) != 2:
            raise ValueError("official lineup payload requires exactly two distinct teams")
        player_ids = [player.source_player_id for team in self.teams for player in team.starters]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("official lineup player IDs must be unique across both teams")


def parse_official_lineup_json(content: bytes | str) -> OfficialLineupParseResult:
    """Parse one archived source payload without accepting caller-supplied identities."""

    if isinstance(content, bytes):
        text = content.decode("utf-8")
    elif isinstance(content, str):
        text = content
    else:
        raise TypeError("official lineup content must be bytes or text")
    try:
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("official lineup content must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("official lineup payload must be an object")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, "official lineup payload")
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != OFFICIAL_LINEUP_SCHEMA_VERSION
    ):
        raise ValueError(f"official lineup schema_version must be {OFFICIAL_LINEUP_SCHEMA_VERSION}")

    source = _text(payload["source"], "source")
    match_mapping_source = _text(payload["match_mapping_source"], "match_mapping_source")
    team_mapping_source = _text(payload["team_mapping_source"], "team_mapping_source")
    player_mapping_source = _text(payload["player_mapping_source"], "player_mapping_source")
    if match_mapping_source != OFFICIAL_LINEUP_MATCH_MAPPING_SOURCE:
        raise ValueError(f"match_mapping_source must be {OFFICIAL_LINEUP_MATCH_MAPPING_SOURCE!r}")
    if team_mapping_source != OFFICIAL_LINEUP_TEAM_MAPPING_SOURCE:
        raise ValueError(f"team_mapping_source must be {OFFICIAL_LINEUP_TEAM_MAPPING_SOURCE!r}")
    if player_mapping_source != source:
        raise ValueError("player_mapping_source must match the official lineup source")

    teams_value = payload["teams"]
    if not isinstance(teams_value, list):
        raise ValueError("official lineup teams must be a list")
    teams = tuple(_parse_team(item) for item in teams_value)
    return OfficialLineupParseResult(
        source=source,
        source_match_id=_text(payload["source_match_id"], "source_match_id"),
        match_mapping_source=match_mapping_source,
        team_mapping_source=team_mapping_source,
        player_mapping_source=player_mapping_source,
        published_at=_timestamp(payload["published_at"], "published_at"),
        teams=teams,
    )


def _parse_team(value: Any) -> OfficialLineupTeam:
    if not isinstance(value, dict):
        raise ValueError("official lineup team must be an object")
    _require_exact_fields(value, _TEAM_FIELDS, "official lineup team")
    starters_value = value["starters"]
    if not isinstance(starters_value, list):
        raise ValueError("official lineup starters must be a list")
    starters = tuple(_parse_player(item) for item in starters_value)
    return OfficialLineupTeam(
        source_team_id=_text(value["source_team_id"], "source_team_id"),
        starters=starters,
    )


def _parse_player(value: Any) -> OfficialLineupPlayer:
    if not isinstance(value, dict):
        raise ValueError("official lineup player must be an object")
    _require_exact_fields(value, _PLAYER_FIELDS, "official lineup player")
    return OfficialLineupPlayer(
        source_player_id=_text(value["source_player_id"], "source_player_id"),
        name=_text(value["name"], "player name"),
    )


def _require_exact_fields(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp") from error
    require_utc(parsed, name)
    return parsed.astimezone(UTC)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be non-empty text")
    _require_text(value, name)
    return value


def _require_text(value: str, name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty text without surrounding whitespace")
