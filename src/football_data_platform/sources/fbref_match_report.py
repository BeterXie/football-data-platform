"""Parse FBref per-match summary tables into source-owned DTOs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from football_data_platform.sources.fbref import ParseDiagnostic

_PLAYER_ID = re.compile(r"/players/([0-9a-z]+)/", re.IGNORECASE)
_SUMMARY_TABLE = re.compile(r"^stats_([0-9a-z]+)_summary$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PlayerReportRow:
    source_player_id: str
    player_name: str
    role: str
    minutes: float
    starter: bool | None
    metrics: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class TeamReport:
    source_team_id: str
    players: tuple[PlayerReportRow, ...]
    aggregated_stats: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class MatchReportParseResult:
    teams: tuple[TeamReport, ...]
    diagnostics: tuple[ParseDiagnostic, ...]


@dataclass(slots=True)
class _Cell:
    text_parts: list[str] = field(default_factory=list)
    href: str | None = None

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


class _SummaryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id: str | None = None
        self.current_row: dict[str, _Cell] | None = None
        self.current_cell: _Cell | None = None
        self.current_stat: str | None = None
        self.rows: dict[str, list[dict[str, _Cell]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and _SUMMARY_TABLE.match(attributes.get("id") or ""):
            self.table_id = attributes["id"]
            self.rows.setdefault(self.table_id, [])
        elif self.table_id is not None and tag == "tr":
            self.current_row = {}
        elif self.current_row is not None and tag in {"td", "th"}:
            self.current_stat = attributes.get("data-stat")
            self.current_cell = _Cell()
        elif self.current_cell is not None and tag == "a":
            self.current_cell.href = attributes.get("href")

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            if self.current_row is not None and self.current_stat:
                self.current_row[self.current_stat] = self.current_cell
            self.current_cell = None
            self.current_stat = None
        elif tag == "tr" and self.current_row is not None and self.table_id is not None:
            if self.current_row.get("player") is not None:
                self.rows[self.table_id].append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.table_id is not None:
            self.table_id = None


class _CommentCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []

    def handle_comment(self, data: str) -> None:
        if "_summary" in data and "<table" in data.lower():
            self.fragments.append(data)


def parse_match_report(content: bytes | str) -> MatchReportParseResult:
    """Parse normal and comment-wrapped FBref player summary tables."""

    html = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    comments = _CommentCollector()
    comments.feed(html)
    parser = _SummaryParser()
    parser.feed(html)
    for fragment in comments.fragments:
        parser.feed(fragment)

    diagnostics: list[ParseDiagnostic] = []
    teams: list[TeamReport] = []
    for table_id, rows in sorted(parser.rows.items()):
        matched = _SUMMARY_TABLE.match(table_id)
        if matched is None:
            continue
        players: list[PlayerReportRow] = []
        reported_totals: dict[str, float | None] = {}
        for row_number, row in enumerate(rows, start=1):
            player_cell = row["player"]
            if player_cell.text.casefold() == "team total":
                reported_totals = {
                    name: _number(row.get(name))
                    for name in ("goals", "xg", "shots", "shots_on_target")
                }
                continue
            player_id_match = _PLAYER_ID.search(player_cell.href or "")
            if player_id_match is None:
                if player_cell.text.casefold() not in {"player", "team total"} and not re.match(
                    r"^\d+ players$", player_cell.text.casefold()
                ):
                    diagnostics.append(
                        ParseDiagnostic(
                            "player_id_missing",
                            f"table {table_id} row {row_number} has no player link",
                            row_number,
                        )
                    )
                continue
            minutes = _number(row.get("minutes"))
            if minutes is None:
                diagnostics.append(
                    ParseDiagnostic(
                        "player_minutes_missing",
                        f"player {player_cell.text} has no minutes",
                        row_number,
                        player_id_match.group(1),
                    )
                )
                continue
            role = row.get("position").text if row.get("position") is not None else "unknown"
            starter_value = _number(row.get("games_starts"))
            metrics = {
                name: _number(cell)
                for name, cell in row.items()
                if name not in {"player", "position", "minutes", "games_starts", "shirtnumber"}
            }
            players.append(
                PlayerReportRow(
                    source_player_id=player_id_match.group(1),
                    player_name=player_cell.text,
                    role=role or "unknown",
                    minutes=minutes,
                    starter=(None if starter_value is None else starter_value > 0),
                    metrics=metrics,
                )
            )
        aggregated = {}
        for metric in ("goals", "xg", "shots", "shots_on_target"):
            reported = reported_totals.get(metric)
            aggregated[metric] = (
                reported
                if reported is not None
                else _sum_available(player.metrics.get(metric) for player in players)
            )
        teams.append(TeamReport(matched.group(1), tuple(players), aggregated))
    if not teams:
        diagnostics.append(
            ParseDiagnostic("player_summary_tables_missing", "no player summary tables found")
        )
    return MatchReportParseResult(tuple(teams), tuple(diagnostics))


def _number(cell: _Cell | None) -> float | None:
    if cell is None or not cell.text:
        return None
    value = cell.text.replace(",", "").removesuffix("%").strip()
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _sum_available(values) -> float | None:
    available = [value for value in values if value is not None]
    return math.fsum(available) if available else None
