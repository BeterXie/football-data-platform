"""Parse FBref per-match tables into source-owned DTOs.

FBref places several report tables in HTML comments on some pages.  The parser therefore
collects both normal and comment-wrapped tables, keeps table-level diagnostics, and only uses
the summary table as the source of player identity/minutes.  Auxiliary tables enrich those
summary rows when present; missing values remain ``None`` rather than being replaced with zero.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser

from football_data_platform.sources.fbref import ParseDiagnostic

REPORT_PARSER_VERSION = "fbref-match-report/2"
REPORT_TABLE_NAMES = (
    "summary",
    "passing",
    "passing_types",
    "defense",
    "possession",
    "misc",
    "keeper",
)
_REPORT_TABLE = re.compile(
    r"^stats_([0-9a-z]+)_(summary|passing_types|passing|defense|possession|misc|keeper)$",
    re.IGNORECASE,
)
_REPORT_TABLE_PREFIX = re.compile(r"^stats_[0-9a-z]+_", re.IGNORECASE)
_PLAYER_ID = re.compile(r"/players/([0-9a-z]+)/", re.IGNORECASE)
_STRUCTURAL_FIELDS = frozenset(
    {
        "player",
        "position",
        "minutes",
        "games_starts",
        "shirtnumber",
        "nationality",
        "age",
        "born",
    }
)
_TOTAL_LABELS = frozenset({"team total", "squad total", "total"})
_MATCH_ID_FROM_URL = re.compile(r"/matches/([0-9a-z]+)/", re.IGNORECASE)
_COMPETITION_URL = re.compile(r"/comps/([0-9a-z]+)/([^/?#]+)/", re.IGNORECASE)
_TEAM_ID_FROM_URL = re.compile(r"/squads/([0-9a-z]+)/", re.IGNORECASE)


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
    # Per-table totals are retained for downstream field-coverage diagnostics.  The existing
    # ``aggregated_stats`` field intentionally remains the small summary contract consumed by
    # the match-report pipeline.
    table_stats: dict[str, dict[str, float | None]] = field(default_factory=dict)
    tables_present: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchReportIdentity:
    """Identity and score facts extracted from the report page itself."""

    source_match_id: str | None = None
    competition_source_id: str | None = None
    season_source_id: str | None = None
    played_on: date | None = None
    home_source_id: str | None = None
    away_source_id: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None


@dataclass(frozen=True, slots=True)
class MatchReportParseResult:
    teams: tuple[TeamReport, ...]
    diagnostics: tuple[ParseDiagnostic, ...]
    parser_version: str = REPORT_PARSER_VERSION
    tables_present: tuple[str, ...] = ()
    identity: MatchReportIdentity = field(default_factory=MatchReportIdentity)


@dataclass(slots=True)
class _Cell:
    text_parts: list[str] = field(default_factory=list)
    href: str | None = None

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


class _ReportTableParser(HTMLParser):
    """Collect rows from report tables, including unsupported table IDs for diagnostics."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id: str | None = None
        self.current_row: dict[str, _Cell] | None = None
        self.current_cell: _Cell | None = None
        self.current_stat: str | None = None
        self.rows: dict[str, list[dict[str, _Cell]]] = {}
        self.table_occurrences: list[str] = []
        self.unsupported_table_ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            candidate = attributes.get("id") or ""
            if _REPORT_TABLE.match(candidate):
                self.table_id = candidate
                self.rows.setdefault(candidate, [])
                self.table_occurrences.append(candidate)
            elif _REPORT_TABLE_PREFIX.match(candidate):
                self.unsupported_table_ids.append(candidate)
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
            if self.current_row:
                self.rows[self.table_id].append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.table_id is not None:
            self.table_id = None


class _CommentCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []

    def handle_comment(self, data: str) -> None:
        if "<table" in data.lower() and "stats_" in data.lower():
            self.fragments.append(data)


class _IdentityParser(HTMLParser):
    """Extract stable identity fields from canonical metadata and the FBref scorebox."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.match_ids: list[str] = []
        self.competition_ids: list[str] = []
        self.season_ids: list[str] = []
        self.team_ids: list[str] = []
        self.score_values: list[int] = []
        self.date_values: list[str] = []
        self._scorebox_div_depth = 0
        self._score_div_depth = 0
        self._score_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "meta" and (attributes.get("property") or "").casefold() in {
            "og:url",
            "twitter:url",
        }:
            self._add_match_id(attributes.get("content"))
        elif tag == "link" and "canonical" in (attributes.get("rel") or "").casefold().split():
            self._add_match_id(attributes.get("href"))

        venue_date = attributes.get("data-venue-date")
        if venue_date:
            self.date_values.append(venue_date)

        if tag == "div":
            if self._scorebox_div_depth:
                self._scorebox_div_depth += 1
            elif "scorebox" in classes:
                self._scorebox_div_depth = 1
            if self._scorebox_div_depth and "score" in classes:
                self._score_div_depth = self._scorebox_div_depth
                self._score_parts = []

        if tag == "a" and self._scorebox_div_depth:
            href = attributes.get("href") or ""
            team = _TEAM_ID_FROM_URL.search(href)
            if team is not None:
                self.team_ids.append(team.group(1))
            competition = _COMPETITION_URL.search(href)
            if competition is not None:
                self.competition_ids.append(competition.group(1))
                self.season_ids.append(competition.group(2))

    def handle_data(self, data: str) -> None:
        if self._score_div_depth:
            self._score_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "div" or not self._scorebox_div_depth:
            return
        if self._score_div_depth == self._scorebox_div_depth:
            self.score_values.extend(_score_numbers(" ".join(self._score_parts)))
            self._score_div_depth = 0
            self._score_parts = []
        self._scorebox_div_depth -= 1

    def _add_match_id(self, value: str | None) -> None:
        matched = _MATCH_ID_FROM_URL.search(value or "")
        if matched is not None:
            self.match_ids.append(matched.group(1))


def parse_match_report(
    content: bytes | str,
    *,
    required_tables: tuple[str, ...] = ("summary",),
) -> MatchReportParseResult:
    """Parse normal and comment-wrapped FBref report tables.

    ``required_tables`` is an explicit caller contract.  It defaults to ``summary`` because
    the summary table is required to establish player identity and minutes.  Auxiliary table
    absence is observable through ``tables_present`` and is never silently interpreted as zero.
    """

    required = tuple(dict.fromkeys(required_tables))
    unknown_required = set(required) - set(REPORT_TABLE_NAMES)
    if unknown_required:
        raise ValueError(f"unknown required report tables: {sorted(unknown_required)}")

    html = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    identity_parser = _IdentityParser()
    identity_parser.feed(html)
    comments = _CommentCollector()
    comments.feed(html)
    parser = _ReportTableParser()
    parser.feed(html)
    for fragment in comments.fragments:
        parser.feed(fragment)

    identity, identity_diagnostics = _parse_identity(identity_parser)
    diagnostics: list[ParseDiagnostic] = list(identity_diagnostics)
    for table_id in parser.unsupported_table_ids:
        diagnostics.append(
            ParseDiagnostic(
                "unsupported_report_table",
                f"unsupported FBref report table {table_id!r}",
                subject_id=table_id,
                table_id=table_id,
            )
        )
    for table_id in sorted(
        {item for item in parser.table_occurrences if parser.table_occurrences.count(item) > 1}
    ):
        diagnostics.append(
            ParseDiagnostic(
                "duplicate_report_table",
                f"FBref report table {table_id!r} occurs more than once",
                subject_id=table_id,
                table_id=table_id,
            )
        )

    # Team state is kept separate from the final DTO so auxiliary rows can be merged into the
    # summary rows regardless of table order in the source document.
    team_state: dict[str, dict[str, object]] = {}
    table_names: list[str] = []
    for table_id, rows in parser.rows.items():
        matched = _REPORT_TABLE.match(table_id)
        if matched is None:
            continue
        source_team_id, table_name = matched.group(1), matched.group(2).lower()
        if table_name not in table_names:
            table_names.append(table_name)
        state = team_state.setdefault(
            source_team_id,
            {
                "tables": [],
                "summary": {},
                "auxiliary": {},
                "table_stats": {},
                "summary_totals": {},
            },
        )
        tables = state["tables"]
        assert isinstance(tables, list)
        if table_name not in tables:
            tables.append(table_name)
        summary = state["summary"]
        auxiliary = state["auxiliary"]
        table_stats = state["table_stats"]
        summary_totals = state["summary_totals"]
        assert isinstance(summary, dict)
        assert isinstance(auxiliary, dict)
        assert isinstance(table_stats, dict)
        assert isinstance(summary_totals, dict)

        table_total: dict[str, float | None] = {}
        for row_number, row in enumerate(rows, start=1):
            player_cell = row.get("player")
            if player_cell is None:
                diagnostics.append(
                    ParseDiagnostic(
                        "report_player_column_missing",
                        f"table {table_id} row {row_number} has no player field",
                        row_number,
                        table_id=table_id,
                    )
                )
                continue
            label = player_cell.text.casefold()
            if _is_total_label(label):
                table_total = _row_metrics(row)
                if table_name == "summary":
                    summary_totals.update(table_total)
                continue
            player_match = _PLAYER_ID.search(player_cell.href or "")
            if player_match is None:
                if label not in {"player", "players"} and not re.match(r"^\d+ players$", label):
                    diagnostics.append(
                        ParseDiagnostic(
                            "player_id_missing",
                            f"table {table_id} row {row_number} has no player link",
                            row_number,
                            table_id=table_id,
                        )
                    )
                continue
            source_player_id = player_match.group(1)
            metrics = _row_metrics(row)
            if table_name == "summary":
                minutes = _number(row.get("minutes"))
                if minutes is None:
                    diagnostics.append(
                        ParseDiagnostic(
                            "player_minutes_missing",
                            f"player {player_cell.text} has no minutes",
                            row_number,
                            source_player_id,
                            table_id,
                        )
                    )
                    continue
                role_cell = row.get("position")
                role = role_cell.text if role_cell is not None else "unknown"
                starter_value = _number(row.get("games_starts"))
                current = summary.get(source_player_id)
                if current is not None:
                    diagnostics.append(
                        ParseDiagnostic(
                            "duplicate_player_row",
                            f"player {source_player_id} appears more than once in {table_id}",
                            row_number,
                            source_player_id,
                            table_id,
                        )
                    )
                    current.metrics.update(metrics)
                    continue
                summary[source_player_id] = PlayerReportRow(
                    source_player_id=source_player_id,
                    player_name=player_cell.text,
                    role=role or "unknown",
                    minutes=minutes,
                    starter=(None if starter_value is None else starter_value > 0),
                    metrics=metrics,
                )
            else:
                auxiliary.setdefault(table_name, {})[source_player_id] = metrics

        if table_name != "summary":
            table_total = table_total or _sum_table_metrics(auxiliary.get(table_name, {}).values())
        elif not summary_totals:
            table_total = {
                name: _sum_available(player.metrics.get(name) for player in summary.values())
                for name in ("goals", "xg", "shots", "shots_on_target")
            }
        table_stats[table_name] = table_total

    teams: list[TeamReport] = []
    for source_team_id, state in team_state.items():
        tables = state["tables"]
        summary = state["summary"]
        auxiliary = state["auxiliary"]
        table_stats = state["table_stats"]
        summary_totals = state["summary_totals"]
        assert isinstance(tables, list)
        assert isinstance(summary, dict)
        assert isinstance(auxiliary, dict)
        assert isinstance(table_stats, dict)
        assert isinstance(summary_totals, dict)
        if "summary" not in tables:
            diagnostics.append(
                ParseDiagnostic(
                    "summary_table_missing",
                    f"team {source_team_id} has no summary table",
                    subject_id=source_team_id,
                )
            )
            # A report team is defined by its summary table.  Auxiliary tables without a
            # summary remain visible through diagnostics but cannot provide canonical player
            # identity/minutes and therefore are not emitted as a TeamReport.
            continue
        for required_name in required:
            if required_name not in tables:
                diagnostics.append(
                    ParseDiagnostic(
                        "required_report_table_missing",
                        f"team {source_team_id} is missing required table {required_name!r}",
                        subject_id=source_team_id,
                    )
                )
        for table_name, rows in auxiliary.items():
            for source_player_id, metrics in rows.items():
                player = summary.get(source_player_id)
                if player is None:
                    diagnostics.append(
                        ParseDiagnostic(
                            "auxiliary_player_not_in_summary",
                            f"player {source_player_id} in {table_name} has no summary row",
                            subject_id=source_player_id,
                            table_id=f"stats_{source_team_id}_{table_name}",
                        )
                    )
                    continue
                player.metrics.update(metrics)
        players = tuple(summary.values())
        core = {
            metric: summary_totals.get(metric)
            for metric in ("goals", "xg", "shots", "shots_on_target")
        }
        for metric in tuple(core):
            if core[metric] is None:
                core[metric] = _sum_available(player.metrics.get(metric) for player in players)
        teams.append(
            TeamReport(
                source_team_id=source_team_id,
                players=players,
                aggregated_stats=core,
                table_stats={name: dict(values) for name, values in table_stats.items()},
                tables_present=tuple(tables),
            )
        )

    if not team_state:
        diagnostics.append(
            ParseDiagnostic("player_summary_tables_missing", "no player summary tables found")
        )
    elif "summary" in required and not any(
        "summary" in state["tables"] for state in team_state.values()
    ):
        diagnostics.append(
            ParseDiagnostic("player_summary_tables_missing", "no player summary tables found")
        )
    return MatchReportParseResult(
        tuple(teams),
        tuple(diagnostics),
        parser_version=REPORT_PARSER_VERSION,
        tables_present=tuple(table_names),
        identity=identity,
    )


def _is_total_label(value: str) -> bool:
    return value in _TOTAL_LABELS or value.startswith("team total")


def _parse_identity(
    parser: _IdentityParser,
) -> tuple[MatchReportIdentity, tuple[ParseDiagnostic, ...]]:
    diagnostics: list[ParseDiagnostic] = []
    match_id = _one_identity_value(
        parser.match_ids,
        label="source match ID",
        missing_code="match_source_id_missing",
        conflict_code="match_source_id_conflict",
        diagnostics=diagnostics,
    )
    competition_id = _one_identity_value(
        parser.competition_ids,
        label="competition source ID",
        missing_code="match_competition_id_missing",
        conflict_code="match_competition_id_conflict",
        diagnostics=diagnostics,
    )
    season_id = _one_identity_value(
        parser.season_ids,
        label="season source ID",
        missing_code="match_season_id_missing",
        conflict_code="match_season_id_conflict",
        diagnostics=diagnostics,
    )
    team_ids = tuple(dict.fromkeys(parser.team_ids))
    if len(team_ids) != 2:
        diagnostics.append(
            ParseDiagnostic(
                "match_team_identity_missing"
                if len(team_ids) < 2
                else "match_team_identity_conflict",
                f"scorebox contains {len(team_ids)} distinct team IDs; expected 2",
            )
        )
        home_team_id = away_team_id = None
    else:
        home_team_id, away_team_id = team_ids

    if len(parser.score_values) != 2:
        diagnostics.append(
            ParseDiagnostic(
                "match_score_missing" if len(parser.score_values) < 2 else "match_score_conflict",
                f"scorebox contains {len(parser.score_values)} score values; expected 2",
            )
        )
        home_goals = away_goals = None
    else:
        home_goals, away_goals = parser.score_values

    played_dates: list[date] = []
    for value in parser.date_values:
        try:
            played_dates.append(date.fromisoformat(value.strip()[:10]))
        except ValueError:
            diagnostics.append(
                ParseDiagnostic(
                    "match_date_invalid",
                    f"invalid scorebox data-venue-date value {value!r}",
                )
            )
    unique_dates = tuple(dict.fromkeys(played_dates))
    if not unique_dates:
        diagnostics.append(ParseDiagnostic("match_date_missing", "scorebox match date is missing"))
        played_on = None
    elif len(unique_dates) > 1:
        diagnostics.append(
            ParseDiagnostic(
                "match_date_conflict",
                f"scorebox contains conflicting dates: {unique_dates!r}",
            )
        )
        played_on = None
    else:
        played_on = unique_dates[0]

    return (
        MatchReportIdentity(
            source_match_id=match_id,
            competition_source_id=competition_id,
            season_source_id=season_id,
            played_on=played_on,
            home_source_id=home_team_id,
            away_source_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
        ),
        tuple(diagnostics),
    )


def _one_identity_value(
    values: list[str],
    *,
    label: str,
    missing_code: str,
    conflict_code: str,
    diagnostics: list[ParseDiagnostic],
) -> str | None:
    unique = tuple(dict.fromkeys(values))
    if not unique:
        diagnostics.append(ParseDiagnostic(missing_code, f"{label} is missing from raw page"))
        return None
    if len(unique) > 1:
        diagnostics.append(
            ParseDiagnostic(conflict_code, f"raw page contains conflicting {label}s: {unique!r}")
        )
        return None
    return unique[0]


def _score_numbers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"(?<!\d)\d+(?!\d)", value))


def _row_metrics(row: dict[str, _Cell]) -> dict[str, float | None]:
    return {name: _number(cell) for name, cell in row.items() if name not in _STRUCTURAL_FIELDS}


def _sum_table_metrics(rows) -> dict[str, float | None]:
    names = {name for row in rows for name in row}
    return {name: _sum_available(row.get(name) for row in rows) for name in names}


def _number(cell: _Cell | None) -> float | None:
    if cell is None or not cell.text:
        return None
    value = cell.text.replace(",", "").strip()
    if value.casefold() in {"-", "—", "–", "n/a", "na", "null"}:
        return None
    value = value.removesuffix("%").strip()
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _sum_available(values) -> float | None:
    available = [value for value in values if value is not None]
    return math.fsum(available) if available else None
