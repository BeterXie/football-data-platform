"""Generic FBref schedule acquisition and parsing."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.domain.models import MatchStatus

FBREF_BASE_URL = "https://fbref.com"
COLLECTOR_VERSION = "fbref-schedule/1"
_TEAM_ID = re.compile(r"/squads/([0-9a-z]+)/", re.IGNORECASE)
_MATCH_ID = re.compile(r"/matches/([0-9a-z]+)/", re.IGNORECASE)
_SCORE = re.compile(r"^\s*(\d+)\s*[\-\u2013\u2014]\s*(\d+)\s*$")


@dataclass(frozen=True, slots=True)
class ScheduleMatch:
    source_fixture_id: str
    source_match_id: str | None
    round_name: str | None
    kickoff_at: datetime | None
    home_source_id: str
    home_name: str
    away_source_id: str
    away_name: str
    status: MatchStatus
    home_goals: int | None
    away_goals: int | None
    report_url: str | None
    fixture_known_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ParseDiagnostic:
    code: str
    message: str
    row_number: int | None = None
    subject_id: str | None = None
    table_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduleParseResult:
    matches: tuple[ScheduleMatch, ...]
    diagnostics: tuple[ParseDiagnostic, ...]
    rows_seen: int
    fixture_known_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class FetchDiagnostic:
    code: str
    url: str
    http_status: int | None
    message: str
    observed_at: datetime
    body: bytes | None = None

    @property
    def response_body(self) -> bytes | None:
        """Compatibility alias for callers that describe the response explicitly."""

        return self.body

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "url": self.url,
            "http_status": self.http_status,
            "message": self.message,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
        }
        if self.body is not None:
            # Keep command summaries bounded; callers archive ``body`` itself as raw evidence.
            payload["body_present"] = True
            payload["body_size_bytes"] = len(self.body)
            payload["body_preview"] = self.body[:512].decode("utf-8", errors="replace")
        return payload


class FBrefFetchError(RuntimeError):
    def __init__(self, diagnostic: FetchDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class FBrefAccessBlockedError(FBrefFetchError):
    """The source presented an access-control challenge or denial."""


@dataclass(slots=True)
class _Cell:
    text_parts: list[str] = field(default_factory=list)
    href: str | None = None

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


class _CommentCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_fragments: list[str] = []

    def handle_comment(self, data: str) -> None:
        if "<table" in data.lower():
            self.table_fragments.append(data)


class _ScheduleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_schedule_table = False
        self.current_row: dict[str, _Cell] | None = None
        self.current_cell: _Cell | None = None
        self.current_stat: str | None = None
        self.rows: list[dict[str, _Cell]] = []
        self.fixture_known_at_values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            table_id = attributes.get("id") or ""
            if table_id.startswith("sched_"):
                self.in_schedule_table = True
                fixture_known_at = attributes.get("data-fdp-fixture-known-at")
                if fixture_known_at:
                    self.fixture_known_at_values.append(fixture_known_at)
        elif self.in_schedule_table and tag == "tr":
            self.current_row = {}
            fixture_known_at = attributes.get("data-fdp-fixture-known-at")
            if fixture_known_at:
                self.current_row["__fdp_fixture_known_at"] = _Cell(text_parts=[fixture_known_at])
        elif self.in_schedule_table and self.current_row is not None and tag in {"td", "th"}:
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
        elif tag == "tr" and self.in_schedule_table and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_schedule_table:
            self.in_schedule_table = False


def schedule_url(
    competition: CompetitionDefinition,
    season: SeasonDefinition,
) -> str:
    """Build the registered FBref schedule URL without competition-specific branches."""

    source = season.source("fbref")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", competition.name).strip("-")
    return (
        f"{FBREF_BASE_URL}/en/comps/{source.competition_id}/{source.season_id}/"
        f"schedule/{source.season_id}-{slug}-Scores-and-Fixtures"
    )


def fetch_schedule(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    observed_at: datetime | None = None,
) -> tuple[bytes, datetime]:
    """Fetch one schedule page or raise with a machine-readable diagnostic."""

    observed_at = observed_at or datetime.now(UTC)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "football-data-platform/0.1 "
                "(zero-budget research collector; contact local operator)"
            )
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content = response.read()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as error:
        code = "blocked_by_access_control" if error.code in {403, 429} else "http_error"
        body = _read_http_error_body(error)
        diagnostic = FetchDiagnostic(
            code=code,
            url=url,
            http_status=error.code,
            message=f"FBref request failed with HTTP {error.code}",
            observed_at=observed_at,
            body=body,
        )
        exception_type = (
            FBrefAccessBlockedError if code == "blocked_by_access_control" else FBrefFetchError
        )
        raise exception_type(diagnostic) from error
    except (OSError, TimeoutError) as error:
        raise FBrefFetchError(
            FetchDiagnostic(
                code="network_error",
                url=url,
                http_status=None,
                message=f"FBref request failed: {error}",
                observed_at=observed_at,
            )
        ) from error

    if status in {403, 429} or _looks_like_access_challenge(content):
        raise FBrefAccessBlockedError(
            FetchDiagnostic(
                code="blocked_by_access_control",
                url=url,
                http_status=status,
                message="FBref returned an access-control challenge",
                observed_at=observed_at,
                body=content,
            )
        )
    return content, observed_at


def parse_schedule(
    content: bytes | str,
    *,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    page_url: str,
) -> ScheduleParseResult:
    """Parse every league schedule row, including FBref comment-wrapped tables."""

    html = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    if _looks_like_access_challenge(html.encode("utf-8")):
        raise FBrefAccessBlockedError(
            FetchDiagnostic(
                code="blocked_by_access_control",
                url=page_url,
                http_status=None,
                message="FBref payload contains an access-control challenge",
                observed_at=datetime.now(UTC),
                body=html.encode("utf-8"),
            )
        )

    comments = _CommentCollector()
    comments.feed(html)
    table_parser = _ScheduleTableParser()
    table_parser.feed(html)
    for fragment in comments.table_fragments:
        table_parser.feed(fragment)

    diagnostics: list[ParseDiagnostic] = []
    matches: list[ScheduleMatch] = []
    seen_fixture_ids: set[str] = set()
    for row_number, row in enumerate(table_parser.rows, start=1):
        try:
            match, kickoff = _parse_row(
                row,
                competition=competition,
                season=season,
                page_url=page_url,
            )
        except ValueError as error:
            diagnostics.append(ParseDiagnostic("invalid_schedule_row", str(error), row_number))
            continue
        if match is None:
            continue
        if kickoff is not None and kickoff.diagnostic_code is not None:
            diagnostics.append(
                ParseDiagnostic(
                    kickoff.diagnostic_code,
                    kickoff.diagnostic_message or "fixture kickoff is unknown",
                    row_number,
                    match.source_fixture_id,
                )
            )
        if match.source_fixture_id in seen_fixture_ids:
            diagnostics.append(
                ParseDiagnostic(
                    "duplicate_fixture_id",
                    f"duplicate fixture {match.source_fixture_id}",
                    row_number,
                    match.source_fixture_id,
                )
            )
            continue
        seen_fixture_ids.add(match.source_fixture_id)
        matches.append(match)

    if not table_parser.rows:
        diagnostics.append(
            ParseDiagnostic("schedule_table_missing", "no FBref schedule table found")
        )
    return ScheduleParseResult(
        tuple(matches),
        tuple(diagnostics),
        len(table_parser.rows),
        _fixture_known_at(table_parser.fixture_known_at_values),
    )


def _fixture_known_at(values: list[str]) -> datetime | None:
    """Parse optional replay metadata carried by the bundled schedule fixture."""

    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError("schedule fixture has conflicting known-at metadata")
    try:
        value = datetime.fromisoformat(values[0].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("schedule fixture known-at metadata must be an ISO timestamp") from error
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError("schedule fixture known-at metadata must use UTC")
    return value.astimezone(UTC)


def _parse_row(
    row: dict[str, _Cell],
    *,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
    page_url: str,
) -> tuple[ScheduleMatch | None, _KickoffParse | None]:
    home = row.get("home_team")
    away = row.get("away_team")
    if home is None and away is None:
        return None, None
    if home is None or away is None:
        raise ValueError("fixture has only one team")
    home_source_id = _id_from_href(home.href, _TEAM_ID, "home team")
    away_source_id = _id_from_href(away.href, _TEAM_ID, "away team")
    if not home.text or not away.text:
        raise ValueError("fixture has an empty team name")

    round_name = _optional_text(row, "round")
    match_report = row.get("match_report")
    report_url = (
        urljoin(page_url, match_report.href)
        if match_report is not None and match_report.href
        else None
    )
    source_match_id = (
        _match_id_from_href(match_report.href)
        if match_report is not None and match_report.href
        else None
    )
    source_fixture_id = source_match_id or ":".join(
        (season.source("fbref").season_id, home_source_id, away_source_id)
    )
    kickoff = _parse_kickoff(row, competition.timezone)
    score_text = _optional_text(row, "score")
    home_goals: int | None = None
    away_goals: int | None = None
    status = MatchStatus.SCHEDULED
    if score_text:
        score = _SCORE.match(score_text)
        if score is not None:
            home_goals, away_goals = (int(score.group(1)), int(score.group(2)))
            status = MatchStatus.FINISHED
        elif "postpon" in score_text.casefold():
            status = MatchStatus.POSTPONED
        elif "cancel" in score_text.casefold():
            status = MatchStatus.CANCELLED
        else:
            raise ValueError(f"unrecognized score value {score_text!r}")

    return (
        ScheduleMatch(
            source_fixture_id=source_fixture_id,
            source_match_id=source_match_id,
            round_name=round_name,
            kickoff_at=kickoff.value,
            home_source_id=home_source_id,
            home_name=home.text,
            away_source_id=away_source_id,
            away_name=away.text,
            status=status,
            home_goals=home_goals,
            away_goals=away_goals,
            report_url=report_url,
            fixture_known_at=_row_fixture_known_at(row),
        ),
        kickoff,
    )


@dataclass(frozen=True, slots=True)
class _KickoffParse:
    value: datetime | None
    diagnostic_code: str | None = None
    diagnostic_message: str | None = None


def _parse_kickoff(row: dict[str, _Cell], timezone_name: str) -> _KickoffParse:
    date_text = _optional_text(row, "date")
    time_text = _optional_text(row, "start_time") or _optional_text(row, "time")
    if not date_text:
        return _KickoffParse(
            None,
            "kickoff_missing",
            "fixture has no date; kickoff is unknown",
        )
    if not time_text:
        return _KickoffParse(
            None,
            "kickoff_missing",
            "fixture has no start_time; kickoff is unknown",
        )
    try:
        local = datetime.strptime(f"{date_text} {time_text[:5]}", "%Y-%m-%d %H:%M")
    except ValueError:
        return _KickoffParse(
            None,
            "kickoff_invalid",
            f"invalid kickoff {date_text!r} {time_text!r}; kickoff is unknown",
        )
    return _KickoffParse(local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(UTC))


def _row_fixture_known_at(row: dict[str, _Cell]) -> datetime | None:
    cell = row.get("__fdp_fixture_known_at")
    if cell is None:
        return None
    return _fixture_known_at([cell.text])


def _read_http_error_body(error: urllib.error.HTTPError) -> bytes | None:
    try:
        body = error.read()
    except (OSError, ValueError):
        return None
    return body if isinstance(body, bytes) else None


def _id_from_href(href: str | None, pattern: re.Pattern[str], label: str) -> str:
    match = pattern.search(href or "")
    if match is None:
        raise ValueError(f"{label} link has no source ID")
    return match.group(1)


def _match_id_from_href(href: str) -> str | None:
    match = _MATCH_ID.search(href)
    return None if match is None else match.group(1)


def _optional_text(row: dict[str, _Cell], key: str) -> str | None:
    cell = row.get(key)
    return cell.text if cell is not None and cell.text else None


def _looks_like_access_challenge(content: bytes) -> bool:
    sample = content[:200_000].lower()
    markers = (b"cf-chl-", b"cloudflare", b"challenge-platform")
    return any(marker in sample for marker in markers)
