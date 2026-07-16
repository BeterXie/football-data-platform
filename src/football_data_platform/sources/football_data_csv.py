"""Results-only adapter for Football-Data's public league CSV files."""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from football_data_platform.config import CompetitionDefinition, SeasonDefinition
from football_data_platform.domain.models import MatchStatus
from football_data_platform.sources.fbref import (
    FetchDiagnostic,
    ParseDiagnostic,
    ScheduleMatch,
    ScheduleParseResult,
)

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
COLLECTOR_VERSION = "football-data-results/1"


class FootballDataFetchError(RuntimeError):
    def __init__(self, diagnostic: FetchDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def results_url(season: SeasonDefinition) -> str:
    source = season.source("football-data")
    return f"{FOOTBALL_DATA_BASE_URL}/{source.season_id}/{source.competition_id}.csv"


def fetch_results_csv(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    observed_at: datetime | None = None,
) -> tuple[bytes, datetime]:
    observed_at = observed_at or datetime.now(UTC)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "football-data-platform/0.1 (results backfill)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read(), observed_at
    except urllib.error.HTTPError as error:
        raise FootballDataFetchError(
            FetchDiagnostic(
                code="http_error",
                url=url,
                http_status=error.code,
                message=f"Football-Data request failed with HTTP {error.code}",
                observed_at=observed_at,
            )
        ) from error
    except (OSError, TimeoutError) as error:
        raise FootballDataFetchError(
            FetchDiagnostic(
                code="network_error",
                url=url,
                http_status=None,
                message=f"Football-Data request failed: {error}",
                observed_at=observed_at,
            )
        ) from error


def parse_results_csv(
    content: bytes | str,
    *,
    competition: CompetitionDefinition,
    season: SeasonDefinition,
) -> ScheduleParseResult:
    """Parse only fixture/result columns; betting-odds columns are ignored."""

    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    reader = csv.DictReader(io.StringIO(text))
    diagnostics: list[ParseDiagnostic] = []
    matches: list[ScheduleMatch] = []
    seen: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        try:
            home_name = _required(row, "HomeTeam")
            away_name = _required(row, "AwayTeam")
            kickoff_at = _kickoff(row, competition.timezone)
            source_fixture_id = (
                f"{season.source('football-data').season_id}:{home_name}:{away_name}"
            )
            if source_fixture_id in seen:
                diagnostics.append(
                    ParseDiagnostic(
                        "duplicate_fixture_id",
                        f"duplicate fixture {source_fixture_id}",
                        row_number,
                        source_fixture_id,
                    )
                )
                continue
            seen.add(source_fixture_id)
            home_goals = _optional_int(row.get("FTHG"))
            away_goals = _optional_int(row.get("FTAG"))
            if (home_goals is None) != (away_goals is None):
                raise ValueError("fixture has only one full-time score")
            status = MatchStatus.FINISHED if home_goals is not None else MatchStatus.SCHEDULED
            matches.append(
                ScheduleMatch(
                    source_fixture_id=source_fixture_id,
                    source_match_id=None,
                    round_name=None,
                    kickoff_at=kickoff_at,
                    home_source_id=home_name,
                    home_name=home_name,
                    away_source_id=away_name,
                    away_name=away_name,
                    status=status,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    report_url=None,
                )
            )
        except ValueError as error:
            diagnostics.append(ParseDiagnostic("invalid_results_row", str(error), row_number))
    if reader.fieldnames is None:
        diagnostics.append(ParseDiagnostic("csv_header_missing", "CSV has no header"))
    return ScheduleParseResult(tuple(matches), tuple(diagnostics), max(reader.line_num - 1, 0))


def _kickoff(row: dict[str, str | None], timezone_name: str) -> datetime:
    date_text = _required(row, "Date")
    time_text = (row.get("Time") or "").strip()
    if not time_text:
        raise ValueError("fixture is missing Time")
    parsed_date = None
    for pattern in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed_date = datetime.strptime(date_text, pattern).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        raise ValueError(f"invalid Date {date_text!r}")
    try:
        parsed_time = datetime.strptime(time_text[:5], "%H:%M").time()
    except ValueError as error:
        raise ValueError(f"invalid Time {time_text!r}") from error
    return datetime.combine(parsed_date, parsed_time, tzinfo=ZoneInfo(timezone_name)).astimezone(
        UTC
    )


def _required(row: dict[str, str | None], name: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise ValueError(f"fixture is missing {name}")
    return value


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"invalid full-time goal value {value!r}") from error
    if parsed < 0:
        raise ValueError("full-time goals must be non-negative")
    return parsed
