"""External source adapters."""

from football_data_platform.sources.fbref_match_report import MatchReportIdentity
from football_data_platform.sources.official_lineup import (
    OFFICIAL_LINEUP_PARSER_VERSION,
    OFFICIAL_LINEUP_SCHEMA_VERSION,
    OfficialLineupParseResult,
    OfficialLineupPlayer,
    OfficialLineupTeam,
    parse_official_lineup_json,
)
from football_data_platform.sources.prematch import (
    AdapterDiagnostic,
    NewsEvidenceDTO,
    OfficialLineupDTO,
    PrematchEventDTO,
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
    adapt_injury_event,
    adapt_news_evidence,
    adapt_official_lineup,
    adapt_suspension_event,
    classify_confirmation,
    validate_official_lineups,
)

__all__ = [
    "AdapterDiagnostic",
    "NewsEvidenceDTO",
    "OFFICIAL_LINEUP_PARSER_VERSION",
    "OFFICIAL_LINEUP_SCHEMA_VERSION",
    "OfficialLineupDTO",
    "OfficialLineupParseResult",
    "OfficialLineupPlayer",
    "OfficialLineupTeam",
    "PrematchEventDTO",
    "SourceDescriptor",
    "SourceKind",
    "SourceRegistry",
    "adapt_injury_event",
    "adapt_news_evidence",
    "adapt_official_lineup",
    "adapt_suspension_event",
    "classify_confirmation",
    "parse_official_lineup_json",
    "validate_official_lineups",
    "MatchReportIdentity",
]
