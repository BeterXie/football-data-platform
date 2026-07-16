"""External source adapters."""

from football_data_platform.sources.fbref_match_report import MatchReportIdentity
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
    "OfficialLineupDTO",
    "PrematchEventDTO",
    "SourceDescriptor",
    "SourceKind",
    "SourceRegistry",
    "adapt_injury_event",
    "adapt_news_evidence",
    "adapt_official_lineup",
    "adapt_suspension_event",
    "classify_confirmation",
    "validate_official_lineups",
    "MatchReportIdentity",
]
