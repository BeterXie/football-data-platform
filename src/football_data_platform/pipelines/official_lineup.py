"""Archive, normalize, and replay one paired official-lineup JSON payload."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from football_data_platform.domain.ids import MatchId, RawAssetId
from football_data_platform.domain.models import require_utc
from football_data_platform.sources.official_lineup import OFFICIAL_LINEUP_PARSER_VERSION
from football_data_platform.sources.prematch import SourceRegistry
from football_data_platform.storage.canonical import (
    CanonicalConflictError,
    CanonicalStore,
    OfficialLineupContractEvidence,
)
from football_data_platform.storage.facts import (
    CanonicalFactStore,
    OfficialLineupRawParseError,
    verify_official_lineup_contract,
)
from football_data_platform.storage.raw import RawArchive


@dataclass(frozen=True, slots=True)
class OfficialLineupIngestResult:
    raw_asset_id: str
    contract_id: str
    match_id: MatchId
    match_version: int
    fact_ids: tuple[str, ...]


class OfficialLineupIngestError(ValueError):
    """Rejected official-lineup evidence that remains available in the raw archive."""

    def __init__(self, code: str, message: str, *, raw_asset_id: RawAssetId) -> None:
        super().__init__(message)
        self.code = code
        self.raw_asset_id = raw_asset_id


def ingest_official_lineup_json(
    content: bytes,
    *,
    source: str,
    source_match_id: str,
    page_url: str,
    observed_at: datetime,
    archive: RawArchive,
    canonical: CanonicalStore,
    source_registry: SourceRegistry,
) -> OfficialLineupIngestResult:
    """Archive raw bytes before parsing and persist their verified paired XIs."""

    require_utc(observed_at, "observed_at")
    asset = archive.archive(
        content,
        source=source,
        source_id=source_match_id,
        url=page_url,
        observed_at=observed_at,
        target_event_time=None,
        collector_version=OFFICIAL_LINEUP_PARSER_VERSION,
        media_type="application/json",
    )
    canonical.register_raw_asset(asset)
    facts = CanonicalFactStore(canonical, source_registry=source_registry)
    try:
        contract, _ = facts.append_official_lineup_asset(
            archive=archive,
            raw_asset_id=asset.id,
        )
    except OfficialLineupRawParseError as error:
        raise OfficialLineupIngestError(
            "official_lineup_parse_failed",
            str(error),
            raw_asset_id=asset.id,
        ) from error
    except (CanonicalConflictError, KeyError, TypeError, ValueError) as error:
        raise OfficialLineupIngestError(
            "official_lineup_validation_failed",
            str(error),
            raw_asset_id=asset.id,
        ) from error
    try:
        contract = verify_official_lineup_contract(
            contract.contract_id,
            archive=archive,
            canonical=canonical,
        )
    except (CanonicalConflictError, KeyError, TypeError, ValueError) as error:
        raise OfficialLineupIngestError(
            "official_lineup_contract_replay_failed",
            str(error),
            raw_asset_id=asset.id,
        ) from error

    return OfficialLineupIngestResult(
        raw_asset_id=asset.id.value,
        contract_id=contract.contract_id,
        match_id=contract.match_id,
        match_version=contract.match_version,
        fact_ids=contract.fact_ids,
    )


def replay_official_lineup_contract(
    contract_id: str,
    *,
    archive: RawArchive,
    canonical: CanonicalStore,
) -> OfficialLineupContractEvidence:
    """Compatibility entry point for the shared storage-level verifier."""

    return verify_official_lineup_contract(
        contract_id,
        archive=archive,
        canonical=canonical,
    )
