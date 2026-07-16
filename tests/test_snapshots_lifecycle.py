from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from football_data_platform.config import load_competition_registry
from football_data_platform.domain import snapshots as snapshot_contract
from football_data_platform.domain.ids import (
    CompetitionId,
    MatchId,
    PlayerId,
    SeasonId,
    SnapshotId,
    TeamId,
)
from football_data_platform.domain.lifecycle import (
    LifecycleState,
    MatchAvailability,
    PlayerObservationAvailability,
    Qualification,
    SnapshotAvailability,
    assess_lifecycle,
)
from football_data_platform.domain.models import MatchStatus
from football_data_platform.domain.snapshots import (
    CaptureMode,
    SnapshotFeature,
    SnapshotSourceValidation,
    SnapshotType,
    build_snapshot,
    verify_snapshot,
)
from football_data_platform.features.lineup import (
    LINEUP_DELTA_INPUT_TRANSFORM_V3,
    lineup_delta_input_payload,
)
from football_data_platform.features.team_baseline import (
    TeamMatchProcess,
    build_team_baseline,
    expected_goals_from_baseline,
    team_baseline_payload,
)
from football_data_platform.sources.prematch import (
    OfficialLineupDTO,
    SourceDescriptor,
    SourceKind,
    SourceRegistry,
)
from football_data_platform.storage.canonical import CanonicalStore
from football_data_platform.storage.derived import DerivedArchive
from football_data_platform.storage.facts import CanonicalFactStore
from football_data_platform.storage.layout import DataLayout
from football_data_platform.storage.raw import (
    ArchiveConflictError,
    ChecksumMismatchError,
    RawArchive,
)

KICKOFF = datetime(2025, 8, 16, 14, 0, tzinfo=UTC)
OBSERVED_AT = KICKOFF + timedelta(days=1)
HOME = TeamId("team:home")
AWAY = TeamId("team:away")
MATCH = MatchId("match:test")
ROOT = Path(__file__).parents[1]


def _official_lineup_fact_store(canonical: CanonicalStore) -> CanonicalFactStore:
    return CanonicalFactStore(
        canonical,
        source_registry=SourceRegistry(
            (
                SourceDescriptor(
                    "official-lineup-test",
                    SourceKind.OFFICIAL_LINEUP,
                    "official-lineup-test",
                    official=True,
                ),
            )
        ),
    )


class _CompositeSnapshotValidator:
    def __init__(
        self,
        derived: DerivedArchive,
        overrides: dict[str, SnapshotSourceValidation],
    ) -> None:
        self.derived = derived
        self.overrides = overrides

    def validate_snapshot_source(self, source_ref: str) -> SnapshotSourceValidation:
        return self.overrides.get(source_ref) or self.derived.validate_snapshot_source(source_ref)


def _stores(tmp_path: Path, *, observed_at: datetime = OBSERVED_AT):
    layout = DataLayout(tmp_path / "data")
    raw = RawArchive(layout)
    asset = raw.archive(
        b"fixture evidence",
        source="test-source",
        source_id="fixture-evidence",
        url="fixture://evidence",
        observed_at=observed_at,
        target_event_time=None,
        collector_version="test-collector/1",
        media_type="application/octet-stream",
    )
    return raw, DerivedArchive(layout), asset


def _strict_official_lineup_context(tmp_path: Path, *, fact_count: int = 11):
    layout = DataLayout(tmp_path / "strict-data")
    observed_at = KICKOFF - timedelta(hours=1)
    raw = RawArchive(layout)
    asset = raw.archive(
        b"official lineup evidence",
        source="official-lineup-test",
        source_id="official-lineup",
        url="fixture://official-lineup",
        observed_at=observed_at,
        target_event_time=KICKOFF,
        collector_version="test/1",
        media_type="application/json",
    )
    canonical = CanonicalStore(layout.canonical / "platform.sqlite3")
    canonical.initialize()
    canonical.register_registry(
        load_competition_registry(ROOT / "config" / "competitions.toml"),
        registered_at=observed_at,
    )
    canonical.register_raw_asset(asset)
    teams = tuple(
        canonical.resolve_or_create_team(
            source="fbref",
            source_id=f"strict-team-{index}",
            canonical_name=f"Strict Team {index}",
            competition_id=CompetitionId("competition:eng.1"),
            observed_at=observed_at,
            raw_asset_id=asset.id,
        )
        for index in range(2)
    )
    match, version = canonical.resolve_or_create_match(
        source="fbref-schedule",
        source_id="strict-fixture",
        competition_id=CompetitionId("competition:eng.1"),
        season_id=SeasonId("season:eng.1.2025-26"),
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        kickoff_at=KICKOFF,
        status=MatchStatus.SCHEDULED,
        observed_at=observed_at,
        raw_asset_id=asset.id,
    )
    players = tuple(
        canonical.resolve_or_create_player(
            source="official-lineup-test",
            source_id=f"strict-player-{index}",
            canonical_name=f"Strict Player {index}",
            observed_at=observed_at,
            raw_asset_id=asset.id,
        ).id
        for index in range(11)
    )
    facts = _official_lineup_fact_store(canonical)
    facts.append_official_lineup(
        OfficialLineupDTO(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[0].id,
            player_ids=players,
            source="official-lineup-test",
            published_at=observed_at,
            observed_at=observed_at,
            raw_asset_id=asset.id,
            url="fixture://official-lineup",
        )
    )
    if fact_count < len(players):
        with canonical.connect() as connection:
            placeholders = ",".join("?" for _ in players[fact_count:])
            connection.execute(
                "DELETE FROM lineup_facts WHERE match_id = ? AND team_id = ? "
                f"AND player_id IN ({placeholders})",
                (
                    match.id.value,
                    teams[0].id.value,
                    *(player_id.value for player_id in players[fact_count:]),
                ),
            )
    return canonical, raw, DerivedArchive(layout), asset, teams, match, version, players


def _ready_delta_item(
    player_ids: tuple[PlayerId, ...],
    raw_ref: str,
    profile_ref: str,
    reference_lineup_ref: str,
) -> dict:
    return {
        "quality_status": "ready",
        "dimension_deltas": {"attack": 0.0},
        "missing_fields": [],
        "input_refs": [profile_ref],
        "not_applicable_fields": [],
        "blocking_reasons": [],
        "transform_version": "lineup-delta/2",
        "role_contract_versions": ["lineup-role-metrics/1"],
        "dimension_sample_sizes": {"attack": [11, 11]},
        "starter_ids": [player_id.value for player_id in player_ids],
        "reference_starter_ids": [player_id.value for player_id in player_ids],
        "player_profile_refs": {player_id.value: profile_ref for player_id in player_ids},
        "profile_window": "recent_form",
        "profile_window_version": "recent/1",
        "lineup_input_refs": [raw_ref],
        "reference_lineup_ref": reference_lineup_ref,
    }


def _source(derived: DerivedArchive, asset, value, transform: str) -> str:
    return derived.write_snapshot_source(
        value=value,
        input_refs=(asset.id,),
        transform_version=transform,
        generated_at=asset.observed_at,
    )


def test_official_lineup_v2_round_trips_exact_canonical_facts(tmp_path: Path) -> None:
    _, _, derived, asset, teams, match, version, players = _strict_official_lineup_context(tmp_path)

    source_ref = derived.write_official_lineup_source(
        match_id=match.id,
        match_version=version.version,
        team_id=teams[0].id,
        player_ids=players,
        known_at=asset.observed_at,
        observed_at=asset.observed_at,
        raw_asset_id=asset.id,
    )

    validation = derived.validate_snapshot_source(source_ref)
    assert validation.transform_version == "official-lineup-input/2"
    assert validation.value == [player_id.value for player_id in players]
    assert validation.source_context == {
        "match_id": match.id.value,
        "match_version": version.version,
        "team_id": teams[0].id.value,
        "player_ids": [player_id.value for player_id in players],
        "known_at": asset.observed_at.isoformat().replace("+00:00", "Z"),
        "observed_at": asset.observed_at.isoformat().replace("+00:00", "Z"),
    }


@pytest.mark.parametrize(
    "mismatch",
    ("raw", "match", "version", "team", "players", "known_at", "observed_at"),
)
def test_official_lineup_v2_rejects_noncanonical_context(tmp_path: Path, mismatch: str) -> None:
    canonical, raw, derived, asset, teams, match, version, players = (
        _strict_official_lineup_context(tmp_path)
    )
    arguments = {
        "match_id": match.id,
        "match_version": version.version,
        "team_id": teams[0].id,
        "player_ids": players,
        "known_at": asset.observed_at,
        "observed_at": asset.observed_at,
        "raw_asset_id": asset.id,
    }
    if mismatch == "raw":
        unrelated = raw.archive(
            b"unrelated report or news evidence",
            source="news-test",
            source_id="unrelated-report",
            url="fixture://unrelated-report",
            observed_at=asset.observed_at,
            target_event_time=KICKOFF,
            collector_version="test/1",
            media_type="text/html",
        )
        canonical.register_raw_asset(unrelated)
        arguments["raw_asset_id"] = unrelated.id
    elif mismatch == "match":
        arguments["match_id"] = MatchId("match:wrong")
    elif mismatch == "version":
        arguments["match_version"] = version.version + 1
    elif mismatch == "team":
        arguments["team_id"] = teams[1].id
    elif mismatch == "players":
        arguments["player_ids"] = (*players[:-1], PlayerId("player:wrong"))
    elif mismatch == "known_at":
        arguments["known_at"] = asset.observed_at - timedelta(minutes=1)
    else:
        arguments["observed_at"] = asset.observed_at + timedelta(minutes=1)

    with pytest.raises(ArchiveConflictError, match="canonical lineup facts"):
        derived.write_official_lineup_source(**arguments)


def test_official_lineup_v2_rejects_missing_canonical_starter_fact(tmp_path: Path) -> None:
    _, _, derived, asset, teams, match, version, players = _strict_official_lineup_context(
        tmp_path, fact_count=10
    )

    with pytest.raises(ArchiveConflictError, match="canonical lineup facts"):
        derived.write_official_lineup_source(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[0].id,
            player_ids=players,
            known_at=asset.observed_at,
            observed_at=asset.observed_at,
            raw_asset_id=asset.id,
        )


def test_ready_lineup_snapshot_accepts_strict_official_sources_and_ready_delta(
    tmp_path: Path,
) -> None:
    canonical, _, derived, asset, teams, match, version, home_players = (
        _strict_official_lineup_context(tmp_path)
    )
    facts = _official_lineup_fact_store(canonical)
    away_players = tuple(
        canonical.resolve_or_create_player(
            source="official-lineup-test",
            source_id=f"strict-away-player-{index}",
            canonical_name=f"Strict Away Player {index}",
            observed_at=asset.observed_at,
            raw_asset_id=asset.id,
        ).id
        for index in range(11)
    )
    facts.append_official_lineup(
        OfficialLineupDTO(
            match_id=match.id,
            match_version=version.version,
            team_id=teams[1].id,
            player_ids=away_players,
            source="official-lineup-test",
            published_at=asset.observed_at,
            observed_at=asset.observed_at,
            raw_asset_id=asset.id,
            url="fixture://official-lineup",
        )
    )
    official_sources = {
        team.id.value: derived.write_official_lineup_source(
            match_id=match.id,
            match_version=version.version,
            team_id=team.id,
            player_ids=players,
            known_at=asset.observed_at,
            observed_at=asset.observed_at,
            raw_asset_id=asset.id,
        )
        for team, players in zip(teams, (home_players, away_players), strict=True)
    }
    baseline_ref = "derived-source:" + "a" * 64
    context_ref = "derived-source:" + "b" * 64
    delta_ref = "derived-source:" + "c" * 64
    baseline = {
        "artifact_id": "team-baseline:" + "a" * 64,
        "lambda_home": 1.5,
        "lambda_away": 1.0,
    }
    context = {"days_since_previous_match": 6.0}
    home_profile_ref = "player-profile:" + "d" * 64
    away_profile_ref = "player-profile:" + "e" * 64
    delta = {
        teams[0].id.value: _ready_delta_item(
            home_players,
            asset.id.value,
            home_profile_ref,
            official_sources[teams[0].id.value],
        ),
        teams[1].id.value: _ready_delta_item(
            away_players,
            asset.id.value,
            away_profile_ref,
            official_sources[teams[1].id.value],
        ),
    }
    overrides = {
        baseline_ref: SnapshotSourceValidation(
            baseline_ref,
            "derived",
            "team-baseline-input/2",
            asset.observed_at,
            baseline,
            (asset.id.value,),
        ),
        context_ref: SnapshotSourceValidation(
            context_ref,
            "derived",
            "match-context-input/1",
            asset.observed_at,
            context,
            (asset.id.value,),
        ),
        delta_ref: SnapshotSourceValidation(
            delta_ref,
            "derived",
            LINEUP_DELTA_INPUT_TRANSFORM_V3,
            asset.observed_at,
            delta,
            tuple(sorted((asset.id.value, home_profile_ref, away_profile_ref))),
            asset.observed_at,
        ),
    }
    features = (
        SnapshotFeature("team_baseline", baseline, asset.observed_at, baseline_ref, "baseline"),
        SnapshotFeature("match_context", context, asset.observed_at, context_ref, "context"),
        SnapshotFeature("lineup_delta", delta, asset.observed_at, delta_ref, "lineup-delta"),
        *(
            SnapshotFeature(
                "official_lineup_confirmed",
                [player_id.value for player_id in players],
                asset.observed_at,
                official_sources[team.id.value],
                f"official-lineup:{team.id.value}",
                team.id.value,
            )
            for team, players in zip(teams, (home_players, away_players), strict=True)
        ),
    )

    snapshot = build_snapshot(
        match_id=match.id,
        match_version=version.version,
        snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
        as_of=asset.observed_at,
        scheduled_kickoff_used=KICKOFF,
        feature_spec_version="prematch-features/2",
        features=features,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
        source_validator=_CompositeSnapshotValidator(derived, overrides),
    )

    assert snapshot.quality_status == "ready"


def test_ready_lineup_snapshot_rejects_legacy_official_sources(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home_players = tuple(PlayerId(f"player:home-{index}") for index in range(11))
    away_players = tuple(PlayerId(f"player:away-{index}") for index in range(11))
    home_reference_ref = "derived-source:" + "f" * 64
    away_reference_ref = "derived-source:" + "9" * 64
    official = tuple(
        _lineup_feature(
            derived,
            asset,
            team_id=team_id,
            value=[player_id.value for player_id in players],
            known_at=as_of,
        )
        for team_id, players in ((HOME, home_players), (AWAY, away_players))
    )
    delta_ref = "derived-source:" + "c" * 64
    delta = {
        HOME.value: _ready_delta_item(
            home_players,
            asset.id.value,
            "player-profile:" + "d" * 64,
            home_reference_ref,
        ),
        AWAY.value: _ready_delta_item(
            away_players,
            asset.id.value,
            "player-profile:" + "e" * 64,
            away_reference_ref,
        ),
    }
    validation = SnapshotSourceValidation(
        delta_ref,
        "derived",
        LINEUP_DELTA_INPUT_TRANSFORM_V3,
        asset.observed_at,
        delta,
        (
            asset.id.value,
            home_reference_ref,
            away_reference_ref,
            "player-profile:" + "d" * 64,
            "player-profile:" + "e" * 64,
        ),
        as_of,
    )

    with pytest.raises(ValueError, match="requires official-lineup-input/2"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/2",
            features=(
                *_t24_features(derived, asset),
                SnapshotFeature("lineup_delta", delta, as_of, delta_ref, "lineup-delta"),
                *official,
            ),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=_CompositeSnapshotValidator(derived, {delta_ref: validation}),
        )


def _t24_features(
    derived: DerivedArchive,
    asset,
    *,
    known_at: datetime | None = None,
) -> tuple[SnapshotFeature, SnapshotFeature]:
    feature_known_at = known_at or KICKOFF - timedelta(days=2)
    baseline_artifact = build_team_baseline(
        (
            TeamMatchProcess(
                "match-snapshot-baseline",
                HOME.value,
                AWAY.value,
                KICKOFF - timedelta(days=30),
                KICKOFF - timedelta(days=3),
                1.5,
                1.0,
                asset.id.value,
            ),
        ),
        as_of=KICKOFF - timedelta(days=2),
        half_life_days=90,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline_artifact)
    lambda_home, lambda_away = expected_goals_from_baseline(
        baseline_artifact,
        home_team_id=HOME.value,
        away_team_id=AWAY.value,
    )
    baseline = {
        "artifact_id": baseline_artifact.artifact_id,
        "artifact": team_baseline_payload(baseline_artifact),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
    context = {"days_since_previous_match": 6.0}
    return (
        SnapshotFeature(
            "team_baseline",
            baseline,
            feature_known_at,
            _source(derived, asset, baseline, "team-baseline-input/2"),
            "team-baseline",
        ),
        SnapshotFeature(
            "match_context",
            context,
            feature_known_at,
            _source(derived, asset, context, "match-context-input/1"),
            "context:rest-days",
        ),
    )


def _snapshot_arguments(derived: DerivedArchive) -> dict:
    return {
        "match_id": MATCH,
        "match_version": 1,
        "snapshot_type": SnapshotType.T24H,
        "as_of": KICKOFF - timedelta(hours=24),
        "scheduled_kickoff_used": KICKOFF,
        "feature_spec_version": "prematch-features/1",
        "home_team_id": HOME,
        "away_team_id": AWAY,
        "source_validator": derived,
    }


def test_t24_snapshot_rejects_future_information(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=24)

    with pytest.raises(ValueError, match="known after"):
        build_snapshot(
            features=_t24_features(derived, asset, known_at=as_of + timedelta(minutes=1)),
            **_snapshot_arguments(derived),
        )


def test_team_baseline_source_rejects_fake_artifact_identity(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline = build_team_baseline(
        (
            TeamMatchProcess(
                "match-a",
                "team:home",
                "team:away",
                KICKOFF - timedelta(days=30),
                KICKOFF - timedelta(days=30),
                1.5,
                1.0,
                asset.id.value,
            ),
        ),
        as_of=KICKOFF - timedelta(days=1),
        half_life_days=90,
        iterations=2,
    ).artifact
    derived.write_team_baseline(baseline)
    payload = {
        "artifact_id": "team-baseline:" + "f" * 64,
        "artifact": team_baseline_payload(baseline),
        "lambda_home": 1.5,
        "lambda_away": 1.0,
    }
    source_ref = derived.write_snapshot_source(
        value=payload,
        input_refs=(asset.id,),
        transform_version="team-baseline-input/2",
        generated_at=asset.observed_at,
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        derived.validate_snapshot_source(source_ref)


def test_raw_observation_time_cannot_self_authorize_captured_mode(tmp_path: Path) -> None:
    as_of = KICKOFF - timedelta(hours=24)
    _, derived, asset = _stores(tmp_path, observed_at=as_of)

    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )

    assert snapshot.capture_mode is CaptureMode.RECONSTRUCTED


def test_source_validator_rejects_modified_raw_content(tmp_path: Path) -> None:
    raw, derived, asset = _stores(tmp_path)
    features = _t24_features(derived, asset)
    raw.layout.raw_object_path(asset.checksum).write_bytes(b"modified")

    with pytest.raises(ChecksumMismatchError, match="checksum"):
        build_snapshot(features=features, **_snapshot_arguments(derived))


def test_source_validator_rejects_missing_or_mismatched_derived_source(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, context = _t24_features(derived, asset)
    mismatched = replace(
        baseline,
        value={**baseline.value, "lambda_home": 9.0},
    )

    with pytest.raises(ValueError, match="derived source value"):
        build_snapshot(features=(mismatched, context), **_snapshot_arguments(derived))
    wrong_transform = replace(
        context,
        source_ref=_source(derived, asset, context.value, "unexpected-transform/1"),
    )
    with pytest.raises(ValueError, match="source transform"):
        build_snapshot(
            features=(baseline, wrong_transform),
            **_snapshot_arguments(derived),
        )
    with pytest.raises(FileNotFoundError):
        build_snapshot(
            features=(replace(baseline, source_ref="derived-source:" + "f" * 64), context),
            **_snapshot_arguments(derived),
        )


def test_feature_spec_computes_preview_instead_of_trusting_missing_fields(
    tmp_path: Path,
) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, _ = _t24_features(derived, asset)

    snapshot = build_snapshot(features=(baseline,), **_snapshot_arguments(derived))

    assert snapshot.quality_status == "preview"
    assert snapshot.missing_fields == ("feature:match_context",)


def _lineup_feature(
    derived: DerivedArchive,
    asset,
    *,
    team_id: TeamId,
    value,
    known_at: datetime,
) -> SnapshotFeature:
    return SnapshotFeature(
        "official_lineup_confirmed",
        value,
        known_at,
        _source(derived, asset, value, "official-lineup-input/1"),
        f"official-lineup:{team_id.value}",
        team_id.value,
    )


def _v3_lineup_delta_feature(
    derived: DerivedArchive,
    asset,
    *,
    home_starters: tuple[str, ...],
    away_starters: tuple[str, ...],
    known_at: datetime,
) -> SnapshotFeature:
    value = {
        team_id.value: lineup_delta_input_payload(
            starter_ids=starters,
            reference_starter_ids=None,
            profiles=(),
            lineup_input_refs=(asset.id.value,),
        )
        for team_id, starters in ((HOME, home_starters), (AWAY, away_starters))
    }
    source_ref = derived.write_snapshot_source(
        value=value,
        input_refs=(asset.id,),
        transform_version=LINEUP_DELTA_INPUT_TRANSFORM_V3,
        generated_at=asset.observed_at,
        known_at=known_at,
    )
    return SnapshotFeature("lineup_delta", value, known_at, source_ref, "lineup-delta")


def test_lineups_snapshot_requires_both_official_lineups(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home = _lineup_feature(
        derived,
        asset,
        team_id=HOME,
        value=[f"player:home-{index}" for index in range(11)],
        known_at=as_of,
    )

    with pytest.raises(ValueError, match="both teams"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/2",
            features=(*_t24_features(derived, asset), home),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_lineups_snapshot_rejects_null_or_invalid_player_ids(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    away = _lineup_feature(
        derived,
        asset,
        team_id=AWAY,
        value=[f"player:away-{index}" for index in range(11)],
        known_at=as_of,
    )
    arguments = {
        "match_id": MATCH,
        "match_version": 1,
        "snapshot_type": SnapshotType.LINEUPS_CONFIRMED,
        "as_of": as_of,
        "scheduled_kickoff_used": KICKOFF,
        "feature_spec_version": "prematch-features/2",
        "home_team_id": HOME,
        "away_team_id": AWAY,
        "source_validator": derived,
    }

    for invalid in (None, ["not-a-player"] * 11):
        home = _lineup_feature(
            derived,
            asset,
            team_id=HOME,
            value=invalid,
            known_at=as_of,
        )
        with pytest.raises(ValueError, match="11 unique platform player IDs"):
            build_snapshot(
                features=(*_t24_features(derived, asset), home, away),
                **arguments,
            )


def test_lineups_snapshot_rejects_delta_starters_different_from_official_xi(
    tmp_path: Path,
) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home_players = tuple(f"player:home-{index}" for index in range(11))
    away_players = tuple(f"player:away-{index}" for index in range(11))
    delta_home = (*home_players[:-1], "player:home-replacement")
    home = _lineup_feature(derived, asset, team_id=HOME, value=list(home_players), known_at=as_of)
    away = _lineup_feature(derived, asset, team_id=AWAY, value=list(away_players), known_at=as_of)
    delta = _v3_lineup_delta_feature(
        derived,
        asset,
        home_starters=delta_home,
        away_starters=away_players,
        known_at=as_of,
    )

    with pytest.raises(ValueError, match="starter_ids do not match official lineup"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/2",
            features=(*_t24_features(derived, asset), delta, home, away),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_lineups_snapshot_rejects_delta_raw_different_from_official_lineup(
    tmp_path: Path,
) -> None:
    raw, derived, official_asset = _stores(tmp_path)
    delta_asset = raw.archive(
        b"unrelated raw evidence",
        source="test-source",
        source_id="unrelated-evidence",
        url="fixture://unrelated-evidence",
        observed_at=official_asset.observed_at,
        target_event_time=None,
        collector_version="test-collector/1",
        media_type="application/octet-stream",
    )
    as_of = KICKOFF - timedelta(hours=1)
    home_players = tuple(f"player:home-{index}" for index in range(11))
    away_players = tuple(f"player:away-{index}" for index in range(11))
    home = _lineup_feature(
        derived, official_asset, team_id=HOME, value=list(home_players), known_at=as_of
    )
    away = _lineup_feature(
        derived, official_asset, team_id=AWAY, value=list(away_players), known_at=as_of
    )
    delta = _v3_lineup_delta_feature(
        derived,
        delta_asset,
        home_starters=home_players,
        away_starters=away_players,
        known_at=as_of,
    )

    with pytest.raises(ValueError, match="lineup_input_refs do not match official lineup source"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/2",
            features=(*_t24_features(derived, official_asset), delta, home, away),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_lineups_snapshot_rejects_ready_delta_without_dimensions(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home = _lineup_feature(
        derived,
        asset,
        team_id=HOME,
        value=[f"player:home-{index}" for index in range(11)],
        known_at=as_of,
    )
    away = _lineup_feature(
        derived,
        asset,
        team_id=AWAY,
        value=[f"player:away-{index}" for index in range(11)],
        known_at=as_of,
    )
    empty_ready = {
        team_id.value: {
            "quality_status": "ready",
            "dimension_deltas": {},
            "missing_fields": [],
            "not_applicable_fields": [],
            "blocking_reasons": [],
            "input_refs": [f"player-profile:{team_id.value}"],
            "transform_version": "lineup-delta/2",
            "role_contract_versions": ["lineup-role-metrics/1"],
            "dimension_sample_sizes": {},
        }
        for team_id in (HOME, AWAY)
    }
    delta = SnapshotFeature(
        "lineup_delta",
        empty_ready,
        as_of,
        _source(derived, asset, empty_ready, "lineup-delta-input/1"),
        "lineup-delta",
    )

    with pytest.raises(ValueError, match="requires dimensions"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/2",
            features=(*_t24_features(derived, asset), delta, home, away),
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_legacy_lineup_payload_is_v1_read_only_but_remains_verifiable(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    as_of = KICKOFF - timedelta(hours=1)
    home = _lineup_feature(
        derived,
        asset,
        team_id=HOME,
        value=[f"player:home-{index}" for index in range(11)],
        known_at=as_of,
    )
    away = _lineup_feature(
        derived,
        asset,
        team_id=AWAY,
        value=[f"player:away-{index}" for index in range(11)],
        known_at=as_of,
    )
    legacy_value = {
        team_id.value: {
            "quality_status": "preview",
            "dimension_deltas": {},
            "missing_fields": ["reference_lineup"],
        }
        for team_id in (HOME, AWAY)
    }
    delta = SnapshotFeature(
        "lineup_delta",
        legacy_value,
        as_of,
        _source(derived, asset, legacy_value, "lineup-delta-input/1"),
        "lineup-delta",
    )
    features = (*_t24_features(derived, asset), delta, home, away)

    missing = snapshot_contract._assess_readiness(
        "prematch-features/1",
        SnapshotType.LINEUPS_CONFIRMED,
        features,
        HOME,
        AWAY,
    )
    assert missing == (
        f"{AWAY.value}:reference_lineup",
        f"{HOME.value}:reference_lineup",
    )
    with pytest.raises(ValueError, match="read-only legacy"):
        build_snapshot(
            match_id=MATCH,
            match_version=1,
            snapshot_type=SnapshotType.LINEUPS_CONFIRMED,
            as_of=as_of,
            scheduled_kickoff_used=KICKOFF,
            feature_spec_version="prematch-features/1",
            features=features,
            home_team_id=HOME,
            away_team_id=AWAY,
            source_validator=derived,
        )


def test_snapshot_is_content_identified_and_idempotently_archived(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )

    assert derived.write_snapshot(snapshot) == derived.write_snapshot(snapshot)
    assert derived.load_snapshot_payload(snapshot)["capture_mode"] == "reconstructed"


def test_snapshot_identity_is_independent_of_feature_input_order(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    first, second = _t24_features(derived, asset)

    ordered = build_snapshot(features=(first, second), **_snapshot_arguments(derived))
    reversed_order = build_snapshot(features=(second, first), **_snapshot_arguments(derived))

    assert ordered.id == reversed_order.id


def test_snapshot_verifier_rejects_tampered_identity_and_quality(tmp_path: Path) -> None:
    _, derived, asset = _stores(tmp_path)
    baseline, context = _t24_features(derived, asset)
    ready = build_snapshot(features=(baseline, context), **_snapshot_arguments(derived))
    preview = build_snapshot(features=(baseline,), **_snapshot_arguments(derived))

    with pytest.raises(ValueError, match="snapshot identity"):
        verify_snapshot(
            replace(ready, id=SnapshotId("snapshot:" + "f" * 64)),
            source_validator=derived,
        )
    with pytest.raises(ValueError, match="snapshot quality"):
        verify_snapshot(
            replace(preview, quality_status="ready", missing_fields=()),
            source_validator=derived,
        )


def test_qualifications_are_independent_but_cannot_skip_incomplete_archive() -> None:
    eleven_home = frozenset(f"home-player-{index}" for index in range(11))
    eleven_away = frozenset(f"away-player-{index}" for index in range(11))
    availability = MatchAvailability(
        match_status=MatchStatus.FINISHED,
        team_ids=(HOME.value, AWAY.value),
        snapshots=(
            SnapshotAvailability(
                SnapshotType.T24H,
                CaptureMode.RECONSTRUCTED,
                "ready",
            ),
        ),
        result_90_present=True,
        team_stat_fields={
            HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            AWAY.value: frozenset({"goals", "shots"}),
        },
        starters={HOME.value: eleven_home, AWAY.value: eleven_away},
        player_observation_ids=eleven_home | eleven_away,
    )

    assessment = assess_lifecycle(
        availability,
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    by_name = {result.qualification: result for result in assessment.qualifications}

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)
    assert not by_name[Qualification.SCORE_MODEL].passed
    assert "unverified_snapshot_evidence" in by_name[Qualification.SCORE_MODEL].reason_codes
    assert not by_name[Qualification.TEAM_BASELINE].passed
    assert by_name[Qualification.PLAYER_PROFILE].passed
    assert "missing_team_stat:team:away:xg" in by_name[Qualification.TEAM_BASELINE].reason_codes


def test_finished_but_incomplete_archive_is_pending() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={},
            starters={},
            player_observation_ids=frozenset(),
        ),
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)


def test_empty_team_stats_and_starters_cannot_archive_as_complete() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset(),
                AWAY.value: frozenset(),
            },
            starters={
                HOME.value: frozenset(),
                AWAY.value: frozenset(),
            },
            player_observation_ids=frozenset(),
        ),
        evaluated_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert assessment.reason_codes == ("post_match_archive_incomplete",)


def test_unverified_snapshot_summary_cannot_advance_lifecycle() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.SCHEDULED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(
                SnapshotAvailability(
                    SnapshotType.LINEUPS_CONFIRMED,
                    CaptureMode.CAPTURED,
                    "ready",
                ),
            ),
            result_90_present=False,
            team_stat_fields={},
            starters={},
            player_observation_ids=frozenset(),
        ),
        evaluated_at=OBSERVED_AT,
    )
    score = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.SCORE_MODEL
    )

    assert assessment.state is LifecycleState.DISCOVERED
    assert "unverified_snapshot_evidence" in score.reason_codes


def test_persisted_verified_snapshot_advances_only_after_it_was_observed(
    tmp_path: Path,
) -> None:
    snapshot_time = KICKOFF - timedelta(hours=24)
    _, derived, asset = _stores(tmp_path, observed_at=snapshot_time)
    snapshot = build_snapshot(
        features=_t24_features(derived, asset),
        **_snapshot_arguments(derived),
    )
    derived.write_snapshot(snapshot)
    availability = MatchAvailability(
        match_status=MatchStatus.SCHEDULED,
        team_ids=(HOME.value, AWAY.value),
        snapshots=(SnapshotAvailability.from_snapshot(snapshot),),
        result_90_present=False,
        team_stat_fields={},
        starters={},
        player_observation_ids=frozenset(),
        match_id=MATCH.value,
        match_version=1,
    )

    before = assess_lifecycle(
        availability,
        evaluated_at=snapshot_time - timedelta(seconds=1),
        snapshot_validator=derived,
    )
    available = assess_lifecycle(
        availability,
        evaluated_at=snapshot_time,
        snapshot_validator=derived,
    )

    assert before.state is LifecycleState.DISCOVERED
    assert available.state is LifecycleState.T24_READY


def test_empty_player_observations_do_not_pass_player_profile_readiness() -> None:
    starters = {
        HOME.value: frozenset(f"player:home-{index}" for index in range(11)),
        AWAY.value: frozenset(f"player:away-{index}" for index in range(11)),
    }
    details = {
        player_id: PlayerObservationAvailability(
            team_id=team_id,
            role="",
            minutes=0,
            metric_fields=frozenset(),
            known_at=OBSERVED_AT,
        )
        for team_id, player_ids in starters.items()
        for player_id in player_ids
    }
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters=starters,
            player_observation_ids=frozenset(details),
            player_observations=details,
        ),
        evaluated_at=OBSERVED_AT,
    )
    player_profile = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.PLAYER_PROFILE
    )

    assert not player_profile.passed
    assert any(
        code.startswith("missing_player_role_or_minutes:") for code in player_profile.reason_codes
    )
    assert any(code.startswith("missing_player_metrics:") for code in player_profile.reason_codes)


@pytest.mark.parametrize(
    ("metric_fields", "known_at", "expected_reason"),
    (
        (frozenset({"key_passes"}), OBSERVED_AT, "missing_player_metric:"),
        (frozenset({"shots"}), None, "missing_player_known_at:"),
    ),
)
def test_player_profile_readiness_requires_role_metrics_and_known_time(
    metric_fields: frozenset[str],
    known_at: datetime | None,
    expected_reason: str,
) -> None:
    starters = {
        HOME.value: frozenset(f"player:home-{index}" for index in range(11)),
        AWAY.value: frozenset(f"player:away-{index}" for index in range(11)),
    }
    details = {
        player_id: PlayerObservationAvailability(
            team_id=team_id,
            role="FW",
            minutes=90,
            metric_fields=frozenset({"shots"}),
            known_at=OBSERVED_AT,
        )
        for team_id, player_ids in starters.items()
        for player_id in player_ids
    }
    first_player = sorted(details)[0]
    details[first_player] = replace(
        details[first_player], metric_fields=metric_fields, known_at=known_at
    )
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters=starters,
            player_observation_ids=frozenset(details),
            player_observations=details,
        ),
        evaluated_at=OBSERVED_AT,
    )
    player_profile = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.PLAYER_PROFILE
    )

    assert not player_profile.passed
    assert any(
        code.startswith(expected_reason) and first_player in code
        for code in player_profile.reason_codes
    )


@pytest.mark.parametrize(
    ("role", "minutes", "expected_reason"),
    (
        ("", 90, "missing_player_role:"),
        ("FW", 0, "missing_player_minutes:"),
        ("unknown", 90, "missing_player_role_contract:"),
    ),
)
def test_player_profile_readiness_rejects_missing_or_unsupported_role_contract(
    role: str, minutes: float, expected_reason: str
) -> None:
    starters = {
        HOME.value: frozenset(f"player:home-{index}" for index in range(11)),
        AWAY.value: frozenset(f"player:away-{index}" for index in range(11)),
    }
    details = {
        player_id: PlayerObservationAvailability(
            team_id=team_id,
            role="FW",
            minutes=90,
            metric_fields=frozenset({"shots"}),
            known_at=OBSERVED_AT,
        )
        for team_id, player_ids in starters.items()
        for player_id in player_ids
    }
    first_player = sorted(details)[0]
    details[first_player] = replace(details[first_player], role=role, minutes=minutes)
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters=starters,
            player_observation_ids=frozenset(details),
            player_observations=details,
        ),
        evaluated_at=OBSERVED_AT,
    )
    player_profile = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.PLAYER_PROFILE
    )

    assert not player_profile.passed
    assert any(
        code.startswith(expected_reason) and first_player in code
        for code in player_profile.reason_codes
    )


def test_unknown_readiness_ruleset_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported readiness ruleset"):
        assess_lifecycle(
            MatchAvailability(
                match_status=MatchStatus.SCHEDULED,
                team_ids=(HOME.value, AWAY.value),
                snapshots=(),
                result_90_present=False,
                team_stat_fields={},
                starters={},
                player_observation_ids=frozenset(),
            ),
            evaluated_at=OBSERVED_AT,
            ruleset_version="readiness/unknown",
        )


def test_future_availability_cutoff_cannot_be_used_for_an_earlier_assessment() -> None:
    assessment = assess_lifecycle(
        MatchAvailability(
            match_status=MatchStatus.FINISHED,
            team_ids=(HOME.value, AWAY.value),
            snapshots=(),
            result_90_present=True,
            team_stat_fields={
                HOME.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
                AWAY.value: frozenset({"goals", "xg", "shots", "shots_on_target"}),
            },
            starters={},
            player_observation_ids=frozenset(),
            as_of=OBSERVED_AT + timedelta(hours=1),
        ),
        evaluated_at=OBSERVED_AT,
    )
    team_baseline = next(
        result
        for result in assessment.qualifications
        if result.qualification is Qualification.TEAM_BASELINE
    )

    assert assessment.state is LifecycleState.FINISHED_STATS_PENDING
    assert "availability_as_of_after_evaluation" in team_baseline.reason_codes
