# Football Data Platform Delivery Plan

## Scope

The first delivery is the 2025-26 Premier League vertical slice. It proves
the contracts and orchestration needed to expand to the other top-five
leagues and seasons from 2021-22 onward. It does not include in-play models,
automated wagering, real-money execution, a public API, or a web application.

The authoritative product design remains the 2026-07-16 design in the legacy
`world-cup-predictor` repository until its documentation migration is handled
as a separate, mechanical change.

## Work Breakdown

| ID | Module | Deliverable | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| FDP-001 | Repository foundation | Installable Python package, locked runtime contract, CLI entry point, test layout | None | Clean install and CLI help succeed |
| FDP-002 | Competition registry | Config-driven competition/season/source definitions, including Premier League 2025-26 | FDP-001 | Registry resolves by stable platform ID; no World Cup branches |
| FDP-003 | Raw evidence store | Immutable content-addressed payloads and manifests with source, URL, observed time, checksum, collector version, and diagnostics | FDP-001 | Repeating the same write is idempotent; conflicting mutation is impossible |
| FDP-004 | Canonical store | Versioned SQLite schema for identities, source mappings, fixtures, lineups, stats, evidence, events, and market snapshots | FDP-001 | Foreign keys and uniqueness constraints reject ambiguous facts |
| FDP-005 | FBref adapter | Generic schedule and match-report discovery/parser with offline replay and access-control diagnostics | FDP-002, FDP-003 | Premier League fixture parses; blocked online access is machine-readable |
| FDP-005A | Results fallback | Football-Data schedule/result-only adapter with strict odds-column exclusion | FDP-002, FDP-003 | Real 2025-26 feed yields 20 teams/380 matches; football-model contracts contain no odds |
| FDP-006 | Normalization | Raw FBref records mapped to platform competition, season, team, player, and match IDs | FDP-004, FDP-005 | Replay produces the same IDs and canonical row counts |
| FDP-007 | Prematch snapshots | Immutable `t24h` and `lineups-confirmed` snapshots with `captured`/`reconstructed` mode and lineage | FDP-004, FDP-006 | Every included fact has `known_at <= as_of`; late facts are rejected |
| FDP-008 | Lifecycle/readiness | Validator-computed lifecycle and independent score/team/player training qualifications | FDP-004, FDP-007 | Exclusion reasons are explicit; missing values are never silently zero-filled |
| FDP-009 | Derived football features | Versioned team baseline, lineup delta/context inputs, and role-based player profiles | FDP-006, FDP-008 | Derived records cite canonical input/version and quality status |
| FDP-010 | Score model | One normalized Dixon-Coles score grid and all result/market probability aggregations | FDP-009 | Formal four-cell tau tests and probability conservation pass |
| FDP-011 | Prediction/evaluation contract | Unified prediction, settlement, real-market benchmark, and model-run schemas | FDP-008, FDP-010 | No legacy log adapter is needed inside evaluators; captured/reconstructed cohorts remain separate |
| FDP-012 | Governance | Champion/challenger registry and shadow evaluation without an invented promotion threshold | FDP-011 | Promotion refuses to run until an explicit reviewed policy is supplied |
| FDP-013 | Orchestration/reporting | Idempotent local pipeline, run manifest, diagnostics, static data/model report | FDP-005 through FDP-012 | One command replays the offline slice twice with stable outputs |
| FDP-014 | Season gate | Coverage and integrity validator for the 380-match Premier League season | FDP-006, FDP-013 | Missing/duplicate fixtures fail with actionable diagnostics |

## Acceptance Gates

### Gate A: Contract integrity

- Stable platform IDs are never inferred downstream from display names.
- `raw`, `canonical`, and `derived` data cannot be confused by path or API.
- Every mutable football fact is versioned; every derived artifact has lineage.
- UTC aware timestamps are required at all knowledge and observation boundaries.

### Gate B: Temporal integrity

- Snapshot builders reject facts learned after `as_of`.
- Historical reconstructions cannot be labelled as captured.
- Rescheduled fixtures retain match identity and create a new match version.
- Training qualification is task-specific and validator-computed.

### Gate C: Model integrity

- Dixon-Coles uses `tau(0,1) = 1 + lambda_home * rho` and
  `tau(1,0) = 1 + lambda_away * rho`.
- HAD, handicap, totals, and exact-score probabilities derive from one
  normalized grid using identical lambdas, rho, and truncation.
- The football model has no market-odds input.
- Evaluation uses real outcomes and, when present, real timestamped market
  snapshots; synthetic odds cannot pass the market gate.

### Gate D: Operational integrity

- Network/access-control failures are recorded, not interpreted as empty data.
- Commands are idempotent and resumable and return meaningful exit codes.
- Structured output is authoritative; Markdown/HTML reports are rebuildable.
- The offline vertical slice runs in CI without network access.

## Delivery Sequence

1. Freeze schemas and storage invariants (FDP-001 through FDP-004).
2. Prove acquisition and canonical replay (FDP-005 and FDP-006).
3. Prove temporal lifecycle contracts (FDP-007 and FDP-008).
4. Add derived features and the baseline model (FDP-009 and FDP-010).
5. Close prediction, evaluation, and governance loops (FDP-011 and FDP-012).
6. Integrate the CLI, reports, and season-wide coverage gate (FDP-013 and FDP-014).

Expansion to more competitions is accepted only by adding registry entries and
source mappings. League-specific copies of the pipeline are a failed review.

## Current Acceptance Status

| Gate | Status | Evidence |
| --- | --- | --- |
| Repository, registry, and three-layer storage | Accepted | Package install, CLI, immutable raw archive, SQLite canonical catalog |
| Generic FBref parsing | Accepted offline | Schedule and match-report golden fixtures; no tournament-specific branches |
| FBref live access | Blocked and observable | HTTP 403 becomes persisted `blocked_by_access_control`, exit code 3 |
| Premier League 2025-26 fixture/result catalog | Accepted | Real fallback feed produced 20 teams, 380 matches, and 380 result facts |
| Full FBref per-match statistics collection | Open | The full collection gate remains false until source IDs/reports are collected |
| Temporal snapshots and readiness | Accepted | Reconstructed T-24h and lineup-preview artifacts; independent qualification reasons |
| Team/player/lineup/context features | Accepted for contract slice | Versioned baseline/profile/delta builders preserve missingness and reject future facts |
| Dixon-Coles score distribution | Accepted | Formal four-cell regression tests and one normalized grid for every market view |
| Prediction/evaluation schema | Accepted | One prediction document per artifact; unknown/tampered schemas fail; market is separate |
| End-to-end replay and report | Accepted | Fixed-time golden run replays with stable run, snapshot, and prediction IDs |

The implementation milestone is accepted as a vertical contract slice. It does not claim that the
FBref 380-match statistics archive, prospective captured snapshots, or five-league 2021-22 onward
backfill is complete.
