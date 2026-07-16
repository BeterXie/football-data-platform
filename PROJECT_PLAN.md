# Football Data Platform Delivery Plan

## Scope

The first delivery is the 2025-26 Premier League vertical slice. It proves
the contracts and orchestration needed to expand to the other top-five
leagues and seasons from 2021-22 onward. It does not include in-play models,
automated wagering, real-money execution, a public API, or a web application.

The authoritative product design is
`docs/superpowers/specs/2026-07-16-football-data-platform-design.md` in this repository. The
superseded 2026-05-28 design is retained only as historical context.

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

This table reports the evidence boundary after the 2026-07-16 review. The remediation work,
dependencies, and full phase-one roadmap are tracked in
[docs/repair-and-optimization-plan.md](docs/repair-and-optimization-plan.md). "Not accepted" does
not mean that no implementation exists; it means the current evidence does not prove the stated
gate.

| Gate | Status | Evidence |
| --- | --- | --- |
| Repository and raw evidence foundation | Verified at current unit scope | Package install, CLI, and content-addressed raw replay are covered; full run/derived lineage remains open |
| Canonical catalog and cross-source identities | Contract gates implemented; production evidence open | Stable mappings, participant constraints, migrations, and cross-source replay are covered; a full-season reconciliation is not recorded |
| Generic FBref parsing | Contract gates implemented; golden-sample only | Offline fixtures parse and forged identity/score reports are rejected; live schema coverage remains unproven |
| FBref live access | Blocked and observable | HTTP 403 becomes persisted `blocked_by_access_control`, exit code 3 |
| Premier League 2025-26 fixture/result catalog | Gate implemented; coverage evidence open | The fallback sample reuses platform identities and the validator checks 20/380 structure and persisted attempts; the bundled feed is not a full-season capture |
| Full FBref per-match statistics collection | Open | The full collection gate remains false until source IDs/reports are collected |
| Temporal snapshots and three readiness qualifications | Contract gates implemented; captured production open | Snapshot provenance, UTC cutoffs, lifecycle completeness, and independent qualifications reject the previous bypasses; prospective captured runs are not recorded |
| Prematch news, injuries, suspensions, and official lineups | Contract adapters/tests implemented; live collection open | Evidence independence, publication cutoffs, official-source checks, and atomic XI writes are covered; production source runs are not proven |
| Player profiles and lineup deltas | Versioned contract implemented; full coverage open | Windowed role metrics, availability/load fields, missing/N/A states, and preview deltas are validated; full player history is not recorded |
| Team baseline and context composition | Versioned contract implemented; calibration evidence open | Away-neutral coordinates, contribution keys, lineage, and final lambda composition are tested; prospective calibration is not established |
| Dixon-Coles score distribution | Verified at mathematical unit scope | Formal four-cell tau and normalized-grid tests pass; this does not accept upstream feature construction or season coverage |
| Prediction, real-market evaluation, and governance | Contract gates implemented; prospective policy/sample open | Future/incomplete/synthetic markets, unknown results, and invalid promotion metrics are rejected; no reviewed prospective promotion cohort is proven |
| Run manifests and static reports | Golden/derived manifests implemented; all-command lineage open | Derived artifacts and successful/failed golden runs are content-addressed; broader CLI/report registry coverage remains open |
| Versioned training datasets and model-run artifacts | Contract/storage/tests implemented; production evidence open | Per-sample qualification, leakage, captured raw evidence, model cohorts, and tamper checks are covered; no production model run is recorded |
| Paper betting ledger | Contract/storage/tests implemented; prospective operation open | Real/open market gates, exposure recomputation, append-only revisions, and deterministic settlement are covered; no real prospective ledger cohort is recorded |
| End-to-end vertical slice | Replay only; not accepted | The fixed-time two-match golden run is deterministic but is not 20 teams/380 matches with one persisted attempt per fixture |
| Five leagues from 2021-22 and context competitions | Open | Required by the authoritative design; no completion evidence is currently recorded |

No production vertical slice or phase-one milestone is accepted at this baseline. Individual unit
contracts may remain verified while the broader gate stays open.
