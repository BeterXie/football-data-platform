# Football Data Platform Engineering Rules

This repository implements the football data and prediction platform described in
`docs/superpowers/specs/2026-07-16-football-data-platform-design.md`.

## Current Delivery Boundary

- Prove the Premier League 2025-26 vertical slice first.
- The bundled two-match golden sample is a replay test, not full-season coverage.
- Full-season acceptance requires 20 teams, 380 matches, and a recorded collection attempt for
  every fixture. Missing or blocked source data must remain visible.
- Phase one excludes in-play prediction, automated betting, real-money execution, public APIs,
  and paid sports-data services.

## Architecture Invariants

- Keep `raw`, `canonical`, and `derived` data separate. Models never parse source HTML.
- Raw evidence is immutable and content-addressed. Canonical and derived records cite raw or
  versioned input references.
- Platform IDs are stable; display names and provider IDs are never downstream join keys.
- Every knowledge boundary uses timezone-aware UTC. Snapshot facts require `known_at <= as_of`.
- Historical evidence is `reconstructed`; it must never be labelled `captured`.
- `score-model-ready`, `team-baseline-ready`, and `player-profile-ready` are independent validator
  results. Never add a manually writable global `ready` flag or fill missing values with zero.
- Team baseline, lineup delta, and match context are distinct contributions. A contribution key
  may be applied only once.

## Model And Market Rules

- The football model predicts one normalized 90-minute joint score distribution, excluding extra
  time and penalties.
- All result, handicap, total-goals, and exact-score probabilities aggregate the same
  `DixonColesGrid` with identical lambdas, rho, and truncation.
- Formal low-score terms are:
  `tau(0,1)=1+lambda_home*rho` and `tau(1,0)=1+lambda_away*rho`.
- Market odds are never football-model inputs. Evaluation accepts only real timestamped quotes;
  synthetic odds cannot establish model quality or ROI.
- The Football-Data adapter is results-only. Its CSV odds columns are intentionally ignored and
  must not leak into team baselines, player profiles, snapshots, or expected-goals features.
- Market absence does not delete a football prediction. It marks the market benchmark unavailable.
- Challenger promotion requires an explicit reviewed policy and prospective captured samples.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

Use `fdp run-golden --observed-at 2026-07-16T08:00:00Z` for deterministic end-to-end replay.
Production commands must be idempotent, resumable, emit structured diagnostics, and return a
nonzero exit code when an acceptance gate fails.
