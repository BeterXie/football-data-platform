# Football Data Platform

A source-agnostic football data platform for reproducible collection, canonical facts, and
derived analytics. The first vertical slice targets the 2025-26 Premier League, while the
competition registry and domain contracts remain reusable across competitions and seasons.

- [Authoritative design](docs/superpowers/specs/2026-07-16-football-data-platform-design.md)
- [ADR and documentation index](docs/README.md)
- [Domain glossary](docs/domain-glossary.md)
- [Delivery plan and acceptance gates](PROJECT_PLAN.md)

## Development

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

The runtime dependency set is intentionally small; `tzdata` supplies reproducible IANA timezone
rules on Windows. Competition and source season identifiers live in
`config/competitions.toml`; business code should resolve them through
`load_competition_registry()` instead of embedding provider-specific identifiers.

## Data layers

- `raw`: immutable source evidence and checksummed payloads.
- `canonical`: normalized facts keyed by platform-owned entity IDs.
- `derived`: reproducible features, datasets, models, predictions, and evaluations.

`DataLayout.ensure()` creates these directories below a caller-selected data root. `RawArchive`
deduplicates payload bytes by SHA-256 while retaining a separate immutable manifest for every
distinct observation.

## CLI

Initialize a local catalog:

```powershell
fdp init --data-root data
```

Replay the bundled offline golden slice with a fixed observation time:

```powershell
fdp run-golden --data-root data/golden --observed-at 2026-07-16T08:00:00Z
```

The command emits a JSON run pointer and writes structured artifacts plus a Markdown report below
the selected data root. The two-match golden fixture proves end-to-end replay only; the independent
season gate still requires 20 teams, 380 matches, and a recorded collection attempt per fixture.

Probe the registered FBref URL without treating access controls as empty data:

```powershell
fdp diagnose-fbref --data-root data
```

An HTTP denial or Cloudflare challenge returns exit code `3` and persists a machine-readable
`blocked_by_access_control` diagnostic. Bypassing a CAPTCHA is not a supported collection path.

When FBref schedule access is blocked, the registered Football-Data CSV can seed real fixture and
90-minute result identities without feeding its odds columns into the football model:

```powershell
fdp backfill-results --data-root data/premier-league-2025-26
```

This command can pass the 20-team/380-match schedule gate, but it deliberately leaves the FBref
per-match statistics-attempt gate open. Football-Data is a schedule/result fallback, not a
replacement for FBref team and player process statistics.
