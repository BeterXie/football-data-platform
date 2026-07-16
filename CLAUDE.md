# Repository Guidance

Read and follow [AGENTS.md](AGENTS.md) before changing code or data contracts. The same critical
constraints apply to every agent:

- preserve `raw/canonical/derived` separation and lineage;
- enforce UTC and `known_at <= as_of`;
- never label historical reconstruction as captured;
- keep task-specific readiness independent;
- derive every market view from the one corrected Dixon-Coles grid;
- keep market odds outside the football model;
- do not add in-play, automated betting, or real-money behavior in phase one;
- do not describe the two-match golden fixture as the completed 380-match season.

Run the test, Ruff, format-check, and compile commands in `AGENTS.md` before accepting work.
