# World Cup Team Win Rate Prediction System — Design Spec

> [!IMPORTANT]
> 本文档是历史初始设计，已被 [2026-07-16 足球数据与预测平台总体设计](2026-07-16-football-data-platform-design.md) 替代。保留本文仅用于理解项目演进，不再作为当前范围、数据契约或模型验收依据。

## Overview

Predict single-match win/draw/loss probabilities and full-tournament outcomes (champion, stage advancement) for the men's FIFA World Cup finals, targeting betting-grade accuracy. Uses player-level club data to solve the sparse-national-team-data problem, with a hybrid statistical + ML model architecture.

## Data Architecture

### Layer 1: Player Data (Core)

Per-player features sourced primarily from FBref, Transfermarkt, and FIFA Index:

- Demographics: age, position, preferred foot
- Club performance (last 1-2 seasons): minutes played, goals, assists, pass completion %, tackles, interceptions, progressive carries, xG, xA
- Market value: current Transfermarkt valuation + historical trend
- National team experience: caps, tournament appearances, NT goals
- Injury status: recent injuries, current availability

### Layer 2: Team Data

- Latest squad roster (23-26 players)
- Last 2 years of international A-match results with opponent strength
- FIFA ranking + trend
- Manager's preferred formation and playing style (possession %, counter-attack tendency)
- Match context: venue, travel distance, rest days

### Layer 3: Match Context

- Head-to-head history between the two teams
- Venue type (neutral/home)
- Referee profile (card rate, penalty rate)
- Match importance (group stage, knockout, dead rubber)

## Model Architecture

### Phase 1: Player Ability Score

1. Extract per-player performance indicators from club data, weighted by league strength coefficient (e.g., Premier League goals carry more weight than Ligue 1 goals).
2. Transfermarkt market value used as a cross-validation signal.
3. Output: a composite ability score per player.

### Phase 2: Team Strength Aggregation

- Group players by position (FW/MF/DF/GK) and formation.
- Compute three team scores: attacking power, midfield control, defensive power.
- Starters weighted by expected minutes; substitutes at ~0.3x.

### Phase 3: Single-Match Prediction

1. **Base layer**: Bivariate Poisson distribution modeling goal counts for both teams. Team A goals ~ Poisson(lambda_A), Team B goals ~ Poisson(lambda_B). Lambda parameters derived from team attacking/defensive scores + opponent strength.
2. **Correction layer**: Lightweight XGBoost model that takes the Poisson output + match context features (H2H, venue, referee, stakes) and outputs fine-tuned win/draw/loss probabilities.
3. **Output**: P(home win), P(draw), P(away win).

### Phase 4: Tournament Simulation

- Monte Carlo simulation following the real World Cup format (group stage → knockouts).
- Each match sampled from Phase 3 probabilities.
- 10,000 simulation runs → champion probability, stage advancement probabilities per team, group qualification probability.

## Evaluation

- **Test set**: 2022 Qatar World Cup (time-series split — all prior data used for training).
- **Metrics**: Log Loss and Brier Score (probability-calibrated metrics, more meaningful than raw accuracy for betting use).
- **Baseline**: Compare against market-implied probabilities (betting odds) and a pure ELO-only model.

## Tech Stack

- Python 3.x
- pandas, numpy: data processing
- scipy: statistical modeling (Poisson distributions)
- xgboost: correction-layer ML model
- matplotlib/plotly: visualization
- Jupyter Notebook: primary development environment

## Project Structure

```
world-cup-predictor/
├── data/               # Raw and processed data
│   ├── raw/
│   └── processed/
├── notebooks/          # Jupyter notebooks for each phase
│   ├── 01_data_collection.ipynb
│   ├── 02_player_ability.ipynb
│   ├── 03_team_strength.ipynb
│   ├── 04_match_prediction.ipynb
│   └── 05_tournament_sim.ipynb
├── src/                # Reusable Python modules
│   ├── data/
│   ├── features/
│   ├── models/
│   └── simulation/
├── docs/
└── README.md
```

## Scope & Non-Goals

**In scope:**
- Men's FIFA World Cup finals only
- Single-match and full-tournament prediction
- Jupyter-based workflow initially

**Out of scope (for now):**
- Women's World Cup
- Qualifying stages
- Web UI / API service
- Real-time live betting updates
- Scraping infrastructure (use existing APIs/public datasets first)
