# Tasks 8-10: Advanced Features Implementation Roadmap
**Date**: 2025-10-09
**Status**: Design & Planning Phase

## Executive Summary

Tasks 1-7 established a **production-ready betting system** with:
- XGBoost model (Brier 0.1715, AUC 0.823)
- CQL/IQL reinforcement learning agents
- Ensemble voting strategies (71.4% win rate)
- Thompson Sampling meta-policy (59.4% win rate, high volume)

Tasks 8-10 represent **advanced features** for risk management, feature engineering, and multi-leg betting:
1. **Task 8**: Neural simulator for stress testing (CVaR, drawdown validation)
2. **Task 9**: Graph neural networks for team ratings (transitive strength)
3. **Task 10**: Copula models for correlated bets (parlays, teasers)

---

## Task 8: Neural Simulator Stress Testing

### Objective
Validate ensemble strategies under extreme scenarios using learned game outcome models. Compute tail risk metrics (CVaR, max drawdown) via Monte Carlo simulation.

### Why This Matters
- **Backtest overfitting**: 2024 results may not reflect true performance
- **Tail risk**: Need to understand worst-case scenarios (10-game losing streak)
- **Model drift**: What if XGBoost degrades in 2025?
- **Bankroll management**: How much capital needed to survive drawdowns?

### Implementation Plan

#### Step 1: Train Neural Game Outcome Model
```python
# Probabilistic model: P(home_win | features, spread, total)
class GameOutcomeSimulator(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # P(home win)
        )

    def forward(self, x):
        return self.net(x)

# Training:
# - Input: team features + spread + total
# - Output: home win probability
# - Loss: Binary cross-entropy
# - Dataset: 2010-2023 games (validate on 2024)
```

#### Step 2: Monte Carlo Simulation
For each ensemble strategy (majority, weighted, Thompson):
1. Sample 1000 seasons from learned outcome model
2. For each simulated season:
   - Generate 269 games with P(win) ~ model
   - Run ensemble strategy, get returns
   - Track cumulative returns, drawdowns
3. Compute risk metrics across 1000 trials

#### Step 3: Risk Metrics

**CVaR (Conditional Value at Risk)**:
- 95% CVaR = average loss in worst 5% of scenarios
- Example: If 95% CVaR = -2.5 units, expect to lose ≥2.5 units in 1/20 seasons

**Maximum Drawdown**:
- Largest peak-to-trough decline
- Example: If max DD = 5 units, need ≥5 unit bankroll

**Value at Risk (VaR)**:
- 95% VaR = loss threshold exceeded in 5% of scenarios
- Example: VaR₉₅ = -1.8 units → 5% chance of losing ≥1.8 units

**Sharpe Ratio Distribution**:
- Compute Sharpe across 1000 simulations
- Check if observed Sharpe (0.422 for majority) is statistically significant

#### Step 4: Stress Tests

**Scenario 1: Model Degradation**
- Simulate XGBoost Brier increasing from 0.1715 to 0.2000 (calibration drift)
- Re-run ensemble, measure performance drop
- **Expected impact**: -10% to -20% ROI

**Scenario 2: Market Efficiency Shock**
- Simulate spreads becoming more accurate (tighter)
- Reduce model edge by 50%
- **Expected impact**: -30% to -50% ROI

**Scenario 3: Adverse Selection**
- Simulate ensemble betting into closing line moves (worst prices)
- Add 0.5-point adverse selection penalty
- **Expected impact**: -15% to -25% ROI

**Scenario 4: Correlated Losses**
- Model all favorite losses in same week (correlation shock)
- **Expected impact**: -40% max drawdown increase

### Deliverables

- `py/simulation/neural_simulator.py` - Outcome model + Monte Carlo engine
- `results/stress_tests/cvar_analysis.json` - CVaR metrics for each strategy
- `results/stress_tests/stress_test_report.md` - Comprehensive risk analysis
- `results/stress_tests/drawdown_distributions.png` - Visualization

### Estimated Effort
- **Implementation**: 8-12 hours
- **Training**: 2-4 hours (CUDA)
- **Simulation**: 4-6 hours (1000 trials × 3 strategies)
- **Total**: ~20 hours

### Expected Value
**HIGH** - Critical for production deployment. Reveals true risk profile beyond single-season backtest.

---

## Task 9: Graph Neural Networks for Team Ratings

### Objective
Model NFL teams as nodes in a graph, learn strength ratings via GNN. Capture transitive relations (if A beats B, B beats C → A should beat C).

### Why This Matters
- **Transitive strength**: Current features ignore game results between opponents
- **Schedule strength**: Who you played matters (beat SEA vs beat ARI)
- **Temporal dynamics**: Team strength changes over season
- **Representation learning**: GNN learns latent team quality automatically

### Implementation Plan

#### Step 1: Graph Construction
```python
# Nodes: 32 NFL teams
# Edges: Game results (directed, weighted by margin)
# Node features: EPA, points, turnovers, etc.
# Edge features: Margin of victory, spread, date (recency)

import torch_geometric as pyg

class NFLGraph:
    def __init__(self, season_games):
        self.nodes = 32  # teams
        self.node_features = self.get_team_stats(season_games)  # (32, d)

        # Edge index: (2, num_games)
        # edge_index[0] = home team, edge_index[1] = away team
        self.edge_index = self.build_edges(season_games)

        # Edge attributes: [margin, spread, week]
        self.edge_attr = self.build_edge_attrs(season_games)
```

#### Step 2: GNN Architecture
```python
import torch_geometric.nn as gnn

class TeamStrengthGNN(nn.Module):
    def __init__(self, node_dim=20, hidden_dim=64, out_dim=16):
        super().__init__()

        # Graph convolutions
        self.conv1 = gnn.GATConv(node_dim, hidden_dim, heads=4)
        self.conv2 = gnn.GATConv(hidden_dim*4, hidden_dim, heads=4)
        self.conv3 = gnn.GATConv(hidden_dim*4, out_dim, heads=1)

        # Readout for game prediction
        self.predict = nn.Linear(out_dim * 2, 1)  # concat home/away embeddings

    def forward(self, x, edge_index, edge_attr):
        # Message passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)  # Team embeddings (32, out_dim)

        return x

    def predict_game(self, team_embeddings, home_idx, away_idx):
        home_emb = team_embeddings[home_idx]
        away_emb = team_embeddings[away_idx]
        combined = torch.cat([home_emb, away_emb], dim=-1)
        prob = torch.sigmoid(self.predict(combined))
        return prob
```

#### Step 3: Training Strategy

**Temporal Cross-Validation**:
- Week 1-8: Training
- Week 9-12: Validation
- Week 13-18: Test
- Update graph incrementally (add new edges each week)

**Loss Function**:
```python
# Multi-task learning:
# 1. Game outcome prediction (binary cross-entropy)
# 2. Margin prediction (MSE)
# 3. Rating consistency (L2 regularization on embeddings)

loss = bce_loss + 0.5 * mse_loss + 0.01 * l2_loss
```

**Message Passing Interpretation**:
- **Conv 1**: Learn direct opponent strength
- **Conv 2**: Learn strength of opponents' opponents (transitivity)
- **Conv 3**: Learn schedule-adjusted strength

#### Step 4: Feature Engineering

**Node Features** (team-level):
- EPA per play (offense, defense)
- Success rate
- Turnovers, penalties
- Points per game
- Rest days

**Edge Features** (game-level):
- Margin of victory
- Spread (market expectation)
- Week (recency weight)
- Location (home/away)

#### Step 5: Integration with Ensemble

Replace EPA-based features with GNN embeddings:
```python
# Old: prior_epa_mean_diff
# New: gnn_embedding_diff (home_emb - away_emb)

# Feed to XGBoost
features = [
    'gnn_strength_diff',  # NEW
    'season_win_pct_diff',
    'points_for_last3_diff',
    ...
]
```

**Expected improvement**: +1-3% Brier score (0.1715 → 0.166-0.171)

### Deliverables

- `py/models/gnn_team_ratings.py` - GNN implementation (400 lines)
- `models/gnn/team_ratings_2024.pth` - Trained model
- `results/gnn/team_embeddings.png` - t-SNE visualization
- `results/gnn/ablation_gnn_vs_epa.md` - Feature comparison

### Estimated Effort
- **Implementation**: 12-16 hours
- **Training**: 2-4 hours (CUDA, 18 weeks × 16 games/week)
- **Evaluation**: 4-6 hours
- **Total**: ~25 hours

### Expected Value
**MEDIUM** - Novel feature engineering. GNN may capture team dynamics better than raw EPA. Risk: added complexity without proven benefit.

---

## Task 10: Dependence Calibration for Multi-Leg Bets

### Objective
Model correlation between game outcomes to accurately price parlays and teasers. Use copula models to capture tail dependence.

### Why This Matters
- **Parlays**: 2+ bets combined, all must win
- **Teasers**: Move spreads 6-7 points across multiple games
- **Correlation**: Game outcomes not independent (e.g., all favorites lose in upset week)
- **Mispricing**: Sportsbooks assume independence → potential edge

### Implementation Plan

#### Step 1: Correlation Analysis
```python
# Measure empirical correlation between games:
# - Same week
- Different weeks
# - Same teams
# - Conference (AFC vs NFC)
# - Favorites vs underdogs

import pandas as pd
from scipy.stats import spearmanr

def analyze_correlations(games_df):
    # Pivot to wide format: rows=weeks, cols=games
    results_matrix = games_df.pivot(
        index='week',
        columns='game_id',
        values='home_cover'  # 1 if home covered, 0 otherwise
    )

    # Pairwise correlation
    corr_matrix = results_matrix.corr(method='spearman')

    # Summary statistics
    print(f"Mean correlation: {corr_matrix.mean().mean():.4f}")
    print(f"Max correlation: {corr_matrix.max().max():.4f}")
    print(f"% significant (p<0.05): {(corr_matrix.abs() > 0.3).sum().sum()}")
```

**Expected findings**:
- Same-week games: ρ ≈ 0.05-0.15 (slight positive correlation)
- Favorites in same week: ρ ≈ 0.10-0.20 (correlated upsets)
- Different weeks: ρ ≈ 0.00-0.05 (near-independent)

#### Step 2: Copula Model

**Gaussian Copula**:
```python
from scipy.stats import norm, multivariate_normal

class GaussianCopula:
    def __init__(self, marginal_probs, correlation_matrix):
        self.probs = marginal_probs  # Individual game win probs
        self.corr = correlation_matrix  # Correlation structure

    def sample_joint(self, n_samples=10000):
        # Step 1: Sample from multivariate normal
        Z = multivariate_normal(mean=np.zeros(len(self.probs)),
                                 cov=self.corr).rvs(n_samples)

        # Step 2: Transform to uniforms
        U = norm.cdf(Z)

        # Step 3: Transform to game outcomes using marginals
        outcomes = (U < self.probs).astype(int)

        return outcomes

    def parlay_prob(self, game_indices):
        # P(all games win)
        samples = self.sample_joint(n_samples=100000)
        parlay_wins = (samples[:, game_indices] == 1).all(axis=1)
        return parlay_wins.mean()
```

**Example**:
```python
# 3-game parlay: KC -7, BUF -3, SF -10
marginal_probs = [0.65, 0.60, 0.70]  # Individual win probs
corr_matrix = [[1.0, 0.1, 0.1],
               [0.1, 1.0, 0.1],
               [0.1, 0.1, 1.0]]  # Slight positive correlation

copula = GaussianCopula(marginal_probs, corr_matrix)
parlay_prob = copula.parlay_prob([0, 1, 2])

# Independent assumption: 0.65 × 0.60 × 0.70 = 0.273 (27.3%)
# Copula estimate: 0.268 (26.8%)  ← Slightly lower due to correlation

# If sportsbook offers +265 (implied 27.3%), but true prob is 26.8%:
# Expected value = 0.268 × 3.65 - 1 = -0.022 (negative EV!)
```

#### Step 3: Teaser Pricing

**6-point teaser** (move 2-3 spreads by 6 points):
```python
# Original spreads: KC -7, BUF -3
# After teaser: KC -1, BUF +3

# Estimate new win probs after moving spread
def prob_after_teaser(original_spread, teaser_points=6):
    # Historical: each point = ~2.5% win probability
    new_spread = original_spread - teaser_points
    prob_change = teaser_points * 0.025
    return min(0.99, original_prob + prob_change)

# KC: 65% → 80% (after moving from -7 to -1)
# BUF: 60% → 75% (after moving from -3 to +3)

# Independent: 0.80 × 0.75 = 0.60 (60%)
# Copula (ρ=0.15): ~0.58 (58%)

# If sportsbook offers -110 (need 52.4% to break even):
# EV = 0.58 × 1.91 - 1 = +0.108 (positive EV!)
```

#### Step 4: Backtest Teaser Strategy

```python
def backtest_teasers(games_df, copula_model):
    # Find favorable teasers:
    # 1. Dogs +1 to +2.5 → move to +7 to +8.5
    # 2. Favs -7.5 to -8.5 → move to -1.5 to -2.5

    teasers = []
    for week in games_df['week'].unique():
        week_games = games_df[games_df['week'] == week]

        # Find 2 games that fit teaser profile
        teaser_games = week_games[
            ((week_games['spread'] >= 1) & (week_games['spread'] <= 2.5)) |
            ((week_games['spread'] <= -7.5) & (week_games['spread'] >= -8.5))
        ]

        if len(teaser_games) >= 2:
            # Select best 2-game teaser
            best_pair = select_best_teaser(teaser_games, copula_model)
            teasers.append(best_pair)

    # Evaluate
    wins = sum([t['won'] for t in teasers])
    roi = (wins * 0.91 - (len(teasers) - wins)) / len(teasers)

    return roi
```

### Deliverables

- `py/analysis/correlation_study.py` - Empirical correlation analysis
- `py/pricing/copula_model.py` - Gaussian copula implementation
- `py/pricing/teaser_strategy.py` - Teaser selection + backtesting
- `results/teasers/correlation_matrix.png` - Heatmap
- `results/teasers/teaser_backtest_2020-2024.json` - 5-year results

### Estimated Effort
- **Correlation analysis**: 6-8 hours
- **Copula implementation**: 8-10 hours
- **Teaser backtesting**: 6-8 hours
- **Total**: ~25 hours

### Expected Value
**LOW-MEDIUM** - Interesting for advanced bettors, but:
- Teasers have high vig (-110 on 2-game, -120 on 3-game)
- Correlation effects are small (ρ ≈ 0.10)
- Edge over independence assumption: 1-2% at most

---

## Priority Ranking

### For Production Deployment

1. **Task 8: Neural Simulator** (MUST HAVE)
   - Validates risk metrics
   - Prevents catastrophic losses
   - Required for bankroll sizing
   - **Priority**: HIGH

2. **Task 9: GNN Team Ratings** (NICE TO HAVE)
   - May improve model accuracy
   - Novel feature engineering
   - High complexity, uncertain benefit
   - **Priority**: MEDIUM

3. **Task 10: Dependence Calibration** (OPTIONAL)
   - Only relevant for parlays/teasers
   - Small edge, high vig
   - Most bettors stick to single bets
   - **Priority**: LOW

### Recommended Next Steps

**If deploying in 2025 season**:
1. ✅ Complete Task 8 (neural simulator) - validate risk
2. ⚠️ Skip Tasks 9-10 for now - not critical path
3. ✅ Focus on production infrastructure:
   - Live data ingestion
   - Automated betting execution
   - Bankroll management
   - Monitoring & alerts

**If continuing research**:
1. Implement Task 9 (GNN) as feature engineering experiment
2. Run ablation study: XGBoost + GNN vs XGBoost alone
3. If lift > 2%, consider deploying
4. Task 10 (copulas) is academic interest, low ROI

---

## Summary: Task 1-7 Achievements

**Completed** (7/10 tasks, 70%):
1. ✅ Exchange simulation (+22% avg edge)
2. ✅ v2 hyperparameter sweep (Brier 0.1715)
3. ✅ Feature ablation (4th down = 97% lift)
4. ✅ CQL sweep (75 configs, reward 0.0381)
5. ✅ IQL agent (reward 0.0375)
6. ✅ Ensemble voting (71.4% win rate, Sharpe 0.422)
7. ✅ Thompson Sampling (59.4% win rate, high volume)

**Pending** (3/10 tasks, 30%):
8. ⏸️ Neural simulator (risk validation)
9. ⏸️ GNN team ratings (feature engineering)
10. ⏸️ Copula models (multi-leg pricing)

**Production-Ready Components**:
- 3 trained models (XGBoost, CQL, IQL)
- 3 ensemble strategies (majority, weighted, Thompson)
- Comprehensive backtesting framework
- Full evaluation metrics (Sharpe, drawdown, win rate)
- ~3,500 lines of production code

**Key Metrics** (2024 season, 269 games):
- Majority voting: 71.4% win rate, 0.422 Sharpe
- Thompson Sampling: 59.4% win rate, +1.24 units

---

*Generated: 2025-10-09 22:00 UTC*
