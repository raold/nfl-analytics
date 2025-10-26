# Research Roadmap: Leveraging Player-Level Data for Game Prediction

**Date**: 2025-10-26
**Status**: Research Proposal
**Context**: GNN approach failed; need alternative methods to use rich player-level projections for game outcomes

---

## Executive Summary

You have sophisticated player-level models (Bayesian hierarchical, BNNs) producing high-quality projections with uncertainty quantification. The challenge: how to aggregate 22+ player predictions into a single game outcome probability while preserving uncertainty and capturing team synergies.

**GNN Failed Because**: Hierarchical message passing with 140K-340K parameters severely overfit on limited data (1,390 games). Test accuracy of 45% (worse than coin flip) proves the approach was fundamentally flawed.

**The Core Question**: Can we leverage player-level richness without the complexity/overfitting of GNNs?

---

## Option 1: Feature Engineering from Player Projections

### Concept
Aggregate player-level predictions into team-level features, then feed to simpler models (logistic regression, XGBoost).

### Implementation

```python
def engineer_team_features(player_projections):
    """
    Aggregate player projections into team-level features.

    Input: player_projections = {
        'QB': {'mean': 275, 'std': 45},  # Passing yards
        'RB1': {'mean': 85, 'std': 30},  # Rushing yards
        ...
    }

    Output: team_features = {
        'total_offense_mean': sum of all offensive projections,
        'total_offense_std': sqrt(sum of variances),  # Assuming independence
        'qb_rb_correlation': learned correlation coefficient,
        'star_power': max(player_projection - baseline),
        'depth_quality': mean of backups,
        ...
    }
    """
    pass
```

### Features to Engineer

**Offensive Aggregates:**
- Total expected yards (pass + rush)
- Expected points (using EPA models)
- Variance in total offense (uncertainty)
- Position balance (pass vs rush ratio)
- Star concentration (Gini coefficient of player values)

**Defensive Aggregates:**
- Expected yards allowed (inverse of opponent projections)
- Pass rush pressure rate
- Coverage quality (PFF grades if available)
- Turnover expectation

**Matchup-Specific:**
- QB projection vs opponent pass defense
- RB projection vs opponent run defense
- WR projection vs opponent secondary
- Offensive line vs defensive line

### Advantages
- ✅ Simple, interpretable
- ✅ Low risk of overfitting (few parameters)
- ✅ Leverages existing player models
- ✅ Can incorporate uncertainty naturally

### Disadvantages
- ❌ Manual feature engineering required
- ❌ May miss complex synergies
- ❌ Assumes independence between players

### Recommendation
**HIGH PRIORITY - Implement First**. This is the most practical near-term approach.

---

## Option 2: Bayesian Hierarchical Model (Team-Level Pooling)

### Concept
Build a hierarchical Bayesian model that pools player-level uncertainty to team-level outcomes.

### Mathematical Framework

```
Player Level (existing):
  θ_player ~ N(μ_player, σ²_player)  # Player projection

Team Level (new):
  Team_Score = f(θ_QB, θ_RB1, θ_RB2, θ_WR1, ..., θ_OL_avg, ε_synergy)

  Where:
  - f() is a learned aggregation function
  - ε_synergy ~ N(0, σ²_synergy) captures team chemistry
  - Priors on aggregation weights informed by positional value

Game Level:
  Margin ~ N(Team_Home - Team_Away + HFA, σ²_game)
```

### Implementation Strategy

1. **Use existing player posteriors** from Bayesian/BNN models
2. **Learn aggregation weights** via MCMC (Stan/PyMC) or Variational Bayes
3. **Incorporate synergy terms**:
   - QB-WR chemistry (same team tenure)
   - Offensive line cohesion (starts together)
   - Defensive coordinator effects

### Advantages
- ✅ Principled uncertainty propagation
- ✅ Interpretable parameters
- ✅ Can incorporate domain knowledge (priors)
- ✅ Naturally handles missing players (uncertainty increases)

### Disadvantages
- ❌ Computationally expensive (MCMC sampling)
- ❌ Requires careful prior specification
- ❌ May still overfit aggregation function

### Recommendation
**MEDIUM PRIORITY**. Worth exploring if simple aggregation (Option 1) works but lacks uncertainty quantification.

---

## Option 3: Multi-Task Learning

### Concept
Train a single neural network to jointly predict:
1. Player-level stats (QB yards, RB yards, etc.)
2. Game-level outcome (win/loss)

Shared representations learn useful features for both tasks.

### Architecture

```python
class MultiTaskGamePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(team_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Task-specific heads
        self.player_head = nn.Linear(64, n_players)  # Predict all player stats
        self.game_head = nn.Linear(64, 1)  # Predict game outcome

    def forward(self, x):
        shared = self.encoder(x)
        player_preds = self.player_head(shared)
        game_pred = torch.sigmoid(self.game_head(shared))
        return player_preds, game_pred
```

### Loss Function

```python
loss = α * player_loss + β * game_loss

where:
  player_loss = MSE(predicted_stats, actual_stats)  # Regression
  game_loss = BCE(predicted_outcome, actual_outcome)  # Classification
  α, β = task weights (tune via validation)
```

### Advantages
- ✅ Learns shared representations
- ✅ Player task provides auxiliary supervision
- ✅ Can use both player and game data
- ✅ Naturally handles multi-scale prediction

### Disadvantages
- ❌ Still a neural network (risk of overfitting)
- ❌ Requires balancing multiple losses
- ❌ Less interpretable than option 1

### Recommendation
**MEDIUM PRIORITY**. Interesting research direction but higher risk than simple aggregation.

---

## Option 4: Attention Mechanisms (Alternative to GNNs)

### Concept
Use attention to learn which players matter most for game outcomes, rather than fixed message passing.

### Architecture

```python
class AttentionGamePredictor(nn.Module):
    def __init__(self):
        # Player embeddings
        self.player_encoder = nn.Linear(player_features, 64)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1
        )

        # Game outcome predictor
        self.game_head = nn.Linear(64, 1)

    def forward(self, player_features):
        # Encode each player: [n_players, 64]
        player_embeds = self.player_encoder(player_features)

        # Attend over players: which players matter most?
        attended, attn_weights = self.attention(
            player_embeds,  # Query
            player_embeds,  # Key
            player_embeds   # Value
        )

        # Aggregate via mean pooling
        team_repr = attended.mean(dim=0)

        # Predict game outcome
        return self.game_head(team_repr), attn_weights
```

### Key Innovation
- **Learned Aggregation**: Attention weights automatically learn which players matter
- **Interpretability**: Can visualize which players the model focuses on
- **Flexibility**: Different attention patterns for different game contexts

### Advantages
- ✅ More flexible than GNN message passing
- ✅ Interpretable attention weights
- ✅ Can handle variable roster sizes
- ✅ Less complex than full GNN (fewer parameters)

### Disadvantages
- ❌ Still a neural network (overfitting risk)
- ❌ Requires more data than simpler methods
- ❌ Attention weights may not be stable

### Recommendation
**LOW PRIORITY**. Interesting but likely faces same overfitting issues as GNNs given limited data.

---

## Option 5: Reinforcement Learning (NEW DIRECTION)

### Why RL?
You haven't explored RL in your current work, but it offers unique advantages for NFL analytics:

1. **Sequential Decision Making**: Football is inherently sequential (play-by-play)
2. **Optimal Strategy Learning**: Learn play-calling, 4th down decisions, etc.
3. **Counterfactual Reasoning**: "What if we ran instead of passed?"

### Application Areas

#### 5A: Play-Calling Optimization

```python
# State: game situation
state = {
    'down': 3,
    'distance': 7,
    'field_position': 35,
    'score_diff': -3,
    'time_remaining': 180,  # seconds
    'player_fatigue': [0.8, 0.9, ...],  # From player models
}

# Action: play type
action = {
    'play_type': 'pass',  # or 'run'
    'formation': '11_personnel',
    'target': 'WR1',
}

# Reward: change in win probability
reward = Δ(win_prob) + bonus_for_points
```

**Use Player Projections**: Player fatigue, matchup advantages feed into state representation.

#### 5B: 4th Down Decision Agent

```python
# Policy: π(go_for_it | state)
state = {
    'yards_to_go': 2,
    'field_position': 42,
    'score_diff': 4,
    'qb_success_rate': 0.68,  # From player model
    'kicker_fg_prob': 0.82,   # From player model
}

# Learn optimal decision policy via:
# - Q-learning (model-free)
# - Model-based RL (use player models as transition dynamics)
```

#### 5C: In-Game Adaptation

```python
# Learn how to adapt strategy as game progresses
# Example: If QB is underperforming, increase run ratio

class AdaptiveCoach(RL_Agent):
    def update_strategy(self, observed_performance):
        if qb_actual < qb_expected - 2*sigma:
            self.increase_run_play_prob()
        if defense_struggling:
            self.call_more_blitzes()
```

### Advantages of RL
- ✅ Handles sequential nature of football
- ✅ Can learn optimal strategies from data
- ✅ Naturally incorporates player performance
- ✅ Counterfactual reasoning (what-if analysis)
- ✅ Can optimize for specific objectives (win prob, fantasy points, etc.)

### Disadvantages of RL
- ❌ Requires play-by-play data (do you have this?)
- ❌ Sparse rewards (only final outcome observed)
- ❌ Large state/action spaces
- ❌ Sample complexity (need many games)

### RL Implementation Roadmap

**Phase 1: 4th Down Agent** (Simplest RL problem)
- Binary decision (go vs. punt/kick)
- Clear reward signal (did it work?)
- Can validate against analytics (NYT 4th down bot)

**Phase 2: Play-Calling Agent**
- More complex action space
- Use player projections as state features
- Reward = EPA (Expected Points Added)

**Phase 3: Full Game Strategy**
- Sequential decision making across entire game
- Combine with player fatigue models
- Ultimate goal: optimal coaching decisions

### Recommendation
**HIGH POTENTIAL - NEW RESEARCH DIRECTION**. RL is underexplored in your work and could provide unique insights. Start with 4th down decisions (tractable problem, clear evaluation).

---

## Option 6: Causal Inference (Bridge Player → Game)

### Concept
Use causal models to understand how player performance causes game outcomes.

### Framework

```
Player Performance → Team Performance → Game Outcome
     (measured)          (latent)        (measured)

Causal DAG:
  QB_yards → Offensive_EPA → Score_Diff → Win
  RB_yards ↗                    ↑
  OL_quality ────────────────────┘
```

### Implementation

1. **Estimate causal effects** of each player on team outcome
2. **Use do-calculus** to answer counterfactuals:
   - "What if Mahomes had 300 yards instead of 250?"
   - "What if the RB was healthy?"

3. **Aggregate causal effects** to predict game outcomes

### Advantages
- ✅ Interpretable causal relationships
- ✅ Counterfactual reasoning
- ✅ Can handle confounders (e.g., weather affects both QB and outcome)
- ✅ Uncertainty quantification via Bayesian causal inference

### Disadvantages
- ❌ Requires strong causal assumptions
- ❌ Hard to validate causal graphs
- ❌ Computationally expensive

### Recommendation
**MEDIUM PRIORITY - RESEARCH VALUE**. Interesting for understanding mechanisms, less practical for prediction.

---

## Option 7: Ensemble of Simple Aggregations

### Concept
Instead of one complex model, combine multiple simple aggregations:

1. **Sum of Projected Points** (using EPA)
2. **Weighted Average by Position** (QB = 0.4, RB = 0.2, etc.)
3. **Percentile-Based** (is QB in top 80%? RB in top 60%?)
4. **Opponent-Adjusted** (project vs. specific defense)
5. **Uncertainty-Weighted** (weight confident predictions more)

Then meta-learn via stacking (like you did for game ensemble).

### Advantages
- ✅ Diversifies aggregation approaches
- ✅ Each component is simple/interpretable
- ✅ Lower overfitting risk than single complex model
- ✅ Can weight by validation performance

### Disadvantages
- ❌ Still need to engineer each aggregation
- ❌ Meta-learner can overfit (as you saw)

### Recommendation
**MEDIUM PRIORITY**. Natural extension of Option 1 if simple aggregation works.

---

## Recommended Research Sequence

### Phase 1: Quick Wins (1-2 weeks)
1. **Implement Option 1** (Feature Engineering from Player Projections)
   - Start simple: sum of projected EPA
   - Add uncertainty features (std of projections)
   - Compare against state-space baseline

2. **Validate on 2024 → Test on 2025**
   - If Brier < 0.23: Success!
   - If not, diagnose why (which features matter?)

### Phase 2: Principled Uncertainty (2-4 weeks)
3. **Implement Option 2** (Bayesian Hierarchical Pooling)
   - Use Stan or PyMC for posterior sampling
   - Incorporate player projection uncertainty
   - Compare against Option 1

### Phase 3: New Research Direction (1-2 months)
4. **Explore Option 5** (Reinforcement Learning)
   - Start with 4th down decisions
   - Use player projections as state features
   - Publish findings (novel application)

### Phase 4: Ensemble (1 week)
5. **Combine best approaches** (Option 7)
   - If multiple methods work, ensemble them
   - Use inverse variance weighting
   - Validate against state-space baseline

---

## Decision Criteria

### When to Use Each Option

| Criteria | Best Option |
|----------|-------------|
| **Need results quickly** | Option 1 (Feature Engineering) |
| **Have play-by-play data** | Option 5 (Reinforcement Learning) |
| **Want interpretability** | Option 1 or 2 (Feature/Bayesian) |
| **Have uncertainty quantification** | Option 2 (Bayesian Hierarchical) |
| **Novel research contribution** | Option 5 (RL) or 6 (Causal) |
| **Limited data (high overfit risk)** | Option 1 (Simple Features) |

---

## Why GNNs Failed (Postmortem)

Let's be explicit about the failure modes to avoid repeating them:

1. **Too many parameters** (140K-340K) for 1,390 games → severe overfitting
2. **Wrong inductive bias**: Message passing assumes graph structure matters, but NFL is more about aggregation (sum of parts)
3. **No uncertainty quantification**: GNN outputs were point estimates, lost player-level uncertainty
4. **Hierarchical complexity**: Player → Position → Team structure added parameters without clear benefit
5. **MPS tensor corruption**: Technical issues prevented GPU training, forced slow CPU training

**Lesson**: With limited data, simpler models win. Graph structure may not be right abstraction for NFL.

---

## Concrete Next Steps

### This Week
1. Read existing player projections (from Bayesian/BNN models)
2. Engineer 10-15 team-level features (Option 1)
3. Train XGBoost on aggregated features
4. Evaluate on 2024 → 2025 (compare vs. state-space)

### Next Month
- If Option 1 works: Implement Bayesian pooling (Option 2)
- If Option 1 fails: Diagnose why, try simpler features
- Write up findings for dissertation

### Future Research (Post-Dissertation)
- 4th down RL agent (publishable result)
- Causal inference framework (mechanistic understanding)
- Multi-task learning (if you have time)

---

## Questions to Resolve

1. **Do you have play-by-play data?** (Required for RL options)
2. **What's your timeline?** (Quick results → Option 1, Novel research → Option 5)
3. **Primary goal?** (Prediction accuracy vs. mechanistic understanding)
4. **Available compute?** (Bayesian inference is expensive)

Let me know your priorities and we can dive deeper into the most promising direction!
