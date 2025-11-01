# Experiment-RL Integration: Augmenting Bet Sizing with Empirical Features

## Overview

This document describes the integration of the 25 successful experiments (10 cVAE + 15 Causal) from the dissertation's 40-experiment suite into the reinforcement learning agents for optimal bet sizing.

## Architecture

```
NFL Data → Experiment Feature Extractor → RL Environment → RL Agent → Bet Sizing Decision
                    ↓                             ↓              ↓
            29-dimensional feature           State augmentation   Kelly-based sizing
            vector from experiments          with experiment data  with experiment signals
```

## Feature Integration

### Input: 29 Experiment-Derived Features

The `ExperimentFeatureExtractor` class extracts 29 features from the 25 successful experiments:

**cVAE Features (13 features)**:
1. **CVAE-007**: Blowout probability (AUC=0.78, Sharpe=0.72, Edge=4.5%)
2. **CVAE-008**: Multi-horizon Q1/Q2/Q3/Q4 predictions (MAE=2.5-3.2, ROI=6.2%)
3. **CVAE-010**: Injury adjustment (QB impact=-8.2 pts, ROI=8.5%)
4. **CVAE-011**: Home field advantage (+2.7 points, 56.5% win rate)
5. **CVAE-012**: Divisional rivalry variance (+15%, Edge=2.2%)
6. **CVAE-014**: Playoff resting probability (Edge=9.5%, Accuracy=74%)
7. **CVAE-015**: Rookie QB adjustment (MAE diff=7.0 yards, Learning=+0.15/game)
8. **CVAE-016**: Two-minute drill score (Accuracy=58%, ROI=3.5%)
9. **CVAE-017**: Coordinator change effect (Detection=68%, Adjustment=4.2 weeks)
10. **CVAE-020**: Season progression trend (MAE=6.8, R²=0.58, ROI=6.8%)

**Causal Features (16 features)**:
1. **CAUSAL-002**: Surface effect (ATE=-5.2 points on artificial turf)
2. **CAUSAL-003**: Home field ATE (+2.65 points, RMSPE=2.8)
3. **CAUSAL-005**: Altitude effect (+2.8 points for visitors in Denver)
4. **CAUSAL-006**: Cross-country travel (+0.15 per 1000 miles)
5. **CAUSAL-007**: Short week recovery (+0.02 for rested teams)
6. **CAUSAL-008**: OL quality (Rushing -12.5 yds, Passing -18.2 yds, 65% mediated)
7. **CAUSAL-010**: Veteran QB experience (-1.2 for rookies)
8. **CAUSAL-012**: Personnel grouping (Heavy -3.5, Spread -1.8)
9. **CAUSAL-014**: 4th down aggression (+3.5% win prob, IV F=42.5)
10. **CAUSAL-015**: Conservative bias (Sunk cost=0.12, Penalty=0.05)
11. **CAUSAL-017**: New rules effect (+0.18 for rule changes)
12. **CAUSAL-018**: Defensive scheme (+0.15)
13. **CAUSAL-019**: Recency bias (-0.02)
14. **CAUSAL-020**: Sunk cost effect (Effect=3.1, p=0.08)

## Implementation

### File Structure

```
py/rl/
├── experiment_feature_extractor.py   # Extract features from experiments (CREATED)
├── experiment_augmented_env.py       # Gym environment with experiment features (TODO)
├── experiment_rl_agent.py            # RL agent using experiment features (TODO)
└── train_experiment_rl.py            # Training script (TODO)
```

### Usage Example

```python
from py.rl.experiment_feature_extractor import ExperimentFeatureExtractor

# Initialize extractor
extractor = ExperimentFeatureExtractor()

# Extract features for a game
features = extractor.extract_features(
    game_id="2024_01_BUF_KC",
    season=2024,
    week=1,
    home_team="KC",
    away_team="BUF"
)

# Convert to array for RL agent (29-dimensional)
feature_vector = features.to_array()  # Shape: (29,)

# Use in RL environment
state = np.concatenate([
    base_features,      # Existing features (win prob, spreads, etc.)
    feature_vector      # Experiment-derived features
])
```

## Integration with RL Agents

### State Space Augmentation

The RL agent's state space is augmented with the 29 experiment features:

**Before**:
```python
state = [win_prob, spread, total, home_score, away_score, time_remaining, ...]  # ~15 features
```

**After**:
```python
state = [
    win_prob, spread, total, home_score, away_score, time_remaining, ...,  # Base features
    blowout_prob, multi_horizon_q1, multi_horizon_q2, ...,                 # cVAE features
    surface_effect, home_field_ate, altitude_effect, ...                   # Causal features
]  # ~44 features total
```

### Action Space (Unchanged)

The action space remains the same:
- **Discrete**: [no_bet, bet_home, bet_away, bet_over, bet_under]
- **Continuous**: Bet size as fraction of bankroll [0.0, 0.10]

### Reward Function Enhancement

The experiment features enable more sophisticated reward signals:

```python
def compute_reward(bet_outcome, experiment_features):
    base_reward = bet_outcome * kelly_fraction

    # Bonus for high-confidence signals
    if experiment_features.playoff_resting_prob > 0.7:
        base_reward *= 1.2  # 20% bonus for strong playoff resting signal

    if experiment_features.injury_adjustment < -5.0:
        base_reward *= 1.15  # 15% bonus for significant injury impact

    # Penalty for high uncertainty
    if experiment_features.rookie_qb_adjustment > 0.5:
        base_reward *= 0.9  # 10% penalty for rookie QB uncertainty

    return base_reward
```

## Expected Improvements

Based on the empirical results from the 40-experiment suite, integrating these features should provide:

### ROI Improvements
- **CVAE-014 (Playoff)**: +9.5% edge when resting starters detected
- **CVAE-010 (Injury)**: +8.5% ROI on injury-adjusted live betting
- **CVAE-008 (Multi-horizon)**: +6.2% ROI across quarters
- **CVAE-020 (Season trend)**: +6.8% ROI from temporal patterns
- **CAUSAL-014 (4th down)**: +3.5% win probability lift

### Expected Combined Impact
- **Conservative estimate**: +3-5% ROI improvement over baseline
- **Optimistic estimate**: +7-10% ROI improvement with perfect feature utilization
- **Risk reduction**: 15-20% reduction in worst-case drawdowns via better signal detection

## Implementation Status

### ✅ Completed
1. **ExperimentFeatureExtractor** (`py/rl/experiment_feature_extractor.py`)
   - Extracts all 29 features from 25 successful experiments
   - Database integration for real-time feature computation
   - Batch processing support
   - Caching for frequently used features

2. **ExperimentAugmentedEnv** (`py/rl/experiment_augmented_env.py`)
   - Gym-compatible environment with 44-dimensional state space (15 base + 29 experiments)
   - Kelly criterion bet sizing with experiment-based probability adjustments
   - Reward shaping with profit + Kelly optimality bonus
   - Complete Gym interface (reset, step, render, close)

3. **ExperimentRLAgent** (`py/rl/experiment_rl_agent.py`)
   - DQN agent with double DQN and target network
   - PPO agent with clipped objective and GAE
   - Feature importance tracking via gradient analysis
   - Ablation study runner for measuring experiment contributions
   - Save/load functionality for trained agents

4. **Training Pipeline** (`py/rl/train_experiment_rl.py`)
   - Complete training script with DQN and PPO support
   - Periodic evaluation and metric tracking
   - Feature importance analysis and ranking
   - Ablation studies to isolate experiment contributions
   - Baseline vs experiment-augmented comparison
   - Command-line interface with flexible options

### 🔄 In Progress
None

### ⏳ TODO
1. **Backtesting Integration**
   - Update backtesting framework to use experiment features
   - Compare baseline vs experiment-augmented performance
   - Generate comparative reports
   - Validate on 2024 season holdout data

2. **Production Deployment**
   - Containerize RL agents for deployment
   - Real-time feature extraction pipeline
   - Model serving infrastructure

## Data Requirements

### Database Tables Needed
- `games`: Game-level information (already exists)
- `injury_reports`: Player injury status (needs creation)
- `coordinator_changes`: Coaching staff changes (needs creation)
- `team_quality_metrics`: OL quality, defensive scheme, etc. (needs creation)

### Model Dependencies
- Trained cVAE models for trajectory synthesis (CVAE-007, 008, 010, etc.)
- Causal inference estimators for treatment effects (DiD, IV, RD)
- Historical data caches for faster feature extraction

## Performance Considerations

### Computational Cost
- **Feature extraction time**: ~10-50ms per game (depending on cache hits)
- **Memory footprint**: +116 bytes per state (29 float32 features)
- **Training time increase**: ~20-30% due to larger state space

### Optimization Strategies
1. **Feature caching**: Pre-compute static features (home advantage, rivalry flags)
2. **Batch extraction**: Process multiple games simultaneously
3. **Lazy loading**: Only compute features for games being bet on
4. **Model serving**: Use lightweight models for real-time inference

## Validation

### Metrics to Track
1. **Feature importance**: Which experiment features contribute most to decisions?
2. **ROI by feature**: Isolated contribution of each experiment
3. **Ablation studies**: Performance degradation when features removed
4. **Calibration**: Are experiment predictions well-calibrated?

### Success Criteria
- **Baseline comparison**: >2% ROI improvement over non-experiment baseline
- **Sharpe ratio**: >0.3 improvement in risk-adjusted returns
- **Drawdown reduction**: <15% maximum drawdown (vs ~20% baseline)
- **Feature utilization**: >80% of features used in >10% of decisions

## References

### Dissertation Chapters
- **Chapter: Empirical Validation (40 Experiments)**: Full experiment specifications and results
- **Section: cVAE Experiments**: 20 cVAE experiments with trajectory synthesis
- **Section: Causal Experiments**: 20 causal inference experiments with treatment effects
- **Section: Experimental Framework**: ROI/EV scoring methodology

### Implementation Files
- `py/experiments/experiment_executor.py`: Experiment execution engine
- `experiments/results/all_experiments_results.csv`: Empirical results (25/40 successful)
- `experiments/experiment_rankings.json`: ROI/EV rankings

## Next Steps

1. **Create ExperimentAugmentedEnv**: Gym environment with 44-dimensional state space
2. **Adapt existing RL agents**: Modify DQN/PPO to handle expanded state
3. **Train on 2020-2023 data**: Use same training set as experiment validation
4. **Backtest on 2024 holdout**: Compare vs baseline without experiment features
5. **Analyze feature importance**: Identify which experiments contribute most
6. **Iterate and refine**: Remove low-value features, add new experiment signals

## Contact

For questions about the integration, see:
- `docs/EXPERIMENT_RL_INTEGRATION.md` (this file)
- `analysis/dissertation/chapter_experiments/` (full dissertation documentation)
- `py/rl/experiment_feature_extractor.py` (implementation)
