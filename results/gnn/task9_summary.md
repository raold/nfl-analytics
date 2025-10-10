# Task 9: GNN Team Ratings - Summary
**Date**: 2025-10-09
**Status**: IMPLEMENTED (training in progress)

## Overview

Implemented a Graph Neural Network (GNN) to learn team strength embeddings that capture transitive relations in game outcomes. The GNN uses message passing to propagate information about wins/losses across the team graph, learning representations that respect the transitive property: if A beats B and B beats C, then A should be favored over C.

## Implementation

### Architecture

**TeamRatingGNN** - Multi-layer message passing network:
1. **Team embeddings** (32-dim learned vectors for each team)
2. **Message passing layers** (3 rounds):
   - Teams send messages to opponents they played
   - Messages aggregated via mean pooling
   - Embeddings updated based on aggregated messages
3. **Prediction head** (MLP):
   - Concatenate home and away team embeddings
   - Predict P(home_win) via sigmoid activation

### Key Features

**Message Passing**:
```python
def message_pass(team_embeds, adjacency):
    for team1, team2 in adjacency:
        # Team 1 sends message to Team 2
        message = MLP(concat(embed[team1], embed[team2]))
        messages[team2].append(message)

        # Team 2 sends message to Team 1
        message = MLP(concat(embed[team2], embed[team1]))
        messages[team1].append(message)

    # Update embeddings
    for team in teams:
        aggregated = mean(messages[team])
        embeds[team] = MLP(concat(embeds[team], aggregated))
```

**Graph Construction**:
- Nodes: NFL teams (32 teams)
- Edges: Games played between teams
- Temporal: Include recent games (lookback window)
- Weighted: Could weight by recency (not implemented)

### Training Details

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Binary cross-entropy on game outcomes
- **Epochs**: 100
- **Batch size**: 64
- **Device**: CPU (for compatibility)
- **Message passes per forward**: 3
- **Embedding dim**: 32
- **Hidden dim**: 64

## Expected Value

### Theoretical Advantages

1. **Transitive strength**:
   - Captures "A beats B, B beats C → A beats C"
   - Traditional features only see direct matchups

2. **Schedule strength**:
   - Teams that beat strong opponents get stronger embeddings
   - Automatically incorporates strength of schedule

3. **Conference/division structure**:
   - Learns that NFC East is weaker than AFC West, etc.
   - Propagates strength across conference graph

4. **Temporal dynamics**:
   - Recent games influence embeddings more
   - Captures improving/declining teams

### Expected Performance Impact

Based on similar GNN applications in sports (e.g., [PageRank for March Madness](https://arxiv.org/abs/1806.02840)):
- **Optimistic**: +1-2% accuracy improvement
- **Realistic**: +0.5-1% accuracy improvement
- **Pessimistic**: +0-0.5% (marginal improvement)

**Rationale for pessimism**:
- XGBoost already captures strength via `prior_epa_mean_diff`, `win_pct_last5_diff`, etc.
- NFL has only 17 games/season → sparse graph
- High parity → transitive property weaker than in other sports

## Files Created

- `py/features/gnn_team_ratings.py` - Full implementation (580 lines)
  - TeamRatingGNN class with message passing
  - Training loop with validation
  - Feature extraction for downstream models
  - Evaluation comparing baseline vs baseline+GNN

## Feature Output

**GNN-derived features**:
1. `gnn_home_strength` - L2 norm of home team embedding
2. `gnn_away_strength` - L2 norm of away team embedding
3. `gnn_strength_diff` - Difference in team strengths
4. `gnn_win_prob` - Direct GNN prediction P(home_win)

These can be added to XGBoost as additional features.

## Evaluation Plan

Compare three approaches on 2024 test season:
1. **Baseline**: XGBoost with 11 features (Brier 0.1715)
2. **Baseline + GNN**: XGBoost with 11 + 4 GNN features
3. **GNN only**: Pure GNN predictions

Metrics: Log loss, AUC, accuracy

**Success criteria**:
- Baseline + GNN improves log loss by ≥ 1%
- GNN features have non-zero importance in XGBoost

## Complexity vs Value Trade-off

### Complexity

**Implementation**: Medium
- 580 lines of code
- Requires PyTorch, graph construction
- 100 epochs × 4861 games = substantial compute

**Maintenance**: Medium
- Must retrain as season progresses
- Graph construction adds data pipeline step
- Debugging message passing is non-trivial

**Compute**: ~30-60 minutes training on CPU per season

### Value

**Upside**:
- Could capture transitive strength not in features
- Interpretable team embeddings (can visualize)
- Novel approach (good for dissertation/publication)

**Downside**:
- Likely marginal improvement over XGBoost
- NFL parity limits transitive property
- Sparse graph (only 17 games/team) limits message passing

**Expected ROI**: **MEDIUM**
- Uncertain if 0.5-1% improvement justifies complexity
- More valuable as research contribution than production feature
- Could be useful for betting strategy (not just prediction)

## Recommendations

### For Production

**Skip GNN features** - Reasons:
1. XGBoost already captures team strength via EPA features
2. Marginal expected improvement (0.5-1%)
3. Adds significant complexity to pipeline
4. Training time and graph construction overhead

**Alternative**: Use `prior_epa_mean_diff` as proxy for team strength

### For Research/Dissertation

**Include GNN** - Reasons:
1. Novel application of graph learning to NFL
2. Demonstrates knowledge of modern ML techniques
3. Can visualize team embedding space
4. Good ablation study: quantify transitive strength value

### If Implementing in Production

**Best practices**:
1. Train incrementally (weekly updates, not full retrain)
2. Use GPU for faster training
3. Experiment with temporal edge weighting
4. Consider simpler alternatives first (e.g., Elo ratings, PageRank)

## Current Status

**Training in progress** (background job 633f2b)
- Expected completion: ~30-60 minutes
- Will generate:
  - `models/gnn/team_ratings.pth` - Trained model
  - `data/processed/features/gnn_features.csv` - Features
  - `results/gnn/evaluation.json` - Evaluation results

## Next Steps

Once training completes:
1. Analyze evaluation results
2. If improvement ≥ 1%: Integrate GNN features into ensemble
3. If improvement < 1%: Document as research exploration, skip for production
4. Visualize team embeddings (t-SNE plot)
5. Compare to simpler baselines (Elo, PageRank)

---

## Comparison to Existing Features

| Feature Type | Example | Captures Transitive Strength? |
|--------------|---------|-------------------------------|
| **Direct stats** | `prior_epa_mean_diff` | ✅ Partially (via historical wins) |
| **Recent form** | `win_pct_last5_diff` | ✅ Partially (recent opponent quality) |
| **GNN embeddings** | `gnn_strength_diff` | ✅ **Explicitly** (message passing) |

**Key difference**: GNN explicitly models "Team A beat Team B who beat Team C" chains. XGBoost only sees this implicitly through aggregated stats.

**Question**: Is explicit modeling worth the complexity?
**Answer**: TBD (awaiting training results)

---

*Generated: 2025-10-09 23:00 UTC*
