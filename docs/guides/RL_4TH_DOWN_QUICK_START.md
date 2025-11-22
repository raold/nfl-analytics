# RL 4th Down Agent - Quick Start Guide

**Purpose**: Get up and running with the 4th down CQL agent in <5 minutes
**Audience**: Future sessions, collaborators, dissertation reviewers
**Status**: Ready for Week 2 (hyperparameter tuning)

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Activate environment
source .venv/bin/activate && export PYTHONPATH=.

# 2. Train agent (50 epochs, ~5 minutes on M4)
python scripts/train_fourth_down_cql.py \
    --mode single \
    --epochs 50 \
    --alpha 1.0 \
    --lr 1e-4 \
    --hidden-dims 128 64 32

# 3. Check results
cat models/fourth_down_*/evaluation_report.json
```

**Expected Output**:
- Agreement rate: ~65-70% (baseline)
- Reward lift: ~0.2-0.5
- Training time: 3-5 minutes

---

## 📊 Dataset Overview

**Already Extracted**: `data/fourth_down_dataset_*.csv`

```
Train: 99,112 plays (1999-2023)
Test:   5,885 plays (2024-2025)
Total: 104,997 plays (26 seasons)
```

**Actions**:
- 0: Go (pass/run) - 14.3%
- 1: Punt - 62.0%
- 2: Field Goal - 23.7%

**State**: 10 features (distance, field position, score, time, WP, etc.)
**Reward**: Δ win probability + bonuses

---

## 🎯 Week 2 Tasks

### 1. Hyperparameter Grid Search (Priority 1)

**Recommended**:
```bash
python scripts/train_fourth_down_cql.py \
    --mode grid \
    --n-trials 20 \
    --epochs 300 \
    --output-dir models/fourth_down_grid
```

**Searches**:
- alpha: [0.1, 0.3, 1.0, 3.0]
- lr: [1e-5, 1e-4, 1e-3]
- hidden_dims: [[128,64], [256,128,64], [512,256,128]]
- batch_size: [128, 256, 512]

**Runtime**: ~24 hours (20 trials × 300 epochs × ~4 min/trial)

**Output**: `models/fourth_down_grid/grid_search_results.json`

### 2. Train Best Model (Priority 2)

After grid search, train best hyperparameters for longer:

```bash
# Extract best hyperparameters
python -c "import json; results = json.load(open('models/fourth_down_grid/grid_search_results.json')); best = max(results, key=lambda x: x['reward_lift']); print(best['hyperparameters'])"

# Train with best config
python scripts/train_fourth_down_cql.py \
    --mode single \
    --epochs 1000 \
    --alpha <BEST_ALPHA> \
    --lr <BEST_LR> \
    --hidden-dims <BEST_HIDDEN_DIMS> \
    --output-dir models/fourth_down_final
```

**Runtime**: ~8 hours (1000 epochs)

### 3. Evaluate on Test Set (Priority 3)

```bash
python py/rl/fourth_down_agent.py \
    --test-data data/fourth_down_dataset_test.csv \
    --load models/fourth_down_final/cql_agent_final.pth \
    --evaluate
```

**Outputs**:
- `models/fourth_down_final/evaluation_report.json`
- Agreement rate, reward lift, Q-value stats
- Action-specific performance

---

## 📁 File Locations

**Code**:
- `py/rl/fourth_down_mdp.py` - MDP definition (state/action/reward)
- `py/rl/fourth_down_agent.py` - CQL agent training/evaluation
- `scripts/train_fourth_down_cql.py` - Training pipeline

**Data**:
- `data/fourth_down_dataset_train.csv` - Training set
- `data/fourth_down_dataset_test.csv` - Test set
- `data/fourth_down_dataset_metadata.json` - Statistics

**Models** (created during training):
- `models/fourth_down_*/cql_agent_final.pth` - Trained model
- `models/fourth_down_*/training_log.json` - Loss curves
- `models/fourth_down_*/evaluation_report.json` - Metrics

**Documentation**:
- `docs/milestones/RL_4TH_DOWN_PHASE1_3_COMPLETE.md` - Phases 1-3 summary
- `docs/guides/RL_4TH_DOWN_QUICK_START.md` - This file

---

## 🔍 Key Metrics

**Agreement Rate**: % match with actual coaches' decisions
- Baseline: 65-70% (random policy ~33%, always punt ~62%)
- Target: >85% (matches NYT bot)
- Stretch: >90%

**Reward Lift**: Expected reward improvement over actual decisions
- Baseline: 0.0 (actual coaches)
- Target: >0.5 (5% WP improvement)
- Stretch: >1.0 (10% WP improvement)

**Win Probability Impact**: Counterfactual analysis on 2024 season
- Target: +0.02 WP per 4th down (2% average improvement)
- Stretch: +0.05 WP per 4th down (5% average improvement)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'py.rl'"

**Fix**: Set PYTHONPATH
```bash
export PYTHONPATH=.
```

### "AssertionError: NaN values in states/rewards"

**Fix**: Regenerate dataset (handles NULL values)
```bash
python py/rl/fourth_down_mdp.py --output data/fourth_down_dataset.csv
```

### Training is slow

**Options**:
1. Reduce epochs: `--epochs 100` (faster testing)
2. Increase batch size: `--batch-size 512` (fewer updates)
3. Use smaller network: `--hidden-dims 64 32` (faster forward pass)

### Agreement rate is low (<60%)

**Diagnosis**:
- Check Q-value range (should be [-10, 10])
- Try higher alpha (more conservative): `--alpha 3.0`
- Train longer: `--epochs 1000`

---

## 📖 Next Steps After Week 2

**Week 3**: NYT Bot Comparison
- Scrape NYT 4th Down Bot recommendations
- Match to 2024 test set
- Compute agreement: CQL vs NYT vs coaches

**Week 4**: Counterfactual Analysis
- Simulate "what if team X followed CQL policy?"
- Estimate win probability changes
- Identify most impactful divergences

**Weeks 5-6**: Dissertation Writing
- 15-20 page section for Chapter 7
- Publication-ready figures and tables
- Case studies of interesting decisions

---

## 🎓 Dissertation Integration

**Chapter**: Chapter 7 (Reinforcement Learning)
**Section**: Section 7.4 (Fourth Down Decision-Making)
**Length**: 15-20 pages
**Contribution**: Novel application of CQL to NFL 4th down decisions

**Required Elements**:
- Introduction & motivation
- Related work (Burke, Romer, NYT bot)
- MDP formulation
- CQL algorithm
- Experimental results
- Baseline comparisons
- Counterfactual analysis
- Discussion & limitations

**Quality Standards**: See `docs/guides/DISSERTATION_STANDARDS.md`

---

## 💡 Tips

1. **Start with grid search**: Don't waste time on manual tuning
2. **Monitor Q-values**: Should be in [-10, 10] range (reward-scaled)
3. **Check agreement by context**: Short distance (1-2 yards) should have higher go rate
4. **Compare vs always punt**: Agreement ~62% means agent is just copying coaches
5. **Look for divergence patterns**: When does RL differ from coaches? (field position, score, time)

---

## 🔗 References

**Algorithm**: Kumar et al. (2020) - Conservative Q-Learning for Offline RL
**Domain**: Burke et al. (2009) - Fourth Down Study
**Baseline**: NYT 4th Down Bot (2017)
**Related**: `docs/milestones/CQL_COMPLETE_SUMMARY.md` (betting agent, +91% ROI)

---

**Last Updated**: November 22, 2025
**Status**: ✅ Ready for Week 2
**Next Action**: Run hyperparameter grid search
