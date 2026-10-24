# Backtesting Framework

**Location:** `nfl_analytics/backtests/`
**Purpose:** Historical validation and performance testing of betting models
**Total Modules:** 12 Python files

---

## 📋 Overview

This directory contains backtesting scripts for validating NFL betting models on historical data. All backtests use **walk-forward validation** to prevent look-ahead bias and simulate realistic betting scenarios.

**Key Principle:** Only use information that would have been available at bet placement time.

---

## 🎯 Backtest Types

### 1. Model Comparison

Compare performance across different model versions.

| File | Purpose | Models Compared |
|------|---------|-----------------|
| **compare_v1_v2_5.py** | v1.0 vs v2.5 comparison | Baseline vs Enhanced |
| **comprehensive_model_comparison.py** | Multi-version comparison | v1.0, v2.0, v2.5, v3.0 |
| **model_version_comparison.py** | Version-specific metrics | All versions |

**Usage:**
```bash
# Compare v1.0 vs v2.5
uv run python nfl_analytics/backtests/compare_v1_v2_5.py \
  --start-season 2022 \
  --end-season 2024

# Comprehensive comparison
uv run python nfl_analytics/backtests/comprehensive_model_comparison.py \
  --output analysis/results/model_comparison.csv
```

**Expected Output:**
```
Model Version Comparison (2022-2024):
====================================
v1.0 Baseline:   ROI +1.59%, Win Rate 55.0%, Sharpe 1.2
v2.5 Enhanced:   ROI +3.8%,  Win Rate 58.2%, Sharpe 1.9
v3.0 Advanced:   ROI +5.4%,  Win Rate 59.8%, Sharpe 2.3

Improvement v1.0→v3.0: +3.81pp ROI, +4.8pp Win Rate
```

### 2. Bayesian Props Backtesting

Validate Bayesian hierarchical models for player props.

| File | Focus | Features |
|------|-------|----------|
| **bayesian_props_backtest_2024.py** | 2024 holdout | Single season validation |
| **bayesian_props_multiyear_backtest.py** | Multi-season | 2020-2024 walk-forward |

**Usage:**
```bash
# Backtest 2024 season
uv run python nfl_analytics/backtests/bayesian_props_backtest_2024.py

# Multi-year walk-forward
uv run python nfl_analytics/backtests/bayesian_props_multiyear_backtest.py \
  --start-year 2020 \
  --end-year 2024 \
  --retrain-frequency weekly
```

**Key Features:**
- Weekly model retraining
- As-of-date feature construction
- Uncertainty-adjusted bet sizing
- Tracking of Bayesian credible intervals

### 3. Game Outcome Backtesting

Test spread, total, and moneyline predictions.

| File | Focus |
|------|-------|
| **game_outcome_backtest.py** | Game-level predictions |

**Metrics Tracked:**
- Against-the-spread (ATS) performance
- Over/Under accuracy
- Moneyline ROI
- Closing line value (CLV)

**Usage:**
```bash
uv run python nfl_analytics/backtests/game_outcome_backtest.py \
  --seasons 2020 2021 2022 2023 2024 \
  --min-edge 0.03 \
  --kelly-fraction 0.25
```

### 4. Ensemble Backtesting

Validate ensemble methods.

| File | Purpose |
|------|---------|
| **comprehensive_ensemble_backtest.py** | Test all ensemble configurations |

**Ensemble Configurations Tested:**
- Simple average
- Inverse variance weighting
- Stacked meta-learner
- Thompson sampling

**Usage:**
```bash
uv run python nfl_analytics/backtests/comprehensive_ensemble_backtest.py \
  --methods average weighted stacked thompson \
  --output analysis/results/ensemble_comparison.csv
```

### 5. Phase 5 Comparison Backtests

Recent experimental backtests (Oct 2025).

| File | Experiment |
|------|------------|
| **phase5_comparison_backtest.py** | Phase 5 feature comparison |
| **phase5_comparison_backtest_optimized.py** | Optimized Phase 5 |
| **phase5_comprehensive_comparison.py** | Comprehensive Phase 5 |
| **phase5_three_way_comparison.py** | 3-way comparison |
| **realistic_backtest.py** | Realistic constraints |

**Note:** Phase 5 includes advanced defensive features and feature interactions.

---

## 🚀 Quick Start

### Run Standard Backtest

```bash
# Backtest Bayesian props model (2024)
uv run python nfl_analytics/backtests/bayesian_props_backtest_2024.py

# View results
cat analysis/results/bayesian_props_2024_metrics.csv
```

### Compare Model Versions

```bash
# Compare v1.0 vs v2.5 vs v3.0
uv run python nfl_analytics/backtests/comprehensive_model_comparison.py \
  --start-season 2022 \
  --end-season 2024 \
  --output analysis/results/version_comparison.csv
```

---

## 📊 Backtest Metrics

### Standard Metrics

All backtests report these metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **ROI** | Return on investment | >+5% |
| **Win Rate** | Percentage of winning bets | >55% |
| **Sharpe Ratio** | Risk-adjusted return | >2.0 |
| **Max Drawdown** | Largest peak-to-trough decline | <-10% |
| **Profit Factor** | Gross profit / gross loss | >1.5 |
| **Avg Bet Size** | Average wager as % of bankroll | 1-3% |

### Advanced Metrics

| Metric | Description |
|--------|-------------|
| **Closing Line Value (CLV)** | How often we beat closing line |
| **Calibration Score** | Prediction accuracy calibration |
| **Kelly Criterion Utilization** | Actual vs optimal bet sizing |
| **Correlation-Adjusted Return** | Returns accounting for bet correlation |

---

## ⚙️ Backtest Configuration

### Walk-Forward Validation

All backtests use walk-forward to prevent look-ahead bias.

```python
# Example walk-forward setup
train_window = 52  # weeks
test_window = 4    # weeks

for test_start in range(train_window, len(data), test_window):
    # Train on data before test period
    train_end = test_start
    train_data = data[train_end - train_window:train_end]

    # Test on next period
    test_data = data[test_start:test_start + test_window]

    # Train model
    model.fit(train_data)

    # Predict and evaluate
    predictions = model.predict(test_data)
    evaluate_predictions(predictions, test_data)
```

### Betting Constraints

Realistic betting constraints applied:

```python
BETTING_CONSTRAINTS = {
    'min_edge': 0.03,           # Minimum 3% edge to bet
    'max_bet_size': 0.05,       # Max 5% of bankroll per bet
    'kelly_fraction': 0.25,     # Quarter-Kelly for safety
    'max_correlated_exposure': 0.15,  # Max 15% on correlated bets
    'transaction_cost': 0.001,  # 0.1% transaction cost
    'min_odds': -300,           # Don't bet heavy favorites
    'max_odds': +400            # Don't bet long shots
}
```

---

## 📈 Example Results

### Bayesian Props Backtest (2024)

```
Bayesian Hierarchical Props - 2024 Holdout Results
==================================================
Total Bets: 487
Win Rate: 58.3% (284 wins, 203 losses)
ROI: +4.2%
Sharpe Ratio: 1.9
Max Drawdown: -11.3%

By Prop Type:
  Passing Yards: ROI +5.1%, Win Rate 59.2%
  Rushing Yards: ROI +3.8%, Win Rate 57.1%
  Receptions:    ROI +4.6%, Win Rate 60.3%
  Receiving Yds: ROI +3.2%, Win Rate 56.8%

Calibration: 91.2% (predictions within 1σ interval)
```

### Model Comparison (2022-2024)

```
Model Version Comparison (2022-2024)
====================================
v1.0 Baseline:
  ROI: +1.59%
  Win Rate: 55.0%
  Sharpe: 1.2
  Bets: 1,247

v2.5 Enhanced:
  ROI: +3.8%
  Win Rate: 58.2%
  Sharpe: 1.9
  Bets: 1,089

v3.0 Advanced:
  ROI: +5.4%
  Win Rate: 59.8%
  Sharpe: 2.3
  Bets: 976

Best Performer: v3.0 (+3.81pp ROI vs v1.0)
Most Selective: v3.0 (22% fewer bets, higher edge threshold)
```

---

## 🧪 Custom Backtests

### Create Custom Backtest

```python
import pandas as pd
from backtests.baseline_glm import BacktestHarness

class MyBacktest(BacktestHarness):
    def __init__(self, start_season, end_season):
        super().__init__(start_season, end_season)
        self.my_model = MyModel()

    def run(self):
        results = []
        for season in range(self.start_season, self.end_season + 1):
            # Load data
            data = self.load_season_data(season)

            # Train model
            self.my_model.fit(data)

            # Generate predictions
            predictions = self.my_model.predict(data)

            # Evaluate
            metrics = self.evaluate(predictions, data)
            results.append(metrics)

        return pd.DataFrame(results)

# Run custom backtest
backtest = MyBacktest(2020, 2024)
results = backtest.run()
print(results)
```

---

## ⚠️ Common Pitfalls

### 1. Look-Ahead Bias

**Bad:**
```python
# Using future data (WRONG!)
model.fit(all_data)  # Includes future games
predictions = model.predict(test_data)
```

**Good:**
```python
# Only use past data (CORRECT)
train_data = all_data[all_data['kickoff'] < test_date]
model.fit(train_data)
predictions = model.predict(test_data)
```

### 2. Ignoring Transaction Costs

Always account for:
- Sportsbook vig (typically -110 odds = 4.5% hold)
- Transaction fees
- Slippage (line movement)

### 3. Overfitting to Backtest Period

- Use out-of-sample testing
- Validate on multiple time periods
- Don't tune hyperparameters on test set

### 4. Unrealistic Bet Sizing

- Don't bet 100% Kelly (too aggressive)
- Use quarter-Kelly or smaller
- Account for correlation between bets

---

## 📊 Backtest Validation

### Statistical Significance Testing

```python
from scipy import stats

# Permutation test for ROI significance
def permutation_test(returns1, returns2, n_permutations=10000):
    """Test if ROI difference is statistically significant."""
    observed_diff = returns1.mean() - returns2.mean()

    permuted_diffs = []
    for _ in range(n_permutations):
        # Randomly shuffle labels
        combined = np.concatenate([returns1, returns2])
        np.random.shuffle(combined)

        perm_diff = combined[:len(returns1)].mean() - \
                    combined[len(returns1):].mean()
        permuted_diffs.append(perm_diff)

    # p-value
    p_value = (np.abs(permuted_diffs) >= np.abs(observed_diff)).mean()
    return p_value

# Example
p_val = permutation_test(v3_returns, v1_returns)
print(f"p-value: {p_val:.4f}")  # <0.05 = significant
```

### Robustness Checks

1. **Multiple time periods** - Test on 2020-2021, 2022-2023, 2024
2. **Different bet sizes** - Quarter-Kelly, half-Kelly, fixed fraction
3. **Edge thresholds** - 2%, 3%, 5% minimum edge
4. **Subgroup analysis** - By prop type, by team, by weather

---

## 🔗 Related Documentation

- **[PROJECT_STATUS.md](../../PROJECT_STATUS.md)** - Current backtest status
- **[docs/ADVANCED_BAYESIAN_V3.md](../../docs/ADVANCED_BAYESIAN_V3.md)** - Bayesian validation results
- **[nfl_analytics/models/README.md](../models/README.md)** - Model training
- **[nfl_analytics/ensemble/README.md](../ensemble/README.md)** - Ensemble validation

---

**Last Updated:** November 1, 2025
**Maintained by:** NFL Analytics Team
**Total Modules:** 12 Python files
