# Phase 2.2 MoE BNN API Documentation

**Version:** 1.0
**Date:** October 17, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [MixtureExpertsBNN Class](#mixturexpertsbnn-class)
3. [BNNMoEProductionPipeline Class](#bnnmoeproductionpipeline-class)
4. [Data Formats](#data-formats)
5. [Examples](#examples)

---

## Overview

This document describes the public API for Phase 2.2 Mixture-of-Experts Bayesian Neural Network.

### Key Classes

1. **`MixtureExpertsBNN`** - Model training and inference
2. **`BNNMoEProductionPipeline`** - Production betting pipeline

### Import Paths

```python
from models.bnn_mixture_experts_v2 import MixtureExpertsBNN
from production.bnn_moe_production_pipeline import BNNMoEProductionPipeline
```

---

## MixtureExpertsBNN Class

### Constructor

```python
MixtureExpertsBNN(
    n_features: int,
    n_experts: int = 3,
    expert_hidden_size: int = 16
)
```

**Parameters:**
- `n_features` (int): Number of input features (default: 4)
- `n_experts` (int): Number of expert networks (default: 3)
- `expert_hidden_size` (int): Hidden layer size for each expert (default: 16)

**Returns:** `MixtureExpertsBNN` instance

**Example:**
```python
model = MixtureExpertsBNN(
    n_features=4,
    n_experts=3,
    expert_hidden_size=16
)
```

---

### load()

```python
@classmethod
MixtureExpertsBNN.load(filepath: str | Path) -> MixtureExpertsBNN
```

Load trained model from disk.

**Parameters:**
- `filepath` (str | Path): Path to saved model file (.pkl)

**Returns:** `MixtureExpertsBNN` instance with restored parameters

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `pickle.UnpicklingError`: If file is corrupted

**Example:**
```python
model = MixtureExpertsBNN.load('models/bayesian/bnn_mixture_experts_v2.pkl')
```

---

### build_model()

```python
def build_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> pm.Model
```

Construct PyMC model graph.

**Parameters:**
- `X_train` (np.ndarray): Training features, shape (N, D)
- `y_train` (np.ndarray): Training targets (log-transformed), shape (N,)

**Returns:** `pm.Model` - PyMC model object

**Side Effects:** Sets `self.model`

**Example:**
```python
X_train = np.random.randn(1000, 4)
y_train = np.random.randn(1000)
model_graph = model.build_model(X_train, y_train)
```

---

### train()

```python
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4
) -> az.InferenceData
```

Train model using NUTS MCMC sampling.

**Parameters:**
- `X_train` (np.ndarray): Training features (standardized), shape (N, D)
- `y_train` (np.ndarray): Training targets (log-transformed), shape (N,)
- `n_samples` (int): Posterior samples per chain (default: 2000)
- `n_tune` (int): Burn-in samples per chain (default: 1000)
- `n_chains` (int): Number of independent chains (default: 4)

**Returns:** `az.InferenceData` - Posterior samples trace

**Side Effects:**
- Sets `self.trace`
- Prints convergence diagnostics

**Training Time:** ~2.5 hours for default parameters

**Example:**
```python
trace = model.train(
    X_train,
    y_train,
    n_samples=2000,
    n_tune=1000,
    n_chains=4
)
```

---

### predict()

```python
def predict(X_test: np.ndarray) -> Dict[str, np.ndarray]
```

Generate predictions with uncertainty intervals.

**Parameters:**
- `X_test` (np.ndarray): Test features (standardized), shape (M, D)

**Returns:** `Dict[str, np.ndarray]` with keys:
- `'mean'`: Point estimates, shape (M,)
- `'std'`: Standard deviations, shape (M,)
- `'lower_90'`: 90% CI lower bound, shape (M,)
- `'upper_90'`: 90% CI upper bound, shape (M,)
- `'samples'`: Full posterior samples, shape (n_posterior, M)

**Prediction Time:** ~1-2 seconds per sample

**Example:**
```python
predictions = model.predict(X_test)
print(f"Mean: {predictions['mean']}")
print(f"90% CI: [{predictions['lower_90']}, {predictions['upper_90']}]")
```

---

### evaluate_calibration()

```python
def evaluate_calibration(
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]
```

Evaluate model calibration on test set.

**Parameters:**
- `X_test` (np.ndarray): Test features (standardized), shape (M, D)
- `y_test` (np.ndarray): Test targets (original scale), shape (M,)

**Returns:** `Dict[str, float]` with keys:
- `'mae'`: Mean Absolute Error (yards)
- `'rmse'`: Root Mean Squared Error (yards)
- `'coverage_90'`: 90% CI coverage percentage
- `'coverage_68'`: ±1σ coverage percentage
- `'target_90'`: Target 90% coverage (90.0)
- `'target_68'`: Target 68% coverage (68.0)

**Example:**
```python
metrics = model.evaluate_calibration(X_test, y_test)
print(f"MAE: {metrics['mae']:.2f} yards")
print(f"90% Coverage: {metrics['coverage_90']:.1f}%")
```

---

### save()

```python
def save(
    filepath: str | Path,
    X_train_shape: Tuple[int, int] = None,
    y_train_shape: Tuple[int,] = None
) -> None
```

Save model to disk.

**Parameters:**
- `filepath` (str | Path): Output path for model file
- `X_train_shape` (Tuple[int, int], optional): Training data shape
- `y_train_shape` (Tuple[int,], optional): Training target shape

**Saves:**
- Model trace (posterior samples)
- Model configuration
- Scaler parameters (mean, scale)
- Training data shapes

**Example:**
```python
model.save('models/bayesian/my_model.pkl')
```

---

## BNNMoEProductionPipeline Class

### Constructor

```python
BNNMoEProductionPipeline(
    model_path: str = None,
    db_config: Dict = None,
    bankroll: float = 10000.0,
    kelly_fraction: float = 0.15,
    max_bet_fraction: float = 0.03,
    min_edge: float = 0.03,
    min_confidence: float = 0.80
)
```

**Parameters:**
- `model_path` (str, optional): Path to trained model (default: auto-detect)
- `db_config` (Dict, optional): Database connection config
- `bankroll` (float): Current bankroll in dollars (default: $10,000)
- `kelly_fraction` (float): Fraction of Kelly to bet (default: 0.15 = 15%)
- `max_bet_fraction` (float): Max bet as fraction of bankroll (default: 0.03 = 3%)
- `min_edge` (float): Minimum edge required to bet (default: 0.03 = 3%)
- `min_confidence` (float): Minimum confidence threshold (default: 0.80 = 80%)

**Returns:** `BNNMoEProductionPipeline` instance

**Side Effects:** Loads model from disk (~2-3 seconds)

**Example:**
```python
pipeline = BNNMoEProductionPipeline(
    model_path='models/bayesian/bnn_mixture_experts_v2.pkl',
    bankroll=10000.0,
    kelly_fraction=0.15,
    min_edge=0.03
)
```

---

### predict_with_uncertainty()

```python
def predict_with_uncertainty(
    players_df: pd.DataFrame
) -> pd.DataFrame
```

Generate predictions with uncertainty for players.

**Parameters:**
- `players_df` (pd.DataFrame): Player features with columns:
  - `player_id` (str)
  - `player_name` (str)
  - `season` (int)
  - `week` (int)
  - `carries` (int)
  - `avg_rushing_l3` (float)
  - `season_avg` (float)

**Returns:** `pd.DataFrame` with additional columns:
- `pred_mean` (float): Predicted rushing yards
- `pred_std` (float): Standard deviation
- `pred_lower_90` (float): 90% CI lower bound
- `pred_upper_90` (float): 90% CI upper bound
- `pred_interval_width` (float): Width of 90% CI
- `pred_cv` (float): Coefficient of variation

**Example:**
```python
players = pd.DataFrame({
    'player_id': ['00-0034844'],
    'player_name': ['Saquon Barkley'],
    'season': [2025],
    'week': [11],
    'carries': [20],
    'avg_rushing_l3': [85.0],
    'season_avg': [95.0]
})

predictions = pipeline.predict_with_uncertainty(players)
print(predictions[['player_name', 'pred_mean', 'pred_std']])
```

---

### select_bets()

```python
def select_bets(
    predictions_df: pd.DataFrame,
    prop_lines_df: pd.DataFrame
) -> pd.DataFrame
```

Select profitable bets using Kelly criterion.

**Parameters:**
- `predictions_df` (pd.DataFrame): Predictions from `predict_with_uncertainty()`
- `prop_lines_df` (pd.DataFrame): Prop lines with columns:
  - `player_id` (str)
  - `rushing_line` (float): O/U line (yards)
  - `over_odds` (int): American odds for OVER
  - `under_odds` (int): American odds for UNDER
  - `over_implied_prob` (float): Implied probability for OVER
  - `under_implied_prob` (float): Implied probability for UNDER

**Returns:** `pd.DataFrame` with bet recommendations:
- `player_id` (str)
- `player_name` (str)
- `rushing_line` (float)
- `bet_side` (str): 'over' or 'under'
- `prediction_mean` (float)
- `prediction_std` (float)
- `prob_win` (float): Win probability
- `edge` (float): Edge over market (0-1)
- `confidence` (float): Confidence score (0-1)
- `odds` (int): American odds
- `bet_amount` (float): Recommended bet size ($)
- `expected_value` (float): Expected profit ($)

**Filtering:**
- Only bets with `edge >= min_edge`
- Only bets with `confidence >= min_confidence`
- Only bets with `bet_amount >= $10`

**Example:**
```python
bets = pipeline.select_bets(predictions_df, prop_lines_df)
print(bets[['player_name', 'bet_side', 'edge', 'bet_amount']])
```

---

### run_pipeline()

```python
def run_pipeline(
    season: int,
    week: int = None,
    player_ids: List[str] = None
) -> Dict
```

Run complete end-to-end pipeline.

**Parameters:**
- `season` (int): NFL season (e.g., 2025)
- `week` (int, optional): Week number (None = all weeks)
- `player_ids` (List[str], optional): Specific players to analyze

**Returns:** `Dict` with keys:
- `'success'` (bool): Whether pipeline succeeded
- `'predictions'` (int): Number of predictions generated
- `'bets'` (int): Number of bets selected
- `'output_files'` (Dict[str, Path]): Paths to output files
  - `'predictions'`: CSV with all predictions
  - `'bets'`: CSV with bet recommendations
  - `'summary'`: TXT with human-readable summary
  - `'stats'`: JSON with performance stats
- `'error'` (str, optional): Error message if failed

**Side Effects:**
- Queries database
- Saves output files to `output/bnn_moe_recommendations/`
- Updates `self.performance` metrics

**Example:**
```python
result = pipeline.run_pipeline(season=2025, week=11)

if result['success']:
    print(f"✅ {result['predictions']} predictions, {result['bets']} bets")
    print(f"Output: {result['output_files']['bets']}")
else:
    print(f"❌ Error: {result['error']}")
```

---

## Data Formats

### Input: Player Features DataFrame

```python
pd.DataFrame({
    'player_id': str,          # NFL player ID
    'player_name': str,        # Player name
    'season': int,             # NFL season (YYYY)
    'week': int,               # Week number (1-18)
    'team': str,               # Team abbreviation
    'carries': int,            # Rushing attempts (5-30)
    'avg_rushing_l3': float,   # 3-game rolling average (yards)
    'season_avg': float,       # Season-to-date average (yards)
})
```

### Output: Predictions DataFrame

```python
pd.DataFrame({
    # Input fields (copied)
    'player_id': str,
    'player_name': str,
    'season': int,
    'week': int,
    'team': str,
    'carries': int,
    'avg_rushing_l3': float,
    'season_avg': float,

    # Predictions
    'pred_mean': float,          # Point estimate (yards)
    'pred_std': float,           # Standard deviation (yards)
    'pred_lower_90': float,      # 90% CI lower (yards)
    'pred_upper_90': float,      # 90% CI upper (yards)
    'pred_interval_width': float, # CI width (yards)
    'pred_cv': float,            # Coefficient of variation (std/mean)
})
```

### Output: Bet Recommendations DataFrame

```python
pd.DataFrame({
    'player_id': str,
    'player_name': str,
    'team': str,
    'rushing_line': float,       # O/U line (yards)
    'bet_side': str,             # 'over' or 'under'

    # Predictions
    'prediction_mean': float,
    'prediction_std': float,
    'prediction_lower_90': float,
    'prediction_upper_90': float,

    # Betting metrics
    'prob_win': float,           # Win probability (0-1)
    'market_prob': float,        # Market implied probability (0-1)
    'edge': float,               # Edge over market (0-1)
    'confidence': float,         # Confidence score (0-1)
    'odds': int,                 # American odds (e.g., -110)
    'bookmaker': str,            # Sportsbook name

    # Bet sizing
    'bet_fraction': float,       # Fraction of bankroll (0-1)
    'bet_amount': float,         # Recommended bet ($)
    'expected_value': float,     # Expected profit ($)

    'game_time': datetime,       # Game start time
})
```

---

## Examples

### Example 1: Load Model and Make Predictions

```python
import numpy as np
from models.bnn_mixture_experts_v2 import MixtureExpertsBNN

# Load trained model
model = MixtureExpertsBNN.load('models/bayesian/bnn_mixture_experts_v2.pkl')

# Prepare features (standardized)
X_test = model.scaler.transform(np.array([
    [20, 85.0, 95.0, 11]  # carries, avg_l3, season_avg, week
]))

# Generate predictions
predictions = model.predict(X_test)

print(f"Prediction: {predictions['mean'][0]:.1f} ± {predictions['std'][0]:.1f} yards")
print(f"90% CI: [{predictions['lower_90'][0]:.1f}, {predictions['upper_90'][0]:.1f}]")
```

**Output:**
```
Prediction: 82.3 ± 28.1 yards
90% CI: [38.2, 142.1]
```

---

### Example 2: Run Full Production Pipeline

```python
from production.bnn_moe_production_pipeline import BNNMoEProductionPipeline

# Initialize pipeline
pipeline = BNNMoEProductionPipeline(
    model_path='models/bayesian/bnn_mixture_experts_v2.pkl',
    bankroll=10000.0,
    kelly_fraction=0.15,
    min_edge=0.03,
    min_confidence=0.80
)

# Run for upcoming week
result = pipeline.run_pipeline(season=2025, week=11)

if result['success']:
    print(f"✅ Generated {result['predictions']} predictions")
    print(f"✅ Selected {result['bets']} bets")

    # Load bet recommendations
    import pandas as pd
    bets = pd.read_csv(result['output_files']['bets'])

    # Show top 3 bets
    top_3 = bets.nlargest(3, 'edge')
    for _, bet in top_3.iterrows():
        print(f"\n{bet['player_name']}: {bet['bet_side'].upper()} {bet['rushing_line']}")
        print(f"  Prediction: {bet['prediction_mean']:.1f} yards")
        print(f"  Edge: {bet['edge']*100:.2f}% | Bet: ${bet['bet_amount']:.2f}")
```

---

### Example 3: Custom Risk Configuration

```python
# Ultra-conservative settings
pipeline_conservative = BNNMoEProductionPipeline(
    bankroll=10000.0,
    kelly_fraction=0.10,    # 10% Kelly (very conservative)
    max_bet_fraction=0.02,  # Max 2% per bet
    min_edge=0.05,          # Require 5% edge
    min_confidence=0.90     # Require 90% confidence
)

# Aggressive settings
pipeline_aggressive = BNNMoEProductionPipeline(
    bankroll=10000.0,
    kelly_fraction=0.25,    # 25% Kelly (aggressive)
    max_bet_fraction=0.05,  # Max 5% per bet
    min_edge=0.02,          # Accept 2% edge
    min_confidence=0.70     # Accept 70% confidence
)
```

---

### Example 4: Batch Predictions

```python
import pandas as pd

# Prepare batch of players
players = pd.DataFrame({
    'player_id': ['00-0034844', '00-0032764', '00-0038555'],
    'player_name': ['Saquon Barkley', 'Derrick Henry', 'Tank Bigsby'],
    'season': [2025, 2025, 2025],
    'week': [11, 11, 11],
    'team': ['PHI', 'BAL', 'JAX'],
    'carries': [20, 22, 18],
    'avg_rushing_l3': [85.0, 90.0, 65.0],
    'season_avg': [95.0, 88.0, 72.0]
})

# Generate predictions
predictions = pipeline.predict_with_uncertainty(players)

# Show results
print(predictions[['player_name', 'pred_mean', 'pred_std', 'pred_lower_90', 'pred_upper_90']])
```

**Output:**
```
     player_name  pred_mean  pred_std  pred_lower_90  pred_upper_90
0  Saquon Barkley       82.3      28.1           38.2          142.1
1   Derrick Henry       88.5      30.2           42.1          152.3
2     Tank Bigsby       68.2      25.4           33.5          118.6
```

---

### Example 5: Evaluate Calibration

```python
# Load test data
X_test, y_test = load_test_data()  # Your data loading function

# Evaluate calibration
metrics = model.evaluate_calibration(X_test, y_test)

# Display results
print("Calibration Metrics:")
print(f"  MAE: {metrics['mae']:.2f} yards")
print(f"  RMSE: {metrics['rmse']:.2f} yards")
print(f"  90% Coverage: {metrics['coverage_90']:.1f}% (target: {metrics['target_90']:.0f}%)")
print(f"  ±1σ Coverage: {metrics['coverage_68']:.1f}% (target: {metrics['target_68']:.0f}%)")

# Check if well-calibrated
if 85 <= metrics['coverage_90'] <= 95:
    print("\n✓ Model is well-calibrated!")
else:
    print("\n⚠️ Model may need recalibration")
```

---

## Error Handling

### Common Exceptions

```python
# FileNotFoundError
try:
    model = MixtureExpertsBNN.load('nonexistent.pkl')
except FileNotFoundError as e:
    print(f"Model file not found: {e}")

# Database connection error
try:
    result = pipeline.run_pipeline(season=2025, week=11)
except psycopg2.OperationalError as e:
    print(f"Database connection failed: {e}")

# Invalid predictions
try:
    predictions = model.predict(X_test)
except ValueError as e:
    print(f"Prediction error (check input shape): {e}")
```

---

## Performance Considerations

### Prediction Latency

| Operation | Latency | Notes |
|-----------|---------|-------|
| Model load | 2-3 sec | One-time cost |
| Single prediction | 1-2 sec | 8,000 posterior samples |
| Batch (30 players) | 30-60 sec | Sequential processing |

**Optimization:** Use batch processing for multiple predictions

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Model loaded | 1.3 GB | Full posterior trace |
| During prediction | +200 MB | Temporary arrays |
| Peak (30 players) | ~2 GB | Scales with batch size |

**Recommendation:** Use machines with 4+ GB RAM

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 17, 2025 | Initial production release |

---

## See Also

- [Technical Documentation](./PHASE_2_2_TECHNICAL_DOCUMENTATION.md)
- [Deployment Guide](./PHASE_2_2_DEPLOYMENT_GUIDE.md)
- [Production Summary](../PHASE_2.2_PRODUCTION_SUMMARY.md)

---

**Document Version:** 1.0
**Last Updated:** October 17, 2025
**Status:** Production-Ready ✓
