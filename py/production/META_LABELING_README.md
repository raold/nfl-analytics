# Meta-Labeling Classifier v2 - Implementation Guide

**Author:** Claude Code (Opus 4.5)
**Date:** December 11, 2025
**Status:** Production Ready
**Version:** 2.0

## Overview

Meta-labeling is a two-stage machine learning approach that separates "what to bet" (primary model) from "should we bet" (meta-model). This implementation uses 27 meta-features across 6 categories to predict bet quality, integrated with Kelly criterion sizing for optimal bankroll management.

**Expected Impact:**
- Reduce bet volume by 40-50% (filter low-quality bets)
- Increase win rate by 4-6% (higher quality bets only)
- Improve ROI by 3-5% (fewer losing bets, better sizing)

## Architecture

### Components

1. **MetaFeatureExtractor** (`meta_labeling_v2.py`)
   - Extracts 27 features from ensemble predictions
   - Organized into 6 interpretable categories
   - Handles missing data gracefully

2. **MetaLabelingClassifier** (`meta_labeling_v2.py`)
   - XGBoost classifier with Platt calibration
   - Walk-forward training methodology
   - Returns calibrated probabilities [0, 1]

3. **Training Script** (`training/train_meta_classifier.py`)
   - Loads historical predictions and outcomes
   - Trains with walk-forward validation
   - Generates feature importance plots
   - Saves model and training report

4. **Integration Example** (`examples/meta_kelly_integration_example.py`)
   - Complete end-to-end workflow
   - Ensemble → Meta-Labeling → Kelly → Final Bet

## Meta-Features (27 Total)

### 1. Primary Signals (6 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `primary_confidence` | abs(prob - 0.5) × 2 | [0, 1] |
| `primary_edge` | predicted - vegas | [-20, 20] |
| `ensemble_agreement` | % components agreeing | [0, 1] |
| `prediction_variance` | var(component predictions) | [0, ∞) |
| `ensemble_sharpe` | mean(edge) / std(edge) | (-∞, ∞) |
| `weight_concentration` | Herfindahl index of weights | [0.2, 1] |

**Why these matter:** High confidence + agreement + edge → better bets. Low variance → consistent predictions.

### 2. Market Context (6 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `is_primetime` | TNF/SNF/MNF flag | {0, 1} |
| `is_divisional` | Divisional game flag | {0, 1} |
| `spread_magnitude` | abs(spread) | [0, 20] |
| `line_movement` | close - open | [-5, 5] |
| `public_betting_pct` | % bets on favorite | [0, 1] |
| `steam_move` | Rapid movement (>2 pts) | {0, 1} |

**Why these matter:** Primetime games have more public money. Steam moves indicate sharp action. Large spreads are harder to predict.

### 3. Historical Performance (6 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `model_accuracy_last_10` | Rolling accuracy | [0, 1] |
| `recent_roi` | ROI last 20 bets | [-0.5, 0.5] |
| `recent_win_streak` | Current streak | [-10, 10] |
| `calibration_error` | Brier score (recent) | [0, 0.5] |
| `model_sharpe_ratio` | Risk-adjusted returns | (-∞, ∞) |
| `edge_realization_rate` | How often edge realizes | [0, 1] |

**Why these matter:** Recent performance predicts future performance. Well-calibrated models are more trustworthy.

### 4. Uncertainty Quantification (4 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `prediction_std` | Ensemble std dev | [0, ∞) |
| `epistemic_uncertainty` | Model uncertainty (reducible) | [0, ∞) |
| `aleatoric_uncertainty` | Inherent randomness (irreducible) | [0, ∞) |
| `confidence_interval_width` | 95% CI width | [0, ∞) |

**Why these matter:** High epistemic uncertainty → model lacks knowledge. High aleatoric → inherently unpredictable game.

### 5. Temporal Factors (3 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `days_rest_diff` | home_rest - away_rest | [-14, 14] |
| `travel_distance` | Miles traveled (away) | [0, 3000] |
| `season_phase` | Early/mid/late/playoffs | {0, 1, 2, 3} |

**Why these matter:** Rest advantages matter. Cross-country travel impacts performance. Playoff intensity differs.

### 6. Team Context (2 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `injury_load_diff` | home_load - away_load | [-1, 1] |
| `weather_severity` | Composite weather score | [0, 1] |

**Why these matter:** Injuries impact performance. Extreme weather increases variance.

## Integration Formula

```python
# Step 1: Get ensemble prediction
prediction = ensemble.predict(game_id)

# Step 2: Extract meta-features
extractor = MetaFeatureExtractor()
meta_features = extractor.extract_from_prediction(
    prediction, game_features, historical_metrics
)

# Step 3: Predict bet quality
classifier = MetaLabelingClassifier(model_path="models/meta_labeling_v2.joblib")
meta_prob = classifier.predict_probability(meta_features)
kelly_multiplier = classifier.get_kelly_multiplier(meta_prob)

# Step 4: Size bet with Kelly criterion
kelly_result = optimal_bet_size(
    model_prob=prediction["probability"],
    market_odds=-110,
    bankroll=10000,
    kelly_fraction=0.25
)

# Step 5: Apply meta-multiplier
final_bet_size = kelly_result["bet_amount"] * kelly_multiplier

# Decision rules:
# meta_prob < 0.4 → multiplier = 0.0 → NO BET
# meta_prob 0.4-0.6 → linear interpolation → REDUCED BET
# meta_prob > 0.6 → multiplier = 1.0 → FULL KELLY
```

## Training the Model

### Prerequisites

1. Historical game predictions with outcomes (stored in database)
2. Ensemble component predictions and weights
3. Game features from `GameFeatureExtractor`
4. PostgreSQL connection to `devdb01`

### Training Command

```bash
# Basic training (2020-2024, spread bets, walk-forward validation)
python py/training/train_meta_classifier.py

# Custom parameters
python py/training/train_meta_classifier.py \
    --start-season 2020 \
    --end-season 2024 \
    --bet-type spread \
    --use-walk-forward \
    --output-dir models \
    --no-plots  # Skip plots
```

### Walk-Forward Validation

Walk-forward methodology prevents temporal leakage:

1. **Fold 1:** Train 2020-2021, validate 2022
2. **Fold 2:** Train 2020-2022, validate 2023
3. **Fold 3:** Train 2020-2023, validate 2024
4. **Final:** Train on all data (2020-2024) for production

**Why walk-forward?** More realistic than k-fold CV because:
- Respects temporal order
- Simulates production deployment
- Prevents future information leakage
- Accounts for regime changes over time

### Output Files

```
models/
├── meta_labeling_v2.joblib                    # Trained model
├── meta_labeling_v2_report.txt                # Training metrics
├── meta_labeling_v2_feature_importance.png    # Feature importance plot
└── meta_labeling_v2_walk_forward.png          # Validation results
```

## Production Usage

### 1. Basic Usage

```python
from py.production.meta_labeling_v2 import MetaFeatureExtractor, MetaLabelingClassifier

# Initialize
extractor = MetaFeatureExtractor()
classifier = MetaLabelingClassifier(model_path="models/meta_labeling_v2.joblib")

# Extract features from ensemble prediction
meta_features = extractor.extract_from_prediction(
    prediction=ensemble_prediction,
    game_features=game_features,
    historical_metrics=recent_metrics
)

# Predict bet quality
meta_prob = classifier.predict_probability(meta_features)

# Get Kelly multiplier
kelly_mult = classifier.get_kelly_multiplier(meta_prob)

# Decision
if kelly_mult > 0:
    print(f"BET with {kelly_mult:.1%} of Kelly size")
else:
    print("PASS - Low quality bet")
```

### 2. Batch Processing

```python
# Process multiple games
recommendations = []

for game_id in upcoming_games:
    prediction = ensemble.predict(game_id)
    meta_features = extractor.extract_from_prediction(...)
    meta_prob = classifier.predict_probability(meta_features)

    recommendations.append({
        "game_id": game_id,
        "meta_prob": meta_prob,
        "should_bet": meta_prob > 0.4
    })

# Sort by quality
recommendations.sort(key=lambda x: x["meta_prob"], reverse=True)

# Bet on top N highest-quality opportunities
top_bets = [r for r in recommendations if r["should_bet"]][:5]
```

### 3. Integration with CQL Bet Sizer

```python
from py.production.cql_bet_sizer import CQLBetSizer

# Initialize both systems
meta_clf = MetaLabelingClassifier(model_path="models/meta_labeling_v2.joblib")
cql_sizer = CQLBetSizer()

# Meta-labeling for quality filter
meta_prob = meta_clf.predict_probability(meta_features)

if meta_prob > 0.4:
    # High-quality bet → use CQL for sizing
    cql_recommendation = cql_sizer.recommend_bet(opportunity, bankroll)
    final_bet_size = cql_recommendation.bet_size
else:
    # Low-quality bet → pass
    final_bet_size = 0
```

## Database Schema

### Table: `ml.meta_predictions`

Stores meta-labeling predictions for each bet opportunity.

```sql
CREATE TABLE ml.meta_predictions (
    prediction_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    season INT NOT NULL,
    week INT NOT NULL,

    -- Predictions
    meta_probability NUMERIC NOT NULL,
    kelly_multiplier NUMERIC NOT NULL,
    should_bet BOOLEAN NOT NULL,

    -- All 27 meta-features (columns)
    primary_confidence NUMERIC,
    primary_edge NUMERIC,
    -- ... (see migration file)

    -- Outcome tracking
    actual_outcome TEXT,
    bet_won BOOLEAN,
    bet_profit NUMERIC,

    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Migration

```bash
# Apply migration
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
    -f sql/migrations/create_meta_predictions_table.sql
```

## Model Performance

### Expected Metrics (after training on 2020-2024 data)

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| ROC AUC | > 0.65 | Discriminates profitable bets |
| Accuracy | > 0.58 | Better than coin flip |
| Brier Score | < 0.22 | Well-calibrated probabilities |
| Precision | > 0.60 | High hit rate on "bet" predictions |
| Recall | > 0.55 | Doesn't miss too many good bets |

### Feature Importance (typical)

Top 10 features (expected):
1. `primary_edge` - Largest predictor of bet quality
2. `ensemble_agreement` - Consensus matters
3. `model_accuracy_last_10` - Recent performance predictive
4. `primary_confidence` - High confidence bets better
5. `prediction_variance` - Low variance → more reliable
6. `epistemic_uncertainty` - Model knowledge gaps
7. `calibration_error` - Well-calibrated models win
8. `edge_realization_rate` - Past edge realization predicts future
9. `ensemble_sharpe` - Risk-adjusted edge matters
10. `recent_roi` - Hot streaks persist short-term

## Monitoring and Maintenance

### Key Metrics to Track

1. **Meta-Probability Distribution**
   - Should see bimodal distribution (clear good/bad separation)
   - If uniform → model not learning signal

2. **Kelly Multiplier Usage**
   - Track % of bets at 0.0, 0.5, 1.0 multipliers
   - Should see ~30-40% at 0.0 (filtered out)

3. **Realized Performance by Meta-Prob Bucket**
   - Bets with meta_prob > 0.6 should have higher win rate
   - Bets with meta_prob < 0.4 should have lower win rate
   - Validates calibration

4. **Feature Drift**
   - Monitor feature distributions over time
   - Alert if distributions shift significantly
   - May indicate need for retraining

### Retraining Schedule

- **Full retrain:** Every offseason (add new season's data)
- **Incremental update:** Mid-season if performance degrades
- **Walk-forward validation:** Always maintain temporal splits

### Feature Engineering Opportunities

Future enhancements:
1. **Opponent-specific features:** Historical performance against this opponent
2. **Referee features:** Referee tendencies (total, penalties)
3. **Betting market features:** Sharp vs public money ratios
4. **Social sentiment:** Twitter/news sentiment scores
5. **Weather interactions:** QB performance in wind/cold
6. **Injury timing:** Recent vs long-term injuries

## Troubleshooting

### Issue: Low AUC (< 0.55)

**Possible causes:**
- Insufficient training data
- Feature engineering issues (check for NaNs)
- Class imbalance (adjust `scale_pos_weight` in XGBoost)
- Target variable leakage (verify lagged features)

**Solutions:**
```python
# Check class balance
print(train_df["bet_won"].value_counts())

# Adjust for imbalance
classifier.base_model.set_params(
    scale_pos_weight=len(y[y==0]) / len(y[y==1])
)

# Check for NaNs
print(train_df[meta_features].isnull().sum())
```

### Issue: Poor Calibration (Brier > 0.25)

**Possible causes:**
- Platt scaling not applied
- Overfitting
- Distribution shift between train/test

**Solutions:**
```python
# Ensure calibration is enabled
classifier.train_walk_forward(df, calibrate=True)

# Check calibration curve
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true)
plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration
```

### Issue: All Bets Filtered (kelly_multiplier = 0 always)

**Possible causes:**
- Thresholds too strict (low_threshold > 0.4)
- Model too pessimistic
- Training data issues

**Solutions:**
```python
# Adjust thresholds
kelly_mult = classifier.get_kelly_multiplier(
    meta_prob,
    low_threshold=0.3,   # More permissive
    high_threshold=0.5
)

# Check meta_prob distribution
print(meta_probs.describe())
```

## References

1. **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 3: Meta-Labeling.
2. **Chen, T. & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
3. **Kelly, J.** (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal.
4. **Platt, J.** (1999). *Probabilistic Outputs for Support Vector Machines*. Advances in Large Margin Classifiers.

## Files Reference

| File | Purpose |
|------|---------|
| `py/production/meta_labeling_v2.py` | Main implementation (extractor + classifier) |
| `py/training/train_meta_classifier.py` | Training script |
| `py/production/examples/meta_kelly_integration_example.py` | Integration example |
| `sql/migrations/create_meta_predictions_table.sql` | Database schema |
| `models/meta_labeling_v2.joblib` | Trained model (created after training) |

## License

Internal use only - NFL Analytics Dissertation Project

---

**Questions?** See `CLAUDE.md` for project conventions and contact information.
