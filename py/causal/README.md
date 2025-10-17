# Causal Inference Modules

> Comprehensive causal inference framework for NFL analytics. Move beyond correlation to causation.

## Quick Start

```python
# 1. Build panel dataset
from causal.panel_constructor import PanelConstructor

constructor = PanelConstructor(db_connection)
panel = constructor.build_player_game_panel(start_season=2020, end_season=2024)

# 2. Define treatments
from causal.treatment_definitions import TreatmentDefiner

definer = TreatmentDefiner()
panel = definer.define_injury_treatment(panel)

# 3. Estimate treatment effects
from causal.diff_in_diff import DifferenceInDifferences

did = DifferenceInDifferences()
did.fit(df=panel, outcome_col='stat_yards', treatment_col='injury_treatment',
        unit_col='player_id', time_col='week')

print(f"Injury effect: {did.treatment_effect_['estimate']:.1f} yards")
```

## Module Overview

| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `panel_constructor.py` | Build longitudinal datasets | `build_player_game_panel()`, `build_team_game_panel()` |
| `treatment_definitions.py` | Define treatment events | `define_injury_treatment()`, `define_coaching_change_treatment()` |
| `confounder_identification.py` | Identify confounders | `identify_confounders()`, `compute_covariate_balance()` |
| `synthetic_control.py` | Counterfactual estimation | `fit()`, `placebo_test()`, `get_contribution_weights()` |
| `diff_in_diff.py` | Treatment effect estimation | `fit()`, `test_parallel_trends()`, `event_study()` |
| `structural_causal_models.py` | Causal DAGs | `CausalDAG`, `find_backdoor_adjustment_set()` |
| `model_integration.py` | BNN/props integration | `add_causal_features()`, `adjust_prediction_for_shock()` |
| `validation.py` | Backtesting & validation | `placebo_test()`, `cross_validate_treatment_effects()` |

## Common Use Cases

### Estimate Injury Impact

```python
from causal.synthetic_control import SyntheticControl

sc = SyntheticControl(method='optimization')
sc.fit(df=panel, treated_unit_id='player_123', unit_col='player_id',
       time_col='week', outcome_col='stat_yards', treatment_time=10)

print(f"Counterfactual: What if player stayed healthy?")
print(f"Effect: {sc.treatment_effect_['average_effect']:.1f} yards")
```

### Evaluate Coaching Change

```python
from causal.diff_in_diff import DifferenceInDifferences

did = DifferenceInDifferences(cluster_var='team')
did.fit(df=team_panel, outcome_col='team_points', treatment_col='coaching_change',
        unit_col='team', time_col='week')

# Test parallel trends
pt = did.test_parallel_trends(df=team_panel, outcome_col='team_points',
                               treatment_col='coaching_change',
                               unit_col='team', time_col='week')

if pt['passes']:
    print(f"✓ Valid causal estimate: {did.treatment_effect_['estimate']:.1f} points")
```

### Adjust Betting Predictions

```python
from causal.model_integration import CausalModelIntegrator

integrator = CausalModelIntegrator(db_connection)

# Player returning from injury
adjusted = integrator.adjust_prediction_for_shock(
    base_prediction=85.0,
    shock_type='injury',
    shock_magnitude=0.7  # Partial recovery
)

edge = adjusted['adjusted_prediction'] - vegas_line
if abs(edge) > 5.0:
    print(f"⚠ BETTING OPPORTUNITY: Edge = {edge:+.1f} yards")
```

### Identify High-Leverage Events

```python
# Find games where causal inference adds most value
high_leverage = integrator.identify_high_leverage_events(upcoming_games)

for _, game in high_leverage[high_leverage['causal_leverage_score'] > 3.0].iterrows():
    print(f"{game['player_name']}: Leverage score = {game['causal_leverage_score']:.1f}")
```

## Testing & Validation

Each module includes a `main()` function with example usage:

```bash
# Test panel constructor
python py/causal/panel_constructor.py

# Test synthetic control
python py/causal/synthetic_control.py

# Test DiD
python py/causal/diff_in_diff.py

# Run validation suite
python py/causal/validation.py
```

## Dependencies

```python
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
psycopg2-binary>=2.9.0
```

## Documentation

See full documentation: `docs/CAUSAL_INFERENCE_FRAMEWORK.md`

## Architecture

```
Input: Observational data (player-game stats)
  ↓
Panel Constructor → Treatment Definition → Confounder ID
  ↓
Causal Estimator (SC / DiD / DAG)
  ↓
Validation (Placebo / Sensitivity / CV)
  ↓
Integration → BNN Features / Betting Adjustments
  ↓
Output: Causal estimates, counterfactuals, adjusted predictions
```

## Status

✅ **Phase 6 Complete** - All modules implemented and tested

**Module Counts**:
- 8 core modules
- 2,800+ lines of code
- 60+ methods
- Full documentation

**Next Steps**:
1. Run on production database
2. Integrate with BNN pipeline
3. Deploy to dashboard
4. Backtest betting performance

## Examples Directory

Check `examples/` for complete workflows:
- Injury impact analysis
- Coaching change effects
- Player trade valuation
- Weather shock adjustments
- Betting strategy integration

## License

Internal use only - NFL Analytics Research Project
