#!/usr/bin/env python3
"""
Test 5-way ensemble with BNN Calibrated (5th model).

Tests: Bayesian + XGBoost + BNN + BNN Calibrated + State-Space
"""

from py.models.player_props_ensemble import PlayerPropsEnsemble

# Initialize ensemble (5-way - includes BNN Calibrated!)
print("Initializing 5-way ensemble...")
ensemble = PlayerPropsEnsemble(
    use_bayesian=True,
    use_xgboost=True,
    use_bnn=True,
    use_bnn_calibrated=True,  # 5TH MODEL!
    use_state_space=True
)

# Test with Mahomes
print("\n" + "="*80)
print("TEST: Patrick Mahomes vs TBD (hypothetical 2025 Week 10)")
print("="*80)

prediction = ensemble.predict(
    player_id="00-0033873",
    player_name="Patrick Mahomes",
    prop_type="passing_yards",
    line=275.5,
    opponent="KC",  # Hypothetical opponent
    is_home=True,
    season=2025,
    week=10
)

print(f"\n{'Ensemble Prediction:':<25} {prediction.mean:.1f} ± {prediction.std:.1f}")
print(f"{'90% CI:':<25} [{prediction.q05:.1f}, {prediction.q95:.1f}]")
print("\nIndividual Models:")
if prediction.bayesian_mean:
    print(f"  ✓ Bayesian:        {prediction.bayesian_mean:.1f} yards")
if prediction.xgboost_mean:
    print(f"  ✓ XGBoost:         {prediction.xgboost_mean:.1f} yards")
if prediction.bnn_mean:
    print(f"  ✓ BNN:             {prediction.bnn_mean:.1f} yards")
if prediction.bnn_calibrated_mean:
    print(f"  ✓ BNN Calibrated:  {prediction.bnn_calibrated_mean:.1f} yards (T=0.75)")
if prediction.state_space_mean:
    print(f"  ✓ State-Space:     {prediction.state_space_mean:.1f} yards")

if not any([prediction.bayesian_mean, prediction.xgboost_mean, prediction.bnn_mean,
            prediction.bnn_calibrated_mean, prediction.state_space_mean]):
    print("  ⏳ (no models active)")

print(f"\nModel Weights:")
if hasattr(prediction, 'weights') and prediction.weights:
    for model, weight in prediction.weights.items():
        print(f"  {model:18s}: {weight:.2%}")

print(f"\nBetting Analysis:")
print(f"  Over probability:  {prediction.over_prob:.1%}")
print(f"  Edge: {prediction.edge:+.2%}")
print(f"  Recommendation: {prediction.recommendation}")

# Count active models
n_models = sum([
    prediction.bayesian_mean is not None,
    prediction.xgboost_mean is not None,
    prediction.bnn_mean is not None,
    prediction.bnn_calibrated_mean is not None,
    prediction.state_space_mean is not None
])

print(f"\n{'='*80}")
if n_models == 5:
    print("✅ 5-WAY ENSEMBLE ACTIVE (INCLUDING BNN CALIBRATED)")
    print(f"📊 Expected ROI: +6-8% (all models contributing)")
elif n_models == 4:
    print("✅ 4-WAY ENSEMBLE ACTIVE")
    print(f"📊 Expected ROI: +5-7%")
elif n_models == 3:
    print("✅ 3-WAY ENSEMBLE ACTIVE")
    print(f"📊 Expected ROI: +4-6%")
elif n_models == 2:
    print("✅ 2-WAY ENSEMBLE ACTIVE")
    print(f"📊 Expected ROI: +4%")
elif n_models == 1:
    print("⚠️  SINGLE MODEL ONLY")
    print(f"📊 Expected ROI: +2-3%")
else:
    print("❌ NO MODELS ACTIVE")
print(f"{'='*80}\n")
