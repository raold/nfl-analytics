#!/bin/bash
#
# Ensemble Orchestrator - Manage 5-way ensemble (Bayesian + XGBoost + BNN + State-Space + CQL)
#
# Usage:
#   ./scripts/orchestrate_ensemble.sh --train                  # Train all models
#   ./scripts/orchestrate_ensemble.sh --optimize               # Optimize ensemble weights
#   ./scripts/orchestrate_ensemble.sh --status                 # Get ensemble status
#   ./scripts/orchestrate_ensemble.sh --predict FEATURES.json  # Generate prediction
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" == "--train" ]; then
    TRAINING_WINDOW="${2:-2018-2024}"

    echo "=== Training 5-Way Ensemble ==="
    echo "Training window: $TRAINING_WINDOW"
    echo ""
    echo "Models:"
    echo "  1. Bayesian Hierarchical (brms/Stan)"
    echo "  2. XGBoost (Python)"
    echo "  3. Bayesian Neural Network (PyMC/PyTorch)"
    echo "  4. State-Space Model (Stan)"
    echo "  5. CQL Reinforcement Learning (PyTorch)"
    echo ""

    python3 -c "
from py.utils.ensemble_orchestrator import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator()

print('Starting training...')
print()

# Train all models
results = orchestrator.train_all_models(training_window='$TRAINING_WINDOW')

# Print results
successful = sum(results.values())
total = len(results)

print()
print('=' * 50)
print(f'Training complete: {successful}/{total} models succeeded')
print('=' * 50)
print()

for model_type, success in results.items():
    status = '✅' if success else '❌'
    print(f'{status} {model_type.value}')

# Save state
orchestrator.save_state()

print()
if successful == total:
    print('✅ All models trained successfully')
    print('Next: ./scripts/orchestrate_ensemble.sh --optimize')
else:
    print('⚠️  Some models failed - check logs above')
    exit(1)
"

elif [ "$1" == "--optimize" ]; then
    echo "=== Optimizing Ensemble Weights ==="
    echo ""

    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Error: Please provide predictions and actuals data"
        echo "Usage: ./scripts/orchestrate_ensemble.sh --optimize PREDICTIONS.pkl ACTUALS.pkl"
        echo ""
        echo "Or use validation data:"
        echo "  python3 -c 'from py.utils.ensemble_orchestrator import EnsembleOrchestrator; ...'"
        exit 1
    fi

    PREDICTIONS_FILE="$2"
    ACTUALS_FILE="$3"

    python3 -c "
import pickle
from py.utils.ensemble_orchestrator import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator()

# Load predictions and actuals
with open('$PREDICTIONS_FILE', 'rb') as f:
    predictions = pickle.load(f)

with open('$ACTUALS_FILE', 'rb') as f:
    actuals = pickle.load(f)

print('Optimizing ensemble weights using CVX...')
print()

# Optimize weights
weights = orchestrator.optimize_weights(predictions, actuals)

print('Optimized weights:')
for model_type, weight in weights.items():
    print(f'  {model_type.value}: {weight:.4f}')

# Save state with optimized weights
orchestrator.save_state()

print()
print('✅ Weight optimization complete')
print('   Weights saved to models/ensemble_state.json')
"

elif [ "$1" == "--status" ]; then
    echo "=== Ensemble Status ==="
    echo ""

    python3 -c "
import json
from py.utils.ensemble_orchestrator import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator()

# Get performance metrics
metrics = orchestrator.get_performance_metrics()

print(f'Timestamp: {metrics[\"timestamp\"]}')
print()
print('Models:')

for model_name, model_info in metrics['models'].items():
    print(f'  {model_name}:')
    print(f'    Status: {model_info[\"status\"]}')
    print(f'    Weight: {model_info[\"weight\"]:.4f}')
    print(f'    Trained: {model_info[\"trained_date\"]}')

    if model_info['metrics']:
        print(f'    Metrics: {model_info[\"metrics\"]}')

    print()

# Save status to file
with open('ensemble_status.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('✅ Status saved to ensemble_status.json')
"

elif [ "$1" == "--predict" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide features file"
        echo "Usage: ./scripts/orchestrate_ensemble.sh --predict FEATURES.json"
        exit 1
    fi

    FEATURES_FILE="$2"

    if [ ! -f "$FEATURES_FILE" ]; then
        echo "Error: Features file not found: $FEATURES_FILE"
        exit 1
    fi

    echo "=== Generating Ensemble Prediction ==="
    echo "Features: $FEATURES_FILE"
    echo ""

    python3 -c "
import json
from py.utils.ensemble_orchestrator import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator()

# Load features
with open('$FEATURES_FILE', 'r') as f:
    features = json.load(f)

# Generate prediction
prediction = orchestrator.predict(features, use_weights=True)

print('Ensemble Prediction:')
print(f'  Combined: {prediction[\"ensemble_prediction\"]:.2f}')
print()
print('Individual Predictions:')
for model_name, pred in prediction['individual_predictions'].items():
    weight = prediction['weights'][model_name]
    print(f'  {model_name}: {pred:.2f} (weight: {weight:.4f})')

print()
print('✅ Prediction complete')
"

else
    echo "=== Ensemble Orchestrator ==="
    echo ""
    echo "5-Way Ensemble:"
    echo "  1. Bayesian Hierarchical (brms/Stan)"
    echo "  2. XGBoost (Python)"
    echo "  3. Bayesian Neural Network (PyMC/PyTorch)"
    echo "  4. State-Space Model (Stan)"
    echo "  5. CQL Reinforcement Learning (PyTorch)"
    echo ""
    echo "Usage:"
    echo "  ./scripts/orchestrate_ensemble.sh --train [WINDOW]    # Train all models"
    echo "  ./scripts/orchestrate_ensemble.sh --optimize PRED ACT # Optimize weights"
    echo "  ./scripts/orchestrate_ensemble.sh --status            # Get status"
    echo "  ./scripts/orchestrate_ensemble.sh --predict FEATURES  # Predict"
    echo ""
    echo "Examples:"
    echo "  ./scripts/orchestrate_ensemble.sh --train 2018-2024"
    echo "  ./scripts/orchestrate_ensemble.sh --status"
    echo "  ./scripts/orchestrate_ensemble.sh --predict game_features.json"
    echo ""
    echo "Targets:"
    echo "  - ROI: >5%"
    echo "  - Win Rate: >59%"
    echo "  - Sharpe Ratio: >2.2"
    echo "  - Max Drawdown: <-8%"
    exit 1
fi

echo ""
echo "✅ Complete"
