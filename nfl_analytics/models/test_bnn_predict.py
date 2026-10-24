"""
Quick test to verify the broadcast shape bug is fixed
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from bnn_simpler_v2 import SimplerBNNv2

print("Testing BNN v2 prediction fix...")
print("=" * 60)

# Load saved model
model_path = _PROJECT_ROOT / "models/bayesian/bnn_simpler_v2.pkl"
model = SimplerBNNv2.load(model_path)

# Create small test data (different size than training)
X_test = np.random.randn(10, 4)  # 10 samples, 4 features

print(f"\nTest data shape: {X_test.shape}")
print("Attempting prediction...")

try:
    predictions = model.predict(X_test)
    print("✓ SUCCESS: Prediction completed without errors!")
    print(f"  Predicted mean shape: {predictions['mean'].shape}")
    print(f"  Sample predictions: {predictions['mean'][:3]}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
