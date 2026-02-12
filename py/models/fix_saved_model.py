"""
Fix the existing saved model by adding training shapes
"""

import pickle
from pathlib import Path

model_path = Path("/Users/dro/rice/nfl-analytics/models/bayesian/bnn_simpler_v2.pkl")

print("Loading existing model...")
with open(model_path, "rb") as f:
    save_obj = pickle.load(f)

print(f"Keys in saved object: {save_obj.keys()}")

# Add training shapes (we know from the training output: 2663 samples, 4 features)
save_obj["X_train_shape"] = (2663, 4)
save_obj["y_train_shape"] = (2663,)
save_obj["scaler_mean"] = None  # We'll need to refit if we need scaler
save_obj["scaler_scale"] = None

# Re-save with updated shapes
with open(model_path, "wb") as f:
    pickle.dump(save_obj, f)

print("✓ Model updated with training shapes")
print(f"  X_train_shape: {save_obj['X_train_shape']}")
print(f"  y_train_shape: {save_obj['y_train_shape']}")
