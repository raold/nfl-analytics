#!/usr/bin/env python3
"""
Production-ready CQL model loading and inference utilities.

Supports:
- Loading single best model or full ensemble
- Batch prediction with uncertainty quantification
- Confidence filtering (only predict when ensemble agrees)
- Export predictions to CSV/JSON

Usage:
    from py.models.load_cql_ensemble import CQLEnsemble

    # Load ensemble
    ensemble = CQLEnsemble(models_dir="models/cql")
    ensemble.load_best_model("805ae9f0")
    ensemble.load_ensemble_models([
        'ee237922', 'a19dc3fe', '90fe41f9', ...
    ])

    # Predict with uncertainty
    state = {
        'spread_close': 7.0,
        'total_close': 48.5,
        'epa_gap': 0.15,
        'market_prob': 0.65,
        'p_hat': 0.72,
        'edge': 0.07
    }

    prediction = ensemble.predict_with_confidence(state, threshold=0.05)

    if prediction['action'] != 'skip':
        print(f"Bet {prediction['bet_size']:.2%} with Q-value {prediction['q_value']:.3f}")
        print(f"Confidence: {prediction['confidence']:.2%}")
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================================
# Q-Network Architecture (must match cql_agent.py)
# ============================================================================

class QNetwork(nn.Module):
    """MLP for Q(s, a) estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# ============================================================================
# CQL Ensemble Class
# ============================================================================

class CQLEnsemble:
    """Production-ready CQL ensemble for betting decisions."""

    def __init__(
        self,
        models_dir: Union[str, Path] = "models/cql",
        state_cols: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        """
        Initialize CQL ensemble.

        Args:
            models_dir: Directory containing trained models
            state_cols: State feature column names
            device: Device for inference (cpu, cuda, mps)
        """
        self.models_dir = Path(models_dir)
        self.device = device
        self.state_cols = state_cols or [
            "spread_close", "total_close", "epa_gap",
            "market_prob", "p_hat", "edge"
        ]

        self.best_model: Optional[QNetwork] = None
        self.best_model_id: Optional[str] = None
        self.ensemble_models: List[QNetwork] = []
        self.ensemble_model_ids: List[str] = []

        # Action space: {no-bet (0), small (1), medium (2), large (3)}
        self.action_space = {
            0: {'name': 'no-bet', 'size': 0.0},
            1: {'name': 'small', 'size': 0.01},
            2: {'name': 'medium', 'size': 0.03},
            3: {'name': 'large', 'size': 0.05}
        }

    def load_model(self, model_id: str) -> tuple[QNetwork, dict]:
        """Load a single CQL model."""
        model_path = self.models_dir / model_id
        metadata_path = model_path / "metadata.json"
        checkpoint_path = model_path / "best_checkpoint.pth"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        config = metadata["config"]
        state_dim = len(config["state_cols"])
        action_dim = 4  # {no-bet, small, medium, large}

        model = QNetwork(state_dim, action_dim, config["hidden_dims"])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["q_network"])
        model.eval()

        return model, metadata

    def load_best_model(self, model_id: str = "805ae9f0"):
        """Load best single model."""
        self.best_model, metadata = self.load_model(model_id)
        self.best_model_id = model_id
        print(f"✅ Loaded best model: {model_id} (loss={metadata['latest_metrics']['loss']:.4f})")

    def load_ensemble_models(self, model_ids: Optional[List[str]] = None):
        """Load ensemble models."""
        if model_ids is None:
            # Default Phase 3 ensemble (seed 42-61)
            model_ids = [
                'ee237922', 'a19dc3fe', '90fe41f9', 'aa67f6f5',
                'c46a91c3', 'cd7d1ed9', 'dc57c8a2', 'df2233d9',
                'fbaa0f3f', 'fef28489', '655b2be4', '1e76793f',
                '3a5be1ef', '3ea3746c', '33ad8155', '487eb7aa',
                '74b1acbf', '80e26617', '88c895db', '090d1bd4'
            ]

        for model_id in model_ids:
            try:
                model, _ = self.load_model(model_id)
                self.ensemble_models.append(model)
                self.ensemble_model_ids.append(model_id)
            except FileNotFoundError:
                print(f"⚠️  Skipping {model_id}: not found")

        print(f"✅ Loaded ensemble: {len(self.ensemble_models)} models")

    def _prepare_state(self, state: Union[Dict, pd.Series, np.ndarray]) -> torch.Tensor:
        """Convert state to tensor."""
        if isinstance(state, dict):
            state_array = np.array([state[col] for col in self.state_cols])
        elif isinstance(state, pd.Series):
            state_array = state[self.state_cols].values
        else:
            state_array = state

        return torch.FloatTensor(state_array).unsqueeze(0)

    def predict_single(self, state: Union[Dict, pd.Series, np.ndarray]) -> Dict:
        """Predict using best single model."""
        if self.best_model is None:
            raise RuntimeError("Best model not loaded. Call load_best_model() first.")

        state_tensor = self._prepare_state(state)

        with torch.no_grad():
            q_values = self.best_model(state_tensor).numpy()[0]

        # Best action
        best_action = int(q_values.argmax())
        action_info = self.action_space[best_action]

        return {
            'action': best_action,
            'action_name': action_info['name'],
            'bet_size': action_info['size'],
            'q_values': q_values.tolist(),
            'q_best': float(q_values[best_action]),
            'q_no_bet': float(q_values[0]),
            'q_advantage': float(q_values[best_action] - q_values[0]),
            'model_id': self.best_model_id
        }

    def predict_ensemble(
        self,
        state: Union[Dict, pd.Series, np.ndarray],
        confidence_threshold: float = 0.05
    ) -> Dict:
        """Predict using ensemble with uncertainty quantification."""
        if len(self.ensemble_models) == 0:
            raise RuntimeError("Ensemble not loaded. Call load_ensemble_models() first.")

        state_tensor = self._prepare_state(state)

        # Get predictions from all models
        q_values_list = []
        with torch.no_grad():
            for model in self.ensemble_models:
                q_values = model(state_tensor).numpy()[0]
                q_values_list.append(q_values)

        q_values_array = np.array(q_values_list)  # (n_models, 4)

        # Ensemble statistics
        q_mean = q_values_array.mean(axis=0)
        q_std = q_values_array.std(axis=0)

        # Best action according to ensemble mean
        best_action = int(q_mean.argmax())
        action_info = self.action_space[best_action]

        # Uncertainty
        best_q_std = q_std[best_action]
        high_confidence = best_q_std < confidence_threshold

        # Decision
        if best_action == 0 or not high_confidence:
            decision = 'skip'
            bet_size = 0.0
        else:
            decision = 'bet'
            bet_size = action_info['size']

        return {
            'action': best_action if high_confidence else 0,
            'action_name': action_info['name'] if high_confidence else 'skip',
            'decision': decision,
            'bet_size': bet_size,
            'q_mean': q_mean.tolist(),
            'q_std': q_std.tolist(),
            'q_best': float(q_mean[best_action]),
            'q_no_bet': float(q_mean[0]),
            'q_advantage': float(q_mean[best_action] - q_mean[0]),
            'uncertainty': float(best_q_std),
            'confidence': float(1 / (1 + best_q_std)),
            'high_confidence': bool(high_confidence),
            'ensemble_size': len(self.ensemble_models)
        }

    def predict_batch(
        self,
        states: pd.DataFrame,
        use_ensemble: bool = True,
        confidence_threshold: float = 0.05
    ) -> pd.DataFrame:
        """Predict on batch of states."""
        predictions = []

        for _, state in states.iterrows():
            if use_ensemble:
                pred = self.predict_ensemble(state, confidence_threshold)
            else:
                pred = self.predict_single(state)

            predictions.append(pred)

        return pd.DataFrame(predictions)

    def export_predictions(
        self,
        states: pd.DataFrame,
        output_path: Union[str, Path],
        use_ensemble: bool = True,
        format: str = "csv"
    ):
        """Export predictions to file."""
        predictions = self.predict_batch(states, use_ensemble)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            predictions.to_csv(output_path, index=False)
        elif format == "json":
            predictions.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"✅ Exported {len(predictions)} predictions to {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using CQL ensemble in production."""
    # Initialize ensemble
    ensemble = CQLEnsemble(models_dir="models/cql")

    # Load models
    ensemble.load_best_model("805ae9f0")
    ensemble.load_ensemble_models()  # Defaults to Phase 3 ensemble

    # Example game state
    state = {
        'spread_close': 7.0,
        'total_close': 48.5,
        'epa_gap': 0.15,
        'market_prob': 0.65,
        'p_hat': 0.72,
        'edge': 0.07
    }

    # Single model prediction
    print("\n" + "="*60)
    print("SINGLE MODEL PREDICTION")
    print("="*60)
    pred_single = ensemble.predict_single(state)
    print(f"Action: {pred_single['action_name']}")
    print(f"Bet size: {pred_single['bet_size']:.2%}")
    print(f"Q-value: {pred_single['q_best']:.3f}")
    print(f"Advantage: {pred_single['q_advantage']:.3f}")

    # Ensemble prediction
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION (with uncertainty)")
    print("="*60)
    pred_ensemble = ensemble.predict_ensemble(state, confidence_threshold=0.05)
    print(f"Decision: {pred_ensemble['decision']}")
    print(f"Action: {pred_ensemble['action_name']}")
    print(f"Bet size: {pred_ensemble['bet_size']:.2%}")
    print(f"Q-value: {pred_ensemble['q_best']:.3f} ± {pred_ensemble['uncertainty']:.3f}")
    print(f"Confidence: {pred_ensemble['confidence']:.2%}")
    print(f"High confidence: {pred_ensemble['high_confidence']}")


if __name__ == "__main__":
    example_usage()
