#!/usr/bin/env python3
"""
Ensemble Model Stacking with Learned Weights

Instead of fixed 70/30 weights, learn optimal ensemble weights using:
1. Logistic regression on model predictions
2. XGBoost meta-learner
3. Weighted average with learned weights

Usage:
    python py/models/ensemble_stacking.py --v2-model models/xgboost/v2_1/model.json \
                                          --v3-model models/xgboost/v3_production/model.json
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnsembleStacker:
    """Learn optimal ensemble weights for combining models."""

    def __init__(self, v2_model_path: str, v3_model_path: str, features_csv: str):
        """Initialize ensemble stacker."""
        self.v2_model_path = Path(v2_model_path)
        self.v3_model_path = Path(v3_model_path)
        self.features_csv = features_csv

        # Load models
        logger.info(f"Loading v2.1 model from {self.v2_model_path}")
        self.v2_model = xgb.Booster()
        self.v2_model.load_model(str(self.v2_model_path))

        with open(self.v2_model_path.parent / "metadata.json") as f:
            self.v2_metadata = json.load(f)

        logger.info(f"Loading v3 model from {self.v3_model_path}")
        self.v3_model = xgb.Booster()
        self.v3_model.load_model(str(self.v3_model_path))

        with open(self.v3_model_path.parent / "metadata.json") as f:
            self.v3_metadata = json.load(f)

        self.v2_features = self.v2_metadata["training_data"]["features"]
        self.v3_features = self.v3_metadata["training_data"]["features"]

        logger.info(f"v2.1: {len(self.v2_features)} features")
        logger.info(f"v3: {len(self.v3_features)} features")

    def load_data(self, test_seasons: list[int]) -> pd.DataFrame:
        """Load feature data for test seasons."""
        logger.info(f"Loading data from {self.features_csv}")
        df = pd.read_csv(self.features_csv)

        # Filter to completed games in test seasons
        df = df[(df["season"].isin(test_seasons)) & (df["home_score"].notna())].copy()

        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        logger.info(f"Loaded {len(df)} games from seasons {test_seasons}")
        return df

    def get_base_predictions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from both base models.

        Returns:
            (v2_predictions, v3_predictions)
        """
        logger.info("Generating base model predictions...")

        # v2.1 predictions
        X_v2 = df[self.v2_features].fillna(0).values
        dmatrix_v2 = xgb.DMatrix(X_v2, feature_names=self.v2_features)
        v2_preds = self.v2_model.predict(dmatrix_v2)

        # v3 predictions
        X_v3 = df[self.v3_features].fillna(0).values
        dmatrix_v3 = xgb.DMatrix(X_v3, feature_names=self.v3_features)
        v3_preds = self.v3_model.predict(dmatrix_v3)

        logger.info(f"✓ Generated {len(v2_preds)} predictions from each model")

        return v2_preds, v3_preds

    def train_stacking_models(
        self, v2_preds: np.ndarray, v3_preds: np.ndarray, y_true: np.ndarray, val_split: float = 0.2
    ) -> dict:
        """
        Train multiple stacking approaches.

        Args:
            v2_preds: v2.1 model predictions
            v3_preds: v3 model predictions
            y_true: True labels
            val_split: Validation split ratio

        Returns:
            Dict of stacking models and their performance
        """
        logger.info("\nTraining stacking models...")

        # Split data
        n = len(y_true)
        n_val = int(n * val_split)
        indices = np.random.permutation(n)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train = np.column_stack([v2_preds[train_idx], v3_preds[train_idx]])
        y_train = y_true[train_idx]
        X_val = np.column_stack([v2_preds[val_idx], v3_preds[val_idx]])
        y_val = y_true[val_idx]

        results = {}

        # 1. Fixed weights (baseline)
        logger.info("\n[1] Fixed Weights (70% v3, 30% v2.1):")
        fixed_train = 0.7 * v3_preds[train_idx] + 0.3 * v2_preds[train_idx]
        fixed_val = 0.7 * v3_preds[val_idx] + 0.3 * v2_preds[val_idx]

        results["fixed"] = {
            "train_brier": float(brier_score_loss(y_train, fixed_train)),
            "val_brier": float(brier_score_loss(y_val, fixed_val)),
            "train_acc": float(accuracy_score(y_train, (fixed_train > 0.5).astype(int))),
            "val_acc": float(accuracy_score(y_val, (fixed_val > 0.5).astype(int))),
            "weights": {"v2": 0.3, "v3": 0.7},
        }

        logger.info(
            f"  Train Brier: {results['fixed']['train_brier']:.4f}, Acc: {results['fixed']['train_acc']:.1%}"
        )
        logger.info(
            f"  Val Brier: {results['fixed']['val_brier']:.4f}, Acc: {results['fixed']['val_acc']:.1%}"
        )

        # 2. Logistic Regression Stacking
        logger.info("\n[2] Logistic Regression Stacking:")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)

        lr_train_pred = lr_model.predict_proba(X_train)[:, 1]
        lr_val_pred = lr_model.predict_proba(X_val)[:, 1]

        results["logistic"] = {
            "train_brier": float(brier_score_loss(y_train, lr_train_pred)),
            "val_brier": float(brier_score_loss(y_val, lr_val_pred)),
            "train_acc": float(accuracy_score(y_train, (lr_train_pred > 0.5).astype(int))),
            "val_acc": float(accuracy_score(y_val, (lr_val_pred > 0.5).astype(int))),
            "coefficients": {
                "v2": float(lr_model.coef_[0][0]),
                "v3": float(lr_model.coef_[0][1]),
                "intercept": float(lr_model.intercept_[0]),
            },
            "model": lr_model,
        }

        logger.info(f"  Coef v2: {lr_model.coef_[0][0]:.4f}, v3: {lr_model.coef_[0][1]:.4f}")
        logger.info(
            f"  Train Brier: {results['logistic']['train_brier']:.4f}, Acc: {results['logistic']['train_acc']:.1%}"
        )
        logger.info(
            f"  Val Brier: {results['logistic']['val_brier']:.4f}, Acc: {results['logistic']['val_acc']:.1%}"
        )

        # 3. Optimal weighted average (grid search)
        logger.info("\n[3] Optimal Weighted Average (Grid Search):")
        best_weight = 0.7
        best_brier = float("inf")

        for w in np.arange(0.0, 1.01, 0.05):
            ensemble_val = w * v3_preds[val_idx] + (1 - w) * v2_preds[val_idx]
            brier = brier_score_loss(y_val, ensemble_val)
            if brier < best_brier:
                best_brier = brier
                best_weight = w

        optimal_train = best_weight * v3_preds[train_idx] + (1 - best_weight) * v2_preds[train_idx]
        optimal_val = best_weight * v3_preds[val_idx] + (1 - best_weight) * v2_preds[val_idx]

        results["optimal"] = {
            "train_brier": float(brier_score_loss(y_train, optimal_train)),
            "val_brier": float(brier_score_loss(y_val, optimal_val)),
            "train_acc": float(accuracy_score(y_train, (optimal_train > 0.5).astype(int))),
            "val_acc": float(accuracy_score(y_val, (optimal_val > 0.5).astype(int))),
            "weights": {"v2": float(1 - best_weight), "v3": float(best_weight)},
        }

        logger.info(f"  Optimal weights: v2={1-best_weight:.2f}, v3={best_weight:.2f}")
        logger.info(
            f"  Train Brier: {results['optimal']['train_brier']:.4f}, Acc: {results['optimal']['train_acc']:.1%}"
        )
        logger.info(
            f"  Val Brier: {results['optimal']['val_brier']:.4f}, Acc: {results['optimal']['val_acc']:.1%}"
        )

        # 4. XGBoost Meta-Learner
        logger.info("\n[4] XGBoost Meta-Learner:")
        dtrain_meta = xgb.DMatrix(X_train, label=y_train, feature_names=["v2_pred", "v3_pred"])
        dval_meta = xgb.DMatrix(X_val, label=y_val, feature_names=["v2_pred", "v3_pred"])

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,  # Shallow tree for simple combination
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "device": "cpu",
        }

        evals = [(dtrain_meta, "train"), (dval_meta, "val")]
        meta_model = xgb.train(
            params,
            dtrain_meta,
            num_boost_round=50,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        xgb_train_pred = meta_model.predict(dtrain_meta)
        xgb_val_pred = meta_model.predict(dval_meta)

        results["xgboost_meta"] = {
            "train_brier": float(brier_score_loss(y_train, xgb_train_pred)),
            "val_brier": float(brier_score_loss(y_val, xgb_val_pred)),
            "train_acc": float(accuracy_score(y_train, (xgb_train_pred > 0.5).astype(int))),
            "val_acc": float(accuracy_score(y_val, (xgb_val_pred > 0.5).astype(int))),
            "model": meta_model,
        }

        logger.info(
            f"  Train Brier: {results['xgboost_meta']['train_brier']:.4f}, Acc: {results['xgboost_meta']['train_acc']:.1%}"
        )
        logger.info(
            f"  Val Brier: {results['xgboost_meta']['val_brier']:.4f}, Acc: {results['xgboost_meta']['val_acc']:.1%}"
        )

        return results

    def evaluate_on_test(
        self, stacking_results: dict, v2_preds: np.ndarray, v3_preds: np.ndarray, y_true: np.ndarray
    ) -> dict:
        """Evaluate all stacking approaches on full test set."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("=" * 80)

        eval_results = {}

        # Individual models
        logger.info("\nIndividual Models:")
        v2_brier = brier_score_loss(y_true, v2_preds)
        v2_acc = accuracy_score(y_true, (v2_preds > 0.5).astype(int))
        logger.info(f"  v2.1: Brier {v2_brier:.4f}, Accuracy {v2_acc:.1%}")

        v3_brier = brier_score_loss(y_true, v3_preds)
        v3_acc = accuracy_score(y_true, (v3_preds > 0.5).astype(int))
        logger.info(f"  v3:   Brier {v3_brier:.4f}, Accuracy {v3_acc:.1%}")

        eval_results["v2"] = {"brier": float(v2_brier), "accuracy": float(v2_acc)}
        eval_results["v3"] = {"brier": float(v3_brier), "accuracy": float(v3_acc)}

        # Ensemble models
        logger.info("\nEnsemble Models:")

        # Fixed
        fixed_pred = 0.7 * v3_preds + 0.3 * v2_preds
        fixed_brier = brier_score_loss(y_true, fixed_pred)
        fixed_acc = accuracy_score(y_true, (fixed_pred > 0.5).astype(int))
        logger.info(f"  Fixed (70/30):      Brier {fixed_brier:.4f}, Accuracy {fixed_acc:.1%}")
        eval_results["fixed"] = {"brier": float(fixed_brier), "accuracy": float(fixed_acc)}

        # Optimal
        weights = stacking_results["optimal"]["weights"]
        optimal_pred = weights["v3"] * v3_preds + weights["v2"] * v2_preds
        optimal_brier = brier_score_loss(y_true, optimal_pred)
        optimal_acc = accuracy_score(y_true, (optimal_pred > 0.5).astype(int))
        logger.info(
            f"  Optimal ({weights['v3']:.2f}/{weights['v2']:.2f}): Brier {optimal_brier:.4f}, Accuracy {optimal_acc:.1%}"
        )
        eval_results["optimal"] = {"brier": float(optimal_brier), "accuracy": float(optimal_acc)}

        # Logistic
        X = np.column_stack([v2_preds, v3_preds])
        lr_pred = stacking_results["logistic"]["model"].predict_proba(X)[:, 1]
        lr_brier = brier_score_loss(y_true, lr_pred)
        lr_acc = accuracy_score(y_true, (lr_pred > 0.5).astype(int))
        logger.info(f"  Logistic Stacking:  Brier {lr_brier:.4f}, Accuracy {lr_acc:.1%}")
        eval_results["logistic"] = {"brier": float(lr_brier), "accuracy": float(lr_acc)}

        # XGBoost Meta
        dmeta = xgb.DMatrix(X, feature_names=["v2_pred", "v3_pred"])
        xgb_pred = stacking_results["xgboost_meta"]["model"].predict(dmeta)
        xgb_brier = brier_score_loss(y_true, xgb_pred)
        xgb_acc = accuracy_score(y_true, (xgb_pred > 0.5).astype(int))
        logger.info(f"  XGBoost Meta:       Brier {xgb_brier:.4f}, Accuracy {xgb_acc:.1%}")
        eval_results["xgboost_meta"] = {"brier": float(xgb_brier), "accuracy": float(xgb_acc)}

        # Find best
        best_method = min(eval_results.items(), key=lambda x: x[1]["brier"])
        logger.info(
            f"\n✓ Best Method: {best_method[0].upper()} (Brier {best_method[1]['brier']:.4f})"
        )

        return eval_results

    def save_ensemble(self, stacking_results: dict, test_results: dict, output_dir: str):
        """Save ensemble configuration."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save logistic regression model
        import joblib

        joblib.dump(stacking_results["logistic"]["model"], output_path / "logistic_stacker.pkl")

        # Save XGBoost meta-learner
        stacking_results["xgboost_meta"]["model"].save_model(str(output_path / "xgboost_meta.json"))

        # Save metadata
        metadata = {
            "v2_model": str(self.v2_model_path),
            "v3_model": str(self.v3_model_path),
            "stacking_approaches": {
                "fixed": stacking_results["fixed"],
                "optimal": stacking_results["optimal"],
                "logistic": {k: v for k, v in stacking_results["logistic"].items() if k != "model"},
            },
            "test_results": test_results,
        }

        with open(output_path / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Ensemble saved to {output_path}/")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ensemble stacking with learned weights")
    parser.add_argument("--v2-model", default="models/xgboost/v2_1/model.json")
    parser.add_argument("--v3-model", default="models/xgboost/v3_production/model.json")
    parser.add_argument(
        "--features-csv", default="data/processed/features/asof_team_features_v3.csv"
    )
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2024])
    parser.add_argument("--output-dir", default="models/ensemble/v1")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize
    stacker = EnsembleStacker(args.v2_model, args.v3_model, args.features_csv)

    # Load data
    df = stacker.load_data(args.test_seasons)

    # Get base predictions
    v2_preds, v3_preds = stacker.get_base_predictions(df)
    y_true = df["home_win"].values

    # Train stacking models
    stacking_results = stacker.train_stacking_models(v2_preds, v3_preds, y_true)

    # Evaluate on test
    test_results = stacker.evaluate_on_test(stacking_results, v2_preds, v3_preds, y_true)

    # Save
    stacker.save_ensemble(stacking_results, test_results, args.output_dir)

    print("\n" + "=" * 80)
    print("ENSEMBLE STACKING COMPLETE")
    print("=" * 80)
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
