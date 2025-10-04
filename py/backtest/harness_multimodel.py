"""Comprehensive multi-model backtest harness with ensemble diagnostics.

Features
--------
- Walk-forward evaluation for GLM, Gradient Boosting, and State-Space models
- Automatic equal-weight and stacked (logistic) ensembles for every model subset
- Season-level and overall metrics (Brier, LogLoss, Accuracy, ROI)
- Bootstrap confidence intervals for metrics (season-resampled)
- Prediction covariance matrix and residual correlations between models
- Feature importance snapshots (GLM coefficients / odds ratios, boosting importances)
- Optional export of per-game predictions and diagnostics artefacts

Usage example
-------------
::

  python py/backtest/harness_multimodel.py \
    --features-csv analysis/features/asof_team_features.csv \
    --seasons 2003-2024 \
    --threshold 0.55 \
    --output-csv analysis/results/multimodel_comparison.csv \
    --per-season-csv analysis/results/multimodel_per_season.csv \
    --predictions-csv analysis/results/multimodel_predictions.csv \
    --diagnostics-dir analysis/results/multimodel_diagnostics
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.baseline_glm import DEFAULT_FEATURE_COLUMNS as GLM_FEATURES
from models.state_space import StateSpaceRatings

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


def parse_seasons(spec: str) -> list[int]:
    """Parse seasons argument supporting comma lists or ranges (e.g., "2003-2024")."""
    seasons: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(token))
    return sorted(sorted(set(seasons)))


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_feature_data(path: Path, seasons: Sequence[int]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["kickoff"])
    if seasons:
        df = df[df["season"].isin(seasons)].copy()
    df = df[df["home_cover"].notna()].copy()
    df.sort_values(["season", "week", "kickoff", "game_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ----------------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------------


class BaseModel:
    name: str

    def fit(self, train_df: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GLMModel(BaseModel):
    name = "glm"

    def __init__(self, feature_cols: Sequence[str]):
        self.feature_cols = list(feature_cols)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        X = train_df[self.feature_cols].fillna(0.0)
        y = train_df["home_cover"].astype(int).to_numpy()
        self.pipeline.fit(X, y)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].fillna(0.0)
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        coef = self.pipeline.named_steps["logit"].coef_[0]
        importance = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "coefficient": coef,
                "odds_ratio": np.exp(coef),
            }
        )
        importance.sort_values("coefficient", key=np.abs, ascending=False, inplace=True)
        return importance


class GradientBoostingModel(BaseModel):
    name = "xgb"

    def __init__(self, feature_cols: Sequence[str]):
        self.feature_cols = list(feature_cols)
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        X = train_df[self.feature_cols].fillna(0.0)
        y = train_df["home_cover"].astype(int).to_numpy()
        self.model.fit(X, y)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        importance = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            }
        )
        importance.sort_values("importance", ascending=False, inplace=True)
        return importance


class StateSpaceModel(BaseModel):
    name = "state"

    def __init__(self, q: float = 3.0, r: float = 13.5):
        self.q = q
        self.r = r
        self.model = StateSpaceRatings(q=q, r=r)

    def fit(self, train_df: pd.DataFrame) -> None:
        self.model = StateSpaceRatings(q=self.q, r=self.r)
        minimal = train_df[
            [
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        ].copy()
        minimal["margin"] = minimal["home_score"] - minimal["away_score"]
        self.model.fit(minimal)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in test_df.iterrows():
            probs.append(
                self.model.predict_prob_ats(
                    row["home_team"],
                    row["away_team"],
                    row["spread_close"],
                )
            )
        return np.asarray(probs)


MODEL_FACTORIES = {
    "glm": lambda features: GLMModel(features),
    "xgb": lambda features: GradientBoostingModel(features),
    "state": lambda _features: StateSpaceModel(),
}


# ----------------------------------------------------------------------------
# Walk-forward prediction engine
# ----------------------------------------------------------------------------


def walk_forward_predictions(
    df: pd.DataFrame,
    base_models: list[str],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    records: list[dict[str, object]] = []

    for idx in range(1, len(seasons)):
        season = seasons[idx]
        train_seasons = seasons[:idx]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == season]

        # Pre-populate per-game record
        season_records = {
            row["game_id"]: {
                "game_id": row["game_id"],
                "season": row["season"],
                "week": row["week"],
                "actual": int(row["home_cover"]),
            }
            for _, row in test_df.iterrows()
        }

        for model_key in base_models:
            factory = MODEL_FACTORIES[model_key]
            model = factory(feature_cols)
            model.fit(train_df)
            probs = model.predict_proba(test_df)
            for game_id, prob in zip(test_df["game_id"], probs):
                season_records[game_id][model_key] = float(prob)

        records.extend(
            rec for rec in season_records.values() if isinstance(rec, dict) and rec.get("game_id")
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Ensemble creation
# ----------------------------------------------------------------------------


def all_combinations(items: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for r in range(2, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield combo


def add_equal_weight_ensembles(pred_df: pd.DataFrame, base_models: list[str]) -> None:
    for combo in all_combinations(base_models):
        label = f"ens_mean_{'_'.join(combo)}"
        pred_df[label] = pred_df[list(combo)].mean(axis=1)


def add_stack_ensembles(pred_df: pd.DataFrame, base_models: list[str]) -> None:
    seasons = sorted(pred_df["season"].unique())
    for combo in all_combinations(base_models):
        label = f"ens_stack_{'_'.join(combo)}"
        pred_df[label] = np.nan

        for idx in range(1, len(seasons)):
            train_mask = pred_df["season"].isin(seasons[:idx])
            test_mask = pred_df["season"] == seasons[idx]
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            model = LogisticRegression(max_iter=2000, solver="lbfgs")
            model.fit(pred_df.loc[train_mask, list(combo)], pred_df.loc[train_mask, "actual"])
            preds = model.predict_proba(pred_df.loc[test_mask, list(combo)])[:, 1]
            pred_df.loc[test_mask, label] = preds

        # For any rows still NaN (e.g., first test season), fall back to equal-weight mean
        mean_label = f"ens_mean_{'_'.join(combo)}"
        if mean_label in pred_df:
            pred_df[label] = pred_df[label].fillna(pred_df[mean_label])


# ----------------------------------------------------------------------------
# Metrics & diagnostics
# ----------------------------------------------------------------------------


def compute_roi(
    probs: np.ndarray, actuals: np.ndarray, threshold: float, decimal_payout: float
) -> float:
    mask = probs >= threshold
    if mask.sum() == 0:
        return 0.0
    wins = actuals[mask].sum()
    losses = mask.sum() - wins
    profit = wins * (decimal_payout - 1.0) - losses
    return float(profit / mask.sum())


def overall_metrics(
    pred_df: pd.DataFrame, model_cols: list[str], threshold: float, decimal_payout: float
) -> pd.DataFrame:
    rows = []
    actual = pred_df["actual"].to_numpy()
    for col in model_cols:
        probs = pred_df[col].to_numpy()
        preds = (probs >= threshold).astype(int)
        safe_probs = np.clip(probs, 1e-9, 1 - 1e-9)
        rows.append(
            {
                "model": col,
                "n_games": len(actual),
                "brier": brier_score_loss(actual, safe_probs),
                "logloss": log_loss(actual, safe_probs),
                "accuracy": accuracy_score(actual, preds),
                "roi": compute_roi(probs, actual, threshold, decimal_payout),
            }
        )
    return pd.DataFrame(rows)


def per_season_metrics(
    pred_df: pd.DataFrame, model_cols: list[str], threshold: float, decimal_payout: float
) -> pd.DataFrame:
    rows = []
    for season, group in pred_df.groupby("season"):
        actual = group["actual"].to_numpy()
        for col in model_cols:
            probs = group[col].to_numpy()
            preds = (probs >= threshold).astype(int)
            safe_probs = np.clip(probs, 1e-9, 1 - 1e-9)
            rows.append(
                {
                    "season": season,
                    "model": col,
                    "n_games": len(actual),
                    "brier": brier_score_loss(actual, safe_probs),
                    "logloss": log_loss(actual, safe_probs),
                    "accuracy": accuracy_score(actual, preds),
                    "roi": compute_roi(probs, actual, threshold, decimal_payout),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(
    pred_df: pd.DataFrame,
    model_cols: list[str],
    threshold: float,
    decimal_payout: float,
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    seasons = pred_df["season"].unique()
    results: dict[str, dict[str, tuple[float, float]]] = {}

    metrics = {"brier", "logloss", "accuracy", "roi"}
    for model in model_cols:
        results[model] = {metric: (math.nan, math.nan) for metric in metrics}

    if len(seasons) < 2 or n_bootstrap <= 0:
        return results

    metric_samples = {model: {metric: [] for metric in metrics} for model in model_cols}
    for _ in range(n_bootstrap):
        sample_seasons = rng.choice(seasons, size=len(seasons), replace=True)
        sample_df = pred_df[pred_df["season"].isin(sample_seasons)]
        overall = overall_metrics(sample_df, model_cols, threshold, decimal_payout)
        for _, row in overall.iterrows():
            model = row["model"]
            metric_samples[model]["brier"].append(row["brier"])
            metric_samples[model]["logloss"].append(row["logloss"])
            metric_samples[model]["accuracy"].append(row["accuracy"])
            metric_samples[model]["roi"].append(row["roi"])

    for model in model_cols:
        for metric in metrics:
            samples = metric_samples[model][metric]
            if samples:
                results[model][metric] = (
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                )
    return results


def prediction_covariance(pred_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    residuals = {col: pred_df[col] - pred_df["actual"] for col in model_cols}
    res_df = pd.DataFrame(residuals)
    cov = res_df.corr()
    return cov


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-model backtest and diagnostics harness")
    ap.add_argument(
        "--features-csv",
        default="analysis/features/asof_team_features.csv",
        help="Prepared features CSV",
    )
    ap.add_argument(
        "--seasons", required=True, help="Seasons (comma or dash syntax, e.g., 2003-2024)"
    )
    ap.add_argument("--base-models", default="glm,xgb,state", help="Comma-separated base models")
    ap.add_argument("--threshold", type=float, default=0.55, help="Decision threshold for ROI")
    ap.add_argument(
        "--decimal-payout", type=float, default=1.91, help="Decimal payout (e.g., 1.91 for -110)"
    )
    ap.add_argument(
        "--output-csv",
        default="analysis/results/multimodel_comparison.csv",
        help="Overall metrics CSV",
    )
    ap.add_argument("--per-season-csv", default=None, help="Optional per-season metrics CSV")
    ap.add_argument("--predictions-csv", default=None, help="Optional per-game predictions CSV")
    ap.add_argument(
        "--diagnostics-dir", default=None, help="Directory for diagnostics artefacts (JSON/CSV)"
    )
    ap.add_argument(
        "--bootstrap-samples", type=int, default=500, help="Bootstrap iterations for CIs"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seasons = parse_seasons(args.seasons)
    base_models = [m.strip() for m in args.base_models.split(",") if m.strip()]
    unknown = [m for m in base_models if m not in MODEL_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown base models: {unknown}")

    feature_cols = GLM_FEATURES
    df = load_feature_data(Path(args.features_csv), seasons)
    if df.empty:
        raise ValueError("No games found for the specified seasons.")

    print(f"Loaded {len(df)} games across {df['season'].nunique()} seasons")

    predictions = walk_forward_predictions(df, base_models, feature_cols)
    add_equal_weight_ensembles(predictions, base_models)
    add_stack_ensembles(predictions, base_models)

    model_columns = [
        col for col in predictions.columns if col not in {"game_id", "season", "week", "actual"}
    ]

    overall = overall_metrics(predictions, model_columns, args.threshold, args.decimal_payout)
    ensure_dir(Path(args.output_csv))
    overall.sort_values("brier", inplace=True)
    overall.to_csv(args.output_csv, index=False)
    print(f"Overall metrics -> {args.output_csv}")

    per_season_csv = args.per_season_csv
    if per_season_csv:
        per_season_df = per_season_metrics(
            predictions, model_columns, args.threshold, args.decimal_payout
        )
        ensure_dir(Path(per_season_csv))
        per_season_df.to_csv(per_season_csv, index=False)
        print(f"Per-season metrics -> {per_season_csv}")

    if args.predictions_csv:
        ensure_dir(Path(args.predictions_csv))
        predictions.to_csv(args.predictions_csv, index=False)
        print(f"Per-game predictions -> {args.predictions_csv}")

    if args.diagnostics_dir:
        diag_dir = Path(args.diagnostics_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Bootstrap CIs
        ci = bootstrap_confidence_intervals(
            predictions,
            model_columns,
            threshold=args.threshold,
            decimal_payout=args.decimal_payout,
            n_bootstrap=args.bootstrap_samples,
        )
        (diag_dir / "metric_cis.json").write_text(json.dumps(ci, indent=2))

        # Covariance / correlation of residuals
        cov = prediction_covariance(predictions, model_columns)
        cov.to_csv(diag_dir / "residual_correlation.csv")

        # Feature importance snapshots (fit on all seasons)
        glm_model = GLMModel(feature_cols)
        glm_model.fit(df)
        glm_imp = glm_model.feature_importance(df)
        glm_imp.to_csv(diag_dir / "glm_importance.csv", index=False)

        xgb_model = GradientBoostingModel(feature_cols)
        xgb_model.fit(df)
        xgb_imp = xgb_model.feature_importance()
        xgb_imp.to_csv(diag_dir / "xgb_importance.csv", index=False)

        print(f"Diagnostics written to {diag_dir}")

    print("\n=== Overall Results ===")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
