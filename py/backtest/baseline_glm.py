"""Baseline logistic backtest for spread (ATS) modeling.

Uses the as-of feature dataset to run a walk-forward (leave-one-season-out)
logistic regression. Supports optional probability calibration (Platt sigmoid
or isotonic) via cross-validated calibration on the training fold. Reports
per-season metrics plus an overall summary.
Optionally writes season metrics, predictions, and a TeX table for inclusion in
analysis/dissertation.

Example:
  python py/backtest/baseline_glm.py \
      --features spread_close,epa_diff_prior,prior_epa_mean_diff,rest_diff \
      --start-season 2003 --end-season 2024 \
      --output-csv analysis/results/glm_baseline_metrics.csv \
      --preds-csv analysis/results/glm_baseline_preds.csv \
      --tex analysis/dissertation/figures/out/glm_baseline_table.tex
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DECIMAL_PAYOUT_DEFAULT = 1.9091  # Approx -110 vig

DEFAULT_FEATURE_COLUMNS = [
    "epa_diff_prior",
    "plays_diff_prior",
    "prior_epa_mean_diff",
    "epa_pp_last3_diff",
    "prior_margin_avg_diff",
    "win_pct_last5_diff",
    "season_win_pct_diff",
    "season_point_diff_avg_diff",
    "rest_diff",
    "prior_games_diff",
    "points_for_last3_diff",
    "points_against_last3_diff",
    "home_travel_change",
    "away_travel_change",
    "home_prev_result",
    "away_prev_result",
    "qb_change_diff",
    "home_qb_change",
    "away_qb_change",
    "qb_team_games_diff",
    "home_qb_team_games",
    "away_qb_team_games",
    "qb_total_games_diff",
    "home_qb_total_games",
    "away_qb_total_games",
    "coach_change_diff",
    "home_coach_change",
    "away_coach_change",
    "coach_team_games_diff",
    "home_coach_team_games",
    "away_coach_team_games",
    "coach_total_games_diff",
    "home_coach_total_games",
    "away_coach_total_games",
    "home_surface_grass",
    "home_surface_artificial",
    "home_roof_dome",
    "home_roof_outdoors",
    "surface_grass_diff",
    "surface_artificial_diff",
    "roof_dome_diff",
    "roof_outdoors_diff",
]


@dataclass
class Metrics:
    season: int | str
    games: int
    pushes: int
    brier: float
    log_loss: float
    hit_rate: float
    roi: float


@dataclass
class ModelBundle:
    estimator: object  # sklearn estimator with predict_proba
    feature_columns: Sequence[str]


def parse_feature_list(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_FEATURE_COLUMNS)
    return [col.strip() for col in raw.split(",") if col.strip()]


def load_features(csv_path: str, start: int, end: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["kickoff"])
    df = df[(df["season"] >= start) & (df["season"] <= end)].copy()
    return df


def prepare_dataset(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    required_columns = list(feature_columns) + ["home_cover", "home_margin", "spread_close"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from features dataset: {missing}")
    df = df.dropna(subset=["home_cover"])
    numeric_cols = list(feature_columns)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    df["home_cover_train"] = df["home_cover"].astype(int)
    return df


def fit_model(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    calibration: str = "none",
    cv_folds: int = 5,
) -> ModelBundle:
    X = train[list(feature_columns)].to_numpy(dtype=float)
    y = train["home_cover_train"].to_numpy(dtype=int)
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    if calibration and calibration.lower() != "none":
        method = "sigmoid" if calibration.lower() in {"platt", "sigmoid"} else "isotonic"
        try:
            est = CalibratedClassifierCV(estimator=base, method=method, cv=cv_folds)
        except TypeError:
            est = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv_folds)
        est.fit(X, y)
    else:
        est = base.fit(X, y)
    return ModelBundle(estimator=est, feature_columns=feature_columns)


def evaluate_model(
    bundle: ModelBundle,
    test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    X_test = test[list(bundle.feature_columns)].to_numpy(dtype=float)
    preds = bundle.estimator.predict_proba(X_test)[:, 1]
    out = test.copy()
    out["pred_cover_prob"] = preds
    out["bet_home"] = out["pred_cover_prob"] >= threshold
    return out


def compute_metrics(
    season: int | str,
    pred_df: pd.DataFrame,
    threshold: float,
    decimal_payout: float,
) -> Metrics:
    mask_valid = ~pred_df["home_cover"].isna()
    games = int(mask_valid.sum())
    pushes = int(pred_df.shape[0] - games)
    if games == 0:
        return Metrics(season, 0, pushes, np.nan, np.nan, np.nan, np.nan)

    y_true = pred_df.loc[mask_valid, "home_cover"].astype(int)
    y_pred = pred_df.loc[mask_valid, "pred_cover_prob"].clip(1e-6, 1 - 1e-6)
    bets_home = pred_df.loc[mask_valid, "bet_home"].astype(bool)

    brier = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)

    home_covers = y_true.astype(bool)
    correct = bets_home == home_covers
    hit_rate = correct.mean()

    payout_win = decimal_payout - 1.0
    roi_series = np.where(correct, payout_win, -1.0)
    roi = roi_series.mean()

    return Metrics(season, games, pushes, float(brier), float(ll), float(hit_rate), float(roi))


def run_backtest(
    df: pd.DataFrame,
    start_season: int,
    feature_columns: Sequence[str],
    threshold: float,
    decimal_payout: float,
) -> tuple[list[Metrics], pd.DataFrame]:
    results: list[Metrics] = []
    predictions: list[pd.DataFrame] = []
    seasons = sorted(df["season"].unique())
    for season in seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if season < start_season or train.empty or test.empty:
            continue
        # Fit model (with optional calibration handled inside)
        bundle = fit_model(
            train,
            feature_columns,
            calibration=globals().get("_CAL_METHOD", "none"),
            cv_folds=globals().get("_CAL_FOLDS", 5),
        )
        pred_df = evaluate_model(bundle, test, threshold)
        metrics = compute_metrics(season, pred_df, threshold, decimal_payout)
        results.append(metrics)
        pred_enriched = pred_df[
            [
                "game_id",
                "season",
                "week",
                "kickoff",
                "home_team",
                "away_team",
                "spread_close",
                "total_close",
                "home_cover",
                "pred_cover_prob",
                "bet_home",
            ]
        ].copy()
        pred_enriched["season_eval"] = season
        predictions.append(pred_enriched)
    combined_preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return results, combined_preds


def metrics_to_df(metrics_list: list[Metrics]) -> pd.DataFrame:
    return pd.DataFrame([m.__dict__ for m in metrics_list])


def append_overall_row(
    metrics_df: pd.DataFrame,
    overall_metrics: Metrics,
) -> pd.DataFrame:
    overall_row = pd.DataFrame([overall_metrics.__dict__])
    return pd.concat([metrics_df, overall_row], ignore_index=True)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def format_metric(value: float, precision: int = 4) -> str:
    if np.isnan(value):
        return "--"
    return f"{value:.{precision}f}"


def write_tex(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["season", "games", "pushes", "brier", "log_loss", "hit_rate", "roi"]
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Baseline GLM backtest]{Baseline GLM backtest metrics by season.}",
        "  \\label{tab:glm-baseline}",
        "  \\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}",
        "  \\begin{tabular}{@{} l r r r r r r @{} }",
        "    \\toprule",
        "    Season & Games & Pushes & Brier & LogLoss & HitRate & ROI \\\\ ",
        "    \\midrule",
    ]
    for _, row in df[cols].iterrows():
        season_val = row["season"]
        if isinstance(season_val, (np.integer, int)):
            season_fmt = f"{int(season_val)}"
        else:
            season_fmt = str(season_val)
        line = (
            "      {season} & {games:d} & {pushes:d} & {brier} & {log_loss} & {hit_rate} & {roi} \\\\"
        ).format(
            season=season_fmt,
            games=int(row["games"]),
            pushes=int(row["pushes"]),
            brier=format_metric(row["brier"]),
            log_loss=format_metric(row["log_loss"]),
            hit_rate=format_metric(row["hit_rate"]),
            roi=format_metric(row["roi"]),
        )
        lines.append(line)
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_predictions(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def reliability_curve(
    pred_df: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pred_df.copy()
    df = df.dropna(subset=["home_cover", "pred_cover_prob"]).copy()
    df["home_cover"] = df["home_cover"].astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["pred_cover_prob"], bins=bins, include_lowest=True, right=True)
    grp = df.groupby("bin", observed=True)
    out = grp.apply(
        lambda g: pd.Series(
            {
                "count": len(g),
                "pred_mean": g["pred_cover_prob"].mean(),
                "obs_rate": g["home_cover"].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()
    out["bin_low"] = out["bin"].apply(lambda iv: float(iv.left) if hasattr(iv, "left") else np.nan)
    out["bin_high"] = out["bin"].apply(
        lambda iv: float(iv.right) if hasattr(iv, "right") else np.nan
    )
    return out[["bin", "bin_low", "bin_high", "count", "pred_mean", "obs_rate"]]


def write_reliability_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def write_reliability_plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed; skipping reliability plot.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use bin midpoints for x
    x = (df["bin_low"].to_numpy(dtype=float) + df["bin_high"].to_numpy(dtype=float)) / 2.0
    y = df["obs_rate"].to_numpy(dtype=float)
    w = df["count"].to_numpy(dtype=float)
    plt.figure(figsize=(4.5, 4.0), dpi=150)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Ideal")
    plt.scatter(x, y, s=np.clip(w, 10, 120), alpha=0.8, color="#2a6fbb", label="Observed")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed rate")
    plt.title("Reliability Curve (Overall)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    default_features = ",".join(DEFAULT_FEATURE_COLUMNS)
    ap = argparse.ArgumentParser(description="Baseline GLM ATS backtest")
    ap.add_argument(
        "--features",
        default=default_features,
        help=f"Comma-separated feature columns (default: {default_features})",
    )
    ap.add_argument(
        "--features-csv",
        default="analysis/features/asof_team_features.csv",
        help="CSV containing as-of features",
    )
    ap.add_argument(
        "--start-season",
        type=int,
        default=2003,
        help="First season to evaluate (train uses seasons < start)",
    )
    ap.add_argument("--end-season", type=int, default=2024, help="Last season to include")
    ap.add_argument(
        "--min-season",
        dest="min_season",
        type=int,
        default=2001,
        help="Earliest season to load from dataset",
    )
    ap.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for betting home (else away)",
    )
    ap.add_argument(
        "--decimal-payout",
        type=float,
        default=DECIMAL_PAYOUT_DEFAULT,
        help="Decimal payout used for ROI (-110 -> 1.9091)",
    )
    ap.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Optional probability calibration",
    )
    ap.add_argument("--cv-folds", type=int, default=5, help="CV folds for calibration")
    ap.add_argument("--cal-bins", type=int, default=10, help="Bins for reliability curve")
    ap.add_argument("--output-csv", help="Optional CSV output path for per-season metrics")
    ap.add_argument("--preds-csv", help="Optional CSV output path for per-game predictions")
    ap.add_argument("--tex", help="Optional TeX table output path")
    ap.add_argument("--cal-csv", help="Optional CSV path for reliability curve (overall)")
    ap.add_argument("--cal-plot", help="Optional PNG path for reliability curve (overall)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = parse_feature_list(args.features)
    df = load_features(args.features_csv, args.min_season, args.end_season)
    df = prepare_dataset(df, feature_columns)
    # Store calibration settings in module globals for run_backtest to pick up
    global _CAL_METHOD, _CAL_FOLDS
    _CAL_METHOD = args.calibration
    _CAL_FOLDS = args.cv_folds
    metrics_list, preds_df = run_backtest(
        df,
        start_season=args.start_season,
        feature_columns=feature_columns,
        threshold=args.decision_threshold,
        decimal_payout=args.decimal_payout,
    )
    if not metrics_list:
        print("No seasons evaluated.")
        return

    metrics_df = metrics_to_df(metrics_list)
    if not preds_df.empty:
        overall_metrics = compute_metrics(
            "Overall", preds_df, args.decision_threshold, args.decimal_payout
        )
        metrics_df = append_overall_row(metrics_df, overall_metrics)
        # Reliability outputs
        rel = reliability_curve(preds_df, n_bins=args.cal_bins)
        if args.cal_csv:
            write_reliability_csv(rel, args.cal_csv)
        if args.cal_plot:
            write_reliability_plot(rel, args.cal_plot)
    print("Features used:", feature_columns)
    print(metrics_df)

    if args.output_csv:
        write_csv(metrics_df, args.output_csv)
    if args.preds_csv and not preds_df.empty:
        write_predictions(preds_df, args.preds_csv)
    if args.tex:
        write_tex(metrics_df, args.tex)


if __name__ == "__main__":
    main()
