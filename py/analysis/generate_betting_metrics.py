#!/usr/bin/env python3
"""
Generate comprehensive betting performance metrics from backtest predictions.

Calculates from multimodel_predictions.csv:
- CLV distribution (closing line value in bps)
- Realized edge vs closing line
- Sharpe ratio per model
- MAR ratio (return / max drawdown)
- Sortino ratio
- Win rate, ATS cover rate
- Bankroll trajectories

Generates LaTeX tables for dissertation Chapter 8.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_clv(row, model_prob_col):
    """Calculate closing line value in basis points."""
    # CLV = model prob - implied prob from closing line
    # Implied prob from American odds: if odds > 0: 100/(odds+100), if odds < 0: -odds/(-odds+100)

    # For spreads, use home_cover as target
    # Assume closing line is fair (50% no-vig)
    # CLV = |model_prob - 0.5| if correct, -(model_prob - 0.5) if wrong

    model_prob = row[model_prob_col]
    actual = row["actual"]

    # Directional edge: +EV if model says >50% and correct, or <50% and incorrect
    if actual == 1:
        edge = model_prob - 0.5
    else:
        edge = (1 - model_prob) - 0.5

    return edge * 10000  # Convert to basis points


def calculate_sharpe(returns):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    # Assume ~17 weeks per season, ~16 games per week = 272 bets per year
    return (returns.mean() / returns.std()) * np.sqrt(272)


def calculate_sortino(returns):
    """Calculate Sortino ratio (downside deviation)."""
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    # Guard against division by very small std (causes overflow)
    if downside_std < 1e-10:
        return 0.0
    return (returns.mean() / downside_std) * np.sqrt(272)


def calculate_mar(returns):
    """Calculate MAR ratio (return / max drawdown)."""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    if max_dd == 0:
        return 0.0
    total_return = cumulative.iloc[-1] - 1
    return total_return / max_dd


def main():
    """Generate betting performance metrics and tables."""
    print("\n🔄 Generating betting performance metrics from backtest predictions...\n")

    # Load predictions
    pred_path = Path("analysis/results/multimodel_predictions.csv")
    if not pred_path.exists():
        print(f"❌ Predictions file not found: {pred_path}")
        return

    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} predictions across {df['season'].nunique()} seasons")

    # Get model columns (exclude metadata and actual)
    model_cols = [
        c
        for c in df.columns
        if c
        not in [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "home_cover",
            "home_score",
            "away_score",
            "spread_close",
            "actual",
        ]
    ]

    print(f"Found {len(model_cols)} model predictions: {model_cols[:3]}...")

    # Calculate CLV for each model
    results = []

    for model in model_cols:
        # CLV in bps
        df[f"{model}_clv"] = df.apply(lambda row: calculate_clv(row, model), axis=1)

        # Binary bet outcomes (assume -110 odds, need >52.4% to profit)
        # Simulate betting when model prob > 52.4%
        threshold = 0.524
        df[f"{model}_bet"] = (df[model] > threshold) | (df[model] < (1 - threshold))
        df[f"{model}_return"] = 0.0

        # Calculate returns for bets
        for idx, row in df[df[f"{model}_bet"]].iterrows():
            prob = row[model]
            actual = row["actual"]

            # Bet home if prob > 52.4%, away if prob < 47.6%
            if prob > threshold:
                # Bet home
                won = actual == 1
            else:
                # Bet away
                won = actual == 0

            # Return: +0.91 if win (risk 1.1 to win 1), -1.1 if lose
            df.loc[idx, f"{model}_return"] = 0.91 if won else -1.1

        # Aggregate metrics
        clv_mean = df[f"{model}_clv"].mean()
        clv_median = df[f"{model}_clv"].median()
        clv_std = df[f"{model}_clv"].std()

        # Betting metrics (only on games where bet placed)
        bet_df = df[df[f"{model}_bet"]].copy()
        n_bets = len(bet_df)

        if n_bets > 0:
            returns = bet_df[f"{model}_return"]
            win_rate = (returns > 0).sum() / n_bets
            avg_return = returns.mean()
            sharpe = calculate_sharpe(returns)
            sortino = calculate_sortino(returns)
            mar = calculate_mar(returns)

            # Total P&L
            total_pnl = returns.sum()
            roi = total_pnl / n_bets  # Per-bet ROI
        else:
            win_rate = 0.0
            avg_return = 0.0
            sharpe = 0.0
            sortino = 0.0
            mar = 0.0
            total_pnl = 0.0
            roi = 0.0

        results.append(
            {
                "model": model,
                "n_games": len(df),
                "n_bets": n_bets,
                "clv_mean_bps": clv_mean,
                "clv_median_bps": clv_median,
                "clv_std_bps": clv_std,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_pnl": total_pnl,
                "roi_pct": roi * 100,
                "sharpe": sharpe,
                "sortino": sortino,
                "mar": mar,
            }
        )

    # Convert to DataFrame and sort by Sharpe
    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Save to CSV
    out_dir = Path("analysis/dissertation/figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "betting_performance_metrics.csv", index=False)
    print("✅ Saved betting_performance_metrics.csv")

    # Generate LaTeX table (top 5 models by Sharpe)
    top5 = results_df.head(5)

    with open(out_dir / "betting_performance_table.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write(
            "  \\caption[Betting performance metrics]{Betting performance metrics for top 5 models by Sharpe ratio (2004--2024). Assumes -110 odds, bets placed when model prob $>$ 52.4\\\\%.}\n"
        )
        f.write("  \\label{tab:betting-performance}\n")
        f.write("  \\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.12}\n")
        f.write("  \\begin{tabular}{@{} l r r r r r @{} }\n")
        f.write("    \\toprule\n")
        f.write(
            "    \\textbf{Model} & \\textbf{N Bets} & \\textbf{Win Rate} & \\textbf{ROI \\%} & \\textbf{Sharpe} & \\textbf{Sortino} \\\\\n"
        )
        f.write("    \\midrule\n")

        for _, row in top5.iterrows():
            model_short = (
                row["model"]
                .replace("ens_stack_", "")
                .replace("ens_mean_", "MEAN(")
                .replace("_", "+")
                .upper()
            )
            if "MEAN(" in model_short:
                model_short += ")"

            f.write(
                f"    {model_short[:25]} & {row['n_bets']:,} & {row['win_rate']*100:.1f}\\% & {row['roi_pct']:+.2f}\\% & {row['sharpe']:.2f} & {row['sortino']:.2f} \\\\\n"
            )

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Generated betting_performance_table.tex")

    # Generate CLV distribution table
    clv_summary = results_df[["model", "clv_mean_bps", "clv_median_bps", "clv_std_bps"]].head(5)

    with open(out_dir / "clv_distribution_table.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write(
            "  \\caption[CLV distribution by model]{Closing line value (CLV) distribution by model (basis points, 2004--2024).}\n"
        )
        f.write("  \\label{tab:clv-distribution}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.12}\n")
        f.write("  \\begin{tabular}{@{} l r r r @{} }\n")
        f.write("    \\toprule\n")
        f.write(
            "    \\textbf{Model} & \\textbf{Mean CLV (bps)} & \\textbf{Median CLV (bps)} & \\textbf{Std CLV (bps)} \\\\\n"
        )
        f.write("    \\midrule\n")

        for _, row in clv_summary.iterrows():
            model_short = (
                row["model"]
                .replace("ens_stack_", "")
                .replace("ens_mean_", "MEAN(")
                .replace("_", "+")
                .upper()
            )
            if "MEAN(" in model_short:
                model_short += ")"

            f.write(
                f"    {model_short[:25]} & {row['clv_mean_bps']:+.1f} & {row['clv_median_bps']:+.1f} & {row['clv_std_bps']:.1f} \\\\\n"
            )

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Generated clv_distribution_table.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("BETTING PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\nTop Model: {results_df.iloc[0]['model']}")
    print(f"  Sharpe Ratio: {results_df.iloc[0]['sharpe']:.2f}")
    print(f"  Win Rate: {results_df.iloc[0]['win_rate']*100:.1f}%")
    print(f"  ROI: {results_df.iloc[0]['roi_pct']:+.2f}%")
    print(f"  CLV: {results_df.iloc[0]['clv_mean_bps']:+.1f} bps")
    print(f"  Total Bets: {results_df.iloc[0]['n_bets']:,}")

    # Save summary JSON
    summary = {
        "top_model": results_df.iloc[0]["model"],
        "top_sharpe": float(results_df.iloc[0]["sharpe"]),
        "top_win_rate": float(results_df.iloc[0]["win_rate"]),
        "top_roi": float(results_df.iloc[0]["roi_pct"]),
        "top_clv_bps": float(results_df.iloc[0]["clv_mean_bps"]),
        "n_bets": int(results_df.iloc[0]["n_bets"]),
    }

    with open(out_dir / "betting_performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Saved betting_performance_summary.json\n")


if __name__ == "__main__":
    main()
