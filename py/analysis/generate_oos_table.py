#!/usr/bin/env python3
"""
Generate out-of-sample (OOS) results table by season.

Creates Table 10.2 referenced in Chapter 8.
Shows top 3 models (GLM, Stack, XGB) across recent seasons (2015-2024).
"""

from pathlib import Path

import pandas as pd


def main():
    """Generate OOS performance table."""
    print("\n🔄 Generating OOS results table...\n")

    # Load per-season results
    results_path = Path("analysis/results/multimodel_per_season.csv")
    df = pd.read_csv(results_path)

    # Select top 3 models
    top_models = ["glm", "ens_stack_glm_xgb_state", "xgb"]
    df_top = df[df["model"].isin(top_models)].copy()

    # Filter to recent seasons (2015-2024)
    df_recent = df_top[df_top["season"] >= 2015].copy()

    # Format for LaTeX
    out_dir = Path("analysis/dissertation/figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "oos_record_table.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write(
            "  \\caption[Out-of-sample results by season]{Out-of-sample performance by season: GLM baseline, stacked ensemble, and XGBoost (2015--2024).}\n"
        )
        f.write("  \\label{tab:oos-record}\n")
        f.write("  \\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.08}\n")
        f.write("  \\begin{tabular}{@{} c l r r r r @{} }\n")
        f.write("    \\toprule\n")
        f.write(
            "    \\textbf{Season} & \\textbf{Model} & \\textbf{N} & \\textbf{Brier $\\downarrow$} & \\textbf{Accuracy} & \\textbf{ROI \\%} \\\\\n"
        )
        f.write("    \\midrule\n")

        # Group by season
        for season in sorted(df_recent["season"].unique()):
            season_data = df_recent[df_recent["season"] == season]

            for idx, (_, row) in enumerate(season_data.iterrows()):
                # Model short name
                model_map = {
                    "glm": "GLM",
                    "ens_stack_glm_xgb_state": "Stack(All)",
                    "xgb": "XGBoost",
                }
                model_short = model_map[row["model"]]

                # Format metrics
                n_games = int(row["n_games"])
                brier = f"{row['brier']:.4f}"
                acc = f"{row['accuracy']*100:.1f}\\%"
                roi = f"{row['roi']*100:+.1f}\\%"

                # Write row
                if idx == 0:
                    f.write(
                        f"    {season} & {model_short} & {n_games} & {brier} & {acc} & {roi} \\\\\n"
                    )
                else:
                    f.write(f"     & {model_short} & {n_games} & {brier} & {acc} & {roi} \\\\\n")

            # Add midrule between seasons (except last)
            if season < sorted(df_recent["season"].unique())[-1]:
                f.write("    \\midrule\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Generated oos_record_table.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("OOS TABLE SUMMARY")
    print("=" * 60)
    print(f"Seasons covered: {df_recent['season'].min()} - {df_recent['season'].max()}")
    print(f"Models: {len(top_models)}")
    print(f"Total rows: {len(df_recent)}")
    print("\nBest Brier by season (Stack model):")

    stack_data = df_recent[df_recent["model"] == "ens_stack_glm_xgb_state"]
    for _, row in stack_data.iterrows():
        print(
            f"  {row['season']}: {row['brier']:.4f} (Acc={row['accuracy']*100:.1f}%, ROI={row['roi']*100:+.1f}%)"
        )


if __name__ == "__main__":
    main()
