"""Backtest harness: run baseline GLM over multiple configs.

Runs a small grid of feature lists and decision thresholds using the
`baseline_glm` module, then writes a summary CSV and optional TeX table.

Example:
  python py/backtest/harness.py \
    --features-csv analysis/features/asof_team_features.csv \
    --start-season 2003 --end-season 2024 \
    --output-csv analysis/results/glm_harness_metrics.csv \
    --tex analysis/dissertation/figures/out/glm_harness_table.tex
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import baseline_glm as glm  # type: ignore
except Exception:  # pragma: no cover
    # Allow running as a standalone script
    import importlib

    glm = importlib.import_module("backtest.baseline_glm")


@dataclass
class Config:
    name: str
    features: Sequence[str]
    threshold: float


DEFAULT_CONFIGS: list[Config] = [
    Config(
        name="core_form",
        features=[
            "epa_diff_prior",
            "prior_epa_mean_diff",
            "epa_pp_last3_diff",
            "prior_margin_avg_diff",
            "season_point_diff_avg_diff",
            "rest_diff",
        ],
        threshold=0.5,
    ),
    Config(
        name="core_plus_recent",
        features=[
            "epa_diff_prior",
            "prior_epa_mean_diff",
            "epa_pp_last3_diff",
            "prior_margin_avg_diff",
            "season_point_diff_avg_diff",
            "rest_diff",
            "points_for_last3_diff",
            "points_against_last3_diff",
            "win_pct_last5_diff",
        ],
        threshold=0.5,
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backtest harness for GLM variants")
    ap.add_argument(
        "--features-csv",
        default="analysis/features/asof_team_features.csv",
        help="CSV containing as-of features",
    )
    ap.add_argument("--start-season", type=int, default=2003)
    ap.add_argument("--end-season", type=int, default=2024)
    ap.add_argument("--min-season", type=int, default=2001)
    ap.add_argument("--decimal-payout", type=float, default=glm.DECIMAL_PAYOUT_DEFAULT)
    ap.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="none")
    ap.add_argument(
        "--calibrations",
        default=None,
        help="Comma-separated calibration methods to sweep (overrides --calibration)",
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument(
        "--thresholds",
        default="0.45,0.50,0.55",
        help="Comma-separated decision thresholds to sweep",
    )
    ap.add_argument("--cal-bins", type=int, default=10, help="Bins for reliability curve")
    ap.add_argument(
        "--cal-out-dir",
        default=None,
        help="Directory to write reliability plots/CSVs per run and per season",
    )
    ap.add_argument(
        "--panel-config", default=None, help="Config name to build per-season reliability panel for"
    )
    ap.add_argument(
        "--panel-cal", default=None, help="Calibration to build per-season reliability panel for"
    )
    ap.add_argument(
        "--panel-threshold",
        type=float,
        default=None,
        help="Threshold to build per-season reliability panel for",
    )
    ap.add_argument(
        "--panel-tex", default=None, help="Output TeX path for per-season reliability panel"
    )
    ap.add_argument(
        "--panel2-cal", default=None, help="Second calibration for side-by-side panel (e.g., none)"
    )
    ap.add_argument(
        "--panel-combo-tex",
        default=None,
        help="Output TeX path for side-by-side (cal1 vs cal2) panel",
    )
    ap.add_argument("--output-csv", help="Output CSV for combined metrics")
    ap.add_argument("--tex", help="Optional per-season TeX table path")
    ap.add_argument("--tex-overall", help="Optional overall comparison TeX table path")
    return ap.parse_args()


def write_tex(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "config",
        "calibration",
        "season",
        "games",
        "ece",
        "brier",
        "log_loss",
        "hit_rate",
        "roi",
    ]
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[GLM variants (harness)]{Baseline GLM variants by season.}",
        "  \\label{tab:glm-harness}",
        "  \\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}",
        "  \\begin{tabular}{@{} l l r r r r r r r @{} }",
        "    \\toprule",
        "    Config & Cal & Season & Games & ECE & Brier & LogLoss & HitRate & ROI \\\\ ",
        "    \\midrule",
    ]
    for _, row in df[cols].iterrows():
        line = (
            "      {config} & {cal} & {season} & {games:d} & {ece:.4f} & {brier:.4f} & {log_loss:.4f} & {hit_rate:.4f} & {roi:.4f} \\\\"
        ).format(
            config=str(row["config"]).replace("_", "\\_"),
            cal=row.get("calibration", "none"),
            season=int(row["season"]) if str(row["season"]).isdigit() else row["season"],
            games=int(row["games"]),
            ece=row.get("ece", float("nan")),
            brier=row["brier"],
            log_loss=row["log_loss"],
            hit_rate=row["hit_rate"],
            roi=row["roi"],
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


def write_tex_overall(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "config",
        "calibration",
        "threshold",
        "ece",
        "mce",
        "brier",
        "log_loss",
        "hit_rate",
        "roi",
    ]
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[GLM overall comparison]{Overall metrics by config and threshold.}",
        "  \\label{tab:glm-harness-overall}",
        "  \\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}",
        "  \\begin{tabular}{@{} l l r r r r r r r @{} }",
        "    \\toprule",
        "    Config & Cal & Thr & ECE & MCE & Brier & LogLoss & HitRate & ROI \\\\ ",
        "    \\midrule",
    ]
    for _, row in df[cols].iterrows():
        line = (
            "      {config} & {cal} & {thr:.2f} & {ece:.4f} & {mce:.4f} & {brier:.4f} & {log_loss:.4f} & {hit_rate:.4f} & {roi:.4f} \\\\"
        ).format(
            config=str(row["config"]).replace("_", "\\_"),
            cal=row.get("calibration", "none"),
            thr=float(row["threshold"]),
            ece=row["ece"],
            mce=row["mce"],
            brier=row["brier"],
            log_loss=row["log_loss"],
            hit_rate=row["hit_rate"],
            roi=row["roi"],
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


def write_panel_tex(image_paths: list[str], out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{figure}[t]",
        "  \\centering",
        f"  \\caption[{title}]{{{title}}}",
    ]
    per_row = 4
    for i, img in enumerate(image_paths):
        lines.append(f"  \\includegraphics[width=0.22\\linewidth]{{{img}}}")
        if (i + 1) % per_row == 0:
            lines.append("  \\par\\vspace{2pt}")
    lines.append("\\end{figure}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_dual_panel_tex(
    left_images: list[str],
    right_images: list[str],
    out_path: str,
    left_title: str,
    right_title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def panel_lines(images: list[str]) -> list[str]:
        lines: list[str] = []
        per_row = 4
        for i, img in enumerate(images):
            lines.append(f"    \\includegraphics[width=0.22\\linewidth]{{{img}}}")
            if (i + 1) % per_row == 0:
                lines.append("    \\par\\vspace{2pt}")
        return lines

    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{figure}[t]",
        "  \\centering",
        "  \\begin{minipage}[t]{0.49\\linewidth}",
        f"    \\caption*{{\\footnotesize {left_title}}}",
    ]
    lines.extend(panel_lines(left_images))
    lines.extend(
        [
            "  \\end{minipage}\\hfill",
            "  \\begin{minipage}[t]{0.49\\linewidth}",
            f"    \\caption*{{\\footnotesize {right_title}}}",
        ]
    )
    lines.extend(panel_lines(right_images))
    lines.extend(
        [
            "  \\end{minipage}",
            "\\end{figure}",
        ]
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_df = glm.load_features(args.features_csv, args.min_season, args.end_season)
    out_rows: list[pd.DataFrame] = []
    overall_rows: list[pd.DataFrame] = []
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    calibrations = [
        m.strip()
        for m in (args.calibrations.split(",") if args.calibrations else [args.calibration])
        if m.strip()
    ]
    # For building a side-by-side panel after loop
    combo_left: list[str] | None = None
    combo_right: list[str] | None = None
    combo_left_title = combo_right_title = None

    for cfg in DEFAULT_CONFIGS:
        df = glm.prepare_dataset(base_df, cfg.features)
        for cal in calibrations:
            glm._CAL_METHOD = cal
            glm._CAL_FOLDS = args.cv_folds
            for thr in thresholds:
                metrics_list, preds_df = glm.run_backtest(
                    df,
                    start_season=args.start_season,
                    feature_columns=cfg.features,
                    threshold=thr,
                    decimal_payout=args.decimal_payout,
                )
                if not metrics_list:
                    continue
                metrics_df = glm.metrics_to_df(metrics_list)
                if not preds_df.empty:
                    # Reliability (overall)
                    rel = glm.reliability_curve(preds_df, n_bins=args.cal_bins)
                    # ECE/MCE
                    total = float(rel["count"].sum()) or 1.0
                    ece = float(
                        (rel["count"] * (rel["pred_mean"] - rel["obs_rate"]).abs()).sum() / total
                    )
                    mce = float((rel["pred_mean"] - rel["obs_rate"]).abs().max())
                    # Per-season ECE
                    season_ece = {}
                    for s, g in preds_df.groupby("season"):
                        rel_s = glm.reliability_curve(g, n_bins=args.cal_bins)
                        tot_s = float(rel_s["count"].sum()) or 1.0
                        season_ece[int(s)] = float(
                            (rel_s["count"] * (rel_s["pred_mean"] - rel_s["obs_rate"]).abs()).sum()
                            / tot_s
                        )
                    overall = glm.compute_metrics("Overall", preds_df, thr, args.decimal_payout)
                    overall_df = pd.DataFrame([overall.__dict__])
                    overall_df.insert(0, "config", cfg.name)
                    overall_df.insert(1, "calibration", cal)
                    overall_df.insert(2, "threshold", thr)
                    overall_df["ece"] = ece
                    overall_df["mce"] = mce
                    overall_rows.append(overall_df)
                    # Attach per-season ECE to metrics_df rows
                    metrics_df["ece"] = metrics_df["season"].map(
                        lambda x: season_ece.get(int(x), np.nan)
                    )
                    metrics_df = pd.concat(
                        [
                            metrics_df,
                            overall_df.drop(columns=["config", "calibration", "threshold"]),
                        ],
                        ignore_index=True,
                    )
                    # Write overall reliability CSV/plot if out dir provided
                    if args.cal_out_dir:
                        os.makedirs(args.cal_out_dir, exist_ok=True)
                        base = f"rel_{cfg.name}_{cal}_thr{thr:.2f}"
                        csv_path = os.path.join(args.cal_out_dir, base + "_overall.csv")
                        try:
                            rel.to_csv(csv_path, index=False)
                        except Exception:
                            pass
                        try:
                            glm.write_reliability_plot(
                                rel, os.path.join(args.cal_out_dir, base + "_overall.png")
                            )
                        except Exception:
                            pass
                        # Per-season reliability outputs
                        for s, g in preds_df.groupby("season"):
                            rel_s = glm.reliability_curve(g, n_bins=args.cal_bins)
                            csv_s = os.path.join(args.cal_out_dir, f"{base}_s{s}.csv")
                            try:
                                rel_s.to_csv(csv_s, index=False)
                            except Exception:
                                pass
                            try:
                                glm.write_reliability_plot(
                                    rel_s, os.path.join(args.cal_out_dir, f"{base}_s{s}.png")
                                )
                            except Exception:
                                pass
                        # Panel TeX (single) for selected config/cal/threshold
                        if (
                            args.panel_tex
                            and args.panel_config == cfg.name
                            and (args.panel_cal or cal) == cal
                            and (
                                args.panel_threshold is None
                                or abs(args.panel_threshold - thr) < 1e-9
                            )
                        ):
                            # Collect season images in order and write a figure panel
                            imgs = []
                            seasons_sorted = sorted(set(preds_df["season"]))
                            for s in seasons_sorted:
                                rel_img = os.path.join(
                                    args.cal_out_dir or ".",
                                    f"{base}_s{s}.png",
                                )
                                # Path relative to figures/out (two levels up from there)
                                rel_path = os.path.relpath(
                                    rel_img,
                                    start=os.path.join(
                                        ROOT, "analysis", "dissertation", "figures", "out"
                                    ),
                                )
                                imgs.append(rel_path.replace("\\", "/"))
                            title = f"Per-season reliability: {cfg.name}, {cal}, thr={thr:.2f}"
                            write_panel_tex(imgs, args.panel_tex, title)

                        # Collect images for dual panel if requested
                        if (
                            args.panel_combo_tex
                            and args.panel_config == cfg.name
                            and (
                                args.panel_threshold is None
                                or abs(args.panel_threshold - thr) < 1e-9
                            )
                            and args.panel2_cal is not None
                            and args.cal_out_dir
                        ):
                            seasons_sorted = sorted(set(preds_df["season"]))
                            imgs = []
                            for s in seasons_sorted:
                                rel_img = os.path.join(
                                    args.cal_out_dir,
                                    f"rel_{cfg.name}_{cal}_thr{thr:.2f}_s{s}.png",
                                )
                                rel_path = os.path.relpath(
                                    rel_img,
                                    start=os.path.join(
                                        ROOT, "analysis", "dissertation", "figures", "out"
                                    ),
                                )
                                imgs.append(rel_path.replace("\\", "/"))
                            if cal == (args.panel_cal or cal):
                                combo_left = imgs
                                combo_left_title = f"{cfg.name} — {cal}, thr={thr:.2f}"
                            if cal == args.panel2_cal:
                                combo_right = imgs
                                combo_right_title = f"{cfg.name} — {cal}, thr={thr:.2f}"
                metrics_df.insert(0, "config", cfg.name)
                metrics_df.insert(1, "calibration", cal)
                metrics_df.insert(2, "threshold", thr)
                out_rows.append(metrics_df)

    if not out_rows:
        print("No results produced.")
        return
    final = pd.concat(out_rows, ignore_index=True)
    print(final.head())
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        final.to_csv(args.output_csv, index=False)
    if args.tex:
        write_tex(final, args.tex)
    if args.tex_overall and overall_rows:
        overall = pd.concat(overall_rows, ignore_index=True)
        write_tex_overall(overall, args.tex_overall)

    # Emit dual panel TeX if requested and both sides collected
    if (
        args.panel_combo_tex
        and combo_left
        and combo_right
        and combo_left_title
        and combo_right_title
    ):
        write_dual_panel_tex(
            combo_left, combo_right, args.panel_combo_tex, combo_left_title, combo_right_title
        )


if __name__ == "__main__":
    main()
