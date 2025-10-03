#!/usr/bin/env python3
"""
Failure analysis utilities for weekly policy recommendations and realized trades.

Computes:
  - Share of zero-bet weeks by season
  - Primary gate attribution for zero-bet weeks (OPE vs Acceptance)
  - Writes LaTeX rows for Chapter 8 Table (zero weeks summary)

Expected inputs (CSV/Parquet):
  stakes: one row per (season, week, market) with at least
    - season (int or str)
    - week (int or str)
    - stake (float; 0 means no capital placed for that market)
    - accepted (bool/int; optional). If omitted, zero-week detection relies on stake==0 sums.
    - gate_primary (str; optional) in {"OPE","Acceptance","None","Unknown"}

  trades: optional, one row per executed trade for richer post-mortems (not required here)

Usage:
  python py/ops/report_failure_analysis.py \
    --stakes data/weekly_stakes.csv \
    --output-tex analysis/dissertation/results/zero_weeks_rows.tex

Notes:
  - Only the zero-week summary is emitted to LaTeX. Richer failure
    tagging can be added alongside if you pass a trades file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def read_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".json"}:
        return pd.read_json(p, lines=p.name.endswith(".jsonl"))
    raise ValueError(f"Unsupported input format: {p.suffix}")


def mk_zero_week_table(
    stakes: pd.DataFrame,
    season_col: str = "season",
    week_col: str = "week",
    stake_col: str = "stake",
    gate_col: str = "gate_primary",
) -> pd.DataFrame:
    # Normalize column names
    for col in (season_col, week_col, stake_col):
        if col not in stakes.columns:
            raise KeyError(f"Column '{col}' missing in stakes input")

    df = stakes.copy()
    # Pull relevant columns only
    cols = {season_col: "season", week_col: "week", stake_col: "stake"}
    df = df.rename(columns=cols)
    if gate_col in df.columns:
        df = df.rename(columns={gate_col: "gate"})
    else:
        df["gate"] = "Unknown"

    # Aggregate to week level
    grp = df.groupby(["season", "week"], as_index=False).agg(
        total_stake=("stake", "sum"),
        any_rows=("stake", "size"),
    )
    week_df = grp.merge(
        # primary gate per week by majority on zero-stake rows; fallback Unknown
        df.assign(is_zero=df["stake"].fillna(0) == 0)
        .query("is_zero")
        .groupby(["season", "week"])  # type: ignore[attr-defined]
        .agg(primary_gate=("gate", lambda s: s.value_counts().idxmax() if len(s) else "Unknown"))
        .reset_index(),
        on=["season", "week"],
        how="left",
    )

    week_df["is_zero_week"] = week_df["total_stake"].fillna(0) <= 0
    # Fill primary gate Unknown when non-zero week
    week_df.loc[~week_df["is_zero_week"], "primary_gate"] = "None"

    # Summarize by season
    def pct(x: int, d: int) -> int:
        return int(round(100 * x / d)) if d else 0

    rows = []
    for season, w in week_df.groupby("season"):
        weeks = int(w.shape[0])
        zero_weeks = int(w["is_zero_week"].sum())
        w0 = w[w["is_zero_week"]]
        ope = int((w0["primary_gate"].str.lower() == "ope").sum())
        acc = int((w0["primary_gate"].str.lower() == "acceptance").sum())
        rows.append(
            {
                "Season": season,
                "Weeks": weeks,
                "ZeroWeeksPct": pct(zero_weeks, weeks),
                "OPEGatePct": pct(ope, weeks),
                "AcceptancePct": pct(acc, weeks),
                "Notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values("Season")


def write_latex_rows(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, r in df.iterrows():
        line = f"{r['Season']} & {r['Weeks']} & {r['ZeroWeeksPct']} & {r['OPEGatePct']} & {r['AcceptancePct']} & {r['Notes']} \\\\"  # noqa: E501
        lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute zero-bet weeks table for LaTeX")
    ap.add_argument("--stakes", required=True, help="Path to stakes CSV/Parquet/JSON")
    ap.add_argument("--output-tex", default="analysis/dissertation/results/zero_weeks_rows.tex")
    ap.add_argument("--season-col", default="season")
    ap.add_argument("--week-col", default="week")
    ap.add_argument("--stake-col", default="stake")
    ap.add_argument("--gate-col", default="gate_primary")
    args = ap.parse_args()

    stakes = read_frame(args.stakes)
    tab = mk_zero_week_table(
        stakes,
        season_col=args.season_col,
        week_col=args.week_col,
        stake_col=args.stake_col,
        gate_col=args.gate_col,
    )
    write_latex_rows(tab, args.output_tex)
    print(f"Wrote LaTeX rows -> {args.output_tex}")


if __name__ == "__main__":
    main()

