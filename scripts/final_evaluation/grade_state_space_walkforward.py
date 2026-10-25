#!/usr/bin/env python3
"""
Grade the Kalman state-space walk-forward in models/state_space/predictions_historical.csv
against the closing line with market_benchmark.

The stored file (written 2026-02-11 by nfl_analytics/models/generate_state_space_predictions.py)
holds only P(home win) for 1,782 games of 2019 through 2025 week 7. Its generator ran one
StateSpaceRatings filter (models/state_space/config.json: q=2, r=15, home-field advantage 3,
initial variance 100, HFA variance 4) week by week from a cold start in 2019 (every team at 0):
it predicted every game of a week, then added process noise and updated on that week's
margins, so each prediction uses only earlier weeks. P(home win) = Phi(mu / sd), where mu is
the filter's expected home margin and sd^2 = var_home + var_away + hfa_variance + r^2.

Margins are recovered by running the same filter again on the same games, here from devdb01
in a read-only session: the refit must reproduce every stored probability (to 1e-12) before
its mu is used, and mu must equal sd * Phi^-1(stored p). Two differences from today's
devdb01 are needed for that: the stored run wrote the 2019 Raiders as LV (its game ids read
2019_01_DEN_LV), where devdb01 has OAK, and it did not have 2025_06_LA_BAL, so that game is
left out of the refit as it was left out of the original.

The predictions are sign-safe: P(home win) involves no line, mu is the expected HOME margin in
the nflverse sign of spread_close, and market_benchmark refuses predictions that correlate
<= 0 with the line. Grading is the market_benchmark CLI (run in-process) on regular-season
games, 2019-2024, 2025 (weeks 1-7 only) and 2019-2025.

Caveat: (q, r, HFA) are the grid point with the lowest 2024 Brier in
models/state_space/hyperparameter_search_results.json (27 points, filter run on 2019-2023 and
scored on 2024), so 2019-2024 is not out of sample for those three numbers. Only the 107
games of 2025 come after the tuning.

Outputs (docs/investigations/2026-10-final-evaluation/results/):
  state_space_walkforward_margins.csv            recovered margins, one row per stored game
  state_space_walkforward_reproduction.csv       refit vs stored file
  state_space_walkforward_<seasons>.{md,json}    market_benchmark reports
  state_space_walkforward_summary.csv

Usage:
  uv run --no-sync python scripts/final_evaluation/grade_state_space_walkforward.py
  (DB connection from the DB_* environment variables; see nfl_analytics/utils/db_config.py)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import psycopg
from scipy.stats import norm

from nfl_analytics.evaluation.market_benchmark.__main__ import main as benchmark_cli
from nfl_analytics.models.state_space import StateSpaceRatings
from nfl_analytics.utils.db_config import get_db_config

REPO = Path(__file__).resolve().parents[2]
STORED = REPO / "models/state_space/predictions_historical.csv"
CONFIG = REPO / "models/state_space/config.json"
OUT_DIR = REPO / "docs/investigations/2026-10-final-evaluation/results"
PREFIX = "state_space_walkforward"
WINDOWS = ("2019-2024", "2025", "2019-2025")
READ_ONLY = "-c default_transaction_read_only=on"
STORED_CODES = {"OAK": "LV"}  # the stored run's team codes, where they differ from devdb01
TOLERANCE = 1e-12

GAMES_SQL = """
SELECT game_id, season, week, game_type, home_team, away_team, home_score, away_score
FROM games
WHERE season BETWEEN 2019 AND 2025 AND home_score IS NOT NULL AND away_score IS NOT NULL
"""


def fetch_games() -> pd.DataFrame:
    with psycopg.connect(**get_db_config(), options=READ_ONLY) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            if cur.fetchone()[0] != "on":
                raise RuntimeError("could not open a read-only session")
            cur.execute(GAMES_SQL)
            cols = [d.name for d in cur.description]
            games = pd.DataFrame(cur.fetchall(), columns=cols)
    games["margin"] = games["home_score"] - games["away_score"]
    # the id the generator built: f"{season}_{week:02d}_{away}_{home}" with its team codes
    home = games["home_team"].replace(STORED_CODES)
    away = games["away_team"].replace(STORED_CODES)
    games["stored_game_id"] = [
        f"{s}_{w:02d}_{a}_{h}" for s, w, a, h in zip(games["season"], games["week"], away, home)
    ]
    games["stored_home"], games["stored_away"] = home, away
    return games


def refit(games: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The generator's loop: predict a whole week, then predict_step(), then update."""
    model = StateSpaceRatings(
        q=cfg["q"],
        r=cfg["r"],
        hfa_init=cfg["hfa_init"],
        init_variance=cfg["init_variance"],
        hfa_variance=cfg["hfa_variance"],
    )
    rows = []
    for _, week in games.sort_values(["season", "week"]).groupby(["season", "week"], sort=True):
        for g in week.itertuples():
            mu, sd = model.predict_margin(g.stored_home, g.stored_away)
            p = model.predict_prob_home_win(g.stored_home, g.stored_away)
            rows.append((g.stored_game_id, mu, sd, p))
        model.predict_step()
        for g in week.itertuples():
            model.update_game(g.stored_home, g.stored_away, g.margin)
    return pd.DataFrame(rows, columns=["stored_game_id", "mu", "sd", "p_refit"])


def run_cli(argv: list[str]) -> tuple[dict, str]:
    outputs = []
    for extra in (["--json"], []):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = benchmark_cli([*argv, *extra])
        if rc != 0:
            raise RuntimeError(f"market_benchmark {' '.join(argv)} returned {rc}")
        outputs.append(buf.getvalue())
    return json.loads(outputs[0]), outputs[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    os.environ["PGOPTIONS"] = READ_ONLY  # for the market_benchmark --db connections
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stored = pd.read_csv(STORED)
    cfg = json.loads(CONFIG.read_text())
    games = fetch_games()
    assert stored["game_id"].is_unique and games["stored_game_id"].is_unique
    unmatched = sorted(set(stored["game_id"]) - set(games["stored_game_id"]))
    if unmatched:
        print(f"stored games not in devdb01: {unmatched}", file=sys.stderr)
        return 1
    last = stored.sort_values(["season", "week"]).iloc[-1]
    upto = (games["season"] < last["season"]) | (
        (games["season"] == last["season"]) & (games["week"] <= last["week"])
    )
    absent = sorted(games.loc[upto & ~games["stored_game_id"].isin(stored["game_id"]), "game_id"])
    used = games[games["stored_game_id"].isin(stored["game_id"])]

    fit = refit(used, cfg)
    x = stored.rename(columns={"game_id": "stored_game_id", "prediction": "p_stored"}).merge(
        fit, on="stored_game_id", validate="one_to_one"
    )
    x = x.merge(
        used[["stored_game_id", "game_id", "game_type", "home_team", "away_team"]],
        on="stored_game_id",
        suffixes=("_stored", ""),
    )
    p_diff = float((x["p_refit"] - x["p_stored"]).abs().max())
    mu_diff = float((x["sd"] * norm.ppf(x["p_stored"]) - x["mu"]).abs().max())
    # the stored target is the home win (margin > 0) of the same game
    margin = used.set_index("stored_game_id")["margin"].loc[x["stored_game_id"]].to_numpy()
    target_ok = bool((x["target"].to_numpy() == (margin > 0)).all())
    checks = pd.DataFrame(
        [
            ("stored rows", len(stored), ""),
            ("stored rows matched to devdb01 games", len(x), "2019 Raiders ids read LV"),
            (
                "devdb01 games of the window absent from the stored run",
                len(absent),
                " ".join(absent),
            ),
            ("max |P(home win) refit - stored|", f"{p_diff:.2g}", f"must be <= {TOLERANCE}"),
            ("max |sd * Phi^-1(stored p) - refit mu|", f"{mu_diff:.2g}", "points of margin"),
            ("stored target == (margin > 0) for every row", target_ok, ""),
            ("pred_home_margin range", f"{x['mu'].min():+.2f} to {x['mu'].max():+.2f}", ""),
        ],
        columns=["check", "value", "note"],
    )
    checks.to_csv(args.out_dir / f"{PREFIX}_reproduction.csv", index=False)
    print(checks.to_string(index=False))
    if p_diff > TOLERANCE or not target_ok:
        print("refit does not reproduce the stored predictions: not graded", file=sys.stderr)
        return 1

    out = x.assign(pred_home_margin=x["mu"], pred_sd=x["sd"], pred_home_win_prob=x["p_stored"])
    out = out.sort_values(["season", "week", "game_id"])[
        [
            "game_id",
            "stored_game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "pred_home_margin",
            "pred_sd",
            "pred_home_win_prob",
        ]
    ]
    margins_csv = args.out_dir / f"{PREFIX}_margins.csv"
    out.to_csv(margins_csv, index=False)  # full precision: the stored p round-trips exactly
    print(f"wrote {margins_csv.name}: {len(out)} games")

    rows = []
    for seasons in WINDOWS:
        argv = ["--predictions-csv", str(margins_csv), "--db", "--seasons", seasons]
        result, md = run_cli(argv)
        # the CLI names the run after the CSV path; keep the outputs free of local paths
        result["name"] = margins_csv.name
        md = md.replace(str(margins_csv), margins_csv.name)
        (args.out_dir / f"{PREFIX}_{seasons}.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out_dir / f"{PREFIX}_{seasons}.md").write_text(md)
        a, s = result["ats"], result["spread_accuracy"]
        reg, wp = result["information_regression"], result["win_prob_skill"]
        rows.append(
            {
                "seasons": seasons,
                "games": result["games"],
                "record": f"{a['wins']}-{a['losses']}-{a['pushes']}",
                "win_rate": a["win_rate"],
                "win_rate_lo": a["win_rate_ci95"][0],
                "win_rate_hi": a["win_rate_ci95"][1],
                "roi": a["roi"],
                "roi_lo": a["roi_ci95"][0],
                "roi_hi": a["roi_ci95"][1],
                "model_mae": s["model_mae"],
                "market_mae": s["market_mae"],
                "dm_p": s["dm_p_value_two_sided"],
                "model_log_loss": wp["model_log_loss"],
                "market_log_loss": wp["market_log_loss"],
                "log_loss_diff_lo": wp["log_loss_diff_ci95"][0],
                "log_loss_diff_hi": wp["log_loss_diff_ci95"][1],
                "info_coef": reg["model_gap_coef"],
                "info_lo": reg["model_gap_ci95"][0],
                "info_hi": reg["model_gap_ci95"][1],
                "pred_line_corr": result["pred_line_corr"],
                "warnings": " | ".join(result["warnings"]),
            }
        )
        print(md)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / f"{PREFIX}_summary.csv", index=False, float_format="%.6g")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
