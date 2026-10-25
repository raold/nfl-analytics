#!/usr/bin/env python3
"""
Grade Game Outcome v2's 2025 predictions against the closing market (the canonical table).

Input: data/game_outcome_v2_predictions_2025.csv, written by export_game_outcome_v2_2025.py
(285 games, one prediction run created 2026-02-12, after the season; the market and score
columns are nflverse schedule data). No database is needed.

Definitions (every number in the final evaluation's v2 tables uses them):
  margin    home_score - away_score
  line      spread_close, nflverse spread_line: the expected HOME margin, positive = home favored
  pred      pred_home_spread, the model's expected HOME margin (same sign as the line)
  edge      pred - line, taken to 0.1 point (pred has one decimal, the line is a half point)
  cover     the home team covers iff margin - line > 0; margin == line is a push
  ATS bet   the side the edge favours (home if edge > 0, away if edge < 0); it wins when that
            side covers. Pushes are refunded and left out of win rates and ROI.
  win rate  wins / (wins + losses), Wilson 95% interval
  ROI       profit per settled bet. At -110 a win pays +100/110 and a loss costs 1 (break-even
            52.38%). "Listed" uses each side's nflverse closing spread price instead.
            Intervals resample whole weeks (2,000 draws, seed 0), as market_benchmark does.
  MAE       mean |margin - pred| against mean |margin - line|; Diebold-Mariano p (two-sided)
  winners   straight-up picks; the tie (2025_04_GB_DAL, 40-40) is left out. A picker with no
            pick for a game (predicted margin exactly 0, or both moneylines -110) is counted
            as a miss in "correct / decided"; McNemar's exact test compares each picker with
            the closing-spread favorite on the same games
  log-loss  model pred_home_win_prob against the closing moneyline with the vig removed
            proportionally and by Shin's method; ties left out; interval on the per-game
            difference by week blocks
  info      market_benchmark.information_regression: margin = a + b * line + c * (pred - line),
            week-clustered errors; c is what the model knows beyond the line

The regular-season row at |edge| >= 1 is market_benchmark.run_benchmark itself, the code
behind `python -m nfl_analytics.evaluation.market_benchmark --predictions-csv ... --db`.
--cli-db and --cli-nflverse-csv run that CLI and check that it agrees.

Outputs (docs/investigations/2026-10-final-evaluation/results/):
  game_outcome_v2_2025.md                         every table, as rendered in the README
  game_outcome_v2_2025.csv                        the same numbers, one row per metric
  game_outcome_v2_2025_market_benchmark_reg.md    market_benchmark report, regular season
  game_outcome_v2_2025_market_benchmark_reg.json

Usage:
  uv run --no-sync python scripts/final_evaluation/grade_game_outcome_v2_2025.py \\
      [--cli-db] [--cli-nflverse-csv games.csv]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nfl_analytics.evaluation.market_benchmark import (
    prepare_market,
    run_benchmark,
    to_json,
    to_markdown,
)
from nfl_analytics.evaluation.market_benchmark.__main__ import main as benchmark_cli
from nfl_analytics.evaluation.market_benchmark.metrics import (
    ats_summary,
    grade_ats,
    information_regression,
    spread_accuracy,
    wilson_interval,
    win_prob_skill,
)
from nfl_analytics.utils.spread_convention import model_edge

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data/game_outcome_v2_predictions_2025.csv"
OUT_DIR = REPO / "docs/investigations/2026-10-final-evaluation/results"
PREFIX = "game_outcome_v2_2025"
PRED, PROB = "pred_home_spread", "pred_home_win_prob"
N_BOOT, SEED = 2000, 0  # market_benchmark's week-block bootstrap settings
MARKETS = {"proportional": "market_home_win_prob", "Shin": "market_home_win_prob_shin"}

ROWS: list[dict] = []  # tidy output: one row per (section, slice, metric)


def record(section: str, slice_: str, metric: str, value, ci=(None, None), n=None, note=""):
    lo, hi = ci
    ROWS.append(
        {
            "section": section,
            "slice": slice_,
            "n": n,
            "metric": metric,
            "value": value,
            "ci_low": lo,
            "ci_high": hi,
            "note": note,
        }
    )


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    # kickoff in the run's own table is the game date at midnight ET; kickoff_utc is the
    # true kickoff, which market_benchmark compares with prediction_timestamp
    m = prepare_market(df.drop(columns=["kickoff"]).rename(columns={"kickoff_utc": "kickoff"}))
    m["edge"] = model_edge(m[PRED], m["spread_close"]).round(1)
    # the thresholds use the exact edge; check that the float edge market_benchmark computes
    # puts every game on the same side of each threshold
    raw = model_edge(m[PRED], m["spread_close"]).abs()
    for t in (1.0, 2.0):
        assert ((raw >= t) == (m["edge"].abs() >= t)).all()
        assert ((raw > t) == (m["edge"].abs() > t)).all()
    assert (m["edge"] != 0).all(), "every game has an edge, so every game is a bet"
    assert m["game_id"].is_unique and len(m) == 285
    return m


def week_block_ci(df: pd.DataFrame, values: np.ndarray) -> tuple[float, float]:
    """95% interval for the mean of per-game values, resampling whole (season, week) blocks."""
    blocks = df.assign(_v=values).groupby(["season", "week"])["_v"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    means = blocks[draws, 0].sum(axis=1) / blocks[draws, 1].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# --- against the spread ---------------------------------------------------------------------


def ats_slice(m: pd.DataFrame, keep: pd.Series, label: str, definition: str) -> dict:
    """ATS record of the bets in ``keep``: ROI at -110 and at the listed closing prices."""
    sub = m.loc[keep]
    flat = sub.assign(home_spread_odds=np.nan, away_spread_odds=np.nan)  # grade_ats: NaN = -110
    s110 = ats_summary(grade_ats(flat, PRED, min_edge=0), n_boot=N_BOOT, seed=SEED)
    slist = ats_summary(grade_ats(sub, PRED, min_edge=0), n_boot=N_BOOT, seed=SEED)
    assert (s110["wins"], s110["losses"]) == (slist["wins"], slist["losses"])
    placed = s110["wins"] + s110["losses"] + s110["pushes"]
    row = {
        "slice": label,
        "definition": definition,
        "games": int(len(sub)),
        "placed": placed,
        "record": f"{s110['wins']}-{s110['losses']}-{s110['pushes']}",
        "win_rate": s110["win_rate"],
        "win_ci": s110["win_rate_ci95"],
        "p_breakeven": s110["p_value_beats_breakeven"],
        "roi_110": s110["roi"],
        "roi_110_ci": s110["roi_ci95"],
        "roi_listed": slist["roi"],
        "roi_listed_ci": slist["roi_ci95"],
        "breakeven_listed": slist["breakeven_win_rate"],
    }
    n = placed
    record("ats", label, "bets placed (incl. pushes)", placed, n=n, note=definition)
    record("ats", label, "wins", s110["wins"], n=n)
    record("ats", label, "losses", s110["losses"], n=n)
    record("ats", label, "pushes", s110["pushes"], n=n)
    record("ats", label, "win rate", s110["win_rate"], s110["win_rate_ci95"], n=n)
    record("ats", label, "p(win rate > 52.38%), one-sided", s110["p_value_beats_breakeven"], n=n)
    record("ats", label, "ROI at -110", s110["roi"], s110["roi_ci95"], n=n)
    record("ats", label, "ROI at listed closing prices", slist["roi"], slist["roi_ci95"], n=n)
    record("ats", label, "mean break-even at listed prices", slist["breakeven_win_rate"], n=n)
    return row


def ats_rows(m: pd.DataFrame) -> list[dict]:
    e = m["edge"].abs()
    reg = m["game_type"] == "REG"
    every = pd.Series(True, index=m.index)
    return [
        ats_slice(m, every, "All 285 games", "every game, side of the edge"),
        ats_slice(m, e >= 1, "|edge| >= 1", "market_benchmark's default threshold"),
        ats_slice(m, e >= 2, "|edge| >= 2", ""),
        ats_slice(m, e > 2, "|edge| > 2 (old headline)", "strictly above 2 points"),
        ats_slice(m, reg, "Regular season, every game", ""),
        ats_slice(m, reg & (e >= 1), "Regular season, |edge| >= 1", "the market_benchmark run"),
        ats_slice(m, ~reg, "Playoffs, every game", "13 games"),
        ats_slice(m, ~reg & (m["game_type"] != "SB"), "Playoffs before the Super Bowl", ""),
    ]


# --- accuracy, straight up, probabilities, information ----------------------------------------


def mae_row(sub: pd.DataFrame, label: str) -> dict:
    s = spread_accuracy(sub, PRED)
    err_model = (sub["home_margin"] - sub[PRED]).abs()
    err_line = (sub["home_margin"] - sub["spread_close"]).abs()
    closer = float((err_model < err_line).mean())  # equal errors count as not closer
    record("mae", label, "model MAE", s["model_mae"], n=s["games"])
    record("mae", label, "closing-line MAE", s["market_mae"], n=s["games"])
    record("mae", label, "Diebold-Mariano p, two-sided", s["dm_p_value_two_sided"], n=s["games"])
    record("mae", label, "share of games with the model closer than the line", closer)
    return {"slice": label, **s, "closer": closer}


PICKERS = {
    "model: P(home win) > 0.5": lambda d: np.sign(d[PROB] - 0.5),
    "model: sign of predicted margin": lambda d: np.sign(d[PRED]),
    "closing-spread favorite": lambda d: np.sign(d["spread_close"]),
    "de-vigged moneyline favorite": lambda d: np.sign(d["market_home_win_prob"] - 0.5),
}


def straight_up(sub: pd.DataFrame, label: str) -> list[dict]:
    d = sub[sub["home_margin"] != 0]
    y = np.sign(d["home_margin"])
    right = {name: (pick(d) == y) & (pick(d) != 0) for name, pick in PICKERS.items()}
    fav = right["closing-spread favorite"]
    rows = []
    for name, pick in PICKERS.items():
        made = int((pick(d) != 0).sum())
        k = int(right[name].sum())
        n = len(d)
        # exact McNemar test against the closing favorite on the same games
        b = int((right[name] & ~fav).sum())
        c = int((~right[name] & fav).sum())
        p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else math.nan
        rows.append(
            {
                "slice": label,
                "picker": name,
                "decided": n,
                "picks": made,
                "correct": k,
                "rate": k / n,
                "ci": wilson_interval(k, n),
                "rate_of_picks": k / made if made else math.nan,
                "only_this_right": b,
                "only_favorite_right": c,
                "mcnemar_p": p,
            }
        )
        record("straight_up", label, f"{name}: correct of decided", k / n, wilson_interval(k, n), n)
        record("straight_up", label, f"{name}: picks made", made, n=n)
        record("straight_up", label, f"{name}: correct of picks made", rows[-1]["rate_of_picks"])
        if name != "closing-spread favorite":
            record("straight_up", label, f"{name}: McNemar p vs favorite", p, n=b + c)
    return rows


def probability(sub: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for vig, col in MARKETS.items():
        w = win_prob_skill(sub, PROB, col, n_boot=N_BOOT, seed=SEED)
        d = sub.dropna(subset=[PROB, col])  # the games win_prob_skill scores
        d = d[d["home_margin"] != 0]
        y = (d["home_margin"] > 0).astype(int).to_numpy()
        pm, pk = d[PROB].to_numpy(float), d[col].to_numpy(float)
        brier_ci = week_block_ci(d, (pm - y) ** 2 - (pk - y) ** 2)
        row = {"slice": label, "devig": vig, **w, "brier_diff_ci95": brier_ci}
        rows.append(row)
        n = w["games"]
        record("probability", label, "model log-loss", w["model_log_loss"], n=n)
        record("probability", label, f"market log-loss ({vig})", w["market_log_loss"], n=n)
        record(
            "probability",
            label,
            f"log-loss model - market ({vig})",
            w["model_log_loss"] - w["market_log_loss"],
            w["log_loss_diff_ci95"],
            n,
        )
        record("probability", label, "model Brier", w["model_brier"], n=n)
        record("probability", label, f"market Brier ({vig})", w["market_brier"], n=n)
        record(
            "probability",
            label,
            f"Brier model - market ({vig})",
            w["model_brier"] - w["market_brier"],
            brier_ci,
            n,
        )
    return rows


def information(sub: pd.DataFrame, label: str) -> dict:
    r = information_regression(sub, PRED)
    n = len(sub)
    record(
        "information",
        label,
        "coef on (pred - line)",
        r["model_gap_coef"],
        r["model_gap_ci95"],
        n,
        r["note"] or f"{r['clusters']} week clusters",
    )
    record("information", label, "p-value, coef on (pred - line)", r["model_gap_p_value"], n=n)
    record("information", label, "coef on line", r["line_coef"], n=n)
    return {"slice": label, "games": n, **r}


# --- rendering --------------------------------------------------------------------------------


def cell(text: str) -> str:
    """A Markdown table cell: '|' inside a cell would end it."""
    return text.replace("|", "\\|")


def pct(x, signed=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def ci(pair, signed=False, digits=None) -> str:
    lo, hi = pair
    if lo is None or math.isnan(lo):
        return "n/a"
    if digits is not None:
        return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"
    return f"[{pct(lo, signed)}, {pct(hi, signed)}]"


def render(ats, mae, su, prob, info, bench) -> str:
    out = ["# Game Outcome v2, 2025: graded against the closing market", ""]
    out += [
        "Generated by `scripts/final_evaluation/grade_game_outcome_v2_2025.py` from "
        "`data/game_outcome_v2_predictions_2025.csv`. Definitions are in the script's docstring.",
        "",
        "## Spread accuracy",
        "",
        "| Games | n | Model MAE | Closing-line MAE | Diebold-Mariano p | Model closer |",
        "|---|---|---|---|---|---|",
    ]
    for r in mae:
        out.append(
            f"| {r['slice']} | {r['games']} | {r['model_mae']:.2f} | {r['market_mae']:.2f} | "
            f"{r['dm_p_value_two_sided']:.3f} | {pct(r['closer'])} |"
        )
    out += [
        "",
        "## Against the spread",
        "",
        "| Bets | Placed | W-L-P | Win rate [95% CI] | p vs 52.38% | ROI at -110 [95% CI] | "
        "ROI at listed prices [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ats:
        out.append(
            f"| {cell(r['slice'])} | {r['placed']} | {r['record']} | {pct(r['win_rate'])} "
            f"{ci(r['win_ci'])} | {r['p_breakeven']:.2f} | {pct(r['roi_110'], True)} "
            f"{ci(r['roi_110_ci'], True)} | {pct(r['roi_listed'], True)} "
            f"{ci(r['roi_listed_ci'], True)} |"
        )
    out += [
        "",
        "## Straight up (ties left out)",
        "",
        "| Games | Picker | Correct / decided [95% CI] | Picks made | Correct / picks | "
        "Only this right / only the favorite right | McNemar p vs favorite |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in su:
        split, p = f"{r['only_this_right']} / {r['only_favorite_right']}", r["mcnemar_p"]
        p_txt = "n/a (same picks)" if math.isnan(p) else f"{p:.2g}"
        if r["picker"] == "closing-spread favorite":
            split, p_txt = "", ""
        out.append(
            f"| {r['slice']} | {r['picker']} | {r['correct']}/{r['decided']} = {pct(r['rate'])} "
            f"{ci(r['ci'])} | {r['picks']} | {pct(r['rate_of_picks'])} | {split} | {p_txt} |"
        )
    out += [
        "",
        "## Win probability against the de-vigged closing moneyline (ties left out)",
        "",
        "Coin-flip references: log-loss ln 2 = 0.6931, Brier 0.2500.",
        "",
        "| Games | n | De-vig | Log-loss model / market | Difference [95% CI] | "
        "Brier model / market | Difference [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in prob:
        out.append(
            f"| {r['slice']} | {r['games']} | {r['devig']} | {r['model_log_loss']:.4f} / "
            f"{r['market_log_loss']:.4f} | {r['model_log_loss'] - r['market_log_loss']:+.4f} "
            f"{ci(r['log_loss_diff_ci95'], digits=4)} | {r['model_brier']:.4f} / "
            f"{r['market_brier']:.4f} | {r['model_brier'] - r['market_brier']:+.4f} "
            f"{ci(r['brier_diff_ci95'], digits=4)} |"
        )
    out += [
        "",
        "## Information beyond the line",
        "",
        "OLS margin = a + b * line + c * (pred - line), errors clustered by week.",
        "",
        "| Games | n | c (95% CI) | p | b (line) | Weeks |",
        "|---|---|---|---|---|---|",
    ]
    for r in info:
        out.append(
            f"| {r['slice']} | {r['games']} | {r['model_gap_coef']:+.3f} "
            f"{ci(r['model_gap_ci95'], digits=3)} | {r['model_gap_p_value']:.2f} | "
            f"{r['line_coef']:.3f} | {r['clusters']} |"
        )
    out += [
        "",
        "## market_benchmark report, regular season, |edge| >= 1",
        "",
        "Saved as `game_outcome_v2_2025_market_benchmark_reg.md` / `.json`:",
        "",
        bench.rstrip(),
        "",
    ]
    return "\n".join(out)


# --- CLI cross-check --------------------------------------------------------------------------


def comparable(result: dict | str) -> dict:
    """market_benchmark JSON without the name (the CLI names the run after the CSV path)."""
    r = json.loads(result if isinstance(result, str) else to_json(result))
    r.pop("name", None)
    return r


def cli_check(source: list[str], expected: dict) -> bool:
    argv = [
        "--predictions-csv",
        str(CSV),
        "--pred-col",
        PRED,
        "--seasons",
        "2025",
        "--json",
        *source,
    ]
    shown = " ".join(a.replace(str(REPO) + "/", "") for a in argv)
    if "--db" in source:  # libpq reads PGOPTIONS at connect time: a read-only session
        os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = benchmark_cli(argv)
    if rc != 0:
        print(f"CLI check FAILED: market_benchmark {shown} returned {rc}")
        return False
    same = comparable(buf.getvalue()) == expected
    print(f"CLI check: market_benchmark {shown}: {'identical' if same else 'DIFFERENT'}")
    return same


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cli-db", action="store_true", help="also run the CLI with --db")
    ap.add_argument("--cli-nflverse-csv", type=Path, help="also run the CLI on this games.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    m = load()
    reg, post = m[m["game_type"] == "REG"], m[m["game_type"] != "REG"]
    slices = [(m, "All games"), (reg, "Regular season"), (post, "Playoffs")]
    ats = ats_rows(m)
    # the dissertation's playoff row (MAE 8.43) leaves out the Super Bowl
    pre_sb = (post[post["game_type"] != "SB"], "Playoffs before the Super Bowl")
    mae = [mae_row(s, label) for s, label in [*slices, pre_sb]]
    su = [r for s, label in [*slices, pre_sb] for r in straight_up(s, label)]
    prob = [r for s, label in slices for r in probability(s, label)]
    info = [information(s, label) for s, label in slices[:2]]  # 13 playoff games: 4 weeks

    bench = run_benchmark(reg, "Game Outcome v2, 2025 regular season", PRED, min_edge=1.0)
    bench_md = to_markdown(bench)
    a, label = bench["ats"], "regular season, |edge| >= 1"
    for key in ("wins", "losses", "pushes"):
        record("market_benchmark", label, key, a[key], n=a["bets"])
    record("market_benchmark", label, "win rate", a["win_rate"], a["win_rate_ci95"], a["bets"])
    record("market_benchmark", label, "ROI at listed closing prices", a["roi"], a["roi_ci95"])
    record(
        "market_benchmark",
        label,
        "retroactive share",
        bench["retroactive_share"],
        note="prediction_timestamp (the run's created_at) at or after kickoff",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{PREFIX}_market_benchmark_reg.md").write_text(bench_md)
    (args.out_dir / f"{PREFIX}_market_benchmark_reg.json").write_text(to_json(bench) + "\n")
    report = render(ats, mae, su, prob, info, bench_md)
    (args.out_dir / f"{PREFIX}.md").write_text(report)
    pd.DataFrame(ROWS).to_csv(args.out_dir / f"{PREFIX}.csv", index=False, float_format="%.6g")
    print(report)
    print(f"wrote {PREFIX}.md, {PREFIX}.csv, {PREFIX}_market_benchmark_reg.{{md,json}}")

    ok = True
    expected = comparable(bench)
    if args.cli_db:
        ok &= cli_check(["--db"], expected)
    if args.cli_nflverse_csv:
        ok &= cli_check(["--nflverse-csv", str(args.cli_nflverse_csv.resolve())], expected)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
