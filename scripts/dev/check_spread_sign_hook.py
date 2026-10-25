#!/usr/bin/env python3
"""Self-test for the spread-sign-convention pre-commit hook.

`pre-commit run spread-sign-convention --all-files` only proves that the tree has no hits; it
would also pass if the pattern matched nothing. This script compiles the hook's entry the way
pre-commit's pygrep does (bytes, line by line) and checks it against known-wrong lines (mostly
from the 2026-10 sign survey) and correct ones (nflverse sign: spread_close > 0 = home favored).

Usage: uv run python scripts/dev/check_spread_sign_hook.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"

# Sportsbook-style math on nflverse lines: the hook must flag each of these.
WRONG = [
    "outcome = 1 if (margin + spread) > 0 else 0",  # backtests/realistic_backtest.py:233
    "adjusted_margin = mean_margin + spread",  # models/state_space.py:228 before 8fb5425
    "actuals_ats.append(1.0 if (margin_true + spread) > 0 else 0.0)",  # state_space.py:278
    "    home_score - away_score + spread > 0",  # state_space.py docstring before the fix
    "spread_target = if_else(margin + spread_close > 0, 1, 0)",  # notebooks/10_*.qmd:82
    "CASE WHEN g.home_score + g.spread_close > g.away_score THEN 1 ELSE 0 END",  # migration 005
    "CASE WHEN (g.home_score - g.away_score) + g.spread_close > 0 THEN 1",  # migration 100
    'df["home_cover"] = (df["home_score"] - df["away_score"] + df["spread_close"]) > 0',
    "home_cover <- df$margin + df$spread_line > 0",
    "covered = score_diff + spread_line > 0",
    'ok = diff + d["spread_close"] > 0',
    "hit = margin + vegas_spread > 0",
    "c=margin+spread_close>0",
    'c = margin + float(row["spread_close"]) > 0',
    "covers = margins + spreads > 0",
    "NEW.spread_cover := (NEW.actual_spread + NEW.sportsbook_spread) > 0;",  # migration 048
    "covered = actual + spread_close > 0",
    "cover_prob = mean(margin > -spread_close),",  # notebooks/11_*.qmd:83
    "push_prob = mean(margin == -spread_close)",  # notebooks/11_*.qmd:84
    "covered = home_margin >= -spread_close",
    "CASE WHEN (g.home_score - g.away_score) > -g.spread_close THEN 1 END",
    'c = d["margin"] > -d["spread_close"]',
    "CASE WHEN spread_close < 0 THEN 1 ELSE 0 END as home_is_favorite,",  # feature_loader:131
    'home_favorite = df["spread_close"] < 0',
    "is_home_fav <- spread_line < 0",
    "home_favored = vegas_spread <= 0",
    'games["implied_home_total"] = games["total_close"] / 2 - games["spread_close"] / 2',
    "lambda_home <- total_line/2 - spread_line/2",  # notebooks/11_*.qmd:51
    "mu_home = (total_close - spread_close) / 2",
    "home_mean = total_close / 2.0 - spread_close / 2.0",
    # the sportsbook cover rule with the operands reversed (2026-10 review)
    "home_cover = (spread_close + margin) > 0",
    'df["cover"] = df["spread_close"] + df["home_margin"] > 0',
    "covered = np.where(g.spread_close + g.home_score - g.away_score > 0, 1, 0)",
    "CASE WHEN g.spread_close + g.home_score > g.away_score THEN 1 ELSE 0 END",
    "home_cover <- df$spread_line + df$margin > 0",
    "hit = vegas_spread + score_diff >= 0",
    'c = float(row["spread_close"]) + row["actual_margin"] > 0',
]

# Correct nflverse math, away-side forms and look-alikes: the hook must pass each of these.
RIGHT = [
    "    return home_margin - spread_close",
    "CASE WHEN (g.home_score - g.away_score) - g.spread_close > 0 THEN 1",
    "CASE WHEN rd.team = g.home_team THEN g.spread_close ELSE -g.spread_close END",
    "    NEW.spread_edge := NEW.predicted_spread - NEW.sportsbook_spread;",
    'games["implied_away_total"] = games["total_close"] / 2 - games["spread_close"] / 2',
    'games["implied_home_total"] = games["total_close"] / 2 + games["spread_close"] / 2',
    "mu_away = (total_close - spread_close) / 2",
    'favorite = np.where(df["spread_close"] < 0, df["away_team"], df["home_team"])',
    "away_favored = spread_close < 0",
    "home_underdog = spread_close < 0",
    'fav_ok = ((v["spread_close"] > 0) == won) & decided & fav',
    "away_covers = away_margin > -spread_close",
    "CASE WHEN (g.away_score - g.home_score) > -g.spread_close THEN 1 END",
    "        if score_diff <= -8:  # Down by 2 scores",
    'week5 = week5.assign(pred_home_margin=week5["spread_close"] + 0.5)',
    'results["predicted_home_margin"] = results["spread_close"] + predicted_cover_margin',
    "upper = pred_margin + spread_std * 1.96",
    "            home_win = 1 if actual_spread > 0 else 0",
    '    mkt.add_argument("--market-csv", help="CSV export of games (scores + spread_close)")',
    'old = (m["margin"] + m["spread_close"] > 0).astype(int)  # sign-ok: reproduces the old rule',
    # reversed operands that are correct: the away side covers, or the line is negated
    "away_covers = spread_close - home_margin > 0",
    'away_cover = df["spread_close"] - df["home_margin"] > 0',
    "home_cover = -spread_close + home_margin > 0",
    'home_cover = (-df["spread_close"] + df["home_margin"]) > 0',
    "CASE WHEN g.spread_close - (g.home_score - g.away_score) > 0 THEN 1 END",
]


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    hook = next(h for h in hooks if h["id"] == "spread-sign-convention")
    entry = hook["entry"]
    if hook["language"] != "pygrep" or not entry.startswith("^(?!.*sign-ok)"):
        print("spread-sign-convention must be pygrep with an entry starting ^(?!.*sign-ok)")
        return 1
    pattern = re.compile(entry.encode())  # as pre_commit.languages.pygrep compiles it
    missed = [line for line in WRONG if not pattern.search(line.encode())]
    flagged = [line for line in RIGHT if pattern.search(line.encode())]
    for line in missed:
        print(f"not flagged (should be): {line}")
    for line in flagged:
        print(f"flagged (should not be): {line}")
    print(
        f"spread-sign-convention: {len(WRONG) - len(missed)}/{len(WRONG)} wrong lines flagged, "
        f"{len(RIGHT) - len(flagged)}/{len(RIGHT)} correct lines passed"
    )
    return 1 if missed or flagged else 0


if __name__ == "__main__":
    sys.exit(main())
