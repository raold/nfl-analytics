#!/usr/bin/env python3
"""
Locate and re-check the edge and headline claims listed in the retraction ledger
(docs/investigations/2026-10-final-evaluation/retraction_ledger.md).

Outputs (docs/investigations/2026-10-final-evaluation/results/):
  claims_locations.csv  every line in README.md, PROJECT_STATUS.md, CLAUDE.md, docs/, the
                        dissertation sources, models/**/*.json and data/*.json (plus a few named
                        source artifacts) that matches a claim's patterns, with whether a
                        dissertation file is part of the main.tex build and whether the number
                        is already flagged there (the line says retracted, invalid, sign bug ...,
                        or the file is a correction document such as ERRATA.md or POSTMORTEMS.md)
  claims_summary.csv    one row per claim: where it appears, ranked by prominence
  claims_checks.csv     re-computed quantities, one row each, from devdb01 (read-only session)
                        and committed files
  claims_checks_nflverse.csv
                        the rows that need nflverse schedules (written only with
                        --nflverse-games, so a run without it leaves the file alone)

Grading follows nfl_analytics/utils/spread_convention.py: games.spread_close is the expected
home margin (positive = home favored); home covers iff margin - spread_close > 0; pushes are
excluded; model edge = predicted home margin - spread_close. ROI is at -110 (+100/110 per win,
-1 per loss) unless a row says otherwise.

Usage:
  uv run --no-sync python scripts/final_evaluation/check_claims.py
  uv run --no-sync python scripts/final_evaluation/check_claims.py --nflverse-games games.csv
  (--nflverse-games adds the 1999-2025 wind check from nflverse schedules; DB connection from
  the DB_* environment variables, see nfl_analytics/utils/db_config.py)

The re-checks read artifacts that the 2026-10 prune removed from the working tree (simulated
A/B results, ensemble backtests, state-space metric JSONs, ...). To reproduce the committed
outputs, run the script from a worktree of the tag taken just before the prune:
  git worktree add ../nfl-analytics-pre-prune archive/pre-prune-2026-10
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nfl_analytics.utils.db_config import get_db_config
from nfl_analytics.utils.spread_convention import home_cover, home_cover_margin, model_edge

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/investigations/2026-10-final-evaluation/results"
WIN_PROFIT = 100 / 110
BREAKEVEN = 110 / 210
SEED = 20261024
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # old code -> code the next season uses

# ----------------------------------------------------------------------------
# Claims and where to look for them
# ----------------------------------------------------------------------------

PCT = r"\$?\s*\\?%"  # "74.9%", LaTeX "74.9\%" and "$-7.5$\%"
FLAGGED = re.compile(
    r"(?i)retract|refuted|invalidat|sign (bug|error)|not reproduc|correction|predates|"
    r"superseded|do not cite|never cite|wrong sign|fabricat|simulated|placeholder|leaked|"
    r"NOT achieved|caveat"
)
# Documents whose purpose is to correct the record: a claim quoted there is not asserted
CORRECTION_DOCS = ("analysis/dissertation/ERRATA.md", "docs/POSTMORTEMS.md", "docs/HISTORY.md")
CORRECTION_PREFIX = "docs/investigations/2026-10-"  # the salvage-pass evaluations


@dataclass
class Claim:
    cid: str
    title: str
    verdict: str
    patterns: list[str]
    require: str | None = None  # a matching line must also match this
    exclude: str | None = None  # ... and must not match this
    extra_files: list[str] = field(default_factory=list)  # source artifacts outside the scope

    def matches(self, line: str) -> re.Match | None:
        for pat in self.patterns:
            m = re.search(pat, line)
            if m:
                if self.require and not re.search(self.require, line):
                    return None
                if self.exclude and re.search(self.exclude, line):
                    return None
                return m
        return None


CLAIMS = [
    Claim(
        "C01",
        "Power ratings 74.9% ATS / +43% ROI on the 2025 holdout (Dec 2025)",
        "sign bug",
        [r"74\.9" + PCT, r"\+43(\.0)?" + PCT, r"75\.3" + PCT, r"10\^?\{?-68|10⁻⁶⁸"],
    ),
    Claim(
        "C02",
        "Game Outcome v2, 2025: -3.2% ATS ROI on 208 bets",
        "correct (only at |edge| > 2 strictly)",
        [r"208\s+(qualified\s+)?bets", r"(?<![\d.])(−|-|\$-)\s?3\.2" + PCT],
        require=r"(?i)ATS|ROI|bets|208|season",
    ),
    Claim(
        "C03",
        "Game Outcome v2, 2025: 56.5% win-probability accuracy",
        "correct count, below the market",
        [r"56\.5" + PCT],
        require=r"(?i)win[ -]?prob|11\.05",
        exclude=r"(?i)cvae|home teams|home win rate",
    ),
    Claim(
        "C04",
        "Game Outcome v2, 2025: spread MAE 11.05 vs closing line 9.67",
        "correct",
        [r"\b11\.05\b"],
        require=r"(?i)MAE|spread|9\.67",
    ),
    Claim(
        "C05",
        "README baselines 2019-2024: power ratings 609-608-31, Elo 581-596-31",
        "correct",
        [r"609-608-31", r"581-596-31"],
    ),
    Claim(
        "C06",
        "Cleaned GLM baseline 2004-2024: 51.0% ATS, -7.5% ROI, Sharpe -1.22 (3,826 bets)",
        "wrong payoff (vig charged twice); wrong provenance",
        [r"(?<![\d.])(−|-|\$-)\s?7\.(45|5)" + PCT, r"(?<![\d.])(−|-|\$-)\s?1\.22\b", r"3,826"],
        require=r"(?i)ROI|Sharpe|baseline|win|clean|3,826|GLM|ATS",
    ),
    Claim(
        "C07",
        "Stacked ensemble Brier 0.2515: 'best-in-class' / 'top tier' calibration",
        "mislabeled metric (worse than a constant 0.5)",
        [r"0\.2515", r"(?i)best[- ]in[- ]class calibration", r"(?i)top tier of published"],
        exclude=r"core\\?_(plus|form)|epa_per_play|home win rate|0\.1641",
    ),
    Claim(
        "C08",
        "+14.9 bps closing-line value (CLV)",
        "mislabeled metric (not CLV)",
        [r"14\.9\s*(bps|basis)", r"CLV\s*=\s*\+?14\.9", r"GLM & \+14\.9", r"Stacked\).*14\.9"],
    ),
    Claim(
        "C09",
        "Props MAE 44.25 (equal weight) vs 44.91 (Dirichlet), 2025 holdout",
        "same-game data (the game's own pass attempts)",
        [r"44\.25", r"44\.91"],
    ),
    Claim(
        "C10",
        "Props +2.96% ROI, 53.89% win rate; 'player props markets show exploitable edges'",
        "simulated lines; leaked CQL component",
        [r"2\.96" + PCT, r"53\.89", r"(?i)props markets show exploitable edges?"],
    ),
    Claim(
        "C11",
        "Dirichlet v8.0 props weights: 31.5% MAE improvement (89.42 -> 61.26)",
        "tuned on the test season; in-sample components",
        [r"31\.5" + PCT, r"61\.26", r"89\.42"],
        require=r"(?i)MAE|improv|Dirichlet|v8|61\.26|89\.42",
        exclude=r"(?i)Actionable Signals|BAL vs HOU",
    ),
    Claim(
        "C12",
        "v6.0 'games' ensemble: 54.5% ATS cover, +4.0% ROI (2022-2024)",
        "in-sample; no generating script; v6 predicts passing yards",
        [r"54\.5" + PCT, r"\+4\.0" + PCT],
        require=r"(?i)cover|spread accuracy|ATS|v6|ROI",
        exclude=r"(?i)State-Space 54|XGB|Model Size",
    ),
    Claim(
        "C13",
        "ADR-007 canonical training window: 47.77 yards MAE",
        "same-game data; window picked on the 2025 test weeks",
        [r"47\.77"],
    ),
    Claim(
        "C14",
        "ADR-009 v6 baseline: 34.00 yards MAE (2022-2024)",
        "in-sample",
        [r"34\.00"],
    ),
    Claim(
        "C15",
        "ADR-013 rest: home +3 days 69.2% ATS; 25-point swing; r = 0.17",
        "sign bug",
        [
            r"69\.2" + PCT,
            r"(?i)25[- ]?(percentage[- ])?points? swing|25\s?pp",
            r"(?i)(pearson r|r\s*=|rest)[^\n]{0,12}0\.1698",
        ],
        exclude=r"269\.2",
    ),
    Claim(
        "C16",
        "ADR-014/015: v2.1 10.70 vs v2 10.45 MAE; Kalman/GP features have negative value",
        "tuned on the test set; compares different seasons",
        [r"\b10\.70\b", r"423 features"],
        exclude=r"(?i)closing-line|2007-2013|1869",
    ),
    Claim(
        "C17",
        "Game Outcome v2 2024 validation: MAE 10.45, AUC 0.691, 62.9%",
        "tuned on the validation set (early stopping)",
        [r"\b10\.45\b", r"\b0\.691\b", r"62\.9" + PCT],
    ),
    Claim(
        "C18",
        "State-space ATS accuracy 84.6% (2024) / 74.7% (2025)",
        "sign bug (2024 also in-sample)",
        [r"84\.6" + PCT, r"74\.7" + PCT, r"0\.74725", r"0\.84561"],
        require=r"(?i)ATS|accuracy",
        exclude=r"(?i)hit rate|Position-average",
    ),
    Claim(
        "C19",
        "2019-2025 game-outcome ensemble CSV (~71% ATS, 81-84% straight-up)",
        "same-season data",
        [
            r"2019[–-]2025 ensemble CSV",
            r"predictions_all_games_2019_2025",
            r"5-Component Ensemble",
            r"0\.157498",
        ],
    ),
    Claim(
        "C20",
        "models/ensemble backtest: 76.8% win rate on 910 trades, bankroll 9.2e33",
        "sign bug; same-season data",
        [r"76\.8" + PCT, r"910 trades", r"9\.2\d*e\+?33", r"0\.76813"],
        require=r"(?i)win|trade|bankroll|0\.768",
        exclude=r"(?i)coverage",
    ),
    Claim(
        "C21",
        "models/ensemble realistic backtest: 63.2% win rate on 796 trades",
        "sign bug",
        [r"0\.63190", r"796 trades", r"63\.2" + PCT + r"\s*win"],
    ),
    Claim(
        "C22",
        "Bayesian + XGBoost 2024 test: 71.3% win rate, +18.9% expected ROI (178 bets)",
        "simulated XGBoost; sign bug; static in-sample ratings",
        [r"0\.71348", r"71\.3" + PCT + r"\s*win", r"18\.948"],
        extra_files=["analysis/ensemble/bayesian_xgb_2024_test.json"],
    ),
    Claim(
        "C23",
        "Semantic A/B: 75.4% vs 75.9% accuracy, +43.9% ROI, p = 3e-11",
        "simulated from the outcomes",
        [r"0\.75384", r"75\.4" + PCT, r"43\.909", r"3\.4\d*e-11"],
        exclude=r"Wong",
    ),
    Claim(
        "C24",
        "Semantic A/B that promoted v2: +4.6% win accuracy",
        "look-ahead data (news as of the run date)",
        [r"\+4\.6" + PCT + r"\s*win", r"ab_test_lift", r"\b0\.046\b"],
    ),
    Claim(
        "C25",
        "Semantic features 'underperformed baseline' (58.5% -> 51.8%, p = 0.21); quick A/B",
        "no valid test (constant control; look-ahead inputs)",
        [
            r"(?i)semantic features (\\emph\{)?underperform",
            r"p\s*=\s*0\.21",
            r"quick_ab_test_semantic",
            r"Control \(no semantic",
        ],
    ),
    Claim(
        "C26",
        "Game Outcome v2: 52.1% 'ats_win_pct' / 'correct side'",
        "mislabeled metric (straight-up side)",
        [r"52\.1" + PCT],
        require=r"(?i)ats|side|win_pct|Ensemble v2",
        exclude=r"(?i)PFF|uncertainty|_std",
    ),
    Claim(
        "C27",
        "67.6% ATS in the 3-4 point edge 'sweet spot'",
        "post-hoc subgroup of the test season; misattributed to power ratings",
        [r"67\.6" + PCT, r"(?i)sweet spot", r"(?i)3[–-]4[ -]?(pt|point)s?\s+(edge\s+)?window"],
        exclude=r"(?i)wildcard",
    ),
    Claim(
        "C28",
        "Edge C: 70% win rate, 33.6% ROI (n = 24)",
        "hard-coded",
        [
            r"estimated_win_rate",
            r"estimated_roi_pct",
            r"(?i)edge c (validated|analysis|sample)",
            r"0\.70 (for|from) Edge C",
        ],
        extra_files=["scripts/edge_c_dashboard.py", "scripts/calculate_edge_c_bet_size.py"],
    ),
    Claim(
        "C29",
        "Playoffs behave differently: 8.43 playoff MAE vs 11.14 regular season",
        "correct numbers, wrong inference (12 games; the line improves as much)",
        [r"\b8\.43\b"],
        require=r"(?i)playoff|MAE|post",
    ),
    Claim(
        "C30",
        "2025 playoffs: 70% ATS on 10 bets, +33.6% ROI",
        "correct count, 10 bets (p = 0.17)",
        [r"70\s*\\?%\s*ATS on 10", r"\+33\.6" + PCT, r"Playoffs & 10 & 70"],
    ),
    Claim(
        "C31",
        "Nov-2025 backtests: 62.4% / 60.2% / 62.8% ATS, ROI 4.47M% / 51,106% / 1,320,124%",
        "not reproducible; leak in a feature file that was never committed",
        [
            r"62\.4" + PCT,
            r"60\.2" + PCT,
            r"62\.8" + PCT,
            r"51,?106",
            r"4\.47\s*M",
            r"1,?320,?124",
        ],
        require=r"(?i)ATS|accuracy|win|ROI|backtest|leak",
        extra_files=[
            "analysis/backtest/fixed_backtest_results.json",
            "analysis/backtest/recent_seasons_test_results.json",
            "analysis/backtest/clean_backtest_results.json",
            "analysis/backtest/full_backtest_results.json",
        ],
    ),
    Claim(
        "C32",
        "Weather models 91.8-95.3% accuracy; 'wind myth busted' (r = 0.004)",
        "unverifiable (no generator in git); wind conclusion wrong",
        [
            r"(?i)myth",
            r"(?i)HYPOTHESIS REJECTED",
            r"0\.9183|0\.9525|0\.918 & 0\.795",
            r"9[1-5]\.\d" + PCT + r"\s*accuracy",
        ],
        extra_files=[
            "analysis/results/weather_and_rl_summary.md",
            "analysis/results/multimodel_weather_comparison.csv",
        ],
    ),
    Claim(
        "C33",
        "21% zero-bet weeks (OPE / simulator gating)",
        "hard-coded",
        [r"21" + PCT + r"\s*(of weeks|abstention)", r"(?i)zero-bet weeks?", r"19 \(21\\?%\)"],
    ),
    Claim(
        "C34",
        "40 experiments executed against real NFL data",
        "hard-coded (most metrics are literals)",
        [r"40 experiments", r"(?i)executed against real NFL data", r"(?i)validated 40"],
    ),
    Claim(
        "C35",
        "CVaR benchmark and utilization-adjusted Sharpe tables (RL Sharpe 1.73)",
        "hard-coded",
        [
            r"(?i)utilization[- ]adjusted",
            r"(?i)CVaR Optimal",
            r"cvar_benchmark",
            r"RL Policy & 1\.73",
        ],
    ),
    Claim(
        "C36",
        "Props ROI on simulated lines: +5.6% (59.7%, Sharpe 2.3), +6.2% (60.5%), v5.x 5.5-7.5%",
        "simulated lines (line = actual yards + noise)",
        [r"\+5\.6" + PCT, r"\+6\.2\b", r"59\.7" + PCT, r"60\.5\b", r"5\.5[–-]7\.5" + PCT],
        require=r"(?i)ROI|win|5-Way|ensemble|props|v5",
        exclude=r"(?i)Model Size|confidence interval was estimated",
    ),
    Claim(
        "C37",
        "XGBoost props v1: 12.13 yards MAE (2024 holdout)",
        "same-game data; early stopping on the test season",
        [r"\b12\.13\b", r"12\.128"],
    ),
    Claim(
        "C38",
        "ROI projection -7.5% -> +0.9% to +2.3% with missing signals",
        "speculative projection on the wrong baseline",
        [r"(?i)ROI projection", r"(?i)Realistic Path", r"\+0\.9" + PCT + r"\s*ROI"],
    ),
    Claim(
        "C39",
        "CQL agent: 100% win rate, +24% ROI (Nov 2025)",
        "target leakage (already retracted, ADR-016)",
        [r"(?i)CQL[^.]{0,40}100" + PCT, r"100" + PCT + r"\s*win rate", r"\+24" + PCT],
        exclude=r"\+0\.34 pp",
    ),
    Claim(
        "C40",
        "Teasers: +1683 bps EV, +21.7% ROI; '+5-9% ROI lane'",
        "sign bug",
        [r"\b1683(\.1)?\b", r"\b21\.70\b", r"\+5-{1,2}9\\?%", r"(?i)teaser EV difference"],
        extra_files=["notebooks/analysis/dissertation/figures/out/teaser_ev_oos_table.tex"],
    ),
    Claim(
        "C41",
        "Offline RL / OPE gate on data/rl_logged.csv: median DR 0.2301, policy accepted",
        "same-game data; rewards graded with the old cover rule",
        [r"DR=0\.2301", r"& 0\.230[01] &", r"rl_logged"],
    ),
    Claim(
        "C42",
        "Power ratings walk-forward: 57.7% ATS / +10.2% ROI (2025); 51.8% (2020-2025)",
        "correct, but one season of noise",
        [r"57\.7" + PCT, r"\+10\.2" + PCT, r"51\.8" + PCT],
        require=r"(?i)power|ATS|2025|2020|ROI",
        exclude=r"(?i)lm_|defaults|58\.5",
    ),
    Claim(
        "C43",
        "Incremental XGBoost +21% vs 'static baseline' (Nov 10, 2025)",
        "wrong baseline (already retracted, POSTMORTEMS section 2)",
        [r"\+21" + PCT],
        require=r"(?i)incremental|static",
    ),
    Claim(
        "C44",
        "XGBoost v4 Ridge-residual features (MAE 4.76)",
        "target leakage (already retracted, ADR-003)",
        [r"88(\.31)?\s*\\?%", r"\b4\.76\b"],
        require=r"(?i)ridge|v4|residual|MAE|improv",
        exclude=r"4\.71%|coverage",
    ),
]


# ----------------------------------------------------------------------------
# Locating claims
# ----------------------------------------------------------------------------


def dissertation_build_files() -> set[str]:
    """Repo-relative .tex files reached from main.tex through \\input / \\include."""
    main_dir = REPO / "analysis/dissertation/main"
    inc = re.compile(r"\\(?:input|include)\{([^}#\\]+)\}")
    seen: set[Path] = set()
    stack = [main_dir / "main.tex"]
    while stack:
        f = stack.pop().resolve()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for line in f.read_text(errors="replace").splitlines():
            code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
            for m in inc.finditer(code):
                target = main_dir / m.group(1).strip()  # paths resolve from the build directory
                if target.suffix != ".tex":
                    target = target.with_name(target.name + ".tex")
                stack.append(target)
    return {str(p.relative_to(REPO)) for p in seen if REPO in p.parents}


def scope_files(extra: list[str]) -> list[str]:
    files = ["README.md", "PROJECT_STATUS.md", "CLAUDE.md"]
    skip = "docs/investigations/2026-10-final-evaluation/"
    files += sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "docs").rglob("*.md")
        if not str(p.relative_to(REPO)).startswith(skip)
    )
    dis = REPO / "analysis/dissertation"
    files += sorted(str(p.relative_to(REPO)) for ext in ("*.tex", "*.md") for p in dis.rglob(ext))
    for base, pattern in (("models", "**/*.json"), ("data", "*.json")):
        for p in sorted((REPO / base).glob(pattern)):
            if p.stat().st_size <= 300_000:  # skip model dumps
                files.append(str(p.relative_to(REPO)))
    return files + [f for f in extra if f not in files]


def prominence(path: str, in_build: bool | None) -> int:
    order = {"README.md": 1, "PROJECT_STATUS.md": 2, "CLAUDE.md": 3}
    if path in order:
        return order[path]
    if path == "analysis/dissertation/main/main.tex":
        return 4  # abstract, summary tables and front matter
    if path.startswith("analysis/dissertation/"):
        return 5 if in_build else 9
    if path.startswith("docs/"):
        return 6
    if path.startswith(("models/", "data/")):
        return 7
    return 8


def locate(build: set[str]) -> pd.DataFrame:
    extra = sorted({f for c in CLAIMS for f in c.extra_files})
    rows = []
    for rel in scope_files(extra):
        path = REPO / rel
        if not path.is_file():
            continue
        in_build = rel in build if rel.startswith("analysis/dissertation/") else None
        correction_doc = rel in CORRECTION_DOCS or rel.startswith(CORRECTION_PREFIX)
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for claim in CLAIMS:
                if rel in extra and rel not in claim.extra_files:
                    continue  # artifacts are only searched for their own claim
                m = claim.matches(line)
                if not m:
                    continue
                start = max(0, m.start() - 70)
                says_so = bool(FLAGGED.search(line))
                rows.append(
                    {
                        "claim": claim.cid,
                        "file": rel,
                        "line": lineno,
                        "in_dissertation_build": in_build,
                        "flags_it_already": says_so or correction_doc,
                        "flag_reason": (
                            "correction document"
                            if correction_doc
                            else "line says so" if says_so else ""
                        ),
                        "prominence": prominence(rel, in_build),
                        "excerpt": " ".join(line[start : start + 200].split()),
                    }
                )
    return pd.DataFrame(rows)


def compact(locs: pd.DataFrame, limit: int = 12) -> str:
    parts = []
    for rel, grp in locs.groupby("file", sort=False):
        parts.append(f"{rel}:{','.join(str(n) for n in grp['line'])}")
    more = len(parts) - limit
    return "; ".join(parts[:limit]) + (f"; +{more} more files" if more > 0 else "")


def summarize(locs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for claim in CLAIMS:
        loc = locs[locs["claim"] == claim.cid].sort_values(["prominence", "file", "line"])
        asserted = loc[~loc["flags_it_already"]]
        rows.append(
            {
                "claim": claim.cid,
                "title": claim.title,
                "verdict": claim.verdict,
                "best_prominence": int(loc["prominence"].min()) if len(loc) else 99,
                "lines": len(loc),
                "lines_asserting": len(asserted),
                "lines_flagging": len(loc) - len(asserted),
                "dissertation_build_lines": int(loc["in_dissertation_build"].eq(True).sum()),
                "asserted_at": compact(asserted),
                "flagged_at": compact(loc[loc["flags_it_already"]]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["best_prominence", "lines_asserting"], ascending=[True, False])


# ----------------------------------------------------------------------------
# Database and grading helpers
# ----------------------------------------------------------------------------


class ReadOnlyDB:
    def __init__(self):
        cfg = get_db_config()
        self.conn = psycopg.connect(**cfg, options="-c default_transaction_read_only=on")
        self.conn.read_only = True

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
        for c in df.columns:  # numeric -> float (Decimal from NUMERIC columns)
            if df[c].dtype == object and len(df) and isinstance(df[c].dropna().iloc[0], float):
                df[c] = df[c].astype(float)
        return df

    def close(self):
        self.conn.close()


GAMES_SQL = """
SELECT game_id, season, week, game_type, kickoff, home_team, away_team,
       home_score::float AS home_score, away_score::float AS away_score,
       spread_close::float AS spread_close, total_close::float AS total_close,
       home_moneyline::float AS home_moneyline, away_moneyline::float AS away_moneyline,
       home_rest::float AS home_rest, away_rest::float AS away_rest, roof
FROM games
WHERE home_score IS NOT NULL AND spread_close IS NOT NULL
"""


def load_games(db: ReadOnlyDB) -> pd.DataFrame:
    g = db.query(GAMES_SQL)
    g["margin"] = g["home_score"] - g["away_score"]
    g["cover_margin"] = home_cover_margin(g["margin"], g["spread_close"])
    g["cover"] = home_cover(g["margin"], g["spread_close"])
    return g


def wilson(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def ats(edge, cover_margin, min_edge: float = 0.0, strict: bool = False) -> tuple[int, int, int]:
    """W-L-P of backing the side of the edge (home if edge > 0) against cover_margin."""
    edge = np.asarray(edge, dtype=float)
    cm = np.asarray(cover_margin, dtype=float)
    size = np.abs(edge) > min_edge if strict else np.abs(edge) >= min_edge
    sel = size & (edge != 0) & ~np.isnan(edge) & ~np.isnan(cm)
    res = np.sign(edge[sel]) * np.sign(cm[sel])
    return int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())


def fmt_record(wins: int, losses: int, pushes: int) -> str:
    n = wins + losses
    if n == 0:
        return f"{wins}-{losses}-{pushes}, no decided bets"
    lo, hi = wilson(wins, n)
    roi = (wins * WIN_PROFIT - losses) / n
    return f"{wins}-{losses}-{pushes}, {wins / n:.1%} [{lo:.1%}, {hi:.1%}], ROI {roi:+.1%}"


def devig(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    def implied(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    with np.errstate(divide="ignore", invalid="ignore"):
        ph, pa = implied(home_ml.to_numpy(float)), implied(away_ml.to_numpy(float))
    return pd.Series(ph / (ph + pa), index=home_ml.index)


class Results:
    def __init__(self):
        self.rows = []

    def add(self, claim, check, value, n=None, source="", note=""):
        if isinstance(value, float):
            value = f"{value:.4g}"
        self.rows.append(
            {"claim": claim, "check": check, "value": value, "n": n, "source": source, "note": note}
        )

    def failed(self, claim, check, err):
        self.add(claim, check, "NOT RUN", note=f"{type(err).__name__}: {err}")


# ----------------------------------------------------------------------------
# Checks against devdb01
# ----------------------------------------------------------------------------

V2_SQL = """
SELECT p.game_id, p.prediction_run_id, p.pred_home_spread, p.pred_home_win_prob, p.vegas_spread,
       p.created_at::text AS created_at
FROM ml.game_outcome_predictions p
WHERE p.season = 2025
"""


def check_v2_2025(db, games, res: Results):
    src = "ml.game_outcome_predictions JOIN games"
    v = db.query(V2_SQL).merge(games, on="game_id")
    v = v.astype({"pred_home_spread": float, "pred_home_win_prob": float, "vegas_spread": float})
    edge = model_edge(v["pred_home_spread"], v["spread_close"])
    cm = v["cover_margin"]
    reg = v["game_type"] == "REG"
    post = v["week"].between(19, 21)
    res.add(
        "C04",
        "rows / prediction runs / created_at",
        f"{len(v)} / {v['prediction_run_id'].nunique()} / "
        f"{v['created_at'].min()[:16]} to {v['created_at'].max()[:16]}",
        source=src,
    )
    res.add(
        "C04",
        "vegas_spread equals games.spread_close",
        f"{int((v['vegas_spread'] == v['spread_close']).sum())} of {len(v)}",
        source=src,
    )
    for label, mask in (("all games", v.index == v.index), ("regular season", reg)):
        mae = (v["margin"] - v["pred_home_spread"])[mask].abs().mean()
        mkt = (v["margin"] - v["spread_close"])[mask].abs().mean()
        res.add(
            "C04",
            f"spread MAE model vs closing line, {label}",
            f"{mae:.2f} vs {mkt:.2f}",
            int(mask.sum()),
            src,
        )

    res.add(
        "C02", "ATS all games (side = sign(pred - line))", fmt_record(*ats(edge, cm)), len(v), src
    )
    for label, kw in (
        ("|edge| > 2", {"min_edge": 2, "strict": True}),
        ("|edge| >= 2", {"min_edge": 2}),
        ("|edge| >= 1", {"min_edge": 1}),
    ):
        res.add("C02", f"ATS {label}", fmt_record(*ats(edge, cm, **kw)), source=src)
    rec = ats(edge[reg], cm[reg], min_edge=1)
    res.add(
        "C02",
        "ATS regular season, |edge| >= 1 (market_benchmark's rule)",
        fmt_record(*rec),
        source=src,
    )
    bug_edge, bug_cm = v["pred_home_spread"] + v["spread_close"], v["margin"] + v["spread_close"]  # sign-ok: reproduces the retracted rule on purpose
    wins, losses, _ = ats(bug_edge, bug_cm)
    res.add(
        "C02",
        "old rule (pred + line vs margin + line) agreement",
        f"{wins / (wins + losses):.1%}",
        wins + losses,
        src,
        "what a margin + spread_close grader reports for this model",  # sign-ok: reproduces the retracted rule on purpose
    )

    for label, kw, mask in (
        ("regular season |edge| > 2", {"min_edge": 2, "strict": True}, reg),
        ("playoffs (weeks 19-21) |edge| > 2", {"min_edge": 2, "strict": True}, post),
        ("all postseason incl. Super Bowl |edge| > 2", {"min_edge": 2, "strict": True}, ~reg),
    ):
        rec = ats(edge[mask], cm[mask], **kw)
        note = ""
        if rec[0] + rec[1] and "playoffs" in label:
            p_one = stats.binomtest(rec[0], rec[0] + rec[1], 0.5, "greater").pvalue
            note = f"one-sided binomial p vs 50% = {p_one:.2f}"
        res.add("C30", f"ATS {label}", fmt_record(*rec), source=src, note=note)

    # Straight-up winner and win-probability quality
    decided = v["margin"] != 0
    won = v["margin"] > 0
    pick = v["pred_home_win_prob"] > 0.5
    correct = (pick == won) & decided
    res.add(
        "C03",
        "straight-up winner, model P(home win) > 0.5",
        f"{int(correct.sum())}/{len(v)} = {correct.sum() / len(v):.1%}; "
        f"{int(correct.sum())}/{int(decided.sum())} = {correct.sum() / decided.sum():.1%} excl. ties",
        len(v),
        src,
    )
    fav = v["spread_close"] != 0
    fav_ok = ((v["spread_close"] > 0) == won) & decided & fav
    res.add(
        "C03",
        "straight-up winner, closing-spread favorite",
        f"{int(fav_ok.sum())}/{int((decided & fav).sum())} = {fav_ok.sum() / (decided & fav).sum():.1%}",
        int((decided & fav).sum()),
        src,
    )
    ml = v["home_moneyline"].notna() & v["away_moneyline"].notna() & decided
    p_mkt = devig(v.loc[ml, "home_moneyline"], v.loc[ml, "away_moneyline"])
    p_mod = v.loc[ml, "pred_home_win_prob"].clip(1e-6, 1 - 1e-6)
    y = won[ml].astype(float)

    def ll(p):
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    res.add(
        "C03",
        "win-prob log-loss model vs de-vigged moneyline",
        f"{ll(p_mod):.4f} vs {ll(p_mkt):.4f}",
        int(ml.sum()),
        src,
    )
    res.add(
        "C03",
        "win-prob Brier model vs de-vigged moneyline",
        f"{((p_mod - y) ** 2).mean():.4f} vs {((p_mkt - y) ** 2).mean():.4f}",
        int(ml.sum()),
        src,
    )
    side = (np.sign(v["pred_home_spread"]) == np.sign(v["margin"])) & decided
    res.add(
        "C26",
        "spread head straight-up side, sign(pred) == sign(margin)",
        f"{int(side.sum())}/{int(decided.sum())} = {side.sum() / decided.sum():.1%}",
        int(decided.sum()),
        src,
        "the number stored as ats_win_pct",
    )
    res.add("C26", "ATS all games for comparison", fmt_record(*ats(edge, cm)), len(v), src)

    metrics = db.query(
        "SELECT run_id, metric_name, metric_value::float AS metric_value, metadata::text AS metadata "
        "FROM ml.model_metrics WHERE metric_name IN ('ats_win_pct', 'ats_roi_pct', 'spread_mae', "
        "'win_prob_accuracy', 'win_prob_auc') ORDER BY run_id, metric_name"
    )
    for _, m in metrics.iterrows():
        cid = {"ats_win_pct": "C26", "ats_roi_pct": "C02"}.get(m["metric_name"], "C17")
        res.add(
            cid,
            f"ml.model_metrics run {m['run_id']} {m['metric_name']} (stored)",
            f"{m['metric_value']:g}",
            source="ml.model_metrics",
            note=m["metadata"],
        )

    # Edge buckets (ws3/ws5)
    a = edge.abs()
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10), (10, np.inf)):
        mask = (a >= lo) & (a < hi)
        res.add(
            "C27",
            f"ATS with {lo} <= |edge| < {hi}",
            fmt_record(*ats(edge[mask], cm[mask])),
            int(mask.sum()),
            src,
        )

    # Playoffs vs regular season
    for label, mask in (
        ("regular season", reg),
        ("playoffs weeks 19-21", post),
        ("Super Bowl", v["week"] == 22),
    ):
        mae = (v["margin"] - v["pred_home_spread"])[mask].abs().mean()
        mkt = (v["margin"] - v["spread_close"])[mask].abs().mean()
        res.add(
            "C29",
            f"spread MAE model vs closing line, {label}",
            f"{mae:.2f} vs {mkt:.2f}",
            int(mask.sum()),
            src,
        )
    err = (v["margin"] - v["pred_home_spread"])[reg].abs().to_numpy()
    k = int(post.sum())
    target = (v["margin"] - v["pred_home_spread"])[post].abs().mean()
    rng = np.random.default_rng(SEED)
    draws = np.argsort(rng.random((20_000, len(err))), axis=1)[:, :k]
    share = float((err[draws].mean(axis=1) <= target).mean())
    res.add(
        "C29",
        f"share of random {k}-game regular-season subsets with model MAE <= {target:.2f}",
        f"{share:.2f}",
        20_000,
        src,
    )
    mkt_all = games.assign(
        post=games["game_type"] != "REG", err=(games["margin"] - games["spread_close"]).abs()
    )
    for flag, grp in mkt_all.groupby("post"):
        res.add(
            "C29",
            f"closing-line MAE 1999-2025, {'postseason' if flag else 'regular season'}",
            f"{grp['err'].mean():.2f}",
            len(grp),
            "games",
        )

    # ADR-014/015: v2.1's test window was 2025 weeks 3-13 (163 games, Vegas MAE 10.00)
    w313 = reg & v["week"].between(3, 13)
    mae = (v["margin"] - v["pred_home_spread"])[w313].abs().mean()
    mkt = (v["margin"] - v["spread_close"])[w313].abs().mean()
    res.add(
        "C16",
        "stored v2 on 2025 weeks 3-13: spread MAE model vs closing line",
        f"{mae:.2f} vs {mkt:.2f}",
        int(w313.sum()),
        src,
        "ADR-014 compares v2.1 on these games (10.70) with v2's 2024 validation MAE (10.45)",
    )


def prior_season_ratings(games: pd.DataFrame, season: int) -> dict[str, float]:
    """Dec-9 method: each team's average point differential in last season's regular season."""
    prev = games[(games["season"] == season - 1) & (games["game_type"] == "REG")]
    rows = pd.concat(
        [
            pd.DataFrame({"team": prev["home_team"], "m": prev["margin"]}),
            pd.DataFrame({"team": prev["away_team"], "m": -prev["margin"]}),
        ]
    )
    ratings = {t: float(r) for t, r in rows.groupby("team")["m"].mean().items()}
    for old, new in RELOCATED.items():  # a relocated team keeps last season's rating
        if old in ratings and new not in ratings:
            ratings[new] = ratings[old]
    return ratings


def check_power_ratings_dec9(games, res: Results):
    src = "games (Dec-9 method rebuilt: last season's point differential per game + 2.5 HFA)"
    frames = []
    for season in range(2019, 2026):
        r = prior_season_ratings(games, season)
        s = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
        s = s[s["home_team"].isin(r) & s["away_team"].isin(r)]
        s["pred"] = s["home_team"].map(r) - s["away_team"].map(r) + 2.5
        frames.append(s)
    d = pd.concat(frames)
    for label, mask in (
        ("2025 weeks 1-14", (d["season"] == 2025) & (d["week"] <= 14)),
        ("2020-2025", d["season"].between(2020, 2025)),
        ("2019-2024", d["season"].between(2019, 2024)),
    ):
        s = d[mask]
        for t in (0, 1, 2, 3):
            bug = ats(s["pred"] + s["spread_close"], s["margin"] + s["spread_close"], min_edge=t)  # sign-ok: reproduces the retracted rule on purpose
            ok = ats(model_edge(s["pred"], s["spread_close"]), s["cover_margin"], min_edge=t)
            res.add(
                "C01",
                f"{label}, |edge| >= {t}: old rule (pred + line, margin + line)",
                fmt_record(*bug),
                source=src,
            )
            res.add("C01", f"{label}, |edge| >= {t}: correct rule", fmt_record(*ok), source=src)
    per_season = []
    for season, s in d.groupby("season"):
        old = ats(s["pred"] + s["spread_close"], s["margin"] + s["spread_close"])  # sign-ok: reproduces the retracted rule on purpose
        new = ats(model_edge(s["pred"], s["spread_close"]), s["cover_margin"])
        per_season.append(
            f"{season}: {old[0] / (old[0] + old[1]):.1%} vs {new[0] / (new[0] + new[1]):.1%}"
        )
    res.add(
        "C01", "per season, all bets: old rule vs correct rule", "; ".join(per_season), source=src
    )


def check_power_ratings_repo(games, res: Results):
    try:
        from nfl_analytics.models import power_ratings_model as prm
    except Exception as exc:  # psycopg2 missing, module moved
        res.failed("C42", "nfl_analytics.models.power_ratings_model.backtest_season", exc)
        return
    g = games[games["season"].between(2020, 2025)].copy()
    tot = np.zeros(3, dtype=int)
    for season in range(2020, 2026):
        r = prm.backtest_season(g, season)
        rec = np.array([r.ats_wins, r.ats_losses, r.ats_pushes])
        tot += rec
        if season == 2025:
            res.add(
                "C42",
                "repo power_ratings_model 2025 (weeks >= 5)",
                fmt_record(*rec),
                r.qualified_bets,
                "games",
                f"model MAE {r.pred_mae:.2f} vs line {r.vegas_mae:.2f}",
            )
    res.add("C42", "repo power_ratings_model 2020-2025 pooled", fmt_record(*tot), source="games")


def check_readme_baselines(res: Results):
    try:
        from nfl_analytics.evaluation.market_benchmark.baselines import (
            elo_walkforward,
            power_ratings_walkforward,
        )
    except Exception as exc:
        res.failed("C05", "market_benchmark baselines", exc)
        return
    m = pd.read_csv(REPO / "data/historical_weather_2018_2024.csv")
    m = m.dropna(subset=["home_score", "away_score", "spread_close"]).drop_duplicates("game_id")
    m = m.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    m["home_margin"] = m["home_score"] - m["away_score"]
    cm = home_cover_margin(m["home_margin"], m["spread_close"])
    test = m["season"].between(2019, 2024)
    for name, fn in (("power ratings", power_ratings_walkforward), ("Elo", elo_walkforward)):
        pred = fn(m)
        rec = ats(model_edge(pred, m["spread_close"])[test], cm[test], min_edge=1)
        res.add(
            "C05",
            f"README baseline {name}, 2019-2024, |edge| >= 1",
            fmt_record(*rec),
            int(test.sum()),
            "data/historical_weather_2018_2024.csv",
        )


def check_rest(db, games, res: Results):
    g = games[games["home_rest"].notna() & games["away_rest"].notna()].copy()
    g["rd"] = g["home_rest"] - g["away_rest"]
    cats = pd.cut(
        g["rd"],
        [-99, -3, -1, 0, 2, 99],
        labels=["away +3 or more", "away +1..2", "equal", "home +1..2", "home +3 or more"],
    )
    g["cat"] = cats
    g["old_cover"] = np.where(
        g["margin"] + g["spread_close"] > 0,  # sign-ok: reproduces the retracted rule on purpose
        1.0,
        np.where(g["margin"] + g["spread_close"] < 0, 0.0, np.nan),  # sign-ok: reproduces the retracted rule on purpose
    )
    src = "games.home_rest / away_rest (nflverse)"
    for label, mask, col in (
        ("2025, correct rule", g["season"] == 2025, "cover"),
        ("2025, old rule", g["season"] == 2025, "old_cover"),
        ("2006-2024, correct rule", g["season"].between(2006, 2024), "cover"),
    ):
        s = g[mask & g[col].notna()]
        parts = [
            f"{c}: {grp[col].mean():.1%} (n={len(grp)})"
            for c, grp in s.groupby("cat", observed=True)
        ]
        res.add(
            "C15", f"home cover rate by rest difference, {label}", "; ".join(parts), len(s), src
        )
    for label, mask in (
        ("2025", g["season"] == 2025),
        ("2006-2024", g["season"].between(2006, 2024)),
    ):
        s = g[mask & g["cover"].notna()]
        res.add(
            "C15",
            f"corr(rest difference, ATS margin), {label}",
            f"{np.corrcoef(s['rd'], s['cover_margin'])[0, 1]:+.3f}",
            len(s),
            src,
        )

    # ADR-013's own query: home team's days since its previous HOME game, 3-14 days, 2025
    h = db.query("""
        SELECT game_id, week, home_team, (home_score - away_score)::float AS margin,
               spread_close::float AS spread_close,
               (kickoff AT TIME ZONE 'America/New_York')::date
                 - LAG((kickoff AT TIME ZONE 'America/New_York')::date)
                     OVER (PARTITION BY home_team ORDER BY kickoff) AS home_rest
        FROM games
        WHERE season = 2025 AND home_score IS NOT NULL AND spread_close IS NOT NULL
        """)
    h = h[h["home_rest"].notna() & h["home_rest"].between(3, 14)].copy()
    h["home_rest"] = h["home_rest"].astype(float)
    h["bucket"] = pd.cut(h["home_rest"], [0, 5, 7, 10, 15], labels=["3-5", "6-7", "8-10", "11+"])
    for label, mask in (("weeks <= 16", h["week"] <= 16), ("all weeks", h["week"] > 0)):
        s = h[mask]
        for rule, cov in (
            ("correct", (s["margin"] - s["spread_close"] > 0).astype(float)),
            ("old", (s["margin"] + s["spread_close"] > 0).astype(float)),  # sign-ok: reproduces the retracted rule on purpose
        ):
            parts = [
                f"{b}: {cov[s['bucket'] == b].mean():.1%} (n={int((s['bucket'] == b).sum())})"
                for b in s["bucket"].cat.categories
            ]
            r = np.corrcoef(s["home_rest"], cov)[0, 1]
            res.add(
                "C15",
                f"ADR-013 query (home-game rest, pushes as no-cover), {label}, {rule} rule",
                f"r = {r:+.3f}; " + "; ".join(parts),
                len(s),
                "games (kickoff dates in America/New_York)",
                "ADR-013 printed r = 0.1698 on 139 games; 8-10d 64.3%, 11+d 59.3%",
            )


def check_state_space_metrics(games, res: Results):
    from nfl_analytics.models.state_space import StateSpaceRatings

    cfg = json.loads((REPO / "models/state_space/config.json").read_text())
    stored = pd.read_csv(REPO / "models/state_space/predictions_historical.csv")
    test_ids = set(stored.loc[stored["season"] == 2025, "game_id"])
    g = games[games["season"].between(2019, 2025)].copy()
    for c in ("home_team", "away_team"):
        g.loc[g["season"] == 2019, c] = g.loc[g["season"] == 2019, c].replace(RELOCATED)
    g = g.sort_values(["season", "week", "kickoff"])
    model = StateSpaceRatings(
        q=cfg["q"],
        r=cfg["r"],
        hfa_init=cfg["hfa_init"],
        init_variance=cfg["init_variance"],
        hfa_variance=cfg["hfa_variance"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(g[g["season"].between(2019, 2024)].copy())  # train_state_space fits train + val
    stored_metrics = {
        "2024 (val, in-sample)": json.loads(
            (REPO / "models/state_space/val_metrics.json").read_text()
        ),
        "2025 first 107 games (test)": json.loads(
            (REPO / "models/state_space/test_metrics.json").read_text()
        ),
    }
    for label, s in (
        ("2024 (val, in-sample)", g[g["season"] == 2024]),
        ("2025 first 107 games (test)", g[g["game_id"].isin(test_ids)]),
    ):
        mu_sd = [model.predict_margin(h, a) for h, a in zip(s["home_team"], s["away_team"])]
        mu = np.array([x[0] for x in mu_sd])
        sd = np.array([x[1] for x in mu_sd])
        line = s["spread_close"].to_numpy()
        margin = s["margin"].to_numpy()
        p_old = norm.cdf((mu + line) / sd)
        y_old = (margin + line > 0).astype(float)  # pushes counted as no cover, as stored
        p_fix = norm.cdf((mu - line) / sd)
        y = home_cover(margin, line)
        keep = ~np.isnan(y)
        st = stored_metrics[label]
        res.add(
            "C18",
            f"{label}: stored accuracy_ats",
            f"{st['accuracy_ats']:.4f}",
            st.get("n_games"),
            "models/state_space/*_metrics.json",
        )
        res.add(
            "C18",
            f"{label}: old formula and old label (reproduction)",
            f"accuracy {np.mean((p_old >= 0.5) == (y_old == 1)):.4f}, Brier {np.mean((p_old - y_old) ** 2):.3f}",
            len(s),
            "games + StateSpaceRatings(config.json) fit on 2019-2024",
        )
        res.add(
            "C18",
            f"{label}: correct formula and label",
            f"accuracy {np.mean((p_fix[keep] >= 0.5) == (y[keep] == 1)):.4f}, "
            f"Brier {np.mean((p_fix[keep] - y[keep]) ** 2):.3f}, "
            f"MAE {np.mean(np.abs(margin - mu)):.2f} vs line {np.mean(np.abs(margin - line)):.2f}",
            int(keep.sum()),
            "games + StateSpaceRatings(config.json) fit on 2019-2024",
        )


def check_ensemble_csv(games, res: Results):
    p = pd.read_csv(REPO / "models/ensemble/game_outcome/predictions_all_games_2019_2025.csv")
    d = p.merge(games[["game_id", "margin", "spread_close", "cover_margin"]], on="game_id")
    src = "models/ensemble/game_outcome/predictions_all_games_2019_2025.csv + games"
    decided = d["margin"] != 0
    res.add(
        "C19",
        "target equals home straight-up win",
        f"{((d['target'] == 1) == (d['margin'] > 0))[decided].mean():.4f}",
        int(decided.sum()),
        src,
    )
    groups = (
        ("2019-2023 ('train')", d["season"].between(2019, 2023)),
        ("2024 ('val')", d["season"] == 2024),
        ("2025 ('test')", d["season"] == 2025),
    )
    for label, mask in groups:
        s = d[mask & decided]
        su = ((s["ensemble_prediction"] >= 0.5) == (s["margin"] > 0)).mean()
        fav = ((s["spread_close"] > 0) == (s["margin"] > 0))[s["spread_close"] != 0].mean()
        res.add(
            "C19",
            f"straight-up accuracy {label}: ensemble vs closing favorite",
            f"{su:.1%} vs {fav:.1%}",
            len(s),
            src,
        )
        pseudo = 13.5 * norm.ppf(d.loc[mask, "ensemble_prediction"].clip(1e-3, 1 - 1e-3))
        rec = ats(model_edge(pseudo, d.loc[mask, "spread_close"]), d.loc[mask, "cover_margin"], 1)
        res.add(
            "C19",
            f"ATS {label}, margin = 13.5 * Phi^-1(p), |edge| >= 1",
            fmt_record(*rec),
            int(mask.sum()),
            src,
        )


def check_trade_logs(games, res: Results):
    for cid, path in (
        ("C20", "models/ensemble/backtest/trades.csv"),
        ("C21", "models/ensemble/realistic_backtest/trades.csv"),
    ):
        t = pd.read_csv(REPO / path).merge(
            games[["game_id", "margin", "spread_close", "cover_margin"]], on="game_id", how="left"
        )
        side = np.where(t["bet_side"] == "home", 1.0, -1.0)
        rec = ats(side, t["cover_margin"])
        old = np.sign(t["margin"] + t["spread_close"]) * side > 0  # sign-ok: reproduces the retracted rule on purpose
        res.add(
            cid,
            "stored outcome agreement with old rule / correct rule",
            f"{(t['outcome'] == old.astype(int)).mean():.3f} / "
            f"{(t['outcome'] == (np.sign(t['cover_margin']) * side > 0).astype(int)).mean():.3f}",
            len(t),
            path,
        )
        res.add(cid, "trades regraded with the correct rule", fmt_record(*rec), len(t), path)
        by = []
        for season, s in t.groupby("season"):
            wins, losses, _ = ats(np.where(s["bet_side"] == "home", 1.0, -1.0), s["cover_margin"])
            by.append(f"{season}: {wins / (wins + losses):.1%} (n={wins + losses})")
        res.add(cid, "regraded win rate by season", "; ".join(by), len(t), path)


def check_bayes_xgb(games, res: Results):
    path = "analysis/ensemble/bayesian_xgb_2024_test.json"
    data = json.loads((REPO / path).read_text())
    g = pd.DataFrame(data["games"])
    res.add("C22", "stored results", json.dumps(data["results"]), source=path)
    np.random.seed(42)  # reproduces the simulate branch of ensemble_bayesian_xgb.py
    sim = np.clip(g["bayesian_prob_home"] + np.random.normal(0, 0.05, len(g)), 0.01, 0.99)
    res.add(
        "C22",
        "xgb_prob_home vs bayesian_prob_home + N(0, 0.05) with seed 42 (max abs diff)",
        f"{np.abs(sim - g['xgb_prob_home']).max():.2e}",
        len(g),
        path,
        "nfl_analytics/production/ensemble_bayesian_xgb.py simulate branch",
    )
    old = g["home_cover_margin"] == g["home_margin"] + g["spread_close"]  # sign-ok: reproduces the retracted rule on purpose
    res.add(
        "C22",
        "stored home_cover_margin equals margin + spread_close (old rule)",  # sign-ok: reproduces the retracted rule on purpose
        f"{old.mean():.3f}",
        len(g),
        path,
    )
    res.add(
        "C22",
        "distinct Bayesian ratings per team over the season",
        f"{g.groupby('home_team')['home_bayesian_rating'].nunique().max()}",
        len(g),
        path,
        "one static rating per team",
    )
    bets = g.loc[g["should_bet"].astype(bool), ["game_id", "bet_side"]].merge(
        games[["game_id", "cover_margin"]], on="game_id"
    )
    side = np.where(bets["bet_side"] == "home", 1.0, -1.0)
    res.add(
        "C22",
        "bets regraded with the correct rule",
        fmt_record(*ats(side, bets["cover_margin"])),
        len(bets),
        path,
    )


def check_semantic_ab(games, res: Results):
    path = "data/semantic_ab_test_results.json"
    d = json.loads((REPO / path).read_text())
    res.add(
        "C23",
        "stored baseline / semantic accuracy",
        f"{d['baseline']['accuracy']:.4f} / {d['semantic']['accuracy']:.4f}",
        d["games"],
        path,
    )
    src = (REPO / "scripts/ab_test_semantic_features.py").read_text()
    for needle in (
        "baseline_accuracy = 0.749",
        'games["home_covered"].values.copy()',
        "np.random.random",
    ):
        res.add(
            "C23",
            f"generator contains `{needle}`",
            str(needle in src),
            source="scripts/ab_test_semantic_features.py",
        )
    q = pd.read_csv(REPO / "data/quick_ab_test_semantic.csv")
    qa = q.merge(games[["game_id", "spread_close", "cover_margin"]], on="game_id")
    res.add(
        "C25",
        "quick A/B control_spread distinct values",
        ", ".join(f"{x:g}" for x in sorted(q["control_spread"].unique())),
        len(q),
        "data/quick_ab_test_semantic.csv",
    )
    res.add(
        "C25",
        "quick A/B straight-up accuracy control / treatment",
        f"{q['control_correct'].mean():.1%} / {q['treatment_correct'].mean():.1%}",
        len(q),
        "data/quick_ab_test_semantic.csv",
    )
    for arm in ("control_spread", "treatment_spread"):
        rec = ats(model_edge(qa[arm], qa["spread_close"]), qa["cover_margin"])
        res.add(
            "C25",
            f"quick A/B ATS, {arm}",
            fmt_record(*rec),
            len(qa),
            "data/quick_ab_test_semantic.csv + games",
        )


def check_props(db, res: Results):
    src = "mart.player_game_stats (passing, attempts >= 10, week >= 1)"
    p = db.query("""
        SELECT player_id, season, week, stat_attempts::float AS att, stat_yards::float AS yds
        FROM mart.player_game_stats
        WHERE stat_category = 'passing' AND season IN (2024, 2025) AND week >= 1
          AND stat_attempts >= 10
        """)
    ly = p[p["season"] == 2024]
    ypa = ly["yds"].sum() / ly["att"].sum()
    cur = p[p["season"] == 2025].copy()
    cur["zero_skill"] = cur["att"] * ypa
    cur["const"] = ly["yds"].mean()
    hist = p.sort_values(["season", "week"])
    prior = []
    for _, r in cur.iterrows():
        h = hist[
            (hist["player_id"] == r["player_id"])
            & ((hist["season"] < 2025) | (hist["week"] < r["week"]))
        ]
        prior.append(h["yds"].mean() if len(h) else np.nan)
    cur["prior_mean"] = prior
    for label, mask in (
        ("2025 weeks 1-12", cur["week"] <= 12),
        ("2025 weeks 1-5", cur["week"] <= 5),
    ):
        s = cur[mask]
        res.add(
            "C09",
            f"zero-skill same-game attempts x 2024 league YPA ({ypa:.2f}), {label}",
            f"MAE {np.abs(s['yds'] - s['zero_skill']).mean():.2f}",
            len(s),
            src,
            "uses the game's own attempts, as the props models did",
        )
        res.add(
            "C09",
            f"constant 2024 mean yards, {label}",
            f"MAE {np.abs(s['yds'] - s['const']).mean():.2f}",
            len(s),
            src,
        )
        k = s["prior_mean"].notna()
        res.add(
            "C09",
            f"player's mean over earlier games (pre-game only), {label}",
            f"MAE {np.abs(s.loc[k, 'yds'] - s.loc[k, 'prior_mean']).mean():.2f}",
            int(k.sum()),
            src,
        )
    tabs = db.query(
        "SELECT table_schema || '.' || table_name AS t FROM information_schema.tables "
        "WHERE table_name ILIKE '%%prop%%' OR table_name ILIKE '%%player_line%%'"
    )
    res.add(
        "C36",
        "tables named like prop/player_line in devdb01",
        ", ".join(tabs["t"]) if len(tabs) else "none",
        len(tabs),
        "information_schema.tables",
    )
    mk = db.query("SELECT market_type, count(*) AS n FROM odds_history GROUP BY 1 ORDER BY 1")
    res.add(
        "C36",
        "odds_history market types (row counts)",
        ", ".join(f"{r.market_type}: {r.n}" for r in mk.itertuples()),
        int(mk["n"].sum()),
        "public.odds_history (aggregate only)",
    )


def check_nov2025_backtests(games, res: Results):
    try:
        from nfl_analytics.features import asof_features as af

        cfg = get_db_config()
        with psycopg.connect(**cfg, options="-c default_transaction_read_only=on") as conn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                team = af.fetch_team_game_features(conn)
        feats = af.pivot_to_games(af.compute_team_history(team))
    except Exception as exc:
        res.failed("C31", "as-of feature rebuild (nfl_analytics.features.asof_features)", exc)
        return
    cols = [
        "spread_close",
        "total_close",
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "margin_last3_diff",
        "prior_win_pct_diff",
        "rest_diff",
        "qb_change_diff",
        "coach_change_diff",
    ]
    cols = [c for c in cols if c in feats.columns]
    d = feats[feats["season"].between(1999, 2024) & feats["spread_close"].notna()].copy()
    d = d[d["home_score"].notna()]
    margin = d["home_score"] - d["away_score"]
    labels = {
        "correct label (pushes dropped)": home_cover(margin, d["spread_close"]),
        "old label margin + spread > 0 (pushes as 0)": (margin + d["spread_close"] > 0).astype(  # sign-ok: reproduces the retracted rule on purpose
            float
        ),
    }
    src = "as-of features rebuilt from devdb01 with nfl_analytics/features/asof_features.py"
    for name, y in labels.items():
        k = y.notna().to_numpy()
        x = d.loc[k, cols].fillna(0).to_numpy()
        yy = y[k].astype(int).to_numpy()
        s = d.loc[k, "season"].to_numpy()
        for design, tr, te in (
            (
                "train 1999-2019 -> test 2020-2024 (fixed_backtest, printed 60.2%)",
                s < 2020,
                s >= 2020,
            ),
            (
                "train 2022-2023 -> test 2024 (recent_seasons_test, printed 62.8%)",
                (s >= 2022) & (s <= 2023),
                s == 2024,
            ),
        ):
            sc = StandardScaler().fit(x[tr])
            m = LogisticRegression(max_iter=2000, random_state=42).fit(sc.transform(x[tr]), yy[tr])
            acc = float((m.predict(sc.transform(x[te])) == yy[te]).mean())
            lo, hi = wilson(int(round(acc * te.sum())), int(te.sum()))
            res.add(
                "C31",
                f"{design}, {name}",
                f"accuracy {acc:.1%} [{lo:.1%}, {hi:.1%}]",
                int(te.sum()),
                src,
                f"features: {', '.join(cols)}",
            )


def check_weather(db, res: Results, nflverse: Path | None):
    w = db.query("""
        SELECT g.season, g.roof, w.wind::float AS wind, w.temp::float AS temp,
               (g.home_score + g.away_score)::float AS total, g.total_close::float AS total_close
        FROM weather w JOIN games g USING (game_id)
        WHERE w.data_source = 'historical_csv' AND g.total_close IS NOT NULL
          AND g.home_score IS NOT NULL
        """)
    src = "weather (data_source historical_csv = nflverse mph, 2018-2024) JOIN games"

    def unders(s, label, source):
        s = s[s["total"] != s["total_close"]]
        n = len(s)
        u = int((s["total"] < s["total_close"]).sum())
        p = stats.binomtest(u, n, 0.5, alternative="greater").pvalue if n else math.nan
        res.add(
            "C32", f"unders, {label}", f"{u}/{n} = {u / n:.1%}, one-sided p = {p:.4f}", n, source
        )

    out = w[w["roof"].isin(["outdoors", "open"]) & w["wind"].notna()]
    res.add(
        "C32",
        "corr(wind, total points), outdoor 2018-2024",
        f"{np.corrcoef(out['wind'], out['total'])[0, 1]:+.3f}",
        len(out),
        src,
    )
    res.add(
        "C32",
        "corr(wind, total - closing total), outdoor 2018-2024",
        f"{np.corrcoef(out['wind'], out['total'] - out['total_close'])[0, 1]:+.3f}",
        len(out),
        src,
    )
    unders(out, "outdoor 2018-2024, all", src)
    unders(out[out["wind"] >= 15], "outdoor 2018-2024, wind >= 15 mph", src)
    mw = pd.read_csv(REPO / "analysis/results/multimodel_weather_comparison.csv")
    res.add(
        "C32",
        "multimodel_weather_comparison.csv accuracy by model",
        ", ".join(f"{r.model}: {r.accuracy:.3f}" for r in mw.itertuples()),
        int(mw["n_games"].iloc[0]),
        "analysis/results/multimodel_weather_comparison.csv",
        "same harness without weather: 50-51% (walkforward_2004_2024_metrics.csv)",
    )
    if nflverse is None:
        return
    nv = pd.read_csv(nflverse, low_memory=False)
    nv = nv[nv["result"].notna() & nv["total_line"].notna() & nv["roof"].isin(["outdoors", "open"])]
    nv = nv.assign(total=nv["home_score"] + nv["away_score"], total_close=nv["total_line"])
    nsrc = f"nflverse schedules ({nflverse.name}), outdoor/open roofs"
    unders(nv, "nflverse 1999-2025 outdoor, all", nsrc)
    unders(nv[nv["wind"] >= 15], "nflverse 1999-2025 outdoor, wind >= 15 mph", nsrc)
    unders(
        nv[(nv["wind"] >= 15) & (nv["season"] <= 2012)], "nflverse 1999-2012, wind >= 15 mph", nsrc
    )
    unders(
        nv[(nv["wind"] >= 15) & (nv["season"] > 2012)], "nflverse 2013-2025, wind >= 15 mph", nsrc
    )


def check_teasers(games, res: Results):
    s = games[games["season"].between(2020, 2024)]
    legs = np.concatenate([s["cover_margin"] + 6, -s["cover_margin"] + 6])
    legs = legs[legs != 0]
    p = float((legs > 0).mean())
    res.add("C40", "6-point teaser legs won, both sides, 2020-2024", f"{p:.1%}", len(legs), "games")
    for price, dec in (("1.8 (notebook 04)", 1.8), ("-120", 1 + 100 / 120)):
        res.add(
            "C40",
            f"two-leg 6-point teaser EV at decimal {price}, independent legs",
            f"{dec * p * p - 1:+.1%}",
            len(legs),
            "games",
            f"break-even leg rate {dec ** -0.5:.1%}",
        )


def check_rl_logged(games, res: Results):
    path = "data/rl_logged.csv"
    r = pd.read_csv(REPO / path).merge(
        games[["game_id", "margin", "cover_margin"]], on="game_id", suffixes=("", "_db")
    )
    act = r[r["action"] == 1]
    old = act["home_score"] - act["away_score"] + act["spread_close"] > 0  # sign-ok: reproduces the retracted rule on purpose
    res.add(
        "C41",
        "action = 1 rows: reward r > 0 agrees with old rule / correct rule",
        f"{((act['r'] > 0) == old).mean():.3f} / {((act['r'] > 0) == (act['cover_margin'] > 0)).mean():.3f}",
        len(act),
        path,
    )
    res.add(
        "C41",
        "corr(p_hat, home margin of the same game)",
        f"{np.corrcoef(r['p_hat'], r['margin'])[0, 1]:+.3f}",
        len(r),
        path,
        "p_hat is built from the predicted game's own EPA",
    )
    pseudo = 13.5 * norm.ppf(r["p_hat"].clip(1e-3, 1 - 1e-3))
    rec = ats(model_edge(pseudo, r["spread_close"]), r["cover_margin"], 1)
    res.add(
        "C41",
        "ATS of p_hat as a margin (13.5 * Phi^-1(p)), |edge| >= 1",
        fmt_record(*rec),
        len(r),
        path,
    )


# ----------------------------------------------------------------------------
# Checks of committed files (hard-coded or simulated results)
# ----------------------------------------------------------------------------


def table_numbers(tex: str, first_col: str) -> list[str]:
    for line in tex.splitlines():
        if line.strip().startswith(first_col):
            return re.findall(r"-?\d+\.\d+", line)
    return []


def check_static(res: Results):
    gen = (REPO / "scripts/utils/generate_missing_tables.py").read_text()
    out = REPO / "analysis/dissertation/figures/out"
    for table, rows in (
        (
            "cvar_benchmark_table.tex",
            ["Equal Weight", "Min Variance", "Max Sharpe", "Risk Parity", "\\textbf{CVaR Optimal}"],
        ),
        (
            "utilization_adjusted_sharpe_table.tex",
            [
                "Buy \\& Hold SPY",
                "Static Kelly",
                "Dynamic Kelly",
                "RL Policy",
                "\\textbf{RL + Risk Gates}",
            ],
        ),
    ):
        tex = (out / table).read_text()
        nums = [n for row in rows for n in table_numbers(tex, row)]
        found = sum(1 for n in nums if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", gen))
        res.add(
            "C35",
            f"{table}: printed numbers present as literals in generate_missing_tables.py",
            f"{found} of {len(nums)}",
            len(nums),
            "scripts/utils/generate_missing_tables.py",
        )
    tex = (out / "cvar_benchmark_table.tex").read_text()
    bad = 0
    for row in (
        "Equal Weight",
        "Min Variance",
        "Max Sharpe",
        "Risk Parity",
        "\\textbf{CVaR Optimal}",
    ):
        nums = [float(x) for x in table_numbers(tex, row)]
        if len(nums) == 5 and nums[4] > nums[2]:  # worst week milder than the 95% tail mean
            bad += 1
    res.add(
        "C35",
        "cvar_benchmark_table rows where Worst% is milder than CVaR95% (impossible)",
        f"{bad} of 5",
        5,
        "analysis/dissertation/figures/out/cvar_benchmark_table.tex",
    )

    zgen = (REPO / "nfl_analytics/analysis/generate_zero_bet_weeks.py").read_text()
    reads = bool(re.search(r"read_sql|read_csv|connect\(|open\(", zgen))
    res.add(
        "C33",
        "generate_zero_bet_weeks.py reads any data / says 'plausible'",
        f"{reads} / {'plausible' in zgen}",
        source="nfl_analytics/analysis/generate_zero_bet_weeks.py",
    )

    tree = ast.parse((REPO / "nfl_analytics/experiments/experiment_executor.py").read_text())
    kinds = {"all literal": 0, "mixed": 0, "all computed": 0, "no metrics dict": 0}
    no_data = 0
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and re.match(r"_(cvae|causal)_\d{3}_", fn.name)):
            continue
        values = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(isinstance(t, ast.Name) and t.id == "metrics" for t in node.targets):
                    values += node.value.values
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                if node.value.elts and isinstance(node.value.elts[0], ast.Dict):
                    values += node.value.elts[0].values
        lit = sum(
            isinstance(v, ast.Constant)
            or (isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant))
            for v in values
        )
        if not values:
            kinds["no metrics dict"] += 1
        else:
            kinds["all literal" if lit == len(values) else "mixed" if lit else "all computed"] += 1
        calls = {
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        no_data += not (calls & {"read_sql", "_query_df", "execute", "read_csv"})
    total = sum(kinds.values())
    res.add(
        "C34",
        "experiment functions by metrics-dict values",
        ", ".join(f"{k}: {v}" for k, v in kinds.items()) + f"; no data read: {no_data}",
        total,
        "nfl_analytics/experiments/experiment_executor.py (AST)",
    )

    edge = (REPO / "scripts/edge_c_dashboard.py").read_text()
    m = re.search(r"EDGE_C_STATS\s*=\s*\{(.*?)\n\}", edge, re.S)
    res.add(
        "C28",
        "EDGE_C_STATS literal",
        " ".join(m.group(1).split())[:200] if m else "not found",
        source="scripts/edge_c_dashboard.py",
    )

    meta = json.loads((REPO / "models/xgboost/player_props_v1_metadata.json").read_text())
    same_game = [
        c for c in meta["feature_cols"] if c in ("log_attempts", "yards_per_attempt", "td_rate")
    ]
    res.add(
        "C37",
        "player_props_v1 test_mae / same-game features",
        f"{meta['test_mae']:.3f} / {', '.join(same_game)}",
        source="models/xgboost/player_props_v1_metadata.json",
        note="yards = attempts x yards_per_attempt of the same game",
    )

    for path, needle in (
        ("nfl_analytics/backtest/historical_validation_5way.py", "np.random.uniform(-15, 15)"),
        (
            "nfl_analytics/backtest/historical_validation_4way_optimized.py",
            "np.random.uniform(-15, 15)",
        ),
        ("nfl_analytics/backtests/comprehensive_ensemble_backtest.py", "np.random.normal(1"),
        (
            "nfl_analytics/backtests/bayesian_props_backtest_2024.py",
            "* np.random.uniform(",  # prop_line = bayes_prediction * U(0.95, 1.05)
        ),
    ):
        f = REPO / path
        hit = (
            [i for i, line in enumerate(f.read_text().splitlines(), 1) if needle in line]
            if f.is_file()
            else []
        )
        res.add(
            "C36",
            f"{path} builds the prop line from noise (`{needle}`)",
            (
                f"lines {', '.join(map(str, hit))}"
                if hit
                else ("file missing" if not f.is_file() else "not found")
            ),
            source=path,
        )

    for path in (
        "analysis/backtest/fixed_backtest_results.json",
        "analysis/backtest/recent_seasons_test_results.json",
    ):
        d = json.loads((REPO / path).read_text())
        res.add(
            "C31",
            f"{path} test_accuracy (stored)",
            f"{d['test_accuracy']:.4f}",
            d.get("test_games"),
            path,
        )
    for cid, path, key in (
        ("C20", "models/ensemble/backtest/summary.json", "win_rate"),
        ("C21", "models/ensemble/realistic_backtest/summary.json", "win_rate"),
    ):
        d = json.loads((REPO / path).read_text())
        res.add(
            cid,
            f"{path} {key} / final_bankroll (stored)",
            f"{d[key]:.4f} / {d['final_bankroll']:.3g}",
            d.get("n_trades"),
            path,
        )
    ope = (out / "ope_grid_table.tex").read_text()
    m = re.search(r"median DR=([\d.]+)", ope)
    res.add(
        "C41",
        "ope_grid_table.tex median DR (printed)",
        m.group(1) if m else "not found",
        source="analysis/dissertation/figures/out/ope_grid_table.tex",
    )

    v2 = REPO / "models/production/game_outcome_v2"
    meta = json.loads((v2 / "metadata.json").read_text())["validation_metrics"]
    for model, metric in (("spread_model.json", "spread_mae"), ("total_model.json", "total_mae")):
        attrs = json.loads((v2 / model).read_text())["learner"]["attributes"]
        res.add(
            "C17",
            f"{model} early-stopping best_score vs metadata validation {metric}",
            f"{float(attrs['best_score']):.6f} vs {meta[metric]:.6f} "
            f"(best_iteration {attrs['best_iteration']})",
            source="models/production/game_outcome_v2/",
            note="the validation metric is the early-stopping objective on the same season",
        )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--nflverse-games", type=Path, help="nflverse schedules games.csv (optional)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build = dissertation_build_files()
    locs = locate(build)
    locs.to_csv(args.out_dir / "claims_locations.csv", index=False)
    summary = summarize(locs)
    summary.to_csv(args.out_dir / "claims_summary.csv", index=False)
    print(f"dissertation build: {len(build)} .tex files; claim lines found: {len(locs)}")
    print(
        summary[
            ["claim", "best_prominence", "lines", "lines_asserting", "lines_flagging"]
        ].to_string(index=False)
    )

    res = Results()
    db = ReadOnlyDB()
    try:
        games = load_games(db)
        checks = [
            ("C02-C04,C16,C26,C27,C29,C30", lambda: check_v2_2025(db, games, res)),
            ("C01", lambda: check_power_ratings_dec9(games, res)),
            ("C42", lambda: check_power_ratings_repo(games, res)),
            ("C05", lambda: check_readme_baselines(res)),
            ("C15", lambda: check_rest(db, games, res)),
            ("C18", lambda: check_state_space_metrics(games, res)),
            ("C19", lambda: check_ensemble_csv(games, res)),
            ("C20,C21", lambda: check_trade_logs(games, res)),
            ("C22", lambda: check_bayes_xgb(games, res)),
            ("C23,C25", lambda: check_semantic_ab(games, res)),
            ("C09,C36", lambda: check_props(db, res)),
            ("C31", lambda: check_nov2025_backtests(games, res)),
            ("C32", lambda: check_weather(db, res, args.nflverse_games)),
            ("C40", lambda: check_teasers(games, res)),
            ("C41", lambda: check_rl_logged(games, res)),
            ("static", lambda: check_static(res)),
        ]
        for name, fn in checks:
            try:
                fn()
            except Exception as exc:  # keep going; the CSV records what failed
                res.failed(name, "check raised", exc)
    finally:
        db.close()
    out = pd.DataFrame(res.rows)
    from_nflverse = out["source"].str.startswith("nflverse")
    out[~from_nflverse].to_csv(args.out_dir / "claims_checks.csv", index=False)
    if args.nflverse_games is not None:
        out[from_nflverse].to_csv(args.out_dir / "claims_checks_nflverse.csv", index=False)
    with pd.option_context("display.max_colwidth", 140, "display.width", 250):
        print(out[["claim", "check", "value", "n"]].to_string(index=False))
    print(f"\nWrote claims_locations.csv, claims_summary.csv, claims_checks*.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
