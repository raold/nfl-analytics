#!/usr/bin/env python3
"""
Power Ratings Model - Point Differential Baseline

Calculates team power ratings from exponentially weighted point differential
and applies a fixed 2.5-point home field advantage to generate spread predictions.

Investigation finding: "Simple point differential + 2.5 HFA achieved +43% ROI
on 2025 holdout. Single highest-ROI opportunity with lowest effort."

Spread convention (nflverse): positive = home favored, actual = home - away score.

Usage:
    python nfl_analytics/models/power_ratings_model.py --seasons 2020-2025
    python nfl_analytics/models/power_ratings_model.py --seasons 2025 --min-edge 1.5
    python nfl_analytics/models/power_ratings_model.py --seasons 2020-2025 --sweep
"""

import argparse
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HFA_DEFAULT = 2.5
HALF_LIFE_DEFAULT = 10
MIN_GAMES_DEFAULT = 4


@dataclass
class BacktestResult:
    """Single season backtest metrics."""
    season: int
    total_games: int
    qualified_bets: int
    ats_wins: int
    ats_losses: int
    ats_pushes: int
    ats_pct: float
    roi_pct: float
    pred_mae: float
    vegas_mae: float
    edge_threshold: float
    per_week: dict = field(default_factory=dict)


def get_db_connection() -> psycopg2.extensions.connection:
    """Create database connection using env vars with dev fallback."""
    return psycopg2.connect(
        dbname=os.environ.get("NFL_DB_NAME", "devdb01"),
        user=os.environ.get("NFL_DB_USER", "dro"),
        password=os.environ.get("NFL_DB_PASSWORD", "sicillionbillions"),
        host=os.environ.get("NFL_DB_HOST", "localhost"),
        port=int(os.environ.get("NFL_DB_PORT", "5544")),
    )


def load_games(seasons: list[int]) -> pd.DataFrame:
    """Load completed games with scores and spreads from the database."""
    ph = ",".join(["%s"] * len(seasons))
    query = f"""
        SELECT game_id, season, week, game_type,
               home_team, away_team, home_score, away_score, spread_close
        FROM games
        WHERE season IN ({ph}) AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY season, week, kickoff
    """
    cols = ["game_id", "season", "week", "game_type",
            "home_team", "away_team", "home_score", "away_score", "spread_close"]
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, seasons)
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    logger.info("Loaded %d games across seasons %s", len(df), seasons)
    return df


def compute_power_ratings(
    games_df: pd.DataFrame, season: int, through_week: int,
    half_life: float = HALF_LIFE_DEFAULT,
) -> dict[str, float]:
    """Compute exponentially weighted point differential ratings per team.

    Uses games in the given season through `through_week`. Recent games receive
    higher weight via exponential decay controlled by `half_life` (in games).
    """
    decay = np.log(2) / half_life
    mask = (games_df["season"] == season) & (games_df["week"] <= through_week)
    sg = games_df[mask]
    if sg.empty:
        return {}

    ratings: dict[str, float] = {}
    for team in set(sg["home_team"]) | set(sg["away_team"]):
        home_g = sg[sg["home_team"] == team]
        away_g = sg[sg["away_team"] == team]
        margins = np.concatenate([
            (home_g["home_score"] - home_g["away_score"]).values,
            (away_g["away_score"] - away_g["home_score"]).values,
        ]).astype(float)
        weeks = np.concatenate([home_g["week"].values, away_g["week"].values]).astype(float)
        if len(margins) == 0:
            continue
        weights = np.exp(-decay * (through_week - weeks))
        ratings[team] = float(np.average(margins, weights=weights))
    return ratings


def predict_spread(home_rating: float, away_rating: float, hfa: float = HFA_DEFAULT) -> float:
    """Predict home margin of victory: home_rating - away_rating + HFA."""
    return home_rating - away_rating + hfa


def _roi_from_pct(pct: float) -> float:
    """ROI at -110 odds given win percentage."""
    return (pct * (100 / 110) - (1 - pct)) * 100


def backtest_season(
    games_df: pd.DataFrame, season: int, half_life: float = HALF_LIFE_DEFAULT,
    hfa: float = HFA_DEFAULT, min_games: int = MIN_GAMES_DEFAULT, min_edge: float = 0.0,
) -> BacktestResult:
    """Walk-forward backtest for one season.

    Each week: compute ratings from prior weeks, predict spreads, compare to Vegas.
    Bet the side where |model_spread - vegas_spread| >= min_edge.
    """
    sg = games_df[games_df["season"] == season]
    weeks = sorted(sg["week"].unique())
    stats = {"games": 0, "bets": 0, "w": 0, "l": 0, "p": 0}
    pred_err, vegas_err, per_week = [], [], {}

    for week in weeks:
        if week < min_games + 1:
            continue
        ratings = compute_power_ratings(games_df, season, week - 1, half_life)
        wk_bets, wk_wins = 0, 0

        for _, g in sg[sg["week"] == week].iterrows():
            home, away = g["home_team"], g["away_team"]
            actual = float(g["home_score"] - g["away_score"])
            vegas = g["spread_close"]
            if home not in ratings or away not in ratings or vegas is None or np.isnan(vegas):
                continue

            pred = predict_spread(ratings[home], ratings[away], hfa)
            stats["games"] += 1
            pred_err.append(abs(pred - actual))
            vegas_err.append(abs(vegas - actual))

            edge = pred - vegas
            if abs(edge) < min_edge:
                continue
            stats["bets"] += 1
            wk_bets += 1

            # Bet home if edge > 0, else bet away
            home_covers = actual > vegas
            home_fails = actual < vegas
            if (edge > 0 and home_covers) or (edge < 0 and home_fails):
                stats["w"] += 1
                wk_wins += 1
            elif (edge > 0 and home_fails) or (edge < 0 and home_covers):
                stats["l"] += 1
            else:
                stats["p"] += 1

        if wk_bets > 0:
            per_week[week] = {"bets": wk_bets, "wins": wk_wins}

    decided = stats["w"] + stats["l"]
    pct = stats["w"] / decided if decided > 0 else 0.0
    return BacktestResult(
        season=season, total_games=stats["games"], qualified_bets=stats["bets"],
        ats_wins=stats["w"], ats_losses=stats["l"], ats_pushes=stats["p"],
        ats_pct=pct, roi_pct=_roi_from_pct(pct) if decided > 0 else 0.0,
        pred_mae=float(np.mean(pred_err)) if pred_err else 0.0,
        vegas_mae=float(np.mean(vegas_err)) if vegas_err else 0.0,
        edge_threshold=min_edge, per_week=per_week,
    )


def run_backtest(
    seasons: list[int], half_life: float = HALF_LIFE_DEFAULT, hfa: float = HFA_DEFAULT,
    min_games: int = MIN_GAMES_DEFAULT, min_edge: float = 0.0,
) -> list[BacktestResult]:
    """Run walk-forward backtests across multiple seasons."""
    games_df = load_games(seasons)
    return [backtest_season(games_df, s, half_life, hfa, min_games, min_edge)
            for s in sorted(seasons)]


def print_results(results: list[BacktestResult], hfa: float = HFA_DEFAULT,
                  half_life: float = HALF_LIFE_DEFAULT) -> None:
    """Print formatted backtest results table."""
    edge = results[0].edge_threshold if results else 0.0
    print(f"\n{'=' * 80}")
    print("POWER RATINGS MODEL - BACKTEST RESULTS")
    print(f"Parameters: HFA={hfa}, half_life={half_life}, min_edge={edge}")
    print("=" * 80)
    print(f"{'Season':>6} | {'Games':>5} | {'Bets':>4} | {'ATS W-L-P':>10} | "
          f"{'ATS%':>6} | {'ROI%':>7} | {'Pred MAE':>8} | {'Vegas MAE':>9}")
    print("-" * 80)

    tw, tl, tp = 0, 0, 0
    pred_maes, vegas_maes = [], []
    for r in results:
        rec = f"{r.ats_wins}-{r.ats_losses}-{r.ats_pushes}"
        print(f"{r.season:>6} | {r.total_games:>5} | {r.qualified_bets:>4} | {rec:>10} | "
              f"{r.ats_pct:>5.1%} | {r.roi_pct:>+6.1f}% | {r.pred_mae:>8.2f} | {r.vegas_mae:>9.2f}")
        tw += r.ats_wins; tl += r.ats_losses; tp += r.ats_pushes
        if r.pred_mae > 0: pred_maes.append(r.pred_mae)
        if r.vegas_mae > 0: vegas_maes.append(r.vegas_mae)

    print("-" * 80)
    td = tw + tl
    tpct = tw / td if td > 0 else 0.0
    troi = _roi_from_pct(tpct) if td > 0 else 0.0
    tb = sum(r.qualified_bets for r in results)
    trec = f"{tw}-{tl}-{tp}"
    pm = float(np.mean(pred_maes)) if pred_maes else 0.0
    vm = float(np.mean(vegas_maes)) if vegas_maes else 0.0
    print(f"{'TOTAL':>6} | {'':>5} | {tb:>4} | {trec:>10} | "
          f"{tpct:>5.1%} | {troi:>+6.1f}% | {pm:>8.2f} | {vm:>9.2f}")
    print("=" * 80)
    print(f"\nBreakeven at -110: 52.4%")
    print(f"Overall ATS: {tpct:.1%} ({'PROFITABLE' if tpct > 0.524 else 'UNPROFITABLE'})")
    print(f"Model MAE vs Vegas MAE: {pm:.2f} vs {vm:.2f}\n")


def parse_seasons(s: str) -> list[int]:
    """Parse '2020-2025' or '2024' into list of ints."""
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main() -> None:
    """CLI entry point for power ratings backtests."""
    parser = argparse.ArgumentParser(description="Power Ratings: point diff + HFA baseline")
    parser.add_argument("--seasons", default="2020-2025", help="e.g. '2020-2025' or '2024'")
    parser.add_argument("--half-life", type=float, default=HALF_LIFE_DEFAULT,
                        help=f"Decay half-life in games (default: {HALF_LIFE_DEFAULT})")
    parser.add_argument("--hfa", type=float, default=HFA_DEFAULT,
                        help=f"Home field advantage points (default: {HFA_DEFAULT})")
    parser.add_argument("--min-games", type=int, default=MIN_GAMES_DEFAULT,
                        help=f"Min games before rating trusted (default: {MIN_GAMES_DEFAULT})")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Min model-vs-Vegas edge to bet (default: 0.0)")
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep across HFA/HL/edge")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons)

    if args.sweep:
        print("\n=== PARAMETER SWEEP ===\n")
        print(f"{'HFA':>5} | {'HL':>4} | {'Edge':>4} | {'Bets':>5} | {'ATS%':>6} | {'ROI%':>7}")
        print("-" * 50)
        best_roi, best_params = -999.0, {}
        games_df = load_games(seasons)
        for hfa in [1.5, 2.0, 2.5, 3.0, 3.5]:
            for hl in [5, 8, 10, 14, 18]:
                for edge in [0.0, 1.0, 1.5, 2.0, 3.0]:
                    res = [backtest_season(games_df, s, hl, hfa, args.min_games, edge)
                           for s in sorted(seasons)]
                    w = sum(r.ats_wins for r in res)
                    l = sum(r.ats_losses for r in res)
                    b = sum(r.qualified_bets for r in res)
                    d = w + l
                    if d < 20:
                        continue
                    pct = w / d
                    roi = _roi_from_pct(pct)
                    print(f"{hfa:>5.1f} | {hl:>4} | {edge:>4.1f} | {b:>5} | {pct:>5.1%} | {roi:>+6.1f}%")
                    if roi > best_roi:
                        best_roi = roi
                        best_params = {"hfa": hfa, "half_life": hl, "min_edge": edge}
        print(f"\nBest: HFA={best_params.get('hfa')}, HL={best_params.get('half_life')}, "
              f"Edge={best_params.get('min_edge')}, ROI={best_roi:+.1f}%")
    else:
        results = run_backtest(seasons, args.half_life, args.hfa, args.min_games, args.min_edge)
        print_results(results, hfa=args.hfa, half_life=args.half_life)


if __name__ == "__main__":
    main()
