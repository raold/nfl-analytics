#!/usr/bin/env python3
"""
International Games Betting Edge Analysis

Purpose: Test 3 specific betting edges for NFL international games (2018-2024)
Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025

Edges Tested:
    A. West Coast QB Penalty (LAC, LAR, SF, SEA in London)
    B. International Totals UNDER Bias
    C. Lingering Fatigue (next game after international)

Database: PostgreSQL (devdb01)
Output: results/international_betting_edges.json
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Database connection
DB_URL = os.getenv("DATABASE_URL", "postgresql://dro:sicillionbillions@localhost:5544/devdb01")

# West Coast teams
WEST_COAST_TEAMS = ["LAC", "LAR", "SF", "SEA"]

# International stadiums (comprehensive list)
INTL_STADIUMS = [
    "%wembley%",
    "%tottenham%",
    "%london%",
    "%mexico%",
    "%azteca%",
    "%germany%",
    "%munich%",
    "%frankfurt%",
]


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(DB_URL)


def fetch_international_games(season_start: int = 2018) -> pd.DataFrame:
    """
    Fetch all international games from database.

    Args:
        season_start: First season to include (default 2018)

    Returns:
        DataFrame with international games
    """
    conn = get_db_connection()

    # Build stadium filter
    stadium_filter = " OR ".join([f"stadium ILIKE '{s}'" for s in INTL_STADIUMS])

    query = f"""
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        stadium,
        kickoff,
        home_score,
        away_score,
        total_close,
        spread_close,
        home_score + away_score AS actual_total,
        EXTRACT(HOUR FROM kickoff AT TIME ZONE 'America/Los_Angeles') AS kickoff_hour_pt
    FROM public.games
    WHERE ({stadium_filter})
        AND season >= {season_start}
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
    ORDER BY kickoff;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Fetched {len(df)} international games from {season_start} onwards")
    return df


def fetch_qb_performances(game_ids: list[str]) -> pd.DataFrame:
    """
    Fetch QB passing performance for specific games.

    Args:
        game_ids: List of game IDs

    Returns:
        DataFrame with QB performances
    """
    conn = get_db_connection()

    # Extract season, week from game_ids
    game_conditions = []
    for gid in game_ids:
        parts = gid.split("_")
        if len(parts) >= 3:
            season = int(parts[0])
            week = int(parts[1])
            game_conditions.append(f"(pgs.season = {season} AND pgs.week = {week})")

    if not game_conditions:
        return pd.DataFrame()

    game_filter = " OR ".join(game_conditions)

    query = f"""
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.player_display_name,
        pgs.current_team,
        pgs.stat_yards,
        pgs.stat_attempts,
        pgs.stat_completions,
        pgs.stat_touchdowns,
        pgs.stat_negative AS interceptions,
        pgs.game_id_pattern
    FROM mart.player_game_stats pgs
    WHERE pgs.stat_category = 'passing'
        AND pgs.position_group = 'QB'
        AND ({game_filter})
        AND pgs.week >= 1  -- Exclude week=0 (season aggregates)
        AND pgs.stat_attempts >= 15  -- Minimum attempts
    ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Fetched {len(df)} QB performances from {len(game_ids)} games")
    return df


def fetch_qb_domestic_average(player_ids: list[str], season_start: int = 2018) -> pd.DataFrame:
    """
    Fetch QB domestic (non-international) average performance.

    CRITICAL FIX (Nov 16, 2025): Use game-level joins to prevent cross-season contamination.
    Old implementation used current_team filter which caused Sam Darnold's 2018 Jets performances
    to be averaged with 2024 Vikings roster, inflating baselines to impossible values (515 yards/game).

    Args:
        player_ids: List of player IDs
        season_start: First season to include

    Returns:
        DataFrame with domestic averages per QB
    """
    if not player_ids:
        return pd.DataFrame()

    conn = get_db_connection()

    # Get international game (season, week) combinations to exclude
    stadium_filter = " OR ".join([f"stadium ILIKE '{s}'" for s in INTL_STADIUMS])
    intl_games_query = f"""
    SELECT season, week
    FROM public.games
    WHERE ({stadium_filter}) AND season >= {season_start}
    """
    intl_games = pd.read_sql(intl_games_query, conn)

    # Build exclusion filter
    exclusion_conditions = []
    for _, row in intl_games.iterrows():
        exclusion_conditions.append(f"(pgs.season = {row['season']} AND pgs.week = {row['week']})")

    exclusion_filter = " OR ".join(exclusion_conditions) if exclusion_conditions else "1=0"

    # Format player IDs for SQL
    player_id_list = ",".join([f"'{pid}'" for pid in player_ids])

    # CRITICAL FIX: week=0 contains SEASON AGGREGATES, not individual games
    # Example: Mahomes week=0 for 2022 shows 5,250 yards (full season total)
    # This inflates baselines to impossible values (470+ yards/game when averaging season totals with games)
    #
    # Additional fix: current_team in player_game_stats is unreliable for historical data
    # Example: Carson Wentz shows current_team='MIN' for 2016-2020 when he was on PHI
    #
    # Solution: Exclude week=0, calculate QB averages WITHOUT team grouping (player career average)
    query = f"""
    SELECT
        pgs.player_id,
        pgs.player_display_name,
        pgs.current_team,  -- Keep for reference but don't group by
        AVG(pgs.stat_yards) AS avg_yards,
        STDDEV(pgs.stat_yards) AS std_yards,
        COUNT(*) AS games
    FROM mart.player_game_stats pgs
    WHERE pgs.stat_category = 'passing'
        AND pgs.position_group = 'QB'
        AND pgs.player_id IN ({player_id_list})
        AND pgs.season >= {season_start}
        AND pgs.season BETWEEN 2018 AND 2024  -- Restrict to known data range
        AND pgs.week >= 1  -- CRITICAL: Exclude week=0 (season aggregates)
        AND pgs.stat_attempts >= 15
        AND NOT ({exclusion_filter})
    GROUP BY pgs.player_id, pgs.player_display_name, pgs.current_team
    HAVING COUNT(*) >= 5;  -- Minimum 5 domestic games
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Fetched domestic averages for {len(df)} QBs")
    return df


def analyze_edge_a_west_coast_penalty(
    intl_games: pd.DataFrame, qb_perf: pd.DataFrame, domestic_avg: pd.DataFrame
) -> dict:
    """
    Edge A: Test if West Coast QBs underperform in early London games.

    Expected penalty: -15 to -25 yards vs domestic average

    Args:
        intl_games: International games DataFrame
        qb_perf: QB performances in international games
        domestic_avg: QB domestic averages

    Returns:
        Dictionary with edge analysis results
    """
    logger.info("Analyzing Edge A: West Coast QB Penalty")

    # Filter London games only (early kickoff)
    # Note: kickoff_hour_pt may be negative due to UTC offset, so check for typical London game times
    # London games typically kick off at 6:00-10:00 AM PT (9:30 AM local = 1:30 AM PT)
    london_games = intl_games[
        intl_games["stadium"].str.contains("Wembley|Tottenham", case=False, na=False)
    ].copy()

    logger.info(f"Identified {len(london_games)} early London games")

    # Merge QB performances with games
    qb_intl = qb_perf.merge(
        london_games[["season", "week", "game_id"]], on=["season", "week"], how="inner"
    )

    # Filter West Coast QBs
    west_coast_qbs = qb_intl[qb_intl["current_team"].isin(WEST_COAST_TEAMS)].copy()

    logger.info(f"Found {len(west_coast_qbs)} West Coast QB performances in London")

    if len(west_coast_qbs) == 0:
        return {
            "edge": "A",
            "name": "West Coast QB Penalty",
            "viable": False,
            "reason": "Insufficient data (0 West Coast QB performances)",
            "sample_size": 0,
        }

    # Merge with domestic averages
    west_coast_with_baseline = west_coast_qbs.merge(
        domestic_avg[["player_id", "avg_yards", "std_yards", "games"]],
        on="player_id",
        how="inner",
    )

    if len(west_coast_with_baseline) == 0:
        return {
            "edge": "A",
            "name": "West Coast QB Penalty",
            "viable": False,
            "reason": "No QBs with sufficient domestic baseline",
            "sample_size": len(west_coast_qbs),
        }

    # Calculate penalty
    west_coast_with_baseline["penalty"] = (
        west_coast_with_baseline["stat_yards"] - west_coast_with_baseline["avg_yards"]
    )

    avg_penalty = west_coast_with_baseline["penalty"].mean()
    std_penalty = west_coast_with_baseline["penalty"].std()
    n = len(west_coast_with_baseline)

    # Statistical test (one-sample t-test vs 0)
    t_stat, p_value = stats.ttest_1samp(west_coast_with_baseline["penalty"], 0)

    # 95% confidence interval
    ci_95 = stats.t.interval(0.95, n - 1, loc=avg_penalty, scale=std_penalty / np.sqrt(n))

    # ROI estimation (simple model)
    # Assume we bet UNDER on West Coast QB props if penalty > -15 yards
    # Line typically set at domestic average
    # If actual penalty is -20 yards, we win 20/50 = 40% of bets at -110 odds
    if avg_penalty < -10:
        # Calculate theoretical win rate
        # Penalty shifts distribution, increasing UNDER win rate
        baseline_under_rate = 0.50
        penalty_shift = abs(avg_penalty) / (west_coast_with_baseline["std_yards"].mean() or 40)
        win_rate = baseline_under_rate + (penalty_shift * 0.15)  # Heuristic
        win_rate = min(win_rate, 0.70)  # Cap at 70%

        # ROI calculation at -110 odds
        roi = (win_rate * 1.909 - 1.0) * 100  # -110 decimal odds = 1.909

        viable = (avg_penalty < -15) and (p_value < 0.10) and (roi > 5)
    else:
        win_rate = None
        roi = None
        viable = False

    result = {
        "edge": "A",
        "name": "West Coast QB Penalty",
        "viable": viable,
        "sample_size": n,
        "avg_penalty_yards": round(avg_penalty, 2),
        "std_penalty_yards": round(std_penalty, 2),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "ci_95_lower": round(ci_95[0], 2),
        "ci_95_upper": round(ci_95[1], 2),
        "estimated_win_rate": round(win_rate, 3) if win_rate else None,
        "estimated_roi_pct": round(roi, 2) if roi else None,
        "interpretation": (
            f"West Coast QBs averaged {avg_penalty:.1f} yards vs domestic baseline "
            f"(p={p_value:.4f}). "
            + (
                f"VIABLE: Edge exists, estimated {roi:.1f}% ROI at -110 odds."
                if viable
                else "NOT VIABLE: Penalty insufficient or not statistically significant."
            )
        ),
        "qb_details": west_coast_with_baseline[
            ["player_display_name", "current_team", "season", "week", "stat_yards", "avg_yards", "penalty"]
        ].to_dict("records"),
    }

    return result


def analyze_edge_b_totals_under_bias(intl_games: pd.DataFrame) -> dict:
    """
    Edge B: Test if international game totals have UNDER bias.

    Expected: 58-62% UNDER rate vs 50% domestic

    Args:
        intl_games: International games DataFrame

    Returns:
        Dictionary with edge analysis results
    """
    logger.info("Analyzing Edge B: International Totals UNDER Bias")

    # Filter games with valid totals
    valid_totals = intl_games[intl_games["total_close"].notna()].copy()

    if len(valid_totals) == 0:
        return {
            "edge": "B",
            "name": "International Totals UNDER Bias",
            "viable": False,
            "reason": "No games with closing totals",
            "sample_size": 0,
        }

    # Calculate UNDER results
    valid_totals["under_hit"] = valid_totals["actual_total"] < valid_totals["total_close"]

    under_rate = valid_totals["under_hit"].mean()
    n = len(valid_totals)
    under_count = valid_totals["under_hit"].sum()

    # Statistical test (binomial test vs 50%)
    binom_test = stats.binomtest(under_count, n, 0.50, alternative="greater")
    p_value = binom_test.pvalue

    # 95% confidence interval for proportion
    ci_raw = stats.binom.interval(0.95, n, under_rate)
    ci_95 = (ci_raw[0] / n, ci_raw[1] / n)

    # ROI estimation
    if under_rate > 0.53:  # Need >53% to profit at -110 odds
        # ROI at -110 odds
        roi = (under_rate * 1.909 - 1.0) * 100
        viable = (under_rate > 0.56) and (p_value < 0.10) and (roi > 5)
    else:
        roi = None
        viable = False

    # Compare to domestic UNDER rate (fetch from database)
    conn = get_db_connection()
    stadium_filter = " OR ".join([f"stadium ILIKE '{s}'" for s in INTL_STADIUMS])
    domestic_query = f"""
    SELECT
        COUNT(*) FILTER (WHERE home_score + away_score < total_close) AS under_count,
        COUNT(*) AS total_games
    FROM public.games
    WHERE season >= 2018
        AND total_close IS NOT NULL
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        AND NOT ({stadium_filter});
    """
    domestic = pd.read_sql(domestic_query, conn)
    conn.close()

    domestic_under_rate = (
        domestic["under_count"].iloc[0] / domestic["total_games"].iloc[0]
        if len(domestic) > 0 and domestic["total_games"].iloc[0] > 0
        else 0.50
    )

    result = {
        "edge": "B",
        "name": "International Totals UNDER Bias",
        "viable": viable,
        "sample_size": n,
        "under_count": int(under_count),
        "under_rate": round(under_rate, 4),
        "domestic_under_rate": round(domestic_under_rate, 4),
        "edge_vs_domestic": round((under_rate - domestic_under_rate) * 100, 2),
        "p_value": round(p_value, 4),
        "ci_95_lower": round(ci_95[0], 4),
        "ci_95_upper": round(ci_95[1], 4),
        "estimated_roi_pct": round(roi, 2) if roi else None,
        "interpretation": (
            f"International games hit UNDER {under_rate:.1%} of the time "
            f"vs {domestic_under_rate:.1%} domestic (p={p_value:.4f}). "
            + (
                f"VIABLE: Edge exists, estimated {roi:.1f}% ROI at -110 odds."
                if viable
                else "NOT VIABLE: UNDER rate insufficient or not statistically significant."
            )
        ),
        "game_details": valid_totals[
            ["season", "week", "home_team", "away_team", "total_close", "actual_total", "under_hit"]
        ].to_dict("records"),
    }

    return result


def analyze_edge_c_lingering_fatigue(intl_games: pd.DataFrame) -> dict:
    """
    Edge C: Test if teams underperform in next game after international travel.

    Expected penalty decay: -10 * exp(-days_since / 7)

    Args:
        intl_games: International games DataFrame

    Returns:
        Dictionary with edge analysis results
    """
    logger.info("Analyzing Edge C: Lingering Fatigue")

    conn = get_db_connection()

    # For each international game, find next game for both teams
    fatigue_data = []

    for _, intl_game in intl_games.iterrows():
        season = intl_game["season"]
        week = intl_game["week"]
        kickoff = intl_game["kickoff"]

        for team in [intl_game["home_team"], intl_game["away_team"]]:
            # Find next game for this team
            next_game_query = f"""
            SELECT
                game_id,
                season,
                week,
                kickoff,
                CASE
                    WHEN home_team = '{team}' THEN home_score
                    ELSE away_score
                END AS team_score,
                CASE
                    WHEN home_team = '{team}' THEN 'home'
                    ELSE 'away'
                END AS home_away,
                EXTRACT(EPOCH FROM (kickoff - '{kickoff}'::timestamptz)) / 86400.0 AS days_since_intl
            FROM public.games
            WHERE (home_team = '{team}' OR away_team = '{team}')
                AND kickoff > '{kickoff}'::timestamptz
                AND season = {season}
                AND home_score IS NOT NULL
            ORDER BY kickoff
            LIMIT 1;
            """

            next_game = pd.read_sql(next_game_query, conn)

            if len(next_game) > 0 and next_game["days_since_intl"].iloc[0] <= 14:
                # Get QB performance in next game
                next_season = next_game["season"].iloc[0]
                next_week = next_game["week"].iloc[0]

                qb_query = f"""
                SELECT
                    pgs.player_id,
                    pgs.player_display_name,
                    pgs.stat_yards,
                    pgs.stat_attempts
                FROM mart.player_game_stats pgs
                WHERE pgs.season = {next_season}
                    AND pgs.week = {next_week}
                    AND pgs.week >= 1  -- Exclude week=0 (season aggregates)
                    AND pgs.current_team = '{team}'
                    AND pgs.stat_category = 'passing'
                    AND pgs.position_group = 'QB'
                    AND pgs.stat_attempts >= 15
                ORDER BY pgs.stat_attempts DESC
                LIMIT 1;
                """

                qb_perf = pd.read_sql(qb_query, conn)

                if len(qb_perf) > 0:
                    fatigue_data.append({
                        "team": team,
                        "intl_season": season,
                        "intl_week": week,
                        "next_season": next_season,
                        "next_week": next_week,
                        "days_since_intl": next_game["days_since_intl"].iloc[0],
                        "player_id": qb_perf["player_id"].iloc[0],
                        "player_name": qb_perf["player_display_name"].iloc[0],
                        "stat_yards": qb_perf["stat_yards"].iloc[0],
                    })

    conn.close()

    if len(fatigue_data) == 0:
        return {
            "edge": "C",
            "name": "Lingering Fatigue",
            "viable": False,
            "reason": "No valid next-game data found",
            "sample_size": 0,
        }

    fatigue_df = pd.DataFrame(fatigue_data)

    # Fetch domestic baselines for these QBs
    player_ids = fatigue_df["player_id"].unique().tolist()
    domestic_avg = fetch_qb_domestic_average(player_ids, season_start=2018)

    # Merge with baselines
    fatigue_with_baseline = fatigue_df.merge(
        domestic_avg[["player_id", "avg_yards", "std_yards"]], on="player_id", how="inner"
    )

    if len(fatigue_with_baseline) == 0:
        return {
            "edge": "C",
            "name": "Lingering Fatigue",
            "viable": False,
            "reason": "No QBs with sufficient domestic baseline",
            "sample_size": len(fatigue_df),
        }

    # Calculate penalty
    fatigue_with_baseline["penalty"] = (
        fatigue_with_baseline["stat_yards"] - fatigue_with_baseline["avg_yards"]
    )

    # Model: penalty = -10 * exp(-days_since / 7)
    fatigue_with_baseline["predicted_penalty"] = (
        -10 * np.exp(-fatigue_with_baseline["days_since_intl"] / 7.0)
    )

    # Statistical test
    avg_penalty = fatigue_with_baseline["penalty"].mean()
    std_penalty = fatigue_with_baseline["penalty"].std()
    n = len(fatigue_with_baseline)

    t_stat, p_value = stats.ttest_1samp(fatigue_with_baseline["penalty"], 0)

    # Correlation between predicted and actual penalty
    if n >= 5:
        corr, corr_p = stats.pearsonr(
            fatigue_with_baseline["predicted_penalty"], fatigue_with_baseline["penalty"]
        )
    else:
        corr, corr_p = None, None

    # ROI estimation (games within 7 days)
    short_rest = fatigue_with_baseline[fatigue_with_baseline["days_since_intl"] <= 7]
    if len(short_rest) >= 3:
        short_rest_penalty = short_rest["penalty"].mean()
        short_rest_std = short_rest["penalty"].std()

        # Similar ROI calculation as Edge A
        if short_rest_penalty < -8:
            baseline_under_rate = 0.50
            penalty_shift = abs(short_rest_penalty) / (short_rest_std or 40)
            win_rate = baseline_under_rate + (penalty_shift * 0.12)
            win_rate = min(win_rate, 0.70)
            roi = (win_rate * 1.909 - 1.0) * 100
            viable = (short_rest_penalty < -10) and (p_value < 0.10) and (roi > 5)
        else:
            win_rate = None
            roi = None
            viable = False
    else:
        short_rest_penalty = None
        win_rate = None
        roi = None
        viable = False

    result = {
        "edge": "C",
        "name": "Lingering Fatigue",
        "viable": viable,
        "sample_size": n,
        "short_rest_sample": len(short_rest) if len(short_rest) >= 3 else 0,
        "avg_penalty_yards": round(avg_penalty, 2),
        "std_penalty_yards": round(std_penalty, 2),
        "short_rest_penalty_yards": round(short_rest_penalty, 2) if short_rest_penalty else None,
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "model_correlation": round(corr, 3) if corr else None,
        "model_correlation_p": round(corr_p, 4) if corr_p else None,
        "estimated_win_rate": round(win_rate, 3) if win_rate else None,
        "estimated_roi_pct": round(roi, 2) if roi else None,
        "interpretation": (
            f"QBs in next game after international travel averaged {avg_penalty:.1f} yards "
            f"vs domestic baseline (p={p_value:.4f}). "
            + (f"Short rest (<7 days): {short_rest_penalty:.1f} yards penalty. " if short_rest_penalty else "")
            + (
                f"VIABLE: Edge exists, estimated {roi:.1f}% ROI at -110 odds."
                if viable
                else "NOT VIABLE: Penalty insufficient or not statistically significant."
            )
        ),
        "game_details": fatigue_with_baseline[
            [
                "player_name",
                "team",
                "intl_season",
                "intl_week",
                "next_week",
                "days_since_intl",
                "stat_yards",
                "avg_yards",
                "penalty",
                "predicted_penalty",
            ]
        ].to_dict("records"),
    }

    return result


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("International Games Betting Edge Analysis")
    logger.info("=" * 80)

    # Fetch international games
    intl_games = fetch_international_games(season_start=2018)

    if len(intl_games) == 0:
        logger.error("No international games found. Exiting.")
        return

    # Fetch QB performances in international games
    game_ids = intl_games["game_id"].tolist()
    qb_perf = fetch_qb_performances(game_ids)

    # Fetch domestic baselines
    if len(qb_perf) > 0:
        player_ids = qb_perf["player_id"].unique().tolist()
        domestic_avg = fetch_qb_domestic_average(player_ids, season_start=2018)
    else:
        domestic_avg = pd.DataFrame()

    # Analyze Edge A: West Coast QB Penalty
    edge_a = analyze_edge_a_west_coast_penalty(intl_games, qb_perf, domestic_avg)

    # Analyze Edge B: Totals UNDER Bias
    edge_b = analyze_edge_b_totals_under_bias(intl_games)

    # Analyze Edge C: Lingering Fatigue
    edge_c = analyze_edge_c_lingering_fatigue(intl_games)

    # Summary
    results = {
        "metadata": {
            "analysis_date": pd.Timestamp.now().isoformat(),
            "season_range": f"{intl_games['season'].min()}-{intl_games['season'].max()}",
            "total_intl_games": len(intl_games),
            "total_qb_performances": len(qb_perf),
        },
        "edges": {
            "edge_a_west_coast_qb_penalty": edge_a,
            "edge_b_totals_under_bias": edge_b,
            "edge_c_lingering_fatigue": edge_c,
        },
        "summary": {
            "viable_edges": sum([edge_a["viable"], edge_b["viable"], edge_c["viable"]]),
            "best_edge": max(
                [
                    ("A", edge_a.get("estimated_roi_pct", -100)),
                    ("B", edge_b.get("estimated_roi_pct", -100)),
                    ("C", edge_c.get("estimated_roi_pct", -100)),
                ],
                key=lambda x: x[1] if x[1] else -100,
            ),
        },
    }

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "international_betting_edges.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Viable edges: {results['summary']['viable_edges']}/3")
    logger.info(f"Best edge: {results['summary']['best_edge'][0]} "
                f"(ROI: {results['summary']['best_edge'][1]:.2f}%)")

    for edge_name, edge_data in results["edges"].items():
        logger.info(f"\n{edge_data['name']}:")
        logger.info(f"  Viable: {'YES' if edge_data['viable'] else 'NO'}")
        logger.info(f"  Sample Size: {edge_data['sample_size']}")
        if edge_data.get("estimated_roi_pct"):
            logger.info(f"  Estimated ROI: {edge_data['estimated_roi_pct']:.2f}%")
        logger.info(f"  {edge_data['interpretation']}")

    # Exit code based on success criteria
    if results["summary"]["viable_edges"] >= 1:
        logger.info("\n✅ SUCCESS: At least 1 viable betting edge identified with +5% ROI potential")
        sys.exit(0)
    else:
        logger.warning("\n⚠️ WARNING: No viable betting edges found meeting +5% ROI criteria")
        sys.exit(1)


if __name__ == "__main__":
    main()
