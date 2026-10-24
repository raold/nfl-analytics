#!/usr/bin/env python3
"""
Injury Weight Validation Script
===============================

Empirically validates the injury impact weights used in game_feature_extractor.py:
- QB_OUT_SPREAD_IMPACT: -5.0 points
- KEY_POSITION_SPREAD_IMPACT: -2.0 points
- OLINE_INJURY_SPREAD_IMPACT: -0.5 points per injury

Methodology:
1. Extract games with known injury data (2020-2024)
2. Compare actual ATS results against predicted adjustments
3. Calculate optimal weights via regression
4. Compare to current hardcoded values

Usage:
    python analysis/validation/validate_injury_weights.py

Output:
    - Summary statistics
    - Optimal weight estimates
    - Recommendation for py/config/constants.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import psycopg

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from py.config.constants import INJURY_WEIGHTS


@dataclass
class ValidationResult:
    """Results from injury weight validation."""

    # Sample size
    n_games: int
    n_games_with_qb_injury: int
    n_games_with_key_position_injury: int

    # Current weights (from constants.py)
    current_qb_weight: float
    current_key_position_weight: float
    current_oline_weight: float

    # Empirically estimated weights
    estimated_qb_weight: float
    estimated_key_position_weight: float
    estimated_oline_weight: float

    # Model fit
    r_squared: float
    mae: float

    # Recommendation
    recommendation: str


def get_connection() -> psycopg.Connection:
    """Create database connection."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def load_injury_game_data(conn: psycopg.Connection) -> pd.DataFrame:
    """
    Load games with injury data for validation.

    Returns DataFrame with:
    - Game metadata (season, week, teams, spread, margin)
    - Injury flags (qb_out, key_position_out, oline_injuries for both teams)
    - ATS result (home_covered)
    """
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.spread_close,
        g.home_score - g.away_score as home_margin,
        (g.home_score - g.away_score) - g.spread_close as home_ats_margin,
        CASE WHEN (g.home_score - g.away_score) - g.spread_close > 0 THEN 1
             WHEN (g.home_score - g.away_score) - g.spread_close < 0 THEN 0
             ELSE 0.5 END as home_covered,
        -- Home team injury data
        COALESCE(h_inj.qb_out::int, 0) as home_qb_out,
        COALESCE(h_inj.key_position_out::int, 0) as home_key_position_out,
        COALESCE(h_inj.oline_injuries, 0) as home_oline_injuries,
        COALESCE(h_inj.injury_severity_index, 0) as home_injury_severity,
        -- Away team injury data
        COALESCE(a_inj.qb_out::int, 0) as away_qb_out,
        COALESCE(a_inj.key_position_out::int, 0) as away_key_position_out,
        COALESCE(a_inj.oline_injuries, 0) as away_oline_injuries,
        COALESCE(a_inj.injury_severity_index, 0) as away_injury_severity
    FROM games g
    LEFT JOIN mart.team_injury_load h_inj
        ON g.season = h_inj.season
        AND g.week = h_inj.week
        AND g.home_team = h_inj.team
    LEFT JOIN mart.team_injury_load a_inj
        ON g.season = a_inj.season
        AND g.week = a_inj.week
        AND g.away_team = a_inj.team
    WHERE g.season >= 2020
        AND g.home_score IS NOT NULL
        AND g.spread_close IS NOT NULL
        AND (h_inj.team IS NOT NULL OR a_inj.team IS NOT NULL)
    ORDER BY g.season, g.week
    """

    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()

    return pd.DataFrame(data, columns=columns)


def calculate_injury_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add injury differential features for regression."""
    df = df.copy()

    # QB injury differential (home - away: positive = home QB out, bad for home)
    df["qb_injury_diff"] = df["home_qb_out"] - df["away_qb_out"]

    # Key position injury differential
    df["key_position_diff"] = df["home_key_position_out"] - df["away_key_position_out"]

    # O-line injury differential
    df["oline_injury_diff"] = df["home_oline_injuries"] - df["away_oline_injuries"]

    # Total severity differential
    df["severity_diff"] = df["home_injury_severity"] - df["away_injury_severity"]

    return df


def estimate_optimal_weights(df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """
    Estimate optimal injury weights via OLS regression.

    Model: home_ats_margin ~ qb_diff * w1 + key_pos_diff * w2 + oline_diff * w3

    Returns:
        Tuple of (qb_weight, key_pos_weight, oline_weight, r_squared, mae)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    # Prepare features
    X = df[["qb_injury_diff", "key_position_diff", "oline_injury_diff"]].values
    y = df["home_ats_margin"].values

    # Fit regression (weights should be negative since injuries hurt team)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Predictions and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Coefficients (negate since we want positive values for spread impact)
    qb_weight = -model.coef_[0]
    key_pos_weight = -model.coef_[1]
    oline_weight = -model.coef_[2]

    return qb_weight, key_pos_weight, oline_weight, r2, mae


def validate_current_weights(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Validate current weights by calculating predicted vs actual ATS margin.

    Returns:
        Tuple of (correlation, mae)
    """
    # Calculate predicted adjustment using current weights
    df = df.copy()
    df["predicted_adjustment"] = (
        df["qb_injury_diff"] * (-INJURY_WEIGHTS.QB_OUT_SPREAD_IMPACT) +
        df["key_position_diff"] * (-INJURY_WEIGHTS.KEY_POSITION_SPREAD_IMPACT) +
        df["oline_injury_diff"] * (-INJURY_WEIGHTS.OLINE_INJURY_SPREAD_IMPACT)
    )

    correlation = df["predicted_adjustment"].corr(df["home_ats_margin"])
    mae = np.mean(np.abs(df["predicted_adjustment"] - df["home_ats_margin"]))

    return correlation, mae


def run_validation() -> ValidationResult:
    """Run full validation and return results."""
    print("=" * 60)
    print("Injury Weight Validation")
    print("=" * 60)

    # Load data
    print("\n1. Loading game and injury data...")
    conn = get_connection()
    df = load_injury_game_data(conn)
    conn.close()

    print(f"   Loaded {len(df)} games with injury data (2020-2024)")

    # Add differential features
    df = calculate_injury_differential_features(df)

    # Summary statistics
    n_qb_injury = len(df[(df["home_qb_out"] == 1) | (df["away_qb_out"] == 1)])
    n_key_injury = len(df[(df["home_key_position_out"] > 0) | (df["away_key_position_out"] > 0)])

    print(f"   Games with QB injury: {n_qb_injury}")
    print(f"   Games with key position injury: {n_key_injury}")

    # Validate current weights
    print("\n2. Validating current weights from constants.py...")
    print(f"   Current QB weight: {INJURY_WEIGHTS.QB_OUT_SPREAD_IMPACT}")
    print(f"   Current key position weight: {INJURY_WEIGHTS.KEY_POSITION_SPREAD_IMPACT}")
    print(f"   Current O-line weight: {INJURY_WEIGHTS.OLINE_INJURY_SPREAD_IMPACT}")

    corr, current_mae = validate_current_weights(df)
    print(f"   Correlation with ATS margin: {corr:.3f}")
    print(f"   MAE: {current_mae:.2f} points")

    # Estimate optimal weights
    print("\n3. Estimating optimal weights via regression...")
    qb_w, key_w, oline_w, r2, opt_mae = estimate_optimal_weights(df)
    print(f"   Estimated QB weight: {qb_w:.2f}")
    print(f"   Estimated key position weight: {key_w:.2f}")
    print(f"   Estimated O-line weight: {oline_w:.2f}")
    print(f"   R-squared: {r2:.4f}")
    print(f"   MAE: {opt_mae:.2f} points")

    # Generate recommendation
    print("\n4. Recommendation...")

    qb_diff_pct = abs(qb_w - INJURY_WEIGHTS.QB_OUT_SPREAD_IMPACT) / INJURY_WEIGHTS.QB_OUT_SPREAD_IMPACT * 100
    key_diff_pct = abs(key_w - INJURY_WEIGHTS.KEY_POSITION_SPREAD_IMPACT) / INJURY_WEIGHTS.KEY_POSITION_SPREAD_IMPACT * 100

    if qb_diff_pct < 30 and key_diff_pct < 50:
        recommendation = "KEEP CURRENT WEIGHTS - Empirical estimates within acceptable range"
    elif r2 < 0.01:
        recommendation = "KEEP CURRENT WEIGHTS - Low R² suggests injuries have limited predictive value for ATS"
    else:
        recommendation = f"CONSIDER UPDATING - QB: {qb_w:.1f}, Key: {key_w:.1f}, OL: {oline_w:.2f}"

    print(f"   {recommendation}")

    # QB injury analysis
    print("\n5. QB Injury Deep Dive...")
    qb_games = df[(df["home_qb_out"] == 1) | (df["away_qb_out"] == 1)]
    if len(qb_games) > 10:
        # Games where home QB is out
        home_qb_out = df[df["home_qb_out"] == 1]
        away_qb_out = df[df["away_qb_out"] == 1]

        if len(home_qb_out) > 0:
            print(f"   Home QB out ({len(home_qb_out)} games):")
            print(f"     Avg ATS margin: {home_qb_out['home_ats_margin'].mean():.2f}")
            print(f"     Home cover rate: {home_qb_out['home_covered'].mean():.1%}")

        if len(away_qb_out) > 0:
            print(f"   Away QB out ({len(away_qb_out)} games):")
            print(f"     Avg ATS margin: {away_qb_out['home_ats_margin'].mean():.2f}")
            print(f"     Home cover rate: {away_qb_out['home_covered'].mean():.1%}")
    else:
        print("   Insufficient QB injury data for deep analysis")

    print("\n" + "=" * 60)

    return ValidationResult(
        n_games=len(df),
        n_games_with_qb_injury=n_qb_injury,
        n_games_with_key_position_injury=n_key_injury,
        current_qb_weight=INJURY_WEIGHTS.QB_OUT_SPREAD_IMPACT,
        current_key_position_weight=INJURY_WEIGHTS.KEY_POSITION_SPREAD_IMPACT,
        current_oline_weight=INJURY_WEIGHTS.OLINE_INJURY_SPREAD_IMPACT,
        estimated_qb_weight=qb_w,
        estimated_key_position_weight=key_w,
        estimated_oline_weight=oline_w,
        r_squared=r2,
        mae=opt_mae,
        recommendation=recommendation,
    )


if __name__ == "__main__":
    try:
        result = run_validation()
        print(f"\nValidation complete. See py/config/constants.py for weight definitions.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install scikit-learn pandas psycopg")
        sys.exit(1)
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
