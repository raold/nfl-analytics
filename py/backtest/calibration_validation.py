"""
Simple calibration validation script for 2025 holdout data.

Generates ensemble predictions and calculates 90% PI coverage without requiring prop odds.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection string
DB_CONNECTION = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def fetch_qb_games(conn, start_season: int, end_season: int) -> pd.DataFrame:
    """Fetch QB passing games for calibration validation."""
    query = """
    SELECT
        np.player_id,
        np.player_display_name as player_name,
        np.season,
        np.week,
        np.attempts,
        np.pass_yards as actual_yards,
        np.completions,
        np.pass_touchdowns as passing_tds,
        CASE WHEN np.player_id = g.home_qb_id THEN 1 ELSE 0 END as is_home,
        COALESCE(ABS(g.spread_close), 0.0) as spread_abs,
        COALESCE(g.total_close, 47.0) as total_line,
        COALESCE(CAST(g.temp AS NUMERIC), 70.0) as temp,
        COALESCE(CAST(g.wind AS NUMERIC), 5.0) as wind,
        CASE
            WHEN g.roof = 'outdoors' AND
                 (g.weather_condition IN ('rain', 'snow', 'wind') OR COALESCE(CAST(g.wind AS NUMERIC), 0) >= 15)
            THEN 1
            ELSE 0
        END as is_bad_weather
    FROM nextgen_passing np
    INNER JOIN games g ON
        (np.player_id = g.home_qb_id OR np.player_id = g.away_qb_id)
        AND g.season = np.season
        AND g.week = np.week
    WHERE np.season BETWEEN %s AND %s
        AND np.attempts >= 15
        AND np.pass_yards IS NOT NULL
    ORDER BY np.season, np.week, np.player_id
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (start_season, end_season))
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df['game_id'] = df.apply(lambda x: f"{x['season']}_{x['week']}_{x['player_id']}", axis=1)

    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add L3 lagged features (last 3 games) for each player."""
    df = df.sort_values(['player_id', 'season', 'week'])

    for player_id in df['player_id'].unique():
        mask = df['player_id'] == player_id
        player_df = df[mask].copy()

        # Calculate rolling averages (last 3 games)
        player_df['yards_L3'] = player_df['actual_yards'].rolling(3, min_periods=1).mean().shift(1)
        player_df['yards_per_attempt_L3'] = (
            player_df['actual_yards'] / player_df['attempts']
        ).rolling(3, min_periods=1).mean().shift(1)
        player_df['td_rate_L3'] = (
            player_df['passing_tds'] / player_df['attempts']
        ).rolling(3, min_periods=1).mean().shift(1)

        # Update main dataframe
        df.loc[mask, 'yards_L3'] = player_df['yards_L3'].values
        df.loc[mask, 'yards_per_attempt_L3'] = player_df['yards_per_attempt_L3'].values
        df.loc[mask, 'td_rate_L3'] = player_df['td_rate_L3'].values

    # Fill missing lagged features with league averages
    df['yards_L3'] = df['yards_L3'].fillna(240.0)
    df['yards_per_attempt_L3'] = df['yards_per_attempt_L3'].fillna(7.0)
    df['td_rate_L3'] = df['td_rate_L3'].fillna(0.04)

    return df


def calculate_calibration_metrics(df: pd.DataFrame) -> dict:
    """Calculate calibration metrics."""
    n = len(df)

    # 90% PI coverage
    within_90 = ((df['actual_yards'] >= df['lower_90']) &
                 (df['actual_yards'] <= df['upper_90'])).sum()
    coverage_90 = within_90 / n

    # Binomial test: H0: coverage = 0.90
    binom_pvalue = stats.binomtest(within_90, n, 0.90, alternative='two-sided').pvalue

    # Prediction error metrics
    errors = df['actual_yards'] - df['pred_mean']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    # Interval width (sharpness)
    interval_width = (df['upper_90'] - df['lower_90']).mean()

    return {
        'n_games': n,
        'coverage_90': coverage_90,
        'within_90': within_90,
        'binom_pvalue': binom_pvalue,
        'mae': mae,
        'rmse': rmse,
        'mean_interval_width': interval_width,
    }


def main():
    logger.info("=" * 80)
    logger.info("ENSEMBLE CALIBRATION VALIDATION (2025 HOLDOUT)")
    logger.info("=" * 80)

    # Initialize
    conn = psycopg2.connect(DB_CONNECTION)
    ensemble = EnhancedEnsemble4Way()

    # Fetch 2025 QB games
    logger.info("\n📥 Fetching 2025 QB games...")
    df = fetch_qb_games(conn, 2025, 2025)
    logger.info(f"✅ Fetched {len(df):,} QB games")
    logger.info("   Seasons: 2025-2025")
    logger.info(f"   Players: {df['player_id'].nunique()}")

    # Add lagged features
    logger.info("\n🔄 Adding lagged features...")
    df = add_lagged_features(df)

    # Generate ensemble predictions
    logger.info("\n🤖 Generating ensemble predictions...")
    predictions = []

    for idx, row in df.iterrows():
        try:
            pred = ensemble.predict(
                player_id=row['player_id'],
                season=row['season'],
                week=row['week']
            )

            # Calculate 90% PI (using normal approximation)
            lower_90 = pred['ensemble_mean'] - 1.645 * pred['ensemble_std']
            upper_90 = pred['ensemble_mean'] + 1.645 * pred['ensemble_std']

            predictions.append({
                'pred_mean': pred['ensemble_mean'],
                'pred_std': pred['ensemble_std'],
                'lower_90': lower_90,
                'upper_90': upper_90,
            })

            if (idx + 1) % 50 == 0:
                logger.info(f"   Progress: {idx + 1}/{len(df)} ({(idx+1)/len(df)*100:.0f}%)")

        except Exception as e:
            logger.error(f"❌ Prediction failed for game {row['game_id']}: {e}")
            predictions.append({
                'pred_mean': np.nan,
                'pred_std': np.nan,
                'lower_90': np.nan,
                'upper_90': np.nan,
            })

    # Combine predictions with actuals
    pred_df = pd.DataFrame(predictions)
    df = pd.concat([df, pred_df], axis=1)

    # Drop failed predictions
    df = df.dropna(subset=['pred_mean'])
    logger.info(f"\n✅ Generated {len(df):,} valid predictions")

    # Calculate calibration metrics
    logger.info("\n📊 Calculating calibration metrics...")
    metrics = calculate_calibration_metrics(df)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION RESULTS (2025 HOLDOUT)")
    logger.info("=" * 80)
    logger.info(f"Games: {metrics['n_games']}")
    logger.info(f"\n90% PI Coverage: {metrics['within_90']}/{metrics['n_games']} ({metrics['coverage_90']*100:.1f}%)")
    logger.info("  Target: 85-95% (ideal: 90%)")

    if 0.85 <= metrics['coverage_90'] <= 0.95:
        logger.info("  ✅ CALIBRATED (within target range)")
    elif metrics['coverage_90'] < 0.85:
        logger.info("  ❌ UNDERCALIBRATED (ensemble overconfident)")
    else:
        logger.info("  ⚠️  OVERCALIBRATED (ensemble underconfident)")

    logger.info(f"\nBinomial Test: p = {metrics['binom_pvalue']:.4f}")
    logger.info("  (H0: coverage = 90%, α = 0.05)")

    logger.info("\nPrediction Error:")
    logger.info(f"  MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} yards")

    logger.info("\nInterval Sharpness:")
    logger.info(f"  Mean Width: {metrics['mean_interval_width']:.1f} yards")
    logger.info("=" * 80)

    # Save results
    output_path = "analysis/results/ensemble_4way_calibration_2025_holdout.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Results saved to: {output_path}")

    # Close connection
    conn.close()

    return metrics


if __name__ == "__main__":
    main()
