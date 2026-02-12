#!/usr/bin/env python3
"""
Bayesian Optimization for BNN Hyperparameters

Uses Optuna to systematically search the hyperparameter space for improved
calibration. The objective is to maximize 90% CI coverage while maintaining
reasonable prediction accuracy.

Search space:
- Prior std: [0.3, 1.5]
- Hidden units: [8, 32]
- Player effect sigma: [0.1, 0.5]
- Noise sigma: [0.2, 0.5]
- Feature groups: combinations of Vegas, Environment, Opponent

Target: Find configuration achieving >50% coverage (2x current best)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psycopg2
import pymc as pm
from optuna.storages import RDBStorage
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append("/Users/dro/rice/nfl-analytics")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
STUDY_DIR = Path("/Users/dro/rice/nfl-analytics/experiments/optimization")
STUDY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STUDY_DIR / "optuna_study.db"

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def load_rushing_data_with_features(
    start_season: int,
    end_season: int,
    include_vegas: bool = False,
    include_env: bool = False,
    include_opp: bool = False,
) -> pd.DataFrame:
    """Load rushing data with optional feature groups"""
    conn = psycopg2.connect(**DB_CONFIG)

    # Build query based on requested features
    query_parts = [
        f"""
        WITH rushing_data AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.current_team as team,
                pgs.player_position as position,
                pgs.stat_yards,
                pgs.stat_attempts as carries,
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id
                    ORDER BY pgs.season, pgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_rushing_l3,
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id, pgs.season
                    ORDER BY pgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_avg
            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN {start_season} AND {end_season}
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5
              AND pgs.stat_yards IS NOT NULL
        )
    """
    ]

    # Add opponent defense CTE if needed
    if include_opp:
        query_parts.append(
            """,
        opponent_defense AS (
            SELECT
                pgs.season,
                pgs.week,
                pgs.current_team as defense_team,
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.current_team, pgs.season
                    ORDER BY pgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as rush_yds_allowed_avg,
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.current_team
                    ORDER BY pgs.season, pgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as rush_yds_allowed_l3
            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN {start_season} AND {end_season}
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5
        ),
        opponent_ranks AS (
            SELECT
                season, week, defense_team,
                rush_yds_allowed_avg, rush_yds_allowed_l3,
                ROW_NUMBER() OVER (
                    PARTITION BY season, week
                    ORDER BY rush_yds_allowed_avg ASC NULLS LAST
                ) as rush_rank
            FROM (SELECT DISTINCT season, week, defense_team, rush_yds_allowed_avg, rush_yds_allowed_l3 FROM opponent_defense) t
        )
        """
        )

    # Main SELECT
    select_parts = ["rd.*"]

    if include_vegas:
        select_parts.extend(
            [
                "CASE WHEN rd.team = g.home_team THEN g.spread_close ELSE -g.spread_close END as spread_close",
                "g.total_close",
            ]
        )

    if include_env:
        select_parts.extend(
            [
                "COALESCE(g.roof = 'dome', false) as is_dome",
                "COALESCE(g.surface = 'fieldturf' OR g.surface = 'astroturf', false) as is_turf",
                "COALESCE(g.temp, 70.0) as temp",
                "COALESCE(g.wind, 0.0) as wind",
            ]
        )

    if include_opp:
        select_parts.extend(
            [
                "COALESCE(opp.rush_yds_allowed_avg, 100.0) as opp_rush_yds_allowed_avg",
                "COALESCE(opp.rush_rank, 16.0) as opp_rush_rank",
                "COALESCE(opp.rush_yds_allowed_l3, 100.0) as opp_rush_yds_l3",
            ]
        )

    query_parts.append(
        f"""
        SELECT {', '.join(select_parts)}
        FROM rushing_data rd
    """
    )

    if include_vegas or include_env:
        query_parts.append(
            """
        LEFT JOIN games g
            ON rd.season = g.season
            AND rd.week = g.week
            AND (rd.team = g.home_team OR rd.team = g.away_team)
        """
        )

    if include_opp:
        query_parts.append(
            """
        LEFT JOIN opponent_ranks opp
            ON rd.season = opp.season
            AND rd.week = opp.week
            AND (
                (rd.team = g.home_team AND g.away_team = opp.defense_team)
                OR (rd.team = g.away_team AND g.home_team = opp.defense_team)
            )
        """
        )

    # Add WHERE clause for Vegas features if needed
    if include_vegas:
        query_parts.append(
            """
        WHERE rd.stat_yards IS NOT NULL
          AND g.spread_close IS NOT NULL
          AND g.total_close IS NOT NULL
        """
        )

    query_parts.append("ORDER BY rd.season, rd.week, rd.stat_yards DESC")

    query = "\n".join(query_parts).format(start_season=start_season, end_season=end_season)

    df = pd.read_sql(query, conn)
    conn.close()

    # Handle missing values
    df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
        df.groupby("position")["stat_yards"].transform("median")
    )
    df["season_avg"] = df["season_avg"].fillna(
        df.groupby("position")["stat_yards"].transform("median")
    )

    logger.info(f"Loaded {len(df)} rushing performances")
    return df


def train_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    player_idx_train: np.ndarray,
    X_test: np.ndarray,
    player_idx_test: np.ndarray,
    hidden_dim: int = 16,
    prior_std: float = 0.5,
    player_sigma: float = 0.2,
    noise_sigma: float = 0.3,
    n_samples: int = 1000,
    n_chains: int = 2,
) -> dict[str, np.ndarray]:
    """
    Train BNN and return test predictions.

    Returns:
        Dict with 'mean', 'std', 'q05', 'q50', 'q95'
    """
    n_features = X_train.shape[1]
    n_players = len(np.unique(player_idx_train))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Log transform target
    y_train_log = np.log1p(y_train)

    # Build model
    with pm.Model():
        # Input
        X_input = pm.Data("X_input", X_train_scaled)
        player_input = pm.Data("player_input", player_idx_train)

        # Hierarchical player effects
        player_effect_mu = pm.Normal("player_effect_mu", mu=0, sigma=0.1)
        player_effect_sigma = pm.HalfNormal("player_effect_sigma", sigma=player_sigma)
        player_effects = pm.Normal(
            "player_effects", mu=player_effect_mu, sigma=player_effect_sigma, shape=n_players
        )

        # Single hidden layer
        W1 = pm.Normal("W1", mu=0, sigma=prior_std, shape=(n_features, hidden_dim))
        b1 = pm.Normal("b1", mu=0, sigma=prior_std, shape=hidden_dim)
        hidden = pm.math.dot(X_input, W1) + b1
        hidden = pm.math.maximum(0, hidden)  # ReLU

        # Output layer
        W_out = pm.Normal("W_out", mu=0, sigma=prior_std, shape=(hidden_dim, 1))
        b_out = pm.Normal("b_out", mu=0, sigma=prior_std)

        # Prediction
        mu_network = pm.math.dot(hidden, W_out).flatten() + b_out
        mu = mu_network + player_effects[player_input]

        # Noise
        sigma = pm.HalfNormal("sigma", sigma=noise_sigma)

        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_log)

        pm.Deterministic("prediction", mu)

        # Sample
        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            target_accept=0.95,
            max_treedepth=12,
            progressbar=False,
            return_inferencedata=True,
            cores=min(n_chains, 2),
        )

        # Predict on test set
        pm.set_data({"X_input": X_test_scaled, "player_input": player_idx_test})

        posterior_predictive = pm.sample_posterior_predictive(
            trace, var_names=["prediction"], progressbar=False
        )

    # Extract predictions
    pred_samples = posterior_predictive.posterior_predictive["prediction"].values
    pred_samples_exp = np.expm1(pred_samples)

    predictions = {
        "mean": pred_samples_exp.mean(axis=(0, 1)),
        "std": pred_samples_exp.std(axis=(0, 1)),
        "q05": np.quantile(pred_samples_exp, 0.05, axis=(0, 1)),
        "q50": np.quantile(pred_samples_exp, 0.50, axis=(0, 1)),
        "q95": np.quantile(pred_samples_exp, 0.95, axis=(0, 1)),
    }

    return predictions


class BNNObjective:
    """Objective function for Optuna optimization"""

    def __init__(self):
        """Load data once for all trials"""
        logger.info("Loading base training and test data...")
        # We'll reload with different features per trial
        self.base_features = ["carries", "avg_rushing_l3", "season_avg", "week"]

    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for single trial"""
        # Suggest hyperparameters
        prior_std = trial.suggest_float("prior_std", 0.3, 1.5, log=True)
        hidden_units = trial.suggest_int("hidden_units", 8, 32, step=8)
        player_sigma = trial.suggest_float("player_sigma", 0.1, 0.5)
        noise_sigma = trial.suggest_float("noise_sigma", 0.2, 0.5)

        # Feature selection
        use_vegas = trial.suggest_categorical("use_vegas", [True, False])
        use_env = trial.suggest_categorical("use_environment", [True, False])
        use_opp = trial.suggest_categorical("use_opponent", [True, False])

        logger.info(f"\nTrial {trial.number}: Testing configuration:")
        logger.info(f"  Prior std: {prior_std:.3f}")
        logger.info(f"  Hidden units: {hidden_units}")
        logger.info(f"  Player sigma: {player_sigma:.3f}")
        logger.info(f"  Noise sigma: {noise_sigma:.3f}")
        logger.info(f"  Features: Vegas={use_vegas}, Env={use_env}, Opp={use_opp}")

        try:
            # Load data with requested features
            df_train = load_rushing_data_with_features(
                2020, 2023, include_vegas=use_vegas, include_env=use_env, include_opp=use_opp
            )
            df_test = load_rushing_data_with_features(
                2024, 2024, include_vegas=use_vegas, include_env=use_env, include_opp=use_opp
            )

            if len(df_train) == 0 or len(df_test) == 0:
                logger.error("No data loaded!")
                return -1.0

            # Build feature list
            features = self.base_features.copy()
            if use_vegas:
                features.extend(["spread_close", "total_close"])
            if use_env:
                features.extend(["is_dome", "is_turf", "temp", "wind"])
            if use_opp:
                features.extend(["opp_rush_yds_allowed_avg", "opp_rush_rank", "opp_rush_yds_l3"])

            # Prepare data
            X_train = df_train[features].fillna(0).values
            y_train = df_train["stat_yards"].values

            X_test = df_test[features].fillna(0).values
            y_test = df_test["stat_yards"].values

            # Player encoding
            unique_players = df_train["player_id"].unique()
            player_encoder = {pid: idx for idx, pid in enumerate(unique_players)}

            player_idx_train = df_train["player_id"].map(player_encoder).values
            player_idx_test = (
                df_test["player_id"]
                .map(player_encoder)
                .fillna(len(player_encoder) - 1)
                .astype(int)
                .values
            )

            # Train BNN
            predictions = train_bnn(
                X_train,
                y_train,
                player_idx_train,
                X_test,
                player_idx_test,
                hidden_dim=hidden_units,
                prior_std=prior_std,
                player_sigma=player_sigma,
                noise_sigma=noise_sigma,
                n_samples=1000,
                n_chains=2,
            )

            # Calculate metrics
            coverage_90 = (
                np.mean((y_test >= predictions["q05"]) & (y_test <= predictions["q95"])) * 100
            )

            mae = np.mean(np.abs(y_test - predictions["mean"]))
            interval_width = np.mean(predictions["q95"] - predictions["q05"])

            logger.info(
                f"  Results: Coverage={coverage_90:.1f}%, MAE={mae:.1f}, Width={interval_width:.1f}"
            )

            # Composite score
            coverage_score = min(coverage_90 / 90.0, 1.5)
            mae_score = max(0, 1.0 - (mae - 18.0) / 10.0)
            width_score = max(0, 1.0 - (interval_width - 17) / 80.0)

            score = 0.7 * coverage_score + 0.2 * mae_score + 0.1 * width_score

            # Store metrics
            trial.set_user_attr("coverage_90", coverage_90)
            trial.set_user_attr("mae", mae)
            trial.set_user_attr("interval_width", interval_width)
            trial.set_user_attr("num_features", len(features))

            logger.info(f"  Composite score: {score:.3f}")

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return -1.0


def run_optimization(
    n_trials: int = 50, timeout: int = None, study_name: str = "bnn_hyperparameter_search"
) -> optuna.Study:
    """Run Bayesian optimization study"""
    logger.info("=" * 80)
    logger.info("STARTING BAYESIAN OPTIMIZATION FOR BNN HYPERPARAMETERS")
    logger.info("=" * 80)
    logger.info(f"Study: {study_name}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Database: {DB_PATH}")
    logger.info("")

    # Create storage
    storage = RDBStorage(
        url=f"sqlite:///{DB_PATH}", engine_kwargs={"connect_args": {"timeout": 30}}
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Create objective
    objective = BNNObjective()

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)

    # Best trial
    if len(study.trials) > 0 and study.best_trial is not None:
        best_trial = study.best_trial
        logger.info(f"\nBest trial: #{best_trial.number}")
        logger.info(f"  Score: {best_trial.value:.3f}")
        logger.info(f"  Coverage: {best_trial.user_attrs['coverage_90']:.1f}%")
        logger.info(f"  MAE: {best_trial.user_attrs['mae']:.1f} yards")
        logger.info(f"  Interval Width: {best_trial.user_attrs['interval_width']:.1f} yards")
        logger.info("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # Save results
        save_results(study)

    return study


def save_results(study: optuna.Study) -> None:
    """Save optimization results to files"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best parameters as JSON
    best_params_path = STUDY_DIR / f"best_params_{timestamp}.json"
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "trial_number": study.best_trial.number,
                "score": study.best_trial.value,
                "params": study.best_trial.params,
                "metrics": {
                    "coverage_90": study.best_trial.user_attrs["coverage_90"],
                    "mae": study.best_trial.user_attrs["mae"],
                    "interval_width": study.best_trial.user_attrs["interval_width"],
                    "num_features": study.best_trial.user_attrs["num_features"],
                },
            },
            f,
            indent=2,
        )
    logger.info(f"\n✓ Best parameters saved to: {best_params_path}")

    # Save all trials as CSV
    df_trials = study.trials_dataframe()
    trials_path = STUDY_DIR / f"all_trials_{timestamp}.csv"
    df_trials.to_csv(trials_path, index=False)
    logger.info(f"✓ All trials saved to: {trials_path}")

    # Save summary
    summary_path = STUDY_DIR / f"optimization_summary_{timestamp}.md"
    with open(summary_path, "w") as f:
        f.write("# BNN Hyperparameter Optimization Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Study**: {study.study_name}\n\n")
        f.write(f"**Trials Completed**: {len(study.trials)}\n\n")

        f.write("## Best Configuration\n\n")
        f.write(f"- **Trial**: #{study.best_trial.number}\n")
        f.write(f"- **Score**: {study.best_trial.value:.3f}\n")
        f.write(f"- **90% Coverage**: {study.best_trial.user_attrs['coverage_90']:.1f}%\n")
        f.write(f"- **MAE**: {study.best_trial.user_attrs['mae']:.1f} yards\n")
        f.write(f"- **CI Width**: {study.best_trial.user_attrs['interval_width']:.1f} yards\n\n")

        f.write("### Hyperparameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(study.best_trial.params, indent=2))
        f.write("\n```\n\n")

        f.write("## Top 5 Trials\n\n")
        sorted_trials = sorted(
            study.trials, key=lambda t: t.value if t.value else -999, reverse=True
        )[:5]
        f.write("| Rank | Trial | Score | Coverage | MAE | Width | Features |\n")
        f.write("|------|-------|-------|----------|-----|-------|----------|\n")
        for i, trial in enumerate(sorted_trials, 1):
            if trial.value is not None:
                f.write(
                    f"| {i} | #{trial.number} | {trial.value:.3f} | "
                    f"{trial.user_attrs.get('coverage_90', 0):.1f}% | "
                    f"{trial.user_attrs.get('mae', 0):.1f} | "
                    f"{trial.user_attrs.get('interval_width', 0):.1f} | "
                    f"{trial.user_attrs.get('num_features', 0)} |\n"
                )

    logger.info(f"✓ Summary saved to: {summary_path}")


def main():
    """Main optimization workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="BNN Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument(
        "--study-name", type=str, default="bnn_hyperparameter_search", help="Study name"
    )

    args = parser.parse_args()

    # Run optimization
    run_optimization(n_trials=args.trials, timeout=args.timeout, study_name=args.study_name)

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Review optimization_summary_*.md for best configuration")
    logger.info("2. Retrain model with best hyperparameters")
    logger.info("3. Validate on holdout test set")
    logger.info("4. Compare to baseline (31.3% coverage)")
    logger.info("")


if __name__ == "__main__":
    main()
