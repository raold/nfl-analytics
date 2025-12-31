"""
Kalman Filter for Team EPA Trend Detection.

Implements a Local Level (Random Walk + Noise) state-space model
for tracking team EPA trends and detecting regime changes.

Based on Gemini consultation (Dec 28, 2025):
- Local Level Model: μ_t = μ_{t-1} + w_t (process), y_t = μ_t + v_t (observation)
- CUSUM on standardized innovations for regime detection
- Univariate filter per team for simplicity

Usage:
    filter = KalmanEPAFilter()
    features = filter.get_team_features(team='KC', season=2025, week=16)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import psycopg2

logger = logging.getLogger(__name__)

# Import centralized DB config
try:
    from py.utils.db_config import get_db_config
except ImportError:
    # Fallback for standalone usage
    def get_db_config():
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "dbname": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
        }


@dataclass
class KalmanState:
    """State of a Kalman filter for one team."""
    mu: float = 0.0  # Filtered state estimate
    P: float = 0.1   # State error covariance
    cusum_up: float = 0.0   # CUSUM for upward shifts
    cusum_down: float = 0.0  # CUSUM for downward shifts
    innovations: list = field(default_factory=list)
    n_updates: int = 0


class KalmanEPAFilter:
    """
    Kalman Filter for team EPA regime detection.

    Parameters calibrated for NFL game-level EPA averages:
    - σ_v² (observation noise): ~0.025 (game-to-game EPA variance)
    - σ_w² (process noise): ~0.001 (slow underlying changes)

    CUSUM parameters:
    - k=0.5: Reference value for 1-sigma shift detection
    - h=4.0: Control limit (balance false alarms vs detection speed)
    """

    # Noise parameters (empirically tuned for NFL EPA)
    SIGMA_V_SQ = 0.025  # Observation noise variance
    SIGMA_W_SQ = 0.001  # Process noise variance (4% of σ_v²)

    # CUSUM parameters
    CUSUM_K = 0.5  # Reference value
    CUSUM_H = 4.0  # Control limit

    def __init__(self, sigma_v_sq: float = None, sigma_w_sq: float = None, db_config: dict = None):
        """
        Initialize filter with optional custom noise parameters.

        Args:
            sigma_v_sq: Observation noise variance (default: 0.025)
            sigma_w_sq: Process noise variance (default: 0.001)
        """
        if sigma_v_sq is not None:
            self.SIGMA_V_SQ = sigma_v_sq
        if sigma_w_sq is not None:
            self.SIGMA_W_SQ = sigma_w_sq

        self._team_states: dict[str, KalmanState] = {}
        self._cache: dict[tuple, dict] = {}

    def _get_team_epa_history(
        self, team: str, season: int, week: int
    ) -> list[tuple[int, int, float]]:
        """
        Get team's EPA history up to (but not including) the specified week.

        Returns:
            List of (season, week, epa) tuples
        """
        query = """
        WITH team_game_epa AS (
            SELECT
                g.season,
                g.week,
                g.game_id,
                CASE
                    WHEN g.home_team = %(team)s THEN 'home'
                    ELSE 'away'
                END as side,
                AVG(p.epa) as avg_epa
            FROM games g
            JOIN plays p ON g.game_id = p.game_id
            WHERE (g.home_team = %(team)s OR g.away_team = %(team)s)
              AND p.epa IS NOT NULL
              AND ((g.season < %(season)s) OR (g.season = %(season)s AND g.week < %(week)s))
              AND g.season >= %(season)s - 2  -- Last 2 seasons max
            GROUP BY g.season, g.week, g.game_id,
                     CASE WHEN g.home_team = %(team)s THEN 'home' ELSE 'away' END
        )
        SELECT season, week, avg_epa
        FROM team_game_epa
        ORDER BY season, week
        """

        try:
            db_config = get_db_config()
            conn = psycopg2.connect(**db_config)
            with conn.cursor() as cur:
                cur.execute(query, {"team": team, "season": season, "week": week})
                result = [(row[0], row[1], float(row[2])) for row in cur.fetchall()]
            conn.close()
            return result
        except Exception as e:
            logger.warning(f"Failed to get EPA history for {team}: {e}")
            return []

    def _run_filter(self, observations: list[float]) -> KalmanState:
        """
        Run Kalman filter on a sequence of observations.

        Args:
            observations: List of EPA values (game averages)

        Returns:
            Final KalmanState after processing all observations
        """
        if not observations:
            return KalmanState()

        # Initialize with first observation
        state = KalmanState(
            mu=observations[0],
            P=self.SIGMA_V_SQ * 10,  # High initial uncertainty
        )

        for y in observations[1:]:
            # Prediction step
            mu_pred = state.mu
            P_pred = state.P + self.SIGMA_W_SQ

            # Update step
            innovation = y - mu_pred
            S = P_pred + self.SIGMA_V_SQ  # Innovation variance
            K = P_pred / S  # Kalman gain

            state.mu = mu_pred + K * innovation
            state.P = (1 - K) * P_pred

            # Standardized innovation for CUSUM
            z = innovation / np.sqrt(S)
            state.innovations.append(z)

            # Update CUSUM
            state.cusum_up = max(0, state.cusum_up + z - self.CUSUM_K)
            state.cusum_down = min(0, state.cusum_down + z + self.CUSUM_K)

            state.n_updates += 1

        return state

    def get_team_features(
        self, team: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get Kalman-filtered features for a team.

        Returns dict with:
        - kalman_epa: Current filtered EPA estimate
        - kalman_epa_uncertainty: State uncertainty (sqrt of P)
        - kalman_innovation_last: Most recent standardized innovation
        - kalman_cusum_up: CUSUM for upward regime detection
        - kalman_cusum_down: CUSUM for downward regime detection
        - kalman_regime_change: 1 if regime change detected, 0 otherwise
        - kalman_regime_direction: +1 (up), -1 (down), 0 (none)
        - kalman_n_observations: Number of games used

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dict of Kalman-based features
        """
        cache_key = (team, season, week)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get historical EPA
        history = self._get_team_epa_history(team, season, week)

        if len(history) < 3:
            # Not enough data for meaningful filter
            features = {
                "kalman_epa": 0.0,
                "kalman_epa_uncertainty": 0.5,
                "kalman_innovation_last": 0.0,
                "kalman_cusum_up": 0.0,
                "kalman_cusum_down": 0.0,
                "kalman_regime_change": 0,
                "kalman_regime_direction": 0,
                "kalman_n_observations": len(history),
            }
            self._cache[cache_key] = features
            return features

        # Extract EPA values and run filter
        observations = [epa for _, _, epa in history]
        state = self._run_filter(observations)

        # Detect regime change
        regime_up = state.cusum_up > self.CUSUM_H
        regime_down = state.cusum_down < -self.CUSUM_H
        regime_change = 1 if (regime_up or regime_down) else 0
        regime_direction = 1 if regime_up else (-1 if regime_down else 0)

        features = {
            "kalman_epa": round(state.mu, 4),
            "kalman_epa_uncertainty": round(np.sqrt(state.P), 4),
            "kalman_innovation_last": round(state.innovations[-1] if state.innovations else 0.0, 4),
            "kalman_cusum_up": round(state.cusum_up, 4),
            "kalman_cusum_down": round(state.cusum_down, 4),
            "kalman_regime_change": regime_change,
            "kalman_regime_direction": regime_direction,
            "kalman_n_observations": state.n_updates + 1,
        }

        self._cache[cache_key] = features
        return features

    def get_matchup_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """
        Get Kalman-based matchup features.

        Returns features for both teams plus differentials.
        """
        home_features = self.get_team_features(home_team, season, week)
        away_features = self.get_team_features(away_team, season, week)

        # Rename with home/away prefix
        features = {}
        for key, value in home_features.items():
            features[f"home_{key}"] = value
        for key, value in away_features.items():
            features[f"away_{key}"] = value

        # Add differentials
        features["kalman_epa_diff"] = round(
            home_features["kalman_epa"] - away_features["kalman_epa"], 4
        )
        features["kalman_momentum_diff"] = round(
            home_features["kalman_cusum_up"] + home_features["kalman_cusum_down"] -
            away_features["kalman_cusum_up"] - away_features["kalman_cusum_down"], 4
        )

        return features


def test_filter():
    """Test the Kalman filter on 2025 data."""
    print("\n" + "=" * 60)
    print("KALMAN EPA FILTER TEST")
    print("=" * 60)

    filter = KalmanEPAFilter()

    # Test a few teams
    test_cases = [
        ("KC", 2025, 16),
        ("DET", 2025, 16),
        ("PHI", 2025, 16),
        ("NE", 2025, 16),
    ]

    for team, season, week in test_cases:
        features = filter.get_team_features(team, season, week)
        print(f"\n{team} (Week {week}):")
        print(f"  Kalman EPA: {features['kalman_epa']:.4f} ± {features['kalman_epa_uncertainty']:.4f}")
        print(f"  CUSUM: up={features['kalman_cusum_up']:.2f}, down={features['kalman_cusum_down']:.2f}")
        print(f"  Regime: {features['kalman_regime_direction']} (change={features['kalman_regime_change']})")
        print(f"  Observations: {features['kalman_n_observations']}")

    # Test matchup
    print("\n" + "=" * 60)
    print("MATCHUP TEST: KC vs DET")
    print("=" * 60)
    matchup = filter.get_matchup_features("KC", "DET", 2025, 16)
    print(f"EPA diff (home - away): {matchup['kalman_epa_diff']:.4f}")
    print(f"Momentum diff: {matchup['kalman_momentum_diff']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_filter()
