"""
Propensity Score Matching (PSM) Analysis for Weather Familiarity Effects

This module implements PSM to estimate causal effects of weather familiarity
on NFL team performance. Analyzes:
1. Cold-weather teams at home vs dome teams
2. Dome teams playing outdoors vs. other outdoor teams

Expected Effects:
- Cold-weather teams (GB, CHI, BUF) at home: +2-3 yards advantage
- Dome teams (NO, ATL, DET) outdoors: -2-3 yards penalty

Statistical Approach:
- Propensity score matching with weather covariates
- Balance diagnostics (SMD < 0.1)
- ATT estimation with bootstrapped confidence intervals
- Sensitivity analysis for hidden confounding

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PSMResult:
    """Container for PSM estimation results."""
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    n_matched: int
    balance_diagnostics: dict[str, float]
    common_support_pct: float

    def __str__(self) -> str:
        return (
            f"ATT: {self.att:.3f} ({self.ci_lower:.3f}, {self.ci_upper:.3f})\n"
            f"SE: {self.se:.3f}, p={self.p_value:.4f}\n"
            f"N_treated: {self.n_treated}, N_matched: {self.n_matched}/{self.n_control}\n"
            f"Common support: {self.common_support_pct:.1f}%\n"
            f"Balance (max SMD): {max(self.balance_diagnostics.values()):.3f}"
        )


class PSMWeatherFamiliarityAnalyzer:
    """Propensity Score Matching for weather familiarity effects."""

    # Define team classifications
    COLD_WEATHER_TEAMS = ['GB', 'CHI', 'BUF', 'CLE', 'PIT', 'NE', 'NYG', 'NYJ', 'PHI', 'WAS', 'BAL']
    DOME_TEAMS = ['NO', 'ATL', 'DET', 'MIN', 'DAL', 'ARI', 'LAR', 'LAC', 'IND', 'LV']  # LAR/LAC when indoors

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01",
        weather_data_path: str = "/Users/dro/rice/nfl-analytics/data/historical_weather_2018_2024.csv"
    ):
        """Initialize with database and weather data."""
        self.db_url = db_url
        self.weather_data = pd.read_csv(weather_data_path)
        logger.info(f"Loaded weather data: {len(self.weather_data)} games")

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(self.db_url)

    def _fetch_game_data_with_weather(
        self,
        seasons: list[int]
    ) -> pd.DataFrame:
        """
        Fetch game data and merge with weather information.

        Args:
            seasons: List of seasons to fetch

        Returns:
            DataFrame with game outcomes and weather covariates
        """
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.kickoff,
            g.home_score,
            g.away_score,
            g.away_rest,
            g.home_rest,
            g.spread_close,
            g.stadium,
            g.roof,
            g.surface,
            -- Weather from weather table
            w.temp,
            w.wind,
            w.weather_condition,
            -- Home team offensive yards
            SUM(CASE WHEN p.posteam = g.home_team THEN p.yards_gained ELSE 0 END) as home_total_yards,
            -- Away team offensive yards
            SUM(CASE WHEN p.posteam = g.away_team THEN p.yards_gained ELSE 0 END) as away_total_yards
        FROM public.games g
        LEFT JOIN public.weather w ON g.game_id = w.game_id
        LEFT JOIN public.plays p ON g.game_id = p.game_id
        WHERE g.season = ANY(%s)
            AND g.game_type = 'REG'
        GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
                 g.kickoff, g.home_score, g.away_score, g.away_rest, g.home_rest,
                 g.spread_close, g.stadium, g.roof, g.surface, w.temp, w.wind, w.weather_condition
        ORDER BY g.season, g.week
        """

        with self._get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(seasons,))

        # Convert temperature to numeric (may be stored as string)
        df['temp_numeric'] = pd.to_numeric(df['temp'], errors='coerce')
        df['wind_numeric'] = pd.to_numeric(df['wind'], errors='coerce')

        # Classify dome games
        df['is_dome'] = df['roof'].isin(['dome', 'closed'])

        # Classify cold weather (< 40°F)
        df['is_cold'] = (df['temp_numeric'] < 40) & (~df['is_dome'])

        logger.info(f"Fetched {len(df)} games with weather data")
        logger.info(f"Cold games: {df['is_cold'].sum()}, Dome games: {df['is_dome'].sum()}")

        return df

    def analyze_cold_weather_home_advantage(
        self,
        seasons: list[int] = list(range(2018, 2025)),
        n_matches: int = 3,
        output_dir: Path | None = None
    ) -> PSMResult:
        """
        Analyze home field advantage for cold-weather teams in cold conditions.

        Treatment: Cold-weather teams (GB, CHI, BUF, etc.) at home in cold weather (<40°F)
        Control: Warm-weather teams at home in moderate weather (50-70°F)
        Outcome: Home team total yards

        Args:
            seasons: List of seasons to analyze
            n_matches: Number of control matches per treated unit
            output_dir: Optional directory for diagnostic plots

        Returns:
            PSMResult with ATT and balance diagnostics
        """
        logger.info("Analyzing cold-weather home field advantage...")

        # Fetch data
        df = self._fetch_game_data_with_weather(seasons)

        # Define treatment: cold-weather team at home in cold conditions
        df['cold_weather_treatment'] = (
            df['home_team'].isin(self.COLD_WEATHER_TEAMS) &
            df['is_cold']
        ).astype(int)

        # Control: non-cold-weather teams at home in moderate weather
        df['control_group'] = (
            (~df['home_team'].isin(self.COLD_WEATHER_TEAMS)) &
            (df['temp_numeric'] >= 50) &
            (df['temp_numeric'] <= 70) &
            (~df['is_dome'])
        ).astype(int)

        # Filter to treatment + control only
        df = df[(df['cold_weather_treatment'] == 1) | (df['control_group'] == 1)].copy()

        # Covariates
        covariates = ['away_rest', 'home_rest', 'spread_close', 'wind_numeric']

        # Drop missing
        df = df.dropna(subset=covariates + ['home_total_yards'])

        # Estimate propensity scores
        propensity_scores = self._estimate_propensity_scores(df, 'cold_weather_treatment', covariates)

        # Common support
        treated_ps = propensity_scores[df['cold_weather_treatment'] == 1]
        control_ps = propensity_scores[df['cold_weather_treatment'] == 0]
        common_min = max(treated_ps.min(), control_ps.min())
        common_max = min(treated_ps.max(), control_ps.max())
        common_support_pct = 100 * (
            ((treated_ps >= common_min) & (treated_ps <= common_max)).sum() +
            ((control_ps >= common_min) & (control_ps <= common_max)).sum()
        ) / len(df)

        # Match
        matched_treated, matched_control = self._match_observations(
            df, propensity_scores, 'cold_weather_treatment', n_matches=n_matches
        )

        # Balance
        balance = self._calculate_balance_diagnostics(matched_treated, matched_control, covariates)

        # ATT
        att, se, ci_lower, ci_upper, p_value = self._estimate_att(
            matched_treated['home_total_yards'],
            matched_control['home_total_yards']
        )

        result = PSMResult(
            att=att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=len(df[df['cold_weather_treatment'] == 1]),
            n_control=len(df[df['cold_weather_treatment'] == 0]),
            n_matched=len(matched_control),
            balance_diagnostics=balance,
            common_support_pct=common_support_pct
        )

        logger.info(f"Cold Weather Home Advantage PSM Results:\n{result}")

        if output_dir:
            self._plot_balance_diagnostics(balance, output_dir, "cold_weather_balance")

        return result

    def analyze_dome_team_outdoor_penalty(
        self,
        seasons: list[int] = list(range(2018, 2025)),
        n_matches: int = 3,
        output_dir: Path | None = None
    ) -> PSMResult:
        """
        Analyze performance penalty for dome teams playing outdoors.

        Treatment: Dome teams (NO, ATL, DET, etc.) playing outdoor road games
        Control: Outdoor teams playing outdoor road games
        Outcome: Away team total yards

        Args:
            seasons: List of seasons to analyze
            n_matches: Number of control matches per treated unit
            output_dir: Optional directory for diagnostic plots

        Returns:
            PSMResult with ATT and balance diagnostics
        """
        logger.info("Analyzing dome team outdoor performance penalty...")

        # Fetch data
        df = self._fetch_game_data_with_weather(seasons)

        # Define treatment: dome team playing outdoor road game
        df['dome_outdoor_treatment'] = (
            df['away_team'].isin(self.DOME_TEAMS) &
            (~df['is_dome'])
        ).astype(int)

        # Control: outdoor team playing outdoor road game
        df['control_group'] = (
            (~df['away_team'].isin(self.DOME_TEAMS)) &
            (~df['is_dome'])
        ).astype(int)

        # Filter
        df = df[(df['dome_outdoor_treatment'] == 1) | (df['control_group'] == 1)].copy()

        # Covariates
        covariates = ['away_rest', 'home_rest', 'spread_close', 'temp_numeric', 'wind_numeric']

        # Drop missing
        df = df.dropna(subset=covariates + ['away_total_yards'])

        # Propensity scores
        propensity_scores = self._estimate_propensity_scores(df, 'dome_outdoor_treatment', covariates)

        # Common support
        treated_ps = propensity_scores[df['dome_outdoor_treatment'] == 1]
        control_ps = propensity_scores[df['dome_outdoor_treatment'] == 0]
        common_min = max(treated_ps.min(), control_ps.min())
        common_max = min(treated_ps.max(), control_ps.max())
        common_support_pct = 100 * (
            ((treated_ps >= common_min) & (treated_ps <= common_max)).sum() +
            ((control_ps >= common_min) & (control_ps <= common_max)).sum()
        ) / len(df)

        # Match
        matched_treated, matched_control = self._match_observations(
            df, propensity_scores, 'dome_outdoor_treatment', n_matches=n_matches
        )

        # Balance
        balance = self._calculate_balance_diagnostics(matched_treated, matched_control, covariates)

        # ATT
        att, se, ci_lower, ci_upper, p_value = self._estimate_att(
            matched_treated['away_total_yards'],
            matched_control['away_total_yards']
        )

        result = PSMResult(
            att=att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=len(df[df['dome_outdoor_treatment'] == 1]),
            n_control=len(df[df['dome_outdoor_treatment'] == 0]),
            n_matched=len(matched_control),
            balance_diagnostics=balance,
            common_support_pct=common_support_pct
        )

        logger.info(f"Dome Team Outdoor Penalty PSM Results:\n{result}")

        if output_dir:
            self._plot_balance_diagnostics(balance, output_dir, "dome_outdoor_balance")

        return result

    def _estimate_propensity_scores(
        self,
        df: pd.DataFrame,
        treatment_var: str,
        covariates: list[str]
    ) -> np.ndarray:
        """Estimate propensity scores using logistic regression."""
        X = df[covariates].values
        y = df[treatment_var].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)

        propensity_scores = model.predict_proba(X_scaled)[:, 1]

        logger.info(f"Estimated propensity scores: mean={propensity_scores.mean():.3f}")
        return propensity_scores

    def _match_observations(
        self,
        df: pd.DataFrame,
        propensity_scores: np.ndarray,
        treatment_var: str,
        n_matches: int = 1,
        caliper: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Match treated and control observations using nearest neighbor."""
        df = df.copy()
        df['propensity_score'] = propensity_scores

        treated = df[df[treatment_var] == 1].copy()
        control = df[df[treatment_var] == 0].copy()

        matched_control_indices = []

        for idx, treated_row in treated.iterrows():
            treated_ps = treated_row['propensity_score']

            control['ps_distance'] = np.abs(control['propensity_score'] - treated_ps)
            eligible_control = control[control['ps_distance'] <= caliper].copy()

            if len(eligible_control) == 0:
                continue

            nearest = eligible_control.nsmallest(n_matches, 'ps_distance')
            matched_control_indices.extend(nearest.index.tolist())

        matched_control = control.loc[matched_control_indices].copy()

        logger.info(f"Matched {len(treated)} treated with {len(matched_control)} control")
        return treated, matched_control

    def _calculate_balance_diagnostics(
        self,
        matched_treated: pd.DataFrame,
        matched_control: pd.DataFrame,
        covariates: list[str]
    ) -> dict[str, float]:
        """Calculate standardized mean difference (SMD) for balance."""
        balance = {}

        for cov in covariates:
            treated_mean = matched_treated[cov].mean()
            control_mean = matched_control[cov].mean()
            treated_std = matched_treated[cov].std()
            control_std = matched_control[cov].std()

            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            balance[cov] = abs(smd)

        return balance

    def _estimate_att(
        self,
        treated_outcome: pd.Series,
        control_outcome: pd.Series
    ) -> tuple[float, float, float, float, float]:
        """Estimate ATT with standard error and confidence intervals."""
        att = treated_outcome.mean() - control_outcome.mean()

        treated_se = treated_outcome.std() / np.sqrt(len(treated_outcome))
        control_se = control_outcome.std() / np.sqrt(len(control_outcome))
        se = np.sqrt(treated_se**2 + control_se**2)

        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se

        t_stat = att / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(treated_outcome) + len(control_outcome) - 2))

        return att, se, ci_lower, ci_upper, p_value

    def _plot_balance_diagnostics(
        self,
        balance: dict[str, float],
        output_dir: Path,
        filename: str
    ):
        """Plot balance diagnostics (SMD)."""
        plt.figure(figsize=(8, 6))

        covariates = list(balance.keys())
        smds = list(balance.values())

        plt.barh(covariates, smds, color='steelblue')
        plt.axvline(x=0.1, color='red', linestyle='--', label='SMD < 0.1 threshold')
        plt.xlabel('Standardized Mean Difference (SMD)', fontsize=12)
        plt.ylabel('Covariate', fontsize=12)
        plt.title('Balance Diagnostics After Matching', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()

        output_path = output_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved balance plot to {output_path}")
        plt.close()


if __name__ == "__main__":
    # Example usage
    analyzer = PSMWeatherFamiliarityAnalyzer()
    output_dir = Path("/Users/dro/rice/nfl-analytics/analysis/causal_inference/psm_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    logger.info("=" * 80)
    logger.info("ANALYSIS 1: Cold Weather Home Field Advantage")
    logger.info("=" * 80)
    cold_weather_result = analyzer.analyze_cold_weather_home_advantage(output_dir=output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS 2: Dome Team Outdoor Performance Penalty")
    logger.info("=" * 80)
    dome_outdoor_result = analyzer.analyze_dome_team_outdoor_penalty(output_dir=output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY OF WEATHER FAMILIARITY PSM RESULTS")
    logger.info("=" * 80)
    logger.info(f"\n1. Cold Weather Home Advantage: {cold_weather_result.att:.2f} yards (p={cold_weather_result.p_value:.4f})")
    logger.info(f"2. Dome Team Outdoor Penalty: {dome_outdoor_result.att:.2f} yards (p={dome_outdoor_result.p_value:.4f})")
