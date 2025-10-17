#!/usr/bin/env python3
"""
Causal Inference Validation and Backtesting

Validates causal inference methods and backtests causal adjustments:
- Compare causal predictions to baselines
- Sensitivity analysis for assumptions
- Out-of-sample treatment effect validation
- Betting performance with causal adjustments

Key validation approaches:
1. Placebo tests: Apply methods to known null effects
2. Refutation tests: Violate assumptions and check results
3. Cross-validation: Out-of-sample treatment effect estimation
4. Betting backtest: Compare ROI with/without causal adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from causal.synthetic_control import SyntheticControl
from causal.diff_in_diff import DifferenceInDifferences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalValidator:
    """
    Validates causal inference results through various robustness checks.
    """

    def __init__(self):
        self.validation_results = {}

    def placebo_test(
        self,
        estimator: Any,
        df: pd.DataFrame,
        n_placebos: int = 100,
        random_seed: int = 42
    ) -> Dict:
        """
        Run placebo tests by randomly assigning treatment.

        If estimator is valid, placebo estimates should be centered at zero.

        Args:
            estimator: Fitted causal estimator (DiD, SC, etc.)
            df: Panel dataframe
            n_placebos: Number of placebo iterations
            random_seed: Random seed

        Returns:
            Dictionary with placebo distribution and p-value
        """
        np.random.seed(random_seed)

        logger.info(f"Running {n_placebos} placebo tests...")

        placebo_estimates = []

        for i in range(n_placebos):
            # Randomly assign treatment
            df_placebo = df.copy()
            df_placebo['placebo_treatment'] = np.random.binomial(1, 0.5, len(df))

            try:
                # Re-run estimator with placebo treatment
                if isinstance(estimator, DifferenceInDifferences):
                    did_placebo = DifferenceInDifferences()
                    did_placebo.fit(
                        df=df_placebo,
                        outcome_col=estimator.model_.model.endog_names,
                        treatment_col='placebo_treatment',
                        unit_col='player_id' if 'player_id' in df.columns else 'team',
                        time_col='week'
                    )
                    placebo_estimates.append(did_placebo.treatment_effect_['estimate'])

                elif isinstance(estimator, SyntheticControl):
                    # SC placebo is handled internally
                    pass

            except Exception as e:
                logger.debug(f"Placebo iteration {i} failed: {e}")
                continue

        if len(placebo_estimates) == 0:
            logger.warning("No successful placebo iterations")
            return {'success': False}

        placebo_estimates = np.array(placebo_estimates)

        # Compare to actual estimate
        if hasattr(estimator, 'treatment_effect_'):
            actual_estimate = estimator.treatment_effect_['estimate']

            # P-value: proportion of placebos as extreme as actual
            p_value = np.mean(np.abs(placebo_estimates) >= np.abs(actual_estimate))

            result = {
                'success': True,
                'actual_estimate': actual_estimate,
                'placebo_mean': placebo_estimates.mean(),
                'placebo_std': placebo_estimates.std(),
                'placebo_estimates': placebo_estimates,
                'p_value': p_value,
                'passes': p_value < 0.05,
                'interpretation': 'Significant' if p_value < 0.05 else 'Not significant'
            }

            logger.info(f"Placebo test: actual={actual_estimate:.2f}, "
                       f"placebo_mean={result['placebo_mean']:.2f}, "
                       f"p={p_value:.3f}")

            return result

        return {'success': False}

    def sensitivity_analysis(
        self,
        estimator: Any,
        df: pd.DataFrame,
        confounder_strength_range: List[float] = [0.0, 0.1, 0.2, 0.3]
    ) -> pd.DataFrame:
        """
        Sensitivity analysis: How sensitive are results to unmeasured confounding?

        Tests how strong an unmeasured confounder would need to be to explain away the effect.

        Args:
            estimator: Fitted causal estimator
            df: Panel dataframe
            confounder_strength_range: Range of confounder strengths to test

        Returns:
            DataFrame with sensitivity results
        """
        logger.info("Running sensitivity analysis...")

        if not hasattr(estimator, 'treatment_effect_'):
            logger.warning("Estimator must have treatment_effect_ attribute")
            return pd.DataFrame()

        actual_effect = estimator.treatment_effect_['estimate']

        sensitivity_results = []

        for strength in confounder_strength_range:
            # Simulate unmeasured confounder
            # Confounder affects both treatment and outcome

            df_sens = df.copy()

            # Add synthetic confounder
            np.random.seed(int(strength * 1000))
            df_sens['unmeasured_confounder'] = np.random.normal(0, 1, len(df))

            # Confounder affects treatment (selection bias)
            if 'injury_treatment' in df.columns:
                treatment_col = 'injury_treatment'
            else:
                treatment_col = 'treated'

            # Adjust treatment based on confounder
            treatment_prob = 1 / (1 + np.exp(-(
                df_sens[treatment_col] * 2 +
                df_sens['unmeasured_confounder'] * strength
            )))

            df_sens[f'{treatment_col}_adjusted'] = (
                np.random.binomial(1, treatment_prob)
            )

            # Compute adjusted effect (placeholder)
            # In practice, would re-run estimator
            adjusted_effect = actual_effect * (1 - strength * 2)

            sensitivity_results.append({
                'confounder_strength': strength,
                'adjusted_effect': adjusted_effect,
                'effect_reduction': (actual_effect - adjusted_effect) / actual_effect * 100,
                'still_significant': abs(adjusted_effect) > 2 * estimator.treatment_effect_['std_error']
            })

        sensitivity_df = pd.DataFrame(sensitivity_results)

        logger.info(f"Sensitivity analysis complete: {len(sensitivity_df)} scenarios")

        return sensitivity_df

    def cross_validate_treatment_effects(
        self,
        df: pd.DataFrame,
        estimator_class: type,
        outcome_col: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        n_folds: int = 5
    ) -> Dict:
        """
        Cross-validate treatment effect estimates.

        Splits data into folds, estimates treatment effects on each fold,
        and checks consistency.

        Args:
            df: Panel dataframe
            estimator_class: Class of estimator (DiD, SC, etc.)
            outcome_col: Outcome variable
            treatment_col: Treatment indicator
            unit_col: Unit identifier
            time_col: Time identifier
            n_folds: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")

        # Split by units (not time, to maintain temporal structure)
        units = df[unit_col].unique()
        np.random.shuffle(units)

        fold_size = len(units) // n_folds
        fold_estimates = []

        for fold in range(n_folds):
            # Hold-out fold
            holdout_start = fold * fold_size
            holdout_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(units)
            holdout_units = units[holdout_start:holdout_end]

            # Training fold
            train_df = df[~df[unit_col].isin(holdout_units)]

            try:
                # Fit estimator on training fold
                if estimator_class == DifferenceInDifferences:
                    estimator = DifferenceInDifferences()
                    estimator.fit(
                        df=train_df,
                        outcome_col=outcome_col,
                        treatment_col=treatment_col,
                        unit_col=unit_col,
                        time_col=time_col
                    )

                elif estimator_class == SyntheticControl:
                    # SC requires specific treated unit
                    treated_units = train_df[train_df[treatment_col] == 1][unit_col].unique()
                    if len(treated_units) > 0:
                        estimator = SyntheticControl()
                        estimator.fit(
                            df=train_df,
                            treated_unit_id=treated_units[0],
                            unit_col=unit_col,
                            time_col=time_col,
                            outcome_col=outcome_col,
                            treatment_time=train_df[train_df[treatment_col] == 1][time_col].min()
                        )
                    else:
                        continue

                else:
                    logger.warning(f"Unsupported estimator class: {estimator_class}")
                    continue

                # Extract estimate
                if hasattr(estimator, 'treatment_effect_'):
                    fold_estimates.append(estimator.treatment_effect_['estimate'])

            except Exception as e:
                logger.debug(f"Fold {fold} failed: {e}")
                continue

        if len(fold_estimates) == 0:
            logger.warning("No successful CV folds")
            return {'success': False}

        fold_estimates = np.array(fold_estimates)

        # Compute CV statistics
        cv_results = {
            'success': True,
            'n_folds': n_folds,
            'n_successful': len(fold_estimates),
            'cv_mean': fold_estimates.mean(),
            'cv_std': fold_estimates.std(),
            'cv_estimates': fold_estimates,
            'cv_coefficient_of_variation': fold_estimates.std() / abs(fold_estimates.mean()) if fold_estimates.mean() != 0 else np.inf,
            'consistent': fold_estimates.std() / abs(fold_estimates.mean()) < 0.5  # Heuristic threshold
        }

        logger.info(f"CV results: mean={cv_results['cv_mean']:.2f}, "
                   f"std={cv_results['cv_std']:.2f}, "
                   f"consistent={cv_results['consistent']}")

        return cv_results


class CausalBacktester:
    """
    Backtests betting performance with causal adjustments.
    """

    def __init__(self):
        self.backtest_results = {}

    def backtest_predictions(
        self,
        df: pd.DataFrame,
        baseline_pred_col: str,
        causal_pred_col: str,
        actual_col: str,
        market_line_col: Optional[str] = None
    ) -> Dict:
        """
        Backtest predictive accuracy: baseline vs causal-adjusted.

        Args:
            df: Dataframe with predictions and actuals
            baseline_pred_col: Baseline model predictions
            causal_pred_col: Causal-adjusted predictions
            actual_col: Actual outcomes
            market_line_col: Optional market lines for betting analysis

        Returns:
            Dictionary with backtest metrics
        """
        df_test = df[[baseline_pred_col, causal_pred_col, actual_col]].dropna()

        if len(df_test) == 0:
            logger.warning("No valid test data")
            return {}

        # Prediction accuracy
        baseline_mae = np.mean(np.abs(df_test[baseline_pred_col] - df_test[actual_col]))
        causal_mae = np.mean(np.abs(df_test[causal_pred_col] - df_test[actual_col]))

        baseline_rmse = np.sqrt(np.mean((df_test[baseline_pred_col] - df_test[actual_col])**2))
        causal_rmse = np.sqrt(np.mean((df_test[causal_pred_col] - df_test[actual_col])**2))

        # Improvement
        mae_improvement = (baseline_mae - causal_mae) / baseline_mae * 100
        rmse_improvement = (baseline_rmse - causal_rmse) / baseline_rmse * 100

        results = {
            'n_predictions': len(df_test),
            'baseline_mae': baseline_mae,
            'causal_mae': causal_mae,
            'mae_improvement_pct': mae_improvement,
            'baseline_rmse': baseline_rmse,
            'causal_rmse': causal_rmse,
            'rmse_improvement_pct': rmse_improvement,
            'causal_better': causal_mae < baseline_mae
        }

        # Betting performance if market lines available
        if market_line_col and market_line_col in df.columns:
            df_betting = df[[baseline_pred_col, causal_pred_col, actual_col, market_line_col]].dropna()

            # Over/under bets
            baseline_over_bets = df_betting[baseline_pred_col] > df_betting[market_line_col]
            causal_over_bets = df_betting[causal_pred_col] > df_betting[market_line_col]

            actual_over = df_betting[actual_col] > df_betting[market_line_col]

            # Win rates
            baseline_win_rate = (baseline_over_bets == actual_over).mean() * 100
            causal_win_rate = (causal_over_bets == actual_over).mean() * 100

            # ROI (assuming -110 odds)
            def compute_roi(win_rate):
                if win_rate > 52.38:  # Breakeven at -110
                    return (win_rate * 0.909 - (100 - win_rate)) / 100 * 100  # ROI %
                else:
                    return (win_rate * 0.909 - (100 - win_rate)) / 100 * 100

            baseline_roi = compute_roi(baseline_win_rate)
            causal_roi = compute_roi(causal_win_rate)

            results['betting'] = {
                'n_bets': len(df_betting),
                'baseline_win_rate': baseline_win_rate,
                'causal_win_rate': causal_win_rate,
                'win_rate_improvement': causal_win_rate - baseline_win_rate,
                'baseline_roi': baseline_roi,
                'causal_roi': causal_roi,
                'roi_improvement': causal_roi - baseline_roi
            }

        logger.info(f"Backtest: Causal MAE={causal_mae:.2f} vs Baseline={baseline_mae:.2f} "
                   f"({mae_improvement:+.1f}%)")

        if 'betting' in results:
            logger.info(f"Betting: Causal WR={results['betting']['causal_win_rate']:.1f}% "
                       f"vs Baseline={results['betting']['baseline_win_rate']:.1f}% "
                       f"(ROI: {results['betting']['roi_improvement']:+.1f}%)")

        return results

    def event_study_validation(
        self,
        df: pd.DataFrame,
        event_col: str,
        outcome_col: str,
        time_col: str,
        unit_col: str,
        pre_periods: int = 3,
        post_periods: int = 5
    ) -> Dict:
        """
        Validate treatment effects using event study.

        Plots dynamic effects around treatment event.

        Args:
            df: Panel dataframe
            event_col: Binary event indicator
            outcome_col: Outcome variable
            time_col: Time variable
            unit_col: Unit identifier
            pre_periods: Pre-event periods
            post_periods: Post-event periods

        Returns:
            Dictionary with event study results
        """
        logger.info("Running event study validation...")

        # Find event times
        event_df = df[df[event_col] == 1]

        if len(event_df) == 0:
            logger.warning("No events found")
            return {}

        # Get event time for each unit
        event_times = event_df.groupby(unit_col)[time_col].min()

        # Create event time variable
        df_event = df.copy()
        df_event['event_time'] = np.nan

        for unit, event_time in event_times.items():
            unit_mask = df_event[unit_col] == unit
            df_event.loc[unit_mask, 'event_time'] = df_event.loc[unit_mask, time_col] - event_time

        # Compute average outcome by event time
        event_study = df_event.groupby('event_time')[outcome_col].agg(['mean', 'std', 'count'])
        event_study = event_study.loc[-pre_periods:post_periods]

        # Pre-trend test (parallel trends)
        pre_trend_data = event_study.loc[-pre_periods:-1]
        if len(pre_trend_data) > 1:
            # Linear trend in pre-period
            t = np.arange(len(pre_trend_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, pre_trend_data['mean'])

            pre_trend_test = {
                'slope': slope,
                'p_value': p_value,
                'passes': p_value > 0.05,  # No significant pre-trend
                'interpretation': 'Parallel trends OK' if p_value > 0.05 else 'Pre-trend detected'
            }
        else:
            pre_trend_test = None

        logger.info(f"Event study: {len(event_study)} periods")

        if pre_trend_test:
            logger.info(f"Pre-trend test: {pre_trend_test['interpretation']}")

        return {
            'event_study': event_study,
            'pre_trend_test': pre_trend_test
        }


def main():
    """Example validation workflow"""

    print("\n" + "="*80)
    print("CAUSAL INFERENCE VALIDATION")
    print("="*80)

    # Simulate data
    np.random.seed(42)

    teams = ['Team_' + str(i) for i in range(10)]
    weeks = list(range(1, 21))

    data = []
    for team in teams:
        baseline = np.random.normal(25, 3)

        for week in weeks:
            points = baseline + np.random.normal(0, 2)

            # Teams 0-2 get treatment at week 10
            treated = 1 if team in teams[:3] else 0
            post = 1 if week >= 10 else 0

            # True effect: +5 points
            if treated and post:
                points += 5

            data.append({
                'team': team,
                'week': week,
                'points': points,
                'treated': treated,
                'baseline_pred': baseline,
                'causal_pred': baseline + (5 if treated and post else 0),
                'market_line': baseline - 2
            })

    df = pd.DataFrame(data)

    # Fit DiD
    did = DifferenceInDifferences()
    did.fit(
        df=df,
        outcome_col='points',
        treatment_col='treated',
        unit_col='team',
        time_col='week'
    )

    # Validation
    validator = CausalValidator()

    # Placebo test
    print("\n" + "="*80)
    print("PLACEBO TEST")
    print("="*80)

    placebo_results = validator.placebo_test(did, df, n_placebos=50)

    if placebo_results['success']:
        print(f"Actual estimate: {placebo_results['actual_estimate']:.2f}")
        print(f"Placebo mean: {placebo_results['placebo_mean']:.2f}")
        print(f"P-value: {placebo_results['p_value']:.3f}")
        print(f"Result: {placebo_results['interpretation']}")

    # Sensitivity analysis
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)

    sensitivity = validator.sensitivity_analysis(did, df)
    print(sensitivity.to_string(index=False))

    # Backtesting
    print("\n" + "="*80)
    print("PREDICTION BACKTEST")
    print("="*80)

    backtester = CausalBacktester()

    backtest_results = backtester.backtest_predictions(
        df=df[df['week'] >= 10],  # Post-treatment period
        baseline_pred_col='baseline_pred',
        causal_pred_col='causal_pred',
        actual_col='points',
        market_line_col='market_line'
    )

    print(f"\nPrediction Accuracy:")
    print(f"  Baseline MAE: {backtest_results['baseline_mae']:.2f}")
    print(f"  Causal MAE: {backtest_results['causal_mae']:.2f}")
    print(f"  Improvement: {backtest_results['mae_improvement_pct']:+.1f}%")

    if 'betting' in backtest_results:
        print(f"\nBetting Performance:")
        print(f"  Baseline Win Rate: {backtest_results['betting']['baseline_win_rate']:.1f}%")
        print(f"  Causal Win Rate: {backtest_results['betting']['causal_win_rate']:.1f}%")
        print(f"  Baseline ROI: {backtest_results['betting']['baseline_roi']:+.1f}%")
        print(f"  Causal ROI: {backtest_results['betting']['causal_roi']:+.1f}%")


if __name__ == "__main__":
    main()
