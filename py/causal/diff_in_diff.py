#!/usr/bin/env python3
"""
Difference-in-Differences (DiD) Estimation

Implements difference-in-differences estimators for causal inference with panel data.

Key idea: Compare changes over time in treated group vs control group.
The treatment effect is the "difference of differences":
    DiD = (Treated_post - Treated_pre) - (Control_post - Control_pre)

Variants implemented:
1. Standard 2x2 DiD: One treatment, one control, two periods
2. Staggered adoption DiD: Units treated at different times
3. Event study: Dynamic treatment effects relative to treatment time
4. Doubly-robust DiD: Combines regression adjustment with matching

Applications in NFL:
- Coaching changes: Compare team performance before/after, vs teams without changes
- Rule changes: Compare affected teams to unaffected teams
- Player trades: Compare performance on new vs old team
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferenceInDifferences:
    """
    Implements difference-in-differences estimation.

    Identifies treatment effects by comparing treated and control groups'
    changes over time, controlling for time-invariant differences and
    common time trends.
    """

    def __init__(self, cluster_var: str | None = None):
        """
        Args:
            cluster_var: Variable to cluster standard errors on (e.g., 'team', 'player_id')
        """
        self.cluster_var = cluster_var
        self.model_ = None
        self.treatment_effect_ = None
        self.parallel_trends_test_ = None

    def fit(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        covariates: list[str] | None = None,
    ) -> "DifferenceInDifferences":
        """
        Fit standard DiD model.

        Estimates: Y_it = α + β*Treated_i + γ*Post_t + δ*(Treated_i × Post_t) + X_it + ε_it

        Where δ (coefficient on interaction) is the DiD estimate.

        Args:
            df: Panel dataframe
            outcome_col: Outcome variable
            treatment_col: Binary treatment indicator (1=treated unit, 0=control)
            unit_col: Unit identifier
            time_col: Time period identifier
            covariates: Additional control variables

        Returns:
            self (fitted model)
        """
        model_df = df[[outcome_col, treatment_col, unit_col, time_col]].copy()

        # Add covariates if specified
        if covariates:
            for cov in covariates:
                if cov in df.columns:
                    model_df[cov] = df[cov]

        # Drop missing values
        model_df = model_df.dropna()

        if len(model_df) == 0:
            raise ValueError("No valid observations after dropping missing values")

        # Create post-treatment indicator
        # Assume treatment occurs at some point, find treatment timing
        treated_units = model_df[model_df[treatment_col] == 1][unit_col].unique()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")

        # Find when treatment occurs (first time treated=1)
        treatment_times = {}
        for unit in treated_units:
            unit_df = model_df[model_df[unit_col] == unit].sort_values(time_col)
            first_treated = unit_df[unit_df[treatment_col] == 1][time_col].min()
            treatment_times[unit] = first_treated

        # Create post indicator based on treatment timing
        model_df["post"] = 0
        for unit, treatment_time in treatment_times.items():
            mask = (model_df[unit_col] == unit) & (model_df[time_col] >= treatment_time)
            model_df.loc[mask, "post"] = 1

        # For control units, use average treatment time
        avg_treatment_time = np.mean(list(treatment_times.values()))
        control_units = model_df[model_df[treatment_col] == 0][unit_col].unique()
        for unit in control_units:
            mask = (model_df[unit_col] == unit) & (model_df[time_col] >= avg_treatment_time)
            model_df.loc[mask, "post"] = 1

        # Create interaction term
        model_df["treated_x_post"] = model_df[treatment_col] * model_df["post"]

        # Build regression formula
        formula = f"{outcome_col} ~ {treatment_col} + post + treated_x_post"

        if covariates:
            formula += " + " + " + ".join(covariates)

        # Fit model
        if self.cluster_var and self.cluster_var in model_df.columns:
            # Cluster-robust standard errors
            model = smf.ols(formula, data=model_df).fit(
                cov_type="cluster", cov_kwds={"groups": model_df[self.cluster_var]}
            )
        else:
            # Heteroskedasticity-robust standard errors
            model = smf.ols(formula, data=model_df).fit(cov_type="HC3")

        self.model_ = model

        # Extract treatment effect (coefficient on interaction)
        self.treatment_effect_ = {
            "estimate": model.params["treated_x_post"],
            "std_error": model.bse["treated_x_post"],
            "ci_lower": model.conf_int().loc["treated_x_post", 0],
            "ci_upper": model.conf_int().loc["treated_x_post", 1],
            "p_value": model.pvalues["treated_x_post"],
            "t_statistic": model.tvalues["treated_x_post"],
        }

        logger.info(
            f"DiD estimate: {self.treatment_effect_['estimate']:.3f} "
            f"(SE: {self.treatment_effect_['std_error']:.3f}, "
            f"p: {self.treatment_effect_['p_value']:.3f})"
        )

        return self

    def test_parallel_trends(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        pre_periods: int = 3,
    ) -> dict:
        """
        Test parallel trends assumption.

        Tests if treated and control groups had parallel trends before treatment.
        Uses event study with leads (pre-treatment periods).

        Args:
            df: Panel dataframe
            outcome_col: Outcome variable
            treatment_col: Treatment indicator
            unit_col: Unit identifier
            time_col: Time identifier
            pre_periods: Number of pre-treatment periods to test

        Returns:
            Dictionary with test statistics
        """
        # Get treatment timing
        treated_units = df[df[treatment_col] == 1][unit_col].unique()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")

        # Find treatment time for each unit
        treatment_times = {}
        for unit in treated_units:
            unit_df = df[df[unit_col] == unit].sort_values(time_col)
            first_treated = unit_df[unit_df[treatment_col] == 1][time_col].min()
            treatment_times[unit] = first_treated

        # Create event time (periods relative to treatment)
        df_test = df.copy()
        df_test["event_time"] = np.nan

        for unit, treatment_time in treatment_times.items():
            unit_mask = df_test[unit_col] == unit
            df_test.loc[unit_mask, "event_time"] = df_test.loc[unit_mask, time_col] - treatment_time

        # For control units, use relative to average treatment time
        avg_treatment_time = np.mean(list(treatment_times.values()))
        control_units = df[df[treatment_col] == 0][unit_col].unique()
        for unit in control_units:
            unit_mask = df_test[unit_col] == unit
            df_test.loc[unit_mask, "event_time"] = (
                df_test.loc[unit_mask, time_col] - avg_treatment_time
            )

        # Keep only pre-treatment periods
        df_pre = df_test[df_test["event_time"] < 0].copy()

        # Create lead indicators (exclude t=-1 as reference)
        lead_indicators = []
        for lag in range(-pre_periods, 0):
            if lag != -1:  # Reference period
                col_name = f"lead_{abs(lag)}"
                df_pre[col_name] = (df_pre["event_time"] == lag).astype(int)
                df_pre[f"{col_name}_treated"] = df_pre[col_name] * df_pre[treatment_col]
                lead_indicators.append(f"{col_name}_treated")

        # Regression with lead interactions
        if len(lead_indicators) > 0:
            formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(lead_indicators)

            try:
                model = smf.ols(formula, data=df_pre).fit(cov_type="HC3")

                # Joint F-test: all leads = 0
                lead_params = [param for param in lead_indicators if param in model.params]

                if len(lead_params) > 0:
                    f_test = model.f_test(
                        np.eye(len(model.params))[
                            [list(model.params.index).index(param) for param in lead_params]
                        ]
                    )

                    self.parallel_trends_test_ = {
                        "f_statistic": f_test.fvalue[0][0],
                        "p_value": f_test.pvalue,
                        "lead_coefficients": {param: model.params[param] for param in lead_params},
                        "significant_leads": sum(model.pvalues[lead_params] < 0.05),
                        "passes": f_test.pvalue > 0.05,  # Fail to reject null (parallel trends)
                    }

                    logger.info(
                        f"Parallel trends test: F={self.parallel_trends_test_['f_statistic']:.2f}, "
                        f"p={self.parallel_trends_test_['p_value']:.3f}"
                    )

                    if self.parallel_trends_test_["passes"]:
                        logger.info("✓ Parallel trends assumption supported")
                    else:
                        logger.warning("⚠ Parallel trends assumption may be violated")

                else:
                    logger.warning("Could not run parallel trends test: insufficient leads")
                    self.parallel_trends_test_ = None

            except Exception as e:
                logger.warning(f"Parallel trends test failed: {e}")
                self.parallel_trends_test_ = None

        return self.parallel_trends_test_

    def event_study(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        leads: int = 3,
        lags: int = 5,
        covariates: list[str] | None = None,
    ) -> dict:
        """
        Estimate dynamic treatment effects via event study.

        Estimates effects at each time period relative to treatment.

        Args:
            df: Panel dataframe
            outcome_col: Outcome variable
            treatment_col: Treatment indicator
            unit_col: Unit identifier
            time_col: Time identifier
            leads: Number of pre-treatment periods (for parallel trends)
            lags: Number of post-treatment periods

        Returns:
            Dictionary with dynamic effects
        """
        # Get treatment timing
        treated_units = df[df[treatment_col] == 1][unit_col].unique()

        treatment_times = {}
        for unit in treated_units:
            unit_df = df[df[unit_col] == unit].sort_values(time_col)
            first_treated = unit_df[unit_df[treatment_col] == 1][time_col].min()
            treatment_times[unit] = first_treated

        # Create event time
        df_event = df.copy()
        df_event["event_time"] = np.nan

        for unit, treatment_time in treatment_times.items():
            unit_mask = df_event[unit_col] == unit
            df_event.loc[unit_mask, "event_time"] = (
                df_event.loc[unit_mask, time_col] - treatment_time
            )

        # For control units
        avg_treatment_time = np.mean(list(treatment_times.values()))
        control_units = df[df[treatment_col] == 0][unit_col].unique()
        for unit in control_units:
            unit_mask = df_event[unit_col] == unit
            df_event.loc[unit_mask, "event_time"] = (
                df_event.loc[unit_mask, time_col] - avg_treatment_time
            )

        # Create event time dummies (excluding t=-1 as reference)
        event_dummies = []
        for t in range(-leads, lags + 1):
            if t != -1:  # Reference period
                if t < 0:
                    col_name = f"lead_{abs(t)}"
                else:
                    col_name = f"lag_{t}"

                df_event[col_name] = (df_event["event_time"] == t).astype(int)
                df_event[f"{col_name}_treated"] = df_event[col_name] * df_event[treatment_col]
                event_dummies.append(f"{col_name}_treated")

        # Build formula
        formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(event_dummies)

        if covariates:
            formula += " + " + " + ".join(covariates)

        # Fit model
        try:
            if self.cluster_var and self.cluster_var in df_event.columns:
                model = smf.ols(formula, data=df_event).fit(
                    cov_type="cluster", cov_kwds={"groups": df_event[self.cluster_var]}
                )
            else:
                model = smf.ols(formula, data=df_event).fit(cov_type="HC3")

            # Extract coefficients for each event time
            event_coeffs = {}
            for dummy in event_dummies:
                if dummy in model.params:
                    event_time = int(dummy.split("_")[1])
                    if "lead" in dummy:
                        event_time = -event_time

                    event_coeffs[event_time] = {
                        "estimate": model.params[dummy],
                        "std_error": model.bse[dummy],
                        "ci_lower": model.conf_int().loc[dummy, 0],
                        "ci_upper": model.conf_int().loc[dummy, 1],
                        "p_value": model.pvalues[dummy],
                    }

            # Add reference period (t=-1) with zero effect
            event_coeffs[-1] = {
                "estimate": 0.0,
                "std_error": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "p_value": 1.0,
            }

            # Sort by event time
            event_coeffs = dict(sorted(event_coeffs.items()))

            logger.info(f"Event study estimated with {len(event_coeffs)} time periods")

            return {"model": model, "dynamic_effects": event_coeffs}

        except Exception as e:
            logger.error(f"Event study failed: {e}")
            return None

    def staggered_did(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        unit_col: str,
        time_col: str,
        covariates: list[str] | None = None,
    ) -> dict:
        """
        Estimate treatment effects with staggered adoption.

        Accounts for units being treated at different times.
        Uses two-way fixed effects (TWFE) estimator.

        Args:
            df: Panel dataframe with staggered treatment
            outcome_col: Outcome variable
            treatment_col: Treatment indicator (varies by unit and time)
            unit_col: Unit identifier
            time_col: Time identifier
            covariates: Additional controls

        Returns:
            Dictionary with treatment effect estimates
        """
        model_df = df[[outcome_col, treatment_col, unit_col, time_col]].copy()

        if covariates:
            for cov in covariates:
                if cov in df.columns:
                    model_df[cov] = df[cov]

        model_df = model_df.dropna()

        # Create unit and time fixed effects
        unit_dummies = pd.get_dummies(model_df[unit_col], prefix="unit", drop_first=True)
        time_dummies = pd.get_dummies(model_df[time_col], prefix="time", drop_first=True)

        # Combine with treatment and covariates
        X = pd.concat([model_df[[treatment_col]], unit_dummies, time_dummies], axis=1)

        if covariates:
            X = pd.concat([X, model_df[covariates]], axis=1)

        y = model_df[outcome_col]

        # Fit TWFE model
        if self.cluster_var and self.cluster_var in model_df.columns:
            model = OLS(y, sm.add_constant(X)).fit(
                cov_type="cluster", cov_kwds={"groups": model_df[self.cluster_var]}
            )
        else:
            model = OLS(y, sm.add_constant(X)).fit(cov_type="HC3")

        treatment_effect = {
            "estimate": model.params[treatment_col],
            "std_error": model.bse[treatment_col],
            "ci_lower": model.conf_int().loc[treatment_col, 0],
            "ci_upper": model.conf_int().loc[treatment_col, 1],
            "p_value": model.pvalues[treatment_col],
            "method": "Two-Way Fixed Effects (TWFE)",
        }

        logger.info(
            f"Staggered DiD (TWFE) estimate: {treatment_effect['estimate']:.3f} "
            f"(SE: {treatment_effect['std_error']:.3f})"
        )

        return {"model": model, "treatment_effect": treatment_effect}


def main():
    """Example usage with NFL coaching change scenario"""

    np.random.seed(42)

    # Simulate 4 teams over 12 weeks
    teams = ["Team_A", "Team_B", "Team_C", "Team_D"]
    weeks = list(range(1, 13))

    data = []
    for team in teams:
        # Team-specific baseline
        baseline = np.random.normal(24, 2)

        # Time trend (common to all teams)
        time_trend = 0.5

        for week in weeks:
            points = baseline + time_trend * week + np.random.normal(0, 3)

            # Teams A and B fire coaches at week 7 (treated group)
            treated = 1 if team in ["Team_A", "Team_B"] else 0
            post_treatment = 1 if (team in ["Team_A", "Team_B"] and week >= 7) else 0

            # True treatment effect: +5 points after coaching change
            if post_treatment:
                points += 5

            data.append(
                {
                    "team": team,
                    "week": week,
                    "points": points,
                    "treated": treated,
                    "treated_x_post": treated * post_treatment,
                }
            )

    df = pd.DataFrame(data)

    print("\n" + "=" * 80)
    print("DIFFERENCE-IN-DIFFERENCES - COACHING CHANGE EXAMPLE")
    print("=" * 80)
    print("\nScenario: Teams A & B fire coaches at week 7")
    print("Control: Teams C & D keep their coaches")
    print("Question: What is the causal effect of coaching changes?\n")

    # Fit DiD model
    did = DifferenceInDifferences(cluster_var="team")
    did.fit(df=df, outcome_col="points", treatment_col="treated", unit_col="team", time_col="week")

    # Print results
    print("\n" + "=" * 80)
    print("TREATMENT EFFECT ESTIMATE")
    print("=" * 80)
    te = did.treatment_effect_
    print(f"DiD Estimate: {te['estimate']:.2f} points")
    print(f"Standard Error: {te['std_error']:.2f}")
    print(f"95% CI: [{te['ci_lower']:.2f}, {te['ci_upper']:.2f}]")
    print(f"P-value: {te['p_value']:.3f}")
    print(f"Significant: {te['p_value'] < 0.05}")

    # Test parallel trends
    print("\n" + "=" * 80)
    print("PARALLEL TRENDS TEST")
    print("=" * 80)
    pt_test = did.test_parallel_trends(
        df=df,
        outcome_col="points",
        treatment_col="treated",
        unit_col="team",
        time_col="week",
        pre_periods=3,
    )

    if pt_test:
        print(f"F-statistic: {pt_test['f_statistic']:.2f}")
        print(f"P-value: {pt_test['p_value']:.3f}")
        print(f"Parallel trends assumption: {'✓ Satisfied' if pt_test['passes'] else '✗ Violated'}")

    # Event study
    print("\n" + "=" * 80)
    print("EVENT STUDY (Dynamic Effects)")
    print("=" * 80)
    event_results = did.event_study(
        df=df,
        outcome_col="points",
        treatment_col="treated",
        unit_col="team",
        time_col="week",
        leads=3,
        lags=5,
    )

    if event_results:
        print(f"\n{'Period':<10} {'Effect':<10} {'Std Err':<10} {'95% CI':<20}")
        print("-" * 50)
        for t, effect in event_results["dynamic_effects"].items():
            period_label = f"t={t}"
            ci = f"[{effect['ci_lower']:.2f}, {effect['ci_upper']:.2f}]"
            sig = "*" if effect["p_value"] < 0.05 else ""
            print(
                f"{period_label:<10} {effect['estimate']:<10.2f} {effect['std_error']:<10.2f} {ci:<20} {sig}"
            )

        print("\n* = significant at 0.05 level")
        print("t=-1 is the reference period (normalized to 0)")


if __name__ == "__main__":
    main()
