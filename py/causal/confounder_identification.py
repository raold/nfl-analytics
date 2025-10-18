#!/usr/bin/env python3
"""
Confounder Identification and Adjustment

Identifies confounding variables that affect both treatment assignment and outcomes,
and provides methods for adjustment to enable valid causal inference.

Key concepts:
- Confounders: Variables that affect both treatment and outcome, creating spurious associations
- Backdoor criterion: Conditioning on a set of variables that blocks all backdoor paths
- Overlap/Positivity: Ensuring treated and control units have similar covariate distributions
- Balance: Checking that covariates are similar between treatment and control groups
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfounderIdentifier:
    """
    Identifies potential confounders and assesses covariate balance.

    A variable is a confounder if:
    1. It affects treatment assignment (selection bias)
    2. It affects the outcome (prognostic)
    3. It's not a mediator (not on causal path from treatment to outcome)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.identified_confounders = {}

    def identify_confounders(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        candidate_confounders: list[str],
        known_mediators: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Identify confounders from candidate variables.

        Tests each candidate for:
        1. Association with treatment (predicts treatment assignment)
        2. Association with outcome (predicts outcome)

        Args:
            df: Panel dataframe
            treatment_col: Treatment indicator column
            outcome_col: Outcome variable
            candidate_confounders: List of potential confounding variables
            known_mediators: Variables on causal path (exclude from confounders)

        Returns:
            Dictionary mapping variables to confounder evidence
        """
        if known_mediators is None:
            known_mediators = []

        confounders = {}

        for var in candidate_confounders:
            if var in known_mediators:
                continue

            if var not in df.columns:
                logger.warning(f"Variable {var} not found in dataframe")
                continue

            # Test association with treatment
            treatment_assoc = self._test_treatment_association(df, var, treatment_col)

            # Test association with outcome
            outcome_assoc = self._test_outcome_association(df, var, outcome_col, treatment_col)

            # Variable is a confounder if associated with both
            is_confounder = treatment_assoc["significant"] and outcome_assoc["significant"]

            confounders[var] = {
                "is_confounder": is_confounder,
                "treatment_association": treatment_assoc,
                "outcome_association": outcome_assoc,
                "strength": self._compute_confounder_strength(treatment_assoc, outcome_assoc),
            }

        # Sort by strength
        self.identified_confounders[treatment_col] = {
            k: v
            for k, v in sorted(confounders.items(), key=lambda x: x[1]["strength"], reverse=True)
        }

        n_confounders = sum(1 for v in confounders.values() if v["is_confounder"])
        logger.info(f"Identified {n_confounders} confounders for {treatment_col}")

        return self.identified_confounders[treatment_col]

    def _test_treatment_association(self, df: pd.DataFrame, var: str, treatment_col: str) -> dict:
        """
        Test if variable predicts treatment assignment.

        Uses correlation for continuous, chi-square for categorical.
        """
        clean_df = df[[var, treatment_col]].dropna()

        if len(clean_df) < 30:
            return {"significant": False, "statistic": np.nan, "p_value": 1.0}

        # Determine variable type
        if pd.api.types.is_numeric_dtype(clean_df[var]):
            # Point-biserial correlation for continuous variable
            treated = clean_df[clean_df[treatment_col] == 1][var]
            control = clean_df[clean_df[treatment_col] == 0][var]

            if len(treated) < 5 or len(control) < 5:
                return {"significant": False, "statistic": np.nan, "p_value": 1.0}

            statistic, p_value = stats.mannwhitneyu(treated, control, alternative="two-sided")
            effect_size = abs(treated.mean() - control.mean()) / clean_df[var].std()

        else:
            # Chi-square for categorical variable
            contingency = pd.crosstab(clean_df[var], clean_df[treatment_col])
            statistic, p_value, _, _ = stats.chi2_contingency(contingency)
            effect_size = np.sqrt(statistic / len(clean_df))

        return {
            "significant": p_value < self.alpha,
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
        }

    def _test_outcome_association(
        self, df: pd.DataFrame, var: str, outcome_col: str, treatment_col: str
    ) -> dict:
        """
        Test if variable predicts outcome, controlling for treatment.

        Uses partial correlation or regression coefficient.
        """
        clean_df = df[[var, outcome_col, treatment_col]].dropna()

        if len(clean_df) < 30:
            return {"significant": False, "statistic": np.nan, "p_value": 1.0}

        # Compute partial correlation: corr(var, outcome | treatment)
        # Using residual method
        try:
            # Residualize outcome on treatment
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            lr.fit(clean_df[[treatment_col]], clean_df[outcome_col])
            outcome_resid = clean_df[outcome_col] - lr.predict(clean_df[[treatment_col]])

            # Residualize variable on treatment
            if pd.api.types.is_numeric_dtype(clean_df[var]):
                lr.fit(clean_df[[treatment_col]], clean_df[var])
                var_resid = clean_df[var] - lr.predict(clean_df[[treatment_col]])

                # Correlation of residuals
                corr, p_value = stats.pearsonr(var_resid, outcome_resid)
                statistic = corr
                effect_size = abs(corr)
            else:
                # For categorical, use ANOVA on residuals
                groups = [outcome_resid[clean_df[var] == cat] for cat in clean_df[var].unique()]
                statistic, p_value = stats.f_oneway(*groups)
                effect_size = np.sqrt(statistic / len(clean_df))

        except Exception as e:
            logger.warning(f"Could not compute outcome association for {var}: {e}")
            return {"significant": False, "statistic": np.nan, "p_value": 1.0}

        return {
            "significant": p_value < self.alpha,
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
        }

    def _compute_confounder_strength(self, treatment_assoc: dict, outcome_assoc: dict) -> float:
        """
        Compute overall strength of confounding.

        Uses product of effect sizes as heuristic.
        """
        if not (treatment_assoc["significant"] and outcome_assoc["significant"]):
            return 0.0

        return treatment_assoc["effect_size"] * outcome_assoc["effect_size"]

    def get_minimal_adjustment_set(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """
        Get minimal set of confounders to adjust for valid causal inference.

        This is a heuristic approach. Ideally would use causal graph (backdoor criterion).
        Here we use greedy selection based on confounder strength.

        Returns:
            List of confounder variable names
        """
        if treatment_col not in self.identified_confounders:
            logger.warning(f"Run identify_confounders() first for {treatment_col}")
            return []

        confounders = self.identified_confounders[treatment_col]

        # Get all significant confounders
        adjustment_set = [var for var, info in confounders.items() if info["is_confounder"]]

        logger.info(f"Minimal adjustment set for {treatment_col}: {adjustment_set}")

        return adjustment_set

    def check_positivity(
        self, df: pd.DataFrame, treatment_col: str, confounders: list[str], min_overlap: float = 0.1
    ) -> dict:
        """
        Check positivity/overlap assumption.

        Ensures that for all covariate patterns, there is a positive probability
        of receiving both treatment and control.

        Args:
            df: Panel dataframe
            treatment_col: Treatment indicator
            confounders: List of confounding variables to check
            min_overlap: Minimum proportion of units in overlap region

        Returns:
            Dictionary with overlap diagnostics
        """
        clean_df = df[[treatment_col] + confounders].dropna()

        if len(clean_df) < 50:
            return {"violation": True, "reason": "Insufficient sample size"}

        # Fit propensity score model
        X = clean_df[confounders]
        y = clean_df[treatment_col]

        try:
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X, y)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning(f"Could not fit propensity score model: {e}")
            return {"violation": True, "reason": "Propensity score model failed"}

        # Check for extreme propensity scores
        extreme_low = (propensity_scores < 0.1).sum()
        extreme_high = (propensity_scores > 0.9).sum()

        # Common support region
        treated_ps = propensity_scores[y == 1]
        control_ps = propensity_scores[y == 0]

        overlap_min = max(treated_ps.min(), control_ps.min())
        overlap_max = min(treated_ps.max(), control_ps.max())

        in_overlap = ((propensity_scores >= overlap_min) & (propensity_scores <= overlap_max)).sum()

        overlap_prop = in_overlap / len(propensity_scores)

        # Check for violations
        violation = (
            extreme_low > len(propensity_scores) * 0.05
            or extreme_high > len(propensity_scores) * 0.05
            or overlap_prop < min_overlap
        )

        return {
            "violation": violation,
            "extreme_low_count": extreme_low,
            "extreme_high_count": extreme_high,
            "overlap_proportion": overlap_prop,
            "overlap_range": (overlap_min, overlap_max),
            "propensity_scores": propensity_scores,
            "recommendation": (
                "Trim extreme propensity scores" if violation else "Overlap sufficient"
            ),
        }

    def compute_covariate_balance(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounders: list[str],
        weights: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Compute standardized mean differences for covariate balance.

        Args:
            df: Panel dataframe
            treatment_col: Treatment indicator
            confounders: List of confounders to check balance
            weights: Optional weights (e.g., from propensity score weighting)

        Returns:
            DataFrame with balance statistics for each confounder
        """
        clean_df = df[[treatment_col] + confounders].dropna()

        if weights is not None:
            if len(weights) != len(clean_df):
                logger.warning("Weights length mismatch, ignoring weights")
                weights = None

        balance_stats = []

        for var in confounders:
            treated_mask = clean_df[treatment_col] == 1

            if weights is None:
                treated_mean = clean_df.loc[treated_mask, var].mean()
                control_mean = clean_df.loc[~treated_mask, var].mean()
                treated_std = clean_df.loc[treated_mask, var].std()
                control_std = clean_df.loc[~treated_mask, var].std()
            else:
                treated_weights = weights[treated_mask]
                control_weights = weights[~treated_mask]

                treated_mean = np.average(clean_df.loc[treated_mask, var], weights=treated_weights)
                control_mean = np.average(clean_df.loc[~treated_mask, var], weights=control_weights)
                treated_std = np.sqrt(
                    np.average(
                        (clean_df.loc[treated_mask, var] - treated_mean) ** 2,
                        weights=treated_weights,
                    )
                )
                control_std = np.sqrt(
                    np.average(
                        (clean_df.loc[~treated_mask, var] - control_mean) ** 2,
                        weights=control_weights,
                    )
                )

            # Standardized mean difference
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            # Balance assessment (< 0.1 is good, < 0.25 is acceptable)
            if abs(smd) < 0.1:
                balance_status = "Good"
            elif abs(smd) < 0.25:
                balance_status = "Acceptable"
            else:
                balance_status = "Poor"

            balance_stats.append(
                {
                    "variable": var,
                    "treated_mean": treated_mean,
                    "control_mean": control_mean,
                    "treated_std": treated_std,
                    "control_std": control_std,
                    "standardized_mean_diff": smd,
                    "abs_smd": abs(smd),
                    "balance_status": balance_status,
                }
            )

        balance_df = pd.DataFrame(balance_stats)
        balance_df = balance_df.sort_values("abs_smd", ascending=False)

        n_poor = (balance_df["balance_status"] == "Poor").sum()
        logger.info(f"Covariate balance: {n_poor}/{len(confounders)} variables have poor balance")

        return balance_df

    def suggest_instrumental_variables(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        candidate_instruments: list[str],
    ) -> dict[str, dict]:
        """
        Suggest instrumental variables for addressing unmeasured confounding.

        Valid instrument must:
        1. Strongly predict treatment (relevance)
        2. Not directly affect outcome except through treatment (exclusion)
        3. Not be confounded with outcome (as-if random)

        Args:
            df: Panel dataframe
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            candidate_instruments: List of potential instruments

        Returns:
            Dictionary mapping instruments to validity assessments
        """
        instruments = {}

        for var in candidate_instruments:
            if var not in df.columns:
                continue

            clean_df = df[[var, treatment_col, outcome_col]].dropna()

            if len(clean_df) < 50:
                continue

            # Test relevance (F-statistic > 10 is rule of thumb)
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()

            # First stage: instrument -> treatment
            lr.fit(clean_df[[var]], clean_df[treatment_col])
            treatment_pred = lr.predict(clean_df[[var]])

            # F-statistic for instrument
            ss_total = ((clean_df[treatment_col] - clean_df[treatment_col].mean()) ** 2).sum()
            ss_resid = ((clean_df[treatment_col] - treatment_pred) ** 2).sum()
            f_stat = ((ss_total - ss_resid) / 1) / (ss_resid / (len(clean_df) - 2))

            # Test exclusion restriction (partial correlation with outcome | treatment)
            lr.fit(clean_df[[treatment_col]], clean_df[outcome_col])
            outcome_resid = clean_df[outcome_col] - lr.predict(clean_df[[treatment_col]])

            lr.fit(clean_df[[treatment_col]], clean_df[var])
            var_resid = clean_df[var] - lr.predict(clean_df[[treatment_col]])

            partial_corr, p_value = stats.pearsonr(var_resid, outcome_resid)

            # Assess instrument validity
            is_relevant = f_stat > 10
            is_excludable = p_value > 0.05  # No direct effect on outcome

            instruments[var] = {
                "is_valid_instrument": is_relevant and is_excludable,
                "f_statistic": f_stat,
                "is_relevant": is_relevant,
                "partial_correlation_with_outcome": partial_corr,
                "is_excludable": is_excludable,
                "strength": "Strong" if f_stat > 20 else "Weak" if f_stat > 10 else "Very Weak",
            }

        n_valid = sum(1 for v in instruments.values() if v["is_valid_instrument"])
        logger.info(f"Found {n_valid} valid instrumental variables")

        return instruments

    def detect_colliders(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str, candidate_variables: list[str]
    ) -> list[str]:
        """
        Detect collider variables that should NOT be conditioned on.

        A collider is affected by both treatment and outcome. Conditioning on
        colliders induces spurious associations (M-bias).

        Args:
            df: Panel dataframe
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            candidate_variables: Variables to test

        Returns:
            List of detected colliders
        """
        colliders = []

        for var in candidate_variables:
            if var not in df.columns:
                continue

            clean_df = df[[var, treatment_col, outcome_col]].dropna()

            if len(clean_df) < 30:
                continue

            # Test if treatment predicts variable
            treatment_assoc = self._test_treatment_association(clean_df, var, treatment_col)

            # Test if outcome predicts variable
            outcome_pred = self._test_treatment_association(clean_df, var, outcome_col)

            # Collider if both associations significant
            if treatment_assoc["significant"] and outcome_pred["significant"]:
                colliders.append(var)

        logger.info(f"Detected {len(colliders)} collider variables: {colliders}")
        logger.warning("DO NOT condition on colliders - they induce spurious associations")

        return colliders


def main():
    """Example usage of confounder identifier"""

    # Mock data
    np.random.seed(42)
    n = 1000

    # True confounder: player ability affects both injury risk and performance
    player_ability = np.random.normal(0, 1, n)

    # True mediator: plays_per_game is on causal path
    plays_per_game = 50 + 10 * player_ability + np.random.normal(0, 5, n)

    # Treatment: injury (affected by ability)
    injury_prob = 1 / (1 + np.exp(-(0.3 * player_ability + np.random.normal(0, 0.5, n))))
    injury = (injury_prob > 0.5).astype(int)

    # Outcome: yards (affected by ability and injury)
    yards = 70 + 20 * player_ability - 15 * injury + np.random.normal(0, 10, n)

    # Collider: pro_bowl_selection (affected by both injury and yards)
    pro_bowl = ((yards > 90) & (injury == 0)).astype(int)

    # Irrelevant variable
    shoe_size = np.random.normal(10, 1, n)

    mock_df = pd.DataFrame(
        {
            "player_ability": player_ability,
            "plays_per_game": plays_per_game,
            "injury": injury,
            "yards": yards,
            "pro_bowl": pro_bowl,
            "shoe_size": shoe_size,
            "team_wins": np.random.randint(6, 12, n),  # Another confounder
        }
    )

    identifier = ConfounderIdentifier()

    # Identify confounders
    print("\n" + "=" * 80)
    print("CONFOUNDER IDENTIFICATION")
    print("=" * 80)

    confounders = identifier.identify_confounders(
        mock_df,
        treatment_col="injury",
        outcome_col="yards",
        candidate_confounders=["player_ability", "plays_per_game", "shoe_size", "team_wins"],
        known_mediators=["plays_per_game"],
    )

    print("\nIdentified Confounders:")
    for var, info in confounders.items():
        if info["is_confounder"]:
            print(f"  ✓ {var}: strength={info['strength']:.3f}")
        else:
            print(f"  ✗ {var}: NOT a confounder")

    # Get adjustment set
    print("\n" + "=" * 80)
    print("MINIMAL ADJUSTMENT SET")
    print("=" * 80)

    adjustment_set = identifier.get_minimal_adjustment_set(mock_df, "injury", "yards")
    print(f"Adjust for: {adjustment_set}")

    # Check positivity
    print("\n" + "=" * 80)
    print("POSITIVITY CHECK")
    print("=" * 80)

    positivity = identifier.check_positivity(mock_df, "injury", adjustment_set)
    print(f"Overlap violation: {positivity['violation']}")
    print(f"Overlap proportion: {positivity['overlap_proportion']:.1%}")
    print(f"Recommendation: {positivity['recommendation']}")

    # Check balance
    print("\n" + "=" * 80)
    print("COVARIATE BALANCE")
    print("=" * 80)

    balance = identifier.compute_covariate_balance(mock_df, "injury", adjustment_set)
    print(balance[["variable", "standardized_mean_diff", "balance_status"]])

    # Detect colliders
    print("\n" + "=" * 80)
    print("COLLIDER DETECTION")
    print("=" * 80)

    colliders = identifier.detect_colliders(
        mock_df, "injury", "yards", ["pro_bowl", "player_ability", "shoe_size"]
    )
    print(f"Detected colliders: {colliders}")
    print("⚠️  Do NOT condition on these variables!")


if __name__ == "__main__":
    main()
