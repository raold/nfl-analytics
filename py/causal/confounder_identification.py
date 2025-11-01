"""
Confounder Identification and Adjustment

Methods for identifying and adjusting for confounding.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ConfounderAdjuster:
    """Methods for confounder adjustment"""

    def propensity_score_matching(
        self,
        df: pd.DataFrame,
        treatment: str,
        confounders: List[str],
        caliper: float = 0.1
    ) -> pd.DataFrame:
        """
        Match treated units to controls using propensity scores.

        Args:
            df: Panel data
            treatment: Treatment column
            confounders: List of confounder variables
            caliper: Maximum propensity score difference for matching

        Returns:
            Matched dataset
        """
        from sklearn.linear_model import LogisticRegression

        # Estimate propensity scores
        X = df[confounders].values
        y = df[treatment].values

        model = LogisticRegression()
        model.fit(X, y)

        df['propensity_score'] = model.predict_proba(X)[:, 1]

        # Match each treated unit to nearest control
        treated = df[df[treatment] == 1].copy()
        control = df[df[treatment] == 0].copy()

        matched_indices = []

        for idx, treated_row in treated.iterrows():
            ps_treated = treated_row['propensity_score']

            # Find nearest control within caliper
            control['ps_diff'] = np.abs(control['propensity_score'] - ps_treated)
            nearest = control[control['ps_diff'] <= caliper].nsmallest(1, 'ps_diff')

            if len(nearest) > 0:
                matched_indices.append((idx, nearest.index[0]))

        logger.info(f"Matched {len(matched_indices)} treated units to controls")

        # Build matched dataset
        matched_df = []
        for t_idx, c_idx in matched_indices:
            matched_df.append(df.loc[t_idx])
            matched_df.append(df.loc[c_idx])

        return pd.DataFrame(matched_df)

    def inverse_propensity_weighting(
        self,
        df: pd.DataFrame,
        treatment: str,
        confounders: List[str]
    ) -> pd.DataFrame:
        """
        Weight observations by inverse of propensity score.

        Returns:
            DataFrame with 'ipw_weight' column
        """
        from sklearn.linear_model import LogisticRegression

        X = df[confounders].values
        y = df[treatment].values

        model = LogisticRegression()
        model.fit(X, y)

        df['propensity_score'] = model.predict_proba(X)[:, 1]

        # IPW weight = 1/p(T|X) for treated, 1/(1-p(T|X)) for control
        df['ipw_weight'] = np.where(
            df[treatment] == 1,
            1 / df['propensity_score'],
            1 / (1 - df['propensity_score'])
        )

        # Stabilize weights
        df['ipw_weight'] = df['ipw_weight'] / df['ipw_weight'].mean()

        logger.info(f"IPW weights: mean={df['ipw_weight'].mean():.2f}, std={df['ipw_weight'].std():.2f}")

        return df
