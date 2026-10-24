#!/usr/bin/env python3
"""
Enforcement Framework Phase 3 - Demo Script
===========================================

This script demonstrates how Phase 3 enforcement rules catch common bugs.

Run with:
    python nfl_analytics/utils/enforcement_router.py examples/enforcement_phase3_demo.py

Expected output: 4 BLOCKING issues detected

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


class BuggyValidationScript:
    """
    Example validation script with bugs that Phase 3 rules would catch.

    **INTENTIONALLY BUGGY** - Do not use in production!
    """

    def __init__(self):
        """Initialize with hardcoded password (Rule #4: security_audit)."""
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": os.getenv("DB_PASSWORD"),  # ✅ GOOD - from env
        }

    def bug_1_api_contract_violation(self):
        """
        Bug #1: Deprecated ensemble parameter (Rule #15).

        Would cause: TypeError - unexpected keyword argument 'use_adaptive_weights'
        """
        from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

        # ❌ BLOCKING: api_contract_validation
        # Deprecated parameter: use_adaptive_weights
        ensemble = EnhancedEnsemble4Way(use_adaptive_weights=True)  # Deprecated in v5

        return ensemble

    def bug_2_sql_result_not_validated(self, team: str):
        """
        Bug #2: SQL result accessed without empty check (Rule #16).

        Would cause: IndexError if query returns 0 rows
        """
        conn = psycopg2.connect(**self.db_config)

        query = f"""
        SELECT elo_rating
        FROM mart.team_elo
        WHERE team = '{team}'
        ORDER BY season DESC, week DESC
        LIMIT 1
        """

        # ❌ BLOCKING: sql_result_validation
        # DataFrame 'elo_rating' accessed without empty check
        elo_rating = pd.read_sql_query(query, conn)
        rating = elo_rating.iloc[0]["elo_rating"]  # Crashes if empty

        conn.close()
        return rating

    def bug_3_r_prediction_nan_not_checked(self, model, newdata):
        """
        Bug #3: R model prediction not validated for NaN (Rule #17).

        Would cause: NaN propagates to ensemble mean, corrupts predictions
        """
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        # ❌ WARNING: r_prediction_validation
        # R model predictions not validated for NaN
        # rpy2 calls not wrapped in try/except
        r_pred = ro.r("posterior_predict")(model, newdata=newdata, ndraws=100)
        prediction = np.array(r_pred).mean()  # Returns NaN if model failed

        return prediction

    def bug_4_ensemble_components_not_validated(self, player_id: str, season: int, week: int):
        """
        Bug #4: Ensemble components not validated for minimum count (Rule #18).

        Would cause: Poor predictions with only 2/4 components
        """
        from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

        ensemble = EnhancedEnsemble4Way(db_password=os.getenv("DB_PASSWORD"))

        # Get predictions from all components
        p1 = ensemble._predict_state_space(player_id, season, week)
        p2 = None  # Component failed to load
        p3 = None  # Component failed to load
        p4 = ensemble._predict_xgboost(player_id, season, week)

        # ❌ BLOCKING: ensemble_components_check
        # Ensemble predictions not validated for minimum components
        # No logging for component failures
        valid_predictions = [p1, p2, p3, p4]  # Contains None values
        ensemble_mean = np.nanmean([p for p in valid_predictions if p is not None])

        return ensemble_mean

    def corrected_example_1(self):
        """✅ CORRECT: Proper API contract usage."""
        from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

        ensemble = EnhancedEnsemble4Way(
            db_password=os.getenv("DB_PASSWORD"),
            # No deprecated parameters
        )

        return ensemble

    def corrected_example_2(self, team: str):
        """✅ CORRECT: SQL result validation before access."""
        conn = psycopg2.connect(**self.db_config)

        query = f"""
        SELECT elo_rating
        FROM mart.team_elo
        WHERE team = '{team}'
        ORDER BY season DESC, week DESC
        LIMIT 1
        """

        elo_rating = pd.read_sql_query(query, conn)

        # ✅ Fail-fast validation
        if len(elo_rating) == 0:
            conn.close()
            return None

        rating = elo_rating.iloc[0]["elo_rating"]
        conn.close()
        return rating

    def corrected_example_3(self, model, newdata):
        """✅ CORRECT: R prediction NaN validation."""
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        try:
            # ✅ Wrapped in try/except
            r_pred = ro.r("posterior_predict")(model, newdata=newdata, ndraws=100)
            prediction = np.array(r_pred).mean()

            # ✅ NaN validation
            if np.isnan(prediction):
                print("⚠️ R model returned NaN prediction")
                return None

            return prediction

        except Exception as e:
            print(f"❌ R prediction failed: {e}")
            return None

    def corrected_example_4(self, player_id: str, season: int, week: int):
        """✅ CORRECT: Ensemble component validation."""
        from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

        ensemble = EnhancedEnsemble4Way(db_password=os.getenv("DB_PASSWORD"))

        # Get predictions from all components
        predictions = []

        p1 = ensemble._predict_state_space(player_id, season, week)
        if p1 is None:
            print("⚠️ State-space component failed")  # ✅ Logging
        predictions.append(p1)

        p2 = None  # Simulate failure
        if p2 is None:
            print("⚠️ Hierarchical component failed")  # ✅ Logging
        predictions.append(p2)

        p3 = None  # Simulate failure
        if p3 is None:
            print("⚠️ XGBoost component failed")  # ✅ Logging
        predictions.append(p3)

        p4 = ensemble._predict_xgboost(player_id, season, week)
        if p4 is None:
            print("⚠️ XGBoost component failed")  # ✅ Logging
        predictions.append(p4)

        # ✅ Minimum component validation
        valid_predictions = [p for p in predictions if p is not None]
        if len(valid_predictions) < 3:
            raise RuntimeError(
                f"Only {len(valid_predictions)}/4 components available. "
                f"Minimum 3 required for reliable predictions."
            )

        ensemble_mean = np.mean(valid_predictions)
        return ensemble_mean


def main():
    """Run demo to show enforcement rules in action."""
    print("=" * 70)
    print("Enforcement Framework Phase 3 - Demo")
    print("=" * 70)
    print()
    print("This script demonstrates bugs that Phase 3 rules would catch.")
    print()
    print("Run enforcement validation:")
    print("  python nfl_analytics/utils/enforcement_router.py examples/enforcement_phase3_demo.py")
    print()
    print("Expected output:")
    print("  ❌ BLOCKING: api_contract_validation (Bug #1)")
    print("  ❌ BLOCKING: sql_result_validation (Bug #2)")
    print("  ⚠️  WARNING: r_prediction_validation (Bug #3)")
    print("  ❌ BLOCKING: ensemble_components_check (Bug #4)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
