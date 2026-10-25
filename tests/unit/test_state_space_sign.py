"""Spread sign convention in the state-space ratings model.

games.spread_close is nflverse spread_line: the expected HOME margin, positive when the home
team is favored. Home covers iff (home_score - away_score) - spread_close > 0; pushes have no
ATS result. Games below are real (values from games / nflverse schedules).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_analytics.models.state_space import StateSpaceRatings, evaluate_model

# (game_id, home, away, home_score, away_score, spread_close)
BUF_AT_BAL_2018 = ("2018_01_BUF_BAL", "BAL", "BUF", 47, 3, 7.5)  # BAL -7.5, won by 44: covered
DET_AT_KC_2023 = ("2023_01_DET_KC", "KC", "DET", 20, 21, 4.0)  # KC -4, lost by 1: not covered
SEA_AT_DEN_2018 = ("2018_01_SEA_DEN", "DEN", "SEA", 27, 24, 3.0)  # DEN -3, won by 3: push


def _model_with_margin(home: str, away: str, mean_margin: float) -> StateSpaceRatings:
    """Ratings that make predict_margin(home, away) return mean_margin."""
    model = StateSpaceRatings(hfa_init=2.5)
    model.ratings[home] = (mean_margin - model.hfa, 4.0)
    model.ratings[away] = (0.0, 4.0)
    return model


def _games(*rows) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["game_id", "home_team", "away_team", "home_score", "away_score", "spread_close"],
    )
    df["margin"] = df["home_score"] - df["away_score"]
    return df


class TestPredictProbAts:
    def test_model_above_line_favors_home_cover(self):
        model = _model_with_margin("BAL", "BUF", 10.0)
        assert model.predict_prob_ats("BAL", "BUF", 7.5) > 0.5

    def test_model_below_line_favors_away_cover(self):
        model = _model_with_margin("BAL", "BUF", 10.0)
        assert model.predict_prob_ats("BAL", "BUF", 12.5) < 0.5

    def test_probability_is_phi_of_edge(self):
        model = _model_with_margin("BAL", "BUF", 10.0)
        mean, std = model.predict_margin("BAL", "BUF")
        assert mean == pytest.approx(10.0)
        assert model.predict_prob_ats("BAL", "BUF", 7.5) == pytest.approx(
            norm.cdf((10.0 - 7.5) / std)
        )

    def test_model_equal_to_line_is_coin_flip(self):
        model = _model_with_margin("KC", "DET", 4.0)
        assert model.predict_prob_ats("KC", "DET", 4.0) == pytest.approx(0.5)

    def test_home_underdog_line(self):
        # Negative spread_close = home underdog: predicting a 1-point home loss still beats +3.5
        model = _model_with_margin("PHI", "KC", -1.0)
        assert model.predict_prob_ats("PHI", "KC", -3.5) > 0.5


class TestEvaluateModelLabels:
    def test_pinned_games_graded_with_nflverse_rule(self):
        # P(cover) > 0.5 for both games, so ATS accuracy is the share of real covers:
        # BUF@BAL covered, DET@KC did not (the old sportsbook-style rule called it a cover).
        model = StateSpaceRatings(hfa_init=2.5)
        model.ratings.update({"BAL": (20.0, 4.0), "BUF": (0.0, 4.0)})
        model.ratings.update({"KC": (20.0, 4.0), "DET": (0.0, 4.0)})
        df = _games(BUF_AT_BAL_2018, DET_AT_KC_2023)

        metrics = evaluate_model(model, df)

        p_bal = model.predict_prob_ats("BAL", "BUF", 7.5)
        p_kc = model.predict_prob_ats("KC", "DET", 4.0)
        assert p_bal > 0.5 and p_kc > 0.5
        assert metrics["n_ats"] == 2
        assert metrics["accuracy_ats"] == pytest.approx(0.5)
        assert metrics["brier_ats"] == pytest.approx(((p_bal - 1.0) ** 2 + (p_kc - 0.0) ** 2) / 2)

    def test_push_is_excluded_from_ats_metrics(self):
        model = StateSpaceRatings(hfa_init=2.5)
        model.ratings.update({"BAL": (20.0, 4.0), "BUF": (0.0, 4.0)})
        model.ratings.update({"KC": (20.0, 4.0), "DET": (0.0, 4.0)})
        model.ratings.update({"DEN": (5.0, 4.0), "SEA": (0.0, 4.0)})
        with_push = evaluate_model(model, _games(BUF_AT_BAL_2018, DET_AT_KC_2023, SEA_AT_DEN_2018))
        without = evaluate_model(model, _games(BUF_AT_BAL_2018, DET_AT_KC_2023))

        assert with_push["n_games"] == 3
        assert with_push["n_ats"] == 2
        for key in ("brier_ats", "logloss_ats", "accuracy_ats"):
            assert with_push[key] == pytest.approx(without[key])
        # Win-probability metrics still use all three games
        assert with_push["brier_win"] != pytest.approx(without["brier_win"])

    def test_missing_spread_skips_ats(self):
        model = StateSpaceRatings()
        df = _games(BUF_AT_BAL_2018, DET_AT_KC_2023)
        df["spread_close"] = np.nan
        metrics = evaluate_model(model, df)
        assert "accuracy_ats" not in metrics
        assert metrics["n_games"] == 2
