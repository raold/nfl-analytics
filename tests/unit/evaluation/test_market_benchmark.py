"""
Market benchmark tests on real games (data/historical_weather_2018_2024.csv, nflverse lines).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_analytics.evaluation.market_benchmark import (
    ImplausibleResultError,
    check_home_cover_rate,
    check_plausibility,
    elo_walkforward,
    load_market_from_csv,
    market_baseline,
    power_ratings_walkforward,
    run_benchmark,
    to_markdown,
)
from nfl_analytics.evaluation.market_benchmark.metrics import (
    block_bootstrap_roi,
    closing_line_value,
    grade_ats,
    information_regression,
    wilson_interval,
)
from nfl_analytics.evaluation.market_benchmark.odds import (
    american_to_decimal,
    breakeven_win_rate,
    devig_proportional,
    devig_shin,
)

DATA = Path(__file__).resolve().parents[3] / "data" / "historical_weather_2018_2024.csv"


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"{DATA} not available")
    return load_market_from_csv(DATA)


@pytest.fixture(scope="module")
def evaluated(market) -> pd.DataFrame:
    """2019-2024 with walk-forward power ratings (2018 seeds the preseason priors)."""
    df = market.assign(pred_home_margin=power_ratings_walkforward(market))
    return df[df["season"] >= 2019].reset_index(drop=True)


class TestOdds:
    def test_decimal_and_breakeven(self):
        assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
        assert american_to_decimal(+150) == pytest.approx(2.5)
        assert breakeven_win_rate(-110) == pytest.approx(0.5238, abs=1e-4)

    def test_devig_symmetric_market_is_even(self):
        p = 1 / american_to_decimal(-110)
        assert devig_proportional(p, p) == pytest.approx((0.5, 0.5))
        assert devig_shin(p, p) == pytest.approx((0.5, 0.5))

    def test_shin_shades_longshot_more_than_proportional(self):
        fav, dog = 1 / american_to_decimal(-300), 1 / american_to_decimal(+250)
        prop, shin = devig_proportional(fav, dog), devig_shin(fav, dog)
        assert sum(shin) == pytest.approx(1.0)
        assert shin[0] > prop[0]  # favourite gets more of the probability under Shin


class TestStatistics:
    def test_wilson_interval_known_value(self):
        lo, hi = wilson_interval(50, 100)
        assert lo == pytest.approx(0.4038, abs=1e-3)
        assert hi == pytest.approx(0.5962, abs=1e-3)

    def test_bootstrap_interval_contains_point_estimate(self, evaluated):
        bets = grade_ats(evaluated, min_edge=1.0)
        roi = bets.loc[bets["result"] != "push", "profit"].mean()
        lo, hi = block_bootstrap_roi(bets, n_boot=500, seed=1)
        assert lo < roi < hi
        assert block_bootstrap_roi(bets, n_boot=500, seed=1) == (lo, hi)  # reproducible

    def test_closing_line_value_signs(self):
        bets = pd.DataFrame(
            {"side": ["home", "away"], "spread_close": [4.0, 4.0], "spread_at_bet": [3.0, 3.0]}
        )
        # Backing home at 3 that closes at 4 gained a point; backing away at 3 lost one
        assert list(closing_line_value(bets)) == [1.0, -1.0]


class TestBaselines:
    def test_market_baseline_finds_no_edge(self, market):
        df = market.assign(pred_home_margin=market_baseline(market))
        assert grade_ats(df, min_edge=0.5).empty
        assert information_regression(df) is None

    def test_power_ratings_have_no_edge(self, evaluated):
        result = run_benchmark(evaluated, "power_ratings")
        assert result["ats"]["bets"] > 1000
        assert 0.46 <= result["ats"]["win_rate"] <= 0.54
        lo, hi = result["information_regression"]["model_gap_ci95"]
        assert lo < 0 < hi  # nothing beyond what the line already knows
        assert "power_ratings vs closing market" in to_markdown(result)

    def test_elo_is_walk_forward_and_sane(self, market):
        preds = elo_walkforward(market)
        first_week = market["week"] == market["week"].min()
        # Before any game is played, every team sits at the same rating: HFA only
        assert preds[first_week & (market["season"] == market["season"].min())].nunique() == 1
        assert np.corrcoef(preds.dropna(), market.loc[preds.notna(), "spread_close"])[0, 1] > 0.5


class TestGuardrails:
    def test_leaked_predictions_are_refused(self, evaluated):
        # Predicting the final margin is perfect hindsight; the benchmark must not report it
        leaky = evaluated.assign(pred_home_margin=evaluated["home_margin"].astype(float))
        with pytest.raises(ImplausibleResultError):
            run_benchmark(leaky, "leaky")

    def test_the_dec_2025_claim_would_be_refused(self):
        with pytest.raises(ImplausibleResultError, match="74.1%"):
            check_plausibility({"bets": 1477, "win_rate": 0.741, "roi": 0.41})

    def test_small_samples_are_not_judged(self):
        assert check_plausibility({"bets": 40, "win_rate": 0.70, "roi": 0.30}) == []

    def test_sportsbook_signed_predictions_are_rejected(self, evaluated):
        flipped = evaluated.assign(pred_home_margin=-evaluated["pred_home_margin"])
        with pytest.raises(ValueError, match="sportsbook sign"):
            run_benchmark(flipped, "flipped")

    def test_buggy_cover_grading_fails_the_base_rate_check(self, market):
        # Simulate a market frame whose lines were stored in sportsbook sign
        flipped = market.assign(spread_close=-market["spread_close"])
        with pytest.raises((ImplausibleResultError, ValueError)):
            check_home_cover_rate(flipped)
