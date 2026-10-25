"""
Market benchmark tests on real games (data/historical_weather_2018_2024.csv, nflverse lines).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nfl_analytics.evaluation.market_benchmark import (
    ImplausibleResultError,
    baseline_predictions,
    check_home_cover_rate,
    check_market_sanity,
    check_plausibility,
    elo_walkforward,
    load_market_from_csv,
    market_baseline,
    market_sanity,
    power_ratings_walkforward,
    run_benchmark,
    to_json,
    to_markdown,
)
from nfl_analytics.evaluation.market_benchmark import __main__ as cli
from nfl_analytics.evaluation.market_benchmark.baselines import HFA_POINTS
from nfl_analytics.evaluation.market_benchmark.metrics import (
    block_bootstrap_roi,
    closing_line_value,
    grade_ats,
    information_regression,
    spread_accuracy,
    wilson_interval,
)
from nfl_analytics.evaluation.market_benchmark.report import season_span
from nfl_analytics.evaluation.market_benchmark.odds import (
    american_to_decimal,
    breakeven_win_rate,
    devig_proportional,
    devig_shin,
)

DATA = Path(__file__).resolve().parents[3] / "data" / "historical_weather_2018_2024.csv"


def _strict_json(text: str):
    """json.loads that fails on NaN / Infinity tokens instead of accepting them."""

    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


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

    def test_diebold_mariano_needs_variance(self, market, evaluated):
        # The line against itself: the same error in every game, nothing to test
        same = market.assign(pred_home_margin=market["spread_close"])
        assert spread_accuracy(same)["dm_p_value_two_sided"] == 1.0
        # One game (MIN 28-12 ATL, line 3.5, power ratings 3.67) has no variance to test
        # against; this used to report p = 0.0
        one = evaluated[evaluated["game_id"] == "2019_01_ATL_MIN"]
        assert spread_accuracy(one)["model_mae"] != spread_accuracy(one)["market_mae"]
        assert np.isnan(spread_accuracy(one)["dm_p_value_two_sided"])
        # Line + 0.5 on the 2024 games the home team won by more than the line + 0.5: the
        # prediction is exactly half a point closer in all of them, a constant difference (p = 0.0
        # before, "maximal significance" for a differential with no variance)
        won = market[
            (market["season"] == 2024) & (market["home_margin"] - market["spread_close"] > 0.5)
        ]
        s = spread_accuracy(won.assign(pred_home_margin=won["spread_close"] + 0.5))
        assert s["games"] == 127 and s["market_mae"] - s["model_mae"] == pytest.approx(0.5)
        assert np.isnan(s["dm_p_value_two_sided"])
        # Rendered as n/a, not as p = 0.000
        assert "| Spread MAE: model / market | 12.33 / 12.50 | DM p = n/a |" in to_markdown(
            run_benchmark(one, "power_ratings, one game")
        )


class TestInformationRegression:
    def test_power_ratings_have_a_finite_interval(self, evaluated):
        reg = information_regression(evaluated)
        assert reg["note"] is None
        assert reg["model_gap_ci95"][1] - reg["model_gap_ci95"][0] < 1

    def test_interval_uses_the_t_reference_of_its_p_value(self, evaluated):
        # 2019 weeks 1-4 are 4 week clusters: the p-value uses t(3), so the interval must too.
        # With +/-1.96 SE it was [-0.528, -0.070], excluding 0 next to p = 0.083.
        early = evaluated[(evaluated["season"] == 2019) & (evaluated["week"] <= 4)]
        reg = information_regression(early)
        lo, hi = reg["model_gap_ci95"]
        assert reg["clusters"] == 4
        assert (hi - lo) / 2 == pytest.approx(stats.t.ppf(0.975, 3) * reg["model_gap_se"])
        assert reg["model_gap_p_value"] == pytest.approx(0.083, abs=1e-3)
        assert lo < 0 < hi
        assert (round(lo, 3), round(hi, 3)) == (-0.671, 0.073)

    def test_the_line_itself_is_not_identified(self, market):
        reg = information_regression(market.assign(pred_home_margin=market_baseline(market)))
        assert np.isnan(reg["model_gap_coef"]) and np.isnan(reg["model_gap_ci95"][0])
        assert "constant" in reg["note"]
        shifted = market.assign(pred_home_margin=market["spread_close"] + 1)  # line + 1 point
        assert "constant" in information_regression(shifted)["note"]

    def test_constant_prediction_is_rank_deficient_not_exploding(self, evaluated):
        # Home by HFA_POINTS in every game: pred - line = 2.5 - line is collinear with the line.
        # Before the rank check this returned a CI of +/-6.1e6 (SE 3.1e6) on these games.
        flat = evaluated.assign(pred_home_margin=HFA_POINTS)
        reg = information_regression(flat)
        assert np.isnan(reg["model_gap_coef"]) and np.isnan(reg["model_gap_se"])
        assert "linear function of the line" in reg["note"]
        result = run_benchmark(flat, "home by 2.5")
        assert any("constant" in w for w in result["warnings"])
        assert "not identified" in to_markdown(result)


class TestBaselines:
    def test_market_baseline_finds_no_edge(self, market):
        df = market.assign(pred_home_margin=market_baseline(market))
        assert grade_ats(df, min_edge=0.5).empty
        assert "not identified" in information_regression(df)["note"]

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

    def test_weak_but_correctly_signed_model_gets_a_warning_not_a_refusal(self, market):
        # A rest-advantage model: home by HFA_POINTS plus a point per extra day of rest
        # (real home_rest / away_rest). It correlates only +0.04 with the closing line.
        raw = pd.read_csv(DATA).set_index("game_id")
        rest = market["game_id"].map(raw["home_rest"] - raw["away_rest"])
        df = market.assign(pred_home_margin=HFA_POINTS + rest)
        df = df[df["season"] >= 2019]
        result = run_benchmark(df, "rest advantage")
        assert 0 < result["pred_line_corr"] < 0.2
        assert any("correlates only" in w for w in result["warnings"])
        assert "Warnings:" in to_markdown(result)

    def test_the_bundled_market_passes_the_market_checks(self, market):
        s = check_market_sanity(market[market["season"] >= 2019])
        assert (s["home_covers"], s["decided_games"]) == (756, 1557)  # 48.6%
        assert s["home_cover_p_value"] > 0.5 and s["line_margin_corr"] > 0.4
        assert check_home_cover_rate(market) == pytest.approx(875 / 1804)  # 2018-2024

    def test_buggy_cover_grading_fails_the_base_rate_check(self, market):
        # Derived from the real rows: spread_close negated, i.e. the same lines in sportsbook
        # sign (grading them with the nflverse rule is the old margin-plus-line bug)
        flipped = market.assign(spread_close=-market["spread_close"])
        with pytest.raises(ImplausibleResultError) as err:
            check_home_cover_rate(flipped)
        assert "sportsbook sign" in str(err.value)  # line sign check
        assert "binomial p" in str(err.value)  # 1042/1828 home covers, p ~ 1e-11

    def test_correct_sign_wrong_covers_is_refused_by_the_binomial_test(self, market):
        # Derived from the real rows: home-field advantage added a second time to lines that
        # already include it. The sign is still right, so only the cover base rate catches it.
        doubled = market.assign(spread_close=market["spread_close"] + HFA_POINTS)
        s = market_sanity(doubled)
        assert s["line_margin_corr"] > 0.4 and s["home_cover_rate"] < 0.45
        with pytest.raises(ImplausibleResultError, match="binomial p") as err:
            check_market_sanity(doubled)
        assert "correlates" not in str(err.value)  # the line-sign check passed


class TestReport:
    def test_json_is_strict(self, market):
        # The line against itself places no bets: win rate, ROI and the regression are NaN
        df = market.assign(pred_home_margin=market_baseline(market))
        result = run_benchmark(df, "market")
        text = to_json(result)
        assert "NaN" not in text and "Infinity" not in text
        parsed = _strict_json(text)
        assert parsed["ats"]["win_rate"] is None
        assert parsed["information_regression"]["model_gap_coef"] is None
        assert parsed["market_sanity"]["decided_games"] == 1804
        extremes = {"inf": float("inf"), "neg": np.float64("-inf"), "n": np.int64(3)}
        assert _strict_json(to_json(extremes)) == {"inf": None, "neg": None, "n": 3}

    def test_season_span(self):
        assert season_span(range(2019, 2025)) == "2019-2024"
        assert season_span([2025]) == "2025"
        assert season_span([2024, 2019]) == "2019, 2024"
        assert season_span([2019, 2021, 2022, 2023]) == "2019, 2021-2023"

    def test_readme_baseline_rows_reproduce(self, capsys):
        if not DATA.exists():
            pytest.skip(f"{DATA} not available")
        # README.md "Results, honestly": the two baseline rows, through the documented CLI
        expected = {
            "power_ratings": ["| ATS record | 609-608-31 |", "| ROI | -4.5% | -8.9% to +0.0% |"],
            "elo": ["| ATS record | 581-596-31 |", "| ROI | -5.8% | -11.5% to -0.4% |"],
        }
        rates = {"power_ratings": "| Win rate | 50.0% |", "elo": "| Win rate | 49.4% |"}
        for name, rows in expected.items():
            argv = ["--baseline", name, "--market-csv", str(DATA), "--seasons", "2019-2024"]
            assert cli.main(argv) == 0
            out = capsys.readouterr().out
            for row in rows + [rates[name]]:
                assert row in out, (name, row)


class TestCli:
    def test_db_baselines_warm_up_on_the_prior_season(
        self, evaluated, monkeypatch, capsys, tmp_path
    ):
        requested = []

        class _Conn:
            def close(self):
                pass

        def fake_db(seasons, conn):  # the real games, as devdb01 would return them
            requested.append(list(seasons))
            return load_market_from_csv(DATA, seasons)

        monkeypatch.setattr(cli, "load_market_from_db", fake_db)
        monkeypatch.setattr("nfl_analytics.utils.db_config.get_connection", lambda: _Conn())
        assert cli.main(["--baseline", "elo", "--db", "--seasons", "2019-2024"]) == 0
        db_out = capsys.readouterr().out
        assert requested == [list(range(2018, 2025))]  # 2018 seeds the ratings
        csv_args = ["--baseline", "elo", "--market-csv", str(DATA), "--seasons", "2019-2024"]
        assert cli.main(csv_args) == 0
        assert capsys.readouterr().out == db_out
        assert "| ATS record | 581-596-31 |" in db_out

        # A model's predictions need no warm-up: only the requested season is loaded
        path = tmp_path / "power_2024.csv"
        evaluated.loc[evaluated["season"] == 2024, ["game_id", "pred_home_margin"]].to_csv(
            path, index=False
        )
        assert cli.main(["--predictions-csv", str(path), "--db", "--seasons", "2024"]) == 0
        assert requested[-1] == [2024]
        assert "Season 2024, 272 games" in capsys.readouterr().out

        # Seasons with a gap: the baseline still runs through the seasons in between
        assert cli.main(["--baseline", "elo", "--db", "--seasons", "2019,2024"]) == 0
        assert requested[-1] == list(range(2018, 2025))
        capsys.readouterr()

    def test_baselines_run_through_a_gap_in_the_seasons(self, capsys):
        # --seasons 2019,2024 grades those two seasons, with 2020-2023 feeding the ratings in
        # between. Before, only 2018, 2019 and 2024 were loaded: all 272 Elo predictions for
        # 2024 differed (by up to 13.1 points) and the header read "Seasons 2019-2024".
        argv = ["--baseline", "elo", "--market-csv", str(DATA), "--seasons", "2019,2024"]
        assert cli.main(argv + ["--json"]) == 0
        got = _strict_json(capsys.readouterr().out)
        full = load_market_from_csv(DATA, range(2018, 2025))
        full = full.join(baseline_predictions("elo", full))
        want = run_benchmark(full[full["season"].isin([2019, 2024])], "elo")
        assert got["seasons"] == [2019, 2024] and got["games"] == want["games"] == 528
        for key in ("wins", "losses", "pushes"):
            assert got["ats"][key] == want["ats"][key]
        assert got["spread_accuracy"]["model_mae"] == pytest.approx(
            want["spread_accuracy"]["model_mae"], abs=1e-12
        )
        assert cli.main(argv) == 0
        assert "Seasons 2019, 2024, 528 games" in capsys.readouterr().out

    def test_sources_are_mutually_exclusive(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--baseline", "elo", "--db", "--market-csv", str(DATA), "--seasons", "2024"])
        with pytest.raises(SystemExit):  # devdb01 grades the regular season only
            cli.main(["--baseline", "elo", "--db", "--seasons", "2024", "--game-types", "REG,WC"])
        capsys.readouterr()
        with pytest.raises(SystemExit):  # the bundled export has no game_type column
            argv = ["--baseline", "elo", "--market-csv", str(DATA), "--seasons", "2024"]
            cli.main(argv + ["--game-types", "REG,WC"])
        assert "needs a game_type column" in capsys.readouterr().err
