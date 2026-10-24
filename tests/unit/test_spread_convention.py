"""
Pin the nflverse spread sign convention (positive spread_close = home favored).

The sportsbook rule ``margin + spread_close > 0`` applied to nflverse data produced the
"74.9% ATS / +43% ROI" power-ratings claim (Dec 2025). These tests use real games from
data/historical_weather_2018_2024.csv so a regression to that rule fails loudly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_analytics.utils.spread_convention import (
    assert_nflverse_convention,
    book_home_line,
    home_cover,
    home_cover_margin,
    implied_home_win_prob,
    model_edge,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "historical_weather_2018_2024.csv"


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"{DATA} not available")
    df = pd.read_csv(
        DATA,
        usecols=[
            "game_id",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_close",
        ],
    )
    df = df.dropna().drop_duplicates("game_id")
    df["home_margin"] = df["home_score"] - df["away_score"]
    return df


def _game(games: pd.DataFrame, game_id: str) -> pd.Series:
    return games.loc[games["game_id"] == game_id].iloc[0]


class TestKnownGames:
    def test_bal_favored_and_covered(self, games):
        # BAL (home) -7.5 vs BUF, won 47-3
        g = _game(games, "2018_01_BUF_BAL")
        assert g["spread_close"] == 7.5
        assert home_cover(g["home_margin"], g["spread_close"]) == 1.0
        assert book_home_line(g["spread_close"]) == -7.5

    def test_chi_favored_and_lost(self, games):
        # CHI (home) -3.5 vs GB, lost 3-10
        g = _game(games, "2019_01_GB_CHI")
        assert g["spread_close"] == 3.5
        assert g["home_margin"] == -7
        assert home_cover(g["home_margin"], g["spread_close"]) == 0.0

    def test_favorite_wins_but_fails_to_cover(self, games):
        # NO (home) -10 vs CLE, won 21-18: no cover. The sportsbook rule
        # (margin + spread_close > 0) wrongly grades this as a home cover.
        g = _game(games, "2018_02_CLE_NO")
        assert g["spread_close"] == 10.0
        assert g["home_margin"] == 3
        assert home_cover(g["home_margin"], g["spread_close"]) == 0.0
        assert g["home_margin"] + g["spread_close"] > 0


class TestHelpers:
    def test_push_is_nan(self):
        assert np.isnan(home_cover(3, 3.0))

    def test_series_index_preserved(self):
        s = home_cover(
            pd.Series([10, -3, 7], index=[5, 6, 7]), pd.Series([7.0, -3.0, 3.0], index=[5, 6, 7])
        )
        assert list(s.index) == [5, 6, 7]
        assert s.iloc[0] == 1.0 and np.isnan(s.iloc[1]) and s.iloc[2] == 1.0

    def test_missing_line_is_nan(self):
        assert np.isnan(home_cover(7, np.nan))

    def test_edge_and_margin_signs(self):
        assert home_cover_margin(10, 7.0) == 3.0
        assert model_edge(9.0, 7.0) == 2.0  # model likes home by 2 more than the market

    def test_implied_prob_increases_with_home_favoritism(self):
        p = implied_home_win_prob([-7.0, 0.0, 7.0])
        assert p[0] < 0.5 < p[2]
        assert p[1] == pytest.approx(0.5)


class TestPopulation:
    def test_pooled_home_cover_rate_is_near_half(self, games):
        cover = home_cover(games["home_margin"], games["spread_close"]).dropna()
        assert 0.46 <= cover.mean() <= 0.53

    def test_sportsbook_rule_is_implausible(self, games):
        buggy = games["home_margin"] + games["spread_close"]
        rate = (buggy[buggy != 0] > 0).mean()
        assert rate > 0.54  # documents why the old rule inflated results

    def test_convention_check_passes_on_real_data(self, games):
        assert_nflverse_convention(games)

    def test_convention_check_rejects_flipped_data(self, games):
        flipped = games.assign(spread_close=-games["spread_close"])
        with pytest.raises(ValueError, match="nflverse"):
            assert_nflverse_convention(flipped)

    def test_prior_season_power_ratings_have_no_edge(self, games):
        """Prior-season point differential + 2.5 HFA is ~50% ATS once graded correctly."""
        team_games = pd.concat(
            [
                games[["season", "home_team", "home_margin"]].set_axis(
                    ["season", "team", "margin"], axis=1
                ),
                games[["season", "away_team", "home_margin"]]
                .set_axis(["season", "team", "margin"], axis=1)
                .assign(margin=lambda d: -d["margin"]),
            ]
        )
        ratings = team_games.groupby(["season", "team"])["margin"].mean().reset_index()
        ratings["season"] += 1
        x = games.merge(
            ratings.rename(columns={"team": "home_team", "margin": "r_home"}),
            on=["season", "home_team"],
        ).merge(
            ratings.rename(columns={"team": "away_team", "margin": "r_away"}),
            on=["season", "away_team"],
        )
        pred = x["r_home"] - x["r_away"] + 2.5
        edge = model_edge(pred, x["spread_close"])
        cover = home_cover(x["home_margin"], x["spread_close"])
        bets = (edge.abs() >= 1) & cover.notna()
        ats = (np.where(edge[bets] > 0, cover[bets], 1 - cover[bets])).mean()
        assert bets.sum() > 1000
        assert 0.45 <= ats <= 0.55
