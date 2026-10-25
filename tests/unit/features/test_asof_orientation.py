"""Team-side orientation and calendar handling of the as-of features, pinned to real games.

games.spread_close is the expected HOME margin (nflverse spread_line, positive = home
favored). A team's own line is +spread_close at home and -spread_close on the road, so
prev_spread_close = +3.5 means the team was a 3.5-point favorite in its previous game,
wherever it was played. Rest days are measured on the US/Eastern wall clock.

Rows are real 2023 games (games table = nflverse schedules; EPA sums from plays, rounded).
Kickoffs are the true UTC instants stored in games.kickoff.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_analytics.features import asof_features
from nfl_analytics.features.asof_features_unified import AsofFeatureExtractor
from nfl_analytics.features.kickoff_time import kickoff_wall_clock

# game_id, kickoff UTC, home, away, home_score, away_score, spread_close, total_close,
# (home epa_sum, plays), (away epa_sum, plays)
GAMES = [
    # KC 4-pt home favorite (Thu 20:20 ET), lost 20-21
    (
        "2023_01_DET_KC",
        "2023-09-08 00:20",
        "KC",
        "DET",
        20,
        21,
        4.0,
        53.0,
        (-6.4393, 81),
        (-3.9771, 87),
    ),
    # KC 3.5-pt road favorite at JAX (JAX home underdog: spread_close -3.5)
    (
        "2023_02_KC_JAX",
        "2023-09-17 17:00",
        "JAX",
        "KC",
        9,
        17,
        -3.5,
        50.5,
        (-9.3693, 84),
        (0.4837, 81),
    ),
    # KC 13-pt home favorite vs CHI
    (
        "2023_03_CHI_KC",
        "2023-09-24 20:25",
        "KC",
        "CHI",
        41,
        10,
        13.0,
        48.0,
        (15.7567, 92),
        (-17.8663, 68),
    ),
    # ATL 2.5-pt road favorite at TEN, Sun 13:00 EDT
    (
        "2023_08_ATL_TEN",
        "2023-10-29 17:00",
        "TEN",
        "ATL",
        28,
        23,
        -2.5,
        35.0,
        (0.6294, 89),
        (-1.2936, 87),
    ),
    # ATL 3.5-pt home favorite vs MIN, Sun 13:00 EST (DST ended that morning)
    (
        "2023_09_MIN_ATL",
        "2023-11-05 18:00",
        "ATL",
        "MIN",
        28,
        31,
        3.5,
        38.5,
        (-6.6380, 92),
        (-1.7912, 92),
    ),
]


def _team_rows() -> pd.DataFrame:
    """Team-game rows shaped like SQL_TEAM_BASE output."""
    rows = []
    for gid, kick, home, away, hs, as_, sc, tc, (h_epa, h_n), (a_epa, a_n) in GAMES:
        for team, is_home, pf, pa, epa, n in (
            (home, True, hs, as_, h_epa, h_n),
            (away, False, as_, hs, a_epa, a_n),
        ):
            rows.append(
                {
                    "game_id": gid,
                    "team": team,
                    "is_home": is_home,
                    "season": int(gid[:4]),
                    "week": int(gid[5:7]),
                    "kickoff": pd.Timestamp(kick, tz="UTC"),
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "spread_close": sc,
                    "total_close": tc,
                    "points_for": pf,
                    "points_against": pa,
                    "stadium": None,
                    "roof": "outdoors",
                    "surface": None,
                    "qb_id": None,
                    "qb_name": None,
                    "coach_name": None,
                    "margin": pf - pa,
                    "epa_sum": epa,
                    "plays": n,
                }
            )
    return pd.DataFrame(rows)


def _row(df: pd.DataFrame, game_id: str, team: str) -> pd.Series:
    match = df[(df["game_id"] == game_id) & (df["team"] == team)]
    assert len(match) == 1
    return match.iloc[0]


@pytest.fixture(params=["asof_features", "asof_features_unified"])
def history(request) -> pd.DataFrame:
    if request.param == "asof_features":
        return asof_features.compute_team_history(_team_rows())
    return AsofFeatureExtractor(feature_level="base").compute_team_history(_team_rows())


class TestPrevSpreadOrientation:
    def test_previous_home_favorite_is_positive(self, history):
        # KC week 2: previous game was a 4-point home favorite
        assert _row(history, "2023_02_KC_JAX", "KC")["prev_spread_close"] == pytest.approx(4.0)

    def test_previous_road_favorite_is_positive(self, history):
        # KC week 3: previous game was a 3.5-point road favorite (home-oriented value: -3.5)
        assert _row(history, "2023_03_CHI_KC", "KC")["prev_spread_close"] == pytest.approx(3.5)
        assert _row(history, "2023_09_MIN_ATL", "ATL")["prev_spread_close"] == pytest.approx(2.5)

    def test_first_game_has_no_previous_line(self, history):
        assert np.isnan(_row(history, "2023_02_KC_JAX", "JAX")["prev_spread_close"])

    def test_pivot_keeps_team_side(self):
        wide = asof_features.pivot_to_games(asof_features.compute_team_history(_team_rows()))
        game = wide[wide["game_id"] == "2023_03_CHI_KC"].iloc[0]
        assert game["spread_close"] == pytest.approx(13.0)  # game line stays home-oriented
        assert game["home_prev_spread_close"] == pytest.approx(3.5)


class TestRestDaysWallClock:
    def test_rest_across_dst_change_is_whole_days(self, history):
        # Sun 13:00 EDT -> Sun 13:00 EST is 7 days on the wall clock (7 days 1 hour elapsed)
        assert _row(history, "2023_09_MIN_ATL", "ATL")["rest_days"] == pytest.approx(7.0)

    def test_thursday_to_sunday_rest(self, history):
        # Thu 20:20 ET -> Sun 13:00 ET (9 days 16 h 40 min)
        expected = 9 + 16 / 24 + 40 / 1440
        assert _row(history, "2023_02_KC_JAX", "KC")["rest_days"] == pytest.approx(expected)


class TestKickoffWallClock:
    def test_utc_instants_become_eastern_wall_clock(self):
        kick = pd.Series(pd.to_datetime(["2023-09-08 00:20", "2023-11-05 18:00"], utc=True))
        local = kickoff_wall_clock(kick)
        assert list(local.dt.strftime("%Y-%m-%d %H:%M")) == ["2023-09-07 20:20", "2023-11-05 13:00"]
        assert list(local.dt.day_name()) == ["Thursday", "Sunday"]

    def test_naive_values_are_left_unchanged(self):
        kick = pd.Series(pd.to_datetime(["2023-09-07 20:20"]))
        assert kickoff_wall_clock(kick).iloc[0] == pd.Timestamp("2023-09-07 20:20")

    def test_travel_thursday_flag_uses_eastern_weekday(self, tmp_path):
        # DET@KC kicked off Thu 20:20 ET, which is Friday 00:20 UTC
        from nfl_analytics.features.enhanced_travel_features import EnhancedTravelFeatures

        df = pd.DataFrame(
            {
                "game_id": ["2023_01_DET_KC", "2023_02_KC_JAX"],
                "kickoff": pd.to_datetime(["2023-09-08 00:20", "2023-09-17 17:00"], utc=True),
            }
        )
        missing = str(tmp_path / "missing.csv")  # no stadium data needed for this flag
        travel = EnhancedTravelFeatures(missing, missing, missing)
        out = travel.calculate_thursday_night_features(df)
        assert out["is_thursday_night"].tolist() == [1, 0]
