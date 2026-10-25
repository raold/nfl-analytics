"""
nflverse schedule adapter, the 2026 holdout, predict mode and closing-line value, on real rows
of nflverse games.csv (tests/fixtures/nflverse/README.md has the source and download time).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_analytics.evaluation.market_benchmark import (
    ImplausibleResultError,
    baseline_params,
    baseline_predictions,
    check_market_sanity,
    load_market_from_csv,
    load_market_from_nflverse,
    market_sanity,
    predict_week,
    read_nflverse_games,
    run_benchmark,
    to_markdown,
    write_predictions,
)
from nfl_analytics.evaluation.market_benchmark import __main__ as cli
from nfl_analytics.evaluation.market_benchmark import predict as predict_mod
from nfl_analytics.evaluation.market_benchmark.market import MARKET_COLUMNS
from nfl_analytics.evaluation.market_benchmark.metrics import closing_line_value, win_prob_skill
from nfl_analytics.evaluation.market_benchmark.predict import OUTPUT_COLUMNS

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nflverse"
GAMES = FIXTURES / "games_2025_2026w8.csv"  # 2025 season + 2026 weeks 1-8 (+ one 1999 game)
LINES_20251207 = FIXTURES / "lines_2025w15_20251207.csv"  # 2025 week 15 lines on 2025-12-07
DOWNLOADED = pd.Timestamp("2026-10-24 20:54:34", tz="UTC")  # games.csv download (16:54 EDT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return read_nflverse_games(GAMES)


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    """Every 2026 game final at download: weeks 1-6 and the week 7 Thursday game."""
    return load_market_from_nflverse(GAMES, seasons=[2026])


def _first_kickoff(games: pd.DataFrame, season: int, week: int) -> pd.Timestamp:
    return games.loc[(games["season"] == season) & (games["week"] == week), "kickoff"].min()


class TestAdapter:
    def test_kickoff_is_eastern_wall_clock_converted_to_utc(self, games):
        kick = games.set_index("game_id")["kickoff"]
        assert kick["2026_01_NE_SEA"] == pd.Timestamp("2026-09-10 00:20", tz="UTC")  # 20:20 EDT
        assert kick["2026_08_CAR_GB"] == pd.Timestamp("2026-10-30 00:15", tz="UTC")  # 20:15 EDT
        assert kick["2026_08_BAL_BUF"] == pd.Timestamp("2026-11-01 18:00", tz="UTC")  # 13:00 EST
        assert pd.isna(kick["1999_01_MIN_ATL"])  # nflverse has no gametime for 1999

    def test_lines_keep_the_nflverse_sign(self, games):
        row = games.set_index("game_id").loc["2026_01_NE_SEA"]  # SEA (home) favored by 3
        assert (row["spread_close"], row["total_close"]) == (3.0, 44.5)
        assert row["home_moneyline"] < 0 < row["away_moneyline"]

    def test_only_completed_regular_season_games_are_graded_by_default(self, games):
        market = load_market_from_nflverse(GAMES)
        raw = pd.read_csv(GAMES)
        done = raw[(raw["game_type"] == "REG") & raw["home_score"].notna()]
        assert sorted(market["game_id"]) == sorted(done["game_id"])
        assert int((market["season"] == 2026).sum()) == 94  # no week 7-8 game still to play
        everything = load_market_from_nflverse(GAMES, seasons=[2025], game_types=None)
        assert len(everything) == 285
        assert set(everything["game_type"]) == {"REG", "WC", "DIV", "CON", "SB"}

    def test_rejects_a_file_that_is_not_an_nflverse_schedule(self):
        export = Path(__file__).resolve().parents[3] / "data" / "historical_weather_2018_2024.csv"
        if not export.exists():
            pytest.skip(f"{export} not available")
        with pytest.raises(ValueError, match="not an nflverse schedule"):
            read_nflverse_games(export)


@pytest.fixture(scope="module")
def export(games, tmp_path_factory) -> Path:
    """The fixture's rows as a CSV export of the games table, game_type included (devdb01's
    game_type, scores and lines were repaired from this nflverse file)."""
    path = tmp_path_factory.mktemp("export") / "games.csv"
    games[MARKET_COLUMNS + ["game_type"]].to_csv(path, index=False)
    return path


class TestCsvExport:
    def test_regular_season_only_by_default(self, export):
        assert len(load_market_from_csv(export, [2025])) == 272
        everything = load_market_from_csv(export, [2025], game_types=None)
        assert len(everything) == 285
        assert set(everything["game_type"]) == {"REG", "WC", "DIV", "CON", "SB"}

    def test_an_export_grades_like_the_nflverse_file(self, export, capsys):
        # The export used to keep every game type, so the 2025 playoffs fed the Elo warm-up for
        # 2026 from an export but not from nflverse or devdb01: 87 of the 94 predictions differed
        reg = load_market_from_csv(export, [2025, 2026])
        allg = load_market_from_csv(export, [2025, 2026], game_types=None)
        pred = {
            k: m.join(baseline_predictions("elo", m)).set_index("game_id")["pred_home_margin"]
            for k, m in (("reg", reg), ("all", allg))
        }
        ids = reg.loc[reg["season"] == 2026, "game_id"]
        assert len(ids) == 94 and int((pred["reg"][ids] != pred["all"][ids]).sum()) == 87
        argv = ["--baseline", "elo", "--seasons", "2026"]
        assert cli.main(argv + ["--market-csv", str(export)]) == 0
        from_export = capsys.readouterr().out
        assert cli.main(argv + ["--nflverse-csv", str(GAMES)]) == 0
        assert from_export == capsys.readouterr().out
        # --game-types applies to an export that has a game_type column
        argv = ["--baseline", "elo", "--market-csv", str(export), "--seasons", "2025", "--json"]
        assert cli.main(argv + ["--game-types", "REG,WC,DIV,CON,SB"]) == 0
        assert json.loads(capsys.readouterr().out)["games"] == 285


class TestHoldoutGuardrails:
    def test_2026_holdout_passes(self, holdout):
        s = check_market_sanity(holdout)
        # 45.5% home covers used to trip the fixed 46-53% band; at n = 88 that is p = 0.52
        assert (s["home_covers"], s["decided_games"]) == (40, 88)
        assert s["home_cover_p_value"] == pytest.approx(0.524, abs=1e-3)
        assert s["sign_checked"] and s["line_margin_corr"] > 0.3

    def test_2026_holdout_in_sportsbook_sign_is_refused(self, holdout):
        # Derived from the real rows above: spread_close negated (the same lines in sportsbook
        # sign). The cover rate alone cannot tell at this sample size; the correlation can.
        flipped = holdout.assign(spread_close=-holdout["spread_close"])
        s = market_sanity(flipped)
        assert s["home_cover_p_value"] > 0.05 and s["line_margin_corr"] < -0.3
        with pytest.raises(ImplausibleResultError, match="sportsbook sign"):
            check_market_sanity(flipped)

    def test_one_week_is_judged_only_on_what_it_can_show(self, holdout):
        # 2026 week 5: 4/14 home covers (28.6%) failed the old fixed band; at n = 14 it is
        # p = 0.18. With 15 games the line-sign check (30 games) is skipped with a warning.
        week5 = holdout[holdout["week"] == 5]
        s = check_market_sanity(week5)
        assert (s["home_covers"], s["decided_games"]) == (4, 14) and not s["sign_checked"]
        assert s["home_cover_p_value"] == pytest.approx(0.181, abs=1e-3)
        week5 = week5.assign(pred_home_margin=week5["spread_close"] + 0.5)
        result = run_benchmark(week5, "line + 0.5")
        assert any("line-sign check" in w for w in result["warnings"])

    def test_grading_cli_reports_the_holdout(self, capsys):
        argv = ["--baseline", "elo", "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "Season 2026, 94 games" in out
        assert "| Pooled home cover rate | 45.5% | 40/88, binomial p = 0.524 vs 49% |" in out
        assert "| Log-loss: model / market |" in out  # Elo's win probability vs the moneyline
        assert cli.main(argv + ["--json"]) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["ats"]["bets"] + parsed["ats"]["pushes"] > 0
        assert parsed["win_prob_skill"]["games"] == 93  # ties excluded: 2026_05_NYG_WAS 27-27


class TestPredict:
    @pytest.mark.parametrize("baseline", ["elo", "power_ratings"])
    def test_predictions_equal_the_walk_forward_grading(self, games, baseline):
        # Just before 2026 week 5 kicks off, predict mode must give exactly the walk-forward
        # values the grading path computes for those games afterwards
        as_of = _first_kickoff(games, 2026, 5) - pd.Timedelta(minutes=1)
        preds = predict_week(games, baseline, 2026, 5, as_of=as_of).set_index("game_id")
        market = load_market_from_nflverse(GAMES, seasons=[2025, 2026])
        graded = market.join(baseline_predictions(baseline, market)).set_index("game_id")
        assert len(preds) == 15
        cols = ["pred_home_margin"] + (["pred_home_win_prob"] if baseline == "elo" else [])
        for col in cols:
            np.testing.assert_allclose(preds[col], graded.loc[preds.index, col], rtol=0, atol=1e-12)

    def test_only_games_not_started_are_written(self, games):
        week7 = predict_week(games, "elo", 2026, 7, as_of=DOWNLOADED)
        raw = pd.read_csv(GAMES)
        played = raw.loc[(raw["season"] == 2026) & (raw["week"] == 7) & raw["home_score"].notna()]
        assert len(played) == 1 and week7.attrs["skipped"] == list(played["game_id"])
        assert len(week7) == 13 and played["game_id"].iloc[0] not in set(week7["game_id"])
        kick = pd.to_datetime(week7["kickoff_utc"], utc=True)
        assert (kick > DOWNLOADED).all()
        assert week7.attrs["fit_games"] == 272 + 93  # 2025 regular season + 2026 weeks 1-6

    def test_output_records_line_time_baseline_and_constants(self, games):
        week8 = predict_week(games, "power_ratings", 2026, 8, as_of=DOWNLOADED)
        assert len(week8) == 14 and "pred_home_win_prob" not in week8  # no probability model
        lines = pd.read_csv(GAMES).set_index("game_id")["spread_line"]
        assert (week8["spread_at_prediction"] == week8["game_id"].map(lines)).all()
        assert set(week8["baseline"]) == {"power_ratings"}
        assert json.loads(week8["baseline_params"].iloc[0]) == baseline_params("power_ratings")
        made = pd.to_datetime(week8["prediction_timestamp"], utc=True)
        assert (made <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=1)).all()

    def test_cli_writes_hashes_and_never_overwrites(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(predict_mod, "_utc_now", lambda: DOWNLOADED)
        out = tmp_path / "preds_2026_w08.csv"
        argv = ["predict", "--baseline", "elo", "--nflverse-csv", str(GAMES)]
        argv += ["--season", "2026", "--week", "8", "--out", str(out)]
        assert cli.main(argv) == 0
        printed = capsys.readouterr().out
        assert f"output sha256 {_sha256(out)}" in printed
        assert f"input  sha256 {_sha256(GAMES)}" in printed
        written = pd.read_csv(out)
        assert list(written.columns) == OUTPUT_COLUMNS
        assert len(written) == 14 and set(written["input_sha256"]) == {_sha256(GAMES)}
        assert set(written["prediction_timestamp"]) == {"2026-10-24T20:54:34+00:00"}
        with pytest.raises(SystemExit):  # a prediction record is not silently replaced
            cli.main(argv)

    def test_a_week_already_played_writes_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(predict_mod, "_utc_now", lambda: DOWNLOADED)
        out = tmp_path / "w05.csv"
        argv = ["predict", "--baseline", "elo", "--nflverse-csv", str(GAMES)]
        assert cli.main(argv + ["--season", "2026", "--week", "5", "--out", str(out)]) == 1
        assert not out.exists() and "nothing written" in capsys.readouterr().err


class TestClosingLineValue:
    def test_sign_on_real_line_moves(self):
        # nflverse lines for two real games: on 2025-12-07 (lines file) and at the close
        early = pd.read_csv(LINES_20251207).set_index("game_id")["spread_line_20251207"]
        close = load_market_from_nflverse(GAMES, seasons=[2025]).set_index("game_id")
        close = close["spread_close"]
        assert (early["2025_15_LAC_KC"], close["2025_15_LAC_KC"]) == (4.5, 5.5)  # KC -4.5 -> -5.5
        assert (early["2025_15_BUF_NE"], close["2025_15_BUF_NE"]) == (
            1.5,
            -2.5,
        )  # NE -1.5 -> BUF -2.5
        games = ["2025_15_LAC_KC"] * 2 + ["2025_15_BUF_NE"] * 2
        bets = pd.DataFrame(
            {
                "side": ["home", "away", "home", "away"],
                "spread_close": close[games].to_numpy(),
                "spread_at_bet": early[games].to_numpy(),
            }
        )
        # KC backed at 4.5 closed at 5.5: +1 (LAC +4.5 instead of +5.5: -1).
        # BUF backed at +1.5 closed at -2.5: +4 (NE -1.5 instead of +2.5: -4).
        assert list(closing_line_value(bets)) == [1.0, -1.0, -4.0, 4.0]

    def test_report_gives_clv_for_bets_chosen_at_the_line_when_predicted(self):
        # Predictions that equal the eventual closing line, made on 2025-12-07 at that day's
        # line: every bet is on the side the line later moved to, so each one gains exactly
        # the move and the report must show positive CLV on every bet
        market = load_market_from_nflverse(GAMES, seasons=[2025])
        early = pd.read_csv(LINES_20251207)
        df = market.merge(early.rename(columns={"spread_line_20251207": "spread_at_prediction"}))
        df = df.assign(
            pred_home_margin=df["spread_close"],
            prediction_timestamp="2025-12-07T20:42:32+00:00",  # time of the backup holding them
        )
        result = run_benchmark(df, "closing line known on 2025-12-07")
        move = (df["spread_close"] - df["spread_at_prediction"]).abs()
        clv = result["closing_line_value"]
        assert clv["bets"] == int((move >= 1).sum()) == 9
        assert clv["mean_points"] == pytest.approx(move[move >= 1].mean())
        assert clv["share_positive"] == 1.0
        assert np.isnan(clv["mean_points_ci95"][0])  # one week: no week-block interval
        assert result["retroactive_share"] == 0.0  # every week 15 game kicked off later
        assert result["ats"]["bets"] == 0  # no edge against the close itself
        assert "Closing-line value" in to_markdown(result)

    def test_predictions_file_round_trip(self, games, tmp_path, capsys):
        # Week 5 predicted as of just before its first kickoff, but written now: graded
        # afterwards they are all retroactive, and with closing lines as the line at
        # prediction their closing-line value is exactly zero
        as_of = _first_kickoff(games, 2026, 5) - pd.Timedelta(minutes=1)
        path = tmp_path / "w05.csv"
        write_predictions(predict_week(games, "elo", 2026, 5, as_of=as_of), path, _sha256(GAMES))
        argv = ["--predictions-csv", str(path), "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        assert cli.main(argv + ["--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["games"] == 15 and result["retroactive_share"] == 1.0
        assert result["closing_line_value"]["bets"] > 0
        assert result["closing_line_value"]["mean_points"] == 0
        assert result["win_prob_skill"]["games"] > 0
        assert cli.main(argv) == 0
        assert "not a prospective test" in capsys.readouterr().out

    def test_grading_before_the_games_are_played_says_so(self, games, tmp_path, capsys):
        path = tmp_path / "w08.csv"
        write_predictions(predict_week(games, "elo", 2026, 8, as_of=DOWNLOADED), path, "x")
        argv = ["--predictions-csv", str(path), "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        assert cli.main(argv) == 1
        assert "None of the 14 games" in capsys.readouterr().err

    def test_duplicate_predictions_are_refused(self, games, tmp_path):
        as_of = _first_kickoff(games, 2026, 5) - pd.Timedelta(minutes=1)
        preds = predict_week(games, "elo", 2026, 5, as_of=as_of)
        path = tmp_path / "twice.csv"
        write_predictions(pd.concat([preds, preds.head(1)]), path, "x")
        argv = ["--predictions-csv", str(path), "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        with pytest.raises(ValueError, match="more than one prediction"):
            cli.main(argv)


class TestSmallSamples:
    """Prospective files are graded a week, or a game, at a time."""

    def test_one_week_has_no_log_loss_interval(self, games, tmp_path, capsys):
        # One week resampled by week blocks is always itself: this printed the zero-width
        # interval "diff -0.0304 to -0.0304", which excludes 0
        as_of = _first_kickoff(games, 2026, 5) - pd.Timedelta(minutes=1)
        path = tmp_path / "w05.csv"
        write_predictions(predict_week(games, "elo", 2026, 5, as_of=as_of), path, _sha256(GAMES))
        argv = ["--predictions-csv", str(path), "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        assert cli.main(argv + ["--json"]) == 0
        wp = json.loads(capsys.readouterr().out)["win_prob_skill"]
        # 15 games, less the 27-27 tie 2026_05_NYG_WAS
        assert wp["games"] == 14 and wp["log_loss_diff_ci95"] == [None, None]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "| Log-loss: model / market | 0.6198 / 0.6502 | n/a (one week) |" in out

    def test_holdout_log_loss_interval_is_unchanged(self):
        # 7 weeks: the same week-block bootstrap as before, the figures in the final evaluation
        market = load_market_from_nflverse(GAMES, seasons=[2025, 2026])
        graded = market.join(baseline_predictions("elo", market))
        wp = win_prob_skill(graded[graded["season"] == 2026])
        lo, hi = wp["log_loss_diff_ci95"]
        assert lo < wp["model_log_loss"] - wp["market_log_loss"] < hi
        assert (round(lo, 4), round(hi, 4)) == (-0.0328, 0.0386)

    def test_one_game_has_no_diebold_mariano_test(self, games, tmp_path, capsys):
        # A week graded after its first final game: one loss difference has no variance, so
        # there is no test. This printed "DM p = 0.000" next to "ATS record 1-0-0".
        as_of = _first_kickoff(games, 2026, 6) - pd.Timedelta(minutes=1)
        preds = predict_week(games, "elo", 2026, 6, as_of=as_of)
        path = tmp_path / "one.csv"
        write_predictions(preds[preds["game_id"] == "2026_06_SEA_DEN"], path, _sha256(GAMES))
        argv = ["--predictions-csv", str(path), "--nflverse-csv", str(GAMES), "--seasons", "2026"]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert "| ATS record | 1-0-0 | |" in out
        assert "| Spread MAE: model / market | 3.13 / 5.50 | DM p = n/a |" in out
