# Market Benchmark of Baselines (2026-10-24)

Reproducible, database-free check of the retracted "74.9% ATS" claim and of what simple models
achieve against NFL closing lines, plus the tool changes that let the benchmark grade the 2026
holdout and run prospective tests. Data: `data/historical_weather_2018_2024.csv` (nflverse
closing spreads; 2018 seeds the preseason ratings). Command for each 2019-2024 section:

```bash
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline <name> \
    --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024
```

Moneylines are not in this CSV, so the log-loss comparison against the de-vigged market needs
the database or an nflverse schedule file (both sections further down).

## Options

| Option | What it does |
|---|---|
| `--market-csv PATH` | CSV export of `games` (the bundled file above) |
| `--db` | devdb01 `games` (DB_* environment variables) |
| `--nflverse-csv PATH` | nflverse `games.csv` from the [schedules release](https://github.com/nflverse/nflverse-data/releases/tag/schedules); `spread_line` becomes `spread_close`, `total_line` `total_close`, kickoff = `gameday` + `gametime` read as America/New_York and converted to UTC (NaT without `gametime`, i.e. 1999) |
| `--game-types REG,WC,...` | with `--nflverse-csv` only; default `REG`. Only completed games with a line are graded |
| `--predictions-csv PATH` | `game_id` plus `--pred-col` (expected home margin); optional `pred_home_win_prob`, `prediction_timestamp`, `spread_at_prediction`. More than one row per game is refused |
| `predict ...` | subcommand, see [Predict mode](#predict-mode) |

The three market sources are mutually exclusive. `--baseline` loads the season before the first
graded one for its preseason ratings from every source; until this change `--db` cold-started.
The Elo baseline now also carries its win probability (the Elo expectation
`1 / (1 + 10^(-diff/400))`), so Elo is graded on log-loss whenever the source has moneylines. The
nflverse kickoff conversion agrees with devdb01 `games.kickoff` (repaired today from the same
nflverse fields) on all 7,017 games of 2000-2025 that have a kickoff.

## 2019-2024 baselines (bundled CSV)

The ATS, ROI, MAE, placebo and information rows are byte-identical to the output before these
changes, and so are both README rows (`609-608-31, 50.0%, -4.5% (-8.9% to 0.0%)` and
`581-596-31, 49.4%, -5.8% (-11.5% to -0.4%)`). Only the guardrail rows are new: the fixed
"sanity: 46-53%" band became the binomial test and the sign check described below. A unit test
(`test_readme_baseline_rows_reproduce`) now pins the README rows through the CLI.

### market vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 0-0-0 | |
| Win rate | n/a | n/a to n/a |
| Breakeven | 52.4% | |
| ROI | n/a | n/a to n/a |
| P(win rate > breakeven) | p = n/a | |
| Spread MAE: model / market | 9.84 / 9.84 | DM p = 1.000 |
| Placebo win rate (shuffled) | 49.8% | should be ~50% |
| Pooled home cover rate | 48.6% | 756/1557, binomial p = 0.742 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | n/a | not identified: pred - line is constant (the prediction is the line plus a constant), so it carries no information beyond the line |

### power_ratings vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 609-608-31 | |
| Win rate | 50.0% | 47.2% to 52.8% |
| Breakeven | 52.4% | |
| ROI | -4.5% | -8.9% to +0.0% |
| P(win rate > breakeven) | p = 0.952 | |
| Spread MAE: model / market | 10.42 / 9.84 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.7% | should be ~50% |
| Pooled home cover rate | 48.6% | 756/1557, binomial p = 0.742 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | +0.079 | -0.076 to +0.233 |

### elo vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 581-596-31 | |
| Win rate | 49.4% | 46.5% to 52.2% |
| Breakeven | 52.4% | |
| ROI | -5.8% | -11.5% to -0.4% |
| P(win rate > breakeven) | p = 0.982 | |
| Spread MAE: model / market | 10.24 / 9.84 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.5% | should be ~50% |
| Pooled home cover rate | 48.6% | 756/1557, binomial p = 0.742 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | +0.100 | -0.121 to +0.322 |

## Guardrails

### Market checks (before any model is graded)

The old check refused any frame whose pooled home cover rate fell outside a fixed 46-53%,
whatever its size. That blocked the only unseen holdout (2026 to date: 40/88 = 45.5%) and 4 of
the 6 complete 2026 weeks graded alone, while it could not tell sportsbook-signed lines from
nflverse ones on small samples. Two checks replace it:

- **Home cover rate, exact binomial test.** nflverse 1999-2025 (all game types) has 3,473 home
  covers in 7,081 decided games, 0.4905, with 195 pushes excluded (regular seasons alone:
  3,319/6,778 = 0.4897). The check runs a two-sided exact binomial test of the pooled covers
  against p0 = 0.49 and refuses when p < 0.001, so the tolerated range shrinks as the sample
  grows instead of being fixed.
- **Line sign.** Lines in nflverse sign (expected home margin, positive = home favored)
  correlate positively with the home margin: +0.43 over the 1999-2025 regular seasons. Lines in
  sportsbook sign give exactly the opposite. The check refuses corr(spread_close, home_margin)
  <= 0 from 30 games on. Below 30 games it is skipped and the report says so. In the 1999-2025
  regular seasons, 2 of the 255 two-week windows with at least 30 games have corr <= 0 (0.8%),
  which is the false refusal rate at the smallest sample the check judges.

| Market frame | Home covers | Binomial p | corr(line, margin) | Outcome |
|---|---|---|---|---|
| Bundled CSV 2019-2024 | 756/1557 = 48.6% | 0.742 | +0.45 | passes |
| same rows, `spread_close` negated (sportsbook sign) | 882/1577 = 55.9% | 4e-08 | -0.45 | refused by both checks |
| same rows, home-field advantage (2.5) added to the line a second time | 608/1567 = 38.8% | 5e-16 | +0.45 | refused by the binomial test |
| 2026 to date (weeks 1-6 and the week 7 Thursday game) | 40/88 = 45.5% | 0.524 | +0.33 | passes (the old band refused it) |
| same rows, `spread_close` negated | 53/92 = 57.6% | 0.117 | -0.33 | refused by the sign check only |
| 2026 week 5 alone | 4/14 = 28.6% | 0.181 | (15 games) | passes, sign check skipped with a warning |

The refusal messages say what is probably wrong. For the 2026 rows negated:

> spread_close correlates -0.33 with the home margin over 94 games. Lines in nflverse sign
> (expected home margin, positive = home favored) correlate positively (+0.43 over nflverse
> 1999-2025), so these lines are probably in sportsbook sign (home favorite negative). Negate
> them before grading (nfl_analytics/utils/spread_convention.py)

and for the double-counted home-field advantage, where the sign is right but the covers are not:

> Pooled home cover rate 38.8% (608/1567 decided games) is not what a fair market produces:
> two-sided exact binomial p = 5e-16 against 49% (refused below 0.001). The cover grading is
> probably wrong: the home team covers when home_margin - spread_close > 0 with nflverse lines
> (spread_convention.home_cover); otherwise the lines are in sportsbook sign, or the scores and
> lines come from different games

The negated and shifted frames are derived from the real rows for these checks; they are not
data. The market checks cannot be overridden (`--allow-implausible` only covers the win-rate and
ROI ceilings), because every number graded against a broken market is wrong.

### Prediction checks

- **Sign of the predictions.** corr(pred, line) <= 0 is refused as sportsbook-signed
  predictions. Between 0 and 0.2 the result is reported with a warning instead of being refused:
  a correctly signed weak model is a result, not an error. Example from real columns: home by 2.5
  plus a point per day of extra rest correlates +0.04 with the line over 2019-2024 and is graded
  with the warning.
- **Information regression.** When pred - line is constant (the line itself, or the line plus a
  constant) or an exact linear function of the line (a constant prediction such as "home by
  2.5"), the regression is rank deficient; its coefficients are NaN and a note says why. Before,
  home by 2.5 on 2019-2024 produced a coefficient interval of ±6.1 million (SE 3.1 million).
  Fewer than 2 weeks is also reported as not estimable.
- **Intervals.** Week-block bootstrap intervals (ROI, closing-line value) need at least two
  weeks; with one week they are n/a rather than a zero-width interval.
- **JSON.** `--json` output is strict JSON: NaN and infinities are `null`.

## Warm start for `--db` (2025)

`--db` used to load only the requested seasons, so 2025 baselines started every team at the
same rating. Same code, devdb01 2025 regular season (272 games), cold vs warm (2024 seeds the
ratings, now the default):

| Baseline | Start | ATS | Win rate (95% CI) | ROI (95% CI) | MAE model / market | Log-loss model / market |
|---|---|---|---|---|---|---|
| Elo | cold | 100-106-0 | 48.5% (41.8% to 55.3%) | -7.0% (-19.1% to +4.2%) | 10.30 / 9.72 | 0.6420 / 0.6094 |
| Elo | warm | 91-105-1 | 46.4% (39.6% to 53.4%) | -11.3% (-20.2% to -2.1%) | 10.20 / 9.72 | 0.6368 / 0.6094 |
| Power ratings | cold | 106-106-1 | 50.0% (43.3% to 56.7%) | -4.6% (-14.8% to +5.6%) | 10.38 / 9.72 | |
| Power ratings | warm | 96-110-1 | 46.6% (39.9% to 53.4%) | -10.7% (-21.7% to +1.0%) | 10.40 / 9.72 | |

The warm start improves Elo's MAE and log-loss and changes which games clear the 1-point edge
(Elo settles 196 bets instead of 206); the win rates fall by 2.1 (Elo) and 3.4 (power ratings)
points, well inside their intervals. Neither baseline beats the line either way. The cold rows
equal the `--db` output before this change exactly.

## 2026 holdout

nflverse `games.csv` downloaded 2026-10-24 16:54 EDT (sha256 `5d882d05...0693`): 94 completed
2026 games (weeks 1-6 and `2026_07_NE_CHI`), graded with 2025 as the warm-up season:

```bash
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline elo \
    --nflverse-csv games.csv --seasons 2026
```

| Baseline | ATS | Win rate (95% CI) | ROI (95% CI) | MAE model / market (DM p) | Log-loss model / market |
|---|---|---|---|---|---|
| market | 0-0-0 | n/a | n/a | 9.24 / 9.24 (1.000) | |
| Elo | 35-30-4 | 53.8% (41.9% to 65.4%) | +3.0% (-23.7% to +27.4%) | 9.42 / 9.24 (0.574) | 0.6615 / 0.6601 (diff -0.0328 to +0.0386) |
| Power ratings | 30-45-4 | 40.0% (29.7% to 51.3%) | -23.8% (-53.4% to +5.0%) | 10.27 / 9.24 (0.008) | |

All three pass the market checks (40/88 home covers, p = 0.524; corr +0.33). Before this change
the benchmark refused all three ("Pooled home cover rate 45.5% is outside 46%-53%"). At 65-75
settled bets the ROI intervals are 51-58 points wide: these rows show the tool works on the
holdout, not that either baseline has an edge. The Elo MAE depends on the warm-up length:
started in 2018 (as in the survey's holdout script) it is 9.51, with the CLI's one-season
warm-up 9.42; power ratings use only the previous season and give 10.27 either way.

## Predict mode

```bash
uv run python -m nfl_analytics.evaluation.market_benchmark predict --baseline elo \
    --nflverse-csv games.csv --season 2026 --week 8 --out preds_2026_w08_elo.csv
```

- Fits the walk-forward baseline (`elo` or `power_ratings`) on every completed regular-season
  game of the previous season and of the earlier weeks of the target season: the same inputs
  the grading path uses, so a prediction equals the value the grader computes for that game
  later (unit-tested on 2026 week 5).
- Writes one row per target-week game with no score and a kickoff in the future: `game_id`,
  `season`, `week`, `kickoff_utc`, `home_team`, `away_team`, `pred_home_margin`,
  `pred_home_win_prob` (Elo only), `spread_at_prediction` (the file's `spread_line`, empty if
  not posted), `prediction_timestamp` (UTC time of the run, ISO 8601), `baseline`,
  `baseline_params` (JSON of the constants) and `input_sha256` (of the exact bytes the fit read).
  It prints the sha256 of the input and of the output file, and refuses to replace an existing
  output file unless `--overwrite` is given. Commit the file before kickoff to fix the record.
- `spread_at_prediction` is the line in the nflverse file as of its download, not at the
  second the command ran; the input hash pins which file that was.
- Run on 2026-10-24 23:12 UTC: 14 week 8 games, fit on 366 completed games (2025 regular
  season and 94 of 2026). The 13 week 7 games still to be played were not in the fit. Elo
  output sha256 `10f43776003f89f550f5767f70eed8e9b33587a8487a1325dc1a9543f63c6e3a`, power
  ratings `c8448d60c26a4f7b366568b4c586eda926b04f3449e6dd2a519cd237e2693fb8`. These files are
  outside the repo.

Grade the file once the games are final; before that the grader says none of them is completed:

```bash
uv run python -m nfl_analytics.evaluation.market_benchmark \
    --predictions-csv preds_2026_w08_elo.csv --nflverse-csv games.csv --seasons 2026
```

The report shows the retroactive share (prediction_timestamp at or after kickoff; rows without
both timestamps are left out and counted in a warning) or confirms that every prediction
preceded kickoff, plus the closing-line value below.

## Closing-line value

`spread_at_prediction` turns a prediction file into bets placed when the prediction was made:
|pred - spread_at_prediction| >= `--min-edge`, with the side taken from that edge (choosing the
side against the closing line would use information from after the prediction). Each bet's
closing-line value is `metrics.closing_line_value`: under the nflverse sign, backing home gains
`spread_close - spread_at_prediction` points and backing away gains the opposite. The report
gives the mean per bet with a week-block bootstrap interval and the share of bets that beat the
close. It converges much faster than ATS ROI, which is why it matters for weekly prospective
records.

Worked example on real lines (nflverse `spread_line` on 2025-12-07, preserved in that day's
devdb01 backup, against the closing line; tests/fixtures/nflverse/README.md):

| Game | 2025-12-07 | Close | Bet at 2025-12-07 | CLV |
|---|---|---|---|---|
| 2025_15_LAC_KC (KC home) | 4.5 | 5.5 | KC (home) | +1 |
| | | | LAC (away) | -1 |
| 2025_15_BUF_NE (NE home) | 1.5 | -2.5 | BUF (away) | +4 |
| | | | NE (home) | -4 |

KC at 4.5 needed a 5-point win where the close needed 6; BUF at +1.5 could lose by 1 where the
close made BUF a 2.5-point favorite. End to end: a "prediction" equal to the eventual closing
line, made at the 2025-12-07 lines for the 16 week-15 games, bets 9 of them and gains +3.39
points per bet, every bet positive (`test_report_gives_clv_for_bets_chosen_at_the_line_when_predicted`).

## Reading the results

- **Self-test:** the closing line placed zero bets against itself and matched its own MAE
  (DM p = 1). Shuffled-prediction placebos sit at ~50%, and the pooled home cover rate is 48.6%
  (p = 0.74 against 49%), so the grading is unbiased.
- **Power ratings** (the model behind the retracted claim) go 609-608-31. The 95% interval on
  the win rate (47.2%-52.8%) is centred on 50% and only just reaches the 52.4% breakeven at its
  upper edge; the coefficient on (prediction − line) has an interval spanning zero, so the
  ratings carry no information the line lacks. Graded with the old sign error the same predictions
  "win" 74.1% (tests/unit/test_spread_convention.py).
- **Elo** is no better (49.4%).
- At ~1,200 bets the ROI interval is still about ±4.5 points wide; at 208 bets (the 2025
  season) it is about ±13, and on the 65-75 bets of the 2026 holdout ±25 to ±29. ROI alone
  cannot separate a small real edge from noise; log-loss against the de-vigged moneyline and
  closing-line value converge much faster.
