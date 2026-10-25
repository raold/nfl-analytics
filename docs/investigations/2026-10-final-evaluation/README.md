# Final Evaluation (2026-10-24)

This directory holds the project's final grade of its models against the NFL closing line. Every
result on this page comes from a committed script; each script's output is saved in
[`results/`](results/), and the commands are under [Reproduce](#reproduce). The spread convention
throughout is nflverse's: `spread_close` is the expected home margin (positive = home favored),
the home team covers when margin - line > 0, and a model's edge is prediction - line.

## Result

No model in this project beats the closing line. That holds for every measure the data can
test: the accuracy of the predicted margin (MAE), win-probability log-loss against the de-vigged
closing moneyline, the against-the-spread (ATS) record at real prices, and information beyond the
line (the coefficient on prediction - line when the final margin is regressed on both).

- **The line is more accurate than every model that predicts a margin.** Game Outcome v2 (382
  features) missed the 2025 games by 11.05 points on average; the closing line missed by 9.67
  (285 games, Diebold-Mariano p = 2e-6), and v2 was closer than the line in 39.6% of games. Two
  simple walk-forward baselines did better than v2 but still worse than the line: in the 2025
  regular season Elo's MAE was 10.20 and the power ratings' 10.40, against 11.14 for v2 and 9.72
  for the line. Over 2019-2025 Elo, the power ratings and the Kalman state-space model scored
  10.23, 10.41 and 10.63 against 9.83 (9.82 on the state-space games), all with p < 1e-6.
- **Win probabilities are worse than the market's.** v2's log-loss is 0.6871 against 0.6070 for
  the proportionally de-vigged moneyline (0.6068 with Shin's method); the difference, +0.0801
  (95% CI +0.0505 to +0.1089), leaves v2 barely ahead of a coin flip (0.6931). Elo scores 0.6385
  against 0.6084 over 2019-2025, the state-space model 0.6430 against 0.6084.
- **No ATS record clears break-even.** Betting the side of its edge in every 2025 game, v2 went
  139-145-1 (48.9%). The old headline, -3.2% ROI on 208 bets, is the subset with |edge| strictly
  above 2 points (105-102-1); market_benchmark's default (regular season, |edge| >= 1) gives
  117-124-1 and -7.1%. Over 2019-2025, Elo and the power ratings lost 5.3% and 4.3% at the
  listed closing prices, with intervals that exclude zero. The 2004-2024 walk-forward GLM won
  51.0% of 3,826 bets for an ROI of -2.6% ([walkforward_2004_2024.md](walkforward_2004_2024.md)).
- **Where a model disagrees with the line, the line is right.** Every information-regression
  coefficient on prediction - line has an interval that includes zero (v2: -0.49, -1.36 to
  +0.37), except the state-space model on 2025 weeks 1-7, whose coefficient is negative
  (-0.67, -1.28 to -0.07).

**What "no edge" means given the intervals.** It is a measurement, not a proof that no edge
exists. ATS records are noisy: one season of v2 (284 settled bets) has a win-rate interval of
43.2-54.7% and an ROI interval of -15.2% to +1.6% at -110, which cannot tell a 2-point edge from
none. Larger samples are tighter and still centered below break-even: over 2019-2025 Elo (1,365
settled bets) and the power ratings (1,422) have win-rate intervals of 46.4-51.7% and
47.0-52.2%, and ROI intervals of -10.5% to -0.3% and -8.7% to -0.2%; the 2004-2024 GLM has
49.4-52.6% on 3,826 bets. So an edge large enough to beat the vig is ruled out for the baselines
and is at the far edge of the GLM's interval, while a small positive edge cannot be excluded by
ATS records alone. MAE and log-loss settle the question with far fewer games: every model is
measurably worse than the line on MAE in every multi-season sample, and on log-loss so are Elo
and the state-space model over 2019-2025 and v2 in 2025, with intervals that exclude zero. The
2026 holdout (94
games, 65-75 settled bets per baseline) is too small for ATS: Elo's 35-30-4 (+3.0%, -23.7% to
+27.4%) is noise, and its MAE (9.42 against 9.24) and log-loss (0.6615 against 0.6601) are
slightly worse than the market's.

## Game Outcome v2, 2025: the canonical numbers

Source: [`data/game_outcome_v2_predictions_2025.csv`](../../../data/game_outcome_v2_predictions_2025.csv),
exported from devdb01 `ml.game_outcome_predictions` joined to `games` (285 games, one prediction
run created 2026-02-12 14:46 UTC, after the season, so every prediction is retroactive). Its
market and score columns equal nflverse `games.csv` for 285 of 285 games, and
`grade_game_outcome_v2_2025.py` grades it without the database. Full tables:
[`results/game_outcome_v2_2025.md`](results/game_outcome_v2_2025.md).

Columns: `id` through `created_at` are the stored row. `pred_home_spread` is the expected home
margin, `vegas_spread` and `vegas_total` equal `spread_close` and `total_close`, `actual_spread`
is the home margin, and `kickoff` is the game date at midnight Eastern written as UTC, not the
kickoff time. `game_type` through `under_odds` come from `games` (nflverse schedule data, not
Odds API data): `kickoff_utc` is the true kickoff, then scores, closing spread and total
(nflverse sign), moneylines and spread and total prices. `prediction_timestamp` repeats
`created_at` under the name market_benchmark reads.

Definitions:

- *pred* is `pred_home_spread`, the model's expected home margin despite its name (the model's
  target is home score - away score); *edge* = pred - line, exact to 0.1 point.
- An ATS bet takes the side of the edge and wins if that side covers; the push (1 game) is
  refunded and left out of win rates and ROI. "Placed" counts it. Win-rate intervals are Wilson
  95%. ROI is profit per settled bet: at -110 a win pays +100/110 and a loss costs 1 (break-even
  52.38%); "listed" uses each side's nflverse closing price. ROI intervals resample whole weeks.
- Winners: the 40-40 tie (2025_04_GB_DAL) is left out; a picker without a pick for a game counts
  as a miss. McNemar's exact test compares the model with the closing-spread favorite on the
  same games.
- Log-loss and Brier use the model's `pred_home_win_prob` against the closing moneyline with the
  vig removed proportionally (Shin's method in brackets); intervals on the per-game difference
  resample weeks.

| Bets | Placed | W-L-P | Win rate [95% CI] | p vs 52.38% | ROI at -110 [95% CI] | ROI at listed prices [95% CI] |
|---|---|---|---|---|---|---|
| All 285 games | 285 | 139-145-1 | 48.9% [43.2%, 54.7%] | 0.89 | -6.6% [-15.2%, +1.6%] | -6.3% [-14.9%, +1.9%] |
| \|edge\| >= 1 | 254 | 124-129-1 | 49.0% [42.9%, 55.1%] | 0.87 | -6.4% [-18.0%, +3.7%] | -6.2% [-18.0%, +4.0%] |
| \|edge\| >= 2 | 210 | 105-104-1 | 50.2% [43.5%, 57.0%] | 0.75 | -4.1% [-16.8%, +7.8%] | -4.0% [-16.8%, +8.2%] |
| \|edge\| > 2 (old headline) | 208 | 105-102-1 | 50.7% [44.0%, 57.5%] | 0.71 | -3.2% [-16.1%, +8.8%] | -3.1% [-16.2%, +9.2%] |
| Regular season, every game | 272 | 131-140-1 | 48.3% [42.5%, 54.3%] | 0.92 | -7.7% [-16.6%, -0.2%] | -7.4% [-16.3%, +0.1%] |
| Regular season, \|edge\| >= 1 | 242 | 117-124-1 | 48.5% [42.3%, 54.8%] | 0.90 | -7.3% [-19.3%, +3.2%] | -7.1% [-19.2%, +3.4%] |
| Playoffs, every game | 13 | 8-5-0 | 61.5% [35.5%, 82.3%] | 0.35 | +17.5% [-36.4%, +36.4%] | +16.9% [-34.9%, +36.6%] |
| Playoffs before the Super Bowl | 12 | 8-4-0 | 66.7% [39.1%, 86.2%] | 0.24 | +27.3% [-4.5%, +43.2%] | +26.7% [-2.4%, +43.1%] |

The regular-season row at |edge| >= 1 is market_benchmark's own report
([`results/game_outcome_v2_2025_market_benchmark_reg.md`](results/game_outcome_v2_2025_market_benchmark_reg.md)):
the CLI with `--db` or `--nflverse-csv` gives an identical JSON report. It also flags every
prediction as made after kickoff.

| Measure | All games | Regular season | Playoffs |
|---|---|---|---|
| Games | 285 | 272 | 13 |
| Spread MAE, model / line (Diebold-Mariano p) | 11.05 / 9.67 (2e-6) | 11.14 / 9.72 (2e-6) | 9.12 / 8.58 (0.59) |
| Model closer to the margin than the line | 39.6% | 39.3% | 46.2% |
| Winners: model P(home win) > 0.5 | 162/284 = 57.0% [51.2%, 62.7%] | 156/271 = 57.6% | 6/13 = 46.2% |
| Winners: closing-spread favorite | 187/284 = 65.8% [60.2%, 71.1%] | 177/271 = 65.3% | 10/13 = 76.9% |
| Model right where the favorite was wrong / the reverse (McNemar p) | 40 / 65 (0.019) | 39 / 60 (0.044) | 1 / 5 (0.22) |
| Log-loss, model / market (Shin) | 0.6871 / 0.6070 (0.6068) | 0.6868 / 0.6094 (0.6094) | 0.6934 / 0.5571 (0.5530) |
| Log-loss difference [95% CI] | +0.0801 [+0.0505, +0.1089] | +0.0774 [+0.0463, +0.1082] | +0.1363 [+0.0412, +0.2498] |
| Brier, model / market (Shin) | 0.2467 / 0.2109 (0.2110) | 0.2465 / 0.2121 (0.2124) | 0.2515 / 0.1852 (0.1835) |
| Information: coefficient on pred - line [95% CI] (week clusters) | -0.492 [-1.358, +0.374] (22) | -0.572 [-1.494, +0.350] (18) | not estimated |

The de-vigged moneyline favorite picks the same winners as the spread favorite (one game priced
at -110 both ways gets no pick). The sign of v2's predicted margin picks 148 of 284 winners
(52.1%; 9 predictions are exactly 0), the number `ml.model_metrics` stored as `ats_win_pct`.

## Walk-forward baselines (market_benchmark)

`grade_baselines.py` runs the market_benchmark CLI for each window: devdb01 (read-only) for
2019-2024, 2025 and 2019-2025, and for the 2026 holdout the nflverse `games.csv` downloaded
2026-10-25 01:28:03 UTC (upstream Last-Modified 01:16:43 UTC; 2,186,477 bytes, sha256
`07b41111a46458065937e402656f0636743a53513bade967789709395c16a1b2`; not in the repo), whose 94
completed regular-season games are weeks 1-6 and `2026_07_NE_CHI`. Only regular-season games are
graded; devdb01's `game_type` was repaired from nflverse on 2026-10-24. Each baseline warms up on
the season before the first one graded (2019-2025 runs on from 2018 without a restart, so its
2025 games are not the same predictions as the 2025-only row). Bets at |edge| >= 1; ROI at the
listed closing prices. All reports: [`results/baselines_summary.md`](results/baselines_summary.md)
and `results/baselines_*.{md,json}`.

| Seasons (source) | Baseline | Games | ATS W-L-P | Win rate [95% CI] | ROI [95% CI] | MAE model / line (DM p) | Log-loss model / market | Info coef [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 2019-2024 (devdb01) | Elo | 1599 | 584-593-31 | 49.6% [46.8%, 52.5%] | -4.2% [-10.1%, +1.3%] | 10.23 / 9.84 (0.000) | 0.6386 / 0.6082 | +0.098 [-0.123, +0.320] |
| 2019-2024 (devdb01) | Power ratings | 1599 | 609-607-31 | 50.1% [47.3%, 52.9%] | -3.2% [-7.6%, +1.3%] | 10.42 / 9.84 (0.000) | | +0.079 [-0.076, +0.234] |
| 2025 (devdb01) | Elo | 272 | 91-105-1 | 46.4% [39.6%, 53.4%] | -11.3% [-20.2%, -2.1%] | 10.20 / 9.72 (0.006) | 0.6368 / 0.6094 | -0.344 [-1.009, +0.321] |
| 2025 (devdb01) | Power ratings | 272 | 96-110-1 | 46.6% [39.9%, 53.4%] | -10.7% [-21.7%, +1.0%] | 10.40 / 9.72 (0.002) | | -0.206 [-0.620, +0.207] |
| 2019-2025 (devdb01) | Elo | 1871 | 670-695-32 | 49.1% [46.4%, 51.7%] | -5.3% [-10.5%, -0.3%] | 10.23 / 9.83 (0.000) | 0.6385 / 0.6084 | +0.045 [-0.169, +0.258] |
| 2019-2025 (devdb01) | Power ratings | 1871 | 705-717-32 | 49.6% [47.0%, 52.2%] | -4.3% [-8.7%, -0.2%] | 10.41 / 9.83 (0.000) | | +0.038 [-0.106, +0.183] |
| 2026 holdout (nflverse) | Elo | 94 | 35-30-4 | 53.8% [41.9%, 65.4%] | +3.0% [-23.7%, +27.4%] | 9.42 / 9.24 (0.574) | 0.6615 / 0.6601 | +0.174 [-0.683, +1.031] |
| 2026 holdout (nflverse) | Power ratings | 94 | 30-45-4 | 40.0% [29.7%, 51.3%] | -23.8% [-53.4%, +5.0%] | 10.27 / 9.24 (0.008) | | -0.229 [-0.940, +0.483] |

The `market` baseline (the line itself) places no bets and equals its own MAE in every window
(9.84, 9.72, 9.83, 9.24), as a correct grader must. The Elo log-loss differences have intervals
of +0.0208 to +0.0399 (2019-2025) and -0.0328 to +0.0386 (2026). The guardrails pass everywhere:
pooled home covers 891/1828 (2019-2025) and 40/88 (2026), and the line correlates positively
with the margin.

**Why the devdb01 rows for 2019-2024 differ from the README rows** (power ratings 609-608-31,
-4.5%; Elo 581-596-31, -5.8%, from `data/historical_weather_2018_2024.csv`): that CSV writes the
Raiders as LV in 2018-2019, where nflverse and devdb01 have OAK, and it has no spread prices.
market_benchmark keys ratings by team code, so on devdb01 the Raiders restart at the mean in
2020, and devdb01's listed prices carry less vig than -110 before 2023 (mean break-even 51.7-51.8%
instead of 52.4%; [market facts](../2026-10-market-facts/README.md) section 4). Recoding OAK as LV and pricing every bet at -110 reproduces the README rows
exactly ([`results/baselines_reconcile_2019_2024.csv`](results/baselines_reconcile_2019_2024.csv)):

| Baseline | Market frame | ATS W-L-P | ROI |
|---|---|---|---|
| Elo | bundled CSV (README) | 581-596-31 | -5.8% |
| Elo | devdb01 | 584-593-31 | -4.2% |
| Elo | devdb01, Raiders as LV | 581-596-31 | -4.7% |
| Elo | devdb01, Raiders as LV, every bet at -110 | 581-596-31 | -5.8% |
| Power ratings | bundled CSV (README) | 609-608-31 | -4.5% |
| Power ratings | devdb01 | 609-607-31 | -3.2% |
| Power ratings | devdb01, Raiders as LV | 609-608-31 | -3.3% |
| Power ratings | devdb01, Raiders as LV, every bet at -110 | 609-608-31 | -4.5% |

## Kalman state-space walk-forward

[`models/state_space/predictions_historical.csv`](../../../models/state_space/predictions_historical.csv)
holds P(home win) for 1,782 games of 2019 through 2025 week 7, from one `StateSpaceRatings`
filter run week by week from a cold start in 2019 (q = 2, r = 15, home-field advantage 3): every
game of a week is predicted, then the filter updates on that week's margins. The file has no
margins and no line, so it is sign-safe as stored; `grade_state_space_walkforward.py` recovers the
margins by running the same filter on the same games from devdb01. The refit reproduces every
stored probability (largest difference 1.1e-16) and its expected home margin mu equals
sd x Phi^-1(p) to 1.4e-14, where sd^2 = var_home + var_away + 4 + 15^2. Two details of the
stored run had to be matched: it wrote the 2019 Raiders as LV, and it did not have
`2025_06_LA_BAL`. The margins are in
[`results/state_space_walkforward_margins.csv`](results/state_space_walkforward_margins.csv); on
the graded games they correlate +0.84 with the line.

(q, r, home-field advantage) is the grid point with the lowest 2024 Brier score among 27 in
`models/state_space/hyperparameter_search_results.json` (filter run on 2019-2023, scored on
2024), so 2019-2024 is not out of sample for those three numbers. Only the 107 games of 2025 come
after the tuning. Regular-season games, bets at |edge| >= 1, ROI at listed prices:

| Seasons | Games | ATS W-L-P | Win rate [95% CI] | ROI [95% CI] | MAE model / line (DM p) | Log-loss model / market [diff CI] | Info coef [95% CI] |
|---|---|---|---|---|---|---|---|
| 2019-2024 | 1599 | 633-653-31 | 49.2% [46.5%, 52.0%] | -4.7% [-8.9%, -0.4%] | 10.66 / 9.84 (8e-14) | 0.6437 / 0.6082 [+0.0254, +0.0455] | +0.024 [-0.107, +0.155] |
| 2025, weeks 1-7 | 107 | 39-42-0 | 48.1% [37.6%, 58.9%] | -7.6% [-23.8%, +7.7%] | 10.23 / 9.44 (0.063) | 0.6325 / 0.6104 [-0.0039, +0.0476] | -0.671 [-1.275, -0.066] |
| 2019-2025 | 1706 | 672-695-31 | 49.2% [46.5%, 51.8%] | -4.9% [-9.2%, -0.4%] | 10.63 / 9.82 (1e-14) | 0.6430 / 0.6084 [+0.0255, +0.0445] | -0.004 [-0.135, +0.127] |

These are not the numbers in `models/state_space/test_metrics.json` (MAE 10.58 on 107 games of
2025, "ATS 74.7%"), which come from a different evaluation of the model with the old cover rule;
[retraction_ledger.md](retraction_ledger.md) row C18 has its corrected values.

## 2004-2024 walk-forward and the retraction ledger

[walkforward_2004_2024.md](walkforward_2004_2024.md) regrades the October 2025 harness behind the
dissertation's betting tables (5,529 games): its labels are correct, the state-space column had
the sign bug, and the printed ROI charged the vig twice. Graded correctly, the GLM wins 51.0%
[49.4%, 52.6%] of 3,826 bets for an ROI of -2.6% [-5.1%, +0.0%]; the most favorable one-sided
p-value against break-even among the valid columns is 0.86 betting every game (0.50 under the
dissertation's rule), and no model's Brier score is below a constant 0.5 forecast (best 0.2507).
[retraction_ledger.md](retraction_ledger.md) lists the 44 numbers this project presented as an
edge or a headline, where each still appears, what is wrong with it and the corrected value.

## Prospective 2026 predictions

[`prospective/`](prospective/) holds walk-forward Elo and power-rating predictions for the 27
games of 2026 weeks 7 and 8 that had not kicked off, made with market_benchmark's predict mode at
2026-10-25 01:30 UTC from the `games.csv` above (its sha256 is in every row), about 12 hours
before the first of them (`2026_07_PIT_NO`, 13:30 UTC). `prospective/SHA256SUMS` pins the four
files. Rerunning predict mode on the same input into another directory reproduces every column
except `prediction_timestamp` (checked on 2026-10-25 at about 02:00 UTC). They are baselines
with no claimed edge, and they only become third-party evidence once the files or their digests
are published before kickoff. [`prospective/README.md`](prospective/README.md) lists the
predictions and how to grade them, including closing-line value against the line at prediction
time.

## Related write-ups

- [Line movement and closing-line value, 2020-2024](../2026-10-line-movement/README.md)
  (licensed `odds_history`, aggregates only): where the consensus moved a point or more before
  the close, the side it moved toward covered 59.2% at the early line but 50.4% at the close, so
  the close is efficient and a bet's value is the number it got. Shopping about 15 books gains
  0.34 points per side (51.5% ATS), still short of 52.4%; the panel has no Rams or night games.
- [Market facts from nflverse schedules, 1999-2025](../2026-10-market-facts/README.md): the
  closing spread shows no detectable bias (home covers 49.0% [47.9, 50.2]) and the de-vigged
  moneyline is calibrated. Wind unders (57.0% at 15 mph or more) and Wong teaser legs (74.7%) do
  not survive a Holm correction as profitable bets (Holm p 0.056 and 0.138 or more).
- [Market benchmark](../2026-10-market-benchmark.md): the grading tool, its guardrails and its
  options.

## Earlier numbers that do not match

- **"56.5% win-probability accuracy"** (README, CLAUDE.md, ADR-017/018, dissertation) does not
  reproduce from the stored run: the model picks 162 winners, 57.0% of 284 decided games (56.8%
  of 285 if the tie counts as a miss, the figure in ledger C03), against 65.8% (187/284) for the
  closing favorite. (`models/production/game_outcome_v2/README.md` first gave 66.0% for the
  favorite; it now uses these figures.) Nor do the dissertation's
  57.0% (regular season) and 58.3% (playoffs before the Super Bowl): here 156/271 = 57.6% and
  5/12 = 41.7%. Its ATS rates 48.2%, 66.7% and 48.8% do reproduce, with the push counted as a
  loss: 131 of 272, 8 of 12 (8-4-0) and 139 of 285.
- **"-3.2% ROI on 208 bets"** holds only for |edge| strictly above 2, and its 208 bets include the
  push. Ledger C02 gives the regular season at |edge| >= 1 as -7.3%: that is at -110, and
  market_benchmark's -7.1% uses the listed prices. Both are in the ATS table.
- **The salvage survey's v2 table** (critique of 2026-10-24, not in the repo) reproduces except
  for two details: its all-games ROI of -6.5% divides by 285 bets including the push (per
  settled bet it is -6.6%), and its 65.9% for the closing favorite is 187/284 = 65.85%, here
  65.8%. Its 2025 baseline figures (power ratings 47.9%, Elo 50.0%) bet every game of every
  week with no edge threshold, so they are not the market_benchmark grade above.
- **"8.43 playoff MAE vs 11.14 regular season"** (CLAUDE.md lesson 23, dissertation) is the 12
  playoff games before the Super Bowl (8.425 exactly). The line was also better than usual on
  those games (8.33, against 9.72 in the regular season) and still ahead of v2, and 12 games say
  little: a random 13-game subset of the 2025 regular season gives the line an MAE at least as
  low as its playoff 8.58 29.7% of the time ([market facts](../2026-10-market-facts/README.md)
  section 8). With the Super Bowl the playoffs are 9.12 for v2 and 8.58 for the line.
- **README baseline rows vs devdb01**: the Raiders' team code and the -110 pricing, above.

## Limitations

- v2's 2025 record is one stored run made after the season, so it is retroactive by
  construction. The model was fit on 2022-2023 with 2024 as its early-stopping season, its
  semantic training inputs were computed with look-ahead, and it cannot be retrained from
  devdb01 (`models/production/game_outcome_v2/README.md`).
- market_benchmark's baselines key ratings by team code, so a relocation (OAK to LV in 2020)
  restarts a team when the source uses nflverse codes. The effect on 2019-2025 is shown above; it
  does not touch 2025-2026. Mapping franchise codes in `baselines.py` would remove it.
- market_benchmark's information-regression interval is ±1.96 standard errors while its p-value
  uses a t distribution with (weeks - 1) degrees of freedom. With few weeks they disagree: the
  state-space 2025 interval (7 weeks) excludes zero, yet p = 0.073.
- The 2026 holdout is one nflverse download. nflverse replaces the file in place, so a later
  download can revise these rows; the sha256 above identifies the version graded.

## Reproduce

From a clean shell at the repository root (`cd /home/dro/rice/nfl-analytics`). The scripts that
read devdb01 take the `DB_*` variables (defaults in `nfl_analytics/utils/db_config.py`) and force a
read-only session. `games.csv` is
<https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv>; the numbers
above use the download with sha256 `07b41111...b2`.

```bash
# Game Outcome v2, 2025 (no database); the optional flags rerun the market_benchmark CLI on the
# same predictions with devdb01 and with games.csv and check that its report is identical
uv run --no-sync python scripts/final_evaluation/grade_game_outcome_v2_2025.py \
    [--cli-db] [--cli-nflverse-csv games.csv]

# rebuild data/game_outcome_v2_predictions_2025.csv from devdb01 (byte-identical today)
uv run --no-sync python scripts/final_evaluation/export_game_outcome_v2_2025.py \
    --nflverse-csv games.csv --out /tmp/v2_check.csv

# walk-forward baselines: devdb01 2019-2024, 2025, 2019-2025; nflverse 2026 holdout (~25 s)
uv run --no-sync python scripts/final_evaluation/grade_baselines.py --nflverse-csv games.csv

# Kalman state-space walk-forward (devdb01)
uv run --no-sync python scripts/final_evaluation/grade_state_space_walkforward.py

# 2004-2024 walk-forward and the retraction ledger
uv run --no-sync python scripts/final_evaluation/regrade_walkforward_2004_2024.py
uv run --no-sync python scripts/final_evaluation/check_claims.py

# check the prospective files (do not rerun predict mode into prospective/)
(cd docs/investigations/2026-10-final-evaluation/prospective && sha256sum -c SHA256SUMS)
```

| Table | Script | Output in `results/` |
|---|---|---|
| Game Outcome v2, 2025 | `grade_game_outcome_v2_2025.py` | `game_outcome_v2_2025.{md,csv}`, `game_outcome_v2_2025_market_benchmark_reg.{md,json}` |
| Walk-forward baselines, README reconciliation | `grade_baselines.py` | `baselines_summary.{md,csv}`, `baselines_{db,nflverse}_*.{md,json}`, `baselines_reconcile_2019_2024.csv`, `baselines_provenance.csv` |
| Kalman state-space | `grade_state_space_walkforward.py` | `state_space_walkforward_*` |
| 2004-2024 walk-forward | `regrade_walkforward_2004_2024.py` | `walkforward_2004_2024_*` |
| Retraction ledger | `check_claims.py` | `claims_*` |
| Prospective 2026 | `prospective_2026.py` | `prospective/` |

The devdb01 rows depend on the database: `results/baselines_provenance.csv` records a sha256 of
the 2,127 graded 2018-2025 rows (`abda0867...`), so a later run can tell whether the data changed.
