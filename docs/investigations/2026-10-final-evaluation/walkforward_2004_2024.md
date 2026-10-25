# 2004-2024 Walk-Forward ATS Harness, Regraded (2026-10-24)

The October 2025 multi-model harness predicted, for 5,529 games in 2004-2024, the probability that
the home team covers the closing spread. Those predictions feed the dissertation's betting,
multi-model, CLV and per-season tables (chapter 9). This note regrades every column with the
correct cover rule, the correct -110 payoff and a sign-fixed state-space model.

**Result.** The labels are correct and the null result holds. No column beats the 52.4%
break-even at -110. Betting every game, the most favorable one-sided p-value against break-even
is 0.86. Under the dissertation's own bet threshold, the best column lands exactly on break-even
(p = 0.50). No model column has a Brier score below a constant 0.5 forecast. Against the closing
line these models have no ATS skill. The dissertation's headline numbers come from this harness and
are wrong in three ways:

| Printed claim | What is wrong | Corrected |
|---|---|---|
| GLM 51.0%, ROI -7.45%, Sharpe -1.22 on 3,826 bets (`betting_performance_table.tex`) | the script pays +0.91 per win and -1.10 per loss, which charges the vig twice | 51.0% [49.4%, 52.6%], ROI **-2.6%** [-5.1%, +0.0%], Sharpe **-0.45** |
| Every "State" and "Stack/Mean(...State)" row | `StateSpaceRatings.predict_prob_ats` used Phi((mu + line)/sd) | state alone: Brier 0.3197 -> **0.2749**, ATS 49.8% -> 50.6% |
| Stack Brier 0.2515 "best-in-class"; GLM "CLV +14.9 bps" | 0.2515 is worse than a constant 0.5 (0.2500); the "CLV" is (p - 0.5) signed by the result | best corrected Brier 0.2507, still above 0.2500 and above the 0.2498 base-rate forecast; the "CLV" has t = 1.35, p = 0.18, and no line movement is involved |

With the sign fixed, the best columns are the mean of GLM and state (51.7%, two-sided p = 0.01
against 50%) and the three-model stack (51.3%, p = 0.05). Neither result is evidence of skill.
The corrected stacks bet the road team in 90-93% of games, and in 2004-2024 home teams covered
only 48.7% of the time. So "always bet the road team" also wins 51.3%, at ROI -2.1%, using the
labels alone. The mean(GLM, state) column is the best of 18 columns tried. Its p = 0.01 does not
survive a multiple-comparison correction, and it is still below break-even, with ROI -1.4%
[-3.1%, +0.4%]. A regular-season-only grade gives the same picture.

## Inputs

| Input | What it is |
|---|---|
| `analysis/results/multimodel_predictions.csv` | 5,529 rows: `actual` (home covered), `glm`, `xgb`, `state`, 4 equal-weight means and 4 logistic stacks |
| `analysis/results/glm_baseline_preds.csv` | the same GLM run with `spread_close`, `home_cover` and `pred_cover_prob` (matches the `glm` column to 2.6e-13) |
| devdb01 `games` | scores and closing spreads for 2003-2024, read in a read-only session. Per surveys 12 and 18, and the coordinator's repair today, scores and lines match nflverse exactly. The repair of `game_type` and `kickoff` does not affect anything used here |

Both CSVs were committed in e7f36bd (2025-10-04). They came from
`nfl_analytics/backtest/harness_multimodel.py`, which was then at `py/backtest/`. The harness ran
with seasons 2003-2024. For each test season from 2004 on, it trained on every earlier season,
so every prediction is out of sample by season. The harness can no longer be re-run. Its input
`analysis/features/asof_team_features.csv` is not in git at the version used: the only committed
copy (b90e9d6) has 3 of the GLM's 42 feature columns and no `kickoff`. Its import
`backtest.baseline_glm.DEFAULT_FEATURE_COLUMNS` also no longer exists. The `glm` and `xgb`
columns are therefore graded as published. Their labels are verified below. Their features could
not be re-audited, but leakage could only have flattered a model, and this is a null result.
"XGB" is scikit-learn's `GradientBoostingClassifier` (200 trees, depth 3), not XGBoost.

## Labels

From `results/walkforward_2004_2024_label_check.csv`:

| Check | Result |
|---|---|
| Rows matched to devdb01 games | 5,529 of 5,529 |
| Pushes among them | 0 (the harness dropped pushes) |
| `actual` == (margin - spread_close > 0) | **100.0%** |
| `actual` == old rule (margin + spread_close > 0) | 73.1% |
| `actual` == home straight-up win | 83.3% |
| Pooled home cover rate | 48.7% |
| `glm_baseline_preds.csv` spread_close vs devdb01 | identical (max diff 0) |
| `glm_baseline_preds.csv` home_cover vs the correct rule | 100.0% |
| Playoff games included | 237 (by today's corrected `game_type`) |

## State-space sign fix

The harness's `StateSpaceModel` called `StateSpaceRatings.predict_prob_ats(home, away,
spread_close)`, which returned Phi((mu + s)/sigma), where mu and sigma are the Kalman filter's
predicted home margin and its standard deviation. With nflverse lines (s = expected home margin,
positive = home favored), home covers iff margin - s > 0, so the cover probability is
**Phi((mu - s)/sigma)**. A cover probability should be nearly uncorrelated with the line. The
published state column correlates **+0.795** with `spread_close`. For comparison, the fixed state
column correlates -0.081, GLM +0.006 and XGB +0.049 (`results/walkforward_2004_2024_reproduction.csv`).

I refit `StateSpaceRatings(q=3.0, r=13.5)` exactly as the harness did. Each fit used all non-push
games from 2003 through the season before the test season, and the ratings stayed frozen during
the test season. The script uses only the class's `fit` and `predict_margin` and computes both
probability formulas itself. Another salvage task fixed `predict_prob_ats` on 2026-10-24, and
this keeps the regrade independent of that change. The old formula reproduces the published
column to a maximum difference of **2.2e-16** on all 5,529 games, so `fit` and `predict_margin`
behave as they did in October 2025. Rebuilding the ensembles with the harness's procedure
reproduces every published ensemble column to 5.2e-15 or better. The procedure is equal-weight means, plus a
`LogisticRegression(lbfgs)` stack fit on all earlier test seasons, with the first season falling
back to the mean (`results/walkforward_2004_2024_reproduction.csv`). The six ensembles that
include state were then rebuilt from the fixed column.

## Results

Payoff at -110 is +100/110 per win and -1 per loss, so break-even is 52.38%. The win-rate
interval is Wilson 95%. The ROI interval resamples the 21 seasons with replacement (10,000
draws). Brier and log-loss are on the cover probability, compared with a constant 0.5 (Brier
0.2500, log-loss ln 2 = 0.6931). The interval on Brier - 0.25 is a season-block bootstrap. The
full-sample base-rate forecast (0.4871) has Brier 0.2498. "Home share" is the fraction of bets on
the home side. All numbers are from `results/walkforward_2004_2024_metrics.csv`, rows with
`games_subset` = all games.

### Every game, side = sign(p - 0.5)

| Column | Bets | Home share | W-L | ATS win % [95% CI] | p vs 50% | ROI at -110 [95% CI] | Brier (vs 0.25) [CI] | Log-loss (vs ln 2) |
|---|---|---|---|---|---|---|---|---|
| GLM | 5,529 | 43% | 2788-2741 | 50.4% [49.1%, 51.7%] | 0.54 | -3.7% [-6.1%, -1.2%] | 0.2552 (+0.0052) [+0.0023, +0.0093] | 0.7055 (+0.0123) |
| XGB (sklearn GBM) | 5,529 | 39% | 2770-2759 | 50.1% [48.8%, 51.4%] | 0.89 | -4.4% [-6.6%, -2.1%] | 0.2643 (+0.0143) [+0.0094, +0.0203] | 0.7260 (+0.0329) |
| Mean(GLM, XGB) | 5,529 | 39% | 2742-2787 | 49.6% [48.3%, 50.9%] | 0.55 | -5.3% [-7.3%, -3.4%] | 0.2567 (+0.0067) [+0.0037, +0.0104] | 0.7078 (+0.0147) |
| Stack(GLM, XGB) | 5,529 | 4% | 2809-2720 | 50.8% [49.5%, 52.1%] | 0.24 | -3.0% [-4.9%, -1.0%] | 0.2517 (+0.0017) [-0.0003, +0.0052] | 0.6973 (+0.0042) |
| State-space, sign fixed | 5,529 | 51% | 2800-2729 | 50.6% [49.3%, 52.0%] | 0.35 | -3.3% [-5.5%, -1.1%] | 0.2749 (+0.0249) [+0.0206, +0.0295] | 0.7538 (+0.0607) |
| Mean(GLM, State) fixed | 5,529 | 48% | 2857-2672 | 51.7% [50.4%, 53.0%] | 0.01 | -1.4% [-3.1%, +0.4%] | 0.2569 (+0.0069) [+0.0044, +0.0096] | 0.7081 (+0.0149) |
| Mean(XGB, State) fixed | 5,529 | 47% | 2789-2740 | 50.4% [49.1%, 51.8%] | 0.52 | -3.7% [-5.7%, -1.6%] | 0.2601 (+0.0101) [+0.0075, +0.0129] | 0.7150 (+0.0219) |
| Mean(GLM, XGB, State) fixed | 5,529 | 45% | 2776-2753 | 50.2% [48.9%, 51.5%] | 0.77 | -4.1% [-6.4%, -1.8%] | 0.2556 (+0.0056) [+0.0034, +0.0080] | 0.7051 (+0.0119) |
| Stack(GLM, State) fixed | 5,529 | 8% | 2831-2698 | 51.2% [49.9%, 52.5%] | 0.08 | -2.2% [-3.7%, -0.8%] | 0.2507 (+0.0007) [-0.0003, +0.0025] | 0.6948 (+0.0017) |
| Stack(XGB, State) fixed | 5,529 | 7% | 2831-2698 | 51.2% [49.9%, 52.5%] | 0.08 | -2.2% [-3.9%, -0.7%] | 0.2511 (+0.0011) [-0.0003, +0.0036] | 0.6957 (+0.0026) |
| Stack(GLM, XGB, State) fixed | 5,529 | 10% | 2839-2690 | 51.3% [50.0%, 52.7%] | 0.05 | -2.0% [-3.9%, +0.0%] | 0.2510 (+0.0010) [-0.0003, +0.0031] | 0.6953 (+0.0022) |
| *Reference: prior seasons' home cover rate* | 5,529 | 5% | 2819-2710 | 51.0% [49.7%, 52.3%] | 0.15 | -2.7% [-4.5%, -0.9%] | 0.2499 (-0.0001) [-0.0003, +0.0002] | 0.6930 (-0.0002) |
| *Reference: always the road team* | 5,529 | 0% | 2836-2693 | 51.3% [50.0%, 52.6%] | 0.06 | -2.1% [-3.7%, -0.4%] | 0.2498 (-0.0002) [-0.0004, +0.0001] | 0.6928 (-0.0003) |
| State-space, published (invalid) | 5,529 | 63% | 2754-2775 | 49.8% [48.5%, 51.1%] | 0.79 | -4.9% [-6.9%, -2.7%] | 0.3197 (+0.0697) [+0.0623, +0.0771] | 0.9086 (+0.2155) |
| Mean(GLM, XGB, State) published (invalid) | 5,529 | 57% | 2778-2751 | 50.2% [48.9%, 51.6%] | 0.73 | -4.1% [-6.4%, -1.7%] | 0.2615 (+0.0115) [+0.0084, +0.0148] | 0.7176 (+0.0244) |
| Stack(GLM, XGB, State) published (invalid) | 5,529 | 5% | 2807-2722 | 50.8% [49.5%, 52.1%] | 0.26 | -3.1% [-5.1%, -1.0%] | 0.2515 (+0.0015) [-0.0002, +0.0045] | 0.6966 (+0.0034) |

The other four published state ensembles are also in the CSV. All are below 51.0% except the two
stacks, each at 51.0%.

The reference rows use only labels from earlier games. "Prior seasons' home cover rate" predicts
every game of season S with the cover rate of all games from 2003 to S-1. "Always the road team"
uses the full-sample rate. That rate peeks at the test seasons, but it is only a benchmark. The
stacks' 51% is the road-team base rate rather than anything learned from the models. No model
column has a lower Brier score than the walk-forward base rate (0.2499).

### The dissertation's betting rule: p > 0.524 (home) or p < 0.476 (away)

| Column | Bets | W-L | ATS win % [95% CI] | ROI at -110 [95% CI] | Sharpe (x sqrt 272) |
|---|---|---|---|---|---|
| GLM | 3,826 | 1952-1874 | 51.0% [49.4%, 52.6%] | -2.6% [-5.1%, +0.0%] | -0.45 |
| XGB | 4,419 | 2212-2207 | 50.1% [48.6%, 51.5%] | -4.4% [-6.8%, -2.0%] | -0.77 |
| Mean(GLM, XGB) | 3,963 | 2011-1952 | 50.7% [49.2%, 52.3%] | -3.1% [-5.4%, -0.8%] | -0.54 |
| Stack(GLM, XGB) | 1,139 | 570-569 | 50.0% [47.1%, 52.9%] | -4.5% [-8.1%, -0.4%] | -0.77 |
| State-space, sign fixed | 4,934 | 2510-2424 | 50.9% [49.5%, 52.3%] | -2.9% [-4.8%, -0.9%] | -0.50 |
| Mean(GLM, State) fixed | 4,465 | 2292-2173 | 51.3% [49.9%, 52.8%] | -2.0% [-4.4%, +0.5%] | -0.35 |
| Mean(GLM, XGB, State) fixed | 4,251 | 2163-2088 | 50.9% [49.4%, 52.4%] | -2.9% [-5.2%, -0.5%] | -0.49 |
| Stack(GLM, State) fixed | 1,400 | 734-666 | 52.4% [49.8%, 55.0%] | +0.1% [-3.3%, +4.3%] | +0.02 |
| Stack(XGB, State) fixed | 1,231 | 631-600 | 51.3% [48.5%, 54.0%] | -2.1% [-6.1%, +2.5%] | -0.37 |
| Stack(GLM, XGB, State) fixed | 1,594 | 809-785 | 50.8% [48.3%, 53.2%] | -3.1% [-6.5%, +0.5%] | -0.54 |

Under this rule, the sign-fixed GLM+State stack breaks even (52.4% on 1,400 bets, 91% of them on
the road team). This is the best of 18 columns under a second rule. Its interval runs from 49.8%
to 55.0%, and one-sided p = 0.50 against break-even.

### By season (every game; win rate)

From `results/walkforward_2004_2024_per_season.csv`:

| Season | Games | GLM | XGB | State (fixed) | Stack(GLM, XGB, State) fixed | Prior-seasons base rate |
|---|---|---|---|---|---|---|
| 2004 | 261 | 50.6% | 49.4% | 50.6% | 50.6% | 46.7% |
| 2005 | 257 | 52.1% | 47.1% | 51.8% | 51.0% | 50.6% |
| 2006 | 259 | 49.0% | 49.4% | 45.2% | 52.1% | 51.7% |
| 2007 | 262 | 54.2% | 46.9% | 53.8% | 49.2% | 49.6% |
| 2008 | 261 | 50.2% | 47.5% | 55.9% | 56.3% | 54.4% |
| 2009 | 259 | 56.0% | 49.8% | 53.3% | 55.2% | 53.3% |
| 2010 | 262 | 50.4% | 53.1% | 48.1% | 50.8% | 50.8% |
| 2011 | 256 | 49.2% | 50.0% | 51.6% | 50.0% | 50.0% |
| 2012 | 262 | 53.1% | 48.5% | 47.3% | 52.3% | 53.1% |
| 2013 | 260 | 50.4% | 49.2% | 53.5% | 46.5% | 47.7% |
| 2014 | 261 | 48.3% | 51.0% | 51.3% | 52.9% | 51.3% |
| 2015 | 257 | 49.8% | 47.9% | 51.8% | 55.3% | 53.3% |
| 2016 | 262 | 56.5% | 53.1% | 50.8% | 54.6% | 50.0% |
| 2017 | 259 | 48.3% | 53.3% | 51.0% | 51.0% | 48.6% |
| 2018 | 258 | 46.9% | 52.3% | 50.8% | 51.9% | 52.7% |
| 2019 | 257 | 50.2% | 53.7% | 46.3% | 51.8% | 56.0% |
| 2020 | 269 | 52.8% | 46.8% | 48.7% | 48.0% | 50.6% |
| 2021 | 281 | 52.0% | 54.4% | 54.4% | 51.2% | 51.6% |
| 2022 | 274 | 48.9% | 52.9% | 49.3% | 49.6% | 50.7% |
| 2023 | 271 | 46.1% | 51.3% | 50.6% | 49.8% | 50.2% |
| 2024 | 281 | 44.5% | 44.5% | 47.7% | 48.8% | 48.0% |

Each column clears break-even in only 5 or 6 of 21 seasons. A 50% coin over about 260 games
would land above 52.4% in roughly one season in five. 2024 is the worst season for both GLM and
XGB, at 44.5%.

## What the dissertation prints, and the corrected values

`results/walkforward_2004_2024_dissertation_check.csv` reproduces every printed number from the
published CSV, using each table's own rule. All 187 numbers match to the printed precision
(20 in the betting table, 32 in the multimodel table, 15 in the CLV table, 120 in the per-season
tables). Each row also carries the corrected value.

**`figures/out/betting_performance_table.tex`** (bets when p > 0.524 or p < 0.476; the generator,
`nfl_analytics/analysis/generate_betting_metrics.py:141-142`, pays +0.91 / -1.10):

| Row | Printed: bets, win, ROI, Sharpe | Corrected |
|---|---|---|
| GLM | 3,826, 51.0%, -7.45%, -1.22 | 3,826, 51.0%, -2.60%, -0.45 |
| MEAN(GLM+XGB) | 3,963, 50.7%, -8.00%, -1.31 | 3,963, 50.7%, -3.12%, -0.54 |
| XGB | 4,419, 50.1%, -9.39%, -1.54 | 4,419, 50.1%, -4.44%, -0.77 |
| GLM+XGB (stack) | 1,139, 50.0%, -9.41%, -1.54 | 1,139, 50.0%, -4.46%, -0.77 |
| GLM+STATE (stack) | 1,100, 50.0%, -9.50%, -1.56 | 1,400, 52.4%, +0.09%, +0.02 (state fixed) |

The table's "Sortino 0.00" is an artifact. With a constant loss of -1.10, the downside deviation
is 0, and the generator returns 0. The Sharpe ratio scales i.i.d. per-bet returns by sqrt(272)
and is not a portfolio statistic. It is shown only because the table prints it.

**`figures/out/multimodel_comparison_table.tex`** uses the harness's own definitions. Accuracy is
(p >= 0.55) == outcome. ROI covers home bets only at p >= 0.55, with decimal payout 1.91. The
GLM, XGB and Mean(GLM+XGB) rows are unchanged. The state rows change as follows:

| Row | Printed: Brier, log-loss, accuracy, ROI | Sign-fixed state, stack refit |
|---|---|---|
| Stack(GLM+XGB+State) | 0.2515, 0.6966, 51.1%, -13.5% | 0.2510, 0.6953, 51.3%, -5.5% |
| Stack(GLM+State) | 0.2517, 0.6971, 51.2%, -7.9% | 0.2507, 0.6948, 51.2%, -10.2% |
| Stack(XGB+State) | 0.2519, 0.6976, 51.0%, -17.2% | 0.2511, 0.6957, 51.2%, -7.9% |
| Mean(GLM+XGB+State) | 0.2615, 0.7176, 49.7%, -8.3% | 0.2556, 0.7051, 52.0%, -1.4% |

These ROIs rest on 85-141 home bets for the stacks. They also ignore the away side, so they say
nothing about skill. Use the tables above instead.

**`figures/out/clv_distribution_table.tex`**: the generator's "CLV" is 10,000 x (p - 0.5) when
home covers and 10,000 x (0.5 - p) otherwise. It measures outcomes, not prices. For GLM, the mean
is +14.9 bps, the SD 820.6, t = 1.35, two-sided p = 0.18. Every bet is assumed at the closing
price, so true closing-line value is zero by construction. The other rows have t between 0.16 and
0.75.

**`figures/out/per_season_top3_table.tex` and `oos_record_table.tex`**: the GLM and XGB rows
reproduce. "Stack" is the state-bugged `ens_stack_glm_xgb_state`. Its printed "ROI +0.0%" in
every season means no bets at all: the stack never reached p >= 0.55 as published. After the fix
it does so once, in 2017.

**Text that cites these tables** (abstract `main.tex:358,371,389,407`; chapter 9
`:110-112,168,188-195,724`; chapter 12 `:7,15-18,55,157,161,166-168,205`) should read: *GLM
walk-forward 2004-2024: 51.0% ATS on 3,826 bets (95% CI 49.4-52.6%), ROI -2.6% at -110, no model
better than a coin flip on Brier (best 0.2507 vs 0.2500); no closing-line value was measured.*
These are October 2025 walk-forward results, not results from the "December 2025 leakage
investigation" the text credits. See `retraction_ledger.md`, rows C06-C08, for every location.

## Limitations

- The GLM and XGB features cannot be regenerated, so their as-of construction is unaudited. A
  leak would have inflated these numbers, not hidden an edge.
- The harness included 237 playoff games. Grading the 5,292 regular-season games alone
  (`games_subset` = regular season only) changes no conclusion: GLM 50.6% [49.3%, 52.0%], the
  fixed three-model stack 51.4% [50.0%, 52.7%], mean(GLM, state) 51.5%, always-road 51.3%. Every
  ROI is between -5.6% and -1.6%, and the best upper bound is +0.5%.
- With only 21 seasons, the season-block bootstrap is coarse. Its ROI intervals are about as
  wide as the Wilson intervals imply. At -110, ROI = 1.909 x win rate - 1, so GLM's +/-1.3 points
  of win rate correspond to about +/-2.5 points of ROI, which matches [-6.1%, -1.2%] around
  -3.7%.
- Eighteen columns and two betting rules were graded. The p-values in the tables are per column
  and not adjusted.

## Reproduce

```bash
cd /home/dro/rice/nfl-analytics
uv run --no-sync python scripts/final_evaluation/regrade_walkforward_2004_2024.py
```

This runs in under 20 s from a clean shell. The DB settings come from `DB_*` environment
variables, with the defaults in `nfl_analytics/utils/db_config.py`, and the session is forced
read-only. The script writes `results/walkforward_2004_2024_{label_check, reproduction, metrics,
per_season, dissertation_check, predictions_corrected}.csv`. The last file holds the per-game
state-space mean and SD, the published and fixed state probabilities, and the six rebuilt
ensembles, for regenerating the chapter 9 tables.
