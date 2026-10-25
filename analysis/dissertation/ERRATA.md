# Dissertation errata

Written 2026-10-24 during the repository salvage pass (branch `repair/2026-10`, HEAD `5c597af`).
No chapter has been edited. This file records what is wrong in the dissertation built from
`main/main.tex`, why, what the evidence supports instead, and what a corrected build needs.

- **Sources.** Line numbers are for the current `.tex` sources, re-checked with `rg` on
  2026-10-24. Since the February 2026 build (`412be16`) the sources have changed only through
  the documentation prune (`0c5cbc9`) and the `py/` → `nfl_analytics/` path codemod (`2b5bdc4`,
  `cefd8b6`). Chapter and appendix files are cited by basename (each is unique under
  `analysis/dissertation/`); generated tables as `figures/out/<name>`. Full paths are in §8.
- **Code references** are repo-relative and refer to commit `5c597af`. Other salvage commits
  may fix, move or delete that code (a sign fix to `nfl_analytics/models/state_space.py` was in
  progress in the working tree when this was written).
- **The PDF.** `main/main.pdf` (tracked, 6.6 MB, built at `412be16`) contains every error below.
- **Numbers.** Values in the "Corrected" column were recomputed for this file on 2026-10-24,
  from tracked files or from the dev database after that day's repair of `games.game_type` and
  `games.kickoff` (scores and lines were not touched and match nflverse for all 7,276 games
  1999–2025). §7 says how each was computed. Quoted claims are the dissertation's own numbers.
  Where something could not be reproduced the row says so.
- **Convention.** `games.spread_close` is nflverse `spread_line`: the expected home margin,
  positive when the home team is favored (`2018_01_BUF_BAL` = 7.5; BAL won 47–3 and covered).
  Home covers iff `(home_score − away_score) − spread_close > 0`; pushes are excluded. ATS
  figures here use that rule and −110 pricing: +100/110 per unit on a win, −1 on a loss,
  break-even 52.38%.

## 1. What went wrong

The dissertation's central conclusion is right: models built on public data do not beat the
NFL closing spread. Much of the evidence printed for and around that conclusion is wrong, in
six recurring ways.

1. **Spread sign convention.** About twenty code paths graded covers with the sportsbook rule
   `margin + spread_close > 0`, which is correct only when a home favorite carries a negative
   number. Under that rule both the side bet and the graded result mostly follow the sign of
   the spread, so agreeing with the market looks like skill: a "model" that copies the closing
   line wins 77.4% of 2019–2025 games (§7 R21). The rule produced the retracted December 2025
   power-ratings result (74.9% ATS, +43% ROI), which the dissertation still prints on 15 lines,
   relabelled as "holdout accuracy" or "straight-up accuracy". It also produced the state-space
   ATS equation and table, the `State` column of the 2004–2024 multimodel tables, the 910-trade
   "76.8%" backtest, the reinforcement-learning rewards, a Diebold–Mariano table built in a
   notebook, and the ATS SQL in Chapter 10. The check that would have caught it is the pooled
   home cover rate: 49.05% of 7,081 decided games with the correct rule, 59.09% with the wrong one.
2. **Target and same-game leakage.** Several results used information from the game being
   predicted or from the season being reported: Ridge-residual features (ADR-003) and CQL
   rewards (ADR-016), both already retracted in the text; same-season team aggregates in the
   2019–2025 game ensemble; the predicted game's own EPA in the RL logged dataset; the predicted
   game's pass attempts in every player-props model; semantic news features read as of the run
   date for games already played; and early stopping on the season then reported as validation.
3. **Hard-coded, synthetic and mock content.** §3 lists 18 tables and 5 figures in the build
   that were typed into scripts ("plausible", "synthetic but realistic", "(mock)"), computed
   from random numbers or simulated data, computed with a formula that returns the wrong
   probability (one table), or cannot be traced to a generator and input in the repository
   (three tables). 29 of the "40 experiments" return literal constants. The Chapter 8
   acceptance table's "41 weekly evaluations" were made by adding noise to a single test run.
4. **Prop lines simulated from outcomes.** The database holds no historical player-prop lines.
   The props backtests set the line to the actual yards plus uniform noise, or to the model's
   own prediction, so every props ROI, win rate and Sharpe ratio is meaningless.
5. **Vig charged twice.** The betting-metrics script pays +0.91 per unit on a win and −1.1 on a
   loss, which moves break-even to 54.7%. The headline "51.0% win rate, −7.5% ROI, Sharpe −1.22"
   is −2.6% ROI and Sharpe −0.45 at −110. It is also credited to the wrong study: it comes from
   the October 2025 GLM season walk-forward over 2004–2024, not the December 2025 leakage
   investigation.
6. **"CLV" and Brier mislabelled.** The "+14.9 bps CLV" is the model's probability distance from
   0.5, signed by the outcome. The backtests assume every bet is placed at the closing line, so
   true closing-line value is zero by construction, and the statistic itself is not
   distinguishable from zero (t = 1.35, p = 0.18). The "best-in-class" ATS Brier score of 0.2515
   is worse than a constant 0.5 forecast (0.2500); "matches Vegas (0.250)" compares against that
   coin-flip value.

Why it happened: the spread convention had no single definition and no test pinned to a real
game; no number had to pass a base-rate or plausibility check before it went into the text;
tables were produced by scripts that read no data, with no record of which script, input and
commit made each one; design targets and projections were written in the past tense; and
figures were copied between chapters and documents instead of being regenerated. The
convention fix and its history are in `docs/POSTMORTEMS.md` §4. The repository now has
`nfl_analytics/utils/spread_convention.py` (pinned to real games) and
`nfl_analytics/evaluation/market_benchmark/` (grades predictions against the close and refuses
implausible results).

## 2. Errata

IDs are referred to in §3–§6. "R" numbers point to the recomputations in §7.

### A. Headline numbers

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| A1 | `main.tex:411` (Abstract); `chapter_09_results_discussion.tex:88` (twice), `:102`, `:116`, `:120`; `chapter_12_conclusion.tex:19`, `:179`; `chapter_11_lessons_learned.tex:12`; `section_3_5_phase2_advanced_stats.tex:349`; `subsection_phase5_exhaustive_features.tex:238` | "74.9% spread prediction accuracy on the 2025 holdout (p < 0.001)"; ch09:120 "predict which team covers the spread" | 74.9% is the December 2025 power-ratings ATS rate (`docs-archive-2026-10:docs/milestones/2025_HOLDOUT_VALIDATION_COMPLETE_DEC9.md:21`: "Win Rate 74.9%, ROI +43.0%, 179 bets", method "2024 point differential + 2.5 home field advantage") graded with the sportsbook rule. +43.0% is just the −110 ROI of a 74.9% win rate. The documented method graded the same way scores 71.8–78.5% in every season 2019–2025 (all games, edge thresholds 0–3 points): the signature of the bug. The exact 179-bet sample cannot be rebuilt: the edge threshold is not documented, and the 2025 lines in the database at the time may have included the EPA-derived placeholders written on 2025-11-21 for weeks 12–18 (POSTMORTEMS §4). | Delete. Graded correctly, the same method is 43.1% ATS in 2025 (122/283) and 48.0–50.5% in each season 2019–2024. The 2025 holdout result is Game Outcome v2: 139–145 ATS (48.9%, 95% CI 43.2–54.7) on 284 decided games, and 105–102–1 (50.7%, ROI −3.2%) on bets with an edge beyond ±2 points. | R1, R5 |
| A2 | `appendix_o_model_specifications.tex:794`, `:816`, `:825`; `appendix_m_architecture_evolution.tex:42`; `appendix_n_orthogonalization.tex:1118` | "74.9% straight-up winner prediction accuracy (2025)" | Relabelling of A1; neither the power-ratings model nor Game Outcome v2 picked anywhere near 74.9% of 2025 winners. | Delete. Walk-forward power ratings picked 146/221 (66.1%) of 2025 winners against 141/221 (63.8%) for the closing-line favorite on the same games; Game Outcome v2 picked 162/284 (57.0%) against 187/284 (65.8%) for the favorite. | R19, R5 |
| A3 | `main.tex:358`, `:371`, `:389`, `:407`; `chapter_01_intro.tex:33`, `:45`, `:71`, `:107`, `:142`; `chapter_09_results_discussion.tex:5`, `:86`, `:110–112`, `:120`, `:195`, `:724`; `chapter_12_conclusion.tex:7`, `:15–18`, `:161`, `:166–168`, `:183`, `:205`; `chapter_11_lessons_learned.tex:12`, `:202`; `figures/out/betting_performance_table.tex`; the rest of the 67 lines listed in §8 (A3) | "Cleaned baseline from the December 2025 leakage investigation: 51.0% win rate, −7.5% ROI, Sharpe −1.22 (3,826 bets)" | (1) The payoff charges the vig twice: `nfl_analytics/analysis/generate_betting_metrics.py:141-142` books +0.91 on a win and −1.1 on a loss (break-even 54.7%). (2) Wrong provenance: the table was generated on 2025-10-07 (`2ece208`, last changed `5a75005`, 2025-10-19) from `analysis/results/multimodel_predictions.csv` (`e7f36bd`, 2025-10-04), the 2004–2024 season walk-forward. (3) The labels in that file are correct (100% agreement with `margin − spread_close > 0`). | "GLM season walk-forward, 2004–2024 (October 2025): 1,952–1,874 on 3,826 bets, 51.0% (95% CI 49.4–52.6); ROI −2.6% at −110; Sharpe −0.45 with the same annualisation; p = 0.21 against 50% and p = 0.09 against break-even. No skill." The other rows of `tab:betting-performance` at −110: MEAN(GLM+XGB) −3.1%, XGB −4.4%, GLM+XGB stack −4.5%, GLM+STATE stack −4.6% (the last uses the sign-bugged State column, B3). | R2 |
| A4 | `chapter_09_results_discussion.tex:383` | "Sharpe ratio improvement: −1.22 → +0.35 (bootstrap estimate)" | Starts from the double-vig Sharpe; the "+0.35" is a projection with no data behind it. | Delete. | R2, F7 |
| A5 | `main.tex:358`, `:375`, `:386`, `:389`, `:391`, `:402`, `:405`, `:407`; `chapter_01_intro.tex:33`, `:41`, `:46`, `:142`; `appendix_m_architecture_evolution.tex:35`, `:63`, `:538`, `:645`, `:667` | "Player props showed +2.96% ROI (53.89% win rate)"; "player props markets show exploitable edges"; the thesis of "differential market efficiency across bet types" (ch01:41, main:405) | No market test of props was possible: `odds_history` has only `closing`, `opening` and `game_lines` rows and there is no prop-line table. The props backtests price bets against the actual yards plus U(−15, 15) (`nfl_analytics/backtest/historical_validation_5way.py:158-172`, `nfl_analytics/backtest/historical_validation_4way_optimized.py:633-638`) or against the model's own prediction (`nfl_analytics/backtest/ensemble_5way_backtest.py:314-317`). The 4-way ensemble behind +2.96% also included the leaked CQL component (ADR-016). | "No market test of player props was possible because no historical prop lines exist; props results are limited to error and coverage metrics (and see D2)." Drop the differential-efficiency thesis. `chapter_12_conclusion.tex:7`, `:24`, `:205` and `chapter_11_lessons_learned.tex:12` already call +2.96% invalidated; they should add this reason. | R6 |
| A6 | `main.tex:389`; `chapter_12_conclusion.tex:138`; `appendix_o_model_specifications.tex:823` | "The simple power ratings model achieves 67.6% ATS in a 3–4 point edge sweet spot" | It is a Game Outcome v2 bucket, not power ratings (`docs-archive-2026-10:docs/investigation/ws3_signal_analysis.md:11`, `:106`), chosen after seeing the 2025 results. | Delete, or report as noise: v2 2025 ATS by absolute edge [0,1) 15–16, [1,2) 19–25, [2,3) 17–13, [3,4) 23–11 (67.6%, CI 50.8–80.9), [4,5) 9–12, [5,7) 15–34 (30.6%), ≥7 41–34. One of seven post-hoc buckets and not monotone. | R5 |

### B. Sign convention

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| B1 | `section_state_space_model.tex:93–97` (`eq:ats-probability`) | P(home covers) = Φ((θ_h − θ_a + HFA + ℓ)/σ) | With ℓ = `spread_close` (nflverse sign) the sign of ℓ is wrong. The code it documents had the same error: `nfl_analytics/models/state_space.py:225` (docstring), `:228` (`mean_margin + spread`), `:278` (label `margin_true + spread > 0`). | Φ((θ_h − θ_a + HFA − s)/σ) with s = `spread_close`, or define ℓ as the sportsbook line (ℓ = −`spread_close`). | code |
| B2 | `section_state_space_model.tex:104–128` (`tab:state-space-test-performance`), `:130` | "ATS accuracy 84.6% (2024 validation) and 74.7% (2025 weeks 1–7); ATS Brier 0.134 / 0.168" | Graded with the sportsbook rule. The validation column is also in-sample: `nfl_analytics/models/train_state_space.py:211` fits on train + validation, `:220` scores the validation set. | Out of sample (fit 2018–2023 → 2024; fit 2018–2024 → 2025 weeks 1–7): 48.4% ATS (n = 281) and 49.1% (n = 108), ATS Brier 0.282 and 0.275, both worse than 0.25. The repo's buggy evaluation gives 70.2% and 71.3% on the same splits. Out of sample the 2024 win accuracy is 58.2% and MAE 12.00 (the table's in-sample 75.1% and 9.15). | R3 |
| B3 | `figures/out/multimodel_comparison_table.tex` (Stack(GLM+XGB+State), Stack(GLM+State), Stack(XGB+State), Mean(GLM+XGB+State)); `figures/out/betting_performance_table.tex` (GLM+STATE); `figures/out/oos_record_table.tex` and `figures/out/per_season_top3_table.tex` (Stack(All)); `chapter_09_results_discussion.tex:159`, `:174`, `:177`, `:712` | State-space and stacked results, 2004–2024; "stacked ensembles outperform individual models"; "per-season Brier stable (0.2486–0.2511)" | The `state` column was priced by `predict_prob_ats` (`nfl_analytics/backtest/harness_multimodel.py:197-207` → `nfl_analytics/models/state_space.py:228`). A walk-forward refit with Φ((μ + spread_close)/σ) reproduces it to 2.2e-16. It correlates +0.795 with `spread_close` (mean p = 0.826 for home favorites of 7+, 0.213 for home underdogs of 7+) and has Brier 0.3197. | With the sign fixed the state model has Brier 0.2749, accuracy 50.6%, 50.9% on 4,934 bets, ROI −2.9%; Mean(GLM+XGB+State) moves from Brier 0.2615 to 0.2556. Every model and ensemble stays worse than a constant 0.5 forecast, so "stable per-season Brier" means stably uninformative. Stacks need re-fitting (not done here). GLM and XGB rows are valid. | R4, R2 |
| B4 | `chapter_04_baseline_modeling.tex:63` (`figures/out/dm_test_table.tex`, `tab:dm-test`) | "Decayed weighting beats recent-only on the 2024 holdout (DM p = 0.04)" | Written by `notebooks/00_timeframe_ablation.qmd:302`, `:309`, whose target is `margin + spread_close > 0` (`:123`). | Invalid. Rerun with the correct label or delete. | code |
| B5 | `section_game_ensemble.tex:197–202` | "910 trades, 76.8% win rate ... further confirm overfitting" | The stored trades (`models/ensemble/backtest/trades.csv`) agree 99.9% with the sportsbook rule, and their predictions used whole-season aggregates of the season being predicted (`git show c43e449:py/ensemble/game_outcome_ensemble.py`, lines 47–100 and 117–130). | Graded correctly the trades go 552–331–27 (62.5%, CI 59.3–65.6): still implausible, because of the same-season leak. Attribute the result to the sign bug plus target leakage, not overfitting. | R15 |
| B6 | `chapter_10_production_deployment.tex:121–122`, `:313`, `:642` | ATS margin SQL `AVG(home_score - away_score + spread_close)` | Wrong sign (the road-team version at `:393–394` is correct). | Use `home_score - away_score - spread_close`. With the correct sign (regular season 2010–2024), equal-rest TNF home teams average −0.44 ATS points (n = 247); the dissertation's sign gives +2.7 to +5.8 for every rest group. | R10 |
| B7 | `chapter_06_rl_design.tex:52`, `:103`, `:112` (inputs `figures/out/ope_grid_table.tex`, `tab:ope-grid`, and `figures/out/rl_agent_comparison_table.tex`) | Off-policy evaluation grid "Accept = Yes, median DR = 0.2301"; DQN vs PPO training comparison | The input `data/rl_logged.csv` (`nfl_analytics/rl/ope_gate.py:11`, `nfl_analytics/rl/dqn_agent.py:31`) grades rewards with the old rule (r > 0 agrees with `margin + spread_close > 0` on 652 of 652 bet rows, with the correct rule on 76.1%), and its `p_hat` uses the predicted game's own EPA (`home_epa_mean`/`away_epa_mean` equal each team's mean EPA in that game for the three games checked; corr(p_hat, home margin) = +0.898). | Invalid: delete, or rerun on a leak-free dataset with correctly graded rewards. `tab:rl-vs-baseline` (`chapter_06_rl_design.tex:331–353`) does not use this file (see H4). | R12 |

### C. Betting economics and framing

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| C1 | `chapter_09_results_discussion.tex:159`, `:168`, `:177`, `:188`, `:724`; `chapter_12_conclusion.tex:55`, `:156`; `chapter_05_advanced_bayesian.tex:271`; `alternative_approaches.tex:30`, `:41`; `chapter_10_production_deployment.tex:754`; `figures/out/benchmark_comparison_table.tex`, `figures/out/benchmark_significance_table.tex` (ch09:170–171) | "Brier 0.2515 places us in the top tier ... outperforms FiveThirtyEight ELO (0.253) and matches Vegas (0.250)"; "captures 96% of achievable calibration (0.250 theoretical minimum)"; "XGBoost v2 (Brier 0.2515)" | These are ATS (home-cover) Brier scores. 0.25 is what a constant 0.5 forecast scores, not a minimum; the base rate scores 0.2498. Every model in `tab:multimodel-comparison` is worse (0.2515 to 0.2643; State 0.3197). The external numbers are typed into `nfl_analytics/analysis/benchmark_comparison.py:26-96` without sources and refer to other targets; the significance test uses a fixed formula for the standard error (`:111`). 0.2515 is the three-model stack, not XGBoost v2. | "No model shows ATS skill: every ATS Brier score is at least 0.2515, worse than a constant 0.5 forecast (0.2500), and every log-loss is at least 0.6966 (coin flip 0.6931)." The GLM-to-best-stack gain is 0.0037 Brier between two no-skill forecasts. Delete the benchmark tables or rebuild them on the same target with cited sources. | R2 |
| C2 | `figures/out/clv_distribution_table.tex` (ch09:437); `chapter_09_results_discussion.tex:189`, `:193`, `:724`; `chapter_12_conclusion.tex:55`, `:112`, `:157`; `chapter_01_intro.tex:131` (definition) | "Positive CLV (+14.9 bps): beating closing lines"; "we're on the right side of closing line value more often than not"; CLV defined as "difference between market-implied and model-implied edge" | `nfl_analytics/analysis/generate_betting_metrics.py:24-42` computes (p − 0.5) signed by the outcome, a directional score, not closing-line value. Every bet is assumed at the closing line, so real CLV is 0 by construction. The GLM mean is 14.9 bps with SD 820.6 over 5,529 games: t = 1.35, p = 0.18; no other model reaches t = 0.8. | "No closing-line value was measured." Rename the statistic (e.g. outcome-signed edge) and report it as not different from zero. CLV compares the price a bet was placed at with the closing price; fix ch01:131. | R2 |
| C3 | `chapter_09_results_discussion.tex:180`, `:633`, `:710`, `:718`; `appendix_consolidated.tex:383`; `chapter_07_uncertainty_risk_betting.tex:228` | "Market microstructure features contribute over 40% of CLV capture in ablation studies"; models "excel" at CLV capture | Cites the mock core-ablation table (F6); no CLV was measured (C2). | Delete. | §3 |
| C4 | `chapter_08_simulation.tex:209–216`, `:271–272`, `:305` | "CLV in passing weeks +12.4 bps (95% CI 8.1–16.7) vs +5.8 bps in failing weeks, p = 0.018" | No generator; the "weeks" are synthetic (F10). At a per-game SD near 820 bps, 5,529 games give a 95% half-width of ±21.6 bps; the printed ±4.3 bps would need about 140,000 games. | Delete. | R2, R11 |
| C5 | `chapter_12_conclusion.tex:101` | "the 4.5% vigorish creates a fixed barrier that can be overcome with a superior model" | Contradicts the evidence and the chapter's own conclusion. | Delete, or state it as an open question. | — |
| C6 | `section_postseason_validation.tex:64` | "The full-season ATS ROI of −3.2% is an improvement over the cleaned GLM baseline (−7.5%) ... postseason play may create exploitable edges" | Wrong baseline (A3); the playoff "edge" is 7–3 on 10 bets. | Compare with the GLM walk-forward's −2.6% (different seasons and models; both no-skill). Delete the playoff-edge suggestion: 7–3 has one-sided p = 0.17, and on the 12 non-Super-Bowl playoff games the model's MAE (8.43) was still worse than the line's (8.33). | R5 |
| C7 | `chapter_07_uncertainty_risk_betting.tex:44–48` (`eq:cvar-prog`) | The CVaR program "limits tail risk while allowing Kelly-like growth on the interior" | As written the program minimises CVaR alone: when zero stakes are feasible it returns zero stakes unless some stake mix is profitable even in its worst tail scenarios. The implementation has the same form (`nfl_analytics/risk/cvar_lp.py:83-107`: non-negative stakes, cap, budget, objective CVaR only). | Maximise expected return subject to a CVaR budget, or add a return term or floor. | code |

### D. Player props

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| D1 | `section_5way_ensemble_production.tex:163–165`, `:171`, `:183`, `:263`, `:307–309`; `subsection_5way_ensemble_validation.tex:111–121`, `:146` (`temporal_roi_evolution.pdf`); `canonical_window_and_state_space_approval.tex:326–327`; `section_ensemble_evolution.tex:23–46` (`tab:ensemble-evolution`), `:130–131`, `:219–221`, `:556`; `appendix_m_architecture_evolution.tex:6`, `:22–40` (`tab:performance-trajectory`), `:63`, `:138–139`, `:194–195`, `:538`, `:645`, `:652–653`, `:667` | "+5.6% ROI, 59.7% win rate, Sharpe 2.3 (bootstrap CIs)"; "+6.2%, 60.5%, Sharpe 2.5"; "v5.x 5.5–7.5% ROI"; "+5–7% expected"; per-version ROI and win rates | Lines simulated from outcomes or from the model's own prediction (A5). The printed figures are not even consistent with each other: at −110 a 59.7% win rate is +14.0% ROI, not +5.6%; 60.5% is +15.5%, not +6.2%. | Delete every props ROI, win rate and Sharpe ratio. Keep error and coverage metrics only with D2's caveat. | R6, R7 |
| D2 | `main.tex:376`; `chapter_09_results_discussion.tex:106–107`; `chapter_12_conclusion.tex:20`; `section_3_5_phase2_advanced_stats.tex:350`; `appendix_o_model_specifications.tex:834–849` (`tab:props-holdout-2025`); `v3_ensemble_section.tex:6`; 47.77 at `canonical_window_and_state_space_approval.tex:102` and `appendix_c_ensemble_ablation.tex:74`; full list in §8 (D2) | "Props ensemble MAE 44.25 yards; equal weights beat Dirichlet (44.91)"; "Scenario D achieves the lowest MAE (47.77)" (ADR-007 window choice); "competitive with market" | The props models condition on the game's realised pass attempts: `R/train_hierarchical_v2_canonical.R:53`, `:134` and `R/generate_2025_predictions_all_v2.R:47`, `:96` take `stat_attempts` from the game being predicted. A formula with no model, same-game attempts × 2024 league yards per attempt, scores 45.28 on 2025 weeks 1–12 (n = 330) and 48.44 on weeks 1–5 (n = 155). No market comparison exists (A5). The 47.77 window was itself chosen on the 2025 holdout (`docs/adr/007-canonical-training-window.md:116–124`). | "Props MAE figures use same-game attempts and are not forecasts." Forecasts without same-game information score 55.79 (constant 2024 league mean) and 61.02 (player's season-to-date mean, else prior-season mean) on the same 330 games. The 44.25 vs 44.91 difference is within noise of the no-model baseline. | R13 |

### E. Leakage, look-ahead and unverifiable results

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| E1 | `section_data_leakage_investigation.tex:8–177` (numbers at `:8`, `:14`, `:40`, `:58`, `:86–105`, `:113–126`, `:140`, `:146`, `:162`, `:170–172`) | "62.4% → 60.2% win rate"; "62.8% on 2024 (179/285)"; "4.47M% → 51,106% ROI"; "spread-only baseline 52.2%"; "season-level features → 55–57% expected" | The producing backtests (`nfl_analytics/backtest/full_historical_backtest.py`, `clean_historical_backtest.py`, `clean_backtest_FIXED.py`, `recent_seasons_test.py`) read `models/state_space/predictions.csv`, `data/processed/features/asof_clean_backtest.csv` and `data/processed/features/asof_FIXED_v2.csv`, none of which is in the repository, and the first two are on the sign-bug list in POSTMORTEMS §4. | The same designs (logistic regression, same features and splits) on as-of features rebuilt from the repaired database with the current code: train < 2020 → test 2020–2024 50.7% (n = 1,376, log-loss 0.695, worse than a coin flip); train 2022–2023 → test 2024 46.3% (n = 281). Rewrite the section as a sign-bug-plus-leakage case study with these figures; delete 52.2% and "55–57% expected" (unverifiable). | R16 |
| E2 | `section_unified_baseline_comparison.tex:30`, `:43` | "5-Component Ensemble: MAE 8.3, AUC 0.715, 62.1% ATS — Overfit" | An MAE of 8.3 against the closing line's 9.7 in the same table is the signature of leakage, not overfitting; the game-ensemble predictions of that period used whole-season aggregates of the predicted season (B5). | Relabel as target leakage. | B5 |
| E3 | `section_unified_baseline_comparison.tex:34`, `:51`, `:107`; `chapter_04_baseline_modeling.tex:203`; `section_dirichlet_stacking.tex:272–273`, `:348`, `:621–623`, `:647`, `:656`; `section_ensemble_evolution.tex:454`; `section_3_5_phase2_advanced_stats.tex:288–302`; `chapter_09_results_discussion.tex:103` | v6.0 "54.5% ATS cover, +4.0% ROI (2022–2024)"; "54.1% ATS, MAE 10.1 = production ceiling"; "equal weights 55.2%"; "Spread MAE 4.5"; "Spread MAE 10.25, near-market" | No evaluation script for v6.0 on game spreads was found in the 2026-10 survey. The components were trained on the ADR-007 canonical window (2018–2024, `docs/adr/007-canonical-training-window.md:67`), so 2022–2024 is in-sample, and they include the leaked XGBoost v4 (ADR-003). A spread MAE of 4.5 is impossible next to the closing line's 9.49 over 2022–2024. | Delete, or replace with a market_benchmark run on seasons the components never saw. | R20 |
| E4 | `section_semantic_stack.tex:29`, `:34`, `:368`, `:376–394` | "Semantic features underperformed the baseline (58.5% → 51.8%, p = 0.21)"; "+1.5–2.5% win-rate uplift target" | Semantic features were read as of the run date for games already played: `nfl_analytics/features/game_feature_extractor.py:823` calls `extract_features` without `as_of`, and `nfl_analytics/semantic/features/semantic_features.py:462` defaults to `datetime.now()`. | "The semantic A/B test is not interpretable (look-ahead in the semantic features)." The no-benefit direction stands as a non-result; drop the uplift target. | code |
| E5 | `chapter_03_data_foundation.tex:314–328`, `:678`; `chapter_09_results_discussion.tex:588–618` (`tab:weather-effects`, `tab:extreme-weather`); `chapter_12_conclusion.tex:39`, `:54` | "Wind does not suppress scoring (r = 0.0038, p = 0.90, 1,017 games)"; "high wind (> 40 kph) ... null effects"; "Weather has no predictive value" | The analysis used the database weather table, whose Meteostat ingest multiplies wind speeds that Meteostat already reports in km/h by 3.6 (`nfl_analytics/weather_meteostat.py:199-200`), so every km/h threshold is mislabelled. With nflverse game-day wind the null does not replicate. | Over the same seasons (2020–2025, outdoor or open roofs, n = 990): r(wind, total points) = −0.096 (p = 0.002) and r(wind, total − closing total) = −0.081 (p = 0.011). Over 1999–2025 (n = 5,121) the total-minus-line residual falls 0.185 points per mph (SE 0.035), and games at 15 mph or more went under 57.0% of the time (384/674). Restate: wind lowers scoring slightly. The residual association uses observed, not forecast, wind and a post-hoc threshold, so it is not a validated betting edge. Also fix the temperature p-value (G6). | R9 |
| E6 | `section_postseason_validation.tex:8`, `:34`, `:68` | "Production model trained on 2022–2024 data"; "playoff performance (spread MAE 8.43) outperformed regular season (11.14)"; "the in-season evaluation benefited from live pipelines feeding NextGen tracking stats, ESPN QBR ..." | The model is fit on 2022–2023 with early stopping on 2024 (`nfl_analytics/models/game_outcome_model_fixed.py:156-159`); its stored `best_score` (10.4499) equals the reported 2024 validation MAE, so the 2024 figures are tuned on 2024. The playoffs were easier for everyone. None of v2's three models splits on any of its 46 NextGen or 20 QBR features. | "Fit on 2022–2023, early-stopped on 2024." On the 12 playoff games the line's MAE was 8.33, so the model was still worse than the market (8.43). Missing NextGen/QBR data cannot explain the compressed predictions; the spread model does split on nine line-movement features, which had no 2025 odds snapshots to read. | R5, R14, R20 |
| E7 | `section_fourth_down_cql.tex:22` | State includes "Quarter (1–4, overtime)" | `plays.quarter` is NULL for all 108,866 fourth-down plays in 1999–2024, and `nfl_analytics/rl/fourth_down_mdp.py:148-152` maps NULL to the second quarter, so the quarter feature is constant in training. | State the limitation, or backfill the quarter from nflverse and retrain. | R18 |
| E8 | `chapter_05_advanced_bayesian.tex:86–110` (`tab:bayes-performance`), `:179–186`, `:226–227`; `section_unified_baseline_comparison.tex:22–27` | Bayesian "ATS 52.7%, win rate 54.0%, ROI +1.59%" (2024); comparison rows "state-space 51.8% / 51.2% ATS", "Bayesian hierarchical 52.7%", "GameOutcomeModel 53.1%" | No generator found. One season of about 163 bets: 88/163 is p = 0.35 against 50%. The state-space rows do not match the out-of-sample, correctly graded 2024 figure (48.4%, B2). Game Outcome v2's 2024 is its early-stopping season (E6). | Mark as unverified and all within noise of 50%, or recompute with market_benchmark. `chapter_05_advanced_bayesian.tex:110` already calls +1.59% superseded, but against the wrong baseline (A3). | R3, R7 |
| E9 | `appendix_consolidated.tex:433–467` (`tab:bootstrap-worst-case`), `:669–863` (incl. `:851`, `:862`); `appendix_c_ensemble_ablation.tex:286` | Bootstrap stress test "on actual 2024 season bet outcomes"; "+1.59% → +5.0–7.0% ROI"; "+$260 expected profit"; "~0.5% ROI gain" | `nfl_analytics/simulation/bootstrap_stress_test.py` reads a results JSON that is not in the repository; the ROI path starts from a superseded baseline and adds projected gains. | Delete, or label as historical projections. | — |

### F. Hard-coded, synthetic and mock content (details and generators in §3)

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| F1 | `chapter_09_results_discussion.tex:134–153`; `section_experiment_framework.tex:4`, `:10`, `:28`, `:76`, `:119`, `:143`, `:172`; `section_cvae_experiments.tex:79`, `:92`, `:124`, `:205–228`; `section_causal_experiments.tex`; `section_experiment_augmented_rl.tex:8`, `:36–40`, `:150–171`, `:244–247`, `:283` | "40 experiments executed against real NFL data (not synthetic/placeholder data)"; "9.5% edge when starters rest, 8.5% live betting ROI, 6.2% ROI"; experiment-augmented RL "edges +2.2% to +9.5%" | 29 of 40 experiment functions in `nfl_analytics/experiments/experiment_executor.py` return dicts of numeric literals (8 mixed, 3 computed). Examples: `:444` `live_bet_roi: 0.062` under "Simplified - would need play-by-play" (`:437`); `:478` `0.085` under "Would need injury data - simplified" (`:474`); `:559` `resting_starters_edge: 0.095`. 21 return a literal sample count. No live odds exist. | Delete, or move to an appendix titled "design sketches (no results)". Keep `causal_001/003/004` only if re-run and reported with intervals. | R8 |
| F2 | `chapter_09_results_discussion.tex:442–444` (`tab:zero-weeks`); `chapter_12_conclusion.tex:45`, `:68`, `:159`; `chapter_07_uncertainty_risk_betting.tex:145`; `chapter_06_rl_design.tex:381` | "21% of weeks produce zero bets due to OPE gating or simulator rejection" | Rates typed into `nfl_analytics/analysis/generate_zero_bet_weeks.py` ("generates plausible zero-bet week statistics"); no policy was run. | Delete. | §3 |
| F3 | `chapter_06_rl_design.tex:385` (`tab:utilization_adjusted_sharpe`); `chapter_12_conclusion.tex:56` | "RL improved Sharpe by ~0.1–0.2 over Kelly baselines (1.73 vs 1.42–1.58)" | Typed into `scripts/utils/generate_missing_tables.py`; the table note's formula (Sharpe × utilisation) contradicts `chapter_06_rl_design.tex:380` (× √(W/52)); no RL policy was trained on clean data (ADR-016, B7). | Delete. | §3 |
| F4 | `chapter_07_uncertainty_risk_betting.tex:51` and `chapter_06_rl_design.tex:386–387` (`cvar_benchmark_table`); `chapter_08_simulation.tex:248` (`slippage_model_table`), `:109` (`mc_convergence_table`) | Portfolio CVaR comparison; slippage fits on "24,567 Pinnacle fills"; Gelman–Rubin and ESS diagnostics | Typed constants; the generators read no data. The CVaR table is internally impossible: in every row the worst week is milder than the CVaR95. | Delete. | §3 |
| F5 | `chapter_09_results_discussion.tex:682–697` (`tab:diebold-mariano`, `tab:bootstrap-ci`, `tab:multiple-testing`) | "Ensemble vs GLM DM −18.8 (p < 0.0001)"; "ensemble Brier 0.1125 [0.1099, 0.1152]"; "CLV > 0, p = 0.007" | Computed from random labels with "predictions" built from those labels; p-values typed in. | Delete. The real GLM-to-best-stack difference is 0.0037 Brier between two no-skill forecasts; the real "CLV" p is 0.18. | R2, §3 |
| F6 | `chapter_09_results_discussion.tex:635–638` (`tab:core-ablation`) | "Reweighting improves Brier from 0.2552 to 0.2515; microstructure adds ~0.2% ROI" | Generator caption says "(mock)": multimodel rows relabelled as RL/IQL/t-copula configurations. | Delete. | §3 |
| F7 | `chapter_09_results_discussion.tex:314–398` (`figures/tikz/roi_projection_chart.tex` is input at line 371); `chapter_03_data_foundation.tex:248–268` (`tab:roi_projections_features`) | "Missing-signal integration moves the system to +0.9% to +2.3% ROI" | Guessed per-feature gains added linearly to the double-vig baseline. | Delete. Chapter 3's table is labelled speculative; if kept, start from −2.6% and say it is not a result. | — |
| F8 | `chapter_07_uncertainty_risk_betting.tex:202–228` (`tab:multi-model-reliability`, Table 7.5 in the build); `:53–77` (`tab:teaser-copula-impact`); `chapter_12_conclusion.tex:43`, `:65` | "XGBoost Brier 0.1715 ... all classes ROI −7.5% to −6.8%"; teaser EVs of −50% to −89%; "ignoring dependence overstates teaser EV by 2–5 percentage points" | Table 7.5 is the hand-typed fallback of an `\IfFileExists` whose file does not exist; it contradicts `tab:multimodel-comparison` (XGB 0.2643, GLM 0.2552) and cites a "2023–2025 holdout" that is not in that data. The teaser table comes from `nfl_analytics/analysis/teaser_pricing_comparison.py`, which returns the probability that a leg fails. | Delete both tables and the copula-EV claim. | §3 |
| F9 | `chapter_07_uncertainty_risk_betting.tex:150–155` (`fig:bankroll-hist`), `:164–169` (`fig:bankroll-trajectories`) | "Distribution of final bankroll outcomes under the drawdown-screened policy" | Pure simulations with a fixed 51% win probability and an assumed edge; no policy and no bets. | Label as illustrations with stated assumptions, or delete. | §3 |
| F10 | `chapter_08_simulation.tex:195–216`, `:261–306` (`tab:sim-acceptance`, `fig:sim-acceptance-rates`, `fig:sim-acceptance-vs-live`) | "Pass rates across 41 weekly evaluations (2023–2025): margin 0%, key mass 78%, dependence 63%, home win rate 100%"; "failing acceptance correlates with degraded live metrics" | The 41 weeks (18 + 18 + 5) were synthesised by adding noise to one run; every cell of the table reproduces exactly from that synthetic file. The "live" relationship is simulated from the pass rate. | Report the single real run (`data/sim_acceptance.csv`): margin EMD 3.63 (threshold 2.0) fails, total EMD 4.68 (3.0) fails, 3-point key mass 0.068 (0.05) fails; 6-, 7- and 10-point masses, Kendall τ and home win rate pass. The simulator fails acceptance; that conclusion stands, the weekly rates and the CLV link do not. | R11 |
| F11 | `section_causal_inference.tex:16`, `:22`, `:133` (`tab:treatment-effects`), `:145`, `:159`, `:186–199` (`tab:causal-betting-backtest`), `:240`, `:257` | "Causal-adjusted +5.1% ROI, 56.8% win rate; injury returns 62%, +8.1%"; "3–6% ROI improvements observed in historical backtests"; "the 2022 McCaffrey trade created exploitable edges" | No generator in the repository. The ROI column contradicts the win rates at −110 (62.0% → +18.4%, 56.8% → +8.4%). A win rate for a yards model needs prop lines, which do not exist. The treatment table lists 7 coaching changes, the backtest 40. | Delete (unverifiable). | R7 |
| F12 | `chapter_05_advanced_bayesian.tex:122–137` (`tab:ensemble-performance`) | "Ensemble (both agree) 55.0%, +2.60% ... validates the wisdom of crowds principle" | The text itself says the XGBoost side is "synthetic XGBoost with 5% noise". | Delete the table and the conclusion at `:137`. | — |

### G. Factual errors

| ID | Location | Claim | Why it is affected | Corrected statement or value | Evidence |
|---|---|---|---|---|---|
| G1 | `section_postseason_validation.tex:4`, `:23`, `:36`, `:38`; `chapter_12_conclusion.tex:21`; `chapter_09_results_discussion.tex:116` | "Super Bowl LIX"; "The model predicted a near toss-up (NE +1.4, ... win probability 49.4% home)" | The 2025-season final is Super Bowl LX (`2025_22_SEA_NE`, 2026-02-08, SEA 29, NE 13; NE the designated home team, `spread_close` −4.5). "NE +1.4" reads as NE the underdog; the model's expected home margin was +1.4 (NE favored by 1.4) while its win probability for NE was 49.4%. | "Super Bowl LX"; "expected margin NE by 1.4 points (win probability 49.4%)". | R5, R6 |
| G2 | `section_postseason_validation.tex:21–25` (`tab:full-season-2025`) | "Win Prob Acc.: regular season 57.0%, playoffs 58.3%, full season 56.5%"; "ATS Rate 48.2% / 48.8%" | The playoff cell is the complement: the model picked 5 of 12 playoff winners (41.7%). Regular season 156/271 (57.6%; 156/272 = 57.4% counting the tie as a miss); full season 162/284 (57.0%; 162/285 = 56.8%). The ATS rates count the push as a loss. | Playoffs 41.7%; regular season 57.6%; full season 57.0%; ATS 139–145–1 (48.9% of decided games). The spread and total MAE cells reproduce. | R5 |
| G3 | `chapter_10_production_deployment.tex:134–145` ("Example output"), `:284–306`, `:322`, `:325–349`, `:615` | "On Thursday, November 14, 2024, the New York Giants defeat the Philadelphia Eagles 21–17 despite closing as +7 underdogs" (`2024_11_NYG_PHI`); invented query output and SHAP analysis of that game | No such game. The Thursday 2024-11-14 game was `2024_11_WAS_PHI` (PHI 26–18, PHI favored by 4.5); both 2024 NYG–PHI games were PHI wins (28–3 at NYG, 20–13 at PHI). The real TNF query output has 247 equal-rest TNF games and none with a ±3-day rest gap. | Replace with real query output, or label the scenario as hypothetical. Weekday filters must use `kickoff AT TIME ZONE 'America/New_York'` (TNF kicks off after midnight UTC). | R6, R10 |
| G4 | `chapter_09_results_discussion.tex:205–259` (`tab:missing_indexes`) | "8 critical missing indexes" on `pbp`, `schedules`, `player_stats`; `idx_pbp_player_time`; "p95 latency 1850 ms → 145 ms" | None of these tables or that index exist in the database (play-by-play is `plays`; schedules are `games`). | Delete, or re-measure on the real tables. | R6 |
| G5 | `appendix_o_model_specifications.tex:794`, `:796–813` (`tab:spread-holdout-2025`), `:819–821` | "Prior-season point differentials and 2.5-point HFA"; season rows 2020–2024 at 51.4/50.4/51.6/52.8/50.5%; "the 5 prior seasons averaged 51.3%" | The 2025 row (57.7%) and the 2020–2025 total (51.8%, −1.1%) reproduce from the within-season walk-forward model (`nfl_analytics/models/power_ratings_model.py`), not from prior-season ratings (which give 43.1% in 2025, A1). The 2020–2024 rows do not reproduce, and the printed columns sum to 686/1,310 = 52.4%, not the printed 51.8%. | Replace with the reproduced records: 2020 103–103–0 (50.0%), 2021 108–109–4 (49.8%), 2022 108–105–7 (50.7%), 2023 100–112–9 (47.2%), 2024 120–98–3 (55.0%), 2025 127–93–1 (57.7%, CI 51.1–64.1); total 666–620–24 (51.8%, CI 49.1–54.5, ROI −1.1%). Add that 2019 was 45.1% and 2019–2024 49.7%. | R1, R7 |
| G6 | `figures/out/weather_effects_comparison_table.tex:12` | "Temperature: r = 0.0548, p = 0.7357 (n = 1,021)" | For r = 0.0548 and n = 1,021 the two-sided p is 0.080 (`chapter_09_results_discussion.tex:593` and `chapter_12_conclusion.tex:54` already print 0.08). | p = 0.080. | R7 |
| G7 | `main.tex:361`; `literature_gaps.tex:53` | "300,000+ NFL plays" | The `plays` table has 1,279,628 rows (1999–2025). | "About 1.28 million plays." | R6 |
| G8 | `main.tex:438` | "dependencies are pinned via requirements.txt (Python) and renv.lock (R)" | Neither exists at the top level (only per-service `api/requirements.txt` and `dashboard/requirements.txt`); Python dependencies are pinned by `pyproject.toml` and `uv.lock`; R packages are not pinned. | State that. | git |
| G9 | `chapter_09_results_discussion.tex:38` | XGBoost v4 Ridge Residuals "DISABLED per ADR-015" | ADR-015 covers the Kalman/GP features; Ridge Residuals is ADR-003. | "per ADR-003". | `docs/adr/` |
| G10 | `chapter_12_conclusion.tex:140` | "Model performance varies systematically by week: Weeks 10–13 show the best ATS performance ... Weeks 14–18 the worst" | v2 2025 ATS by phase: weeks 1–4 29–35, 5–9 33–38, 10–13 32–26 (55.2%, CI 42.5–67.3), 14–18 37–41; no phase differs from 50%, and 14–18 is not the worst. | Delete. | R5 |
| G11 | `chapter_03_data_foundation.tex:328` | "XGBoost accuracy improved marginally from 94.9% to 95.3%" | Implausible for any NFL game outcome (the 2025 closing-line favorite won 65.8%); no source found. | Delete. | R5 |

### H. Already retracted or caveated in the text

| ID | Location | Status |
|---|---|---|
| H1 | `chapter_09_results_discussion.tex:18–42`; `chapter_01_intro.tex:54`, `:69`; `appendix_n_orthogonalization.tex` | Ridge Residuals "88% improvement" retracted as target leakage. Keep as a case study; fix the ADR number (G9) and the 74.9% sentence at `appendix_n_orthogonalization.tex:1118` (A2). |
| H2 | `main.tex:396`, `:409`; `chapter_01_intro.tex:71`, `:119`; `subsection_5way_ensemble_validation.tex:134` | CQL "100% win rate" retracted (ADR-016). Keep. |
| H3 | `chapter_09_results_discussion.tex:44–55`; `appendix_incremental_xgb_investigation.tex` | Incremental-XGBoost baseline error. Keep. |
| H4 | `chapter_06_rl_design.tex:331–353` (`tab:rl-vs-baseline`) | Random 50.5%/−3.5% and market 50.9%/−2.8% on 2023–2024 are consistent with −110 pricing. Keep; "Max DD 0.0%" for a losing strategy cannot be right and needs recomputing. |
| H5 | `section_profit_optimization_results.tex:44–92`; `chapter_05_advanced_bayesian.tex:110`; `section_ensemble_evolution.tex:134`; `appendix_m_architecture_evolution.tex:42` | Already marked superseded or invalidated, but the caveats cite the double-vig baseline (A3). |
| H6 | `chapter_12_conclusion.tex:7`, `:24`, `:205`; `chapter_11_lessons_learned.tex:12` | +2.96% props ROI already called invalidated; add that no prop lines exist (A5). |

## 3. Hard-coded, synthetic and mock tables and figures

Generator lines refer to commit `5c597af`.

| Table or figure | Used at | Generator (file:line) | What it is |
|---|---|---|---|
| `figures/out/zero_weeks_table.tex` (`tab:zero-weeks`) | `chapter_09_results_discussion.tex:444` | `nfl_analytics/analysis/generate_zero_bet_weeks.py:9` ("This script generates plausible zero-bet week statistics"), `:24-50` (rates per season); also `nfl_analytics/analysis/generate_results_tables.py:193-219` ("placeholder") | Typed rates (28/22/17/22/17%, total 21%). |
| `figures/out/utilization_adjusted_sharpe_table.tex` | `chapter_06_rl_design.tex:385` | `scripts/utils/generate_missing_tables.py:117-132` (same file opens its first table with "Create synthetic but realistic data", `:26`) | Typed Sharpe ratios and utilisation. |
| `figures/out/cvar_benchmark_table.tex` | `chapter_07_uncertainty_risk_betting.tex:51`; `chapter_06_rl_design.tex:386-387` | `scripts/utils/generate_missing_tables.py:168-178` | Typed; worst week milder than CVaR95 in every row. |
| `figures/out/slippage_model_table.tex` | `chapter_08_simulation.tex:248` | `nfl_analytics/analysis/generate_slippage_model.py:11-25` (reads no data; fill counts typed, including 24,567 Pinnacle fills) | Typed. |
| `figures/out/mc_convergence_table.tex` | `chapter_08_simulation.tex:109` | `nfl_analytics/analysis/generate_slippage_model.py:116-125` | Typed Gelman–Rubin, ESS and MCSE values. |
| `figures/out/sim_acceptance_table.tex` (`tab:sim-acceptance`) | `chapter_08_simulation.tex:261`; text `:195-216` | No generator in the repository writes this version (`nfl_analytics/analysis/generate_slippage_model.py:65-112` writes a different, also typed one). All eight cells reproduce from `analysis/results/sim_acceptance.csv`, which `nfl_analytics/sim/expand_acceptance_by_week.py:13-57` creates by adding unseeded Gaussian noise to the single run in `data/sim_acceptance.csv` (weeks 18 + 18 + 5 = 41). | Synthetic weekly records. |
| `figures/out/sim_acceptance_rates.png`, `figures/out/sim_acceptance_vs_live_perf.png` | `chapter_08_simulation.tex:281-306` | `nfl_analytics/viz/improved_acceptance_plots.py:28-33` (reads the synthetic weekly file), `:111-130` (CLV simulated as 10 + 15 × pass rate + N(0, 5)); `notebooks/90_simulator_acceptance.qmd:34` writes the same files from `analysis/results/live_metrics.csv`, made by `nfl_analytics/sim/generate_live_metrics.py:226-236` (noise added to the model spread, which is then used as the closing line) | Synthetic. |
| `figures/out/bootstrap_ci_table.tex`, `figures/out/diebold_mariano_table.tex` | `chapter_09_results_discussion.tex:685`, `:691` | `nfl_analytics/analysis/statistical_significance.py:230-247` ("Simulate data for demonstration"; random binomial labels; "predictions" = label × 0.7 + noise) | Computed from random data. |
| `figures/out/multiple_testing_table.tex` | `chapter_09_results_discussion.tex:697` | `nfl_analytics/analysis/statistical_significance.py:361-371` | Typed p-values (including "CLV > 0: 0.007"). |
| `figures/out/core_ablation_table.tex` | `chapter_09_results_discussion.tex:638` | `nfl_analytics/analysis/generate_results_tables.py:222-262` (caption "Core ablation grid (mock)" at `:239`) | Multimodel rows relabelled as ablation configurations. |
| `figures/out/benchmark_comparison_table.tex`, `figures/out/benchmark_significance_table.tex` | `chapter_09_results_discussion.tex:170-171` | `nfl_analytics/analysis/benchmark_comparison.py:26-96` (typed values, external figures without sources; the "Stacked" row carries the GLM's 14.9 bps), `:111` (fixed-formula standard error) | Typed. |
| `figures/out/causal_treatment_effects_table.tex`, `figures/out/causal_betting_backtest_table.tex` | `section_causal_inference.tex:133`, `:188` | None found | Unverifiable; ROI inconsistent with win rates (F11). |
| `figures/tikz/roi_projection_chart.tex` | `chapter_09_results_discussion.tex:371` | Hand-written TikZ | Projection drawn as a chart. |
| Inline `tab:multi-model-reliability` (Table 7.5) | `chapter_07_uncertainty_risk_betting.tex:206-220` | Hand-typed fallback of the `\IfFileExists` at `:203` (its target does not exist) | Contradicts `tab:multimodel-comparison` (F8). |
| Inline `tab:teaser-copula-impact` | `chapter_07_uncertainty_risk_betting.tex:57-75` | `nfl_analytics/analysis/teaser_pricing_comparison.py:85-104` (docstring says P(margin > −teased line); the code returns P(margin < −teased line)), `:239` (payout 1.5 labelled −120) | Wrong probability and payout. |
| Inline `tab:ensemble-performance` | `chapter_05_advanced_bayesian.tex:122-135` | Text at `:122`: "Bayesian + synthetic XGBoost with 5% noise" | Simulated. |
| `figures/bankroll_hist.png`, `figures/bankroll_trajectories.png` | `chapter_07_uncertainty_risk_betting.tex:152`, `:166` | `nfl_analytics/viz/bankroll_histogram.py:20-60`, `nfl_analytics/viz/bankroll_trajectories.py:20-50` (win probability fixed at 0.51, edge drawn from N(0.014, 0.01)); committed with the PNGs in `2ece208` | Simulated from assumptions. |
| "40 experiments" tables and text | `chapter_experiments/*.tex`; `section_experiment_augmented_rl.tex` | `nfl_analytics/experiments/experiment_executor.py` (29 of 40 functions return literal metrics, e.g. `:428`, `:444`, `:478`, `:559-560`) | Typed (F1). |
| `tab:bootstrap-worst-case` | `appendix_consolidated.tex:445-458` | `nfl_analytics/simulation/bootstrap_stress_test.py` (input JSON not in the repository) | Unverifiable (E9). |

## 4. Chapter soundness

"Not re-run" means no defect was found, but the result was not reproduced in this pass.

| Chapter | Sound as written | Must change |
|---|---|---|
| Executive Summary, Abstract (`main.tex:352–413`) | Research question; the "no" for spread betting | Rewrite around §5: A1, A3, A5, A6, G7; the "differential market efficiency" thesis goes. |
| 1 Introduction | Research questions, scope, glossary apart from CLV | Thesis and findings (`:33`, `:41`, `:45–47`, `:71`, `:107`, `:142`); CLV definition (`:131`). |
| 2 Literature review | Sound; contains no project results | Duplicate bib entries (`wang2024causal`/`wang2024causaldeep`, `xia2024sports`/`xia2024sportsllm`); `literature_gaps.tex:53` play count. |
| 3 Data foundation | Ingestion, schema and as-of design; data counts other than G7 | `section_3_5_phase2_advanced_stats.tex:288–302`, `:349–350`; `subsection_phase5_exhaustive_features.tex:238`; `chapter_03_data_foundation.tex:328` (G11), weather test `:314–328` and `:678` (E5), ROI projection table `:248–268` (F7). |
| 4 Baseline modeling | GLM and state-space theory (after B1); GNN negative result (not re-run) | B1, B2, B4, B5, E2, E3, E8; `v3_ensemble_section.tex:169`, `:211`, `:353` projections. |
| 5 Advanced Bayesian | Hierarchical model theory; BNN dead-end narrative (not re-run) | E8, F12, D1, D2, E3, C1 (`:271`). |
| 6 RL design | MDP and OPE theory; `tab:rl-vs-baseline` (H4) | B7 (OPE grid, DQN/PPO); F3; F4; experiment-augmented RL (F1). |
| 7 Uncertainty, risk, betting | Kelly maths; conformal and evidential method descriptions (not re-run; props inputs carry D2) | C7 (CVaR program); F4, F8, F9; profit-optimisation caveat numbers (H5). |
| 8 Simulation | Skellam, copula and key-number methods; the conclusion that the simulator fails acceptance | F10 (report the single real run); C4; F4. |
| 9 Results | 2025 postseason validation (after G1, G2, E6, C6); GLM/XGB walk-forward rows read as a null (C1, A3); Ridge and baseline-error case studies (H1, H3); BNN and GNN negatives (not re-run) | A1, A3, A4, A6, B3, C1–C3, D1, D2, E1, E5, F1, F2, F5–F7, F11, G4, G9. |
| 10 Production deployment | Architecture description | B6, G3, E4, C1 (`:754`), A3 (11 lines, §8). |
| 11 Lessons learned | Lessons, in spirit | Numbers at `:12`, `:198`, `:202` (A1, A3). |
| 12 Conclusion | — | Rewrite around §5: A1, A3, A5, A6, C1, C2, C5, E5, F2, F3, F8, G1, G10. |
| Experiments (`chapter_experiments/`) | Experiment-design methodology, as a design | All results (F1). |
| Appendices | I (BNN), N (as a leakage case study), incremental-XGB investigation, C (MAE parts, with D2), O model specifications outside `:787–849`, notation, acronyms | O `:787–849` (A2, A6, D2, G5); M ROI tables (D1, A2); consolidated stress test and projections (E9) and `:383` (C3); `alternative_approaches.tex:30`, `:41` (C1). |

## 5. The defensible thesis

What the evidence supports: **public-data models do not beat the NFL closing spread, and an
evaluation pipeline without convention tests, base-rate checks and table provenance reports
edges that do not exist.** The contribution is a well-documented negative result plus a case
study of how one pipeline produced false edges and caught some of them.

| Strand | Evidence | Status |
|---|---|---|
| No ATS skill over 21 seasons | GLM season walk-forward 2004–2024 (`analysis/results/multimodel_predictions.csv`, labels verified): 51.0% on 3,826 bets (CI 49.4–52.6), −2.6% ROI; every ATS Brier ≥ 0.2515 (coin flip 0.2500) | Regenerate the betting, multimodel, OOS and per-season tables with the correct payoff and without the State rows (A3, B3, C1). |
| No edge on a true holdout | Game Outcome v2, 2025 (`ml.game_outcome_predictions`): MAE 11.05 vs the line's 9.67; 48.9% ATS; log-loss 0.687 vs the de-vigged moneyline's 0.607; −3.2% ROI on the 2-point filter | As written in `section_postseason_validation.tex` after G1, G2, E6, C6; add intervals. |
| A zero-feature baseline does no better | Walk-forward power ratings: 49.7% in 2019–2024, 51.8% in 2020–2025; 2025's 57.7% is one season (2019 was 45.1%) | `appendix_o` after G5, with intervals. |
| The market is hard to model | The Dixon–Coles/copula simulator fails its own acceptance test (single real run) | Chapter 8 after F10. |
| How false edges arise | Sign convention: 74.9% → 43.1%, state-space 70–71% → 48–49%, a market copy "wins" 77.4% under the wrong rule, pooled cover 49.05% vs 59.09%, three recurrences (POSTMORTEMS §4). Leakage: Ridge residuals, CQL, same-season aggregates (910 trades still 62.5% after regrading), same-game EPA, same-game attempts (props MAE 44.25 vs a no-model 45.28), early stopping on the reported season, look-ahead semantic features. Evaluation: wrong baseline, double vig, mislabelled CLV, Brier framed without the 0.25 reference, simulated prop lines, typed tables. | Ch9 case studies (H1, H3) as written; the rest is new material for a rewritten chapter built from this file and POSTMORTEMS §4. |
| Guardrails | `spread_convention.py` pinned to real games; market_benchmark plausibility caps and home-cover guardrail; migration 104 for the cover views | Describe in the rewrite. |

Parts that support this as written: Chapter 2; Chapter 3's data and as-of architecture (minus
the listed lines); `section_postseason_validation.tex` (minus G1, G2, E6, C6); the Chapter 9
Ridge-residual and baseline-error case studies; the BNN and GNN negative results (not re-run);
Chapter 11's lessons; Appendices I, N and the incremental-XGB investigation.

## 6. What a corrected build requires

Nothing below has been done. The order is roughly by impact.

**Retractions and corrected numbers**

- [ ] Remove 74.9% from all 15 lines (A1, A2). Cite Game Outcome v2's 2025 record where a holdout is meant, and the walk-forward power-ratings records with intervals where power ratings are meant.
- [ ] Regenerate `tab:betting-performance` and `tab:clv-distribution` through `nfl_analytics/evaluation/market_benchmark`, or fix `nfl_analytics/analysis/generate_betting_metrics.py:141-142` to +100/110 and −1; replace −7.5% and −1.22 on all the lines in §8 (A3), and credit the October 2025 GLM walk-forward instead of the December 2025 leakage investigation.
- [ ] Fix `eq:ats-probability` and the state-space code (B1), rerun `harness_multimodel` with `--base-models glm,xgb,state`, re-fit the stacks and regenerate `tab:multimodel-comparison`, `tab:oos-record`, `tab:per-season-top3` and `tab:betting-performance`, or drop every State row (B3).
- [ ] Replace `tab:state-space-test-performance` with out-of-sample, correctly graded figures and drop the in-sample validation column (B2).
- [ ] Restate every Brier and log-loss claim against the no-skill values 0.25 and 0.693; drop or rebuild the external benchmark tables with sources on the same target (C1).
- [ ] Remove "CLV" claims or rename the statistic; say that no closing-line value was measured; fix the CLV definition (C2–C4).
- [ ] Remove every props ROI, win rate and Sharpe ratio and the props-edge thesis (A5, D1). Label props MAE as conditioned on same-game attempts, or rerun with pre-game attempt projections and compare against the 55.79 constant baseline (D2).
- [ ] Remove or quarantine every item in §3, and the claims built on them (F1–F12).
- [ ] Chapter 8: replace `tab:sim-acceptance` and both acceptance figures with the single real run (F10); delete the CLV-by-week claims (C4).
- [ ] Weather (E5): fix `nfl_analytics/weather_meteostat.py:199-200` and re-ingest, or use nflverse game-day wind; rewrite `chapter_03_data_foundation.tex:314–328` and `:678`, `chapter_09_results_discussion.tex:588–618`, and `chapter_12_conclusion.tex:39` and `:54`; fix the temperature p-value (G6).
- [ ] Chapter 9 data-leakage section: rewrite as sign bug plus leakage with the R16 figures, or mark it unverifiable (E1).
- [ ] Chapter 4: relabel the 5-component ensemble as leakage (E2), reattribute the 910-trade result (B5), delete or recompute the v6.0 figures out of sample (E3), delete the v3 projections, regenerate or delete `tab:dm-test` (B4).
- [ ] Chapter 10: fix the ATS SQL sign (B6); replace the NYG–PHI scenario and the example outputs with real query output, using `kickoff AT TIME ZONE 'America/New_York'` for weekday filters (G3); delete or re-measure the index study (G4); state that the semantic A/B is not interpretable (E4).
- [ ] Postseason section: Super Bowl LX (G1); playoff win-probability accuracy 41.7% and the other cells (G2); training window, playoff comparison with the line, and the NextGen/QBR explanation (E6).
- [ ] Appendix O: replace `tab:spread-holdout-2025` with the reproduced records and intervals and fix the method description (G5); remove 67.6% (A6); caveat `tab:props-holdout-2025` (D2).
- [ ] Chapter 6: drop or rerun the OPE grid and the DQN/PPO comparison on leak-free data (B7); recompute "Max DD" in `tab:rl-vs-baseline` (H4).
- [ ] Chapter 7: fix `eq:cvar-prog` (C7); delete Table 7.5's fallback, the teaser table and the copula-EV claim (F8); label or delete the bankroll figures (F9); correct the baseline in the profit-optimisation caveats (H5).
- [ ] Chapter 5: delete `tab:ensemble-performance` (F12); mark `tab:bayes-performance` and the comparison table unverified (E8); delete "96% of achievable calibration" (C1).
- [ ] Factual fixes: G7 play count, G8 dependency statement, G9 ADR number, G10 season-phase claim, G11 XGBoost accuracy, C5.
- [ ] Rewrite the Executive Summary, Abstract, Chapter 1 thesis and contributions, Chapter 11's numbers and Chapter 12 around §5.

**Provenance and checks**

- [ ] Give every generated table a header naming its generator, input files (with hashes) and commit, as `figures/out/ope_grid_table.tex` already does; do not build tables whose generator reads no data.
- [ ] Grade every ATS result through `nfl_analytics/utils/spread_convention.py`; widen the pre-commit spread hook to aliased forms (`margin + spread`) and to `.qmd`/`.Rmd` files.
- [ ] Pass every ATS or ROI number through the market_benchmark guardrails (pooled home cover near 49%; refuse more than 60% ATS or 15% ROI on 100+ bets) and report intervals for every rate.
- [ ] Record which seasons each model was trained, tuned and early-stopped on next to every result.

**Build**

- [ ] Fix the overfull boxes the codemod introduced (a scratch build of the current sources logs 100 overfull `\hbox` warnings against 87 in the February log; use `\path{}` or `\allowbreak` for long `\texttt` paths).
- [ ] Decide what to do with `main/main.pdf`: replace it with a corrected build that carries this errata, or untrack it and publish builds as release assets. Untrack the generated `main/main.toc`, `main/main.lof` and `main/main.lot`.
- [ ] Optionally add a CI job that runs `latexmk` (no workflow builds the dissertation today).

## 7. How the corrected values were computed

All checks ran on 2026-10-24 in read-only database sessions or from tracked files, with code at
`5c597af` except where an item says otherwise (R16). The scripts and their raw outputs are kept
with the salvage-pass records, outside the repository; each item below gives enough to redo it.

- **R1 — Power ratings and the 74.9% figure.** Input: `games` (repaired). (a) `nfl_analytics/models/power_ratings_model.backtest_season` with its defaults (HFA 2.5, half-life 10, bets from week 5, correct grading), seasons 2019–2025: 2019 88–107–9 (45.1%); 2020 103–103–0 (50.0%); 2021 108–109–4 (49.8%); 2022 108–105–7 (50.7%); 2023 100–112–9 (47.2%); 2024 120–98–3 (55.0%); 2025 127–93–1 (57.7%, Wilson 95% CI 51.1–64.1, ROI +10.2%); 2020–2025 666–620–24 (51.8%, CI 49.1–54.5, ROI −1.1%); 2019–2024 627–634–32 (49.7%). (b) The December 2025 method: rating = the team's mean point differential over the previous season's regular-season games; prediction = home rating − away rating + 2.5; all games of the season; edge thresholds 0, 1, 2 and 3 points; pushes and zero edges excluded. Wrong rule (edge = pred + `spread_close`, cover = margin + `spread_close`): 71.8–78.5% in every season 2019–2025 (2025: 206/280 = 73.6% at threshold 0). Correct rule: 2025 122/283 = 43.1%; 2019–2024 48.0–50.5% per season at threshold 0. (c) Pooled home cover, decided games: 1999–2025 49.05% (n = 7,081) correct vs 59.09% wrong; 2020–2025 49.94% vs 55.72% (n = 1,660).
- **R2 — Betting table, Brier and "CLV".** Input: `analysis/results/multimodel_predictions.csv` (5,529 games, 2004–2024); its `actual` label agrees 100% with `margin − spread_close > 0` against the repaired `games` (73.1% with the wrong rule). Bets when p > 0.524 or p < 0.476, as in `nfl_analytics/analysis/generate_betting_metrics.py:124-125`; Sharpe = mean/SD × √272, as in `nfl_analytics/analysis/generate_betting_metrics.py:45-50`. GLM: 3,826 bets, 1,952–1,874 (51.02%, Wilson CI 49.4–52.6); script payoff ROI −7.45%, Sharpe −1.22 (identical to the published table); −110 payoff ROI −2.60%, Sharpe −0.45; two-sided binomial p = 0.21 against 0.5 and 0.09 against 0.5238. Other published rows at −110: MEAN(GLM+XGB) −3.12% (Sharpe −0.54), XGB −4.44% (−0.77), GLM+XGB stack −4.46% (−0.77), GLM+STATE stack −4.55% (−0.78). Brier: stack of GLM, XGB and State 0.2515 (lowest), GLM 0.2552, XGB 0.2643, State 0.3197; constant 0.5 0.2500; base rate 0.2498. "CLV" (`nfl_analytics/analysis/generate_betting_metrics.py:24-42`, averaged over all games): GLM +14.9 bps, median +8.1, SD 820.6, t = 1.35, p = 0.18; every other column has absolute t below 0.8. A 95% half-width of 4.3 bps at that SD needs about 140,000 games; 5,529 games give ±21.6 bps.
- **R3 — State-space table.** `StateSpaceRatings()` defaults from `nfl_analytics/models/state_space.py` at `5c597af`, fit on 2018–2023 and scored on 2024 (285 games, 281 decided); fit on 2018–2024 and scored on 2025 weeks 1–7 (108). The repo's `evaluate_model` (wrong rule in probability and label): ATS accuracy 70.2% and 71.3%, ATS Brier 0.191 and 0.178. Correct rule, P(cover) = Φ((μ − s)/σ): 48.4% and 49.1%, ATS Brier 0.282 and 0.275. 2024 out of sample: win accuracy 58.2%, MAE 12.00.
- **R4 — The `state` column.** Walk-forward refit, `StateSpaceRatings(q=3.0, r=13.5)` trained on seasons before s and scored on season s, 2004–2024: Φ((μ + `spread_close`)/σ) reproduces the published column (maximum difference 2.2e-16). Published column: correlation with `spread_close` +0.795; mean p 0.826 for home favorites of 7+ and 0.213 for home underdogs of 7+; Brier 0.3197 (0.1895 against wrong-rule labels). Sign fixed: Brier 0.2749, log-loss 0.754, accuracy 50.6%, 50.9% on 4,934 bets, ROI −2.9%. Mean(GLM+XGB+State) Brier 0.2615 as published, 0.2556 with the fixed column.
- **R5 — Game Outcome v2, 2025.** `ml.game_outcome_predictions` (285 rows, one run created 2026-02-12) joined to `games`: `vegas_spread` equals `spread_close` and the scores match for 285 of 285. Spread MAE model / line: regular season (272) 11.14 / 9.72; playoff weeks 19–21 (12) 8.43 (exactly 8.425) / 8.33; Super Bowl 17.40 / 11.50; all 285 11.05 / 9.67. Total MAE 11.05 regular season, 11.11 overall. The model was closer than the line in 39.6% of games; predictions range from −2.6 to +8.7 (SD 2.3). ATS with edge = prediction − line, pushes excluded: all 139–145 (48.9%, CI 43.2–54.7); regular season 131–140 (48.3%); playoff weeks 19–21 8–4. Edge beyond ±2: regular season 98–99–1 (49.7%, ROI −5.0%), postseason including the Super Bowl 7–3 (one-sided p = 0.17), all 105–102–1 (50.7%, ROI −3.2%). Edge of 5 or more: 56–68. Absolute-edge buckets and season phases are as quoted in A6 and G10. Win-probability accuracy (p > 0.5 against the winner; one tie, `2025_04_GB_DAL`): regular season 156/271, playoffs 5/12, all 162/284; closing-line favorite 187/284. Log-loss 0.687 against 0.607 for the proportionally de-vigged moneyline (n = 284). Super Bowl: `2025_22_SEA_NE`, `pred_home_spread` +1.4, `pred_home_win_prob` 0.494.
- **R6 — Database facts.** `plays`: 1,279,628 rows; `games`: 7,276 (1999–2025). No `2024_11_NYG_PHI`; the Thursday 2024-11-14 20:15 ET game is `2024_11_WAS_PHI` (PHI 26–18, `spread_close` 4.5); 2024 NYG–PHI games: `2024_07_PHI_NYG` (PHI 28–3) and `2024_18_NYG_PHI` (PHI 20–13). No tables named `pbp`, `schedules` or `player_stats`; no index `idx_pbp_player_time`. `odds_history.market_type` takes only `closing`, `opening` and `game_lines`; no prop or player-line tables exist. Super Bowl LX: `2025_22_SEA_NE`, 2026-02-08, NE 13 – SEA 29, `spread_close` −4.5. After migration 104, `mart.game_summary` gives a home cover rate of 0.4905 over 7,081 games.
- **R7 — Arithmetic on printed values.** ROI at −110 = p × 100/110 − (1 − p). Causal table: 62.0% → +18.4% (printed +8.1%), 56.8% → +8.4% (+5.1%), 59.0% → +12.6% (+6.4%), 58.0% → +10.7% (+5.8%). 5-way props: 59.7% → +14.0% (+5.6%), 60.5% → +15.5% (+6.2%). 74.9% → +43.0%. Script payoff break-even 54.73%. Weather table: r = 0.0548, n = 1,021 → p = 0.080 (printed 0.7357); r = 0.0038, n = 1,017 → p = 0.90. Appendix O table: printed wins 686 of 1,310 games = 52.4% (printed 51.8%). Chapter 5: 88/163 → two-sided p = 0.35.
- **R8 — Experiments.** AST scan of `nfl_analytics/experiments/experiment_executor.py`: for each of the 40 `_cvae_NNN_*` and `_causal_NNN_*` methods, the values of the returned metrics dict were classed as numeric literals or expressions. 29 are all literal, 8 mixed, 3 computed (`_causal_001_weather_impact`, `_causal_003_home_field`, `_causal_004_rest_days`); 21 return a literal sample count; 9 of the all-literal functions run a database query used only for the count.
- **R9 — Wind.** nflverse `schedules/games.csv` (downloaded 2026-10-24), roofs `outdoors` or `open`, games with a wind reading and a total line, pushes on the total excluded. 2020–2025 (n = 990): Pearson r(wind, total points) = −0.096 (p = 0.002), r(wind, total − `total_line`) = −0.081 (p = 0.011). 1999–2025 (n = 5,121): r = −0.099 and −0.073; OLS slope of (total − line) on wind −0.185 points per mph (SE 0.035); wind ≥ 15 mph: unders 384/674 = 57.0% (one-sided p = 0.0002). By era at ≥ 15 mph: 57.3%, 58.2%, 55.9%, 55.5% (1999–2006, 2007–2013, 2014–2019, 2020–2025).
- **R10 — Chapter 10 query, corrected.** Regular season 2010–2024, weekday from `kickoff AT TIME ZONE 'America/New_York'`, rest gaps of −3, 0 and +3 days. Correct sign: TNF with equal rest n = 247, mean ATS margin −0.44 (SD 12.51); other games −3: 0.11 (n = 228), 0: 0.02 (n = 2,315), +3: 0.26 (n = 172); no TNF game has a ±3-day gap. With the dissertation's sign the same groups average +2.70 to +5.79.
- **R11 — Chapter 8 acceptance table.** From `analysis/results/sim_acceptance.csv` (41 weeks), grouped as the table groups tests: margin (margin and total EMD) pass 0.0%, median deviation 4.068; key mass (3/6/7/10 points) 78.0%, 0.007; dependence (Kendall τ) 63.4%, 0.093; home win rate 100.0%, 0.043. These match every printed cell. The real run is `data/sim_acceptance.csv` (eight tests, values as in F10).
- **R12 — RL logged dataset.** `data/rl_logged.csv`, 1,675 rows, 652 with `action = 1`: reward sign versus cover rules as in B7; `home_epa_mean`/`away_epa_mean` compared with `AVG(plays.epa)` by possession team for `2020_10_SEA_LA`, `2021_17_HOU_SF` and `2023_05_NYG_MIA` (equal to 7 decimals); corr(`p_hat`, home margin) = +0.898.
- **R13 — Props baselines.** `mart.player_game_stats`, passing rows with at least 10 attempts, weeks ≥ 1 (week 0 holds season totals), 2025 weeks 1–12 (the view ends at week 12): same-game attempts × 2024 league yards per attempt MAE 45.28 (n = 330; weeks 1–5 48.44, n = 155); constant 2024 mean 55.79; the player's mean in earlier 2025 weeks, else his 2024 mean, else the league mean, 61.02.
- **R14 — v2 feature use.** The XGBoost JSON dumps in `models/production/game_outcome_v2/` (`spread_model`, `total_model`, `win_prob_model`; 88/89/108 trees, best iterations 37/38/57) were parsed for split features: 0 of 46 NextGen (`_ngs_`) and 0 of 20 QBR (`_qbr_`) features appear in any split of any model; 8–9 line-movement features and 10–17 semantic features do. The spread model's stored `best_score` (10.449868) equals `metadata.json`'s validation `spread_mae`.
- **R15 — 910 trades.** `models/ensemble/backtest/trades.csv` (910 trades, stored win rate 76.8%) joined to `games`: stored outcomes agree with the wrong rule on 99.9% of decided trades. Correct rule: 552–331–27 (62.5%, CI 59.3–65.6, ROI +19.3%); by season 2022 56.9%, 2023 63.5%, 2024 64.8%, 2025 68.9%.
- **R16 — November 2025 backtest designs.** As-of features rebuilt with `python -m nfl_analytics.features.asof_features` (working tree of 2026-10-24) against the repaired database; 7,081 labelled games, pooled home cover 0.4905, label equal to the correct rule. Logistic regression on `spread_close`, `total_close`, `prior_epa_mean_diff`, `epa_pp_last3_diff`, `margin_last3_diff`, `prior_win_pct_diff`, `rest_diff`, `qb_change_diff`, `coach_change_diff` (standardised): train before 2020 → test 2020–2024 accuracy 50.7% (n = 1,376, log-loss 0.6949); train 2022–2023 → test 2024 46.3% (n = 281, log-loss 0.7095). The 2026-10 survey, on features built before the database repair, got 51.4% and 47.7%.
- **R17 — Locations.** One `rg` pattern per erratum over the 110 `.tex` files the build reads (taken from a scratch build's `main.fls`); results in §8.
- **R18 — `plays.quarter`.** Share NULL among fourth-down plays: 1.000 for 1999–2024 (108,866 plays), 0.000 for 2025 (4,290).
- **R19 — Straight-up accuracy.** Walk-forward power ratings (defaults; games from week 5; ties and pick'em lines excluded): 2025 146/221 (66.1%) vs the closing-line favorite's 141/221 (63.8%); 2024 152/221 (68.8%) vs 164/221 (74.2%).
- **R20 — Line accuracy and odds coverage.** Closing-line MAE against the home margin: 9.49 for 2022–2024 (n = 854), 9.67 for 2025 (n = 285). `odds_history` has opening and closing snapshots for 207–217 games a season in 2020–2024 and none for 2025 (one 2025 game has a `game_lines` row).
- **R21 — Why the wrong rule inflates ATS.** A "model" that copies the closing line (pred = `spread_close`) graded with edge = pred + `spread_close` and cover = margin + `spread_close`: 77.4% of 1,933 decided games in 2019–2025 (74.7–80.3% by season); 77.7% over 1999–2025. Under the correct rule its edge is zero and it never bets.

## 8. Location index

Generated by R17. Paths are relative to `analysis/dissertation/`. Hits that do not belong to
the erratum are listed as excluded.

**A1 (74.9%)** — 15 lines, pattern `74\.9\\?%`:
`main/main.tex:411`; `chapter_03_data_foundation/section_3_5_phase2_advanced_stats.tex:349`;
`chapter_03_data_foundation/subsection_phase5_exhaustive_features.tex:238`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:88,102,116,120`;
`chapter_11_lessons_learned/chapter_11_lessons_learned.tex:12`;
`chapter_12_conclusion/chapter_12_conclusion.tex:19,179`;
`appendix/appendix_m_architecture_evolution.tex:42`; `appendix/appendix_n_orthogonalization.tex:1118`;
`appendix/appendix_o_model_specifications.tex:794,816,825`.

**A3 (−7.5% / −7.45% ROI baseline)** — 67 lines, pattern `-7\.5\\?%|-7\.5\$|-7\.45\\?%`:
`main/main.tex:358,371,389,407`;
`chapter_01_intro/chapter_01_intro.tex:33,45,71,107,142`;
`chapter_03_data_foundation/chapter_03_data_foundation.tex:256,268`;
`chapter_05_advanced_bayesian/chapter_05_advanced_bayesian.tex:106`;
`chapter_05_advanced_bayesian/section_dirichlet_stacking.tex:273,348`;
`chapter_05_advanced_bayesian/section_ensemble_evolution.tex:134,454`;
`chapter_06_rl_design/chapter_06_rl_design.tex:463`;
`chapter_06_rl_design/section_experiment_augmented_rl.tex:150,160,168`;
`chapter_07_uncertainty_risk_betting/chapter_07_uncertainty_risk_betting.tex:228,238`;
`chapter_07_uncertainty_risk_betting/section_profit_optimization_results.tex:44,61,77,171`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:5,86,111,120,317,329,330,341,372,378,398,724`;
`chapter_09_results_discussion/section_5way_ensemble_production.tex:16,171,291,307`;
`chapter_09_results_discussion/section_data_leakage_investigation.tex:162`;
`chapter_09_results_discussion/section_postseason_validation.tex:64`;
`chapter_10_production_deployment/chapter_10_production_deployment.tex:750,978,982,997,1002,1007,1009,1024,1038,1068,1072`;
`chapter_11_lessons_learned/chapter_11_lessons_learned.tex:12,198,202`;
`chapter_12_conclusion/chapter_12_conclusion.tex:7,18,161,166,183,205`;
`appendix/appendix_m_architecture_evolution.tex:42`;
`figures/out/betting_performance_table.tex:11`; `figures/tikz/roi_projection_chart.tex:21`.
Sharpe −1.22 (pattern `-1\.22`, 8 lines): `chapter_07_uncertainty_risk_betting/chapter_07_uncertainty_risk_betting.tex:214`;
`chapter_07_uncertainty_risk_betting/section_profit_optimization_results.tex:62`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:112,195,383,724`;
`chapter_12_conclusion/chapter_12_conclusion.tex:168`; `figures/out/betting_performance_table.tex:11`.
Excluded: `appendix/appendix_m_architecture_evolution.tex:6` (the props "+5.5--7.5% ROI" range, D1).

**A5 (props +2.96% / 53.89% / "exploitable edges" / "differential market efficiency")** —
pattern `2\.96|53\.89|exploitable edges|differential market efficiency`:
`main/main.tex:358,375,386,389,391,402,405,407`; `chapter_01_intro/chapter_01_intro.tex:33,41,46,142`;
`chapter_11_lessons_learned/chapter_11_lessons_learned.tex:12`; `chapter_12_conclusion/chapter_12_conclusion.tex:7,24,205`;
`appendix/appendix_m_architecture_evolution.tex:35,42,63,538,645,667`; `appendix/appendix_n_orthogonalization.tex:1118`.
Excluded: `chapter_04_baseline_modeling/section_game_outcome_model.tex:246` (RMSE 12.96);
`chapter_09_results_discussion/section_causal_inference.tex:22` (F11);
`chapter_09_results_discussion/section_postseason_validation.tex:64` (C6).

**C1 (Brier 0.2515)** — pattern `0\.2515`:
`chapter_05_advanced_bayesian/chapter_05_advanced_bayesian.tex:271`;
`chapter_07_uncertainty_risk_betting/chapter_07_uncertainty_risk_betting.tex:214,223`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:168,188,614,636,724`;
`chapter_09_results_discussion/section_causal_inference.tex:159`;
`chapter_10_production_deployment/chapter_10_production_deployment.tex:754`;
`chapter_12_conclusion/chapter_12_conclusion.tex:55`; `appendix/alternative_approaches.tex:30,41`;
`figures/out/benchmark_comparison_table.tex:4`; `figures/out/core_ablation_table.tex:13`;
`figures/out/multimodel_comparison_table.tex:11`.

**C2–C4 ("CLV")** — pattern `14\.9\s*(bps|basis)|CLV\s*=\s*\+14\.9|\(CLV = \+14\.9|\+14\.9 bps|& \+14\.9|CLV capture|12\.4 bps`:
`chapter_07_uncertainty_risk_betting/chapter_07_uncertainty_risk_betting.tex:228`;
`chapter_08_simulation/chapter_08_simulation.tex:165,211,271,305,311`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:180,189,633,710,718,724`;
`chapter_12_conclusion/chapter_12_conclusion.tex:55,112,157`;
`appendix/appendix_consolidated.tex:383,872`; `figures/out/clv_distribution_table.tex:11`.

**D2 (props MAE 44.25 / 44.91 / 47.77)** — 43 lines, pattern `44\.25|44\.91|47\.77`:
`main/main.tex:376`; `chapter_03_data_foundation/section_3_5_phase2_advanced_stats.tex:350`;
`chapter_04_baseline_modeling/v3_ensemble_section.tex:6`;
`chapter_05_advanced_bayesian/section_dirichlet_stacking.tex:622,645,658`;
`chapter_05_advanced_bayesian/section_ensemble_evolution.tex:112,129,202,280,286,287,311`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:106`;
`chapter_09_results_discussion/section_5way_ensemble_production.tex:45,56,142`;
`chapter_12_conclusion/chapter_12_conclusion.tex:20`;
`sections/canonical_window_and_state_space_approval.tex:102,110,141,242,287`;
`appendix/appendix_c_ensemble_ablation.tex:74,85,88,99,100,172,208,219,720,734`;
`appendix/appendix_m_architecture_evolution.tex:35,399,615,645,667`;
`appendix/appendix_o_model_specifications.tex:149,727,843,844,849`.

**E3 (v6.0 54.5% / 55.2% / 54.1%)** — pattern `54\.5\\%|55\.2\\%|54\.1\\%`:
`chapter_03_data_foundation/section_3_5_phase2_advanced_stats.tex:298`;
`chapter_04_baseline_modeling/chapter_04_baseline_modeling.tex:203`;
`chapter_04_baseline_modeling/section_unified_baseline_comparison.tex:34,51,107`;
`chapter_05_advanced_bayesian/section_dirichlet_stacking.tex:348,621,623,647,656`;
`chapter_05_advanced_bayesian/section_ensemble_evolution.tex:454`.
Excluded: `chapter_04_baseline_modeling/section_player_to_game.tex:174` and `figures/out/oos_record_table.tex:28` (unrelated accuracies).

**E5 (weather null)** — pattern `0\.0038|r ?= ?0\.004|[Ww]eather has no predictive|not supported by the data|wind suppresses scoring`:
`chapter_03_data_foundation/chapter_03_data_foundation.tex:319,326,678`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:593`;
`chapter_12_conclusion/chapter_12_conclusion.tex:39,54`; `figures/out/weather_effects_comparison_table.tex:11`.

**F1 ("40 experiments", "real NFL data")** — pattern `40 (distinct )?experiments|real NFL data`:
`chapter_experiments/section_experiment_framework.tex:4,10,28,76,119,143,172`;
`chapter_06_rl_design/section_experiment_augmented_rl.tex:4,283`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:136,138,140,147,153`.

**F2 (zero-bet weeks)** — pattern `21\\% (zero|of weeks|abstention)|zero-bet weeks|abstention rate`:
`chapter_06_rl_design/chapter_06_rl_design.tex:381`;
`chapter_07_uncertainty_risk_betting/chapter_07_uncertainty_risk_betting.tex:145`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:443`;
`chapter_12_conclusion/chapter_12_conclusion.tex:45,68,159`; `figures/out/zero_weeks_table.tex:4`.

**G1 (Super Bowl LIX)** — pattern `Super Bowl[~ ]?LIX`:
`chapter_09_results_discussion/section_postseason_validation.tex:4,23,36,38`;
`chapter_09_results_discussion/chapter_09_results_discussion.tex:116`;
`chapter_12_conclusion/chapter_12_conclusion.tex:21`.
