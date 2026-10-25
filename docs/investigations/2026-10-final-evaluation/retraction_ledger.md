# Retraction Ledger (2026-10-24)

This ledger has one row for every number this project presented as an edge or a headline
result. Each row gives where the number appears now, what is wrong with it, the corrected value
where one exists, and how it was checked. Rows are ordered by where the number is most visible:
README, then PROJECT_STATUS, CLAUDE.md, the dissertation abstract and front matter
(`main.tex`), the dissertation chapters, docs, and finally data and model files.

**Bottom line.** No number in this project shows an edge over the NFL closing line. Every edge
claim that could be traced comes from one of six causes: the spread sign bug, target leakage or
same-game data, in-sample tuning, simulated or hard-coded results, a mislabeled metric, or a
wrong payoff. Three claims cannot be traced: the Nov-2025 backtests (C31), the weather accuracy
(C32) and the semantic A/B (C25). Their inputs or generators are gone, and the parts that can be
re-run show no edge. The numbers that do hold up all say the same thing:

- Game Outcome v2 on 2025: MAE 11.05 vs the line's 9.67, and ATS 48.9% (C02-C04).
- The README baselines: 50.0% and 49.4% ATS (C05).
- The 2004-2024 GLM walk-forward: 51.0%, ROI -2.6% (C06; see `walkforward_2004_2024.md`).
- Power ratings: 51.8% over 2020-2025 (C42).

## How the rows were built

- **Locations** come from `scripts/final_evaluation/check_claims.py`. It scans `README.md`,
  `PROJECT_STATUS.md`, `CLAUDE.md`, `docs/**/*.md`, the dissertation sources
  (`analysis/dissertation/**/*.tex` and `*.md`), `models/**/*.json` (files up to 300 KB) and
  `data/*.json`, plus named source artifacts, for each claim's patterns.
  - `results/claims_locations.csv` has every matching line.
  - `results/claims_summary.csv` has one row per claim.
  - A line counts as *flagged* when it already says the number is retracted, invalid, a sign
    bug and so on, or when it sits in a correction document (`analysis/dissertation/ERRATA.md`,
    `docs/POSTMORTEMS.md`, `docs/HISTORY.md`, `docs/investigations/2026-10-*`). This is a
    keyword heuristic.
  - The table lists where the number is still **asserted**. The CSV holds the rest.
  - This is a snapshot of the tree at about 19:45 EDT on 2026-10-24, while other salvage tasks
    were still editing docs and writing `ERRATA.md`. Re-run the script to refresh it.
- **"(not built)"** marks a dissertation file that `main.tex` never reaches through `\input` or
  `\include`. The script finds 111 built `.tex` files.
- **Short paths:**
  - `main.tex` = `analysis/dissertation/main/main.tex`
  - `chNN` = the chapter file `chapter_NN_*/chapter_NN_*.tex`; `chNN/x` = section `x.tex` in that
    chapter directory
  - `app/x` = `analysis/dissertation/appendix/[appendix_]x.tex`
  - `fig/x` = `analysis/dissertation/figures/out/x.tex`
  - `diss/` = any other path under `analysis/dissertation/`
  - `ADR-NNN` = `docs/adr/NNN-*.md`
  - `ws*` = `docs/investigations/2026-02-season-review/ws*.md`
- **Checks:**
  - "check_claims Cnn" means rows with that claim id in `results/claims_checks.csv`. The wind
    rows are in `results/claims_checks_nflverse.csv`.
  - "walk-forward" means `regrade_walkforward_2004_2024.py` and `walkforward_2004_2024.md`.
  - "Per survey" means I did not re-run the check. The named survey report (in the salvage
    scratch area) holds the evidence. The usual reason is that the inputs no longer exist.
  - Database checks ran in a read-only session against devdb01 after today's repair of
    `game_type` and `kickoff`.
- **Grading:** home covers iff margin - spread_close > 0, pushes excluded; edge = predicted home
  margin - spread_close; ROI at -110, i.e. +100/110 per win and -1 per loss. Win-rate intervals
  are Wilson 95%.

## Ledger

| ID | Claim | Where it is still asserted (flagged elsewhere) | Verdict | Corrected value | Check |
|---|---|---|---|---|---|
| C01 | Power ratings (2024 point differential + 2.5 HFA): **74.9% ATS, +43% ROI** on 179 bets of the 2025 holdout; 75.3% over 2020-25, p < 1e-68 (Dec 9, 2025) | main.tex:411 (abstract: "74.9% spread prediction accuracy"); ch09:88,102,116,120; ch12:19,179; ch03/section_3_5_phase2_advanced_stats:349; ch03/subsection_phase5_exhaustive_features:238; app/o_model_specifications:794,816,825 and app/n_orthogonalization:1118 (relabelled "straight-up accuracy"); ADR-017:111,117; ws3_model_performance:54. Flagged in 20 lines, including README.md:32, PROJECT_STATUS.md:10, CLAUDE.md:143,763 | **sign bug** | Same method graded correctly: 2025 weeks 1-14 **43.0%** (89-118-1), 2020-2025 48.5% (765-813-33). The old rule reproduces the claim on its own: 74.0-77.7% in 2025 at every edge threshold 0-3, and 71-74% in every season 2019-2025. It is also not a straight-up rate: v2 picks winners 57.0% of the time, the closing favorite 65.8% | check_claims C01, C03 |
| C02 | Game Outcome v2, 2025: **-3.2% ATS ROI on 208 bets** (stored by hand in `ml.model_metrics` run 17) | README.md:20; PROJECT_STATUS.md:8; CLAUDE.md:16,760,761,839; ch09/section_postseason_validation:56,64; ch09/subsection_5way_ensemble_validation:111,146; ch12:21; ADR-017:12; ws2_prediction_compression:137; +3 files | **correct, but only at a strict \|edge\| > 2** | \|edge\| > 2: 105-102-1, 50.7% [44.0%, 57.5%], -3.2%. All games: **139-145-1, 48.9%** [43.2%, 54.7%], ROI -6.6%. \|edge\| >= 2: -4.1%. Regular season, \|edge\| >= 1: 117-124-1, -7.3%. The same predictions graded with the old rule show 75.4% | check_claims C02 |
| C03 | Game Outcome v2, 2025: **56.5% win-probability accuracy** (given with no comparison) | README.md:21; CLAUDE.md:12; ch09/section_postseason_validation:25; ADR-017:13; ADR-018:14 | **correct count, below the market** | 162/285 = 56.8% (57.0% excluding the tie). The closing-spread favorite won **65.8%** (187/284). Log-loss 0.6871 vs 0.6070 for the de-vigged moneyline; Brier 0.2467 vs 0.2109 | check_claims C03 |
| C04 | Game Outcome v2, 2025: spread MAE **11.05 vs 9.67** for the closing line | README.md:19; PROJECT_STATUS.md:7; CLAUDE.md:12,760; ch09/section_postseason_validation:34; ADR-017:11; ws2/ws3/ws4 | **correct** | 11.05 vs 9.67 (285 games, one run, `vegas_spread` = `spread_close` for 285 of 285); regular season 11.14 vs 9.72. All predictions were created in one run on 2026-02-12, after the season. CLAUDE.md:835's "Week 16 showed 10.45" mislabels the 2024 validation MAE (C17) | check_claims C04 |
| C05 | README baselines, 2019-2024: power ratings 609-608-31 (50.0%, -4.5%); Elo 581-596-31 (49.4%, -5.8%) | README.md:29,30 | **correct** | Reproduced exactly from `data/historical_weather_2018_2024.csv`: 609-608-31, 50.0% [47.2%, 52.8%]; 581-596-31, 49.4% [46.5%, 52.2%] | check_claims C05 |
| C19 | 2019-2025 game-outcome ensemble predictions: **~71% ATS**, 81-84% straight-up. Dissertation "5-Component Ensemble 8.3 MAE, AUC 0.715"; PROJECT_STATUS item 4 says to benchmark it | PROJECT_STATUS.md:38; ch04/section_game_outcome_model:285; ch04/section_player_to_game:234; ch04/section_unified_baseline_comparison:30,59; models/ensemble/game_outcome/val_metrics.json:3; fig/baseline_comparison_table:15 (not built) | **same-season data**: full-season stats of the season being predicted (c43e449, per survey 16) | Straight-up 81.5% (2019-23) and 77.2% (2024), vs 65.5% and 70.5% for the closing favorite, which is impossible without leakage. ATS via margin = 13.5 x Phi^-1(p): 71.3% and 65.5%. The 2025 part, with partial-season stats, is 58.5% straight-up (favorite 68.9%) and 48.1% ATS. Do not benchmark it | check_claims C19 |
| C09 | Props MAE **44.25** (equal weight) vs **44.91** (Dirichlet v8.0), 2025 holdout | CLAUDE.md:9; main.tex:376 (abstract table); ch09:106; ch12:20; app/o_model_specifications:843,844,849; app/m_architecture_evolution:35,645,667; ch03/section_3_5_phase2_advanced_stats:350; ch05/section_dirichlet_stacking:622,645,658; +2 files | **same-game data**: the models use the game's own pass attempts (per survey 16) | A no-skill formula, the game's attempts x 2024 league yards per attempt (7.19), scores **45.28** (2025 weeks 1-12, n = 330). Honest pre-game baselines score 55.79 (constant) and 57.26 (player's earlier games). There are no prop lines to test against | check_claims C09 |
| C11 | Dirichlet v8.0 props weights: **31.5% MAE improvement** (89.42 -> 61.26 yards) | CLAUDE.md:35; ch05/section_dirichlet_stacking (15 lines); ch05/section_ensemble_evolution:412,423,424,454,559; docs/api/API_REFERENCE.md:49,425,641; 2 figures (not built) | **in-sample / tuned on the test season**: 108 configs chosen by 2024 MAE, the reported metric; components include the leaked XGBoost v4 | None; drop it. On 2025 the equal weights did better (C09) | per survey 16, not re-run (inputs `results/validation_v6_recovered_xgb/` missing) |
| C12 | v6.0 "games" ensemble: **54.5% ATS cover, +4.0% ROI** (2022-2024) | ch03/section_3_5_phase2_advanced_stats:298,300; ch05/section_dirichlet_stacking:272,621,656; ch05/section_ensemble_evolution:454; ADR-010:14,199; docs/api/API_REFERENCE.md:47,48; 3 files not built. Flagged at CLAUDE.md:32 | **in-sample** (components trained on 2018-2024); no generating script exists; v6 predicts passing yards, not spreads | None; the graded game model is v2: 48.9% ATS (C02) | per surveys 16 and 19, not re-run (no script) |
| C17 | Game Outcome v2 "validation" (2024): MAE **10.45**, AUC 0.691, 62.9% | CLAUDE.md:835; ch04/section_game_outcome_model:287,340,342; ADR-014:38,57,79; ADR-015:33,35; ws4_ensemble_assessment:132,136,137,352; ws7_infrastructure:172,175; models/registry.json:416,420 | **tuned on the validation set** | The early-stopping `best_score` of `spread_model.json` is 10.449868, identical to the reported validation `spread_mae` (best_iteration 37); total MAE 10.080146 vs 10.080147. No untouched 2024 number exists; v2's out-of-sample record is 2025 (C02-C04) | check_claims C17 |
| C16 | ADR-014/015: v2.1 (Kalman + GP) **10.70 vs v2 10.45**, so the Kalman/GP features have "negative value"; CLAUDE.md lesson 17 | CLAUDE.md:827; ADR-014:46,57,105,137,167; ADR-015:33,57,105 | **tuned on the test set** (v2.1 early-stopped on 2025, per survey 16) and **compares different seasons** (10.45 is v2's 2024 validation MAE) | On the same games (2025 weeks 3-13, n = 162) stored v2 scores **11.09**, vs **10.00** for the line. The test-tuned 10.70 of v2.1 shows nothing either way. Disabling Kalman/GP can stand on engineering grounds (the GP never trains) | check_claims C16 |
| C25 | "Semantic features underperformed baseline" (58.5% -> 51.8%, p = 0.21); CLAUDE.md A/B sample output; the quick A/B | CLAUDE.md:474,825; ch10:1074; ch10/section_semantic_stack:34,384,394; ch03/section_3_5_phase2_advanced_stats:351; ws3_model_performance:130; ws3_signal_analysis:167,176 | **no valid test**: no artifact backs 58.5/51.8/p = 0.21; `scripts/ab_test_semantic_2025.py` compares a prediction with itself; the quick A/B control is the constant +2.5 | Quick A/B: control = 2.5 for all 195 games. Straight-up 54.9% in both arms. ATS: control 89-89-1 (50.0%), treatment 101-93-1 (52.1% [45.1%, 59.0%]), which is noise. The semantic inputs were also look-ahead (C24) | check_claims C25; per surveys 19 and 22 |
| C29 | "Playoffs behave differently: **8.43** playoff MAE vs **11.14** regular season" (CLAUDE.md lesson 23; ADR-017 playoff feature plan) | CLAUDE.md:836; ch09/section_postseason_validation:22,34; ADR-017:14,80 | **correct numbers, wrong inference** (12 games) | 2025 weeks 19-21: model **8.42 vs line 8.33**, so the line improved as well; regular season 11.14 vs 9.72. A random 12-game regular-season subset does this well 13% of the time. Historically the line is no sharper in the postseason: MAE 10.42 (n = 309) vs 10.26 (n = 6,967), 1999-2025 | check_claims C29 |
| C06 | "Cleaned baseline from the December 2025 leakage investigation": **51.0% ATS, -7.5% ROI, Sharpe -1.22** (3,826 bets) | main.tex:358,371,389,407 (abstract); ch01:33,45,71,107,142; ch03:256; ch05:106; ch09:5,86,111,112,120,195,317,341,372,378,383,398,724; ch12:18,161,183,205; fig/betting_performance_table:11; more in other files (60 asserting lines in 21 files, 29 flagged) | **wrong payoff** (the generator pays +0.91/-1.10, charging the vig twice) and **wrong provenance** (it is the October 2025 2004-2024 GLM walk-forward) | **51.0% [49.4%, 52.6%] on 3,826 bets, ROI -2.6% [-5.1%, +0.0%], Sharpe -0.45**. Betting every game: 50.4%, ROI -3.7% [-6.1%, -1.2%] | walk-forward |
| C10 | Props **+2.96% ROI, 53.89%** win rate: "player props markets show exploitable edges" | main.tex:358,375,386,389,391,402,407; ch01:33,41,46,142; ch12:205; app/m_architecture_evolution:35,63,538,645,667; app/n_orthogonalization:1118. Flagged at ch11:12, ch12:7,24 | **simulated lines**, with a **target-leaked** CQL component (per survey 18) | None. devdb01 has no prop-line table, and `odds_history` holds only game markets (closing 59,920, opening 57,314, game_lines 32 rows). No market test of props is possible | check_claims C36; per survey 18 |
| C27 | **67.6% ATS** in the 3-4 point edge "sweet spot" (credited to power ratings) | main.tex:389; app/o_model_specifications:823; ch12:138; ws3_signal_analysis:106,118; ws5_betting_strategy:47,54,280 | **post-hoc subgroup of the test season** (selected in-sample); it is a Game Outcome v2 bucket, not power ratings | 3 <= \|edge\| < 4: 23-11-1, 67.6% [50.8%, 80.9%], one of 8 buckets. Its neighbors: 4-5 pts 9-12 (42.9%), 5-7 pts 15-34 (30.6%). All games: 48.9% | check_claims C27 |
| C42 | Power ratings walk-forward: **57.7% ATS, +10.2% ROI** in 2025 (N = 221); 51.8%, -1.1% over 2020-2025 | main.tex:372,389,391,407; ch01:47,50,142; ch05:226; ch12:205; app/o_model_specifications:809,819,821,825; ws3_model_performance:48,53,54,63 | **correct**, but one season of noise | 2025: 127-93-1, 57.7% [51.1%, 64.1%]. 2020-2025: 666-620-24, **51.8%** [49.1%, 54.5%], ROI -1.1%. A different walk-forward implementation (market_benchmark's) gives about 48% for 2025 (survey 01). Cite the multi-season figure | check_claims C42 |
| C39 | CQL bet sizing: **100% win rate, +91% ROI, "+24% improvement"** | main.tex:396,409; ch01:71; app/o_model_specifications:677; ch09/subsection_5way_ensemble_validation:134; ADR-016:17; ws5_betting_strategy:215; 3 files not built. Flagged at ch01:119 | **target leakage** (already retracted, ADR-016) | None | per ADR-016, not re-run |
| C44 | XGBoost v4 Ridge-residual features: **88.31% improvement**, MAE 4.70-4.76 | main.tex:396,409; ch01:54,69,71; app/n_orthogonalization (20 lines); app/i_bnn_investigation:28,67,461,629,688; app/c_ensemble_ablation:820; ch04/section_xgboost_v4_ridge:4; +9 files | **target leakage** (already retracted, ADR-003) | Honest MAE about 59 (ADR-003, per survey 16). Keep only as the leakage case study | per ADR-003, not re-run |
| C07 | Stacked ensemble **Brier 0.2515**: "best-in-class", "top tier", beats FiveThirtyEight (0.253), matches Vegas (0.250) | ch09:168,188,614,636,724; ch12:55; ch05:271; ch07:214,223; ch10:754; ch09/section_causal_inference:159; fig/benchmark_comparison_table:4; app/alternative_approaches:30,41; +3 files | **mislabeled metric**: 0.2515 is worse than a constant 0.5. The external figures are hard-coded and use a different target (survey 18) | Every column is above 0.2500. Published stacks 0.2515-0.2519; with the sign fix 0.2507-0.2511. The label-only base-rate forecast scores 0.2498 (0.2499 walk-forward) | walk-forward |
| C08 | **+14.9 bps closing-line value** (CLV) | ch09:189,724; ch12:55,112,157; fig/clv_distribution_table:11; fig/benchmark_comparison_table:12; app/master_todos:330,513,731 (not built) | **mislabeled metric**: it is 10,000 x (p - 0.5) signed by the result. Bets are assumed at the closing price, so CLV is zero by construction | GLM: mean +14.9 bps, SD 820.6, t = 1.35, p = 0.18. No CLV was measured | walk-forward |
| C18 | State-space ATS accuracy **84.6% (2024) / 74.7% (2025)**, ATS Brier 0.134/0.168 | ch04/section_state_space_model:125; models/state_space/val_metrics.json:9; models/state_space/test_metrics.json:9; ch09/section_state_space_results:56 (not built) | **sign bug**; 2024 is also **in-sample** (fit on 2019-2024) | The old formula and label reproduce 0.8456 exactly. The stored 0.7473 is 68/91, so it was computed on 91 of the 107 games, presumably those that had lines then; on all 107 the old rule gives 0.7383. Correct formula and label: 2024 (in-sample) 65.8%; 2025 first 107 games **52.3%**, Brier 0.270, MAE 10.64 vs the line's 9.44. Walk-forward 2004-2024: 50.6% | check_claims C18; walk-forward |
| C13 | ADR-007 canonical training window: **47.77 yards** MAE (2025 weeks 1-5) | app/c_ensemble_ablation (10 lines); app/m_architecture_evolution:399,615; app/o_model_specifications:149,727; ch05/section_ensemble_evolution:112,129,202,280,286,287,311; ch09/section_5way_ensemble_production:45,56,142; diss/sections/canonical_window_and_state_space_approval.tex (5 lines); docs/STATE_SPACE_FINAL_DECISION.md (15 lines); docs/TRAINING_WINDOW_ABLATION_RESULTS.md; +11 files | **same-game data** and **tuned on the test weeks** (per survey 16) | The no-skill attempts x league YPA formula scores **48.44** on 2025 weeks 1-5 (n = 155). A pre-game player mean scores 56.82 | check_claims C09 (weeks 1-5 rows) |
| C36 | Props ROI on simulated lines: **+5.6%** (59.7% win, Sharpe 2.3, CI [4.2, 7.0]), **+6.2%** (60.5%, Sharpe 2.5), v5.x 5.5-7.5% | ch09/section_5way_ensemble_production:164,263,307,308; ch09/subsection_5way_ensemble_validation:101,111,115,146; ch05/section_ensemble_evolution:220; docs/guides/DEVELOPER_WORKFLOW.md:183; docs/guides/TESTING_GUIDE.md:20,264; 2 files not built | **simulated**: the "line" is built from the outcome or the prediction plus noise | None. Code: `historical_validation_5way.py:166` (actual yards + U(-15, 15)), `historical_validation_4way_optimized.py:635`, `comprehensive_ensemble_backtest.py:243,248,270`, `bayesian_props_backtest_2024.py:341`. No prop-line table in devdb01 | check_claims C36 |
| C34 | **40 experiments** "executed against real NFL data", with edges up to 9.5% | ch09:138,140,147,153; ch06/section_experiment_augmented_rl:4,283; diss/chapter_experiments/section_experiment_framework.tex:28,76,119,172; diss/chapter_experiments/chapter_experiments.tex (not built) | **hard-coded** | AST scan of `nfl_analytics/experiments/experiment_executor.py`: 29 of 40 functions return only literal metrics, 8 mix literals and computed values, 3 compute them; 22 read no data | check_claims C34 |
| C33 | **21% zero-bet weeks** (OPE and simulator gating) | ch12:45,68,159; ch09:442,443; ch06:381; ch07:145; fig/zero_weeks_table:4,17; diss/results/zero_weeks_table.tex:6 | **hard-coded** | None. `nfl_analytics/analysis/generate_zero_bet_weeks.py` reads no data and says it generates "plausible" statistics | check_claims C33 |
| C35 | CVaR benchmark table and utilization-adjusted Sharpe table (RL policy **1.73** vs Kelly 1.42-1.58) | ch06:386,387; ch07:51; fig/cvar_benchmark_table:4,15; fig/utilization_adjusted_sharpe_table:3,14; app/notation_glossary:270; docs/REPRODUCIBILITY.md:130 | **hard-coded** | None. All 25 and 20 printed numbers are literals in `scripts/utils/generate_missing_tables.py`. The CVaR table is impossible as printed: in all 5 rows the worst week is milder than the 95% CVaR | check_claims C35 |
| C31 | Nov-2025 backtests: **62.4%** win and ROI 51,106% (4.47M% before the EPA fix); **60.2%** after the kickoff fix; **62.8%** on 2024; ROI 1,320,124% (`full_backtest_results.json`) | ch09/section_data_leakage_investigation:8,14,40,58,113,116,120,140,170; analysis/backtest/*.json | **not reproducible**: the cover label in that as-of builder (e12856b) was already correct, and its feature CSV was never committed | The same designs on today's as-of builder and repaired DB: train 1999-2019 -> test 2020-2024 **50.7%** [48.0%, 53.3%] (n = 1,376); train 2022-23 -> 2024 46.3% (n = 281). With the old cover label they would be 76.8% and 79.6%, so the sign bug does not explain 60%. A leak in the lost feature file is the likely cause | check_claims C31 |
| C20 | `models/ensemble/backtest`: **76.8% win rate** on 910 trades, final bankroll **9.2e33** | ch04/section_game_ensemble:200 (blamed on "overfitting"); models/ensemble/backtest/summary.json:3,4 | **sign bug** (stored outcomes agree with margin + spread 99.9% of the time) on top of **same-season data** (the C19 predictions) | Regraded: 552-331-27, 62.5% [59.3%, 65.6%]. That is still impossible, because of the leak. By season: 56.9, 63.5, 64.8, 68.9% | check_claims C20 |
| C24 | Semantic features: **+4.6% win accuracy** in the A/B that promoted v2 | ch04/section_game_outcome_model:291,349; models/registry.json:414,424 | **look-ahead data**: semantic features were extracted "as of now" for games already played (per survey 16) | None | per survey 16, not re-run (the semantic schema is absent from devdb01) |
| C30 | 2025 playoffs: **70% ATS on 10 bets, +33.6%** | ch09/section_postseason_validation:54; ADR-017:14 | **correct count**, 10 bets | Postseason including the Super Bowl at \|edge\| > 2: 7-3, 70.0% [39.7%, 89.2%], one-sided p = 0.17 | check_claims C30 |
| C38 | ROI projection: -7.5% -> **+0.9%** (realistic) / +2.3% with "missing signals"; Sharpe -1.22 -> +0.35 | ch09:314,372,383; ch08:61; diss/figures/tikz/roi_projection_chart.tex:1,43 | **speculative projection** from the wrong baseline | None. The measured baseline is -2.6% (C06), and a projection is not evidence | per survey 18 |
| C41 | OPE gate accepts the RL policy: median DR **0.2301** [0.1991, 0.2609], ESS 1674 | fig/ope_grid_table:7,15-23; docs/REPRODUCIBILITY.md:133 | **same-game data**, plus rewards graded with the old cover rule | `data/rl_logged.csv`: `p_hat` correlates +0.898 with the same game's margin; read as a margin it wins 87.6% ATS. Rewards agree with the old rule 100.0% and with the correct rule 76.1%. The OPE result is invalid | check_claims C41 |
| C40 | Teasers: two-leg EV **+1683 bps** (notebook 04), **+21.7% ROI** (`teaser_ev_oos_table`), a "+5-9% ROI lane" | ch07:59 (caption); notebooks/analysis/dissertation/figures/out/teaser_ev_oos_table.tex:9; diss/chapter_11_market_optimization/chapter_11_market_optimization.tex:248,280 (not built); fig/teaser_* tables (not built) | **sign bug**: nflverse lines read as sportsbook sign (surveys 15, 23) | 6-point teaser legs won **70.7%** in 2020-2024 (both sides, n = 2,788). A two-leg teaser therefore has EV **-10.0%** at decimal 1.8 and -8.4% at -120; break-even per leg is 74.5% and 73.9% | check_claims C40 |
| C14 | ADR-009: v6 baseline **34.00 yards** MAE (2022-2024), "proven production" | app/c_ensemble_ablation:609,681; ADR-009:25,109,134,442 | **in-sample** (components trained on 2018-2024) | None | per survey 16, not re-run (script deleted) |
| C43 | Incremental XGBoost **+21%** vs the "static baseline" (Nov 10, 2025) | app/o_model_specifications:628; ch09:46; ADR-008:22,666 | **wrong baseline** (already retracted, POSTMORTEMS section 2) | None | per POSTMORTEMS section 2, not re-run |
| C15 | ADR-013 rest: **r = 0.17**, a "25-point swing"; home +3 days rest **69.2% ATS** | ADR-013:36,48,117,148,169 (the file has a correction banner); ADR-014:138; ws3_model_performance:124; ws5_betting_strategy:103 | **sign bug** | nflverse rest, 2025, home +3 days or more: **41.4%** (n = 29) correctly graded, 69.0% under the old rule. 2006-2024: 49.0% (n = 512), corr(rest difference, ATS margin) **+0.009** (n = 5,011). ADR-013's own query does not reproduce its table on the repaired DB (126 games, not 139). Graded correctly it gives r = +0.137 with no monotone pattern | check_claims C15 |
| C26 | Game Outcome v2: **52.1%** `ats_win_pct` / "correct side (spread)" | ws3_model_performance:25,29,64; ws7_infrastructure:179; `ml.model_metrics` run 17 | **mislabeled metric**: the spread head's straight-up side, sign(pred) = sign(margin) | 148/284 = 52.1% is straight-up. The actual ATS record is 139-145-1, **48.9%** | check_claims C26 |
| C37 | XGBoost props v1: **12.13 yards** MAE on 2024, "may actually work" | ws6_props_assessment:54,224; models/xgboost/player_props_v1_metadata.json:29 | **same-game data** (features `log_attempts`, `yards_per_attempt`, `td_rate` of the predicted game) and early stopping on the test season | None. Yards = attempts x yards per attempt is an identity | check_claims C37; per survey 16 |
| C23 | Semantic A/B: **75.4% vs 75.9%** accuracy, +43.9% ROI, p = 3.4e-11 | data/semantic_ab_test_results.json:6,8,9 | **simulated from the outcomes** | None. `scripts/ab_test_semantic_features.py` sets `baseline_accuracy = 0.749`, copies `games["home_covered"]` and flips labels with `np.random.random` | check_claims C23 |
| C21 | `models/ensemble/realistic_backtest`: **63.2%** win rate on 796 trades | models/ensemble/realistic_backtest/summary.json:3 | **sign bug** (stored outcomes match the old rule 100%) | Regraded: 382-389-25, **49.5%** [46.0%, 53.1%], ROI -5.4% | check_claims C21 |
| C28 | Edge C ("lingering fatigue"): **70% win rate, 33.6% ROI**, n = 24 | scripts/edge_c_dashboard.py:27,37,38,180,181,240,241; scripts/calculate_edge_c_bet_size.py:146,240,265 | **hard-coded** (`EDGE_C_STATS` literal, marked "PENDING BASELINE FIX") | None. 33.6% is simply the -110 ROI implied by 70% | check_claims C28 |
| C32 | Weather models **91.8-95.3% accuracy**, +80% ROI; "**wind myth busted**" (r = 0.004, hypothesis rejected) | analysis/results/weather_and_rl_summary.md:134,219,250,251; analysis/results/multimodel_weather_comparison.csv:2,3; diss/chapter_12_profit_optimization/chapter_12_profit_optimization.tex:10 (not built); fig/multimodel_weather_table:10 (not built) | Accuracy: **unverifiable** (no generator in git) and impossible: the same harness without weather scores 50-51%. Wind: **wrong** | Outdoor games with wind >= 15 mph went **under 57.0%** (384/674, one-sided p = 0.0002, nflverse 1999-2025): 58.0% in 1999-2012 and 55.4% in 2013-2025. devdb01 2018-2024: 57.1% (76/133, p = 0.059); corr(wind, total - closing total) = -0.071 (n = 1,147). This is game-time wind, so it is a historical association, not a tradeable edge | check_claims C32 (nflverse rows) |
| C22 | Bayesian + XGBoost 2024 test: **71.3% win rate**, +18.9% expected ROI (178 bets) | analysis/ensemble/bayesian_xgb_2024_test.json:11,12 | **simulated** XGBoost, **sign bug** labels and **in-sample** ratings | `xgb_prob_home` = Bayesian probability + N(0, 0.05) with seed 42, reproduced to 1.1e-16. `home_cover_margin` = margin + spread for all 285 games. One static rating per team. Bets regraded with the correct rule: 100-76-2, 56.8% [49.4%, 63.9%], still in-sample | check_claims C22 |

## Verdict counts

Counts are by primary verdict. Several rows have more than one cause.

| Verdict | Rows |
|---|---|
| sign bug | C01, C15, C18, C20, C21, C40 |
| target leakage | C39, C44 (both already retracted) |
| same-season or same-game data | C09, C13, C19, C24, C37, C41 |
| in-sample or tuned on the test set | C11, C12, C14, C16, C17, C27 |
| simulated or hard-coded | C10, C22, C23, C28, C33, C34, C35, C36, C38 |
| mislabeled metric | C07, C08, C26 |
| wrong payoff | C06 |
| wrong baseline | C43 (already retracted) |
| not reproducible or no valid test | C25, C31, C32 (accuracy) |
| correct | C02 (threshold-specific), C03 (below the market), C04, C05, C29 (numbers only), C30, C42 |

## Notes

- **C01, the headline retraction.** The Dec 9 document says the method is "2024 point
  differential + 2.5 home field advantage" on 2025 weeks 1-14. The 179-bet count does not
  reproduce at any threshold: 175-204 bets at edges of 3 to 0 points. One possible reason is
  that `games.spread_close` held EPA-derived placeholder lines for 2025 weeks 12-18 from
  2025-11-21 until a later ingest replaced them (POSTMORTEMS section 4). The mechanism
  reproduces at every threshold and in every season, so the retraction does not depend on the
  exact count. The README, PROJECT_STATUS and CLAUDE.md already retract it. The dissertation
  abstract (`main.tex:411`), chapters 3, 9 and 12, and appendix O still present it.
- **C06-C08 and C18** are regraded in `walkforward_2004_2024.md`. That note also covers the
  sign-fixed state-space column. With the fix, the best columns reach 51.2-51.7%. They are still
  below break-even and roughly match a constant "always bet the road team" strategy, which wins
  51.3% here because home teams covered only 48.7% of 2004-2024 games.
- **C31.** The commit that produced 60.2% (e12856b, 2025-11-02) already graded covers correctly
  (`margin - spread_close`, pushes NaN). Its builder is identical to today's `asof_features.py`
  apart from formatting. The run used the working tree, which also holds another task's
  uncommitted kickoff edits. The lost input `data/processed/features/asof_FIXED_v2.csv` had
  1,403 test games where today's build has 1,376 non-push games. That suggests a different
  label or game set, which cannot be checked now.
- **C15.** ADR-013's query measures each home team's days since its previous *home* game. The
  numbers in ws5 ("69.2% for home +3 days") match the old rule applied to nflverse rest
  differences (69.0%, n = 29).
- **Locations are a snapshot.** Line numbers will move as other salvage tasks edit these
  files. Run `check_claims.py` again before using them for edits.

## Not re-run, and why

| Row | Reason | Evidence |
|---|---|---|
| C11, C12, C14 | Inputs or scripts are not in the repo (`results/validation_v6_recovered_xgb/`, `validate_ensemble_v7_multiyear.py`) | survey 16-leakage-sweep; 19-docs-consistency |
| C24 | The `semantic.*` schema and the Dec-2025 A/B inputs are not in devdb01 | survey 16-leakage-sweep |
| C38 | A projection, not a measurement | survey 18-dissertation |
| C39, C43, C44 | Already retracted in ADR-016, POSTMORTEMS section 2 and ADR-003 | those documents |
| C10 (the leaked CQL component), C37 (the trainer's same-game and early-stopping code) | Cited from the surveys. The prop-line absence, the simulated-line code sites (C36) and the v1 feature list were re-checked | surveys 16 and 18 |

## Reproduce

```bash
cd /home/dro/rice/nfl-analytics
uv run --no-sync python scripts/final_evaluation/check_claims.py      # under 20 s
uv run --no-sync python -I scripts/final_evaluation/check_claims.py --nflverse-games <games.csv>
uv run --no-sync python scripts/final_evaluation/regrade_walkforward_2004_2024.py
```

The second command adds the 1999-2025 wind rows from nflverse schedules
(`github.com/nflverse/nflverse-data/releases/download/schedules/games.csv`; the copy used here
was downloaded on 2026-10-24 at 16:54 EDT). Both scripts connect through the `DB_*` environment
variables, with the defaults in `nfl_analytics/utils/db_config.py`, and force a read-only
session. They read no row-level odds: the only `odds_history` query is an aggregate count by
market type.
