# 2025 Season Review (February 2026)

Seven workstreams written on 2026-02-12, after the 2025 season, to find out why Game Outcome v2
lost to the closing line. They are kept as written. The October 2026 salvage checked them against
the code, the saved models and the repaired database; this page says what still holds. The graded
result that supersedes their numbers is the
[final evaluation](../2026-10-final-evaluation/README.md): v2's spread MAE 11.05 against 9.67 for
the line, ATS 139-145-1 (48.9%), winners 57.0% against 65.8% for the closing favorite.

Files and scripts these documents cite that are no longer in the tree are at tag
`archive/pre-prune-2026-10`. Retraction-ledger rows are cited as C01-C44
([ledger](../2026-10-final-evaluation/retraction_ledger.md)).

## Two corrections that apply to several workstreams

- **The compression diagnosis.** v2's 2025 predictions span -2.6 to +8.7 points with a standard
  deviation of 2.30, against 6.18 for the closing line (correlation 0.36). WS1 and WS2 compare that
  2.30 with the spread of the outcomes (14.16), and WS1-WS3 blame missing QBR and NextGen data. But none
  of v2's three boosters splits on any of its 20 QBR or 46 NextGen inputs (counted from the saved
  trees), so those gaps cannot have moved a single prediction. The inputs v2 does use that had no
  2025 data are the line-movement (`lm_`) group: `odds_history` has no 2025 snapshots, so they sat
  at their defaults for every 2025 game. (`espn_qbr` has since been loaded: 10,709 rows.)
- **Post-hoc subgroups.** The "sweet spots" (the 2-4 or 3-4 point edge window, home favorites with
  a 2-4 point edge, Thursday games, Wild Card games) were picked after seeing the 2025 results, on
  6-65 bets each, among many buckets; WS5 says itself that none survives a multiple-comparison
  correction. They are not evidence of an edge (C27). The neighbors of the 3-4 point bucket
  (23-11-1) are 9-12 (4-5 points) and worse beyond.

## By workstream

| File | Still useful | Corrected |
|---|---|---|
| [ws1_data_layer_audit.md](ws1_data_layer_audit.md) | the inventory of empty and stale inputs for 2025 | the root cause of the compression (above) |
| [ws2_prediction_compression.md](ws2_prediction_compression.md) | the description of the compressed predictions | the root cause; "ATS ROI could move from -3.2% to +1-3%" is speculation, and -3.2% holds only at \|edge\| > 2 (C02) |
| [ws3_model_performance.md](ws3_model_performance.md) | MAE against the line; the power-ratings walk-forward, 51.8% over 2020-2025 (C42) | "Correct side 52.1%" is straight-up side accuracy, not ATS (C26); the 2025 57.7% for power ratings is one season of noise (C42); the rest "25pp ATS swing" is the sign bug (C15) |
| [ws3_signal_analysis.md](ws3_signal_analysis.md) | the edge-bucket table (it reproduces) | the 3-4 point "sweet spot" (C27); "Line Movement OK (from odds_history)": 2025 has no snapshots; the semantic A/B "p = 0.21" has no artifact behind it (C25); the rest table is another post-hoc split (its "most profitable scenario" is 13 of 20 games) |
| [ws4_ensemble_assessment.md](ws4_ensemble_assessment.md) | the map of the two systems: v6 predicts player passing yards, Game Outcome v2 predicts games | v2's "validation" numbers (10.45, AUC 0.691) are scores on its early-stopping season (C17) |
| [ws5_betting_strategy.md](ws5_betting_strategy.md) | the edge-threshold table (139-145-1 on all games; it reproduces exactly) | the 2-4 point bucket, TNF 71.4% and the Kelly sizing are post-hoc (above); "69.2% for home +3 days rest" is the sign bug (C15) |
| [ws6_props_assessment.md](ws6_props_assessment.md) | the inventory of missing props artifacts | line 22 has it backwards: the v6 ensemble predicts passing yards (WS4); the XGBoost props v1 "12.13 MAE" uses the predicted game's own yards per attempt and early-stopped on its test season (C37); no prop lines exist to test props against (C10) |
| [ws7_infrastructure.md](ws7_infrastructure.md) | the account of a season run by hand | "ats_win_pct 52.1%" is straight-up side accuracy (C26); its Makefile pipeline was retired in the 2026-10 prune, its health gate checks QBR and NextGen inputs v2 never uses, and the feedback and betting tests it added went with their modules |
