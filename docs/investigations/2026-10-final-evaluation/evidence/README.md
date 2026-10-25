# Negative-result evidence (moved here 2026-10-24)

Three small artifacts from the Nov 2025 model experiments. They were committed under `models/`
and are kept here because each one records a model that did not beat the market. The rest of
`models/` was removed in the 2026-10 prune (tag `archive/pre-prune-2026-10` has it). The files
are unchanged; only their paths moved. Every number below is read directly from these files.

| File | Old path |
|---|---|
| `market_blend_config.json` | `models/xgboost/market_blend_config.json` |
| `xgboost_v6_walkforward_20251127_131505_metadata.json` | `models/xgboost/xgboost_v6_walkforward_20251127_131505_metadata.json` |
| `gnn_ablation_summary.csv` | `models/experiments/gnn/ablation/ablation_summary.csv` |

## XGBoost v6 market blend (2025-11-27)

`market_blend_config.json` is the result of a grid search for the weight on the XGBoost v6
spread model (`xgboost_v6_advanced_spread_20251127_124409.json`, not kept) in a blend with the
closing line. The optimum was **market weight 1.0, model weight 0.0**: any weight on the model
made the error worse. Its `validation_mae` of 9.61 is exactly the closing line's own 2024 MAE in
the walk-forward file below.

## XGBoost v6 walk-forward (2025-11-27)

`xgboost_v6_walkforward_20251127_131505_metadata.json` is a season-by-season walk-forward with a
4-season training window and 43 features. Six of the features are the market itself
(`spread_close`, `total_close`, both moneylines and both implied probabilities). Even with the
closing line as an input, the model's absolute error on the final margin was worse than the
line's every season:

| Test season | Training seasons | Test games | Model MAE | Closing-line MAE | Difference |
|---|---|---|---|---|---|
| 2022 | 2018–2021 | 271 | 9.31 | 8.74 | +0.57 |
| 2023 | 2019–2022 | 272 | 10.59 | 9.90 | +0.69 |
| 2024 | 2020–2023 | 272 | 10.32 | 9.61 | +0.71 |
| Overall | | 815 | 10.08 | 9.42 | +0.66 |

## Hierarchical GNN ablation

`gnn_ablation_summary.csv` holds 9 configurations of the hierarchical GNN (embedding size,
hidden size, message-passing rounds, QB–receiver "chemistry" edges). The model predicted whether
the home team wins (`train_hierarchical_gnn.py`, not kept). Every test accuracy is a multiple of
1/271, so the test set was 271 games.

- Test Brier score ranged from 0.2746 to 0.2849. Every configuration is worse than a constant
  50% forecast, which scores 0.25.
- Test accuracy ranged from 46.9% (127/271) to 52.8% (143/271), and test AUC from 0.541 to 0.577.
- Validation Brier ranged from 0.2150 to 0.2372, so the ranking on the validation set did not
  carry over: the best configuration on validation ("Large Model", 0.2150) scored 0.2846 on test.
