# NFL Analytics Model Registry (Database Export)

**Generated:** 2025-11-13 14:39:27
**Source:** PostgreSQL `ml.model_registry`
**Total Models:** 23

---

## Production Models ✅ (12 models)

| ID | Model Name | Type | Version | Status | Description |
|----|------------|------|---------|--------|-------------|
| 1 | bnn_multioutput_v1 | archived | v1 | archived | Archived 2025-11-05: Data leakage from Vegas lines (84.7% SH... |
| 2 | bnn_passing_v1 | bnn | v1.0 | completed | PyMC-based BNN for passing yards prediction. MAE: 58.70 yard... |
| 3 | bnn_calibrated_v2 | bnn | v2.0 | completed | Temperature-scaled BNN (T=0.75) for optimal calibration. Cov... |
| 4 | passing_informative_priors_v1 | hierarchical | v1.0 | completed | Hierarchical Bayesian model with empirically-derived informa... |
| 5 | receiving_qb_chemistry_v1 | hierarchical | v1.0 | completed | QB-WR dyadic effects model with 2,168 QB-WR pairs. Impact: +... |
| 6 | cql_agent_v1 | rl | v1.0 | completed | Conservative Q-Learning for bet sizing. 24% improvement over... |
| 7 | state_space_v1 | state_space | v1.0 | completed | Time-varying player ratings using LOESS-approximated Kalman ... |
| 11 | state_space_v2 | state_space | v2.0 | production | Bayesian Random Walk State-Space v2 |
| 12 | bnn_multioutput_v3_1 | bayesian_neural_network | v3.1 | production | Multi-Output BNN v3.1 (Calibrated) |
| 13 | hierarchical_v2 | hierarchical_bayesian | v2.0 | production | Hierarchical Bayesian Passing v2 |
| 14 | cql_bet_sizer | reinforcement_learning | v1.0 | production | Conservative Q-Learning Bet Sizer |
| 16 | informative_priors_v2 | hierarchical_bayesian | v2.0 | production | Bayesian Informative Priors v2 |

---

## Experimental Models ⚠️ (11 models)

| ID | Model Name | Type | Version | Status | Description |
|----|------------|------|---------|--------|-------------|
| 8 | bnn_rushing_v1 | bnn | v1.0 | experimental | v1.0 baseline rushing BNN. 53MB, PyMC-based. |
| 9 | bnn_simpler_v2 | bnn | v2.0 | experimental | Simplified BNN architecture. 10MB, useful baseline. |
| 10 | bnn_mixture_experts_v2 | bnn | v2.0 | archived | Mixture of experts BNN. 1.3GB experimental model. |
| 15 | xgboost_baseline | gradient_boosting | v1 | staging | Baseline model for ensemble comparison. Not used in producti... |
| 17 | bnn_nuts | archived | v1 | archived | Archived 2025-10-15: Timeout after 6+ hours per configuratio... |
| 18 | bnn_mc_dropout | archived | v1 | archived | Archived 2025-10-20: Gradient explosion → NaN losses after e... |
| 19 | bnn_temperature_unscaled | archived | v1 | archived | Archived 2025-10-25: Severe undercalibration (26% coverage v... |
| 20 | bnn_laplace | archived | v1 | archived | Archived 2025-11-01: 9/9 configurations failed with Hessian ... |
| 21 | bnn_multioutput_passing_v2 | archived | v2 | archived | Archived 2025-11-09: NaN losses from epoch 0 despite Week 0 ... |
| 22 | mcp_test_model | test | 1.0.0 | trained | MCP Phase 1 integration test model |
| 28 | xgboost_v2_canonical | gradient_boosting | v2.0 | archived | XGBoost v2 Canonical |

---

## Query Examples

### View all production models
```sql
SELECT model_name, model_type, version, description
FROM ml.model_registry
WHERE is_production = true
ORDER BY model_name;
```

### View model with metrics
```sql
SELECT
    mr.model_name,
    mr.model_type,
    mm.metric_name,
    mm.metric_value
FROM ml.model_registry mr
LEFT JOIN ml.model_metrics mm ON mr.model_id = mm.model_id
WHERE mr.is_production = true
ORDER BY mr.model_name, mm.metric_name;
```

---

**Note:** This file is auto-generated from the database. Do not edit manually.
**Update:** Run `python scripts/sync_model_registry.py --export-md`
