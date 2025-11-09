#!/usr/bin/env Rscript
#
# State-Space Model GO/NO-GO Decision
#
# Decision rule:
#   - If state-space v2 MAE ≤ best baseline MAE → GO (include in ensemble)
#   - If state-space v2 MAE > best baseline MAE → NO-GO (exclude, document)
#
# If close (within 1 yard MAE), run Diebold-Mariano test for statistical significance.
#

library(tidyverse)
library(glue)

cat("======================================================================\n")
cat("STATE-SPACE MODEL GO/NO-GO DECISION\n")
cat("======================================================================\n\n")

# ============================================================================
# STEP 1: Load metrics
# ============================================================================

cat("STEP 1: Loading 2025 validation metrics...\n\n")

metrics <- read_csv("analysis/results/all_models_2025_comparison_v2.csv", show_col_types = FALSE)

print(metrics, n = Inf)
cat("\n")

# ============================================================================
# STEP 2: Identify best baseline and state-space performance
# ============================================================================

cat("STEP 2: Comparing state-space vs baselines...\n\n")

# Baselines: hierarchical and informative priors
baselines <- metrics %>% filter(model %in% c("hierarchical_v2", "informative_priors_v2"))
state_space <- metrics %>% filter(model == "state_space_v2")

best_baseline <- baselines %>% slice_min(mae, n = 1)

cat(sprintf("Best baseline: %s\n", best_baseline$model))
cat(sprintf("  MAE: %.2f yards\n", best_baseline$mae))
cat(sprintf("  RMSE: %.2f yards\n", best_baseline$rmse))
cat(sprintf("  Coverage: %.1f%%\n\n", best_baseline$coverage_90))

cat(sprintf("State-space v2:\n"))
cat(sprintf("  MAE: %.2f yards\n", state_space$mae))
cat(sprintf("  RMSE: %.2f yards\n", state_space$rmse))
cat(sprintf("  Coverage: %.1f%%\n\n", state_space$coverage_90))

delta_mae <- state_space$mae - best_baseline$mae

cat(sprintf("Delta MAE: %.2f yards", delta_mae))
if (delta_mae < 0) {
  cat(" (state-space is BETTER)\n\n")
} else {
  cat(" (state-space is WORSE)\n\n")
}

# ============================================================================
# STEP 3: Make decision
# ============================================================================

cat("STEP 3: Making GO/NO-GO decision...\n\n")

# Decision thresholds
CLOSE_THRESHOLD <- 1.0  # Within 1 yard = "close"

if (abs(delta_mae) <= CLOSE_THRESHOLD) {
  cat("❗ Models are CLOSE (delta MAE ≤ 1.0 yards)\n")
  cat("   Recommend Diebold-Mariano test for statistical significance\n\n")

  # Load predictions for DM test
  cat("   Running Diebold-Mariano test...\n")

  ss_preds <- read_csv("analysis/results/state_space_v2_2025_holdout.csv", show_col_types = FALSE)
  baseline_preds <- read_csv(
    glue("analysis/results/{best_baseline$model}_2025_holdout.csv"),
    show_col_types = FALSE
  )

  # Extract errors
  e1 <- ss_preds$abs_error
  e2 <- baseline_preds$abs_error

  # Loss differential
  d <- e1^2 - e2^2

  # DM test statistic
  n <- length(d)
  d_bar <- mean(d)
  s_d <- sd(d)
  dm_stat <- sqrt(n) * d_bar / s_d
  p_value <- 2 * pnorm(-abs(dm_stat))

  cat(sprintf("   DM statistic: %.3f\n", dm_stat))
  cat(sprintf("   p-value: %.4f\n\n", p_value))

  if (p_value < 0.05) {
    if (dm_stat < 0) {
      decision <- "GO"
      reason <- glue("State-space significantly outperforms {best_baseline$model} (p={round(p_value, 4)})")
    } else {
      decision <- "NO-GO"
      reason <- glue("{best_baseline$model} significantly outperforms state-space (p={round(p_value, 4)})")
    }
  } else {
    if (state_space$mae <= best_baseline$mae) {
      decision <- "GO"
      reason <- glue("Tie with {best_baseline$model} (no significant difference), include for diversity")
    } else {
      decision <- "NO-GO"
      reason <- glue("Tie with {best_baseline$model} (no significant difference), exclude for simplicity")
    }
  }

} else if (delta_mae < 0) {
  decision <- "GO"
  reason <- glue("State-space outperforms best baseline by {abs(round(delta_mae, 2))} yards MAE")
} else {
  decision <- "NO-GO"
  reason <- glue("Best baseline ({best_baseline$model}) outperforms state-space by {round(delta_mae, 2)} yards MAE")
}

# ============================================================================
# STEP 4: Document decision
# ============================================================================

cat("======================================================================\n")
cat(glue("FINAL DECISION: {decision}\n"))
cat("======================================================================\n\n")

cat(glue("Rationale: {reason}\n\n"))

# Additional context
cat("Additional Context:\n")
cat(sprintf("  - State-space MAE: %.2f yards\n", state_space$mae))
cat(sprintf("  - Best baseline MAE: %.2f yards (%s)\n", best_baseline$mae, best_baseline$model))
cat(sprintf("  - State-space coverage: %.1f%%\n", state_space$coverage_90))
cat(sprintf("  - Best baseline coverage: %.1f%%\n", best_baseline$coverage_90))
cat(sprintf("  - State-space interval width: %.1f yards\n", state_space$mean_interval_width))
cat(sprintf("  - Best baseline interval width: %.1f yards\n\n", best_baseline$mean_interval_width))

# ============================================================================
# STEP 5: Save decision to file
# ============================================================================

cat("STEP 5: Saving decision...\n")

decision_doc <- glue("
# State-Space Model GO/NO-GO Decision

**Date**: {Sys.Date()}
**Validation Period**: 2025 weeks 1-6 (148 games, 37 QBs)

## Decision: **{decision}**

### Rationale
{reason}

## Performance Comparison

| Model | MAE (yards) | RMSE (yards) | Coverage 90% | Interval Width | Decision |
|-------|-------------|--------------|--------------|----------------|----------|
| {best_baseline$model} | {round(best_baseline$mae, 2)} | {round(best_baseline$rmse, 2)} | {round(best_baseline$coverage_90, 1)}% | {round(best_baseline$mean_interval_width, 1)} | Baseline |
| state_space_v2 | {round(state_space$mae, 2)} | {round(state_space$rmse, 2)} | {round(state_space$coverage_90, 1)}% | {round(state_space$mean_interval_width, 1)} | **{decision}** |

**Delta MAE**: {round(delta_mae, 2)} yards

## Implications

### If GO:
- Include state-space model in 4-way ensemble (hierarchical + informative priors + BNN + state-space)
- Expected ensemble improvement: +0.1-0.3 pp ROI (model diversity)
- Adds time-varying player skill dynamics to ensemble

### If NO-GO:
- Exclude state-space model from ensemble (use 3-way: hierarchical + informative priors + BNN)
- Document as negative result (state-space didn't add value beyond hierarchical models)
- Archive code for future research

## Next Steps

1. Update ensemble configuration in `py/ensemble/enhanced_ensemble_v3.py`
2. Run full ensemble backtest on 2022-2024
3. Optimize ensemble weights using inverse variance or meta-learner
4. Deploy to production with monitoring

## Training Details

**Canonical window**: 2015-2024 (uniform) = 2018-2024 + H=4 decay on 2015-2017

**All v2 models**:
- Hierarchical Bayesian v2: {round(metrics$mae[metrics$model == 'hierarchical_v2'], 2)} yards MAE
- Informative Priors v2: {round(metrics$mae[metrics$model == 'informative_priors_v2'], 2)} yards MAE
- State-Space v2: {round(state_space$mae, 2)} yards MAE

**Decision made**: {Sys.time()}
**Decision maker**: Claude Code (Sonnet 4.5)
")

write_file(decision_doc, "docs/STATE_SPACE_FINAL_DECISION.md")

cat("✓ Decision saved to: docs/STATE_SPACE_FINAL_DECISION.md\n\n")

cat("======================================================================\n")
cat("✅ DECISION PROCESS COMPLETE\n")
cat("======================================================================\n\n")

cat(glue("Final verdict: **{decision}**\n"))
cat(glue("Rationale: {reason}\n\n"))

# Return decision for downstream use
if (decision == "GO") {
  cat("✅ State-space model approved for ensemble inclusion\n\n")
} else {
  cat("❌ State-space model excluded from ensemble (documented as negative result)\n\n")
}
")
