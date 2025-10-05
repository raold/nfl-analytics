# LaTeX Table Integration Manifest
**Generated**: October 4, 2025
**Purpose**: Track which auto-generated tables are integrated into dissertation chapters

---

## Summary Statistics

- **Total tables available**: 34
- **Tables currently referenced**: 19
- **Integration rate**: 56% (19/34)
- **Orphaned tables**: 15

---

## Tables Currently Integrated ✅

### Chapter 3: Data Foundation
- `copula_gof_table.tex` - Copula goodness-of-fit tests
- `tail_dependence_table.tex` - Tail dependence estimates

### Chapter 4: Baseline Modeling
- `glm_baseline_table.tex` - GLM walk-forward validation results
- `glm_harness_overall.tex` - Harness calibration summary
- `keymass_chisq_table.tex` - Key-number chi-square tests
- `oos_record_table.tex` - Out-of-sample record

### Chapter 5: RL Design
- `ope_grid_table.tex` - OPE grid search results
- `ess_table.tex` - Effective sample size
- `rl_vs_baseline_table.tex` - RL agent comparison to baseline
- `rl_agent_comparison_table.tex` - DQN vs PPO comparison
- `utilization_adjusted_sharpe_table.tex` - Adj. Sharpe ratios
- `dm_test_table.tex` - Diebold-Mariano tests

### Chapter 6: Uncertainty, Risk & Betting
- `cvar_benchmark_table.tex` - CVaR portfolio benchmarks
- `teaser_ev_oos_table.tex` - Teaser EV out-of-sample
- `teaser_ev_sensitivity_table.tex` - Teaser sensitivity analysis
- `reweighting_ablation_table.tex` - IPF reweighting ablation

### Chapter 7: Simulation
- `sim_acceptance_table.tex` - Simulator acceptance metrics
- `sim_fail_deviation_table.tex` - Failure mode deviations

### Chapter 8: Results & Discussion
- *(Cross-references to earlier tables)*

---

## Orphaned Tables (Available but Not Integrated) ⚠️

### GLM Calibration Variants
1. `glm_baseline_table_cal_platt.tex` - Platt-calibrated GLM
2. `glm_reliability_panel.tex` - Reliability diagrams
3. `glm_reliability_panel_dual.tex` - Dual reliability plot
4. `glm_reliability_panel_single.tex` - Single reliability plot

### GLM Harness Variants (Size)
5. `glm_harness_table_small.tex` - Compact harness table
6. `glm_harness_table_small3.tex` - 3-column harness
7. `glm_harness_table_small4.tex` - 4-column harness
8. `glm_harness_overall_small.tex` - Compact overall summary
9. `glm_harness_overall_small3.tex` - 3-column overall
10. `glm_harness_overall_small4.tex` - 4-column overall

### Multi-Model Comparisons
11. `multimodel_table.tex` - Multi-model performance
12. `multimodel_weather_table.tex` - Weather impact comparison

### Teaser Pricing
13. `teaser_copula_impact_table.tex` - Copula vs independence pricing

### Helper Tables
14. `oos_record_rows.tex` - Raw OOS record data (helper)
15. `_test_table.tex` - Test/debug table

---

## Integration Recommendations

### High Priority (Include in Dissertation)

**Chapter 4 - Baseline Modeling:**
- `glm_baseline_table_cal_platt.tex` - Shows Platt calibration impact
- `glm_reliability_panel.tex` - Visual calibration validation
- `multimodel_table.tex` - Compare GLM to XGBoost/RF

**Chapter 6 - Risk & Betting:**
- `teaser_copula_impact_table.tex` - Demonstrates copula pricing delta

**Chapter 8 - Results:**
- `multimodel_weather_table.tex` - Weather impact on models

### Medium Priority (Optional/Appendix)

- Size variants (`_small`, `_small3`, `_small4`) - Use if space-constrained
- `oos_record_rows.tex` - Raw data backup

### Low Priority (Can Omit)

- `_test_table.tex` - Debug file, not for publication
- Duplicate calibration plots (choose one: `glm_reliability_panel.tex`)

---

## Action Plan

### Step 1: Integrate High-Priority Tables (1-2 hours)

**Chapter 4 additions:**
```latex
\subsection{Multi-Model Comparison}
\IfFileExists{../figures/out/multimodel_table.tex}{
  \input{../figures/out/multimodel_table.tex}
  Multi-model performance across GLM, XGBoost, and Random Forest...
}{}

\subsection{Calibration Validation}
\IfFileExists{../figures/out/glm_reliability_panel.tex}{
  \input{../figures/out/glm_reliability_panel.tex}
  Reliability diagrams show strong calibration...
}{}
```

**Chapter 6 additions:**
```latex
\subsection{Copula Pricing Impact}
\IfFileExists{../figures/out/teaser_copula_impact_table.tex}{
  \input{../figures/out/teaser_copula_impact_table.tex}
  The delta between copula and independence assumptions is minimal (ρ=0.020)...
}{}
```

**Chapter 8 additions:**
```latex
\subsection{Weather Impact on Predictions}
\IfFileExists{../figures/out/multimodel_weather_table.tex}{
  \input{../figures/out/multimodel_weather_table.tex}
  Wind hypothesis rejection (r=0.004, p=0.90) is confirmed across all models...
}{}
```

### Step 2: Choose Between Size Variants

For tight page limits, replace:
- `glm_harness_table.tex` → `glm_harness_table_small3.tex`
- `glm_harness_overall.tex` → `glm_harness_overall_small3.tex`

### Step 3: Clean Up Test Files

Move to archive:
```bash
mkdir -p analysis/dissertation/figures/out/archive
mv analysis/dissertation/figures/out/_test_table.tex analysis/dissertation/figures/out/archive/
```

---

## Table File Locations

All auto-generated tables live in:
```
analysis/dissertation/figures/out/*.tex
```

Referenced from chapters via:
```latex
\IfFileExists{../figures/out/TABLENAME.tex}{
  \input{../figures/out/TABLENAME.tex}
}{}
```

---

## Validation Checklist

Before final PDF build:
- [ ] All high-priority tables integrated
- [ ] No `\input{}` statements referencing missing files
- [ ] Table captions are descriptive
- [ ] Cross-references to tables work (`\ref{tab:...}`)
- [ ] All tables compile without errors
- [ ] No overfull hbox warnings for tables
- [ ] Table numbering is sequential within chapters

---

## Notes

- **Small variants**: Generated for tight journal submissions, may not be needed for dissertation
- **Reliability panels**: Choose ONE of the three (recommend: `glm_reliability_panel.tex`)
- **Test files**: Archive before defense, not for publication
- **Integration rate**: Target 80%+ for comprehensive dissertation (currently 56%)

---

**Last Updated**: October 4, 2025
**Next Review**: After integrating high-priority tables
