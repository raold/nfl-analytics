# Undefined References TODO

**Generated:** January 1, 2026
**Status:** Tabled for future work
**Context:** Identified during Chapter 3-4 comprehensive review

## Summary

| Category | Count | Priority |
|----------|-------|----------|
| Undefined Labels | 14 | Medium |
| Undefined Citations | 171 | High |
| **Total** | **185** | |

---

## Undefined Labels (14)

### Chapter Labels (2)
| Label | Likely Location | Action Needed |
|-------|-----------------|---------------|
| `chap:baseline` | Should be `chap:methods` | Update reference |
| `chap:results` | Chapter 9 or 10 | Add label to results chapter |

### Section Labels (10)
| Label | Description | Action Needed |
|-------|-------------|---------------|
| `sec:cvar-stakes` | CVaR stakes section | Create or rename |
| `sec:did-injuries` | Difference-in-differences for injuries | Create or rename |
| `sec:ensemble-stacking` | Ensemble stacking section | Create or rename |
| `sec:evidential-live` | Evidential for live betting | Create or rename |
| `sec:honest-failures` | Honest failures section | Create or rename |
| `sec:hybrid-architecture` | Hybrid architecture section | Create or rename |
| `sec:llm-sports` | LLM for sports section | Create or rename |
| `sec:mlops-production` | MLOps production section | Create or rename |
| `sec:online-features` | Online features section | Create or rename |
| `sec:sequence-attention` | Sequence attention section | Create or rename |

### Subsection Labels (1)
| Label | Description | Action Needed |
|-------|-------------|---------------|
| `subsec:hansen1982` | Hansen 1982 lit review subsection | Add to lit review |

### Table Labels (3)
| Label | Description | Action Needed |
|-------|-------------|---------------|
| `tab:rl_vs_baseline` | RL vs baseline comparison | Create table |
| `tab:uq-comparison` | UQ methods comparison | Create table |
| `tab:weather-effects` | Weather effects table | Create table |

### Figure Labels (2)
| Label | Description | Action Needed |
|-------|-------------|---------------|
| `fig:pareto-baseline` | Baseline Pareto frontier | Generate figure |
| `fig:uq-pareto` | UQ Pareto frontier | Generate figure |

---

## Undefined Citations (171)

### High-Priority (Core Methods)
These are cited multiple times and critical for methodology:

```
burkner2017          - brms package
carpenter2017        - Stan
chen2016xgboost      - XGBoost
gelman2013           - Bayesian Data Analysis
glickman1998         - State-space NFL
harville1980         - Poisson score models
kelly1956            - Kelly criterion
levine2020           - Offline RL
sutton2018           - RL textbook
wolpert1992          - Stacked generalization
```

### Medium-Priority (Sports Analytics)
```
burke2009epa         - EPA methodology
dixon1997            - Bivariate Poisson
lopez2020big         - Big Data Bowl
nflverse2023         - nflverse data
romer2006            - 4th down analysis
stern1991            - Spread-to-win
tango2007book        - The Book (baseball)
yurko2020going       - Going deep
```

### Low-Priority (Supporting)
Remaining 150+ citations - see `/tmp/undefined_cites.txt` for full list.

---

## Recommended Actions

### Immediate (Before Submission)
1. Add all 171 citations to `references.bib`
2. Create/fix the 14 undefined labels
3. Generate missing figures (`fig:pareto-baseline`, `fig:uq-pareto`)

### Deferred (Can Wait)
- Section restructuring for orphaned labels
- Table generation for missing tables

---

## Notes

- Many citations may already exist with slightly different keys (e.g., `glickman1998` vs `glickman1998state`)
- Some labels may be in orphaned/unused files
- The `feast2021` and `tecton2022` citations DO exist but weren't being found initially
