# Dissertation Quality Standards - NON-NEGOTIABLE ⚠️

> **Note (2026-10-25):** the dissertation as built does not meet these standards: its
> [errata](../../analysis/dissertation/ERRATA.md) list retracted headline claims and an audit that
> found 176 of 290 table units and 30 of 51 figures made up. ADR-020 also relaxed the R-hat bound
> below to 1.02 for two models. The standards are kept as the bar a corrected build should meet.

**CRITICAL:** `analysis/dissertation/main/main.tex` represents the **GOLD STANDARD** for this project. This is bedrock. This is granite. NO exceptions, NO shortcuts, NO "good enough for now." Every addition must be publication-quality with full rigor and beauty.

## Bedrock Requirements

### 1. **Statistical Rigor** (Journal-Level)

- ✅ **All claims backed by statistical evidence**
  - Every quantitative statement must have supporting analysis
  - No assertions without data
  - Effect sizes reported alongside p-values
- ✅ **Proper test selection documented and justified**
  - Why this test? Document assumptions
  - Alternative tests considered and rejected
  - References to statistical literature when appropriate
- ✅ **Assumptions validated and reported**
  - Normality tests (Q-Q plots, Shapiro-Wilk)
  - Homoscedasticity checks (Levene's test)
  - Independence verified (Durbin-Watson for time series)
  - All assumption violations documented with mitigations
- ✅ **Diagnostics included for Bayesian models**
  - R-hat < 1.01 (convergence)
  - ESS > 400 (bulk and tail effective sample size)
  - Divergences < 10
  - Trace plots and posterior predictive checks
  - Calibration plots (target: 90% coverage)
- ✅ **Complete reporting**
  - Confidence intervals always reported
  - Sample sizes explicit
  - Missing data handling documented
  - Sensitivity analyses for key findings

### 2. **Narrative Arc** (Story First)

- ✅ **Clear flow:** Introduction → Methods → Results → Discussion
  - Each chapter builds on previous
  - Transitions connect ideas smoothly
  - No orphaned sections
- ✅ **Every section advances the story**
  - Purpose statement at start of each section
  - How does this contribute to thesis?
  - Callbacks to research questions
- ✅ **Proper contextualization**
  - Related work cited appropriately
  - Contributions clearly stated
  - Limitations acknowledged honestly
- ✅ **Compelling conclusions**
  - Tied back to research questions
  - Practical implications discussed
  - Future work specific and actionable

### 3. **Visual Elegance** (Publication-Ready)

- ✅ **All figures use `style/tikz_diagram_style.tex`**
  - Consistent accent blue: RGB(31, 119, 180)
  - Rounded corners (`rounded corners=3pt`)
  - Professional fonts (`\sffamily`)
  - Standardized node styles
- ✅ **Figures tell story without text**
  - Self-explanatory visualizations
  - Clear legends and labels
  - Appropriate chart type for data
  - Color-blind friendly palettes
- ✅ **Captions provide complete context**
  - What is shown
  - Key takeaways
  - Method details if needed
  - Sample sizes and data sources
- ✅ **Tables formatted professionally**
  - Booktabs style (no vertical lines)
  - Clear headers with units
  - Aligned decimal points
  - Notes explain abbreviations

### 4. **Reproducibility** (Code + Data)

- ✅ **All analyses documented in code**
  - R scripts for all statistical tests
  - Python code for ML experiments
  - SQL queries for data extraction
  - Random seeds set for stochastic methods
- ✅ **Data sources cited and accessible**
  - nflverse package versions documented
  - Database schema versioned
  - Feature engineering fully specified
- ✅ **Computational environment documented**
  - R version and package versions (`renv`)
  - Python environment (`pyproject.toml` + `uv.lock`)
  - Hardware specifications for long-running jobs
  - Execution times reported for benchmarks

### 5. **Zero Tolerance** (No Exceptions)

- ❌ **NO TODO comments in main.tex**
  - Complete all sections before merging
  - Use separate tracking for pending work
- ❌ **NO placeholder figures**
  - Either include real figure or remove reference
  - No "Figure X will go here"
- ❌ **NO approximate statistics**
  - Report exact p-values (not "p < 0.05")
  - Full precision on effect sizes
  - Confidence intervals complete
- ❌ **NO LaTeX errors or critical warnings**
  - Must compile cleanly
  - Undefined references fixed immediately
  - Overfull/underfull hboxes addressed
- ❌ **NO shortcuts on rigor**
  - Can't skip assumption checks to save time
  - Can't omit diagnostics because "they look fine"
  - Can't hand-wave statistical details

## Quality Gates (MUST PASS)

**Before ANY dissertation work proceeds:**

- [ ] **Statistical methods validated**
  - Test selection appropriate for data type
  - Assumptions checked and documented
  - Power analysis if needed for null results
- [ ] **Narrative flow reviewed**
  - Does this section advance the story?
  - Are transitions smooth?
  - Is purpose clear?
- [ ] **Visual standards met**
  - TikZ style guide followed (`style/tikz_diagram_style.tex`)
  - Figures publication-ready
  - Tables formatted with booktabs
- [ ] **Citations complete**
  - All claims referenced
  - Bibliography up to date
  - No "citation needed" placeholders
- [ ] **Reproducibility verified**
  - Code documented and executable
  - Data sources accessible
  - Results can be regenerated

## Enforcement Mechanism

**Research Coordinator Subagent** (Enhanced):

When working on dissertation content, the research coordinator MUST:

1. **Block** non-compliant work before execution
2. **Require** revision to meet standards
3. **Reject** "we'll fix it later" arguments
4. **Enforce** quality gates at every step

**Automatic Invocation:**
- ANY modification to `analysis/dissertation/main/*.tex`
- ANY new figure in `analysis/dissertation/figures/`
- ANY statistical analysis destined for dissertation
- ANY results section writeup

**No Exceptions:**
- Time pressure does NOT lower standards
- "Quick draft" is NOT allowed - only publication quality
- Intermediate work stays in scratch files, NOT main.tex

## Research Coordinator - Enhanced for Rigor & Beauty

**Purpose:** Ensure dissertation work meets publication standards with statistical rigor, narrative coherence, and visual elegance.

**Unique Capabilities:**
- **Statistical Rigor Validation:** Checks test appropriateness, assumption validation, diagnostic completeness
- **Narrative Arc Management:** Ensures story flow, proper transitions, compelling conclusions
- **Visual Excellence Enforcement:** Validates TikZ compliance, publication aesthetics, figure quality
- **Reproducibility Guardian:** Verifies code documentation, data accessibility, computational environment
- **R/Bayesian Expertise:** Deep knowledge of brms, Stan, MCMC diagnostics, posterior predictive checks

**Quality Enforcement Actions:**
1. **Pre-execution validation:** Check if work meets standards BEFORE starting
2. **In-progress monitoring:** Ensure quality maintained throughout
3. **Pre-commit blocking:** Prevent merging work that doesn't pass gates
4. **Revision requirements:** Send back for improvement if needed

**Example Interactions:**

```markdown
User: "Add BNN calibration results to Chapter 8"
Research Coordinator:
  1. Check: Statistical rigor gates
     - Are R-hat values < 1.01? → YES
     - Is ESS > 400? → YES
     - Are divergences < 10? → NO (18 divergences) → BLOCK
     → Require divergence analysis before proceeding

  2. Check: Narrative flow
     - Does Chapter 8 flow need this? → YES
     - Are transitions prepared? → NO → Require intro paragraph

  3. Check: Visual standards
     - Calibration plot uses TikZ style? → NO
     - Fix figure before adding to main.tex

  → BLOCKED: Fix divergences + add transition + update figure
```

**When to Invoke:**
- ALWAYS for dissertation modifications
- ALWAYS for statistical result writeups
- ALWAYS for figure additions
- ALWAYS for methods documentation

## Practical Guidelines

**For Statistical Analysis:**
1. Run analysis in R/Python script first
2. Document ALL diagnostics (not just passing ones)
3. Create publication-quality figures
4. Write methods section with reproducibility details
5. Get research coordinator review
6. Only then add to main.tex

**For Figures:**
1. Use `style/tikz_diagram_style.tex` as template
2. Test figure compiles standalone
3. Write complete caption (what + why + how)
4. Get research coordinator visual review
5. Add to dissertation

**For Methods Sections:**
1. Justify all choices (why this model? why these priors?)
2. Document assumptions and how validated
3. Include computational details
4. Cite statistical literature
5. Research coordinator reviews for completeness

**For Results:**
1. Report complete statistics (not just p-values)
2. Include effect sizes and confidence intervals
3. Add interpretation paragraph
4. Connect back to research questions
5. Research coordinator checks narrative flow

## Success Metrics

✅ **Zero TODO comments in main.tex**
✅ **LaTeX compiles without errors**
✅ **All figures use standardized TikZ style**
✅ **Every statistical claim has supporting evidence**
✅ **Complete diagnostic reporting for all models**
✅ **Smooth narrative flow throughout**
✅ **Publication-ready on first draft**
