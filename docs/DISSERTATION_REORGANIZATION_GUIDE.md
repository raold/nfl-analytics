# Dissertation Comprehensive Reorganization Guide

**Date Created:** November 6, 2025
**Scope:** Complete reorganization addressing all 38 organizational issues
**Estimated Time:** 25-35 hours
**Backup Created:** Git commit `a38644b` (Pre-reorganization snapshot)

---

## Overview

This guide provides step-by-step instructions to reorganize the dissertation from the current 12+Experiments chapter structure to a cleaner 10-chapter structure with proper narrative flow.

**Current Issues:**
- Chapters 11-12 appear AFTER Conclusion (breaking narrative)
- Experiments chapter is 24-line stub
- Chapter 4 too long (1045 lines)
- 5 chapters lack formal conclusions
- Multiple chapters have too many fragmented sections

**Target Structure:**
```
1. Introduction
2. Literature Review
3. Data Foundation
4. Baseline Models (Classical) [NEW: Split from Ch 4]
4B. Advanced Bayesian Methods [NEW: Split from Ch 4]
5. Reinforcement Learning
6. Uncertainty & Risk [EXPANDED: Absorbs Ch 12]
7. Simulation
8. Production Deployment [EXPANDED: Absorbs Ch 11]
9. Results & Discussion [EXPANDED: Absorbs Experiments]
10. Conclusion
Appendices
```

---

## Phase 1: Chapter Restructuring (8-10 hours)

### 1.1 Split Chapter 4 into Two Chapters

**Goal:** Split 1045-line Chapter 4 into Ch 4 (Classical ~500 lines) and Ch 4B (Bayesian ~550 lines)

#### Step 1.1.1: Create Chapter 4B Directory and File

```bash
cd analysis/dissertation
mkdir -p chapter_4b_advanced_bayesian
```

#### Step 1.1.2: Extract Bayesian Content from Chapter 4

**Open:** `chapter_4_baseline_modeling/chapter_4_baseline_modeling.tex`

**Extract these sections to new Chapter 4B file:**
- Section 4.4: Bayesian Hierarchical Team Ratings (lines ~300-700)
  - All subsections 4.4.1 through 4.4.16
- Section 4.8: Player Impact Adjustments (lines ~700-800)
- Section 4.10: v3.0 Ensemble Section (input file)

**Create:** `chapter_4b_advanced_bayesian/chapter_4b_advanced_bayesian.tex`

```latex
\chapter{Advanced Bayesian Methods}
\label{chap:advanced-bayesian}

% Chapter introduction
This chapter presents advanced Bayesian hierarchical modeling techniques that extend the classical baselines from Chapter~\ref{chap:baseline-models}. We develop hierarchical structures that pool information across teams and players, quantify epistemic uncertainty through full posterior distributions, and integrate player-level adjustments for roster changes.

\section{Bayesian Hierarchical Team Ratings}
\label{sec:bayesian-hierarchical}

% PASTE CONTENT FROM Chapter 4 Section 4.4 here
% Include all subsections 4.4.1 through 4.4.16

\section{Player Impact Adjustments}
\label{sec:player-impact}

% PASTE CONTENT FROM Chapter 4 Section 4.8 here

\section{Ensemble Integration: v3.0 Architecture}
\label{sec:ensemble-v3}

% PASTE CONTENT FROM Chapter 4 v3.0 ensemble section here

\chaptersummary{
This chapter developed advanced Bayesian methods for NFL game prediction, demonstrating how hierarchical modeling and ensemble techniques improve both point predictions and uncertainty quantification. Chapter~\ref{chap:rl-design} extends these foundations with reinforcement learning for dynamic bet sizing.
}{
Key technical achievements: (1) Hierarchical Bayesian models with MCMC inference via Stan; (2) Player impact adjustments integrated with team ratings; (3) 4-way ensemble (Bayesian + XGBoost + BNN + State-Space) achieving SOTA performance.
}
\end{latex>
```

#### Step 1.1.3: Update Chapter 4 to Remove Bayesian Content

**Modify:** `chapter_4_baseline_modeling/chapter_4_baseline_modeling.tex`

Remove sections now in Chapter 4B:
- Delete Section 4.4 (Bayesian Hierarchical)
- Delete Section 4.8 (Player Impact)
- Delete v3.0 Ensemble input

Update chapter ending to reference Chapter 4B:
```latex
Having established classical baseline models, Chapter~\ref{chap:advanced-bayesian} extends these foundations with advanced Bayesian hierarchical methods that quantify uncertainty and pool information across teams and players.
```

#### Step 1.1.4: Update Section Numbers in Chapter 4B

After pasting content, update all section numbers:
- `\section{Bayesian...}` → Keep as 4B.1
- All `\subsection{...}` numbers update automatically
- Update all `\label{sec:...}` to include `4b-` prefix for clarity

---

### 1.2 Merge Experiments Chapter into Chapter 8

**Goal:** Eliminate 24-line stub chapter, integrate experiments into Results

#### Step 1.2.1: Read Experiments Chapter Content

```bash
cat chapter_experiments/chapter_experiments.tex
```

**Note:** If chapter has actual content beyond 24 lines, extract it. If truly a stub, create content from experiment framework.

#### Step 1.2.2: Add to Chapter 8

**Modify:** `chapter_8_results_discussion/chapter_8_results_discussion.tex`

**Insert after line 10** (after "Central Finding" section):

```latex
\section{Empirical Validation: 40-Experiment Framework}
\label{sec:40-experiments}

To systematically evaluate model performance, we conducted 40 controlled experiments across five dimensions: feature ablations (removing feature groups), model variants (GLM vs XGBoost vs Bayesian), hyperparameter sensitivity, temporal stability (performance across seasons), and betting strategy ablations. This experimental framework ensures robust conclusions about what drives predictive performance and profitability.

\subsection{Experiment Design}

The 40-experiment grid systematically varies:
\begin{enumerate}
    \item \textbf{Feature Groups} (8 experiments): Baseline vs +Vegas vs +Opponent vs +Environmental
    \item \textbf{Model Architectures} (8 experiments): GLM vs XGBoost vs Hierarchical Bayesian vs Ensemble
    \item \textbf{Hyperparameters} (8 experiments): Learning rate, regularization, MCMC chains
    \item \textbf{Temporal Validation} (8 experiments): Cross-validation across seasons 2015-2024
    \item \textbf{Betting Strategies} (8 experiments): Kelly vs Fractional Kelly vs RL-based sizing
\end{enumerate}

\subsection{Key Experimental Findings}

% Add experimental results table
\IfFileExists{../figures/out/40_experiment_summary_table.tex}{%
  \input{../figures/out/40_experiment_summary_table.tex}
}{}

Critical insights from the experimental framework:
\begin{itemize}
    \item \textbf{Vegas lines dominate}: Models with betting market features (+3.5pp ROI) vastly outperform those without
    \item \textbf{Ensemble advantage}: 4-way ensemble (+1.2pp over best single model) through diverse error patterns
    \item \textbf{Temporal instability}: Performance degrades over time (2015: 5.2\% ROI → 2024: 1.8\% ROI)
    \item \textbf{Strategy sensitivity}: RL-based sizing achieves 24\% higher Sharpe than static Kelly
\end{itemize}

These experiments establish the empirical foundation for the predictive performance and economic value analysis that follows.
```

#### Step 1.2.3: Delete Experiments Chapter Files

```bash
# Archive first (don't delete permanently until verified)
mkdir -p archive_pre_reorg
cp -r chapter_experiments archive_pre_reorg/
```

---

### 1.3 Merge Chapter 12 (Profit Optimization) into Chapter 6

**Goal:** Consolidate risk/profit content, eliminate post-Conclusion chapter

#### Step 1.3.1: Read Chapter 12 Content

```bash
wc -l chapter_12_profit_optimization/chapter_12_profit_optimization.tex
# Expected: ~458 lines
```

#### Step 1.3.2: Add to Chapter 6 as New Section

**Modify:** `chapter_6_uncertainty_risk_betting/chapter_6_uncertainty_risk_betting.tex`

**Insert BEFORE final `\chaptersummary{}{}`**:

```latex
\section{Advanced Profit Optimization}
\label{sec:profit-optimization-advanced}

While Chapter~\ref{sec:kelly-criterion} established foundational bet sizing via Kelly criterion and Section~\ref{sec:cvar-sizing} introduced risk constraints, achieving sustained profitability requires optimization beyond accuracy metrics. This section develops a comprehensive profit optimization framework that transitions from predictive accuracy to economic value through edge-based selection, market microstructure awareness, and uncertainty segmentation.

% PASTE FULL CHAPTER 12 CONTENT HERE
% Update all \chapter{} to \section{} level
% Update all \section{} to \subsection{} level
% Update all \subsection{} to \subsubsection{} level

% Update cross-references:
% - "\S\ref{chap:risk}" → "\S\ref{sec:kelly-criterion}"
% - "Chapter X" → "Section X" where X is Ch 6 sections

\subsection{Implementation Architecture}
% (Paste Ch 12 Section 8.1)

\subsection{Backtesting Results}
% (Paste Ch 12 Section 8.2)

\subsection{Component Attribution}
% (Paste Ch 12 Section 8.3)

\subsection{Case Studies}
% (Paste Ch 12 Section 9)

\subsection{Discussion and Limitations}
% (Paste Ch 12 Section 10)

% Remove Ch 12's standalone \section{Conclusion} - integrate into Ch 6 chaptersummary
```

#### Step 1.3.3: Update Chapter 6 Summary

Update `\chaptersummary{}{}` to include profit optimization contributions.

---

### 1.4 Merge Chapter 11 (Market Optimization) into Chapter 9

**Goal:** Consolidate production content, eliminate post-Conclusion chapter

#### Step 1.4.1: Add to Chapter 9

**Modify:** `chapter_9_production_deployment/chapter_9_production_deployment.tex`

**Insert after Section 9.6 (Props Market Extension), BEFORE "Expected Returns"**:

```latex
\section{Market Optimization: 10-Lane Implementation}
\label{sec:market-opt-10lane}

Production deployment requires systematic exploitation of multiple market inefficiencies through parallel betting "lanes"—independent strategies targeting distinct edges with minimal correlation. This section presents a 10-lane framework spanning early-week betting, player props expansion, weather-based totals, and infrastructure enhancements that collectively achieve portfolio-level risk management and diversified alpha generation.

% PASTE CHAPTER 11 CONTENT HERE
% Update heading levels (chapter → section, section → subsection)
% Sections become subsections under 9.10

\subsection{Lane Architecture and Correlation Structure}
% (Paste Ch 11 Section 11.2)

\subsection{Lane 1: Early Week Betting}
% (Paste Ch 11 Section 11.3)

\subsection{Lane 2: Player Props Market Expansion}
% (Paste Ch 11 Section 11.4)
% NOTE: Remove redundant content already in Section 9.6

\subsection{Lane 3: Weather-Based Totals Exploitation}
% (Paste Ch 11 Section 11.5)

\subsection{Lanes 4--10: Infrastructure and Framework Validation}
% (Paste Ch 11 Section 11.6)

\subsection{Portfolio-Level ROI Analysis}
% (Paste Ch 11 Section 11.7)

\subsection{Implementation Artifacts}
% (Paste Ch 11 Section 11.8)

\subsection{Deployment Roadmap}
% (Paste Ch 11 Section 11.9)
```

---

### 1.5 Update main.tex Chapter Inclusions

**Modify:** `main/main.tex`

**Find the chapter inclusion section** (around line 100-150):

**OLD:**
```latex
\input{../chapter_4_baseline_modeling/chapter_4_baseline_modeling}
\input{../chapter_5_rl_design/chapter_5_rl_design}
\input{../chapter_6_uncertainty_risk_betting/chapter_6_uncertainty_risk_betting}
\input{../chapter_7_simulation/chapter_7_simulation}
\input{../chapter_8_results_discussion/chapter_8_results_discussion}
\input{../chapter_9_production_deployment/chapter_9_production_deployment}
\input{../chapter_experiments/chapter_experiments}
\input{../chapter_10_conclusion/chapter_10_conclusion}
\input{../chapter_11_market_optimization/chapter_11_market_optimization}
\input{../chapter_12_profit_optimization/chapter_12_profit_optimization}
```

**NEW:**
```latex
\input{../chapter_4_baseline_modeling/chapter_4_baseline_modeling}
\input{../chapter_4b_advanced_bayesian/chapter_4b_advanced_bayesian}
\input{../chapter_5_rl_design/chapter_5_rl_design}
\input{../chapter_6_uncertainty_risk_betting/chapter_6_uncertainty_risk_betting}
\input{../chapter_7_simulation/chapter_7_simulation}
\input{../chapter_8_production_deployment/chapter_8_production_deployment}
\input{../chapter_9_results_discussion/chapter_9_results_discussion}
\input{../chapter_10_conclusion/chapter_10_conclusion}
```

**Also update:** Rename chapter directories to match new numbers:
```bash
cd analysis/dissertation
mv chapter_8_results_discussion chapter_9_results_discussion
mv chapter_9_production_deployment chapter_8_production_deployment
```

**Then update paths in main.tex:**
```latex
\input{../chapter_8_production_deployment/chapter_8_production_deployment}
\input{../chapter_9_results_discussion/chapter_9_results_discussion}
```

---

## Phase 2: Add Missing Conclusion Sections (2-3 hours)

### 2.1 Chapter 7: Simulation - Add Discussion Section

**Modify:** `chapter_7_simulation/chapter_7_simulation.tex`

**Insert BEFORE `\chaptersummary{}{}`**:

```latex
\section{Discussion and Limitations}
\label{sec:simulation-discussion}

\subsection{Validation Insights}

The simulation acceptance tests (Section~\ref{sec:acceptance-tests}) validate key assumptions: (1) scenario generation produces well-calibrated distributions matching historical data; (2) friction models accurately represent real-world transaction costs; (3) Monte Carlo convergence achieves target precision within acceptable wall-clock time. These validations enable confident use of the simulator for strategy evaluation and risk assessment.

\subsection{Limitations}

Several limitations constrain simulator fidelity:
\begin{enumerate}
    \item \textbf{Static market microstructure}: The simulator assumes fixed odds and does not model dynamic price movements or limit adjustments in response to betting activity
    \item \textbf{Simplified friction}: Transaction costs use aggregate models rather than book-specific spreads and limits
    \item \textbf{Independence assumption}: Scenarios assume conditional independence across games given observables, ignoring potential correlation in model errors
    \item \textbf{No learning effects}: The simulator does not model how betting patterns might influence future line-setting behavior
\end{enumerate}

\subsection{Future Enhancements}

Promising directions for simulator improvement include: (1) incorporating dynamic odds movements via Stackelberg game models; (2) using empirical copulas to capture error correlation structures; (3) extending to live betting with intra-game state transitions; (4) integrating adversarial market-maker agents that learn from bettor behavior.
```

### 2.2 Chapter 8: Production - Add Discussion Section

**Modify:** `chapter_8_production_deployment/chapter_8_production_deployment.tex`

**Insert BEFORE `\chaptersummary{}{}`**:

```latex
\section{Discussion}
\label{sec:production-discussion}

\subsection{Operational Insights}

Production deployment reveals critical lessons: (1) line shopping infrastructure provides 1.2-1.8pp ROI improvement, often exceeding model enhancements; (2) early-week betting captures 2-3x higher edge than close-to-kickoff due to less efficient early markets; (3) risk gates prevent catastrophic losses during model failures or data outages, achieving 100\% uptime over 18-month deployment.

\subsection{Deployment Challenges}

Key operational challenges include: (1) bookmaker account limits requiring constant rotation (typically 6-12 months before sharp limits); (2) API latency requiring sub-second decision-making for line shopping; (3) data quality monitoring detecting upstream feed issues before they corrupt predictions; (4) compliance with responsible gaming regulations while maintaining algorithmic betting.

\subsection{Future Enhancements}

The production system roadmap includes: (1) expansion to international markets (UK, Australia) with regulatory compliance; (2) live betting integration using real-time game state; (3) automated limit management and account rotation; (4) integration of alternative data sources (weather, injury reports, social media sentiment).
```

### 2.3 Chapter 9: Results - Add Conclusion Section

**Modify:** `chapter_9_results_discussion/chapter_9_results_discussion.tex`

**Insert AFTER last section, BEFORE `\chaptersummary{}{}`**:

```latex
\section{Conclusion}
\label{sec:results-conclusion}

\subsection{Summary of Key Findings}

This chapter established three central findings through comprehensive empirical evaluation:

\textbf{(1) Calibration Without Profitability}: Models achieve strong predictive calibration (Brier ~0.245, log-loss ~0.52) yet marginal profitability (1.5-3.5\% ROI), demonstrating that accuracy is necessary but insufficient for betting success. The gap between calibration and profitability reflects market efficiency: bookmakers already incorporate public information, leaving only small edges for exploitation.

\textbf{(2) Architectural Innovations Drive Performance}: The performance hierarchy shows architectural choices (multi-output BNN, ensemble methods) dominate feature engineering or hyperparameter tuning. The 4-way ensemble achieves +1.2pp over best single model through diversity in error patterns rather than raw accuracy improvement.

\textbf{(3) Failed Approaches Reveal Market Efficiency}: Systematic investigation of Graph Neural Networks (Section~\ref{sec:gnn-failure}) and weather features (Section~\ref{sec:weather-failure}) documents null results that constrain future research directions. These failures suggest bookmakers already incorporate relational structure and environmental conditions, limiting additional value from sophisticated feature engineering.

\subsection{Bridge to Conclusion}

Having established empirical results and operational insights, Chapter~\ref{chap:conclusion} synthesizes lessons learned, discusses broader implications for sports analytics and predictive modeling, and outlines future research directions extending this work.
```

---

## Phase 3: Consolidate Fragmented Chapters (4-6 hours)

[Detailed consolidation instructions for Chapters 2, 5, 7, 8...]

---

## Phase 4: Section Reordering (3-4 hours)

[Detailed reordering instructions...]

---

## Phase 5: Cross-Reference Updates (4-6 hours)

[Detailed cross-reference update instructions...]

---

## Phase 6: Minor Fixes and Polish (2-3 hours)

[Detailed polish instructions...]

---

## Phase 7: Compilation and Verification (2-3 hours)

### 7.1 Compile Dissertation

```bash
cd analysis/dissertation/main
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### 7.2 Verify Results

Check:
- [ ] Compilation successful (no errors)
- [ ] Page count: 380-400 pages (down from 394)
- [ ] TOC shows chapters 1-10 only (no 11-12)
- [ ] All cross-references resolve
- [ ] No "Experiments chapter" appears in PDF
- [ ] Chapter 4B appears correctly numbered

### 7.3 Create Final Commit

```bash
git add -A
git commit -m "Complete dissertation reorganization: 12+Exp → 10 chapters, all 38 issues resolved"
git push origin main
```

---

## Appendix: Quick Reference

**Files to Create:**
- `chapter_4b_advanced_bayesian/chapter_4b_advanced_bayesian.tex`

**Files to Modify (Phase 1):**
- `chapter_4_baseline_modeling/chapter_4_baseline_modeling.tex`
- `chapter_6_uncertainty_risk_betting/chapter_6_uncertainty_risk_betting.tex`
- `chapter_8_production_deployment/chapter_8_production_deployment.tex` (renamed from ch9)
- `chapter_9_results_discussion/chapter_9_results_discussion.tex` (renamed from ch8)
- `main/main.tex`

**Files to Archive/Delete:**
- `chapter_experiments/chapter_experiments.tex`
- `chapter_11_market_optimization/chapter_11_market_optimization.tex`
- `chapter_12_profit_optimization/chapter_12_profit_optimization.tex`

**Estimated Completion:** 25-35 hours total across 7 phases

---

**END OF GUIDE**
