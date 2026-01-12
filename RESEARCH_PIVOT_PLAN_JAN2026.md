# Research Pivot Plan: From Complexity to Profitability
**Date:** January 11, 2026
**Status:** DRAFT
**Context:** Post-Dissertation "Deep Dive" & Leakage Remediation

---

## Executive Summary: The "Ferrari vs. Camry" Paradox

The comprehensive audit of the `main.tex` dissertation reveals a stark reality: we have built a "Ferrari" analytics system (TimescaleDB, 5-way ensembles, RL pipelines) that is being outperformed by a "Toyota Camry" (Simple Power Ratings).

*   **Complex Ensemble (Spread):** 51.0% Win Rate (Post-Leakage) -> **-7.5% ROI**
*   **Simple Power Ratings (Spread):** 74.9% Win Rate (2025 Holdout) -> **+43.0% ROI**
*   **Player Props:** +2.96% ROI (Diluted by noise)

This document outlines a strategic pivot from "Operational Optimization" (seeking marginal gains via hyper-tuning) to "Structural Realignment" (deploying what actually works).

---

## Vector A: The "Simplicity" Pivot (Spread Markets)

**Hypothesis:** Spread markets are efficient relative to play-by-play noise (which complex models overfit) but inefficient relative to season-level fundamentals (which simple models capture).

### 1. Deploy the "Toyota" (Power Ratings)
*   **Action:** Immediately promote the "Simple Power Ratings" model (2024 Point Diff + 2.5 HFA) to production for spread betting.
*   **Isolation:** Run as a standalone strategy pod, distinct from the 5-way ensemble.
*   **Research Task:** Test intra-season decay. Does the 2024 Point Diff signal fade by Week 8 of 2025? If so, implement a crossover point to *current season* differential.

### 2. The "Veto" Classifier
*   **Insight:** Feature pruning experiments showed that removing 75% of features improved **Win AUC** (Classification) while hurting **Spread MAE** (Regression).
*   **Action:** Train a lightweight XGBoost Classifier using only the **Top 25% features** (86 features).
*   **Logic:** Use this as a "Veto Gate". Only bet the Power Ratings spread pick if the Classifier's Win Probability agrees (e.g., > 55% Win Prob for a Moneyline/Spread correlation).

---

## Vector B: The "Stability" Pivot (Props Markets)

**Hypothesis:** The +2.96% ROI on props is an aggregate of "Signal" (QB/WR) and "Noise" (RB).

### 1. Positional Segmentation
*   **Finding:** Dissertation notes "QB/WR props are highly predictable (0.85 stability)" while "RB props are essentially noise (0.50 stability)."
*   **Action:** **Stop betting RB Rushing Yards immediately.** The 0.50 stability implies it is a coin flip.
*   **Reallocation:** Allocate 100% of the Props Bankroll to **QB Passing Yards** and **WR Receptions**.
*   **Model Split:** Train distinct models for each position group rather than a monolithic "Props Ensemble."

### 2. Same Game Parlays (SGP) - The "Correlation" Edge
*   **Insight:** We have a Copula framework (Chapter 7/8). Public books often price SGPs as independent events.
*   **Method:** Identify highly correlated pairs (e.g., `QB Passing Yards Over` + `WR1 Receiving Yards Over`).
*   **Execution:** Use Copula simulations to price the *joint probability*. If `P(A & B) > Book_Price(A & B)`, bet the SGP. This exploits the "Correlation Smile" that books often flatten.

---

## Vector C: The "Execution" Pivot (Operational Alpha)

**Hypothesis:** -7.5% ROI on spreads is partly due to the "4.5% vig barrier." Operational improvements can claw back 1-2%.

### 1. Automated "Steam Chasing"
*   **Insight:** Line shopping adds +0.5% EV. Steam chasing can add +1.5-2.0%.
*   **Action:** Monitor **Tier 1 Books** (Pinnacle/Circa). When they move (e.g., -3 to -3.5), *immediately* bet the "stale" -3 at **Tier 3 Books** (DraftKings/MGM) before they update.
*   **Requirement:** Low-latency polling of Tier 1 feeds.

### 2. The "Holdout" Truth (Purged CV)
*   **Action:** Implement **Purged K-Fold Cross-Validation**.
*   **Logic:** The "Post-Leakage Reality" (51% Win Rate) implies the current training pipeline still has subtle leakage. Purged CV drops the *season following* the test set from the training set to ensure no "future" rule changes or trends leak into the "past."

---

## Summary of Immediate Next Steps

| Priority | Strategy | Estimated ROI Impact | Complexity |
| :--- | :--- | :--- | :--- |
| **P0** | **Deploy "Power Ratings" Model** | **+20% to +40%** | Low |
| **P1** | **Cut RB Props / Double WR/QB** | **+2% to +5%** | Low |
| **P2** | **Implement SGP (Correlation) Betting** | **+5% to +10%** | High |
| **P3** | **Steam Chasing Automation** | **+1% to +3%** | Medium |
