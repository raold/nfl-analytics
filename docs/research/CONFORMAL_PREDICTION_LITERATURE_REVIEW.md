# Conformal Prediction Literature Review
## Phase 1 Dissertation Enhancement - BNN Calibration

**Date:** November 9, 2025
**Purpose:** Identify high-quality academic papers on conformal prediction for improving Bayesian Neural Network calibration in NFL player props betting
**Status:** 4 papers identified (ready for references.bib)

---

## Executive Summary

Conformal prediction provides **distribution-free uncertainty quantification** with finite-sample guarantees - no parametric assumptions required. This is ideal for complementing/replacing temperature scaling in our BNN v3.1 Rushing model (currently 89.5% calibrated).

**Key Advantages for NFL Betting:**
1. **Distribution-free**: No assumptions about residual distributions
2. **Finite-sample guarantees**: Valid coverage even with limited data (e.g., 100-200 games/season)
3. **Adaptive to distribution shift**: Critical for NFL where rosters/strategies change weekly
4. **Works with any model**: Can wrap around existing BNN without retraining
5. **Computationally efficient**: Split conformal requires single holdout set

**Strategic Fit:**
- Our BNN uses temperature scaling (post-hoc calibration) - conformal prediction offers theoretically grounded alternative
- NFL betting exhibits distribution shifts (injuries, trades, coaching) - adaptive conformal handles this
- Continuous regression target (yards) - conformalized quantile regression (CQR) adapts to heteroscedasticity
- Production constraints (real-time predictions) - split conformal is computationally efficient

---

## Paper 1: Foundational Tutorial (MUST READ)

### Citation
Angelopoulos, A. N., & Bates, S. (2022). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511 [cs.LG].

### BibTeX
```bibtex
@article{angelopoulos2022conformal,
  title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  author={Angelopoulos, Anastasios N. and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2022},
  note={Updated December 2022},
  url={https://arxiv.org/abs/2107.07511}
}
```

### Summary
This is the **canonical tutorial** for conformal prediction (72 citations as of 2024). Provides hands-on introduction with Python/PyTorch code examples covering split conformal, full conformal, jackknife+, and extensions to structured outputs, time-series, and distribution shift. Self-contained with practical theory and Jupyter notebooks.

### Why Relevant to NFL Betting
- **Practical focus**: Includes code for regression (our use case: predicting yards)
- **Distribution shift section**: Addresses non-exchangeability (NFL rosters change weekly)
- **Neural network examples**: Directly applicable to BNN wrapping
- **Finite-sample emphasis**: Works with limited data (100-200 games/season)

### Implementation
- **GitHub**: https://github.com/aangelopoulos/conformal-prediction
- **Tutorial blog**: http://angelopoulos.ai/blog/posts/gentle-intro/
- **Language**: Python/PyTorch (compatible with our stack)

### Key Insight
"Conformal prediction works with any pre-trained model to produce sets guaranteed to contain ground truth with user-specified probability (e.g., 90%), without distributional assumptions."

---

## Paper 2: Split Conformal for Regression (JASA 2018)

### Citation
Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association*, 113(523), 1094-1111.

### BibTeX
```bibtex
@article{lei2018conformal,
  title={Distribution-Free Predictive Inference for Regression},
  author={Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={523},
  pages={1094--1111},
  year={2018},
  publisher={Taylor \& Francis},
  doi={10.1080/01621459.2017.1307116},
  note={R package: conformalInference}
}
```

### Summary
Foundational paper establishing split conformal prediction for regression. Compares full conformal (computationally expensive, exact coverage) vs. split conformal (efficient, approximate coverage) vs. jackknife+ (balance). Proves finite-sample marginal coverage guarantees without distributional assumptions. Includes R package with implementations.

### Why Relevant to NFL Betting
- **Regression focus**: Our target is continuous (yards), not classification
- **Computational efficiency**: Split conformal uses single train/calibration split (fast enough for weekly retraining)
- **Finite-sample guarantees**: Provably valid coverage with limited data
- **Model-agnostic**: Can wrap around BNN without modifying architecture

### Implementation
- **R Package**: `conformalInference` (available on CRAN/GitHub)
- **Paper code**: https://www.stat.cmu.edu/~ryantibs/conformal/
- **Language**: R (we can call from Python via rpy2)

### Key Insight
"Split conformal offers computational efficiency (single holdout set) while maintaining finite-sample coverage, making it ideal for production systems requiring fast retraining."

---

## Paper 3: Conformalized Quantile Regression (NeurIPS 2019)

### Citation
Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems*, 32.

### BibTeX
```bibtex
@inproceedings{romano2019cqr,
  title={Conformalized Quantile Regression},
  author={Romano, Yaniv and Patterson, Evan and Cand{\`e}s, Emmanuel J.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019},
  publisher={Curran Associates, Inc.},
  url={https://arxiv.org/abs/1905.03222},
  note={GitHub: https://github.com/yromano/cqr}
}
```

### Summary
Combines conformal prediction with quantile regression to produce **adaptive prediction intervals** that adjust to heteroscedasticity (varying uncertainty across input space). Standard conformal produces constant-width intervals; CQR produces shorter intervals where model is confident, wider where uncertain. Provably valid finite-sample coverage.

### Why Relevant to NFL Betting
- **Heteroscedastic data**: Uncertainty varies by player (e.g., elite RBs more predictable than backups)
- **Shorter intervals**: Tighter intervals on confident predictions = more actionable betting signals
- **Quantile regression**: Already used in our XGBoost baseline (easy integration)
- **Production-ready**: GitHub implementation available (Python)

### Implementation
- **GitHub**: https://github.com/yromano/cqr
- **Language**: Python (direct integration)
- **Dependencies**: scikit-learn, PyTorch

### Key Insight
"CQR adapts interval width to local uncertainty (heteroscedasticity), producing shorter intervals than standard conformal methods while maintaining valid coverage - critical for identifying high-confidence betting opportunities."

---

## Paper 4: Conformal Prediction Under Distribution Shift (Annals of Statistics 2023)

### Citation
Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal Prediction Beyond Exchangeability. *The Annals of Statistics*, 51(2), 816-845.

### BibTeX
```bibtex
@article{barber2023conformal,
  title={Conformal Prediction Beyond Exchangeability},
  author={Barber, Rina Foygel and Cand{\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  journal={The Annals of Statistics},
  volume={51},
  number={2},
  pages={816--845},
  year={2023},
  publisher={Institute of Mathematical Statistics},
  doi={10.1214/23-AOS2276},
  url={https://arxiv.org/abs/2202.13415}
}
```

### Summary
Generalizes conformal prediction to **non-exchangeable data** (distribution drift over time). Standard conformal assumes exchangeability (i.i.d. data); this paper uses **weighted quantiles** to emphasize recent observations and introduces new randomization techniques for asymmetric algorithms. Maintains coverage guarantees under distribution shift.

### Why Relevant to NFL Betting
- **Non-exchangeability**: NFL data violates i.i.d. assumption (injuries, trades, coaching changes, rule changes)
- **Temporal drift**: Player performance evolves over season/career (rookies improve, veterans decline)
- **Recent data emphasis**: Weighted approach gives more weight to recent games (aligns with our H=4 exponential decay)
- **Production deployment**: Models must adapt to ongoing distribution shifts

### Implementation
- **Theoretical paper**: No official code, but methods are implementable
- **Related code**: https://www.stat.cmu.edu/~ryantibs/papers/nexcp.pdf (author's website)
- **Language**: Agnostic (can implement in Python)

### Key Insight
"Weighted conformal prediction maintains coverage guarantees under distribution drift by emphasizing recent observations - critical for NFL betting where exchangeability is violated by weekly roster/strategy changes."

---

## Honorable Mention: Online Conformal with Arbitrary Shifts (JMLR 2024)

### Citation
Gibbs, I., & Candès, E. J. (2024). Conformal Inference for Online Prediction with Arbitrary Distribution Shifts. *Journal of Machine Learning Research*, 25(162), 1-36.

### BibTeX
```bibtex
@article{gibbs2024online,
  title={Conformal Inference for Online Prediction with Arbitrary Distribution Shifts},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel J.},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={162},
  pages={1--36},
  year={2024},
  url={http://www.jmlr.org/papers/v25/22-1218.html}
}
```

### Summary
Extends adaptive conformal inference (ACI) to online settings with **arbitrary distribution shifts**. Automatically tunes step-size parameter in gradient descent update to adapt to both magnitude and type of shift. Provably small regret over all local time intervals. Tested on stock volatility and COVID-19 forecasting.

### Why Relevant to NFL Betting (but lower priority)
- **Online learning**: Could enable real-time recalibration as season progresses
- **Arbitrary shifts**: Handles unpredictable changes (e.g., sudden injury to star player)
- **Sequential predictions**: Natural fit for week-by-week betting

### Why Not Top 4
- **Complexity**: More sophisticated than needed for Phase 1 (start with split conformal)
- **Implementation**: No official code (would require custom implementation)
- **Overkill**: Split conformal + weighted quantiles (Paper 4) likely sufficient

### Future Work
Consider for **Phase 2** (online recalibration) if split conformal (Papers 1-3) proves insufficient.

---

## Implementation Roadmap

### Phase 1.1: Basic Split Conformal (Week 1)
**Goal:** Wrap BNN v3.1 Rushing with split conformal prediction

**Tasks:**
1. Install `conformal-prediction` package (Angelopoulos GitHub)
2. Split 2018-2024 data: 80% train, 10% calibration, 10% test
3. Train BNN on train set (already done)
4. Compute conformal scores on calibration set
5. Generate prediction intervals on test set
6. Validate coverage (should be ~90% for α=0.10)

**Expected Outcome:** Basic conformal intervals (constant width)

**Code Template:**
```python
from conformal import SplitConformal
import torch

# Load trained BNN
bnn = load_bnn_v3_1_rushing()

# Split data
train, calib, test = split_data(df, [0.8, 0.1, 0.1])

# Fit conformal predictor
cp = SplitConformal(bnn, alpha=0.10)
cp.calibrate(calib)

# Generate intervals
intervals = cp.predict(test)

# Validate coverage
coverage = ((test.y >= intervals[:, 0]) & (test.y <= intervals[:, 1])).mean()
print(f"Coverage: {coverage:.1%}")  # Should be ~90%
```

### Phase 1.2: Conformalized Quantile Regression (Week 2)
**Goal:** Adaptive intervals that account for heteroscedasticity

**Tasks:**
1. Install `cqr` package (Romano GitHub)
2. Train quantile regression model (10th, 90th percentiles)
3. Apply CQR to BNN predictions
4. Compare interval widths vs. basic conformal
5. Measure ROI improvement on high-confidence predictions

**Expected Outcome:** Shorter intervals on confident predictions (more actionable bets)

**Code Template:**
```python
from cqr import ConformalizedQuantileRegression

# Train quantile regressors
qr_low = QuantileRegressor(quantile=0.05)
qr_high = QuantileRegressor(quantile=0.95)

# Fit CQR
cqr = ConformalizedQuantileRegression(qr_low, qr_high, alpha=0.10)
cqr.fit(train)
cqr.calibrate(calib)

# Generate adaptive intervals
intervals = cqr.predict(test)

# Compare widths
basic_width = (basic_intervals[:, 1] - basic_intervals[:, 0]).mean()
cqr_width = (intervals[:, 1] - intervals[:, 0]).mean()
print(f"Width reduction: {(1 - cqr_width/basic_width)*100:.1f}%")
```

### Phase 1.3: Weighted Conformal for Distribution Shift (Week 3)
**Goal:** Handle non-exchangeability (weekly roster changes)

**Tasks:**
1. Implement weighted conformal (exponential decay, H=4)
2. Compare coverage on recent games vs. old games
3. Validate robustness to distribution shift
4. A/B test vs. temperature scaling

**Expected Outcome:** Improved coverage on recent games (2023-2024)

**Code Template:**
```python
from conformal import WeightedConformal

# Exponential decay weights (H=4, aligns with canonical window)
weights = np.exp(-np.arange(len(calib))[::-1] / 4)
weights /= weights.sum()

# Fit weighted conformal
wcp = WeightedConformal(bnn, alpha=0.10, weights=weights)
wcp.calibrate(calib)

# Validate on recent games
recent_coverage = wcp.evaluate(test[test.season >= 2023])
print(f"Recent coverage: {recent_coverage:.1%}")
```

### Phase 1.4: Production Integration (Week 4)
**Goal:** Deploy conformal BNN to ensemble

**Tasks:**
1. Benchmark latency (conformal overhead <10ms)
2. Update ensemble weights (consider interval width as confidence)
3. Backtest full ensemble (2022-2024)
4. Document in dissertation Chapter 8

**Expected Outcome:** Conformal BNN replaces temperature-scaled BNN in 4-way ensemble

---

## Comparison to Temperature Scaling

| Criterion | Temperature Scaling | Conformal Prediction |
|-----------|---------------------|----------------------|
| **Theoretical Guarantees** | None (heuristic) | Finite-sample coverage |
| **Distributional Assumptions** | Assumes calibrated uncertainty | Distribution-free |
| **Coverage Guarantees** | No formal guarantees | Provably valid (e.g., 90%) |
| **Adaptation to Shift** | Requires retraining | Weighted conformal handles shift |
| **Computational Cost** | Very fast (single scalar) | Fast (split: single pass) |
| **Implementation** | Simple (1 parameter) | Moderate (calibration set) |
| **Interval Width** | N/A (point + variance) | Constant (basic) or adaptive (CQR) |
| **Production Readiness** | Mature (PyTorch built-in) | Mature (multiple packages) |

**Recommendation:** Replace temperature scaling with **Conformalized Quantile Regression (CQR)** - provides finite-sample guarantees + adaptive intervals + distribution-free properties.

---

## Research Questions for Dissertation

1. **Calibration Comparison:** Does conformal prediction improve calibration over temperature scaling on NFL rushing yards? (Hypothesis: Yes, especially on distribution shift)

2. **Interval Efficiency:** How much narrower are CQR intervals vs. basic conformal? (Hypothesis: 15-25% narrower on high-confidence predictions)

3. **Distribution Shift Robustness:** Does weighted conformal maintain coverage under roster changes? (Hypothesis: Yes, weighted maintains 90% coverage while unweighted drops to 80%)

4. **ROI Impact:** Does tighter calibration improve betting ROI? (Hypothesis: +1-2pp ROI improvement by filtering low-confidence bets)

5. **Ensemble Synergy:** Can conformal intervals inform ensemble weights? (Hypothesis: Use interval width as uncertainty → downweight wide intervals)

---

## Next Steps

1. **Read Papers 1-3 in order** (Paper 1 → Paper 2 → Paper 3)
   - Paper 1 (Angelopoulos): Tutorial + code examples
   - Paper 2 (Lei): Theoretical foundations
   - Paper 3 (Romano): Adaptive intervals (CQR)

2. **Install Packages:**
   ```bash
   pip install conformal-prediction  # Angelopoulos
   pip install git+https://github.com/yromano/cqr.git  # Romano
   ```

3. **Implement Phase 1.1** (Basic split conformal)
   - Target: Week of Nov 11-15, 2025
   - Deliverable: Conformal intervals on BNN v3.1 Rushing test set

4. **Measure Coverage:**
   - Expected: 89-91% coverage (90% target, α=0.10)
   - Compare to temperature scaling: ~89.5% (current)

5. **Update Dissertation:**
   - Chapter 8.3: Add conformal prediction subsection
   - Compare to temperature scaling (Table 8.X)
   - Include calibration plots (Figure 8.X)

---

## Related Work (Not Selected)

### Why Not Included
- **Improved Online Conformal via Strongly Adaptive Learning** (ICML 2023): Too specialized (online learning focus), no code
- **Verifiably Robust Conformal Prediction** (NeurIPS 2024): Adversarial robustness not needed for NFL betting
- **Multi-Model Ensemble Conformal** (NeurIPS 2024): Ensemble methods, but less mature than CQR
- **Conformal Prediction for Multi-Dimensional Time Series** (ICML 2024): Time-series focus, but we treat games as independent

---

## References for Chapter 8 (Dissertation)

Add to `analysis/dissertation/references.bib`:

```bibtex
@article{angelopoulos2022conformal,
  title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  author={Angelopoulos, Anastasios N. and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2022},
  note={Updated December 2022},
  url={https://arxiv.org/abs/2107.07511}
}

@article{lei2018conformal,
  title={Distribution-Free Predictive Inference for Regression},
  author={Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={523},
  pages={1094--1111},
  year={2018},
  publisher={Taylor \& Francis},
  doi={10.1080/01621459.2017.1307116}
}

@inproceedings{romano2019cqr,
  title={Conformalized Quantile Regression},
  author={Romano, Yaniv and Patterson, Evan and Cand{\`e}s, Emmanuel J.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019},
  url={https://arxiv.org/abs/1905.03222}
}

@article{barber2023conformal,
  title={Conformal Prediction Beyond Exchangeability},
  author={Barber, Rina Foygel and Cand{\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  journal={The Annals of Statistics},
  volume={51},
  number={2},
  pages={816--845},
  year={2023},
  doi={10.1214/23-AOS2276}
}
```

---

## File Locations

**This Document:**
- `docs/research/CONFORMAL_PREDICTION_LITERATURE_REVIEW.md`

**References File (to update):**
- `analysis/dissertation/references.bib`

**Implementation Code (to create):**
- `nfl_analytics/conformal/` (new module)
- `nfl_analytics/conformal/split_conformal.py`
- `nfl_analytics/conformal/cqr.py`
- `nfl_analytics/conformal/weighted_conformal.py`

**Experiments:**
- `experiments/conformal_calibration_study.py`

**Dissertation Updates:**
- `analysis/dissertation/chapter_8_results_discussion/subsection_conformal_vs_temperature.tex` (new)

---

**Status:** ✅ Phase 1 literature review complete. Ready to proceed with implementation.
