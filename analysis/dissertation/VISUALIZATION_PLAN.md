# Visualization Plan for BNN Calibration Study

## Overview
Create visualizations that tell the story of the calibration investigation and highlight key findings.

## Recommended Visualizations

### 1. Coverage Comparison Bar Chart ⭐ PRIORITY
**Purpose**: Show at-a-glance comparison of all methods against the 90% target

**Design**:
- Horizontal bar chart with methods on y-axis, coverage on x-axis
- Red dashed vertical line at 90% (target)
- Color code bars:
  - Red: <50% coverage (severely under-calibrated)
  - Orange: 50-80% (under-calibrated)
  - Yellow: 80-90% (approaching target)
  - Green: 90-95% (well-calibrated)
- Annotate each bar with exact coverage percentage

**Data**:
```
BNN Baseline: 26.2%
BNN Vegas: 29.7%
BNN Environment: 29.7%
BNN Opponent: 31.3%
Conformal: 84.5%
Quantile Reg: 89.4%
Multi-output BNN: 92.0%
```

**Key Insight**: Visual gap between standard BNNs (~30%) and well-calibrated methods (~90%)

---

### 2. Calibration-Sharpness Trade-off Scatter Plot ⭐ PRIORITY
**Purpose**: Visualize the fundamental trade-off between coverage and interval width

**Design**:
- X-axis: Average 90% CI width (yards)
- Y-axis: Actual 90% coverage (%)
- Horizontal line at 90% (target coverage)
- Each method as a point, sized by training time
- Arrows showing "ideal direction" (top-left: narrow + calibrated)
- Pareto frontier highlighted

**Data**:
```
BNN Baseline: (17, 26.2%, 25min)
BNN Vegas: (17, 29.7%, 25min)
BNN Opponent: (17, 31.3%, 30min)
Conformal: (66, 84.5%, 2min)
Quantile Reg: (106, 89.4%, 2min)
Multi-output: (?, 92.0%, 240min)
```

**Key Insight**: Multi-output BNN potentially achieves both good coverage and reasonable sharpness

---

### 3. Feature Ablation Progression Chart
**Purpose**: Show incremental impact of feature engineering

**Design**:
- Line chart with feature groups on x-axis
- Y-axis: 90% coverage
- Horizontal target line at 90%
- Points connected by line showing progression:
  Baseline → +Vegas → +Environment → +Opponent
- Annotate each point with coverage and $\Delta$ from baseline

**Key Insight**: Diminishing returns from feature engineering (5.1pp total improvement)

---

### 4. Reliability Diagram (Calibration Curve)
**Purpose**: Show calibration quality across different confidence levels

**Design**:
- X-axis: Predicted probability (from model)
- Y-axis: Observed frequency
- Diagonal line = perfect calibration
- Plot curves for each method showing deviation from diagonal
- Histogram below showing data density at each confidence level

**Data needed**: Would require binning predictions and calculating empirical coverage

**Key Insight**: Well-calibrated methods stay close to diagonal; BNNs systematically underestimate

---

### 5. Prediction Interval Visualization (Sample Cases)
**Purpose**: Show concrete examples of intervals from different methods

**Design**:
- Show 3-5 sample predictions with ground truth
- For each prediction, show intervals from all methods
- Visualize as error bars with:
  - Point estimate (dot)
  - 90% CI (thick bar)
  - Actual value (vertical line)
- Use color to indicate if interval contains truth

**Key Insight**: BNN intervals look precise but often miss; well-calibrated intervals are wider but more reliable

---

### 6. Training Time vs. Coverage Bubble Chart
**Purpose**: Show efficiency trade-offs

**Design**:
- X-axis: Training time (log scale)
- Y-axis: 90% coverage
- Bubble size: CI width
- Quadrants:
  - Top-left: Fast + calibrated (ideal)
  - Bottom-left: Fast but poor calibration
  - Top-right: Slow but calibrated
  - Bottom-right: Slow and poor (worst)

**Key Insight**: Quantile regression achieves excellent coverage with minimal compute

---

### 7. Prior Sensitivity Heatmap
**Purpose**: Show robustness to prior choice

**Design**:
- Heatmap with prior $\sigma$ values on one axis
- Coverage metric as color intensity
- Show that coverage barely changes across prior settings

**Data**: From Phase 2 prior sensitivity analysis

**Key Insight**: Prior choice is not the problem

---

## Implementation Priority

**High Priority** (include in dissertation):
1. Coverage Comparison Bar Chart - main result visualization
2. Calibration-Sharpness Scatter - shows fundamental trade-off
3. Reliability Diagram - shows quality of uncertainty estimates

**Medium Priority** (supplementary material):
4. Feature Ablation Progression - supports Phase 1 narrative
5. Prediction Interval Examples - intuitive understanding

**Low Priority** (can omit):
6. Training Time Bubble Chart - efficiency is secondary to calibration
7. Prior Sensitivity Heatmap - already shown in tables

---

## Technical Implementation Notes

### Tools
- **Matplotlib/Seaborn**: For standard charts (bars, scatter, lines)
- **PGFPlots** (via TikZ): For publication-quality LaTeX-native figures
- **ArviZ**: For Bayesian-specific diagnostics (if needed)

### Style Guidelines
- **Color palette**: Use colorblind-friendly palette (e.g., Tableau 10 or Viridis)
- **Font**: Match dissertation font (likely Computer Modern or similar)
- **Size**: Figure width should match \textwidth or \columnwidth
- **Labels**: Clear axis labels with units
- **Legend**: Positioned to not obscure data
- **Caption**: Detailed caption explaining what the figure shows

### Data Sources
All results are in:
- `experiments/calibration/*.json` (metrics for each method)
- `BNN_CALIBRATION_STUDY_RESULTS.md` (summary table)

---

## Recommended Figure Sequence in Dissertation

**Section: Calibration Crisis and Diagnosis**
- Figure 1: Coverage Comparison Bar Chart
  Caption: "90\% CI coverage across all methods. Standard BNNs severely under-calibrated (~30\%) regardless of features. Well-calibrated methods achieve 85-92\% coverage."

**Section: Feature Engineering Attempts**
- Figure 2: Feature Ablation Progression
  Caption: "Incremental coverage improvement from feature engineering. Adding opponent defense features yields best single-output BNN (31.3\%) but remains far from target."

**Section: Alternative UQ Methods**
- Figure 3: Calibration-Sharpness Trade-off
  Caption: "Trade-off between calibration (coverage) and sharpness (interval width). Multi-output BNN achieves strong calibration while maintaining point accuracy."

**Section: Model Reliability**
- Figure 4: Reliability Diagram
  Caption: "Calibration curves showing predicted vs. observed coverage. Well-calibrated methods (green) track diagonal; BNNs (red) systematically underestimate uncertainty."

---

## Next Steps
1. Generate Coverage Comparison Bar Chart (highest priority)
2. Create Calibration-Sharpness Scatter Plot
3. If time permits: Reliability Diagram
4. Export all figures as PDF for LaTeX inclusion
