## state_space_walkforward_margins.csv vs closing market

Seasons 2025-2025, 107 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 39-42-0 | |
| Win rate | 48.1% | 37.6% to 58.9% |
| Breakeven | 52.2% | |
| ROI | -7.6% | -23.8% to +7.7% |
| P(win rate > breakeven) | p = 0.801 | |
| Spread MAE: model / market | 10.23 / 9.44 | DM p = 0.063 |
| Placebo win rate (shuffled) | 46.9% | should be ~50% |
| Pooled home cover rate | 52.3% | 56/107, binomial p = 0.500 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | -0.671 | -1.275 to -0.066 |
| Log-loss: model / market | 0.6325 / 0.6104 | diff -0.0039 to +0.0476 |

