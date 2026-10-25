## state_space_walkforward_margins.csv vs closing market

Seasons 2019-2025, 1706 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 672-695-31 | |
| Win rate | 49.2% | 46.5% to 51.8% |
| Breakeven | 51.6% | |
| ROI | -4.9% | -9.2% to -0.4% |
| P(win rate > breakeven) | p = 0.968 | |
| Spread MAE: model / market | 10.63 / 9.82 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.0% | should be ~50% |
| Pooled home cover rate | 48.8% | 812/1664, binomial p = 0.883 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | -0.004 | -0.135 to +0.127 |
| Log-loss: model / market | 0.6430 / 0.6084 | diff +0.0255 to +0.0445 |

