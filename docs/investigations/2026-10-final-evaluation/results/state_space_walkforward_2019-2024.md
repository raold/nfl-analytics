## state_space_walkforward_margins.csv vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 633-653-31 | |
| Win rate | 49.2% | 46.5% to 52.0% |
| Breakeven | 51.6% | |
| ROI | -4.7% | -8.9% to -0.4% |
| P(win rate > breakeven) | p = 0.958 | |
| Spread MAE: model / market | 10.66 / 9.84 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.2% | should be ~50% |
| Pooled home cover rate | 48.6% | 756/1557, binomial p = 0.742 vs 49% |
| Line vs home margin correlation | +0.45 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | +0.024 | -0.107 to +0.155 |
| Log-loss: model / market | 0.6437 / 0.6082 | diff +0.0254 to +0.0455 |

