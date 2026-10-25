## Game Outcome v2, 2025 regular season vs closing market

Season 2025, 272 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 117-124-1 | |
| Win rate | 48.5% | 42.3% to 54.8% |
| Breakeven | 52.3% | |
| ROI | -7.1% | -19.2% to +3.4% |
| P(win rate > breakeven) | p = 0.888 | |
| Spread MAE: model / market | 11.14 / 9.72 | DM p = 0.000 |
| Placebo win rate (shuffled) | 49.4% | should be ~50% |
| Pooled home cover rate | 49.8% | 135/271, binomial p = 0.808 vs 49% |
| Line vs home margin correlation | +0.50 | sign check: must be > 0 from 30 games |
| Model info beyond the line (coef on pred - line) | -0.572 | -1.565 to +0.420 |
| Log-loss: model / market | 0.6868 / 0.6094 | diff +0.0463 to +0.1082 |

**100% of predictions were made at or after kickoff (retroactive); this is not a prospective test.**
