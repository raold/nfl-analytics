# Market Benchmark of Baselines (2026-10-24)

Reproducible, database-free check of the retracted "74.9% ATS" claim and of what simple models
achieve against NFL closing lines. Data: `data/historical_weather_2018_2024.csv` (nflverse
closing spreads; 2018 seeds the preseason ratings). Command for each section:

```bash
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline <name> \
    --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024
```

Moneylines are not in this CSV, so the log-loss comparison against the de-vigged market needs
the database run (open item in PROJECT_STATUS.md).

### market vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 0-0-0 | |
| Win rate | n/a | n/a to n/a |
| Breakeven | 52.4% | |
| ROI | n/a | n/a to n/a |
| P(win rate > breakeven) | p = n/a | |
| Spread MAE: model / market | 9.84 / 9.84 | DM p = 1.000 |
| Placebo win rate (shuffled) | 49.8% | should be ~50% |
| Pooled home cover rate | 48.6% | sanity: 46-53% |


### power_ratings vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 609-608-31 | |
| Win rate | 50.0% | 47.2% to 52.8% |
| Breakeven | 52.4% | |
| ROI | -4.5% | -8.9% to +0.0% |
| P(win rate > breakeven) | p = 0.952 | |
| Spread MAE: model / market | 10.42 / 9.84 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.7% | should be ~50% |
| Pooled home cover rate | 48.6% | sanity: 46-53% |
| Model info beyond the line (coef on pred - line) | +0.079 | -0.076 to +0.233 |


### elo vs closing market

Seasons 2019-2024, 1599 games, bets at |edge| >= 1.0 pts.

| Metric | Value | 95% CI |
|---|---|---|
| ATS record | 581-596-31 | |
| Win rate | 49.4% | 46.5% to 52.2% |
| Breakeven | 52.4% | |
| ROI | -5.8% | -11.5% to -0.4% |
| P(win rate > breakeven) | p = 0.982 | |
| Spread MAE: model / market | 10.24 / 9.84 | DM p = 0.000 |
| Placebo win rate (shuffled) | 50.5% | should be ~50% |
| Pooled home cover rate | 48.6% | sanity: 46-53% |
| Model info beyond the line (coef on pred - line) | +0.100 | -0.121 to +0.322 |


## Reading the results

- **Self-test:** the closing line placed zero bets against itself and matched its own MAE
  (DM p = 1). Shuffled-prediction placebos sit at ~50%, and the pooled home cover rate is 48.6%,
  so the grading is unbiased.
- **Power ratings** (the model behind the retracted claim) go 609-608-31. The 95% interval on
  the win rate (47.2%-52.8%) is centred on 50% and only just reaches the 52.4% breakeven at its
  upper edge; the coefficient on (prediction − line) has an interval spanning zero, so the
  ratings carry no information the line lacks. Graded with the old sign error the same predictions
  "win" 74.1% (tests/unit/test_spread_convention.py).
- **Elo** is no better (49.4%).
- At ~1,200 bets the ROI interval is still about ±4.5 points wide; at 208 bets (the 2025
  season) it is about ±13. ROI alone cannot separate a small real edge from noise; log-loss
  against the de-vigged moneyline and closing-line value converge much faster.
