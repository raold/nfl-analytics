# Market Facts from nflverse Schedules (2026-10-24)

How NFL closing lines have behaved since 1999, measured from the public nflverse schedule
file, with the completed games of the 2026 season as an out-of-sample check.

> **Every number here is a historical association measured after the fact against closing
> lines. None of it is a betting edge.** Bets are placed before the close, at other lines and
> prices. The 15 mph wind threshold and the era boundaries were chosen by an earlier survey
> that had already seen these data; the Wong teaser ranges are a published rule. Nothing here
> was tested prospectively.
> For the two patterns that look exploitable (wind unders and Wong teasers), the sections below
> say what a prospective test would need. At the rate qualifying games occur, either test
> would take decades.

## Data and reproduction

- Source: nflverse schedules,
  <https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv>, downloaded
  2026-10-24 16:54 EDT (file mtime 2026-10-24T20:54:34Z). 2,186,406 bytes, sha256
  `5d882d05c73ab11758cfe70e71e994e50b53e39b974943368dfee500f9a50693`, 7,548 rows, seasons
  1999-2026.
- Historical window 1999-2025: 7,276 completed games. Holdout: the 94 games of 2026 completed
  by the download (weeks 1-7, the last on 2026-10-22) out of 272 scheduled.
- Command (a few seconds; seeded, so the outputs are identical for the same input file):

```bash
uv run --no-sync python scripts/market_facts.py --games <path to games.csv> \
    --out docs/investigations/2026-10-market-facts/results
```

The script ([scripts/market_facts.py](../../../scripts/market_facts.py)) writes one CSV per
table to [results/](results/), plus [results/summary.md](results/summary.md), its printed
report. Every number on this page is copied from that report or its CSVs. A newer download
changes the 2026 rows, and the older rows too if nflverse revises them; compare the sha256
in [results/provenance.csv](results/provenance.csv).

**Conventions.** `spread_line` is the expected home margin, positive when the home team is
favoured (2018_01_BUF_BAL = +7.5; the script checks this at load). The home team covers when
home margin - spread_line > 0, computed with
[nfl_analytics/utils/spread_convention.py](../../../nfl_analytics/utils/spread_convention.py).
Pushes are excluded from every rate. "Non-neutral" drops the 94 neutral-site games of
1999-2025: 27 Super Bowls and 67 others, mostly international games (39 at the three London
stadiums alone; [results/01_neutral_sites.csv](results/01_neutral_sites.csv)). "Outdoor" means
`roof` is `outdoors` or `open`. "Playoffs" is every game whose type is not REG.

**Methods.** All intervals are 95%:

- rates: Wilson
- means: t
- differences of rates: Newcombe hybrid score
- differences of means: Welch
- Brier, log-loss and ROI: week-block bootstrap (2,000 draws)
- regression slopes: week-clustered (CR1) standard errors

Three facts are framed as tests (wind, teasers, playoffs). Their eight p-values form one
family with Holm's step-down correction; see [Tests](#tests-and-the-holm-correction). The
exact binomial tests treat games as independent, so each is repeated with a week-clustered
standard error as a sensitivity check. Everything else is descriptive: an estimate and its
interval, without a p-value.

## Summary

Each row gives the 1999-2025 result unless noted, and the 2026 games so far as the holdout.

| Section | Fact | Estimate [95% CI] | n | 2026 holdout |
|---|---|---|---|---|
| 1 | Home team covers the closing spread (non-neutral) | 49.0% [47.9, 50.2] | 6,988 decided | 47.6% [37.3, 58.2], n=84 |
| 1 | Closing-line MAE vs the final margin | 10.27 [10.08, 10.46] pts | 7,276 | 9.24 [7.73, 10.74], n=94 |
| 1 | Slope of home margin on the closing spread (1 = calibrated) | 1.043 [0.994, 1.092] | 7,276 | 0.799 [0.324, 1.275] |
| 2 | Games decided by exactly 6, 2015-2025 vs 1999-2014 | 6.9% vs 5.3% (+1.62 [+0.50, +2.77] pct pts) | 3,028 / 4,248 | 9.6% [5.1, 17.2] |
| 2 | Games decided by exactly 10, same periods | 4.8% vs 6.0% (-1.29 [-2.33, -0.23] pct pts) | 3,028 / 4,248 | 4.3% [1.7, 10.4] |
| 2 | Exactly 3 / exactly 7, same periods | 14.8% vs 15.3% / 8.7% vs 9.3% (no clear change) | 3,028 / 4,248 | 17.0% / 11.7% |
| 3 | Home edge priced by the close (mean spread, non-neutral) | 2.57 (1999-2006) to 1.74 (2021-2025) pts | 2,102 / 1,392 | 1.79, n=89 |
| 3 | 2020, played mostly without crowds: home margin vs spread | 0.17 [-1.54, 1.88] vs 1.17 [0.39, 1.95] pts | 265 | |
| 4 | De-vigged moneyline: log-loss gain over a constant (2006-2025) | 0.0768 [0.0672, 0.0859] | 5,281 | 0.0361 [-0.0014, 0.0697], n=93 |
| 4 | Moneyline recalibration slope (1 = calibrated) | 1.035 [0.956, 1.113] | 5,281 | 0.796 [0.304, 1.288] |
| 5 | Totals: over rate | 49.4% [48.3, 50.6] | 7,170 decided | 51.1% [41.1, 60.9], n=94 |
| 6 | Unders at game-time wind of 15 mph or more | 57.0% [53.2, 60.7] | 674 decided | no qualifying game yet (n=0) |
| 6 | (total - closing total) per mph of wind | -0.181 [-0.249, -0.114] | 5,205 | -0.048 [-2.491, +2.396], n=44 (weeks 1-4 only) |
| 7 | Wong teaser legs (six points) | 74.7% [72.3, 76.9] | 1,391 decided | 73.5% [56.9, 85.4], n=34 |
| 8 | Closing-line MAE, playoffs minus regular season | +0.16 [-0.83, +1.15] pts | 309 / 6,967 | no playoff games yet |

After the Holm correction (8 tests), the association between wind and the closing total
survives: unders above 50% (Holm p = 0.001) and a non-zero slope (Holm p = 1.5e-06). None of
the tests that would mean a profit survives:

- Unders above the -110 break-even: Holm p = 0.056.
- Wong legs above the two-team break-even at -110, -120 or -130: Holm p of 0.138 or more.
- A difference in closing-line accuracy between playoffs and the regular season: Holm p = 1.000.

With week-clustered standard errors instead of independent games, every Holm decision stays the
same (unders above -110: Holm p = 0.054; Wong legs at -110: 0.173).

## 1. Closing-spread calibration

The closing spread shows no detectable bias or miscalibration in any era.

- In each of the four eras, the interval on the home cover rate contains 50%.
- The pooled rate is 49.0% [47.9, 50.2] over 6,988 decided non-neutral games.
- The mean of (home margin - line) is +0.09 [-0.21, +0.40] points.
- The regression of home margin on the line has slope 1.043 [0.994, 1.092]; 1 means calibrated.

Closing-line MAE fell from 10.42 (1999-2006) and 10.70 (2007-2013) to 9.77 (2020-2025).

With the neutral-site games included, 7,081 games are decided and 195 push, and 49.05% are
home covers (0.490467 in [results/01_spread_cover_rate.csv](results/01_spread_cover_rate.csv)).
A read-only query of the repaired `mart.game_summary` view (migration 104) on 2026-10-24
returned the same 7,276 games, 7,081 decided, 195 pushes and 0.490467, so the database and
nflverse agree:

```sql
SELECT count(*), count(home_covered), round(avg(home_covered::numeric), 6),
       count(*) FILTER (WHERE home_cover_margin = 0)
FROM mart.game_summary
WHERE season BETWEEN 1999 AND 2025 AND home_score IS NOT NULL AND spread_close IS NOT NULL;
```

| Window       | Sites              | Games   | Pushes   | Home cover         |
|:-------------|:-------------------|:--------|:---------|:-------------------|
| 1999-2006    | non-neutral        | 2102    | 66       | 49.4% [47.2, 51.5] |
| 2007-2013    | non-neutral        | 1851    | 47       | 48.9% [46.6, 51.2] |
| 2014-2019    | non-neutral        | 1572    | 48       | 48.2% [45.7, 50.7] |
| 2020-2025    | non-neutral        | 1657    | 33       | 49.6% [47.2, 52.1] |
| 1999-2025    | non-neutral        | 7182    | 194      | 49.0% [47.9, 50.2] |
| 1999-2025    | all, incl. neutral | 7276    | 195      | 49.0% [47.9, 50.2] |
| 2026 holdout | non-neutral        | 89      | 5        | 47.6% [37.3, 58.2] |

The next table covers every game with a line, so it includes neutral sites.

| Window       | Games   | MAE (pts)            | Mean of margin - line   | Slope of margin on line   |
|:-------------|:--------|:---------------------|:------------------------|:--------------------------|
| 1999-2006    | 2112    | 10.42 [10.07, 10.77] | +0.18 [-0.38, +0.75]    | 1.007 [0.912, 1.103]      |
| 2007-2013    | 1869    | 10.70 [10.32, 11.09] | +0.06 [-0.56, +0.68]    | 1.085 [0.995, 1.176]      |
| 2014-2019    | 1602    | 10.10 [9.69, 10.50]  | -0.23 [-0.87, +0.41]    | 1.039 [0.929, 1.150]      |
| 2020-2025    | 1693    | 9.77 [9.38, 10.15]   | +0.32 [-0.28, +0.93]    | 1.041 [0.940, 1.141]      |
| 1999-2025    | 7276    | 10.27 [10.08, 10.46] | +0.09 [-0.21, +0.40]    | 1.043 [0.994, 1.092]      |
| 2026 holdout | 94      | 9.24 [7.73, 10.74]   | -1.48 [-3.89, +0.92]    | 0.799 [0.324, 1.275]      |

<details>
<summary>Closing-line MAE by season</summary>

| Season         | Games   | MAE (pts)            | Mean of margin - line   |
|:---------------|:--------|:---------------------|:------------------------|
| 1999           | 259     | 10.37 [9.35, 11.39]  | +0.74 [-0.89, +2.37]    |
| 2000           | 259     | 10.71 [9.71, 11.71]  | +0.49 [-1.16, +2.14]    |
| 2001           | 259     | 10.26 [9.26, 11.26]  | -0.16 [-1.77, +1.45]    |
| 2002           | 267     | 10.36 [9.32, 11.41]  | +0.27 [-1.36, +1.90]    |
| 2003           | 267     | 10.54 [9.52, 11.56]  | +1.06 [-0.57, +2.68]    |
| 2004           | 267     | 10.39 [9.47, 11.31]  | +0.05 [-1.51, +1.60]    |
| 2005           | 267     | 9.99 [8.98, 11.00]   | +0.95 [-0.62, +2.52]    |
| 2006           | 267     | 10.75 [9.77, 11.73]  | -1.91 [-3.52, -0.30]    |
| 2007           | 267     | 10.75 [9.79, 11.71]  | +0.27 [-1.35, +1.88]    |
| 2008           | 267     | 11.11 [10.11, 12.12] | -0.28 [-1.95, +1.39]    |
| 2009           | 267     | 11.28 [10.21, 12.34] | -0.25 [-1.98, +1.48]    |
| 2010           | 267     | 10.75 [9.69, 11.82]  | -0.67 [-2.35, +1.00]    |
| 2011           | 267     | 10.51 [9.45, 11.57]  | +0.96 [-0.69, +2.61]    |
| 2012           | 267     | 10.72 [9.65, 11.78]  | +0.00 [-1.67, +1.67]    |
| 2013           | 267     | 9.81 [8.82, 10.79]   | +0.39 [-1.15, +1.93]    |
| 2014           | 267     | 11.28 [10.20, 12.37] | +0.16 [-1.58, +1.91]    |
| 2015           | 267     | 10.12 [9.14, 11.10]  | -0.54 [-2.10, +1.03]    |
| 2016           | 267     | 9.04 [8.14, 9.95]    | +0.68 [-0.74, +2.10]    |
| 2017           | 267     | 10.08 [9.06, 11.10]  | +0.47 [-1.12, +2.05]    |
| 2018           | 267     | 9.87 [8.85, 10.89]   | -0.26 [-1.83, +1.30]    |
| 2019           | 267     | 10.18 [9.19, 11.17]  | -1.87 [-3.43, -0.31]    |
| 2020           | 269     | 9.79 [8.82, 10.76]   | -0.96 [-2.48, +0.57]    |
| 2021           | 285     | 10.67 [9.69, 11.64]  | +0.12 [-1.47, +1.70]    |
| 2022           | 284     | 8.78 [7.92, 9.65]    | +0.29 [-1.06, +1.63]    |
| 2023           | 285     | 9.98 [8.98, 10.99]   | +1.13 [-0.40, +2.66]    |
| 2024           | 285     | 9.70 [8.74, 10.66]   | +0.68 [-0.80, +2.16]    |
| 2025           | 285     | 9.67 [8.79, 10.55]   | +0.61 [-0.82, +2.04]    |
| 2026 (holdout) | 94      | 9.24 [7.73, 10.74]   | -1.48 [-3.89, +0.92]    |

</details>

Two seasons out of 27 (2006 and 2019) have a mean-error interval that excludes zero, about
what chance gives at 95%. The 2026 games so far (MAE 9.24, mean error -1.48) are within the
range of single seasons since 1999 (MAE 8.78-11.28, mean error -1.91 to +1.13).

## 2. Key numbers

The table gives the share of all completed games (REG and POST) decided by exactly |margin|
points. From 2015, extra points have been snapped from the 15-yard line.

| abs(margin)   | 1999-2014 (n=4248)   | 2015-2025 (n=3028)   | Change, 2015-2025 minus 1999-2014 (pct pts)   | 2026 holdout (n=94)   |
|:--------------|:---------------------|:---------------------|:----------------------------------------------|:----------------------|
| 3             | 15.3% [14.3, 16.4]   | 14.8% [13.6, 16.1]   | -0.50 [-2.15, +1.18]                          | 17.0% [10.8, 25.9]    |
| 7             | 9.3% [8.4, 10.2]     | 8.7% [7.7, 9.7]      | -0.59 [-1.91, +0.76]                          | 11.7% [6.7, 19.8]     |
| 6             | 5.3% [4.7, 6.0]      | 6.9% [6.1, 7.9]      | +1.62 [+0.50, +2.77]                          | 9.6% [5.1, 17.2]      |
| 10            | 6.0% [5.4, 6.8]      | 4.8% [4.1, 5.6]      | -1.29 [-2.33, -0.23]                          | 4.3% [1.7, 10.4]      |
| 1             | 3.9% [3.4, 4.5]      | 4.5% [3.8, 5.3]      | +0.62 [-0.31, +1.58]                          | 1.1% [0.2, 5.8]       |
| 2             | 3.7% [3.1, 4.3]      | 4.6% [3.9, 5.4]      | +0.92 [-0.00, +1.88]                          | 5.3% [2.3, 11.9]      |
| 8             | 3.3% [2.8, 3.9]      | 4.3% [3.6, 5.1]      | +0.97 [+0.09, +1.90]                          | 5.3% [2.3, 11.9]      |

Since 2015:

- Margins of 6 and 8 are more common and margins of 10 less common; those three intervals
  exclude zero.
- Margins of 2 have risen too, but that interval just reaches zero.
- The two largest key numbers barely moved: 3 is 14.8% against 15.3%, and 7 is 8.7% against
  9.3%. Both intervals include zero.

The seven intervals are not adjusted for multiplicity. The years since 2015 also differ in
other rules and in coaching decisions, so this is an association with the period, not an
estimate of the PAT rule's effect. The full distribution of margins is in
[results/02_margin_distribution.csv](results/02_margin_distribution.csv).

## 3. Home-field advantage

The table covers non-neutral games; all figures are in points.

| Window       | Games   | Mean home margin   | Mean closing spread   | Margin - spread      | Home cover         |
|:-------------|:--------|:-------------------|:----------------------|:---------------------|:-------------------|
| 1999-2006    | 2102    | 2.74 [2.12, 3.36]  | 2.57 [2.32, 2.81]     | +0.17 [-0.40, +0.74] | 49.4% [47.2, 51.5] |
| 2007-2013    | 1851    | 2.72 [2.02, 3.41]  | 2.58 [2.30, 2.86]     | +0.14 [-0.48, +0.76] | 48.9% [46.6, 51.2] |
| 2014-2019    | 1572    | 2.05 [1.34, 2.76]  | 2.29 [2.00, 2.58]     | -0.24 [-0.88, +0.41] | 48.2% [45.7, 50.7] |
| 2020         | 265     | 0.17 [-1.54, 1.88] | 1.17 [0.39, 1.95]     | -1.00 [-2.53, +0.53] | 49.4% [43.5, 55.4] |
| 2021-2025    | 1392    | 2.25 [1.50, 3.00]  | 1.74 [1.42, 2.06]     | +0.51 [-0.15, +1.18] | 49.7% [47.0, 52.3] |
| 1999-2025    | 7182    | 2.39 [2.06, 2.73]  | 2.30 [2.16, 2.43]     | +0.10 [-0.21, +0.40] | 49.0% [47.9, 50.2] |
| 2026 holdout | 89      | 0.75 [-1.86, 3.36] | 1.79 [0.72, 2.86]     | -1.04 [-3.52, +1.44] | 47.6% [37.3, 58.2] |

The home edge priced by the close (the mean closing spread) and the edge on the field moved
together:

| Period | Priced home edge (pts) | Realised home margin (pts) |
|---|---|---|
| 1999-2013 | about 2.6 | 2.74 and 2.72 |
| 2014-2019 | 2.29 | 2.05 |
| 2020 | 1.17 | 0.17 |
| 2021-2025 | 1.74 | 2.25 |

Over 1999-2025 the realised margin minus the spread is +0.10 [-0.21, +0.40] points, so the
close tracked home advantage.

2020 was played mostly without crowds. The closing spread gave home teams 1.17 points that
season, about half the 2.29 of 2014-2019, and home teams still fell about a point short of
even that (-1.00 [-2.53, +0.53], an interval that includes zero).

In 2021-2025 home teams beat the spread by +0.51 [-0.15, +1.18] points on average. That fits
a market that cut the home edge slightly too far after 2020, but the interval includes zero,
and home teams covered only 49.7% [47.0, 52.3] of those games. A mean above zero with a cover
rate below 50% means the surplus came from the size of some home wins, not their frequency,
and backing every home team would still have lost to the vig.

Results by season are in
[results/03_home_field_by_season.csv](results/03_home_field_by_season.csv).

## 4. Moneyline calibration and the favourite-longshot pattern

Moneylines are in the file from 2006. Coverage is partial in 2006 (220 games), 2008 (194)
and 2009 (253), and nearly complete in every other season. Probabilities are de-vigged
proportionally. Ties are excluded (14 in 2006-2025 and 1 in 2026), because a tie refunds a
two-way moneyline bet.

| Window       | Games   | Constant                        | Brier: market           | Brier: constant   | Log-loss: market        | Log-loss: constant   | Log-loss gain vs constant   | Recalibration slope b   |
|:-------------|:--------|:--------------------------------|:------------------------|:------------------|:------------------------|:---------------------|:----------------------------|:------------------------|
| 2006-2025    | 5281    | 0.558 (in-sample home win rate) | 0.2111 [0.2069, 0.2157] | 0.2466            | 0.6095 [0.5999, 0.6195] | 0.6863               | 0.0768 [0.0672, 0.0859]     | 1.035 [0.956, 1.113]    |
| 2026 holdout | 93      | 0.558 (2006-2025 home win rate) | 0.2328 [0.2197, 0.2462] | 0.2515            | 0.6601 [0.6317, 0.6892] | 0.6962               | 0.0361 [-0.0014, 0.0697]    | 0.796 [0.304, 1.288]    |

The de-vigged closing moneyline is a well-calibrated forecast:

- It beats the constant (the in-sample home win rate) by 0.0768 [0.0672, 0.0859] in log-loss.
- Regressing outcomes on its log-odds gives slope b = 1.035 [0.956, 1.113]. A slope of 1 means
  calibrated; a favourite-longshot bias would push b above 1.
- In the reliability table below, every bin with more than 7 games has an observed win rate
  within 3.6 points of its price. One bin of ten ([0.6, 0.7), 1,093 games) has an interval
  that excludes its price; with ten bins, chance gives about that many misses.

| p(home) bin   | Games   | Mean p(home)   | Home win rate      | Observed - priced (pct pts)   |
|:--------------|:--------|:---------------|:-------------------|:------------------------------|
| [0.0, 0.1)    | 7       | 0.093          | 0.0% [0.0, 35.4]   | -9.3                          |
| [0.1, 0.2)    | 107     | 0.163          | 19.6% [13.2, 28.1] | +3.4                          |
| [0.2, 0.3)    | 380     | 0.256          | 25.5% [21.4, 30.1] | -0.1                          |
| [0.3, 0.4)    | 661     | 0.353          | 34.5% [31.0, 38.2] | -0.8                          |
| [0.4, 0.5)    | 716     | 0.446          | 43.7% [40.1, 47.4] | -0.9                          |
| [0.5, 0.6)    | 867     | 0.555          | 52.8% [49.5, 56.1] | -2.6                          |
| [0.6, 0.7)    | 1093    | 0.648          | 61.8% [58.8, 64.6] | -3.1                          |
| [0.7, 0.8)    | 957     | 0.748          | 76.1% [73.3, 78.7] | +1.2                          |
| [0.8, 0.9)    | 451     | 0.843          | 86.3% [82.8, 89.1] | +2.0                          |
| [0.9, 1.0)    | 42      | 0.917          | 95.2% [84.2, 98.7] | +3.5                          |

The next table gives the flat ROI of backing every side, at its listed moneyline, bucketed by
that side's de-vigged win probability. The holdout rows are in
[results/04_moneyline_roi_by_bucket.csv](results/04_moneyline_roi_by_bucket.csv).

| Window       | p(side)                  | Bets   | Mean p   | Win rate           | Flat ROI              | ROI if de-vigged price exact   |
|:-------------|:-------------------------|:-------|:---------|:-------------------|:----------------------|:-------------------------------|
| 2006-2025    | [0.0, 0.2)               | 607    | 0.152    | 14.0% [11.5, 17.0] | -14.4% [-31.6, 2.3]   | -2.6%                          |
| 2006-2025    | [0.2, 0.3)               | 1337   | 0.253    | 24.4% [22.2, 26.8] | -6.3% [-15.1, 2.7]    | -2.7%                          |
| 2006-2025    | [0.3, 0.4)               | 1754   | 0.352    | 36.8% [34.6, 39.1] | 1.7% [-3.8, 7.6]      | -2.6%                          |
| 2006-2025    | [0.4, 0.5)               | 1556   | 0.445    | 45.4% [43.0, 47.9] | -0.4% [-6.1, 5.0]     | -2.7%                          |
| 2006-2025    | [0.5, 0.6)               | 1610   | 0.553    | 54.4% [52.0, 56.8] | -4.3% [-8.8, -0.3]    | -2.7%                          |
| 2006-2025    | [0.6, 0.7)               | 1754   | 0.648    | 63.2% [60.9, 65.4] | -5.1% [-8.3, -2.0]    | -2.6%                          |
| 2006-2025    | [0.7, 0.8)               | 1337   | 0.747    | 75.6% [73.2, 77.8] | -1.5% [-4.5, 1.5]     | -2.7%                          |
| 2006-2025    | [0.8, 1.0)               | 607    | 0.848    | 86.0% [83.0, 88.5] | -1.3% [-4.7, 1.7]     | -2.6%                          |
| 2006-2025    | all favourites (p > 0.5) | 5254   | 0.669    | 66.4% [65.1, 67.7] | -3.5% [-5.5, -1.5]    | -2.6%                          |
| 2006-2025    | all underdogs (p < 0.5)  | 5254   | 0.331    | 33.6% [32.3, 34.9] | -2.8% [-6.8, 1.2]     | -2.6%                          |
| 2026 holdout | all favourites (p > 0.5) | 93     | 0.648    | 62.4% [52.2, 71.5] | -7.2% [-17.0, 4.8]    | -4.1%                          |
| 2026 holdout | all underdogs (p < 0.5)  | 93     | 0.352    | 37.6% [28.5, 47.8] | 7.2% [-10.1, 23.5]    | -4.1%                          |

The vig alone costs 2.6-2.7% in every bucket (the last column). Against that baseline:

- The heaviest longshots lost most: -14.4% [-31.6, +2.3] below 0.2, and -6.3% [-15.1, +2.7]
  at 0.2-0.3.
- Heavy favourites lost least: -1.5% and -1.3% above 0.7.

That is the classic favourite-longshot direction at the extremes. The middle runs the other
way, though: small favourites (0.5-0.7) lost 4.3% and 5.1%, the only bucket intervals that
exclude zero, while small underdogs (0.3-0.5) were near break-even. (The [0.5, 0.6) bucket
also holds both sides of the 27 games priced exactly even.) No bucket's interval lies above
zero, so nothing here pays.

**Price source.** The character of nflverse's prices changed between 2022 and 2023. The mean
moneyline overround is 1.68-2.90% in every season from 2006 to 2022 and 4.27-4.28% from 2023
on. Spread and total prices jump from under 3% to about 4.7% at the same point
([results/04_price_overround_by_season.csv](results/04_price_overround_by_season.csv)).
Smaller steps come earlier: the moneyline overround goes from 1.79% in 2008 to 2.39% in 2009,
and the spread overround from 1.96% in 2012 to 2.45% in 2013. Flat ROI at listed prices
therefore mixes price regimes. The 2026 holdout is priced like 2023-2025, which is why its
vig-only ROI is -4.1%.

**Data notes.** One game (2020_01_LV_CAR) has a negative overround, which means a pricing
error in the source; it is kept. In 141 games both moneylines are negative (pick'em pricing).

## 5. Totals

The table compares the final total points with the closing total.

| Window       | Games   | Pushes   | Over rate          | Mean total - line (pts)   |
|:-------------|:--------|:---------|:-------------------|:--------------------------|
| 1999-2006    | 2112    | 40       | 48.7% [46.6, 50.9] | +0.85 [+0.27, +1.44]      |
| 2007-2013    | 1869    | 35       | 51.6% [49.3, 53.9] | +1.24 [+0.63, +1.85]      |
| 2014-2019    | 1602    | 13       | 48.5% [46.1, 51.0] | -0.07 [-0.73, +0.58]      |
| 2020-2025    | 1693    | 18       | 48.8% [46.4, 51.2] | +0.65 [+0.03, +1.27]      |
| 1999-2025    | 7276    | 106      | 49.4% [48.3, 50.6] | +0.70 [+0.39, +1.01]      |
| 2026 holdout | 94      | 0        | 51.1% [41.1, 60.9] | +1.16 [-1.50, +3.82]      |

On the over/under question the closing total is fair: overs win 49.4% [48.3, 50.6] of 7,170
decided games. Final totals still average 0.70 [0.39, 1.01] points above the line. Positive
on average but under 50% overs means the misses are skewed: when a game goes over, it goes
over by more than it falls short when it goes under. That does not make overs a bet. 49.4% is
below the 52.38% break-even at -110, and no era's over rate is outside 48.5-51.6%.

## 6. Wind and the closing total

The sample is outdoor or open-roof games with a closing total and a wind value: 5,205 games in
1999-2025. Wind is missing for 46% of outdoor games in 2022, 21% in 2023 and 34% of the 2026
games so far. In 2026 the gap is the latest weeks: weeks 1-4 have a wind value for 44 of 45
outdoor games, weeks 5-7 for none of 22
([results/06_wind_coverage_holdout_by_week.csv](results/06_wind_coverage_holdout_by_week.csv)).
The nflverse `wind` field is in mph and records the conditions at the game, not a forecast.

| Window       | Wind (mph)   | Games   | Pushes   | Under rate         | Mean total - line (pts)   | Flat ROI on unders at -110   | At listed price   |
|:-------------|:-------------|:--------|:---------|:-------------------|:--------------------------|:-----------------------------|:------------------|
| 1999-2006    | >= 15        | 243     | 4        | 57.3% [51.0, 63.4] | -1.35 [-3.11, +0.41]      | 9.4%                         | 9.6%              |
| 2007-2013    | >= 15        | 191     | 2        | 58.2% [51.1, 65.0] | -1.25 [-3.29, +0.79]      | 11.1%                        | 13.9%             |
| 2014-2019    | >= 15        | 129     | 2        | 55.9% [47.2, 64.2] | -1.79 [-3.94, +0.36]      | 6.7%                         | 9.0%              |
| 2020-2025    | >= 15        | 121     | 2        | 55.5% [46.5, 64.1] | -1.60 [-3.94, +0.75]      | 5.9%                         | 6.7%              |
| 1999-2025    | >= 15        | 684     | 10       | 57.0% [53.2, 60.7] | -1.45 [-2.47, -0.43]      | 8.8%                         | 10.2%             |
| 1999-2025    | all          | 5205    | 84       | 51.0% [49.7, 52.4] | +0.55 [+0.18, +0.92]      | -2.6%                        | -1.2%             |
| 2026 holdout | >= 15        | 0       | 0        | n/a                | n/a                       | n/a                          | n/a               |

The listed price is nflverse's `under_odds`, available from 2006; -110 is used where it is
missing. Before 2023 those prices carried a smaller margin than -110 (section 4).

| Wind (mph)   | Games   | Mean closing total   | Mean points   | Mean total - line (pts)   | Under rate         |
|:-------------|:--------|:---------------------|:--------------|:--------------------------|:-------------------|
| 0-5          | 1579    | 43.1                 | 44.6          | +1.53 [+0.85, +2.22]      | 48.0% [45.5, 50.5] |
| 6-10         | 2161    | 42.9                 | 43.5          | +0.68 [+0.13, +1.24]      | 50.1% [48.0, 52.3] |
| 11-14        | 781     | 42.6                 | 42.5          | -0.06 [-1.00, +0.88]      | 54.4% [50.9, 57.9] |
| 15-19        | 497     | 42.1                 | 40.5          | -1.59 [-2.78, -0.39]      | 57.3% [52.9, 61.7] |
| 20+          | 187     | 40.9                 | 39.9          | -1.07 [-3.04, +0.89]      | 56.0% [48.8, 63.0] |

| Window       | Games                 | Slope of (total - closing total) on wind, pts per mph |
|:-------------|:----------------------|:------------------------|
| 1999-2006    | 1654 (all with wind)  | -0.154 [-0.275, -0.032] |
| 2007-2013    | 1356 (all with wind)  | -0.175 [-0.294, -0.057] |
| 2014-2019    | 1192 (all with wind)  | -0.237 [-0.387, -0.087] |
| 2020-2025    | 1003 (all with wind)  | -0.203 [-0.380, -0.026] |
| 1999-2025    | 5205 (all with wind)  | -0.181 [-0.249, -0.114] |
| 1999-2025    | 5202 (wind <= 40 mph) | -0.188 [-0.260, -0.116] |
| 1999-2025    | 5121 (pushes dropped) | -0.185 [-0.253, -0.116] |
| 2026 holdout | 44 (all with wind)    | -0.048 [-2.491, +2.396] |

The closing total does fall with wind, from 43.1 points at 0-5 mph to 40.9 at 20 mph or more.
Final totals fall faster, from 44.6 to 39.9. The residual, total minus closing total, falls by
0.181 [0.114, 0.249] points per mph of game-time wind. At 15 mph or more, 57.0% [53.2, 60.7] of
674 decided games went under, against 51.0% [49.7, 52.4] for all outdoor games with wind data.
Flat ROI on those unders was +8.8% at -110.

How the three wind tests come out:

| Test | Hypothesis | p | Holm p | Week-clustered p (Holm p) | Survives Holm? |
|---|---|---|---|---|---|
| W1 | unders above 50% | 1.7e-04 | 0.001 | 1.7e-04 (0.001) | yes |
| W3 | slope not zero | 1.9e-07 | 1.5e-06 | already week-clustered | yes |
| W2 | unders above the 52.38% break-even at -110 | 0.009 | 0.056 | 0.009 (0.054) | no |

Windy games often share a week, so the week-clustered version is the one to trust if they
differ; here they agree.

### Caveats

- **Game-time wind, not the forecast at the close.** A bettor has only a forecast, and the
  closing total already reflects it. Observed wind is information the close could not have
  had. A forecast is a noisy version of the observation, so a forecast-based rule would most
  likely show a smaller effect.
- **The threshold was chosen after the fact.** 15 mph is a common rule of thumb, but the
  survey that flagged the effect picked it after seeing these data. At 20 mph or more the
  under rate is 56.0% [48.8, 63.0] (184 decided games). Holm corrects for the eight declared
  tests, not for the thresholds, roof definitions and eras that were examined before them.
- **Is the effect decaying?**
  - The point estimate of the 15 mph under rate fell from 57.3% (1999-2006) and 58.2%
    (2007-2013) to 55.9% (2014-2019) and 55.5% (2020-2025).
  - ROI at -110 fell from 9.4-11.1% to 5.9-6.7%.
  - The era intervals all overlap, and the per-mph slope has not shrunk (-0.154, -0.175,
    -0.237, -0.203).
  - These data cannot tell a decaying effect from a stable one.
- **Data quality.** Three games list implausible winds: 2023_13_SF_PHI (44 mph),
  2008_02_TEN_CIN (70 mph) and 2016_13_NYG_PIT (71 mph). They are kept; dropping winds above
  40 mph moves the slope to -0.188 [-0.260, -0.116].
- **2026.** None of the 44 outdoor games with wind data has reached 15 mph yet (n = 0), and
  weeks 5-7 have no wind values yet, so there is no out-of-sample check. The 2026 slope rests
  on four weeks of data.

### What a prospective test would need

1. The wind forecast available before the bet, archived with its timestamp.
2. The total and under price available at that moment, plus the closing total, so that
   closing-line value can be measured.
3. A rule fixed in advance: forecast of 15 mph or more, outdoor or open roof, flat stakes.
4. A sample size set in advance.

Against the -110 break-even, detecting a true under rate of 57.0% (the 1999-2025 estimate)
with 80% power at one-sided alpha 0.05 takes 727 bets. At the 2020-2025 rate of 55.5% it takes
1,619. About 20 games a season qualify (19.8 in 2020-2025, an undercount because wind is
missing in 2022-2023), so the test would take 37 or 82 seasons. Closing-line value would give
an answer sooner, but it needs timestamped line history.

## 7. Wong teaser legs

A six-point teaser moves each leg's line six points toward the bettor, and a two-team teaser
pays only if both legs win. Wong legs are the ones the teaser carries across both 3 and 7:

- favourites of -7.5 to -8.5, teased to -1.5 to -2.5;
- underdogs of +1.5 to +2.5, teased to +7.5 to +8.5.

Legs are graded on the closing spread. A teased line equal to the final margin is a push and
is excluded.

The two-team break-even per leg is 72.37% at -110, 73.85% at -120 and 75.18% at -130.

| Window       | Legs            | Decided   | Pushes   | Win rate           | 2-team ROI at -110   | 2-team ROI at -120   | 2-team ROI at -130   |
|:-------------|:----------------|:----------|:---------|:-------------------|:---------------------|:---------------------|:---------------------|
| 1999-2006    | Wong (all)      | 387       | 1        | 72.4% [67.7, 76.6] | -0.1%                | -4.0%                | -7.4%                |
| 2007-2013    | Wong (all)      | 203       | 2        | 75.4% [69.0, 80.8] | 8.4%                 | 4.1%                 | 0.5%                 |
| 2014-2019    | Wong (all)      | 354       | 3        | 75.4% [70.7, 79.6] | 8.6%                 | 4.3%                 | 0.6%                 |
| 2020-2025    | Wong (all)      | 447       | 0        | 75.8% [71.7, 79.6] | 9.8%                 | 5.4%                 | 1.8%                 |
| 1999-2025    | Wong (all)      | 1391      | 6        | 74.7% [72.3, 76.9] | 6.5%                 | 2.3%                 | -1.3%                |
| 1999-2025    | Wong favourites | 491       | 2        | 73.1% [69.0, 76.8] | 2.1%                 | -2.0%                | -5.4%                |
| 1999-2025    | Wong underdogs  | 900       | 4        | 75.6% [72.6, 78.3] | 9.0%                 | 4.7%                 | 1.0%                 |
| 1999-2025    | all other legs  | 12905     | 250      | 68.5% [67.7, 69.3] | -10.3%               | -13.9%               | -16.9%               |
| 2026 holdout | Wong (all)      | 34        | 0        | 73.5% [56.9, 85.4] | 3.2%                 | -0.9%                | -4.3%                |
| 2026 holdout | Wong favourites | 10        | 0        | 60.0% [31.3, 83.2] | -31.3%               | -34.0%               | -36.3%               |
| 2026 holdout | Wong underdogs  | 24        | 0        | 79.2% [59.5, 90.8] | 19.6%                | 14.9%                | 10.9%                |
| 2026 holdout | all other legs  | 151       | 3        | 72.2% [64.6, 78.7] | -0.5%                | -4.5%                | -7.8%                |

The ROI columns are win_rate^2 x decimal price - 1, which assumes the two legs are
independent and share one win rate.

Wong legs won 74.7% [72.3, 76.9] of 1,391 decided legs, against 68.5% [67.7, 69.3] for every
other six-point leg, so the key-number structure is real. Whether it pays depends on the price:

| Test | Price | Break-even per leg | p | Holm p | Week-clustered p (Holm p) |
|---|---|---|---|---|---|
| T1 | -110 | 72.37% | 0.028 | 0.138 | 0.035 (0.173) |
| T2 | -120 | 73.85% | 0.248 | 0.994 | 0.255 (1.000) |
| T3 | -130 | 75.18% | 0.676 | 1.000 | 0.649 (1.000) |

None of the three survives the Holm correction, with or without week clustering. Underdog
legs (75.6%, 900 legs) did better than favourite legs (73.1%, 491). Every era since 2007 sits
at 75.4-75.8% (72.4% in 1999-2006), so there is no sign of decay; the binding constraint is
the price.

### What a prospective test would need

A prospective test needs the teaser price actually offered and the lines available when the
bet is placed, not the close, with the leg rules and the price fixed in advance. Whether any
book still offers two-team six-point teasers at -110 was not checked. The sample sizes below
assume the 2020-2025 rate of 74.5 Wong legs a season:

| Price | True leg rate assumed | Legs needed | Seasons |
|---|---|---|---|
| -110 | 74.7% (1999-2025) | 2,255 | 30 |
| -110 | 75.8% (2020-2025) | 1,001 | 13 |
| -120 | 74.7% (1999-2025) | 16,816 | 226 |
| -120 | 75.8% (2020-2025) | 2,980 | 40 |

## 8. Playoffs vs the regular season

| Window       | REG games   | REG MAE              | POST games   | POST MAE            | POST - REG           | Permutation p   |
|:-------------|:------------|:---------------------|:-------------|:--------------------|:---------------------|:----------------|
| 1999-2025    | 6967        | 10.26 [10.07, 10.46] | 309          | 10.42 [9.45, 11.39] | +0.16 [-0.83, +1.15] | 0.744           |
| 2025         | 272         | 9.72 [8.83, 10.62]   | 13           | 8.58 [3.59, 13.56]  | -1.15 [-6.19, +3.90] | 0.599           |
| 2026 holdout | 94          | 9.24 [7.73, 10.74]   | 0            | n/a                 | n/a                  | n/a             |

The closing line is as accurate in the playoffs as in the regular season:

- Over 1999-2025, playoff MAE minus regular-season MAE is +0.16 [-0.83, +1.15] points. Test P1
  shuffles the playoff label within each season: p = 0.744.
- In 2025 the line's playoff MAE was 8.58 against 9.72 in the regular season (P2: p = 0.599).
  A random 13-game subset of the 2025 regular season does at least that well 29.7% of the
  time.

So the 2025 lesson "Playoffs behave differently - 8.43 playoff MAE vs 11.14 regular season"
([CLAUDE.md](../../../CLAUDE.md) lesson 23, as of 2026-10-24), and the playoff-feature
rationale built on it in [ADR-017](../../adr/017-2026-season-roadmap.md), describe 13 games
that the market also forecast unusually well. They are not a property of the model. No 2026 playoff games have
been played. Results by season and phase are in
[results/08_mae_by_season_phase.csv](results/08_mae_by_season_phase.csv).

## Tests and the Holm correction

The test family is declared in `test_family()` in the script. Adjusted p-values follow
Holm's step-down procedure ([results/tests_holm.csv](results/tests_holm.csv)).

| ID   | Hypothesis                                                     | Window    | n    | Estimate   | Null   | p       | Holm p   | Reject (Holm, 5%)   |
|:-----|:---------------------------------------------------------------|:----------|:-----|:-----------|:-------|:--------|:---------|:--------------------|
| W1   | under rate at wind >= 15 mph > 0.5000                          | 1999-2025 | 674  | 0.5697     | 0.5000 | 1.7e-04 | 0.001    | yes                 |
| W2   | under rate at wind >= 15 mph > 0.5238                          | 1999-2025 | 674  | 0.5697     | 0.5238 | 0.009   | 0.056    | no                  |
| W3   | slope of (total - closing total) on wind != 0 (points per mph) | 1999-2025 | 5205 | -0.1815    | 0.0000 | 1.9e-07 | 1.5e-06  | yes                 |
| T1   | Wong leg win rate > two-team break-even at -110 (0.7237)       | 1999-2025 | 1391 | 0.7469     | 0.7237 | 0.028   | 0.138    | no                  |
| T2   | Wong leg win rate > two-team break-even at -120 (0.7385)       | 1999-2025 | 1391 | 0.7469     | 0.7385 | 0.248   | 0.994    | no                  |
| T3   | Wong leg win rate > two-team break-even at -130 (0.7518)       | 1999-2025 | 1391 | 0.7469     | 0.7518 | 0.676   | 1.000    | no                  |
| P1   | closing-line MAE differs between playoffs and regular season   | 1999-2025 | 7276 | 0.1588     | 0.0000 | 0.744   | 1.000    | no                  |
| P2   | closing-line MAE differs between playoffs and regular season   | 2025      | 285  | -1.1455    | 0.0000 | 0.599   | 1.000    | no                  |

Method for each test:

- W1, W2, T1-T3: exact binomial, one-sided.
- W3: OLS with week-clustered standard errors, two-sided.
- P1: permutation within seasons, two-sided.
- P2: permutation within 2025, two-sided.

The binomial tests assume independent games. As a sensitivity check, W1, W2 and T1-T3 are
repeated as one-sided Wald z tests with week-clustered standard errors; W3 is already
week-clustered and P1/P2 are permutation tests, so they keep their p-values. Holm is applied to
the eight again (`p_sensitivity` and `p_holm_sensitivity` in the CSV), and no decision changes:

| ID   | Sensitivity test                      | p       | Holm p   | Reject (Holm, 5%)   |
|:-----|:--------------------------------------|:--------|:---------|:--------------------|
| W1   | Wald z, week-clustered SE, one-sided  | 1.7e-04 | 0.001    | yes                 |
| W2   | Wald z, week-clustered SE, one-sided  | 0.009   | 0.054    | no                  |
| W3   | same test (SE already week-clustered) | 1.9e-07 | 1.5e-06  | yes                 |
| T1   | Wald z, week-clustered SE, one-sided  | 0.035   | 0.173    | no                  |
| T2   | Wald z, week-clustered SE, one-sided  | 0.255   | 1.000    | no                  |
| T3   | Wald z, week-clustered SE, one-sided  | 0.649   | 1.000    | no                  |
| P1   | same test (permutation)               | 0.744   | 1.000    | no                  |
| P2   | same test (permutation)               | 0.599   | 1.000    | no                  |

The sample sizes for prospective tests are in
[results/prospective_power.csv](results/prospective_power.csv): one-sided alpha 0.05, power
0.80, normal approximation.

## Survey numbers checked

The 2026-10-24 survey computed these facts on a byte-identical copy of the file (same sha256).
Every survey number checked here reproduces. Where its figures differ from this page, its
"1999-2025" or "2015+" windows included the 2026 games played so far.

| Survey figure | Survey value | This page | Verdict |
|---|---|---|---|
| Pooled home cover, non-neutral | 49.02%, n=7,072 | 49.0% [47.9, 50.2], n=6,988 (1999-2025) | Reproduces. The survey's n includes 84 decided 2026 games (6,988 + 84; 3,427 + 40 covers = 3,467 / 7,072) |
| Key margins, 1999-2014 | 3: 15.3, 7: 9.3, 6: 5.3, 10: 6.1, 1: 3.9, 2: 3.7, 8: 3.3 (%) | same; 10 is 6.05%, shown here as 6.0 | Reproduces |
| Key margins, 2015 on | 14.9, 8.8, 7.0, 4.7, 4.4, 4.6, 4.3 (2015-2026 so far) | 14.8, 8.7, 6.9, 4.8, 4.5, 4.6, 4.3 (2015-2025) | Consistent; the survey's window adds 94 games of 2026 |
| Home margin / spread, 1999-2009, 2010-2019, 2020, 2021-2025 | 2.71/2.60, 2.34/2.37, 0.17/1.17, 2.25/1.74 | 2020 and 2021-2025 identical (table 3); the decades are the game-weighted means of results/03_home_field_by_season.csv | Reproduces |
| Closing-line MAE, REG / POST | 10.25 (n=7,061) / 10.42 (n=309) | 10.26 (n=6,967) / 10.42 (n=309) | Reproduces; the survey's REG n includes the 94 games of 2026 |
| 2025 REG / POST MAE; random-subset probability | 9.72 / 8.58; 0.297 | 9.72 / 8.58; 0.297 | Reproduces |
| De-vigged moneyline | n=5,374 (2006-2026 so far): Brier 0.2115 vs 0.2467, log-loss 0.610, favourites -3.6%, underdogs -2.7% | n=5,281 (2006-2025): 0.2111 vs 0.2466, 0.6095, -3.5%, -2.8% | Consistent; the survey adds 93 games of 2026. Its "underdogs below 25%: -9.2%" bucket is not used here |
| Totals | over 49.5% (n=7,264), mean +0.71 | 49.4% of 7,170 decided, mean +0.70 (1999-2025) | Consistent; (3,545 + 48) / 7,264 = 49.46% with 2026 |
| Unders at 15 mph or more | 384/674 = 57.0%, p = 0.0002, +8.8% at -110; eras 57.3/58.2/55.9/55.5% | same; one-sided exact p = 1.7e-04 | Reproduces |
| Wind slope | -0.185 (se 0.035), n=5,121 | -0.181 [-0.249, -0.114], n=5,205 | The survey dropped the 84 pushes; without them this page also gives -0.185 [-0.253, -0.116], n=5,121 (table 6c) |
| Wong legs | 74.7% (n=1,391), other legs 68.5%, eras 72.4/75.4/75.4/75.8%, favourites 73.1% (491), underdogs 75.6% (900) | same | Reproduces |

Not re-examined here: the survey's 32 °F and below result (46.9% unders) and the injury
regression, which needs the database.

## Caveats that apply throughout

- **Closing lines only.** There is no line movement, no bet-time price and no limit on stake
  size. The file does not document nflverse's line and price sources, and the prices changed
  character in 2023 (section 4).
- **Post hoc choices.** The eras, the threshold and the buckets were set knowing an earlier
  survey's results. Holm controls the eight declared tests only.
- **A small holdout.** The 2026 holdout is 94 games over seven weeks. It can catch a gross
  failure but cannot confirm a small effect, and several of its cells are empty (no wind of
  15 mph or more, no playoffs).
- **Weather.** Wind is the observation at game time, and coverage is incomplete in 2022, 2023
  and 2026 (no 2026 values after week 4 yet).

## Files

All files are in [results/](results/):

- [summary.md](results/summary.md): the full printed report.
- [provenance.csv](results/provenance.csv): input file hash, size and row counts.
- `01_spread_cover_rate.csv`, `01_spread_error.csv`, `01_spread_error_by_season.csv`,
  `01_neutral_sites.csv`
- `02_key_numbers.csv`, `02_margin_distribution.csv`
- `03_home_field.csv`, `03_home_field_by_season.csv`
- `04_moneyline_skill.csv`, `04_moneyline_reliability.csv`, `04_moneyline_roi_by_bucket.csv`,
  `04_price_overround_by_season.csv`
- `05_totals.csv`
- `06_wind_threshold.csv`, `06_wind_bins.csv`, `06_wind_regression.csv`,
  `06_wind_coverage_by_season.csv`, `06_wind_coverage_holdout_by_week.csv`,
  `06_wind_above_40mph.csv`
- `07_teaser_legs.csv`
- `08_playoff_vs_regular.csv`, `08_mae_by_season_phase.csv`
- `tests_holm.csv`, `prospective_power.csv`
