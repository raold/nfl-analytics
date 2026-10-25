# Prospective 2026 predictions: weeks 7 and 8

These are walk-forward Elo and power-rating predictions for the 27 regular-season games of 2026
weeks 7 and 8 that had not kicked off at 2026-10-25 01:30 UTC (21:30 EDT on Oct 24). They were
made with `market_benchmark`'s predict mode before any of these games was played, so they can be
graded as a prospective test once the games are final. They are baselines with no claimed edge:
over 2019-2025 and the 2026 games played so far, both lose to the closing line
([../README.md](../README.md)).

| File | Games | Baseline | sha256 |
|---|---|---|---|
| `preds_2026_w07_elo.csv` | 13 (week 7 except the Thursday game) | Elo | `bdfeba23d7e1ecf9c77f0d358ff8c2f87e1d40df9c53e68cf296ccd8f1de26f9` |
| `preds_2026_w07_power_ratings.csv` | 13 | power ratings | `553cca7c238e7e63463e4d0fb11083364d35155b02c742e3a776842c0a0d173d` |
| `preds_2026_w08_elo.csv` | 14 (all of week 8) | Elo | `15c82824eb1e75f4c051532251f619ded34c162a017f17e150a8254301310eff` |
| `preds_2026_w08_power_ratings.csv` | 14 | power ratings | `8d612f6a15489a517a2994ab11aead3076db2f85d5bb596f3fe0b0be3f17ba73` |

`SHA256SUMS` holds the same four digests: `sha256sum -c SHA256SUMS` in this directory checks the
files.

## How and when they were made

Input: nflverse `games.csv` from
<https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv>, downloaded
2026-10-25 01:28:03 UTC (upstream `Last-Modified: Sun, 25 Oct 2026 01:16:43 GMT`), 2,186,477
bytes, sha256 `07b41111a46458065937e402656f0636743a53513bade967789709395c16a1b2`. It is not in
the repo. nflverse replaces this file in place, so the digest (also in every row's
`input_sha256`) is what identifies the version used. That file has 272 games for 2026, 94 of
them final (weeks 1-6 and `2026_07_NE_CHI` on Thursday Oct 22), and lines for weeks 1-8.

Command, run from the repo root in a clean shell (`env -i HOME=$HOME PATH=$PATH`):

```bash
uv run --no-sync python scripts/final_evaluation/prospective_2026.py \
    --nflverse-csv games.csv --weeks 7,8 \
    --out-dir docs/investigations/2026-10-final-evaluation/prospective
```

The script runs this once per week and baseline, then writes `SHA256SUMS`:

```bash
python -m nfl_analytics.evaluation.market_benchmark predict --baseline elo \
    --nflverse-csv games.csv --season 2026 --week 7 \
    --out docs/investigations/2026-10-final-evaluation/prospective/preds_2026_w07_elo.csv
```

Run log (started 2026-10-25T01:30:45Z):

```text
13 predictions for 2026 week 7 (elo, fit on 365 completed games); skipped 1 games already started or completed: 2026_07_NE_CHI
13 predictions for 2026 week 7 (power_ratings, fit on 365 completed games); skipped 1 games already started or completed: 2026_07_NE_CHI
14 predictions for 2026 week 8 (elo, fit on 366 completed games); skipped 0 games already started or completed
14 predictions for 2026 week 8 (power_ratings, fit on 366 completed games); skipped 0 games already started or completed
```

The `prediction_timestamp` in the files runs from 2026-10-25T01:30:47Z to 01:30:51Z. The first of
these games, `2026_07_PIT_NO`, kicks off at 13:30 UTC (09:30 ET), 11 h 59 min later. The first
week 8 game, `2026_08_CAR_GB`, kicks off 2026-10-30 00:15 UTC (Thursday 20:15 ET).

## What the predictions are

- **Fit.** Each baseline is fitted walk-forward on every completed regular-season game of 2025
  (272) and of the earlier 2026 weeks: weeks 1-6 (93 games) for week 7, plus `2026_07_NE_CHI` for
  week 8. These are the same inputs the grader uses for a week whose earlier games are all
  played.
- **Constants.** Both baselines run with fixed constants from
  `nfl_analytics/evaluation/market_benchmark/baselines.py`, copied into every row
  (`baseline_params`). No code estimates them from data, and nothing was tuned on 2026. The 2026
  games enter only as results that update the ratings, which is what a walk-forward baseline does.
  - Elo: K = 20, home-field advantage 48 Elo points (1.92 points of margin), each preseason a
    third of the way back to 1505, the margin-of-victory multiplier
    ln(|margin| + 1) x 2.2 / (0.001 x winner's Elo edge + 2.2), and 25 Elo points per point of
    margin. Apart from the home-field value, this is the update FiveThirtyEight published for its
    NFL Elo. `pred_home_win_prob` is the Elo expectation 1 / (1 + 10^(-diff/400)).
  - Power ratings: a team's rating is (this season's point margins + 4 x two-thirds of last
    season's average margin) / (games played + 4). The prediction is home rating - away rating
    + 2.5.
- **Sign.** `pred_home_margin` is the expected HOME margin, and `spread_at_prediction` is nflverse
  `spread_line` in the input file (expected home margin, positive = home favored), the same sign
  as `games.spread_close`.
- **Week 8 predates week 7.** The week 8 files were made before 13 of week 7's 14 games were
  played, so they do not use those results. They are a valid prospective record, but they are not
  the values the walk-forward grader computes for week 8 once week 7 is in. A fresher week-8
  record can be made after the Monday game (2026-10-26) and before Thursday's 2026-10-29 20:15 ET
  kickoff with `--weeks 8 --suffix _after_w07` and a new download. Grade each file on its own; the
  grader refuses a file with two predictions for one game.
- **Neutral site.** `2026_07_PIT_NO` is listed at a neutral site. The baselines give the listed
  home team (NO) the full home-field advantage there, as they do in every graded season.
- **Line at prediction.** `spread_at_prediction` is the line in nflverse's file at its download,
  not a price that could be bet at that second. The grader uses it for closing-line value.

## The predictions

Edges and bet sides here are against the line at prediction. The grader also bets at
|pred - closing line| >= 1, which can select different games.

| Game | Kickoff (ET) | Line at prediction | Elo margin | Elo P(home win) | Elo bet at edge >= 1 | Power margin | Power bet at edge >= 1 |
|---|---|---|---|---|---|---|---|
| 2026_07_PIT_NO (neutral site) | Sun 10-25 09:30 | -3.0 | -2.3 | 0.417 | no bet | -2.4 | no bet |
| 2026_07_CIN_BAL | Sun 10-25 13:00 | -3.0 | +3.1 | 0.610 | BAL (+6.1) | +4.6 | BAL (+7.6) |
| 2026_07_CLE_TEN | Sun 10-25 13:00 | -2.5 | -3.1 | 0.389 | no bet | -1.9 | no bet |
| 2026_07_IND_MIN | Sun 10-25 13:00 | +6.5 | +9.0 | 0.784 | MIN (+2.5) | +8.8 | MIN (+2.3) |
| 2026_07_MIA_NYJ | Sun 10-25 13:00 | +4.5 | +2.8 | 0.600 | MIA (-1.7) | +6.5 | NYJ (+2.0) |
| 2026_07_NYG_HOU | Sun 10-25 13:00 | +10.0 | +5.8 | 0.697 | NYG (-4.2) | +5.6 | NYG (-4.4) |
| 2026_07_SF_ATL | Sun 10-25 13:00 | -2.5 | -6.2 | 0.291 | SF (-3.7) | -7.1 | SF (-4.6) |
| 2026_07_TB_CAR | Sun 10-25 13:00 | +3.0 | +5.9 | 0.701 | CAR (+2.9) | +6.2 | CAR (+3.2) |
| 2026_07_DEN_ARI | Sun 10-25 16:05 | -4.5 | -6.3 | 0.286 | DEN (-1.8) | -5.1 | no bet |
| 2026_07_GB_DET | Sun 10-25 16:25 | +3.5 | +2.0 | 0.572 | GB (-1.5) | +4.8 | DET (+1.3) |
| 2026_07_LA_LV | Sun 10-25 16:25 | -3.0 | -2.5 | 0.410 | no bet | -2.9 | no bet |
| 2026_07_KC_SEA | Sun 10-25 20:20 | +3.0 | +4.4 | 0.655 | SEA (+1.4) | +1.2 | KC (-1.8) |
| 2026_07_DAL_PHI | Mon 10-26 20:15 | -4.5 | +3.4 | 0.621 | PHI (+7.9) | +1.7 | PHI (+6.2) |
| 2026_08_CAR_GB | Thu 10-29 20:15 | +3.0 | +2.2 | 0.578 | no bet | +1.8 | CAR (-1.2) |
| 2026_08_ARI_DAL | Sun 11-01 13:00 | +6.5 | +3.2 | 0.613 | ARI (-3.3) | +8.4 | DAL (+1.9) |
| 2026_08_ATL_TB | Sun 11-01 13:00 | +1.5 | +0.0 | 0.501 | ATL (-1.5) | +2.0 | no bet |
| 2026_08_BAL_BUF | Sun 11-01 13:00 | +3.0 | +5.4 | 0.684 | BUF (+2.4) | +4.1 | BUF (+1.1) |
| 2026_08_CLE_PIT | Sun 11-01 13:00 | +4.5 | +4.9 | 0.669 | no bet | +6.7 | PIT (+2.2) |
| 2026_08_IND_JAX | Sun 11-01 13:00 | +7.5 | +9.3 | 0.793 | JAX (+1.8) | +10.8 | JAX (+3.3) |
| 2026_08_LV_NYJ | Sun 11-01 13:00 | -1.5 | +0.3 | 0.510 | NYJ (+1.8) | +1.3 | NYJ (+2.8) |
| 2026_08_MIN_DET | Sun 11-01 13:00 | +2.5 | -1.5 | 0.447 | MIN (-4.0) | +0.3 | MIN (-2.2) |
| 2026_08_TEN_CIN | Sun 11-01 13:00 | +10.0 | +8.5 | 0.773 | TEN (-1.5) | +10.8 | no bet |
| 2026_08_LAC_LA | Sun 11-01 16:05 | +7.0 | +7.3 | 0.741 | no bet | +10.6 | LA (+3.6) |
| 2026_08_KC_DEN | Sun 11-01 16:25 | -1.5 | +3.5 | 0.624 | DEN (+5.0) | -4.1 | KC (-2.6) |
| 2026_08_NE_MIA | Sun 11-01 16:25 | -8.5 | -7.2 | 0.261 | MIA (+1.3) | -9.7 | NE (-1.2) |
| 2026_08_PHI_WAS | Sun 11-01 20:20 | +3.5 | -0.9 | 0.468 | PHI (-4.4) | -0.7 | PHI (-4.2) |
| 2026_08_CHI_SEA | Mon 11-02 20:15 | +3.0 | +3.7 | 0.631 | no bet | +4.4 | SEA (+1.4) |

The table was generated from the four CSV files; the CSV files are the record.

## When the timestamp becomes verifiable

`prediction_timestamp` comes from this machine's clock, and a file on a laptop can be edited at
any time. A git commit date is also set locally and proves nothing. The record becomes verifiable
by a third party only once the files, or their digests, are published somewhere that records when
it received them, before the games kick off. Pushing a commit with this directory to GitHub would
do it, since GitHub logs when each push arrives (the repository's Activity view). Posting the
`SHA256SUMS` digests publicly would also work. Whether to publish before the next kickoff,
2026-10-25 09:30 ET, is the owner's decision. A game that kicks off before publication still has
a prediction on file, but it does not count as prospective evidence for anyone else.

## Grading after the games

Download a current `games.csv` once the games are final (week 7 ends with the Monday game on
2026-10-26, week 8 on 2026-11-02), then from the repo root:

```bash
T=$(mktemp -d)
curl -sSfL -o $T/games.csv https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
P=docs/investigations/2026-10-final-evaluation/prospective
(cd $P && sha256sum -c SHA256SUMS)
for f in $P/preds_2026_w0[78]_*.csv; do
    uv run --no-sync python -m nfl_analytics.evaluation.market_benchmark \
        --predictions-csv "$f" --nflverse-csv $T/games.csv --seasons 2026
done
```

Each report gives the ATS record and ROI at |pred - closing line| >= 1, MAE against the closing
line, log-loss against the de-vigged moneyline (Elo only), closing-line value for bets chosen at
`spread_at_prediction`, and whether every prediction preceded kickoff. Games not yet final are
left out. If none is final the grader says so and exits with status 1. With 13-14 games per file,
the line-sign check (it needs 30 games) is skipped with a warning, and every interval is very
wide. One week of a baseline says nothing about skill; the record matters only as the first
entries of a longer prospective series.

To grade both weeks of one baseline together:

```bash
{ cat $P/preds_2026_w07_elo.csv; tail -n +2 $P/preds_2026_w08_elo.csv; } > $T/elo_w07_w08.csv
uv run --no-sync python -m nfl_analytics.evaluation.market_benchmark \
    --predictions-csv $T/elo_w07_w08.csv --nflverse-csv $T/games.csv --seasons 2026
```
