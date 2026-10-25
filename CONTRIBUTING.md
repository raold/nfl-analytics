# Contributing

The project is wound down (see [PROJECT_STATUS.md](PROJECT_STATUS.md)); this repository is kept
as a research archive. Changes that fix an error, improve the record or extend the grading tool
are welcome. Read [CLAUDE.md](CLAUDE.md#critical-rules) first: the spread sign convention, the
kickoff time zone and the database rules there apply to every change.

## Before you open a pull request

- `uv sync --extra dev`, then `uv run pytest -m "not integration"`,
  `uv run ruff check --select E9,F63,F7,F82 nfl_analytics tests scripts` and
  `uv run black --check nfl_analytics tests`. CI runs exactly these, plus the temporal-safety
  tests, the `spread-sign-convention` hook, a secret scan and an offline check of the links in
  every Markdown file.
- `uvx pre-commit@4.7.0 install` runs the same checks locally on every commit. Do not use
  `--no-verify`.
- Grade covers with `nfl_analytics/utils/spread_convention.py`, never with `margin + spread`. A
  line that reproduces the old rule on purpose needs a `# sign-ok: <reason>` marker.
- Add tests for new code; pin conventions with real games.

## Results and claims

- Every number in a doc, table or figure must come from a committed script that reads committed
  data, or from a query whose text is in the doc. No typed-in or simulated numbers presented as
  results; label simulations and projections as such.
- Grade any model against the closing line with `nfl_analytics.evaluation.market_benchmark`, and
  report intervals. Say which seasons a model was trained, tuned and early-stopped on.
- Put an investigation under `docs/investigations/<yyyy-mm>-<topic>/` with its script and output,
  and link it from [docs/README.md](docs/README.md).

## Data and large files

- Git LFS (`.gitattributes`) tracks `backups/*.dump`, `models/**/*.rds` and `models/**/*.pkl`;
  at present it holds only the two 2025-12-07 database dumps in `backups/` (their `odds_history`
  table is empty). Other binaries, `analysis/dissertation/main/main.pdf` among them, are regular
  git blobs; keep them small.
- Do not commit generated outputs, database dumps or row-level odds. `public.odds_history` is
  licensed from The Odds API and must not be redistributed: private exports go to
  `backups/local/` (gitignored).
- Prefer small CSV or Markdown summaries over binary artifacts.

## Secrets

Never commit credentials. Configuration comes from environment variables (`.env.example` lists
them; `.env` is gitignored); document variable names, not values. The development database
password is public and must not be reused anywhere.

## Commits

Plain commit messages that say what changed and why; no co-author or AI attribution lines. The
main branch is `main`.
