# Tests

The suite after the October 2026 prune: tests for the grading tool, the spread convention, the
leakage detector, the kept feature code and the frozen Game Outcome v2 loader. Tests of removed
code went with it (tag `archive/pre-prune-2026-10`).

## Running

```bash
uv sync --extra dev
uv run pytest -m "not integration"            # offline; what CI runs
uv run pytest -m integration                  # 133 tests; need devdb01 with data and DB_PASSWORD
```

On 2026-10-25 the offline run gave 227 passed and 1 skipped (`test_evidential_regression.py`
needs the `torch` extra). `pyproject.toml` adds coverage options to every run; pass `--no-cov`
to skip them.

`tests/{integration,infrastructure,e2e}` are marked `integration` automatically
(`tests/conftest.py`), and with `-m "not integration"` their modules are not even imported. Unit
tests get a placeholder `DB_PASSWORD`; integration tests use the real environment.

The infrastructure tests leave devdb01 alone. `test_backup_restore.py` runs
`scripts/maintenance/backup.sh` (a `pg_dump` of `DB_NAME`, default devdb01) into a temporary
directory; its restore test restores only into the database named by `POSTGRES_DB` and skips
unless that name contains "test", and the full round-trip comparison is skipped (run it by hand).
`test_migrations.py` checks the migration files and reads devdb01's schema; its tests that apply
migrations are skipped (they need a clean `testdb`).

## What is here

| Path | What it checks |
|---|---|
| `unit/test_spread_convention.py` | the nflverse sign convention on real games (BUF@BAL 2018, GB@CHI 2019, CLE@NO 2018), the pooled home cover rate, that the old rule inflates it, and that prior-season power ratings are about 50% ATS once graded correctly |
| `unit/evaluation/test_market_benchmark.py` | grading, ROI at real prices, guardrails, the information regression, and the README baseline rows through the CLI |
| `unit/evaluation/test_market_benchmark_nflverse.py` | the `--nflverse-csv` path, predict mode and closing-line value, on the real rows in `fixtures/nflverse/` |
| `unit/evaluation/test_holm_step_down.py` | the Holm step-down procedure in `nfl_analytics/analysis/statistical_tests.py` |
| `unit/test_state_space_sign.py` | the state-space ATS probability and evaluation after the sign fix (pinned games, a push) |
| `unit/test_game_outcome_import.py` | the v2 model and its prediction service import without shap or credentials; the DO NOT DEPLOY guard |
| `unit/test_data_leakage_prevention.py`, `unit/test_data_leakage_detector_comprehensive.py` | `nfl_analytics/utils/data_leakage_detector.py` |
| `unit/features/` | as-of features and their home/away orientation, the game feature extractor (mocked), line-movement features, temporal safety of team histories |
| `unit/test_evidential_regression.py` | the evidential regression model (needs torch) |
| `integration/` | database integrity, the game feature extractor on pinned games (with an as-of leak check), the SQL validator, the temporal-leakage suites |
| `infrastructure/` | backup and restore scripts, migration files and devdb01's schema (see above) |
| `sql/test_schema.sql`, `sql/test_data_quality.sql` | ad-hoc checks to run with psql |
| `fixtures/nflverse/` | real nflverse rows for the benchmark tests ([README](fixtures/nflverse/README.md)) |
| `fixtures/temporal_leakage_scenarios.py` | known-good and known-bad scenarios for the leakage tests |

## CI

`.github/workflows/ci.yml` runs the ruff subset (`E9,F63,F7,F82`) on `nfl_analytics tests
scripts`, `black --check nfl_analytics tests`, and `pytest -m "not integration"` on Python 3.11,
plus a gitleaks scan of the checked-out tree (reviewed false positives in `.gitleaksignore`).
`temporal-safety.yml` runs `test_spread_convention.py` and the two leakage-detector test files,
the guard's self-test (`scripts/dev/check_spread_sign_hook.py`: known wrong lines must be flagged,
correct ones must pass), then the `spread-sign-convention` pre-commit hook over the whole tree.
`check-links.yml` checks the relative links of every Markdown file with lychee (offline).
`make lint` runs the same lint steps locally. No workflow has a database, so nothing marked
`integration` runs in CI.

## Integration status

`tests/integration` against devdb01 on 2026-10-25, in a read-only session: 66 passed, 20 failed,
8 errors, 3 skipped. The failures are stale expectations, not data problems: the game feature
extractor returns 338 features where the tests expect 100, `odds_history` has no `market` column,
pandas rejects the `H` frequency alias, a `benchmark` fixture is missing, and `LeakageIssue` is
no longer subscriptable. In `test_data_pipeline.py` one failing query aborts the shared
transaction, which fails the rest of that class. `tests/infrastructure` (27 tests) was not run in
that pass.

## Writing tests

- Pin conventions with real games from the database or nflverse, not synthetic examples.
- Grade covers through `nfl_analytics/utils/spread_convention.py`; the sign hook rejects
  `margin + spread` (mark a deliberate reproduction of the old rule `# sign-ok: <reason>`).
- Seed every random generator a test uses; an unseeded fixture made
  `test_win_percentage_excludes_current_result` flaky after the prune.
- Keep unit tests offline: mock connections, and mark anything that needs a service
  `@pytest.mark.integration` or put it under `tests/integration/`.
