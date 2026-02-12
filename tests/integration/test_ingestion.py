import os
import subprocess
import time
from pathlib import Path

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_DSN = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5544)),
    "dbname": os.environ.get("POSTGRES_DB", "devdb01"),
    "user": os.environ.get("POSTGRES_USER", "dro"),
    "password": os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
}


def run(command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def wait_for_db(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with psycopg.connect(**POSTGRES_DSN):
                return
        except psycopg.OperationalError:
            time.sleep(2)
    raise TimeoutError("Database not available after waiting")


@pytest.fixture(scope="module", autouse=True)
def postgres_container():
    env = os.environ.copy()
    run(["docker", "compose", "up", "-d", "pg"], cwd=REPO_ROOT, env=env)
    wait_for_db()
    yield
    run(["docker", "compose", "down"], cwd=REPO_ROOT, env=env)


def apply_schema():
    ddl = (REPO_ROOT / "db" / "001_init.sql").read_text()
    with psycopg.connect(**POSTGRES_DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)


def get_game_count():
    with psycopg.connect(**POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM games;")
            return cur.fetchone()[0]


def explain_season_week():
    with psycopg.connect(**POSTGRES_DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off;")
            cur.execute(
                "EXPLAIN SELECT * FROM games WHERE season = %s AND week = %s;",
                (2024, 1),
            )
            plan = "\n".join(row[0] for row in cur.fetchall())
    return plan


def refresh_mart():
    with psycopg.connect(**POSTGRES_DSN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW mart.game_summary;")


def test_schedule_ingestion_idempotent():
    apply_schema()

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_HOST": POSTGRES_DSN["host"],
            "POSTGRES_PORT": str(POSTGRES_DSN["port"]),
            "POSTGRES_DB": POSTGRES_DSN["dbname"],
            "POSTGRES_USER": POSTGRES_DSN["user"],
            "POSTGRES_PASSWORD": POSTGRES_DSN["password"],
            "NFLVERSE_SNAPSHOT_PATH": str(
                REPO_ROOT / "data" / "raw" / "nflverse_schedules_1999_2024.rds"
            ),
        }
    )

    run(["Rscript", "--vanilla", "data/ingest_schedules.R"], cwd=REPO_ROOT, env=env)
    first_count = get_game_count()
    assert first_count > 0

    run(["Rscript", "--vanilla", "data/ingest_schedules.R"], cwd=REPO_ROOT, env=env)
    second_count = get_game_count()

    assert first_count == second_count, "Ingestion should be idempotent"

    plan = explain_season_week()
    assert "games_season_week_idx" in plan

    refresh_mart()
    with psycopg.connect(**POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM mart.game_summary;")
            rows = cur.fetchone()[0]
            assert rows >= first_count, "Game summary view should have rows"
