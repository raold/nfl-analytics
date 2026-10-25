"""GameFeatureExtractor on pinned real games against the live database (read-only session).

Marked integration (CI deselects it). Skips cleanly when DB_PASSWORD is unset or the database
is unreachable. For each pinned game it checks that extraction raises nothing, the feature set
against the frozen Game Outcome v2 inputs, the kickoff-derived features (US/Eastern wall
clock), and that no outcome data from a game on or after the target week reaches the
extractors: every result row fetched through psycopg2 is recorded, and rows that identify a
game (game_id, or season + week) at or after the target week may only carry schedule or
market columns.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import psycopg2
import psycopg2.extensions
import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
V2_NAMES = json.loads((REPO / "models/production/game_outcome_v2/feature_names.json").read_text())
# Rest/back-to-back features GFE adds that v2 does not use
EXTRA_REST = {
    "home_days_rest",
    "away_days_rest",
    "is_home_short_week",
    "is_away_short_week",
    "is_home_long_week",
    "is_away_long_week",
}

# Real games (games table = nflverse schedules): game_id -> (home, away, season, week, expected)
PINNED = {
    # Sun 13:00 ET; KC played Thu 20:20 ET in week 1, so 9.69 days of rest
    "2023_02_KC_JAX": (
        "JAX",
        "KC",
        2023,
        2,
        {"pt_is_early_sunday": 1, "pt_is_primetime": 0, "away_days_rest": 9.69},
    ),
    # Sunday night; HOU played Thu 20:15 ET ten days earlier, across the DST change
    "2024_10_DET_HOU": (
        "HOU",
        "DET",
        2024,
        10,
        {"pt_is_sunday_night": 1, "pt_game_slot_encoded": 2, "home_days_rest": 10.0},
    ),
    # Sunday night in 2025, whose kickoffs were date-only before the 2026-10-24 repair
    "2025_10_PIT_LAC": (
        "LAC",
        "PIT",
        2025,
        10,
        {"pt_is_sunday_night": 1, "pt_is_primetime": 1, "pt_game_slot_encoded": 2},
    ),
}

# Columns that carry game results or in-game stats (as opposed to schedule or market data)
OUTCOME_COLUMN = re.compile(
    r"score|epa|yard|point|margin|result|wpa|success|cpoe|turnover|sack|qbr|touchdown|"
    r"interception|fumble|penalt|plays|win",
    re.IGNORECASE,
)


def _db_config() -> dict:
    password = os.environ.get("DB_PASSWORD")
    if not password:
        pytest.skip("DB_PASSWORD not set")
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "5544")),
        "dbname": os.environ.get("DB_NAME", "devdb01"),
        "user": os.environ.get("DB_USER", "dro"),
        "password": password,
    }


@pytest.fixture(scope="module")
def db_config():
    config = _db_config()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PGOPTIONS", "-c default_transaction_read_only=on")  # every session read-only
        try:
            psycopg2.connect(connect_timeout=5, **config).close()
        except psycopg2.OperationalError as exc:
            pytest.skip(f"database unreachable: {exc}")
        yield config


@pytest.fixture(scope="module")
def game_weeks(db_config):
    with psycopg2.connect(**db_config) as conn, conn.cursor() as cur:
        cur.execute("SELECT game_id, season, week FROM games")
        return {gid: (season, week) for gid, season, week in cur.fetchall()}


@pytest.fixture(scope="module")
def extractor(db_config):
    from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

    return GameFeatureExtractor(db_config=db_config)


class _Recorder:
    """psycopg2 connection factory that records every query and every row it returns."""

    def __init__(self):
        self.records: list[dict] = []
        recorder = self

        class RecordingCursorMixin:
            def execute(self, query, vars=None):
                self._rec = {"sql": str(query), "params": vars, "columns": [], "rows": []}
                recorder.records.append(self._rec)
                return super().execute(query, vars)

            def _keep(self, rows):
                if self.description:
                    self._rec["columns"] = [d[0] for d in self.description]
                self._rec["rows"].extend(rows)

            def fetchone(self):
                row = super().fetchone()
                if row is not None:
                    self._keep([row])
                return row

            def fetchmany(self, size=None):
                rows = super().fetchmany(size) if size is not None else super().fetchmany()
                self._keep(rows)
                return rows

            def fetchall(self):
                rows = super().fetchall()
                self._keep(rows)
                return rows

            def __iter__(self):
                while True:
                    row = self.fetchone()
                    if row is None:
                        return
                    yield row

        class RecordingConnection(psycopg2.extensions.connection):
            def cursor(self, *args, **kwargs):
                base = kwargs.pop("cursor_factory", None) or psycopg2.extensions.cursor
                factory = type(f"Recording{base.__name__}", (RecordingCursorMixin, base), {})
                return super().cursor(*args, cursor_factory=factory, **kwargs)

        self.connection_factory = RecordingConnection


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    real_connect = psycopg2.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("connection_factory", rec.connection_factory)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return rec


def _row_value(row, columns, name):
    if isinstance(row, dict):
        return row.get(name)
    return row[columns.index(name)]


def _leak_check(records, game_weeks, season, week):
    """(rows checked, violations): outcome-bearing rows from games at or after the target."""
    checked, violations = 0, []
    for rec in records:
        cols = rec["columns"]
        if not rec["rows"] or not cols:
            continue
        outcome_cols = [c for c in cols if OUTCOME_COLUMN.search(c)]
        for row in rec["rows"]:
            if "game_id" in cols:
                gid = _row_value(row, cols, "game_id")
                if gid not in game_weeks:
                    continue
                row_season, row_week = game_weeks[gid]
            elif "season" in cols and "week" in cols:
                row_season = _row_value(row, cols, "season")
                row_week = _row_value(row, cols, "week")
            else:
                continue
            if row_season is None or row_week is None:
                continue
            checked += 1
            at_or_after = row_season > season or (row_season == season and row_week >= week)
            if at_or_after and outcome_cols:
                violations.append((rec["sql"].split()[:12], row_season, row_week, outcome_cols))
    return checked, violations


@pytest.mark.parametrize("game_id", sorted(PINNED))
def test_pinned_game_features(game_id, extractor, recorder, game_weeks):
    home, away, season, week, expected = PINNED[game_id]
    assert game_weeks[game_id] == (season, week)

    features = extractor.extract_game_features(home, away, season, week, use_cache=False)

    # Feature set: every v2 input except the removed sem_* group and the syn_* group (its
    # coaching_staff query currently fails), plus the six rest/BTB extras
    names = set(features)
    assert not any(n.startswith(("sem_", "kal_", "gp_")) for n in names)
    syn = {n for n in V2_NAMES if n.startswith("syn_")}
    expected_names = {n for n in V2_NAMES if not n.startswith("sem_")} - syn
    assert expected_names <= names, sorted(expected_names - names)[:10]
    assert names - expected_names - syn == EXTRA_REST
    assert len(features) == len(expected_names) + len(EXTRA_REST) + len(names & syn)
    assert len(expected_names) == 332

    # Kickoff-derived features use the US/Eastern wall clock
    for key, value in expected.items():
        assert features[key] == pytest.approx(value), key

    # Offensive/defensive extractors report only earlier games
    assert extractor.validate_temporal_safety(home, away, season, week)

    # No outcome-bearing row from a game at or after the target week
    checked, violations = _leak_check(recorder.records, game_weeks, season, week)
    assert checked > 0, "no game-identified rows were fetched; the leakage check saw nothing"
    assert not violations, violations[:3]


def test_leak_check_flags_the_target_games_own_result(db_config, recorder, game_weeks):
    """Negative control: fetching a pinned game's final score must be reported."""
    game_id = "2024_10_DET_HOU"
    _, _, season, week, _ = PINNED[game_id]
    with psycopg2.connect(**db_config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT game_id, home_score, away_score FROM games WHERE game_id = %s", (game_id,)
        )
        cur.fetchall()
    checked, violations = _leak_check(recorder.records, game_weeks, season, week)
    assert checked == 1
    assert len(violations) == 1 and violations[0][3] == ["home_score", "away_score"]
