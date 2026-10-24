#!/usr/bin/env python3
"""
Utilities for generating and storing GameOutcomeModel predictions.

Provides a reusable function that both CLI scripts and the FastAPI service can
call to kick off a full-season (or single-week) prediction run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

from nfl_analytics.models.game_outcome_model_fixed import GameOutcomeModel
from nfl_analytics.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DB_CONFIG = settings.database.psycopg_kwargs()
TABLE_NAME = settings.game_outcome_table
DEFAULT_MODEL_DIR = "models/production/game_outcome_v2"


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_games(target_season: int) -> pd.DataFrame:
    query = """
    SELECT DISTINCT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.kickoff,
        g.home_score,
        g.away_score,
        g.spread_close as vegas_spread,
        g.total_close as vegas_total
    FROM public.games g
    WHERE g.season = %s
        AND g.week <= 22
    ORDER BY g.week, g.kickoff
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn, params=(target_season,))
        return df


def _store_predictions(predictions_df: pd.DataFrame, prediction_run_id: uuid4) -> int:
    """Store predictions to database with proper connection management."""
    records = []
    for _, row in predictions_df.iterrows():
        records.append(
            {
                "prediction_run_id": str(prediction_run_id),
                "game_id": str(row["game_id"]),
                "season": _safe_int(row["season"]),
                "week": _safe_int(row["week"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "kickoff": row["kickoff"],
                "pred_home_spread": _safe_float(row["pred_home_spread"]),
                "pred_total": _safe_float(row["pred_total"]),
                "pred_home_win_prob": _safe_float(row["pred_home_win_prob"]),
                "spread_std": _safe_float(row["spread_std"]),
                "total_std": _safe_float(row["total_std"]),
                "vegas_spread": _safe_float(row["vegas_spread"]),
                "vegas_total": _safe_float(row["vegas_total"]),
                "actual_home_score": _safe_int(row.get("actual_home_score")),
                "actual_away_score": _safe_int(row.get("actual_away_score")),
                "actual_spread": _safe_float(row.get("actual_spread")),
                "actual_total": _safe_float(row.get("actual_total")),
                "model_version": row.get("model_version"),
                "inference_time_ms": _safe_float(row.get("inference_time_ms")),
            }
        )

    if not records:
        return 0

    insert_query = f"""
    INSERT INTO {TABLE_NAME} (
        prediction_run_id,
        game_id, season, week, home_team, away_team, kickoff,
        pred_home_spread, pred_total, pred_home_win_prob,
        spread_std, total_std,
        vegas_spread, vegas_total,
        actual_home_score, actual_away_score, actual_spread, actual_total,
        model_version, inference_time_ms
    ) VALUES (
        %(prediction_run_id)s, %(game_id)s, %(season)s, %(week)s, %(home_team)s, %(away_team)s, %(kickoff)s,
        %(pred_home_spread)s, %(pred_total)s, %(pred_home_win_prob)s,
        %(spread_std)s, %(total_std)s,
        %(vegas_spread)s, %(vegas_total)s,
        %(actual_home_score)s, %(actual_away_score)s, %(actual_spread)s, %(actual_total)s,
        %(model_version)s, %(inference_time_ms)s
    )
    """

    # FIXED: Use context manager to prevent connection leaks
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            try:
                execute_batch(cur, insert_query, records, page_size=100)
                conn.commit()
                return len(records)
            except psycopg2.Error:
                conn.rollback()
                logger.exception("Failed to store predictions")
                raise


def run_game_outcome_predictions(
    target_season: int,
    selected_week: Optional[int] = None,
    model_dir: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate predictions for a season (optionally a single week)."""
    games_df = _fetch_games(target_season)
    if games_df.empty:
        raise ValueError(f"No games found for season {target_season}")

    if selected_week:
        games_df = games_df[games_df["week"] == selected_week]
        if games_df.empty:
            raise ValueError(f"No games for season {target_season} week {selected_week}")

    weeks = sorted(games_df["week"].unique())
    prediction_run_id = uuid4()
    if verbose:
        logger.info(
            "Starting GameOutcomeModel prediction run %s for season %s (weeks: %s)",
            prediction_run_id,
            target_season,
            weeks,
        )

    model = GameOutcomeModel(model_dir=model_dir or DEFAULT_MODEL_DIR)
    model.load_models()

    all_predictions: List[Dict[str, Any]] = []

    for week in weeks:
        week_games = games_df[games_df["week"] == week]
        batch_payload = [
            {
                "game_id": game["game_id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "season": int(game["season"]),
                "week": int(game["week"]),
                "kickoff": game["kickoff"],
            }
            for _, game in week_games.iterrows()
        ]

        batch_predictions = model.predict_games(batch_payload)
        predictions_by_id = {
            pred["game_id"]: pred for pred in batch_predictions if pred.get("game_id")
        }

        for _, game in week_games.iterrows():
            pred = predictions_by_id.get(game["game_id"])
            if not pred:
                logger.warning("Missing prediction for %s", game["game_id"])
                continue

            # Skip predictions that failed (have error key instead of results)
            if "error" in pred:
                logger.warning("Prediction failed for %s: %s", game["game_id"], pred["error"])
                continue

            all_predictions.append(
                {
                    "prediction_run_id": str(prediction_run_id),
                    "game_id": game["game_id"],
                    "season": int(game["season"]),
                    "week": int(game["week"]),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "kickoff": game["kickoff"],
                    "pred_home_spread": _safe_float(pred["home_spread"]),
                    "pred_total": _safe_float(pred["total_points"]),
                    "pred_home_win_prob": _safe_float(pred["home_win_probability"]),
                    "spread_std": _safe_float(pred["spread_std"]),
                    "total_std": _safe_float(pred["total_std"]),
                    "vegas_spread": _safe_float(game["vegas_spread"]),
                    "vegas_total": _safe_float(game["vegas_total"]),
                    "actual_home_score": _safe_int(game["home_score"]),
                    "actual_away_score": _safe_int(game["away_score"]),
                    "actual_spread": (
                        _safe_float(game["home_score"] - game["away_score"])
                        if pd.notna(game["home_score"])
                        else None
                    ),
                    "actual_total": (
                        _safe_float(game["home_score"] + game["away_score"])
                        if pd.notna(game["home_score"])
                        else None
                    ),
                    "model_version": model.metadata.get("version", "v1.0.0"),
                    "inference_time_ms": _safe_float(pred.get("inference_time_ms")),
                }
            )

    predictions_df = pd.DataFrame(all_predictions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path("data") / f"game_predictions_{target_season}_{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(csv_path, index=False)

    stored = 0
    if not dry_run and not predictions_df.empty:
        stored = _store_predictions(predictions_df, prediction_run_id)

    summary = {
        "prediction_run_id": str(prediction_run_id),
        "season": target_season,
        "weeks": weeks,
        "total_games": int(len(predictions_df)),
        "stored_games": stored,
        "csv_path": str(csv_path),
        "dry_run": dry_run,
    }

    if verbose:
        logger.info(
            "Completed run %s: %s games (stored=%s, csv=%s)",
            prediction_run_id,
            summary["total_games"],
            stored,
            csv_path,
        )

    return summary
