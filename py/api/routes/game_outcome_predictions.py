#!/usr/bin/env python3
"""
GameOutcomeModel prediction APIs.

Exposes prediction runs and stored predictions from ml.game_outcome_predictions so
the dashboard (and other clients) can consume them via the authenticated API.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import List, Optional
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime

from api.auth import require_api_key, APIKey
from py.config.settings import get_settings
from py.services.game_outcome_runs import run_game_outcome_predictions

settings = get_settings()

router = APIRouter(prefix="/game-outcomes", tags=["GameOutcomeModel"])


class PredictionRun(BaseModel):
    prediction_run_id: UUID
    total_games: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    season_start: Optional[int]
    season_end: Optional[int]
    model_version: Optional[str]
    season_start: Optional[int]
    season_end: Optional[int]


class GameOutcomePrediction(BaseModel):
    prediction_id: int
    prediction_run_id: Optional[UUID]
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff: Optional[datetime]
    pred_home_spread: Optional[float]
    pred_total: Optional[float]
    pred_home_win_prob: Optional[float]
    spread_std: Optional[float]
    total_std: Optional[float]
    vegas_spread: Optional[float]
    vegas_total: Optional[float]
    actual_home_score: Optional[int]
    actual_away_score: Optional[int]
    actual_spread: Optional[float]
    actual_total: Optional[float]
    model_version: Optional[str]
    prediction_timestamp: Optional[datetime]
    inference_time_ms: Optional[float]
    spread_edge: Optional[float]
    total_edge: Optional[float]


class PredictionRunTriggerRequest(BaseModel):
    season: Optional[int] = None
    week: Optional[int] = None
    dry_run: bool = False


class PredictionRunTriggerResponse(BaseModel):
    prediction_run_id: str
    season: int
    weeks: List[int]
    total_games: int
    stored_games: int
    csv_path: str
    dry_run: bool


def _get_connection():
    """Helper to open a psycopg connection using shared settings."""
    return psycopg.connect(settings.database.dsn, row_factory=dict_row)


@router.get("/runs", response_model=List[PredictionRun])
async def list_prediction_runs(_: APIKey = Depends(require_api_key)):
    """Return all recorded prediction vintages."""
    query = """
        SELECT
            prediction_run_id,
            total_games,
            started_at,
            completed_at,
            season_start,
            season_end,
            model_version
        FROM ml.game_outcome_prediction_runs
        ORDER BY completed_at DESC NULLS LAST
    """
    with _get_connection() as conn:
        rows = conn.execute(query).fetchall()
    return [PredictionRun(**row) for row in rows]


@router.get("/predictions")
async def list_predictions(
    season: Optional[int] = Query(None, description="Season year filter"),
    week: Optional[int] = Query(None, ge=1, le=22),
    team: Optional[str] = Query(None, min_length=2, max_length=4),
    prediction_run_id: Optional[UUID] = Query(
        None, description="Specify a run to return historical predictions"
    ),
    _: APIKey = Depends(require_api_key),
):
    """Return stored predictions filtered by season/week/team/run."""
    table_name = (
        settings.game_outcome_table
        if prediction_run_id
        else "ml.game_outcome_predictions_latest"
    )

    base_query = f"""
        SELECT
            prediction_id,
            prediction_run_id,
            game_id,
            season,
            week,
            home_team,
            away_team,
            kickoff,
            pred_home_spread,
            pred_total,
            pred_home_win_prob,
            spread_std,
            total_std,
            vegas_spread,
            vegas_total,
            actual_home_score,
            actual_away_score,
            actual_spread,
            actual_total,
            model_version,
            prediction_timestamp,
            inference_time_ms,
            (pred_home_spread - vegas_spread) AS spread_edge,
            (pred_total - vegas_total) AS total_edge
        FROM {table_name}
        WHERE 1=1
    """

    params = {}

    if prediction_run_id:
        base_query += " AND prediction_run_id = %(prediction_run_id)s"
        params["prediction_run_id"] = str(prediction_run_id)

    if season:
        base_query += " AND season = %(season)s"
        params["season"] = season

    if week:
        base_query += " AND week = %(week)s"
        params["week"] = week

    if team:
        base_query += " AND (home_team = %(team)s OR away_team = %(team)s)"
        params["team"] = team.upper()

    base_query += " ORDER BY week, kickoff NULLS LAST"

    try:
        with _get_connection() as conn:
            rows = conn.execute(base_query, params).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {exc}") from exc

    return {
        "count": len(rows),
        "predictions": [GameOutcomePrediction(**row) for row in rows],
    }


@router.post("/runs", response_model=PredictionRunTriggerResponse)
async def trigger_prediction_run(
    request: PredictionRunTriggerRequest,
    _: APIKey = Depends(require_api_key),
):
    """Trigger a new prediction run via API."""
    season = request.season or datetime.utcnow().year + 1
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(
        None,
        partial(
            run_game_outcome_predictions,
            target_season=season,
            selected_week=request.week,
            dry_run=request.dry_run,
            verbose=True,
        ),
    )
    return PredictionRunTriggerResponse(**summary)
