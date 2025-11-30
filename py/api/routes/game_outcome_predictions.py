#!/usr/bin/env python3
"""
GameOutcomeModel prediction APIs.

Exposes prediction runs and stored predictions from ml.game_outcome_predictions so
the dashboard (and other clients) can consume them via the authenticated API.
"""

from __future__ import annotations

import asyncio
from functools import partial
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
LATEST_VIEW = "ml.game_outcome_predictions_latest"
TABLE_NAME = settings.game_outcome_table
VALID_STATES = {"upcoming", "in_progress", "final"}


def _determine_default_season(conn: psycopg.Connection) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(season), EXTRACT(YEAR FROM NOW())::INT) FROM ml.game_outcome_predictions"
    ).fetchone()
    return int(row[0])


def _normalize_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.lower()
    if normalized not in VALID_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid game_state '{value}'. Allowed: {sorted(VALID_STATES)}",
        )
    return normalized


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
    home_moneyline: Optional[float]
    away_moneyline: Optional[float]
    home_spread_odds: Optional[float]
    away_spread_odds: Optional[float]
    over_odds: Optional[float]
    under_odds: Optional[float]
    odds_last_fetched: Optional[datetime]
    actual_home_score: Optional[int]
    actual_away_score: Optional[int]
    actual_spread: Optional[float]
    actual_total: Optional[float]
    game_state: str
    game_status: Optional[str]
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
    game_state: Optional[str] = Query(
        None,
        description="Filter by game state: upcoming, in_progress, final",
    ),
    _: APIKey = Depends(require_api_key),
):
    """Return stored predictions filtered by season/week/team/run."""
    table_name = TABLE_NAME if prediction_run_id else LATEST_VIEW
    normalized_state = _normalize_state(game_state)

    game_state_expr = """
        CASE
            WHEN COALESCE(g.game_status, '') ILIKE 'final%%'
                 OR (g.home_score IS NOT NULL AND g.away_score IS NOT NULL)
                THEN 'final'
            WHEN g.kickoff <= NOW()
                 AND (g.home_score IS NULL OR g.away_score IS NULL)
                THEN 'in_progress'
            ELSE 'upcoming'
        END
    """

    base_query = f"""
        SELECT
            gp.prediction_id,
            gp.prediction_run_id,
            gp.game_id,
            gp.season,
            gp.week,
            gp.home_team,
            gp.away_team,
            g.kickoff,
            gp.pred_home_spread,
            gp.pred_total,
            gp.pred_home_win_prob,
            gp.spread_std,
            gp.total_std,
            g.spread_close AS vegas_spread,
            g.total_close AS vegas_total,
            g.home_moneyline,
            g.away_moneyline,
            g.home_spread_odds,
            g.away_spread_odds,
            g.over_odds,
            g.under_odds,
            g.odds_last_fetched,
            gp.actual_home_score,
            gp.actual_away_score,
            gp.actual_spread,
            gp.actual_total,
            gp.model_version,
            gp.prediction_timestamp,
            gp.inference_time_ms,
            (gp.pred_home_spread - g.spread_close) AS spread_edge,
            (gp.pred_total - g.total_close) AS total_edge,
            {game_state_expr} AS game_state,
            g.game_status
        FROM {table_name} gp
        JOIN public.games g ON g.game_id = gp.game_id
        WHERE 1=1
    """

    params = {}

    try:
        with _get_connection() as conn:
            effective_season = season or _determine_default_season(conn)
            base_query += " AND gp.season = %(season)s"
            params["season"] = effective_season

            if prediction_run_id:
                base_query += " AND gp.prediction_run_id = %(prediction_run_id)s"
                params["prediction_run_id"] = str(prediction_run_id)

            if week:
                base_query += " AND gp.week = %(week)s"
                params["week"] = week

            if team:
                base_query += " AND (gp.home_team = %(team)s OR gp.away_team = %(team)s)"
                params["team"] = team.upper()

            if normalized_state:
                base_query += " AND " + game_state_expr + " = %(state)s"
                params["state"] = normalized_state

            base_query += " ORDER BY gp.week, g.kickoff NULLS LAST"
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
