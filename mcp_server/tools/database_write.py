"""
Database write operations for NFL analytics MCP server.
Provides safe INSERT/UPDATE operations for model registry and predictions.
"""

import json
import logging
from typing import Any

import psycopg2

from mcp_server.tools.database import get_db_connection

logger = logging.getLogger(__name__)


def insert_model_metadata(
    model_name: str,
    version: str,
    model_type: str,
    model_family: str,
    model_path: str,
    description: str | None = None,
    notes: str | None = None,
    **kwargs
) -> dict[str, Any]:
    """
    Insert model metadata into ml.model_registry.

    Args:
        model_name: Name of the model (e.g., "bnn_passing_v3.1")
        version: Version string (e.g., "3.1.0")
        model_type: Type of model (e.g., "bayesian_neural_network")
        model_family: Framework/family (e.g., "pymc", "brms", "xgboost")
        model_path: Path to model file
        description: Human-readable description
        notes: Additional notes (can include JSON for metrics)
        **kwargs: Additional fields (metadata_path, checkpoint_dir, etc.)

    Returns:
        Dictionary with:
            - success: bool
            - model_id: int (if successful)
            - error: error message (if failed)
    """
    try:
        logger.info(f"Inserting model metadata: {model_name} v{version}")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO ml.model_registry
                (model_name, version, model_type, model_family, model_path,
                 description, notes, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'trained', NOW(), NOW())
                RETURNING model_id
                """

                params = (
                    model_name,
                    version,
                    model_type,
                    model_family,
                    model_path,
                    description,
                    notes
                )

                cursor.execute(sql, params)
                model_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Model registered successfully: model_id={model_id}")

                return {
                    "success": True,
                    "model_id": model_id,
                    "message": f"Model {model_name} v{version} registered with ID {model_id}"
                }

    except psycopg2.errors.UniqueViolation:
        error_msg = f"Model {model_name} v{version} already exists"
        logger.warning(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


def update_model_status(
    model_id: int,
    status: str,
    notes: str | None = None
) -> dict[str, Any]:
    """
    Update model status in registry.

    Args:
        model_id: ID of the model
        status: New status ('trained', 'validated', 'deployed', 'deprecated')
        notes: Optional notes about the status change

    Returns:
        Success/error dictionary
    """
    valid_statuses = {'trained', 'validated', 'deployed', 'deprecated', 'failed'}

    if status not in valid_statuses:
        return {
            "success": False,
            "error": f"Invalid status. Must be one of: {valid_statuses}"
        }

    try:
        logger.info(f"Updating model {model_id} status to: {status}")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Update status
                sql_update = """
                UPDATE ml.model_registry
                SET status = %s, updated_at = NOW()
                WHERE model_id = %s
                RETURNING model_name, version
                """

                cursor.execute(sql_update, (status, model_id))
                result = cursor.fetchone()

                if not result:
                    return {
                        "success": False,
                        "error": f"Model ID {model_id} not found"
                    }

                model_name, version = result

                # Log status change if notes provided
                if notes:
                    sql_log = """
                    INSERT INTO ml.model_audit_log
                    (model_id, action, details, created_at)
                    VALUES (%s, %s, %s, NOW())
                    """
                    log_details = json.dumps({
                        "status_change": status,
                        "notes": notes
                    })
                    cursor.execute(sql_log, (model_id, 'status_change', log_details))

                conn.commit()

                logger.info(f"Model {model_name} v{version} status updated to: {status}")

                return {
                    "success": True,
                    "message": f"Model {model_name} v{version} status updated to {status}"
                }

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }


def insert_prediction_batch(
    model_id: int,
    game_week: str,
    predictions: list
) -> dict[str, Any]:
    """
    Insert a batch of predictions for a game week.

    Args:
        model_id: ID of the model making predictions
        game_week: Week identifier (e.g., "2024-W09")
        predictions: List of dicts with keys:
            - game_id: str
            - player_id: str
            - prop_type: str (e.g., "passing_yards")
            - prediction: float
            - uncertainty_lower: float (optional)
            - uncertainty_upper: float (optional)

    Returns:
        Success/error dictionary with count of inserted predictions
    """
    try:
        logger.info(f"Inserting {len(predictions)} predictions for week {game_week}")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO ml.predictions
                (model_id, game_id, player_id, prop_type, prediction,
                 uncertainty_lower, uncertainty_upper, game_week, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """

                for pred in predictions:
                    params = (
                        model_id,
                        pred['game_id'],
                        pred['player_id'],
                        pred['prop_type'],
                        pred['prediction'],
                        pred.get('uncertainty_lower'),
                        pred.get('uncertainty_upper'),
                        game_week
                    )
                    cursor.execute(sql, params)

                conn.commit()

                logger.info(f"Successfully inserted {len(predictions)} predictions")

                return {
                    "success": True,
                    "count": len(predictions),
                    "message": f"Inserted {len(predictions)} predictions for week {game_week}"
                }

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
