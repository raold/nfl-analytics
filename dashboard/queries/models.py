"""
Model Queries

SQL queries for ml.model_registry and ml.training_runs tables.

Author: Claude Code
Date: November 2025
"""

import pandas as pd


class ModelQueries:
    """Centralized queries for model registry and training metadata."""

    @staticmethod
    def model_registry(conn) -> pd.DataFrame:
        """
        Fetch all models from ml.model_registry.

        Returns:
            DataFrame with columns:
                - model_id, model_name, model_version
                - model_type, model_family, status
                - created_at, updated_at
                - description, performance_summary (from notes)
        """
        query = """
        SELECT
            model_id,
            model_name,
            version as model_version,
            model_type,
            model_family,
            status,
            created_at,
            updated_at,
            description,
            notes as performance_summary
        FROM ml.model_registry
        ORDER BY updated_at DESC
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def model_by_id(conn, model_id: int) -> pd.DataFrame | None:
        """
        Fetch single model from registry by ID.

        Args:
            conn: Database connection
            model_id: Model ID

        Returns:
            Single-row DataFrame or empty DataFrame if not found
        """
        query = """
        SELECT
            model_id,
            model_name,
            version as model_version,
            model_type,
            model_family,
            status,
            created_at,
            updated_at,
            description,
            notes as performance_summary
        FROM ml.model_registry
        WHERE model_id = %s
        """

        return pd.read_sql_query(query, conn, params=(model_id,))

    @staticmethod
    def models_by_status(conn, status: str = "production") -> pd.DataFrame:
        """
        Fetch models filtered by status.

        Args:
            conn: Database connection
            status: Model status ('production', 'staging', 'active', 'archived', etc.)

        Returns:
            DataFrame with model metadata
        """
        query = """
        SELECT
            model_id,
            model_name,
            version as model_version,
            model_type,
            model_family,
            status,
            created_at,
            updated_at,
            description,
            notes as performance_summary
        FROM ml.model_registry
        WHERE status = %s
        ORDER BY updated_at DESC
        """

        return pd.read_sql_query(query, conn, params=(status,))

    @staticmethod
    def training_runs(conn, model_id: int | None = None, limit: int = 10) -> pd.DataFrame:
        """
        Fetch recent training runs.

        Args:
            conn: Database connection
            model_id: Optional filter for specific model
            limit: Maximum runs to return (default: 10)

        Returns:
            DataFrame with columns:
                - run_id, model_id, started_at, completed_at
                - status, final_metrics, training_duration_seconds
        """
        query = """
        SELECT
            run_id,
            model_id,
            started_at,
            completed_at,
            status,
            final_metrics,
            training_duration_seconds
        FROM ml.training_runs
        """

        params = []
        if model_id:
            query += " WHERE model_id = %s"
            params.append(model_id)

        query += f" ORDER BY started_at DESC LIMIT {limit}"

        return pd.read_sql_query(query, conn, params=params if params else None)

    @staticmethod
    def latest_training_run(conn, model_id: int) -> pd.DataFrame | None:
        """
        Fetch latest training run for a specific model.

        Args:
            conn: Database connection
            model_id: Model ID

        Returns:
            Single-row DataFrame or empty DataFrame if no runs found
        """
        query = """
        SELECT
            run_id,
            model_id,
            started_at,
            completed_at,
            status,
            final_metrics,
            training_duration_seconds
        FROM ml.training_runs
        WHERE model_id = %s
        ORDER BY started_at DESC
        LIMIT 1
        """

        return pd.read_sql_query(query, conn, params=(model_id,))

    @staticmethod
    def model_family_comparison(conn, model_family: str) -> pd.DataFrame:
        """
        Compare all versions of a specific model family.

        Args:
            conn: Database connection
            model_family: Model family (e.g., 'XGBoost', 'BNN', 'State-Space')

        Returns:
            DataFrame with all versions ordered by created_at DESC
        """
        query = """
        SELECT
            model_id,
            model_name,
            version as model_version,
            model_type,
            status,
            created_at,
            updated_at,
            description,
            notes as performance_summary
        FROM ml.model_registry
        WHERE model_family = %s
        ORDER BY created_at DESC
        """

        return pd.read_sql_query(query, conn, params=(model_family,))

    @staticmethod
    def ensemble_components(conn) -> pd.DataFrame:
        """
        Fetch models that are components of production ensemble.

        Returns:
            DataFrame with ensemble component models and their weights
        """
        query = """
        SELECT
            mr.model_id,
            mr.model_name,
            mr.version as model_version,
            mr.status,
            mr.updated_at,
            COALESCE(
                CASE
                    WHEN mr.model_name LIKE '%State-Space%' THEN 0.30
                    WHEN mr.model_name LIKE '%Hierarchical%' THEN 0.30
                    WHEN mr.model_name LIKE '%XGBoost%' THEN 0.25
                    WHEN mr.model_name LIKE '%Informative%' THEN 0.15
                    ELSE NULL
                END,
                0
            ) as ensemble_weight
        FROM ml.model_registry mr
        WHERE mr.status IN ('production', 'active')
          AND (
              mr.model_name LIKE '%State-Space%'
              OR mr.model_name LIKE '%Hierarchical%'
              OR mr.model_name LIKE '%XGBoost%'
              OR mr.model_name LIKE '%Informative%'
          )
        ORDER BY ensemble_weight DESC NULLS LAST
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def training_run_history(conn, model_id: int, limit: int = 50) -> pd.DataFrame:
        """
        Fetch training run history for a model with summary stats.

        Args:
            conn: Database connection
            model_id: Model ID
            limit: Maximum runs to return (default: 50)

        Returns:
            DataFrame with run history and performance metrics
        """
        query = f"""
        SELECT
            run_id,
            model_id,
            started_at,
            completed_at,
            status,
            final_metrics,
            training_duration_seconds,
            EXTRACT(EPOCH FROM (completed_at - started_at)) / 3600 as duration_hours
        FROM ml.training_runs
        WHERE model_id = %s
        ORDER BY started_at DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn, params=(model_id,))

    @staticmethod
    def model_update_timeline(conn, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch timeline of model updates.

        Args:
            conn: Database connection
            days_back: Number of days to look back (default: 30)

        Returns:
            DataFrame with columns:
                - model_id, model_name, version, status
                - updated_at, days_since_update
        """
        query = f"""
        SELECT
            model_id,
            model_name,
            version as model_version,
            status,
            updated_at,
            CAST(EXTRACT(DAY FROM NOW() - updated_at) AS INTEGER) as days_since_update
        FROM ml.model_registry
        WHERE updated_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY updated_at DESC
        """

        return pd.read_sql_query(query, conn)
