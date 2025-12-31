"""
Centralized Database Configuration for NFL Analytics.

This module provides a single source of truth for database connection settings,
supporting both local development (localhost:5544) and Docker (nfl-analytics-pg-1:5432).

Usage:
    from py.utils.db_config import get_db_config, get_db_url

    # Get config dict
    config = get_db_config()
    conn = psycopg2.connect(**config)

    # Get URL string
    url = get_db_url()
    engine = create_engine(url)

Environment Variables:
    DB_HOST: Database host (default: localhost)
    DB_PORT: Database port (default: 5544)
    DB_NAME: Database name (default: devdb01)
    DB_USER: Database user (default: dro)
    DB_PASSWORD: Database password (default: sicillionbillions)

Docker Note:
    When running in Docker containers, set:
    - DB_HOST=nfl-analytics-pg-1
    - DB_PORT=5432
"""

import os
import logging

logger = logging.getLogger(__name__)

# Default values for local development
_DEFAULTS = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_db_config() -> dict:
    """
    Get database configuration dictionary.

    Returns:
        Dict with host, port, dbname, user, password keys.
        Values come from environment variables or defaults.
    """
    config = {
        "host": os.getenv("DB_HOST", _DEFAULTS["host"]),
        "port": int(os.getenv("DB_PORT", _DEFAULTS["port"])),
        "dbname": os.getenv("DB_NAME", _DEFAULTS["dbname"]),
        "user": os.getenv("DB_USER", _DEFAULTS["user"]),
        "password": os.getenv("DB_PASSWORD", _DEFAULTS["password"]),
    }

    # Log if running in non-default mode (e.g., Docker)
    if config["host"] != _DEFAULTS["host"]:
        logger.debug(f"Using non-default DB host: {config['host']}:{config['port']}")

    return config


def get_db_url() -> str:
    """
    Get database URL string.

    Returns:
        PostgreSQL connection URL string.
    """
    config = get_db_config()
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}"


def get_connection():
    """
    Get a psycopg2 connection using centralized config.

    Returns:
        psycopg2.connection object
    """
    import psycopg2
    config = get_db_config()
    return psycopg2.connect(**config)
