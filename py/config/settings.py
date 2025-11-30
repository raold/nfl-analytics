#!/usr/bin/env python3
"""
Centralized configuration helpers for NFL Analytics.

Provides a single place to load environment-driven settings so that secrets
are not hardcoded across scripts, models, and dashboards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Dict, Any


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch environment variable with optional default."""
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    return value or None


@dataclass(frozen=True)
class DatabaseSettings:
    """Database configuration loaded from environment variables."""

    host: str = _get_env("DB_HOST", "localhost")
    port: int = int(_get_env("DB_PORT", "5544"))
    name: str = _get_env("DB_NAME", "devdb01")
    user: str = _get_env("DB_USER", "dro")
    password: Optional[str] = _get_env("DB_PASSWORD")

    def require_password(self) -> str:
        """Return password or raise a helpful error if missing."""
        if not self.password:
            raise RuntimeError(
                "DB_PASSWORD environment variable must be set. "
                "Use `export DB_PASSWORD=...` or configure your secrets manager."
            )
        return self.password

    @property
    def dsn(self) -> str:
        """Return a psycopg/SQLAlchemy style DSN."""
        return (
            f"postgresql://{self.user}:{self.require_password()}@"
            f"{self.host}:{self.port}/{self.name}"
        )

    def psycopg_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for psycopg2/psycopg connect methods."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.name,
            "user": self.user,
            "password": self.require_password(),
        }


@dataclass(frozen=True)
class APIIntegrationSettings:
    """Settings for API consumers (dashboard, CLI tools, etc.)."""

    base_url: Optional[str] = _get_env("GAME_PREDICTIONS_API_URL")
    api_key: Optional[str] = _get_env("GAME_PREDICTIONS_API_KEY")


@dataclass(frozen=True)
class Settings:
    """Root settings object."""

    database: DatabaseSettings = DatabaseSettings()
    api: APIIntegrationSettings = APIIntegrationSettings()
    game_outcome_table: str = os.getenv(
        "GAME_OUTCOME_TABLE", "ml.game_outcome_predictions"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
