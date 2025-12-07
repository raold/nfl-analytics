"""
Configuration management for MCP server.
Loads from environment variables and .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")

# ESPN Configuration
ESPN_API_BASE_URL = os.getenv(
    "ESPN_API_BASE_URL",
    "https://sports.core.api.espn.com/v2"
)

# MCP Server Configuration
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "3000"))
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
MCP_CACHE_DIR = PROJECT_ROOT / "cache" / "mcp"
MCP_LOG_DIR = PROJECT_ROOT / "logs" / "mcp"

# Ensure directories exist
MCP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Query limits for safety
MAX_QUERY_ROWS = 10000
MAX_QUERY_TIME_SECONDS = 30

# News fetching limits
MAX_NEWS_ARTICLES = 100
NEWS_CACHE_SECONDS = 3600  # 1 hour

# Logging setup
import logging

logging.basicConfig(
    level=getattr(logging, MCP_LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MCP_LOG_DIR / "mcp_server.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"MCP Server configuration loaded from {PROJECT_ROOT}")
logger.info(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'Not configured'}")
logger.info(f"Cache directory: {MCP_CACHE_DIR}")
