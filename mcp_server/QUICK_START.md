# MCP Server Quick Start Guide

## What We've Built So Far

### Phase 1 MVP: Database + ESPN News

**Status**: Partial implementation (60% complete)

### Completed Components:

1. **Environment Setup** ✅
   - MCP SDK installed (`mcp[cli]>=1.2.0`)
   - Dependencies: `requests`, `beautifulsoup4`, `lxml`
   - Directory structure created

2. **Configuration** ✅
   - `config.py`: Handles .env loading, database URL, logging
   - Automatically creates log and cache directories
   - Safety limits configured (max query rows, timeout)

3. **Package Structure** ✅
   - `__init__.py`: Package metadata
   - Logging configured to both file and console

### Files Created:
```
mcp_server/
├── __init__.py          ✅ Package initialization
├── config.py            ✅ Configuration management
├── MCP_BUILD_STATUS.md  ✅ Build tracking
├── QUICK_START.md       ✅ This file
└── tools/
    └── (empty - to be created)
```

---

## Next Steps to Complete Phase 1 MVP

### 1. Create `mcp_server/tools/__init__.py`
Simple package init file

### 2. Create `mcp_server/tools/database.py`
PostgreSQL query tool with:
- Safe SQL execution (parameterized queries)
- Connection pooling
- Query timeout enforcement
- Row limit enforcement
- Error handling and logging

**Key Function**: `query_database(sql: str) -> dict`

###3. Create `mcp_server/tools/news_fetcher.py`
ESPN news scraper with:
- Team name normalization (e.g., "PHI" → "Eagles")
- RSS feed parsing
- HTML scraping fallback
- Caching (1-hour TTL)
- Rate limiting

**Key Functions**:
- `fetch_espn_news(team: str, limit: int = 10) -> list[dict]`
- `search_all_nfl_news(keywords: list[str], hours: int = 24) -> list[dict]`

### 4. Create `mcp_server/server.py`
FastMCP server with:
- Tool registration
- HTTP server on port 3000
- Request logging
- Error handling

**Example structure**:
```python
from mcp import FastMCP
from mcp_server.tools.database import query_database
from mcp_server.tools.news_fetcher import fetch_espn_news

app = FastMCP("nfl-analytics")

@app.tool()
async def query_db(sql: str):
    """Execute SQL query against NFL database"""
    return query_database(sql)

@app.tool()
async def get_team_news(team: str, limit: int = 10):
    """Fetch recent ESPN news for a team"""
    return fetch_espn_news(team, limit)

if __name__ == "__main__":
    app.run(port=3000)
```

---

## Testing the MVP

Once all files are created, test with:

```bash
# Start MCP server
uv run python -m mcp_server.server

# In another terminal, test with Claude Code or API client
# Example query: "Show me recent Eagles news and their last 5 games EPA"
```

Expected workflow:
1. Claude calls `get_team_news("Eagles", 10)` → Gets recent articles
2. Claude calls `query_db("SELECT * FROM games WHERE home_team = 'PHI' OR away_team = 'PHI' ORDER BY game_date DESC LIMIT 5")` → Gets game data
3. Claude synthesizes results into natural language response

---

## Environment Variables Needed

Add to `.env`:
```bash
# Your API key (already have this)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Database (already configured)
DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/devdb01

# ESPN (no auth needed)
ESPN_API_BASE_URL=https://sports.core.api.espn.com/v2

# MCP Server
MCP_SERVER_PORT=3000
MCP_LOG_LEVEL=INFO
```

---

## What's Next After MVP

### Phase 2: Statistical Integration
- `tools/statistics.py`: Permutation tests, effect size, bootstrap CIs
- `tools/narrative.py`: End-to-end narrative validation

### Phase 3: Social Monitoring
- `tools/social_monitor.py`: X.com NFL Portal scraping

### Phase 4: Model Integration
- `tools/models.py`: XGBoost inference + SHAP explanations

### Phase 5: Dashboard
- `dashboard/pages/04_MCP_Explorer.py`: Streamlit visualization

---

## Current Status

**Phase 1**: 60% complete (config done, need tools + server)
**Estimated time to MVP**: 1-2 hours of focused work

**Next action**: Create the 4 remaining files listed above

---

Last Updated: 2025-10-30
