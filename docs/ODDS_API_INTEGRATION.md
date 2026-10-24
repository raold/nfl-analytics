# Odds API Integration Guide

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 8, 2025
**Version:** 1.0
**Status:** Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup Instructions](#setup-instructions)
4. [Configuration](#configuration)
5. [Running the Service](#running-the-service)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [API Details](#api-details)
8. [Database Schema](#database-schema)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

The NFL Analytics odds fetching system provides real-time betting odds data from The Odds API with production-grade reliability, monitoring, and error handling.

### Key Features

- **Automated Updates**: Polls API every 15 minutes (configurable) during game days
- **Retry Logic**: Exponential backoff with jitter for failed requests
- **Rate Limiting**: Respects API quota limits with intelligent throttling
- **Line Movement Detection**: Alerts on significant line changes (>5% threshold)
- **Deduplication**: Prevents redundant database inserts
- **Monitoring**: Comprehensive metrics and health checks
- **Graceful Shutdown**: Handles SIGTERM/SIGINT signals properly

### System Components

1. **Odds Update Service** (`py/data/odds_update_service.py`)
   - Continuous polling service with error handling
   - Line movement detection and alerting
   - Metrics tracking and logging

2. **Prop Odds Fetcher** (`py/data/fetch_prop_odds.py`)
   - Player prop odds parsing and enrichment
   - Player ID matching and game mapping
   - Database insertion with deduplication

3. **Historical Odds Extractor** (`etl/extract/odds_api.py`)
   - One-time historical data fetching
   - Backfill capability for past dates

4. **Docker Service** (`containers/Dockerfile.odds-fetcher`)
   - Containerized deployment
   - Health checks and restart policies

---

## Architecture

### Data Flow Diagram

```
┌─────────────────────┐
│   The Odds API      │
│  (External Service) │
└──────────┬──────────┘
           │ HTTPS/REST
           │ (15 min interval)
           ▼
┌─────────────────────┐
│  Odds Update        │
│  Service            │◄──── Environment Variables
│  (Docker Container) │      (API Key, Config)
└──────────┬──────────┘
           │
           ├──────────────────┐
           ▼                  ▼
┌─────────────────────┐  ┌──────────────────┐
│  odds_history       │  │  prop_lines      │
│  (TimescaleDB)      │  │  _history        │
│  820,080 rows       │  │  (PostgreSQL)    │
└─────────────────────┘  └──────────────────┘
           │                  │
           └────────┬─────────┘
                    ▼
           ┌─────────────────────┐
           │  Grafana Dashboard  │
           │  Prometheus Metrics │
           │  Alertmanager       │
           └─────────────────────┘
```

### Service Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Odds Update Service                          │
│                                                            │
│  ┌────────────────┐      ┌──────────────────┐            │
│  │  API Client    │─────►│  Retry Logic     │            │
│  │  (requests)    │      │  (exp. backoff)  │            │
│  └────────────────┘      └──────────────────┘            │
│           │                       │                       │
│           ▼                       ▼                       │
│  ┌────────────────┐      ┌──────────────────┐            │
│  │  Parser        │─────►│  Enrichment      │            │
│  │  (JSON→Dict)   │      │  (Player IDs)    │            │
│  └────────────────┘      └──────────────────┘            │
│           │                       │                       │
│           ▼                       ▼                       │
│  ┌────────────────┐      ┌──────────────────┐            │
│  │  Line Movement │      │  Deduplication   │            │
│  │  Detection     │      │  Logic           │            │
│  └────────────────┘      └──────────────────┘            │
│           │                       │                       │
│           ▼                       ▼                       │
│  ┌────────────────┐      ┌──────────────────┐            │
│  │  Alerting      │      │  Database Insert │            │
│  │  (Webhook)     │      │  (PostgreSQL)    │            │
│  └────────────────┘      └──────────────────┘            │
│                                                            │
│  ┌───────────────────────────────────────────┐            │
│  │  Metrics Tracking                         │            │
│  │  - API calls (success/failure)            │            │
│  │  - Quota usage                            │            │
│  │  - Line movements detected                │            │
│  │  - Props inserted/duplicates              │            │
│  └───────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Prerequisites

1. **The Odds API Key**
   - Sign up at https://the-odds-api.com
   - Get free tier (500 requests/month) or paid plan
   - Note your API key

2. **PostgreSQL Database**
   - PostgreSQL 14+ with TimescaleDB extension
   - Running on `localhost:5544` (or configure custom)
   - Database: `devdb01`

3. **Docker & Docker Compose**
   - Docker Engine 20.10+
   - Docker Compose 2.0+

### Step 1: Set Environment Variables

Create a `.env` file in the project root:

```bash
# The Odds API Configuration
ODDS_API_KEY=your_api_key_here

# Database Configuration (optional, defaults shown)
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=sicillionbillions

# Service Configuration (optional)
ODDS_POLL_INTERVAL=900          # 15 minutes
LINE_MOVE_THRESHOLD=5.0         # 5% line movement alert
LOG_LEVEL=INFO

# Alerting (optional)
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**⚠️ SECURITY WARNING:** Never commit `.env` file to version control!

### Step 2: Verify Database Schema

Ensure the odds tables exist:

```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "\d odds_history"

psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "\d prop_lines_history"
```

If tables don't exist, run migrations:

```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/migrations/009_odds_optimization.sql
```

### Step 3: Test API Connection

Test your API key works:

```bash
# One-time test fetch
python py/data/odds_update_service.py --once --dry-run
```

Expected output:
```
[INFO] OddsUpdateService initialized
[INFO] Poll interval: 900s
[INFO] Fetched 16 upcoming NFL events
[INFO] DRY RUN: Would insert 247 prop records
[INFO] Poll complete: 0 props inserted in 12.3s
```

### Step 4: Build Docker Image

```bash
# Build odds fetcher image
docker build -f containers/Dockerfile.odds-fetcher -t nfl-analytics-odds-fetcher .
```

---

## Configuration

### Service Configuration

All configuration via environment variables (see `.env` above).

#### Key Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `ODDS_API_KEY` | (required) | Your API key from The Odds API |
| `POLL_INTERVAL` | 900 | Polling interval in seconds (15 min) |
| `LINE_MOVE_THRESHOLD` | 5.0 | Line movement alert threshold (%) |
| `MAX_RETRIES` | 5 | Maximum retry attempts on failure |
| `REQUEST_TIMEOUT` | 30 | HTTP request timeout (seconds) |

#### Polling Schedule

The service uses intelligent polling based on game proximity:

| Time Until Game | Polling Frequency | Rationale |
|----------------|-------------------|-----------|
| > 1 week | Every 24 hours | Odds rarely change |
| 3-7 days | Every 6 hours | Some line movement |
| 1-3 days | Every 2 hours | Increasing activity |
| < 24 hours | Every 15 minutes | High volatility |

This minimizes API usage while capturing critical line movements.

### Bookmaker Selection

Default bookmakers (can customize in `OddsServiceConfig`):
- **DraftKings** (`draftkings`)
- **FanDuel** (`fanduel`)
- **BetMGM** (`betmgm`)
- **Caesars** (`caesars`)
- **Pinnacle** (`pinnacle`) - Sharp book
- **BetOnline** (`betonlineag`)

To add/remove bookmakers, edit `py/data/odds_update_service.py`:

```python
class OddsServiceConfig:
    def __post_init__(self):
        if self.bookmakers is None:
            self.bookmakers = [
                "draftkings",
                "fanduel",
                # Add more here
            ]
```

### Market Selection

Default prop markets:
- `player_pass_yds` - Passing yards
- `player_pass_tds` - Passing touchdowns
- `player_rush_yds` - Rushing yards
- `player_reception_yds` - Receiving yards
- `player_receptions` - Receptions

Available markets (see `py/data/fetch_prop_odds.py` for full list):
- `player_pass_completions`
- `player_pass_interceptions`
- `player_rush_attempts`
- `player_anytime_td`
- `player_first_td`
- `player_kicking_points`

---

## Running the Service

### Option 1: Docker Compose (Recommended)

```bash
# Start with odds profile
docker compose --profile odds up -d

# Check logs
docker compose logs -f odds-fetcher

# Stop service
docker compose --profile odds down
```

### Option 2: Standalone Docker

```bash
# Run container
docker run -d \
  --name odds-fetcher \
  --env-file .env \
  --network nfl-analytics \
  -v $(pwd)/logs/odds:/app/logs/odds \
  nfl-analytics-odds-fetcher

# View logs
docker logs -f odds-fetcher

# Stop container
docker stop odds-fetcher
docker rm odds-fetcher
```

### Option 3: Direct Python (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run continuously
python py/data/odds_update_service.py

# Run once (testing)
python py/data/odds_update_service.py --once

# Dry run (no database writes)
python py/data/odds_update_service.py --dry-run
```

### Graceful Shutdown

The service handles shutdown signals gracefully:

```bash
# Send SIGTERM (Docker stop)
docker compose --profile odds stop odds-fetcher

# Or SIGINT (Ctrl+C in terminal)
# Press Ctrl+C when running directly
```

On shutdown, the service will:
1. Stop polling loop
2. Log final metrics summary
3. Save metrics to JSON file
4. Close database connections
5. Exit cleanly

---

## Monitoring & Alerting

### Logs

Service logs are written to:
- **Console**: `stdout` (visible in `docker logs`)
- **File**: `logs/odds/odds_service_YYYYMMDD.log`

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Example log output:
```
2025-11-08 10:30:15 [INFO] [odds_update_service] Starting odds poll
2025-11-08 10:30:16 [INFO] [odds_update_service] Fetched 14 upcoming NFL events
2025-11-08 10:30:17 [INFO] [odds_update_service] API requests remaining: 487
2025-11-08 10:30:42 [WARNING] [odds_update_service] LINE MOVEMENT: Patrick Mahomes passing_yards @ draftkings: 285.5 → 275.5 (-3.5%)
2025-11-08 10:32:18 [INFO] [odds_update_service] Poll complete: 347 props inserted in 123.4s
```

### Metrics

Service tracks comprehensive metrics (see `ServiceMetrics` dataclass):

#### API Metrics
- `api_calls_total` - Total API calls made
- `api_calls_success` - Successful API calls
- `api_calls_failed` - Failed API calls
- `api_requests_remaining` - Quota remaining this month
- `api_requests_used` - Quota used this month

#### Data Metrics
- `events_fetched` - NFL events fetched
- `props_parsed` - Player props parsed
- `props_inserted` - Props inserted into database
- `props_duplicates` - Duplicate props skipped

#### Line Movement Metrics
- `line_moves_detected` - Significant line movements (>5%)
- `alerts_sent` - Alerts sent via webhook

#### Error Metrics
- `errors_total` - Total errors encountered
- `retries_total` - Total retry attempts

#### Timing Metrics
- `last_poll_at` - Timestamp of last poll
- `last_success_at` - Timestamp of last successful poll
- `poll_duration_avg` - Average poll duration (seconds)

Metrics are:
1. Logged to console on shutdown
2. Saved to `logs/odds/metrics_YYYYMMDD_HHMMSS.json`
3. Available via Prometheus (see below)

### Grafana Dashboard

Use the SQL queries in `db/queries/odds_monitoring_dashboard.sql` to create Grafana panels.

**Example Panels:**

1. **API Usage Gauge**
   - Query: Current month API calls / quota limit
   - Visualization: Gauge (0-100%)
   - Alert: > 90% usage

2. **Odds Freshness Heatmap**
   - Query: Hours since last odds update per game
   - Visualization: Table with color gradient
   - Alert: > 2 hours for games within 24h

3. **Line Movement Timeline**
   - Query: Line movements > 5% over time
   - Visualization: Time series
   - Alert: > 10 movements in 1 hour

4. **Bookmaker Coverage**
   - Query: Events covered by each bookmaker
   - Visualization: Bar chart
   - Alert: Coverage < 75%

### Prometheus Integration

The `postgres_exporter` can scrape odds metrics from PostgreSQL.

**1. Configure postgres_exporter queries** (`prometheus/postgres_exporter_queries.yaml`):

```yaml
# Add this to your queries file
odds_api_calls_total:
  query: |
    SELECT
      api_name as api,
      calls_made as value
    FROM api_usage_tracker
    WHERE month = DATE_TRUNC('month', CURRENT_DATE)
  metrics:
    - api:
        usage: "LABEL"
        description: "API name"
    - value:
        usage: "GAUGE"
        description: "Total API calls this month"

odds_api_quota_remaining:
  query: |
    SELECT
      api_name as api,
      quota_limit - calls_made as value
    FROM api_usage_tracker
    WHERE month = DATE_TRUNC('month', CURRENT_DATE)
  metrics:
    - api:
        usage: "LABEL"
    - value:
        usage: "GAUGE"
        description: "API quota remaining"

odds_snapshot_freshness_hours:
  query: |
    SELECT
      g.game_id,
      EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 as value
    FROM games g
    LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
    WHERE g.kickoff > NOW() AND g.kickoff < NOW() + INTERVAL '7 days'
    GROUP BY g.game_id
  metrics:
    - game_id:
        usage: "LABEL"
    - value:
        usage: "GAUGE"
        description: "Hours since last odds snapshot"
```

**2. Add alert rules** (`prometheus/alert_rules.yml`):

```yaml
groups:
  - name: odds_api_alerts
    interval: 5m
    rules:
      - alert: OddsAPIQuotaCritical
        expr: odds_api_quota_remaining < 50
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Odds API quota nearly exhausted"
          description: "Only {{ $value }} API calls remaining this month"

      - alert: StaleOddsData
        expr: odds_snapshot_freshness_hours > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Odds data stale for game {{ $labels.game_id }}"
          description: "No odds update for {{ $value }} hours"

      - alert: OddsServiceDown
        expr: up{job="odds-fetcher"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Odds fetching service is down"
```

### Webhook Alerts

Configure Slack/Discord webhook for real-time alerts:

```bash
# Set in .env
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

Alerts sent for:
- **Line movements** > 5%
- **API errors** (rate limits, timeouts)
- **Critical failures** (database down, auth errors)

Example alert payload:
```json
{
  "text": "[NFL Odds Alert] [WARNING] Patrick Mahomes passing_yards moved +7.2% at draftkings",
  "timestamp": "2025-11-08T10:30:42Z"
}
```

---

## API Details

### The Odds API Documentation

Official docs: https://the-odds-api.com/liveapi/guides/v4/

### Endpoints Used

#### 1. List Events

```
GET https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events
```

Parameters:
- `apiKey` (required): Your API key
- `dateFormat`: `iso` (ISO 8601 timestamps)

Response:
```json
[
  {
    "id": "e9b5c8f3a1d2...",
    "sport_key": "americanfootball_nfl",
    "commence_time": "2025-11-10T18:00:00Z",
    "home_team": "Kansas City Chiefs",
    "away_team": "Buffalo Bills"
  }
]
```

#### 2. Get Odds for Event

```
GET https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds
```

Parameters:
- `apiKey` (required): Your API key
- `regions`: `us` (US bookmakers)
- `markets`: Comma-separated (e.g., `player_pass_yds,player_rush_yds`)
- `bookmakers`: Comma-separated (e.g., `draftkings,fanduel`)
- `oddsFormat`: `american` (e.g., -110, +150)
- `dateFormat`: `iso`

Response:
```json
{
  "id": "e9b5c8f3a1d2...",
  "sport_key": "americanfootball_nfl",
  "commence_time": "2025-11-10T18:00:00Z",
  "home_team": "Kansas City Chiefs",
  "away_team": "Buffalo Bills",
  "bookmakers": [
    {
      "key": "draftkings",
      "title": "DraftKings",
      "last_update": "2025-11-08T10:30:15Z",
      "markets": [
        {
          "key": "player_pass_yds",
          "last_update": "2025-11-08T10:30:15Z",
          "outcomes": [
            {
              "name": "Over",
              "description": "Patrick Mahomes",
              "price": -115,
              "point": 285.5
            },
            {
              "name": "Under",
              "description": "Patrick Mahomes",
              "price": -105,
              "point": 285.5
            }
          ]
        }
      ]
    }
  ]
}
```

### Rate Limits

**Free Tier:**
- 500 requests/month
- No rate limit per second
- Resets on 1st of month

**Paid Tiers:**
- 1,000 - 10,000 requests/month
- Higher tiers available

**Best Practices:**
1. Cache responses (don't re-fetch same data)
2. Use event list to get all events in one call
3. Batch requests with appropriate markets/bookmakers
4. Respect `x-requests-remaining` header

### API Response Headers

```
x-requests-remaining: 487
x-requests-used: 13
```

Track these to avoid quota exhaustion!

### Bookmaker Keys

| Bookmaker | Key | Tier | Notes |
|-----------|-----|------|-------|
| DraftKings | `draftkings` | Sharp | High limits |
| FanDuel | `fanduel` | Sharp | High limits |
| BetMGM | `betmgm` | Sharp | MGM network |
| Caesars | `caesars` | Sharp | Caesars network |
| Pinnacle | `pinnacle` | Sharpest | Lowest vig |
| BetOnline | `betonlineag` | Offshore | International |

**Sharp vs Recreational:**
- **Sharp books**: Low vig, high limits, fast line moves (Pinnacle, DraftKings)
- **Recreational books**: Higher vig, lower limits, slower lines (many others)

---

## Database Schema

### odds_history Table

**Purpose:** Historical odds snapshots (h2h, spreads, totals)

```sql
CREATE TABLE odds_history (
    event_id TEXT NOT NULL,
    sport_key TEXT,
    commence_time TIMESTAMPTZ,
    home_team TEXT,
    away_team TEXT,
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,
    market_key TEXT NOT NULL,
    market_last_update TIMESTAMPTZ,
    outcome_name TEXT NOT NULL,
    outcome_price DOUBLE PRECISION,
    outcome_point DOUBLE PRECISION,
    snapshot_at TIMESTAMPTZ NOT NULL,
    book_last_update TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_id, bookmaker_key, market_key, outcome_name, snapshot_at)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('odds_history', 'snapshot_at');
```

**Key Fields:**
- `event_id` - The Odds API event identifier
- `market_key` - Market type (`h2h`, `spreads`, `totals`)
- `outcome_name` - Outcome (team name, `Over`, `Under`)
- `outcome_price` - American odds (e.g., -110, +150)
- `outcome_point` - Point value (spread/total line)
- `snapshot_at` - When odds were fetched (time series key)

**Indexes:**
- Primary key on `(event_id, bookmaker_key, market_key, outcome_name, snapshot_at)`
- BRIN index on `snapshot_at` (TimescaleDB optimization)
- Partial indexes on `market_key` for fast filtering

**Current Data:**
- **820,080 rows**
- Date range: 2023-09-01 to 2025-02-09
- 16 bookmakers, 3 markets

### prop_lines_history Table

**Purpose:** Player prop betting lines (passing yards, TDs, etc.)

```sql
CREATE TABLE prop_lines_history (
    event_id TEXT NOT NULL,
    game_id TEXT,
    sport_key TEXT DEFAULT 'americanfootball_nfl',
    commence_time TIMESTAMPTZ,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    player_position TEXT,
    player_team TEXT,
    prop_type TEXT NOT NULL,
    market_key TEXT NOT NULL,
    line_value DOUBLE PRECISION NOT NULL,
    over_odds DOUBLE PRECISION,
    under_odds DOUBLE PRECISION,
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,
    snapshot_at TIMESTAMPTZ NOT NULL,
    market_last_update TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    over_implied_prob DOUBLE PRECISION,
    under_implied_prob DOUBLE PRECISION,
    book_hold DOUBLE PRECISION,
    line_move_since_open DOUBLE PRECISION,
    is_exchange BOOLEAN DEFAULT false,
    back_price DOUBLE PRECISION,
    lay_price DOUBLE PRECISION,
    liquidity DOUBLE PRECISION,
    bookmaker_tier TEXT,
    PRIMARY KEY (event_id, player_id, prop_type, bookmaker_key, snapshot_at)
);
```

**Key Fields:**
- `player_id` - GSIS player ID (matched from `rosters_weekly`)
- `prop_type` - Prop type (`passing_yards`, `rushing_yards`, etc.)
- `line_value` - Prop line (e.g., 285.5 passing yards)
- `over_odds` / `under_odds` - American odds for Over/Under
- `over_implied_prob` / `under_implied_prob` - Implied probabilities
- `book_hold` - Bookmaker vig/juice percentage
- `line_move_since_open` - Line movement from opening line

**Derived Fields:**
- `over_implied_prob = 100 / (over_odds + 100)` (for positive odds)
- `over_implied_prob = abs(over_odds) / (abs(over_odds) + 100)` (for negative odds)
- `book_hold = (over_implied_prob + under_implied_prob) - 1.0`

**Indexes:**
- Primary key on `(event_id, player_id, prop_type, bookmaker_key, snapshot_at)`
- Index on `(player_id, prop_type, snapshot_at DESC)` for player queries
- Index on `(bookmaker_key, prop_type, snapshot_at DESC)` for bookmaker queries
- Partial index on `book_hold` for low-hold bets (<3%)

### games Table (Odds Integration)

The `games` table was extended with odds tracking columns:

```sql
ALTER TABLE games
ADD COLUMN odds_api_event_id TEXT,
ADD COLUMN odds_last_fetched TIMESTAMPTZ,
ADD COLUMN odds_fetch_priority INT DEFAULT 0,
ADD COLUMN odds_snapshots_count INT DEFAULT 0;
```

**Matching Logic:**
1. Exact match on team names + kickoff date
2. Fuzzy match using `odds_api_name` field from `team_mappings`
3. Date-only match with partial team name (fallback)

See `match_odds_events_to_games()` function in migration `009_odds_optimization.sql`.

### api_usage_tracker Table

**Purpose:** Track API quota usage by month

```sql
CREATE TABLE api_usage_tracker (
    id SERIAL PRIMARY KEY,
    month DATE NOT NULL,
    api_name VARCHAR(50) NOT NULL DEFAULT 'the-odds-api',
    calls_made INT NOT NULL DEFAULT 0,
    quota_limit INT,
    cost_per_call NUMERIC(10,4),
    total_cost NUMERIC(10,2),
    last_call_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(month, api_name)
);
```

Updated automatically by odds service on each poll.

---

## Troubleshooting

### Common Issues

#### 1. "ODDS_API_KEY not found in config or environment"

**Cause:** API key not set in environment.

**Solution:**
```bash
# Set in .env file
echo "ODDS_API_KEY=your_api_key_here" >> .env

# Or export directly
export ODDS_API_KEY=your_api_key_here
```

#### 2. "Rate limit hit. Reset at: 1731024000"

**Cause:** API quota exhausted for the month.

**Solution:**
- Wait until reset date (1st of next month)
- Reduce `POLL_INTERVAL` to use fewer requests
- Upgrade to paid API tier
- Check `api_usage_tracker` table for current usage

```sql
SELECT * FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE);
```

#### 3. "Could not find player ID for: John Doe"

**Cause:** Player name in API doesn't match `rosters_weekly` table.

**Solution:**
- Check player name spelling: `SELECT full_name FROM rosters_weekly WHERE full_name ILIKE '%doe%';`
- API might use nickname (e.g., "Pat Mahomes" vs "Patrick Mahomes")
- Player might not be in roster yet (rookies, recent signings)

**Workaround:** Props without player IDs are filtered out but logged.

#### 4. "Could not match event to game"

**Cause:** Event from API doesn't match any game in `games` table.

**Solution:**
- Check date alignment: `SELECT * FROM games WHERE kickoff::date = '2025-11-10';`
- Run matching function manually: `SELECT * FROM match_odds_events_to_games();`
- Verify team names match: `SELECT * FROM team_mappings;`

#### 5. Database Connection Errors

**Cause:** PostgreSQL not running or wrong credentials.

**Solution:**
```bash
# Check if database is up
docker compose ps pg-1

# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"

# Restart database if needed
docker compose restart pg-1
```

#### 6. Service Keeps Restarting

**Cause:** Uncaught exception in service loop.

**Solution:**
```bash
# Check logs for stack trace
docker compose logs odds-fetcher | tail -100

# Common causes:
# - Invalid API key
# - Database connection issues
# - Network timeout

# Run in dry-run mode to test
docker compose run --rm odds-fetcher python py/data/odds_update_service.py --once --dry-run
```

### Debug Mode

Enable verbose logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or override at runtime
docker compose run --rm -e LOG_LEVEL=DEBUG odds-fetcher
```

### Health Checks

Check service health:

```bash
# Docker health check
docker inspect odds-fetcher | grep -A5 Health

# Manual checks
# 1. Log file exists and is recent
ls -lh logs/odds/odds_service_$(date +%Y%m%d).log

# 2. Database has recent odds
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) FROM odds_history WHERE snapshot_at > NOW() - INTERVAL '1 hour';"

# 3. API usage tracker updated
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT * FROM api_usage_tracker WHERE month = DATE_TRUNC('month', CURRENT_DATE);"
```

---

## Best Practices

### API Usage Optimization

1. **Batch Requests**
   - Fetch multiple markets in one call (use comma-separated `markets` param)
   - Fetch multiple bookmakers in one call (use comma-separated `bookmakers` param)
   - One request can return 100+ props

2. **Intelligent Polling**
   - Poll more frequently for games within 24 hours
   - Reduce frequency for games >1 week out
   - Stop polling after game starts (odds finalized)

3. **Cache Responses**
   - Don't re-fetch unchanged data
   - Compare `market_last_update` timestamps
   - Use database as cache (check latest snapshot before fetching)

4. **Monitor Quota**
   - Set alerts at 75%, 90% usage
   - Reduce polling if approaching limit
   - Track `x-requests-remaining` header

### Data Quality

1. **Deduplication**
   - Use `ON CONFLICT DO NOTHING` in inserts
   - Primary key prevents exact duplicates
   - Log duplicate counts for monitoring

2. **Validation**
   - Reject odds with missing critical fields (`player_id`, `line_value`)
   - Filter outliers (odds > 10,000 unlikely)
   - Validate implied probabilities sum reasonably (should be > 1.0 for book hold)

3. **Enrichment**
   - Always match player names to GSIS IDs
   - Map events to `game_id` in `games` table
   - Calculate derived fields (`book_hold`, `implied_prob`)

### Error Handling

1. **Retry Strategy**
   - Exponential backoff with jitter
   - Max 5 retries
   - Different handling for rate limits vs transient errors

2. **Graceful Degradation**
   - Skip individual event errors, continue polling
   - Log errors but don't crash service
   - Alert on repeated failures

3. **Circuit Breaker**
   - Stop polling if API consistently fails (>10 consecutive errors)
   - Send critical alert
   - Require manual restart

### Monitoring

1. **Key Metrics**
   - API quota usage (alert at >90%)
   - Odds freshness (alert if >2 hours stale for games <24h out)
   - Line movements (track frequency and magnitude)
   - Error rate (alert if >5% of requests fail)

2. **Alerting Thresholds**
   - **Critical**: API quota exhausted, service down, database unreachable
   - **Warning**: Stale odds, missing bookmaker, high error rate
   - **Info**: Significant line movements, quota milestones

3. **Dashboard Panels**
   - API usage gauge (real-time)
   - Odds coverage heatmap (by game)
   - Line movement timeline (last 24h)
   - Bookmaker reliability scores

### Security

1. **API Key Protection**
   - Store in `.env` file (never commit)
   - Use Docker secrets in production
   - Rotate keys periodically

2. **Database Access**
   - Use read-only credentials for reporting
   - Restrict write access to service account
   - Enable SSL for remote connections

3. **Webhook Security**
   - Use HTTPS for alert webhooks
   - Validate webhook signatures (if supported)
   - Rate limit alert sending

---

## Appendix

### File Locations

```
nfl-analytics/
├── py/data/
│   ├── odds_update_service.py      # Main service (765 lines)
│   ├── fetch_prop_odds.py           # Prop fetcher (638 lines)
│   └── ingest_odds_history.py       # Historical ingestion
├── etl/extract/
│   └── odds_api.py                  # ETL extractor (381 lines)
├── containers/
│   └── Dockerfile.odds-fetcher      # Docker image
├── docker-compose.yml               # Service definition (lines 231-262)
├── db/
│   ├── migrations/
│   │   └── 009_odds_optimization.sql # Schema (391 lines)
│   └── queries/
│       └── odds_monitoring_dashboard.sql # Queries (650+ lines)
├── docs/
│   └── ODDS_API_INTEGRATION.md      # This file
└── logs/odds/
    └── odds_service_YYYYMMDD.log    # Daily logs
```

### External Resources

- **The Odds API**: https://the-odds-api.com
  - Documentation: https://the-odds-api.com/liveapi/guides/v4/
  - Dashboard: https://the-odds-api.com/account/

- **TimescaleDB**: https://www.timescale.com
  - Hypertables: https://docs.timescale.com/use-timescale/latest/hypertables/

- **Grafana Dashboards**: https://grafana.com/grafana/dashboards/
  - PostgreSQL datasource: https://grafana.com/docs/grafana/latest/datasources/postgres/

### Related Documentation

- `docs/database/DATABASE_SCHEMA_AUDIT.md` - Full database schema documentation
- `docs/ADVANCED_BAYESIAN_V3.md` - Model integration with odds data
- `CLAUDE.md` - Project overview and architecture

### Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-08 | 1.0 | Initial release - Production-ready odds service |

---

**Questions or Issues?**

Check logs first:
```bash
docker compose logs odds-fetcher | tail -100
```

Run diagnostic query:
```sql
SELECT * FROM mart.odds_coverage;
```

Test API key:
```bash
curl "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey=YOUR_KEY"
```

For further assistance, see troubleshooting section or check project documentation in `docs/`.
