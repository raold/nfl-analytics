# Dashboard Updates - November 8, 2025

## Summary

Updated the Streamlit dashboard (`dashboard/app.py`) with comprehensive system status, model tracking, and data summary pages.

## New Features

### 1. System Status Page

**Access**: Sidebar → "System Status"

**Features**:
- **Docker Services Health**: Real-time health checks for all 7 services
  - PostgreSQL (port 5544)
  - Redis (port 6379)
  - Prometheus (port 9090)
  - Grafana (port 3000)
  - Streamlit (port 8501)
  - Postgres Exporter (port 9187)
  - Node Exporter (port 9100)

- **Database Statistics**:
  - Total games count (7,263)
  - Total plays count (1,254,173)
  - Season coverage (1999-2025)
  - Database size
  - Latest game timestamp

- **Canonical Training Window**:
  - Configuration details (2018-2024 + H=4 decay on 2015-2017)
  - Validation results (MAE: 47.77 yards, Coverage: 86.5%)
  - Statistical significance (p=0.0069)

- **Monitoring Links**:
  - Grafana dashboard (port 3000)
  - Prometheus (port 9090)
  - Database connection info

### 2. Model Status Page

**Access**: Sidebar → "Model Status"

**Features**:
- **Model Registry**: Live data from `ml.model_registry` table
  - Total models, active models, in production, archived
  - Model details table with version, type, framework, status

- **Model Files**: Filesystem scan of trained models
  - Total model files count
  - Total size in MB
  - Largest file
  - Detailed table with filename, size, last modified
  - Handles both Docker (/app/models) and local paths

- **Recent Training Runs**: Last 10 runs from `ml.training_runs`
  - Run ID, model ID, timestamps, status, metrics, duration

- **4-Way Ensemble Configuration**:
  - Component status (Hierarchical Bayesian, XGBoost, BNN, State-Space v2)
  - Combination methods
  - Expected performance metrics (ROI, Win Rate, Sharpe, Drawdown)

### 3. Data Summary Page

**Access**: Sidebar → "Data Summary"

**Features**:
- **Overview Metrics**: Games, plays, season coverage
- **Schema Statistics**: Table counts by schema (public, mart, ml, predictions, etc.)
- **Recent Data Ingestion**: Last 10 weeks of data by season/week/type
- **Data Quality Notes**:
  - Key findings (✅ validated data points)
  - Important notes (column naming conventions, team abbreviations)

## Navigation

The dashboard now uses a **sidebar radio button** navigation:
- **Predictions** (default): Original prediction monitoring page
- **System Status**: New system health and status page
- **Model Status**: New model registry and files page
- **Data Summary**: New data overview and quality page

## Technical Details

### New Functions Added

1. `check_service_health(host, port, service_name)`: Socket-based health check
2. `fetch_database_stats()`: Query database for comprehensive statistics
3. `fetch_model_registry()`: Query `ml.model_registry` table
4. `fetch_training_runs()`: Query `ml.training_runs` table
5. `render_system_status_page()`: Render system status page
6. `render_model_status_page()`: Render model status page
7. `render_data_summary_page()`: Render data summary page

### Docker Integration

Updated `docker-compose.yml`:
- Added models directory mount: `./models:/app/models:ro` (read-only)
- Ensures dashboard can access trained model files in Docker environment

### Path Handling

Model file scanning handles multiple paths:
- `/app/models` (Docker container)
- `models` (local development)
- `../models` (relative path)

Gracefully falls back with informative messages if models directory not found.

## Usage

### Local Development

```bash
# Run locally
cd dashboard
streamlit run app.py
```

### Docker Environment

```bash
# Rebuild and restart
docker compose up -d --build streamlit

# View logs
docker compose logs -f streamlit

# Access dashboard
open http://localhost:8501
```

## Refresh Behavior

- **System Status**: Refresh button clears all caches and reloads data
- **Model Status**: Uses cached data (5-minute TTL)
- **Data Summary**: Uses cached data (5-minute TTL)
- **Predictions**: Original behavior unchanged

## Dependencies

No new dependencies required. All features use existing packages:
- `streamlit==1.31.0`
- `pandas==2.2.0`
- `plotly==5.18.0`
- `psycopg[binary]==3.1.18`
- `numpy==1.26.3`
- `matplotlib==3.8.2`

Standard library used for:
- `socket` (health checks)
- `os`, `glob` (file scanning)
- `datetime` (timestamp formatting)

## Testing

```bash
# Syntax check
python -m py_compile dashboard/app.py

# Run locally for testing
cd dashboard
streamlit run app.py

# Check Docker health
docker compose ps
```

## Screenshots

### System Status Page
- Docker services with green/red health indicators
- Database statistics in metric cards
- Training window configuration details
- Links to Grafana, Prometheus, Database

### Model Status Page
- Model registry table from database
- Model files with sizes and timestamps
- Training runs history
- 4-way ensemble configuration

### Data Summary Page
- Overview metrics
- Schema breakdown by table count
- Recent ingestion data
- Data quality checklist

## Future Enhancements

Potential improvements:
1. Add Redis cache statistics
2. Add training progress tracking (if models are training)
3. Add model performance comparison charts
4. Add automated model health checks
5. Add alerts/notifications for failing services
6. Add training window visualization (chart showing decay weights)
7. Add model file version comparison

## Notes

- All database queries use psycopg3 (psycopg package)
- Health checks use socket connections (2-second timeout)
- File scanning is safe (wrapped in try/except)
- Docker volume mounts are read-only for security
- No external API calls (except optional live scores on Predictions page)
- All data is real-time (not cached long-term)

## Related Files

- `dashboard/app.py` - Main dashboard application
- `dashboard/Dockerfile` - Docker build configuration
- `dashboard/requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker services configuration

## Contact

For issues or questions about the dashboard:
- Check logs: `docker compose logs streamlit`
- Check health: `docker compose ps`
- Test locally: `cd dashboard && streamlit run app.py`
