# MCP Integration Quick Start Guide

**Project:** NFL Analytics - Props Betting System
**Date:** November 9, 2025
**Purpose:** Step-by-step implementation guide for top 5 MCP integrations
**Estimated Time:** 8-10 hours (1-2 working days)

---

## Prerequisites

**Requirements:**
- Node.js 18+ installed
- Docker Desktop running
- Grafana API token (create at http://localhost:3000/org/apikeys)
- GitHub personal access token (create at https://github.com/settings/tokens)
- R 4.3+ with mcptools package

**Quick Check:**
```bash
# Verify Node.js
node --version  # Should be 18+

# Verify Docker
docker ps

# Verify R
Rscript --version

# Verify existing MCP server
docker ps | grep mcp
```

---

## Integration 1: PostgreSQL MCP Pro (30 minutes)

**Goal:** Enable read/write database operations, performance analysis, and schema introspection.

### Step 1: Install PostgreSQL MCP Server

```bash
# Install official PostgreSQL MCP server
npm install -g @modelcontextprotocol/server-postgres

# Verify installation
npx @modelcontextprotocol/server-postgres --help
```

### Step 2: Create Configuration File

Create `~/.config/mcp/config.json` (or update existing):

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01"
      ],
      "env": {
        "PG_READ_ONLY": "false"
      }
    }
  }
}
```

### Step 3: Test Connection

```bash
# Test with Claude Desktop (if installed)
# Or test programmatically:

# Create test script
cat > test_postgres_mcp.py << 'EOF'
from mcp.client import Client

async def test():
    async with Client("npx", ["-y", "@modelcontextprotocol/server-postgres",
                               "postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01"]) as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        # Test query
        result = await client.call_tool("query", {
            "sql": "SELECT COUNT(*) FROM games"
        })
        print("Games count:", result)

import asyncio
asyncio.run(test())
EOF

python test_postgres_mcp.py
```

### Step 4: Integrate with Existing MCP Server

**Option A: Extend Existing Server (Recommended)**

Edit `mcp_server/server.py`:

```python
# Add write operations
@mcp.tool()
def insert_model_metadata(
    model_name: str,
    model_version: str,
    model_type: str,
    framework: str,
    description: str,
    performance_summary: dict
) -> dict:
    """
    Insert model metadata into ml.model_registry.

    This enables automated model registration after training.
    """
    query = """
    INSERT INTO ml.model_registry
    (model_name, model_version, model_type, framework, description,
     performance_summary, status, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, 'trained', NOW(), NOW())
    RETURNING model_id
    """
    params = (model_name, model_version, model_type, framework,
              description, json.dumps(performance_summary))

    return query_database(query, params)

@mcp.tool()
def get_schema_info(table_name: Optional[str] = None) -> dict:
    """
    Get schema information for database tables.

    If table_name is provided, returns columns for that table.
    Otherwise, returns all tables.
    """
    if table_name:
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """
        return query_database(query, (table_name,))
    else:
        query = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
        """
        return query_database(query)
```

**Test Integration:**
```bash
# Rebuild Docker container
docker-compose build mcp-server

# Restart with MCP profile
docker-compose --profile mcp up -d mcp-server

# Check logs
docker logs nfl-analytics-mcp
```

### Expected Outcome
- ✅ Read/write database access
- ✅ Schema introspection tools
- ✅ Model registry automation ready

---

## Integration 2: Filesystem MCP (20 minutes)

**Goal:** Programmatic access to model artifacts, logs, and data files.

### Step 1: Install Filesystem MCP Server

```bash
# Install official filesystem MCP server
npm install -g @modelcontextprotocol/server-filesystem
```

### Step 2: Update MCP Configuration

Add to `~/.config/mcp/config.json`:

```json
{
  "mcpServers": {
    "postgres": { ... },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "models",
        "logs",
        "analysis/results"
      ]
    }
  }
}
```

### Step 3: Integrate with Docker

Update `docker-compose.yml`:

```yaml
mcp-server:
  volumes:
    - ./models:/app/models:ro           # Read-only (prevent overwrites)
    - ./logs:/app/logs:rw               # Read-write
    - ./analysis/results:/app/results:ro
  environment:
    - ALLOWED_DIRS=/app/models,/app/logs,/app/results
```

### Step 4: Add Filesystem Tools

Create `mcp_server/tools/filesystem.py`:

```python
"""
Filesystem tools for model artifact management.

Security: Read-only access to models, read-write for logs.
"""
import os
import json
from pathlib import Path
from typing import List, Optional, Dict

MODELS_DIR = Path("/app/models")
LOGS_DIR = Path("/app/logs")
RESULTS_DIR = Path("/app/results")

def list_models(model_type: Optional[str] = None) -> List[Dict]:
    """
    List all trained models in models/ directory.

    Args:
        model_type: Filter by type (bayesian, cql, xgboost)

    Returns:
        List of model dictionaries with metadata
    """
    models = []

    # Scan models directory
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(('.rds', '.pkl', '.pth', '.pt')):
                filepath = Path(root) / file
                stat = filepath.stat()

                # Determine model type from path
                relative_path = filepath.relative_to(MODELS_DIR)
                detected_type = str(relative_path.parts[0]) if len(relative_path.parts) > 1 else 'unknown'

                # Filter by type if specified
                if model_type and detected_type != model_type:
                    continue

                models.append({
                    'filename': str(relative_path),
                    'type': detected_type,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': stat.st_mtime,
                    'extension': filepath.suffix
                })

    return sorted(models, key=lambda x: x['modified'], reverse=True)

def read_model_metadata(model_path: str) -> Dict:
    """
    Read metadata file for a trained model.

    Args:
        model_path: Relative path to model file

    Returns:
        Model metadata dictionary
    """
    base_path = MODELS_DIR / model_path
    metadata_path = base_path.with_suffix('.json')

    if not metadata_path.exists():
        return {'error': f'Metadata file not found: {metadata_path}'}

    with open(metadata_path, 'r') as f:
        return json.load(f)

def read_training_log(model_type: str, limit: int = 100) -> List[str]:
    """
    Read recent training logs for a model type.

    Args:
        model_type: Model type (bayesian, cql, xgboost)
        limit: Number of lines to return

    Returns:
        List of log lines
    """
    log_path = LOGS_DIR / f"{model_type}_training.log"

    if not log_path.exists():
        return [f"Log file not found: {log_path}"]

    with open(log_path, 'r') as f:
        lines = f.readlines()

    return lines[-limit:]
```

Add to `mcp_server/server.py`:

```python
from mcp_server.tools.filesystem import list_models, read_model_metadata, read_training_log

@mcp.tool()
def list_trained_models(model_type: Optional[str] = None) -> list:
    """List all trained models in models/ directory."""
    return list_models(model_type)

@mcp.tool()
def get_model_metadata(model_path: str) -> dict:
    """Get metadata for a specific model."""
    return read_model_metadata(model_path)

@mcp.tool()
def get_training_logs(model_type: str, limit: int = 100) -> list:
    """Get recent training logs."""
    return read_training_log(model_type, limit)
```

### Expected Outcome
- ✅ AI-assisted model artifact inspection
- ✅ Automated model registry sync (227 pending models)
- ✅ Training log analysis

---

## Integration 3: GitHub MCP (25 minutes)

**Goal:** Automated repository management, issue tracking, and documentation updates.

### Step 1: Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` (full control of private repositories)
   - `workflow` (update GitHub Action workflows)
4. Generate token and save securely

### Step 2: Install GitHub MCP Server

```bash
# Install official GitHub MCP server
npm install -g @modelcontextprotocol/server-github

# Store token in .env
echo "GITHUB_TOKEN=ghp_your_token_here" >> .env
```

### Step 3: Update MCP Configuration

Add to `~/.config/mcp/config.json`:

```json
{
  "mcpServers": {
    "postgres": { ... },
    "filesystem": { ... },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Step 4: Integrate with Existing MCP Server

Create `mcp_server/tools/github.py`:

```python
"""
GitHub integration tools for repository management.

Requires GITHUB_TOKEN environment variable.
"""
import os
import requests
from typing import Dict, List, Optional

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "raold"  # Update with your GitHub username
REPO_NAME = "nfl-analytics"

def _headers():
    """Get GitHub API headers with authentication."""
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

def create_issue(title: str, body: str, labels: Optional[List[str]] = None) -> Dict:
    """
    Create a GitHub issue.

    Use cases:
    - Training failure notifications
    - Model performance degradation alerts
    - TODO items from AI conversations
    """
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    data = {
        "title": title,
        "body": body,
        "labels": labels or []
    }

    response = requests.post(url, json=data, headers=_headers())
    return response.json()

def create_pull_request(title: str, body: str, head: str, base: str = "main") -> Dict:
    """
    Create a pull request.

    Use cases:
    - Automated model updates
    - Documentation updates
    - Phase completion PRs
    """
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    data = {
        "title": title,
        "body": body,
        "head": head,
        "base": base
    }

    response = requests.post(url, json=data, headers=_headers())
    return response.json()

def search_code(query: str) -> List[Dict]:
    """
    Search code across repository.

    Use cases:
    - Find usage of specific functions
    - Identify deprecated patterns
    - Locate feature engineering code
    """
    url = f"{GITHUB_API}/search/code"
    params = {
        "q": f"{query} repo:{REPO_OWNER}/{REPO_NAME}"
    }

    response = requests.get(url, params=params, headers=_headers())
    return response.json().get('items', [])
```

Add to `mcp_server/server.py`:

```python
from mcp_server.tools.github import create_issue, create_pull_request, search_code

@mcp.tool()
def create_github_issue(title: str, body: str, labels: Optional[str] = None) -> dict:
    """
    Create a GitHub issue for tracking.

    Example:
        create_github_issue(
            title="BNN training failed: CUDA OOM",
            body="BNN v3.2 training failed with CUDA out of memory error...",
            labels="bug,training"
        )
    """
    label_list = labels.split(",") if labels else []
    return create_issue(title, body, label_list)

@mcp.tool()
def create_github_pr(title: str, body: str, branch: str) -> dict:
    """Create a pull request."""
    return create_pull_request(title, body, branch)

@mcp.tool()
def search_repo_code(query: str) -> list:
    """Search code across repository."""
    return search_code(query)
```

### Step 5: Update docker-compose.yml

```yaml
mcp-server:
  environment:
    - GITHUB_TOKEN=${GITHUB_TOKEN}
```

### Expected Outcome
- ✅ Automated issue creation for training failures
- ✅ PR automation for phase completions
- ✅ Code search for feature usage

---

## Integration 4: Grafana MCP (35 minutes)

**Goal:** Natural language metric queries, dashboard discovery, and alerting.

### Step 1: Create Grafana API Token

1. Navigate to http://localhost:3000
2. Login (admin / ${GRAFANA_PASSWORD})
3. Go to Administration → Service Accounts
4. Create service account: "MCP Integration"
5. Add token with "Viewer" role
6. Copy token and save securely

### Step 2: Install Grafana MCP Server

```bash
# Install official Grafana MCP server
npm install -g @grafana/mcp-grafana

# Store token in .env
echo "GRAFANA_TOKEN=glsa_your_token_here" >> .env
```

### Step 3: Update MCP Configuration

Add to `~/.config/mcp/config.json`:

```json
{
  "mcpServers": {
    "postgres": { ... },
    "filesystem": { ... },
    "github": { ... },
    "grafana": {
      "command": "npx",
      "args": ["-y", "@grafana/mcp-grafana"],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_TOKEN": "${GRAFANA_TOKEN}"
      }
    }
  }
}
```

### Step 4: Integrate with Existing MCP Server

Create `mcp_server/tools/grafana.py`:

```python
"""
Grafana integration tools for monitoring and alerting.

Requires GRAFANA_URL and GRAFANA_TOKEN environment variables.
"""
import os
import requests
from typing import Dict, List, Optional

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN")

def _headers():
    """Get Grafana API headers with authentication."""
    return {
        "Authorization": f"Bearer {GRAFANA_TOKEN}",
        "Content-Type": "application/json"
    }

def query_prometheus(query: str, time_range: str = "1h") -> Dict:
    """
    Query Prometheus metrics via Grafana.

    Use cases:
    - "Show me CPU usage for last hour"
    - "What's the database connection pool utilization?"
    - "How many model predictions in the last 24 hours?"
    """
    url = f"{GRAFANA_URL}/api/datasources/proxy/1/api/v1/query_range"

    # Parse time range (simplified)
    if time_range.endswith('h'):
        hours = int(time_range[:-1])
        step = f"{hours * 60}s"
    else:
        step = "60s"

    params = {
        "query": query,
        "start": f"now-{time_range}",
        "end": "now",
        "step": step
    }

    response = requests.get(url, params=params, headers=_headers())
    return response.json()

def search_dashboards(query: Optional[str] = None) -> List[Dict]:
    """
    Search Grafana dashboards.

    Use cases:
    - "Find dashboards related to model performance"
    - "Show all dashboards with 'NFL' in the title"
    """
    url = f"{GRAFANA_URL}/api/search"
    params = {"type": "dash-db"}

    if query:
        params["query"] = query

    response = requests.get(url, params=params, headers=_headers())
    return response.json()

def get_alert_rules() -> List[Dict]:
    """
    Get all alert rules.

    Use cases:
    - "What alerts are currently firing?"
    - "Show me all configured alerts"
    """
    url = f"{GRAFANA_URL}/api/alert-rules"
    response = requests.get(url, headers=_headers())
    return response.json()
```

Add to `mcp_server/server.py`:

```python
from mcp_server.tools.grafana import query_prometheus, search_dashboards, get_alert_rules

@mcp.tool()
def query_metrics(query: str, time_range: str = "1h") -> dict:
    """
    Query Prometheus metrics via Grafana.

    Examples:
        query_metrics("rate(http_requests_total[5m])", "6h")
        query_metrics("node_cpu_seconds_total", "24h")
    """
    return query_prometheus(query, time_range)

@mcp.tool()
def find_dashboards(search_query: Optional[str] = None) -> list:
    """Search Grafana dashboards."""
    return search_dashboards(search_query)

@mcp.tool()
def list_alerts() -> list:
    """Get all alert rules."""
    return get_alert_rules()
```

### Step 5: Update docker-compose.yml

```yaml
mcp-server:
  environment:
    - GRAFANA_URL=http://nfl-analytics-grafana:3000
    - GRAFANA_TOKEN=${GRAFANA_TOKEN}
  depends_on:
    - grafana
```

### Expected Outcome
- ✅ Natural language metric queries
- ✅ AI-powered dashboard discovery
- ✅ Alert investigation assistance

---

## Integration 5: R MCP (mcptools) (40 minutes)

**Goal:** Interactive R debugging, Bayesian model diagnostics, and statistical computing.

### Step 1: Install mcptools Package

```r
# Install mcptools from CRAN
install.packages("mcptools")

# Or from GitHub for latest version
# install.packages("remotes")
# remotes::install_github("posit-dev/mcptools")
```

### Step 2: Configure R MCP Server

Add to `~/.config/mcp/config.json`:

```json
{
  "mcpServers": {
    "postgres": { ... },
    "filesystem": { ... },
    "github": { ... },
    "grafana": { ... },
    "r-mcptools": {
      "command": "Rscript",
      "args": ["-e", "mcptools::mcp_server()"]
    }
  }
}
```

### Step 3: Start R Session with MCP

```r
# In R console or RStudio
library(mcptools)

# Enable MCP session (allows AI to access variables)
mcptools::mcp_session()

# Load some data for testing
library(readRDS)
model <- readRDS("models/bayesian/simplified_random_walk_v2.rds")
```

### Step 4: Test R MCP Tools

**Example AI Queries:**
- "Check R-hat for all parameters in the model object"
- "Show summary of posterior distributions"
- "Plot trace plots for the first 4 chains"
- "What's the effective sample size for sigma?"

**R Functions Available:**
```r
# Check object in environment
ls()

# Get help
?brms::brm

# Check diagnostics
brms::rhat(model)
brms::neff_ratio(model)

# Plot
plot(model)
```

### Step 5: Create R Helper Functions

Create `R/mcp_helpers.R`:

```r
#' MCP Helper Functions for Bayesian Model Diagnostics
#'
#' Functions designed for easy MCP/AI access to model diagnostics.

#' Check MCMC convergence for a brms model
#' @param model A brms model object
#' @return List with convergence diagnostics
check_convergence <- function(model) {
  list(
    rhat_max = max(brms::rhat(model), na.rm = TRUE),
    rhat_problem = sum(brms::rhat(model) > 1.01, na.rm = TRUE),
    ess_bulk_min = min(brms::neff_ratio(model, "bulk"), na.rm = TRUE),
    ess_tail_min = min(brms::neff_ratio(model, "tail"), na.rm = TRUE),
    divergences = sum(brms::nuts_params(model)$divergent__),
    recommendation = ifelse(
      max(brms::rhat(model), na.rm = TRUE) < 1.01 &&
      min(brms::neff_ratio(model), na.rm = TRUE) > 0.1 &&
      sum(brms::nuts_params(model)$divergent__) < 10,
      "✅ Model converged successfully",
      "⚠️ Convergence issues detected - review diagnostics"
    )
  )
}

#' Get quick model summary for MCP
#' @param model A brms model object
#' @return Formatted summary text
model_summary_for_mcp <- function(model) {
  summary_text <- capture.output(summary(model))
  paste(summary_text, collapse = "\n")
}

#' List all trained Bayesian models
#' @return Data frame of model files with metadata
list_bayesian_models <- function() {
  model_files <- list.files("models/bayesian", pattern = "\\.rds$", full.names = TRUE)

  data.frame(
    filename = basename(model_files),
    size_mb = file.size(model_files) / 1024^2,
    modified = file.mtime(model_files),
    stringsAsFactors = FALSE
  )
}

# Source on package load
if (interactive()) {
  message("MCP helper functions loaded")
  message("Available: check_convergence(), model_summary_for_mcp(), list_bayesian_models()")
}
```

Source in R session:
```r
source("R/mcp_helpers.R")
```

### Expected Outcome
- ✅ Interactive R debugging from AI conversations
- ✅ Bayesian model diagnostics automation
- ✅ Faster convergence issue detection

---

## Verification & Testing

### Test All Integrations

Create `test_mcp_integrations.py`:

```python
"""
Test script for all MCP integrations.

Run: python test_mcp_integrations.py
"""
import asyncio
from mcp.client import Client

async def test_postgres():
    print("\n=== Testing PostgreSQL MCP ===")
    async with Client("npx", ["-y", "@modelcontextprotocol/server-postgres",
                               "postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01"]) as client:
        result = await client.call_tool("query", {"sql": "SELECT COUNT(*) FROM games"})
        print(f"✅ PostgreSQL: {result}")

async def test_filesystem():
    print("\n=== Testing Filesystem MCP ===")
    async with Client("npx", ["-y", "@modelcontextprotocol/server-filesystem",
                               "models"]) as client:
        result = await client.call_tool("list_directory", {"path": "models"})
        print(f"✅ Filesystem: Found {len(result)} items")

async def test_github():
    print("\n=== Testing GitHub MCP ===")
    import os
    token = os.getenv("GITHUB_TOKEN")
    async with Client("npx", ["-y", "@modelcontextprotocol/server-github"]) as client:
        result = await client.call_tool("search_repositories", {"query": "nfl-analytics"})
        print(f"✅ GitHub: {result}")

async def test_grafana():
    print("\n=== Testing Grafana MCP ===")
    async with Client("npx", ["-y", "@grafana/mcp-grafana"]) as client:
        result = await client.call_tool("search_dashboards", {})
        print(f"✅ Grafana: Found {len(result)} dashboards")

async def main():
    print("Testing MCP Integrations...")

    try:
        await test_postgres()
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")

    try:
        await test_filesystem()
    except Exception as e:
        print(f"❌ Filesystem test failed: {e}")

    try:
        await test_github()
    except Exception as e:
        print(f"❌ GitHub test failed: {e}")

    try:
        await test_grafana()
    except Exception as e:
        print(f"❌ Grafana test failed: {e}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
```

Run tests:
```bash
python test_mcp_integrations.py
```

---

## Final Checklist

**Before Production:**
- [ ] All MCP servers tested and working
- [ ] Credentials stored in `.env` (not committed to git)
- [ ] Docker volumes mounted correctly
- [ ] Grafana API token created with viewer role
- [ ] GitHub personal access token with repo scope
- [ ] R mcptools package installed
- [ ] Test queries executed successfully
- [ ] Documentation updated (CLAUDE.md, README.md)
- [ ] MCP server logs verified (no errors)

**Configuration Files:**
- [ ] `~/.config/mcp/config.json` - Client configuration
- [ ] `.env` - Environment variables (GITHUB_TOKEN, GRAFANA_TOKEN)
- [ ] `docker-compose.yml` - Updated with new volumes and env vars
- [ ] `mcp_server/server.py` - Enhanced with new tools

**Expected Outcome:**
- ✅ 5 MCP integrations deployed and tested
- ✅ AI-assisted development workflow enabled
- ✅ Automated model registry, documentation, monitoring

---

## Troubleshooting

### Common Issues

**Issue 1: MCP Server Won't Start**
```bash
# Check Docker logs
docker logs nfl-analytics-mcp

# Check for missing environment variables
docker-compose config | grep -A5 mcp-server

# Restart with verbose logging
docker-compose --profile mcp up mcp-server
```

**Issue 2: PostgreSQL Connection Refused**
```bash
# Verify database is running
docker ps | grep pg-1

# Test connection manually
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 -c "SELECT 1"

# Check docker network
docker network inspect nfl-analytics
```

**Issue 3: GitHub Token Invalid**
```bash
# Test token manually
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Regenerate token if expired
```

**Issue 4: Grafana API Token Permissions**
```bash
# Verify token has correct permissions
curl -H "Authorization: Bearer $GRAFANA_TOKEN" http://localhost:3000/api/org

# Create new token with Viewer role
```

**Issue 5: R MCP Session Not Found**
```r
# Restart R session
# Reinstall mcptools
remove.packages("mcptools")
install.packages("mcptools")

# Check for conflicts
library(mcptools)
mcptools::mcp_session()
```

---

## Next Steps

**After Integration:**
1. **Model Registry Sync:** Use filesystem MCP to register 227 pending models
2. **Documentation Automation:** Use GitHub MCP to update CLAUDE.md after training
3. **Monitoring Setup:** Create Grafana dashboards for MCP server metrics
4. **R Workflow:** Test Bayesian model diagnostics with R MCP
5. **Team Onboarding:** Share MCP tools with team members (if applicable)

**Performance Optimization:**
1. Enable MCP response caching (Redis)
2. Monitor MCP server resource usage
3. Optimize slow database queries
4. Set up alerting for MCP failures

**Documentation:**
1. Update CLAUDE.md with MCP tools section
2. Create MCP usage examples in docs/
3. Document common AI queries for each integration
4. Add MCP troubleshooting guide

---

**Estimated Total Time:** 8-10 hours (1-2 working days)
**Expected ROI:** 20+ hours/month saved in manual operations
**Payback Period:** 1 month

**Questions?** See `docs/MCP_INTEGRATION_AUDIT.md` for comprehensive details.
