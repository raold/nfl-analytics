#!/usr/bin/env bash
#
# Project Restructuring Script
# Moves files to new organized structure
#
# Usage: bash scripts/reorganize_project.sh [--dry-run]
#

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be moved"
    echo ""
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

move_file() {
    local src="$1"
    local dest="$2"
    
    if [[ ! -f "$src" ]]; then
        log_warning "Source file not found: $src"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would move: $src → $dest"
        return 0
    fi
    
    # Create destination directory if needed
    mkdir -p "$(dirname "$dest")"
    
    # Move file
    mv "$src" "$dest"
    log_success "Moved: $src → $dest"
}

create_dir() {
    local dir="$1"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would create directory: $dir"
        return 0
    fi
    
    mkdir -p "$dir"
    log_success "Created directory: $dir"
}

echo "🚀 Starting Project Restructuring"
echo "================================"
echo ""

# Phase 1: Organize Documentation
log_info "Phase 1: Organizing Documentation"
echo ""

# Move reports to docs/reports/
move_file "2025_SEASON_DATA_INGESTION.md" "docs/reports/2025_season_data_ingestion.md"
move_file "BACKFILL_COMPLETE_RESULTS.md" "docs/reports/backfill_complete_results.md"
move_file "DATABASE_AUDIT_REPORT.md" "docs/reports/database_audit_report.md"
move_file "DATABASE_BACKFILL_EXECUTION_SUMMARY.md" "docs/reports/database_backfill_execution_summary.md"
move_file "DATABASE_GAP_ANALYSIS_AND_BACKFILL_PLAN.md" "docs/reports/database_gap_analysis_and_backfill_plan.md"
move_file "DATABASE_PRODUCTION_CERT.md" "docs/reports/database_production_cert.md"
move_file "FEATURE_ENGINEERING_COMPLETE.md" "docs/reports/feature_engineering_complete.md"
move_file "CODEBASE_AUDIT_2025.md" "docs/reports/codebase_audit_2025.md"

# Move agent context docs
move_file "AGENTS.md" "docs/agent_context/AGENTS.md"
move_file "CLAUDE.md" "docs/agent_context/CLAUDE.md"
move_file "GEMINI.md" "docs/agent_context/GEMINI.md"

# Keep project planning docs at root for now (will move after review)
# PROJECT_RESTRUCTURE_PLAN.md stays at root

echo ""
log_success "Phase 1 Complete: Documentation organized"
echo ""

# Phase 2: Organize Data Files
log_info "Phase 2: Organizing Data Files"
echo ""

# Create data subdirectories
create_dir "data/raw/nflverse"
create_dir "data/raw/odds"
create_dir "data/raw/weather"
create_dir "data/processed/features"
create_dir "data/processed/predictions"
create_dir "data/staging"

# Move specific data files
if [[ -f "data/rl_logged.csv" ]]; then
    move_file "data/rl_logged.csv" "data/processed/rl_logged.csv"
fi

if [[ -f "data/bets.csv" ]]; then
    move_file "data/bets.csv" "data/processed/bets.csv"
fi

if [[ -f "data/scenarios.csv" ]]; then
    move_file "data/scenarios.csv" "data/processed/scenarios.csv"
fi

# Move analysis feature files
if [[ -d "analysis/features" ]]; then
    log_info "Moving feature files..."
    find analysis/features -name "*.csv" -type f | while read -r file; do
        filename=$(basename "$file")
        move_file "$file" "data/processed/features/$filename"
    done
fi

echo ""
log_success "Phase 2 Complete: Data files organized"
echo ""

# Phase 3: Organize Scripts
log_info "Phase 3: Organizing Scripts"
echo ""

# Create script subdirectories
create_dir "scripts/dev"
create_dir "scripts/deploy"
create_dir "scripts/maintenance"
create_dir "scripts/analysis"

# Move development scripts
move_file "scripts/dev_setup.sh" "scripts/dev/setup_env.sh"
move_file "scripts/init_dev.sh" "scripts/dev/init_dev.sh"
move_file "scripts/setup_testing.sh" "scripts/dev/setup_testing.sh"
move_file "scripts/install_pytorch.sh" "scripts/dev/install_pytorch.sh"

# Move analysis scripts
move_file "scripts/check_2025_data.R" "scripts/analysis/check_2025_data.R"
move_file "scripts/run_reports.sh" "scripts/analysis/run_reports.sh"
move_file "scripts/make_time_decay_weights.R" "scripts/analysis/make_time_decay_weights.R"

# Move ingestion scripts to new ETL location
log_info "Moving ingestion scripts to R/ingestion/..."
create_dir "R/ingestion"
move_file "data/ingest_schedules.R" "R/ingestion/ingest_schedules.R"
move_file "data/ingest_pbp.R" "R/ingestion/ingest_pbp.R"
move_file "data/ingest_injuries.R" "R/ingestion/ingest_injuries.R"
move_file "scripts/ingest_2025_season.R" "R/ingestion/ingest_2025_season.R"

# Move feature scripts
log_info "Moving R feature scripts..."
create_dir "R/features"
move_file "data/features_epa.R" "R/features/features_epa.R"
move_file "data/baseline_spread.R" "R/features/baseline_spread.R"

echo ""
log_success "Phase 3 Complete: Scripts organized"
echo ""

# Phase 4: Organize Database Files
log_info "Phase 4: Organizing Database Files"
echo ""

create_dir "db/migrations"
create_dir "db/views"
create_dir "db/functions"
create_dir "db/seeds"

# Move migrations (already in db/)
log_info "Database migrations already in place"

echo ""
log_success "Phase 4 Complete: Database files organized"
echo ""

# Phase 5: Organize Python Package
log_info "Phase 5: Organizing Python Package"
echo ""

# Move odds ingestion to ETL
if [[ -f "py/ingest_odds_history.py" ]]; then
    log_info "Note: py/ingest_odds_history.py will be refactored into etl/extract/odds_api.py"
    # Keep original for now, will refactor after ETL framework complete
fi

# Move weather ingestion to ETL
if [[ -f "py/weather_meteostat.py" ]]; then
    log_info "Note: py/weather_meteostat.py will be refactored into etl/extract/weather.py"
    # Keep original for now
fi

echo ""
log_success "Phase 5 Complete: Python package structure ready"
echo ""

# Phase 6: Infrastructure
log_info "Phase 6: Organizing Infrastructure"
echo ""

create_dir "infrastructure/docker"

# Move Docker files
if [[ -f "docker-compose.yaml" ]]; then
    move_file "docker-compose.yaml" "infrastructure/docker/docker-compose.yaml"
fi

if [[ -d "docker" ]]; then
    log_info "Moving docker/ contents to infrastructure/docker/..."
    find docker -type f | while read -r file; do
        filename=$(basename "$file")
        move_file "$file" "infrastructure/docker/$filename"
    done
    if [[ "$DRY_RUN" == "false" ]]; then
        rmdir docker 2>/dev/null || true
    fi
fi

echo ""
log_success "Phase 6 Complete: Infrastructure organized"
echo ""

# Phase 7: Clean up root directory
log_info "Phase 7: Cleaning root directory"
echo ""

# Remove temporary files
if [[ "$DRY_RUN" == "false" ]]; then
    rm -f 1.R 2>/dev/null || true
    rm -f best_VA_sportsbooks.png 2>/dev/null || true
    rm -f naurtf.pdf 2>/dev/null || true
    rm -f policy.json 2>/dev/null || true  # Move to models/ if needed
    log_success "Removed temporary files from root"
fi

echo ""
log_success "Phase 7 Complete: Root directory cleaned"
echo ""

# Generate summary
echo ""
echo "================================"
echo "📊 Restructuring Summary"
echo "================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    log_warning "This was a DRY RUN - no files were actually moved"
    log_info "Run without --dry-run to apply changes"
else
    log_success "Project successfully restructured!"
    echo ""
    echo "📁 New Structure Created:"
    echo "  ├── docs/           (all documentation)"
    echo "  ├── etl/            (data pipelines)"
    echo "  ├── py/             (Python package)"
    echo "  ├── R/              (R scripts)"
    echo "  ├── db/             (database schema)"
    echo "  ├── data/           (organized data storage)"
    echo "  ├── scripts/        (operational scripts)"
    echo "  ├── infrastructure/ (Docker, deployment)"
    echo "  └── tests/          (test suite)"
    echo ""
    log_info "Next steps:"
    echo "  1. Review new structure: tree -L 2 docs/ etl/ scripts/"
    echo "  2. Update import statements in code"
    echo "  3. Update GitHub Actions workflows"
    echo "  4. Update README.md with new structure"
    echo "  5. Test all scripts in new locations"
fi

echo ""
