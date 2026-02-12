#!/bin/bash
# Dynamic NFL Data Ingestion Script
# Auto-detects current season or accepts season override

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
SEASON=""
FORCE=false
SKIP_BACKUP=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --season)
      SEASON="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --skip-backup)
      SKIP_BACKUP=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --season YEAR     Specify season to ingest (default: auto-detect)"
      echo "  --force          Skip confirmation prompts"
      echo "  --skip-backup    Skip creating backup before ingestion"
      echo "  --help           Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                    # Auto-detect current season"
      echo "  $0 --season 2024      # Ingest 2024 season"
      echo "  $0 --season 2025 --force  # Ingest 2025 without prompts"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Function to get current NFL season
get_current_season() {
  local month=$(date +%m)
  local year=$(date +%Y)

  if [ $month -ge 9 ]; then
    # September or later - current year's season
    echo $year
  elif [ $month -le 2 ]; then
    # January/February - previous year's season
    echo $((year - 1))
  else
    # March-August - offseason, use previous season
    echo $((year - 1))
  fi
}

# Determine season to process
if [ -z "$SEASON" ]; then
  SEASON=$(get_current_season)
  echo -e "${BLUE}Auto-detected current NFL season: ${SEASON}${NC}"
else
  echo -e "${BLUE}Processing specified season: ${SEASON}${NC}"
fi

# Validate season range
if [ $SEASON -lt 1999 ] || [ $SEASON -gt 2050 ]; then
  echo -e "${RED}Error: Invalid season ${SEASON}. Must be between 1999 and 2050.${NC}"
  exit 1
fi

# Show what will happen
echo ""
echo -e "${YELLOW}=== NFL Data Ingestion Plan ===${NC}"
echo "Season to ingest: $SEASON"
echo "Current date: $(date '+%Y-%m-%d')"
echo ""
echo "This will:"
echo "  1. Delete existing data for season $SEASON"
echo "  2. Download fresh data from nflverse"
echo "  3. Update games, plays, and rosters tables"
echo "  4. Calculate derived statistics"
echo "  5. Refresh materialized views"
echo ""

# Get current data counts
if command -v psql &> /dev/null; then
  echo -e "${BLUE}Current database state:${NC}"
  psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -t -c "
    SELECT
      'Season $SEASON: ' ||
      COALESCE((SELECT COUNT(*) FROM games WHERE season = $SEASON), 0) || ' games, ' ||
      COALESCE((SELECT COUNT(*) FROM plays WHERE game_id LIKE '${SEASON}_%'), 0) || ' plays'
  " 2>/dev/null || echo "  (Unable to connect to database)"
  echo ""
fi

# Confirm unless --force
if [ "$FORCE" != true ]; then
  read -p "Proceed with ingestion? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Ingestion cancelled.${NC}"
    exit 0
  fi
fi

# Create backup unless --skip-backup
if [ "$SKIP_BACKUP" != true ]; then
  echo -e "${BLUE}Creating backup before ingestion...${NC}"
  if [ -f "scripts/maintenance/backup.sh" ]; then
    bash scripts/maintenance/backup.sh --quiet || echo -e "${YELLOW}Warning: Backup failed${NC}"
  else
    echo -e "${YELLOW}Warning: Backup script not found${NC}"
  fi
  echo ""
fi

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
  echo -e "${RED}Error: R is not installed or not in PATH${NC}"
  exit 1
fi

# Check if required R script exists
SCRIPT_PATH="R/ingestion/ingest_current_season.R"
if [ ! -f "$SCRIPT_PATH" ]; then
  # Fallback to legacy script for specific season
  if [ $SEASON -eq 2025 ] && [ -f "R/ingestion/ingest_2025_season_safe.R" ]; then
    echo -e "${YELLOW}Using legacy 2025 ingestion script${NC}"
    SCRIPT_PATH="R/ingestion/ingest_2025_season_safe.R"
  elif [ -f "R/ingestion/ingest_2025_season.R" ]; then
    echo -e "${YELLOW}Using basic ingestion script (no error handling)${NC}"
    SCRIPT_PATH="R/ingestion/ingest_2025_season.R"
  else
    echo -e "${RED}Error: No ingestion script found${NC}"
    exit 1
  fi
fi

# Run ingestion
echo -e "${GREEN}Starting ingestion...${NC}"
echo ""

# Set up error handling for R script
export R_LOG_DIR="logs/r_etl"
mkdir -p "$R_LOG_DIR"

# Run the R script with season parameter if using dynamic version
if [[ "$SCRIPT_PATH" == *"ingest_current_season.R" ]]; then
  Rscript "$SCRIPT_PATH" --season "$SEASON"
else
  # Legacy scripts don't support parameters
  Rscript "$SCRIPT_PATH"
fi

EXIT_CODE=$?

# Check result
if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo -e "${GREEN}✅ Ingestion completed successfully!${NC}"

  # Show final counts
  if command -v psql &> /dev/null; then
    echo ""
    echo -e "${BLUE}Final database state:${NC}"
    psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -t -c "
      SELECT
        'Season $SEASON: ' ||
        COALESCE((SELECT COUNT(*) FROM games WHERE season = $SEASON), 0) || ' games (' ||
        COALESCE((SELECT COUNT(*) FROM games WHERE season = $SEASON AND home_score IS NOT NULL), 0) || ' completed), ' ||
        COALESCE((SELECT COUNT(*) FROM plays WHERE game_id LIKE '${SEASON}_%'), 0) || ' plays'
    " 2>/dev/null
  fi

  # Check for alerts
  if [ -f "logs/r_etl/alerts.json" ]; then
    UNREAD_ALERTS=$(python3 -c "import json; alerts=json.load(open('logs/r_etl/alerts.json')); print(sum(1 for a in alerts if a.get('status')=='unread'))" 2>/dev/null || echo "0")
    if [ "$UNREAD_ALERTS" -gt 0 ]; then
      echo ""
      echo -e "${YELLOW}⚠️  $UNREAD_ALERTS unread alerts in logs/r_etl/alerts.json${NC}"
    fi
  fi
else
  echo ""
  echo -e "${RED}❌ Ingestion failed with exit code $EXIT_CODE${NC}"
  echo "Check logs in: $R_LOG_DIR"
  exit $EXIT_CODE
fi

# Suggest next steps
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Generate features: make features"
echo "  2. Run backtests: make backtest"
echo "  3. Check data quality: make validate"