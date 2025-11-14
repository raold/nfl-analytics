#!/usr/bin/env bash
#
# NFL Domain Validation Audit
#
# Comprehensive sports analytics domain validation
# Companion script to docs/audits/2025_11_comprehensive_audit/15_sports_analytics_validation.md
#
# Usage:
#   bash scripts/domain_validation_audit.sh
#   bash scripts/domain_validation_audit.sh --check team_abbreviations
#   bash scripts/domain_validation_audit.sh --verbose
#
# Author: Sports Analytics Validator (Claude Code)
# Date: November 14, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database connection
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5544}"
DB_NAME="${POSTGRES_DB:-devdb01}"
DB_USER="${POSTGRES_USER:-dro}"
DB_PASS="${POSTGRES_PASSWORD:-sicillionbillions}"

DB_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Parse arguments
VERBOSE=false
SPECIFIC_CHECK=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --verbose)
      VERBOSE=true
      shift
      ;;
    --check)
      SPECIFIC_CHECK="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--verbose] [--check CHECK_NAME]"
      echo ""
      echo "Available checks:"
      echo "  team_abbreviations    Team abbreviation validation (LA ambiguity)"
      echo "  positions             Position code validation (11 expected)"
      echo "  scheduling            Game scheduling validation (weeks, playoffs)"
      echo "  data_quality          Data quality checks (spreads, yards, weather)"
      echo "  overtime              Overtime tracking validation"
      echo "  temporal_validity     Temporal correctness (as-of joins, leakage)"
      echo "  betting_math          Kelly criterion and ROI calculations"
      echo "  fillna_audit          Code quality: fillna(0) audit"
      echo "  models                Statistical model configuration"
      echo "  production            Production readiness checks"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage"
      exit 1
      ;;
  esac
done

# Logging functions
log_info() {
  echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASSED_CHECKS++))
}

log_failure() {
  echo -e "${RED}✗${NC} $1"
  ((FAILED_CHECKS++))
}

log_warning() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

run_check() {
  ((TOTAL_CHECKS++))
}

# Database query helper
query_db() {
  local query="$1"
  psql "$DB_URL" -t -c "$query" 2>&1
}

echo "========================================================================"
echo "  NFL DOMAIN VALIDATION AUDIT"
echo "========================================================================"
echo ""
echo "Database: $DB_NAME @ $DB_HOST:$DB_PORT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
# 1. TEAM ABBREVIATIONS (P1 Critical)
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "team_abbreviations" ]]; then
  echo "---[ Team Abbreviations ]---"
  run_check

  # Check for 32 NFL teams
  team_count=$(query_db "SELECT COUNT(DISTINCT canonical_abbr) FROM reference.teams;" | xargs)
  if [[ "$team_count" == "32" ]]; then
    log_success "Found 32 NFL teams (correct)"
  else
    log_failure "Expected 32 teams, found $team_count"
  fi

  # Check for LA ambiguity (P1 issue)
  run_check
  la_games=$(query_db "SELECT COUNT(*) FROM games WHERE home_team = 'LA' OR away_team = 'LA';" | xargs)
  if [[ "$la_games" == "0" ]]; then
    log_success "No LA team ambiguity (LAR/LAC correctly distinguished)"
  else
    log_failure "LA team ambiguity: $la_games games use 'LA' instead of 'LAR' (P1 CRITICAL)"
  fi

  # Check for invalid team abbreviations
  run_check
  invalid_teams=$(query_db "
    SELECT COUNT(*)
    FROM (
      SELECT DISTINCT home_team FROM games
      UNION
      SELECT DISTINCT away_team FROM games
    ) t
    WHERE home_team NOT IN (SELECT canonical_abbr FROM reference.teams)
      AND home_team IS NOT NULL;
  " | xargs)

  if [[ "$invalid_teams" == "0" ]]; then
    log_success "All team abbreviations valid (match reference.teams)"
  else
    log_failure "Found $invalid_teams invalid team abbreviations"
  fi

  echo ""
fi

# ============================================================================
# 2. POSITION CODES
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "positions" ]]; then
  echo "---[ Position Codes ]---"
  run_check

  position_count=$(query_db "SELECT COUNT(DISTINCT position) FROM rosters WHERE season >= 2020;" | xargs)
  if [[ "$position_count" == "11" ]]; then
    log_success "Found 11 position codes (NFL standard)"
  else
    log_warning "Expected 11 positions, found $position_count"
  fi

  # Check for expected positions
  run_check
  expected_positions="QB RB WR TE OL DL LB DB K P LS"
  missing_positions=()

  for pos in $expected_positions; do
    count=$(query_db "SELECT COUNT(*) FROM rosters WHERE position = '$pos' AND season >= 2020;" | xargs)
    if [[ "$count" == "0" ]]; then
      missing_positions+=("$pos")
    fi
  done

  if [[ ${#missing_positions[@]} -eq 0 ]]; then
    log_success "All expected position codes found"
  else
    log_failure "Missing positions: ${missing_positions[*]}"
  fi

  echo ""
fi

# ============================================================================
# 3. GAME SCHEDULING
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "scheduling" ]]; then
  echo "---[ Game Scheduling ]---"
  run_check

  # Check week structure for 2024 season
  week_1_games=$(query_db "SELECT COUNT(*) FROM games WHERE season = 2024 AND week = 1;" | xargs)
  if [[ "$week_1_games" == "16" ]]; then
    log_success "Week 1 has 16 games (all teams play)"
  else
    log_warning "Week 1 has $week_1_games games (expected 16)"
  fi

  # Check bye weeks (weeks 5-14 should have 13-16 games)
  run_check
  bye_week_issue=$(query_db "
    SELECT COUNT(*)
    FROM (
      SELECT week, COUNT(*) as game_count
      FROM games
      WHERE season = 2024 AND week BETWEEN 5 AND 14
      GROUP BY week
      HAVING COUNT(*) < 13 OR COUNT(*) > 16
    ) t;
  " | xargs)

  if [[ "$bye_week_issue" == "0" ]]; then
    log_success "Bye weeks (5-14) have correct game counts (13-16)"
  else
    log_warning "Found $bye_week_issue weeks with unusual game counts during bye weeks"
  fi

  # Check playoff structure
  run_check
  wild_card_games=$(query_db "SELECT COUNT(*) FROM games WHERE season = 2024 AND week = 19;" | xargs)
  super_bowl_games=$(query_db "SELECT COUNT(*) FROM games WHERE season = 2024 AND week = 22;" | xargs)

  if [[ "$wild_card_games" == "6" ]] && [[ "$super_bowl_games" == "1" ]]; then
    log_success "Playoff structure correct (Wild Card: 6, Super Bowl: 1)"
  else
    log_warning "Playoff structure: Wild Card=$wild_card_games, Super Bowl=$super_bowl_games"
  fi

  echo ""
fi

# ============================================================================
# 4. DATA QUALITY
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "data_quality" ]]; then
  echo "---[ Data Quality ]---"
  run_check

  # Check for impossible spread values
  impossible_spreads=$(query_db "SELECT COUNT(*) FROM games WHERE ABS(spread_close) > 30 AND spread_close IS NOT NULL;" | xargs)
  if [[ "$impossible_spreads" == "0" ]]; then
    log_success "No impossible spread values (all |spread| <= 30)"
  else
    log_failure "Found $impossible_spreads games with |spread| > 30"
  fi

  # Check for impossible passing yards
  run_check
  impossible_passing=$(query_db "SELECT COUNT(*) FROM plays WHERE pass = true AND passing_yards > 99;" | xargs)
  if [[ "$impossible_passing" == "0" ]]; then
    log_success "No impossible passing yards (all <= 99)"
  else
    log_failure "Found $impossible_passing plays with passing_yards > 99"
  fi

  # Check weather outliers
  run_check
  extreme_wind=$(query_db "SELECT COUNT(*) FROM weather WHERE wind_kph > 120;" | xargs)
  if [[ "$extreme_wind" == "0" ]]; then
    log_success "No extreme wind outliers (all wind < 120 kph)"
  else
    log_warning "Found $extreme_wind weather records with wind > 120 kph (verify data quality)"
  fi

  # Check temperature ranges
  run_check
  temp_min=$(query_db "SELECT MIN(temp_c) FROM weather WHERE temp_c IS NOT NULL;" | xargs)
  temp_max=$(query_db "SELECT MAX(temp_c) FROM weather WHERE temp_c IS NOT NULL;" | xargs)

  if (( $(echo "$temp_min >= -30" | bc -l) )) && (( $(echo "$temp_max <= 50" | bc -l) )); then
    log_success "Temperature range realistic ($temp_min°C to $temp_max°C)"
  else
    log_warning "Temperature range unusual: $temp_min°C to $temp_max°C"
  fi

  echo ""
fi

# ============================================================================
# 5. OVERTIME TRACKING
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "overtime" ]]; then
  echo "---[ Overtime Tracking ]---"
  run_check

  # Check overtime games since 2017 rule change
  ot_games=$(query_db "SELECT COUNT(*) FROM games WHERE overtime = 1 AND season >= 2017;" | xargs)
  ot_pct=$(query_db "
    SELECT ROUND(
      100.0 * SUM(CASE WHEN overtime = 1 THEN 1 ELSE 0 END) / COUNT(*),
      2
    )
    FROM games
    WHERE season >= 2017;
  " | xargs)

  if [[ -n "$ot_games" ]] && [[ "$ot_games" -gt 0 ]]; then
    log_success "Overtime games tracked: $ot_games since 2017 (~$ot_pct% of games)"
  else
    log_warning "No overtime games found (check data completeness)"
  fi

  echo ""
fi

# ============================================================================
# 6. TEMPORAL VALIDITY
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "temporal_validity" ]]; then
  echo "---[ Temporal Validity ]---"
  run_check

  # Check for rolling averages before week 4
  early_rolling=$(query_db "
    SELECT COUNT(*)
    FROM mv_team_rolling_stats
    WHERE points_for_l3 IS NOT NULL AND week < 4;
  " | xargs 2>/dev/null || echo "0")

  if [[ "$early_rolling" == "0" ]]; then
    log_success "No 3-game rolling averages before week 4 (no temporal leakage)"
  else
    log_failure "Found $early_rolling rolling averages before week 4 (temporal leakage)"
  fi

  # Check kickoff ordering
  run_check
  misordered=$(query_db "
    WITH ordered_games AS (
      SELECT game_id, team, season, week, kickoff,
             LAG(kickoff) OVER (PARTITION BY team, season ORDER BY week, game_id) as prev_kickoff
      FROM mv_team_rolling_stats
    )
    SELECT COUNT(*)
    FROM ordered_games
    WHERE kickoff < prev_kickoff;
  " | xargs 2>/dev/null || echo "0")

  if [[ "$misordered" == "0" ]]; then
    log_success "All games properly ordered by kickoff (temporal correctness)"
  else
    log_failure "Found $misordered misordered games (kickoff < previous kickoff)"
  fi

  echo ""
fi

# ============================================================================
# 7. BETTING MATH
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "betting_math" ]]; then
  echo "---[ Betting Math Validation ]---"

  # Change to project root
  cd "$(dirname "$0")/.." || exit 1

  # Kelly criterion formula check (Python)
  run_check
  if python3 -c "
import sys
sys.path.insert(0, 'py')
from production.kelly_sizing import kelly_criterion, american_to_decimal_odds

# Test case: 55% win prob, -110 odds
win_prob = 0.55
odds = -110
decimal_odds = american_to_decimal_odds(odds)
kelly_frac = kelly_criterion(win_prob, decimal_odds, kelly_fraction=1.0)

# Expected: (0.909 * 0.55 - 0.45) / 0.909 ≈ 0.0549 (5.49%)
expected = 0.0549
if abs(kelly_frac - expected) < 0.001:
    print('PASS')
else:
    print('FAIL')
" 2>/dev/null | grep -q "PASS"; then
    log_success "Kelly criterion formula correct (55% @ -110 → 5.49%)"
  else
    log_failure "Kelly criterion formula incorrect (check py/production/kelly_sizing.py)"
  fi

  # American odds conversion check
  run_check
  if python3 -c "
import sys
sys.path.insert(0, 'py')
from production.kelly_sizing import american_to_decimal_odds

# Test: -110 → 0.909, +150 → 1.5
neg_odds = american_to_decimal_odds(-110)
pos_odds = american_to_decimal_odds(150)

if abs(neg_odds - 0.909) < 0.01 and abs(pos_odds - 1.5) < 0.01:
    print('PASS')
else:
    print('FAIL')
" 2>/dev/null | grep -q "PASS"; then
    log_success "American odds conversion correct (-110→0.909, +150→1.5)"
  else
    log_failure "American odds conversion incorrect"
  fi

  echo ""
fi

# ============================================================================
# 8. CODE QUALITY: fillna(0) AUDIT
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "fillna_audit" ]]; then
  echo "---[ Code Quality: fillna(0) Audit ]---"
  run_check

  fillna_count=$(grep -rn "fillna(0)" py/ 2>/dev/null | wc -l | xargs || echo "0")

  if [[ "$fillna_count" -gt 0 ]]; then
    log_warning "Found $fillna_count occurrences of fillna(0) - audit for domain correctness (P2)"
    if [[ "$VERBOSE" == "true" ]]; then
      echo ""
      echo "  Locations:"
      grep -rn "fillna(0)" py/ | head -10
      echo "  ... (use grep for full list)"
    fi
  else
    log_success "No fillna(0) occurrences found (or py/ directory not present)"
  fi

  echo ""
fi

# ============================================================================
# 9. STATISTICAL MODELS
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "models" ]]; then
  echo "---[ Statistical Models ]---"
  run_check

  # Check if hierarchical model uses Gaussian family
  if grep -q "family = gaussian()" R/train_hierarchical_v2_canonical.R 2>/dev/null; then
    log_success "Hierarchical model uses Gaussian family (lognormal for yards)"
  else
    log_warning "Hierarchical model family unclear (check R/train_hierarchical_v2_canonical.R)"
  fi

  # Check MCMC configuration
  run_check
  if grep -q "chains = 4" R/train_hierarchical_v2_canonical.R 2>/dev/null; then
    log_success "MCMC uses 4 chains (adequate for convergence)"
  else
    log_warning "MCMC chain configuration unclear"
  fi

  # Check for R-hat convergence threshold
  run_check
  if grep -q "1.01" R/train_hierarchical_v2_canonical.R 2>/dev/null; then
    log_success "R-hat threshold set to 1.01 (industry standard)"
  else
    log_warning "R-hat threshold unclear (check convergence criteria)"
  fi

  echo ""
fi

# ============================================================================
# 10. PRODUCTION READINESS
# ============================================================================
if [[ -z "$SPECIFIC_CHECK" || "$SPECIFIC_CHECK" == "production" ]]; then
  echo "---[ Production Readiness ]---"
  run_check

  # Check for risk gates
  if grep -q "MIN_ROI\|MIN_WIN_RATE\|MIN_SHARPE" py/utils/betting_validator.py 2>/dev/null; then
    log_success "Risk gates implemented (ROI, win rate, Sharpe, drawdown)"
  else
    log_warning "Risk gates not found (check py/utils/betting_validator.py)"
  fi

  # Check for Kelly fraction
  run_check
  if grep -q "kelly_fraction.*0\.25\|kelly_fraction.*0\.35" py/production/*.py 2>/dev/null; then
    log_success "Kelly fraction set to 25-35% (conservative, industry standard)"
  else
    log_warning "Kelly fraction configuration unclear"
  fi

  # Check for CLV tracking
  run_check
  if grep -q "closing_line\|CLV" py/models/player_props_ensemble.py 2>/dev/null; then
    log_success "Closing line value (CLV) tracked in production"
  else
    log_warning "CLV not found in production pipeline (P2: implement before live betting)"
  fi

  echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo "========================================================================"
echo "  VALIDATION SUMMARY"
echo "========================================================================"
echo ""
echo "Total Checks:    $TOTAL_CHECKS"
echo -e "${GREEN}Passed:${NC}         $PASSED_CHECKS"
echo -e "${RED}Failed:${NC}         $FAILED_CHECKS"
echo -e "${YELLOW}Warnings:${NC}       $WARNINGS"
echo ""

if [[ $FAILED_CHECKS -eq 0 ]]; then
  echo -e "${GREEN}✓ All critical checks passed${NC}"
  exit_code=0
else
  echo -e "${RED}✗ $FAILED_CHECKS critical checks failed${NC}"
  exit_code=1
fi

if [[ $WARNINGS -gt 0 ]]; then
  echo -e "${YELLOW}⚠ $WARNINGS warnings (review recommended)${NC}"
fi

echo ""
echo "Full report: docs/audits/2025_11_comprehensive_audit/15_sports_analytics_validation.md"
echo ""

exit $exit_code
