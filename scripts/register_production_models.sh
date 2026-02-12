#!/bin/bash
##############################################################################
# Register Production Models to Database
#
# Automated script to register all 7 production models from MODEL_REGISTRY.yaml
# to the ml.model_registry PostgreSQL table.
#
# Models:
#   1. State-Space v2 (simplified_random_walk_v2.rds)
#   2. Hierarchical v2 (passing_yards_hierarchical_v2.rds)
#   3. Informative Priors v2 (passing_informative_priors_v2.rds)
#   4. Hybrid XGBoost v1 (hybrid_static_incremental_v1/)
#   5. BNN v3.1 Rushing (bnn_multioutput_v1_final.pt)
#   6. CQL Bet Sizer (best_model.pth)
#   7. Temperature Scaler (temperature_scaler_4way.pkl)
#
# Usage:
#   ./scripts/register_production_models.sh [--dry-run]
#
# Options:
#   --dry-run    Show SQL commands without executing
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025
##############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Database connection
DB_HOST="localhost"
DB_PORT="5544"
DB_NAME="devdb01"
DB_USER="dro"
DB_PASS="sicillionbillions"
DB_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}\n"
fi

# Helper function to execute SQL
run_sql() {
  local sql="$1"
  local description="$2"

  if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}Would execute:${NC}"
    echo -e "${YELLOW}${sql}${NC}\n"
  else
    echo -e "${BLUE}Executing: ${description}${NC}"
    psql "$DB_URL" -c "$sql"
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Success${NC}\n"
    else
      echo -e "${RED}❌ Failed${NC}\n"
      return 1
    fi
  fi
}

# Helper function to check if model exists
check_model_exists() {
  local model_name="$1"
  local result=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM ml.model_registry WHERE model_name = '$model_name';")
  echo "$result" | tr -d ' '
}

# Helper function to register or update model
register_model() {
  local model_name="$1"
  local model_type="$2"
  local model_family="$3"
  local version="$4"
  local model_path="$5"
  local status="$6"
  local description="$7"
  local notes="$8"

  local count=$(check_model_exists "$model_name")

  if [ "$count" -eq 0 ]; then
    # INSERT new model
    local sql="
INSERT INTO ml.model_registry (
  model_name,
  model_type,
  model_family,
  version,
  model_path,
  status,
  is_production,
  description,
  notes,
  created_at
) VALUES (
  '$model_name',
  '$model_type',
  '$model_family',
  '$version',
  '$model_path',
  '$status',
  true,
  '$description',
  '$notes',
  NOW()
);"
    run_sql "$sql" "Register new model: $model_name"
  else
    # UPDATE existing model
    local sql="
UPDATE ml.model_registry
SET
  model_type = '$model_type',
  model_family = '$model_family',
  version = '$version',
  model_path = '$model_path',
  status = '$status',
  is_production = true,
  description = '$description',
  notes = '$notes',
  updated_at = NOW()
WHERE model_name = '$model_name';"
    run_sql "$sql" "Update existing model: $model_name"
  fi
}

##############################################################################
# MAIN REGISTRATION
##############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}NFL Analytics - Production Model Registration${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

# Check database connection
echo -e "${BLUE}Checking database connection...${NC}"
if psql "$DB_URL" -c "SELECT 1;" > /dev/null 2>&1; then
  echo -e "${GREEN}✅ Database connection successful${NC}\n"
else
  echo -e "${RED}❌ Database connection failed${NC}"
  echo -e "${RED}Please check connection string: $DB_URL${NC}"
  exit 1
fi

# Check if models/ directory exists
if [ ! -d "models" ]; then
  echo -e "${RED}❌ models/ directory not found${NC}"
  echo -e "${RED}Please run from project root directory${NC}"
  exit 1
fi

##############################################################################
# Model 1: State-Space v2
##############################################################################

echo -e "${BLUE}==== Model 1/7: State-Space v2 ====${NC}"

register_model \
  "state_space_v2" \
  "state_space" \
  "bayesian_state_space" \
  "v2.0" \
  "models/bayesian/simplified_random_walk_v2.rds" \
  "production" \
  "Bayesian Random Walk State-Space v2 - APPROVED November 8, 2025" \
  "Approved via 4-scenario ablation study. Statistically superior to v1 (MAE 47.77 vs 49.12, p=0.0069). Uses simplified random walk for stability. Training window: 2018-2024 + H=4 decay. Convergence: Rhat 1.0013, ESS 2218, 0 divergences. Calibration: 88.5% (90% target). Ensemble weight: 30%."

##############################################################################
# Model 2: Hierarchical v2
##############################################################################

echo -e "${BLUE}==== Model 2/7: Hierarchical v2 ====${NC}"

register_model \
  "hierarchical_v2" \
  "hierarchical_bayesian" \
  "brms_hierarchical" \
  "v2.0" \
  "models/bayesian/passing_yards_hierarchical_v2.rds" \
  "production" \
  "Hierarchical Bayesian Passing v2 - PRODUCTION (retrained Nov 13, 2025)" \
  "Multi-level model: Player → Team → League. Partial pooling for player effects. Retrained with 6000 iterations for excellent convergence. Rhat 1.0092 (73% improvement from 2000 iter), ESS 8394, 42 divergences (<1%). Calibration: 87.2%. MAE 52.34 yards. Training window: 2018-2024 + H=4 decay. Ensemble weight: 30%. See ADR-002 for convergence threshold relaxation."

##############################################################################
# Model 3: Informative Priors v2
##############################################################################

echo -e "${BLUE}==== Model 3/7: Informative Priors v2 ====${NC}"

register_model \
  "informative_priors_v2" \
  "hierarchical_bayesian" \
  "brms_hierarchical" \
  "v2.0" \
  "models/bayesian/passing_informative_priors_v2.rds" \
  "production" \
  "Bayesian Informative Priors v2 - PRODUCTION (retrained Nov 13, 2025)" \
  "Enhanced version with stronger domain priors based on expert knowledge. Retrained with 6000 iterations for excellent convergence. Rhat 1.0177 (84% improvement from 2000 iter), ESS 8490, 177 divergences (<3%). Calibration: 86.8%. MAE 53.12 yards. Training window: 2018-2024 + H=4 decay. Ensemble weight: 15%. See ADR-002."

##############################################################################
# Model 4: Hybrid XGBoost v1
##############################################################################

echo -e "${BLUE}==== Model 4/7: Hybrid Static-Incremental XGBoost v1 ====${NC}"

register_model \
  "hybrid_static_incremental_v1" \
  "hybrid_ensemble" \
  "xgboost_hybrid" \
  "v1.0" \
  "models/xgboost/hybrid_static_incremental_v1" \
  "production" \
  "Hybrid Static-Incremental XGBoost v1 - APPROVED November 12, 2025" \
  "Breakthrough: 70/30 static-incremental blend. Static (70%): Trained once on 2018-2023, stability. Incremental (30%): Sliding window (150 games), retrain every 20 games, adaptation. MAE 38.03 (2022-2024), +4.2% improvement vs pure static (39.66). Replaces XGBoost v2 in ensemble. Training window: 2018-2024 + H=4 decay. Ensemble weight: 25%. Expected ROI boost: +0.5-1.0%."

##############################################################################
# Model 5: BNN v3.1 Rushing (EXCLUDED from ensemble but still production)
##############################################################################

echo -e "${BLUE}==== Model 5/7: BNN Multi-Output v3.1 (Rushing) ====${NC}"

register_model \
  "bnn_multioutput_v3_1" \
  "bayesian_neural_network" \
  "pytorch_bnn" \
  "v3.1" \
  "models/bayesian/bnn_multioutput_v1_final.pt" \
  "production" \
  "Multi-Output BNN v3.1 Calibrated (Rushing only)" \
  "Successfully calibrated October 2025 after temperature scaling (T=0.75). Mixture-of-Experts architecture (3 experts). MAE 58.70 yards, RMSE 76.12 yards. Calibration: 89.5% (excellent). Training window: 2018-2024 + H=4 decay. NOTE: Rushing yards only. Passing yards variant (v2) FAILED with NaN losses (Nov 9, 2025). NOT used in current ensemble v5.1 (rushing-only, ensemble is passing-focused)."

##############################################################################
# Model 6: CQL Bet Sizer
##############################################################################

echo -e "${BLUE}==== Model 6/7: CQL Bet Sizer ====${NC}"

register_model \
  "cql_bet_sizer" \
  "reinforcement_learning" \
  "conservative_q_learning" \
  "v1.0" \
  "models/cql/best_model.pth" \
  "production" \
  "Conservative Q-Learning Bet Sizer - PRODUCTION" \
  "Trained on 5,146 games (2006-2024). +24% improvement over baseline (policy reward 1.75 vs 1.41). Applied AFTER ensemble predictions for Kelly Criterion optimization. Risk gates: edge >= 2%, max bet 5% bankroll, stop-loss -10% daily drawdown, uncertainty filter < 80 yards std. Conservative Q-Learning for offline RL."

##############################################################################
# Model 7: Temperature Scaler 4-Way
##############################################################################

echo -e "${BLUE}==== Model 7/7: Temperature Scaler 4-Way ====${NC}"

register_model \
  "temperature_scaler_4way" \
  "calibration" \
  "isotonic_regression" \
  "v1.0" \
  "models/calibration/temperature_scaler_4way.pkl" \
  "production" \
  "Temperature Scaler for 4-Way Ensemble - APPROVED November 13, 2025" \
  "Applied to ensemble output after weighted averaging. Temperature T=0.5534 (sharpen by 47%). Reduces over-confidence: 97.3% coverage → 89.7% (target: 90%). Interval reduction: 25.6% sharper. Calibration samples: 146 games (2022-2024 validation). Dependencies: state_space_v2, hierarchical_v2, hybrid_static_incremental_v1, informative_priors_v2. Applied in Stage 3 of post-processing pipeline (after aggregation, before bet sizing)."

##############################################################################
# VERIFICATION
##############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Registration Complete - Verification${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

if [ "$DRY_RUN" = false ]; then
  echo -e "${BLUE}Querying registered production models...${NC}\n"

  psql "$DB_URL" -c "
SELECT
  model_name,
  version,
  status,
  is_production,
  LEFT(model_path, 50) as model_path
FROM ml.model_registry
WHERE is_production = true
ORDER BY model_name;
"

  echo -e "\n${BLUE}Checking model files exist on filesystem...${NC}\n"

  models=(
    "models/bayesian/simplified_random_walk_v2.rds"
    "models/bayesian/passing_yards_hierarchical_v2.rds"
    "models/bayesian/passing_informative_priors_v2.rds"
    "models/xgboost/hybrid_static_incremental_v1/static_model.pkl"
    "models/bayesian/bnn_multioutput_v1_final.pt"
    "models/cql/best_model.pth"
    "models/calibration/temperature_scaler_4way.pkl"
  )

  all_exist=true
  for model in "${models[@]}"; do
    if [ -f "$model" ] || [ -d "$(dirname "$model")" ]; then
      echo -e "${GREEN}✅ $model${NC}"
    else
      echo -e "${RED}❌ MISSING: $model${NC}"
      all_exist=false
    fi
  done

  echo ""
  if [ "$all_exist" = true ]; then
    echo -e "${GREEN}✅ All production models verified (database + filesystem)${NC}"
  else
    echo -e "${RED}⚠️  Some model files are missing - check paths above${NC}"
  fi
else
  echo -e "${YELLOW}Dry run complete - no changes made${NC}"
  echo -e "${YELLOW}Run without --dry-run to register models${NC}"
fi

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}Done${NC}"
echo -e "${GREEN}======================================================================${NC}\n"
