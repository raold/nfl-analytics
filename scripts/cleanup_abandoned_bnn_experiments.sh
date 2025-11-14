#!/bin/bash
##############################################################################
# Cleanup Abandoned BNN Experiments
#
# Deletes 11 abandoned BNN experiment files (3.5GB total).
# All models archived in database with reason="abandoned".
# BNN research line terminated November 9, 2025 after 6 failed approaches.
#
# Files to delete:
#   1. bnn_multioutput_v1.pkl (1.6GB) - Data leakage, archived
#   2. bnn_mixture_experts_v2.pkl (1.3GB) - Abandoned research
#   3. bnn_rushing_opponent.pkl (371MB) - Experiment
#   4. bnn_rushing_improved_v2.pkl (343MB) - Experiment
#   5. bnn_rushing_environment.pkl (302MB) - Experiment
#   6. bnn_nuts_heteroscedastic_sigma0.3_v1.pkl (84MB) - NUTS timeout
#   7. bnn_nuts_heteroscedastic_sigma0.5_v1.pkl (84MB) - NUTS timeout
#   8. bnn_nuts_homoscedastic_sigma0.3_v1.pkl (47MB) - NUTS timeout
#   9. bnn_nuts_homoscedastic_sigma0.5_v1.pkl (47MB) - NUTS timeout
#  10. bnn_advi_sigma0.3_arch32x16_v1.pkl (70MB) - ADVI experiment
#  11. bnn_demo_v1.pkl (13MB) - Demo only
#
# Usage:
#   ./scripts/cleanup_abandoned_bnn_experiments.sh [--dry-run]
#
# Options:
#   --dry-run    List files without deleting
#
# Safety:
#   - Requires confirmation before deletion
#   - Verifies database has archived records
#   - Creates backup list of deleted files
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025
# Reference: docs/audits/2025_11_comprehensive_audit/14_model_registry_audit.md
##############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo -e "${YELLOW}🔍 DRY RUN MODE - No files will be deleted${NC}\n"
fi

# Check we're in project root
if [ ! -d "models/bayesian" ]; then
  echo -e "${RED}❌ Error: models/bayesian directory not found${NC}"
  echo -e "${RED}Please run from project root directory${NC}"
  exit 1
fi

# Files to delete (relative to models/bayesian/)
declare -a FILES=(
  "bnn_multioutput_v1.pkl"
  "bnn_mixture_experts_v2.pkl"
  "bnn_rushing_opponent.pkl"
  "bnn_rushing_improved_v2.pkl"
  "bnn_rushing_environment.pkl"
  "bnn_nuts_heteroscedastic_sigma0.3_v1.pkl"
  "bnn_nuts_heteroscedastic_sigma0.5_v1.pkl"
  "bnn_nuts_homoscedastic_sigma0.3_v1.pkl"
  "bnn_nuts_homoscedastic_sigma0.5_v1.pkl"
  "bnn_advi_sigma0.3_arch32x16_v1.pkl"
  "bnn_demo_v1.pkl"
)

# File sizes (for verification)
declare -A EXPECTED_SIZES=(
  ["bnn_multioutput_v1.pkl"]="1.6G"
  ["bnn_mixture_experts_v2.pkl"]="1.3G"
  ["bnn_rushing_opponent.pkl"]="371M"
  ["bnn_rushing_improved_v2.pkl"]="343M"
  ["bnn_rushing_environment.pkl"]="302M"
  ["bnn_nuts_heteroscedastic_sigma0.3_v1.pkl"]="84M"
  ["bnn_nuts_heteroscedastic_sigma0.5_v1.pkl"]="84M"
  ["bnn_nuts_homoscedastic_sigma0.3_v1.pkl"]="47M"
  ["bnn_nuts_homoscedastic_sigma0.5_v1.pkl"]="47M"
  ["bnn_advi_sigma0.3_arch32x16_v1.pkl"]="70M"
  ["bnn_demo_v1.pkl"]="13M"
)

##############################################################################
# HEADER
##############################################################################

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Cleanup Abandoned BNN Experiments${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

echo -e "${YELLOW}Context:${NC}"
echo -e "  BNN research line abandoned November 9, 2025"
echo -e "  6 consecutive BNN approaches failed (84 person-hours, 26 days)"
echo -e "  Only BNN v3.1 Rushing (58KB) remains in production"
echo -e "  See: analysis/dissertation/main/appendix_bnn_investigation.tex"
echo -e ""

##############################################################################
# SCAN FILES
##############################################################################

echo -e "${BLUE}Scanning for abandoned BNN experiment files...${NC}\n"

total_size=0
existing_files=()
missing_files=()

for file in "${FILES[@]}"; do
  filepath="models/bayesian/$file"

  if [ -f "$filepath" ]; then
    size=$(du -h "$filepath" | cut -f1)
    existing_files+=("$file")
    echo -e "${YELLOW}Found:${NC} $file ($size, expected: ${EXPECTED_SIZES[$file]})"

    # Calculate total size in MB for summary
    size_mb=$(du -m "$filepath" | cut -f1)
    total_size=$((total_size + size_mb))
  else
    missing_files+=("$file")
    echo -e "${BLUE}Skip:${NC} $file (already deleted)"
  fi
done

total_size_gb=$(echo "scale=1; $total_size / 1024" | bc)

echo -e "\n${BLUE}Summary:${NC}"
echo -e "  Files found: ${#existing_files[@]}"
echo -e "  Already deleted: ${#missing_files[@]}"
echo -e "  Total size: ${total_size_gb}GB (~${total_size}MB)"
echo -e ""

# Exit if no files to delete
if [ ${#existing_files[@]} -eq 0 ]; then
  echo -e "${GREEN}✅ All abandoned BNN experiments already deleted${NC}"
  exit 0
fi

##############################################################################
# DATABASE VERIFICATION
##############################################################################

echo -e "${BLUE}Verifying database has archived records...${NC}"

DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

if ! command -v psql &> /dev/null; then
  echo -e "${YELLOW}⚠️  psql not found - skipping database verification${NC}"
else
  archived_count=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM ml.model_registry WHERE status = 'archived' AND model_name LIKE 'bnn_%';" 2>/dev/null | tr -d ' ' || echo "0")

  if [ "$archived_count" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $archived_count archived BNN models in database${NC}\n"
  else
    echo -e "${YELLOW}⚠️  No archived BNN models found in database (expected 6)${NC}"
    echo -e "${YELLOW}Database records may not exist, but safe to delete files${NC}\n"
  fi
fi

##############################################################################
# CONFIRMATION
##############################################################################

if [ "$DRY_RUN" = false ]; then
  echo -e "${RED}======================================================================${NC}"
  echo -e "${RED}⚠️  WARNING: You are about to DELETE ${#existing_files[@]} files (${total_size_gb}GB)${NC}"
  echo -e "${RED}======================================================================${NC}\n"

  echo -e "${YELLOW}Files to delete:${NC}"
  for file in "${existing_files[@]}"; do
    echo -e "  - $file"
  done

  echo -e "\n${YELLOW}This action cannot be undone.${NC}"
  echo -e "${YELLOW}Database archived records will be preserved for historical reference.${NC}\n"

  read -p "Are you sure you want to delete these files? (yes/no): " confirm

  if [ "$confirm" != "yes" ]; then
    echo -e "\n${BLUE}Cancelled by user. No files deleted.${NC}"
    exit 0
  fi
fi

##############################################################################
# DELETION
##############################################################################

if [ "$DRY_RUN" = true ]; then
  echo -e "\n${YELLOW}[DRY RUN] Would delete the following files:${NC}\n"
  for file in "${existing_files[@]}"; do
    filepath="models/bayesian/$file"
    size=$(du -h "$filepath" | cut -f1)
    echo -e "  ${YELLOW}rm${NC} $filepath ($size)"
  done
  echo -e "\n${YELLOW}Run without --dry-run to perform deletion${NC}"
else
  echo -e "\n${BLUE}Deleting files...${NC}\n"

  # Create backup list of deleted files
  backup_file="models/bayesian/DELETED_BNN_EXPERIMENTS_$(date +%Y%m%d_%H%M%S).txt"
  echo "# Deleted BNN Experiment Files - $(date)" > "$backup_file"
  echo "# Total: ${#existing_files[@]} files, ${total_size_gb}GB" >> "$backup_file"
  echo "# Reason: BNN research abandoned November 9, 2025" >> "$backup_file"
  echo "" >> "$backup_file"

  deleted_count=0
  for file in "${existing_files[@]}"; do
    filepath="models/bayesian/$file"
    size=$(du -h "$filepath" | cut -f1)

    # Log to backup file
    echo "$filepath ($size)" >> "$backup_file"

    # Delete file
    rm -f "$filepath"

    if [ ! -f "$filepath" ]; then
      echo -e "${GREEN}✅ Deleted:${NC} $file ($size)"
      deleted_count=$((deleted_count + 1))
    else
      echo -e "${RED}❌ Failed to delete:${NC} $file"
    fi
  done

  echo -e "\n${BLUE}Backup list created:${NC} $backup_file"
  echo -e "\n${GREEN}======================================================================${NC}"
  echo -e "${GREEN}Cleanup Complete${NC}"
  echo -e "${GREEN}======================================================================${NC}\n"

  echo -e "${GREEN}✅ Deleted $deleted_count files (${total_size_gb}GB)${NC}"
  echo -e "${BLUE}Remaining BNN files in models/bayesian/:${NC}"

  bnn_remaining=$(find models/bayesian -name "bnn_*.pkl" -o -name "bnn_*.pt" | wc -l | tr -d ' ')
  if [ "$bnn_remaining" -gt 0 ]; then
    echo -e "  ${bnn_remaining} BNN files (includes production bnn_multioutput_v1_final.pt)"
  else
    echo -e "  None (all BNN files deleted)"
  fi

  # Check directory size after deletion
  new_size=$(du -sh models/bayesian | cut -f1)
  echo -e "\n${BLUE}models/bayesian/ size after cleanup:${NC} $new_size"
  echo -e "${GREEN}Expected savings: ~${total_size_gb}GB${NC}\n"
fi

##############################################################################
# ADDITIONAL CLEANUP (BNN multioutput checkpoints)
##############################################################################

if [ -d "models/bayesian/bnn_multioutput_checkpoints" ]; then
  checkpoint_size=$(du -sh models/bayesian/bnn_multioutput_checkpoints | cut -f1)
  checkpoint_count=$(find models/bayesian/bnn_multioutput_checkpoints -name "*.pt" | wc -l | tr -d ' ')

  echo -e "${YELLOW}======================================================================${NC}"
  echo -e "${YELLOW}Additional Cleanup Opportunity${NC}"
  echo -e "${YELLOW}======================================================================${NC}\n"

  echo -e "${YELLOW}Found BNN multioutput checkpoint directory:${NC}"
  echo -e "  Path: models/bayesian/bnn_multioutput_checkpoints/"
  echo -e "  Files: $checkpoint_count checkpoint files"
  echo -e "  Size: $checkpoint_size"
  echo -e "  Final model: bnn_multioutput_v1_final.pt (58KB)"
  echo -e ""
  echo -e "${YELLOW}Recommendation: Delete checkpoint directory to save additional space${NC}"
  echo -e "${YELLOW}Command: rm -rf models/bayesian/bnn_multioutput_checkpoints/${NC}\n"
fi

echo -e "${GREEN}Done.${NC}\n"
