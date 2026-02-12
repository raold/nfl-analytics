#!/bin/bash
# cleanup_root_directory.sh
# Purpose: Move scattered .md/.txt files from root to proper locations per FILE_ORGANIZATION.md
# Date: November 15, 2025
# Status: Ready to execute (dry-run mode by default)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "")/.." \&\& pwd)"
cd "$PROJECT_ROOT"

# Dry run by default (set to false to actually move files)
DRY_RUN=true
if [[ "$1" == "--execute" ]]; then
    DRY_RUN=false
    echo -e "${RED}WARNING: Executing file moves (not a dry run)${NC}"
    echo "Press Ctrl+C within 5 seconds to cancel..."
    sleep 5
else
    echo -e "${BLUE}DRY RUN MODE${NC}"
    echo "Run with --execute flag to actually move files"
    echo ""
fi

# Function to move file
move_file() {
    local src="$1"
    local dst="$2"

    if [[ ! -f "$src" ]]; then
        echo -e "${YELLOW}SKIP${NC}: $src (not found)"
        return
    fi

    # Create destination directory if needed
    local dst_dir=$(dirname "$dst")
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$dst_dir"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${BLUE}WOULD MOVE${NC}: $src → $dst"
    else
        mv "$src" "$dst"
        echo -e "${GREEN}MOVED${NC}: $src → $dst"
    fi
}

echo "=========================================="
echo "NFL Analytics - Root Directory Cleanup"
echo "=========================================="
echo ""

# Create new directories
echo "Creating directories..."
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p docs/experiments
    mkdir -p docs/archive/2025_nov/ab_test_backup
    mkdir -p docs/archive/2025_nov/ensemble_history
    echo -e "${GREEN}Created${NC}: docs/experiments"
    echo -e "${GREEN}Created${NC}: docs/archive/2025_nov/ab_test_backup"
    echo -e "${GREEN}Created${NC}: docs/archive/2025_nov/ensemble_history"
else
    echo -e "${BLUE}Would create${NC}: docs/experiments"
    echo -e "${BLUE}Would create${NC}: docs/archive/2025_nov/ab_test_backup"
    echo -e "${BLUE}Would create${NC}: docs/archive/2025_nov/ensemble_history"
fi
echo ""

# Phase 1: Session summaries → archive
echo "Phase 1: Session Summaries & Status Reports"
echo "--------------------------------------------"
move_file "SESSION_STATUS_NOV14.md" "docs/archive/2025_nov/SESSION_STATUS_2025-11-14.md"
move_file "TRAINING_SCRIPT_CREATION_SUMMARY.md" "docs/archive/2025_nov/TRAINING_SCRIPT_SUMMARY_2025-11-14.md"
move_file "SESSION_COMPLETION_REPORT.md" "docs/archive/2025_nov/SESSION_COMPLETION_2025-11-13.md"
move_file "STATE_SPACE_FIX_REPORT_NOV13.md" "docs/fixes/STATE_SPACE_FIX_NOV13.md"
echo ""

# Phase 2: Delivery reports → appropriate locations
echo "Phase 2: Delivery Reports & Checklists"
echo "---------------------------------------"
move_file "AB_TEST_DELIVERY_REPORT.md" "docs/deployment/AB_TEST_DELIVERY_REPORT.md"
move_file "FINAL_DELIVERABLES_INDEX.md" "docs/archive/2025_nov/DELIVERABLES_INDEX_2025-11-14.md"
move_file "FEATURE_ANALYSIS_DELIVERY.md" "docs/archive/2025_nov/FEATURE_ANALYSIS_DELIVERY_2025-11-14.md"
move_file "VERIFICATION_CHECKLIST.md" "docs/archive/2025_nov/VERIFICATION_CHECKLIST.md"
move_file "TESTING_GUIDE_FINAL.md" "docs/guides/TESTING_GUIDE.md"
echo ""

# Phase 3: Experiment docs → experiments/
echo "Phase 3: Experiment Documentation"
echo "----------------------------------"
move_file "EXPERIMENT_2_DELIVERY.md" "docs/experiments/XGB_EXP2_DELIVERY.md"
move_file "EXPERIMENT_2_FILES_INDEX.txt" "docs/experiments/XGB_EXP2_FILES_INDEX.txt"
move_file "BFS_EXPERIMENT_5_CHECKLIST.md" "docs/experiments/BFS_EXP5_CHECKLIST.md"
move_file "BFS_EXPERIMENTS_EXECUTION_SUMMARY.md" "docs/experiments/BFS_EXPERIMENTS_SUMMARY.md"
move_file "XGBOOST_V3_1_EXECUTION_SUMMARY.md" "docs/experiments/XGBOOST_V3_1_SUMMARY.md"
echo ""

# Phase 4: Quick references → guides/
echo "Phase 4: Quick References & Guides"
echo "-----------------------------------"
move_file "QUICK_RECOVERY_COMMANDS.txt" "docs/guides/DATABASE_RECOVERY_QUICK_REFERENCE.txt"
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${BLUE}DRY RUN COMPLETE${NC}"
    echo ""
    echo "No files were actually moved."
    echo "Run with --execute flag to perform moves:"
    echo "  ./scripts/cleanup_root_directory.sh --execute"
else
    echo -e "${GREEN}CLEANUP COMPLETE${NC}"
    echo ""
    echo "Files moved successfully."
    echo "Run 'git status' to review changes."
    echo "Run 'git add -A && git commit -m \"docs: reorganize root files per FILE_ORGANIZATION.md\"' to commit."
fi
echo ""

# Verify root directory
echo "Root directory files (should only be 8 permitted files):"
echo "---------------------------------------------------------"
ls -1 *.md *.txt 2>/dev/null | while read file; do
    case "$file" in
        README.md|CLAUDE.md|SETUP.md|CONTRIBUTING.md|FILE_ORGANIZATION.md)
            echo -e "${GREEN}✓${NC} $file (permitted)"
            ;;
        *)
            echo -e "${RED}✗${NC} $file (violation - should be moved)"
            ;;
    esac
done

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
if [[ "$DRY_RUN" == true ]]; then
    echo "1. Review moves above"
    echo "2. Run with --execute if satisfied:"
    echo "   ./scripts/cleanup_root_directory.sh --execute"
    echo "3. Commit changes: git commit -am 'docs: reorganize root files'"
else
    echo "1. Review: git status"
    echo "2. Test navigation in moved files"
    echo "3. Commit: git commit -am 'docs: reorganize root files per FILE_ORGANIZATION.md'"
    echo "4. (Optional) Run Phase 2-4 from FILE_ORGANIZATION_SURVEY_NOV15.md"
fi
