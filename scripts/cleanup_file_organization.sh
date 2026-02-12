#!/usr/bin/env bash
#
# File Organization Cleanup Script
#
# Moves misplaced documentation files to correct locations per FILE_ORGANIZATION.md
#
# Usage:
#   ./scripts/cleanup_file_organization.sh           # Dry run (show what would be moved)
#   ./scripts/cleanup_file_organization.sh --execute # Actually move files
#
# Author: Claude Code
# Date: November 13, 2025

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Execution mode
DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

# Permitted files in root
PERMITTED_FILES=(
    "README.md"
    "CLAUDE.md"
    "SETUP.md"
    "CONTRIBUTING.md"
    "FILE_ORGANIZATION.md"
    "LICENSE"
    ".gitignore"
    ".env.example"
)

# Create necessary directories
create_dirs() {
    local dirs=(
        "docs/archive/2025_nov"
        "docs/milestones"
        "docs/deployment"
        "docs/performance"
        "docs/fixes"
        "docs/guides"
        "docs/decisions"
        "docs/models"
        "docs/database"
    )

    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo -e "${BLUE}Creating directory: $dir${NC}"
            if [[ "$DRY_RUN" == false ]]; then
                mkdir -p "$dir"
            fi
        fi
    done
}

# Check if file is permitted in root
is_permitted() {
    local file="$1"
    for permitted in "${PERMITTED_FILES[@]}"; do
        if [[ "$file" == "$permitted" ]]; then
            return 0
        fi
    done
    return 1
}

# Determine correct location for a file
get_correct_location() {
    local file="$1"
    local basename=$(basename "$file")

    # Session summaries
    if [[ "$basename" =~ SESSION_SUMMARY ]]; then
        echo "docs/archive/2025_nov/"
        return
    fi

    # Implementation/milestone completions
    if [[ "$basename" =~ (IMPLEMENTATION|INTEGRATION|PHASE.*COMPLETE|MILESTONE) ]]; then
        echo "docs/milestones/"
        return
    fi

    # Deployment related
    if [[ "$basename" =~ (DEPLOYMENT|PRODUCTION|DOCKER) ]]; then
        echo "docs/deployment/"
        return
    fi

    # Performance/optimization
    if [[ "$basename" =~ (PERFORMANCE|OPTIMIZATION|ENSEMBLE|ABLATION) ]]; then
        echo "docs/performance/"
        return
    fi

    # Fixes/postmortems
    if [[ "$basename" =~ (FIX|POSTMORTEM|BLOCKER|ERROR) ]]; then
        echo "docs/fixes/"
        return
    fi

    # Model documentation
    if [[ "$basename" =~ (MODEL|BNN|XGB|BAYESIAN) ]] && [[ ! "$basename" =~ PERFORMANCE ]]; then
        echo "docs/models/"
        return
    fi

    # Database
    if [[ "$basename" =~ (DATABASE|SCHEMA|MIGRATION) ]]; then
        echo "docs/database/"
        return
    fi

    # Architecture/decisions
    if [[ "$basename" =~ (ARCHITECTURE|ADR|DECISION) ]]; then
        echo "docs/decisions/"
        return
    fi

    # Guides/tutorials
    if [[ "$basename" =~ (GUIDE|TUTORIAL|QUICKSTART|HOW) ]]; then
        echo "docs/guides/"
        return
    fi

    # Temporary/audit reports → archive
    if [[ "$basename" =~ (audit|temp|test|summary|log).*\.(txt|log) ]]; then
        echo "docs/archive/2025_nov/"
        return
    fi

    # Default: archive with date
    echo "docs/archive/2025_nov/"
}

# Move a file
move_file() {
    local src="$1"
    local dest_dir="$2"
    local basename=$(basename "$src")
    local dest="$dest_dir$basename"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}WOULD MOVE:${NC} $src → $dest"
    else
        echo -e "${GREEN}MOVING:${NC} $src → $dest"
        mv "$src" "$dest"
    fi
}

# Main cleanup function
cleanup() {
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}File Organization Cleanup${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}DRY RUN MODE${NC} - No files will be moved"
        echo -e "Run with ${GREEN}--execute${NC} to actually move files"
        echo ""
    else
        echo -e "${RED}EXECUTE MODE${NC} - Files will be moved"
        echo ""
    fi

    # Create necessary directories
    create_dirs
    echo ""

    # Find all .md and .txt files in root
    local count=0
    local files=$(find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.txt" \) ! -name ".*")

    while IFS= read -r file; do
        local basename=$(basename "$file")

        # Skip if file is permitted
        if is_permitted "$basename"; then
            echo -e "${GREEN}✓ KEEP:${NC} $basename (permitted in root)"
            continue
        fi

        # Determine correct location
        local dest_dir=$(get_correct_location "$file")

        # Move file
        move_file "$file" "$dest_dir"
        ((count++))
    done <<< "$files"

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Summary${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "Files that would be moved: ${YELLOW}$count${NC}"
        echo ""
        echo -e "${GREEN}Run with --execute to perform the move${NC}"
    else
        echo -e "Files moved: ${GREEN}$count${NC}"
        echo ""
        echo -e "${GREEN}✓ Cleanup complete!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Review moved files: ls -R docs/"
        echo "  2. Update any broken links"
        echo "  3. Commit changes: git add -A && git commit -m 'docs: reorganize per FILE_ORGANIZATION.md'"
    fi
}

# Usage information
usage() {
    cat << EOF
File Organization Cleanup Script

Usage:
  $(basename "$0")           # Dry run (show what would be moved)
  $(basename "$0") --execute # Actually move files
  $(basename "$0") --help    # Show this help

This script moves misplaced documentation files to correct locations
according to FILE_ORGANIZATION.md.

Permitted files in root:
$(printf '  - %s\n' "${PERMITTED_FILES[@]}")

All other .md and .txt files will be moved to appropriate docs/ subdirectories.

EOF
    exit 0
}

# Main
main() {
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        usage
    fi

    cleanup
}

main "$@"
