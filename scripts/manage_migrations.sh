#!/usr/bin/env bash
#
# Database Migration Management CLI
# ==================================
#
# Wrapper script for DatabaseMigrationAgent utility
#
# Usage:
#   ./scripts/manage_migrations.sh generate <schema_diff.json> [--name migration_name]
#   ./scripts/manage_migrations.sh validate <migration_file>
#   ./scripts/manage_migrations.sh apply <migration_file> [--dry-run]
#   ./scripts/manage_migrations.sh rollback <migration_id>
#   ./scripts/manage_migrations.sh audit [--target-db devdb02]
#   ./scripts/manage_migrations.sh list
#
# Examples:
#   # Generate migration from schema diff
#   ./scripts/manage_migrations.sh generate schema_changes.json --name add_ensemble_tracking
#
#   # Validate migration before applying
#   ./scripts/manage_migrations.sh validate db/migrations/050_add_ensemble_tracking.sql
#
#   # Apply migration (dry run first)
#   ./scripts/manage_migrations.sh apply db/migrations/050_add_ensemble_tracking.sql --dry-run
#   ./scripts/manage_migrations.sh apply db/migrations/050_add_ensemble_tracking.sql
#
#   # Rollback migration
#   ./scripts/manage_migrations.sh rollback 42
#
#   # Audit schema drift between environments
#   ./scripts/manage_migrations.sh audit --target-db production_db
#
#   # List applied migrations
#   ./scripts/manage_migrations.sh list
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python script path
MIGRATION_AGENT="$PROJECT_ROOT/py/utils/database_migration_agent.py"

# Check if migration agent exists
if [ ! -f "$MIGRATION_AGENT" ]; then
    echo -e "${RED}ERROR: Migration agent not found at $MIGRATION_AGENT${NC}"
    exit 1
fi

# Function to display usage
usage() {
    cat << EOF
${BLUE}Database Migration Management CLI${NC}

${GREEN}USAGE:${NC}
    $0 <command> [options]

${GREEN}COMMANDS:${NC}
    generate <schema_diff.json> [--name <name>]
        Generate migration file from schema diff JSON

    validate <migration_file>
        Validate migration before applying

    apply <migration_file> [--dry-run]
        Apply migration with safety checks
        Use --dry-run to test without committing changes

    rollback <migration_id>
        Rollback a specific migration by ID

    audit [--target-db <dbname>] [--target-host <host>] [--target-port <port>]
        Audit schema drift between environments
        Default target: localhost:5544

    list
        List all applied migrations

${GREEN}EXAMPLES:${NC}
    # Generate migration
    $0 generate schema_changes.json --name add_ensemble_tracking

    # Validate migration
    $0 validate db/migrations/050_add_ensemble_tracking.sql

    # Apply migration (dry run first)
    $0 apply db/migrations/050_add_ensemble_tracking.sql --dry-run
    $0 apply db/migrations/050_add_ensemble_tracking.sql

    # Rollback migration
    $0 rollback 42

    # Audit schema drift
    $0 audit --target-db production_db

    # List migrations
    $0 list

${GREEN}EXIT CODES:${NC}
    0 - Success
    1 - Error or validation failed

EOF
}

# Check if command provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

# Execute command
case "$COMMAND" in
    generate)
        if [ $# -eq 0 ]; then
            echo -e "${RED}ERROR: schema_diff.json file required${NC}"
            echo "Usage: $0 generate <schema_diff.json> [--name <name>]"
            exit 1
        fi

        echo -e "${BLUE}Generating migration from schema diff...${NC}"
        python "$MIGRATION_AGENT" generate "$@"
        ;;

    validate)
        if [ $# -eq 0 ]; then
            echo -e "${RED}ERROR: migration_file required${NC}"
            echo "Usage: $0 validate <migration_file>"
            exit 1
        fi

        echo -e "${BLUE}Validating migration: $1${NC}"
        python "$MIGRATION_AGENT" validate "$@"
        ;;

    apply)
        if [ $# -eq 0 ]; then
            echo -e "${RED}ERROR: migration_file required${NC}"
            echo "Usage: $0 apply <migration_file> [--dry-run]"
            exit 1
        fi

        MIGRATION_FILE="$1"
        shift

        # Check for --dry-run flag
        DRY_RUN=false
        while [ $# -gt 0 ]; do
            case "$1" in
                --dry-run)
                    DRY_RUN=true
                    shift
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}DRY RUN MODE - No changes will be committed${NC}"
            python "$MIGRATION_AGENT" apply "$MIGRATION_FILE" --dry-run
        else
            echo -e "${BLUE}Applying migration: $MIGRATION_FILE${NC}"
            echo -e "${YELLOW}WARNING: This will modify the database. Ctrl+C to cancel.${NC}"
            sleep 2
            python "$MIGRATION_AGENT" apply "$MIGRATION_FILE"
        fi
        ;;

    rollback)
        if [ $# -eq 0 ]; then
            echo -e "${RED}ERROR: migration_id required${NC}"
            echo "Usage: $0 rollback <migration_id>"
            exit 1
        fi

        MIGRATION_ID="$1"

        echo -e "${YELLOW}WARNING: Rolling back migration $MIGRATION_ID${NC}"
        echo -e "${YELLOW}This will modify the database. Ctrl+C to cancel.${NC}"
        sleep 2

        python "$MIGRATION_AGENT" rollback "$MIGRATION_ID"
        ;;

    audit)
        echo -e "${BLUE}Auditing schema drift...${NC}"
        python "$MIGRATION_AGENT" audit "$@"
        ;;

    list)
        echo -e "${BLUE}Listing applied migrations...${NC}"
        python "$MIGRATION_AGENT" list
        ;;

    help|--help|-h)
        usage
        exit 0
        ;;

    *)
        echo -e "${RED}ERROR: Unknown command: $COMMAND${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

exit $?
