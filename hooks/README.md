# Git Hooks for NFL Analytics Project

This directory contains Git hook templates for automated validation and enforcement.

## Available Hooks

### 1. pre-commit-frontend-validation

**Purpose:** Validates dashboard SQL queries against current database schema before commit.

**Prevents:**
- The `mart.player_game_stats.game_id` error (occurred DOZENS of times)
- Hardcoded team abbreviations ('LA' instead of 'LAR'/'LAC')
- Missing WHERE clauses on large tables
- SELECT * queries (anti-pattern)

**Installation:**

```bash
# Copy to .git/hooks/
cp hooks/pre-commit-frontend-validation .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Usage:**

Runs automatically on `git commit`. Example output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 FRONTEND SCHEMA VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Running schema validation...

[ERROR] schema_validation: Column 'game_id' does not exist in table 'mart.player_game_stats'
  File: dashboard/app.py
  Line: 1417
  💡 Suggestion: Join to games via (season, week) composite key

❌ Schema validation failed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ COMMIT BLOCKED - Schema validation failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Bypass (emergency only):**

```bash
git commit --no-verify
```

**⚠️ WARNING:** Bypassing removes protection against schema errors.

---

## Hook Installation Guide

### Automatic Installation (Recommended)

```bash
# Install all hooks
./scripts/install_hooks.sh
```

### Manual Installation

```bash
# 1. Copy hook template
cp hooks/pre-commit-frontend-validation .git/hooks/pre-commit

# 2. Make executable
chmod +x .git/hooks/pre-commit

# 3. Test hook
./git/hooks/pre-commit
```

### Verify Installation

```bash
# Check if hook is installed
ls -la .git/hooks/pre-commit

# Test hook manually
./.git/hooks/pre-commit
```

---

## Configuration

### Environment Variables

Hooks use the same environment variables as the main project:

```bash
export DB_HOST=localhost
export DB_PORT=5544
export DB_NAME=devdb01
export DB_USER=dro
export DB_PASSWORD=sicillionbillions  # pragma: allowlist secret
```

### Hook Behavior

Edit hook file to customize behavior:

```bash
# Validate only changed files (faster)
VALIDATE_ONLY_CHANGED=true

# Block on warnings (stricter)
BLOCK_ON_WARNINGS=false

# Files to check
FILES_TO_CHECK=(
    "dashboard/app.py"
    "api/main.py"
)
```

---

## Troubleshooting

### Issue 1: Hook Not Running

**Problem:** `git commit` doesn't trigger validation.

**Fix:**

```bash
# Check if hook is executable
ls -la .git/hooks/pre-commit

# Make executable if needed
chmod +x .git/hooks/pre-commit
```

### Issue 2: "scripts/sync_frontend.sh not found"

**Problem:** Hook can't find validation script.

**Fix:**

```bash
# Ensure sync_frontend.sh exists
ls -la scripts/sync_frontend.sh

# Make executable if needed
chmod +x scripts/sync_frontend.sh
```

### Issue 3: Database Connection Error

**Problem:** Hook fails with "Cannot connect to database".

**Fix:**

```bash
# Start database
docker-compose up -d postgres

# Check connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"
```

### Issue 4: False Positives

**Problem:** Hook blocks valid commits.

**Temporary bypass:**

```bash
git commit --no-verify
```

**Permanent fix:** Report issue and update validation logic in `py/utils/frontend_engineering_agent.py`.

---

## Integration with Other Tools

### Pre-commit Framework

If using [pre-commit](https://pre-commit.com/):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: frontend-schema-validation
        name: Frontend Schema Validation
        entry: ./scripts/sync_frontend.sh validate-schema
        language: system
        pass_filenames: false
        always_run: false
        files: ^(dashboard|api)/.*\.py$
```

### Husky (for Node.js projects)

```json
{
  "husky": {
    "hooks": {
      "pre-commit": "./scripts/sync_frontend.sh validate-schema"
    }
  }
}
```

---

## Best Practices

### 1. Test Hooks Before Committing

```bash
# Dry run validation
./scripts/sync_frontend.sh validate-schema
```

### 2. Keep Schema Snapshot Up-to-Date

```bash
# After database migrations
./scripts/sync_frontend.sh save-snapshot
```

### 3. Don't Bypass Hooks Without Reason

Only use `--no-verify` when:
- Hook has known false positive
- Emergency hotfix required
- Database is temporarily unavailable

Always document why you bypassed the hook in commit message:

```bash
git commit --no-verify -m "fix: critical production bug

Bypassed pre-commit hook due to known false positive in validation.
See issue #123 for details."
```

### 4. Update Hooks Regularly

```bash
# Re-install hooks after pulling changes
./scripts/install_hooks.sh
```

---

## Related Documentation

- [FRONTEND_ENGINEERING_AGENT.md](../docs/FRONTEND_ENGINEERING_AGENT.md) - Complete guide
- [SQL_PATTERNS.md](../docs/database/SQL_PATTERNS.md) - Correct SQL patterns
- [DATABASE_QUICK_REFERENCE.md](../docs/database/DATABASE_QUICK_REFERENCE.md) - Schema reference

---

**Version:** 1.0.0
**Last Updated:** November 22, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
