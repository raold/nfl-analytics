#!/bin/bash
# Script to reorganize root-level documentation files per FILE_ORGANIZATION.md
# Created: November 15, 2025
# Purpose: Move all .md and .txt files from root to appropriate docs/ subdirectories

set -e

echo "================================================"
echo "Documentation Cleanup - November 15, 2025"
echo "================================================"
echo

# Create necessary directories
echo "Creating directory structure..."
mkdir -p docs/archive/2025_nov
mkdir -p docs/milestones
mkdir -p docs/audits
mkdir -p docs/performance
mkdir -p docs/deployment
mkdir -p docs/guides
mkdir -p docs/knowledge
mkdir -p docs/optimization
mkdir -p docs/backtest
mkdir -p docs/fixes

echo "Moving files to appropriate locations..."
echo

# Audit files → docs/audits/
echo "📁 Audit files → docs/audits/"
mv -v ADR_AUDIT_SUMMARY.txt docs/audits/ 2>/dev/null || true
mv -v CLAUDE_MD_AUDIT_*.* docs/audits/ 2>/dev/null || true
mv -v CODE_AUDIT_SUMMARY.md docs/audits/ 2>/dev/null || true
mv -v ARCHITECTURE_AUDIT_*.md docs/audits/ 2>/dev/null || true
mv -v C4_ARCHITECTURE_AUDIT_SUMMARY.md docs/audits/ 2>/dev/null || true
mv -v TEST_AUDIT_*.md docs/audits/ 2>/dev/null || true
mv -v TEST_COVERAGE_*.md docs/audits/ 2>/dev/null || true
mv -v MODEL_MANAGEMENT_AUDIT_NOV15.md docs/audits/ 2>/dev/null || true
mv -v POSTMORTEM_AUDIT_COMPLETE_NOV15.md docs/audits/ 2>/dev/null || true

# API Documentation → docs/archive/2025_nov/ (delivery reports)
echo
echo "📁 API Documentation → docs/archive/2025_nov/"
mv -v API_DOCUMENTATION_*.md docs/archive/2025_nov/ 2>/dev/null || true

# Performance reports → docs/performance/
echo
echo "📁 Performance reports → docs/performance/"
mv -v MCMC_OPTIMIZATION_NOV15_2025.md docs/performance/ 2>/dev/null || true

# Deployment/Testing → docs/deployment/
echo
echo "📁 Deployment/Testing → docs/deployment/"
mv -v AB_TEST_FRAMEWORK_TESTING_COMPLETE.md docs/deployment/ 2>/dev/null || true
mv -v EMERGENCY_DATABASE_RECOVERY.md docs/deployment/ 2>/dev/null || true

# Backtest reports → docs/backtest/
echo
echo "📁 Backtest reports → docs/backtest/"
mv -v BACKTEST_2024_VALIDATION_REPORT.md docs/backtest/ 2>/dev/null || true
mv -v BFS_FINAL_RESULTS.md docs/backtest/ 2>/dev/null || true

# Ensemble health → docs/milestones/
echo
echo "📁 Ensemble health → docs/milestones/"
mv -v ENSEMBLE_HEALTH_*.* docs/milestones/ 2>/dev/null || true
mv -v ENSEMBLE_V6_BACKTEST_QUICK_START.txt docs/milestones/ 2>/dev/null || true

# Infrastructure/Index files → docs/archive/2025_nov/
echo
echo "📁 Infrastructure/Index files → docs/archive/2025_nov/"
mv -v INFRASTRUCTURE_FIXES_INDEX.md docs/archive/2025_nov/ 2>/dev/null || true
mv -v DOCUMENTATION_INDEX.md docs/archive/2025_nov/ 2>/dev/null || true
mv -v DATA_PIPELINE_DOCUMENTATION_INDEX.md docs/archive/2025_nov/ 2>/dev/null || true
mv -v AUDIT_RESULTS_INDEX.md docs/archive/2025_nov/ 2>/dev/null || true

# R Model fixes → docs/archive/2025_nov/
echo
echo "📁 R Model fixes → docs/archive/2025_nov/"
mv -v R_MODEL_FIX_*.md docs/archive/2025_nov/ 2>/dev/null || true
mv -v ELO_BUG_FIX_VERIFICATION.md docs/fixes/ 2>/dev/null || true

# Tech debt → docs/knowledge/
echo
echo "📁 Tech debt → docs/knowledge/"
mv -v TECH_DEBT_*.md docs/knowledge/ 2>/dev/null || true

# Version comparison → docs/archive/2025_nov/
echo
echo "📁 Version comparison → docs/archive/2025_nov/"
mv -v VERSION_COMPARISON_v6.md docs/archive/2025_nov/ 2>/dev/null || true

# Test generation → docs/archive/2025_nov/
echo
echo "📁 Test generation → docs/archive/2025_nov/"
mv -v TEST_GENERATION_PLAN_80PCT.md docs/archive/2025_nov/ 2>/dev/null || true

# Architecture executive summary → docs/
echo
echo "📁 Architecture executive summary → docs/"
mv -v ARCHITECTURE_AUDIT_EXECUTIVE_SUMMARY.md docs/ 2>/dev/null || true

echo
echo "================================================"
echo "✅ Cleanup complete!"
echo "================================================"
echo
echo "Summary:"
echo "  - Audit files → docs/audits/"
echo "  - Session summaries → docs/archive/2025_nov/"
echo "  - Performance reports → docs/performance/"
echo "  - Backtest reports → docs/backtest/"
echo "  - Deployment guides → docs/deployment/"
echo "  - Bug fixes → docs/fixes/"
echo "  - Knowledge synthesis → docs/knowledge/"
echo
echo "Remaining root files:"
ls -la *.md *.txt 2>/dev/null | grep -v "README\|CLAUDE\|SETUP\|CONTRIBUTING\|FILE_ORGANIZATION\|LICENSE" || echo "  None (all cleaned up!)"
echo

# Move remaining files
echo
echo "Moving remaining files..."
echo

# Milestones
mv -v MODEL_REGISTRY_COMPLETE.md docs/milestones/ 2>/dev/null || true
mv -v PHASE_1_2_CRITICAL_FIXES_COMPLETE.md docs/milestones/ 2>/dev/null || true
mv -v XGBOOST_V4_INTEGRATION_COMPLETE.md docs/milestones/ 2>/dev/null || true

# Fixes/Investigations
mv -v R_MODEL_NAN_INVESTIGATION_REPORT.md docs/fixes/ 2>/dev/null || true
mv -v TRACK1_FIXES_SUMMARY.md docs/fixes/ 2>/dev/null || true

# Session summaries
mv -v SESSION_FINAL_SUMMARY_NOV14.md docs/archive/2025_nov/ 2>/dev/null || true
mv -v PROJECT_STATUS.md docs/archive/2025_nov/ 2>/dev/null || true

# Deployment/Validation
mv -v XGBOOST_V4_VALIDATION_COMPLETE.md docs/deployment/ 2>/dev/null || true
mv -v XGBOOST_V4_VALIDATION_SUMMARY.md docs/deployment/ 2>/dev/null || true

# Guides
mv -v XGBOOST_DIAGNOSTIC_QUICK_START.md docs/guides/ 2>/dev/null || true
mv -v CLAUDE_MD_CORRECTIONS_NEEDED.md docs/archive/2025_nov/ 2>/dev/null || true

echo "✅ All remaining files moved!"
