#!/usr/bin/env bash
#
# Security Audit Script for NFL Analytics
#
# Scans for:
# - Hardcoded credentials (passwords, API keys, sicillionbillions)
# - Docker security issues (privileged mode, exposed ports)
# - API security (missing auth, weak CORS)
# - File permissions (sensitive files world-readable)
# - Dependency vulnerabilities (pip-audit)
#
# Usage:
#   ./scripts/security_audit.sh              # Run full audit
#   ./scripts/security_audit.sh --save       # Save report to JSON
#   ./scripts/security_audit.sh --weekly     # Weekly automated scan

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=================================="
echo "NFL Analytics Security Audit"
echo "=================================="
echo "⚠️  Real money betting requires rigorous security"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Run security auditor
python3 -c "
import sys
sys.path.insert(0, '.')

from nfl_analytics.utils.security_auditor import SecurityAuditor

auditor = SecurityAuditor()

print('Running comprehensive security audit...')
print('This may take a few minutes...')
print()

report = auditor.run_full_audit()

print(auditor.generate_report())

# Save report if requested
if '--save' in sys.argv or '--weekly' in sys.argv:
    auditor.save_report()
    print()
    print('✓ Report saved to security_audit_report.json')

# Exit with error if critical issues found
if report.has_critical_issues():
    print()
    print('⛔ CRITICAL SECURITY ISSUES DETECTED')
    print('   Immediate action required before production deployment')
    sys.exit(1)
else:
    print()
    print('✓ No critical security issues found')
    if report.high_count > 0:
        print(f'  ⚠️  {report.high_count} high-priority issues should be addressed')
" "$@"
