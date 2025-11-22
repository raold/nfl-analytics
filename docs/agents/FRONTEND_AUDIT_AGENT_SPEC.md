# Frontend Audit Agent Specification

**Created**: November 21, 2025
**Status**: Design Phase
**Purpose**: Automate comprehensive frontend quality assurance

---

## Problem Statement

**Current State**: Manual, reactive UI/UX issue detection via screenshots
- Issues found piecemeal (white text, missing data, low predictions)
- No systematic validation across all pages
- Quality degrades over time without notice
- Requires manual screenshots → investigate → fix → verify cycle

**Desired State**: Automated, proactive frontend auditing
- Systematic crawl of all dashboard pages
- Automated detection of UI/UX issues
- Data completeness validation
- Visual consistency checks
- Accessibility compliance
- Generates comprehensive audit reports

---

## Agent Capabilities

### 1. Page Discovery & Crawling ✅
- **Auto-discover** all Streamlit pages in `dashboard/`
- **Navigate** programmatically using Streamlit testing framework
- **Screenshot capture** for visual regression testing
- **State management** (season/week selections, filters)

### 2. UI/UX Issue Detection 🎨
- **Contrast Ratios** - WCAG AA compliance (4.5:1 for normal text)
- **White-on-white** detection - Text with <3:1 contrast
- **Color-blind** simulation - Deuteranopia, protanopia, tritanopia
- **Font sizing** - Minimum 12px, headers >18px
- **Button states** - Hover, active, disabled properly styled
- **Responsive layout** - Mobile/tablet/desktop breakpoints

### 3. Data Validation 📊
- **Missing data** - NULL counts, empty tables, "No data" messages
- **Stale data** - Timestamps >24h old for real-time metrics
- **Data accuracy** - Cross-check with database
  - Example: Dashboard shows 14 games, DB has 14 games
  - Example: Predicted score matches `ml.game_predictions` table
- **Prediction sanity** - Realistic ranges (220-300 yards, 20-30 points)

### 4. Error Detection 🐛
- **Console errors** - JavaScript/React errors in browser console
- **HTTP errors** - Failed API calls, 404s, 500s
- **Loading states** - Spinners that never resolve
- **Broken images** - Team logos, player headshots
- **Database errors** - SQL query failures, connection timeouts

### 5. Visual Consistency 🎭
- **Color palette** - Standardized theme colors
- **Typography** - Consistent font families, sizes, weights
- **Spacing** - Padding/margin consistency (8px grid)
- **Component style** - Cards, tables, buttons match design system
- **Icons** - Consistent emoji vs SVG usage

### 6. Accessibility (WCAG 2.1 AA) ♿
- **Keyboard navigation** - Tab order, focus indicators
- **Screen reader** - ARIA labels, alt text
- **Color independence** - Info not conveyed by color alone
- **Text alternatives** - Charts have data tables
- **Semantic HTML** - Proper heading hierarchy

---

## Implementation Architecture

### Technology Stack
- **Selenium** - Browser automation (Chrome/Firefox headless)
- **Playwright** - Modern alternative (faster, better API)
- **Streamlit Testing** - Built-in `st.testing` module
- **PIL/Pillow** - Screenshot comparison, color analysis
- **Beautiful Soup** - HTML parsing
- **Accessibility Insights** - WCAG compliance scanning

### Audit Workflow

```
┌──────────────────────────────────────────┐
│ 1. Page Discovery                        │
│    - Find all .py files in dashboard/   │
│    - Extract page titles from metadata  │
└──────────┬───────────────────────────────┘
           │
┌──────────▼───────────────────────────────┐
│ 2. Automated Navigation                  │
│    - Launch Streamlit in test mode      │
│    - Navigate to each page              │
│    - Interact with widgets (dropdowns)  │
└──────────┬───────────────────────────────┘
           │
┌──────────▼───────────────────────────────┐
│ 3. Visual Capture                        │
│    - Full-page screenshot               │
│    - Element-level snapshots            │
│    - Accessibility tree extraction      │
└──────────┬───────────────────────────────┘
           │
┌──────────▼───────────────────────────────┐
│ 4. Analysis Pipeline                     │
│  ┌────────────────────────────────────┐ │
│  │ UI/UX Analyzer                     │ │
│  │  - Contrast ratio calculator       │ │
│  │  - Color blind simulator           │ │
│  │  - Layout consistency checker      │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Data Validator                     │ │
│  │  - Query database for ground truth │ │
│  │  - Compare displayed vs expected   │ │
│  │  - Sanity check predictions        │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Error Detector                     │ │
│  │  - Console log scraping            │ │
│  │  - HTTP traffic monitoring         │ │
│  │  - Exception catching              │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ Accessibility Scanner              │ │
│  │  - WCAG 2.1 AA compliance          │ │
│  │  - Keyboard navigation test        │ │
│  │  - Screen reader compatibility     │ │
│  └────────────────────────────────────┘ │
└──────────┬───────────────────────────────┘
           │
┌──────────▼───────────────────────────────┐
│ 5. Report Generation                     │
│    - Markdown audit report              │
│    - Annotated screenshots              │
│    - Priority-ranked issues             │
│    - Actionable fix suggestions         │
└──────────────────────────────────────────┘
```

### Output Format

**Audit Report Structure** (`docs/audits/frontend_audit_YYYYMMDD.md`):

```markdown
# Frontend Audit Report

**Date**: November 21, 2025
**Pages Audited**: 12
**Issues Found**: 47
**Critical**: 3 | **High**: 12 | **Medium**: 18 | **Low**: 14

---

## Executive Summary

- ❌ **3 Critical Issues** - Block production deployment
- ⚠️  **12 High Priority** - Degrade user experience significantly
- ℹ️  **32 Lower Priority** - Polish and consistency improvements

### Top Issues
1. **White text on light background** (Game Predictions page, line 2254) - WCAG FAIL
2. **Missing game scores** (Week 12 games) - Data completeness FAIL
3. **Low QB predictions** (~100 yards vs expected ~270) - Sanity check FAIL

---

## Page-by-Page Breakdown

### 🎯 Game Predictions vs Sportsbook

**Overall Score**: 6.2/10 ⚠️

#### UI/UX Issues (5)
1. ❌ **CRITICAL** - White text contrast <3:1 (Model label, line 2254)
   - Current: `color: #6c757d` on `background: rgba(0,0,0,0.03)` = 2.8:1
   - Required: 4.5:1 for WCAG AA
   - Fix: Change to `color: #495057` or darken background

2. ⚠️ **HIGH** - Inconsistent card styling
   - Completed games use white background
   - Live games use yellow background (#fff3cd)
   - Scheduled games mix both
   - Fix: Standardize on semantic colors (green=good, yellow=caution, white=neutral)

...

#### Data Validation (3)
1. ❌ **CRITICAL** - Unrealistic predictions (12/14 games)
   - Josh Allen: 104.9 yards (expected ~270)
   - Root cause: Missing R model predictions for Week 12
   - Fix: Implement future game prediction in R models

...

#### Accessibility (2)
1. ⚠️ **MEDIUM** - Table headers missing scope attributes
   - `<th>` tags need `scope="col"` for screen readers
   - Fix: Add to lines 1995-1998, 2133-2136, 2217-2220

...

---

## Recommendations

### Immediate Actions (Critical)
1. Fix white text contrast violations
2. Generate R predictions for future games OR disable ensemble until available
3. Add missing screen reader attributes

### Short-term (High Priority)
1. Standardize card styling across game states
2. Implement data freshness indicators
3. Add loading state error handling

### Long-term (Medium/Low Priority)
1. Design system documentation
2. Component library refactor
3. Automated visual regression tests

---

## Automated Fix Script

For issues that can be fixed programmatically:

\`\`\`bash
# Run auto-fixer (dry run)
python scripts/audit_frontend.py --fix --dry-run

# Apply fixes
python scripts/audit_frontend.py --fix --apply
\`\`\`

Auto-fixable issues:
- Contrast ratio violations (color adjustments)
- Missing ARIA labels (attribute injection)
- Inconsistent spacing (CSS standardization)

```

---

## Usage

### CLI Interface

```bash
# Full audit (all pages)
python scripts/audit_frontend.py

# Specific page only
python scripts/audit_frontend.py --page "Game Predictions"

# Filter by severity
python scripts/audit_frontend.py --severity critical,high

# Auto-fix mode
python scripts/audit_frontend.py --fix --apply

# CI/CD integration
python scripts/audit_frontend.py --ci --fail-on critical
```

### Python API

```python
from py.agents.frontend_auditor import FrontendAuditor

# Initialize auditor
auditor = FrontendAuditor(
    dashboard_url="http://localhost:8501",
    db_config=DB_CONFIG,
    output_dir="docs/audits/"
)

# Run audit
report = auditor.audit_all_pages()

# Check specific page
issues = auditor.audit_page("Game Predictions")

# Get critical issues only
critical = report.filter_by_severity("critical")

# Generate annotated screenshots
auditor.generate_visual_report(highlight_issues=True)
```

---

## Integration Points

### Pre-Commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python scripts/audit_frontend.py --ci --fail-on critical
```

### GitHub Actions
```yaml
name: Frontend Audit
on: [pull_request]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Frontend Audit
        run: python scripts/audit_frontend.py --ci
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: audit-report
          path: docs/audits/
```

### Weekly Cron (Monitoring)
```bash
# crontab -e
0 6 * * 1 cd /path/to/nfl-analytics && python scripts/audit_frontend.py --email-report
```

---

## Success Metrics

- **Coverage**: 100% of dashboard pages audited
- **Detection Rate**: >95% of known issues caught automatically
- **False Positive Rate**: <5% (minimize noise)
- **Execution Time**: <5 minutes for full audit
- **Actionability**: >80% of issues have auto-fix available

---

## Future Enhancements

### Phase 2: Visual Regression Testing
- Baseline screenshot library
- Pixel-diff detection
- Layout shift monitoring
- Cross-browser testing (Chrome, Firefox, Safari)

### Phase 3: Performance Auditing
- Page load time profiling
- Bundle size analysis
- Lighthouse scores
- Core Web Vitals monitoring

### Phase 4: User Flow Testing
- Critical path validation (view prediction → check accuracy)
- Form submission testing
- Filter interaction testing
- Multi-page workflows

---

**Status**: ✅ DESIGN COMPLETE - Ready for implementation
**Estimated Effort**: 8-12 hours initial build, 2-4 hours/week maintenance
**Expected ROI**: 50+ hours saved over 6 months (vs manual screenshot debugging)
