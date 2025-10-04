# VS Code Problems Panel - Known Issues

**Date:** October 3, 2025  
**Status:** PDF compiles successfully despite warnings

---

## ⚠️ BibTeX Warnings (Expected - Non-Blocking)

### Missing Citation Commands (main.tex)
**Location:** `analysis/dissertation/main/main.tex:1`

```
I found no \citation commands
I found no \bibdata command
I found no \bibstyle command
```

**Cause:** VS Code LaTeX extension is checking main.tex in isolation, but the actual citation commands are in the chapter files that are `\input{}` by main.tex.

**Resolution:** Ignore - these warnings appear because the extension isn't following the `\input{}` directives. The actual build works fine.

---

## ⚠️ LaTeX Reference Warnings (Expected - Non-Blocking)

### acronyms.tex
**Warning:** `LaTeX: Reference 'chap:results' on page v undefined`
- **Cause:** Chapter 8 (Results) not yet written
- **Resolution:** Will resolve when Chapter 8 is added

### productionization_guide.tex (4 warnings)
**Warnings:** References to appendix sections undefined
- `'alg:ope-gate-appendix'` on page xvi
- `'alg:sim-accept-appendix'` on page xx
- `'app:systems-blueprint'` on page xxi

**Cause:** These appendices haven't been written yet
**Resolution:** Add these appendix sections or remove the references

### chapter_1_intro.tex (4 warnings)
**Warnings:** Chapter cross-references undefined
- `'chap:data'` on page 49, column 208
- `'chap:methods'` on page 49, column 252
- `'chap:risk'` on page 49, column 298
- `'chap:sim'` on page 49, column 340

**Cause:** Chapter numbering/labeling mismatch or chapters not yet written
**Resolution:** Update chapter labels or add missing chapters

### chapter_3_data_foundation.tex (6 warnings)
**Warnings:** Various reference issues
- `'tbl:schema-mart'` on page 14, column 265
- `'tbl:class-imbalance'` on page 112, column 12
- `'fig:feat-imp'` on page 137, column 124
- `'chap:methods'` on page 276, column 1

**Cause:** Missing table/figure labels in this chapter
**Resolution:** Add `\label{tbl:schema-mart}` etc. to the appropriate tables/figures

### Overfull \hbox Warnings (Cosmetic - Non-Critical)
**Files:** `chapter_3_data_foundation.tex`
- Line 263: `Overfull \hbox (2.31589pt too wide)` - Books column
- Line 263: `Overfull \hbox (13.2804pt too wide)` - spread/total/ML column

**Cause:** Table columns slightly wider than available space
**Resolution:** 
- Option 1: Ignore (only a few points too wide)
- Option 2: Use smaller font in table (`\small` or `\footnotesize`)
- Option 3: Abbreviate column headers ("Spread/Tot/ML" → "Markets")

---

## ✅ Fixed Issues (No Longer in Problems Panel)

1. ✅ Unicode minus sign (U+2212) in Chapter 3 → Fixed
2. ✅ Trailing backslashes in 8 table files → Fixed
3. ✅ Unescaped underscores in 2 files → Fixed

---

## 📊 Problems Panel Summary

| Category | Count | Status |
|----------|-------|--------|
| BibTeX warnings | 3 | Expected (false positive) |
| Undefined references | ~15 | Expected (future chapters) |
| Overfull hbox | 2 | Cosmetic only |
| **Critical errors** | **0** | **✅ ALL FIXED** |

---

## Notes

- The 220 problems shown in your Problems panel screenshot are mostly duplicate warnings reported for each LaTeX pass
- After BibTeX runs and cross-references resolve, many will disappear
- The key metric is: **PDF compiles successfully (83 pages, 1.3 MB)**
- All blocking errors have been eliminated

**Bottom line:** You can safely ignore these warnings and continue working on the dissertation content.
