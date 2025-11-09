# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting significant architectural and methodological decisions for the NFL Analytics project.

## What are ADRs?

Architecture Decision Records capture important architectural decisions made during development, including:
- The context and problem being addressed
- The decision made and alternatives considered
- The rationale for the decision
- The consequences (both positive and negative)

ADRs provide a historical record of **why** decisions were made, not just **what** was decided.

## ADR Format

Each ADR follows a standard template:

```markdown
# ADR-XXX: [Decision Title]

**Status**: [Proposed|Accepted|Deprecated|Superseded]
**Date**: [Decision date]

## Context
[What is the issue we're seeing?]

## Decision
[What we're doing about it]

## Consequences
[What happens as a result]
```

## Index of ADRs

### Accepted

- **[ADR-001: Canonical Training Window](ADR-001-canonical-training-window.md)** (Nov 8, 2025)
  - **Decision**: Use 2018-2024 (full weight) + 2015-2017 (H=4 exponential decay) for all ensemble models
  - **Impact**: +0.5-1.0 pp ROI improvement, fair model comparison, 47.77 yards MAE
  - **Related**: State-Space v2 approval, ensemble weight optimization

### Proposed

*(No proposed ADRs)*

### Superseded

*(No superseded ADRs)*

## Creating a New ADR

1. Copy the template:
   ```bash
   cp ADR-TEMPLATE.md ADR-XXX-short-title.md
   ```

2. Assign the next available number (XXX)

3. Fill in all sections:
   - Context (problem statement)
   - Options considered (with pros/cons)
   - Decision rationale
   - Consequences (positive and negative)
   - Success metrics
   - Related decisions

4. Submit for review before accepting

5. Update this README index

## ADR Lifecycle

```
Proposed → Accepted → [Deprecated OR Superseded]
         ↓
    Rejected
```

- **Proposed**: Under discussion, not yet implemented
- **Accepted**: Approved and being implemented
- **Deprecated**: No longer recommended, but still in use
- **Superseded**: Replaced by a newer ADR
- **Rejected**: Considered but not accepted

## Related Documentation

- **Audit Reports**: `/docs/milestones/DISSERTATION_*_AUDIT.md`
- **Decision Docs**: `/docs/*_DECISION.md` (standalone decision documents)
- **Milestones**: `/docs/milestones/` (implementation completion records)

## Guidelines

### When to Create an ADR

Create an ADR for decisions that:
- Affect the project's overall architecture
- Have significant impact on ROI or performance
- Require non-trivial tradeoffs
- Will be difficult to change later
- Need to be understood by future developers

### When NOT to Create an ADR

Do NOT create ADRs for:
- Routine implementation details
- Code style preferences
- Temporary workarounds
- Obvious technical choices

### Best Practices

1. **Be Specific**: Provide concrete metrics and validation results
2. **Show Your Work**: Document alternatives considered and why they were rejected
3. **Quantify Impact**: Use numbers (MAE, ROI, p-values) not vague terms ("better", "faster")
4. **Link to Code**: Reference actual implementations
5. **Update When Needed**: If consequences change, update the ADR

## Contact

For questions about ADRs, see:
- **PROJECT_STATUS.md**: Current project status
- **CLAUDE.md**: Comprehensive project documentation
- **docs/ARCHITECTURE_REFERENCE.md**: System architecture
