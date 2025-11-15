# Architecture Decision Records (ADRs)

**Last Updated:** November 15, 2025
**Total ADRs:** 3 (November 2025 infrastructure decisions)
**Status:** Active

---

## What are ADRs?

Architecture Decision Records (ADRs) document significant architectural and design decisions made during the project. Each ADR captures:
- **Context:** Why the decision was needed
- **Decision:** What was chosen and why
- **Alternatives:** What other options were considered
- **Consequences:** Impacts (positive, negative, neutral, risks)
- **Implementation:** How the decision was executed

**Purpose:**
- Preserve institutional knowledge across team changes
- Document rationale for future reference ("Why did we do it this way?")
- Enable informed revisiting of decisions when context changes
- Prevent repeating past mistakes

---

## Index of ADRs

### Infrastructure & Deployment

- **[ADR-001: Docker Compose for Production Stack](./001-docker-compose-infrastructure.md)**
  - **Status:** Accepted (2025-11-08)
  - **Summary:** Use Docker Compose over Kubernetes for orchestrating 6 services (PostgreSQL, Redis, FastAPI, Streamlit, Prometheus, Grafana)
  - **Key Outcome:** 99% developer experience improvement (single command vs 30 manual commands)
  - **Tags:** infrastructure, deployment, docker, orchestration

- **[ADR-002: Git LFS for Model Artifact Storage](./002-git-lfs-migration.md)**
  - **Status:** Accepted (2025-11-14)
  - **Summary:** Migrate from Google Drive sync to Git LFS for version control of model artifacts (300 MB - 5 GB files)
  - **Key Outcome:** Removed 3,500 lines of sync code, native Git workflow, public deployment ready
  - **Tags:** infrastructure, version-control, ml-model, storage

### Machine Learning & Models

- **[ADR-003: XGBoost v4 Ridge Residuals Feature Engineering](./003-xgboost-v4-ridge-residuals.md)**
  - **Status:** Accepted (2025-11-14)
  - **Summary:** Two-stage architecture using Ridge regression for opponent baseline followed by XGBoost learning from residuals
  - **Key Outcome:** 88.31% improvement (40.20 → 4.70 MAE), statistically significant across 2022-2024
  - **Tags:** ml-model, feature-engineering, xgboost, prediction-accuracy

---

## ADR Status Definitions

| Status | Description |
|--------|-------------|
| **Proposed** | Decision under discussion, not yet finalized |
| **Accepted** | Decision finalized and implemented |
| **Rejected** | Decision considered but not chosen |
| **Deprecated** | Decision superseded by newer approach (but still in production) |
| **Superseded** | Decision replaced completely (no longer in production) |

---

## Creating New ADRs

### When to Create an ADR

Create an ADR for decisions that:
- **Impact architecture:** Service boundaries, data flow, technology choices
- **Are difficult to reverse:** Database schema, cloud provider, core algorithms
- **Have significant trade-offs:** Performance vs cost, simplicity vs flexibility
- **Set precedents:** Establish patterns for future similar decisions
- **Affect multiple teams:** Require coordination or shared understanding

**Examples of ADR-worthy decisions:**
- Choosing PostgreSQL over MongoDB
- Using Docker Compose vs Kubernetes
- Selecting XGBoost over Neural Networks
- Adopting microservices architecture
- Migrating to Git LFS for model storage

**Examples of non-ADR decisions:**
- Fixing a bug in data processing
- Updating a library version
- Refactoring a single module
- Changing a configuration parameter

### ADR Template

Use the standard template: [template.md](./template.md)

**Template includes:**
- Context (problem, background, decision drivers)
- Decision (what was chosen and why)
- Alternatives considered (with pros/cons)
- Consequences (positive, negative, neutral, risks)
- Implementation (timeline, success criteria)
- Related decisions (links to other ADRs)
- References (docs, code, papers)

### ADR Numbering

- Format: `ADR-XXX-descriptive-title.md`
- XXX = zero-padded sequential number (001, 002, 003, ...)
- Use kebab-case for descriptive title
- Examples:
  - `ADR-001-docker-compose-infrastructure.md`
  - `ADR-002-git-lfs-migration.md`
  - `ADR-003-xgboost-v4-ridge-residuals.md`

### Workflow

1. **Draft ADR:** Copy `template.md` to `ADR-XXX-title.md`
2. **Fill out template:** Document context, decision, alternatives
3. **Review:** Share with team for feedback
4. **Finalize:** Update status to "Accepted" and merge to main
5. **Update index:** Add entry to this README.md

---

## ADR Statistics

### By Category

| Category | Count | Examples |
|----------|-------|----------|
| Infrastructure | 2 | Docker Compose, Git LFS |
| Machine Learning | 1 | XGBoost v4 Ridge Residuals |
| Database | 0 | (Future: Schema design, migration strategy) |
| API/Services | 0 | (Future: REST vs GraphQL, authentication) |

### By Status

| Status | Count | Percentage |
|--------|-------|------------|
| Accepted | 3 | 100% |
| Proposed | 0 | 0% |
| Rejected | 0 | 0% |
| Deprecated | 0 | 0% |
| Superseded | 0 | 0% |

### Timeline

```
2025-11-08: ADR-001 (Docker Compose)
2025-11-14: ADR-002 (Git LFS Migration)
2025-11-14: ADR-003 (XGBoost v4 Ridge Residuals)
```

---

## Future ADRs (Planned)

### High Priority

1. **ADR-004: Ensemble v6 Configuration**
   - 4-way ensemble weights (State-Space 30%, Hierarchical 25%, XGBoost v4 20%, IP 25%)
   - Expected: Week of November 18, 2025

2. **ADR-005: A/B Testing Framework**
   - MD5 hash-based deterministic routing
   - Success criteria: MAE improvement >5 yards, p <0.05
   - Expected: Week of November 18, 2025

3. **ADR-006: Model Registry Database Schema**
   - ml.model_metadata table design
   - 227 pending model registrations
   - Expected: December 2025

### Medium Priority

4. **ADR-007: Production Deployment Strategy**
   - Cloud platform choice (AWS, GCP, Azure)
   - Single VM vs multi-host orchestration
   - Expected: January 2026

5. **ADR-008: Database Migration Strategy**
   - Schema change workflow
   - Zero-downtime migrations
   - Expected: December 2025

6. **ADR-009: Monitoring and Alerting Architecture**
   - Prometheus + Grafana configuration
   - Alert thresholds and escalation
   - Expected: January 2026

---

## References

### External Resources

- [ADR GitHub](https://adr.github.io/) - ADR methodology and tools
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) - Original ADR proposal (Michael Nygard)
- [ADR Template Markdown](https://github.com/joelparkerhenderson/architecture-decision-record) - Community templates

### Internal Documentation

- [ARCHITECTURE_REFERENCE.md](../ARCHITECTURE_REFERENCE.md) - System architecture overview
- [FILE_ORGANIZATION.md](../../FILE_ORGANIZATION.md) - Documentation placement rules
- [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md) - Full documentation index

---

## Best Practices

### Writing Good ADRs

1. **Be Specific:** Avoid vague statements. Quantify outcomes when possible.
   - ❌ Bad: "Docker Compose is easier to use"
   - ✅ Good: "Docker Compose reduces setup from 30 manual commands to 1 command (99% improvement)"

2. **Document Alternatives:** Show you considered other options (prevents "why didn't we try X?" questions)
   - List 3-5 realistic alternatives
   - Explain pros/cons objectively
   - State why each was rejected

3. **Capture Context:** Future readers need to understand the decision environment
   - What constraints existed? (budget, timeline, team size)
   - What information was available? (benchmarks, research)
   - What assumptions were made? (growth projections, user needs)

4. **Quantify Consequences:** Use metrics where possible
   - ✅ "Reduced codebase by 3,500 lines (8%)"
   - ✅ "88.31% improvement (40.20 → 4.70 MAE)"
   - ✅ "Developer onboarding time: 2-4 hours → 15 minutes"

5. **Link to Evidence:** Reference validation results, benchmarks, research
   - Link to validation reports
   - Cite research papers
   - Reference implementation code

6. **Acknowledge Risks:** No decision is risk-free - document and mitigate
   - List potential risks with likelihood and impact
   - Provide mitigation strategies
   - Set triggers for revisiting decision

### Updating ADRs

**ADRs are immutable once accepted.** If a decision changes:

1. **Create new ADR:** Don't edit the original
2. **Update status:** Mark old ADR as "Deprecated" or "Superseded"
3. **Cross-link:** New ADR should reference old one
4. **Explain why:** Document what changed in context

**Example:**
```
ADR-001: Docker Compose (Accepted)
  → ADR-010: Kubernetes Migration (Accepted)
     Status of ADR-001: Superseded by ADR-010
     Reason: Service count grew from 6 to 25 (exceeded Docker Compose scale)
```

### Review Cadence

**Quarterly Review (Every 3 months):**
- Review all "Accepted" ADRs
- Check if context has changed (new technologies, new requirements)
- Identify candidates for deprecation/superseding

**Annual Audit (Once per year):**
- Validate decisions still align with business goals
- Update consequences with actual outcomes
- Archive obsolete ADRs to `docs/adr/archive/`

---

## Maintenance

**ADR Custodian:** Project Team (rotating quarterly)

**Responsibilities:**
- Ensure new ADRs follow template
- Maintain this index (update counts, links)
- Facilitate quarterly reviews
- Archive obsolete ADRs

**Contact:** See CLAUDE.md for project context and conventions

---

**Version:** 1.0
**Last Updated:** 2025-11-15
**Maintainer:** Claude Code (Sonnet 4.5)
