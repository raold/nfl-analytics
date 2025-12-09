# Enterprise Agents Archive

This directory contains agents that were built for team/multi-tenant/cloud deployment
but are not needed for single-user laptop development.

## Archived Agents

### security_auditor.py (~577 LOC)
**Reason:** No cloud deployment, no multi-tenant, no attack surface  
**Features:** Credential scanning, Docker security, API authentication  
**Notes:** Excellent code, just not needed for local development

### ab_test_monitor.py (~1,200 LOC)  
**Reason:** Not running A/B tests on laptop  
**Features:** Real-time experiment tracking, guardrails, SPRT, Prometheus metrics  
**Notes:** Production-ready, but overkill for solo research

### cache_invalidator.py (~550 LOC)
**Reason:** Caching complexity not needed for local development  
**Features:** Redis cache invalidation, dependency tracking  
**Notes:** May restore if caching becomes critical

## When to Restore

If you ever deploy to cloud or need multi-user support, these agents are ready to go.
They are well-tested and production-ready.

## Total LOC Saved

~3,800 LOC removed from active codebase (20% reduction)

