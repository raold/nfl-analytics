# NFL Analytics Database Migrations

**Generated:** 2025-12-07 17:22:22
**Total Migrations:** 62

## Migration Groups

| Group | Description | Count |
|-------|-------------|-------|
| baseline | Database baseline schema | 1 |
| core | Core infrastructure and extensions | 11 |
| data | Data tables and columns | 15 |
| predictions | Predictions and props schema | 9 |
| trading | Paper trading system | 3 |
| ml | Machine learning schema | 8 |
| tracking | Feature and model tracking | 4 |
| fixes | Bug fixes and corrections | 4 |
| dba | DBA utilities and audit | 3 |
| security | Security and authentication | 1 |
| perf | Performance optimizations | 1 |
| reconciliation | Schema reconciliation | 1 |
| validation | Schema validation | 1 |

---

## All Migrations

### Baseline: Database baseline schema

| Version | File | Description |
|---------|------|-------------|
| 000 | `000_complete_schema.sql` | Complete schema baseline |

### Core: Core infrastructure and extensions

| Version | File | Description |
|---------|------|-------------|
| 001 | `001_init.sql` | Initialize database |
| 002 | `002_timescale.sql` | Enable TimescaleDB |
| 003 | `003_mart_game_weather.sql` | Weather mart views |
| 004 | `004_advanced_features.sql` | Advanced feature tables |
| 005 | `005_enhance_mart_views.sql` | Enhanced mart views |
| 006 | `006_remove_weather_duplication.sql` | Clean weather duplicates |
| 007a | `007_data_quality_log.sql` | Data quality logging |
| 007b | `007_ab_test_monitoring.sql` | A/B test monitoring |
| 008a | `008_add_foreign_keys_simple.sql` | Basic foreign keys |
| 013 | `013_optimize_timescaledb.sql` | TimescaleDB optimization |
| 017 | `017_materialized_views.sql` | Materialized views |

### Data: Data tables and columns

| Version | File | Description |
|---------|------|-------------|
| 008b | `008_nextgen_stats.sql` | NextGen stats tables |
| 009a | `009_odds_optimization.sql` | Odds optimization |
| 009b | `009_advanced_metrics.sql` | Advanced metrics |
| 010a | `010_data_standardization_layer.sql` | Data standardization |
| 010b | `010_complete_nflverse_data.sql` | Complete nflverse data |
| 011 | `011_team_logos.sql` | Team logos |
| 012 | `012_consolidate_team_mappings.sql` | Team mappings |
| 026d | `026_add_nflreadr_alignment_columns.sql` | nflreadr alignment |
| 027b | `027_add_critical_columns.sql` | Critical columns |
| 028 | `028_add_defensive_player_columns.sql` | Defensive columns |
| 029 | `029_add_situational_drive_columns.sql` | Situational columns |
| 030 | `030_add_phase4_critical_columns.sql` | Phase 4 columns |
| 031 | `031_add_phase5_high_signal_columns.sql` | Phase 5 columns |
| 050a | `050_add_coaching_staff_tables.sql` | Coaching staff |
| 900 | `add_international_venues.sql` | International venues |

### Predictions: Predictions and props schema

| Version | File | Description |
|---------|------|-------------|
| 018 | `018_predictions_schema.sql` | Predictions schema (primary) |
| 019a | `019_prop_lines_table.sql` | Prop lines table |
| 019b | `019_v2_model_features.sql` | V2 model features |
| 020 | `020_prop_predictions_table.sql` | Prop predictions |
| 023 | `023_player_hierarchy_schema.sql` | Player hierarchy |
| 024 | `024_bayesian_prop_predictions.sql` | Bayesian props |
| 048a | `048_create_game_predictions.sql` | Game predictions |
| 048b | `048_prediction_accuracy_view.sql` | Accuracy view |
| 048e | `048_create_game_predictions_ADDENDUM.sql` | Predictions addendum |

### Trading: Paper trading system

| Version | File | Description |
|---------|------|-------------|
| 026a | `026_pinnacle_exchange_fields.sql` | Pinnacle fields |
| 026b | `026_paper_trading_schema.sql` | Paper trading schema |
| 027a | `027_paper_trading_tables.sql` | Paper trading tables |

### Ml: Machine learning schema

| Version | File | Description |
|---------|------|-------------|
| 025 | `025_model_registry.sql` | Model registry |
| 026c | `026_canonical_training_window.sql` | Training window |
| 032 | `032_ml_predictions.sql` | ML predictions |
| 033 | `033_fix_convergence_status.sql` | Convergence status fix |
| 049 | `049_adaptive_learning_tracking.sql` | Adaptive learning |
| 101 | `101_stat_type_performance.sql` | Stat type performance |
| 102 | `102_dirichlet_alpha_by_stat_type.sql` | Dirichlet alphas |
| 103 | `103_bayesian_update_log.sql` | Bayesian update log |

### Tracking: Feature and model tracking

| Version | File | Description |
|---------|------|-------------|
| 043 | `043_column_lineage_tracking.sql` | Column lineage |
| 044 | `044_add_ensemble_tracking_columns.sql` | Ensemble tracking |
| 046 | `046_ensemble_version_tracking.sql` | Ensemble versions |
| 047 | `047_r_model_predictions_lookup.sql` | R model lookup |

### Fixes: Bug fixes and corrections

| Version | File | Description |
|---------|------|-------------|
| 021 | `021_fix_week6_kickoff_times.sql` | Week 6 kickoff fix |
| 022 | `022_fix_all_week6_kickoffs.sql` | All Week 6 kickoffs |
| 048c | `048_fix_thursday_night_flag.sql` | Thursday flag fix |
| 048d | `048_create_views_compatibility_patch.sql` | Views compat patch |

### Dba: DBA utilities and audit

| Version | File | Description |
|---------|------|-------------|
| 014 | `014_add_backup_table_pk.sql` | Backup table PK |
| 015 | `015_dba_audit_log_table.sql` | DBA audit log |
| 016 | `016_player_id_mapping_view.sql` | Player ID mapping |

### Security: Security and authentication

| Version | File | Description |
|---------|------|-------------|
| 045 | `045_api_keys_table.sql` | API keys |

### Perf: Performance optimizations

| Version | File | Description |
|---------|------|-------------|
| 050b | `050_dashboard_performance_indexes.sql` | Dashboard indexes |

### Reconciliation: Schema reconciliation

| Version | File | Description |
|---------|------|-------------|
| 100 | `100_schema_reconciliation.sql` | Schema reconciliation |

### Validation: Schema validation

| Version | File | Description |
|---------|------|-------------|
| 999 | `verify_schema.sql` | Schema verification |

---

## Usage

```bash
# Check migration status
python scripts/run_migrations.py --status

# Apply pending migrations
python scripts/run_migrations.py --apply

# Dry run (validate without applying)
python scripts/run_migrations.py --dry-run

# Generate this documentation
python scripts/run_migrations.py --generate-docs
```

## Canonical Ordering

Migrations with duplicate numbers (e.g., 018_*, 026_*, 048_*) are resolved using
suffixes (a, b, c, d, e). The canonical order is defined in `scripts/run_migrations.py`.

## Notes

- Migrations 100+ use the schema_migrations tracking table
- Migrations 000-050 may have been applied manually (legacy)
- Always run `--dry-run` before `--apply` in production
