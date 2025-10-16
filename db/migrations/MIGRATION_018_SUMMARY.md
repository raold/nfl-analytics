# Migration 018 - Database Consistency Fixes

**Date:** October 12, 2025
**Status:** COMPLETED
**Migration File:** `018_critical_fixes_v3.sql`

## Overview

This migration addresses critical database schema inconsistencies identified during the Bayesian model integration. The issues were causing join failures, data mismatches, and preventing proper integration of Bayesian predictions with the XGBoost pipeline.

## Problems Addressed

### 1. Team Abbreviation Inconsistencies
**Issue:** Los Angeles Rams were stored as 'LA' in nflverse data but expected as 'LAR' elsewhere
**Fix:** Created `reference.team_abbreviations` table with canonical mappings across all data sources

### 2. Missing Normalization Functions
**Issue:** No standardized way to convert between different team code conventions
**Fix:** Created `normalize_team_abbr()` function to handle LA→LAR and WSH→WAS conversions

### 3. Player ID Lookup Complexity
**Issue:** Multiple player ID systems (GSIS, PFR, ESPN) without unified lookup
**Fix:** Created `lookup_player_ids()` function to search across all identifier systems

### 4. Data Type Inconsistencies
**Issue:** Column types varied between TEXT and VARCHAR
**Fix:** Standardized to TEXT for:
- `injuries.gsis_id` (was VARCHAR(20))
- `injuries.team` (was VARCHAR(3))
- `mart.bayesian_player_ratings.player_id` (was VARCHAR(50))

### 5. Missing Primary Keys
**Issue:** `mart.rolling_features` had no primary key
**Fix:** Added composite primary key on (team, season, week)

### 6. Helper Views for Common Joins
**Issue:** Complex joins needed for player information lookups
**Fix:** Created two helper views:
- `v_rosters_with_player_info` - Combines rosters_weekly with player names
- `v_games_normalized` - Games with canonical team abbreviations

## Changes Applied

### Tables Created
1. **reference.team_abbreviations**
   - 32 NFL teams with mappings across all data sources
   - Columns: canonical_abbr, nflverse_abbr, espn_abbr, team_name
   - Primary key: canonical_abbr

### Functions Created
1. **normalize_team_abbr(TEXT) → TEXT**
   - Converts team codes to canonical form
   - LA → LAR
   - WSH → WAS
   - All others pass through unchanged

2. **lookup_player_ids(...) → TABLE**
   - Search for player across all ID systems
   - Returns: player_id, gsis_id, pfr_id, espn_id, player_name
   - Accepts any ID type as input

### Views Created
1. **v_rosters_with_player_info**
   - All rosters_weekly columns plus player_name from players table
   - Join on gsis_id = player_id

2. **v_games_normalized**
   - All games columns plus canonical team names and abbreviations
   - Uses normalize_team_abbr() function

### Schema Modifications
- `injuries.gsis_id`: Changed from VARCHAR(20) to TEXT
- `injuries.team`: Changed from VARCHAR(3) to TEXT
- `mart.bayesian_player_ratings.player_id`: Changed from VARCHAR(50) to TEXT
- `mart.rolling_features`: Added PRIMARY KEY (team, season, week)

### Indexes Added
- `idx_rosters_weekly_gsis` on rosters_weekly(gsis_id)
- `idx_players_player_id_hash` on players using HASH (player_id)
- `idx_team_abbr_nflverse` on reference.team_abbreviations(nflverse_abbr)

## Verification Results

All verification checks passed:
- ✓ 32 teams in team_abbreviations table
- ✓ LA→LAR mapping exists
- ✓ 2 helper functions created
- ✓ 2 helper views created
- ✓ Primary key added to rolling_features
- ✓ normalize_team_abbr('LA') returns 'LAR'

## Impact

### Immediate Benefits
1. **Eliminated join failures** - Consistent team codes across all tables
2. **Simplified player lookups** - Single function for all ID systems
3. **Data integrity** - Primary keys prevent duplicates in rolling_features
4. **Type safety** - Consistent TEXT types prevent implicit conversion issues

### For Bayesian Integration
The fixes directly address the errors encountered in `generate_bayesian_predictions.py`:
- Player ID lookups now work consistently
- Team code mismatches eliminated
- Helper views simplify common join patterns

### Performance
- Hash index on players.player_id for O(1) lookups
- Regular indexes on foreign key columns
- Views use efficient joins with proper indexes

## Usage Examples

### Team Normalization
```sql
-- Convert LA to LAR
SELECT normalize_team_abbr('LA');  -- Returns: 'LAR'

-- Get canonical team info
SELECT * FROM reference.team_abbreviations
WHERE nflverse_abbr = 'LA';
```

### Player Lookup
```sql
-- Find player by GSIS ID
SELECT * FROM lookup_player_ids(p_gsis_id => '00-0033873');

-- Find player by PFR ID
SELECT * FROM lookup_player_ids(p_pfr_id => 'MahoPa00');
```

### Helper Views
```sql
-- Get rosters with player names
SELECT season, week, team, full_name, players_name, position
FROM v_rosters_with_player_info
WHERE season = 2024 AND week = 6;

-- Get games with normalized teams
SELECT game_id, home_team, home_team_canonical, home_team_name
FROM v_games_normalized
WHERE season = 2024;
```

## Migration History

### Attempts
1. **018_database_consistency_fixes.sql** - Failed due to incorrect column name assumptions
2. **018_critical_fixes_v2.sql** - Failed due to function dependency on existing views
3. **018_critical_fixes_v3.sql** - SUCCESS - Properly handles view/function dependencies

### Key Lessons
- Always check actual schema with `\d table_name` before writing migrations
- Drop dependent views before dropping/recreating functions they depend on
- Use CASCADE carefully - better to explicitly drop dependencies
- Test team normalization functions immediately after creation

## Next Steps

1. Update existing code to use helper views where applicable
2. Gradually migrate all team-related queries to use normalize_team_abbr()
3. Update documentation to reference canonical team abbreviations
4. Consider adding CHECK constraints for team codes
5. Add foreign key constraints pointing to team_abbreviations table

## Files Modified
- `/Users/dro/rice/nfl-analytics/db/migrations/018_critical_fixes_v3.sql` - Final migration
- `/Users/dro/rice/nfl-analytics/db/audit/DBA_AUDIT_REPORT.md` - Audit findings

## References
- DBA Audit Report: `db/audit/DBA_AUDIT_REPORT.md`
- Migration Log: `db/migrations/018_migration_output.log`
- Verification Queries: Included in migration script STEP 10
