# Sportsbook Lines Backfill Guide

**Purpose:** Backfill missing sportsbook betting lines using EPA-based estimates
**Script:** `scripts/backfill_sportsbook_lines.py`
**Last Updated:** November 21, 2025

## Quick Start

```bash
# Check what would be updated (dry run)
python scripts/backfill_sportsbook_lines.py --dry-run

# Backfill all missing lines
python scripts/backfill_sportsbook_lines.py

# Backfill specific season
python scripts/backfill_sportsbook_lines.py --season 2025

# Backfill specific week
python scripts/backfill_sportsbook_lines.py --season 2025 --week 12
```

## When to Use

**Use this script when:**
- New games added to schedule without betting lines
- Future games need placeholder lines for testing
- Historical games missing sportsbook data

**Do NOT use when:**
- Real sportsbook lines are available (use proper odds API instead)
- Retroactive historical analysis (placeholders don't reflect actual market)

## How It Works

### Methodology
1. **Queries missing games:** Finds all games with `spread_close IS NULL`
2. **Calculates EPA:** Gets recent 8-game EPA for both teams
3. **Estimates spread:** `(home_epa - away_epa) * 28 + 2.5 (home advantage)`
4. **Estimates total:** `45 (league avg) + (combined_epa) * 10`
5. **Converts to moneyline:** Standard spread-to-ML conversion table
6. **Updates database:** Sets all 8 betting line columns

### Data Sources
- **Team EPA:** `mart.team_epa` table (EPA per play, last 8 games)
- **Fallback:** Uses 0.0 (league average) if insufficient data

## Command Reference

### Basic Usage
```bash
python scripts/backfill_sportsbook_lines.py [OPTIONS]
```

### Options
| Option | Description | Example |
|--------|-------------|---------|
| `--season YEAR` | Filter to specific season | `--season 2025` |
| `--week NUM` | Filter to specific week | `--week 12` |
| `--dry-run` | Show what would change (don't commit) | `--dry-run` |
| `--help` | Show help message | `--help` |

### Examples

**Dry run before committing:**
```bash
python scripts/backfill_sportsbook_lines.py --season 2025 --week 12 --dry-run
# Output: DRY RUN: Would update 14 games
```

**Backfill single week:**
```bash
python scripts/backfill_sportsbook_lines.py --season 2025 --week 12
# Output: Successfully updated 14 games
```

**Backfill entire season:**
```bash
python scripts/backfill_sportsbook_lines.py --season 2025
# Output: Successfully updated 94 games
```

**Check all missing lines:**
```bash
python scripts/backfill_sportsbook_lines.py --dry-run
# Output: Found X games missing sportsbook data (across all seasons >= 2023)
```

## Output Format

### Console Logging
```
2025-11-21 05:45:16 - INFO - Found 14 games missing sportsbook data
2025-11-21 05:45:16 - INFO - Processing 2025_12_BUF_HOU: BUF @ HOU
2025-11-21 05:45:16 - INFO -   Spread: -4.5 | Total: 46.0 | ML: -200/+200
...
2025-11-21 05:45:16 - INFO - Successfully updated 14 games
```

### Database Updates
**Table:** `public.games`
**Updated Columns:**
- `spread_close` - Point spread (negative = home favored)
- `total_close` - Over/under total
- `home_moneyline` - Home team ML odds
- `away_moneyline` - Away team ML odds
- `home_spread_odds` - Home spread odds (-110)
- `away_spread_odds` - Away spread odds (-110)
- `over_odds` - Over odds (-110)
- `under_odds` - Under odds (-110)

## Validation Queries

### Check data completeness
```sql
SELECT
    season,
    COUNT(*) as total_games,
    COUNT(spread_close) as has_spread,
    ROUND(100.0 * COUNT(spread_close) / COUNT(*), 1) as completeness_pct
FROM games
WHERE season >= 2023
GROUP BY season
ORDER BY season DESC;
```

### Verify specific week
```sql
SELECT
    game_id,
    home_team,
    away_team,
    spread_close,
    total_close,
    home_moneyline,
    away_moneyline
FROM games
WHERE season = 2025 AND week = 12
ORDER BY game_id;
```

### Find biggest favorites
```sql
SELECT
    game_id,
    home_team || ' vs ' || away_team as matchup,
    spread_close,
    CASE
        WHEN spread_close < 0 THEN home_team || ' -' || ABS(spread_close)
        WHEN spread_close > 0 THEN away_team || ' -' || ABS(spread_close)
        ELSE 'Pick''em'
    END as favorite
FROM games
WHERE season = 2025 AND week = 12
ORDER BY ABS(spread_close) DESC;
```

## Quality Assurance

### Expected Ranges
- **Spread:** -14.0 to +14.0 (typical NFL range)
- **Total:** 40.0 to 55.0 (constrained by script)
- **Moneyline:** -600 to +600 (typical range)
- **Spread/Total Odds:** Always -110

### Known Issues

**LAR Team Abbreviation:**
```
WARNING - No EPA data for LAR, using 0.0
```
- **Cause:** LAR (Los Angeles Rams) not found in `mart.team_epa`
- **Impact:** Defaults to league average EPA (0.0)
- **Fix:** See `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md` for LA abbreviation standardization

### Validation Checklist
- [ ] All games have non-NULL `spread_close`
- [ ] All games have non-NULL `total_close`
- [ ] All games have non-NULL moneylines
- [ ] Spreads are reasonable (-14 to +14)
- [ ] Totals are bounded (40 to 55)
- [ ] Moneylines correlate with spreads
- [ ] Odds set to -110 (standard juice)

## Integration

### API Usage
Once sportsbook lines are populated, API can calculate betting edges:

```python
# Example: Game prediction with betting edge
POST /predict/game
{
  "game_id": "2025_12_BUF_HOU",
  "season": 2025,
  "week": 12
}

# Response includes:
{
  "predicted_spread": -5.2,
  "market_spread": -4.5,
  "spread_edge": -0.7,  # Model predicts HOU by more than market
  ...
}
```

### Risk Management
Betting lines enable portfolio optimization:
- Kelly criterion sizing (model edge vs market odds)
- Exposure tracking (max liability per game)
- +EV bet filtering (positive expected value)

## Troubleshooting

### Script Errors

**ModuleNotFoundError:**
```bash
# Activate virtual environment first
source .venv/bin/activate
python scripts/backfill_sportsbook_lines.py
```

**Database Connection Error:**
```bash
# Verify PostgreSQL is running
docker ps | grep postgres
# Check connection string in script (default: localhost:5544)
```

**No Games Found:**
```bash
# Check if games already have lines
psql -c "SELECT season, week, COUNT(*) FROM games WHERE spread_close IS NULL GROUP BY season, week;"
```

### Data Quality Issues

**Unrealistic Spreads:**
- Check EPA data quality in `mart.team_epa`
- Verify team abbreviations match between games and team_epa tables
- Consider manual override for known mismatches

**Missing EPA Data:**
- Script logs warnings for teams with no EPA history
- Falls back to 0.0 (league average)
- Review team abbreviation consistency

## Maintenance

### Regular Tasks
1. **Weekly Backfill:** Run script on Sunday mornings for new games
2. **Validation:** Spot-check generated lines vs expert consensus
3. **Real Line Replacement:** Update placeholders with actual lines when available

### Monitoring
```sql
-- Check for missing lines
SELECT season, week, COUNT(*)
FROM games
WHERE spread_close IS NULL
  AND season >= 2023
GROUP BY season, week;

-- Audit line quality
SELECT
    season,
    MIN(spread_close) as min_spread,
    MAX(spread_close) as max_spread,
    MIN(total_close) as min_total,
    MAX(total_close) as max_total
FROM games
WHERE season = 2025
GROUP BY season;
```

## References

### Related Documentation
- **Backfill Report:** `docs/archive/2025_nov/SPORTSBOOK_BACKFILL_REPORT_NOV21.md`
- **Database Schema:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **EPA Methodology:** `docs/ADVANCED_BAYESIAN_V3.md`

### Algorithm Details
See `scripts/backfill_sportsbook_lines.py` for:
- EPA query logic
- Spread calculation formula
- Moneyline conversion table
- Total estimation methodology

## Support

**Questions?** Check the following:
1. Script docstrings: `scripts/backfill_sportsbook_lines.py`
2. Backfill report: `docs/archive/2025_nov/SPORTSBOOK_BACKFILL_REPORT_NOV21.md`
3. Database docs: `docs/database/DATABASE_QUICK_REFERENCE.md`

---

**Last Updated:** November 21, 2025
**Status:** Production Ready
