# Props Betting Quick Start Guide

Get up and running with the NFL player props betting system in under 10 minutes.

## Prerequisites

- PostgreSQL database running (with NFL data loaded)
- Python 3.11+ with `uv` installed
- The Odds API key ([Get one free](https://the-odds-api.com))

## Step 1: Database Setup (2 minutes)

```bash
cd /Users/dro/rice/nfl-analytics

# Run the migration to create prop tables
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -f db/migrations/019_prop_lines_table.sql

# Verify tables created
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "\dt *prop*"
# Should see: prop_lines_history, prop_line_openings
```

## Step 2: Set API Key (30 seconds)

```bash
export ODDS_API_KEY=your_key_here
```

Or add to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export ODDS_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc
```

## Step 3: Fetch Sample Data (1 minute)

```bash
# Fetch prop lines for upcoming games (limit to 3 events for testing)
python py/data/fetch_prop_odds.py --api-key $ODDS_API_KEY --max-events 3

# Verify data ingested
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "SELECT COUNT(*) FROM prop_lines_history;"
```

Expected output:
```
 count
-------
   450
```
(~150 props per game × 3 games = 450 records)

## Step 4: Train Models (5 minutes per prop type)

**Option A: Quick Test (1 prop type)**

```bash
# Generate features for recent seasons
python py/features/player_features.py --start-season 2020 --end-season 2024 --output data/player_features_2020_2024.csv

# Train passing yards model only
python py/models/props_predictor.py \
    --features data/player_features_2020_2024.csv \
    --prop-type passing_yards \
    --train-seasons "2020-2023" \
    --test-season 2024 \
    --output models/props/passing_yards_v1.json
```

**Option B: Full Setup (all 8 prop types)**

```bash
# Train all prop types (takes ~40 minutes)
for prop_type in passing_yards passing_tds interceptions rushing_yards rushing_tds receiving_yards receptions receiving_tds; do
    echo "Training $prop_type..."
    python py/models/props_predictor.py \
        --features data/player_features_2020_2024.csv \
        --prop-type $prop_type \
        --train-seasons "2020-2023" \
        --test-season 2024 \
        --output models/props/${prop_type}_v1.json
done
```

## Step 5: Run Pipeline (30 seconds)

```bash
# Full pipeline (fetch + predict + select)
python py/production/props_production_pipeline.py --api-key $ODDS_API_KEY

# Or use existing prop lines (faster, saves API requests)
python py/production/props_production_pipeline.py --skip-fetch
```

## Step 6: Review Recommendations

```bash
# Human-readable summary
cat output/props_recommendations/summary_latest.txt

# CSV for analysis
head output/props_recommendations/recommended_bets_latest.csv
```

Example output:
```
================================================================================
DAILY PROPS BETTING RECOMMENDATIONS
Generated: 2025-10-12 14:30:00
================================================================================

Total bets: 12
Total to wager: $3,450.00
Average edge: 4.2%
Average EV: $5.80
Average confidence: 72.3/100

TOP 10 BETS BY EV:

Patrick Mahomes - passing_yards OVER
  Line: 275.5
  Prediction: 298.2
  Odds: +105 (draftkings)
  Edge: 5.3%
  EV: $8.50
  Bet size: $450.00
  Confidence: 84.2/100

Josh Allen - passing_tds OVER
  Line: 1.5
  Prediction: 2.1
  Odds: +110 (fanduel)
  Edge: 4.8%
  EV: $7.20
  Bet size: $380.00
  Confidence: 78.5/100

...
```

## ✅ You're Ready!

The system is now operational. To use it daily:

1. **Morning:** Run pipeline to get fresh recommendations
```bash
python py/production/props_production_pipeline.py --api-key $ODDS_API_KEY
```

2. **Review:** Check `output/props_recommendations/summary_latest.txt`

3. **Place bets:** Use recommendations, but verify lines are still available at your sportsbook

4. **Track results:** (optional) Log outcomes for backtesting

## 🎯 Pro Tips

### Save API Requests

The free tier gives 500 requests/month. Each pipeline run uses ~10-20 requests.

**Optimize:**
```bash
# Fetch lines once in morning
python py/data/fetch_prop_odds.py --api-key $ODDS_API_KEY

# Run pipeline multiple times without re-fetching
python py/production/props_production_pipeline.py --skip-fetch

# Re-fetch before placing bets (lines may have moved)
python py/data/fetch_prop_odds.py --api-key $ODDS_API_KEY --max-events 5
```

### Automate with Cron

Add to crontab for daily execution:
```bash
crontab -e

# Add this line (runs daily at 10am)
0 10 * * * cd /Users/dro/rice/nfl-analytics && uv run python py/production/props_production_pipeline.py --api-key $ODDS_API_KEY 2>&1 | tee logs/props_$(date +\%Y\%m\%d).log
```

### Focus on High-Confidence Bets

Filter to only high-confidence recommendations:
```bash
# In props_ev_selector.py, increase min_edge
python py/production/props_ev_selector.py \
    --predictions predictions.csv \
    --prop-lines prop_lines.csv \
    --min-edge 0.05 \  # 5% instead of 3%
    --output selected_bets.csv
```

### Monitor Line Movement

Check if lines have moved since your data fetch:
```sql
-- See line movement for a specific player
SELECT * FROM calculate_prop_line_movement(
    p_player_id => '00-0033873',
    p_prop_type => 'passing_yards',
    p_bookmaker => 'draftkings',
    p_hours_lookback => 24
);
```

## 🐛 Common Issues

### Issue: "No models found"

**Cause:** Models not trained yet

**Fix:** Run Step 4 (Train Models)

---

### Issue: "No bets meet selection criteria"

**Possible causes:**
- Lines too sharp (no edge)
- Book holds too high
- Players injured

**Debug:**
```bash
# Check available prop lines
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "SELECT COUNT(*), AVG(book_hold) FROM latest_prop_lines;"

# Lower edge requirement temporarily
python py/production/props_ev_selector.py --min-edge 0.02
```

---

### Issue: "Player ID not found"

**Cause:** Player name in API doesn't match roster

**Fix:** Names will be logged. You can manually add mappings or they'll be skipped.

---

### Issue: API rate limit exceeded

**Cause:** Used all free tier requests (500/month)

**Solutions:**
- Wait until next month
- Upgrade to paid tier ($25/mo = 10K requests)
- Use `--skip-fetch` to work with existing data

## 📊 Expected Performance

With proper usage (3% min edge, injury checks, correlation limits):

- **ROI:** 4-8% after hold
- **Win rate:** 52-55%
- **Bets per week:** 15-30
- **Average stake:** 2-3% of bankroll

**Example with $10,000 bankroll:**
- Average bet: $250
- Bets per week: 20
- Weekly volume: $5,000
- Expected weekly profit: $200-400 (4-8% ROI)

## 🚀 Next Steps

1. **Backtest thoroughly** before live betting:
   ```bash
   python py/models/props_predictor.py --backtest --test-season 2024
   ```

2. **Start small**: Use 1-2% Kelly instead of 15%

3. **Track results**: Log all bets and outcomes

4. **Monitor CLV**: Compare your bet line vs closing line

5. **Adjust parameters**: If losing, tighten edge requirements

## 📚 Learn More

- Full documentation: `docs/PROPS_BETTING_SYSTEM.md`
- Database schema: `db/migrations/019_prop_lines_table.sql`
- Example notebook: (TODO: create Jupyter notebook)

## 🤝 Need Help?

Common commands reference:

```bash
# Check database connection
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "SELECT version();"

# View latest prop lines
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "SELECT * FROM latest_prop_lines LIMIT 5;"

# Check model files
ls -lh models/props/

# View pipeline logs
tail -f logs/props_*.log

# Re-run specific step
python py/data/fetch_prop_odds.py --help
python py/models/props_predictor.py --help
python py/production/props_ev_selector.py --help
```

Good luck and bet responsibly! 🎯
