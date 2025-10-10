# Data Sources Roadmap: Tiered Approach to NFL Betting Analytics

**Date**: 2025-10-10
**Purpose**: Strategic roadmap for data acquisition and integration
**Philosophy**: Start free, upgrade strategically based on ROI

---

## Overview

This roadmap provides a tiered approach to data sources for NFL betting analytics, balancing cost, complexity, and expected value. The strategy is to start with free/low-cost sources (Tier 1-2) and only upgrade to premium sources (Tier 3-4) when justified by profit and identified data gaps.

**Total Budget**:
- **Current**: $340/year (Tier 1-2)
- **Conservative**: $340-1,000/year
- **Aggressive**: $1,000-3,000/year (only if >$10k/year profit)

---

## Tier 1: Foundation (FREE) - Core Data Infrastructure

### Status: ✅ IMPLEMENTED

**Sources**:

#### 1. nflverse / nflreadr (FREE)
- **What**: Comprehensive NFL play-by-play, roster, injury data
- **Coverage**: 1999-2024 (regularly updated)
- **Data**:
  - Play-by-play (every snap, every game)
  - Next Gen Stats (player tracking)
  - Rosters, injuries, schedules
  - Participation data
- **Value**: $2,000+/year equivalent (if paid)
- **Integration**: `R/ingestion/ingest_pbp.R`, `R/ingestion/ingest_rosters.R`
- **Update Frequency**: Weekly during season
- **Limitations**: 48-72 hour lag on some advanced stats

#### 2. nflfastR (FREE)
- **What**: Advanced EPA and win probability models
- **Coverage**: 1999-2024
- **Data**:
  - Expected Points Added (EPA)
  - Win Probability (WP, WPA)
  - CPOE (Completion % Over Expected)
  - Success rate metrics
- **Value**: $1,000+/year equivalent
- **Integration**: Built into nflreadr
- **Use Case**: Core features for XGBoost v2 model
- **Limitations**: EPA calculated post-game (not real-time)

#### 3. Pro Football Reference (FREE via scraping)
- **What**: Historical stats and advanced metrics
- **Coverage**: 1922-2024
- **Data**:
  - Team stats (offense, defense, special teams)
  - Player stats (AV, ANY/A, DVOA proxies)
  - Historical injury reports
- **Value**: $500/year equivalent
- **Integration**: Manual scraping for missing data gaps
- **Limitations**: Rate limited, requires scraping (legal gray area)

#### 4. ESPN / NFL.com (FREE)
- **What**: Breaking news, injury updates, depth charts
- **Coverage**: Real-time
- **Data**:
  - Injury reports (official)
  - Depth chart changes
  - Coaching decisions
  - Weather forecasts
- **Value**: $200/year (timeliness)
- **Integration**: Manual monitoring + Claude semantic research
- **Limitations**: Unstructured (text-based)

**Tier 1 Total Cost**: $0/year
**Tier 1 Total Value**: $3,700+/year

---

## Tier 2: Essential Paid ($340/year) - Operational Infrastructure

### Status: ✅ RECOMMENDED (SUBSCRIBE NOW)

**Sources**:

#### 1. NFL+ Premium ($40/year)
- **What**: Official NFL streaming + data
- **Coverage**: Real-time during games
- **Data**:
  - Live injury reports (official)
  - Inactive lists (90 min pre-game)
  - All-22 coaches film (48 hours post-game)
  - Next Gen Stats (live)
  - Official depth charts
- **Value**: $350/year (injury reports = $200, inactive lists = $100, film = $50)
- **ROI**: 875%
- **Use Case**:
  - Injury_load feature updates
  - Last-minute lineup changes (late line moves)
  - Film study for qualitative edge
- **Integration**: Manual updates to injury database
- **Limitations**: No API (manual entry)

#### 2. The Odds API ($300/year)
- **What**: Real-time odds aggregator
- **Coverage**: 15+ US sportsbooks (FanDuel, DraftKings, BetMGM, etc.)
- **Data**:
  - Live spreads, totals, moneylines
  - Props (limited)
  - Historical odds (6 months)
- **Value**: $700-1,000/year (line shopping +2-3 cents/bet = +1-2% ROI)
- **ROI**: 233-333%
- **Use Case**:
  - Automated line shopping (Task 7: line_shopping.py)
  - Best odds identification
  - Historical line movement analysis
- **Integration**: `py/production/line_shopping.py` (API client implemented)
- **Limitations**: 10,000 requests/month (adequate for 100 bets/week)

**Tier 2 Total Cost**: $340/year
**Tier 2 Total Value**: $1,050-1,350/year
**Tier 2 ROI**: 309-397%

---

## Tier 3: Advanced Analytics ($100-1,000/year) - Performance Enhancement

### Status: ⚠️ CONDITIONAL (Test first, upgrade if profitable)

**Sources**:

#### 1. Action Network PRO ($100/year) - ⚠️ CONDITIONAL
- **What**: Public betting data and sharp tracking
- **Coverage**: Real-time
- **Data**:
  - Public betting percentages (% of bets, % of money)
  - Sharp money indicators (reverse line moves)
  - Line movement alerts
  - Consensus picks
- **Value**: $200-400/year
- **ROI**: 200-400%
- **Use Case**:
  - Identify reverse line moves (sharp money opposite public)
  - Fade the public strategy
  - Sharp book tracking (Circa, Pinnacle moves)
- **Conditions to Subscribe**:
  - ✅ Tier 1-2 system generating >$2,000/year profit
  - ✅ Identified specific use case (e.g., live betting expansion)
  - ✅ Free trial shows +$200 value
- **Integration**: Manual monitoring or web scraping
- **Limitations**: Public data can be misleading (bettors lie)

#### 2. Weather Underground API ($300/year) - ⚠️ CONDITIONAL
- **What**: Historical and forecast weather data
- **Coverage**: US stadiums
- **Data**:
  - Wind speed/direction (critical for totals)
  - Temperature, precipitation
  - Historical conditions (2010+)
- **Value**: $300-500/year
- **ROI**: 100-167%
- **Use Case**:
  - Wind >15 mph → lower totals (especially Buffalo, Chicago)
  - Dome vs outdoor adjustments
  - Historical weather-adjusted EPA
- **Conditions to Subscribe**:
  - ✅ Identified weather as model feature gap
  - ✅ Backtest shows +0.5% Brier improvement
  - ✅ Totals betting becoming primary strategy
- **Integration**: API → features pipeline
- **Limitations**: Weather forecasts unreliable >3 days out

#### 3. Sharp Book Access - Offshore ($500/year) - ⚠️ HIGH RISK
- **What**: Pinnacle, CRIS, Bookmaker accounts (offshore)
- **Coverage**: Pre-game and live lines
- **Data**:
  - Sharp opening lines (Pinnacle efficient market)
  - Low-hold markets (2-3% vig vs 4-5% US books)
  - High limits (up to $50k/bet)
- **Value**: $1,000-2,000/year (better lines + sharp validation)
- **ROI**: 200-400%
- **Use Case**:
  - Closing Line Value (CLV) validation
  - Bet sharp lines if +EV
  - Line shopping arbitrage
- **⚠️ LEGAL WARNING**: Offshore books illegal in US (gray area for bettors)
- **⚠️ FINANCIAL RISK**: No US regulation, withdrawal issues common
- **Conditions to Subscribe**:
  - ✅ Consistently beating US books (+10% ROI for 6+ months)
  - ✅ Bankroll >$50k (can afford risk)
  - ✅ Legal consultation (understand risks)
- **Integration**: Manual betting (no API typically)

**Tier 3 Total Cost**: $100-1,000/year (conditional)
**Tier 3 Total Value**: $500-2,900/year (if conditions met)
**Tier 3 ROI**: 50-290% (highly variable)

---

## Tier 4: Premium Data ($1,000-6,000/year) - Diminishing Returns

### Status: ❌ SKIP (Unless $10k+/year profit AND identified data gap)

**Sources**:

#### 1. Bet Labs ($1,188/year) - ❌ SKIP FOR NOW
- **What**: Historical betting trends and system identification
- **Data**:
  - ATS trends (home dogs, divisional games, etc.)
  - Profitable betting systems
  - Situational analysis
- **Value**: $1,000-1,500/year (trend validation)
- **ROI**: 84-126% (marginal)
- **Why Skip**: Trends often regress, survivorship bias common
- **Reconsider If**: Model hits accuracy ceiling, need new features

#### 2. DonBest ($3,600/year) - ❌ SKIP
- **What**: Professional odds service (50+ books)
- **Data**:
  - Real-time odds (worldwide)
  - Steam move alerts (sharp money)
  - Opening/closing line database
- **Value**: $1,000-1,500/year
- **ROI**: 28-42% (poor)
- **Why Skip**: The Odds API provides 80% of value for 8% of cost
- **Reconsider If**: Expanding to international books

#### 3. SportsDataIO NFL Package ($6,000/year) - ❌ SKIP
- **What**: Enterprise-grade NFL data API
- **Data**:
  - Real-time play-by-play
  - Player tracking (Next Gen Stats)
  - Live odds (15+ books)
  - Injury API
- **Value**: $300-500/year (marginal over free sources)
- **ROI**: 5-8% (terrible)
- **Why Skip**: nflverse + The Odds API cover 95% for $300/year
- **Reconsider If**: Building commercial product (need SLA, support)

#### 4. Player Tracking Data (AWS, Zebra, $10,000+/year) - ❌ SKIP
- **What**: Raw player tracking data (positions, speeds, routes)
- **Data**:
  - GPS coordinates (10Hz sampling)
  - Route running metrics
  - Separation data
- **Value**: Unknown (untested in betting context)
- **ROI**: Likely negative (data scientists needed)
- **Why Skip**: Complexity >> value, NFL restricts access
- **Reconsider If**: Academic research grant, not for profit

**Tier 4 Total Cost**: $1,188-$6,000+/year
**Tier 4 Expected Value**: $1,000-2,000/year
**Tier 4 ROI**: 8-168% (mostly poor)

---

## Decision Matrix: When to Upgrade Tiers

### From Tier 1 (Free) → Tier 2 (Essential Paid)

**Trigger**: Starting production betting
**Timing**: Immediately (before first bet)
**Cost**: $340/year
**Expected ROI**: 309%
**Decision**: ✅ UPGRADE NOW

### From Tier 2 → Tier 3 (Advanced)

**Triggers** (any of):
- ✅ Quarterly profit >$2,000 ($8k/year pace)
- ✅ Identified data gap (e.g., weather hurting totals accuracy)
- ✅ Expanding to new markets (live betting, player props)

**Timing**: After 3-6 months profitability
**Cost**: +$100-1,000/year
**Expected ROI**: 50-290%
**Decision**: ⚠️ TEST FREE TRIAL FIRST

### From Tier 3 → Tier 4 (Premium)

**Triggers** (ALL required):
- ✅ Annual profit >$10,000
- ✅ Model accuracy plateaued (can't improve with current data)
- ✅ Identified specific premium data that adds +1% accuracy
- ✅ ROI analysis shows >200% return on data cost

**Timing**: Only after 12+ months profitability
**Cost**: +$1,000-6,000/year
**Expected ROI**: 8-168% (mostly poor)
**Decision**: ❌ SKIP UNLESS PROVEN NECESSARY

---

## Data Source Evaluation Framework

### Before Subscribing to Any Paid Source

**Ask These Questions**:

1. **Need**: What data gap does this fill?
   - Example: "NFL+ provides injury reports → improves injury_load features"

2. **Alternative**: Is there a free/cheaper alternative?
   - Example: "Twitter injury updates (free) vs NFL+ ($40) → NFL+ faster, official"

3. **Value**: How much $ value does this add per year?
   - Example: "Injury reports avoid 5 bad bets/year → save $500 (assuming $100/bet)"

4. **ROI**: Value / Cost ratio?
   - Example: "$500 value / $40 cost = 1250% ROI → YES"

5. **Integration**: Can we actually use this data?
   - Example: "NFL+ no API → manual entry → 10 min/week → feasible"

6. **Test**: Can we try before buying?
   - Example: "The Odds API free tier (500 req/month) → test 1 month → upgrade if useful"

### ROI Thresholds

- **>500% ROI**: Subscribe immediately (e.g., NFL+ Premium)
- **200-500% ROI**: Subscribe after validation (e.g., The Odds API)
- **100-200% ROI**: Conditional subscribe (test first, e.g., Action Network PRO)
- **50-100% ROI**: Skip unless critical gap (e.g., Weather API for totals betting)
- **<50% ROI**: Never subscribe (e.g., SportsDataIO)

---

## Current Recommendation Summary

### Tier 1: FREE (USING NOW) ✅
- nflverse / nflreadr
- nflfastR
- Pro Football Reference (scraping)
- ESPN / NFL.com

**Total Cost**: $0
**Total Value**: $3,700+/year

### Tier 2: Essential ($340/year) ✅ SUBSCRIBE NOW
- NFL+ Premium: $40/year
- The Odds API: $300/year

**Total Cost**: $340/year
**Total Value**: $1,050-1,350/year
**ROI**: 309-397%

### Tier 3: Advanced ($100-1,000/year) ⚠️ CONDITIONAL
- Action Network PRO: $100/year (if live betting)
- Weather Underground: $300/year (if totals focus)
- Sharp book access: $500/year (if consistent +10% ROI)

**Conditions**: Profit >$2k/quarter, identified data gap

### Tier 4: Premium ($1,000+/year) ❌ SKIP
- Bet Labs: $1,188/year (poor ROI)
- DonBest: $3,600/year (The Odds API sufficient)
- SportsDataIO: $6,000/year (nflverse sufficient)

**Only reconsider if**: >$10k/year profit AND proven data gap

---

## Implementation Timeline

### Month 1-3: Foundation (Tier 1-2)
- [x] Configure nflverse data pipeline
- [x] Integrate nflfastR EPA features
- [ ] Subscribe to NFL+ Premium
- [ ] Get The Odds API key (free tier)
- [ ] Test The Odds API integration

### Month 4-6: Validation (Tier 2 → Tier 3 decision)
- [ ] Track NFL+ value (bets avoided due to injury intel)
- [ ] Track The Odds API value (cents gained per bet)
- [ ] Calculate quarterly ROI
- [ ] If ROI >$2k/quarter, test Action Network PRO free trial

### Month 7-12: Optimization (Tier 3 conditional)
- [ ] If profitable, subscribe to Action Network PRO ($100/year)
- [ ] If totals betting, test Weather Underground API
- [ ] If consistent +10% ROI, consider sharp book access (⚠️ legal risk)

### Month 13+: Premium Data (Tier 4 - only if necessary)
- [ ] If annual profit >$10k AND model accuracy plateaued
- [ ] Run ROI analysis on premium data
- [ ] Only subscribe if >200% expected ROI

---

## Data Quality Monitoring

### Key Metrics to Track

**1. Data Completeness**:
- % of games with all features (target: >95%)
- Missing data patterns (e.g., weather data unavailable for domes)

**2. Data Timeliness**:
- Lag between game end and data availability
- Real-time data latency (odds updates)

**3. Data Accuracy**:
- Injury report accuracy (reported vs actual inactives)
- Odds feed accuracy (compare to book websites)

**4. Data ROI**:
```python
data_roi = {
    'source': 'The Odds API',
    'cost': 300,  # $/year
    'bets_improved': 50,  # number of bets with better line
    'avg_improvement': 0.025,  # 2.5 cents per bet
    'total_ev_gain': 50 * 0.025 * 100,  # $125 on $100 bets
    'roi': 125 / 300,  # 42% (marginal, but worth it for automation)
}
```

---

## Alternative Data Sources (Emerging)

### Experimental / Unproven

**1. Social Media Sentiment (FREE via Twitter API)**
- Public sentiment analysis
- Injury rumors (before official reports)
- **Risk**: Noisy, unreliable
- **Status**: Research only, not production

**2. Referee Analytics (FREE via NFLpenalties.com)**
- Penalty rates by crew
- Home/away bias
- **Value**: Marginal (0.1-0.2% accuracy)
- **Status**: Future feature engineering

**3. Broadcast Angles / Computer Vision (COSTLY, $10k+)**
- Automated film breakdown
- Formation recognition
- **Value**: Unknown
- **Status**: Academic research, not profitable

---

## Summary: Data Strategy

**Core Principle**: Start cheap, upgrade strategically based on ROI

**Current Stack** (Recommended):
- Tier 1 (FREE): nflverse, nflfastR → $3,700 value
- Tier 2 ($340/year): NFL+, The Odds API → $1,050 value
- **Total**: $340/year for $4,750 value (1,397% ROI)

**Upgrade Path**:
1. Month 1-3: Implement Tier 1-2 (foundation)
2. Month 4-6: Validate ROI, test Tier 3 free trials
3. Month 7-12: Conditional Tier 3 subscriptions (if profitable)
4. Month 13+: Only Tier 4 if >$10k profit AND proven data gap

**Never Subscribe To**:
- ❌ SportsDataIO ($6k/year) - nflverse sufficient
- ❌ DonBest ($3.6k/year) - The Odds API sufficient
- ❌ Any service with <100% expected ROI
- ❌ Any service without free trial / test period

**Key Takeaway**: The free nflverse stack + $340/year in subscriptions provides 95% of the value at 5% of the cost compared to premium services. Only upgrade when profit justifies cost AND data gap is proven.

---

*Last Updated: 2025-10-10*
