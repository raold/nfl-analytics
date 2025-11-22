# Market Microstructure Features - Deferred Opportunity

**Status**: 🔴 **DEFERRED** (requires $500/month API subscription)
**Created**: November 22, 2025
**Priority**: Medium-High (revisit after Track A-F completion)
**Expected Impact**: +2-3% ROI improvement
**Timeline**: 6-9 weeks implementation (if approved)

---

## Executive Summary

Market microstructure analysis involves tracking real-time betting line movements, sharp money indicators, and steam moves to identify value before lines tighten. This opportunity was **deferred** due to $500/month API subscription cost, but documented for future implementation after current 6-track parallel execution completes.

**Key Value Proposition**: Capture early line value by detecting sharp money BEFORE public catches up.

---

## Background

### User Decision (November 22, 2025)

**Question**: Implement market microstructure features using real-time odds APIs?

**Answer**: **NO** - "Skip market microstructure (no $500/month API)"

**Context**: Part of 7 modeling opportunities analysis. User approved 6/7 tracks for parallel execution, deferring only this one due to cost.

**Rationale**:
- Cost: $500/month for OddsJam API (or similar)
- Current focus: Exhaust all modeling improvements before adding paid data feeds
- Strategic: Revisit after Track A-F complete (8 weeks, ~Jan 17, 2026)

---

## Technical Overview

### What is Market Microstructure?

Market microstructure studies how betting markets discover prices through order flow, liquidity, and information asymmetry. In sports betting:

1. **Sharp Money**: Bets placed by professional bettors (high win rate, large stakes)
2. **Steam Moves**: Rapid line movements across multiple sportsbooks (indicates sharp consensus)
3. **Line Shopping**: Identifying best available odds across 20+ sportsbooks
4. **Closing Line Value (CLV)**: Beating the closing line consistently (gold standard metric)

### Why It Matters for NFL Props

**Inefficiency Window**: Props lines are softer than main markets (spread/total). Sharp money creates temporary inefficiencies:

| Time | Line Status | Opportunity |
|------|-------------|-------------|
| **Opening** | Soft (bookmaker initial guess) | **HIGH** - First 1-2 hours |
| **Early Action** | Sharp money moves line | **MEDIUM** - Can follow sharp steam |
| **Closing** | Efficient (balanced by sharp action) | **LOW** - Hard to beat |

**Example**:
- **Tuesday 9 AM**: Patrick Mahomes passing yards line opens at 275.5
- **Tuesday 9:30 AM**: Sharp money hits UNDER 275.5 at Pinnacle (steam detected)
- **Tuesday 10 AM**: Line moves to 272.5 across all books
- **Opportunity**: Bet UNDER at 275.5 in 9:00-9:30 window (+3 yards CLV)

---

## Implementation Plan (Deferred)

### Phase 1: Data Infrastructure (Weeks 1-2)

**Goal**: Ingest real-time odds data, store in TimescaleDB

**Components**:
1. **API Integration** (`py/data/market_microstructure_api.py`, ~400 lines)
   - OddsJam API client (REST + WebSocket)
   - Poll every 30 seconds for line updates
   - Store in `ml.market_odds` table (hypertable)

2. **Database Schema** (`db/migrations/050_market_microstructure.sql`)
   ```sql
   CREATE TABLE ml.market_odds (
       timestamp TIMESTAMPTZ NOT NULL,
       player_id VARCHAR(50) NOT NULL,
       sportsbook VARCHAR(50) NOT NULL,
       market_type VARCHAR(20) NOT NULL, -- 'passing_yards', 'rushing_yards', etc.
       line_value NUMERIC(6,2) NOT NULL,
       over_odds INTEGER NOT NULL,
       under_odds INTEGER NOT NULL,
       PRIMARY KEY (timestamp, player_id, sportsbook, market_type)
   );

   SELECT create_hypertable('ml.market_odds', 'timestamp');
   ```

3. **Feature Engineering** (`py/features/market_microstructure_features.py`, ~500 lines)
   - Line velocity: `d(line)/dt` (points per hour)
   - Sharp indicator: Pinnacle/Bookmaker movement
   - Steam detection: 3+ books move ≥1.5 points in <5 minutes
   - CLV calculation: Our prediction vs closing line

**Deliverables**:
- API client with retry logic, rate limiting
- Real-time data pipeline (30-second updates)
- 8 base features engineered

### Phase 2: Sharp Money Detection (Weeks 3-4)

**Goal**: Identify steam moves and sharp consensus

**Features** (10 total):
1. **Steam Indicators**:
   - `steam_detected` (boolean): ≥3 books move ≥1.5 points in <5 min
   - `steam_direction` (1/-1): Over or under steam
   - `steam_magnitude` (float): Average points moved across books

2. **Sharp Book Consensus**:
   - `pinnacle_line_diff` (float): Pinnacle vs consensus (sharp book)
   - `bookmaker_line_diff` (float): Bookmaker vs consensus
   - `sharp_consensus_direction` (1/-1): Sharp books aligned

3. **Line Movement Velocity**:
   - `line_velocity_1h` (float): Points moved in last 1 hour
   - `line_velocity_24h` (float): Points moved since open
   - `reverse_line_movement` (boolean): Line moved AGAINST public %

4. **Liquidity Indicators**:
   - `sportsbook_count` (int): How many books offering line
   - `max_bet_limit` (float): Largest accepted wager (from API)

**Algorithm**:
```python
def detect_steam(odds_df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """
    Detect steam moves (sharp consensus line movements).

    Args:
        odds_df: DataFrame with columns [timestamp, sportsbook, line_value]
        window_minutes: Time window for steam detection (default 5)

    Returns:
        DataFrame with steam_detected, steam_direction, steam_magnitude
    """
    # Group by timestamp, calculate line changes
    line_changes = odds_df.groupby('timestamp').agg({
        'line_value': ['mean', 'std', 'count']
    })

    # Detect: ≥3 books moved ≥1.5 points in <5 minutes
    steam_mask = (
        (line_changes[('line_value', 'count')] >= 3) &
        (line_changes[('line_value', 'std')] >= 1.5) &
        (line_changes.index.diff() <= pd.Timedelta(minutes=window_minutes))
    )

    steam_df = pd.DataFrame({
        'steam_detected': steam_mask,
        'steam_direction': np.sign(line_changes[('line_value', 'mean')].diff()),
        'steam_magnitude': line_changes[('line_value', 'std')]
    })

    return steam_df
```

**Deliverables**:
- Steam detection algorithm (90%+ precision target)
- Sharp consensus identification (Pinnacle + Bookmaker)
- 10 features engineered

### Phase 3: CLV Optimization (Weeks 5-6)

**Goal**: Maximize Closing Line Value (CLV)

**Strategy**:
1. **Predict Closing Line**: Train XGBoost to forecast where line will close
   - Input: Current line, sharp indicators, historical movement patterns
   - Output: Predicted closing line (±2.5 yards 80% confidence)

2. **Optimal Bet Timing**: Place bets when `current_line - predicted_closing_line > 2.0`
   - Example: Current 275.5, predicted closing 272.0 → Bet UNDER now (+3.5 CLV)

3. **CLV Validation**: Backtest on 2022-2024 historical odds data
   - Target: +1.5-2.0 yards average CLV
   - Acceptance: CLV > 0 on 65%+ of bets

**Model Architecture**:
```python
# XGBoost Closing Line Predictor
features = [
    'current_line',
    'pinnacle_line_diff',
    'bookmaker_line_diff',
    'line_velocity_1h',
    'line_velocity_24h',
    'steam_detected',
    'steam_magnitude',
    'reverse_line_movement',
    'time_to_kickoff_hours',
    'day_of_week'
]

target = 'closing_line'  # Line value at kickoff

model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

**Deliverables**:
- Closing line predictor (MAE <1.5 yards)
- Optimal timing algorithm
- CLV backtest report (2022-2024)

### Phase 4: Ensemble Integration (Weeks 7-9)

**Goal**: Integrate market microstructure features into Ensemble v7.5/v8.0

**Approach**:
1. **Add Market Features to XGBoost**: 10 microstructure features → XGBoost v6 (retraining)
2. **CLV-Weighted Ensemble**: Boost components with high historical CLV
3. **Bet Timing Gate**: Only place bets when CLV ≥ +2.0 yards

**Expected Impact**:
- MAE: Minimal change (market features don't predict actual performance)
- **ROI: +2-3%** (better bet timing + CLV capture)
- Win Rate: +1-2pp (avoid negative CLV bets)

**Deliverables**:
- XGBoost v6 with market features
- CLV-weighted ensemble
- Production betting pipeline with timing gate

---

## Cost-Benefit Analysis

### Costs

| Item | Cost | Frequency |
|------|------|-----------|
| **OddsJam API** | $500/month | Monthly |
| **Development Time** | 6-9 weeks | One-time |
| **Infrastructure** | Minimal (TimescaleDB hypertable) | One-time |
| **Maintenance** | 2-4 hours/week | Ongoing |
| **TOTAL (Year 1)** | **$6,000 + dev time** | Annual |

### Benefits (Projected)

| Metric | Baseline (v6.0) | With Market Microstructure | Improvement |
|--------|-----------------|----------------------------|-------------|
| ROI | +6-9% | **+8-12%** | **+2-3%** |
| Average CLV | 0 yards | **+1.5-2.0 yards** | New metric |
| Win Rate | 58-59% | **59-61%** | **+1-2pp** |
| Bet Volume | 100% | 100% | Same |

**Break-Even Analysis**:
- Monthly cost: $500
- Required ROI improvement: +0.5% (at $100,000 annual betting volume)
- Expected ROI improvement: +2-3%
- **Break-even at**: ~$25,000 annual betting volume (easily achieved)

**Net Annual Value** (at $100k volume):
- ROI improvement: +2.5% × $100k = **+$2,500/year**
- Cost: $6,000/year
- **Net loss**: -$3,500/year (NOT PROFITABLE at current scale)

**Profitable at**: $240,000+ annual betting volume

---

## API Providers Comparison

### Option 1: OddsJam (Recommended)

**Price**: $500/month
**Coverage**: 20+ sportsbooks, real-time WebSocket
**Features**:
- Player props (passing, rushing, receiving, TDs)
- Historical odds data (2+ years)
- CLV tracking built-in
- Steam move alerts

**Pros**: Best props coverage, reliable uptime (99.5%)
**Cons**: Most expensive

### Option 2: The Odds API

**Price**: $200/month (500 requests/hour)
**Coverage**: 15+ sportsbooks, REST polling only
**Features**:
- Player props support
- Historical data (1 year)
- No built-in CLV tracking

**Pros**: Lower cost, adequate for testing
**Cons**: No WebSocket (30-60 second polling delay), less historical data

### Option 3: BetOnline API

**Price**: $300/month
**Coverage**: BetOnline only, real-time
**Features**:
- Props coverage
- Line history (6 months)

**Pros**: Mid-tier pricing
**Cons**: Single book (can't detect steam across books)

**Recommendation**: Start with The Odds API ($200/month) for Phase 1-2 validation, upgrade to OddsJam if CLV backtest shows +2%+ ROI.

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **API downtime** | Medium | High | Fallback to polling, cache last known lines |
| **Rate limiting** | Low | Medium | Implement exponential backoff, local caching |
| **Data quality issues** | Medium | High | Validate line consistency across books |
| **Steam detection false positives** | High | Medium | Tune thresholds via backtesting, require 90%+ precision |

### Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Books adjust to our edge** | Medium | High | Diversify across 5+ sportsbooks, use bet masking |
| **CLV doesn't translate to ROI** | Low | High | Validate on 2+ years historical data first |
| **Soft lines disappear** | Low | Medium | Focus on player props (slower market efficiency) |

### Financial Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **$500/month cost not justified** | **HIGH** | Medium | START with The Odds API ($200/month), validate ROI first |
| **Break-even requires $240k+ volume** | Medium | High | Grow betting volume before implementing |
| **Opportunity cost** | Low | Low | Implement after Track A-F (higher ROI opportunities) |

---

## Success Criteria

### Phase 1-2 (API + Features, Weeks 1-4)

- [ ] API uptime: 99%+ (excluding planned maintenance)
- [ ] Data latency: <60 seconds (polling) or <5 seconds (WebSocket)
- [ ] Steam detection precision: 90%+ (validated on 2022-2024 data)
- [ ] Sharp consensus identification: Pinnacle + Bookmaker movement correlated

### Phase 3 (CLV Optimization, Weeks 5-6)

- [ ] Closing line predictor MAE: <1.5 yards (80% confidence)
- [ ] Average CLV: +1.5-2.0 yards (backtested 2022-2024)
- [ ] CLV positive rate: 65%+ of bets
- [ ] Bet timing optimization: CLV improvement vs random timing

### Phase 4 (Ensemble Integration, Weeks 7-9)

- [ ] ROI improvement: +2-3% vs baseline (backtested 2022-2024)
- [ ] Win rate improvement: +1-2pp
- [ ] No MAE degradation (market features don't predict performance)
- [ ] Production pipeline: <500ms inference latency

### Financial (Year 1)

- [ ] ROI > +8% (baseline +6%, need +2% improvement to justify cost)
- [ ] Net profit > $1,000/year (after $6,000 API cost)
- [ ] Break-even achieved by Month 6

---

## Integration with Current Roadmap

### Current 6-Track Parallel Execution (Nov 22 - Jan 17, 2026)

**Tracks**:
- A: Player Projection Aggregation (Week 2 complete)
- B: Rushing/Receiving Props Expansion (Week 3 complete)
- C: Causal Inference Phase 1 (Week 6 complete)
- D: RL 4th Down Decisions (Week 8 complete)
- E: Dirichlet Bayesian Stacking (Week 4 complete)
- F: Drive-Level Prediction (Week 4 complete)

**Market Microstructure Timing**:
1. **Jan 17, 2026**: All 6 tracks complete, Ensemble v8.0 deployed
2. **Jan 20-31**: Backtest v8.0 on full 2022-2024 dataset
3. **Feb 1**: Evaluate v8.0 performance:
   - If ROI ≥ +14%: **Market microstructure becomes TOP PRIORITY**
   - If ROI < +10%: Focus on Track C/D/E/F refinements first

**Rationale**: Exhaust free improvements before paying $500/month for marginal gains.

### Ensemble Versioning

- **v6.1** (Week 2, Dec 6): v6.0 + player aggregation
- **v7.0** (Week 4, Dec 20): v6.1 + Dirichlet stacking
- **v7.5** (Week 6, Jan 3): v7.0 + causal features
- **v8.0** (Week 8, Jan 17): v7.5 + all Track A-F improvements
- **v8.5** (Week 16, Mar 7): **v8.0 + market microstructure** (IF APPROVED)

---

## Dissertation Relevance

### Chapter Alignment

Market microstructure does NOT fit well into current dissertation structure:

**Current Chapters**:
1. Introduction
2. Literature Review
3. Data Foundation
4. Baseline Modeling
4b. Advanced Bayesian Methods
5. Advanced ML (GNN, NLP)
6. Uncertainty Quantification
7. Reinforcement Learning (4th Down)
8. Results & Discussion
9. Production Engineering
10. Lessons Learned

**Potential Addition**: **Chapter 5.5: Market Microstructure & Information Asymmetry**
- Focus: How betting markets aggregate information
- Contribution: Detecting sharp money via line movements
- **Problem**: More finance/econometrics than ML/stats
- **Decision**: Likely NOT dissertation-worthy (practical tool, not research contribution)

**Alternative**: Brief mention in Chapter 9 (Production Engineering) as a production enhancement, not a core research contribution.

---

## Alternatives Considered

### Alternative 1: Build Our Own Odds Scraper

**Cost**: $0 (DIY)
**Effort**: 4-6 weeks development + legal risk
**Legal Risk**: **HIGH** - Terms of Service violations, potential lawsuits

**Verdict**: ❌ **NOT RECOMMENDED** (legal risk outweighs cost savings)

### Alternative 2: Free Odds Sources

**Options**: ESPN API, Yahoo Sports, public feeds
**Coverage**: Limited (often main lines only, not props)
**Update Frequency**: 5-10 minutes (too slow for steam detection)

**Verdict**: ❌ **INSUFFICIENT** (need real-time props data)

### Alternative 3: Delay Until Betting Volume Scales

**Rationale**: Wait until $250k+ annual volume before adding $6k/year cost
**Timeline**: 12-18 months (organic growth)
**Opportunity**: Focus on free improvements (Track A-F) first

**Verdict**: ✅ **BEST APPROACH** (current user decision)

---

## Documentation Maintenance

### Owner

**Primary**: Model Management Subagent
**Secondary**: Prediction Engine Subagent

### Update Triggers

- [ ] Quarterly review (March 1, June 1, Sept 1, Dec 1)
- [ ] When betting volume exceeds $200k annual
- [ ] When API pricing changes (OddsJam, The Odds API)
- [ ] After v8.0 deployment (Jan 17, 2026 → Feb 1 decision point)

### Related Documentation

- **Parallel Roadmap**: `docs/PARALLEL_MODELING_ROADMAP_NOV22.md`
- **Advanced Features**: `docs/research/ADVANCED_FEATURE_ENGINEERING_ROADMAP.md`
- **Ensemble Versioning**: `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md`

---

## Next Steps

### Immediate (Nov 22, 2025)

- [x] Document opportunity for future reference
- [ ] No action (DEFERRED)

### Future Decision Point (Feb 1, 2026)

**IF** v8.0 ROI ≥ +14% AND betting volume ≥ $100k:
1. Subscribe to The Odds API ($200/month trial)
2. Implement Phase 1-2 (API + Features)
3. Backtest CLV on 2022-2024 data
4. **GO/NO-GO Decision**: If backtested ROI improvement ≥ +2%, proceed to Phase 3-4

**ELSE**:
- Defer indefinitely
- Focus on Track C/D/E/F refinements
- Revisit when volume ≥ $250k

---

## Appendix: Example Use Case

### Scenario: Patrick Mahomes Passing Yards (Hypothetical)

**Tuesday 9:00 AM**:
- Line opens: 275.5 yards
- Pinnacle: 275.5 (-110/-110)
- DraftKings: 275.5 (-108/-112)
- FanDuel: 276.0 (-110/-110)

**Tuesday 9:15 AM**:
- **STEAM DETECTED**: Pinnacle moves to 272.5, Bookmaker to 273.0, Circa to 273.5
- Sharp indicator: Pinnacle + Bookmaker moved UNDER simultaneously
- Our system: **BET UNDER 275.5** (before other books adjust)

**Tuesday 9:30 AM**:
- All books move to 272.5-273.5
- **Our CLV**: +3.0 yards (bet at 275.5, closing 272.5)

**Sunday Result**:
- Mahomes: 268 yards
- **Win**: UNDER 275.5 ✅ (CLV helped us beat closing line by 4.5 yards)

**Value**: Even if our model predicted 274 yards (OVER), the market microstructure signal (sharp money going UNDER) provided additional information that improved our edge.

---

**Status**: 🔴 **DEFERRED** until Feb 1, 2026 (post-v8.0 evaluation)
**Next Review**: February 1, 2026 (after Track A-F completion)
**Owner**: Model Management Subagent
**Last Updated**: November 22, 2025
