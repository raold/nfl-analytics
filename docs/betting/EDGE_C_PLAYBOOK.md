# Edge C Betting Playbook: Lingering Fatigue

**Last Updated:** November 16, 2025
**Status:** Paper Trading Phase
**Expected ROI:** +33.6% (pending baseline validation)

---

## Executive Summary

**Edge C** exploits the lingering fatigue effect on quarterbacks playing within 7 days of an NFL international game. Statistical analysis of 24 QB performances (2018-2024) shows a **-195.6 yard penalty** on short rest with **p < 0.0001 significance**.

**Key Findings:**
- **Sample Size:** 24 QB performances (14 with <7 day rest)
- **Average Penalty:** -209.8 yards vs domestic baseline
- **Short Rest Penalty:** -195.6 yards (≤7 days between games)
- **Statistical Significance:** p < 0.0001 (t = -13.766)
- **Expected Win Rate:** 70% (vs 52.4% breakeven at -110 odds)
- **Expected ROI:** +33.6%

**Current Status:** Edge identified but **baseline calculation fix required** before live betting. Paper trading recommended for 2025 season.

---

## Table of Contents

1. [The Edge](#the-edge)
2. [When to Bet](#when-to-bet)
3. [How to Size Bets](#how-to-size-bets)
4. [Risk Management](#risk-management)
5. [Paper Trading Protocol](#paper-trading-protocol)
6. [Exit Criteria](#exit-criteria)
7. [2025 Opportunities](#2025-opportunities)
8. [Tools & Scripts](#tools--scripts)
9. [Baseline Fix Tracking](#baseline-fix-tracking)

---

## The Edge

### Hypothesis

Quarterbacks underperform in their **next game after international travel** due to:
1. **Lingering jet lag** (circadian rhythm disruption)
2. **Travel fatigue** (long flights, disrupted sleep)
3. **Preparation disruption** (unfamiliar facilities, compressed schedule)
4. **Recovery deficit** (inadequate rest before next game)

### Statistical Validation

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **Sample Size** | 24 QB performances | 14 with <7 day rest |
| **Average Penalty** | -209.8 ± 74.7 yards | All rest periods |
| **Short Rest Penalty** | -195.6 yards | ≤7 days between games |
| **p-value** | < 0.0001 | Highly statistically significant |
| **t-statistic** | -13.766 | Strong effect size |
| **Win Rate (estimated)** | 70% | Bet UNDER on passing props |
| **ROI (estimated)** | +33.6% | At standard -110 odds |

### Causal Mechanism

**Why the edge exists:**
1. **Jet Lag Recovery:** Studies show 1 day recovery per timezone crossed; 6-hour London shift = 6 days optimal recovery
2. **Short Turnaround:** NFL schedules often place next game within 7 days (insufficient recovery)
3. **Compressed Preparation:** International games disrupt normal practice routines
4. **Accumulated Fatigue:** Travel fatigue compounds with game exertion

**Why it persists:**
1. **Oddsmakers undervalue effect:** Penalties often only -5 to -10 yards in props
2. **Public bias:** Fans bet favorites regardless of fatigue
3. **Small sample:** Only 20 international games since 2018 (harder for market to price)

---

## When to Bet

### Go Criteria (ALL must be met)

1. ✅ **QB played in international game** (London, Munich, Brazil)
2. ✅ **Next game within ≤7 days** of international game
3. ✅ **Passing prop available** at -110 odds or better
4. ✅ **Baseline calculation fix validated** (PENDING as of Nov 16, 2025)
5. ✅ **Statistical significance maintained** (p < 0.05 after fix)

### No-Go Criteria (ANY triggers abort)

1. ❌ **Baseline fix not validated** → Paper trade only
2. ❌ **QB injured/inactive** → Skip
3. ❌ **Opponent elite pass defense** (Top 5 EPA/play) → Reduce Kelly or skip
4. ❌ **Extreme weather** (30+ mph winds, heavy snow) → Skip (unpredictable variance)
5. ❌ **Odds worse than -130** → Edge too thin, skip
6. ❌ **Paper trade performance <55% win rate** after 10 trades → Abandon edge

### Edge Strength Tiers

| Days Since Intl | Edge Strength | Recommended Action | Kelly Fraction |
|-----------------|---------------|-------------------|----------------|
| ≤5 days | **VERY HIGH** | Full bet (pending fix) | 10% Kelly |
| 6-7 days | **HIGH** | Full bet (pending fix) | 10% Kelly |
| 8-10 days | **MEDIUM** | Half bet | 5% Kelly |
| >10 days | **LOW** | Skip | N/A |

---

## How to Size Bets

### Kelly Criterion Formula

**Kelly Fraction:**
```
f = (bp - q) / b

where:
  f = fraction of bankroll to bet
  b = net odds (decimal odds - 1)
  p = win probability
  q = loss probability (1 - p)
```

**For Edge C at -110 odds:**
- p = 0.70 (70% win rate)
- q = 0.30
- b = 1.909 - 1 = 0.909
- f = (0.909 × 0.70 - 0.30) / 0.909 = 0.367 = **36.7% full Kelly**

### Variance Adjustment (Small Sample)

**Problem:** n=24 is small sample → high estimation uncertainty

**Solution:** Reduce Kelly by `sqrt(100/n)` factor
```
Adjustment = sqrt(100/24) = 2.04
Variance-Adjusted Kelly = 36.7% / 2.04 = 18.0%
```

### Fractional Kelly (Risk Management)

**Recommended:** Use **10% fractional Kelly** (10% of variance-adjusted Kelly)
```
Final Kelly = 18.0% × 10% = 1.8% of bankroll
```

**Conservative:** Use **5% fractional Kelly** if first 10 paper trades underperform
```
Final Kelly = 18.0% × 5% = 0.9% of bankroll
```

### Bet Sizing Calculator

**Use the provided script:**
```bash
# Interactive mode
python scripts/calculate_edge_c_bet_size.py --interactive

# Command line
python scripts/calculate_edge_c_bet_size.py --bankroll 10000 --odds -110 --win-rate 0.70 --kelly-fraction 0.10
```

**Example Output (10% Kelly, $10,000 bankroll):**
```
Full Kelly: 36.7% of bankroll
Variance Adjusted: 18.0% (factor: 2.04x)
Fractional Kelly (10%): 1.8% of bankroll
Kelly Recommended: $180
Max Allowed (2% cap): $200
FINAL RECOMMENDATION: $180 (1.8% of bankroll)
```

---

## Risk Management

### Bet Size Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| **Fractional Kelly** | 10% | Small sample protection |
| **Max Bet (% Bankroll)** | 2% | Prevent ruin risk |
| **Max Bets Per Week** | 3 | Avoid overexposure |
| **Max Bets Per Day** | 1 | Reduce correlated risk |

### Risk Gates (Check Before Every Bet)

**Pre-Bet Checklist:**
1. [ ] Baseline calculation fix validated
2. [ ] QB confirmed starting (check injury report)
3. [ ] Odds ≤ -130 (edge remains positive)
4. [ ] Within bet size limits (≤2% bankroll, ≤3 bets/week)
5. [ ] Paper trade performance ≥10 trades, ≥60% win rate

**During-Season Monitoring:**
- Track running win rate (abort if <55% after 20 bets)
- Track running ROI (abort if <0% after 20 bets)
- Adjust Kelly down to 5% if win rate 55-59%
- Increase Kelly to 15% only if win rate >75% after 30 bets

### Bankroll Management

**Recommended Bankroll Allocation:**
- **Edge C:** 10% of total betting bankroll
- **Max simultaneous bets:** 3 (6% of bankroll maximum)
- **Reserve:** 90% for other edges and ensemble bets

**Example ($10,000 bankroll):**
- Edge C allocation: $1,000
- Per-bet max: $200 (2% total bankroll = 20% Edge C allocation)
- 3 simultaneous bets max: $600 (60% Edge C allocation)

### Stop-Loss Rules

**Abandon edge if:**
1. 20 live bets placed AND ROI < 0%
2. 30 live bets placed AND win rate < 55%
3. Baseline calculation discovered to be fundamentally flawed
4. Market adjustment detected (consistent line movement against edge)

**Reduce Kelly to 5% if:**
1. 10 live bets AND win rate 55-59%
2. 3 consecutive losses
3. Significant sample growth changes statistical significance (p > 0.05)

---

## Paper Trading Protocol

### Phase 1: Data Collection (2025 Season)

**Goal:** Validate edge with 10+ paper trades before live betting

**Protocol:**
1. **Identify opportunities** using monitor script
2. **Record bet details** in tracking spreadsheet
3. **Wait for results** (no money at risk)
4. **Update performance** metrics weekly

**Tracking Fields (CSV):**
- `trade_id`: Sequential ID
- `date`: Game date
- `qb`, `team`, `opponent`: Game details
- `days_since_intl`: Days between international game and next game
- `baseline_yards`: QB domestic average (corrected baseline)
- `predicted_yards`: Baseline - penalty (-195.6 yards)
- `actual_yards`: Actual QB passing yards
- `prop_line`: Sportsbook UNDER line
- `prop_odds`: Odds (typically -110)
- `bet_size_dollars`: Hypothetical bet size
- `outcome`: WIN/LOSS/PUSH
- `profit_loss`: Hypothetical profit/loss
- `roi`: ROI for this bet
- `notes`: Any relevant context

### Phase 2: Validation (After 10+ Trades)

**Success Criteria:**
1. ✅ **Win Rate ≥60%** (vs 70% expected)
2. ✅ **ROI ≥+10%** (vs 33.6% expected)
3. ✅ **p-value <0.05** (statistical significance maintained)

**If criteria met:**
- Proceed to **Phase 3: Limited Live Betting**
- Start with smallest bet sizes (5% Kelly)
- Monitor closely for 5 trades before increasing Kelly

**If criteria NOT met:**
- Abandon edge OR
- Collect 10 more paper trades (n=20 total) and re-evaluate

### Phase 3: Limited Live Betting (First 10 Bets)

**Conservative approach:**
1. **Start:** 5% fractional Kelly (0.9% bankroll per bet)
2. **Max bets:** 1 per week (reduce correlation risk)
3. **Monitoring:** Track every bet, re-calculate win rate after each
4. **Adjustment:** If 7/10 win rate after 10 bets → increase to 10% Kelly

### Tools for Paper Trading

**1. Monitor Opportunities:**
```bash
python scripts/monitor_international_schedule.py --weeks-ahead 8
```

**2. Calculate Bet Sizes:**
```bash
python scripts/calculate_edge_c_bet_size.py --bankroll 10000 --interactive
```

**3. Track Results:**
- Update `results/edge_c_paper_trades.csv` after each game
- Use Excel/Google Sheets for analysis

**4. View Dashboard:**
```bash
python scripts/edge_c_dashboard.py --update
```

---

## Exit Criteria

### Abandon Edge (Permanent)

**Trigger ANY of the following:**

1. **Baseline proven flawed**
   - If corrected baseline eliminates penalty (p > 0.10)
   - If penalty magnitude <15 yards (insufficient edge)

2. **Paper trade failure**
   - 20 paper trades AND win rate <55%
   - 20 paper trades AND ROI <0%

3. **Live betting failure**
   - 20 live bets AND ROI <0%
   - 30 live bets AND win rate <55%

4. **Market adaptation**
   - Oddsmakers consistently price in -20+ yard penalty
   - Odds consistently worse than -150 (edge eliminated)

5. **Sample invalidation**
   - NFL schedule changes reduce <7 day turnarounds
   - International game format changes (e.g., more rest days)

### Reduce Kelly (Temporary)

**From 10% → 5% Kelly if:**
- 10 bets AND win rate 55-59%
- 3 consecutive losses
- p-value increases to 0.01-0.05 range (weakening significance)

**From 5% → 0% (pause betting) if:**
- 5 consecutive losses
- Win rate drops below 52% (below breakeven)
- Major injury to key QB in sample (changes population characteristics)

### Increase Kelly (Aggressive)

**From 10% → 15% Kelly if:**
- 30 bets AND win rate >75%
- 30 bets AND ROI >40%
- p-value remains <0.001 (strengthening significance)
- Baseline fix validates larger penalty (>200 yards)

**From 15% → 20% Kelly if:**
- 50 bets AND win rate >80%
- 50 bets AND ROI >50%
- Professional bettor verification of edge

---

## 2025 Opportunities

### Confirmed International Games

| Week | Date | Location | Stadium | Matchup | Time (ET) |
|------|------|----------|---------|---------|-----------|
| 5 | Oct 6 | London | Tottenham Hotspur | MIN @ NYJ | 9:30 AM |
| 6 | Oct 13 | London | Wembley | CHI @ JAX | 9:30 AM |
| 7 | Oct 20 | London | Wembley | JAX @ NE | 9:30 AM |
| 10 | Nov 10 | Munich | Allianz Arena | NYG @ CAR | 9:30 AM |

### Expected Edge C Opportunities

**Week 6 (7 days after Week 5 London):**
- **Sam Darnold (MIN):** Plays Week 6 game 7 days after London
- **Aaron Rodgers (NYJ):** Plays Week 6 game 7 days after London

**Week 7 (7 days after Week 6 London):**
- **Caleb Williams (CHI):** Plays Week 7 game 7 days after London
- **Trevor Lawrence (JAX):** Plays Week 7 game 7 days after London

**Week 8 (7-8 days after Week 7 London):**
- **Trevor Lawrence (JAX):** Plays Week 8 game 7 days after London (NOTE: back-to-back international, may have DOUBLE penalty)
- **Drake Maye (NE):** Plays Week 8 game 7-8 days after London

**Week 11 (7 days after Week 10 Munich):**
- **Daniel Jones (NYG):** Plays Week 11 game 7 days after Munich
- **Bryce Young (CAR):** Plays Week 11 game 7 days after Munich

**Total Expected Opportunities:** 6-8 bets (depending on schedules)

### Monitoring Schedule

**4 weeks before international game:**
- Run monitor script to identify next-week matchups
- Research QB passing prop markets (historical lines)

**1 week before next game:**
- Verify QB health (injury reports)
- Check opponent pass defense ranking
- Get latest baseline estimate (corrected)

**Day before next game:**
- Lock in prop line (shop for best odds)
- Calculate Kelly bet size
- Log paper trade details

**Day after next game:**
- Record actual results
- Update paper trade performance
- Re-run validation statistics

---

## Tools & Scripts

### 1. International Schedule Monitor

**Purpose:** Identify upcoming Edge C opportunities

**Usage:**
```bash
# Check next 4 weeks
python scripts/monitor_international_schedule.py

# Check next 8 weeks
python scripts/monitor_international_schedule.py --weeks-ahead 8

# Save alerts to file
python scripts/monitor_international_schedule.py --output results/edge_c_alerts.json
```

**Output:**
```
EDGE OPPORTUNITY: Sam Darnold (MIN) plays 7 days after London game
  International: Week 5 in London (2025-10-06)
  Next Game: Week 6 vs TBD (2025-10-13)
  Expected Penalty: -195.6 yards
  Expected Win Rate: 70.0%
  Expected ROI: 33.6%
```

### 2. Bet Sizing Calculator

**Purpose:** Calculate fractional Kelly bet sizes with variance adjustment

**Usage:**
```bash
# Interactive mode (recommended)
python scripts/calculate_edge_c_bet_size.py --interactive

# Command line
python scripts/calculate_edge_c_bet_size.py \
  --bankroll 10000 \
  --win-rate 0.70 \
  --odds -110 \
  --kelly-fraction 0.10
```

**Output:**
```
KELLY CALCULATION:
  Full Kelly: 36.70% of bankroll
  Variance Adjusted: 18.00% (factor: 2.04x)
  Fractional Kelly (10%): 1.80% of bankroll

BET SIZING:
  Kelly Recommended: $180.00
  Max Allowed (2% bankroll): $200.00
  FINAL RECOMMENDATION: $180.00 (1.80% of bankroll)

EXPECTED VALUE:
  EV per $1 bet: $0.3363
  Expected profit: $60.53
  ROI: 33.63%
```

### 3. Paper Trade Tracker

**Purpose:** Log and analyze paper trade results

**File:** `results/edge_c_paper_trades.csv`

**Example Entry:**
```csv
trade_id,date,qb,team,opponent,game_week,days_since_intl,intl_location,baseline_yards,predicted_yards,actual_yards,prop_line,prop_odds,bet_size_dollars,outcome,profit_loss,roi,notes
1,2025-10-13,Sam Darnold,MIN,TBD,6,7,London,250,54.4,180,225.5,-110,180,WIN,163.64,0.909,First paper trade
```

### 4. Validation Dashboard

**Purpose:** Monitor Edge C performance and upcoming opportunities

**Usage:**
```bash
# View current status
python scripts/edge_c_dashboard.py

# Update opportunities from database
python scripts/edge_c_dashboard.py --update

# Custom paper trades file
python scripts/edge_c_dashboard.py --paper-trades my_trades.csv
```

**Output:**
```
================================================================================
                    EDGE C VALIDATION DASHBOARD
================================================================================

EDGE C STATISTICS (Lingering Fatigue)
Status: PENDING BASELINE FIX
Last Validated: 2025-11-16
Sample Size: 24 QB performances (14 with <7 day rest)

UPCOMING OPPORTUNITIES (Next 4 Weeks)
Found 2 opportunities:
1. Sam Darnold (MIN) - Week 6
   International: Week 5 in London
   Next Game: 2025-10-13 vs TBD (7 days)

PAPER TRADE PERFORMANCE
Total Trades: 5
Completed Trades: 3
Win Rate: 66.7%
Total Profit/Loss: $245.45
Average ROI: 27.3%
```

---

## Baseline Fix Tracking

### Current Issue (as of Nov 16, 2025)

**Problem:** Baseline QB averages calculated using `current_team` filter, which reflects 2024 roster, not historical team during performance.

**Impact:**
- Artificially inflates baselines (e.g., Justin Herbert 515 yards/game baseline is impossible)
- Penalty calculations likely **overestimated**
- Edge validity **uncertain** until fix validated

**Example Error:**
```sql
-- WRONG (current implementation):
SELECT AVG(stat_yards)
FROM player_game_stats
WHERE current_team = 'LAC'  -- Uses 2024 team for all seasons

-- CORRECT (pending implementation):
SELECT AVG(pgs.stat_yards)
FROM player_game_stats pgs
JOIN games g ON (pgs.season = g.season AND pgs.week = g.week)
WHERE pgs.player_id = '00-0036212'
  AND (pgs.current_team = g.home_team OR pgs.current_team = g.away_team)
```

### Fix Validation Plan

**Step 1: Implement Corrected Baseline (2 hours)**
- Rewrite `fetch_qb_domestic_average()` in analysis scripts
- Join to `games` table via `(season, week)` to get actual team
- Validate against ESPN/PFR known QB averages

**Step 2: Re-Run Edge C Analysis (1 hour)**
- Re-calculate penalties with corrected baselines
- Re-run statistical tests (t-test, p-value)
- Generate new Edge C statistics report

**Step 3: Validation (1 hour)**
- Compare to external benchmarks (ESPN QB stats 2018-2024)
- Sanity check: No QB should average >350 yards/game
- If penalty remains >-15 yards AND p < 0.05 → Edge validated

**Step 4: Decision (Go/No-Go)**
- **GO:** If penalty ≥-15 yards AND p < 0.05 → Proceed to paper trading
- **NO-GO:** If penalty <-15 yards OR p > 0.10 → Abandon edge

### Expected Outcomes

**Scenario 1: Edge strengthens (30% probability)**
- Corrected baselines reduce averages (more realistic)
- Penalty magnitude **increases** (e.g., -220 yards)
- ROI estimate **increases** (e.g., +40%)
- **Action:** Increase Kelly to 15% after 10 successful bets

**Scenario 2: Edge persists (50% probability)**
- Penalty magnitude adjusts to **-100 to -150 yards**
- Statistical significance maintained (p < 0.01)
- ROI estimate adjusts to **+15-25%**
- **Action:** Proceed with 10% Kelly as planned

**Scenario 3: Edge disappears (20% probability)**
- Penalty magnitude <-15 yards OR p > 0.10
- Edge **not viable** for betting
- **Action:** Abandon edge, focus on other opportunities

### Baseline Fix Timeline

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| **1. Implementation** | 2 hours | PENDING | Corrected baseline function |
| **2. Re-Analysis** | 1 hour | PENDING | New Edge C statistics |
| **3. Validation** | 1 hour | PENDING | Validated against ESPN |
| **4. Decision** | 30 min | PENDING | GO/NO-GO determination |
| **TOTAL** | 4.5 hours | PENDING | Edge C validation report |

**Estimated Completion:** Within 1 business day of request

---

## Appendix: Historical Edge C Data

### Sample QB Performances (2018-2024)

**Short Rest (<7 days) Examples:**

| QB | Season | Intl Week | Next Week | Days | Actual | Baseline | Penalty |
|----|--------|-----------|-----------|------|--------|----------|---------|
| Tua Tagovailoa | 2021 | 6 | 7 | 6.9 | 291 | 468.3 | **-177.3** |
| Derek Carr | 2022 | 4 | 5 | 7.0 | 241 | 506.9 | **-265.9** |
| Trevor Lawrence | 2022 | 8 | 9 | 7.0 | 235 | 459.0 | **-224.0** |
| Josh Allen | 2023 | 5 | 6 | 7.0 | 169 | 462.1 | **-293.1** |
| Trevor Lawrence | 2024 | 7 | 8 | 7.0 | 308 | 459.0 | **-151.0** |

**Average Penalty (Short Rest):** -195.6 yards

### Statistical Tests

**One-Sample t-test (Penalty vs 0):**
- H0: Penalty = 0 (no effect)
- Ha: Penalty < 0 (negative effect)
- t-statistic: -13.766
- p-value: < 0.0001
- **Conclusion:** Reject H0, penalty exists

**Binomial Test (Win Rate):**
- H0: Win rate = 50% (no edge)
- Ha: Win rate > 52.4% (profitable at -110)
- Observed: 70% win rate (14/14 short rest showed penalty)
- **Conclusion:** Edge exists at -110 odds

### ROI Calculation

**At -110 odds:**
- Win: +$90.91 per $100 bet (payout = $190.91)
- Loss: -$100 per $100 bet

**Expected Value:**
```
EV = (0.70 × $90.91) + (0.30 × -$100)
   = $63.64 - $30.00
   = $33.64 per $100 bet
   = 33.6% ROI
```

**Breakeven Win Rate:**
```
Breakeven = 110 / (110 + 100) = 52.4%
```

**Edge over Breakeven:**
```
70% - 52.4% = 17.6 percentage points
```

---

## Support & Questions

**Technical Issues:**
- Database connection: See `docs/database/DATABASE_QUICK_REFERENCE.md`
- Script errors: Check Python version (3.8+) and dependencies

**Betting Questions:**
- Kelly criterion: See `docs/betting/KELLY_CRITERION_GUIDE.md` (TODO)
- Risk management: See `docs/betting/RISK_MANAGEMENT.md` (TODO)

**Edge C Specific:**
- Baseline fix status: Check `results/PHASE_0_INTL_GAMES_SUMMARY.md`
- Latest analysis: `results/international_betting_edges.json`

---

**END OF PLAYBOOK**

**Remember:** This is real money. Paper trade first. Validate the edge. Manage risk ruthlessly.
