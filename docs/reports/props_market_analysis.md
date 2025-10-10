# NFL Player Props: Market Analysis & Strategy

**Date**: 2025-10-10
**Status**: Research & Development Phase
**Implementation**: Q1 2026 (Post-Spread/Total Profitability)

---

## Executive Summary

**Recommendation**: **DEFER** props betting until spread/total models prove profitable for 1+ seasons.

**Reasoning**:
- Props markets are **less efficient** than game-level markets (higher variance, softer lines)
- **Higher ROI potential** (+5-10%) but **higher variance** (3-4× game-level betting)
- **Requires larger bankroll** ($20k+) and **more time** (daily prop research)
- **Data gaps**: Need player-level features, injury tracking, usage trends
- **Market risk**: Props limits are lower ($100-500 vs $1,000+ for spreads)

**Phased Approach**:
1. **Phase 1 (2025)**: Prove spread/total models profitable with $10k bankroll
2. **Phase 2 (2026 Q1)**: Build props infrastructure (features, models, data pipelines)
3. **Phase 3 (2026 Q2)**: Paper trade props for one season (no real money)
4. **Phase 4 (2026 Q3+)**: Live betting with $20k bankroll (if paper trading successful)

---

## Market Overview

### What Are Player Props?

Player props are bets on individual player performance:
- **QB**: Passing yards, passing TDs, interceptions, completions
- **RB**: Rushing yards, rushing TDs, receptions
- **WR/TE**: Receiving yards, receptions, receiving TDs
- **K**: Total points, longest FG, FG made/missed

**Bet Type**: Over/Under (O/U) at a specific line (e.g., "Patrick Mahomes Over 275.5 passing yards")

**Typical Odds**: -110/-110 (fair market) to -120/-105 (unbalanced)

### Market Size

**Total Handle** (2024):
- Game-level (spread/total/ML): $10B+ per season
- Player props: $2-3B per season (~20-30% of game-level)
- Growing rapidly: +25% YoY (driven by same-game parlays, microbetting)

**Sportsbook Breakdown** (Virginia):
- **FanDuel**: 50+ props per game, highest limits ($500)
- **DraftKings**: 40+ props per game, moderate limits ($300)
- **BetMGM/Caesars**: 30+ props per game, lower limits ($200)
- **Smaller books**: 10-20 props per game, very low limits ($50-100)

**Liquidity**:
- **Star players** (Mahomes, Allen, McCaffrey): High liquidity, sharp lines
- **Mid-tier players** (WR2/RB2): Moderate liquidity, softer lines
- **Backup players** (WR3/TE2): Low liquidity, very soft lines (but low limits)

### Market Efficiency

**Compared to Spreads/Totals**:
- **Spreads/Totals**: Very efficient (Pinnacle/Circa sharps move lines quickly)
- **Player Props**: Less efficient (books slower to adjust, fewer sharp bettors)

**Why Props Are Softer**:
1. **More markets**: 50+ props per game → harder for books to price all accurately
2. **Less sharp money**: Most sharp bettors focus on game-level (higher limits)
3. **Recreational focus**: Casual bettors love props (emotion, favorite players)
4. **Parlay bias**: Books shade props for parlays (subsidizes standalone props)

**Expected Edge**:
- **Spreads/Totals**: +1-3% (highly efficient)
- **Player Props**: +5-10% (less efficient, higher variance)

**Holding Periods**:
- **Spreads**: 1 week between games → can compound edge quickly
- **Props**: Daily/weekly → more opportunities but higher time investment

---

## Prop Types & Lines

### Passing Props (QB)

| Prop Type | Typical Line | Variance | Edge Potential |
|-----------|--------------|----------|----------------|
| **Passing Yards** | 250-300 | High | Medium (+3-5%) |
| **Passing TDs** | 1.5-2.5 | Very High | High (+5-10%) |
| **Interceptions** | 0.5-1.5 | Very High | High (+5-10%) |
| **Completions** | 22.5-27.5 | Medium | Low (+1-3%) |

**Best for Modeling**:
- **Passing Yards**: Predictable, large sample size, stable variance
- **Passing TDs**: High variance but exploitable (game script, red zone usage)

**Avoid**:
- **Interceptions**: Too random (one bad throw = huge swing)
- **Completions**: Highly correlated with attempts (use attempts instead)

### Rushing Props (RB)

| Prop Type | Typical Line | Variance | Edge Potential |
|-----------|--------------|----------|----------------|
| **Rushing Yards** | 50-100 | High | Medium (+3-5%) |
| **Rushing TDs** | 0.5-1.5 | Very High | High (+5-10%) |
| **Receptions** | 2.5-5.5 | Medium | Medium (+3-5%) |

**Best for Modeling**:
- **Rushing Yards**: Predictable for bellcow RBs (70%+ snap share)
- **Receptions** (pass-catching RBs): Game script dependent (teams down = more receptions)

**Avoid**:
- **Rushing TDs**: Too random unless RB is goal-line back

### Receiving Props (WR/TE)

| Prop Type | Typical Line | Variance | Edge Potential |
|-----------|--------------|----------|----------------|
| **Receiving Yards** | 40-80 | Very High | High (+5-10%) |
| **Receptions** | 3.5-6.5 | Medium | Medium (+3-5%) |
| **Receiving TDs** | 0.5-1.5 | Very High | High (+5-10%) |

**Best for Modeling**:
- **Receiving Yards** (WR1): High target share, predictable volume
- **Receptions** (slot WRs/TE): Game script dependent, exploitable

**Avoid**:
- **Receiving TDs**: Too random (red zone usage, game flow)
- **Backup WRs**: Low volume, high variance

---

## Edge Sources

### 1. Model Accuracy (+2-4% Edge)

**Data Advantages**:
- **Play-by-play data**: Detailed player usage (snap %, routes run, targets)
- **Rolling averages**: Recent performance (last 3/5 games) better than season-long
- **Opponent strength**: Defensive rankings (pass defense vs WR1, run defense vs RB)
- **Game script**: Implied team total, spread (teams down throw more, teams up run more)

**Model Approach**:
- **XGBoost regression**: Predict exact stat (e.g., 285.3 passing yards)
- **Quantile regression**: Estimate uncertainty (e.g., ±30 yards std)
- **Over/Under probabilities**: Integrate over normal distribution

**Expected Accuracy**:
- **MAE (Mean Absolute Error)**: 25-35 yards for passing/receiving, 10-15 yards for rushing
- **R²**: 0.15-0.25 (player stats are noisy, even best models struggle)
- **Calibration**: Critical for prop betting (need accurate probabilities, not just point estimates)

### 2. Market Inefficiencies (+3-6% Edge)

**Recency Bias**:
- Books overweight last 1-2 games → exploit mean reversion
- Example: WR goes 150 yards → next game line inflated by 10-15 yards → bet under

**Star Player Tax**:
- Books shade lines on star players (Mahomes, Allen, McCaffrey) to attract action
- Opportunity: Bet under on overpriced stars, over on underpriced mid-tier players

**Game Script Mispricing**:
- Books slow to adjust for spread/total changes (e.g., line moves from -3 to -7)
- Opportunity: If team now heavily favored, passing yards likely decrease (running clock)

**Injury Information Edge**:
- Late scratches, surprise inactives → lines don't adjust fast enough
- Opportunity: Monitor injury reports 90 min before kickoff, bet before line moves

**Weather Mispricing**:
- Books undervalue wind impact on passing yards (>15 mph wind → -20-30 passing yards)
- Opportunity: Bet under on passing yards in windy games

### 3. Line Shopping (+1-2% Edge)

**Props Have Wide Spreads**:
- Spreads/totals: 0.5-1 point variance between books
- Props: 2-5 yard/reception variance between books

**Example** (Patrick Mahomes passing yards):
- FanDuel: 282.5
- DraftKings: 279.5
- BetMGM: 285.5
- **Variance**: 6 yards (2% of line)

**Line Shopping ROI**:
- Game-level: +1-2% (narrow spreads)
- Props: +2-3% (wider spreads, less efficient)

---

## Expected ROI & Variance

### ROI Projections

**Conservative Model** (Baseline):
- **Data**: Player features + opponent strength + game script
- **Expected ROI**: +3-5% per bet
- **Assumptions**: Bet only high-confidence props (>5% edge), 2-3 bets/week

**Aggressive Model** (With Edge Opportunities):
- **Data**: Baseline + injury tracking + weather + line shopping + late news
- **Expected ROI**: +7-10% per bet
- **Assumptions**: Bet all +EV props (>3% edge), 10-15 bets/week

**Comparison to Spreads**:
- Spreads: +1-3% ROI, 5-10 bets/week → **+15-30% annual return**
- Props (conservative): +3-5% ROI, 10-15 bets/week → **+30-75% annual return**
- Props (aggressive): +7-10% ROI, 30-50 bets/week → **+210-500% annual return**

**Reality Check**: Aggressive model likely overstates ROI (market adapts, limits decrease, variance increases)

### Variance Analysis

**Variance Sources**:
1. **Injury risk**: Player gets hurt mid-game → instant loss
2. **Game script**: Blowout → star players benched in 4th quarter
3. **Random variance**: One big play (80-yard TD) swings outcome

**Variance Comparison**:
- **Spreads**: 52-55% win rate, 0.8-1.0 std dev per bet
- **Props**: 53-58% win rate, 2.5-3.5 std dev per bet

**Bankroll Impact**:
- **Spreads**: $10k bankroll → 2-3% drawdown risk (Kelly sizing)
- **Props**: $10k bankroll → 10-15% drawdown risk (higher variance)
- **Recommendation**: Use $20k+ bankroll for props (lower Kelly fractions)

### Sharpe Ratio

**Definition**: (Expected Return - Risk-Free Rate) / Standard Deviation

**Calculations**:
- **Spreads**: (3% - 0%) / 5% = 0.6 Sharpe
- **Props (conservative)**: (5% - 0%) / 12% = 0.42 Sharpe
- **Props (aggressive)**: (8% - 0%) / 18% = 0.44 Sharpe

**Interpretation**: Props have higher absolute returns but **lower risk-adjusted returns** than spreads.

---

## Bankroll Requirements

### Minimum Bankroll

**Conservative Approach**:
- **Bet Size**: 1-2% of bankroll per bet (fractional Kelly = 0.25)
- **Weekly Bets**: 10-15 props
- **Weekly Risk**: $2k-3k (10-15 × $200 avg bet)
- **Minimum Bankroll**: $20,000

**Aggressive Approach**:
- **Bet Size**: 2-3% of bankroll per bet (fractional Kelly = 0.5)
- **Weekly Bets**: 30-50 props
- **Weekly Risk**: $10k-15k (40 × $300 avg bet)
- **Minimum Bankroll**: $50,000

**Comparison to Spreads**:
- Spreads: $10k bankroll sufficient (lower variance, fewer bets)
- Props: $20k-50k bankroll recommended (higher variance, more bets)

### Betting Limits

**Sportsbook Limits** (Virginia):
- **FanDuel**: $500 max (star players), $200-300 (mid-tier)
- **DraftKings**: $300 max (star players), $100-200 (mid-tier)
- **BetMGM**: $200 max (star players), $50-100 (mid-tier)

**Limit Risk**:
- If model finds edge on star player prop → can bet up to $1,200 total (across 4 books)
- If model finds edge on mid-tier player → can bet only $400-600 total
- **Problem**: Bankroll grows faster than betting limits (limited upside)

**Recommendation**: Focus on **star player props** (higher limits, more liquidity)

---

## Risk Management

### Position Sizing (Kelly Criterion)

**Kelly Formula**: f* = (bp - q) / b
- f* = fraction of bankroll to bet
- b = decimal odds - 1 (e.g., -110 → b = 0.909)
- p = win probability
- q = 1 - p

**Fractional Kelly** (Recommended):
- **Full Kelly**: Maximizes long-term growth (but high variance)
- **Half Kelly**: 50% of full Kelly (reduces variance by 75%)
- **Quarter Kelly**: 25% of full Kelly (reduces variance by 93%)

**Props Recommendation**: Use **1/8 to 1/4 Kelly** (props have higher variance than spreads)

**Example**:
- Model predicts 60% chance of over 275.5 passing yards
- Odds: -110 (implied 52.4%)
- Edge: 60% - 52.4% = 7.6%
- Full Kelly: 8.4% of bankroll
- **Quarter Kelly**: 2.1% of bankroll ($420 on $20k bankroll)

### Correlation Risk

**Problem**: Props are correlated (QB passing yards ↔ WR receiving yards)

**Example**:
- Bet Patrick Mahomes over 275.5 passing yards
- Bet Travis Kelce over 65.5 receiving yards
- If Mahomes has bad game → both bets lose (correlated risk)

**Solution**:
1. **Limit correlated bets**: Don't bet QB + WR from same team
2. **Reduce bet sizes**: If betting correlated props, use 0.5× Kelly each
3. **Focus on independent props**: Different games, different positions

### Injury Risk

**Problem**: Player gets hurt mid-game → instant loss (no refund)

**Mitigation**:
1. **Avoid injury-prone players**: History of soft-tissue injuries (hamstring, ankle)
2. **Monitor injury reports**: NFL injury report (Friday update), beat reporters (Twitter)
3. **Bet on durable players**: OL-protected QBs, pass-catching RBs (less contact)
4. **Insurance props**: Bet opposite side in-game if injury risk high

---

## Market Selection

### Which Props to Bet

**Tier 1: High Priority** (Bet Frequently)
- **QB Passing Yards** (star QBs): High volume, predictable, large limits
- **RB Rushing Yards** (bellcow RBs): 70%+ snap share, predictable usage
- **WR1 Receiving Yards**: High target share (25%+), predictable

**Tier 2: Medium Priority** (Bet Selectively)
- **QB Passing TDs**: High variance but exploitable (game script, red zone usage)
- **Pass-Catching RB Receptions**: Game script dependent (teams down throw more)
- **TE Receiving Yards** (top-5 TEs): High target share, exploitable

**Tier 3: Low Priority** (Avoid or Rare Bets)
- **Rushing TDs**: Too random (goal-line carries unpredictable)
- **Receiving TDs**: Too random (red zone usage varies)
- **Interceptions**: Too random (one bad throw = huge swing)
- **Backup WRs**: Low volume, high variance, low limits

### Which Props to Avoid

**Red Flags**:
1. **Low limits** (<$100): Not worth the time/effort
2. **Backup players**: Unpredictable usage (injury-dependent)
3. **TD props**: High variance, hard to model
4. **Weather games**: >15 mph wind or heavy rain (too much uncertainty)
5. **Backup QBs**: Small sample size, unpredictable performance

**Contrarian Opportunity**:
- **Unders on star players**: Books shade lines high to attract action → bet under
- **Overs on mid-tier players**: Books shade lines low (less action) → bet over

---

## Expected Performance

### Base Case (Conservative)

**Assumptions**:
- Bet 10-15 props per week (only high-confidence bets, >5% edge)
- Average edge: +5% per bet
- Average bet size: $200 (1% of $20k bankroll)
- Win rate: 55%

**Annual Performance**:
- Total bets: 170-255 per season (17 weeks × 10-15 bets)
- Total wagered: $34k-51k
- Expected profit: $1,700-2,550 (5% × $34k-51k)
- **ROI**: +5%
- **Annual return**: +8.5-12.8% of $20k bankroll

**Risk**:
- Standard deviation: 12% per bet
- Max drawdown: 15-20% (1 in 20 chance)
- 95% confidence interval: -$500 to +$4,800

### Optimistic Case (Aggressive)

**Assumptions**:
- Bet 30-50 props per week (all +EV bets, >3% edge)
- Average edge: +8% per bet
- Average bet size: $300 (1.5% of $20k bankroll)
- Win rate: 58%

**Annual Performance**:
- Total bets: 510-850 per season (17 weeks × 30-50 bets)
- Total wagered: $153k-255k
- Expected profit: $12,240-20,400 (8% × $153k-255k)
- **ROI**: +8%
- **Annual return**: +61-102% of $20k bankroll

**Risk**:
- Standard deviation: 18% per bet
- Max drawdown: 30-40% (1 in 10 chance)
- 95% confidence interval: -$2,000 to +$44,800
- **Bust risk**: 5-10% chance of losing >50% of bankroll

---

## Implementation Roadmap

### Phase 1: Data Infrastructure (Q4 2025)

**Objective**: Build player-level data pipelines

**Tasks**:
1. ✅ `py/features/player_features.py` - Rolling averages, opponent strength, game script
2. ✅ `py/models/props_predictor.py` - XGBoost regression model for prop predictions
3. [ ] `R/ingestion/ingest_player_props.R` - Scrape historical prop lines (The Odds API)
4. [ ] `py/ingestion/ingest_injuries.py` - Real-time injury tracking (NFL API, Twitter)
5. [ ] `py/ingestion/ingest_weather.py` - Weather forecasts (OpenWeatherMap API)

**Deliverables**:
- Historical player features (2010-2024)
- Historical prop lines (2020-2024)
- Injury database (current season)

### Phase 2: Model Development (Q1 2026)

**Objective**: Train and validate props models

**Tasks**:
1. [ ] Train XGBoost models (one per prop type: passing_yards, rushing_yards, etc.)
2. [ ] Hyperparameter sweep (learning rate, depth, subsample)
3. [ ] Backtest on historical data (2022-2024 seasons)
4. [ ] Calibration analysis (are 60% predictions actually 60%?)
5. [ ] Feature importance analysis (which features drive accuracy?)

**Deliverables**:
- Trained models for 8+ prop types
- Backtest report (ROI, Sharpe, win rate, drawdowns)
- Calibration plots (reliability diagrams)

### Phase 3: Paper Trading (Q2-Q3 2026)

**Objective**: Validate models on live data (no real money)

**Tasks**:
1. [ ] Daily prop predictions (Tuesday-Sunday)
2. [ ] Track performance vs closing lines
3. [ ] Identify profitable prop types (passing yards vs TDs)
4. [ ] Refine edge threshold (bet only >5% edge vs >3%)
5. [ ] Test line shopping (compare FanDuel vs DraftKings vs BetMGM)

**Success Criteria**:
- **ROI**: +3% over 1 season (paper trading)
- **Win Rate**: 53%+ (vs break-even 52.4%)
- **Sharpe**: >0.4 (risk-adjusted returns)
- **Calibration**: Brier score <0.23

**If Paper Trading Fails**:
- Revisit model architecture (try neural networks, ensemble methods)
- Collect more data (2-3 seasons of player props lines)
- Focus on higher-confidence props only (>7% edge)

### Phase 4: Live Betting (Q4 2026+)

**Objective**: Deploy props betting with real money

**Tasks**:
1. [ ] Start with small bankroll ($5k) for 4 weeks (stress test)
2. [ ] Scale to $20k bankroll if 4-week test successful
3. [ ] Monitor performance weekly (ROI, Sharpe, drawdowns)
4. [ ] Adjust bet sizing if variance exceeds expectations
5. [ ] Expand to more prop types if profitable (TDs, receptions, etc.)

**Performance Monitoring**:
- **Weekly reports**: ROI, win rate, CLV, largest bets
- **Monthly reviews**: Sharpe ratio, max drawdown, bankroll growth
- **Quarterly audits**: Model drift, recalibration, feature importance

---

## Risk Factors

### Model Risk

**Problem**: Player stats are noisy (even best models struggle)

**Evidence**:
- Best props models achieve R² = 0.20-0.25 (75-80% unexplained variance)
- Passing yards have ±30 yards std dev (15% of typical line)
- One big play (80-yard TD) can swing outcome

**Mitigation**:
- Bet only high-confidence props (>5% edge)
- Use quantile regression for uncertainty estimates
- Diversify across multiple props (10-15 per week)

### Market Risk

**Problem**: Sportsbooks limit winning players

**Evidence**:
- Props limits start at $100-500 (vs $1,000+ for spreads)
- Books aggressively limit props winners (after $5k-10k profit)
- Line shopping becomes harder (can't bet $500 at all books)

**Mitigation**:
- Spread bets across 5+ sportsbooks (FanDuel, DraftKings, BetMGM, Caesars, Circa)
- Use family accounts (legal in Virginia if each person places own bets)
- Focus on star player props (higher limits, less scrutiny)

### Data Risk

**Problem**: Injury news, weather changes → need real-time updates

**Evidence**:
- 10% of players get surprise inactives (90 min before kickoff)
- Weather forecasts change rapidly (wind speed ±5 mph)
- Late scratches can invalidate prop bets (but no refund)

**Mitigation**:
- Subscribe to NFL+ Premium ($40/year) for injury reports
- Monitor beat reporters on Twitter (30-60 min before kickoff)
- Avoid betting props on Friday (wait for Sunday injury updates)

---

## Comparison to Spreads/Totals

| Metric | Spreads/Totals | Player Props |
|--------|----------------|--------------|
| **Market Efficiency** | Very efficient | Less efficient |
| **Expected ROI** | +1-3% | +5-10% |
| **Variance** | Low (0.8-1.0 std) | High (2.5-3.5 std) |
| **Sharpe Ratio** | 0.6 | 0.4 |
| **Betting Limits** | $1,000-5,000 | $100-500 |
| **Bankroll Requirement** | $10k | $20k-50k |
| **Time Investment** | 5-10 hours/week | 15-20 hours/week |
| **Data Requirements** | Team-level | Player-level |
| **Model Complexity** | Medium | High |
| **Injury Risk** | Low (team-level) | High (player-level) |
| **Scalability** | High | Low (limits) |

**Verdict**: Props offer **higher returns** but **higher risk** and **lower scalability**.

---

## Final Recommendation

### Immediate Action (2025)

1. ❌ **Do NOT bet props yet** (focus on spreads/totals until profitable)
2. ✅ **Build infrastructure** (player features, historical props lines)
3. ✅ **Train models** (XGBoost regression for 8+ prop types)
4. ✅ **Paper trade** (Q2-Q3 2026, validate on live data)

### Long-Term Strategy (2026+)

1. **If spread/total models profitable** → Allocate $20k to props (separate bankroll)
2. **If paper trading successful** (>+3% ROI) → Deploy live betting (Q4 2026)
3. **If props profitable** (>+5% ROI for 1 season) → Scale to $50k bankroll (2027)

### Why Defer Props?

1. **Spreads are easier**: Lower variance, higher limits, less time investment
2. **Props require more data**: Player-level features, injury tracking, weather
3. **Props are riskier**: Higher variance → need larger bankroll ($20k+ vs $10k)
4. **Limits are lower**: Can't scale props as much as spreads ($500 vs $5,000 max bets)

**Bottom Line**: Prove profitability on spreads/totals first, then expand to props in 2026.

---

*Last Updated: 2025-10-10*
