# Virginia Sportsbooks: Operational Guide

**Date**: 2025-10-10
**Status**: Active for production deployment

## Overview

Virginia legalized sports betting in April 2020, with online sportsbooks launching in January 2021. The state has one of the most competitive sports betting markets in the US with 15+ active operators and no cap on licenses.

**Key Facts**:
- **Legal Age**: 21+
- **Market Launch**: January 21, 2021
- **Regulatory Body**: Virginia Lottery
- **Tax Rate**: 15% on gross gaming revenue
- **Online Betting**: Legal statewide
- **Retail Betting**: Limited (primarily casino-based)
- **College Betting**: Allowed (except VA colleges)

---

## Licensed Sportsbooks in Virginia (2025)

### Tier 1: Premium Books (Best for Line Shopping)

#### 1. **FanDuel Sportsbook**
- **Market Share**: ~35-40% (largest in VA)
- **Strengths**:
  - Competitive NFL spreads and totals
  - Fast bet settlement
  - Excellent mobile app
  - Same-game parlays (avoid per Task 10)
- **Odds Quality**: A+ (often sharpest lines)
- **Limits**: High ($10k+ for NFL spreads)
- **Promos**: Regular odds boosts (check for value)
- **API/Data**: Limited (manual entry required)

#### 2. **DraftKings Sportsbook**
- **Market Share**: ~25-30%
- **Strengths**:
  - Early lines (posted Sunday night)
  - Props variety (400+ per NFL game)
  - Live betting platform
- **Odds Quality**: A (competitive, occasionally soft on props)
- **Limits**: High ($5k-10k)
- **Promos**: Profit boosts, SGP insurance
- **API/Data**: Limited

#### 3. **BetMGM**
- **Market Share**: ~8-12%
- **Strengths**:
  - Alternate spreads/totals
  - Early week lines (Tuesday AM)
  - Parlay insurance
- **Odds Quality**: B+ (sometimes +1-2 cents on spreads)
- **Limits**: Medium-High ($5k)
- **Promos**: Risk-free bets, parlay insurance
- **API/Data**: None

#### 4. **Caesars Sportsbook**
- **Market Share**: ~6-8%
- **Strengths**:
  - Generous promos for new users
  - Live betting
  - Caesars Rewards integration
- **Odds Quality**: B (competitive but not sharpest)
- **Limits**: Medium ($3k-5k)
- **Promos**: Frequent odds boosts
- **API/Data**: None

#### 5. **ESPN BET (formerly Barstool)**
- **Market Share**: ~5-7%
- **Strengths**:
  - ESPN integration (live stats)
  - Micro-betting (live props)
  - Public betting trends
- **Odds Quality**: B (public-leaning, occasionally soft)
- **Limits**: Medium ($2k-5k)
- **Promos**: ESPN+ integration bonuses
- **API/Data**: None

### Tier 2: Competitive Books (Good for Arbitrage)

#### 6. **bet365**
- **Strengths**: International reputation, extensive props
- **Odds Quality**: B+ (competitive international lines)
- **Limits**: Medium ($3k)

#### 7. **Fanatics Sportsbook**
- **Strengths**: Fanatics rewards, merchandise integration
- **Odds Quality**: B (newer book, occasionally soft)
- **Limits**: Medium ($2k-3k)

#### 8. **BetRivers**
- **Strengths**: iRush Rewards, solid mobile app
- **Odds Quality**: B (regionally competitive)
- **Limits**: Medium ($2k-4k)

#### 9. **Borgata (BetMGM skin)**
- **Strengths**: Separate promos from BetMGM
- **Odds Quality**: B+ (shares BetMGM lines)
- **Limits**: Medium ($3k-5k)

#### 10. **Hard Rock Bet**
- **Strengths**: Unity Rewards, casino integration
- **Odds Quality**: C+ (public-leaning)
- **Limits**: Low-Medium ($1k-2k)

### Tier 3: Niche Books (Specialized Use Cases)

#### 11. **Circa Sports**
- **Strengths**: Sharp lines, survivor pools, contests
- **Odds Quality**: A+ (sharp, Vegas-based)
- **Limits**: High for sharp bettors ($5k+)
- **Notes**: May limit winners quickly

#### 12. **SuperBook**
- **Strengths**: Futures, season-long props
- **Odds Quality**: B+ (Vegas-based, sharp)
- **Limits**: Medium ($2k-3k)

#### 13. **SI Sportsbook**
- **Strengths**: Sports Illustrated content integration
- **Odds Quality**: C+ (recreational-focused)
- **Limits**: Low-Medium ($1k-2k)

#### 14. **Fliff**
- **Strengths**: Social betting, unique markets
- **Odds Quality**: C (niche markets)
- **Limits**: Low ($500-1k)

#### 15. **Underdog Fantasy**
- **Strengths**: Pick'em style, player props focus
- **Odds Quality**: B (DFS-style pricing)
- **Limits**: Low-Medium ($1k)
- **Notes**: Not traditional sportsbook (pick'em format)

---

## Line Shopping Strategy

### Primary Books (Check Every Bet)
1. **FanDuel** - Base comparison (sharpest lines)
2. **DraftKings** - Alternate lines, props
3. **BetMGM** - Early week lines
4. **Circa Sports** - Sharp line validation

### Secondary Books (Check for Arbs/+EV)
5. **Caesars** - Promos, boosts
6. **ESPN BET** - Public soft lines
7. **Fanatics** - New user soft lines
8. **bet365** - International lines

### Workflow
```
1. Model generates prediction → Edge identified
2. Check FanDuel line (baseline)
3. Check DraftKings, BetMGM, Circa (best line?)
4. Check Caesars, ESPN BET (soft line arb?)
5. Place bet at book with best odds
6. Log bet with book_id, odds, timestamp
```

### Expected Improvement from Line Shopping
- **Average**: +2-3 cents per bet (e.g., -110 → -107)
- **Best case**: +5-8 cents (find soft line)
- **Impact on ROI**: +1-2% over season
- **Example**: $100 bet at -110 → $90.91 profit | $100 bet at -107 → $93.46 profit (+$2.55)

---

## Odds Comparison (Typical NFL Sunday)

**Example Game: Chiefs @ Bills**

| Book | Spread | Total | Moneyline (KC) |
|------|--------|-------|----------------|
| FanDuel | KC -3 (-110) | 52.5 (-110) | -155 |
| DraftKings | KC -3 (-108) | 52.5 (-112) | -150 |
| BetMGM | KC -2.5 (-115) | 52 (-110) | -152 |
| Caesars | KC -3 (-110) | 52.5 (-105) | -160 |
| ESPN BET | KC -3 (-112) | 53 (-110) | -158 |
| Circa | KC -3.5 (-110) | 52.5 (-108) | -165 |

**Best Bets**:
- Spread: BetMGM KC -2.5 (-115) - half point cheaper
- Total: Caesars O52.5 (-105) - 5 cents better
- ML: DraftKings KC -150 - 5 cents better

**Line Shopping Profit**:
- Spread bet: 0.5 point value ≈ $20-30 EV on $1k bet
- Total bet: 5 cents ≈ $5 EV on $100 bet
- ML bet: 5 cents ≈ $3 EV on $100 bet

---

## Account Management

### Multi-Book Strategy
**Required**:
- Maintain active accounts at 5-8 books minimum
- Keep $1k-2k balance at each book (total: $5k-16k)
- Rotate books to avoid pattern detection

**Recommended Allocation** ($10k total bankroll):
1. FanDuel: $2k (40% of bets)
2. DraftKings: $2k (30% of bets)
3. BetMGM: $1.5k (15% of bets)
4. Caesars: $1k (10% of bets)
5. ESPN BET: $1k (5% of bets)
6. Circa: $1k (sharp validation, small bets)
7. bet365: $0.5k (arb opportunities)
8. Fanatics: $1k (promo hunting)

### Deposit/Withdrawal Strategy
- **Deposit**: ACH (free), credit card (instant but fees)
- **Withdrawal**: ACH (2-5 days), PayPal (24-48 hours)
- **Frequency**: Weekly consolidation to primary bank
- **Tax**: Track all transactions (Form W-2G if $600+ win on single bet)

### Account Security
- Use unique passwords (1Password/LastPass)
- Enable 2FA on all books
- Never share login credentials
- Use VPN only if traveling (not for geo-spoofing)

---

## Promo Strategy (Maximize EV)

### New User Promos (One-Time)
- **FanDuel**: "Bet $5, Get $200" - +EV if used correctly
- **DraftKings**: "Deposit Match up to $1k" - 20% playthrough
- **Caesars**: "$1k First Bet on Caesars" - risk-free bet
- **Strategy**: Use on +EV bets identified by model, not random

### Recurring Promos (Weekly)
- **Odds Boosts**: Check daily, compare to model fair odds
  - Example: FanDuel boosts Chiefs -3 from -110 to +100
  - Model says fair = -105 → +5 cents value → BET
- **Profit Boosts**: 25-50% boost on winnings
  - Use on highest edge bets
- **Parlay Insurance**: Skip (parlays -EV per Task 10)

### Promo EV Calculation
```python
# Odds boost example
normal_odds = -110  # implied prob: 52.4%
boosted_odds = +100  # implied prob: 50.0%
model_prob = 0.55   # our edge: +5%

# EV calculation
normal_ev = (0.55 * 0.909) - (0.45 * 1) = 0.0495 - 0.45 = +0.0495 (+4.95%)
boosted_ev = (0.55 * 1.0) - (0.45 * 1) = 0.55 - 0.45 = +0.10 (+10%)

# Boost adds +5.05% EV
```

---

## Limits and Account Restrictions

### Expected Limits (Winning Bettors)
- **Recreational books** (ESPN BET, Fanatics): May limit after $5k-10k profit
- **Mainstream books** (FanDuel, DraftKings): Limits after $20k-50k profit
- **Sharp books** (Circa, BetMGM): Higher thresholds but still monitor

### Avoiding Limits
1. **Bet sizing**: Vary bet amounts (don't always bet exact Kelly)
2. **Timing**: Don't always bet immediately when lines open
3. **Bet types**: Mix spreads, totals, moneylines (avoid pattern)
4. **Losing bets**: Occasionally bet slightly -EV bets (camouflage)
5. **Props**: Sprinkle in some props (most books love prop action)
6. **Round numbers**: Bet $237 instead of $250 (looks recreational)

### What to Do When Limited
- **Partial limits**: Accept reduced max bet, continue betting
- **Full limits**: Move to other books, use "bearding" (not recommended legally)
- **Appeal**: Contact customer service (rarely successful)
- **Long-term**: Sharp books (Circa, Pinnacle offshore) accept winners

---

## Data Integration (Line Shopping Automation)

### Manual Entry (Current State)
- Check 4-5 books manually for each bet
- Log odds in spreadsheet
- Time cost: ~5 min per bet
- Human error: Occasional missed best line

### Automated Line Shopping (Future - Task 7)
```python
# py/production/line_shopping.py (to be implemented)
odds_aggregator = OddsAggregator(
    books=['fanduel', 'draftkings', 'betmgm', 'caesars'],
    api_keys={...},
)

best_odds = odds_aggregator.get_best_odds(
    game='KC_vs_BUF',
    bet_type='spread',
    side='KC',
)

# Returns:
# {
#     'book': 'betmgm',
#     'odds': -108,
#     'line': -2.5,
#     'improvement_vs_consensus': +0.5 points,
#     'ev_gain': +$25 on $1000 bet
# }
```

### Data Sources (Free)
- **The Odds API** (theoddsapi.com): 500 free requests/month
- **Action Network** (actionnetwork.com): Public consensus, line movement
- **Covers** (covers.com): Opening/closing line database
- **ESPN** (espn.com): Free live odds display

### Data Sources (Paid)
- **SportsDataIO** ($500/month): Real-time odds feed, 15+ books
- **DonBest** ($300/month): Line movement tracking, steam moves
- **Bet Labs** ($99/month): Historical line analysis
- **NFL Pro** ($40/year): Injury reports, advanced stats (Task 8 analysis)

---

## Tax Considerations

### Federal Tax (IRS)
- **Form W-2G**: Issued if single bet wins $600+ (net $300+ profit)
- **Reporting**: All gambling winnings taxable (even without W-2G)
- **Deductions**: Losses deductible up to winnings (itemize Schedule A)
- **Record Keeping**: Mandatory bet log (date, book, amount, result)

### Virginia State Tax
- **Rate**: 4% on net gambling winnings
- **Reporting**: Include on VA state return (Form 760)
- **Professional Gambler**: If full-time, may file Schedule C (business income)

### Tax Optimization
- **Offset wins/losses**: Realize losses in same tax year as wins
- **Professional status**: Deduct expenses (software, data, travel) if professional
- **Quarterly estimates**: Pay estimated tax if winnings >$1k per quarter
- **CPA consultation**: Recommended if annual handle >$100k

---

## Risk Management

### Geolocation
- **Requirement**: Must be physically in Virginia to place bets
- **Technology**: GPS + WiFi + Cell tower triangulation
- **Travel**: Can withdraw/deposit from anywhere, can't bet outside VA
- **VPN**: Do NOT use (instant account suspension)

### Fraud Prevention
- **Identity verification**: SSN, photo ID, address proof required
- **Deposit limits**: $500-1k daily for new accounts
- **Withdrawal verification**: Enhanced for $10k+ withdrawals
- **Suspicious activity**: Large bet sizing swings, arb patterns flagged

### Bankroll Security
- **FDIC Insurance**: Most books NOT FDIC insured (keep minimal balance)
- **Book Bankruptcy**: Risk if book goes under (historical: 5% of books)
- **Recommendation**: Withdraw weekly, keep only 1-2 weeks bankroll in books

---

## Competitive Landscape (2025)

### Market Trends
- **Consolidation**: M&A activity (ESPN acquires Barstool, etc.)
- **Promos declining**: Less aggressive new user bonuses than 2021-22
- **Vig increasing**: Some books moving from -110 to -115 on NFL spreads
- **Props expansion**: 300% growth in player props since 2021
- **Live betting**: 40% of handle now in-game

### Virginia vs Other States
- **Advantage**: No betting tax on bettors (NJ, NY, IL have state taxes on wins)
- **Advantage**: College betting allowed (unlike IL, LA, NY)
- **Disadvantage**: No in-person betting (unlike NJ, NV)
- **Disadvantage**: No offshore books (unlike unregulated states)

### Future Outlook
- **Expected**: 2-3 more books to launch by 2026
- **Trend**: Books using AI for limit/ban decisions (harder to beat)
- **Opportunity**: Micro-betting, player props (less efficient markets)

---

## Operational Checklist

### Daily Operations
- [ ] Check line movements (8 AM, 12 PM, 6 PM)
- [ ] Review model predictions for upcoming games
- [ ] Check odds boosts across 5+ books
- [ ] Monitor promo emails (FanDuel, DraftKings)
- [ ] Update bet log with previous day results

### Weekly Operations
- [ ] Withdraw winnings (consolidate to primary bank)
- [ ] Rebalance book accounts (maintain $1k-2k each)
- [ ] Review performance (win rate, ROI, by book)
- [ ] Check for new promos/bonuses
- [ ] Backup bet log (export to CSV)

### Monthly Operations
- [ ] Tax tracking (calculate YTD wins/losses)
- [ ] Book account audit (verify balances)
- [ ] Limit monitoring (check if any books restricted)
- [ ] Strategy review (line shopping effectiveness)
- [ ] Data analysis (which books offer best odds?)

---

## Recommended Initial Setup

### Phase 1: Account Creation (Week 1)
1. Sign up for FanDuel, DraftKings, BetMGM (Tier 1)
2. Complete identity verification
3. Claim new user promos
4. Deposit $2k each ($6k total)

### Phase 2: Expansion (Week 2-3)
5. Add Caesars, ESPN BET (Tier 2)
6. Deposit $1k each ($2k total)
7. Test bet workflow on each platform
8. Setup withdrawal methods (ACH, PayPal)

### Phase 3: Advanced (Week 4+)
9. Add Circa, bet365 (sharp books)
10. Implement line shopping workflow
11. Track odds variance by book
12. Optimize account allocation

### Total Capital Required
- **Betting bankroll**: $10k (primary capital)
- **Book floats**: $8k (distributed across books)
- **Reserve**: $2k (cover withdrawals/deposits)
- **Total**: $20k recommended

---

## Summary

**Virginia Sportsbook Landscape**:
- 15+ legal books (highly competitive)
- No state tax on winnings (bettor-friendly)
- Tier 1: FanDuel, DraftKings, BetMGM (must-have)
- Tier 2: Caesars, ESPN BET, bet365 (line shopping)
- Tier 3: Circa, Fanatics, others (specialty use)

**Line Shopping Value**:
- +2-3 cents average per bet
- +1-2% ROI over season
- Required for +EV betting (vig ≈2-5%)

**Best Practices**:
- Maintain 5-8 active accounts
- Check 4-5 books before every bet
- Rotate books to avoid pattern detection
- Use promos strategically (odds boosts, profit boosts)
- Withdraw weekly (minimize book risk)

**Next Steps**:
1. Create accounts (Task 6: DONE)
2. Build line shopping aggregator (Task 7: PENDING)
3. Implement automated odds monitoring (Task 7: PENDING)

---

*Last Updated: 2025-10-10*
