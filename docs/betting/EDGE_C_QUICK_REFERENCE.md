# Edge C Quick Reference Card

**Last Updated:** November 16, 2025
**Status:** Paper Trading Phase

---

## The Edge

**Edge C: Lingering Fatigue**
- QBs underperform -195.6 yards in next game after international travel (≤7 days)
- Statistical significance: p < 0.0001 (n=24)
- Expected win rate: 70%
- Expected ROI: +33.6%

---

## When to Bet

✅ **GO Criteria (ALL must be met):**
1. QB played international game (London/Munich/Brazil)
2. Next game ≤7 days after international game
3. Prop odds ≤ -130
4. Baseline calculation fix validated
5. Paper trade performance ≥60% win rate (after 10 trades)

❌ **NO-GO (ANY triggers abort):**
1. Baseline fix not validated
2. QB injured/inactive
3. Extreme weather (30+ mph winds, snow)
4. Odds worse than -130

---

## How to Size Bets

**Formula:**
```
Fractional Kelly = 10% × Variance-Adjusted Kelly
                 = 10% × 18%
                 = 1.8% of bankroll

Max Bet = min(1.8% bankroll, 2% bankroll)
        = 2% bankroll (capped)
```

**Example ($10,000 bankroll):**
- Kelly recommendation: $180
- Max bet cap: $200
- **Bet size: $180-200**

**Calculator:**
```bash
python scripts/calculate_edge_c_bet_size.py --interactive
```

---

## Risk Management

**Limits:**
- Max bet: 2% of bankroll
- Max bets/week: 3
- Max bets/day: 1

**Exit Criteria:**
- Abandon if ROI <0% after 20 live bets
- Reduce Kelly to 5% if win rate 55-59% after 10 bets
- Pause if 5 consecutive losses

---

## 2025 Opportunities

| Week | International | Location | Potential Targets |
|------|---------------|----------|-------------------|
| 6 | Week 5 (Oct 6) | London | Darnold (MIN), Rodgers (NYJ) |
| 7 | Week 6 (Oct 13) | London | Williams (CHI), Lawrence (JAX) |
| 8 | Week 7 (Oct 20) | London | Lawrence (JAX), Maye (NE) |
| 11 | Week 10 (Nov 10) | Munich | Jones (NYG), Young (CAR) |

**Expected Total:** 6-8 paper trades

---

## Quick Commands

**Monitor Opportunities:**
```bash
python scripts/monitor_international_schedule.py
```

**Calculate Bet Size:**
```bash
python scripts/calculate_edge_c_bet_size.py \
  --bankroll 10000 --win-rate 0.70 --odds -110
```

**View Dashboard:**
```bash
python scripts/edge_c_dashboard.py --update
```

**Track Paper Trade:**
1. Add row to `results/edge_c_paper_trades.csv`
2. After game, update `actual_yards`, `outcome`, `profit_loss`, `roi`

---

## Paper Trading Workflow

**Before Game:**
1. Run monitor script → identify opportunity
2. Calculate bet size → log in CSV
3. Record QB baseline, predicted yards, prop line

**After Game:**
1. Get actual QB passing yards
2. Determine outcome (WIN/LOSS/PUSH)
3. Calculate P&L and ROI
4. Update CSV

**After 10 Trades:**
1. Run dashboard → check performance
2. If win rate ≥60% AND ROI ≥+10% → ready for live betting
3. If not → collect 10 more or abandon edge

---

## Files

| File | Purpose |
|------|---------|
| `scripts/monitor_international_schedule.py` | Alert system |
| `scripts/calculate_edge_c_bet_size.py` | Bet sizing |
| `scripts/edge_c_dashboard.py` | Performance dashboard |
| `results/edge_c_paper_trades.csv` | Paper trade log |
| `docs/betting/EDGE_C_PLAYBOOK.md` | Full playbook (50+ pages) |

---

## Current Status

**Baseline Fix:** PENDING (4 hours estimated)
**Paper Trading:** Ready to start Oct 2025
**Live Betting:** Pending validation (Jan 2026 earliest)

**Blocker:** Baseline calculation uses incorrect `current_team` filter
**Impact:** Penalty magnitude uncertain until fix validated
**Action:** Paper trade only until baseline fix complete

---

## Support

**Questions:** See `docs/betting/EDGE_C_PLAYBOOK.md`
**Issues:** Check dashboard for latest stats
**Analysis:** `results/international_betting_edges.json`

---

**Remember:** This is real money. Paper trade first. Validate the edge. Manage risk ruthlessly.
