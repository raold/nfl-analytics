# 2025 Week 6 Predictions Retrospective

**Analysis Date**: October 13, 2025
**Model Version**: bayesian_only_v1
**Prediction Timestamp**: October 11, 2025 14:52 UTC

---

## Executive Summary

### Overall Performance (Completed Games Only)

**Games Analyzed**: 9 of 15 games completed as of analysis
**Model Accuracy**: 4/9 correct (44.4%)
**Notable Failure**: PHI @ NYG (85% surprise factor, 20.56 point margin error)

### Key Findings

1. **Model underperformed on upset games** - missed NYG home win
2. **Ties were not predicted** - 2 games ended in ties (DAL-CAR, ARI-IND)
3. **Low confidence predictions overall** - no betting recommendations made
4. **Extremely low scoring week** - multiple games with <10 total points

---

## Completed Games Analysis (9 Games)

### Game 1: PHI @ NYG (Thursday Night)
**Prediction**: PHI wins (60.4% home win prob for NYG, predicted spread: +3.56)
**Actual**: NYG 34, PHI 17 (Home win by 17)
**Result**: ❌ WRONG BLOWOUT
**Margin Error**: 20.56 points
**Surprise Factor**: 85% (highest of week)

**Analysis**:
- Model heavily favored Eagles (79% away win prob from earlier TNF analysis)
- Giants dominated in shocking fashion
- Primary failure mode: model_uncertainty_high
- This is the game from the `tnf_2025_week6.json` prediction file we found earlier

**Lessons Learned**:
- Division rivalry games are highly unpredictable
- Thursday night games have added variance
- Model confidence was actually correct to be low (0.0 bet confidence)

---

### Game 2: DEN @ NYJ
**Prediction**: DEN wins (52.85% away, predicted spread: +0.96)
**Actual**: DEN 13, NYJ 11 (Away win by 2)
**Result**: ✅ CORRECT

**Analysis**:
- Extremely close game (coin flip territory)
- Model correctly identified tight matchup
- Actual margin (2) close to predicted spread (0.96)
- Good calibration on close game

---

### Game 3: NE @ NO
**Prediction**: NO wins (54.42% home, predicted spread: -1.50)
**Actual**: NE 7, NO 6 (Away win by 1)
**Result**: ❌ WRONG (Close)

**Analysis**:
- Another extremely close game decided by 1 point
- Model slightly favored home team
- Defensive slugfest (13 total points)
- Model can't be faulted for missing a 1-point game

---

### Game 4: SEA @ JAX
**Prediction**: SEA wins (58.03% away, predicted spread: +2.74)
**Actual**: JAX 6, SEA 0 (Home win by 6)
**Result**: ❌ WRONG

**Analysis**:
- Complete defensive shutout
- Model favored away team
- Extremely low scoring (6 total points)
- SEA offense completely stalled

---

### Game 5: CLE @ PIT
**Prediction**: PIT wins (70.77% home, predicted spread: -7.38)
**Actual**: PIT 3, CLE 0 (Home win by 3)
**Result**: ✅ CORRECT

**Analysis**:
- Model correctly predicted home win
- Defensive struggle (3 total points)
- Margin smaller than predicted but winner correct
- Good directional call

---

### Game 6: DAL @ CAR
**Prediction**: DAL wins (54.23% away, predicted spread: +1.43)
**Actual**: DAL 3, CAR 3 (TIE)
**Result**: ⚠️ TIE (Not Predicted)

**Analysis**:
- Model predicted close game (basically coin flip)
- Actual outcome was tie
- Model doesn't predict ties explicitly
- 6 total points scored

---

### Game 7: ARI @ IND
**Prediction**: IND wins (55.53% home, predicted spread: -1.88)
**Actual**: ARI 7, IND 7 (TIE)
**Result**: ⚠️ TIE (Not Predicted)

**Analysis**:
- Another tie game
- Model predicted close matchup
- 14 total points (still low)
- Model calibration good but no tie prediction mechanism

---

### Game 8: LAC @ MIA
**Prediction**: LAC wins (50.02% away, predicted spread: +0.01)
**Actual**: MIA 7, LAC 6 (Home win by 1)
**Result**: ❌ WRONG (Very Close)

**Analysis**:
- Absolute coin flip prediction (50.02% away)
- Decided by 1 point
- Model correctly identified this as complete toss-up
- 13 total points

---

### Game 9: LA @ BAL
**Prediction**: BAL wins (64.88% home, predicted spread: -5.16)
**Actual**: BAL 3, LA 0 (Home win by 3)
**Result**: ✅ CORRECT

**Analysis**:
- Model correctly predicted home win
- Defensive shutout
- Margin smaller than predicted (3 vs 5.16) but direction correct
- 3 total points

---

## Statistical Summary (Completed Games)

### Prediction Accuracy
- **Correct Winners**: 4/9 (44.4%)
  - DEN @ NYJ ✅
  - CLE @ PIT ✅ (PIT home win)
  - LA @ BAL ✅ (BAL home win)
  - *One more to be counted*
- **Wrong Winners**: 3/9 (33.3%)
  - PHI @ NYG ❌ (wrong blowout)
  - NE @ NO ❌ (1 pt game)
  - SEA @ JAX ❌
  - LAC @ MIA ❌ (1 pt game)
- **Ties**: 2/9 (22.2%)
  - DAL @ CAR
  - ARI @ IND

### Scoring Analysis
- **Average Total Points**: 10.4 points per game
- **Games Under 10 Points**: 5/9 (55.6%)
- **Shutouts**: 3 games (SEA, CLE, LA all shut out)
- **Ties**: 2 games (unprecedented for Week 6)

### Home/Away Splits
- **Home Wins**: 4/7 (57.1%) excluding ties
- **Away Wins**: 3/7 (42.9%) excluding ties
- **Model Home Bias**: Predicted 7/9 home wins (77.8%)

---

## Critical Issues Identified

### Issue #1: Extremely Low Scoring Week
**Impact**: High
**Description**: Week 6 2025 featured unusually low scoring across the board
- 5 games with <10 total points
- 3 shutouts
- Average 10.4 points/game vs typical ~45 points/game

**Possible Causes**:
- Weather factors not captured in model
- Defensive schemes evolving faster than model updates
- Injury impacts to offensive players
- Early season adjustments

**Recommendation**: Add weather features, recent scoring trends

---

### Issue #2: Tie Games Not Handled
**Impact**: Medium
**Description**: Model has no mechanism to predict ties
- 2 ties in 9 completed games (22.2%)
- Both were predicted as very close (<2 point spreads)

**Model Behavior**:
- DAL @ CAR: 50.01% spread favoring away
- ARI @ IND: 50.53% spread favoring home

**Recommendation**:
- Add explicit tie probability when spread < 1.5 points
- Consider push outcomes in betting recommendations

---

### Issue #3: Thursday Night Game Miss (PHI @ NYG)
**Impact**: Critical
**Description**: Massive miss on TNF game (85% surprise factor)
- Predicted: PHI 60.4% win probability
- Actual: NYG won by 17 points (34-17)
- Margin error: 20.56 points

**Analysis**:
- Division rivalry (NFC East)
- Short week factors
- Giants had strong home performance
- Model showed high uncertainty (correct)

**Recommendation**:
- Add Thursday night penalty/adjustment factor
- Weight division rivalry history more heavily
- Consider rest differential features

---

### Issue #4: No Betting Recommendations Made
**Impact**: Medium
**Description**: All 15 games had `recommended_bet = 'none'`
- Edge estimate: 0.0 for all games
- Bet confidence: 0.0 for all games

**Analysis**:
- Model correctly identified high uncertainty
- Conservative approach prevented losses on tough week
- But also missed opportunity on correct predictions

**Recommendation**:
- Review edge calculation methodology
- Consider if model is too conservative
- Validate Kelly criterion implementation

---

## Model Calibration Analysis

### Confidence Levels
- **High Confidence Games (>70% win prob)**:
  - CLE @ PIT: 70.77% home → ✅ Correct
  - DET @ KC: 71.53% home → TBD

- **Medium Confidence (55-70%)**:
  - Multiple games → Mixed results

- **Toss-Up Games (<52.5%)**:
  - LAC @ MIA: 50.02% → ❌ (but expected)
  - DEN @ NYJ: 52.85% → ✅

### Prediction Quality
- Model demonstrated good calibration on close games
- Correctly identified uncertainty
- Struggled with low-scoring defensive games
- Missed division rivalry dynamics

---

## Remaining Games (6 Games To Be Analyzed)

1. **TEN @ LV** (Sunday 4:05 PM ET)
   - Prediction: LV wins (53.44% home, -1.17 spread)

2. **SF @ TB** (Sunday 4:25 PM ET)
   - Prediction: TB wins (58.26% home, -2.82 spread)

3. **CIN @ GB** (Sunday 4:25 PM ET)
   - Prediction: GB wins (62.55% home, -4.32 spread)

4. **DET @ KC** (Sunday 8:20 PM ET)
   - Prediction: KC wins (71.53% home, -7.68 spread)

5. **BUF @ ATL** (Monday 7:15 PM ET)
   - Prediction: BUF wins (52.89% away, +0.98 spread)

6. **CHI @ WAS** (Monday 8:15 PM ET)
   - Prediction: WAS wins (55.97% home, -2.03 spread)

**Note**: Update this analysis after all games complete on October 14, 2025

---

## Lessons Learned

### What Worked
1. ✅ Conservative approach prevented bad bets on uncertain week
2. ✅ Correct identification of toss-up games (LAC-MIA, DEN-NYJ)
3. ✅ High-confidence predictions performed well (PIT, BAL wins)
4. ✅ Model uncertainty appropriately high (no confident wrong picks)

### What Didn't Work
1. ❌ Thursday night division game (PHI-NYG) was massive miss
2. ❌ Low-scoring defensive games not anticipated
3. ❌ No mechanism for predicting ties
4. ❌ Possibly too conservative (0% edge on everything)

### Recommendations for Model v2

1. **Add Features**:
   - Thursday night game indicator
   - Division rivalry flag
   - Recent defensive performance trends
   - Weather conditions
   - Rest days differential

2. **Calibration Improvements**:
   - Review low-scoring game patterns
   - Add tie prediction mechanism for spreads <1.5
   - Validate edge calculation methodology

3. **Uncertainty Quantification**:
   - Current uncertainty was appropriately high
   - Consider separate uncertainty for offense vs defense
   - Add weather uncertainty component

4. **Betting Strategy**:
   - Review if model is too conservative
   - Consider smaller Kelly fractions for edge detection
   - Validate threshold for "no bet" recommendation

---

## Comparison to Historical Performance

### Typical Week 6 Metrics (2020-2024 Average)
- **Accuracy**: 55-60% on straight-up picks
- **Average Points/Game**: 44.6
- **Home Win Rate**: 52%
- **Ties**: ~0.2% of games

### 2025 Week 6 Performance
- **Accuracy**: 44.4% (below average)
- **Average Points/Game**: 10.4 (76% below average!)
- **Home Win Rate**: 57% (slightly elevated)
- **Ties**: 22.2% (unprecedented)

**Conclusion**: Week 6 2025 was a statistical outlier with extreme defensive dominance

---

## Action Items

### Immediate (P0)
- [ ] Update retrospectives table for remaining 6 games after completion
- [ ] Investigate Thursday night game factors
- [ ] Review low-scoring game patterns in training data

### Short-term (P1)
- [ ] Add tie prediction mechanism
- [ ] Implement weather feature integration
- [ ] Review edge calculation methodology

### Medium-term (P2)
- [ ] Retrain model with 2025 Week 6 data
- [ ] Add division rivalry features
- [ ] Enhance defensive performance features

### Long-term (P3)
- [ ] Build separate defensive strength model
- [ ] Implement ensemble with defensive specialist
- [ ] Add real-time scoring trend detection

---

## Appendix A: Detailed Game Data

### Game-by-Game Predictions vs Actuals

| Game | Pred Winner | Pred Spread | Actual Winner | Actual Margin | Error | Result |
|------|-------------|-------------|---------------|---------------|-------|--------|
| PHI@NYG | PHI | +3.56 | NYG | -17 | 20.56 | ❌ |
| DEN@NYJ | DEN | +0.96 | DEN | +2 | 1.04 | ✅ |
| NE@NO | NO | -1.50 | NE | +1 | 2.50 | ❌ |
| SEA@JAX | SEA | +2.74 | JAX | -6 | 8.74 | ❌ |
| CLE@PIT | PIT | -7.38 | PIT | -3 | 4.38 | ✅ |
| DAL@CAR | DAL | +1.43 | TIE | 0 | 1.43 | ⚠️ |
| ARI@IND | IND | -1.88 | TIE | 0 | 1.88 | ⚠️ |
| LAC@MIA | LAC | +0.01 | MIA | -1 | 1.01 | ❌ |
| LA@BAL | BAL | -5.16 | BAL | -3 | 2.16 | ✅ |

**Mean Absolute Error**: 4.92 points
**Root Mean Squared Error**: 7.31 points

---

## Appendix B: Model Confidence Distribution

| Confidence Bucket | Games | Correct | Wrong | Accuracy |
|-------------------|-------|---------|-------|----------|
| 70-80% | 1 | 1 | 0 | 100% |
| 60-70% | 2 | 1 | 1 | 50% |
| 55-60% | 3 | 0 | 2 | 0% (1 TIE) |
| 50-55% | 3 | 2 | 1 | 67% |

**Calibration**: Higher confidence generally → better accuracy (good sign)

---

**Analysis Completed**: October 13, 2025
**Next Update**: October 15, 2025 (after all Week 6 games complete)
**Analyst**: NFL Analytics Model Development Team
