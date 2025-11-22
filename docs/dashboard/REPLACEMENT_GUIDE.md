# HTML to Native Component - Exact Replacement Guide

**Date:** November 22, 2025
**Purpose:** Step-by-step guide for replacing remaining 5 HTML blocks

---

## Status Update

**Completed:** 2/7 (Custom CSS, Service Health Cards)
**Remaining:** 5/7 (Game cards)

**Component Library:** Enhanced `game_prediction_card()` now supports:
- `card_type="scheduled"` - Upcoming games
- `card_type="live"` - In-progress games
- `card_type="completed"` - Finished games
- `card_type="international"` - International games

---

## Remaining Replacements

### Instance #3: Live Game Card (app.py ~line 1733)

**Current Code (13 lines of HTML):**
```python
st.markdown(f"""
<div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; margin: 10px 0;
            border: 3px solid #ff8c00; box-shadow: 0 0 15px rgba(255, 140, 0, 0.3);">
    <div style="text-align: center; margin-bottom: 10px;">
        <div style="display: inline-flex; align-items: center; background-color: #ff8c00;
                   color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
            <span style="font-size: 18px; margin-right: 8px;">🔴</span>
            <span style="font-size: 16px;">LIVE - {status_display}</span>
        </div>
    </div>
    ... [more HTML] ...
</div>
""", unsafe_allow_html=True)
```

**Replacement (1 line):**
```python
game_prediction_card(
    game=game,
    prediction=pred,
    spread_edge=spread_edge,
    total_edge=total_edge,
    card_type="live"
)
```

**Search Pattern:** Look for `unsafe_allow_html=True` around line 1733
**Context:** Inside `for game_data in in_progress_games:` loop

---

### Instance #4: Completed Game Card (app.py ~line 1855)

**Current Code (similar HTML structure):**
```python
st.markdown(f"""
<div style="background-color: #d4edda; padding: 20px; border-radius: 10px; margin: 10px 0;
            border-left: 5px solid #28a745;">
    ... [team logos, scores, prediction info] ...
</div>
""", unsafe_allow_html=True)
```

**Replacement:**
```python
game_prediction_card(
    game=game,
    prediction=pred,
    spread_edge=spread_edge,
    total_edge=total_edge,
    card_type="completed"
)
```

**Search Pattern:** Look for `unsafe_allow_html=True` around line 1855
**Context:** Inside `for game_data in completed_games:` loop

---

### Instance #5: Upcoming Game Card (app.py ~line 1945)

**Current Code:**
```python
st.markdown(f"""
<div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin: 10px 0;
            border-left: 5px solid {edge_color};">
    ... [betting lines, predictions] ...
</div>
""", unsafe_allow_html=True)
```

**Replacement:**
```python
game_prediction_card(
    game=game,
    prediction=pred,
    spread_edge=spread_edge,
    total_edge=total_edge,
    card_type="scheduled"
)
```

**Search Pattern:** Look for `unsafe_allow_html=True` around line 1945
**Context:** Inside `for game_data in upcoming_games:` loop

---

### Instance #6: Top Players Card (app.py ~line 1973)

**Current Code (10 lines):**
```python
st.markdown(f"""
<div style="text-align: center; padding: 15px; background-color: #f8f9fa;
            border-radius: 10px; margin: 10px 0;">
    <img src="{get_team_logo_url(team)}" width="80" style="margin-bottom: 10px;">
    <div style="font-size: 20px; font-weight: bold; color: #495057;">{team}</div>
</div>
""", unsafe_allow_html=True)
```

**Replacement Strategy:**
This is NOT a game card - it's a player/team display. Two options:

**Option A:** Use Streamlit native components directly (recommended):
```python
# Replace HTML with:
with st.container():
    st.image(get_team_logo_url(team), width=80)
    st.markdown(f"**{team}**")
```

**Option B:** Create new `player_team_card()` component:
```python
# In dashboard/components/cards.py:
def player_team_card(team: str, logo_url: str):
    """Render player/team card with logo."""
    with st.container():
        st.image(logo_url, width=80)
        st.markdown(f"### {team}")

# In app.py:
player_team_card(team, get_team_logo_url(team))
```

**Recommendation:** Use Option A (simpler, only 2 lines)

---

### Instance #7: Game Prediction Card (game_predictions_patch.py line 438)

**Current Code (57 lines - LARGEST HTML BLOCK):**
```python
with st.container():
    st.markdown(f"""
    <div style="{card_style} padding: 20px; border-radius: 10px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div style="text-align: center; flex: 1;">
                <img src="{get_team_logo_url(game['away_team'])}" width="60" style="margin-bottom: 5px;">
                <div style="font-size: 24px; font-weight: bold; color: #212529;">{game['away_team']}</div>
            </div>
            <div style="text-align: center; flex: 0.3;">
                <div style="font-size: 20px; color: #6c757d;">@</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <img src="{get_team_logo_url(game['home_team'])}" width="60" style="margin-bottom: 5px;">
                <div style="font-size: 24px; font-weight: bold; color: #212529;">{game['home_team']}</div>
            </div>
        </div>

        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <thead>
                <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 10px; text-align: left; font-weight: bold;">Bet Type</th>
                    <th style="padding: 10px; text-align: center; font-weight: bold;">Sportsbook</th>
                    <th style="padding: 10px; text-align: center; font-weight: bold;">Our Model</th>
                    <th style="padding: 10px; text-align: center; font-weight: bold;">Edge</th>
                </tr>
            </thead>
            <tbody>
                <tr style="{spread_color}">
                    <td style="padding: 10px; font-weight: bold;">Spread ({game['home_team']})</td>
                    <td style="padding: 10px; text-align: center;">{game['spread_close']:+.1f}</td>
                    <td style="padding: 10px; text-align: center;">{pred.get('predicted_spread', 0):+.1f}</td>
                    <td style="padding: 10px; text-align: center; font-weight: bold; color: {...};">{spread_edge:+.1f}</td>
                </tr>
                <tr style="{total_color}">
                    <td style="padding: 10px; font-weight: bold;">Total (O/U)</td>
                    <td style="padding: 10px; text-align: center;">{game['total_close']:.1f}</td>
                    <td style="padding: 10px; text-align: center;">{pred.get('predicted_total', 0):.1f}</td>
                    <td style="padding: 10px; text-align: center; font-weight: bold; color: {...};">{total_edge:+.1f}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 10px; font-weight: bold;">Moneyline ({game['home_team']})</td>
                    <td style="padding: 10px; text-align: center;">{game['home_moneyline']:+.0f}</td>
                    <td style="padding: 10px; text-align: center;" colspan="2">
                        Win Prob: {pred.get('predicted_home_win_prob', 0.5):.1%}
                    </td>
                </tr>
            </tbody>
        </table>

        <div style="margin-top: 15px; padding: 10px; background-color: rgba(0,0,0,0.03); border-radius: 5px;">
            <div style="font-size: 14px; color: #495057; margin-bottom: 5px;">
                <strong>📊 Predicted Score:</strong> {game['away_team']} {pred.get('predicted_away_score', 0):.1f} @ {game['home_team']} {pred.get('predicted_home_score', 0):.1f}
            </div>
            <div style="font-size: 14px; color: #495057;">
                <strong>🎯 QB Passing:</strong> {game['away_team']} {pred.get('away_qb_yards', 0):.0f}±{pred.get('away_qb_yards_std', 0):.0f} yds | {game['home_team']} {pred.get('home_qb_yards', 0):.0f}±{pred.get('home_qb_yards_std', 0):.0f} yds
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**Replacement (5 lines):**
```python
with st.container():
    game_prediction_card(
        game=game,
        prediction=pred,
        spread_edge=spread_edge,
        total_edge=total_edge,
        card_type="scheduled"
    )
```

**Additional Change Required:**
Add import at top of `game_predictions_patch.py`:
```python
from dashboard.components import game_prediction_card
```

---

## Step-by-Step Implementation

### Step 1: Verify Component Library (✅ Complete)
```bash
# Check component library exists
ls -l dashboard/components/cards.py
ls -l dashboard/components/__init__.py

# Check import in app.py
grep "from dashboard.components import" dashboard/app.py
```

### Step 2: Test Component Rendering

Create test script `scripts/test_game_card_component.py`:
```python
import streamlit as st
from dashboard.components import game_prediction_card

# Test data
test_game = {
    'away_team': 'KC',
    'home_team': 'BUF',
    'away_team_display': 'Kansas City Chiefs',
    'home_team_display': 'Buffalo Bills',
    'spread_close': -2.5,
    'total_close': 51.5,
    'home_moneyline': 120,
    'away_score': 24,
    'home_score': 21,
}

test_prediction = {
    'predicted_spread': -3.0,
    'predicted_total': 53.0,
    'predicted_away_score': 27.0,
    'predicted_home_score': 24.0,
    'predicted_home_win_prob': 0.48,
    'away_qb_yards': 285,
    'away_qb_yards_std': 45,
    'home_qb_yards': 265,
    'home_qb_yards_std': 40,
}

st.header("Test: Scheduled Game")
game_prediction_card(test_game, test_prediction, -0.5, 1.5, "scheduled")

st.header("Test: Live Game")
game_prediction_card(test_game, test_prediction, -0.5, 1.5, "live")

st.header("Test: Completed Game")
game_prediction_card(test_game, test_prediction, -0.5, 1.5, "completed")
```

Run test:
```bash
streamlit run scripts/test_game_card_component.py
```

### Step 3: Replace HTML Blocks

For each instance #3-#7:

1. **Locate the HTML block:**
   ```bash
   grep -n "unsafe_allow_html=True" dashboard/app.py
   grep -n "unsafe_allow_html=True" dashboard/game_predictions_patch.py
   ```

2. **Identify the context:**
   - Which loop is it in? (completed_games, in_progress_games, upcoming_games)
   - What variables are available? (game, pred, spread_edge, total_edge)

3. **Replace with component:**
   - Use appropriate `card_type` based on game state
   - Ensure all required variables are passed

4. **Test immediately:**
   ```bash
   streamlit run dashboard/app.py
   ```

5. **Visual verification:**
   - Check layout renders correctly
   - Verify colors/styling matches expectations
   - Test responsive behavior (resize browser)

### Step 4: Final Verification

After all replacements:

```bash
# Should return 0 instances
grep -c "unsafe_allow_html=True" dashboard/app.py
grep -c "unsafe_allow_html=True" dashboard/game_predictions_patch.py

# Should return 0 (no HTML blocks)
grep -c "<div style=" dashboard/app.py
grep -c "<table style=" dashboard/game_predictions_patch.py
```

### Step 5: Create Git Commit

```bash
git add dashboard/app.py dashboard/game_predictions_patch.py dashboard/components/
git commit -m "refactor(dashboard): replace HTML blocks with native Streamlit components

- Remove all unsafe_allow_html usage (7 instances)
- Create reusable component library (dashboard/components/cards.py)
- Replace service health cards with service_health_card()
- Replace all game cards with game_prediction_card()
- Improve responsive design and accessibility
- Simplify maintenance (centralized styling)"
```

---

## Troubleshooting

### Issue: Component not rendering

**Symptom:** Blank space where card should appear
**Fix:** Check for missing required data in `game` dict
```python
# Add defensive checks:
if not game.get('away_team') or not game.get('home_team'):
    st.warning("Missing team data")
    return
```

### Issue: Wrong game state

**Symptom:** Live game showing as scheduled
**Fix:** Verify `card_type` parameter matches actual game status
```python
# Determine card_type from game data:
if game.get('game_status') == 'completed':
    card_type = "completed"
elif game.get('game_status') == 'in_progress':
    card_type = "live"
else:
    card_type = "scheduled"
```

### Issue: Missing predictions

**Symptom:** "No prediction available" shown
**Fix:** Ensure prediction dict is populated before calling component
```python
# Check prediction validity:
if not pred or not pred.get('predicted_spread'):
    pred = None  # Component will handle gracefully
```

---

## Expected Outcomes

### Before
- 7 instances of `unsafe_allow_html=True`
- 250+ lines of HTML/CSS code
- Hardcoded styling
- Poor responsive design
- Difficult to maintain

### After
- 0 instances of `unsafe_allow_html=True`
- ~50 lines of component calls
- Centralized styling in `cards.py`
- Automatic responsive design (Streamlit native)
- Easy to maintain (change once, updates everywhere)

### Metrics
- **Code Reduction:** 80% fewer lines (250 → 50)
- **Maintainability:** 100% improvement (single source of truth)
- **Accessibility:** Native screen reader support
- **Performance:** Faster rendering (no HTML parsing)
- **Security:** Eliminated XSS risk (no `unsafe_allow_html`)

---

## Completion Checklist

- [ ] Instance #3 replaced (live game card)
- [ ] Instance #4 replaced (completed game card)
- [ ] Instance #5 replaced (upcoming game card)
- [ ] Instance #6 replaced (player/team card)
- [ ] Instance #7 replaced (game_predictions_patch.py)
- [ ] All visual tests passed
- [ ] Responsive design verified
- [ ] No console errors
- [ ] Git commit created
- [ ] Documentation updated

---

## Next Steps After Completion

1. **Performance Testing:**
   - Measure page load time before/after
   - Check memory usage (should be lower)

2. **User Feedback:**
   - Deploy to staging
   - Collect feedback on new styling
   - Iterate on component design if needed

3. **Additional Components:**
   - Create `model_status_card()`
   - Create `prediction_chart()`
   - Migrate other pages to native components

4. **Documentation:**
   - Update dashboard README
   - Add component library usage guide
   - Create style guide for consistent design
