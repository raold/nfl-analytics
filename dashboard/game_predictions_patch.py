#!/usr/bin/env python3
"""
Game Predictions Comparison Page - Dashboard Enhancement

This file contains the complete implementation for displaying game-level predictions
vs sportsbook lines. To integrate into dashboard/app.py:

1. Copy render_game_predictions_comparison() function
2. Copy helper functions (fetch_game_predictions_db, etc.)
3. Update main() function to add new page navigation
4. Add imports at top of app.py

Author: Claude Code
Date: 2025-11-21
"""

import streamlit as st
import pandas as pd
import psycopg
from datetime import datetime
from typing import Optional, Dict, List
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import connection pooling
try:
    from dashboard.utils.db import get_db_connection
except ImportError:
    def get_db_connection():
        import os
        DB_CONFIG = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "dbname": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions")
        }
        return psycopg.connect(**DB_CONFIG)

# Import API client
from utils.api_client import APIClient

# Import schema validation
try:
    from dashboard.utils.schema_validator import validate_schema, get_common_schemas, ValidationError
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    import logging
    logging.warning("Schema validation not available - operating without validation")

# Import dashboard constants
from constants import (
    STRONG_SPREAD_EDGE_THRESHOLD,
    MODERATE_SPREAD_EDGE_THRESHOLD,
    STRONG_TOTAL_EDGE_THRESHOLD,
    MODERATE_TOTAL_EDGE_THRESHOLD,
    SPREAD_TO_PROB_LOGISTIC_FACTOR,
    QB_YARDS_TO_SCORE_INTERCEPT,
    QB_YARDS_TO_SCORE_SLOPE,
    API_TIMEOUT_LONG,
    COLOR_SUCCESS_BG,
    COLOR_WARNING_BG,
    COLOR_NEUTRAL_BG,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_NEUTRAL,
)


# ============================================================================
# Helper Functions
# ============================================================================

def fetch_game_predictions_db(season: int, week: int) -> pd.DataFrame:
    """
    Fetch game predictions from ml.game_predictions table with error handling.

    NOTE: As of Nov 21, 2025, this table doesn't exist yet. This function
    is a placeholder for future implementation.

    Args:
        season: NFL season year
        week: NFL week number

    Returns:
        DataFrame with stored predictions, or empty DataFrame if table doesn't exist
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Database config from environment
        DB_CONFIG = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "dbname": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions")
        }

        query = """
        SELECT
            gp.game_id,
            gp.predicted_home_score,
            gp.predicted_away_score,
            gp.predicted_spread,
            gp.predicted_total,
            gp.predicted_home_win_prob,
            gp.spread_edge,
            gp.total_edge,
            gp.predicted_at
        FROM ml.game_predictions gp
        WHERE SUBSTRING(gp.game_id, 1, 4)::int = %s
          AND SUBSTRING(gp.game_id, 6, 2)::int = %s
        ORDER BY gp.predicted_at DESC
        """

        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season, week))

        logger.info(f"Fetched {len(df)} game predictions for {season} Week {week}")
        return df

    except psycopg.Error as e:
        # Table doesn't exist yet or query error - return empty DataFrame
        logger.debug(f"Game predictions table query failed (expected if table doesn't exist): {e}")
        return pd.DataFrame()

    except Exception as e:
        # Unexpected error - log and return empty
        logger.warning(f"Unexpected error in fetch_game_predictions_db: {e}")
        return pd.DataFrame()


def call_game_prediction_api(game_id: str, season: int, week: int) -> Optional[Dict[str, Any]]:
    """
    Call FastAPI /predict/game endpoint to generate predictions on-the-fly.

    Uses validated APIClient with automatic response validation to prevent
    KeyError crashes from missing API fields.

    Args:
        game_id: Game ID (e.g., '2024_12_BAL_LAC')
        season: NFL season year
        week: NFL week number

    Returns:
        dict with prediction data or None if failed
        {
            'game_id': str,
            'home_team': str,
            'away_team': str,
            'home_qb_yards': float,
            'home_qb_yards_std': float,
            'away_qb_yards': float,
            'away_qb_yards_std': float,
            'total_yards': float,
            'predicted_home_score': float,  # estimated from QB yards
            'predicted_away_score': float,  # estimated from QB yards
            'predicted_spread': float,      # home - away
            'predicted_total': float,       # home + away
            'predicted_home_win_prob': float
        }
    """
    import math

    try:
        # Use validated APIClient (prevents KeyError crashes)
        api = APIClient(timeout=API_TIMEOUT_LONG)
        prediction = api.get_game_prediction(game_id, season, week)

        if prediction is None:
            return None

        # Extract QB passing yards (validated, no KeyError risk)
        home_yards = prediction['home_qb_yards']
        home_std = prediction['home_qb_yards_std']
        away_yards = prediction['away_qb_yards']
        away_std = prediction['away_qb_yards_std']
        total_yards = prediction.get('total_yards', 0.0)

        # Estimate scores from QB yards (simple heuristic)
        # Historical correlation: team score ≈ 10 + (QB_yards / 20)
        # This is a simplified model - ideally you'd train a proper score prediction model
        predicted_home_score = 10 + (home_yards / 20) if home_yards else None
        predicted_away_score = 10 + (away_yards / 20) if away_yards else None

        # Calculate spread and total
        predicted_spread = None
        predicted_total = None
        predicted_home_win_prob = None

        if predicted_home_score and predicted_away_score:
            predicted_spread = predicted_home_score - predicted_away_score
            predicted_total = predicted_home_score + predicted_away_score

            # Simple win probability from spread (logistic function)
            # P(home win) = 1 / (1 + exp(-spread / 7))
            predicted_home_win_prob = 1 / (1 + math.exp(-predicted_spread / 7))

        return {
            'game_id': prediction['game_id'],
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'home_qb_yards': home_yards,
            'home_qb_yards_std': home_std,
            'away_qb_yards': away_yards,
            'away_qb_yards_std': away_std,
            'total_yards': total_yards,
            'predicted_home_score': predicted_home_score,
            'predicted_away_score': predicted_away_score,
            'predicted_spread': predicted_spread,
            'predicted_total': predicted_total,
            'predicted_home_win_prob': predicted_home_win_prob,
            'inference_time_ms': prediction.get('processing_ms', 0)
        }

    except Exception as e:
        st.warning(f"Failed to fetch prediction from API: {e}")
        return None


def calculate_edge(predicted: float, sportsbook: float) -> float:
    """
    Calculate edge: difference between our prediction and sportsbook line.

    Args:
        predicted: Our model's prediction
        sportsbook: Sportsbook line

    Returns:
        Edge value (positive means we predict higher than book)
    """
    if predicted is None or sportsbook is None:
        return 0.0
    return predicted - sportsbook


def get_edge_color(edge: float, bet_type: str) -> str:
    """
    Get background color based on edge magnitude.

    Args:
        edge: Edge value
        bet_type: 'spread' or 'total'

    Returns:
        CSS background color string
    """
    abs_edge = abs(edge)

    # Thresholds for significance
    if bet_type == 'spread':
        strong_threshold = STRONG_SPREAD_EDGE_THRESHOLD
        moderate_threshold = MODERATE_SPREAD_EDGE_THRESHOLD
    else:  # total
        strong_threshold = STRONG_TOTAL_EDGE_THRESHOLD
        moderate_threshold = MODERATE_TOTAL_EDGE_THRESHOLD

    if abs_edge >= strong_threshold:
        return "background-color: #d4edda; border: 2px solid #28a745;"  # Green
    elif abs_edge >= moderate_threshold:
        return "background-color: #fff3cd; border: 2px solid #ffc107;"  # Yellow
    else:
        return "background-color: #f8f9fa; border: 2px solid #dee2e6;"  # Gray


def get_team_logo_url(team_abbr: str) -> str:
    """Get team logo URL from ESPN."""
    # Normalize LAR for logo lookup
    if team_abbr == 'LA':
        team_abbr = 'LAR'
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{team_abbr}.png"


# ============================================================================
# Main Rendering Function
# ============================================================================

def render_game_predictions_comparison() -> None:
    """
    Render game-level predictions vs sportsbook lines comparison page.

    Features:
    - Fetches current week games from database
    - Generates predictions via API if not stored in DB
    - Displays sportsbook lines vs model predictions side-by-side
    - Color-codes by edge magnitude
    - Supports filtering by edge size and bet type
    """
    st.header("🎯 Game Predictions vs Sportsbook")

    # Get current NFL week from ESPN API (reuse existing function)
    try:
        from app import get_current_nfl_week
        current_season, current_week = get_current_nfl_week()
    except:
        # Fallback if import fails
        current_season = 2025
        current_week = 12

    st.markdown(f"**Season {current_season}, Week {current_week}**")

    # Filtering controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        min_spread_edge = st.slider(
            "Min Spread Edge (pts)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            help="Only show games with at least this much spread edge"
        )

    with col2:
        min_total_edge = st.slider(
            "Min Total Edge (pts)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
            help="Only show games with at least this much total edge"
        )

    with col3:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch games from database
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5544")),
        "dbname": os.getenv("DB_NAME", "devdb01"),
        "user": os.getenv("DB_USER", "dro"),
        "password": os.getenv("DB_PASSWORD", "sicillionbillions")
    }

    try:
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            g.home_moneyline,
            g.away_moneyline,
            g.kickoff,
            CASE
                WHEN g.home_score IS NOT NULL THEN 'completed'
                WHEN g.kickoff < NOW() THEN 'in_progress'
                ELSE 'scheduled'
            END as game_status
        FROM games g
        WHERE g.season = %s AND g.week = %s AND g.game_type = 'REG'
        ORDER BY g.kickoff ASC
        """

        with get_db_connection() as conn:
            df_games = pd.read_sql_query(query, conn, params=(current_season, current_week))

        # Schema validation
        if SCHEMA_VALIDATION_AVAILABLE and not df_games.empty:
            schemas = get_common_schemas()
            df_games = validate_schema(df_games, schemas['games'], strict=False)

        if df_games.empty:
            st.info(f"No games found for Week {current_week} of {current_season} season.")
            return

        st.markdown(f"**Found {len(df_games)} games**")

        # Process each game
        games_with_predictions = []

        with st.spinner("Generating predictions..."):
            for idx, game in df_games.iterrows():
                game_id = game['game_id']

                # Try to fetch from database first
                df_pred = fetch_game_predictions_db(current_season, current_week)
                db_pred = df_pred[df_pred['game_id'] == game_id] if not df_pred.empty else pd.DataFrame()

                if not db_pred.empty:
                    # Use stored prediction
                    pred = db_pred.iloc[0].to_dict()
                else:
                    # Call API to generate prediction
                    pred = call_game_prediction_api(game_id, current_season, current_week)

                if pred:
                    # Calculate edges
                    spread_edge = calculate_edge(pred.get('predicted_spread'), game['spread_close'])
                    total_edge = calculate_edge(pred.get('predicted_total'), game['total_close'])

                    # Apply filters
                    if abs(spread_edge) >= min_spread_edge or abs(total_edge) >= min_total_edge:
                        games_with_predictions.append({
                            'game': game,
                            'prediction': pred,
                            'spread_edge': spread_edge,
                            'total_edge': total_edge
                        })

        if not games_with_predictions:
            st.warning("No games match the selected edge filters.")
            return

        # Sort by total edge magnitude (descending)
        games_with_predictions.sort(key=lambda x: abs(x['total_edge']), reverse=True)

        # Display game cards
        for game_data in games_with_predictions:
            game = game_data['game']
            pred = game_data['prediction']
            spread_edge = game_data['spread_edge']
            total_edge = game_data['total_edge']

            # Determine card styling based on edge
            spread_color = get_edge_color(spread_edge, 'spread')
            total_color = get_edge_color(total_edge, 'total')

            # Use strongest edge for overall card styling
            if abs(spread_edge) >= 2.0 or abs(total_edge) >= 3.0:
                card_style = "background-color: #d4edda; border-left: 5px solid #28a745;"
            elif abs(spread_edge) >= 1.0 or abs(total_edge) >= 2.0:
                card_style = "background-color: #fff3cd; border-left: 5px solid #ffc107;"
            else:
                card_style = "background-color: #ffffff; border-left: 5px solid #dee2e6;"

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
                                <td style="padding: 10px; text-align: center; font-weight: bold; color: {COLOR_SUCCESS if abs(spread_edge) >= 2.0 else COLOR_WARNING if abs(spread_edge) >= 1.0 else COLOR_NEUTRAL};">{spread_edge:+.1f}</td>
                            </tr>
                            <tr style="{total_color}">
                                <td style="padding: 10px; font-weight: bold;">Total (O/U)</td>
                                <td style="padding: 10px; text-align: center;">{game['total_close']:.1f}</td>
                                <td style="padding: 10px; text-align: center;">{pred.get('predicted_total', 0):.1f}</td>
                                <td style="padding: 10px; text-align: center; font-weight: bold; color: {COLOR_SUCCESS if abs(total_edge) >= 3.0 else COLOR_WARNING if abs(total_edge) >= 2.0 else COLOR_NEUTRAL};">{total_edge:+.1f}</td>
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

        # Summary stats
        st.markdown("---")
        st.subheader("📈 Edge Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_spread_edge = sum(abs(g['spread_edge']) for g in games_with_predictions) / len(games_with_predictions)
            st.metric("Avg Spread Edge", f"{avg_spread_edge:.2f} pts")

        with col2:
            avg_total_edge = sum(abs(g['total_edge']) for g in games_with_predictions) / len(games_with_predictions)
            st.metric("Avg Total Edge", f"{avg_total_edge:.2f} pts")

        with col3:
            strong_edge_count = sum(1 for g in games_with_predictions if abs(g['spread_edge']) >= 2.0 or abs(g['total_edge']) >= 3.0)
            st.metric("Strong Edge Games", f"{strong_edge_count}")

        # Methodology note
        with st.expander("ℹ️ How Predictions Are Generated"):
            st.markdown("""
            **Current Method:**
            - Uses QB passing yards predictions from Ensemble v6.1 model
            - Estimates team scores using historical correlation: Score ≈ 10 + (QB_yards / 20)
            - Calculates spread (home - away) and total (home + away) from score estimates
            - Win probability derived from predicted spread via logistic function

            **Edge Interpretation:**
            - **Spread Edge:** Difference between our predicted spread and sportsbook line
            - **Total Edge:** Difference between our predicted total and sportsbook over/under
            - **Green (Strong):** Spread ≥2pts OR Total ≥3pts
            - **Yellow (Moderate):** Spread ≥1pt OR Total ≥2pts
            - **Gray (Weak):** Below moderate thresholds

            **⚠️ Important Note:**
            This is a simplified scoring model. For production betting, consider:
            1. Training dedicated game outcome models (not QB yards → score regression)
            2. Incorporating team defense, injuries, weather, rest days
            3. Tracking historical accuracy and calibration
            4. Implementing proper bankroll management and risk controls
            """)

    except psycopg.Error as e:
        st.error(f"Database error loading games: {e}")
        import logging
        logging.error(f"Database error in render_game_predictions_comparison: {e}", exc_info=True)

    except ValidationError as e:
        st.warning(f"Schema validation failed for games: {e}")
        import logging
        logging.warning(f"Schema mismatch in render_game_predictions_comparison: {e}")

    except Exception as e:
        st.error(f"Unexpected error loading game predictions: {str(e)}")
        import traceback
        import logging
        logging.error(f"Unexpected error in render_game_predictions_comparison: {e}", exc_info=True)
        st.code(traceback.format_exc())


# ============================================================================
# Integration Instructions
# ============================================================================

"""
TO INTEGRATE INTO dashboard/app.py:

1. Add imports at top of file:
   ```python
   import math
   ```

2. Copy these functions to app.py:
   - fetch_game_predictions_db()
   - call_game_prediction_api()
   - calculate_edge()
   - get_edge_color()
   - render_game_predictions_comparison()

3. Update main() function navigation:
   ```python
   page = st.sidebar.radio(
       "Select Page",
       ["Predictions", "🎯 Game Predictions", "System Status", "Model Status", "Data Summary"],
       index=0
   )

   # Route to appropriate page
   if page == "🎯 Game Predictions":
       render_game_predictions_comparison()
       return
   elif page == "System Status":
       render_system_status_page()
       return
   # ... rest of navigation
   ```

4. Test the integration:
   ```bash
   # Start FastAPI (if not already running)
   uvicorn py.api.main:app --host 0.0.0.0 --port 8000 --reload

   # Start dashboard
   streamlit run dashboard/app.py

   # Navigate to "🎯 Game Predictions" page
   ```

5. Expected behavior:
   - Page loads current NFL week games from database
   - Calls FastAPI /predict/game endpoint for each game
   - Displays sportsbook vs model predictions in cards
   - Color-codes by edge magnitude
   - Supports filtering by edge size
"""
