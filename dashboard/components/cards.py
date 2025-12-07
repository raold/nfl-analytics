#!/usr/bin/env python3
"""
Streamlit Native Component Library

Replaces raw HTML/CSS with Streamlit native components for:
- Better maintainability
- Improved responsive design
- Consistent styling
- No unsafe_allow_html usage

Author: Claude Code
Date: 2025-11-22
"""


import streamlit as st


def service_health_card(name: str, port: int, is_healthy: bool):
    """
    Render service health status card using Streamlit native components.

    Args:
        name: Service name
        port: Port number
        is_healthy: Health status (True = healthy, False = unhealthy)
    """
    with st.container():
        if is_healthy:
            st.success(f"✅ **{name}**")
        else:
            st.error(f"❌ **{name}**")
        st.caption(f"Port: {port}")


def game_prediction_card(
    game: dict,
    prediction: dict | None = None,
    spread_edge: float | None = None,
    total_edge: float | None = None,
    card_type: str = "scheduled"
):
    """
    Render game prediction card with spread/total edges.

    Supports three game states:
    - 'scheduled': Upcoming games with predictions and betting lines
    - 'live': In-progress games with current scores
    - 'completed': Finished games with final scores and results

    Args:
        game: Game data dict (from database)
            Required keys: away_team, home_team, spread_close, total_close
            Optional keys: away_score, home_score, away_team_display, home_team_display
        prediction: Prediction data dict (from API) - optional for completed games
        spread_edge: Calculated spread edge - optional
        total_edge: Calculated total edge - optional
        card_type: 'scheduled', 'live', 'completed', or 'international'
    """
    # Determine card styling based on edge strength (if edges provided)
    if spread_edge is not None and total_edge is not None:
        if abs(spread_edge) >= 2.0 or abs(total_edge) >= 3.0:
            edge_label = "🟢 STRONG EDGE"
        elif abs(spread_edge) >= 1.0 or abs(total_edge) >= 2.0:
            edge_label = "🟡 MODERATE EDGE"
        else:
            edge_label = "⚪ WEAK EDGE"
    else:
        edge_label = None

    with st.container():
        # Game status banner
        if card_type == "live":
            st.error("🔴 LIVE GAME")
        elif card_type == "international":
            st.info("🌍 INTERNATIONAL GAME")
        elif card_type == "completed":
            st.success("✅ FINAL")

        # Team matchup header
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Away team
            away_team_display = game.get('away_team_display', game.get('away_team'))
            st.markdown(f"### {away_team_display}")
            if card_type in ["live", "completed"]:
                away_score = int(game.get('away_score', 0)) if game.get('away_score') is not None else 0
                st.metric("Score", away_score, delta=None)

        with col2:
            st.markdown("### @")

        with col3:
            # Home team
            home_team_display = game.get('home_team_display', game.get('home_team'))
            st.markdown(f"### {home_team_display}")
            if card_type in ["live", "completed"]:
                home_score = int(game.get('home_score', 0)) if game.get('home_score') is not None else 0
                st.metric("Score", home_score, delta=None)

        # Edge indicator (if available)
        if edge_label:
            st.markdown(f"**{edge_label}**")

        # Betting lines comparison table (if prediction available)
        if prediction:
            st.markdown("#### Betting Lines" if card_type == "scheduled" else "#### Prediction vs Result")

            # Create data for table
            betting_data = {
                "Bet Type": ["Spread", "Total", "Moneyline"],
                "Sportsbook": [
                    f"{game.get('spread_close', 0):+.1f}",
                    f"{game.get('total_close', 0):.1f}",
                    f"{game.get('home_moneyline', 0):+.0f}" if game.get('home_moneyline') else "—"
                ],
                "Our Model": [
                    f"{prediction.get('predicted_spread', 0):+.1f}",
                    f"{prediction.get('predicted_total', 0):.1f}",
                    f"Win Prob: {prediction.get('predicted_home_win_prob', 0.5):.1%}"
                ],
                "Edge": [
                    f"{spread_edge:+.1f} pts" if spread_edge is not None else "—",
                    f"{total_edge:+.1f} pts" if total_edge is not None else "—",
                    "—"
                ]
            }

            # Display as dataframe with appropriate styling
            import pandas as pd
            st.dataframe(
                pd.DataFrame(betting_data),
                hide_index=True,
                use_container_width=True
            )

            # Prediction details (expandable)
            with st.expander("📊 Prediction Details"):
                # Predicted scores
                pred_away_score = prediction.get('predicted_away_score', 0)
                pred_home_score = prediction.get('predicted_home_score', 0)
                st.markdown(f"**Predicted Score:** {game['away_team']} {pred_away_score:.1f} @ {game['home_team']} {pred_home_score:.1f}")

                # QB passing yards (if available)
                if prediction.get('away_qb_yards') is not None:
                    away_qb = prediction.get('away_qb_yards', 0)
                    away_std = prediction.get('away_qb_yards_std', 0)
                    home_qb = prediction.get('home_qb_yards', 0)
                    home_std = prediction.get('home_qb_yards_std', 0)
                    st.markdown(f"""
**QB Passing Yards:**
- {game['away_team']}: {away_qb:.0f} ± {away_std:.0f} yds
- {game['home_team']}: {home_qb:.0f} ± {home_std:.0f} yds
                    """)

                # Actual vs predicted (for completed games)
                if card_type == "completed" and game.get('away_score') is not None:
                    actual_spread = game['home_score'] - game['away_score']
                    actual_total = game['home_score'] + game['away_score']
                    pred_spread = prediction.get('predicted_spread', 0)
                    pred_total = prediction.get('predicted_total', 0)

                    spread_error = abs(actual_spread - pred_spread)
                    total_error = abs(actual_total - pred_total)

                    st.markdown(f"""
**Prediction Accuracy:**
- Spread Error: {spread_error:.1f} pts
- Total Error: {total_error:.1f} pts
                    """)


def section_banner(title: str, icon: str = "", banner_type: str = "info"):
    """
    Render section banner/header.

    Args:
        title: Banner title text
        icon: Optional emoji icon
        banner_type: 'info', 'success', 'warning', 'error'
    """
    full_title = f"{icon} {title}" if icon else title

    if banner_type == "success":
        st.success(full_title)
    elif banner_type == "warning":
        st.warning(full_title)
    elif banner_type == "error":
        st.error(full_title)
    else:
        st.info(full_title)


def live_games_banner():
    """Render LIVE GAMES banner."""
    st.error("# 🔴 LIVE GAMES")


def international_games_banner(num_games: int):
    """Render INTERNATIONAL GAMES banner."""
    game_text = "game" if num_games == 1 else "games"
    st.info(f"### 🌍 INTERNATIONAL GAMES - LIVE NOW ({num_games} {game_text})")


def scheduled_games_banner(num_games: int, day_label: str = ""):
    """Render scheduled games banner."""
    game_text = "game" if num_games == 1 else "games"
    if day_label:
        st.success(f"### {day_label} ({num_games} {game_text})")
    else:
        st.success(f"### Scheduled Games ({num_games} {game_text})")


def metric_card(label: str, value: str, delta: str | None = None, help_text: str | None = None):
    """
    Render metric card using Streamlit native components.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change indicator
        help_text: Optional help tooltip text
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


def edge_summary_stats(games_with_predictions: list):
    """
    Render edge summary statistics.

    Args:
        games_with_predictions: List of dicts with 'spread_edge' and 'total_edge' keys
    """
    if not games_with_predictions:
        st.info("No games with predictions available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_spread_edge = sum(abs(g['spread_edge']) for g in games_with_predictions) / len(games_with_predictions)
        st.metric("Avg Spread Edge", f"{avg_spread_edge:.2f} pts")

    with col2:
        avg_total_edge = sum(abs(g['total_edge']) for g in games_with_predictions) / len(games_with_predictions)
        st.metric("Avg Total Edge", f"{avg_total_edge:.2f} pts")

    with col3:
        strong_edge_count = sum(
            1 for g in games_with_predictions
            if abs(g['spread_edge']) >= 2.0 or abs(g['total_edge']) >= 3.0
        )
        st.metric("Strong Edge Games", f"{strong_edge_count}")


def database_stats_row(db_stats: dict):
    """
    Render database statistics row.

    Args:
        db_stats: Dict with database statistics
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Games", f"{db_stats.get('total_games', 0):,}")

    with col2:
        st.metric("Total Plays", f"{db_stats.get('total_plays', 0):,}")

    with col3:
        season_min = db_stats.get('season_min', 'N/A')
        season_max = db_stats.get('season_max', 'N/A')
        st.metric("Season Coverage", f"{season_min}-{season_max}")

    with col4:
        st.metric("Database Size", db_stats.get('db_size', 'N/A'))

    with col5:
        st.metric("Cache Hit Rate", db_stats.get('cache_hit_rate', 'N/A'))
