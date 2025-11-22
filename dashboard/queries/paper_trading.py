"""
Paper Trading Queries

SQL queries for paper trading session and trade history tables.

Author: Claude Code
Date: November 2025
"""

from typing import Optional
import pandas as pd


class PaperTradingQueries:
    """Centralized queries for paper trading tables."""

    @staticmethod
    def active_sessions(
        conn,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch active and recent paper trading sessions.

        Args:
            conn: Database connection
            days_back: Include sessions ended within this many days (default: 7)

        Returns:
            DataFrame with columns:
                - session_id, started_at, ended_at
                - initial_bankroll, final_bankroll, total_bets
                - win_rate, roi, sharpe_ratio, max_drawdown
                - strategy
        """
        query = f"""
        SELECT
            session_id,
            started_at,
            ended_at,
            initial_bankroll,
            final_bankroll,
            total_bets,
            win_rate,
            roi,
            sharpe_ratio,
            max_drawdown,
            strategy
        FROM paper_trading_sessions
        WHERE ended_at IS NULL OR ended_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY started_at DESC
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def session_trades(
        conn,
        session_id: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch all trades for a specific paper trading session.

        Args:
            conn: Database connection
            session_id: Session ID
            limit: Maximum trades to return (default: 1000)

        Returns:
            DataFrame with columns:
                - paper_trade_id, trade_timestamp, player_name, prop_type
                - line_value, bet_side, odds, bet_amount, edge
                - model_prediction, model_std, actual_result
                - outcome ('win', 'loss', 'push', 'pending'), profit_loss
                - bookmaker_key
        """
        query = f"""
        SELECT
            paper_trade_id,
            trade_timestamp,
            player_name,
            prop_type,
            line_value,
            bet_side,
            odds,
            bet_amount,
            edge,
            model_prediction,
            model_std,
            actual_result,
            outcome,
            profit_loss,
            bookmaker_key
        FROM paper_trades
        WHERE session_id = %s
        ORDER BY trade_timestamp DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn, params=(session_id,))

    @staticmethod
    def pending_trades(
        conn,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Fetch all pending trades awaiting settlement.

        Args:
            conn: Database connection
            limit: Maximum trades to return (default: 20)

        Returns:
            DataFrame with columns:
                - paper_trade_id, trade_timestamp, player_name, prop_type
                - line_value, bet_side, bet_amount, bookmaker_key
                - session_id
        """
        query = f"""
        SELECT
            paper_trade_id,
            trade_timestamp,
            player_name,
            prop_type,
            line_value,
            bet_side,
            bet_amount,
            bookmaker_key,
            session_id
        FROM paper_trades
        WHERE outcome = 'pending'
        ORDER BY trade_timestamp DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def performance_by_prop_type(
        conn,
        session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch performance breakdown by prop type.

        Args:
            conn: Database connection
            session_id: Optional session filter (all sessions if None)

        Returns:
            DataFrame with columns:
                - prop_type, total_trades, win_rate
                - roi, total_wagered, total_pnl
                - avg_edge
        """
        query = """
        SELECT
            prop_type,
            COUNT(*) as total_trades,
            ROUND(AVG(CASE WHEN outcome IN ('win', 'push') THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
            ROUND(SUM(profit_loss) / SUM(bet_amount)::numeric, 4) as roi,
            SUM(bet_amount) as total_wagered,
            SUM(profit_loss) as total_pnl,
            ROUND(AVG(edge)::numeric, 2) as avg_edge
        FROM paper_trades
        WHERE outcome IN ('win', 'loss', 'push')
        """

        params = []
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)

        query += " GROUP BY prop_type ORDER BY total_trades DESC"

        return pd.read_sql_query(query, conn, params=params if params else None)

    @staticmethod
    def performance_by_bookmaker(
        conn,
        session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch performance breakdown by bookmaker.

        Args:
            conn: Database connection
            session_id: Optional session filter (all sessions if None)

        Returns:
            DataFrame with columns:
                - bookmaker_key, total_trades, win_rate
                - roi, total_wagered, total_pnl
        """
        query = """
        SELECT
            bookmaker_key,
            COUNT(*) as total_trades,
            ROUND(AVG(CASE WHEN outcome IN ('win', 'push') THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
            ROUND(SUM(profit_loss) / SUM(bet_amount)::numeric, 4) as roi,
            SUM(bet_amount) as total_wagered,
            SUM(profit_loss) as total_pnl
        FROM paper_trades
        WHERE outcome IN ('win', 'loss', 'push')
        """

        params = []
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)

        query += " GROUP BY bookmaker_key ORDER BY total_wagered DESC"

        return pd.read_sql_query(query, conn, params=params if params else None)

    @staticmethod
    def session_summary(
        conn,
        session_id: str
    ) -> pd.DataFrame:
        """
        Fetch summary statistics for a specific session.

        Returns:
            Single-row DataFrame with aggregated metrics
        """
        query = """
        SELECT
            session_id,
            started_at,
            ended_at,
            initial_bankroll,
            final_bankroll,
            final_bankroll - initial_bankroll as total_profit_loss,
            (final_bankroll - initial_bankroll) / initial_bankroll as roi,
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
            SUM(CASE WHEN outcome = 'pending' THEN 1 ELSE 0 END) as pending,
            ROUND(AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
            ROUND(AVG(edge)::numeric, 2) as avg_edge,
            MAX(CASE
                WHEN outcome = 'win' THEN bet_amount * (odds / 100.0)
                WHEN outcome = 'loss' THEN -bet_amount
                ELSE 0
            END) as largest_win,
            MIN(CASE
                WHEN outcome = 'loss' THEN -bet_amount
                ELSE 0
            END) as largest_loss
        FROM paper_trades
        WHERE session_id = %s
        GROUP BY session_id, started_at, ended_at, initial_bankroll, final_bankroll
        """

        return pd.read_sql_query(query, conn, params=(session_id,))

    @staticmethod
    def bankroll_progression(
        conn,
        session_id: str
    ) -> pd.DataFrame:
        """
        Fetch bankroll progression over time for a session.

        Returns:
            DataFrame with columns:
                - trade_timestamp, bet_amount, outcome, profit_loss
                - cumulative_pnl, bankroll
        """
        query = """
        WITH trades_ordered AS (
            SELECT
                trade_timestamp,
                bet_amount,
                outcome,
                profit_loss,
                ROW_NUMBER() OVER (ORDER BY trade_timestamp) as trade_num
            FROM paper_trades
            WHERE session_id = %s
              AND outcome IN ('win', 'loss', 'push')
            ORDER BY trade_timestamp ASC
        ),
        trades_with_cumsum AS (
            SELECT
                trade_timestamp,
                bet_amount,
                outcome,
                profit_loss,
                SUM(profit_loss) OVER (ORDER BY trade_num ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_pnl,
                (SELECT initial_bankroll FROM paper_trading_sessions WHERE session_id = %s) +
                SUM(profit_loss) OVER (ORDER BY trade_num ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as bankroll
            FROM trades_ordered
        )
        SELECT * FROM trades_with_cumsum
        """

        return pd.read_sql_query(query, conn, params=(session_id, session_id))

    @staticmethod
    def edge_analysis(
        conn,
        session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze performance by edge size categories.

        Args:
            conn: Database connection
            session_id: Optional session filter

        Returns:
            DataFrame with columns:
                - edge_category, total_trades, win_rate, avg_profit_loss, roi
        """
        query = """
        WITH edge_categorized AS (
            SELECT
                CASE
                    WHEN edge < -20 THEN 'Negative Edge (<-20)'
                    WHEN edge < -10 THEN 'Weak Under (-20 to -10)'
                    WHEN edge < 0 THEN 'Slight Under (-10 to 0)'
                    WHEN edge < 10 THEN 'Slight Over (0 to 10)'
                    WHEN edge < 20 THEN 'Moderate Over (10 to 20)'
                    ELSE 'Strong Over (>20)'
                END as edge_category,
                outcome,
                profit_loss,
                bet_amount
            FROM paper_trades
            WHERE outcome IN ('win', 'loss', 'push')
        """

        params = []
        if session_id:
            query = query.replace(
                "FROM paper_trades",
                f"FROM paper_trades WHERE session_id = %s"
            )
            params.append(session_id)

        query += """
        )
        SELECT
            edge_category,
            COUNT(*) as total_trades,
            ROUND(AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
            ROUND(AVG(profit_loss)::numeric, 2) as avg_profit_loss,
            ROUND(SUM(profit_loss) / SUM(bet_amount)::numeric, 4) as roi
        FROM edge_categorized
        GROUP BY edge_category
        ORDER BY edge_category
        """

        return pd.read_sql_query(query, conn, params=params if params else None)

    @staticmethod
    def session_comparison(
        conn,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Compare performance across all sessions.

        Args:
            conn: Database connection
            limit: Maximum sessions to return

        Returns:
            DataFrame with columns:
                - session_id, strategy, started_at, ended_at
                - initial_bankroll, final_bankroll, roi, win_rate
                - total_trades, sharpe_ratio, max_drawdown
        """
        query = f"""
        SELECT
            session_id,
            strategy,
            started_at,
            ended_at,
            initial_bankroll,
            final_bankroll,
            ROUND((final_bankroll - initial_bankroll) / initial_bankroll::numeric, 4) as roi,
            ROUND(win_rate::numeric, 3) as win_rate,
            total_bets as total_trades,
            ROUND(sharpe_ratio::numeric, 2) as sharpe_ratio,
            ROUND(max_drawdown::numeric, 4) as max_drawdown
        FROM paper_trading_sessions
        ORDER BY started_at DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn)
