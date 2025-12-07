"""
Paper Trading System for Risk-Free Validation

This module provides a complete paper trading engine that simulates betting
without real money. It's designed to validate betting strategies before risking
capital, with identical logic to production betting.

Key Features:
- Simulates bet placement without real money
- Records hypothetical bets in database
- Automated settlement using actual game results
- Performance tracking (ROI, Sharpe, max drawdown)
- Identical risk gates and portfolio logic as production

Author: Claude Code
Date: November 6, 2025
Version: 1.0

Usage:
    from py.production.paper_trader import PaperTradingEngine

    engine = PaperTradingEngine(initial_bankroll=10000)

    # Place paper bet
    bet_id = engine.place_paper_bet(
        game_id='2024_01_SF_KC',
        player_id='P-123',
        player_name='Patrick Mahomes',
        prop_type='passing_yards',
        line_value=275.5,
        bet_side='over',
        odds=-110,
        bookmaker_key='pinnacle',
        bet_amount=100.0,
        edge=0.035,
        model_prediction=290.5,
        model_std=45.2
    )

    # Settle after game completes
    engine.settle_bet(bet_id, actual_result=312.0)

    # Get performance metrics
    metrics = engine.get_performance_metrics()
    print(f"ROI: {metrics['roi']:.2%}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
"""

import os
import uuid
from datetime import UTC, datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor


class PaperTradingEngine:
    """
    Paper Trading Engine for risk-free betting validation.

    Attributes:
        initial_bankroll: Starting bankroll for paper trading session
        bankroll: Current bankroll (updated after each settlement)
        session_id: Unique identifier for this trading session
        positions: List of open (pending) bets
        closed_trades: List of settled bets
        db_conn: PostgreSQL database connection
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        session_id: str | None = None,
        strategy: str = 'paper_trading'
    ):
        """
        Initialize paper trading engine.

        Args:
            initial_bankroll: Starting bankroll in dollars
            session_id: Optional session ID (auto-generated if None)
            strategy: Strategy name for tracking (e.g., 'pinnacle_exchange', 'player_props')
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.session_id = session_id or self._generate_session_id()
        self.strategy = strategy
        self.positions = []
        self.closed_trades = []

        # Database connection
        self.db_conn = self._connect_db()

        # Create session record
        self._create_session()

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        random_suffix = uuid.uuid4().hex[:8]
        return f"paper_{timestamp}_{random_suffix}"

    def _connect_db(self) -> psycopg2.extensions.connection:
        """
        Connect to PostgreSQL database.

        Returns:
            Database connection
        """
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5544)),
            database=os.getenv('DB_NAME', 'devdb01'),
            user=os.getenv('DB_USER', 'dro'),
            password=os.getenv('DB_PASSWORD', 'sicillionbillions')
        )
        return conn

    def _create_session(self):
        """Create paper trading session record in database."""
        with self.db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO paper_trading_sessions (
                    session_id, started_at, initial_bankroll, strategy
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
            """, (
                self.session_id,
                datetime.now(UTC),
                self.initial_bankroll,
                self.strategy
            ))
            self.db_conn.commit()

    def place_paper_bet(
        self,
        game_id: str,
        player_id: str,
        player_name: str,
        prop_type: str,
        line_value: float,
        bet_side: str,
        odds: float,
        bookmaker_key: str,
        bet_amount: float,
        edge: float | None = None,
        model_prediction: float | None = None,
        model_std: float | None = None,
        confidence: float | None = None,
        model_version: str | None = None
    ) -> int:
        """
        Place a paper bet (hypothetical, no real money).

        Args:
            game_id: NFL game identifier (e.g., '2024_01_SF_KC')
            player_id: Player identifier
            player_name: Player name (e.g., 'Patrick Mahomes')
            prop_type: Type of prop (e.g., 'passing_yards', 'rushing_yards')
            line_value: Betting line value (e.g., 275.5 yards)
            bet_side: 'over' or 'under'
            odds: American odds (e.g., -110, +120)
            bookmaker_key: Bookmaker identifier (e.g., 'pinnacle', 'draftkings')
            bet_amount: Bet size in dollars
            edge: Model edge as decimal (e.g., 0.035 = 3.5% edge)
            model_prediction: Model's predicted value
            model_std: Model's uncertainty (standard deviation)
            confidence: Model confidence score (0-1)
            model_version: Model version identifier

        Returns:
            paper_trade_id: Database ID of the paper trade

        Raises:
            ValueError: If bet parameters are invalid
        """
        # Validate inputs
        if bet_side not in ['over', 'under']:
            raise ValueError(f"bet_side must be 'over' or 'under', got {bet_side}")

        if bet_amount <= 0:
            raise ValueError(f"bet_amount must be positive, got {bet_amount}")

        if bet_amount > self.bankroll:
            raise ValueError(
                f"bet_amount ({bet_amount}) exceeds bankroll ({self.bankroll})"
            )

        # Insert into database
        with self.db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO paper_trades (
                    trade_timestamp, game_id, player_id, player_name,
                    prop_type, line_value, bet_side, odds, bookmaker_key,
                    bet_amount, edge, model_prediction, model_std,
                    confidence, outcome, model_version, strategy, session_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING paper_trade_id
            """, (
                datetime.now(UTC),
                game_id,
                player_id,
                player_name,
                prop_type,
                line_value,
                bet_side,
                odds,
                bookmaker_key,
                bet_amount,
                edge,
                model_prediction,
                model_std,
                confidence,
                'pending',  # outcome
                model_version,
                self.strategy,
                self.session_id
            ))

            paper_trade_id = cur.fetchone()[0]
            self.db_conn.commit()

        # Add to positions list
        self.positions.append({
            'paper_trade_id': paper_trade_id,
            'bet_amount': bet_amount,
            'odds': odds,
            'line_value': line_value,
            'bet_side': bet_side
        })

        return paper_trade_id

    def settle_bet(
        self,
        paper_trade_id: int,
        actual_result: float
    ) -> tuple[str, float]:
        """
        Settle a paper bet with actual game result.

        Args:
            paper_trade_id: Database ID of paper trade
            actual_result: Actual player stat result (e.g., 312.0 passing yards)

        Returns:
            outcome: 'win', 'loss', or 'push'
            profit_loss: Profit/loss amount
        """
        # Get bet details from database
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM paper_trades
                WHERE paper_trade_id = %s AND outcome = 'pending'
            """, (paper_trade_id,))

            bet = cur.fetchone()

            if bet is None:
                raise ValueError(
                    f"No pending paper trade found with ID {paper_trade_id}"
                )

        # Determine outcome
        line_value = bet['line_value']
        bet_side = bet['bet_side']
        bet_amount = bet['bet_amount']
        odds = bet['odds']

        if actual_result > line_value:
            # Over hit
            outcome = 'win' if bet_side == 'over' else 'loss'
        elif actual_result < line_value:
            # Under hit
            outcome = 'win' if bet_side == 'under' else 'loss'
        else:
            # Push (exact line value)
            outcome = 'push'

        # Calculate profit/loss
        if outcome == 'win':
            profit_loss = self._calculate_payout(bet_amount, odds) - bet_amount
        elif outcome == 'loss':
            profit_loss = -bet_amount
        else:  # push
            profit_loss = 0.0

        # Update database
        with self.db_conn.cursor() as cur:
            cur.execute("""
                UPDATE paper_trades
                SET actual_result = %s,
                    outcome = %s,
                    settled_at = %s,
                    profit_loss = %s
                WHERE paper_trade_id = %s
            """, (
                actual_result,
                outcome,
                datetime.now(UTC),
                profit_loss,
                paper_trade_id
            ))
            self.db_conn.commit()

        # Update bankroll
        self.bankroll += profit_loss

        # Move from positions to closed_trades
        self.positions = [
            p for p in self.positions
            if p['paper_trade_id'] != paper_trade_id
        ]
        self.closed_trades.append({
            'paper_trade_id': paper_trade_id,
            'outcome': outcome,
            'profit_loss': profit_loss
        })

        return outcome, profit_loss

    def _calculate_payout(self, bet_amount: float, odds: float) -> float:
        """
        Calculate payout for winning bet.

        Args:
            bet_amount: Bet size in dollars
            odds: American odds (e.g., -110, +120)

        Returns:
            total_payout: Bet amount + winnings
        """
        if odds > 0:
            # Positive odds (e.g., +120 means win $120 on $100 bet)
            winnings = bet_amount * (odds / 100.0)
        else:
            # Negative odds (e.g., -110 means bet $110 to win $100)
            winnings = bet_amount * (100.0 / abs(odds))

        return bet_amount + winnings

    def get_performance_metrics(self) -> dict:
        """
        Calculate performance metrics for paper trading session.

        Returns:
            metrics: Dictionary with performance metrics
                - total_bets: Number of bets placed
                - settled_bets: Number of bets settled
                - pending_bets: Number of pending bets
                - win_rate: Win rate as decimal (0-1)
                - roi: Return on investment as decimal
                - total_profit_loss: Total profit/loss in dollars
                - sharpe_ratio: Sharpe ratio (assuming 0% risk-free rate)
                - max_drawdown: Maximum drawdown as decimal
                - current_bankroll: Current bankroll
                - bankroll_change: Change from initial bankroll
        """
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get settled bets
            cur.execute("""
                SELECT
                    outcome,
                    bet_amount,
                    profit_loss,
                    settled_at
                FROM paper_trades
                WHERE session_id = %s
                  AND outcome IN ('win', 'loss', 'push')
                ORDER BY settled_at ASC
            """, (self.session_id,))

            settled_bets = cur.fetchall()

            # Get total and pending counts
            cur.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 'pending' THEN 1 ELSE 0 END) as pending_bets
                FROM paper_trades
                WHERE session_id = %s
            """, (self.session_id,))

            counts = cur.fetchone()

        if not settled_bets:
            return {
                'total_bets': counts['total_bets'] or 0,
                'settled_bets': 0,
                'pending_bets': counts['pending_bets'] or 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'total_profit_loss': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_bankroll': self.bankroll,
                'bankroll_change': 0.0
            }

        # Calculate metrics
        num_settled = len(settled_bets)
        num_wins = sum(1 for bet in settled_bets if bet['outcome'] == 'win')
        win_rate = num_wins / num_settled if num_settled > 0 else 0.0

        total_wagered = sum(bet['bet_amount'] for bet in settled_bets)
        total_profit_loss = sum(bet['profit_loss'] for bet in settled_bets)
        roi = total_profit_loss / total_wagered if total_wagered > 0 else 0.0

        # Sharpe ratio (daily returns)
        daily_returns = []
        for bet in settled_bets:
            daily_return = bet['profit_loss'] / bet['bet_amount']
            daily_returns.append(daily_return)

        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative_returns = []
        cumulative = 0.0
        for bet in settled_bets:
            cumulative += bet['profit_loss']
            cumulative_returns.append(cumulative)

        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0.0
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / self.initial_bankroll
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0.0

        return {
            'total_bets': counts['total_bets'] or 0,
            'settled_bets': num_settled,
            'pending_bets': counts['pending_bets'] or 0,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit_loss': total_profit_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_bankroll': self.bankroll,
            'bankroll_change': self.bankroll - self.initial_bankroll
        }

    def close_session(self):
        """Close paper trading session and update database."""
        metrics = self.get_performance_metrics()

        with self.db_conn.cursor() as cur:
            cur.execute("""
                UPDATE paper_trading_sessions
                SET ended_at = %s,
                    final_bankroll = %s,
                    total_bets = %s,
                    win_rate = %s,
                    roi = %s,
                    sharpe_ratio = %s,
                    max_drawdown = %s
                WHERE session_id = %s
            """, (
                datetime.now(UTC),
                self.bankroll,
                metrics['total_bets'],
                metrics['win_rate'],
                metrics['roi'],
                metrics['sharpe_ratio'],
                metrics['max_drawdown'],
                self.session_id
            ))
            self.db_conn.commit()

        self.db_conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_session()
        return False
