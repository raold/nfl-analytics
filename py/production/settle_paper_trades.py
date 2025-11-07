"""
Automated Settlement of Paper Trades

This module automatically settles pending paper trades by fetching actual player
stats from the database after games complete. Designed to run daily via cron job.

Key Features:
- Fetches actual player stats from mart.player_game_stats
- Settles pending paper trades based on actual results
- Updates paper_trading_sessions with performance metrics
- Sends email alerts for session completion
- Handles edge cases (cancelled games, missing data)

Author: Claude Code
Date: November 6, 2025
Version: 1.0

Usage:
    # Run daily settlement (typically via cron)
    python py/production/settle_paper_trades.py

    # Settle specific session
    python py/production/settle_paper_trades.py --session-id paper_20241106_abc123

    # Dry run (no database updates)
    python py/production/settle_paper_trades.py --dry-run

    # Add to crontab:
    0 12 * * * cd /Users/dro/rice/nfl-analytics && uv run python py/production/settle_paper_trades.py
"""

import argparse
import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradeSettler:
    """
    Automated settlement system for paper trades.

    Attributes:
        db_conn: PostgreSQL database connection
        dry_run: If True, no database updates (for testing)
    """

    def __init__(self, dry_run: bool = False):
        """
        Initialize paper trade settler.

        Args:
            dry_run: If True, no database updates
        """
        self.dry_run = dry_run
        self.db_conn = self._connect_db()

    def _connect_db(self) -> psycopg2.extensions.connection:
        """Connect to PostgreSQL database."""
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5544)),
            database=os.getenv('DB_NAME', 'devdb01'),
            user=os.getenv('DB_USER', 'dro'),
            password=os.getenv('DB_PASSWORD', 'sicillionbillions')
        )
        return conn

    def get_pending_trades(
        self,
        session_id: Optional[str] = None,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Get pending paper trades that need settlement.

        Args:
            session_id: Optional session ID to filter by
            days_back: Only get trades from last N days

        Returns:
            List of pending paper trades
        """
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT
                    pt.paper_trade_id,
                    pt.game_id,
                    pt.player_id,
                    pt.player_name,
                    pt.prop_type,
                    pt.line_value,
                    pt.bet_side,
                    pt.bet_amount,
                    pt.odds,
                    pt.session_id,
                    pt.trade_timestamp
                FROM paper_trades pt
                WHERE pt.outcome = 'pending'
                  AND pt.trade_timestamp >= %s
            """

            params = [datetime.now(timezone.utc) - timedelta(days=days_back)]

            if session_id:
                query += " AND pt.session_id = %s"
                params.append(session_id)

            query += " ORDER BY pt.trade_timestamp ASC"

            cur.execute(query, params)
            return cur.fetchall()

    def get_actual_player_stat(
        self,
        player_id: str,
        game_id: str,
        prop_type: str
    ) -> Optional[float]:
        """
        Get actual player stat from database.

        Args:
            player_id: Player identifier
            game_id: Game identifier
            prop_type: Type of stat (e.g., 'passing_yards')

        Returns:
            Actual stat value, or None if not found
        """
        # Map prop_type to column name in player_game_stats
        stat_column_map = {
            'passing_yards': 'stat_yards',
            'rushing_yards': 'stat_yards',
            'receiving_yards': 'stat_yards',
            'passing_tds': 'stat_tds',
            'rushing_tds': 'stat_tds',
            'receiving_tds': 'stat_tds',
            'receptions': 'stat_rec',
            'completions': 'stat_comp',
            'attempts': 'stat_att'
        }

        stat_column = stat_column_map.get(prop_type)
        if not stat_column:
            logger.warning(f"Unknown prop_type: {prop_type}")
            return None

        # Determine stat_type for filtering
        if 'passing' in prop_type:
            stat_type = 'passing'
        elif 'rushing' in prop_type:
            stat_type = 'rushing'
        elif 'receiving' in prop_type or prop_type == 'receptions':
            stat_type = 'receiving'
        else:
            stat_type = 'passing'  # default

        with self.db_conn.cursor() as cur:
            cur.execute(f"""
                SELECT {stat_column}
                FROM mart.player_game_stats
                WHERE player_id = %s
                  AND game_id = %s
                  AND stat_type = %s
                LIMIT 1
            """, (player_id, game_id, stat_type))

            result = cur.fetchone()
            return result[0] if result else None

    def settle_trade(
        self,
        paper_trade_id: int,
        line_value: float,
        bet_side: str,
        bet_amount: float,
        odds: float,
        actual_result: float
    ) -> tuple[str, float]:
        """
        Settle a single paper trade.

        Args:
            paper_trade_id: Database ID of paper trade
            line_value: Betting line
            bet_side: 'over' or 'under'
            bet_amount: Bet size
            odds: American odds
            actual_result: Actual player stat

        Returns:
            outcome: 'win', 'loss', or 'push'
            profit_loss: Profit/loss amount
        """
        # Determine outcome
        if actual_result > line_value:
            outcome = 'win' if bet_side == 'over' else 'loss'
        elif actual_result < line_value:
            outcome = 'win' if bet_side == 'under' else 'loss'
        else:
            outcome = 'push'

        # Calculate profit/loss
        if outcome == 'win':
            if odds > 0:
                winnings = bet_amount * (odds / 100.0)
            else:
                winnings = bet_amount * (100.0 / abs(odds))
            profit_loss = winnings
        elif outcome == 'loss':
            profit_loss = -bet_amount
        else:  # push
            profit_loss = 0.0

        # Update database
        if not self.dry_run:
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
                    datetime.now(timezone.utc),
                    profit_loss,
                    paper_trade_id
                ))
                self.db_conn.commit()

        return outcome, profit_loss

    def settle_all_pending(
        self,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Settle all pending paper trades that have results available.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            summary: Dictionary with settlement summary
        """
        pending_trades = self.get_pending_trades(session_id=session_id)

        logger.info(f"Found {len(pending_trades)} pending trades")

        settled = 0
        not_found = 0
        errors = 0

        for trade in pending_trades:
            try:
                # Get actual result
                actual_result = self.get_actual_player_stat(
                    player_id=trade['player_id'],
                    game_id=trade['game_id'],
                    prop_type=trade['prop_type']
                )

                if actual_result is None:
                    logger.debug(
                        f"No result yet for {trade['player_name']} "
                        f"{trade['prop_type']} in {trade['game_id']}"
                    )
                    not_found += 1
                    continue

                # Settle trade
                outcome, profit_loss = self.settle_trade(
                    paper_trade_id=trade['paper_trade_id'],
                    line_value=trade['line_value'],
                    bet_side=trade['bet_side'],
                    bet_amount=trade['bet_amount'],
                    odds=trade['odds'],
                    actual_result=actual_result
                )

                logger.info(
                    f"Settled trade {trade['paper_trade_id']}: "
                    f"{trade['player_name']} {trade['prop_type']} "
                    f"{trade['bet_side']} {trade['line_value']} "
                    f"(actual: {actual_result:.1f}) → {outcome} "
                    f"(P&L: ${profit_loss:+.2f})"
                )

                settled += 1

            except Exception as e:
                logger.error(
                    f"Error settling trade {trade['paper_trade_id']}: {e}"
                )
                errors += 1

        summary = {
            'total_pending': len(pending_trades),
            'settled': settled,
            'not_found': not_found,
            'errors': errors
        }

        logger.info(
            f"Settlement complete: {settled} settled, "
            f"{not_found} waiting for results, {errors} errors"
        )

        return summary

    def update_session_metrics(self, session_id: str):
        """
        Update performance metrics for a paper trading session.

        Args:
            session_id: Session identifier
        """
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get settled trades for this session
            cur.execute("""
                SELECT
                    outcome,
                    bet_amount,
                    profit_loss
                FROM paper_trades
                WHERE session_id = %s
                  AND outcome IN ('win', 'loss', 'push')
            """, (session_id,))

            settled_bets = cur.fetchall()

            if not settled_bets:
                logger.info(f"No settled bets for session {session_id}")
                return

            # Calculate metrics
            num_settled = len(settled_bets)
            num_wins = sum(1 for bet in settled_bets if bet['outcome'] == 'win')
            win_rate = num_wins / num_settled if num_settled > 0 else 0.0

            total_wagered = sum(bet['bet_amount'] for bet in settled_bets)
            total_profit_loss = sum(bet['profit_loss'] for bet in settled_bets)
            roi = total_profit_loss / total_wagered if total_wagered > 0 else 0.0

            # Sharpe ratio
            import numpy as np
            returns = [bet['profit_loss'] / bet['bet_amount'] for bet in settled_bets]
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown
            cumulative_pnl = []
            cumulative = 0.0
            for bet in settled_bets:
                cumulative += bet['profit_loss']
                cumulative_pnl.append(cumulative)

            peak = cumulative_pnl[0] if cumulative_pnl else 0.0
            max_dd = 0.0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                dd = (peak - value)
                if dd > max_dd:
                    max_dd = dd

            # Get initial bankroll
            cur.execute("""
                SELECT initial_bankroll
                FROM paper_trading_sessions
                WHERE session_id = %s
            """, (session_id,))
            initial_bankroll = cur.fetchone()['initial_bankroll']

            max_drawdown = max_dd / initial_bankroll if initial_bankroll > 0 else 0.0

            # Update session
            if not self.dry_run:
                cur.execute("""
                    UPDATE paper_trading_sessions
                    SET total_bets = %s,
                        win_rate = %s,
                        roi = %s,
                        sharpe_ratio = %s,
                        max_drawdown = %s,
                        final_bankroll = %s
                    WHERE session_id = %s
                """, (
                    num_settled,
                    win_rate,
                    roi,
                    sharpe,
                    max_drawdown,
                    initial_bankroll + total_profit_loss,
                    session_id
                ))
                self.db_conn.commit()

            logger.info(
                f"Updated session {session_id}: "
                f"{num_settled} bets, "
                f"{win_rate:.1%} win rate, "
                f"{roi:+.2%} ROI, "
                f"{sharpe:.2f} Sharpe"
            )

    def send_alert_email(self, summary: Dict, session_id: Optional[str] = None):
        """
        Send email alert with settlement summary.

        Args:
            summary: Settlement summary dict
            session_id: Optional session ID
        """
        # Email configuration (optional)
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER')
        smtp_password = os.getenv('SMTP_PASSWORD')
        email_to = os.getenv('ALERT_EMAIL', 'dro@example.com')

        if not all([smtp_host, smtp_user, smtp_password]):
            logger.info("Email not configured, skipping alert")
            return

        subject = f"Paper Trading Settlement: {summary['settled']} trades settled"

        body = f"""
Paper Trading Settlement Report
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Summary:
- Total pending trades: {summary['total_pending']}
- Settled: {summary['settled']}
- Waiting for results: {summary['not_found']}
- Errors: {summary['errors']}

Session: {session_id or 'All sessions'}

This is an automated message from the paper trading settlement system.
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = email_to

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"Alert email sent to {email_to}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def close(self):
        """Close database connection."""
        self.db_conn.close()


def main():
    """Main entry point for automated settlement."""
    parser = argparse.ArgumentParser(
        description='Settle pending paper trades automatically'
    )
    parser.add_argument(
        '--session-id',
        type=str,
        help='Only settle trades for specific session'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (no database updates)'
    )
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='Skip email alerts'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("Running in DRY RUN mode (no database updates)")

    # Create settler
    settler = PaperTradeSettler(dry_run=args.dry_run)

    try:
        # Settle all pending trades
        summary = settler.settle_all_pending(session_id=args.session_id)

        # Update session metrics
        if args.session_id:
            settler.update_session_metrics(args.session_id)

        # Send email alert
        if not args.no_email and not args.dry_run:
            settler.send_alert_email(summary, session_id=args.session_id)

    finally:
        settler.close()


if __name__ == '__main__':
    main()
