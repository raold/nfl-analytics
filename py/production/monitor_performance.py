"""
monitor_performance.py

Live Production Performance Monitoring

Tracks betting performance in real-time:
- ROI, Sharpe ratio, win rate, max drawdown
- Closing Line Value (CLV) analysis
- Model calibration (Brier score, log loss)
- Bankroll tracking and growth
- Alert system for losing streaks, large drawdowns

Usage:
    # Log a bet
    python py/production/monitor_performance.py log \
        --game-id "2024_10_KC_SF" \
        --bet-type spread \
        --side home \
        --line -3.5 \
        --odds -110 \
        --stake 250 \
        --prediction 0.58

    # Update bet result
    python py/production/monitor_performance.py update \
        --bet-id 123 \
        --result win \
        --home-score 28 \
        --away-score 24

    # Generate daily report
    python py/production/monitor_performance.py report daily

    # Generate weekly report
    python py/production/monitor_performance.py report weekly

    # Check for alerts
    python py/production/monitor_performance.py alert
"""

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Bet:
    """Single bet record."""

    bet_id: int
    timestamp: datetime
    game_id: str
    week: int
    season: int
    bet_type: str  # spread, total, moneyline
    side: str  # home, away, over, under
    line: float
    odds: int  # American odds
    stake: float
    prediction: float  # Model's predicted probability
    result: str | None = None  # win, loss, push
    payout: float | None = None
    home_score: int | None = None
    away_score: int | None = None
    closing_line: float | None = None
    clv: float | None = None  # Closing Line Value


@dataclass
class PerformanceMetrics:
    """Performance metrics for a period."""

    period: str  # daily, weekly, monthly, all-time
    n_bets: int
    n_wins: int
    n_losses: int
    n_pushes: int
    total_staked: float
    total_payout: float
    net_profit: float
    roi: float
    win_rate: float
    avg_odds: float
    avg_stake: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    avg_clv: float
    brier_score: float
    log_loss: float


# ============================================================================
# Performance Monitor
# ============================================================================


class PerformanceMonitor:
    """
    Monitor betting performance in production.
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01",
        initial_bankroll: float = 10000.0,
    ):
        """
        Initialize performance monitor.

        Args:
            db_url: Database connection URL
            initial_bankroll: Starting bankroll
        """
        self.engine = create_engine(db_url)
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll

        logger.info(f"Performance monitor initialized (bankroll: ${initial_bankroll:,.0f})")

    def log_bet(
        self,
        game_id: str,
        week: int,
        season: int,
        bet_type: str,
        side: str,
        line: float,
        odds: int,
        stake: float,
        prediction: float,
        is_paper_trade: bool = False,
    ) -> int:
        """
        Log a new bet to the database.

        Args:
            game_id: Game ID
            week: Week number
            season: Season
            bet_type: spread, total, moneyline
            side: home, away, over, under
            line: Betting line
            odds: American odds
            stake: Bet amount
            prediction: Model's predicted probability
            is_paper_trade: True for paper trading (virtual money)

        Returns:
            bet_id
        """
        timestamp = datetime.now()

        # Insert into database
        query = """
            INSERT INTO bets (
                timestamp, game_id, week, season, bet_type, side, line, odds,
                stake, prediction, is_paper_trade, result, payout
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL
            )
            RETURNING bet_id
        """

        with self.engine.connect() as conn:
            result = conn.execute(
                query,
                (
                    timestamp,
                    game_id,
                    week,
                    season,
                    bet_type,
                    side,
                    line,
                    odds,
                    stake,
                    prediction,
                    is_paper_trade,
                ),
            )
            bet_id = result.fetchone()[0]
            conn.commit()

        mode = "PAPER" if is_paper_trade else "LIVE"
        logger.info(
            f"Logged {mode} bet {bet_id}: {game_id} {bet_type} {side} {line} @ {odds} (${stake})"
        )

        return bet_id

    def update_bet(
        self,
        bet_id: int,
        result: str,
        home_score: int,
        away_score: int,
        closing_line: float | None = None,
    ):
        """
        Update bet result after game finishes.

        Args:
            bet_id: Bet ID
            result: win, loss, push
            home_score: Home team final score
            away_score: Away team final score
            closing_line: Closing line (for CLV calculation)
        """
        # Get bet details
        query = "SELECT * FROM bets WHERE bet_id = %s"
        with self.engine.connect() as conn:
            bet = pd.read_sql(query, conn, params=(bet_id,)).iloc[0]

        # Calculate payout
        if result == "win":
            payout = self._calculate_payout(bet["stake"], bet["odds"])
        elif result == "push":
            payout = 0.0  # Get stake back
        else:
            payout = -bet["stake"]

        # Calculate CLV (Closing Line Value)
        if closing_line is not None:
            clv = abs(bet["line"] - closing_line)
        else:
            clv = None

        # Update database
        update_query = """
            UPDATE bets
            SET result = %s, payout = %s, home_score = %s, away_score = %s,
                closing_line = %s, clv = %s
            WHERE bet_id = %s
        """

        with self.engine.connect() as conn:
            conn.execute(
                update_query,
                (result, payout, home_score, away_score, closing_line, clv, bet_id),
            )
            conn.commit()

        # Update bankroll
        self.current_bankroll += payout

        logger.info(
            f"Updated bet {bet_id}: {result} | Payout: ${payout:+.2f} | "
            f"Bankroll: ${self.current_bankroll:,.2f}"
        )

    def get_bets(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        season: int | None = None,
        week: int | None = None,
    ) -> pd.DataFrame:
        """
        Get bets for a specific period.

        Args:
            start_date: Start date (default: all time)
            end_date: End date (default: now)
            season: Filter by season
            week: Filter by week

        Returns:
            Bets dataframe
        """
        query = "SELECT * FROM bets WHERE 1=1"

        params = []
        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)
        if season:
            query += " AND season = %s"
            params.append(season)
        if week:
            query += " AND week = %s"
            params.append(week)

        query += " ORDER BY timestamp DESC"

        with self.engine.connect() as conn:
            bets = pd.read_sql(query, conn, params=tuple(params))

        return bets

    def calculate_metrics(self, bets: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate performance metrics for a set of bets.

        Args:
            bets: Bets dataframe

        Returns:
            PerformanceMetrics
        """
        # Filter to settled bets only
        settled = bets[bets["result"].notna()].copy()

        if len(settled) == 0:
            return PerformanceMetrics(
                period="unknown",
                n_bets=0,
                n_wins=0,
                n_losses=0,
                n_pushes=0,
                total_staked=0.0,
                total_payout=0.0,
                net_profit=0.0,
                roi=0.0,
                win_rate=0.0,
                avg_odds=0.0,
                avg_stake=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                avg_clv=0.0,
                brier_score=0.0,
                log_loss=0.0,
            )

        # Basic metrics
        n_bets = len(settled)
        n_wins = (settled["result"] == "win").sum()
        n_losses = (settled["result"] == "loss").sum()
        n_pushes = (settled["result"] == "push").sum()

        total_staked = settled["stake"].sum()
        total_payout = settled["payout"].sum()
        net_profit = total_payout

        roi = net_profit / total_staked if total_staked > 0 else 0.0
        win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0.0

        avg_odds = settled["odds"].mean()
        avg_stake = settled["stake"].mean()

        # Sharpe ratio
        returns = settled["payout"] / settled["stake"]
        sharpe_ratio = returns.mean() / (returns.std() + 1e-8) if len(returns) > 1 else 0.0

        # Drawdown
        cumulative_payout = settled["payout"].cumsum()
        running_max = cumulative_payout.cummax()
        drawdown = running_max - cumulative_payout
        max_drawdown = drawdown.max()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        # CLV
        avg_clv = settled["clv"].mean() if "clv" in settled.columns else 0.0

        # Calibration metrics
        predictions = settled["prediction"].values
        actuals = (settled["result"] == "win").astype(int).values

        brier_score = np.mean((predictions - actuals) ** 2)
        log_loss = -np.mean(
            actuals * np.log(predictions + 1e-15) + (1 - actuals) * np.log(1 - predictions + 1e-15)
        )

        return PerformanceMetrics(
            period="custom",
            n_bets=n_bets,
            n_wins=n_wins,
            n_losses=n_losses,
            n_pushes=n_pushes,
            total_staked=total_staked,
            total_payout=total_payout,
            net_profit=net_profit,
            roi=roi,
            win_rate=win_rate,
            avg_odds=avg_odds,
            avg_stake=avg_stake,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            avg_clv=avg_clv,
            brier_score=brier_score,
            log_loss=log_loss,
        )

    def generate_report(self, period: str = "weekly") -> dict:
        """
        Generate performance report.

        Args:
            period: daily, weekly, monthly, season, all-time

        Returns:
            Report dictionary
        """
        logger.info(f"Generating {period} performance report...")

        now = datetime.now()

        if period == "daily":
            start_date = now - timedelta(days=1)
        elif period == "weekly":
            start_date = now - timedelta(days=7)
        elif period == "monthly":
            start_date = now - timedelta(days=30)
        elif period == "season":
            start_date = datetime(now.year, 9, 1)  # NFL season starts in September
        else:  # all-time
            start_date = None

        bets = self.get_bets(start_date=start_date)
        metrics = self.calculate_metrics(bets)
        metrics.period = period

        report = {
            "report_date": now.isoformat(),
            "period": period,
            "metrics": asdict(metrics),
            "bankroll": {
                "initial": self.initial_bankroll,
                "current": self.current_bankroll,
                "growth": self.current_bankroll - self.initial_bankroll,
                "growth_pct": (
                    (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100
                ),
            },
        }

        return report

    def check_alerts(self) -> list[str]:
        """
        Check for performance alerts.

        Returns:
            List of alert messages
        """
        alerts = []

        # Get last 7 days of bets
        bets = self.get_bets(start_date=datetime.now() - timedelta(days=7))
        settled = bets[bets["result"].notna()]

        if len(settled) < 5:
            return alerts  # Not enough data

        # Alert 1: Losing streak (5+ consecutive losses)
        last_5_results = settled.tail(5)["result"].tolist()
        if all(r == "loss" for r in last_5_results):
            alerts.append("🚨 ALERT: 5 consecutive losses detected. Consider reducing bet sizes.")

        # Alert 2: Large drawdown (>15% of bankroll)
        metrics = self.calculate_metrics(settled)
        if metrics.current_drawdown > self.initial_bankroll * 0.15:
            alerts.append(
                f"🚨 ALERT: Large drawdown detected (${metrics.current_drawdown:,.0f} = "
                f"{metrics.current_drawdown / self.initial_bankroll:.1%} of bankroll). "
                "Consider pausing betting."
            )

        # Alert 3: Model drift (Brier score > 0.23)
        if metrics.brier_score > 0.23:
            alerts.append(
                f"⚠️ WARNING: Model calibration declining (Brier = {metrics.brier_score:.4f}). "
                "Review recent predictions."
            )

        # Alert 4: Negative CLV (betting worse lines than closing)
        if metrics.avg_clv < -0.5:
            alerts.append(
                f"⚠️ WARNING: Negative CLV detected ({metrics.avg_clv:.2f}). "
                "Consider betting earlier or using better line shopping."
            )

        # Alert 5: Low win rate (<50%)
        if metrics.win_rate < 0.50 and metrics.n_bets >= 20:
            alerts.append(
                f"⚠️ WARNING: Win rate below 50% ({metrics.win_rate:.1%}). "
                "Review model predictions."
            )

        return alerts

    @staticmethod
    def _calculate_payout(stake: float, american_odds: int) -> float:
        """
        Calculate payout from American odds.

        Args:
            stake: Bet amount
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Payout (profit, not including stake)
        """
        if american_odds > 0:
            return stake * (american_odds / 100)
        else:
            return stake * (100 / abs(american_odds))


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Production Performance Monitor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Log bet
    log_parser = subparsers.add_parser("log", help="Log a new bet")
    log_parser.add_argument("--game-id", required=True, help="Game ID")
    log_parser.add_argument("--week", type=int, required=True, help="Week number")
    log_parser.add_argument("--season", type=int, required=True, help="Season")
    log_parser.add_argument("--bet-type", required=True, choices=["spread", "total", "moneyline"])
    log_parser.add_argument("--side", required=True, help="home, away, over, under")
    log_parser.add_argument("--line", type=float, required=True, help="Betting line")
    log_parser.add_argument("--odds", type=int, required=True, help="American odds")
    log_parser.add_argument("--stake", type=float, required=True, help="Bet amount")
    log_parser.add_argument("--prediction", type=float, required=True, help="Model probability")
    log_parser.add_argument(
        "--paper-trade", action="store_true", help="Paper trading mode (virtual money)"
    )

    # Update bet
    update_parser = subparsers.add_parser("update", help="Update bet result")
    update_parser.add_argument("--bet-id", type=int, required=True, help="Bet ID")
    update_parser.add_argument("--result", required=True, choices=["win", "loss", "push"])
    update_parser.add_argument("--home-score", type=int, required=True)
    update_parser.add_argument("--away-score", type=int, required=True)
    update_parser.add_argument("--closing-line", type=float, help="Closing line (for CLV)")

    # Generate report
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument(
        "period",
        choices=["daily", "weekly", "monthly", "season", "all"],
        help="Report period",
    )

    # Check alerts
    subparsers.add_parser("alert", help="Check for performance alerts")

    args = parser.parse_args()

    # Initialize monitor
    monitor = PerformanceMonitor()

    # Execute command
    if args.command == "log":
        bet_id = monitor.log_bet(
            game_id=args.game_id,
            week=args.week,
            season=args.season,
            bet_type=args.bet_type,
            side=args.side,
            line=args.line,
            odds=args.odds,
            stake=args.stake,
            prediction=args.prediction,
            is_paper_trade=args.paper_trade,
        )
        mode = "PAPER TRADE" if args.paper_trade else "LIVE"
        print(f"{mode} bet logged successfully (ID: {bet_id})")

    elif args.command == "update":
        monitor.update_bet(
            bet_id=args.bet_id,
            result=args.result,
            home_score=args.home_score,
            away_score=args.away_score,
            closing_line=args.closing_line,
        )
        print(f"Bet {args.bet_id} updated successfully")

    elif args.command == "report":
        report = monitor.generate_report(period=args.period)

        print("\n" + "=" * 70)
        print(f"PERFORMANCE REPORT - {report['period'].upper()}")
        print("=" * 70)
        print(f"Report Date: {report['report_date']}")
        print()

        metrics = report["metrics"]
        print("Betting Performance:")
        print(f"  Total Bets: {metrics['n_bets']}")
        print(
            f"  Wins: {metrics['n_wins']} | Losses: {metrics['n_losses']} | Pushes: {metrics['n_pushes']}"
        )
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Total Staked: ${metrics['total_staked']:,.2f}")
        print(f"  Net Profit: ${metrics['net_profit']:+,.2f}")
        print(f"  ROI: {metrics['roi']:+.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print()

        print("Risk Metrics:")
        print(f"  Max Drawdown: ${metrics['max_drawdown']:,.2f}")
        print(f"  Current Drawdown: ${metrics['current_drawdown']:,.2f}")
        print(f"  Avg CLV: {metrics['avg_clv']:+.2f} points")
        print()

        print("Model Calibration:")
        print(f"  Brier Score: {metrics['brier_score']:.4f} (lower is better, <0.23 good)")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print()

        bankroll = report["bankroll"]
        print("Bankroll:")
        print(f"  Initial: ${bankroll['initial']:,.2f}")
        print(f"  Current: ${bankroll['current']:,.2f}")
        print(f"  Growth: ${bankroll['growth']:+,.2f} ({bankroll['growth_pct']:+.1f}%)")
        print("=" * 70)

    elif args.command == "alert":
        alerts = monitor.check_alerts()

        if len(alerts) == 0:
            print("✅ No alerts. All systems operating normally.")
        else:
            print("\n" + "=" * 70)
            print("PERFORMANCE ALERTS")
            print("=" * 70)
            for alert in alerts:
                print(f"  {alert}")
            print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
