#!/usr/bin/env python3
"""
Portfolio Optimization for Correlated NFL Props Bets
Uses cvxpy for quadratic programming with Kelly criterion constraints
Handles correlation between props on same game
"""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BetOpportunity:
    """Single betting opportunity"""

    player_id: str
    prop_type: str
    line: float
    odds: float  # Decimal odds (e.g., 1.91)
    predicted_prob: float
    predicted_mean: float
    predicted_std: float
    game_id: str
    season: int
    week: int

    @property
    def edge(self) -> float:
        """Expected value edge"""
        implied_prob = 1.0 / self.odds
        return self.predicted_prob - implied_prob

    @property
    def kelly_fraction(self) -> float:
        """Kelly criterion bet size (uncorrelated)"""
        if self.edge <= 0:
            return 0.0
        return self.edge / (self.odds - 1)


class CorrelatedPortfolioOptimizer:
    """
    Optimize bet sizes for portfolio of correlated props
    Uses quadratic programming to maximize expected log growth
    """

    def __init__(
        self,
        max_bet_size: float = 0.05,  # 5% of bankroll per bet
        max_total_exposure: float = 0.25,  # 25% of bankroll total
        min_edge: float = 0.02,  # 2% minimum edge
        kelly_fraction: float = 0.25,  # Fractional Kelly (quarter Kelly)
        correlation_penalty: float = 1.0,  # How much to penalize correlation
    ):
        self.max_bet_size = max_bet_size
        self.max_total_exposure = max_total_exposure
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.correlation_penalty = correlation_penalty

    def estimate_correlation_matrix(self, bets: list[BetOpportunity]) -> np.ndarray:
        """
        Estimate correlation matrix between bets
        Based on:
        - Same game → high correlation
        - Same player → very high correlation
        - Same position → moderate correlation
        """
        n = len(bets)
        corr_matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                bet_i = bets[i]
                bet_j = bets[j]

                # Same player, different props → 0.7 correlation
                if bet_i.player_id == bet_j.player_id:
                    corr = 0.7

                # Same game, different players → 0.3 correlation
                elif bet_i.game_id == bet_j.game_id:
                    corr = 0.3

                # Same week, different games → 0.1 correlation
                elif bet_i.week == bet_j.week:
                    corr = 0.1

                else:
                    corr = 0.0

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    def optimize_portfolio(
        self, bets: list[BetOpportunity], bankroll: float = 1.0
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Optimize bet sizes using quadratic programming

        Maximizes: E[log(bankroll)] = Σ f_i * edge_i - 0.5 * Σ Σ f_i * f_j * σ_ij
        Subject to:
            - 0 <= f_i <= max_bet_size
            - Σ f_i <= max_total_exposure
            - Only bet on positive edge

        Returns:
            bet_sizes: Optimal fraction of bankroll for each bet
            metrics: Dictionary of portfolio metrics
        """
        n = len(bets)

        # Filter for positive edge bets
        valid_bets = [b for b in bets if b.edge >= self.min_edge]

        if not valid_bets:
            logger.warning("No bets with sufficient edge found")
            return np.zeros(n), {"n_bets": 0, "total_exposure": 0}

        # Get edges and standard deviations
        edges = np.array([b.edge for b in valid_bets])
        np.array([b.odds for b in valid_bets])

        # Estimate variance-covariance matrix
        corr_matrix = self.estimate_correlation_matrix(valid_bets)

        # Standard deviation of bet outcomes
        # For binary outcome: σ = sqrt(p * (1-p))
        probs = np.array([b.predicted_prob for b in valid_bets])
        std_devs = np.sqrt(probs * (1 - probs))

        # Convert correlation to covariance
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

        # Apply correlation penalty
        cov_matrix *= self.correlation_penalty

        # Optimization variables
        f = cp.Variable(len(valid_bets))  # Bet fractions

        # Objective: Maximize expected log growth
        # E[log(1 + Σ f_i * payoff_i)] ≈ Σ f_i * edge_i - 0.5 * f^T * Σ * f
        expected_return = edges @ f
        portfolio_variance = cp.quad_form(f, cov_matrix)

        objective = cp.Maximize(expected_return - 0.5 * portfolio_variance)

        # Constraints
        constraints = [
            f >= 0,  # No negative bets
            f <= self.max_bet_size * self.kelly_fraction,  # Max per bet
            cp.sum(f) <= self.max_total_exposure * self.kelly_fraction,  # Max total
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization failed: {problem.status}")
            return np.zeros(n), {"status": problem.status}

        # Extract solution
        bet_sizes = f.value

        # Map back to original bet list (including filtered bets)
        full_bet_sizes = np.zeros(n)
        valid_indices = [i for i, b in enumerate(bets) if b.edge >= self.min_edge]
        full_bet_sizes[valid_indices] = bet_sizes

        # Calculate metrics
        metrics = {
            "n_bets": np.sum(bet_sizes > 1e-4),
            "total_exposure": np.sum(bet_sizes),
            "expected_return": float(expected_return.value),
            "portfolio_std": float(np.sqrt(portfolio_variance.value)),
            "sharpe_ratio": (
                float(expected_return.value / np.sqrt(portfolio_variance.value))
                if portfolio_variance.value > 0
                else 0
            ),
            "max_bet_size": float(np.max(bet_sizes)),
            "avg_correlation": float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])),
        }

        return full_bet_sizes, metrics

    def get_bet_recommendations(
        self, bets: list[BetOpportunity], bankroll: float, min_bet_amount: float = 10.0
    ) -> pd.DataFrame:
        """
        Get actionable bet recommendations

        Returns:
            DataFrame with bet sizes, edges, and priorities
        """
        bet_sizes, metrics = self.optimize_portfolio(bets, bankroll)

        # Convert to DataFrame
        recommendations = []
        for i, (bet, size) in enumerate(zip(bets, bet_sizes)):
            if size > 1e-4:  # Only include actual bets
                bet_amount = size * bankroll

                if bet_amount >= min_bet_amount:
                    recommendations.append(
                        {
                            "player_id": bet.player_id,
                            "prop_type": bet.prop_type,
                            "line": bet.line,
                            "odds": bet.odds,
                            "predicted_prob": bet.predicted_prob,
                            "edge": bet.edge,
                            "kelly_fraction": size,
                            "bet_amount": bet_amount,
                            "expected_profit": bet_amount * bet.edge,
                            "game_id": bet.game_id,
                            "week": bet.week,
                        }
                    )

        df = pd.DataFrame(recommendations)

        if not df.empty:
            df = df.sort_values("expected_profit", ascending=False)

        return df, metrics


class SimpleKellyOptimizer:
    """
    Simpler Kelly criterion optimizer without correlation
    Faster but less accurate for correlated bets
    """

    def __init__(self, kelly_fraction: float = 0.25, max_bet_size: float = 0.05):
        self.kelly_fraction = kelly_fraction
        self.max_bet_size = max_bet_size

    def optimize_bets(self, bets: list[BetOpportunity], bankroll: float = 1.0) -> np.ndarray:
        """Simple Kelly criterion for each bet independently"""

        bet_sizes = np.array(
            [
                min(b.kelly_fraction * self.kelly_fraction, self.max_bet_size) if b.edge > 0 else 0
                for b in bets
            ]
        )

        return bet_sizes


if __name__ == "__main__":
    # Demo: Optimize portfolio of correlated props
    logger.info("Portfolio Optimization Demo")

    # Create sample bets (correlated props on same game)
    bets = [
        BetOpportunity(
            player_id="mahomes",
            prop_type="passing_yards",
            line=275.5,
            odds=1.91,
            predicted_prob=0.58,
            predicted_mean=290.0,
            predicted_std=35.0,
            game_id="2024_06_KC_BUF",
            season=2024,
            week=6,
        ),
        BetOpportunity(
            player_id="kelce",
            prop_type="receiving_yards",
            line=65.5,
            odds=1.95,
            predicted_prob=0.55,
            predicted_mean=72.0,
            predicted_std=25.0,
            game_id="2024_06_KC_BUF",
            season=2024,
            week=6,
        ),
        BetOpportunity(
            player_id="hill",
            prop_type="receiving_yards",
            line=85.5,
            odds=1.90,
            predicted_prob=0.60,
            predicted_mean=95.0,
            predicted_std=30.0,
            game_id="2024_06_MIA_NE",
            season=2024,
            week=6,
        ),
    ]

    # Optimize without correlation
    simple_optimizer = SimpleKellyOptimizer()
    simple_sizes = simple_optimizer.optimize_bets(bets)

    # Optimize with correlation
    portfolio_optimizer = CorrelatedPortfolioOptimizer()
    corr_sizes, metrics = portfolio_optimizer.optimize_portfolio(bets)

    # Compare
    logger.info("\nBet size comparison:")
    logger.info("Bet                           Simple Kelly  Corr-Adjusted")
    logger.info("-" * 60)
    for i, bet in enumerate(bets):
        logger.info(
            f"{bet.player_id:15} {bet.prop_type:15} "
            f"{simple_sizes[i]:6.2%}       {corr_sizes[i]:6.2%}"
        )

    logger.info("\nPortfolio metrics:")
    logger.info(f"  Total exposure: {metrics['total_exposure']:.2%}")
    logger.info(f"  Expected return: {metrics['expected_return']:.4f}")
    logger.info(f"  Portfolio Std: {metrics['portfolio_std']:.4f}")
    logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Avg correlation: {metrics['avg_correlation']:.3f}")
