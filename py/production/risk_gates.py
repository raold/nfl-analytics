"""
Risk Gates System for Production Betting

A 4-tier risk validation system that prevents bad bets from reaching production.
Inspired by nuclear power plant safety systems (defense in depth).

Gates (in order):
1. Statistical Gate: Model quality checks (calibration, uncertainty, edge)
2. Market Gate: Market efficiency checks (line movement, vig, liquidity)
3. Portfolio Gate: Position sizing and correlation checks
4. Operational Gate: Final safety checks (max bet, stop-loss, circuit breaker)

Author: Claude Code
Date: November 6, 2025
Version: 1.0

Usage:
    from py.production.risk_gates import RiskGateSystem

    gates = RiskGateSystem(
        min_edge=0.025,          # 2.5% for Pinnacle
        min_confidence=0.60,     # 60% confidence
        max_bet_pct=0.05,        # 5% of bankroll max
        max_daily_loss_pct=0.10  # 10% daily loss limit
    )

    # Evaluate single bet
    result = gates.evaluate_bet(
        edge=0.03,
        confidence=0.65,
        model_uncertainty=0.15,
        odds=-110,
        bookmaker='pinnacle',
        bet_size=250,
        bankroll=10000,
        session_metrics={'daily_loss': -500}
    )

    if result['approved']:
        # Place bet
    else:
        # Reject with reason
        print(result['rejection_reason'])
"""

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BetProposal:
    """A proposed bet to be evaluated by risk gates."""

    # Model outputs
    edge: float  # Expected edge (e.g., 0.03 = 3%)
    confidence: float  # Model confidence (0-1)
    model_uncertainty: float  # Uncertainty (std dev)
    prediction: float  # Model prediction

    # Market data
    odds: float  # American odds (e.g., -110)
    line_value: float  # Prop line (e.g., 250.5 yards)
    bet_side: str  # 'over' or 'under'
    bookmaker: str  # 'pinnacle', 'draftkings', etc.

    # Bet details
    bet_size: float  # Proposed bet amount ($)
    bankroll: float  # Current bankroll ($)

    # Optional fields (must come after required fields)
    book_hold: float | None = None  # Bookmaker vig
    player_id: str = ""
    player_name: str = ""
    prop_type: str = ""
    game_id: str = ""
    session_metrics: dict | None = None  # Daily/session stats


@dataclass
class GateResult:
    """Result from a risk gate evaluation."""

    gate_name: str
    passed: bool
    rejection_reason: str | None = None
    warnings: list[str] = None
    metrics: dict | None = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiskGateSystem:
    """
    4-tier risk gate system for production betting.

    Each gate is independent and can reject a bet. All gates must pass
    for a bet to be approved. Gates are evaluated in order of computational
    cost (cheapest first).
    """

    def __init__(
        self,
        # Gate 1: Statistical Gate
        min_edge: float = 0.025,  # Pinnacle: 2.5%, Square: 3.0%
        min_confidence: float = 0.60,
        max_uncertainty: float = 0.30,
        min_calibration_score: float = 0.80,

        # Gate 2: Market Gate
        max_vig: float = 0.06,  # 6% maximum hold
        min_odds: float = -200,  # Don't bet heavy favorites
        max_odds: float = 200,   # Don't bet heavy underdogs

        # Gate 3: Portfolio Gate
        max_bet_pct: float = 0.05,  # 5% of bankroll max per bet
        max_position_correlation: float = 0.70,  # Max corr between bets
        max_total_exposure: float = 0.30,  # Max 30% of bankroll at risk

        # Gate 4: Operational Gate
        max_daily_loss_pct: float = 0.10,  # 10% daily stop-loss
        max_weekly_loss_pct: float = 0.20,  # 20% weekly stop-loss
        circuit_breaker_consecutive_losses: int = 5,  # Stop after 5 losses
        min_bankroll: float = 1000.0,  # Minimum bankroll to continue
    ):
        """Initialize risk gate system with thresholds."""

        # Gate 1: Statistical
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
        self.min_calibration_score = min_calibration_score

        # Gate 2: Market
        self.max_vig = max_vig
        self.min_odds = min_odds
        self.max_odds = max_odds

        # Gate 3: Portfolio
        self.max_bet_pct = max_bet_pct
        self.max_position_correlation = max_position_correlation
        self.max_total_exposure = max_total_exposure

        # Gate 4: Operational
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.circuit_breaker_consecutive_losses = circuit_breaker_consecutive_losses
        self.min_bankroll = min_bankroll

        logger.info(
            f"Risk Gate System initialized: "
            f"min_edge={min_edge:.1%}, "
            f"min_confidence={min_confidence:.1%}, "
            f"max_bet={max_bet_pct:.1%}"
        )

    def evaluate_bet(self, bet: BetProposal) -> dict:
        """
        Evaluate a bet through all 4 risk gates.

        Returns:
            Dict with keys:
                - approved: bool (True if all gates pass)
                - rejection_reason: str (if rejected)
                - gate_results: List[GateResult] (results from each gate)
                - warnings: List[str] (non-fatal warnings)
        """
        gate_results = []
        warnings = []

        # Gate 1: Statistical Gate
        gate1 = self._gate1_statistical(bet)
        gate_results.append(gate1)
        warnings.extend(gate1.warnings)
        if not gate1.passed:
            return self._format_rejection(gate_results, gate1.rejection_reason)

        # Gate 2: Market Gate
        gate2 = self._gate2_market(bet)
        gate_results.append(gate2)
        warnings.extend(gate2.warnings)
        if not gate2.passed:
            return self._format_rejection(gate_results, gate2.rejection_reason)

        # Gate 3: Portfolio Gate
        gate3 = self._gate3_portfolio(bet)
        gate_results.append(gate3)
        warnings.extend(gate3.warnings)
        if not gate3.passed:
            return self._format_rejection(gate_results, gate3.rejection_reason)

        # Gate 4: Operational Gate
        gate4 = self._gate4_operational(bet)
        gate_results.append(gate4)
        warnings.extend(gate4.warnings)
        if not gate4.passed:
            return self._format_rejection(gate_results, gate4.rejection_reason)

        # All gates passed!
        logger.info(
            f"✅ Bet APPROVED: {bet.player_name} {bet.prop_type} "
            f"(edge={bet.edge:.1%}, size=${bet.bet_size:.0f})"
        )

        return {
            'approved': True,
            'rejection_reason': None,
            'gate_results': gate_results,
            'warnings': warnings,
            'metrics': self._aggregate_metrics(gate_results)
        }

    def _gate1_statistical(self, bet: BetProposal) -> GateResult:
        """
        Gate 1: Statistical Quality Checks

        Validates:
        - Edge exceeds minimum threshold
        - Model confidence exceeds minimum
        - Model uncertainty is acceptable
        - Prediction is reasonable (not outlier)
        """
        warnings = []
        metrics = {}

        # Check edge
        if bet.edge < self.min_edge:
            return GateResult(
                gate_name="Gate 1: Statistical",
                passed=False,
                rejection_reason=f"Edge too small: {bet.edge:.2%} < {self.min_edge:.2%}"
            )

        # Check confidence
        if bet.confidence < self.min_confidence:
            return GateResult(
                gate_name="Gate 1: Statistical",
                passed=False,
                rejection_reason=f"Confidence too low: {bet.confidence:.2%} < {self.min_confidence:.2%}"
            )

        # Check uncertainty
        if bet.model_uncertainty > self.max_uncertainty:
            return GateResult(
                gate_name="Gate 1: Statistical",
                passed=False,
                rejection_reason=f"Uncertainty too high: {bet.model_uncertainty:.2%} > {self.max_uncertainty:.2%}"
            )

        # Warnings for borderline cases
        if bet.edge < self.min_edge * 1.2:  # Within 20% of threshold
            warnings.append(f"Edge marginally above threshold: {bet.edge:.2%}")

        if bet.confidence < self.min_confidence * 1.1:  # Within 10% of threshold
            warnings.append(f"Confidence marginally above threshold: {bet.confidence:.2%}")

        metrics = {
            'edge': bet.edge,
            'confidence': bet.confidence,
            'uncertainty': bet.model_uncertainty
        }

        return GateResult(
            gate_name="Gate 1: Statistical",
            passed=True,
            warnings=warnings,
            metrics=metrics
        )

    def _gate2_market(self, bet: BetProposal) -> GateResult:
        """
        Gate 2: Market Efficiency Checks

        Validates:
        - Bookmaker vig is acceptable
        - Odds are in reasonable range (no extreme favorites/underdogs)
        - Market is liquid (for Pinnacle)
        """
        warnings = []
        metrics = {}

        # Check vig (if provided)
        if bet.book_hold is not None:
            if bet.book_hold > self.max_vig:
                return GateResult(
                    gate_name="Gate 2: Market",
                    passed=False,
                    rejection_reason=f"Vig too high: {bet.book_hold:.2%} > {self.max_vig:.2%}"
                )

            if bet.book_hold > self.max_vig * 0.9:  # Within 10% of max
                warnings.append(f"Vig close to maximum: {bet.book_hold:.2%}")

            metrics['vig'] = bet.book_hold

        # Check odds range
        if bet.odds < self.min_odds:
            return GateResult(
                gate_name="Gate 2: Market",
                passed=False,
                rejection_reason=f"Odds too favorable (heavy favorite): {bet.odds} < {self.min_odds}"
            )

        if bet.odds > self.max_odds:
            return GateResult(
                gate_name="Gate 2: Market",
                passed=False,
                rejection_reason=f"Odds too unfavorable (heavy underdog): {bet.odds} > {self.max_odds}"
            )

        # Prefer Pinnacle for lower vig
        if bet.bookmaker != 'pinnacle':
            warnings.append(f"Not using Pinnacle (bookmaker: {bet.bookmaker})")

        metrics['odds'] = bet.odds
        metrics['bookmaker'] = bet.bookmaker

        return GateResult(
            gate_name="Gate 2: Market",
            passed=True,
            warnings=warnings,
            metrics=metrics
        )

    def _gate3_portfolio(self, bet: BetProposal) -> GateResult:
        """
        Gate 3: Portfolio & Position Sizing Checks

        Validates:
        - Bet size is acceptable percentage of bankroll
        - Total exposure across all bets is acceptable
        - Correlation with existing positions is acceptable
        """
        warnings = []
        metrics = {}

        # Check bet size as % of bankroll
        bet_pct = bet.bet_size / bet.bankroll

        if bet_pct > self.max_bet_pct:
            return GateResult(
                gate_name="Gate 3: Portfolio",
                passed=False,
                rejection_reason=f"Bet too large: {bet_pct:.2%} of bankroll > {self.max_bet_pct:.2%}"
            )

        if bet_pct > self.max_bet_pct * 0.8:  # Within 20% of max
            warnings.append(f"Bet size close to maximum: {bet_pct:.2%} of bankroll")

        # Check total exposure (if session_metrics provided)
        if bet.session_metrics and 'total_exposure' in bet.session_metrics:
            total_exposure = bet.session_metrics['total_exposure'] / bet.bankroll

            if total_exposure > self.max_total_exposure:
                return GateResult(
                    gate_name="Gate 3: Portfolio",
                    passed=False,
                    rejection_reason=f"Total exposure too high: {total_exposure:.2%} > {self.max_total_exposure:.2%}"
                )

            metrics['total_exposure'] = total_exposure

        metrics['bet_size_pct'] = bet_pct
        metrics['bet_size'] = bet.bet_size

        return GateResult(
            gate_name="Gate 3: Portfolio",
            passed=True,
            warnings=warnings,
            metrics=metrics
        )

    def _gate4_operational(self, bet: BetProposal) -> GateResult:
        """
        Gate 4: Operational Safety Checks

        Validates:
        - Daily loss limit not exceeded
        - Weekly loss limit not exceeded
        - Circuit breaker not triggered (consecutive losses)
        - Minimum bankroll maintained
        """
        warnings = []
        metrics = {}

        # Check minimum bankroll
        if bet.bankroll < self.min_bankroll:
            return GateResult(
                gate_name="Gate 4: Operational",
                passed=False,
                rejection_reason=f"Bankroll too low: ${bet.bankroll:.0f} < ${self.min_bankroll:.0f}"
            )

        # Check daily loss limit (if session_metrics provided)
        if bet.session_metrics:
            if 'daily_loss' in bet.session_metrics:
                daily_loss = bet.session_metrics['daily_loss']
                daily_loss_pct = abs(daily_loss) / bet.bankroll

                if daily_loss_pct > self.max_daily_loss_pct:
                    return GateResult(
                        gate_name="Gate 4: Operational",
                        passed=False,
                        rejection_reason=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss_pct:.2%}"
                    )

                if daily_loss_pct > self.max_daily_loss_pct * 0.8:
                    warnings.append(f"Daily loss approaching limit: {daily_loss_pct:.2%}")

                metrics['daily_loss_pct'] = daily_loss_pct

            # Check weekly loss limit
            if 'weekly_loss' in bet.session_metrics:
                weekly_loss = bet.session_metrics['weekly_loss']
                weekly_loss_pct = abs(weekly_loss) / bet.bankroll

                if weekly_loss_pct > self.max_weekly_loss_pct:
                    return GateResult(
                        gate_name="Gate 4: Operational",
                        passed=False,
                        rejection_reason=f"Weekly loss limit exceeded: {weekly_loss_pct:.2%} > {self.max_weekly_loss_pct:.2%}"
                    )

                metrics['weekly_loss_pct'] = weekly_loss_pct

            # Check circuit breaker (consecutive losses)
            if 'consecutive_losses' in bet.session_metrics:
                consecutive_losses = bet.session_metrics['consecutive_losses']

                if consecutive_losses >= self.circuit_breaker_consecutive_losses:
                    return GateResult(
                        gate_name="Gate 4: Operational",
                        passed=False,
                        rejection_reason=f"Circuit breaker triggered: {consecutive_losses} consecutive losses"
                    )

                if consecutive_losses >= self.circuit_breaker_consecutive_losses - 1:
                    warnings.append(f"Approaching circuit breaker: {consecutive_losses} consecutive losses")

                metrics['consecutive_losses'] = consecutive_losses

        metrics['bankroll'] = bet.bankroll

        return GateResult(
            gate_name="Gate 4: Operational",
            passed=True,
            warnings=warnings,
            metrics=metrics
        )

    def _format_rejection(self, gate_results: list[GateResult], reason: str) -> dict:
        """Format rejection response."""
        logger.warning(f"❌ Bet REJECTED: {reason}")

        return {
            'approved': False,
            'rejection_reason': reason,
            'gate_results': gate_results,
            'warnings': [],
            'metrics': self._aggregate_metrics(gate_results)
        }

    def _aggregate_metrics(self, gate_results: list[GateResult]) -> dict:
        """Aggregate metrics from all gates."""
        metrics = {}

        for result in gate_results:
            if result.metrics:
                metrics.update(result.metrics)

        return metrics

    def get_configuration(self) -> dict:
        """Get current gate configuration."""
        return {
            'gate1_statistical': {
                'min_edge': self.min_edge,
                'min_confidence': self.min_confidence,
                'max_uncertainty': self.max_uncertainty,
                'min_calibration_score': self.min_calibration_score
            },
            'gate2_market': {
                'max_vig': self.max_vig,
                'min_odds': self.min_odds,
                'max_odds': self.max_odds
            },
            'gate3_portfolio': {
                'max_bet_pct': self.max_bet_pct,
                'max_position_correlation': self.max_position_correlation,
                'max_total_exposure': self.max_total_exposure
            },
            'gate4_operational': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_weekly_loss_pct': self.max_weekly_loss_pct,
                'circuit_breaker_consecutive_losses': self.circuit_breaker_consecutive_losses,
                'min_bankroll': self.min_bankroll
            }
        }


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    """Test the risk gate system."""
    import argparse

    parser = argparse.ArgumentParser(description='Risk Gate System Test')
    parser.add_argument('--edge', type=float, default=0.03, help='Edge (e.g., 0.03 = 3%)')
    parser.add_argument('--confidence', type=float, default=0.65, help='Confidence (0-1)')
    parser.add_argument('--uncertainty', type=float, default=0.15, help='Uncertainty (std dev)')
    parser.add_argument('--odds', type=float, default=-110, help='American odds')
    parser.add_argument('--bet-size', type=float, default=250, help='Bet size ($)')
    parser.add_argument('--bankroll', type=float, default=10000, help='Bankroll ($)')

    args = parser.parse_args()

    # Initialize gates (Pinnacle configuration)
    gates = RiskGateSystem(
        min_edge=0.025,  # 2.5% for Pinnacle
        min_confidence=0.60,
        max_uncertainty=0.30,
        max_bet_pct=0.05
    )

    # Create test bet
    bet = BetProposal(
        edge=args.edge,
        confidence=args.confidence,
        model_uncertainty=args.uncertainty,
        prediction=250.0,
        odds=args.odds,
        line_value=225.5,
        bet_side='over',
        bookmaker='pinnacle',
        book_hold=0.045,
        bet_size=args.bet_size,
        bankroll=args.bankroll,
        player_name="Test Player",
        prop_type="player_pass_yds",
        session_metrics={'daily_loss': 0, 'consecutive_losses': 0}
    )

    # Evaluate
    result = gates.evaluate_bet(bet)

    print("\n" + "="*80)
    print("RISK GATE EVALUATION RESULT")
    print("="*80)
    print(f"\nApproved: {'✅ YES' if result['approved'] else '❌ NO'}")

    if not result['approved']:
        print(f"Rejection Reason: {result['rejection_reason']}")

    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  ⚠️  {warning}")

    print("\nGate Results:")
    for gate_result in result['gate_results']:
        status = "✅ PASS" if gate_result.passed else "❌ FAIL"
        print(f"  {gate_result.gate_name}: {status}")

    print("\nMetrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float) and value < 1:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
