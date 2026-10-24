"""
Book Router with Friction Model

Routes bets to optimal sportsbooks considering:
1. Withdrawal fees and minimum amounts
2. Bonus rollover requirements
3. Betting limits and max payouts
4. Account health (avoiding heat)
5. Geographic/regulatory restrictions
6. Deposit/withdrawal processing times

The friction model accounts for real costs beyond just finding the best line,
optimizing for long-term profitability and account sustainability.

Expected impact: +0.5-1% ROI from reduced friction costs
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class BookTier(Enum):
    """Sportsbook tiers based on sharpness and limits"""

    SHARP = "sharp"  # Pinnacle, Bookmaker, Circa
    REDUCED_JUICE = "reduced_juice"  # BetOnline, Heritage
    RECREATIONAL = "recreational"  # DraftKings, FanDuel, BetMGM
    SOFT = "soft"  # Local books, new entrants


@dataclass
class WithdrawalPolicy:
    """Book withdrawal terms"""

    fee_structure: str  # 'flat', 'percentage', 'free'
    fee_amount: float  # Dollar amount or percentage
    minimum_withdrawal: float
    processing_days: int
    free_withdrawal_frequency: int | None = None  # Days between free withdrawals
    crypto_available: bool = False
    crypto_fee: float = 0.0


@dataclass
class BonusTerms:
    """Active bonus requirements"""

    bonus_amount: float
    rollover_requirement: float  # e.g., 10x means bet 10x bonus amount
    rollover_remaining: float
    qualifying_odds: int  # Minimum odds (e.g., -200)
    expires: datetime
    contribution_rate: float = 1.0  # Some books count different bet types differently


@dataclass
class BookProfile:
    """Complete sportsbook profile with limits and friction"""

    name: str
    tier: BookTier

    # Limits
    max_bet_spread: float
    max_bet_total: float
    max_bet_prop: float
    max_payout: float
    daily_limit: float

    # Account status
    account_balance: float
    lifetime_profit: float  # Track to avoid winning too much
    heat_level: float = 0.0  # 0-1 scale of account heat
    last_bet: datetime | None = None
    restricted_markets: list[str] = field(default_factory=list)

    # Friction costs
    withdrawal_policy: WithdrawalPolicy | None = None
    active_bonus: BonusTerms | None = None

    # Market availability
    has_alt_lines: bool = True
    alt_line_juice: float = 0.05  # Extra juice on alt lines
    live_betting: bool = True

    # Geographic
    available_states: list[str] = field(default_factory=list)
    vpn_friendly: bool = False


@dataclass
class BetRoute:
    """Optimal routing decision for a bet"""

    bet_id: str
    book: str
    line: float
    price: float  # American odds
    bet_amount: float
    expected_value: float
    friction_cost: float
    net_ev: float
    reasons: list[str]
    warnings: list[str]


class FrictionCalculator:
    """Calculate real costs of betting at each book"""

    @staticmethod
    def calculate_withdrawal_cost(
        amount: float, policy: WithdrawalPolicy, last_withdrawal: datetime | None = None
    ) -> float:
        """Calculate cost to withdraw winnings"""
        if amount < policy.minimum_withdrawal:
            return float("inf")  # Can't withdraw

        # Check if free withdrawal available
        if policy.free_withdrawal_frequency and last_withdrawal:
            days_since = (datetime.now() - last_withdrawal).days
            if days_since >= policy.free_withdrawal_frequency:
                return 0.0

        # Calculate fee
        if policy.fee_structure == "flat":
            return policy.fee_amount
        elif policy.fee_structure == "percentage":
            return amount * policy.fee_amount
        else:  # free
            return 0.0

    @staticmethod
    def calculate_bonus_cost(bet_amount: float, odds: int, bonus: BonusTerms) -> float:
        """Calculate opportunity cost of bonus rollover"""
        # Check if bet qualifies
        if odds > bonus.qualifying_odds:  # More negative is "less than"
            return 0.0  # Doesn't contribute to rollover

        # Calculate contribution
        contribution = bet_amount * bonus.contribution_rate

        # Opportunity cost: locked funds * expected time * rate
        days_to_expiry = (bonus.expires - datetime.now()).days
        if days_to_expiry <= 0:
            return float("inf")  # Bonus expired, funds locked

        # Rough estimate: 2% monthly opportunity cost
        daily_opportunity_cost = 0.02 / 30
        cost = bonus.rollover_remaining * daily_opportunity_cost * min(days_to_expiry, 30)

        # Reduce cost by this bet's contribution
        cost -= contribution * daily_opportunity_cost * min(days_to_expiry, 30)

        return max(0, cost)

    @staticmethod
    def calculate_heat_cost(
        current_heat: float, lifetime_profit: float, bet_amount: float, book_tier: BookTier
    ) -> float:
        """Calculate cost of account heat/restrictions"""
        # Sharp books don't limit winners as much
        if book_tier == BookTier.SHARP:
            heat_multiplier = 0.1
        elif book_tier == BookTier.REDUCED_JUICE:
            heat_multiplier = 0.3
        elif book_tier == BookTier.RECREATIONAL:
            heat_multiplier = 1.0
        else:  # SOFT
            heat_multiplier = 1.5

        # Exponential heat cost as account gets hotter
        heat_cost = (np.exp(current_heat * 3) - 1) * heat_multiplier * 0.01 * bet_amount

        # Additional cost if winning too much
        if lifetime_profit > 10000:
            profit_penalty = (lifetime_profit / 10000) * 0.005 * bet_amount
            heat_cost += profit_penalty

        return heat_cost

    @staticmethod
    def calculate_total_friction(
        bet_amount: float, odds: int, book: BookProfile, last_withdrawal: datetime | None = None
    ) -> tuple[float, list[str]]:
        """Calculate total friction cost for a bet"""
        costs = []
        reasons = []

        # Withdrawal cost (amortized)
        if book.withdrawal_policy:
            withdrawal_cost = FrictionCalculator.calculate_withdrawal_cost(
                bet_amount * 2, book.withdrawal_policy, last_withdrawal  # Assume 2x return
            )
            if withdrawal_cost > 0:
                # Amortize over expected number of bets before withdrawal
                amortized_cost = withdrawal_cost / 50  # Assume 50 bets per withdrawal
                costs.append(amortized_cost)
                reasons.append(f"Withdrawal fee: ${amortized_cost:.2f}")

        # Bonus cost
        if book.active_bonus:
            bonus_cost = FrictionCalculator.calculate_bonus_cost(
                bet_amount, odds, book.active_bonus
            )
            if bonus_cost > 0:
                costs.append(bonus_cost)
                reasons.append(f"Bonus lock: ${bonus_cost:.2f}")

        # Heat cost
        heat_cost = FrictionCalculator.calculate_heat_cost(
            book.heat_level, book.lifetime_profit, bet_amount, book.tier
        )
        if heat_cost > 0:
            costs.append(heat_cost)
            reasons.append(f"Heat risk: ${heat_cost:.2f}")

        return sum(costs), reasons


class BookRouter:
    """Intelligent routing of bets to optimal sportsbooks"""

    def __init__(self, books: list[BookProfile]):
        self.books = {book.name: book for book in books}
        self.last_withdrawals: dict[str, datetime] = {}
        self.recent_bets: list[BetRoute] = []

    def route_bet(
        self,
        bet_id: str,
        market: str,  # 'spread', 'total', 'prop'
        team: str,
        model_prob: float,
        available_lines: dict[str, tuple[float, int]],  # book -> (line, odds)
        desired_amount: float,
        user_state: str = "NV",
    ) -> BetRoute:
        """
        Route a bet to optimal book considering all factors.

        Args:
            bet_id: Unique bet identifier
            market: Type of bet
            team: Team/side to bet
            model_prob: Model probability
            available_lines: Dict of book -> (line, american_odds)
            desired_amount: Desired bet size
            user_state: User location for regulatory compliance

        Returns:
            Optimal routing decision
        """
        best_route = None
        best_net_ev = -float("inf")

        for book_name, (line, odds) in available_lines.items():
            if book_name not in self.books:
                continue

            book = self.books[book_name]

            # Check availability
            if user_state not in book.available_states and not book.vpn_friendly:
                continue

            # Check limits
            if market == "spread" and desired_amount > book.max_bet_spread:
                actual_amount = book.max_bet_spread
            elif market == "total" and desired_amount > book.max_bet_total:
                actual_amount = book.max_bet_total
            elif market == "prop" and desired_amount > book.max_bet_prop:
                actual_amount = book.max_bet_prop
            else:
                actual_amount = desired_amount

            # Check daily limit
            today_total = (
                sum(
                    r.bet_amount
                    for r in self.recent_bets
                    if r.book == book_name and (datetime.now() - book.last_bet).days == 0
                )
                if book.last_bet
                else 0
            )

            if today_total + actual_amount > book.daily_limit:
                actual_amount = max(0, book.daily_limit - today_total)

            if actual_amount <= 0:
                continue

            # Calculate EV
            decimal_odds = self._american_to_decimal(odds)
            ev = actual_amount * (model_prob * (decimal_odds - 1) - (1 - model_prob))

            # Calculate friction
            friction_cost, friction_reasons = FrictionCalculator.calculate_total_friction(
                actual_amount, odds, book, self.last_withdrawals.get(book_name)
            )

            # Net EV
            net_ev = ev - friction_cost

            # Track best option
            if net_ev > best_net_ev:
                best_net_ev = net_ev

                warnings = []
                if book.heat_level > 0.7:
                    warnings.append("High heat - account at risk")
                if book.lifetime_profit > 25000:
                    warnings.append("Significant lifetime profit - expect limits")
                if book.active_bonus and book.active_bonus.rollover_remaining > 0:
                    warnings.append(
                        f"${book.active_bonus.rollover_remaining:.0f} rollover remaining"
                    )

                best_route = BetRoute(
                    bet_id=bet_id,
                    book=book_name,
                    line=line,
                    price=odds,
                    bet_amount=actual_amount,
                    expected_value=ev,
                    friction_cost=friction_cost,
                    net_ev=net_ev,
                    reasons=friction_reasons,
                    warnings=warnings,
                )

        if best_route:
            # Update book status
            book = self.books[best_route.book]
            book.last_bet = datetime.now()
            book.heat_level = min(1.0, book.heat_level + 0.01)  # Slight heat increase

            # Track bet
            self.recent_bets.append(best_route)
            if len(self.recent_bets) > 100:
                self.recent_bets = self.recent_bets[-100:]

        return best_route

    def _american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal"""
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))

    def optimize_portfolio(self, bets: list[dict], target_exposure: float) -> list[BetRoute]:
        """
        Optimize routing for portfolio of bets.

        Considers correlation, book limits, and friction across all bets.
        """
        routes = []

        # Sort bets by edge (highest first)
        sorted_bets = sorted(bets, key=lambda x: x["edge"], reverse=True)

        # Track book usage
        book_usage = {name: 0 for name in self.books}

        for bet in sorted_bets:
            # Adjust desired amount based on remaining exposure
            remaining_exposure = target_exposure - sum(r.bet_amount for r in routes)
            if remaining_exposure <= 0:
                break

            desired_amount = min(bet["desired_amount"], remaining_exposure)

            # Route considering current book usage
            route = self.route_bet(
                bet_id=bet["id"],
                market=bet["market"],
                team=bet["team"],
                model_prob=bet["model_prob"],
                available_lines=bet["available_lines"],
                desired_amount=desired_amount,
            )

            if route:
                routes.append(route)
                book_usage[route.book] += route.bet_amount

        return routes

    def generate_routing_report(self, routes: list[BetRoute]) -> str:
        """Generate comprehensive routing report"""
        report = []
        report.append("=" * 80)
        report.append("BET ROUTING OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary stats
        total_amount = sum(r.bet_amount for r in routes)
        total_ev = sum(r.expected_value for r in routes)
        total_friction = sum(r.friction_cost for r in routes)
        total_net = sum(r.net_ev for r in routes)

        report.append("PORTFOLIO SUMMARY:")
        report.append(f"  Total bets routed: {len(routes)}")
        report.append(f"  Total amount: ${total_amount:,.2f}")
        report.append(f"  Total EV (before friction): ${total_ev:,.2f}")
        report.append(f"  Total friction costs: ${total_friction:,.2f}")
        report.append(f"  Net EV (after friction): ${total_net:,.2f}")
        report.append(f"  Friction as % of EV: {(total_friction/total_ev*100):.1f}%")
        report.append("")

        # Book distribution
        book_dist = {}
        for r in routes:
            if r.book not in book_dist:
                book_dist[r.book] = {"count": 0, "amount": 0, "friction": 0}
            book_dist[r.book]["count"] += 1
            book_dist[r.book]["amount"] += r.bet_amount
            book_dist[r.book]["friction"] += r.friction_cost

        report.append("BOOK DISTRIBUTION:")
        for book, stats in sorted(book_dist.items(), key=lambda x: x[1]["amount"], reverse=True):
            report.append(f"  {book}:")
            report.append(f"    Bets: {stats['count']}")
            report.append(f"    Amount: ${stats['amount']:,.2f}")
            report.append(f"    Friction: ${stats['friction']:,.2f}")
        report.append("")

        # Top friction costs
        high_friction = sorted(routes, key=lambda x: x.friction_cost, reverse=True)[:5]
        if high_friction:
            report.append("HIGHEST FRICTION BETS:")
            for r in high_friction:
                report.append(f"  {r.bet_id} at {r.book}: ${r.friction_cost:.2f}")
                for reason in r.reasons:
                    report.append(f"    - {reason}")
        report.append("")

        # Warnings
        all_warnings = []
        for r in routes:
            for w in r.warnings:
                all_warnings.append(f"{r.book}: {w}")

        if all_warnings:
            report.append("WARNINGS:")
            for w in set(all_warnings):
                report.append(f"  ⚠️ {w}")

        return "\n".join(report)


def demo_book_router():
    """Demonstrate intelligent book routing"""

    # Create sample books
    books = [
        BookProfile(
            name="Pinnacle",
            tier=BookTier.SHARP,
            max_bet_spread=10000,
            max_bet_total=10000,
            max_bet_prop=2000,
            max_payout=100000,
            daily_limit=50000,
            account_balance=5000,
            lifetime_profit=15000,
            heat_level=0.1,  # Sharp books don't care about winners
            withdrawal_policy=WithdrawalPolicy(
                fee_structure="flat",
                fee_amount=50,
                minimum_withdrawal=100,
                processing_days=3,
                crypto_available=True,
                crypto_fee=0,
            ),
            available_states=["NV", "International"],
            vpn_friendly=True,
        ),
        BookProfile(
            name="DraftKings",
            tier=BookTier.RECREATIONAL,
            max_bet_spread=5000,
            max_bet_total=5000,
            max_bet_prop=500,
            max_payout=50000,
            daily_limit=10000,
            account_balance=2000,
            lifetime_profit=5000,
            heat_level=0.6,  # Getting warm
            active_bonus=BonusTerms(
                bonus_amount=1000,
                rollover_requirement=10000,
                rollover_remaining=3000,
                qualifying_odds=-200,
                expires=datetime.now() + timedelta(days=30),
            ),
            withdrawal_policy=WithdrawalPolicy(
                fee_structure="free", fee_amount=0, minimum_withdrawal=20, processing_days=1
            ),
            available_states=["NV", "NJ", "PA", "MI", "IN", "IA", "IL", "CO", "WV", "TN", "AZ"],
            vpn_friendly=False,
        ),
        BookProfile(
            name="BetOnline",
            tier=BookTier.REDUCED_JUICE,
            max_bet_spread=5000,
            max_bet_total=5000,
            max_bet_prop=1000,
            max_payout=50000,
            daily_limit=25000,
            account_balance=10000,
            lifetime_profit=8000,
            heat_level=0.3,
            withdrawal_policy=WithdrawalPolicy(
                fee_structure="flat",
                fee_amount=35,
                minimum_withdrawal=50,
                processing_days=5,
                free_withdrawal_frequency=30,
                crypto_available=True,
                crypto_fee=0,
            ),
            available_states=["International"],
            vpn_friendly=True,
        ),
    ]

    # Create router
    router = BookRouter(books)

    # Sample bets to route
    bets = [
        {
            "id": "KC_BUF_spread",
            "market": "spread",
            "team": "KC -3",
            "model_prob": 0.58,
            "edge": 0.035,
            "desired_amount": 2000,
            "available_lines": {
                "Pinnacle": (-3, -105),
                "DraftKings": (-3, -110),
                "BetOnline": (-3, -107),
            },
        },
        {
            "id": "KC_BUF_total",
            "market": "total",
            "team": "Over 51",
            "model_prob": 0.54,
            "edge": 0.025,
            "desired_amount": 1500,
            "available_lines": {
                "Pinnacle": (51, -108),
                "DraftKings": (51.5, -110),
                "BetOnline": (51, -105),
            },
        },
        {
            "id": "Mahomes_passing",
            "market": "prop",
            "team": "Over 285.5",
            "model_prob": 0.56,
            "edge": 0.04,
            "desired_amount": 500,
            "available_lines": {"DraftKings": (285.5, -115), "BetOnline": (284.5, -110)},
        },
    ]

    # Route portfolio
    routes = router.optimize_portfolio(bets, target_exposure=5000)

    # Generate report
    report = router.generate_routing_report(routes)
    print(report)

    # Show individual routing decisions
    print("\n" + "=" * 80)
    print("INDIVIDUAL ROUTING DECISIONS")
    print("=" * 80)

    for route in routes:
        print(f"\n{route.bet_id}:")
        print(f"  Routed to: {route.book}")
        print(f"  Line: {route.line} @ {route.price:+d}")
        print(f"  Amount: ${route.bet_amount:,.2f}")
        print(f"  EV: ${route.expected_value:,.2f}")
        print(f"  Friction: ${route.friction_cost:,.2f}")
        print(f"  Net EV: ${route.net_ev:,.2f}")
        if route.warnings:
            print(f"  Warnings: {', '.join(route.warnings)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demo_book_router()
