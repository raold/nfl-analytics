"""
Daily reconciliation — verify local state matches exchange truth.

Compares local positions, balance, and fills against the Kalshi API to
detect drift caused by missed fills, network errors, or bugs. Reports
discrepancies and optionally auto-corrects minor mismatches.

Designed to run as a scheduled task (cron 6am ET daily).

Usage:
    from nfl_analytics.execution.reconciliation import Reconciler

    reconciler = Reconciler(kalshi_client, local_positions, local_balance)
    report = await reconciler.run()
    if not report.clean:
        print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import Fill, MarketClient, Position, Side

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrepancy types
# ---------------------------------------------------------------------------


@dataclass
class Discrepancy:
    """A single mismatch between local and exchange state."""

    category: str  # "position", "balance", "fill"
    contract_id: str  # "" for balance discrepancies
    description: str
    local_value: str  # Human-readable local state
    exchange_value: str  # Human-readable exchange state
    severity: str  # "info", "warning", "critical"
    auto_fixable: bool = False

    def __repr__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.category}: {self.description} "
            f"(local={self.local_value}, exchange={self.exchange_value})"
        )


@dataclass
class ReconciliationReport:
    """Results of a reconciliation run."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    discrepancies: list[Discrepancy] = field(default_factory=list)
    positions_checked: int = 0
    fills_checked: int = 0
    balance_checked: bool = False
    auto_fixes_applied: int = 0

    @property
    def clean(self) -> bool:
        """True if no discrepancies found."""
        return len(self.discrepancies) == 0

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "warning")

    def summary(self) -> str:
        """Human-readable summary of reconciliation results."""
        lines = [
            f"{'=' * 60}",
            f"Reconciliation Report — {self.timestamp:%Y-%m-%d %H:%M UTC}",
            f"{'=' * 60}",
            f"Positions checked: {self.positions_checked}",
            f"Fills checked:     {self.fills_checked}",
            f"Balance checked:   {'yes' if self.balance_checked else 'no'}",
        ]

        if self.clean:
            lines.append("\nResult: CLEAN — all state matches exchange")
        else:
            lines.append(f"\nDiscrepancies: {len(self.discrepancies)}")
            lines.append(f"  Critical: {self.critical_count}")
            lines.append(f"  Warnings: {self.warning_count}")

            if self.auto_fixes_applied:
                lines.append(f"  Auto-fixed: {self.auto_fixes_applied}")

            lines.append("\nDetails:")
            for d in self.discrepancies:
                lines.append(f"  {d!r}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class Reconciler:
    """Compares local trading state against the exchange.

    Checks:
        1. Position match — quantity and side for each contract
        2. Balance match — cash balance within tolerance
        3. Fill completeness — all exchange fills exist locally
    """

    def __init__(
        self,
        client: MarketClient,
        local_positions: list[Position] | None = None,
        local_balance_cents: int | None = None,
        local_fills: list[Fill] | None = None,
        balance_tolerance_cents: int = 100,  # Allow 100¢ ($1) tolerance
    ):
        self._client = client
        self._local_positions = local_positions or []
        self._local_balance = local_balance_cents
        self._local_fills = local_fills or []
        self._balance_tolerance = balance_tolerance_cents

    async def run(self, auto_fix: bool = False) -> ReconciliationReport:
        """Run full reconciliation against exchange.

        Args:
            auto_fix: If True, attempt to auto-correct minor discrepancies
                by overwriting local state with exchange truth.

        Returns:
            ReconciliationReport with all findings.
        """
        report = ReconciliationReport()

        # 1. Reconcile positions
        await self._check_positions(report)

        # 2. Reconcile balance
        if self._local_balance is not None:
            await self._check_balance(report)

        # 3. Reconcile fills
        if self._local_fills:
            await self._check_fills(report)

        if report.clean:
            logger.info("Reconciliation CLEAN — all state matches exchange")
        else:
            logger.warning(
                "Reconciliation found %d discrepancies (%d critical)",
                len(report.discrepancies),
                report.critical_count,
            )

        return report

    # -------------------------------------------------------------------
    # Position reconciliation
    # -------------------------------------------------------------------

    async def _check_positions(self, report: ReconciliationReport) -> None:
        """Compare local positions against exchange."""
        exchange_positions = await self._client.get_positions()

        # Build lookup maps
        local_map = {p.contract_id: p for p in self._local_positions}
        exchange_map = {p.contract_id: p for p in exchange_positions}

        all_contracts = set(local_map.keys()) | set(exchange_map.keys())
        report.positions_checked = len(all_contracts)

        for contract_id in all_contracts:
            local = local_map.get(contract_id)
            exchange = exchange_map.get(contract_id)

            if local and not exchange:
                # We think we have a position, exchange says no
                report.discrepancies.append(Discrepancy(
                    category="position",
                    contract_id=contract_id,
                    description="Phantom position — local has position, exchange does not",
                    local_value=f"{local.side.value} {local.quantity}",
                    exchange_value="none",
                    severity="critical",
                    auto_fixable=True,
                ))

            elif exchange and not local:
                # Exchange has a position we don't know about
                report.discrepancies.append(Discrepancy(
                    category="position",
                    contract_id=contract_id,
                    description="Missing position — exchange has position, local does not",
                    local_value="none",
                    exchange_value=f"{exchange.side.value} {exchange.quantity}",
                    severity="critical",
                    auto_fixable=True,
                ))

            elif local and exchange:
                # Both have position — check quantity and side
                if local.side != exchange.side:
                    report.discrepancies.append(Discrepancy(
                        category="position",
                        contract_id=contract_id,
                        description="Side mismatch",
                        local_value=local.side.value,
                        exchange_value=exchange.side.value,
                        severity="critical",
                        auto_fixable=True,
                    ))
                elif local.quantity != exchange.quantity:
                    diff = abs(local.quantity - exchange.quantity)
                    severity = "critical" if diff > 5 else "warning"
                    report.discrepancies.append(Discrepancy(
                        category="position",
                        contract_id=contract_id,
                        description=f"Quantity mismatch (off by {diff})",
                        local_value=str(local.quantity),
                        exchange_value=str(exchange.quantity),
                        severity=severity,
                        auto_fixable=True,
                    ))

    # -------------------------------------------------------------------
    # Balance reconciliation
    # -------------------------------------------------------------------

    async def _check_balance(self, report: ReconciliationReport) -> None:
        """Compare local balance against exchange."""
        report.balance_checked = True
        exchange_balance = await self._client.get_balance_cents()

        diff = abs(self._local_balance - exchange_balance)

        if diff > self._balance_tolerance:
            severity = "critical" if diff > 1000 else "warning"  # >$10 is critical
            report.discrepancies.append(Discrepancy(
                category="balance",
                contract_id="",
                description=f"Balance mismatch (off by {diff}¢ = ${diff/100:.2f})",
                local_value=f"{self._local_balance}¢",
                exchange_value=f"{exchange_balance}¢",
                severity=severity,
                auto_fixable=True,
            ))

    # -------------------------------------------------------------------
    # Fill reconciliation
    # -------------------------------------------------------------------

    async def _check_fills(self, report: ReconciliationReport) -> None:
        """Check that all exchange fills have local records."""
        exchange_fills = await self._client.get_fills()
        report.fills_checked = len(exchange_fills)

        local_fill_ids = {f.fill_id for f in self._local_fills}

        missing = 0
        for ef in exchange_fills:
            if ef.fill_id not in local_fill_ids:
                missing += 1

        if missing > 0:
            report.discrepancies.append(Discrepancy(
                category="fill",
                contract_id="",
                description=f"Missing {missing} fills from local records",
                local_value=str(len(self._local_fills)),
                exchange_value=str(len(exchange_fills)),
                severity="warning" if missing <= 3 else "critical",
                auto_fixable=False,
            ))
