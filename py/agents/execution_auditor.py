"""Post-execution reconciliation agent.

Wraps the Reconciler module for automated daily reconciliation checks.
Compares local fills vs exchange fills, detects discrepancies, and
generates audit trail entries.

Usage:
    from py.agents.execution_auditor import ExecutionAuditor, AuditorConfig

    auditor = ExecutionAuditor(client=kalshi_client)
    report = await auditor.run_audit(local_positions=positions, local_balance=balance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from py.execution.reconciliation import Reconciler, ReconciliationReport
from py.markets.base import Fill, MarketClient, Position
from py.monitoring.alert_manager import AlertLevel, AlertManager

logger = logging.getLogger(__name__)


@dataclass
class AuditorConfig:
    """Configuration for execution auditing."""

    balance_tolerance_cents: int = 100
    alert_on_any_discrepancy: bool = True
    auto_fix: bool = False


@dataclass
class AuditResult:
    """Result of a single reconciliation audit."""

    report: ReconciliationReport
    alerts_sent: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_discrepancies(self) -> bool:
        """True if any discrepancies were found."""
        return not self.report.clean

    @property
    def discrepancy_count(self) -> int:
        """Number of discrepancies found."""
        return len(self.report.discrepancies)

    def summary(self) -> str:
        """Human-readable audit summary."""
        lines = [
            f"Execution Audit — {self.timestamp:%Y-%m-%d %H:%M UTC}",
            f"{'=' * 50}",
        ]

        if not self.has_discrepancies:
            lines.append("Result: CLEAN — no discrepancies")
        else:
            lines.append(f"Discrepancies: {self.discrepancy_count}")
            lines.append(f"  Critical: {self.report.critical_count}")
            lines.append(f"  Alerts sent: {self.alerts_sent}")
            for d in self.report.discrepancies:
                lines.append(f"  [{d.severity.upper()}] {d.category}: {d.description}")

        lines.append(f"Positions checked: {self.report.positions_checked}")
        lines.append(f"Fills checked: {self.report.fills_checked}")

        return "\n".join(lines)


class ExecutionAuditor:
    """Automated execution reconciliation agent.

    Wraps Reconciler for scheduled or on-demand audit runs. Dispatches
    alerts for discrepancies and maintains audit history stats.
    """

    def __init__(
        self,
        client: MarketClient,
        alert_manager: AlertManager | None = None,
        config: AuditorConfig | None = None,
    ):
        self._client = client
        self._alert_manager = alert_manager
        self.config = config or AuditorConfig()

        # Stats
        self._total_audits = 0
        self._total_discrepancies = 0
        self._last_audit_timestamp: datetime | None = None

    async def run_audit(
        self,
        local_positions: list[Position] | None = None,
        local_balance_cents: int | None = None,
        local_fills: list[Fill] | None = None,
    ) -> AuditResult:
        """Run a full reconciliation audit.

        Args:
            local_positions: Current known positions to verify.
            local_balance_cents: Current known cash balance to verify.
            local_fills: Local fill records to verify against exchange.

        Returns:
            AuditResult with the reconciliation report and alert count.
        """
        self._total_audits += 1
        self._last_audit_timestamp = datetime.now(UTC)

        reconciler = Reconciler(
            client=self._client,
            local_positions=local_positions,
            local_balance_cents=local_balance_cents,
            local_fills=local_fills,
            balance_tolerance_cents=self.config.balance_tolerance_cents,
        )

        report = await reconciler.run(auto_fix=self.config.auto_fix)

        alerts_sent = 0
        self._total_discrepancies += len(report.discrepancies)

        # Send alerts for discrepancies
        if self._alert_manager is not None and report.discrepancies:
            if self.config.alert_on_any_discrepancy:
                for d in report.discrepancies:
                    alert_level = (
                        AlertLevel.CRITICAL
                        if d.severity == "critical"
                        else AlertLevel.WARNING
                    )
                    sent = self._alert_manager.alert(
                        alert_level,
                        f"Reconciliation: {d.category}",
                        d.description,
                        {
                            "contract_id": d.contract_id,
                            "local": d.local_value,
                            "exchange": d.exchange_value,
                        },
                    )
                    if sent is not None:
                        alerts_sent += 1

        if report.clean:
            logger.info("Execution audit CLEAN — no discrepancies")
        else:
            logger.warning(
                "Execution audit found %d discrepancies (%d critical)",
                len(report.discrepancies),
                report.critical_count,
            )

        return AuditResult(
            report=report,
            alerts_sent=alerts_sent,
        )

    def summary(self) -> dict:
        """Get auditor summary statistics."""
        return {
            "total_audits": self._total_audits,
            "total_discrepancies": self._total_discrepancies,
            "last_audit_timestamp": (
                self._last_audit_timestamp.isoformat()
                if self._last_audit_timestamp
                else None
            ),
        }
