"""Tests for post-execution reconciliation auditor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from py.agents.execution_auditor import AuditorConfig, AuditResult, ExecutionAuditor
from py.execution.reconciliation import ReconciliationReport
from py.markets.base import Fill, MarketClient, Position, Side
from py.monitoring.alert_manager import AlertManager

pytestmark = pytest.mark.anyio


# ============================================================
# Helpers
# ============================================================


def _mock_client(
    positions: list[Position] | None = None,
    balance_cents: int = 50_000,
    fills: list[Fill] | None = None,
) -> AsyncMock:
    """Create a mock MarketClient with preconfigured responses."""
    client = AsyncMock(spec=MarketClient)
    client.get_positions.return_value = positions or []
    client.get_balance_cents.return_value = balance_cents
    client.get_fills.return_value = fills or []
    return client


def _position(
    contract_id: str = "C-1",
    side: Side = Side.YES,
    quantity: int = 10,
) -> Position:
    return Position(
        contract_id=contract_id,
        side=side,
        quantity=quantity,
        avg_entry_cents=50,
    )


def _fill(
    fill_id: str = "f-1",
    order_id: str = "o-1",
    contract_id: str = "C-1",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        contract_id=contract_id,
        side=Side.YES,
        price_cents=50,
        quantity=5,
    )


# ============================================================
# Clean audit
# ============================================================


class TestCleanAudit:
    """No discrepancies when local matches exchange."""

    async def test_clean_audit(self):
        """Matching positions → clean audit."""
        positions = [_position("C-1"), _position("C-2")]
        client = _mock_client(positions=positions)

        auditor = ExecutionAuditor(client=client)
        result = await auditor.run_audit(local_positions=positions)

        assert not result.has_discrepancies
        assert result.discrepancy_count == 0

    async def test_clean_audit_summary(self):
        """Clean audit summary says CLEAN."""
        client = _mock_client()
        auditor = ExecutionAuditor(client=client)
        result = await auditor.run_audit(local_positions=[])
        summary = result.summary()
        assert "CLEAN" in summary


# ============================================================
# Discrepancy detection
# ============================================================


class TestDiscrepancyDetection:
    """Detects mismatches between local and exchange state."""

    async def test_position_discrepancy(self):
        """Local has position, exchange doesn't → discrepancy."""
        client = _mock_client(positions=[])  # Exchange has none
        auditor = ExecutionAuditor(client=client)
        result = await auditor.run_audit(
            local_positions=[_position("C-1")],
        )
        assert result.has_discrepancies
        assert result.discrepancy_count >= 1

    async def test_balance_discrepancy(self):
        """Balance mismatch → discrepancy."""
        client = _mock_client(balance_cents=50_000)
        auditor = ExecutionAuditor(client=client)
        result = await auditor.run_audit(
            local_positions=[],
            local_balance_cents=40_000,  # Off by $100
        )
        assert result.has_discrepancies

    async def test_discrepancy_summary_shows_details(self):
        """Audit result summary includes discrepancy details."""
        client = _mock_client(positions=[])
        auditor = ExecutionAuditor(client=client)
        result = await auditor.run_audit(
            local_positions=[_position("C-1")],
        )
        summary = result.summary()
        assert "Discrepancies:" in summary
        assert "Critical:" in summary


# ============================================================
# Alert dispatch
# ============================================================


class TestAlertDispatch:
    """Alerts sent via AlertManager on discrepancy."""

    async def test_alert_dispatched_on_discrepancy(self):
        """AlertManager receives alerts when discrepancies found."""
        client = _mock_client(positions=[])
        alert_mgr = AlertManager()
        auditor = ExecutionAuditor(client=client, alert_manager=alert_mgr)
        result = await auditor.run_audit(
            local_positions=[_position("C-1")],
        )
        assert result.alerts_sent >= 1
        assert alert_mgr._total_alerts >= 1

    async def test_no_alert_without_manager(self):
        """No alerts dispatched when no AlertManager configured."""
        client = _mock_client(positions=[])
        auditor = ExecutionAuditor(client=client)  # No alert_manager
        result = await auditor.run_audit(
            local_positions=[_position("C-1")],
        )
        assert result.alerts_sent == 0


# ============================================================
# Auto-fix
# ============================================================


class TestAutoFix:
    """Auto-fix flag passed to Reconciler."""

    async def test_auto_fix_passed(self):
        """auto_fix=True in config is forwarded to Reconciler."""
        client = _mock_client(positions=[])
        config = AuditorConfig(auto_fix=True)
        auditor = ExecutionAuditor(client=client, config=config)
        # Just verify it runs without error — auto_fix is passed through
        result = await auditor.run_audit(local_positions=[])
        assert not result.has_discrepancies


# ============================================================
# Summary stats
# ============================================================


class TestSummaryStats:
    """Auditor tracks cumulative statistics."""

    async def test_summary_counts(self):
        """Summary reflects audit count and discrepancies."""
        client = _mock_client(positions=[])
        auditor = ExecutionAuditor(client=client)
        await auditor.run_audit(local_positions=[_position("C-1")])
        await auditor.run_audit(local_positions=[])

        s = auditor.summary()
        assert s["total_audits"] == 2
        assert s["total_discrepancies"] >= 1
        assert s["last_audit_timestamp"] is not None

    async def test_multiple_audits_accumulate(self):
        """Discrepancy count accumulates across audits."""
        client = _mock_client(positions=[])
        auditor = ExecutionAuditor(client=client)
        await auditor.run_audit(local_positions=[_position("C-1")])
        await auditor.run_audit(local_positions=[_position("C-2")])

        s = auditor.summary()
        assert s["total_audits"] == 2
        assert s["total_discrepancies"] >= 2


# ============================================================
# AuditResult properties
# ============================================================


class TestAuditResultProperties:
    """AuditResult dataclass properties."""

    def test_clean_result(self):
        """Clean report → no discrepancies."""
        report = ReconciliationReport()
        result = AuditResult(report=report)
        assert not result.has_discrepancies
        assert result.discrepancy_count == 0

    def test_dirty_result(self):
        """Report with discrepancies → has_discrepancies True."""
        from py.execution.reconciliation import Discrepancy

        report = ReconciliationReport()
        report.discrepancies = [
            Discrepancy("position", "C-1", "test", "a", "b", "critical"),
        ]
        result = AuditResult(report=report)
        assert result.has_discrepancies
        assert result.discrepancy_count == 1
