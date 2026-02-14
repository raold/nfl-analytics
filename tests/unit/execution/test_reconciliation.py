"""Tests for daily reconciliation — local state vs exchange verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from py.markets.base import Fill, MarketClient, Position, Side
from py.execution.reconciliation import Reconciler, ReconciliationReport

pytestmark = pytest.mark.anyio


# ============================================================
# Helpers
# ============================================================


def _mock_client(
    positions: list[Position] | None = None,
    balance_cents: int = 50000,
    fills: list[Fill] | None = None,
) -> AsyncMock:
    """Create a mock MarketClient with preconfigured responses."""
    client = AsyncMock(spec=MarketClient)
    client.get_positions.return_value = positions or []
    client.get_balance_cents.return_value = balance_cents
    client.get_fills.return_value = fills or []
    return client


def _position(contract_id: str = "C-1", side: Side = Side.YES, quantity: int = 10) -> Position:
    return Position(contract_id=contract_id, side=side, quantity=quantity, avg_entry_cents=50)


def _fill(fill_id: str = "f-1", order_id: str = "o-1", contract_id: str = "C-1") -> Fill:
    return Fill(fill_id=fill_id, order_id=order_id, contract_id=contract_id, side=Side.YES, price_cents=50, quantity=5)


# ============================================================
# Clean reconciliation
# ============================================================


class TestCleanReconciliation:
    """No discrepancies when local matches exchange."""

    async def test_positions_match(self):
        """Identical positions → clean."""
        positions = [_position("C-1", Side.YES, 10), _position("C-2", Side.NO, 5)]
        client = _mock_client(positions=positions)

        recon = Reconciler(
            client,
            local_positions=positions,
            local_balance_cents=50000,
        )
        report = await recon.run()

        assert report.clean
        assert report.positions_checked == 2

    async def test_balance_matches(self):
        """Balance within tolerance → clean."""
        client = _mock_client(balance_cents=50000)

        recon = Reconciler(
            client,
            local_positions=[],
            local_balance_cents=50050,  # Within 100¢ tolerance
            balance_tolerance_cents=100,
        )
        report = await recon.run()

        assert report.clean
        assert report.balance_checked

    async def test_fills_match(self):
        """All exchange fills have local records → clean."""
        fills = [_fill("f-1"), _fill("f-2")]
        client = _mock_client(fills=fills)

        recon = Reconciler(
            client,
            local_fills=fills,
        )
        report = await recon.run()

        assert report.clean


# ============================================================
# Position discrepancies
# ============================================================


class TestPositionDiscrepancies:
    """Detect position mismatches."""

    async def test_phantom_position(self):
        """Local has position, exchange doesn't → critical."""
        client = _mock_client(positions=[])  # Exchange says no positions

        recon = Reconciler(
            client,
            local_positions=[_position("C-1", Side.YES, 10)],
        )
        report = await recon.run()

        assert not report.clean
        assert report.critical_count == 1
        assert "Phantom" in report.discrepancies[0].description

    async def test_missing_position(self):
        """Exchange has position, local doesn't → critical."""
        exchange_positions = [_position("C-1", Side.YES, 10)]
        client = _mock_client(positions=exchange_positions)

        recon = Reconciler(
            client,
            local_positions=[],
        )
        report = await recon.run()

        assert not report.clean
        assert report.critical_count == 1
        assert "Missing" in report.discrepancies[0].description

    async def test_quantity_mismatch_small(self):
        """Small quantity mismatch → warning."""
        exchange_pos = [_position("C-1", Side.YES, 12)]
        local_pos = [_position("C-1", Side.YES, 10)]
        client = _mock_client(positions=exchange_pos)

        recon = Reconciler(client, local_positions=local_pos)
        report = await recon.run()

        assert not report.clean
        assert report.warning_count == 1
        assert "mismatch" in report.discrepancies[0].description.lower()

    async def test_quantity_mismatch_large(self):
        """Large quantity mismatch → critical."""
        exchange_pos = [_position("C-1", Side.YES, 50)]
        local_pos = [_position("C-1", Side.YES, 10)]
        client = _mock_client(positions=exchange_pos)

        recon = Reconciler(client, local_positions=local_pos)
        report = await recon.run()

        assert not report.clean
        assert report.critical_count == 1

    async def test_side_mismatch(self):
        """Side mismatch → critical."""
        exchange_pos = [_position("C-1", Side.NO, 10)]
        local_pos = [_position("C-1", Side.YES, 10)]
        client = _mock_client(positions=exchange_pos)

        recon = Reconciler(client, local_positions=local_pos)
        report = await recon.run()

        assert not report.clean
        assert report.critical_count == 1
        assert "Side" in report.discrepancies[0].description


# ============================================================
# Balance discrepancies
# ============================================================


class TestBalanceDiscrepancies:
    """Detect balance mismatches."""

    async def test_balance_mismatch_warning(self):
        """Balance off by moderate amount → warning."""
        client = _mock_client(balance_cents=50000)

        recon = Reconciler(
            client,
            local_positions=[],
            local_balance_cents=49500,  # Off by 500¢
            balance_tolerance_cents=100,
        )
        report = await recon.run()

        assert not report.clean
        assert report.warning_count == 1

    async def test_balance_mismatch_critical(self):
        """Balance off by large amount → critical."""
        client = _mock_client(balance_cents=50000)

        recon = Reconciler(
            client,
            local_positions=[],
            local_balance_cents=45000,  # Off by 5000¢ ($50)
            balance_tolerance_cents=100,
        )
        report = await recon.run()

        assert not report.clean
        assert report.critical_count == 1

    async def test_no_balance_check_when_not_provided(self):
        """Skip balance check when local balance not provided."""
        client = _mock_client()

        recon = Reconciler(client, local_positions=[])
        report = await recon.run()

        assert not report.balance_checked


# ============================================================
# Fill discrepancies
# ============================================================


class TestFillDiscrepancies:
    """Detect missing fills."""

    async def test_missing_fills(self):
        """Exchange has fills not in local records."""
        exchange_fills = [_fill("f-1"), _fill("f-2"), _fill("f-3")]
        local_fills = [_fill("f-1")]  # Missing f-2 and f-3
        client = _mock_client(fills=exchange_fills)

        recon = Reconciler(client, local_fills=local_fills)
        report = await recon.run()

        assert not report.clean
        assert any("Missing 2 fills" in d.description for d in report.discrepancies)


# ============================================================
# Report formatting
# ============================================================


class TestReportFormatting:
    """ReconciliationReport summary output."""

    async def test_clean_summary(self):
        """Clean report shows CLEAN."""
        client = _mock_client()
        recon = Reconciler(client, local_positions=[])
        report = await recon.run()

        summary = report.summary()
        assert "CLEAN" in summary

    async def test_dirty_summary(self):
        """Report with issues shows discrepancy count."""
        client = _mock_client(positions=[])

        recon = Reconciler(
            client,
            local_positions=[_position("C-1")],
        )
        report = await recon.run()

        summary = report.summary()
        assert "Discrepancies:" in summary
        assert "Critical:" in summary

    def test_report_properties(self):
        """Report properties compute correctly."""
        from py.execution.reconciliation import Discrepancy

        report = ReconciliationReport()
        report.discrepancies = [
            Discrepancy("position", "C-1", "test", "a", "b", "critical"),
            Discrepancy("position", "C-2", "test", "a", "b", "warning"),
            Discrepancy("balance", "", "test", "a", "b", "critical"),
        ]

        assert not report.clean
        assert report.critical_count == 2
        assert report.warning_count == 1
