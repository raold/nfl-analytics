"""
Trading CLI for the prediction market system.

Provides commands to start/stop paper and live trading sessions,
check status, and manage positions.

Usage:
    python -m nfl_analytics.cli.trading start --mode paper
    python -m nfl_analytics.cli.trading stop
    python -m nfl_analytics.cli.trading status
    python -m nfl_analytics.cli.trading positions
    python -m nfl_analytics.cli.trading orders --status open
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# PID file for tracking running sessions
PID_FILE = Path.home() / ".cache" / "nfl-analytics" / "trading.pid"
STATE_FILE = Path.home() / ".cache" / "nfl-analytics" / "trading_state.json"


def _ensure_cache_dir():
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_pid() -> int | None:
    """Read the PID of a running trading session."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is actually running
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _write_state(mode: str, started_at: str, **kwargs):
    """Write session state to disk."""
    _ensure_cache_dir()
    state = {"mode": mode, "started_at": started_at, "pid": os.getpid(), **kwargs}
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _read_state() -> dict | None:
    """Read session state from disk."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """NFL Analytics — Prediction Market Trading System."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["paper", "live"]),
    default="paper",
    help="Trading mode (default: paper)",
)
@click.option(
    "--balance",
    type=int,
    default=100_000,
    help="Initial balance in cents (default: 100000 = $1,000)",
)
@click.option(
    "--sport",
    type=click.Choice(["nfl", "nba", "mlb", "nhl"]),
    default="nfl",
    help="Sport to trade (default: nfl)",
)
def start(mode: str, balance: int, sport: str):
    """Start a trading session."""
    existing_pid = _read_pid()
    if existing_pid:
        console.print(f"[yellow]Trading session already running (PID {existing_pid})[/yellow]")
        console.print("Use [bold]stop[/bold] to terminate the existing session first.")
        raise SystemExit(1)

    if mode == "live":
        console.print("[bold red]LIVE TRADING[/bold red] — Real money at risk!")
        if not click.confirm("Are you sure you want to start live trading?"):
            console.print("Aborted.")
            raise SystemExit(0)

        # Check for Kalshi API credentials
        if not os.environ.get("KALSHI_API_KEY"):
            console.print("[red]Error: KALSHI_API_KEY environment variable not set[/red]")
            raise SystemExit(1)

    _ensure_cache_dir()
    PID_FILE.write_text(str(os.getpid()))

    started_at = datetime.now(UTC).isoformat()
    _write_state(mode=mode, started_at=started_at, sport=sport, balance_cents=balance)

    console.print(f"[bold green]Trading session started[/bold green]")
    console.print(f"  Mode:    [cyan]{mode}[/cyan]")
    console.print(f"  Sport:   [cyan]{sport}[/cyan]")
    console.print(f"  Balance: [cyan]${balance / 100:.2f}[/cyan]")
    console.print(f"  PID:     {os.getpid()}")

    if mode == "paper":
        console.print("\n[dim]Paper trading — no real money at risk.[/dim]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    # Run the trading loop
    try:
        asyncio.run(_run_trading_loop(mode, balance, sport))
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    finally:
        PID_FILE.unlink(missing_ok=True)
        console.print("[green]Trading session stopped.[/green]")


async def _run_trading_loop(mode: str, balance_cents: int, sport: str):
    """Main trading loop (placeholder — wired up in Phase 2+)."""
    from nfl_analytics.markets.paper_client import PaperMarketClient

    if mode == "paper":
        client = PaperMarketClient(initial_balance_cents=balance_cents)
        console.print("[dim]Paper client initialized. Waiting for market data...[/dim]")
    else:
        # Phase 3: KalshiClient will be instantiated here
        console.print("[yellow]Live client not yet implemented (Phase 3)[/yellow]")
        return

    # Placeholder: heartbeat loop until strategies are wired
    iteration = 0
    while True:
        await asyncio.sleep(30)
        iteration += 1
        summary = client.summary()
        logger.debug(
            "Heartbeat #%d: balance=$%.2f positions=%d fills=%d",
            iteration,
            summary["balance_dollars"],
            summary["open_positions"],
            summary["total_fills"],
        )


@cli.command()
def stop():
    """Stop a running trading session."""
    pid = _read_pid()
    if pid is None:
        console.print("[yellow]No trading session is running.[/yellow]")
        raise SystemExit(0)

    console.print(f"Stopping trading session (PID {pid})...")
    try:
        os.kill(pid, signal.SIGINT)
        console.print("[green]Stop signal sent.[/green]")
    except ProcessLookupError:
        console.print("[yellow]Process already terminated.[/yellow]")
        PID_FILE.unlink(missing_ok=True)


@cli.command()
def status():
    """Show current trading session status."""
    pid = _read_pid()
    state = _read_state()

    if pid is None:
        console.print("[dim]No active trading session.[/dim]")
        if state:
            console.print(f"\nLast session: {state.get('mode', '?')} mode")
            console.print(f"Started: {state.get('started_at', '?')}")
        raise SystemExit(0)

    table = Table(title="Trading Session Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", "[bold green]RUNNING[/bold green]")
    table.add_row("PID", str(pid))

    if state:
        table.add_row("Mode", state.get("mode", "?"))
        table.add_row("Sport", state.get("sport", "?"))
        table.add_row("Balance", f"${state.get('balance_cents', 0) / 100:.2f}")
        table.add_row("Started", state.get("started_at", "?"))

    console.print(table)


@cli.command()
def positions():
    """Show open positions (reads from database)."""
    console.print("[dim]Position display requires database connection (Phase 2+)[/dim]")
    console.print("Use [bold]status[/bold] for session info.")


@cli.command()
@click.option(
    "--status-filter",
    "status_filter",
    type=click.Choice(["open", "filled", "cancelled", "all"]),
    default="open",
    help="Filter orders by status",
)
def orders(status_filter: str):
    """Show orders (reads from database)."""
    console.print("[dim]Order display requires database connection (Phase 2+)[/dim]")
    console.print("Use [bold]status[/bold] for session info.")


def main():
    cli()


if __name__ == "__main__":
    main()
