#!/usr/bin/env python3
"""
Multi-Book Odds Aggregator and Line Shopping System.

Aggregates odds from multiple Virginia sportsbooks to find best available lines.
Supports manual entry, CSV import, and API integration (when available).

Features:
- Multi-book odds comparison (spread, total, moneyline)
- Best line identification by bet type
- EV calculation with line shopping gains
- Historical odds tracking
- API integration framework (The Odds API, SportsDataIO)

Usage:
    # Manual odds entry
    python py/production/line_shopping.py \
        --game "KC_vs_BUF" \
        --manual-odds fanduel:-3/-110 draftkings:-3/-108 betmgm:-2.5/-115

    # CSV import
    python py/production/line_shopping.py \
        --odds-csv data/odds/week_11_odds.csv \
        --output results/line_shopping/best_lines.json

    # API mode (requires API key)
    python py/production/line_shopping.py \
        --api theoddsapi \
        --api-key YOUR_KEY \
        --sport americanfootball_nfl
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class OddsLine:
    """Single odds line from a sportsbook."""

    book: str
    game_id: str
    bet_type: str  # 'spread', 'total', 'moneyline'
    side: str  # 'home', 'away', 'over', 'under'
    line: float | None  # spread/total value (None for moneyline)
    odds: int  # American odds
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    def get_implied_prob(self) -> float:
        """Convert American odds to implied probability."""
        if self.odds > 0:
            return 100.0 / (self.odds + 100.0)
        else:
            return abs(self.odds) / (abs(self.odds) + 100.0)

    def get_decimal_odds(self) -> float:
        """Convert American odds to decimal odds (payout per $1)."""
        if self.odds > 0:
            return self.odds / 100.0
        else:
            return 100.0 / abs(self.odds)


@dataclass
class BestLine:
    """Best available line across all books."""

    game_id: str
    bet_type: str
    side: str
    best_book: str
    best_line: float | None
    best_odds: int
    consensus_odds: int  # Average odds across books
    improvement_cents: float  # Cents better than consensus
    ev_gain: float  # Expected value gain ($) on $100 bet
    all_books: list[OddsLine]

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "side": self.side,
            "best_book": self.best_book,
            "best_line": self.best_line,
            "best_odds": self.best_odds,
            "consensus_odds": self.consensus_odds,
            "improvement_cents": self.improvement_cents,
            "ev_gain": self.ev_gain,
            "all_books": [b.to_dict() for b in self.all_books],
        }


# ============================================================================
# Odds Aggregator
# ============================================================================


class OddsAggregator:
    """
    Aggregate odds from multiple sportsbooks and find best lines.
    """

    def __init__(self, books: list[str] = None):
        """
        Initialize odds aggregator.

        Args:
            books: List of sportsbook names to track
        """
        self.books = books or [
            "fanduel",
            "draftkings",
            "betmgm",
            "caesars",
            "espnbet",
            "bet365",
            "circa",
            "fanatics",
        ]
        self.odds_data: dict[str, list[OddsLine]] = {}

    def add_odds_line(
        self,
        book: str,
        game_id: str,
        bet_type: str,
        side: str,
        line: float | None,
        odds: int,
    ):
        """Add a single odds line to the aggregator."""
        odds_line = OddsLine(
            book=book,
            game_id=game_id,
            bet_type=bet_type,
            side=side,
            line=line,
            odds=odds,
            timestamp=datetime.now().isoformat(),
        )

        key = f"{game_id}_{bet_type}_{side}"
        if key not in self.odds_data:
            self.odds_data[key] = []

        self.odds_data[key].append(odds_line)

    def parse_manual_odds(self, odds_str: str) -> tuple[str, float, int]:
        """
        Parse manual odds string.

        Format examples:
            "fanduel:-3/-110"  (spread)
            "draftkings:52.5/-110"  (total)
            "betmgm:-150"  (moneyline)

        Returns:
            (book, line, odds)
        """
        parts = odds_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid format: {odds_str}. Use 'book:line/odds'")

        book = parts[0].lower()

        # Parse line and odds
        if "/" in parts[1]:
            line_str, odds_str = parts[1].split("/")
            line = float(line_str)
            odds = int(odds_str)
        else:
            # Moneyline only (no line)
            line = None
            odds = int(parts[1])

        return book, line, odds

    def load_from_csv(self, csv_path: str):
        """
        Load odds from CSV.

        CSV format:
            game_id,book,bet_type,side,line,odds
            KC_vs_BUF,fanduel,spread,KC,-3,-110
            KC_vs_BUF,draftkings,spread,KC,-3,-108
        """
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            self.add_odds_line(
                book=row["book"],
                game_id=row["game_id"],
                bet_type=row["bet_type"],
                side=row["side"],
                line=row["line"] if pd.notna(row.get("line")) else None,
                odds=int(row["odds"]),
            )

    def get_best_line(
        self,
        game_id: str,
        bet_type: str,
        side: str,
    ) -> BestLine | None:
        """
        Find best available line for a specific bet.

        Args:
            game_id: Game identifier
            bet_type: 'spread', 'total', or 'moneyline'
            side: 'home', 'away', 'over', 'under', or team name

        Returns:
            BestLine with best odds and comparison data
        """
        key = f"{game_id}_{bet_type}_{side}"

        if key not in self.odds_data or len(self.odds_data[key]) == 0:
            return None

        lines = self.odds_data[key]

        # Find best odds (highest for + odds, closest to 0 for - odds)
        best_line = max(lines, key=lambda x: x.odds)

        # Calculate consensus (average odds)
        avg_odds = np.mean([line.odds for line in lines])
        consensus_odds = int(avg_odds)

        # Calculate improvement in cents
        # "Cents" = difference in implied probability
        best_prob = best_line.get_implied_prob()
        consensus_prob = american_odds_to_prob(consensus_odds)
        improvement_cents = (consensus_prob - best_prob) * 100  # Convert to cents

        # Calculate EV gain on $100 bet
        # EV = (win_prob * payout) - (loss_prob * stake)
        # Gain = EV_best - EV_consensus (assuming same win prob)
        win_prob = 0.5  # Simplified assumption

        best_decimal = best_line.get_decimal_odds()
        consensus_decimal = american_odds_to_decimal(consensus_odds)

        ev_best = (win_prob * best_decimal * 100) - ((1 - win_prob) * 100)
        ev_consensus = (win_prob * consensus_decimal * 100) - ((1 - win_prob) * 100)
        ev_gain = ev_best - ev_consensus

        return BestLine(
            game_id=game_id,
            bet_type=bet_type,
            side=side,
            best_book=best_line.book,
            best_line=best_line.line,
            best_odds=best_line.odds,
            consensus_odds=consensus_odds,
            improvement_cents=improvement_cents,
            ev_gain=ev_gain,
            all_books=sorted(lines, key=lambda x: x.odds, reverse=True),
        )

    def get_all_best_lines(self) -> list[BestLine]:
        """Get best lines for all tracked bets."""
        best_lines = []

        # Deduplicate keys
        unique_bets = set()
        for key in self.odds_data.keys():
            parts = key.rsplit("_", 2)
            if len(parts) == 3:
                game_id, bet_type, side = parts
                unique_bets.add((game_id, bet_type, side))

        for game_id, bet_type, side in unique_bets:
            best_line = self.get_best_line(game_id, bet_type, side)
            if best_line:
                best_lines.append(best_line)

        return best_lines

    def compare_books(self) -> pd.DataFrame:
        """
        Compare books across all lines.

        Returns:
            DataFrame with book performance metrics
        """
        book_stats = {book: {"best_count": 0, "total_lines": 0} for book in self.books}

        for best_line in self.get_all_best_lines():
            # Count how many times each book had the best line
            book_stats[best_line.best_book]["best_count"] += 1

            # Count total lines per book
            for odds_line in best_line.all_books:
                if odds_line.book in book_stats:
                    book_stats[odds_line.book]["total_lines"] += 1

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "book": book,
                    "best_line_count": stats["best_count"],
                    "total_lines": stats["total_lines"],
                    "best_line_pct": (
                        (stats["best_count"] / stats["total_lines"] * 100)
                        if stats["total_lines"] > 0
                        else 0
                    ),
                }
                for book, stats in book_stats.items()
            ]
        )

        return df.sort_values("best_line_pct", ascending=False)


# ============================================================================
# Utility Functions
# ============================================================================


def american_odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def american_odds_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def decimal_to_american_odds(decimal: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal >= 1.0:
        return int(decimal * 100)
    else:
        return int(-100 / decimal)


# ============================================================================
# API Integration (Future)
# ============================================================================


class TheOddsAPIClient:
    """
    Client for The Odds API (theoddsapi.com).

    Free tier: 500 requests/month
    Paid tier: $25/month for 10,000 requests
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"

    def get_odds(
        self,
        sport: str = "americanfootball_nfl",
        markets: str = "h2h,spreads,totals",
        regions: str = "us",
    ) -> dict:
        """
        Fetch odds from The Odds API.

        NOTE: Requires `requests` library (not included in base install)
        """
        import requests

        url = f"{self.base_url}/sports/{sport}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def parse_to_aggregator(
        self,
        api_response: dict,
        aggregator: OddsAggregator,
    ):
        """Parse The Odds API response into OddsAggregator."""
        for game in api_response:
            game_id = f"{game['away_team']}_vs_{game['home_team']}"

            for bookmaker in game.get("bookmakers", []):
                book_name = bookmaker["title"].lower().replace(" ", "")

                for market in bookmaker.get("markets", []):
                    market_type = market["key"]  # 'h2h', 'spreads', 'totals'

                    for outcome in market.get("outcomes", []):
                        # Determine bet type and side
                        if market_type == "h2h":
                            bet_type = "moneyline"
                            side = outcome["name"]
                            line = None
                        elif market_type == "spreads":
                            bet_type = "spread"
                            side = outcome["name"]
                            line = outcome.get("point", 0)
                        elif market_type == "totals":
                            bet_type = "total"
                            side = outcome["name"].lower()  # 'over' or 'under'
                            line = outcome.get("point", 0)
                        else:
                            continue

                        # Convert odds (The Odds API uses decimal odds)
                        decimal_odds = outcome["price"]
                        american_odds = decimal_to_american_odds(
                            decimal_odds - 1
                        )  # Subtract 1 for payout only

                        aggregator.add_odds_line(
                            book=book_name,
                            game_id=game_id,
                            bet_type=bet_type,
                            side=side,
                            line=line,
                            odds=american_odds,
                        )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-Book Odds Aggregator and Line Shopping")

    # Input methods
    ap.add_argument("--game", help="Game ID (e.g., KC_vs_BUF)")
    ap.add_argument(
        "--manual-odds",
        nargs="+",
        help="Manual odds (format: book:line/odds, e.g., fanduel:-3/-110)",
    )
    ap.add_argument("--odds-csv", help="CSV file with odds data")
    ap.add_argument(
        "--api", choices=["theoddsapi", "sportsdataio"], help="API source (requires API key)"
    )
    ap.add_argument("--api-key", help="API key for odds provider")

    # Bet specification
    ap.add_argument(
        "--bet-type", choices=["spread", "total", "moneyline"], help="Bet type (for manual query)"
    )
    ap.add_argument("--side", help="Bet side (team name, over, under)")

    # Output
    ap.add_argument("--output", help="Output JSON path")
    ap.add_argument("--compare-books", action="store_true", help="Compare book performance")

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print("Multi-Book Odds Aggregator")
    print(f"{'='*80}")

    # Initialize aggregator
    aggregator = OddsAggregator()

    # Load data
    if args.manual_odds:
        print("\nParsing manual odds...")

        if not args.game:
            print("ERROR: --game required with --manual-odds")
            return 1

        for odds_str in args.manual_odds:
            try:
                book, line, odds = aggregator.parse_manual_odds(odds_str)

                # Infer bet type from line
                if line is None:
                    bet_type = "moneyline"
                    side = args.side or "home"
                elif abs(line) < 10:
                    bet_type = "spread"
                    side = args.side or "home"
                else:
                    bet_type = "total"
                    side = args.side or "over"

                aggregator.add_odds_line(
                    book=book,
                    game_id=args.game,
                    bet_type=bet_type,
                    side=side,
                    line=line,
                    odds=odds,
                )

                print(f"  Added: {book} {bet_type} {side} {line}/{odds}")

            except Exception as e:
                print(f"  ERROR parsing '{odds_str}': {e}")

    elif args.odds_csv:
        print(f"\nLoading odds from {args.odds_csv}...")
        aggregator.load_from_csv(args.odds_csv)
        print(f"  Loaded {len(aggregator.odds_data)} unique bet opportunities")

    elif args.api:
        print(f"\nFetching odds from {args.api} API...")

        if args.api == "theoddsapi":
            if not args.api_key:
                print("ERROR: --api-key required for The Odds API")
                return 1

            try:
                client = TheOddsAPIClient(args.api_key)
                api_response = client.get_odds()
                client.parse_to_aggregator(api_response, aggregator)
                print(f"  Fetched odds for {len(api_response)} games")
            except Exception as e:
                print(f"  ERROR: {e}")
                return 1

        else:
            print(f"  API '{args.api}' not yet implemented")
            return 1

    else:
        print("ERROR: Provide --manual-odds, --odds-csv, or --api")
        return 1

    # Find best lines
    print(f"\n{'='*80}")
    print("Best Available Lines")
    print(f"{'='*80}")

    best_lines = aggregator.get_all_best_lines()

    if len(best_lines) == 0:
        print("No odds data to analyze")
        return 1

    for best_line in sorted(best_lines, key=lambda x: x.improvement_cents, reverse=True):
        print(f"\n{best_line.game_id} - {best_line.bet_type.upper()} {best_line.side}")
        print(f"  Best: {best_line.best_book} → {best_line.best_line}/{best_line.best_odds}")
        print(f"  Consensus: {best_line.consensus_odds}")
        print(f"  Improvement: +{best_line.improvement_cents:.1f} cents")
        print(f"  EV Gain: ${best_line.ev_gain:+.2f} on $100 bet")

        print("  All books:")
        for odds_line in best_line.all_books[:5]:  # Top 5
            print(f"    {odds_line.book:15s} {odds_line.line}/{odds_line.odds}")

    # Book comparison
    if args.compare_books:
        print(f"\n{'='*80}")
        print("Book Performance Comparison")
        print(f"{'='*80}")

        comparison = aggregator.compare_books()
        print(comparison.to_string(index=False))

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "best_lines": [bl.to_dict() for bl in best_lines],
                    "book_comparison": (
                        aggregator.compare_books().to_dict("records")
                        if args.compare_books
                        else None
                    ),
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
