#!/usr/bin/env python3
"""
Line Movement Tracker for Early Week Betting (EWB) Strategy.

Tracks NFL line movements from opening (Tuesday/Wednesday) to closing (Sunday)
to identify sharp money, steam moves, and closing line value (CLV).

Key Concepts:
- Opening Line (OL): First line posted (typically Tuesday 9 AM ET)
- Closing Line (CL): Final line before kickoff (typically Sunday 1 PM ET)
- Line Movement: CL - OL (positive = line moved toward favorite)
- Sharp Money: Early action from professional bettors
- Steam Move: Rapid line movement (>1 point in <15 min)
- CLV: Closing Line Value (profit from betting before sharp money)

Early Week Betting (EWB) Strategy:
- Bet Tuesday-Thursday when line first opens
- Hypothesis: Sharp money moves lines Friday-Sunday
- Goal: Lock in favorable number before line moves against you
- Expected edge: +0.5-1.5% vs waiting for closing line

Usage:
    # Track single game
    python py/features/line_movement_tracker.py \
        --game "KC_vs_BUF" \
        --opening-line -3 \
        --closing-line -3.5

    # Analyze historical line movements
    python py/features/line_movement_tracker.py \
        --historical-csv data/odds/historical_lines_2024.csv \
        --output results/line_movement/ewb_analysis.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class LineSnapshot:
    """Snapshot of line at specific time."""
    timestamp: str
    book: str
    spread: float
    total: float
    moneyline_home: int
    moneyline_away: int


@dataclass
class LineMovement:
    """Line movement analysis for a single game."""
    game_id: str
    opening_spread: float
    closing_spread: float
    movement: float  # CL - OL (positive = moved toward favorite)
    movement_direction: str  # 'favorite', 'underdog', 'push'
    opening_total: float
    closing_total: float
    total_movement: float  # CL - OL
    sharp_indicators: List[str]  # List of sharp money signals
    steam_moves: List[Dict]  # List of rapid moves
    ewb_edge: float  # Estimated edge from betting early (%)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CLVAnalysis:
    """Closing Line Value analysis."""
    total_bets: int
    ewb_wins: int  # Times EWB had better line than CL
    ewb_losses: int  # Times EWB had worse line than CL
    ewb_pushes: int  # Times lines stayed same
    avg_clv: float  # Average CLV in points
    clv_dollars: float  # Dollar value of CLV ($)


# ============================================================================
# Line Movement Tracker
# ============================================================================


class LineMovementTracker:
    """
    Track and analyze NFL line movements for EWB strategy.
    """

    def __init__(self):
        self.snapshots: Dict[str, List[LineSnapshot]] = {}
        self.movements: List[LineMovement] = []

    def add_snapshot(
        self,
        game_id: str,
        timestamp: str,
        book: str,
        spread: float,
        total: float,
        moneyline_home: int = None,
        moneyline_away: int = None,
    ):
        """Add a line snapshot to tracker."""
        snapshot = LineSnapshot(
            timestamp=timestamp,
            book=book,
            spread=spread,
            total=total,
            moneyline_home=moneyline_home or 0,
            moneyline_away=moneyline_away or 0,
        )

        if game_id not in self.snapshots:
            self.snapshots[game_id] = []

        self.snapshots[game_id].append(snapshot)

    def load_from_csv(self, csv_path: str):
        """
        Load line snapshots from CSV.

        CSV format:
            game_id,timestamp,book,spread,total,moneyline_home,moneyline_away
            KC_vs_BUF,2024-10-08T09:00:00,fanduel,-3,52.5,-155,+135
            KC_vs_BUF,2024-10-08T12:00:00,fanduel,-3.5,52.5,-165,+145
        """
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            self.add_snapshot(
                game_id=row['game_id'],
                timestamp=row['timestamp'],
                book=row['book'],
                spread=float(row['spread']),
                total=float(row['total']),
                moneyline_home=int(row.get('moneyline_home', 0)),
                moneyline_away=int(row.get('moneyline_away', 0)),
            )

    def identify_sharp_indicators(
        self,
        game_id: str,
        opening_spread: float,
        closing_spread: float,
    ) -> List[str]:
        """
        Identify sharp money indicators.

        Sharp indicators:
        - Line moves against public (reverse line move)
        - Line moves early in week (Tuesday-Wednesday)
        - Line moves significantly (>1 point)
        - Multiple books move simultaneously
        """
        indicators = []

        movement = closing_spread - opening_spread

        # Significant movement (>1 point)
        if abs(movement) >= 1.0:
            indicators.append('significant_move')

        # Check if move was early in week
        if game_id in self.snapshots:
            snapshots = sorted(self.snapshots[game_id], key=lambda x: x.timestamp)

            if len(snapshots) >= 2:
                first_move_time = datetime.fromisoformat(snapshots[1].timestamp)

                # If first move was Tuesday-Wednesday (early week)
                if first_move_time.weekday() in [1, 2]:  # Tuesday=1, Wednesday=2
                    indicators.append('early_week_move')

        # Check for steam move (rapid movement)
        steam_moves = self.detect_steam_moves(game_id)
        if len(steam_moves) > 0:
            indicators.append('steam_move')

        # Reverse line move (move against public assumption)
        # Assumption: Public bets favorites
        if movement > 0:  # Line moved toward favorite
            indicators.append('sharp_on_favorite')
        elif movement < 0:  # Line moved toward underdog
            indicators.append('sharp_on_underdog')

        return indicators

    def detect_steam_moves(self, game_id: str) -> List[Dict]:
        """
        Detect steam moves (rapid line changes).

        Steam move criteria:
        - Line moves >0.5 points in <15 minutes
        - Multiple books move simultaneously
        """
        steam_moves = []

        if game_id not in self.snapshots:
            return steam_moves

        snapshots = sorted(self.snapshots[game_id], key=lambda x: x.timestamp)

        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            # Time difference
            prev_time = datetime.fromisoformat(prev.timestamp)
            curr_time = datetime.fromisoformat(curr.timestamp)
            time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes

            # Line movement
            spread_move = curr.spread - prev.spread

            # Steam move: >0.5 points in <15 min
            if abs(spread_move) >= 0.5 and time_diff <= 15:
                steam_moves.append({
                    'timestamp': curr.timestamp,
                    'time_window_minutes': time_diff,
                    'spread_movement': spread_move,
                    'book': curr.book,
                })

        return steam_moves

    def calculate_ewb_edge(
        self,
        opening_spread: float,
        closing_spread: float,
        bet_early_on: str = 'model',
    ) -> float:
        """
        Calculate EWB edge (value from betting early).

        Args:
            opening_spread: Opening line
            closing_spread: Closing line
            bet_early_on: 'model' (model picks side), 'favorite', 'underdog'

        Returns:
            Edge in percentage (positive = EWB was better)
        """
        movement = closing_spread - opening_spread

        # If model picked favorite at opening
        # and line moved toward favorite (closing spread increased)
        # → We got worse line by betting early (negative edge)

        # If model picked favorite at opening
        # and line moved toward underdog (closing spread decreased)
        # → We got better line by betting early (positive edge)

        # Simplified: Each 0.5 point ≈ 2-3% edge
        # Conservative: Each 0.5 point = 2% edge
        edge_per_half_point = 0.02  # 2%

        # Edge = (how much line moved in our favor) * edge_per_half_point
        # If we bet favorite and line moved toward underdog → positive edge
        # If we bet favorite and line moved toward favorite → negative edge

        # Simplification: Assume we always bet against line movement
        # (i.e., if line moved toward favorite, we bet underdog early)
        edge = abs(movement) * edge_per_half_point * 2  # Convert to percentage

        # Direction: If line moved, EWB had edge
        if abs(movement) < 0.5:
            edge = 0  # No meaningful movement

        return edge

    def analyze_game(
        self,
        game_id: str,
        opening_spread: Optional[float] = None,
        closing_spread: Optional[float] = None,
        opening_total: Optional[float] = None,
        closing_total: Optional[float] = None,
    ) -> LineMovement:
        """
        Analyze line movement for a single game.

        Args:
            game_id: Game identifier
            opening_spread: Opening spread (if not in snapshots)
            closing_spread: Closing spread (if not in snapshots)
            opening_total: Opening total (if not in snapshots)
            closing_total: Closing total (if not in snapshots)

        Returns:
            LineMovement object
        """
        # Get opening and closing from snapshots if not provided
        if game_id in self.snapshots and len(self.snapshots[game_id]) >= 2:
            snapshots = sorted(self.snapshots[game_id], key=lambda x: x.timestamp)
            opening_snap = snapshots[0]
            closing_snap = snapshots[-1]

            opening_spread = opening_spread or opening_snap.spread
            closing_spread = closing_spread or closing_snap.spread
            opening_total = opening_total or opening_snap.total
            closing_total = closing_total or closing_snap.total

        # Calculate movements
        spread_movement = closing_spread - opening_spread
        total_movement = closing_total - opening_total

        # Determine movement direction
        if abs(spread_movement) < 0.5:
            movement_direction = 'push'
        elif spread_movement > 0:
            movement_direction = 'favorite'
        else:
            movement_direction = 'underdog'

        # Identify sharp indicators
        sharp_indicators = self.identify_sharp_indicators(
            game_id, opening_spread, closing_spread
        )

        # Detect steam moves
        steam_moves = self.detect_steam_moves(game_id)

        # Calculate EWB edge
        ewb_edge = self.calculate_ewb_edge(opening_spread, closing_spread)

        return LineMovement(
            game_id=game_id,
            opening_spread=opening_spread,
            closing_spread=closing_spread,
            movement=spread_movement,
            movement_direction=movement_direction,
            opening_total=opening_total,
            closing_total=closing_total,
            total_movement=total_movement,
            sharp_indicators=sharp_indicators,
            steam_moves=steam_moves,
            ewb_edge=ewb_edge,
        )

    def analyze_clv(
        self,
        bet_times: List[str] = ['early'],  # 'early', 'late', 'closing'
    ) -> CLVAnalysis:
        """
        Analyze Closing Line Value (CLV) for betting strategy.

        Args:
            bet_times: When bets were placed ('early', 'late', 'closing')

        Returns:
            CLVAnalysis
        """
        ewb_wins = 0
        ewb_losses = 0
        ewb_pushes = 0
        clv_values = []

        for movement in self.movements:
            # EWB = betting early (opening line)
            # CLV = difference between opening and closing

            if abs(movement.movement) < 0.5:
                ewb_pushes += 1
                clv_values.append(0)
            elif movement.movement > 0:
                # Line moved toward favorite
                # If we bet underdog early → we won (got better line)
                ewb_wins += 1
                clv_values.append(abs(movement.movement))
            else:
                # Line moved toward underdog
                # If we bet favorite early → we won (got better line)
                ewb_wins += 1
                clv_values.append(abs(movement.movement))

        # Calculate CLV
        avg_clv = np.mean(clv_values) if clv_values else 0

        # Dollar value of CLV (assuming $100 per bet, 2% edge per 0.5 points)
        clv_dollars = avg_clv * 0.02 * 2 * 100 * len(clv_values)  # rough estimate

        return CLVAnalysis(
            total_bets=len(self.movements),
            ewb_wins=ewb_wins,
            ewb_losses=ewb_losses,
            ewb_pushes=ewb_pushes,
            avg_clv=avg_clv,
            clv_dollars=clv_dollars,
        )

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for all tracked games."""
        if len(self.movements) == 0:
            return {}

        movements = [m.movement for m in self.movements]
        total_movements = [m.total_movement for m in self.movements]
        ewb_edges = [m.ewb_edge for m in self.movements]

        return {
            'total_games': len(self.movements),
            'avg_spread_movement': np.mean(movements),
            'std_spread_movement': np.std(movements),
            'max_spread_movement': np.max(np.abs(movements)),
            'avg_total_movement': np.mean(total_movements),
            'pct_line_moves_gt_1pt': np.mean(np.abs(movements) >= 1.0) * 100,
            'avg_ewb_edge': np.mean(ewb_edges),
            'total_ewb_edge': np.sum(ewb_edges),
            'games_with_sharp_indicators': sum(1 for m in self.movements if len(m.sharp_indicators) > 0),
            'games_with_steam_moves': sum(1 for m in self.movements if len(m.steam_moves) > 0),
        }


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Line Movement Tracker for Early Week Betting (EWB)'
    )

    # Single game analysis
    ap.add_argument('--game', help='Game ID (e.g., KC_vs_BUF)')
    ap.add_argument('--opening-line', type=float, help='Opening spread')
    ap.add_argument('--closing-line', type=float, help='Closing spread')
    ap.add_argument('--opening-total', type=float, help='Opening total')
    ap.add_argument('--closing-total', type=float, help='Closing total')

    # Historical analysis
    ap.add_argument('--historical-csv', help='CSV with historical line snapshots')
    ap.add_argument('--analyze-clv', action='store_true', help='Analyze CLV')

    # Output
    ap.add_argument('--output', help='Output JSON path')

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print(f"Line Movement Tracker - Early Week Betting (EWB) Analysis")
    print(f"{'='*80}")

    tracker = LineMovementTracker()

    # Single game analysis
    if args.game and args.opening_line is not None and args.closing_line is not None:
        print(f"\nAnalyzing game: {args.game}")

        movement = tracker.analyze_game(
            game_id=args.game,
            opening_spread=args.opening_line,
            closing_spread=args.closing_line,
            opening_total=args.opening_total or 0,
            closing_total=args.closing_total or 0,
        )

        print(f"\nLine Movement:")
        print(f"  Opening spread: {movement.opening_spread}")
        print(f"  Closing spread: {movement.closing_spread}")
        print(f"  Movement: {movement.movement:+.1f} points ({movement.movement_direction})")
        print(f"  EWB edge: {movement.ewb_edge:.2%}")

        if movement.sharp_indicators:
            print(f"\nSharp indicators:")
            for indicator in movement.sharp_indicators:
                print(f"  - {indicator}")

        if movement.steam_moves:
            print(f"\nSteam moves:")
            for steam in movement.steam_moves:
                print(f"  - {steam['timestamp']}: {steam['spread_movement']:+.1f} points in {steam['time_window_minutes']:.0f} min")

        tracker.movements.append(movement)

    # Historical analysis
    elif args.historical_csv:
        print(f"\nLoading historical data from {args.historical_csv}...")
        tracker.load_from_csv(args.historical_csv)
        print(f"  Loaded {len(tracker.snapshots)} games")

        # Analyze each game
        print(f"\nAnalyzing line movements...")
        for game_id in tracker.snapshots.keys():
            movement = tracker.analyze_game(game_id)
            tracker.movements.append(movement)

        # Summary stats
        print(f"\n{'='*80}")
        print(f"Summary Statistics")
        print(f"{'='*80}")

        stats = tracker.get_summary_stats()

        print(f"\nLine Movement:")
        print(f"  Total games: {stats['total_games']}")
        print(f"  Avg spread movement: {stats['avg_spread_movement']:+.2f} points")
        print(f"  Std spread movement: {stats['std_spread_movement']:.2f} points")
        print(f"  Max spread movement: {stats['max_spread_movement']:.2f} points")
        print(f"  Avg total movement: {stats['avg_total_movement']:+.2f} points")
        print(f"  % lines moved >1 point: {stats['pct_line_moves_gt_1pt']:.1f}%")

        print(f"\nEarly Week Betting (EWB):")
        print(f"  Avg EWB edge: {stats['avg_ewb_edge']:.2%}")
        print(f"  Total EWB edge: {stats['total_ewb_edge']:.2%}")
        print(f"  Games with sharp indicators: {stats['games_with_sharp_indicators']} ({stats['games_with_sharp_indicators']/stats['total_games']*100:.1f}%)")
        print(f"  Games with steam moves: {stats['games_with_steam_moves']} ({stats['games_with_steam_moves']/stats['total_games']*100:.1f}%)")

        # CLV analysis
        if args.analyze_clv:
            print(f"\n{'='*80}")
            print(f"Closing Line Value (CLV) Analysis")
            print(f"{'='*80}")

            clv = tracker.analyze_clv()

            print(f"\nCLV Results:")
            print(f"  Total bets: {clv.total_bets}")
            print(f"  EWB wins (better line): {clv.ewb_wins} ({clv.ewb_wins/clv.total_bets*100:.1f}%)")
            print(f"  EWB losses (worse line): {clv.ewb_losses} ({clv.ewb_losses/clv.total_bets*100:.1f}%)")
            print(f"  Pushes (same line): {clv.ewb_pushes} ({clv.ewb_pushes/clv.total_bets*100:.1f}%)")
            print(f"  Avg CLV: {clv.avg_clv:.2f} points")
            print(f"  CLV dollar value: ${clv.clv_dollars:,.0f}")

            print(f"\nRecommendation:")
            if clv.avg_clv > 0.3:
                print(f"  ✅ EWB strategy has significant edge ({clv.avg_clv:.2f} points CLV)")
                print(f"  ✅ Bet early in week (Tuesday-Wednesday)")
            elif clv.avg_clv > 0.1:
                print(f"  ⚠️ EWB strategy has marginal edge ({clv.avg_clv:.2f} points CLV)")
                print(f"  ⚠️ Consider model confidence threshold")
            else:
                print(f"  ❌ EWB strategy has minimal edge ({clv.avg_clv:.2f} points CLV)")
                print(f"  ❌ Wait for closing line or sharp move confirmation")

    else:
        print("ERROR: Provide --game with --opening-line/--closing-line OR --historical-csv")
        return 1

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'movements': [m.to_dict() for m in tracker.movements],
                'summary_stats': tracker.get_summary_stats() if args.historical_csv else None,
                'clv_analysis': asdict(tracker.analyze_clv()) if args.analyze_clv else None,
            }, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
