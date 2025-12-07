"""
In-game betting simulation using drive-level predictions.

Simulate real-time drive prediction and betting opportunities:
1. Next drive total (over/under yards)
2. Next drive scoring probability
3. Updated game total (current + remaining drives)

Backtest on 2024 season to calculate ROI.

Author: NFL Analytics Team
Date: November 22, 2025
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))

from py.models.lstm_drive_prediction import DriveLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InGameBettingSimulator:
    """
    Simulate in-game betting using drive-level predictions.

    Opportunities:
    1. Next drive total (O/U 25 yards)
    2. Next drive scoring (Yes/No)
    3. Updated game total (sum remaining drives)
    """

    def __init__(self, model_path: str, device: str = "cpu", kelly_fraction: float = 0.25):
        """
        Initialize simulator.

        Args:
            model_path: Path to trained LSTM checkpoint
            device: 'cpu' or 'cuda'
            kelly_fraction: Kelly fraction for bet sizing
        """
        self.device = device
        self.kelly_fraction = kelly_fraction

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint["config"]
        self.model = DriveLSTM(self.config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")

        # Betting history
        self.bets = []

    def predict_next_drive(
        self, previous_drives: pd.DataFrame, feature_cols: list[str]
    ) -> dict[str, float]:
        """
        Predict next drive outcome given previous drives.

        Args:
            previous_drives: DataFrame with previous drives in sequence
            feature_cols: Feature column names

        Returns:
            Dictionary with predictions
        """
        # Extract features
        features = previous_drives[feature_cols].values

        # Take last N drives
        if len(features) > self.config.sequence_length:
            features = features[-self.config.sequence_length :]

        # Pad if necessary
        if len(features) < self.config.sequence_length:
            padding = np.zeros((self.config.sequence_length - len(features), features.shape[1]))
            features = np.vstack([padding, features])

        # Convert to tensor
        seq_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model.predict(seq_tensor, return_probs=True)

            yards_pred = outputs["yards"].squeeze().cpu().item()

            # Points probabilities: [P(0), P(3), P(6)]
            points_probs = outputs["points_probs"].squeeze().cpu().numpy()
            points_pred = np.dot([0, 3, 6], points_probs[:3])  # Expected points

            # Scoring probability (3+ points)
            scoring_prob = points_probs[1:3].sum()

        return {"yards_pred": yards_pred, "points_pred": points_pred, "scoring_prob": scoring_prob}

    def simulate_next_drive_bet(
        self,
        prediction: dict[str, float],
        ground_truth: dict[str, float],
        line: float = 25.0,
        odds_over: float = -110,
        odds_under: float = -110,
    ) -> dict:
        """
        Simulate next drive total bet (O/U).

        Args:
            prediction: Model prediction
            ground_truth: Actual outcome
            line: Over/under line (yards)
            odds_over: Odds for over (American)
            odds_under: Odds for under (American)

        Returns:
            Dictionary with bet details and result
        """
        yards_pred = prediction["yards_pred"]
        yards_true = ground_truth["yards"]

        # Compute edge
        predicted_prob_over = 0.5 + (yards_pred - line) / 50  # Simple heuristic
        predicted_prob_over = np.clip(predicted_prob_over, 0.1, 0.9)

        implied_prob_over = self._odds_to_prob(odds_over)
        implied_prob_under = self._odds_to_prob(odds_under)

        edge_over = predicted_prob_over - implied_prob_over
        edge_under = (1 - predicted_prob_over) - implied_prob_under

        # Determine bet
        if edge_over > 0.05:  # 5% edge threshold
            bet_side = "over"
            edge = edge_over
            bet_odds = odds_over
            bet_prob = predicted_prob_over
        elif edge_under > 0.05:
            bet_side = "under"
            edge = edge_under
            bet_odds = odds_under
            bet_prob = 1 - predicted_prob_over
        else:
            return None  # No bet

        # Kelly sizing
        kelly_size = self._kelly_criterion(bet_prob, bet_odds)
        bet_size = kelly_size * self.kelly_fraction

        # Determine outcome
        if bet_side == "over":
            win = yards_true > line
        else:
            win = yards_true < line

        # Calculate profit
        if win:
            profit = bet_size * self._payout(bet_odds)
        else:
            profit = -bet_size

        return {
            "bet_type": "next_drive_total",
            "bet_side": bet_side,
            "line": line,
            "odds": bet_odds,
            "edge": edge,
            "bet_size": bet_size,
            "yards_pred": yards_pred,
            "yards_true": yards_true,
            "win": win,
            "profit": profit,
        }

    def simulate_scoring_bet(
        self,
        prediction: dict[str, float],
        ground_truth: dict[str, float],
        odds_yes: float = -120,
        odds_no: float = +100,
    ) -> dict:
        """
        Simulate next drive scoring bet (Yes/No).

        Args:
            prediction: Model prediction
            ground_truth: Actual outcome
            odds_yes: Odds for scoring
            odds_no: Odds for not scoring

        Returns:
            Dictionary with bet details
        """
        scoring_prob = prediction["scoring_prob"]
        scored = ground_truth["points"] > 0

        # Compute edge
        implied_prob_yes = self._odds_to_prob(odds_yes)
        implied_prob_no = self._odds_to_prob(odds_no)

        edge_yes = scoring_prob - implied_prob_yes
        edge_no = (1 - scoring_prob) - implied_prob_no

        # Determine bet
        if edge_yes > 0.05:
            bet_side = "yes"
            edge = edge_yes
            bet_odds = odds_yes
            bet_prob = scoring_prob
        elif edge_no > 0.05:
            bet_side = "no"
            edge = edge_no
            bet_odds = odds_no
            bet_prob = 1 - scoring_prob
        else:
            return None

        # Kelly sizing
        kelly_size = self._kelly_criterion(bet_prob, bet_odds)
        bet_size = kelly_size * self.kelly_fraction

        # Outcome
        if bet_side == "yes":
            win = scored
        else:
            win = not scored

        # Profit
        if win:
            profit = bet_size * self._payout(bet_odds)
        else:
            profit = -bet_size

        return {
            "bet_type": "next_drive_scoring",
            "bet_side": bet_side,
            "odds": bet_odds,
            "edge": edge,
            "bet_size": bet_size,
            "scoring_prob": scoring_prob,
            "scored": scored,
            "win": win,
            "profit": profit,
        }

    def simulate_game(self, game_drives: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
        """
        Simulate in-game betting for entire game.

        Args:
            game_drives: DataFrame with all drives in game
            feature_cols: Feature column names

        Returns:
            List of bet outcomes
        """
        game_bets = []

        # Iterate through drives
        for i in range(1, len(game_drives)):
            # Previous drives (up to current)
            previous = game_drives.iloc[:i]

            # Predict next drive
            prediction = self.predict_next_drive(previous, feature_cols)

            # Ground truth (current drive)
            current_drive = game_drives.iloc[i]
            ground_truth = {
                "yards": current_drive["yards_gained"],
                "points": current_drive["points_scored"],
            }

            # Simulate bets
            total_bet = self.simulate_next_drive_bet(prediction, ground_truth)
            if total_bet:
                total_bet["game_id"] = current_drive["game_id"]
                total_bet["drive_id"] = current_drive["drive_id"]
                game_bets.append(total_bet)

            scoring_bet = self.simulate_scoring_bet(prediction, ground_truth)
            if scoring_bet:
                scoring_bet["game_id"] = current_drive["game_id"]
                scoring_bet["drive_id"] = current_drive["drive_id"]
                game_bets.append(scoring_bet)

        return game_bets

    def simulate_season(self, drives_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Simulate in-game betting for entire season.

        Args:
            drives_df: DataFrame with all drives
            feature_cols: Feature column names

        Returns:
            DataFrame with all bets
        """
        logger.info("Simulating in-game betting for season")

        all_bets = []

        for game_id in drives_df["game_id"].unique():
            game_drives = drives_df[drives_df["game_id"] == game_id].copy()

            if len(game_drives) < 2:
                continue

            game_bets = self.simulate_game(game_drives, feature_cols)
            all_bets.extend(game_bets)

        bets_df = pd.DataFrame(all_bets)

        logger.info(
            f"Simulated {len(bets_df):,} bets across {drives_df['game_id'].nunique()} games"
        )

        return bets_df

    def _odds_to_prob(self, american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def _payout(self, american_odds: float) -> float:
        """Calculate payout multiplier from American odds."""
        if american_odds > 0:
            return american_odds / 100
        else:
            return 100 / abs(american_odds)

    def _kelly_criterion(self, prob_win: float, odds: float) -> float:
        """Calculate Kelly fraction."""
        payout = self._payout(odds)
        kelly = (prob_win * (1 + payout) - 1) / payout
        return max(0, kelly)  # No negative bets


def run_simulation():
    """Run full in-game betting simulation."""
    logger.info("=" * 80)
    logger.info("IN-GAME BETTING SIMULATION (2024 Season)")
    logger.info("=" * 80)

    # Load drives data
    drives_df = pd.read_csv("data/drives_2018_2024.csv")

    # Filter to 2024 (validation set)
    drives_df["season"] = drives_df["game_id"].str[:4].astype(int)
    val_drives = drives_df[drives_df["season"] == 2024].copy()

    logger.info(
        f"Loaded {len(val_drives):,} drives from {val_drives['game_id'].nunique()} games (2024)"
    )

    # Feature columns
    feature_cols = [
        "field_position",
        "down",
        "ydstogo",
        "game_seconds_remaining",
        "score_differential",
        "drive_number",
        "recent_yards_per_drive",
        "recent_epa_per_play",
        "recent_success_rate",
        "opp_recent_yards_allowed_per_drive",
        "opp_recent_epa_allowed_per_play",
        "opp_recent_success_rate_allowed",
        "points_in_last_2_drives",
        "consecutive_first_downs",
        "drives_since_score",
    ]

    # Normalize features
    val_drives[feature_cols] = val_drives[feature_cols].fillna(0)
    for col in feature_cols:
        val_drives[col] = (val_drives[col] - val_drives[col].mean()) / (
            val_drives[col].std() + 1e-8
        )

    # Initialize simulator
    simulator = InGameBettingSimulator(
        model_path="models/lstm_drive_v1_best.pt", device="cpu", kelly_fraction=0.25
    )

    # Run simulation
    bets_df = simulator.simulate_season(val_drives, feature_cols)

    if len(bets_df) == 0:
        logger.warning("No bets placed (no edge found)")
        return

    # Calculate performance
    total_bets = len(bets_df)
    total_profit = bets_df["profit"].sum()
    total_wagered = bets_df["bet_size"].sum()
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = bets_df["win"].mean() * 100

    logger.info("\n" + "=" * 80)
    logger.info("BETTING PERFORMANCE (2024 Season)")
    logger.info("=" * 80)
    logger.info(f"Total bets: {total_bets:,}")
    logger.info(f"Total wagered: ${total_wagered:.2f}")
    logger.info(f"Total profit: ${total_profit:.2f}")
    logger.info(f"ROI: {roi:.2f}%")
    logger.info(f"Win rate: {win_rate:.1f}%")

    # Breakdown by bet type
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BY BET TYPE")
    logger.info("=" * 80)

    for bet_type in bets_df["bet_type"].unique():
        type_bets = bets_df[bets_df["bet_type"] == bet_type]
        type_profit = type_bets["profit"].sum()
        type_wagered = type_bets["bet_size"].sum()
        type_roi = (type_profit / type_wagered) * 100 if type_wagered > 0 else 0
        type_win_rate = type_bets["win"].mean() * 100

        logger.info(f"\n{bet_type}:")
        logger.info(f"  Bets: {len(type_bets):,}")
        logger.info(f"  Wagered: ${type_wagered:.2f}")
        logger.info(f"  Profit: ${type_profit:.2f}")
        logger.info(f"  ROI: {type_roi:.2f}%")
        logger.info(f"  Win rate: {type_win_rate:.1f}%")

    # Save results
    bets_df.to_csv("data/ingame_betting_simulation_2024.csv", index=False)
    logger.info("\nSaved results to: data/ingame_betting_simulation_2024.csv")


if __name__ == "__main__":
    run_simulation()
