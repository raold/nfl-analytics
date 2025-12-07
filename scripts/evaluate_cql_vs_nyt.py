#!/usr/bin/env python3
"""
CQL vs NYT 4th Down Bot Comparison

Compares our trained CQL agent against a simplified NYT 4th Down Bot
baseline that uses expected value calculations based on:
  1. Conversion probability (from historical data by distance)
  2. Field goal probability (from distance to goal line)
  3. Win probability models (based on field position and score)

The NYT Bot (Burke et al., 2009) recommends actions that maximize
expected win probability, while CQL learns from historical outcomes.

Usage:
    python scripts/evaluate_cql_vs_nyt.py \
        --model models/fourth_down_grid_20251126_112610/trial_018/cql_agent_final.pth

Author: Claude Code (Opus 4.5)
Date: November 26, 2025
"""

import argparse
import json

# Add project root to path
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from py.rl.cql_agent import CQLAgent
from py.rl.fourth_down_mdp import (
    ACTION_DIM,
    ACTION_FIELD_GOAL,
    ACTION_GO,
    ACTION_NAMES,
    ACTION_PUNT,
    STATE_DIM,
)

# ============================================================================
# NYT 4th Down Bot Implementation (Simplified)
# ============================================================================

class NYT4thDownBot:
    """
    Simplified NYT 4th Down Bot based on expected value model.

    The NYT Bot uses historical conversion/FG rates and win probability
    models to recommend the action that maximizes expected win probability.

    Key parameters calibrated from historical NFL data:
      - 4th-and-1 conversion rate: ~70%
      - 4th-and-2 conversion rate: ~60%
      - Field goal success by distance
      - Win probability vs field position

    References:
      - NYT 4th Down Bot: https://www.nytimes.com/interactive/2014/09/04/upshot/4th-down-bot.html
      - Burke (2009): "The 4th Down Study"
      - Romer (2006): "Do Firms Maximize? Evidence from Professional Football"
    """

    def __init__(self):
        """Initialize NYT Bot with calibrated probabilities."""
        # Conversion probabilities by distance (from historical NFL data)
        # These are approximate values - actual NYT bot uses more granular data
        self.conversion_probs = {
            1: 0.70,   # 4th and 1
            2: 0.60,   # 4th and 2
            3: 0.53,   # 4th and 3
            4: 0.48,   # 4th and 4
            5: 0.44,   # 4th and 5
            6: 0.40,   # 4th and 6
            7: 0.37,   # 4th and 7
            8: 0.34,   # 4th and 8
            9: 0.31,   # 4th and 9
            10: 0.28,  # 4th and 10
            # Beyond 10 yards
            15: 0.20,
            20: 0.12,
        }

        # Field goal success probability by distance (yards from goal line)
        # Based on NFL kicker success rates
        self.fg_success_by_distance = {
            20: 0.98,  # PAT distance
            25: 0.95,  # Chip shot
            30: 0.92,
            35: 0.88,
            40: 0.82,
            45: 0.75,
            50: 0.65,
            55: 0.55,
            60: 0.40,
            65: 0.25,
        }

        # Average punt net yards
        self.avg_punt_net = 42

        # Expected points by field position (simplified model)
        # From opponent's goal line
        self.expected_points_by_position = self._build_expected_points_model()

    def _build_expected_points_model(self) -> dict[int, float]:
        """
        Build expected points model by field position.

        Simplified model where expected points increase as you get closer
        to the opponent's goal line.
        """
        ep = {}
        for yardline in range(1, 100):
            # Distance from opponent's goal line
            dist_to_goal = yardline
            if dist_to_goal <= 10:
                ep[yardline] = 4.0 + (10 - dist_to_goal) * 0.3  # Redzone
            elif dist_to_goal <= 20:
                ep[yardline] = 2.5 + (20 - dist_to_goal) * 0.15
            elif dist_to_goal <= 40:
                ep[yardline] = 1.0 + (40 - dist_to_goal) * 0.075
            elif dist_to_goal <= 60:
                ep[yardline] = 0.0 + (60 - dist_to_goal) * 0.05
            else:
                ep[yardline] = -1.0 + (100 - dist_to_goal) * 0.025
        return ep

    def get_conversion_prob(self, distance: int) -> float:
        """Get 4th down conversion probability for given distance."""
        if distance <= 0:
            return 0.90  # 4th and inches
        elif distance in self.conversion_probs:
            return self.conversion_probs[distance]
        else:
            # Interpolate for distances not in table
            keys = sorted(self.conversion_probs.keys())
            for i, k in enumerate(keys[:-1]):
                if keys[i] < distance < keys[i+1]:
                    # Linear interpolation
                    frac = (distance - keys[i]) / (keys[i+1] - keys[i])
                    return self.conversion_probs[keys[i]] * (1 - frac) + \
                           self.conversion_probs[keys[i+1]] * frac
            return 0.10  # Very long distance

    def get_fg_prob(self, distance_to_goal: int) -> float:
        """Get field goal success probability from distance to goal line."""
        fg_distance = distance_to_goal + 17  # Add 7 for snap + 10 for end zone

        if fg_distance > 65:
            return 0.0  # Out of range
        elif fg_distance <= 20:
            return 0.98

        # Interpolate
        keys = sorted(self.fg_success_by_distance.keys())
        for i, k in enumerate(keys[:-1]):
            if keys[i] <= fg_distance <= keys[i+1]:
                frac = (fg_distance - keys[i]) / (keys[i+1] - keys[i])
                return self.fg_success_by_distance[keys[i]] * (1 - frac) + \
                       self.fg_success_by_distance[keys[i+1]] * frac
        return 0.0

    def recommend(
        self,
        distance: int,
        yardline: int,  # From own goal line (own 20 = 20, opponent 20 = 80)
        score_diff: int = 0,
        seconds_remaining: int = 900,
        quarter: int = 3,
    ) -> int:
        """
        Recommend 4th down action based on expected value.

        Args:
            distance: Yards to first down
            yardline: Field position from own goal line (1-99)
            score_diff: Score differential (positive = leading)
            seconds_remaining: Seconds remaining in half
            quarter: Current quarter (1-4)

        Returns:
            Recommended action (0=GO, 1=PUNT, 2=FG)
        """
        # Calculate distance to opponent's goal line
        dist_to_goal = 100 - yardline

        # Expected value of going for it
        conv_prob = self.get_conversion_prob(distance)

        # If convert: gain first down at current position
        ev_convert = self.expected_points_by_position.get(min(99, yardline + distance), 0)
        # If fail: turnover on downs at current position (opponent gets ball)
        ev_fail = -self.expected_points_by_position.get(max(1, 100 - yardline), 0)

        ev_go = conv_prob * ev_convert + (1 - conv_prob) * ev_fail

        # Expected value of punt
        new_opponent_pos = max(1, 100 - yardline - self.avg_punt_net)
        ev_punt = -self.expected_points_by_position.get(new_opponent_pos, 0)

        # Expected value of field goal
        if dist_to_goal <= 55:  # Within FG range
            fg_prob = self.get_fg_prob(dist_to_goal)
            ev_make = 3.0  # 3 points for FG
            ev_miss = -self.expected_points_by_position.get(max(1, 100 - dist_to_goal - 7), 0)
            ev_fg = fg_prob * ev_make + (1 - fg_prob) * ev_miss
        else:
            ev_fg = -999  # Out of range

        # Choose action with highest expected value
        ev_scores = {
            ACTION_GO: ev_go,
            ACTION_PUNT: ev_punt,
            ACTION_FIELD_GOAL: ev_fg,
        }

        # Adjustments for game situation
        # If trailing late, be more aggressive
        if score_diff < 0 and quarter == 4 and seconds_remaining < 300:
            ev_scores[ACTION_GO] += 0.5  # Bonus for going
            ev_scores[ACTION_PUNT] -= 0.5  # Penalty for punting

        # If leading comfortably, be more conservative
        if score_diff > 14 and quarter >= 3:
            ev_scores[ACTION_PUNT] += 0.3
            ev_scores[ACTION_GO] -= 0.3

        return max(ev_scores, key=ev_scores.get)


# ============================================================================
# Comparison Analysis
# ============================================================================

def load_test_data(csv_path: str) -> tuple:
    """Load test dataset."""
    df = pd.read_csv(csv_path)

    state_cols = [f"s{i}" for i in range(STATE_DIM)]
    states = df[state_cols].to_numpy(dtype=np.float32)
    actions = df["action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)

    return states, actions, rewards, df


def decode_state_features(state: np.ndarray) -> dict[str, float]:
    """
    Decode normalized state features back to original values.

    State encoding (from fourth_down_mdp.py):
      s0: down / 4 (always 1.0 for 4th down)
      s1: distance / 20 (yards to go, capped at 20)
      s2: yardline_100 / 100 (field position)
      s3: score_diff / 30 (score differential, capped)
      s4: time_remaining / 3600 (seconds remaining)
      s5: quarter / 3 (normalized)
      s6: timeouts / 3
      s7: posteam_wp (0-1)
      s8: vegas_wp (0-1)
      s9: is_home (0 or 1)
    """
    return {
        "distance": max(1, int(state[1] * 20)),  # Yards to first down
        "yardline": int(state[2] * 100),  # Field position (yards from opp end zone)
        "score_diff": int(state[3] * 30),  # Score differential
        "seconds_remaining": int(state[4] * 3600),  # Seconds remaining
        "quarter": int(state[5] * 3) + 1,  # Quarter (1-4)
    }


def compare_agents(
    cql_agent: CQLAgent,
    nyt_bot: NYT4thDownBot,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
) -> dict[str, Any]:
    """
    Compare CQL agent vs NYT Bot on test set.

    Metrics:
      1. Agreement rate with actual coaching decisions
      2. Agreement between CQL and NYT
      3. Reward analysis for each agent's recommendations
      4. Breakdown by distance, field position, game situation
    """
    print("\n" + "=" * 80)
    print("CQL vs NYT 4th Down Bot Comparison")
    print("=" * 80)

    n_samples = len(states)
    print(f"Test set size: {n_samples:,} plays")

    # Get CQL predictions
    cql_actions = []
    with torch.no_grad():
        for state in states:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(cql_agent.device)
            q_vals = cql_agent.q_network(state_t)
            cql_actions.append(q_vals.argmax(dim=1).item())
    cql_actions = np.array(cql_actions)

    # Get NYT Bot predictions
    nyt_actions = []
    for state in states:
        features = decode_state_features(state)
        nyt_action = nyt_bot.recommend(
            distance=features["distance"],
            yardline=features["yardline"],
            score_diff=features["score_diff"],
            seconds_remaining=features["seconds_remaining"],
            quarter=features["quarter"],
        )
        nyt_actions.append(nyt_action)
    nyt_actions = np.array(nyt_actions)

    # ========== Agreement Metrics ==========
    print("\n--- Agreement Metrics ---")

    cql_coach_agree = (cql_actions == actions).mean()
    nyt_coach_agree = (nyt_actions == actions).mean()
    cql_nyt_agree = (cql_actions == nyt_actions).mean()

    print(f"CQL vs Coach agreement:  {cql_coach_agree:.2%} ({(cql_actions == actions).sum():,}/{n_samples:,})")
    print(f"NYT vs Coach agreement:  {nyt_coach_agree:.2%} ({(nyt_actions == actions).sum():,}/{n_samples:,})")
    print(f"CQL vs NYT agreement:    {cql_nyt_agree:.2%} ({(cql_actions == nyt_actions).sum():,}/{n_samples:,})")

    # ========== Action Distribution ==========
    print("\n--- Action Distribution ---")
    print(f"{'Action':<12} {'Coach':>10} {'CQL':>10} {'NYT':>10}")
    print("-" * 45)

    for action in range(ACTION_DIM):
        coach_pct = (actions == action).mean() * 100
        cql_pct = (cql_actions == action).mean() * 100
        nyt_pct = (nyt_actions == action).mean() * 100
        print(f"{ACTION_NAMES[action]:<12} {coach_pct:>9.1f}% {cql_pct:>9.1f}% {nyt_pct:>9.1f}%")

    # ========== Reward Analysis ==========
    print("\n--- Reward Analysis ---")

    # When each agent agrees with coach (we know the outcome)
    cql_matched_reward = np.mean([rewards[i] if cql_actions[i] == actions[i] else np.nan for i in range(n_samples)])
    nyt_matched_reward = np.mean([rewards[i] if nyt_actions[i] == actions[i] else np.nan for i in range(n_samples)])
    actual_reward = rewards.mean()

    print(f"Average actual reward:       {actual_reward:.4f}")
    print(f"CQL (when matched) reward:   {np.nanmean([rewards[i] if cql_actions[i] == actions[i] else np.nan for i in range(n_samples)]):.4f}")
    print(f"NYT (when matched) reward:   {np.nanmean([rewards[i] if nyt_actions[i] == actions[i] else np.nan for i in range(n_samples)]):.4f}")

    # Reward lift calculation
    cql_lift = cql_matched_reward - actual_reward if not np.isnan(cql_matched_reward) else 0
    nyt_lift = nyt_matched_reward - actual_reward if not np.isnan(nyt_matched_reward) else 0

    # ========== Agreement by Distance ==========
    print("\n--- Agreement by Distance ---")
    print(f"{'Distance':<12} {'CQL-Coach':>12} {'NYT-Coach':>12} {'CQL-NYT':>10} {'N':>8}")
    print("-" * 55)

    distance_bins = [(1, 2), (3, 5), (6, 10), (11, 99)]
    agreement_by_distance = {}

    for dist_min, dist_max in distance_bins:
        # Distance is s1 * 20 (s0 is down/4, s1 is distance/20)
        distances = (states[:, 1] * 20).astype(int)
        mask = (distances >= dist_min) & (distances <= dist_max)

        if mask.sum() > 0:
            cql_agree = (cql_actions[mask] == actions[mask]).mean()
            nyt_agree = (nyt_actions[mask] == actions[mask]).mean()
            cql_nyt = (cql_actions[mask] == nyt_actions[mask]).mean()

            dist_label = f"{dist_min}-{dist_max}" if dist_max < 99 else f"{dist_min}+"
            print(f"{dist_label:<12} {cql_agree:>11.1%} {nyt_agree:>11.1%} {cql_nyt:>9.1%} {mask.sum():>8,}")

            agreement_by_distance[dist_label] = {
                "cql_coach": float(cql_agree),
                "nyt_coach": float(nyt_agree),
                "cql_nyt": float(cql_nyt),
                "n": int(mask.sum()),
            }

    # ========== Disagreement Analysis ==========
    print("\n--- Disagreement Analysis (CQL vs NYT) ---")

    disagree_mask = cql_actions != nyt_actions
    n_disagree = disagree_mask.sum()
    print(f"Total disagreements: {n_disagree:,} ({n_disagree/n_samples:.1%})")

    if n_disagree > 0:
        # When CQL/NYT disagree, who agrees with coach more?
        cql_right = (cql_actions[disagree_mask] == actions[disagree_mask]).sum()
        nyt_right = (nyt_actions[disagree_mask] == actions[disagree_mask]).sum()

        print("When they disagree:")
        print(f"  CQL agrees with coach: {cql_right:,} ({cql_right/n_disagree:.1%})")
        print(f"  NYT agrees with coach: {nyt_right:,} ({nyt_right/n_disagree:.1%})")

        # What type of disagreements?
        print("\nDisagreement patterns (CQL→NYT):")
        for cql_act in range(ACTION_DIM):
            for nyt_act in range(ACTION_DIM):
                if cql_act != nyt_act:
                    pattern_mask = (cql_actions == cql_act) & (nyt_actions == nyt_act)
                    count = pattern_mask.sum()
                    if count > 0:
                        print(f"  {ACTION_NAMES[cql_act]} → {ACTION_NAMES[nyt_act]}: {count:,} ({count/n_samples:.1%})")

    # ========== Compile Results ==========
    results = {
        "n_samples": n_samples,
        "agreement": {
            "cql_coach": float(cql_coach_agree),
            "nyt_coach": float(nyt_coach_agree),
            "cql_nyt": float(cql_nyt_agree),
        },
        "action_distribution": {
            "coach": {ACTION_NAMES[a]: float((actions == a).mean()) for a in range(ACTION_DIM)},
            "cql": {ACTION_NAMES[a]: float((cql_actions == a).mean()) for a in range(ACTION_DIM)},
            "nyt": {ACTION_NAMES[a]: float((nyt_actions == a).mean()) for a in range(ACTION_DIM)},
        },
        "reward": {
            "actual": float(actual_reward),
            "cql_lift": float(cql_lift) if not np.isnan(cql_lift) else None,
            "nyt_lift": float(nyt_lift) if not np.isnan(nyt_lift) else None,
        },
        "agreement_by_distance": agreement_by_distance,
        "n_disagreements": int(n_disagree),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare CQL agent vs NYT 4th Down Bot")
    parser.add_argument(
        "--model",
        default="models/fourth_down_grid_20251126_112610/trial_018/cql_agent_final.pth",
        help="Path to trained CQL model"
    )
    parser.add_argument(
        "--test-data",
        default="data/fourth_down_dataset_test.csv",
        help="Test dataset CSV"
    )
    parser.add_argument(
        "--output",
        default="models/fourth_down_grid_20251126_112610/cql_vs_nyt_comparison.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CQL vs NYT 4th Down Bot Evaluation")
    print("=" * 80)

    # Load test data
    print(f"\nLoading test data from {args.test_data}")
    states, actions, rewards, df = load_test_data(args.test_data)
    print(f"Loaded {len(states):,} test plays")

    # Initialize CQL agent
    print(f"\nLoading CQL model from {args.model}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cql_agent = CQLAgent(
        state_dim=STATE_DIM,
        n_actions=ACTION_DIM,
        device=device,
        hidden_dims=[256, 128, 64],  # Trial 18 architecture
        alpha=1.0,  # Trial 18 alpha
        lr=0.001,  # Trial 18 lr
    )
    cql_agent.load(args.model)
    cql_agent.q_network.eval()
    print(f"CQL model loaded (device: {device})")

    # Initialize NYT Bot
    nyt_bot = NYT4thDownBot()
    print("NYT 4th Down Bot initialized")

    # Run comparison
    results = compare_agents(cql_agent, nyt_bot, states, actions, rewards)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CQL agreement with coaches: {results['agreement']['cql_coach']:.1%}")
    print(f"NYT agreement with coaches: {results['agreement']['nyt_coach']:.1%}")
    print(f"CQL vs NYT agreement:       {results['agreement']['cql_nyt']:.1%}")
    print(f"\nCQL advantage over NYT:     {results['agreement']['cql_coach'] - results['agreement']['nyt_coach']:+.1%}")


if __name__ == "__main__":
    main()
