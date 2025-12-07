"""
Fourth Down Decision MDP Definition

Markov Decision Process formulation for NFL 4th down decisions:
- Go for it (attempt conversion)
- Punt (field position trade)
- Field goal attempt (3 points or turnover)

State Space (10 dimensions):
  1. down (always 4)
  2. distance (yards to first down, 1-20+)
  3. yardline_100 (yards from opponent's end zone, 1-99)
  4. score_differential (our_score - opp_score, -30 to +30)
  5. time_remaining (seconds, 0-3600)
  6. quarter (1-4)
  7. timeouts_remaining (0-3)
  8. posteam_wp (win probability before play, 0-1)
  9. vegas_wp (market-implied win probability, 0-1)
 10. is_home (1 if home team, 0 if away)

Action Space (3 discrete):
  0: Go (attempt conversion)
  1: Punt (field position trade)
  2: Field goal (3-point attempt)

Reward Function:
  Primary: Δ win probability (wp_after - wp_before)
  Bonuses:
    +3.0 for field goal made
    +7.0 for touchdown
    -7.0 for turnover on downs
    -3.0 for missed field goal
  Time decay: Late-game decisions weighted more heavily

Dataset: 110,896 4th down plays from NFL 1999-2024
Training: 1999-2023 (106,617 plays), Test: 2024-2025 (4,279 plays)

References:
  - Burke et al. (2009): "Fourth Down Study"
  - Romer (2006): "Do Firms Maximize? Evidence from Professional Football"
  - NYT 4th Down Bot: https://www.nytimes.com/interactive/2017/upshot/fourth-down-bot.html

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# ============================================================================
# Constants
# ============================================================================

DB_CONN = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

STATE_DIM = 10
ACTION_DIM = 3

# Action mapping
ACTION_GO = 0
ACTION_PUNT = 1
ACTION_FIELD_GOAL = 2

ACTION_NAMES = {
    ACTION_GO: "go",
    ACTION_PUNT: "punt",
    ACTION_FIELD_GOAL: "field_goal",
}

# Play type to action mapping
PLAY_TYPE_TO_ACTION = {
    "pass": ACTION_GO,
    "run": ACTION_GO,
    "punt": ACTION_PUNT,
    "field_goal": ACTION_FIELD_GOAL,
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class FourthDownState:
    """
    State representation for 4th down decision.

    All features are normalized to [0, 1] or [-1, 1] for neural network input.
    """

    down: int  # Always 4 (included for completeness)
    distance: float  # Yards to first down (normalized 0-1, capped at 20)
    yardline_100: float  # Yards from opponent end zone (normalized 0-1)
    score_differential: float  # Our score - opp score (normalized -1 to 1, capped at 30)
    time_remaining: float  # Seconds remaining (normalized 0-1)
    quarter: float  # Quarter (normalized 0-1)
    timeouts_remaining: float  # Timeouts (normalized 0-1)
    posteam_wp: float  # Win prob before play (0-1)
    vegas_wp: float  # Market win prob (0-1)
    is_home: float  # Home team indicator (0 or 1)

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input."""
        return np.array(
            [
                self.down / 4.0,  # Normalize to [0, 1]
                self.distance,
                self.yardline_100,
                self.score_differential,
                self.time_remaining,
                self.quarter,
                self.timeouts_remaining,
                self.posteam_wp,
                self.vegas_wp,
                self.is_home,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_row(cls, row: dict) -> FourthDownState:
        """
        Create state from database row.

        Applies normalization and clipping to ensure features are in valid ranges.
        """
        # Normalize distance (cap at 20 yards)
        distance = min(row.get("ydstogo", 10), 20) / 20.0

        # Normalize yardline_100 (0-100 yards)
        yardline_100 = row.get("yardline_100", 50) / 100.0

        # Normalize score differential (cap at ±30)
        score_diff = row.get("score_differential", 0)
        score_differential = np.clip(score_diff / 30.0, -1.0, 1.0)

        # Normalize time remaining (3600 seconds max)
        time_remaining = min(row.get("game_seconds_remaining", 1800), 3600) / 3600.0

        # Normalize quarter (1-4 → 0-1)
        # Handle NULL quarters (older data): default to Q2 (0.33)
        raw_quarter = row.get("quarter")
        if raw_quarter is None or np.isnan(raw_quarter):
            quarter = 0.33  # Default to middle of game (Q2)
        else:
            quarter = (raw_quarter - 1) / 3.0

        # Normalize timeouts (0-3 → 0-1)
        timeouts = row.get("posteam_timeouts_remaining", 1)
        timeouts_remaining = min(timeouts, 3) / 3.0 if timeouts is not None else 0.33

        # Win probabilities (already 0-1)
        # Handle NULL WP (very old data or edge cases): default to 0.5
        posteam_wp = row.get("wp", 0.5)
        if posteam_wp is None or np.isnan(posteam_wp):
            posteam_wp = 0.5

        vegas_wp = row.get("vegas_wp", 0.5)
        if vegas_wp is None or np.isnan(vegas_wp):
            vegas_wp = 0.5

        # Home team indicator
        is_home = float(row.get("is_home", 0))

        return cls(
            down=4,
            distance=distance,
            yardline_100=yardline_100,
            score_differential=score_differential,
            time_remaining=time_remaining,
            quarter=quarter,
            timeouts_remaining=timeouts_remaining,
            posteam_wp=posteam_wp,
            vegas_wp=vegas_wp,
            is_home=is_home,
        )


@dataclass
class FourthDownTransition:
    """
    State-action-reward transition for 4th down decision.

    Stores (s, a, r, s', done) tuple for offline RL training.
    """

    state: FourthDownState
    action: int  # 0=go, 1=punt, 2=field_goal
    reward: float
    next_state: FourthDownState  # Often same as state (one-step episode)
    done: bool  # Always True (episodic, one decision per 4th down)

    # Metadata for analysis
    game_id: str
    play_id: int
    season: int
    week: int
    posteam: str
    play_type: str
    yards_gained: float
    fourth_down_converted: bool
    touchdown: bool
    field_goal_result: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.to_array().tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.to_array().tolist(),
            "done": self.done,
            "game_id": self.game_id,
            "play_id": self.play_id,
            "season": self.season,
            "week": self.week,
            "posteam": self.posteam,
            "play_type": self.play_type,
            "yards_gained": self.yards_gained,
            "fourth_down_converted": self.fourth_down_converted,
            "touchdown": self.touchdown,
            "field_goal_result": self.field_goal_result,
        }


# ============================================================================
# Reward Function
# ============================================================================


def compute_reward(row: dict) -> float:
    """
    Compute reward for 4th down decision.

    Primary: Δ win probability (wp_after - wp_before)
    Bonuses: Points scored, turnovers avoided

    Args:
        row: Dictionary with play data from database

    Returns:
        reward: Float reward for this state-action transition

    Notes:
        - WP change is the primary signal (ranges -1 to +1)
        - Bonuses are scaled to be comparable to WP changes
        - Late-game plays weighted more heavily (time decay)
    """
    # Primary reward: Δ win probability
    wp_before = row.get("wp", 0.5)
    wpa = row.get("wpa", 0.0)  # Win probability added
    # Handle NULL WPA (very old data): default to 0
    if wpa is None or (isinstance(wpa, float) and np.isnan(wpa)):
        wpa = 0.0
    delta_wp = wpa

    # Base reward from WP change
    reward = delta_wp * 10.0  # Scale to make bonuses comparable

    # Bonus for points scored
    if row.get("touchdown", False):
        reward += 3.0  # Major positive outcome

    if row.get("field_goal_result") == "made":
        reward += 1.5  # Moderate positive outcome

    # Penalty for turnovers
    if row.get("fumble_lost", False) or row.get("interception", False):
        reward -= 3.0  # Major negative outcome

    # Penalty for turnover on downs (failed 4th down conversion)
    if not row.get("fourth_down_converted", False) and row.get("play_type") in ["pass", "run"]:
        # Check if this was actually a go-for-it play
        if row.get("down") == 4:
            reward -= 2.0

    # Penalty for missed field goal
    if row.get("field_goal_result") == "missed":
        reward -= 1.5

    # Time decay: Late-game decisions matter more
    time_remaining = row.get("game_seconds_remaining", 1800)
    time_weight = 1.0 + (3600 - time_remaining) / 3600  # 1.0 to 2.0
    reward *= time_weight

    # Normalize to reasonable range [-10, 10]
    reward = np.clip(reward, -10.0, 10.0)

    return reward


# ============================================================================
# Dataset Extraction
# ============================================================================


def extract_fourth_down_dataset(
    output_path: str | Path = "data/fourth_down_dataset.csv",
    seasons: tuple[int, int] = (1999, 2025),
    include_no_play: bool = False,
) -> pd.DataFrame:
    """
    Extract 4th down plays from database and convert to MDP format.

    Query joins:
      - plays: 4th down plays with outcomes
      - games: season, week, home/away team

    Filters:
      - down = 4
      - play_type IN ('pass', 'run', 'punt', 'field_goal')
      - Exclude aborted plays, QB kneels

    Args:
        output_path: Where to save extracted dataset
        seasons: (start_year, end_year) inclusive
        include_no_play: Include plays with no_play designation (default False)

    Returns:
        df: DataFrame with 110K+ 4th down plays

    Columns:
        Raw features: game_id, play_id, season, week, posteam, play_type
        State features: down, ydstogo, yardline_100, score_differential, etc.
        Outcome features: yards_gained, fourth_down_converted, touchdown, etc.
        MDP features: state (10-dim array), action (0-2), reward (float)
    """
    print(f"Extracting 4th down dataset for seasons {seasons[0]}-{seasons[1]}...")

    query = """
    SELECT
        -- Identifiers
        p.game_id,
        p.play_id,
        g.season,
        g.week,
        p.posteam,
        p.defteam,

        -- State features (raw)
        p.down,
        p.ydstogo,
        p.yardline_100,
        p.score_differential,
        p.game_seconds_remaining,
        p.half_seconds_remaining,
        p.quarter,
        p.wp,
        p.vegas_wp,

        -- Additional state context
        p.goal_to_go,
        p.shotgun,
        p.no_huddle,

        -- Timeouts (not in plays table, need to infer or use 1 as default)
        1 as posteam_timeouts_remaining,  -- Placeholder

        -- Home team indicator
        CASE WHEN p.posteam = g.home_team THEN 1 ELSE 0 END as is_home,

        -- Action (decision made)
        p.play_type,

        -- Outcome features
        p.yards_gained,
        p.fourth_down_converted,
        p.touchdown,
        p.field_goal_result,
        p.fumble_lost,
        p.interception,
        p.wpa,
        p.epa,

        -- Additional outcome context
        p.first_down,
        p.complete_pass,
        p.incomplete_pass,
        p.sack

    FROM public.plays p
    JOIN public.games g ON p.game_id = g.game_id
    WHERE p.down = 4
      AND g.season >= %(start_season)s
      AND g.season <= %(end_season)s
      AND p.play_type IN ('pass', 'run', 'punt', 'field_goal')
    ORDER BY g.season, g.week, p.game_id, p.play_id;
    """

    if include_no_play:
        # Remove play_type filter to include no_play
        query = query.replace("AND p.play_type IN ('pass', 'run', 'punt', 'field_goal')", "")

    # Execute query
    conn = psycopg2.connect(DB_CONN)
    df = pd.read_sql(
        query,
        conn,
        params={"start_season": seasons[0], "end_season": seasons[1]},
    )
    conn.close()

    print(f"Extracted {len(df):,} 4th down plays")
    print(f"Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"Play type distribution:\n{df['play_type'].value_counts()}")

    # Map play types to actions
    df["action"] = df["play_type"].map(PLAY_TYPE_TO_ACTION)

    # Drop rows with unmapped actions (e.g., no_play)
    before_len = len(df)
    df = df.dropna(subset=["action"])
    after_len = len(df)
    if before_len != after_len:
        print(f"Dropped {before_len - after_len} rows with unmapped actions")

    df["action"] = df["action"].astype(int)

    # Compute rewards
    print("Computing rewards...")
    df["reward"] = df.apply(lambda row: compute_reward(row.to_dict()), axis=1)

    # Create state arrays
    print("Creating state representations...")
    states = []
    for _, row in df.iterrows():
        state = FourthDownState.from_row(row.to_dict())
        states.append(state.to_array())

    # Add state as columns (s0, s1, ..., s9)
    state_df = pd.DataFrame(states, columns=[f"s{i}" for i in range(STATE_DIM)])
    df = pd.concat([df.reset_index(drop=True), state_df], axis=1)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total plays: {len(df):,}")
    print(f"Actions: {df['action'].value_counts().to_dict()}")
    print(f"Mean reward: {df['reward'].mean():.4f}")
    print(f"Reward std: {df['reward'].std():.4f}")
    print(f"Reward range: [{df['reward'].min():.4f}, {df['reward'].max():.4f}]")

    # Action distribution by context
    print("\n=== Action Distribution by Context ===")
    for distance_bin in [(1, 2), (3, 5), (6, 10), (11, 100)]:
        mask = (df["ydstogo"] >= distance_bin[0]) & (df["ydstogo"] <= distance_bin[1])
        if mask.sum() > 0:
            dist_str = (
                f"{distance_bin[0]}-{distance_bin[1]}"
                if distance_bin[1] < 100
                else f"{distance_bin[0]}+"
            )
            print(f"\nDistance {dist_str} yards (n={mask.sum():,}):")
            print(df[mask]["play_type"].value_counts(normalize=True))

    return df


def split_train_test(
    df: pd.DataFrame,
    test_seasons: list[int] = [2024, 2025],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets by season.

    Train: All seasons except test_seasons
    Test: test_seasons only

    Args:
        df: Full dataset
        test_seasons: List of seasons to use for testing

    Returns:
        train_df, test_df: Train and test splits
    """
    train_mask = ~df["season"].isin(test_seasons)
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[~train_mask].reset_index(drop=True)

    print("\n=== Train/Test Split ===")
    print(f"Train: {len(train_df):,} plays ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"Test: {len(test_df):,} plays ({test_df['season'].min()}-{test_df['season'].max()})")

    return train_df, test_df


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extract 4th down dataset for offline RL")
    ap.add_argument("--output", default="data/fourth_down_dataset.csv", help="Output CSV path")
    ap.add_argument("--start-season", type=int, default=1999, help="First season to include")
    ap.add_argument("--end-season", type=int, default=2025, help="Last season to include")
    ap.add_argument("--include-no-play", action="store_true", help="Include no_play outcomes")
    args = ap.parse_args()

    # Extract full dataset
    df = extract_fourth_down_dataset(
        output_path=args.output,
        seasons=(args.start_season, args.end_season),
        include_no_play=args.include_no_play,
    )

    # Create train/test split
    train_df, test_df = split_train_test(df)

    # Save splits
    output_path = Path(args.output)
    train_path = output_path.parent / f"{output_path.stem}_train.csv"
    test_path = output_path.parent / f"{output_path.stem}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved train split to {train_path}")
    print(f"Saved test split to {test_path}")

    # Save metadata
    metadata = {
        "total_plays": len(df),
        "train_plays": len(train_df),
        "test_plays": len(test_df),
        "seasons": f"{args.start_season}-{args.end_season}",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "actions": ACTION_NAMES,
        "action_distribution": df["play_type"].value_counts().to_dict(),
        "reward_stats": {
            "mean": float(df["reward"].mean()),
            "std": float(df["reward"].std()),
            "min": float(df["reward"].min()),
            "max": float(df["reward"].max()),
        },
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
