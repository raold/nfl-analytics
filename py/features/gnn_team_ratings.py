"""
Task 9: GNN Team Ratings

Graph Neural Network for learning team strength embeddings that capture
transitive relations: if A beats B and B beats C, then A should beat C.

Uses message passing on a temporal game graph to learn team embeddings.
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class TeamRatingGNN(nn.Module):
    """
    Graph Neural Network for team strength ratings.

    Architecture:
    1. Team embeddings (learned)
    2. Message passing between teams that played each other
    3. Predict game outcome from team embeddings
    """

    def __init__(
        self, n_teams: int, embedding_dim: int = 32, hidden_dim: int = 64, n_message_passes: int = 3
    ):
        super().__init__()

        self.n_teams = n_teams
        self.embedding_dim = embedding_dim
        self.n_message_passes = n_message_passes

        # Team embeddings (learned parameters)
        self.team_embeddings = nn.Embedding(n_teams, embedding_dim)

        # Message passing network
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Aggregation for message passing
        self.update_mlp = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU())

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.team_embeddings.weight)

    def message_pass(
        self, team_embeds: torch.Tensor, adjacency: list[tuple[int, int]]
    ) -> torch.Tensor:
        """
        One round of message passing.

        Args:
            team_embeds: (n_teams, embedding_dim)
            adjacency: List of (team1_id, team2_id) edges

        Returns:
            Updated embeddings
        """
        # Collect messages for each team
        messages = [[] for _ in range(self.n_teams)]

        for team1_id, team2_id in adjacency:
            # Team 1 sends message to Team 2
            concat = torch.cat([team_embeds[team1_id], team_embeds[team2_id]], dim=0)
            message = self.message_mlp(concat)
            messages[team2_id].append(message)

            # Team 2 sends message to Team 1
            concat = torch.cat([team_embeds[team2_id], team_embeds[team1_id]], dim=0)
            message = self.message_mlp(concat)
            messages[team1_id].append(message)

        # Aggregate messages and update embeddings
        new_embeds = team_embeds.clone()
        for team_id in range(self.n_teams):
            if len(messages[team_id]) > 0:
                # Average pooling
                aggregated = torch.stack(messages[team_id]).mean(dim=0)
                # Update
                concat = torch.cat([team_embeds[team_id], aggregated], dim=0)
                new_embeds[team_id] = self.update_mlp(concat)

        return new_embeds

    def forward(
        self,
        home_ids: torch.Tensor,
        away_ids: torch.Tensor,
        adjacency: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """
        Predict game outcomes.

        Args:
            home_ids: (batch_size,) team IDs
            away_ids: (batch_size,) team IDs
            adjacency: Optional game graph for message passing

        Returns:
            P(home_win): (batch_size, 1)
        """
        # Get initial embeddings
        team_embeds = self.team_embeddings.weight

        # Message passing (if adjacency provided)
        if adjacency is not None:
            for _ in range(self.n_message_passes):
                team_embeds = self.message_pass(team_embeds, adjacency)

        # Get embeddings for this batch
        home_embeds = team_embeds[home_ids]  # (batch_size, embedding_dim)
        away_embeds = team_embeds[away_ids]  # (batch_size, embedding_dim)

        # Concatenate and predict
        concat = torch.cat([home_embeds, away_embeds], dim=1)
        prob = self.predictor(concat)

        return prob

    def get_team_strength(self, team_id: int) -> float:
        """Get scalar strength rating for a team."""
        with torch.no_grad():
            embed = self.team_embeddings(torch.tensor(team_id))
            # Use L2 norm as strength
            return embed.norm().item()


def build_temporal_graph(
    games_df: pd.DataFrame, team_to_id: dict[str, int], lookback_weeks: int = 4
) -> list[list[tuple[int, int]]]:
    """
    Build temporal game graphs for each week.

    Each week's graph includes edges from the past lookback_weeks.
    This allows the GNN to aggregate information from recent games.

    Returns:
        List of adjacency lists, one per week
    """
    # Sort by season and week
    games_df = games_df.sort_values(["season", "week"])

    # Build adjacency for each week
    max_week_id = games_df.groupby(["season", "week"]).ngroup().max()
    adjacencies = [[] for _ in range(max_week_id + 1)]

    for idx, row in games_df.iterrows():
        week_id = games_df.loc[:idx].groupby(["season", "week"]).ngroup().max()
        home_id = team_to_id[row["home_team"]]
        away_id = team_to_id[row["away_team"]]

        # Add edge to current and future weeks
        for future_week in range(week_id, min(week_id + lookback_weeks, max_week_id + 1)):
            adjacencies[future_week].append((home_id, away_id))

    return adjacencies


def train_gnn_ratings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    n_message_passes: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str | None = None,
) -> tuple[TeamRatingGNN, dict[str, int]]:
    """
    Train GNN team ratings.

    Returns:
        model: Trained GNN
        team_to_id: Mapping of team names to IDs
    """
    print(f"Training GNN team ratings on {len(train_df)} games...")

    # Build team vocabulary
    all_teams = sorted(set(train_df["home_team"].unique()) | set(train_df["away_team"].unique()))
    team_to_id = {team: idx for idx, team in enumerate(all_teams)}
    n_teams = len(all_teams)

    print(f"  {n_teams} teams")

    # Build temporal graphs (simplified: use all edges)
    print("  Building game graph...")
    train_adjacency = []
    for _, row in train_df.iterrows():
        home_id = team_to_id[row["home_team"]]
        away_id = team_to_id[row["away_team"]]
        train_adjacency.append((home_id, away_id))

    # Prepare training data
    train_home_ids = torch.LongTensor([team_to_id[t] for t in train_df["home_team"]])
    train_away_ids = torch.LongTensor([team_to_id[t] for t in train_df["away_team"]])
    train_outcomes = torch.FloatTensor(
        (train_df["home_score"] > train_df["away_score"]).astype(float).values
    ).unsqueeze(1)

    # Validation data
    val_home_ids = torch.LongTensor([team_to_id.get(t, 0) for t in val_df["home_team"]])
    val_away_ids = torch.LongTensor([team_to_id.get(t, 0) for t in val_df["away_team"]])
    val_outcomes = torch.FloatTensor(
        (val_df["home_score"] > val_df["away_score"]).astype(float).values
    ).unsqueeze(1)

    # Create model
    model = TeamRatingGNN(
        n_teams=n_teams,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_message_passes=n_message_passes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    n_batches = (len(train_df) + batch_size - 1) // batch_size

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        indices = torch.randperm(len(train_df))

        total_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_df))
            batch_indices = indices[start_idx:end_idx]

            # Get batch
            batch_home = train_home_ids[batch_indices].to(device)
            batch_away = train_away_ids[batch_indices].to(device)
            batch_outcomes = train_outcomes[batch_indices].to(device)

            # Forward pass (with message passing every 10 batches for efficiency)
            if batch_idx % 10 == 0:
                pred = model(batch_home, batch_away, adjacency=train_adjacency)
            else:
                pred = model(batch_home, batch_away, adjacency=None)

            # Compute loss
            loss = criterion(pred, batch_outcomes)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_home_ids.to(device), val_away_ids.to(device), adjacency=None)
            val_loss = criterion(val_pred, val_outcomes.to(device)).item()
            val_acc = ((val_pred > 0.5).float() == val_outcomes.to(device)).float().mean().item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load best model
    model.load_state_dict(best_model_state)

    # Save model
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "team_to_id": team_to_id,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "n_message_passes": n_message_passes,
            },
            output_path,
        )
        print(f"Model saved to {output_path}")

    return model, team_to_id


def extract_gnn_features(
    model: TeamRatingGNN, team_to_id: dict[str, int], games_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract GNN-derived team strength features for downstream models.

    Returns:
        DataFrame with GNN features added
    """
    model.eval()

    # Get team strengths
    team_strengths = {}
    for team, team_id in team_to_id.items():
        team_strengths[team] = model.get_team_strength(team_id)

    # Add features to games
    df = games_df.copy()
    df["gnn_home_strength"] = df["home_team"].map(team_strengths).fillna(0.0)
    df["gnn_away_strength"] = df["away_team"].map(team_strengths).fillna(0.0)
    df["gnn_strength_diff"] = df["gnn_home_strength"] - df["gnn_away_strength"]

    # Get direct GNN predictions
    with torch.no_grad():
        home_ids = torch.LongTensor([team_to_id.get(t, 0) for t in df["home_team"]])
        away_ids = torch.LongTensor([team_to_id.get(t, 0) for t in df["away_team"]])

        device = next(model.parameters()).device
        gnn_probs = (
            model(home_ids.to(device), away_ids.to(device), adjacency=None).cpu().numpy().flatten()
        )

        df["gnn_win_prob"] = gnn_probs

    return df


def evaluate_gnn_features(features_df: pd.DataFrame, test_season: int = 2024) -> dict:
    """
    Evaluate if GNN features improve prediction accuracy.

    Compares:
    1. Baseline XGBoost features only
    2. Baseline + GNN features
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    train_df = features_df[features_df["season"] < test_season].copy()
    test_df = features_df[features_df["season"] == test_season].copy()

    y_train = (train_df["home_score"] > train_df["away_score"]).astype(int)
    y_test = (test_df["home_score"] > test_df["away_score"]).astype(int)

    # Baseline features
    baseline_features = [
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "season_win_pct_diff",
        "win_pct_last5_diff",
        "prior_margin_avg_diff",
        "points_for_last3_diff",
        "points_against_last3_diff",
        "rest_diff",
        "week",
    ]

    # GNN features
    gnn_features = ["gnn_strength_diff", "gnn_win_prob"]

    # Train simple logistic regression for comparison
    from sklearn.linear_model import LogisticRegression

    # Baseline only
    X_train_base = train_df[baseline_features].fillna(0)
    X_test_base = test_df[baseline_features].fillna(0)

    lr_base = LogisticRegression(max_iter=1000)
    lr_base.fit(X_train_base, y_train)

    y_pred_base = lr_base.predict_proba(X_test_base)[:, 1]

    base_logloss = log_loss(y_test, y_pred_base)
    base_auc = roc_auc_score(y_test, y_pred_base)
    base_acc = accuracy_score(y_test, (y_pred_base > 0.5).astype(int))

    # Baseline + GNN
    all_features = baseline_features + gnn_features
    X_train_gnn = train_df[all_features].fillna(0)
    X_test_gnn = test_df[all_features].fillna(0)

    lr_gnn = LogisticRegression(max_iter=1000)
    lr_gnn.fit(X_train_gnn, y_train)

    y_pred_gnn = lr_gnn.predict_proba(X_test_gnn)[:, 1]

    gnn_logloss = log_loss(y_test, y_pred_gnn)
    gnn_auc = roc_auc_score(y_test, y_pred_gnn)
    gnn_acc = accuracy_score(y_test, (y_pred_gnn > 0.5).astype(int))

    # GNN only (for reference)
    gnn_only_logloss = log_loss(y_test, test_df["gnn_win_prob"])
    gnn_only_auc = roc_auc_score(y_test, test_df["gnn_win_prob"])
    gnn_only_acc = accuracy_score(y_test, (test_df["gnn_win_prob"] > 0.5).astype(int))

    return {
        "baseline": {
            "logloss": base_logloss,
            "auc": base_auc,
            "accuracy": base_acc,
        },
        "baseline_plus_gnn": {
            "logloss": gnn_logloss,
            "auc": gnn_auc,
            "accuracy": gnn_acc,
        },
        "gnn_only": {
            "logloss": gnn_only_logloss,
            "auc": gnn_only_auc,
            "accuracy": gnn_only_acc,
        },
        "improvement": {
            "logloss_delta": base_logloss - gnn_logloss,
            "logloss_pct": (base_logloss - gnn_logloss) / base_logloss * 100,
            "auc_delta": gnn_auc - base_auc,
            "accuracy_delta": gnn_acc - base_acc,
        },
    }


def main():
    """
    Train GNN team ratings and evaluate feature improvement.
    """
    import argparse

    parser = argparse.ArgumentParser(description="GNN team ratings")
    parser.add_argument("--data", type=str, required=True, help="Path to features CSV")
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/gnn/team_ratings.pth",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--output-features",
        type=str,
        default="data/processed/features/gnn_features.csv",
        help="Output path for features with GNN ratings",
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default="results/gnn/evaluation.json",
        help="Output path for evaluation results",
    )
    parser.add_argument("--test-season", type=int, default=2024, help="Test season")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Task 9: GNN Team Ratings")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Split train/val/test
    train_df = df[df["season"] < args.test_season - 1].copy()
    val_df = df[df["season"] == args.test_season - 1].copy()
    test_df = df[df["season"] == args.test_season].copy()

    print(f"  Train: {len(train_df)} games (seasons < {args.test_season - 1})")
    print(f"  Val: {len(val_df)} games (season {args.test_season - 1})")
    print(f"  Test: {len(test_df)} games (season {args.test_season})")

    # Train GNN
    model, team_to_id = train_gnn_ratings(
        train_df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        device=args.device,
        output_path=args.output_model,
    )

    # Extract features for all data
    print("\nExtracting GNN features...")
    features_df = extract_gnn_features(model, team_to_id, df)

    # Save features
    Path(args.output_features).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output_features, index=False)
    print(f"Features saved to {args.output_features}")

    # Evaluate
    print("\nEvaluating GNN features...")
    results = evaluate_gnn_features(features_df, test_season=args.test_season)

    # Save results
    Path(args.output_results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_results, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Evaluation Results")
    print(f"{'=' * 80}")
    print("\nBaseline (XGBoost features only):")
    print(f"  Log Loss: {results['baseline']['logloss']:.4f}")
    print(f"  AUC: {results['baseline']['auc']:.4f}")
    print(f"  Accuracy: {results['baseline']['accuracy']:.3f}")

    print("\nBaseline + GNN:")
    print(f"  Log Loss: {results['baseline_plus_gnn']['logloss']:.4f}")
    print(f"  AUC: {results['baseline_plus_gnn']['auc']:.4f}")
    print(f"  Accuracy: {results['baseline_plus_gnn']['accuracy']:.3f}")

    print("\nGNN Only (for reference):")
    print(f"  Log Loss: {results['gnn_only']['logloss']:.4f}")
    print(f"  AUC: {results['gnn_only']['auc']:.4f}")
    print(f"  Accuracy: {results['gnn_only']['accuracy']:.3f}")

    print("\nImprovement:")
    print(
        f"  Log Loss: {results['improvement']['logloss_delta']:.4f} ({results['improvement']['logloss_pct']:.2f}%)"
    )
    print(f"  AUC: {results['improvement']['auc_delta']:.4f}")
    print(f"  Accuracy: {results['improvement']['accuracy_delta']:.3f}")

    print(f"\n{'=' * 80}")
    print(f"Complete! Results saved to {args.output_results}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
