"""
Task 8: Neural Simulator for Monte Carlo Risk Validation

Implements a learned game outcome simulator to stress test betting strategies.
Computes CVaR, VaR, and max drawdown under various adverse scenarios.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    strategy_name: str
    n_trials: int
    mean_return: float
    std_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    returns_distribution: List[float]
    drawdown_distribution: List[float]


class GameOutcomeSimulator(nn.Module):
    """
    Neural network to simulate game outcomes given features.

    Learns P(home_win, spread_cover | features) from historical data.
    Used for Monte Carlo simulation of betting strategies.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Direct output heads
        self.win_head = nn.Linear(prev_dim, 1)
        self.cover_head = nn.Linear(prev_dim, 1)
        self.score_head = nn.Linear(prev_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            win_prob: P(home_win)
            cover_prob: P(home_cover_spread)
            scores: [home_score_mean, away_score_mean]
        """
        features = self.network(x)

        win_logit = self.win_head(features)
        cover_logit = self.cover_head(features)
        score_logit = self.score_head(features)

        win_prob = torch.sigmoid(win_logit)
        cover_prob = torch.sigmoid(cover_logit)
        scores = torch.nn.functional.softplus(score_logit)

        return win_prob, cover_prob, scores


class MonteCarloSimulator:
    """
    Monte Carlo simulator for stress testing betting strategies.
    """

    def __init__(self,
                 outcome_model: GameOutcomeSimulator,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.outcome_model = outcome_model.to(device)
        self.outcome_model.eval()
        self.device = device

    def simulate_game(self,
                      features: np.ndarray,
                      spread: float,
                      n_simulations: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate game outcomes.

        Args:
            features: Game features (1D array)
            spread: Point spread (positive = home favored)
            n_simulations: Number of simulations

        Returns:
            home_wins: Binary outcomes (1 = home win, 0 = away win)
            spread_covers: Binary outcomes (1 = home covers, 0 = away covers)
        """
        with torch.no_grad():
            # Repeat features for all simulations
            features_t = torch.FloatTensor(features).unsqueeze(0).repeat(n_simulations, 1).to(self.device)

            # Get probabilities
            win_prob, cover_prob, scores = self.outcome_model(features_t)

            # Sample outcomes
            home_wins = torch.bernoulli(win_prob).cpu().numpy().flatten()
            spread_covers = torch.bernoulli(cover_prob).cpu().numpy().flatten()

        return home_wins.astype(int), spread_covers.astype(int)

    def simulate_season(self,
                        games_df: pd.DataFrame,
                        strategy_actions: np.ndarray,
                        feature_cols: List[str],
                        n_trials: int = 1000,
                        bankroll: float = 100.0,
                        bet_size_pct: float = 0.01,
                        vig: float = 0.0476) -> SimulationResult:
        """
        Monte Carlo simulation of a betting strategy over a season.

        Args:
            games_df: DataFrame with games and features
            strategy_actions: Actions for each game (-3, -2, -1, 0, 1, 2, 3)
            feature_cols: Column names for features
            n_trials: Number of Monte Carlo trials
            bankroll: Starting bankroll
            bet_size_pct: Bet size as fraction of bankroll
            vig: Vigorish (e.g., 0.0476 for -110 odds)

        Returns:
            SimulationResult with statistics
        """
        n_games = len(games_df)

        # Extract features and spreads
        features = games_df[feature_cols].values
        spreads = games_df['spread_close'].values if 'spread_close' in games_df.columns else np.zeros(n_games)

        # Storage for results
        final_returns = []
        max_drawdowns = []

        for trial in range(n_trials):
            bankroll_t = bankroll
            equity_curve = [bankroll]

            for i in range(n_games):
                action = strategy_actions[i]

                if action == 0:
                    # No bet
                    equity_curve.append(bankroll_t)
                    continue

                # Determine bet size
                if abs(action) == 1:
                    bet_pct = bet_size_pct
                elif abs(action) == 2:
                    bet_pct = 2 * bet_size_pct
                else:  # abs(action) == 3
                    bet_pct = 5 * bet_size_pct

                bet_amount = bankroll_t * bet_pct

                # Simulate outcome
                home_win, spread_cover = self.simulate_game(features[i], spreads[i], n_simulations=1)

                # Determine bet result
                if action > 0:
                    # Bet on home
                    outcome = home_win[0]
                else:
                    # Bet on away
                    outcome = 1 - home_win[0]

                # Update bankroll
                if outcome == 1:
                    # Win: +bet_amount * (1 - vig)
                    profit = bet_amount * (1 - vig)
                    bankroll_t += profit
                else:
                    # Loss: -bet_amount
                    bankroll_t -= bet_amount

                equity_curve.append(bankroll_t)

            # Compute trial statistics
            final_return = (bankroll_t - bankroll) / bankroll
            final_returns.append(final_return)

            # Compute max drawdown
            equity_curve = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(drawdown.min())
            max_drawdowns.append(max_dd)

        # Aggregate statistics
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)

        mean_return = final_returns.mean()
        std_return = final_returns.std()
        sharpe = mean_return / std_return if std_return > 0 else 0.0

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(final_returns, 5)  # 5th percentile (95% VaR)
        cvar_95 = final_returns[final_returns <= var_95].mean()  # Expected return in worst 5%

        # Win rate (from actual outcomes)
        win_rate = (final_returns > 0).mean()

        return SimulationResult(
            strategy_name="strategy",
            n_trials=n_trials,
            mean_return=mean_return,
            std_return=std_return,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdowns.mean(),
            var_95=var_95,
            cvar_95=cvar_95,
            returns_distribution=final_returns.tolist(),
            drawdown_distribution=max_drawdowns.tolist(),
        )

    def stress_test(self,
                    games_df: pd.DataFrame,
                    strategy_actions: np.ndarray,
                    feature_cols: List[str],
                    scenarios: Dict[str, Dict],
                    n_trials: int = 1000) -> Dict[str, SimulationResult]:
        """
        Run multiple stress test scenarios.

        Args:
            games_df: DataFrame with games
            strategy_actions: Base strategy actions
            feature_cols: Feature column names
            scenarios: Dict of scenario_name -> scenario_params
                       Each scenario can specify:
                       - model_degradation: float (reduce win prob by this amount)
                       - edge_reduction: float (reduce edge by this fraction)
                       - adverse_selection: float (penalize bets by this amount)
                       - correlation: float (induce correlated losses)
            n_trials: Number of Monte Carlo trials per scenario

        Returns:
            Dict of scenario_name -> SimulationResult
        """
        results = {}

        for scenario_name, scenario_params in scenarios.items():
            print(f"\nRunning scenario: {scenario_name}")
            print(f"  Parameters: {scenario_params}")

            # Apply scenario modifications
            modified_df = games_df.copy()
            modified_actions = strategy_actions.copy()

            # TODO: Implement scenario modifications
            # For now, run baseline simulation

            result = self.simulate_season(
                games_df=modified_df,
                strategy_actions=modified_actions,
                feature_cols=feature_cols,
                n_trials=n_trials,
            )

            result.strategy_name = scenario_name
            results[scenario_name] = result

            print(f"  Mean return: {result.mean_return:.4f}")
            print(f"  Sharpe: {result.sharpe_ratio:.4f}")
            print(f"  Max DD: {result.max_drawdown:.4f}")
            print(f"  VaR(95%): {result.var_95:.4f}")
            print(f"  CVaR(95%): {result.cvar_95:.4f}")

        return results


def train_outcome_model(train_df: pd.DataFrame,
                        feature_cols: List[str],
                        epochs: int = 100,
                        batch_size: int = 64,
                        lr: float = 1e-3,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        output_path: Optional[str] = None) -> GameOutcomeSimulator:
    """
    Train the game outcome simulator.

    Args:
        train_df: Training data with features and outcomes
        feature_cols: Column names for features
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        output_path: Path to save trained model

    Returns:
        Trained GameOutcomeSimulator
    """
    print(f"Training game outcome simulator on {len(train_df)} games...")

    # Prepare data
    X = train_df[feature_cols].values

    # Outcomes
    y_win = (train_df['home_score'] > train_df['away_score']).astype(float).values

    # Spread cover (home team covers if home_score + spread > away_score)
    if 'spread_close' in train_df.columns:
        y_cover = ((train_df['home_score'] + train_df['spread_close']) > train_df['away_score']).astype(float).values
    else:
        y_cover = y_win  # Fallback to win if no spread

    # Scores
    y_scores = train_df[['home_score', 'away_score']].values

    # Convert to tensors
    X_t = torch.FloatTensor(X).to(device)
    y_win_t = torch.FloatTensor(y_win).unsqueeze(1).to(device)
    y_cover_t = torch.FloatTensor(y_cover).unsqueeze(1).to(device)
    y_scores_t = torch.FloatTensor(y_scores).to(device)

    # Create model
    input_dim = X.shape[1]
    model = GameOutcomeSimulator(input_dim=input_dim).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Training loop
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()

        # Shuffle data
        indices = np.random.permutation(n_samples)

        total_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Get batch
            X_batch = X_t[batch_indices]
            y_win_batch = y_win_t[batch_indices]
            y_cover_batch = y_cover_t[batch_indices]
            y_scores_batch = y_scores_t[batch_indices]

            # Forward pass - get logits directly
            features = model.network(X_batch)
            win_logit = model.win_head(features)
            cover_logit = model.cover_head(features)
            score_logit = model.score_head(features)
            scores = torch.nn.functional.softplus(score_logit)

            # Compute losses using logits (more numerically stable)
            loss_win = bce_with_logits_loss(win_logit, y_win_batch)
            loss_cover = bce_with_logits_loss(cover_logit, y_cover_batch)
            loss_scores = mse_loss(scores, y_scores_batch)

            # Combined loss
            loss = loss_win + loss_cover + 0.1 * loss_scores

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_cols': feature_cols,
            'input_dim': input_dim,
        }, output_path)
        print(f"Model saved to {output_path}")

    return model


def load_outcome_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[GameOutcomeSimulator, List[str]]:
    """
    Load a trained outcome model.

    Returns:
        model: Trained GameOutcomeSimulator
        feature_cols: Feature column names
    """
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = checkpoint['input_dim']
    feature_cols = checkpoint['feature_cols']

    model = GameOutcomeSimulator(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, feature_cols


def main():
    """
    Example usage: Train outcome model and run stress tests.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Neural simulator for risk validation')
    parser.add_argument('--mode', type=str, choices=['train', 'simulate', 'stress'], required=True,
                        help='Mode: train model, run simulation, or stress test')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--model', type=str, default='models/simulator/outcome_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--strategy-file', type=str,
                        help='Path to strategy actions JSON (for simulate/stress modes)')
    parser.add_argument('--output', type=str, default='results/simulation/stress_test.json',
                        help='Output path for results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--n-trials', type=int, default=1000,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Define feature columns (same as ensemble)
    feature_cols = [
        'prior_epa_mean_diff', 'epa_pp_last3_diff', 'season_win_pct_diff',
        'win_pct_last5_diff', 'prior_margin_avg_diff', 'points_for_last3_diff',
        'points_against_last3_diff', 'rest_diff', 'week', 'fourth_downs_diff',
        'fourth_down_epa_diff'
    ]

    if args.mode == 'train':
        # Train outcome model
        train_df = df[df['season'] < 2024].copy()

        model = train_outcome_model(
            train_df=train_df,
            feature_cols=feature_cols,
            epochs=args.epochs,
            device=args.device,
            output_path=args.model,
        )

        print(f"\nTraining complete! Model saved to {args.model}")

    elif args.mode == 'simulate':
        # Load model
        model, feature_cols_loaded = load_outcome_model(args.model, device=args.device)
        simulator = MonteCarloSimulator(model, device=args.device)

        # Load strategy actions
        with open(args.strategy_file, 'r') as f:
            strategy_data = json.load(f)

        actions = np.array([bet['action'] for bet in strategy_data['bets']])

        # Filter to test season
        test_df = df[df['season'] == 2024].copy()

        # Run simulation
        result = simulator.simulate_season(
            games_df=test_df,
            strategy_actions=actions,
            feature_cols=feature_cols,
            n_trials=args.n_trials,
        )

        # Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'strategy_name': result.strategy_name,
                'n_trials': result.n_trials,
                'mean_return': result.mean_return,
                'std_return': result.std_return,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
            }, f, indent=2)

        print(f"\nSimulation complete! Results saved to {args.output}")
        print(f"Mean return: {result.mean_return:.4f}")
        print(f"Sharpe: {result.sharpe_ratio:.4f}")
        print(f"VaR(95%): {result.var_95:.4f}")
        print(f"CVaR(95%): {result.cvar_95:.4f}")

    elif args.mode == 'stress':
        # Load model
        model, feature_cols_loaded = load_outcome_model(args.model, device=args.device)
        simulator = MonteCarloSimulator(model, device=args.device)

        # Load strategy actions
        with open(args.strategy_file, 'r') as f:
            strategy_data = json.load(f)

        actions = np.array([bet['action'] for bet in strategy_data['bets']])

        # Filter to test season
        test_df = df[df['season'] == 2024].copy()

        # Define stress test scenarios
        scenarios = {
            'baseline': {},
            'model_degradation': {'model_degradation': 0.05},
            'market_efficiency': {'edge_reduction': 0.5},
            'adverse_selection': {'adverse_selection': 0.5},
            'correlated_losses': {'correlation': 0.3},
        }

        # Run stress tests
        results = simulator.stress_test(
            games_df=test_df,
            strategy_actions=actions,
            feature_cols=feature_cols,
            scenarios=scenarios,
            n_trials=args.n_trials,
        )

        # Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            output_data = {}
            for scenario_name, result in results.items():
                output_data[scenario_name] = {
                    'n_trials': result.n_trials,
                    'mean_return': result.mean_return,
                    'std_return': result.std_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'win_rate': result.win_rate,
                    'max_drawdown': result.max_drawdown,
                    'var_95': result.var_95,
                    'cvar_95': result.cvar_95,
                }
            json.dump(output_data, f, indent=2)

        print(f"\nStress test complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()
