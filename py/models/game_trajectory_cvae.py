"""
Conditional Variational Autoencoder (cVAE) for NFL Game Trajectory Generation.

This model learns to generate realistic game trajectories (scores, plays, momentum)
conditioned on pre-game features. It replaces parametric distributions (Skellam/Poisson)
with a deep generative model that captures complex dependencies and tail behavior.

Architecture:
- Encoder: Maps game trajectory + conditions → latent distribution q(z|x,c)
- Decoder: Generates game trajectory from latent + conditions p(x|z,c)
- Latent space: Captures game "styles" (shootout, defensive struggle, comeback, etc.)

Key advantages over parametric models:
1. Captures non-linear dependencies between quarters
2. Models momentum shifts and game flow
3. Learns empirical key number clustering without manual reweighting
4. Generates full trajectories for derivative pricing (parlays, teasers, live betting)
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GameTrajectory:
    """Represents a complete game trajectory for generation/reconstruction."""

    # Quarter scores (4 values each)
    home_q1: float
    home_q2: float
    home_q3: float
    home_q4: float
    home_ot: float  # 0 if no OT

    away_q1: float
    away_q2: float
    away_q3: float
    away_q4: float
    away_ot: float  # 0 if no OT

    # Key events (binary flags)
    home_defensive_td: bool
    away_defensive_td: bool
    home_special_teams_td: bool
    away_special_teams_td: bool

    # Momentum indicators
    largest_lead: float
    lead_changes: int

    @property
    def home_total(self) -> float:
        return self.home_q1 + self.home_q2 + self.home_q3 + self.home_q4 + self.home_ot

    @property
    def away_total(self) -> float:
        return self.away_q1 + self.away_q2 + self.away_q3 + self.away_q4 + self.away_ot

    @property
    def total_points(self) -> float:
        return self.home_total + self.away_total

    @property
    def margin(self) -> float:
        return self.home_total - self.away_total

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation for model input."""
        return torch.tensor(
            [
                self.home_q1,
                self.home_q2,
                self.home_q3,
                self.home_q4,
                self.home_ot,
                self.away_q1,
                self.away_q2,
                self.away_q3,
                self.away_q4,
                self.away_ot,
                float(self.home_defensive_td),
                float(self.away_defensive_td),
                float(self.home_special_teams_td),
                float(self.away_special_teams_td),
                self.largest_lead,
                float(self.lead_changes),
            ]
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "GameTrajectory":
        """Reconstruct from tensor representation."""
        t = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        return cls(
            home_q1=t[0],
            home_q2=t[1],
            home_q3=t[2],
            home_q4=t[3],
            home_ot=t[4],
            away_q1=t[5],
            away_q2=t[6],
            away_q3=t[7],
            away_q4=t[8],
            away_ot=t[9],
            home_defensive_td=bool(t[10] > 0.5),
            away_defensive_td=bool(t[11] > 0.5),
            home_special_teams_td=bool(t[12] > 0.5),
            away_special_teams_td=bool(t[13] > 0.5),
            largest_lead=t[14],
            lead_changes=int(t[15]),
        )


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for game trajectory generation.

    The model learns p(trajectory|conditions) where conditions include:
    - Spread, total, moneyline odds
    - Team offensive/defensive ratings
    - Weather conditions
    - Historical matchup features
    """

    def __init__(
        self,
        trajectory_dim: int = 16,  # GameTrajectory tensor dimension
        condition_dim: int = 20,  # Pre-game features dimension
        latent_dim: int = 32,  # Latent space dimension
        hidden_dims: list[int] = None,
        beta: float = 1.0,  # KL divergence weight
        device: str = "cpu",
    ):
        super().__init__()

        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Build encoder
        encoder_dims = [trajectory_dim + condition_dim] + hidden_dims
        encoder_layers = []

        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend(
                [
                    nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                    nn.BatchNorm1d(encoder_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build decoder
        decoder_dims = [latent_dim + condition_dim] + list(reversed(hidden_dims))
        decoder_layers = []

        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend(
                [
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )

        self.decoder = nn.Sequential(*decoder_layers)

        # Output heads for different trajectory components
        self.score_head = nn.Linear(decoder_dims[-1], 10)  # 5 quarters × 2 teams
        self.event_head = nn.Sequential(
            nn.Linear(decoder_dims[-1], 4), nn.Sigmoid()  # Binary events
        )
        self.momentum_head = nn.Linear(decoder_dims[-1], 2)  # Lead, changes

        # Score distribution parameters (learned)
        self.score_log_scales = nn.Parameter(torch.zeros(10))

        self.to(device)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory and conditions to latent distribution parameters.

        Args:
            x: Game trajectory tensor [batch, trajectory_dim]
            c: Condition tensor [batch, condition_dim]

        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            log_var: Log variance of latent distribution [batch, latent_dim]
        """
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean [batch, latent_dim]
            log_var: Log variance [batch, latent_dim]

        Returns:
            z: Sampled latent code [batch, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode from latent code and conditions to game trajectory.

        Args:
            z: Latent code [batch, latent_dim]
            c: Condition tensor [batch, condition_dim]

        Returns:
            Dictionary with trajectory components
        """
        inputs = torch.cat([z, c], dim=1)
        h = self.decoder(inputs)

        # Generate trajectory components
        scores = self.score_head(h)
        events = self.event_head(h)
        momentum = self.momentum_head(h)

        # Apply constraints
        scores = F.softplus(scores)  # Ensure non-negative scores
        # Avoid inplace operation by creating a new tensor
        momentum_processed = momentum.clone()
        momentum_processed[:, 1] = torch.round(
            F.softplus(momentum[:, 1])
        )  # Lead changes are integers

        return {
            "scores": scores,
            "events": events,
            "momentum": momentum_processed,
            "scales": torch.exp(self.score_log_scales),
        }

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass through VAE.

        Args:
            x: Game trajectory tensor [batch, trajectory_dim]
            c: Condition tensor [batch, condition_dim]

        Returns:
            Dictionary with reconstruction and distribution parameters
        """
        # Encode
        mu, log_var = self.encode(x, c)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Decode
        outputs = self.decode(z, c)

        # Combine trajectory components
        trajectory = torch.cat([outputs["scores"], outputs["events"], outputs["momentum"]], dim=1)

        return {
            "reconstruction": trajectory,
            "mu": mu,
            "log_var": log_var,
            "scores": outputs["scores"],
            "events": outputs["events"],
            "momentum": outputs["momentum"],
            "scales": outputs["scales"],
        }

    def loss_function(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss with separate components for different trajectory parts.

        Args:
            outputs: Model outputs from forward pass
            targets: Target trajectory tensor
            mask: Optional mask for missing data

        Returns:
            Dictionary with loss components
        """
        batch_size = targets.shape[0]

        # Split targets
        target_scores = targets[:, :10]
        target_events = targets[:, 10:14]
        target_momentum = targets[:, 14:16]

        # Reconstruction losses
        # Score loss (negative log likelihood of Laplace distribution)
        score_diff = outputs["scores"] - target_scores
        score_loss = (
            torch.abs(score_diff) / outputs["scales"] + torch.log(2 * outputs["scales"])
        ).sum(dim=1)

        # Event loss (binary cross entropy)
        event_loss = F.binary_cross_entropy(outputs["events"], target_events, reduction="none").sum(
            dim=1
        )

        # Momentum loss (MSE for continuous, Poisson for count)
        momentum_loss = F.mse_loss(
            outputs["momentum"][:, 0], target_momentum[:, 0], reduction="none"
        )
        momentum_loss += F.poisson_nll_loss(
            outputs["momentum"][:, 1], target_momentum[:, 1], reduction="none"
        )

        # Total reconstruction loss
        recon_loss = score_loss + event_loss + momentum_loss

        # Apply mask if provided
        if mask is not None:
            recon_loss = recon_loss * mask

        recon_loss = recon_loss.mean()

        # KL divergence loss
        kl_loss = (
            -0.5
            * torch.sum(1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp())
            / batch_size
        )

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "score": score_loss.mean(),
            "event": event_loss.mean(),
            "momentum": momentum_loss.mean(),
        }

    def generate(
        self, conditions: torch.Tensor, n_samples: int = 1000, temperature: float = 1.0
    ) -> list[GameTrajectory]:
        """
        Generate game trajectories from conditions.

        Args:
            conditions: Pre-game features [batch, condition_dim]
            n_samples: Number of samples per condition
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            List of GameTrajectory objects
        """
        self.eval()
        trajectories = []

        with torch.no_grad():
            batch_size = conditions.shape[0]

            for _ in range(n_samples // batch_size + 1):
                # Sample from prior
                z = torch.randn(batch_size, self.latent_dim).to(self.device) * temperature

                # Decode
                outputs = self.decode(z, conditions)

                # Convert to trajectories
                trajectory_tensor = torch.cat(
                    [outputs["scores"], outputs["events"], outputs["momentum"]], dim=1
                )

                for i in range(batch_size):
                    if len(trajectories) < n_samples:
                        traj = GameTrajectory.from_tensor(trajectory_tensor[i])
                        trajectories.append(traj)

        return trajectories[:n_samples]

    def compute_game_probabilities(
        self, conditions: torch.Tensor, spread: float, total: float, n_samples: int = 10000
    ) -> dict[str, float]:
        """
        Compute betting probabilities via sampling.

        Args:
            conditions: Pre-game features
            spread: Point spread (negative = home favored)
            total: Total points line
            n_samples: Number of MC samples

        Returns:
            Dictionary with cover, over, and other probabilities
        """
        trajectories = self.generate(conditions, n_samples)

        # Compute statistics
        margins = [t.margin for t in trajectories]
        totals = [t.total_points for t in trajectories]

        # Key numbers
        key_margins = [3, 6, 7, 10, 14]
        key_probs = {}
        for k in key_margins:
            key_probs[f"margin_{k}"] = sum(1 for m in margins if abs(m - k) < 0.5) / n_samples
            key_probs[f"margin_{-k}"] = sum(1 for m in margins if abs(m + k) < 0.5) / n_samples

        return {
            "cover": sum(1 for m in margins if m > -spread) / n_samples,
            "push": sum(1 for m in margins if abs(m + spread) < 0.5) / n_samples,
            "over": sum(1 for t in totals if t > total) / n_samples,
            "under": sum(1 for t in totals if t < total) / n_samples,
            "mean_margin": np.mean(margins),
            "std_margin": np.std(margins),
            "mean_total": np.mean(totals),
            "std_total": np.std(totals),
            **key_probs,
        }


class CVAESimulator:
    """
    High-level interface for using cVAE in Monte Carlo simulation.
    Drop-in replacement for parametric simulators.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model = ConditionalVAE(device=device)

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        """Load trained model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Loaded cVAE model from {path}")

    def simulate_game(
        self, features: dict[str, float], n_samples: int = 10000
    ) -> dict[str, np.ndarray]:
        """
        Simulate game outcomes from pre-game features.

        Args:
            features: Dictionary of pre-game features
            n_samples: Number of simulations

        Returns:
            Dictionary with arrays of simulated outcomes
        """
        # Convert features to condition tensor
        conditions = self._prepare_conditions(features)

        # Generate trajectories
        trajectories = self.model.generate(conditions, n_samples)

        # Extract outcomes
        return {
            "home_scores": np.array([t.home_total for t in trajectories]),
            "away_scores": np.array([t.away_total for t in trajectories]),
            "margins": np.array([t.margin for t in trajectories]),
            "totals": np.array([t.total_points for t in trajectories]),
            "home_q1": np.array([t.home_q1 for t in trajectories]),
            "home_q2": np.array([t.home_q2 for t in trajectories]),
            "home_q3": np.array([t.home_q3 for t in trajectories]),
            "home_q4": np.array([t.home_q4 for t in trajectories]),
            "away_q1": np.array([t.away_q1 for t in trajectories]),
            "away_q2": np.array([t.away_q2 for t in trajectories]),
            "away_q3": np.array([t.away_q3 for t in trajectories]),
            "away_q4": np.array([t.away_q4 for t in trajectories]),
            "lead_changes": np.array([t.lead_changes for t in trajectories]),
            "largest_lead": np.array([t.largest_lead for t in trajectories]),
        }

    def price_derivative(
        self,
        features: dict[str, float],
        derivative_type: str,
        params: dict,
        n_samples: int = 100000,
    ) -> dict[str, float]:
        """
        Price complex derivatives using deep generative model.

        Args:
            features: Pre-game features
            derivative_type: 'parlay', 'teaser', 'same_game_parlay', etc.
            params: Derivative-specific parameters
            n_samples: Number of MC samples

        Returns:
            Pricing results including fair value and confidence intervals
        """
        simulations = self.simulate_game(features, n_samples)

        if derivative_type == "parlay":
            return self._price_parlay(simulations, params)
        elif derivative_type == "teaser":
            return self._price_teaser(simulations, params)
        elif derivative_type == "same_game_parlay":
            return self._price_sgp(simulations, params)
        else:
            raise ValueError(f"Unknown derivative type: {derivative_type}")

    def _prepare_conditions(self, features: dict[str, float]) -> torch.Tensor:
        """Convert feature dictionary to condition tensor."""
        # Standard feature ordering
        feature_names = [
            "spread_close",
            "total_close",
            "moneyline_home",
            "moneyline_away",
            "home_offensive_rating",
            "home_defensive_rating",
            "away_offensive_rating",
            "away_defensive_rating",
            "home_recent_ppg",
            "away_recent_ppg",
            "home_rest_days",
            "away_rest_days",
            "temperature",
            "wind_speed",
            "precipitation",
            "is_division_game",
            "is_primetime",
            "week_of_season",
            "home_win_pct",
            "away_win_pct",
        ]

        conditions = []
        for name in feature_names:
            conditions.append(features.get(name, 0.0))

        return torch.tensor([conditions], dtype=torch.float32).to(self.device)

    def _price_parlay(self, sims: dict[str, np.ndarray], params: dict) -> dict[str, float]:
        """Price multi-game parlay."""
        # Implementation for parlay pricing
        pass

    def _price_teaser(self, sims: dict[str, np.ndarray], params: dict) -> dict[str, float]:
        """Price teaser with adjusted lines."""
        # Implementation for teaser pricing
        pass

    def _price_sgp(self, sims: dict[str, np.ndarray], params: dict) -> dict[str, float]:
        """Price same-game parlay with correlated legs."""
        # Implementation for SGP pricing using quarter-level correlations
        pass


def train_cvae(
    train_data,
    val_data,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ConditionalVAE:
    """
    Train the conditional VAE model.

    Args:
        train_data: Training dataset
        val_data: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device

    Returns:
        Trained model
    """
    model = ConditionalVAE(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_losses = []

        for batch in train_data:
            trajectories, conditions = batch
            trajectories = trajectories.to(device)
            conditions = conditions.to(device)

            optimizer.zero_grad()
            outputs = model(trajectories, conditions)
            losses = model.loss_function(outputs, trajectories)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())

        # Validation loop
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_data:
                trajectories, conditions = batch
                trajectories = trajectories.to(device)
                conditions = conditions.to(device)

                outputs = model(trajectories, conditions)
                losses = model.loss_function(outputs, trajectories)
                val_losses.append(losses["total"].item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                "models/cvae/best_model.pt",
            )

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}"
            )

    return model


if __name__ == "__main__":
    # Example usage
    logger.info("Initializing Game Trajectory cVAE")

    # Create model
    model = ConditionalVAE(device="cpu")

    # Example condition (pre-game features)
    conditions = torch.randn(4, 20)  # 4 games, 20 features

    # Generate trajectories
    trajectories = model.generate(conditions, n_samples=100)

    # Compute probabilities
    probs = model.compute_game_probabilities(
        conditions[0:1], spread=-3.5, total=45.5, n_samples=10000
    )

    print("\nGenerated Game Probabilities:")
    print(f"  Cover -3.5: {probs['cover']:.3f}")
    print(f"  Over 45.5: {probs['over']:.3f}")
    print(f"  Margin on 3: {probs['margin_3']:.3f}")
    print(f"  Margin on 7: {probs['margin_7']:.3f}")
    print(f"  Mean margin: {probs['mean_margin']:.1f}")
    print(f"  Std margin: {probs['std_margin']:.1f}")
