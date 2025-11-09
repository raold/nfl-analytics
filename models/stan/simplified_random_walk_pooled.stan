/**
 * SIMPLIFIED Random Walk State-Space Model (FINAL ATTEMPT)
 *
 * Simplifications from v3:
 * 1. Random walk (no AR coefficient φ) - reduces parameters
 * 2. Pooled process noise Q across all players - stronger regularization
 * 3. No position hierarchy - all players pool to single league mean
 * 4. Stronger priors on all parameters
 *
 * State Equation (skill evolution - random walk):
 *   θ_{i,t} = θ_{i,t-1} + ω_{i,t},  ω_{i,t} ~ N(0, Q)
 *
 * Observation Equation (measured performance):
 *   y_{i,t} = θ_{i,t} + X_{i,t} * β + ε_{i,t},  ε_{i,t} ~ N(0, R)
 *
 * Key changes:
 * - NO φ (AR coefficient) - just random walk
 * - Single Q pooled across all players (not per-position)
 * - Single league mean (no position groups)
 * - Tighter priors to help convergence
 */

data {
  int<lower=1> N_players;                           // Number of players
  array[N_players] int<lower=1> T_player;           // Games per player
  int<lower=1> T_max;                               // Maximum games
  array[N_players, T_max] real y;                   // Observed log(passing_yards + 1)
  array[N_players, T_max] int<lower=0,upper=1> observed;  // Indicator: was this game observed?
  array[N_players, T_max] real log_attempts;        // log(attempts)
  array[N_players, T_max] int<lower=0,upper=1> is_home;  // Home indicator
}

parameters {
  // Global parameters (VERY FEW to help convergence)
  real mu_league;                    // League-wide mean skill
  real<lower=0> sigma_player;        // Between-player variation (initial skill)
  real<lower=0> Q;                   // Process noise (POOLED across all players)
  real<lower=0> R;                   // Measurement noise

  // Fixed effects
  real beta_attempts;                // Effect of attempts
  real beta_home;                    // Home-field advantage

  // Player-specific parameters (non-centered)
  vector[N_players] theta_raw_0;     // Initial skills (raw)
  array[N_players, T_max-1] real omega_raw;  // Innovations (raw)
}

transformed parameters {
  array[N_players, T_max] real theta;  // Time-varying skills

  // Initialize θ_0 ~ N(mu_league, sigma_player)
  for (i in 1:N_players) {
    theta[i, 1] = mu_league + sigma_player * theta_raw_0[i];
  }

  // Random walk: θ_t = θ_{t-1} + Q * ω_t
  for (i in 1:N_players) {
    for (t in 2:T_max) {
      if (t <= T_player[i]) {
        theta[i, t] = theta[i, t-1] + sqrt(Q) * omega_raw[i, t-1];
      } else {
        theta[i, t] = theta[i, t-1];  // No evolution past last game
      }
    }
  }
}

model {
  // STRONG priors to help convergence
  mu_league ~ normal(5.5, 0.2);           // Tighter prior on league mean
  sigma_player ~ normal(0, 0.3);          // Between-player SD
  Q ~ inv_gamma(5, 0.2);                  // Process noise (tighter)
  R ~ inv_gamma(5, 0.5);                  // Measurement noise (tighter)
  beta_attempts ~ normal(1.0, 0.15);      // Attempts effect (tighter)
  beta_home ~ normal(0.03, 0.015);        // Home advantage (tighter)

  // Non-centered priors (standard normal)
  theta_raw_0 ~ std_normal();
  for (i in 1:N_players) {
    for (t in 1:(T_max-1)) {
      omega_raw[i, t] ~ std_normal();
    }
  }

  // Likelihood
  for (i in 1:N_players) {
    for (t in 1:T_player[i]) {
      if (observed[i, t] == 1) {
        real mu_obs = theta[i, t] + beta_attempts * log_attempts[i, t] + beta_home * is_home[i, t];
        y[i, t] ~ normal(mu_obs, R);
      }
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[N_players, T_max] real y_pred;

  for (i in 1:N_players) {
    for (t in 1:T_max) {
      if (t <= T_player[i] && observed[i, t] == 1) {
        real mu_obs = theta[i, t] + beta_attempts * log_attempts[i, t] + beta_home * is_home[i, t];
        y_pred[i, t] = normal_rng(mu_obs, R);
      } else {
        y_pred[i, t] = 0;  // Not observed
      }
    }
  }
}
