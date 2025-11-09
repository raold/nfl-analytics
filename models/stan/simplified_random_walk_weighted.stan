/**
 * Weighted Simplified Random Walk State-Space Model
 *
 * Extension of simplified_random_walk_pooled.stan with observation weights
 * for implementing exponential decay on historical data.
 *
 * Training window: 2018-2024 (weight=1.0) + 2015-2017 (exponential decay)
 * Weight formula: w(s; t, H) = 0.5^((t-s)/H)
 *
 * State Equation (skill evolution - random walk):
 *   θ_{i,t} = θ_{i,t-1} + ω_{i,t},  ω_{i,t} ~ N(0, Q)
 *
 * Observation Equation (measured performance with weights):
 *   y_{i,t} ~ Normal(θ_{i,t} + X_{i,t} * β, R / sqrt(w_{i,t}))
 *   where w_{i,t} is the observation weight
 *
 * Key implementation: Weighted likelihood using inverse weight in SD
 */

data {
  int<lower=1> N_players;                           // Number of players
  array[N_players] int<lower=1> T_player;           // Games per player
  int<lower=1> T_max;                               // Maximum games
  array[N_players, T_max] real y;                   // Observed log(passing_yards + 1)
  array[N_players, T_max] int<lower=0,upper=1> observed;  // Indicator: was this game observed?
  array[N_players, T_max] real log_attempts;        // log(attempts)
  array[N_players, T_max] int<lower=0,upper=1> is_home;  // Home indicator
  array[N_players, T_max] real<lower=0> weights;    // Observation weights (NEW)
}

parameters {
  // Global parameters
  real mu_league;                    // League-wide mean skill
  real<lower=0> sigma_player;        // Between-player variation (initial skill)
  real<lower=0> Q;                   // Process noise (pooled across all players)
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
  // Priors (same as unweighted model)
  mu_league ~ normal(5.5, 0.2);
  sigma_player ~ normal(0, 0.3);
  Q ~ inv_gamma(5, 0.2);
  R ~ inv_gamma(5, 0.5);
  beta_attempts ~ normal(1.0, 0.15);
  beta_home ~ normal(0.03, 0.015);

  // Non-centered priors
  theta_raw_0 ~ std_normal();
  for (i in 1:N_players) {
    for (t in 1:(T_max-1)) {
      omega_raw[i, t] ~ std_normal();
    }
  }

  // WEIGHTED Likelihood
  // Down-weight old observations by reducing their effective precision
  // Implementation: scale SD by 1/sqrt(weight) so variance scales by 1/weight
  for (i in 1:N_players) {
    for (t in 1:T_player[i]) {
      if (observed[i, t] == 1) {
        real mu_obs = theta[i, t] + beta_attempts * log_attempts[i, t] + beta_home * is_home[i, t];
        real weighted_sigma = R / sqrt(weights[i, t]);  // Increase SD for low-weight obs
        y[i, t] ~ normal(mu_obs, weighted_sigma);
      }
    }
  }
}

generated quantities {
  // Posterior predictive checks (using unweighted predictions)
  array[N_players, T_max] real y_pred;

  for (i in 1:N_players) {
    for (t in 1:T_max) {
      if (t <= T_player[i] && observed[i, t] == 1) {
        real mu_obs = theta[i, t] + beta_attempts * log_attempts[i, t] + beta_home * is_home[i, t];
        y_pred[i, t] = normal_rng(mu_obs, R);  // Unweighted for validation
      } else {
        y_pred[i, t] = 0;
      }
    }
  }
}
