/**
 * Hierarchical Bayesian State-Space Model for NFL Player Passing Yards
 *
 * Model: AR(1) state-space with position-level hierarchy
 *
 * State Equation (skill evolution):
 *   θ_{i,t} = φ * θ_{i,t-1} + ω_{i,t},  ω_{i,t} ~ N(0, Q_p)
 *
 * Observation Equation (measured performance):
 *   y_{i,t} = θ_{i,t} + X_{i,t} * β + ε_{i,t},  ε_{i,t} ~ N(0, R)
 *
 * Hierarchy:
 *   League → Position Group → Player → Game
 *
 * Author: Claude Code (Sonnet 4.5)
 * Date: November 8, 2025
 * Version: 3.0 - Proper Bayesian implementation (replaces LOESS approximation)
 */

data {
  // Dimensions
  int<lower=1> N_players;          // Number of players
  int<lower=1> T_max;              // Maximum games per player
  int<lower=1> N_positions;        // Number of position groups (QB=1, RB=2, WR=3, TE=4)
  int<lower=1> N_covariates;       // Number of covariates (attempts, home/away, etc.)

  // Player metadata
  array[N_players] int<lower=1,upper=N_positions> position_group;  // Position group index per player
  array[N_players] int<lower=1,upper=T_max> T_player;  // Number of games per player

  // Observed data (ragged array - players have different game counts)
  array[N_players, T_max] real y;  // Observed log(passing_yards + 1)
  array[N_players, T_max] real<lower=0> attempts;  // Pass attempts (covariate)
  array[N_players, T_max] int<lower=0,upper=1> is_home;  // Home indicator

  // Missing data indicator (1 = observed, 0 = missing)
  array[N_players, T_max] int<lower=0,upper=1> observed;
}

transformed data {
  // Log-transform attempts for better scaling
  array[N_players, T_max] real log_attempts;

  for (i in 1:N_players) {
    for (t in 1:T_max) {
      log_attempts[i, t] = log(attempts[i, t] + 1);
    }
  }
}

parameters {
  // ============================================================
  // HYPERPRIORS (LEAGUE LEVEL)
  // ============================================================
  real mu_league;                  // League-wide mean skill (log-yards)
  real<lower=0> sigma_league;      // League-wide skill volatility

  // ============================================================
  // POSITION GROUP LEVEL
  // ============================================================
  vector[N_positions] mu_position;  // Position group means
  vector<lower=0>[N_positions] Q_position;  // Position-specific process noise (skill drift rate)

  // ============================================================
  // GLOBAL PARAMETERS
  // ============================================================
  // AR(1) coefficient (skill persistence)
  real<lower=0, upper=1> phi;  // φ ~ 0.9 means 90% skill carries forward

  // Measurement noise (game-to-game variation)
  real<lower=0> R;

  // Fixed effects (covariates)
  real beta_attempts;  // Effect of pass attempts
  real beta_home;      // Home field advantage

  // ============================================================
  // PLAYER LEVEL (STATE-SPACE)
  // ============================================================
  // Initial player skills (t=1)
  vector[N_players] theta_raw_0;  // Non-centered parameterization (for better sampling)

  // Process innovations (skill changes over time)
  array[N_players, T_max-1] real omega_raw;  // Non-centered innovations
}

transformed parameters {
  // ============================================================
  // STATE-SPACE: PLAYER SKILLS OVER TIME
  // ============================================================
  array[N_players, T_max] real theta;  // Time-varying player skills

  // Initialize first time period (non-centered)
  for (i in 1:N_players) {
    theta[i, 1] = mu_position[position_group[i]] + sigma_league * theta_raw_0[i];
  }

  // State evolution: AR(1) random walk
  for (i in 1:N_players) {
    for (t in 2:T_max) {
      if (t <= T_player[i]) {  // Only evolve states for observed games
        theta[i, t] = phi * theta[i, t-1] + sqrt(Q_position[position_group[i]]) * omega_raw[i, t-1];
      } else {
        theta[i, t] = theta[i, t-1];  // Freeze skill after last game
      }
    }
  }
}

model {
  // ============================================================
  // PRIORS
  // ============================================================

  // League-level hyperpriors
  mu_league ~ normal(5.5, 0.3);  // log(245 yards) ≈ 5.5
  sigma_league ~ normal(0, 0.2);  // Weakly informative

  // Position-level priors
  mu_position ~ normal(mu_league, sigma_league);  // Hierarchical shrinkage
  Q_position ~ inv_gamma(3, 0.1);  // Process noise: mean ≈ 0.05, std ≈ 0.04

  // Global parameters
  phi ~ beta(18, 2);  // Prior: E[φ] = 0.9 (strong persistence)
  R ~ normal(0, 0.3);  // Measurement noise

  // Fixed effects
  beta_attempts ~ normal(1.0, 0.2);  // Prior: more attempts → more yards
  beta_home ~ normal(0.03, 0.02);  // Prior: slight home advantage

  // Player-level non-centered priors
  theta_raw_0 ~ std_normal();  // Standard normal (non-centered)

  for (i in 1:N_players) {
    for (t in 1:(T_max-1)) {
      omega_raw[i, t] ~ std_normal();  // Standard normal innovations
    }
  }

  // ============================================================
  // LIKELIHOOD
  // ============================================================
  for (i in 1:N_players) {
    for (t in 1:T_player[i]) {
      if (observed[i, t] == 1) {
        // Linear predictor
        real mu_obs = theta[i, t] +
                      beta_attempts * log_attempts[i, t] +
                      beta_home * is_home[i, t];

        // Log-normal likelihood
        y[i, t] ~ normal(mu_obs, R);
      }
    }
  }
}

generated quantities {
  // ============================================================
  // POSTERIOR PREDICTIVE CHECKS & DIAGNOSTICS
  // ============================================================

  // Posterior predictive samples (for one-step-ahead forecasting)
  array[N_players, T_max] real y_pred;

  // Log-likelihood (for model comparison via LOO-CV)
  array[N_players, T_max] real log_lik;

  for (i in 1:N_players) {
    for (t in 1:T_player[i]) {
      real mu_obs = theta[i, t] +
                    beta_attempts * log_attempts[i, t] +
                    beta_home * is_home[i, t];

      if (observed[i, t] == 1) {
        y_pred[i, t] = normal_rng(mu_obs, R);
        log_lik[i, t] = normal_lpdf(y[i, t] | mu_obs, R);
      } else {
        y_pred[i, t] = -999;  // Sentinel value for missing data
        log_lik[i, t] = 0;
      }
    }

    // Sentinel values for unobserved time periods
    for (t in (T_player[i]+1):T_max) {
      y_pred[i, t] = -999;
      log_lik[i, t] = 0;
    }
  }
}
