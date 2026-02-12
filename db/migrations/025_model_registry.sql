-- ============================================================
-- MODEL REGISTRY SCHEMA
-- Centralized tracking for all ML models and training runs
-- ============================================================

-- Create ml schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ml;

-- Main model registry table
CREATE TABLE IF NOT EXISTS ml.model_registry (
  model_id            SERIAL PRIMARY KEY,
  model_name          TEXT NOT NULL UNIQUE,           -- e.g., 'bnn_multioutput_v1'
  model_type          TEXT NOT NULL,                  -- 'bnn', 'xgboost', 'gnn', 'cvae', 'dqn', 'ppo', 'ensemble'
  model_family        TEXT NOT NULL,                  -- 'bayesian', 'gradient_boosting', 'deep_learning', 'reinforcement_learning'
  version             TEXT NOT NULL,                  -- 'v1', 'v2', 'v1.1', etc.

  -- File paths
  model_path          TEXT NOT NULL,                  -- Relative path from project root
  checkpoint_dir      TEXT,                           -- Directory for checkpoints (if applicable)
  metadata_path       TEXT,                           -- Path to metadata JSON file

  -- Status
  status              TEXT NOT NULL DEFAULT 'training', -- 'training', 'completed', 'failed', 'deprecated', 'production'
  is_production       BOOLEAN DEFAULT FALSE,          -- Currently deployed in production

  -- Training metadata
  training_start      TIMESTAMP WITH TIME ZONE,
  training_end        TIMESTAMP WITH TIME ZONE,
  training_duration_seconds REAL,

  -- Dataset info
  train_start_season  INTEGER,
  train_end_season    INTEGER,
  test_start_season   INTEGER,
  test_end_season     INTEGER,
  train_samples       INTEGER,
  test_samples        INTEGER,

  -- Hardware
  hardware            TEXT,                           -- 'RTX_4090', 'M4_CPU', etc.
  gpu_memory_gb       REAL,

  -- Git tracking
  git_commit_hash     TEXT,
  git_branch          TEXT,

  -- Timestamps
  created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Notes
  description         TEXT,
  notes               TEXT
);

-- Training runs table (multiple runs per model)
CREATE TABLE IF NOT EXISTS ml.training_runs (
  run_id              SERIAL PRIMARY KEY,
  model_id            INTEGER REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,
  run_name            TEXT,                           -- e.g., 'hyperparameter_search_trial_42'

  -- Run metadata
  started_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at        TIMESTAMP WITH TIME ZONE,
  duration_seconds    REAL,
  status              TEXT NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed', 'interrupted'
  exit_code           INTEGER,

  -- Hyperparameters (JSONB for flexibility)
  hyperparameters     JSONB,

  -- Training configuration
  epochs              INTEGER,
  batch_size          INTEGER,
  learning_rate       REAL,
  optimizer           TEXT,

  -- Hardware
  hardware_used       TEXT,
  gpu_utilization_avg REAL,
  gpu_memory_peak_gb  REAL,

  -- Results
  final_metrics       JSONB,                          -- Store all final metrics as JSON
  best_epoch          INTEGER,
  checkpoint_saved    BOOLEAN DEFAULT FALSE,

  -- Logs
  log_file_path       TEXT,
  error_message       TEXT
);

-- Model metrics table (time-series metrics during training)
CREATE TABLE IF NOT EXISTS ml.model_metrics (
  run_id              INTEGER REFERENCES ml.training_runs(run_id) ON DELETE CASCADE,
  timestamp           TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

  -- Metric metadata
  metric_name         TEXT NOT NULL,                  -- 'loss', 'mae', 'rmse', 'accuracy', 'auc', etc.
  metric_type         TEXT NOT NULL,                  -- 'train', 'validation', 'test'
  metric_value        REAL NOT NULL,

  -- Training progress
  epoch               INTEGER,
  step                INTEGER,

  -- Additional context
  metadata            JSONB,

  -- Composite primary key including partitioning column
  PRIMARY KEY (timestamp, run_id, metric_name, metric_type)
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable('ml.model_metrics', 'timestamp',
  if_not_exists => TRUE,
  migrate_data => TRUE
);

-- Model evaluation results
CREATE TABLE IF NOT EXISTS ml.model_evaluations (
  evaluation_id       SERIAL PRIMARY KEY,
  model_id            INTEGER REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,

  -- Evaluation metadata
  evaluation_name     TEXT NOT NULL,
  evaluation_type     TEXT NOT NULL,                  -- 'validation', 'test', 'production', 'backtest'
  evaluated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Dataset
  dataset_name        TEXT,
  dataset_size        INTEGER,
  date_range_start    DATE,
  date_range_end      DATE,

  -- Performance metrics
  metrics             JSONB NOT NULL,                 -- All evaluation metrics as JSON

  -- Comparison to baseline
  baseline_model_id   INTEGER REFERENCES ml.model_registry(model_id),
  improvement_pct     REAL,                           -- % improvement over baseline

  -- Statistical significance
  p_value             REAL,
  confidence_interval JSONB,                          -- {'lower': x, 'upper': y}

  -- Notes
  notes               TEXT
);

-- Ensemble configurations
CREATE TABLE IF NOT EXISTS ml.ensemble_configs (
  ensemble_id         SERIAL PRIMARY KEY,
  ensemble_name       TEXT NOT NULL UNIQUE,
  version             TEXT NOT NULL,

  -- Component models (array of model_ids)
  model_ids           INTEGER[] NOT NULL,

  -- Weighting strategy
  weighting_method    TEXT NOT NULL,                  -- 'equal', 'inverse_variance', 'stacked', 'correlation_adjusted'
  model_weights       JSONB,                          -- {model_id: weight}

  -- Meta-learner (if using stacked ensemble)
  meta_learner_type   TEXT,
  meta_learner_params JSONB,

  -- Performance
  performance_metrics JSONB,

  -- Status
  is_active           BOOLEAN DEFAULT TRUE,
  created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hyperparameter search results
CREATE TABLE IF NOT EXISTS ml.hyperparameter_searches (
  search_id           SERIAL PRIMARY KEY,
  model_type          TEXT NOT NULL,
  search_method       TEXT NOT NULL,                  -- 'grid', 'random', 'bayesian', 'optuna'

  -- Search configuration
  search_space        JSONB NOT NULL,                 -- Parameter ranges/options
  num_trials          INTEGER,
  max_duration_hours  REAL,

  -- Results
  best_trial_id       INTEGER,
  best_params         JSONB,
  best_metric_value   REAL,
  best_metric_name    TEXT,

  -- Execution
  started_at          TIMESTAMP WITH TIME ZONE,
  completed_at        TIMESTAMP WITH TIME ZONE,
  status              TEXT,                           -- 'running', 'completed', 'failed'

  -- All trials (array of trial results)
  trials              JSONB,

  -- Visualization data
  convergence_plot_path TEXT,
  importance_plot_path  TEXT
);

-- Model artifacts (checkpoints, weights, etc.)
CREATE TABLE IF NOT EXISTS ml.model_artifacts (
  artifact_id         SERIAL PRIMARY KEY,
  model_id            INTEGER REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,
  run_id              INTEGER REFERENCES ml.training_runs(run_id) ON DELETE CASCADE,

  -- Artifact metadata
  artifact_type       TEXT NOT NULL,                  -- 'checkpoint', 'final_model', 'weights', 'scaler', 'encoder'
  artifact_name       TEXT NOT NULL,
  file_path           TEXT NOT NULL,
  file_size_bytes     BIGINT,

  -- Versioning
  epoch               INTEGER,                        -- For checkpoints
  is_best             BOOLEAN DEFAULT FALSE,

  -- Storage
  storage_backend     TEXT DEFAULT 'local',           -- 'local', 'google_drive', 's3'
  cloud_path          TEXT,

  -- Checksums for integrity
  md5_hash            TEXT,
  sha256_hash         TEXT,

  created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Production deployment tracking
CREATE TABLE IF NOT EXISTS ml.production_deployments (
  deployment_id       SERIAL PRIMARY KEY,
  model_id            INTEGER REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,

  -- Deployment metadata
  deployment_name     TEXT NOT NULL,
  environment         TEXT NOT NULL,                  -- 'staging', 'production', 'a_b_test'
  deployed_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  undeployed_at       TIMESTAMP WITH TIME ZONE,

  -- Traffic
  traffic_percentage  REAL DEFAULT 100.0,             -- For A/B testing

  -- Performance in production
  total_predictions   BIGINT DEFAULT 0,
  avg_latency_ms      REAL,
  error_rate_pct      REAL,

  -- Monitoring
  monitoring_enabled  BOOLEAN DEFAULT TRUE,
  alert_threshold     JSONB,                          -- Performance degradation thresholds

  -- Status
  status              TEXT NOT NULL DEFAULT 'active', -- 'active', 'inactive', 'retired'
  rollback_reason     TEXT,

  notes               TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_registry_type ON ml.model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON ml.model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_production ON ml.model_registry(is_production);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON ml.training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_started ON ml.training_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_run ON ml.model_metrics(run_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_model_evaluations_model ON ml.model_evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_production_deployments_active ON ml.production_deployments(status) WHERE status = 'active';

-- Views for common queries
CREATE OR REPLACE VIEW ml.active_models AS
SELECT
  mr.*,
  COUNT(DISTINCT tr.run_id) as total_runs,
  MAX(tr.completed_at) as last_training_run,
  pd.deployment_id IS NOT NULL as is_deployed
FROM ml.model_registry mr
LEFT JOIN ml.training_runs tr ON mr.model_id = tr.model_id
LEFT JOIN ml.production_deployments pd ON mr.model_id = pd.model_id AND pd.status = 'active'
WHERE mr.status IN ('training', 'completed', 'production')
GROUP BY mr.model_id, pd.deployment_id;

CREATE OR REPLACE VIEW ml.model_leaderboard AS
SELECT
  mr.model_name,
  mr.model_type,
  mr.version,
  mr.status,
  me.metrics->'test_auc' as test_auc,
  me.metrics->'test_accuracy' as test_accuracy,
  me.metrics->'mae' as mae,
  me.metrics->'rmse' as rmse,
  me.evaluated_at,
  mr.is_production
FROM ml.model_registry mr
JOIN ml.model_evaluations me ON mr.model_id = me.model_id
WHERE me.evaluation_type = 'test'
ORDER BY me.evaluated_at DESC;

-- Functions
CREATE OR REPLACE FUNCTION ml.update_model_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER model_registry_update_timestamp
  BEFORE UPDATE ON ml.model_registry
  FOR EACH ROW
  EXECUTE FUNCTION ml.update_model_timestamp();

CREATE TRIGGER ensemble_configs_update_timestamp
  BEFORE UPDATE ON ml.ensemble_configs
  FOR EACH ROW
  EXECUTE FUNCTION ml.update_model_timestamp();

-- Grant permissions
GRANT ALL ON SCHEMA ml TO dro;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO dro;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO dro;

-- Comments
COMMENT ON TABLE ml.model_registry IS 'Central registry for all ML models';
COMMENT ON TABLE ml.training_runs IS 'Individual training runs with hyperparameters and results';
COMMENT ON TABLE ml.model_metrics IS 'Time-series metrics during training (TimescaleDB hypertable)';
COMMENT ON TABLE ml.model_evaluations IS 'Model evaluation results on different datasets';
COMMENT ON TABLE ml.ensemble_configs IS 'Ensemble model configurations and weights';
COMMENT ON TABLE ml.hyperparameter_searches IS 'Hyperparameter optimization search results';
COMMENT ON TABLE ml.model_artifacts IS 'Tracking for model files, checkpoints, and artifacts';
COMMENT ON TABLE ml.production_deployments IS 'Production deployment tracking and monitoring';
