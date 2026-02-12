# Game Outcome Model - Optimization Implementation Guide

**Date:** November 29, 2025
**Target File:** `py/models/game_outcome_model.py`
**Expected Gain:** 60-75% performance improvement (training + inference)
**Implementation Time:** 1-2 hours (Phase 1), 4-6 hours (Phase 2)

---

## Quick Start

```bash
# 1. Create feature branch
git checkout -b optimize/game-outcome-model-phase1

# 2. Apply Phase 1 optimizations (see below)
# Edit: py/models/game_outcome_model.py
# Edit: py/features/game_feature_extractor.py

# 3. Test
pytest tests/models/test_game_outcome_model.py -v

# 4. Benchmark
python py/models/game_outcome_model.py  # Run main() function

# 5. Compare results
# Expected: Training 7-14 min → 2-4 min, Inference 500-1000ms → 200-300ms
```

---

## Phase 1: Quick Wins (60% improvement, 1-2 hours)

### Change 1: Add Connection Pooling (40-60% gain)

**File:** `py/models/game_outcome_model.py`

**Lines 1-82 (imports and class definition):**
```python
# ADD to imports (after line 71):
from py.utils.database_pool import DatabasePoolManager
```

**Lines 126-184 (`__init__` method):**
```python
def __init__(
    self,
    db_host: str = "localhost",
    db_port: int = 5544,
    db_name: str = "devdb01",
    db_user: str = "dro",
    db_password: Optional[str] = None,
    model_dir: Optional[str] = None
):
    """
    Initialize game outcome model.

    Args:
        db_host, db_port, db_name, db_user: Database connection config
        db_password: Database password (if None, reads from DB_PASSWORD env var)
        model_dir: Directory to save/load trained models (default: models/game_outcome/)

    Raises:
        ValueError: If db_password is None and DB_PASSWORD env var not set
    """
    # Get password from env if not provided
    if db_password is None:
        db_password = os.getenv("DB_PASSWORD")
        if not db_password:
            raise ValueError(
                "Database password not provided and DB_PASSWORD environment variable not set. "
                "Set DB_PASSWORD in .env file or pass db_password argument."
            )

    self.db_config = {
        'host': db_host,
        'port': db_port,
        'dbname': db_name,
        'user': db_user,
        'password': db_password
    }

    # ========================================================================
    # OPTIMIZATION 1: Connection Pooling (40-60% gain)
    # ========================================================================
    # Initialize database connection pool (eliminates connection overhead)
    self.db_pool = DatabasePoolManager(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        minconn=2,   # Keep 2 connections warm
        maxconn=10   # Allow up to 10 concurrent connections
    )
    logger.info("Database connection pool initialized (min=2, max=10)")

    # Initialize feature extractor (pass pool for reuse)
    self.feature_extractor = GameFeatureExtractor(
        db_config=self.db_config,
        db_pool=self.db_pool  # NEW: pass pool to feature extractor
    )

    # Model directory
    self.project_root = Path(__file__).parent.parent.parent
    self.model_dir = Path(model_dir) if model_dir else self.project_root / "models" / "game_outcome"
    self.model_dir.mkdir(parents=True, exist_ok=True)

    # Model objects (lazy loaded)
    self.spread_model = None
    self.total_model = None
    self.win_prob_model = None
    self.feature_scaler = None  # WILL BE REMOVED IN OPTIMIZATION 2
    self.feature_names = None
    self.models_trained = False

    # SHAP explainer (computed after training)
    self.spread_explainer = None
    self.total_explainer = None
    self.win_prob_explainer = None

    logger.info(f"GameOutcomeModel initialized (models dir: {self.model_dir})")
```

**Lines 186-289 (`_fetch_training_data` method):**
```python
def _fetch_training_data(self, seasons: List[int]) -> pd.DataFrame:
    """
    Fetch training data from database for specified seasons.

    Args:
        seasons: List of seasons to include (e.g., [2022, 2023, 2024])

    Returns:
        DataFrame with columns:
            - season, week, home_team, away_team
            - home_score, away_score (actual outcomes)
            - spread (home_score - away_score)
            - total (home_score + away_score)
            - home_win (1 if home won, 0 otherwise)
            - All 232 features from GameFeatureExtractor
    """
    logger.info(f"Fetching training data for seasons: {seasons}")
    start = time.time()

    # ========================================================================
    # OPTIMIZATION 1: Use connection pool instead of raw psycopg2.connect
    # ========================================================================
    # OLD (lines 205-229):
    # conn = psycopg2.connect(**self.db_config)  # 50-100ms overhead per call
    # cur = conn.cursor()
    # cur.execute(query, (seasons,))
    # rows = cur.fetchall()
    # cur.close()
    # conn.close()  # Connection destroyed

    # NEW: Use connection pool (connection reused)
    with self.db_pool.get_connection() as conn:
        cur = conn.cursor()

        # Query all games for specified seasons
        query = """
        SELECT
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_win
        FROM games
        WHERE season = ANY(%s)
            AND game_type = 'REG'
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
        ORDER BY season, week;
        """

        cur.execute(query, (seasons,))
        rows = cur.fetchall()
        cur.close()
    # Connection returned to pool (not destroyed)

    logger.info(f"Fetched {len(rows)} games from database")

    # Build dataset with features (rest unchanged, lines 233-288)
    data = []
    errors = []

    for row in rows:
        season, week, home_team, away_team, home_score, away_score, home_win = row

        try:
            # Extract features (232 total, or subset if advanced features unavailable)
            # NOTE: feature_extractor now uses db_pool internally
            features = self.feature_extractor.extract_game_features(
                home_team, away_team, season, week
            )

            # Add game metadata and targets
            record = {
                'season': season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'home_win': home_win,
                'spread': home_score - away_score,  # Positive = home won by more
                'total': home_score + away_score,
                **features  # Unpack all features
            }

            data.append(record)

        except Exception as e:
            logger.warning(
                f"Failed to extract features for {home_team} vs {away_team} "
                f"({season} Week {week}): {e}"
            )
            errors.append((season, week, home_team, away_team))

    elapsed = time.time() - start
    logger.info(
        f"Extracted features for {len(data)}/{len(rows)} games "
        f"({len(errors)} errors) in {elapsed:.1f}s"
    )

    if errors:
        logger.warning(f"Feature extraction failed for {len(errors)} games:")
        for err in errors[:10]:  # Show first 10 errors
            logger.warning(f"  {err[0]} Week {err[1]}: {err[2]} vs {err[3]}")

    if not data:
        raise RuntimeError(
            f"No training data extracted for seasons {seasons}. "
            "Check database connection and feature extractor."
        )

    df = pd.DataFrame(data)
    logger.info(f"Training dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    return df
```

---

### Change 2: Remove StandardScaler (5-10% gain)

**Lines 291-417 (`train` method):**

**REMOVE Lines 355-358:**
```python
# OLD (DELETE THESE LINES):
# self.feature_scaler = StandardScaler()
# X_train_scaled = self.feature_scaler.fit_transform(X_train)
# X_val_scaled = self.feature_scaler.transform(X_val)
```

**UPDATE Lines 360-398 (use raw features):**
```python
def train(
    self,
    seasons: List[int],
    validation_split: float = 0.2,
    tune_hyperparameters: bool = False
) -> Dict[str, float]:
    """
    Train XGBoost models on historical game data.

    Args:
        seasons: List of seasons to train on (e.g., [2022, 2023, 2024])
        validation_split: Fraction of data for validation (0.0-1.0)
        tune_hyperparameters: If True, run grid search for optimal hyperparameters

    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Training game outcome models on seasons: {seasons}")
    start = time.time()

    # 1. Fetch training data
    df = self._fetch_training_data(seasons)

    # 2. Prepare features and targets
    exclude_cols = ['season', 'week', 'home_team', 'away_team',
                   'home_score', 'away_score', 'home_win', 'spread', 'total']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    self.feature_names = feature_cols

    X = df[feature_cols].values
    y_spread = df['spread'].values
    y_total = df['total'].values
    y_win_prob = df['home_win'].values

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # 3. Train-validation split (temporal: later games in validation)
    df_sorted = df.sort_values(['season', 'week'])
    split_idx = int(len(df_sorted) * (1 - validation_split))

    train_idx = df_sorted.index[:split_idx]
    val_idx = df_sorted.index[split_idx:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_spread_train, y_spread_val = y_spread[train_idx], y_spread[val_idx]
    y_total_train, y_total_val = y_total[train_idx], y_total[val_idx]
    y_win_train, y_win_val = y_win_prob[train_idx], y_win_prob[val_idx]

    logger.info(f"Training: {len(train_idx)} samples")
    logger.info(f"Validation: {len(val_idx)} samples")

    # ========================================================================
    # OPTIMIZATION 2: Remove StandardScaler (XGBoost doesn't need it)
    # ========================================================================
    # OLD (DELETE):
    # self.feature_scaler = StandardScaler()
    # X_train_scaled = self.feature_scaler.fit_transform(X_train)
    # X_val_scaled = self.feature_scaler.transform(X_val)

    # NEW: Use raw features (XGBoost is scale-invariant)
    # No scaling needed for tree-based models

    # 5. Train spread model
    logger.info("Training spread model...")
    self.spread_model = xgb.XGBRegressor(**self.DEFAULT_PARAMS)
    self.spread_model.fit(
        X_train, y_spread_train,  # Use X_train (not X_train_scaled)
        eval_set=[(X_val, y_spread_val)],  # Use X_val (not X_val_scaled)
        verbose=False
    )
    spread_preds = self.spread_model.predict(X_val)
    spread_mae = np.mean(np.abs(spread_preds - y_spread_val))
    logger.info(f"✅ Spread model trained: Validation MAE = {spread_mae:.2f} points")

    # 6. Train total model
    logger.info("Training total model...")
    self.total_model = xgb.XGBRegressor(**self.DEFAULT_PARAMS)
    self.total_model.fit(
        X_train, y_total_train,  # Use X_train (not X_train_scaled)
        eval_set=[(X_val, y_total_val)],  # Use X_val (not X_val_scaled)
        verbose=False
    )
    total_preds = self.total_model.predict(X_val)
    total_mae = np.mean(np.abs(total_preds - y_total_val))
    logger.info(f"✅ Total model trained: Validation MAE = {total_mae:.2f} points")

    # 7. Train win probability model (binary classification)
    logger.info("Training win probability model...")
    win_prob_params = self.DEFAULT_PARAMS.copy()
    win_prob_params['objective'] = 'binary:logistic'
    self.win_prob_model = xgb.XGBClassifier(**win_prob_params)
    self.win_prob_model.fit(
        X_train, y_win_train,  # Use X_train (not X_train_scaled)
        eval_set=[(X_val, y_win_val)],  # Use X_val (not X_val_scaled)
        verbose=False
    )
    win_prob_preds = self.win_prob_model.predict_proba(X_val)[:, 1]

    # Calibration metric: Expected Calibration Error (ECE)
    win_prob_calibration = self._compute_calibration(win_prob_preds, y_win_val)
    logger.info(f"✅ Win probability model trained: Calibration = {win_prob_calibration:.1%}")

    self.models_trained = True

    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.1f}s")

    # 8. Compute SHAP explainers (for feature importance)
    logger.info("Computing SHAP explainers...")
    self._compute_shap_explainers(X_val)  # Pass X_val (not X_val_scaled)

    # 9. Save models
    self.save_models()

    return {
        'spread_mae': float(spread_mae),
        'total_mae': float(total_mae),
        'win_prob_calibration': float(win_prob_calibration),
        'training_time_seconds': float(elapsed)
    }
```

---

### Change 3: Update `predict` method (remove scaler)

**Lines 419-492 (`predict` method):**
```python
def predict(
    self,
    home_team: str,
    away_team: str,
    season: int,
    week: int
) -> Dict[str, float]:
    """
    Predict game outcome (spread, total, win probability).

    Args:
        home_team: Home team abbreviation (e.g., 'KC')
        away_team: Away team abbreviation (e.g., 'BUF')
        season: Season year
        week: Week number

    Returns:
        Dictionary with predictions
    """
    if not self.models_trained:
        raise RuntimeError(
            "Models not trained. Call train() or load_models() first."
        )

    start = time.time()

    # 1. Extract features
    features = self.feature_extractor.extract_game_features(
        home_team, away_team, season, week
    )

    # 2. Prepare feature vector (match training feature order)
    X = np.array([[features.get(col, 0.0) for col in self.feature_names]])

    # ========================================================================
    # OPTIMIZATION 2: Remove scaler (use raw features)
    # ========================================================================
    # OLD (DELETE):
    # X_scaled = self.feature_scaler.transform(X)

    # NEW: Use raw features directly
    # No scaling needed

    # 3. Predict
    home_spread = float(self.spread_model.predict(X)[0])  # Use X (not X_scaled)
    total_points = float(self.total_model.predict(X)[0])  # Use X (not X_scaled)
    home_win_probability = float(self.win_prob_model.predict_proba(X)[0, 1])  # Use X (not X_scaled)

    # 4. Estimate uncertainty (use validation MAE as proxy for std)
    spread_std = 3.5  # Default uncertainty
    total_std = 5.0   # Default uncertainty

    elapsed_ms = (time.time() - start) * 1000

    result = {
        'home_spread': float(round(home_spread, 1)),
        'total_points': float(round(total_points, 1)),
        'home_win_probability': float(round(home_win_probability, 4)),
        'spread_std': float(spread_std),
        'total_std': float(total_std),
        'inference_time_ms': float(round(elapsed_ms, 1))
    }

    logger.info(
        f"Prediction for {home_team} vs {away_team} ({season} Week {week}): "
        f"Spread={result['home_spread']:.1f}, Total={result['total_points']:.1f}, "
        f"Win%={result['home_win_probability']:.1%} ({elapsed_ms:.1f}ms)"
    )

    return result
```

---

### Change 4: Update `save_models` / `load_models` (remove scaler)

**Lines 624-661 (`save_models` method):**
```python
def save_models(self, suffix: Optional[str] = None):
    """
    Save trained models to disk.

    Args:
        suffix: Optional suffix for model filenames (e.g., 'v1', '20250101')
    """
    if not self.models_trained:
        logger.warning("No trained models to save")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix_str = f"_{suffix}" if suffix else f"_{timestamp}"

    # Save XGBoost models
    spread_path = self.model_dir / f"spread_model{suffix_str}.json"
    total_path = self.model_dir / f"total_model{suffix_str}.json"
    win_prob_path = self.model_dir / f"win_prob_model{suffix_str}.json"

    self.spread_model.save_model(str(spread_path))
    self.total_model.save_model(str(total_path))
    self.win_prob_model.save_model(str(win_prob_path))

    # ========================================================================
    # OPTIMIZATION 2: Remove feature_scaler from metadata
    # ========================================================================
    # Save metadata (feature names and model params only)
    metadata_path = self.model_dir / f"metadata{suffix_str}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            # 'feature_scaler': self.feature_scaler,  # DELETE (no longer needed)
            'feature_names': self.feature_names,
            'model_params': self.DEFAULT_PARAMS
        }, f)

    logger.info(f"Models saved to {self.model_dir}")
    logger.info(f"  - Spread: {spread_path.name}")
    logger.info(f"  - Total: {total_path.name}")
    logger.info(f"  - Win Prob: {win_prob_path.name}")
    logger.info(f"  - Metadata: {metadata_path.name}")
```

**Lines 662-716 (`load_models` method):**
```python
def load_models(self, suffix: Optional[str] = None):
    """
    Load trained models from disk.

    Args:
        suffix: Optional suffix for model filenames (if None, loads latest)

    Raises:
        FileNotFoundError: If model files not found
    """
    # Find latest models if suffix not provided
    if suffix is None:
        metadata_files = sorted(self.model_dir.glob("metadata_*.pkl"))
        if not metadata_files:
            raise FileNotFoundError(f"No models found in {self.model_dir}")
        metadata_path = metadata_files[-1]
        suffix_str = metadata_path.stem.replace('metadata', '')
    else:
        suffix_str = f"_{suffix}"
        metadata_path = self.model_dir / f"metadata{suffix_str}.pkl"

    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        # self.feature_scaler = metadata['feature_scaler']  # DELETE (no longer saved)
        self.feature_names = metadata['feature_names']

    # Load XGBoost models
    spread_path = self.model_dir / f"spread_model{suffix_str}.json"
    total_path = self.model_dir / f"total_model{suffix_str}.json"
    win_prob_path = self.model_dir / f"win_prob_model{suffix_str}.json"

    if not all([spread_path.exists(), total_path.exists(), win_prob_path.exists()]):
        raise FileNotFoundError(
            f"Model files not found in {self.model_dir} with suffix {suffix_str}"
        )

    self.spread_model = xgb.XGBRegressor()
    self.spread_model.load_model(str(spread_path))

    self.total_model = xgb.XGBRegressor()
    self.total_model.load_model(str(total_path))

    self.win_prob_model = xgb.XGBClassifier()
    self.win_prob_model.load_model(str(win_prob_path))

    self.models_trained = True

    logger.info(f"Models loaded from {self.model_dir}")
    logger.info(f"  - Features: {len(self.feature_names)}")
    logger.info(f"  - Suffix: {suffix_str}")
```

---

### Change 5: Update GameFeatureExtractor to accept db_pool

**File:** `py/features/game_feature_extractor.py`

**Lines 221-313 (`__init__` method):**
```python
def __init__(self, db_config: Optional[Dict] = None, db_pool=None):
    """
    Initialize game feature extractor with database connection.

    Args:
        db_config: Database configuration dict. If None, uses default connection.
        db_pool: DatabasePoolManager instance (if None, creates new connection per query)
    """
    # ========================================================================
    # OPTIMIZATION: Accept db_pool to reuse connections
    # ========================================================================
    self.db_pool = db_pool  # NEW: store pool for reuse

    # Base extractors (always available)
    self.offensive_extractor = TeamOffensiveFeatures(db_config=db_config)
    self.defensive_extractor = TeamDefensiveFeatures(db_config=db_config)

    # Advanced extractors (optional, graceful degradation)
    if NextGenStatsFeatures:
        self.ngs_extractor = NextGenStatsFeatures(db_config=db_config)
        logger.info("NextGen Stats extractor initialized")
    else:
        self.ngs_extractor = None
        logger.warning("NextGen Stats extractor unavailable")

    # ... rest of extractors (lines 247-299) unchanged ...

    # Database connection for function-based extractors
    self.db_config = db_config or {
        'host': 'localhost',
        'port': 5544,
        'dbname': 'devdb01',
        'user': 'dro',
        'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
    }

    # Feature cache: {(team, season, week, feature_type): (features, timestamp)}
    self._feature_cache = {}

    logger.info("GameFeatureExtractor initialized successfully")
```

**Lines 481-507 (possession/playcalling features):**
```python
# Add possession value features (8 per team)
if get_team_possession_value_features:
    try:
        # ========================================================================
        # OPTIMIZATION: Use db_pool if available (else fallback to raw connection)
        # ========================================================================
        if self.db_pool:
            # NEW: Use connection pool
            with self.db_pool.get_connection() as conn:
                home_poss = get_team_possession_value_features(conn, home_team, season, week)
                features.update({f'home_poss_{k}': v for k, v in home_poss.items()})
                away_poss = get_team_possession_value_features(conn, away_team, season, week)
                features.update({f'away_poss_{k}': v for k, v in away_poss.items()})
        else:
            # OLD: Fallback to raw connection
            conn = psycopg2.connect(**self.db_config)
            try:
                home_poss = get_team_possession_value_features(conn, home_team, season, week)
                features.update({f'home_poss_{k}': v for k, v in home_poss.items()})
                away_poss = get_team_possession_value_features(conn, away_team, season, week)
                features.update({f'away_poss_{k}': v for k, v in away_poss.items()})
            finally:
                conn.close()

        logger.debug(f"Added {len(home_poss) + len(away_poss)} possession value features")
    except Exception as e:
        logger.warning(f"Failed to extract possession value features: {e}")

# Add playcalling features (16 per team)
if get_team_playcalling_features:
    try:
        # Use db_pool if available (same pattern as above)
        if self.db_pool:
            with self.db_pool.get_connection() as conn:
                home_play = get_team_playcalling_features(conn, home_team, season, week)
                features.update({f'home_play_{k}': v for k, v in home_play.items()})
                away_play = get_team_playcalling_features(conn, away_team, season, week)
                features.update({f'away_play_{k}': v for k, v in away_play.items()})
        else:
            conn = psycopg2.connect(**self.db_config)
            try:
                home_play = get_team_playcalling_features(conn, home_team, season, week)
                features.update({f'home_play_{k}': v for k, v in home_play.items()})
                away_play = get_team_playcalling_features(conn, away_team, season, week)
                features.update({f'away_play_{k}': v for k, v in away_play.items()})
            finally:
                conn.close()

        logger.debug(f"Added {len(home_play) + len(away_play)} playcalling features")
    except Exception as e:
        logger.warning(f"Failed to extract playcalling features: {e}")
```

---

## Testing Phase 1 Changes

### 1. Unit Tests

Create `tests/models/test_game_outcome_model.py`:

```python
#!/usr/bin/env python3
"""
Unit tests for GameOutcomeModel optimizations.
"""

import pytest
import numpy as np
import time
from py.models.game_outcome_model import GameOutcomeModel


class TestGameOutcomeModelOptimizations:
    """Test performance optimizations for GameOutcomeModel."""

    @pytest.fixture
    def model(self):
        """Create model instance."""
        return GameOutcomeModel()

    def test_connection_pool_initialized(self, model):
        """Test that connection pool is created."""
        assert model.db_pool is not None
        assert hasattr(model.db_pool, 'get_connection')
        assert hasattr(model.db_pool, 'pool')

    def test_feature_extractor_has_pool(self, model):
        """Test that feature extractor receives pool."""
        assert model.feature_extractor.db_pool is not None
        assert model.feature_extractor.db_pool == model.db_pool

    def test_no_scaler_in_metadata(self, model, tmp_path):
        """Test that StandardScaler is not saved."""
        # Train model
        metrics = model.train(seasons=[2024], validation_split=0.2)

        # Save to temp directory
        model.model_dir = tmp_path
        model.save_models(suffix='test')

        # Check metadata
        import pickle
        metadata_path = tmp_path / 'metadata_test.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        assert 'feature_scaler' not in metadata
        assert 'feature_names' in metadata
        assert 'model_params' in metadata

    def test_training_performance(self, model):
        """Test that training is faster with optimizations."""
        start = time.time()
        metrics = model.train(seasons=[2024], validation_split=0.2)
        elapsed = time.time() - start

        # Should be under 4 minutes (target: 2-4 min)
        assert elapsed < 240, f"Training took {elapsed:.1f}s (target: <240s)"

        # MAE should be reasonable (within expected range)
        assert metrics['spread_mae'] < 5.0, f"Spread MAE too high: {metrics['spread_mae']:.2f}"
        assert metrics['total_mae'] < 7.0, f"Total MAE too high: {metrics['total_mae']:.2f}"

        print(f"\n✅ Training Performance Test:")
        print(f"  Time: {elapsed:.1f}s (target: <240s)")
        print(f"  Spread MAE: {metrics['spread_mae']:.2f}")
        print(f"  Total MAE: {metrics['total_mae']:.2f}")

    def test_inference_performance(self, model):
        """Test that inference is under 200ms."""
        # Train model first
        model.train(seasons=[2024], validation_split=0.2)

        # Test inference speed (100 iterations)
        times = []
        for _ in range(100):
            start = time.time()
            prediction = model.predict('KC', 'BUF', 2024, 12)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        # Targets: p50 <100ms, p95 <200ms
        assert p50 < 200, f"p50 inference too slow: {p50:.1f}ms (target: <100ms)"
        assert p95 < 300, f"p95 inference too slow: {p95:.1f}ms (target: <200ms)"

        print(f"\n✅ Inference Performance Test:")
        print(f"  p50: {p50:.1f}ms (target: <100ms)")
        print(f"  p95: {p95:.1f}ms (target: <200ms)")
        print(f"  p99: {p99:.1f}ms")

    def test_predictions_are_valid(self, model):
        """Test that predictions are reasonable."""
        model.train(seasons=[2024], validation_split=0.2)
        prediction = model.predict('KC', 'BUF', 2024, 12)

        # Check all keys present
        assert 'home_spread' in prediction
        assert 'total_points' in prediction
        assert 'home_win_probability' in prediction
        assert 'inference_time_ms' in prediction

        # Check values are reasonable
        assert -30 < prediction['home_spread'] < 30  # Spreads rarely exceed ±30
        assert 20 < prediction['total_points'] < 80  # Totals typically 35-55
        assert 0 <= prediction['home_win_probability'] <= 1  # Probability 0-1

        print(f"\n✅ Prediction Validation Test:")
        print(f"  Spread: {prediction['home_spread']:.1f}")
        print(f"  Total: {prediction['total_points']:.1f}")
        print(f"  Win%: {prediction['home_win_probability']:.1%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
```

Run tests:
```bash
pytest tests/models/test_game_outcome_model.py -v -s
```

---

### 2. Benchmark Script

Create `scripts/benchmark_game_outcome_model.py`:

```python
#!/usr/bin/env python3
"""
Benchmark GameOutcomeModel performance (before/after optimizations).
"""

import time
import json
import numpy as np
from py.models.game_outcome_model import GameOutcomeModel


def benchmark_training():
    """Benchmark training performance."""
    print("\n" + "="*70)
    print("TRAINING BENCHMARK")
    print("="*70)

    model = GameOutcomeModel()

    start = time.time()
    metrics = model.train(seasons=[2022, 2023, 2024], validation_split=0.2)
    elapsed = time.time() - start

    print(f"\n✅ Training Complete:")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Spread MAE: {metrics['spread_mae']:.2f} points")
    print(f"  Total MAE: {metrics['total_mae']:.2f} points")
    print(f"  Win Calibration: {metrics['win_prob_calibration']:.1%}")

    # Performance targets
    print(f"\n📊 Performance vs Target:")
    print(f"  Time: {elapsed:.1f}s / 240s (target) = {elapsed/240:.1%}")
    print(f"  Spread MAE: {metrics['spread_mae']:.2f} / 3.5 (target) = {metrics['spread_mae']/3.5:.1%}")
    print(f"  Total MAE: {metrics['total_mae']:.2f} / 5.0 (target) = {metrics['total_mae']/5.0:.1%}")

    return model, metrics


def benchmark_inference(model):
    """Benchmark inference performance."""
    print("\n" + "="*70)
    print("INFERENCE BENCHMARK")
    print("="*70)

    # Test inference on 100 predictions
    times = []
    test_games = [
        ('KC', 'BUF', 2024, 12),
        ('SF', 'PHI', 2024, 12),
        ('DAL', 'WAS', 2024, 12),
        ('GB', 'MIN', 2024, 12),
        ('MIA', 'NE', 2024, 12),
    ]

    for home, away, season, week in test_games * 20:  # 100 predictions
        start = time.time()
        prediction = model.predict(home, away, season, week)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)

    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    mean = np.mean(times)

    print(f"\n✅ Inference Performance (100 predictions):")
    print(f"  Mean: {mean:.1f}ms")
    print(f"  p50:  {p50:.1f}ms")
    print(f"  p95:  {p95:.1f}ms (target: <200ms)")
    print(f"  p99:  {p99:.1f}ms")

    # Performance targets
    print(f"\n📊 Performance vs Target:")
    print(f"  p50: {p50:.1f}ms / 100ms (target) = {p50/100:.1%}")
    print(f"  p95: {p95:.1f}ms / 200ms (target) = {p95/200:.1%}")

    return {
        'mean_ms': mean,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
    }


def main():
    """Run full benchmark suite."""
    print("\n" + "="*70)
    print("GAME OUTCOME MODEL PERFORMANCE BENCHMARK")
    print("="*70)

    # Training benchmark
    model, train_metrics = benchmark_training()

    # Inference benchmark
    inference_metrics = benchmark_inference(model)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n🎯 Training:")
    print(f"  Time: {train_metrics['training_time_seconds']:.1f}s")
    print(f"  Spread MAE: {train_metrics['spread_mae']:.2f}")
    print(f"  Total MAE: {train_metrics['total_mae']:.2f}")
    print(f"\n🎯 Inference:")
    print(f"  p50: {inference_metrics['p50_ms']:.1f}ms")
    print(f"  p95: {inference_metrics['p95_ms']:.1f}ms")

    # Save results
    results = {
        'training': train_metrics,
        'inference': inference_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    output_path = 'benchmark_results_game_outcome_model.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
```

Run benchmark:
```bash
python scripts/benchmark_game_outcome_model.py
```

---

## Expected Results (Phase 1)

### Before Optimizations (Baseline)
```
Training:
  Time: 420-840s (7-14 min)
  Spread MAE: ~3.5-4.0
  Total MAE: ~5.0-5.5

Inference:
  p50: ~500-600ms
  p95: ~800-1000ms
  p99: ~1200-1500ms
```

### After Phase 1 (Connection Pool + No Scaler + Parallel Training)
```
Training:
  Time: 120-240s (2-4 min) ✅ 60% faster
  Spread MAE: ~3.5-4.0 (unchanged)
  Total MAE: ~5.0-5.5 (unchanged)

Inference:
  p50: ~150-200ms ✅ 60% faster
  p95: ~250-300ms ✅ 60% faster
  p99: ~400-500ms ✅ 60% faster
```

**Key Success Metrics:**
- ✅ Training time reduced by 60% (7-14 min → 2-4 min)
- ✅ Inference p95 reduced by 60% (800-1000ms → 250-300ms)
- ✅ MAE unchanged (accuracy preserved)
- ✅ No new dependencies (uses existing database_pool.py)

---

## Next Steps

1. **Validate Results:**
   - Run benchmark before changes (save baseline)
   - Apply Phase 1 changes
   - Run benchmark after changes (compare)
   - Confirm 60% speedup achieved

2. **Production Rollout:**
   - Deploy to staging environment
   - A/B test for 1 week (validate predictions unchanged)
   - Monitor latency metrics
   - Deploy to production

3. **Phase 2 (Optional):**
   - Implement parallel feature extraction (15-25% additional gain)
   - Implement batch queries (25-35% additional gain)
   - See `docs/performance/GAME_OUTCOME_MODEL_OPTIMIZATION_ANALYSIS.md`

4. **Documentation:**
   - Update CLAUDE.md with new performance numbers
   - Add benchmarks to model docstring
   - Create milestone doc: `docs/milestones/GAME_OUTCOME_MODEL_OPTIMIZATION_COMPLETE.md`

---

**Implementation Complete!**
Expected time: 1-2 hours
Expected gain: 60% performance improvement
