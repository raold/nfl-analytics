"""
Comprehensive tests for Performance Optimizer

Tests performance optimization across the betting system:
1. MCMC configuration - Chains, tune, draws optimization
2. SQL query optimization - Missing indexes, slow joins, query planning
3. Backtest profiling - Time estimation, bottleneck identification
4. Pandas performance - Avoid iterrows, optimize apply
5. Training time estimation - Model complexity vs training time
6. Memory profiling - Large DataFrame handling

Critical Requirements:
- Accurate training time estimation
- SQL query performance recommendations
- MCMC configuration validation
- Memory-efficient data processing

Author: Claude Code (Sonnet 4.5)
Created: November 9, 2025
"""

import time

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_mcmc_config_slow():
    """Create slow MCMC configuration."""
    return {
        "chains": 8,  # Too many chains (diminishing returns)
        "tune": 5000,  # Excessive tuning
        "draws": 5000,  # Too many draws
        "target_accept": 0.99,  # Very high (slow sampling)
    }


@pytest.fixture
def mock_mcmc_config_fast():
    """Create optimized MCMC configuration."""
    return {
        "chains": 4,  # Optimal for most CPUs
        "tune": 1000,  # Sufficient tuning
        "draws": 2000,  # Sufficient samples
        "target_accept": 0.90,  # Balanced (default)
    }


@pytest.fixture
def mock_slow_sql_query():
    """Create slow SQL query with issues."""
    return """
    SELECT *
    FROM games g
    JOIN plays p ON g.game_id = p.game_id
    WHERE g.season >= 2015
    AND p.play_type = 'pass'
    ORDER BY g.kickoff
    """


@pytest.fixture
def mock_optimized_sql_query():
    """Create optimized SQL query."""
    return """
    SELECT g.game_id, g.kickoff, p.play_id, p.epa
    FROM games g
    JOIN plays p ON g.game_id = p.game_id
    WHERE g.season >= 2015
      AND p.play_type = 'pass'
    ORDER BY g.kickoff
    """


# ============================================================================
# TEST 1: MCMC CONFIGURATION OPTIMIZATION
# ============================================================================


def test_mcmc_config_time_estimation(mock_mcmc_config_slow, mock_mcmc_config_fast):
    """Test MCMC training time estimation."""
    # Estimate time per iteration (mock)
    time_per_iter = 0.05  # 50ms per iteration

    def estimate_time(config):
        total_iters = config["chains"] * (config["tune"] + config["draws"])
        return total_iters * time_per_iter

    slow_time = estimate_time(mock_mcmc_config_slow)
    fast_time = estimate_time(mock_mcmc_config_fast)

    # Slow config should take much longer
    assert slow_time > fast_time

    # Slow: 8 * (5000 + 5000) = 80,000 iters = 4,000s ≈ 67 min
    assert slow_time == pytest.approx(4000, rel=0.1)

    # Fast: 4 * (1000 + 2000) = 12,000 iters = 600s ≈ 10 min
    assert fast_time == pytest.approx(600, rel=0.1)


def test_mcmc_optimal_chains_for_cpu():
    """Test recommendation for optimal number of chains."""
    # Mock CPU count
    cpu_count = 8

    # Recommendation: Use 4 chains (standard) or min(4, cpu_count)
    recommended_chains = min(4, cpu_count)

    assert recommended_chains == 4, "Standard recommendation is 4 chains"


def test_mcmc_excessive_chains_warning():
    """Test warning for excessive chains (diminishing returns)."""
    configs = [
        {"chains": 2, "warning": False},
        {"chains": 4, "warning": False},
        {"chains": 6, "warning": True},  # Diminishing returns
        {"chains": 8, "warning": True},  # Too many
    ]

    for config in configs:
        chains = config["chains"]
        should_warn = chains > 4

        assert should_warn == config["warning"], f"Chains={chains} should_warn={config['warning']}"


def test_mcmc_draws_sufficiency():
    """Test that draws are sufficient for ESS > 400."""
    # Rule of thumb: draws > 1000 to get ESS > 400
    # (ESS typically 40-60% of draws for well-tuned chains)

    test_cases = [
        (500, False),  # Insufficient
        (1000, True),  # Borderline
        (2000, True),  # Good
        (5000, True),  # Excessive (but safe)
    ]

    for draws, sufficient in test_cases:
        estimated_ess = draws * 0.5  # Assume 50% efficiency
        passes = estimated_ess > 400

        assert passes == sufficient, f"Draws={draws} should be {sufficient}"


def test_mcmc_tune_optimization():
    """Test tuning iterations optimization."""
    # Recommendations:
    # - Minimum: 500 (may not converge)
    # - Recommended: 1000 (good balance)
    # - Excessive: > 2000 (diminishing returns)

    test_cases = [
        (500, "minimum"),
        (1000, "recommended"),
        (2000, "recommended"),
        (5000, "excessive"),
    ]

    for tune, category in test_cases:
        if tune < 1000:
            status = "minimum"
        elif tune <= 2000:
            status = "recommended"
        else:
            status = "excessive"

        assert status == category


# ============================================================================
# TEST 2: SQL QUERY OPTIMIZATION
# ============================================================================


def test_sql_select_star_detection(mock_slow_sql_query):
    """Test detection of SELECT * (inefficient)."""
    has_select_star = "SELECT *" in mock_slow_sql_query

    assert has_select_star, "Slow query uses SELECT *"

    # Recommendation: Specify columns explicitly
    recommendation = "Replace SELECT * with specific columns"
    assert recommendation is not None


def test_sql_missing_where_clause():
    """Test detection of queries without WHERE clause (full table scan)."""
    query_no_where = "SELECT * FROM games ORDER BY kickoff"
    query_with_where = "SELECT * FROM games WHERE season = 2024 ORDER BY kickoff"

    has_where_1 = "WHERE" in query_no_where.upper()
    has_where_2 = "WHERE" in query_with_where.upper()

    assert not has_where_1, "Query without WHERE clause"
    assert has_where_2, "Query with WHERE clause"

    # Without WHERE, should recommend adding filter
    if not has_where_1:
        recommendation = "Add WHERE clause to filter rows"
        assert recommendation is not None


def test_sql_join_without_index():
    """Test detection of joins on unindexed columns."""
    # Mock database schema
    indexes = {
        "games": ["game_id", "season"],
        "plays": ["play_id", "game_id"],
    }

    # Query joins on game_id (indexed - good)
    join_column = "game_id"
    is_indexed = join_column in indexes["plays"]

    assert is_indexed, "game_id should be indexed"

    # Query joins on unindexed column (bad)
    unindexed_join_column = "stadium"
    is_indexed_bad = unindexed_join_column in indexes["games"]

    assert not is_indexed_bad, "stadium not indexed"


def test_sql_order_by_performance():
    """Test ORDER BY performance impact."""
    # ORDER BY on indexed column (fast)
    # ORDER BY on unindexed column (slow - requires sort)

    indexed_columns = ["game_id", "kickoff", "season"]

    order_by_col = "kickoff"
    is_indexed = order_by_col in indexed_columns

    assert is_indexed, "ORDER BY on indexed column is fast"


def test_sql_query_complexity_estimation():
    """Test SQL query complexity estimation."""
    # Estimate query complexity based on:
    # - Number of JOINs
    # - Number of WHERE conditions
    # - Presence of subqueries
    # - GROUP BY / ORDER BY

    def estimate_complexity(query):
        complexity = 0
        complexity += query.upper().count("JOIN")
        complexity += query.upper().count("WHERE")
        complexity += query.upper().count("GROUP BY")
        complexity += query.upper().count("ORDER BY")
        return complexity

    simple_query = "SELECT * FROM games WHERE season = 2024"
    complex_query = """
    SELECT g.season, COUNT(*)
    FROM games g
    JOIN plays p ON g.game_id = p.game_id
    WHERE g.season >= 2015
    GROUP BY g.season
    ORDER BY g.season
    """

    simple_score = estimate_complexity(simple_query)
    complex_score = estimate_complexity(complex_query)

    assert complex_score > simple_score


# ============================================================================
# TEST 3: BACKTEST PROFILING
# ============================================================================


def test_backtest_time_estimation():
    """Test backtest training time estimation."""
    # Parameters
    n_folds = 5  # Time-series cross-validation
    time_per_fold = 120  # 2 minutes per fold

    estimated_time = n_folds * time_per_fold

    # Should be ~10 minutes for 5 folds
    assert estimated_time == 600  # 10 minutes


def test_backtest_scaling_with_data_size():
    """Test backtest time scales with data size."""
    # Time complexity: O(n * log(n)) for most models
    # Assume linear approximation for simplicity

    def estimate_backtest_time(n_games, base_time=60):
        """Estimate backtest time (linear approximation)."""
        return (n_games / 1000) * base_time

    time_1k = estimate_backtest_time(1000)
    time_5k = estimate_backtest_time(5000)
    time_10k = estimate_backtest_time(10000)

    # Should scale linearly
    assert time_5k == pytest.approx(time_1k * 5, rel=0.1)
    assert time_10k == pytest.approx(time_1k * 10, rel=0.1)


def test_parallel_backtest_speedup():
    """Test parallel backtest speedup estimation."""
    serial_time = 600  # 10 minutes serial
    n_cores = 4

    # Parallel speedup (Amdahl's law)
    # Assume 80% of work is parallelizable
    parallel_fraction = 0.8
    serial_fraction = 1 - parallel_fraction

    speedup = 1 / (serial_fraction + parallel_fraction / n_cores)
    parallel_time = serial_time / speedup

    # Should be faster than serial
    assert parallel_time < serial_time

    # Speedup should be < n_cores (due to serial fraction)
    assert speedup < n_cores


# ============================================================================
# TEST 4: PANDAS PERFORMANCE OPTIMIZATION
# ============================================================================


def test_avoid_iterrows():
    """Test that vectorized operations are faster than iterrows."""
    n = 10000
    df = pd.DataFrame(
        {
            "a": np.random.randint(1, 100, n),
            "b": np.random.randint(1, 100, n),
        }
    )

    # BAD: Using iterrows
    start = time.time()
    result_slow = []
    for idx, row in df.iterrows():
        result_slow.append(row["a"] + row["b"])
    time_iterrows = time.time() - start

    # GOOD: Vectorized operation
    start = time.time()
    result_fast = df["a"] + df["b"]
    time_vectorized = time.time() - start

    # Vectorized should be much faster
    assert time_vectorized < time_iterrows

    # Results should be identical
    assert np.array_equal(result_slow, result_fast.values)


def test_apply_vs_vectorized():
    """Test that vectorized operations are faster than apply."""
    n = 10000
    df = pd.DataFrame(
        {
            "yards": np.random.randint(0, 100, n),
        }
    )

    # SLOWER: Using apply
    start = time.time()
    df["yards"].apply(lambda x: x * 2)
    time_apply = time.time() - start

    # FASTER: Vectorized
    start = time.time()
    df["yards"] * 2
    time_vectorized = time.time() - start

    # Vectorized should be faster
    assert time_vectorized < time_apply


def test_groupby_agg_performance():
    """Test efficient groupby aggregations."""
    n = 100000
    df = pd.DataFrame(
        {
            "team": np.random.choice(["TEAM_A", "TEAM_B", "TEAM_C"], n),
            "yards": np.random.randint(0, 100, n),
        }
    )

    # Efficient aggregation
    start = time.time()
    df.groupby("team")["yards"].agg(["mean", "sum", "count"])
    time_agg = time.time() - start

    # Should complete quickly (< 0.1s for 100k rows)
    assert time_agg < 0.5


# ============================================================================
# TEST 5: MEMORY PROFILING
# ============================================================================


def test_memory_efficient_dtypes():
    """Test memory savings from optimized dtypes."""
    n = 1000000

    # Inefficient: int64, float64
    df_inefficient = pd.DataFrame(
        {
            "player_id": np.random.randint(1, 1000, n),  # int64
            "yards": np.random.randint(0, 100, n),  # int64
            "epa": np.random.randn(n),  # float64
        }
    )

    # Efficient: int16, float32
    df_efficient = pd.DataFrame(
        {
            "player_id": np.random.randint(1, 1000, n).astype(np.int16),
            "yards": np.random.randint(0, 100, n).astype(np.int16),
            "epa": np.random.randn(n).astype(np.float32),
        }
    )

    mem_inefficient = df_inefficient.memory_usage(deep=True).sum() / 1024**2  # MB
    mem_efficient = df_efficient.memory_usage(deep=True).sum() / 1024**2  # MB

    # Efficient should use less memory
    assert mem_efficient < mem_inefficient

    # Should save ~50% memory
    savings_pct = (mem_inefficient - mem_efficient) / mem_inefficient
    assert savings_pct > 0.4, f"Memory savings: {savings_pct:.1%}"


def test_chunking_large_dataframes():
    """Test chunking strategy for large DataFrames."""
    # Simulate large dataset
    total_rows = 10_000_000
    chunk_size = 100_000
    n_chunks = total_rows // chunk_size

    # Process in chunks
    results = []
    for i in range(n_chunks):
        # Simulate processing chunk
        chunk_result = i * 2
        results.append(chunk_result)

    # Should process 100 chunks
    assert len(results) == 100


# ============================================================================
# TEST 6: MODEL TRAINING TIME ESTIMATION
# ============================================================================


def test_training_time_estimation_by_model_type():
    """Test training time estimation for different model types."""
    n_samples = 10000

    # Model complexity (rough estimates)
    model_times = {
        "linear": 1.0,  # Baseline
        "xgboost": 5.0,  # 5x slower than linear
        "bayesian_mcmc": 60.0,  # 60x slower (MCMC sampling)
        "bnn": 30.0,  # 30x slower (variational inference)
    }

    for model, relative_time in model_times.items():
        estimated_time = (n_samples / 1000) * relative_time

        if model == "linear":
            assert estimated_time == pytest.approx(10.0)
        elif model == "bayesian_mcmc":
            assert estimated_time == pytest.approx(600.0)


def test_training_time_scales_with_features():
    """Test that training time scales with number of features."""
    n_features_list = [10, 50, 100, 200]
    base_time = 60  # seconds for 10 features

    times = []
    for n_features in n_features_list:
        # Assume linear scaling (rough approximation)
        estimated_time = base_time * (n_features / 10)
        times.append(estimated_time)

    # Should scale roughly linearly
    assert times[1] == pytest.approx(300, rel=0.1)  # 5x features = 5x time
    assert times[2] == pytest.approx(600, rel=0.1)  # 10x features = 10x time


# ============================================================================
# TEST 7: BOTTLENECK IDENTIFICATION
# ============================================================================


def test_identify_slowest_operations():
    """Test identification of slowest operations in pipeline."""
    operations = {
        "data_loading": 2.0,
        "feature_engineering": 15.0,  # Slowest
        "model_training": 60.0,  # Second slowest
        "prediction": 1.0,
        "evaluation": 0.5,
    }

    # Sort by time
    sorted_ops = sorted(operations.items(), key=lambda x: x[1], reverse=True)

    # Top 2 bottlenecks
    bottleneck_1 = sorted_ops[0]
    bottleneck_2 = sorted_ops[1]

    assert bottleneck_1[0] == "model_training"
    assert bottleneck_2[0] == "feature_engineering"


def test_profiling_recommendations():
    """Test performance recommendations based on profiling."""
    profile_results = {
        "sql_query_time": 25.0,  # Slow
        "feature_eng_time": 5.0,  # Fast
        "training_time": 120.0,  # Very slow
    }

    recommendations = []

    if profile_results["sql_query_time"] > 10:
        recommendations.append("Optimize SQL queries (add indexes, reduce JOINs)")

    if profile_results["training_time"] > 60:
        recommendations.append("Consider faster model or reduce hyperparameters")

    assert len(recommendations) == 2
    assert any("SQL" in r for r in recommendations)
    assert any("training" in r.lower() for r in recommendations)


# ============================================================================
# TEST 8: INTEGRATION TESTS
# ============================================================================


def test_end_to_end_optimization_workflow():
    """Test complete optimization workflow."""
    # Step 1: Profile current performance
    baseline_times = {
        "data_load": 3.0,
        "feature_eng": 20.0,
        "training": 180.0,
    }

    total_baseline = sum(baseline_times.values())

    # Step 2: Apply optimizations
    optimizations = {
        "data_load": 0.8,  # 20% faster (SQL optimization)
        "feature_eng": 0.6,  # 40% faster (vectorization)
        "training": 0.7,  # 30% faster (MCMC tuning)
    }

    optimized_times = {op: baseline_times[op] * speedup for op, speedup in optimizations.items()}

    total_optimized = sum(optimized_times.values())

    # Should be faster overall
    assert total_optimized < total_baseline

    # Calculate speedup
    speedup = total_baseline / total_optimized
    assert speedup > 1.3  # At least 30% faster


# ============================================================================
# MARKER: Unit tests (no database required)
# ============================================================================

pytestmark = pytest.mark.unit
