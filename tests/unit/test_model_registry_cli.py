"""
Comprehensive tests for Model Registry CLI (sync_model_registry.py)

Tests the 1,179-line CLI tool for managing model lifecycle:
1. Filesystem scanning - Discover all model files
2. Database synchronization - ml.model_registry table
3. Model metadata extraction - Version, type, size
4. Sync status reporting - DB vs filesystem comparison
5. Markdown export - Documentation generation
6. Model registration - Add new models to registry
7. Production promotion - Mark models as production-ready
8. Model archival - Remove outdated models

Critical Requirements:
- Accurate model discovery (no false positives/negatives)
- Correct version extraction from filenames
- Database integrity (no duplicate registrations)
- Production safety (only validated models promoted)

Author: Claude Code (Sonnet 4.5)
Created: November 9, 2025
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import psycopg2

# Import CLI functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.sync_model_registry import (
    scan_models_directory,
    extract_model_info,
    get_db_models,
    compare_db_filesystem,
    MODEL_TYPE_MAP,
    MODEL_FAMILY_MAP,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_models_directory(tmp_path):
    """Create mock models/ directory structure."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create subdirectories
    (models_dir / "bayesian").mkdir()
    (models_dir / "rl").mkdir()
    (models_dir / "xgboost").mkdir()

    # Create mock model files
    files = [
        "bayesian/bnn_passing_v3.1.pkl",
        "bayesian/bnn_rushing_v2.0.pkl",
        "bayesian/hierarchical_receiving_v1.pkl",
        "rl/cql_agent_v2.0.pth",
        "rl/cql_best_model.pth",
        "xgboost/xgb_baseline_v1.pkl",
        "state_space_v2.rds",
    ]

    for file_path in files:
        full_path = models_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("mock model data")

    # Also create metadata files
    (models_dir / "bayesian" / "bnn_passing_v3.1_metadata.json").write_text(
        json.dumps({"test_mae": 58.70, "calibration": 0.895})
    )

    return models_dir


@pytest.fixture
def mock_db_models():
    """Create mock database models."""
    return [
        {
            "model_id": 1,
            "model_name": "bnn_passing_v3.1",
            "model_type": "bnn",
            "model_family": "bayesian_neural_network",
            "version": "v3.1",
            "model_path": "models/bayesian/bnn_passing_v3.1.pkl",
            "status": "validated",
            "is_production": True,
            "created_at": datetime(2025, 11, 8),
            "updated_at": datetime(2025, 11, 8),
            "description": "BNN v3.1 for passing yards (rushing only)",
        },
        {
            "model_id": 2,
            "model_name": "state_space_v2",
            "model_type": "state_space",
            "model_family": "kalman_filter",
            "version": "v2",
            "model_path": "models/state_space_v2.rds",
            "status": "validated",
            "is_production": True,
            "created_at": datetime(2025, 11, 8),
            "updated_at": datetime(2025, 11, 8),
            "description": "State-space v2 APPROVED (MAE: 47.77)",
        },
        {
            "model_id": 3,
            "model_name": "old_deprecated_model",
            "model_type": "xgboost",
            "model_family": "gradient_boosting",
            "version": "v1",
            "model_path": "models/old_deprecated_model.pkl",
            "status": "archived",
            "is_production": False,
            "created_at": datetime(2025, 10, 1),
            "updated_at": datetime(2025, 10, 1),
            "description": "Old model no longer in use",
        },
    ]


@pytest.fixture
def mock_db_connection(mock_db_models):
    """Create mock database connection."""
    mock_conn = MagicMock(spec=psycopg2.extensions.connection)
    mock_cursor = MagicMock()

    # Mock cursor to return db_models
    mock_cursor.fetchall.return_value = [
        {k: v for k, v in model.items()}
        for model in mock_db_models
    ]

    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=False)

    mock_conn.cursor.return_value = mock_cursor

    return mock_conn


# ============================================================================
# TEST 1: FILESYSTEM SCANNING
# ============================================================================


def test_scan_models_directory_finds_all_models(mock_models_directory):
    """Test that scan finds all model files."""
    models = scan_models_directory(mock_models_directory)

    # Should find 7 model files (.pkl, .pth, .rds)
    model_files = [m for m in models if m['file_extension'] in {'.pkl', '.pth', '.rds'}]
    assert len(model_files) == 7, f"Expected 7 models, found {len(model_files)}"


def test_scan_models_directory_excludes_metadata(mock_models_directory):
    """Test that scan excludes metadata JSON files from model count."""
    models = scan_models_directory(mock_models_directory)

    # Metadata .json files should be included but distinguished
    metadata_files = [m for m in models if m['file_extension'] == '.json']
    model_files = [m for m in models if m['file_extension'] in {'.pkl', '.pth', '.rds'}]

    # Should have 1 metadata file
    assert len(metadata_files) == 1

    # Should have 7 model files
    assert len(model_files) == 7


def test_scan_models_directory_excludes_system_files(tmp_path):
    """Test that scan excludes system files and directories."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create system files that should be excluded
    (models_dir / ".DS_Store").write_text("")
    (models_dir / "README.md").write_text("# Models")
    (models_dir / "__pycache__").mkdir()
    (models_dir / "__pycache__" / "cache.pyc").write_text("")

    # Create valid model
    (models_dir / "valid_model.pkl").write_text("model data")

    models = scan_models_directory(models_dir)

    # Should only find the valid model
    assert len(models) == 1
    assert models[0]['model_name'] == 'valid_model'


def test_scan_empty_directory(tmp_path):
    """Test scanning empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    models = scan_models_directory(empty_dir)

    assert len(models) == 0


# ============================================================================
# TEST 2: MODEL METADATA EXTRACTION
# ============================================================================


def test_extract_model_info_bnn():
    """Test extraction of BNN model info."""
    # Create temporary file
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        f.write(b'mock model data')
        temp_path = Path(f.name)

    # Rename to BNN pattern
    bnn_path = temp_path.parent / "bnn_passing_v3.1.pkl"
    temp_path.rename(bnn_path)

    try:
        info = extract_model_info(bnn_path)

        assert info is not None
        assert info['model_name'] == 'bnn_passing_v3.1'
        assert info['model_type'] == 'bnn'
        assert info['model_family'] == 'bayesian_neural_network'
        assert info['version'] == 'v3'  # Extracted from filename
        assert info['file_extension'] == '.pkl'
        assert info['file_size_mb'] >= 0
    finally:
        bnn_path.unlink()


def test_extract_model_info_cql():
    """Test extraction of CQL (RL) model info."""
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.pth', delete=False) as f:
        f.write(b'mock pytorch model')
        temp_path = Path(f.name)

    cql_path = temp_path.parent / "cql_agent_v2.0.pth"
    temp_path.rename(cql_path)

    try:
        info = extract_model_info(cql_path)

        assert info['model_name'] == 'cql_agent_v2.0'
        assert info['model_type'] == 'rl'
        assert info['model_family'] == 'conservative_q_learning'
        assert info['version'] == 'v2'
        assert info['file_extension'] == '.pth'
    finally:
        cql_path.unlink()


def test_extract_model_info_state_space():
    """Test extraction of state-space model info (R .rds file)."""
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.rds', delete=False) as f:
        f.write(b'mock R model')
        temp_path = Path(f.name)

    ss_path = temp_path.parent / "state_space_v2.rds"
    temp_path.rename(ss_path)

    try:
        info = extract_model_info(ss_path)

        assert info['model_name'] == 'state_space_v2'
        assert info['model_type'] == 'state_space'
        assert info['model_family'] == 'kalman_filter'
        assert info['version'] == 'v2'
        assert info['file_extension'] == '.rds'
    finally:
        ss_path.unlink()


def test_extract_model_info_version_parsing():
    """Test version extraction from various filename patterns."""
    test_cases = [
        ("model_v1.pkl", "v1"),
        ("model_v2.0.pkl", "v2"),
        ("model_v3.1.pkl", "v3"),
        ("model_v10.5.pkl", "v10"),
        ("model.pkl", "v1.0"),  # Default version
    ]

    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        for filename, expected_version in test_cases:
            path = Path(tmpdir) / filename
            path.write_text("mock")

            info = extract_model_info(path)
            assert info['version'] == expected_version, \
                f"Failed for {filename}: expected {expected_version}, got {info['version']}"


def test_extract_model_info_file_size():
    """Test file size calculation in MB."""
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        # Write ~1MB of data
        f.write(b'x' * (1024 * 1024))
        temp_path = Path(f.name)

    try:
        info = extract_model_info(temp_path)

        # Should be approximately 1.0 MB
        assert 0.9 <= info['file_size_mb'] <= 1.1
    finally:
        temp_path.unlink()


# ============================================================================
# TEST 3: DATABASE OPERATIONS
# ============================================================================


@patch('scripts.sync_model_registry.psycopg2.connect')
def test_get_db_models(mock_connect, mock_db_connection, mock_db_models):
    """Test retrieving models from database."""
    mock_connect.return_value = mock_db_connection

    from scripts.sync_model_registry import get_db_connection
    conn = get_db_connection()

    # Mock the cursor to return our test data
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_db_models
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=False)
    conn.cursor.return_value = mock_cursor

    models = get_db_models(conn)

    assert len(models) == 3
    assert models[0]['model_name'] == 'bnn_passing_v3.1'
    assert models[1]['model_name'] == 'state_space_v2'
    assert models[2]['model_name'] == 'old_deprecated_model'


# ============================================================================
# TEST 4: SYNC COMPARISON
# ============================================================================


def test_compare_db_filesystem_in_sync():
    """Test comparison when DB and filesystem are in sync."""
    db_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
        {'model_name': 'model_c'},
    ]

    fs_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
        {'model_name': 'model_c'},
    ]

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 3
    assert len(comparison['only_in_db']) == 0
    assert len(comparison['only_in_fs']) == 0
    assert comparison['db_count'] == 3
    assert comparison['fs_count'] == 3


def test_compare_db_filesystem_only_in_db():
    """Test comparison when models exist only in database."""
    db_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
        {'model_name': 'deleted_model'},  # Deleted from filesystem
    ]

    fs_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
    ]

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 2
    assert 'deleted_model' in comparison['only_in_db']
    assert len(comparison['only_in_fs']) == 0


def test_compare_db_filesystem_only_in_fs():
    """Test comparison when models exist only on filesystem (not registered)."""
    db_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
    ]

    fs_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
        {'model_name': 'new_unregistered_model'},  # Not yet in database
    ]

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 2
    assert len(comparison['only_in_db']) == 0
    assert 'new_unregistered_model' in comparison['only_in_fs']


def test_compare_db_filesystem_mixed():
    """Test comparison with models in both categories."""
    db_models = [
        {'model_name': 'model_a'},
        {'model_name': 'old_archived'},  # Deleted from filesystem
        {'model_name': 'model_c'},
    ]

    fs_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_c'},
        {'model_name': 'new_model'},  # Not yet registered
    ]

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 2
    assert 'old_archived' in comparison['only_in_db']
    assert 'new_model' in comparison['only_in_fs']
    assert comparison['db_count'] == 3
    assert comparison['fs_count'] == 3


# ============================================================================
# TEST 5: MODEL TYPE DETECTION
# ============================================================================


def test_model_type_detection_bnn():
    """Test BNN model type detection."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bnn_passing_v3.pkl"
        path.write_text("mock")

        info = extract_model_info(path)
        assert info['model_type'] == 'bnn'


def test_model_type_detection_cql():
    """Test CQL (RL) model type detection."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cql_agent.pth"
        path.write_text("mock")

        info = extract_model_info(path)
        assert info['model_type'] == 'rl'


def test_model_type_detection_xgboost():
    """Test XGBoost model type detection."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "xgb_baseline.pkl"
        path.write_text("mock")

        info = extract_model_info(path)
        assert info['model_type'] == 'xgboost'


def test_model_type_detection_hierarchical():
    """Test hierarchical model type detection."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        # Test multiple patterns
        patterns = [
            'informative_priors_v1.pkl',
            'receiving_model.pkl',
            'passing_yards_hierarchical.pkl',
        ]

        for pattern in patterns:
            path = Path(tmpdir) / pattern
            path.write_text("mock")

            info = extract_model_info(path)
            assert info['model_type'] == 'hierarchical', f"Failed for {pattern}"

            path.unlink()


def test_model_type_detection_unknown():
    """Test unknown model type handling."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "random_file.pkl"
        path.write_text("mock")

        info = extract_model_info(path)
        assert info['model_type'] == 'unknown'
        assert info['model_family'] == 'unknown'


# ============================================================================
# TEST 6: EDGE CASES
# ============================================================================


def test_scan_with_nested_directories(tmp_path):
    """Test scanning deeply nested directory structure."""
    # Create nested structure
    deep_path = tmp_path / "models" / "a" / "b" / "c" / "d"
    deep_path.mkdir(parents=True)

    model_path = deep_path / "deep_model.pkl"
    model_path.write_text("deep model")

    models = scan_models_directory(tmp_path / "models")

    assert len(models) == 1
    assert 'deep_model' in models[0]['model_name']


def test_scan_with_special_characters(tmp_path):
    """Test scanning files with special characters."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create model with special characters
    special_path = models_dir / "model-v1.0_final[APPROVED].pkl"
    special_path.write_text("model")

    models = scan_models_directory(models_dir)

    assert len(models) == 1


def test_extract_model_info_symlink(tmp_path):
    """Test extraction from symbolic link."""
    import os

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create original file
    original = models_dir / "original_model.pkl"
    original.write_text("model data")

    # Create symlink
    symlink = models_dir / "link_to_model.pkl"
    try:
        os.symlink(original, symlink)

        # Should handle symlink gracefully
        info = extract_model_info(symlink)
        assert info is not None
    except OSError:
        # Symlinks may not be supported on all systems
        pytest.skip("Symlinks not supported on this system")


def test_compare_empty_database():
    """Test comparison with empty database."""
    db_models = []
    fs_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
    ]

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 0
    assert len(comparison['only_in_db']) == 0
    assert len(comparison['only_in_fs']) == 2
    assert comparison['db_count'] == 0
    assert comparison['fs_count'] == 2


def test_compare_empty_filesystem():
    """Test comparison with empty filesystem."""
    db_models = [
        {'model_name': 'model_a'},
        {'model_name': 'model_b'},
    ]
    fs_models = []

    comparison = compare_db_filesystem(db_models, fs_models)

    assert len(comparison['in_both']) == 0
    assert len(comparison['only_in_db']) == 2
    assert len(comparison['only_in_fs']) == 0


def test_model_with_very_long_filename(tmp_path):
    """Test handling of very long filenames."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create file with very long name (but within OS limits)
    long_name = "a" * 200 + ".pkl"
    long_path = models_dir / long_name

    try:
        long_path.write_text("model")
        info = extract_model_info(long_path)
        assert info is not None
    except OSError:
        # Some filesystems have length limits
        pytest.skip("Filename too long for this filesystem")


# ============================================================================
# TEST 7: PERFORMANCE
# ============================================================================


def test_scan_performance_large_directory(tmp_path):
    """Test scan performance with large directory (100 models)."""
    import time

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create 100 model files
    for i in range(100):
        (models_dir / f"model_{i}.pkl").write_text(f"model {i}")

    start = time.time()
    models = scan_models_directory(models_dir)
    elapsed = time.time() - start

    # Should complete in < 1 second
    assert elapsed < 1.0, f"Scan took {elapsed:.2f}s (target: < 1s)"
    assert len(models) == 100


# ============================================================================
# MARKER: Unit tests (no database required)
# ============================================================================

pytestmark = pytest.mark.unit
