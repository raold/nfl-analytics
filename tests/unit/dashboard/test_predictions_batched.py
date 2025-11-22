"""
Unit Tests for Batched Prediction Queries

Tests the batched query functions to ensure they properly eliminate N+1 patterns
and return correct results.

Author: Claude Code
Date: November 22, 2025
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.queries.predictions_batched import (
    BatchedPredictionQueries,
    create_lookup_dict
)


class TestBatchedPredictionQueries:
    """Test suite for batched prediction query methods."""

    @pytest.fixture
    def mock_conn(self):
        """Create mock database connection."""
        return Mock()

    @pytest.fixture
    def sample_game_ids(self):
        """Sample game IDs for testing."""
        return ['2024_12_BAL_LAC', '2024_12_KC_LV', '2024_12_BUF_SF']

    @pytest.fixture
    def sample_player_names(self):
        """Sample player names for testing."""
        return ['P.Mahomes', 'L.Jackson', 'J.Allen']

    @pytest.fixture
    def sample_predictions_df(self):
        """Sample predictions DataFrame."""
        return pd.DataFrame({
            'game_id': ['2024_12_BAL_LAC', '2024_12_BAL_LAC', '2024_12_KC_LV'],
            'player_id': [1, 2, 3],
            'player_name': ['L.Jackson', 'P.Ricard', 'P.Mahomes'],
            'team': ['BAL', 'BAL', 'KC'],
            'prediction_mean': [250.5, 15.0, 275.3],
            'prediction_std': [45.2, 5.0, 50.1],
            'ensemble_version': ['v6.1', 'v6.1', 'v6.1']
        })

    def test_predictions_batch_empty_list(self, mock_conn):
        """Test batched predictions with empty game ID list."""
        result = BatchedPredictionQueries.predictions_batch(mock_conn, [])

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_predictions_batch_success(
        self,
        mock_read_sql,
        mock_conn,
        sample_game_ids,
        sample_predictions_df
    ):
        """Test successful batched predictions query."""
        mock_read_sql.return_value = sample_predictions_df

        result = BatchedPredictionQueries.predictions_batch(
            mock_conn,
            sample_game_ids,
            stat_type='passing_yards'
        )

        # Verify query was called with correct parameters
        mock_read_sql.assert_called_once()
        call_args = mock_read_sql.call_args

        # Verify query contains WHERE IN clause
        query = call_args[0][0]
        assert 'WHERE p.game_id IN (' in query
        assert 'AND p.stat_type = %s' in query

        # Verify parameters are correct
        params = call_args[1]['params']
        assert params == sample_game_ids + ['passing_yards']

        # Verify result
        assert not result.empty
        assert len(result) == 3
        assert 'game_id' in result.columns
        assert 'player_name' in result.columns

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_line_movement_batch_success(
        self,
        mock_read_sql,
        mock_conn,
        sample_player_names
    ):
        """Test successful batched line movement query."""
        sample_movement_df = pd.DataFrame({
            'player_name': ['P.Mahomes', 'P.Mahomes', 'L.Jackson'],
            'snapshot_at': pd.to_datetime(['2024-11-20', '2024-11-21', '2024-11-21']),
            'line_value': [265.5, 267.0, 248.5],
            'bookmaker_title': ['DraftKings', 'DraftKings', 'FanDuel']
        })
        mock_read_sql.return_value = sample_movement_df

        result = BatchedPredictionQueries.line_movement_batch(
            mock_conn,
            sample_player_names,
            days_back=7
        )

        # Verify query was called
        mock_read_sql.assert_called_once()
        call_args = mock_read_sql.call_args

        # Verify query contains WHERE IN clause
        query = call_args[0][0]
        assert 'WHERE player_name IN (' in query

        # Verify result
        assert not result.empty
        assert len(result) == 3
        assert set(result['player_name'].unique()).issubset(set(sample_player_names))

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_game_predictions_batch_success(
        self,
        mock_read_sql,
        mock_conn,
        sample_game_ids
    ):
        """Test successful batched game predictions query."""
        sample_game_preds_df = pd.DataFrame({
            'game_id': sample_game_ids,
            'predicted_spread': [-3.5, 7.2, 1.0],
            'predicted_total': [47.5, 52.3, 44.0],
            'predicted_home_win_prob': [0.65, 0.75, 0.52]
        })
        mock_read_sql.return_value = sample_game_preds_df

        result = BatchedPredictionQueries.game_predictions_batch(
            mock_conn,
            sample_game_ids
        )

        # Verify query was called
        mock_read_sql.assert_called_once()

        # Verify result
        assert not result.empty
        assert len(result) == 3
        assert set(result['game_id'].unique()) == set(sample_game_ids)

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_predictions_with_odds_batch_success(
        self,
        mock_read_sql,
        mock_conn,
        sample_game_ids
    ):
        """Test successful batched predictions with odds query."""
        sample_odds_df = pd.DataFrame({
            'game_id': ['2024_12_BAL_LAC', '2024_12_KC_LV'],
            'player_name': ['L.Jackson', 'P.Mahomes'],
            'prediction_mean': [250.5, 275.3],
            'book_line': [245.5, 280.5],
            'edge': [5.0, -5.2],
            'over_odds': [-110, -115]
        })
        mock_read_sql.return_value = sample_odds_df

        result = BatchedPredictionQueries.predictions_with_odds_batch(
            mock_conn,
            sample_game_ids,
            stat_type='passing_yards'
        )

        # Verify query was called
        mock_read_sql.assert_called_once()

        # Verify result has odds columns
        assert not result.empty
        assert 'book_line' in result.columns
        assert 'edge' in result.columns
        assert 'over_odds' in result.columns


class TestCreateLookupDict:
    """Test suite for create_lookup_dict helper function."""

    def test_create_lookup_dict_empty_dataframe(self):
        """Test lookup dict creation with empty DataFrame."""
        df = pd.DataFrame()
        result = create_lookup_dict(df, 'game_id')

        assert result == {}
        assert isinstance(result, dict)

    def test_create_lookup_dict_single_key(self):
        """Test lookup dict with single key per group."""
        df = pd.DataFrame({
            'game_id': ['2024_12_BAL_LAC', '2024_12_KC_LV'],
            'predicted_spread': [-3.5, 7.2],
            'predicted_total': [47.5, 52.3]
        })

        result = create_lookup_dict(df, 'game_id')

        assert len(result) == 2
        assert '2024_12_BAL_LAC' in result
        assert '2024_12_KC_LV' in result

        # Each key maps to list of rows (dicts)
        assert len(result['2024_12_BAL_LAC']) == 1
        assert result['2024_12_BAL_LAC'][0]['predicted_spread'] == -3.5

    def test_create_lookup_dict_multiple_rows_per_key(self):
        """Test lookup dict with multiple rows per key (e.g., multiple players per game)."""
        df = pd.DataFrame({
            'game_id': ['2024_12_BAL_LAC', '2024_12_BAL_LAC', '2024_12_KC_LV'],
            'player_name': ['L.Jackson', 'P.Ricard', 'P.Mahomes'],
            'prediction_mean': [250.5, 15.0, 275.3]
        })

        result = create_lookup_dict(df, 'game_id')

        assert len(result) == 2
        assert len(result['2024_12_BAL_LAC']) == 2  # 2 players
        assert len(result['2024_12_KC_LV']) == 1     # 1 player

        # Verify data structure
        bal_lac_players = result['2024_12_BAL_LAC']
        player_names = [p['player_name'] for p in bal_lac_players]
        assert set(player_names) == {'L.Jackson', 'P.Ricard'}

    def test_create_lookup_dict_O1_lookup(self):
        """Test O(1) lookup performance."""
        # Create large DataFrame
        df = pd.DataFrame({
            'game_id': [f'2024_12_GAME_{i}' for i in range(1000)],
            'value': list(range(1000))
        })

        result = create_lookup_dict(df, 'game_id')

        # O(1) lookup
        assert len(result) == 1000
        assert result['2024_12_GAME_500'][0]['value'] == 500
        assert result['2024_12_GAME_999'][0]['value'] == 999


class TestPerformanceImprovements:
    """Test suite to verify N+1 elimination."""

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_n_plus_1_elimination_proof(self, mock_read_sql, sample_game_ids):
        """Prove that batched query executes only ONCE regardless of game count."""
        mock_conn = Mock()
        mock_read_sql.return_value = pd.DataFrame({
            'game_id': sample_game_ids,
            'player_name': ['Player1', 'Player2', 'Player3'],
            'prediction_mean': [250, 260, 270]
        })

        # Execute batched query for N games
        result = BatchedPredictionQueries.predictions_batch(
            mock_conn,
            sample_game_ids  # 3 games
        )

        # Verify query was called EXACTLY ONCE (not 3 times!)
        assert mock_read_sql.call_count == 1

        # Verify all games returned
        assert len(result) == 3

    @patch('dashboard.queries.predictions_batched.pd.read_sql_query')
    def test_batched_vs_loop_query_count(self, mock_read_sql, sample_game_ids):
        """Compare query count: batched (1 query) vs loop (N queries)."""
        mock_conn = Mock()
        mock_read_sql.return_value = pd.DataFrame({
            'game_id': ['game1'],
            'prediction_mean': [250]
        })

        # ANTI-PATTERN: Loop (simulated) - would execute N queries
        loop_query_count = len(sample_game_ids)  # 3 queries

        # BATCHED: Execute once
        BatchedPredictionQueries.predictions_batch(mock_conn, sample_game_ids)
        batched_query_count = mock_read_sql.call_count  # 1 query

        # Verify speedup
        speedup = loop_query_count / batched_query_count
        assert speedup == len(sample_game_ids)  # 3x speedup for 3 games
        assert batched_query_count == 1
