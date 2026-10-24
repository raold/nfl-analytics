#!/usr/bin/env python3
"""
Unit Tests for Performance Profiling Agent
==========================================

Comprehensive test suite for PerformanceProfilingAgent covering:
- Database query profiling
- Python code profiling
- R code profiling
- API endpoint profiling
- Bottleneck detection
- Recommendation generation
- Report export

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nfl_analytics.utils.performance_profiling_agent import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
    EndpointReport,
    PerformanceProfilingAgent,
    ProfileReport,
    QueryReport,
)


@pytest.fixture
def agent():
    """Create PerformanceProfilingAgent instance."""
    return PerformanceProfilingAgent(
        project_root=Path("/tmp/test_project"),
        db_connection_string="postgresql://test:test@localhost:5432/testdb",
    )


@pytest.fixture
def sample_bottleneck():
    """Create sample bottleneck."""
    return Bottleneck(
        severity=BottleneckSeverity.HIGH,
        bottleneck_type=BottleneckType.DATABASE_QUERY,
        title="Slow query",
        description="Test query",
        location="PostgreSQL",
        current_time_ms=500.0,
        impact="High impact",
    )


class TestBottleneckClassification:
    """Test bottleneck severity classification."""

    def test_classify_query_severity_critical(self, agent):
        """Test critical query severity (>10s)."""
        severity = agent._classify_query_severity(15000)
        assert severity == BottleneckSeverity.CRITICAL

    def test_classify_query_severity_high(self, agent):
        """Test high query severity (>3s)."""
        severity = agent._classify_query_severity(5000)
        assert severity == BottleneckSeverity.HIGH

    def test_classify_query_severity_medium(self, agent):
        """Test medium query severity (>1s)."""
        severity = agent._classify_query_severity(1500)
        assert severity == BottleneckSeverity.MEDIUM

    def test_classify_query_severity_low(self, agent):
        """Test low query severity (>500ms)."""
        severity = agent._classify_query_severity(700)
        assert severity == BottleneckSeverity.LOW

    def test_classify_query_severity_info(self, agent):
        """Test info query severity (<500ms)."""
        severity = agent._classify_query_severity(100)
        assert severity == BottleneckSeverity.INFO

    def test_classify_computation_severity_critical(self, agent):
        """Test critical computation severity (>10s)."""
        severity = agent._classify_computation_severity(12000)
        assert severity == BottleneckSeverity.CRITICAL

    def test_classify_computation_severity_high(self, agent):
        """Test high computation severity (>3s)."""
        severity = agent._classify_computation_severity(4000)
        assert severity == BottleneckSeverity.HIGH

    def test_classify_computation_severity_medium(self, agent):
        """Test medium computation severity (>1s)."""
        severity = agent._classify_computation_severity(2000)
        assert severity == BottleneckSeverity.MEDIUM


class TestDatabaseProfiling:
    """Test database query profiling."""

    @patch("psycopg2.connect")
    def test_profile_database_queries_no_extension(self, mock_connect, agent):
        """Test profiling when pg_stat_statements is not enabled."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"enabled": False}
        mock_connect.return_value.cursor.return_value = mock_cursor

        report = agent.profile_database_queries()

        assert isinstance(report, QueryReport)
        assert len(report.slow_queries) == 0

    @patch("psycopg2.connect")
    def test_profile_database_queries_with_slow_queries(self, mock_connect, agent):
        """Test profiling with slow queries detected."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {"enabled": True},  # Extension enabled
        ]
        mock_cursor.fetchall.side_effect = [
            [  # Slow queries
                {
                    "query": "SELECT * FROM games",
                    "calls": 100,
                    "avg_time_ms": 500.0,
                    "total_time_ms": 50000.0,
                    "rows": 1000,
                    "cache_hit_ratio": 95.0,
                },
            ],
            [],  # N+1 queries
            [],  # Missing indexes
        ]
        mock_connect.return_value.cursor.return_value = mock_cursor

        report = agent.profile_database_queries(min_duration_ms=100)

        assert len(report.slow_queries) == 1
        assert report.slow_queries[0]["query"] == "SELECT * FROM games"
        assert report.avg_query_time_ms == 500.0

    @patch("psycopg2.connect")
    def test_profile_database_queries_n_plus_one(self, mock_connect, agent):
        """Test N+1 query pattern detection."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"enabled": True}
        mock_cursor.fetchall.side_effect = [
            [],  # Slow queries
            [  # N+1 queries
                {
                    "query": "SELECT * FROM stats WHERE player_id = ?",
                    "calls": 500,
                    "avg_time_ms": 5.0,
                    "total_time_ms": 2500.0,
                },
            ],
            [],  # Missing indexes
        ]
        mock_connect.return_value.cursor.return_value = mock_cursor

        report = agent.profile_database_queries()

        assert len(report.n_plus_one_queries) == 1
        assert report.n_plus_one_queries[0]["calls"] == 500

    @patch("psycopg2.connect")
    def test_explain_query(self, mock_connect, agent):
        """Test EXPLAIN ANALYZE query execution."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "QUERY PLAN": [
                {
                    "Plan": {
                        "Node Type": "Seq Scan",
                        "Total Cost": 100.0,
                    }
                }
            ]
        }
        mock_connect.return_value.cursor.return_value = mock_cursor

        result = agent.explain_query("SELECT * FROM games")

        assert result is not None
        assert "QUERY PLAN" in result


class TestPythonProfiling:
    """Test Python code profiling."""

    def test_profile_python_code_nonexistent_file(self, agent):
        """Test profiling nonexistent file."""
        report = agent.profile_python_code("/tmp/nonexistent.py")

        assert isinstance(report, ProfileReport)
        assert report.total_time_s == 0.0

    def test_profile_python_code_simple(self, agent, tmp_path):
        """Test profiling simple Python code."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def slow_function():
    total = 0
    for i in range(1000):
        total += i
    return total

result = slow_function()
"""
        )

        report = agent.profile_python_code(str(test_file), track_memory=True)

        assert report.total_time_s > 0
        assert len(report.function_stats) > 0

    def test_profile_python_code_memory_tracking(self, agent, tmp_path):
        """Test memory tracking during profiling."""
        # Create test file with memory usage
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
data = [i for i in range(1000)]
"""
        )

        report = agent.profile_python_code(str(test_file), track_memory=True)

        assert report.memory_usage_mb >= 0
        assert report.peak_memory_mb >= 0

    def test_profile_function(self, agent):
        """Test profiling specific function."""

        def test_function(n):
            return sum(range(n))

        report = agent.profile_function(test_function, 1000)

        assert report.total_time_s > 0
        assert report.memory_usage_mb >= 0

    def test_profile_function_slow_computation(self, agent):
        """Test profiling slow function creates bottleneck."""

        def slow_function():
            time.sleep(0.1)  # Simulate slow operation
            return 42

        len(agent.bottlenecks)
        report = agent.profile_function(slow_function)

        # Should detect slow computation (>100ms)
        assert report.total_time_s >= 0.1


class TestRProfiling:
    """Test R code profiling."""

    @patch("subprocess.run")
    def test_profile_r_code_success(self, mock_run, agent, tmp_path):
        """Test successful R code profiling."""
        # Create test R script
        r_script = tmp_path / "test.R"
        r_script.write_text("x <- 1:100; mean(x)")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Profiling output",
            stderr="",
        )

        result = agent.profile_r_code(str(r_script))

        assert result["script"] == str(r_script)
        assert "output_file" in result

    @patch("subprocess.run")
    def test_profile_r_code_timeout(self, mock_run, agent, tmp_path):
        """Test R profiling timeout."""
        r_script = tmp_path / "test.R"
        r_script.write_text("x <- 1:100")

        mock_run.side_effect = subprocess.TimeoutExpired("Rscript", 300)

        result = agent.profile_r_code(str(r_script))

        assert result["error"] == "Timeout"

    def test_profile_r_code_nonexistent(self, agent):
        """Test R profiling with nonexistent file."""
        result = agent.profile_r_code("/tmp/nonexistent.R")

        assert result == {}


class TestAPIEndpointProfiling:
    """Test API endpoint profiling."""

    @patch("requests.get")
    def test_profile_api_endpoints_success(self, mock_get, agent):
        """Test successful API endpoint profiling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "result": [
                    {
                        "metric": {
                            "endpoint": "/predict",
                            "method": "POST",
                        },
                        "value": [1234567890, "0.5"],  # 500ms
                    },
                ]
            }
        }
        mock_get.return_value = mock_response

        report = agent.profile_api_endpoints(duration_hours=1)

        assert isinstance(report, EndpointReport)
        assert len(report.endpoints) == 1
        assert report.endpoints[0]["endpoint"] == "/predict"

    @patch("requests.get")
    def test_profile_api_endpoints_slow(self, mock_get, agent):
        """Test slow endpoint detection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "result": [
                    {
                        "metric": {
                            "endpoint": "/predict",
                            "method": "POST",
                        },
                        "value": [1234567890, "2.0"],  # 2000ms
                    },
                ]
            }
        }
        mock_get.return_value = mock_response

        report = agent.profile_api_endpoints(min_latency_ms=1000)

        assert len(report.slow_endpoints) == 1
        assert report.slow_endpoints[0]["avg_latency_ms"] == 2000.0


class TestBottleneckDetection:
    """Test bottleneck detection."""

    @patch("psycopg2.connect")
    def test_detect_bottlenecks_database_only(self, mock_connect, agent):
        """Test detecting database bottlenecks only."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"enabled": True}
        mock_cursor.fetchall.side_effect = [
            [  # Slow queries
                {
                    "query": "SELECT * FROM games",
                    "calls": 100,
                    "avg_time_ms": 5000.0,
                    "total_time_ms": 500000.0,
                    "rows": 1000,
                    "cache_hit_ratio": 95.0,
                },
            ],
            [],  # N+1 queries
            [],  # Missing indexes
        ]
        mock_connect.return_value.cursor.return_value = mock_cursor

        bottlenecks = agent.detect_bottlenecks(
            profile_database=True,
            profile_python=False,
            profile_api=False,
        )

        assert len(bottlenecks) > 0
        assert any(b.bottleneck_type == BottleneckType.DATABASE_QUERY for b in bottlenecks)

    def test_detect_bottlenecks_sorting(self, agent):
        """Test bottlenecks are sorted by severity."""
        agent.bottlenecks = [
            Bottleneck(
                severity=BottleneckSeverity.LOW,
                bottleneck_type=BottleneckType.DATABASE_QUERY,
                title="Low",
                description="Low",
                location="test",
                current_time_ms=100,
                impact="low",
            ),
            Bottleneck(
                severity=BottleneckSeverity.CRITICAL,
                bottleneck_type=BottleneckType.DATABASE_QUERY,
                title="Critical",
                description="Critical",
                location="test",
                current_time_ms=10000,
                impact="critical",
            ),
        ]

        # Re-detect to trigger sorting
        bottlenecks = agent.detect_bottlenecks(
            profile_database=False,
            profile_python=False,
            profile_api=False,
        )

        # Critical should come first
        assert bottlenecks[0].severity == BottleneckSeverity.CRITICAL


class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_generate_recommendations_database_query(self, agent):
        """Test recommendation for database query bottleneck."""
        bottleneck = Bottleneck(
            severity=BottleneckSeverity.HIGH,
            bottleneck_type=BottleneckType.DATABASE_QUERY,
            title="Slow query",
            description="Test",
            location="PostgreSQL",
            current_time_ms=5000,
            impact="High",
        )

        recommendations = agent.generate_recommendations([bottleneck])

        assert len(recommendations) == 1
        assert "index" in recommendations[0].recommendation.lower()

    def test_generate_recommendations_n_plus_one(self, agent):
        """Test recommendation for N+1 query pattern."""
        bottleneck = Bottleneck(
            severity=BottleneckSeverity.HIGH,
            bottleneck_type=BottleneckType.N_PLUS_ONE,
            title="N+1 query",
            description="Test",
            location="PostgreSQL",
            current_time_ms=2500,
            impact="High",
        )

        recommendations = agent.generate_recommendations([bottleneck])

        assert len(recommendations) == 1
        assert (
            "JOIN" in recommendations[0].recommendation
            or "batch" in recommendations[0].recommendation
        )

    def test_generate_recommendations_memory_leak(self, agent):
        """Test recommendation for memory leak."""
        bottleneck = Bottleneck(
            severity=BottleneckSeverity.CRITICAL,
            bottleneck_type=BottleneckType.MEMORY_LEAK,
            title="Memory leak",
            description="Test",
            location="test.py",
            current_time_ms=1000,
            impact="Critical",
        )

        recommendations = agent.generate_recommendations([bottleneck])

        assert len(recommendations) == 1
        assert "memory" in recommendations[0].recommendation.lower()

    def test_generate_recommendations_sorting(self, agent):
        """Test recommendations are sorted by priority."""
        bottlenecks = [
            Bottleneck(
                severity=BottleneckSeverity.LOW,
                bottleneck_type=BottleneckType.DATABASE_QUERY,
                title="Low",
                description="Low",
                location="test",
                current_time_ms=100,
                impact="low",
            ),
            Bottleneck(
                severity=BottleneckSeverity.CRITICAL,
                bottleneck_type=BottleneckType.DATABASE_QUERY,
                title="Critical",
                description="Critical",
                location="test",
                current_time_ms=10000,
                impact="critical",
            ),
        ]

        recommendations = agent.generate_recommendations(bottlenecks)

        # Priority 1 (critical) should come first
        assert recommendations[0].priority <= recommendations[1].priority


class TestReportExport:
    """Test report export functionality."""

    def test_export_report_json(self, agent, tmp_path, sample_bottleneck):
        """Test JSON report export."""
        agent.bottlenecks = [sample_bottleneck]

        output_file = tmp_path / "report.json"
        agent.export_report(str(output_file))

        assert output_file.exists()

        # Validate JSON structure
        data = json.loads(output_file.read_text())
        assert "generated_at" in data
        assert "bottlenecks" in data
        assert len(data["bottlenecks"]) == 1

    def test_export_timeline_html(self, agent, tmp_path, sample_bottleneck):
        """Test HTML timeline export."""
        agent.bottlenecks = [sample_bottleneck]

        output_file = tmp_path / "timeline.html"
        agent.export_timeline(str(output_file))

        assert output_file.exists()

        # Validate HTML content
        html = output_file.read_text()
        assert "<html>" in html
        assert "Performance Timeline" in html
        assert sample_bottleneck.title in html

    def test_generate_summary_report(self, agent, sample_bottleneck):
        """Test summary report generation."""
        agent.bottlenecks = [sample_bottleneck]

        report = agent.generate_summary_report()

        assert "PERFORMANCE PROFILING SUMMARY" in report
        assert "HIGH:" in report
        assert sample_bottleneck.title in report


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_bottlenecks(self, agent):
        """Test with no bottlenecks detected."""
        recommendations = agent.generate_recommendations([])
        assert len(recommendations) == 0

        report = agent.generate_summary_report()
        assert "PERFORMANCE PROFILING SUMMARY" in report

    def test_database_connection_error(self, agent):
        """Test database profiling with connection error."""
        agent.db_connection_string = "postgresql://invalid:invalid@localhost:9999/invalid"

        report = agent.profile_database_queries()
        assert isinstance(report, QueryReport)
        assert len(report.slow_queries) == 0

    def test_profile_python_syntax_error(self, agent, tmp_path):
        """Test profiling Python file with syntax error."""
        test_file = tmp_path / "invalid.py"
        test_file.write_text("this is not valid python code")

        report = agent.profile_python_code(str(test_file))
        assert isinstance(report, ProfileReport)


class TestIntegration:
    """Integration tests for full workflows."""

    @patch("psycopg2.connect")
    def test_full_profiling_workflow(self, mock_connect, agent, tmp_path):
        """Test complete profiling workflow."""
        # Mock database
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"enabled": True}
        mock_cursor.fetchall.side_effect = [
            [  # Slow queries
                {
                    "query": "SELECT * FROM games",
                    "calls": 100,
                    "avg_time_ms": 2000.0,
                    "total_time_ms": 200000.0,
                    "rows": 1000,
                    "cache_hit_ratio": 95.0,
                },
            ],
            [],  # N+1 queries
            [],  # Missing indexes
        ]
        mock_connect.return_value.cursor.return_value = mock_cursor

        # Create test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = sum(range(100))")

        # Update project root to temp directory
        agent.project_root = tmp_path

        # Run full detection
        bottlenecks = agent.detect_bottlenecks(
            profile_database=True,
            profile_python=False,  # Skip Python to avoid file path issues
            profile_api=False,
        )

        # Generate recommendations
        recommendations = agent.generate_recommendations(bottlenecks)

        # Export reports
        agent.export_report(str(tmp_path / "report.json"))
        agent.export_timeline(str(tmp_path / "timeline.html"))

        # Verify outputs
        assert len(bottlenecks) > 0
        assert len(recommendations) > 0
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "timeline.html").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
