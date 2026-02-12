"""
Unit tests for NFL Analytics feature modules.

This package contains comprehensive unit tests for:
- TurnoverLuckEngine: 25 features for regression-based betting edge
- PrimetimeFeaturesEngine: 9 features for market attention indicators
- OpponentStrengthEngine: 15 features for Sagarin-style opponent adjustment

All tests use mocked database connections and follow pytest best practices.

Coverage Target: >80% code coverage
Test Isolation: All tests are independent and repeatable
Temporal Safety: Verified no future data leakage in all features

Author: Claude Code (Sonnet 4.5)
Date: December 8, 2025
"""
