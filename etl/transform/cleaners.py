"""
Data cleaning and standardization.

Handles:
- Team name normalization
- Column name variations (qtr vs quarter for 2025 data)
- Null handling and type conversions
- Duplicate detection
- Date/time standardization
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class TeamNameStandardizer:
    """
    Standardizes NFL team names across different data sources.

    Handles variations like:
    - "LA" → "LAR" (Rams) or "LAC" (Chargers)
    - "Washington" → "WAS"
    - "Oakland" → "LV" (Raiders moved in 2020)
    - "San Diego" → "LAC" (Chargers moved in 2017)
    - "St. Louis" → "LAR" (Rams moved in 2016)
    """

    # Standard 3-letter abbreviations
    STANDARD_TEAMS = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    }

    # Historical mappings (team relocations)
    HISTORICAL_MAPPINGS = {
        'OAK': {'before': 2020, 'after': 'LV'},  # Raiders
        'SD': {'before': 2017, 'after': 'LAC'},   # Chargers
        'STL': {'before': 2016, 'after': 'LAR'},  # Rams
    }

    # Common aliases
    ALIAS_MAPPINGS = {
        'LA': None,  # Ambiguous - needs context
        'WSH': 'WAS',
        'WFT': 'WAS',  # Washington Football Team (2020-2021)
        'JAC': 'JAX',
        'SL': 'STL',   # Historical St. Louis
        'GNB': 'GB',
    }

    def __init__(self):
        """Initialize team name standardizer."""
        self.warnings: List[str] = []

    def standardize(
        self,
        team: str,
        season: Optional[int] = None,
        context_teams: Optional[Set[str]] = None
    ) -> str:
        """
        Standardize team abbreviation.

        Args:
            team: Team abbreviation to standardize
            season: Season year (for historical mappings)
            context_teams: Other teams in the dataset (for disambiguating "LA")

        Returns:
            Standardized 3-letter abbreviation

        Raises:
            ValueError: If team cannot be standardized
        """
        if not team or pd.isna(team):
            raise ValueError("Team name is null or empty")

        team = str(team).strip().upper()

        # Already standard
        if team in self.STANDARD_TEAMS:
            return team

        # Check historical mappings
        if team in self.HISTORICAL_MAPPINGS:
            mapping = self.HISTORICAL_MAPPINGS[team]
            if season and season >= mapping['before']:
                return mapping['after']
            return team  # Keep historical abbreviation for old seasons

        # Check aliases
        if team in self.ALIAS_MAPPINGS:
            mapped = self.ALIAS_MAPPINGS[team]
            if mapped:
                return mapped

            # Handle ambiguous "LA"
            if team == 'LA' and context_teams:
                # If only one LA team in context, use that
                la_teams = {'LAR', 'LAC'} & context_teams
                if len(la_teams) == 1:
                    return la_teams.pop()

            warning = f"Ambiguous team abbreviation: {team}"
            logger.warning(warning)
            self.warnings.append(warning)
            return team

        # Unknown team
        warning = f"Unknown team abbreviation: {team}"
        logger.warning(warning)
        self.warnings.append(warning)
        return team

    def standardize_dataframe(
        self,
        df: pd.DataFrame,
        team_columns: List[str],
        season_column: Optional[str] = 'season'
    ) -> pd.DataFrame:
        """
        Standardize team names in a DataFrame.

        Args:
            df: DataFrame to standardize
            team_columns: List of columns containing team names
            season_column: Column with season year (for historical mappings)

        Returns:
            DataFrame with standardized team names
        """
        df_copy = df.copy()

        # Get all unique teams for context
        all_teams = set()
        for col in team_columns:
            if col in df_copy.columns:
                all_teams.update(df_copy[col].dropna().unique())

        # Standardize each column
        for col in team_columns:
            if col not in df_copy.columns:
                continue

            df_copy[col] = df_copy.apply(
                lambda row: self.standardize(
                    row[col],
                    season=row.get(season_column) if season_column in df_copy.columns else None,
                    context_teams=all_teams
                ),
                axis=1
            )

        return df_copy


class DataCleaner:
    """
    General-purpose data cleaner for ETL pipelines.

    Handles:
    - Null value handling
    - Type conversions
    - Duplicate detection
    - Column name standardization
    - Date/time parsing and timezone conversion
    """

    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_stats: Dict[str, Any] = {}

    def handle_column_aliases(
        self,
        df: pd.DataFrame,
        alias_map: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Handle column name variations.

        Example: {'quarter': ['qtr', 'quarter']} will rename 'qtr' to 'quarter'

        Args:
            df: DataFrame to process
            alias_map: Dict mapping standard names to list of aliases

        Returns:
            DataFrame with standardized column names
        """
        df_copy = df.copy()

        for standard_name, aliases in alias_map.items():
            # Find which alias is present (if any)
            found_alias = None
            for alias in aliases:
                if alias in df_copy.columns and alias != standard_name:
                    found_alias = alias
                    break

            # Rename to standard
            if found_alias:
                logger.info(f"Renaming column '{found_alias}' to '{standard_name}'")
                df_copy = df_copy.rename(columns={found_alias: standard_name})

        return df_copy

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'last'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: DataFrame to deduplicate
            subset: Columns to check for duplicates (None = all columns)
            keep: Which duplicate to keep ('first', 'last', False)

        Returns:
            Deduplicated DataFrame
        """
        before_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        after_count = len(df_clean)

        duplicates_removed = before_count - after_count
        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate rows")
            self.cleaning_stats['duplicates_removed'] = duplicates_removed

        return df_clean

    def standardize_datetimes(
        self,
        df: pd.DataFrame,
        datetime_columns: List[str],
        target_tz: str = 'UTC'
    ) -> pd.DataFrame:
        """
        Standardize datetime columns to a specific timezone.

        Args:
            df: DataFrame to process
            datetime_columns: List of datetime columns
            target_tz: Target timezone (default: UTC)

        Returns:
            DataFrame with standardized datetimes
        """
        df_copy = df.copy()

        for col in datetime_columns:
            if col not in df_copy.columns:
                continue

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col])
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
                    continue

            # Convert to target timezone
            try:
                if df_copy[col].dt.tz is None:
                    # Assume UTC if no timezone
                    df_copy[col] = df_copy[col].dt.tz_localize('UTC')
                else:
                    df_copy[col] = df_copy[col].dt.tz_convert('UTC')
            except Exception as e:
                logger.warning(f"Could not convert {col} to {target_tz}: {e}")

        return df_copy

    def fill_nulls(
        self,
        df: pd.DataFrame,
        fill_strategy: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fill null values using specified strategies.

        Args:
            df: DataFrame to process
            fill_strategy: Dict mapping column names to fill values or methods
                          e.g., {'home_score': 0, 'kickoff': 'ffill'}

        Returns:
            DataFrame with nulls filled
        """
        df_copy = df.copy()

        for col, strategy in fill_strategy.items():
            if col not in df_copy.columns:
                continue

            if strategy == 'ffill':
                df_copy[col] = df_copy[col].fillna(method='ffill')
            elif strategy == 'bfill':
                df_copy[col] = df_copy[col].fillna(method='bfill')
            elif strategy == 'mean':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif strategy == 'median':
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                # Assume it's a constant value
                df_copy[col] = df_copy[col].fillna(strategy)

        return df_copy

    def enforce_types(
        self,
        df: pd.DataFrame,
        type_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Enforce data types on columns.

        Args:
            df: DataFrame to process
            type_map: Dict mapping column names to pandas dtypes

        Returns:
            DataFrame with enforced types
        """
        df_copy = df.copy()

        for col, dtype in type_map.items():
            if col not in df_copy.columns:
                continue

            try:
                if dtype == 'datetime64[ns]':
                    df_copy[col] = pd.to_datetime(df_copy[col])
                else:
                    df_copy[col] = df_copy[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert {col} to {dtype}: {e}")

        return df_copy

    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return self.cleaning_stats


# Example usage
if __name__ == "__main__":
    # Test team name standardization
    standardizer = TeamNameStandardizer()

    # Test various team names
    test_cases = [
        ('ARI', 2024, 'ARI'),
        ('LA', 2024, None),  # Ambiguous
        ('OAK', 2019, 'OAK'),  # Before move
        ('OAK', 2020, 'LV'),   # After move
        ('WSH', 2024, 'WAS'),
        ('JAC', 2024, 'JAX'),
    ]

    print("Team Name Standardization Tests:")
    for team, season, expected in test_cases:
        result = standardizer.standardize(team, season)
        status = "✅" if (expected is None or result == expected) else "❌"
        print(f"  {status} {team} ({season}) → {result} (expected: {expected})")

    # Test data cleaner
    print("\nData Cleaner Tests:")
    cleaner = DataCleaner()

    # Sample data with column alias
    df_test = pd.DataFrame({
        'game_id': ['2025_01_SF_PIT', '2025_01_BAL_KC'],
        'qtr': [1, 2],  # Should be renamed to 'quarter'
        'epa': [0.5, -0.3]
    })

    print(f"Before: {df_test.columns.tolist()}")
    df_clean = cleaner.handle_column_aliases(df_test, {'quarter': ['qtr', 'quarter']})
    print(f"After: {df_clean.columns.tolist()}")
