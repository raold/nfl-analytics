#!/usr/bin/env python3
"""
Scrape NFL coaching staff data (OC, DC, HC) from Pro Football Reference.

This script scrapes coaching data for all 32 NFL teams from 2018-2024,
extracting Offensive Coordinator, Defensive Coordinator, and Head Coach
names for use in early-season opponent feature engineering.

Usage:
    python scripts/scrape_coaching_data.py --output data/coaching_staff_2018_2024.csv

Author: Claude Code
Date: November 19, 2025
"""

import argparse
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pro Football Reference team abbreviations (32 teams)
PFR_TEAMS = [
    'crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle',
    'dal', 'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan',
    'sdg', 'ram', 'mia', 'min', 'nwe', 'nor', 'nyg', 'nyj',
    'rai', 'phi', 'pit', 'sfo', 'sea', 'tam', 'oti', 'was'
]

# Mapping from PFR abbreviations to standard NFL abbreviations
PFR_TO_NFL_TEAM_MAP = {
    'crd': 'ARI', 'atl': 'ATL', 'rav': 'BAL', 'buf': 'BUF',
    'car': 'CAR', 'chi': 'CHI', 'cin': 'CIN', 'cle': 'CLE',
    'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gnb': 'GB',
    'htx': 'HOU', 'clt': 'IND', 'jax': 'JAX', 'kan': 'KC',
    'sdg': 'LAC', 'ram': 'LAR', 'mia': 'MIA', 'min': 'MIN',
    'nwe': 'NE', 'nor': 'NO', 'nyg': 'NYG', 'nyj': 'NYJ',
    'rai': 'LV', 'phi': 'PHI', 'pit': 'PIT', 'sfo': 'SF',
    'sea': 'SEA', 'tam': 'TB', 'oti': 'TEN', 'was': 'WAS'
}


class CoachingScraper:
    """Scrape coaching staff data from Pro Football Reference."""

    def __init__(self, start_season: int = 2018, end_season: int = 2024):
        """Initialize scraper with season range."""
        self.start_season = start_season
        self.end_season = end_season
        self.base_url = "https://www.pro-football-reference.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def scrape_team_season(
        self, pfr_team: str, season: int, retry_count: int = 3
    ) -> Optional[Dict[str, str]]:
        """
        Scrape coaching staff for a single team-season.

        Args:
            pfr_team: Pro Football Reference team abbreviation (e.g., 'kan')
            season: Season year (e.g., 2023)
            retry_count: Number of retries on failure

        Returns:
            Dictionary with keys: team, season, head_coach, offensive_coordinator,
            defensive_coordinator, special_teams_coordinator, or None on failure
        """
        url = f"{self.base_url}/teams/{pfr_team}/{season}.htm"
        nfl_team = PFR_TO_NFL_TEAM_MAP[pfr_team]

        for attempt in range(retry_count):
            try:
                logger.info(f"Scraping {nfl_team} {season} (attempt {attempt + 1}/{retry_count})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the coaching staff section
                # Method 1: Look for div with id="all_coaches" or "coaches"
                coaches_div = soup.find('div', id='all_coaches')
                if not coaches_div:
                    coaches_div = soup.find('div', id='coaches')

                if not coaches_div:
                    # Method 2: Search for section containing "Coaches" heading
                    coaches_heading = soup.find('h2', string=lambda s: s and 'Coaches' in s)
                    if coaches_heading:
                        coaches_div = coaches_heading.find_next('div')

                if not coaches_div:
                    logger.warning(f"Could not find coaches section for {nfl_team} {season}")
                    return None

                # Extract coaching staff text
                coaches_text = coaches_div.get_text(separator='\n', strip=True)

                # Parse coaches from text
                coaches_data = {
                    'team': nfl_team,
                    'season': season,
                    'head_coach': self._extract_coach(coaches_text, 'Head Coach'),
                    'offensive_coordinator': self._extract_coach(coaches_text, 'Offensive Coordinator'),
                    'defensive_coordinator': self._extract_coach(coaches_text, 'Defensive Coordinator'),
                    'special_teams_coordinator': self._extract_coach(coaches_text, 'Special Teams Coordinator'),
                }

                # Validate: Head coach should always be present
                if not coaches_data['head_coach']:
                    logger.warning(f"No head coach found for {nfl_team} {season}")
                    # Try alternative parsing method
                    coaches_data['head_coach'] = self._extract_coach_alternative(soup, 'Head Coach')

                # Success
                logger.info(f"✓ {nfl_team} {season}: HC={coaches_data['head_coach']}, "
                           f"OC={coaches_data['offensive_coordinator']}, "
                           f"DC={coaches_data['defensive_coordinator']}")

                # Rate limiting: 3 seconds between requests
                time.sleep(3)
                return coaches_data

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {nfl_team} {season}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    return None
            except Exception as e:
                logger.error(f"Unexpected error for {nfl_team} {season}: {e}")
                return None

        return None

    def _extract_coach(self, text: str, position: str) -> Optional[str]:
        """
        Extract coach name for a given position from coaches text.

        Args:
            text: Full coaches section text
            position: Position title (e.g., "Offensive Coordinator")

        Returns:
            Coach name (e.g., "Eric Bieniemy") or None if not found
        """
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if position in line:
                # Coach name is typically on the next line
                if i + 1 < len(lines):
                    name = lines[i + 1].strip()
                    # Clean up: remove years in parentheses, extra whitespace
                    name = name.split('(')[0].strip()
                    # Skip if name is empty or looks like another position
                    if name and not any(keyword in name.lower() for keyword in [
                        'coordinator', 'coach', 'assistant', 'director'
                    ]):
                        return name
        return None

    def _extract_coach_alternative(self, soup: BeautifulSoup, position: str) -> Optional[str]:
        """
        Alternative method to extract coach name using HTML structure.

        Args:
            soup: BeautifulSoup object
            position: Position title (e.g., "Head Coach")

        Returns:
            Coach name or None if not found
        """
        # Look for <p> or <strong> tags containing position title
        position_elements = soup.find_all(string=lambda s: s and position in s)
        for elem in position_elements:
            parent = elem.parent
            if parent:
                # Try to find name in sibling elements
                next_elem = parent.find_next_sibling()
                if next_elem:
                    name = next_elem.get_text(strip=True)
                    name = name.split('(')[0].strip()
                    if name and len(name) > 3:
                        return name
        return None

    def scrape_all_teams(self) -> pd.DataFrame:
        """
        Scrape coaching data for all teams across all seasons.

        Returns:
            DataFrame with columns: team, season, head_coach, offensive_coordinator,
            defensive_coordinator, special_teams_coordinator
        """
        all_data = []
        total_teams = len(PFR_TEAMS) * (self.end_season - self.start_season + 1)
        completed = 0

        for season in range(self.start_season, self.end_season + 1):
            for pfr_team in PFR_TEAMS:
                coaches_data = self.scrape_team_season(pfr_team, season)
                if coaches_data:
                    all_data.append(coaches_data)
                else:
                    # Still add row with NULLs to track missing data
                    all_data.append({
                        'team': PFR_TO_NFL_TEAM_MAP[pfr_team],
                        'season': season,
                        'head_coach': None,
                        'offensive_coordinator': None,
                        'defensive_coordinator': None,
                        'special_teams_coordinator': None,
                    })

                completed += 1
                logger.info(f"Progress: {completed}/{total_teams} team-seasons complete "
                           f"({100*completed/total_teams:.1f}%)")

        df = pd.DataFrame(all_data)
        return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate scraped coaching data.

        Args:
            df: DataFrame with coaching data

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check for missing head coaches (should be rare)
        missing_hc = df[df['head_coach'].isna()]
        if len(missing_hc) > 0:
            warnings.append(f"Missing head coach for {len(missing_hc)} team-seasons:")
            for _, row in missing_hc.iterrows():
                warnings.append(f"  - {row['team']} {row['season']}")

        # Check for missing OCs (more common, but should be < 5%)
        missing_oc = df[df['offensive_coordinator'].isna()]
        missing_oc_pct = 100 * len(missing_oc) / len(df)
        if missing_oc_pct > 5:
            warnings.append(f"Missing OC for {len(missing_oc)} team-seasons ({missing_oc_pct:.1f}%)")

        # Check for missing DCs
        missing_dc = df[df['defensive_coordinator'].isna()]
        missing_dc_pct = 100 * len(missing_dc) / len(df)
        if missing_dc_pct > 5:
            warnings.append(f"Missing DC for {len(missing_dc)} team-seasons ({missing_dc_pct:.1f}%)")

        # Check for duplicate team-seasons
        duplicates = df[df.duplicated(subset=['team', 'season'], keep=False)]
        if len(duplicates) > 0:
            warnings.append(f"Duplicate team-seasons found: {len(duplicates)} rows")

        # Overall validation
        is_valid = (
            len(missing_hc) < 5 and  # Allow up to 5 missing HCs
            missing_oc_pct < 10 and  # Allow up to 10% missing OCs
            missing_dc_pct < 10 and  # Allow up to 10% missing DCs
            len(duplicates) == 0
        )

        return is_valid, warnings


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Scrape NFL coaching staff data from Pro Football Reference')
    parser.add_argument(
        '--output',
        type=str,
        default='data/coaching_staff_2018_2024.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--start-season',
        type=int,
        default=2018,
        help='Start season (default: 2018)'
    )
    parser.add_argument(
        '--end-season',
        type=int,
        default=2024,
        help='End season (default: 2024)'
    )
    parser.add_argument(
        '--teams',
        type=str,
        nargs='+',
        help='Scrape only specific teams (NFL abbreviations, e.g., KC BUF)'
    )
    args = parser.parse_args()

    # Initialize scraper
    scraper = CoachingScraper(start_season=args.start_season, end_season=args.end_season)

    # Scrape data
    logger.info(f"Starting coaching data scrape ({args.start_season}-{args.end_season})")
    logger.info(f"Teams: {len(PFR_TEAMS)} | Seasons: {args.end_season - args.start_season + 1}")
    logger.info(f"Total team-seasons: {len(PFR_TEAMS) * (args.end_season - args.start_season + 1)}")

    if args.teams:
        # Scrape only specified teams
        logger.info(f"Filtering to teams: {args.teams}")
        # Convert NFL abbreviations back to PFR format
        nfl_to_pfr = {v: k for k, v in PFR_TO_NFL_TEAM_MAP.items()}
        pfr_teams_to_scrape = [nfl_to_pfr[team] for team in args.teams if team in nfl_to_pfr]

        all_data = []
        for season in range(args.start_season, args.end_season + 1):
            for pfr_team in pfr_teams_to_scrape:
                coaches_data = scraper.scrape_team_season(pfr_team, season)
                if coaches_data:
                    all_data.append(coaches_data)
        df = pd.DataFrame(all_data)
    else:
        # Scrape all teams
        df = scraper.scrape_all_teams()

    # Validate data
    logger.info("Validating scraped data...")
    is_valid, warnings = scraper.validate_data(df)

    if warnings:
        logger.warning("Validation warnings:")
        for warning in warnings:
            logger.warning(f"  {warning}")

    if is_valid:
        logger.info("✓ Data validation PASSED")
    else:
        logger.error("✗ Data validation FAILED - review warnings above")

    # Save to CSV
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} team-seasons to {args.output}")

    # Summary statistics
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total team-seasons scraped: {len(df)}")
    logger.info(f"Head coaches found: {df['head_coach'].notna().sum()} / {len(df)} "
               f"({100*df['head_coach'].notna().sum()/len(df):.1f}%)")
    logger.info(f"Offensive coordinators found: {df['offensive_coordinator'].notna().sum()} / {len(df)} "
               f"({100*df['offensive_coordinator'].notna().sum()/len(df):.1f}%)")
    logger.info(f"Defensive coordinators found: {df['defensive_coordinator'].notna().sum()} / {len(df)} "
               f"({100*df['defensive_coordinator'].notna().sum()/len(df):.1f}%)")

    # Sample output
    logger.info("\n=== SAMPLE DATA (first 10 rows) ===")
    print(df.head(10).to_string())


if __name__ == '__main__':
    main()
