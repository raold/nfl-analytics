#!/usr/bin/env python3
"""
Scrape NFL coaching staff data from Wikipedia.

Wikipedia has simpler HTML structure and less aggressive bot protection compared
to Pro Football Reference. This scraper targets:
- Individual team season pages (e.g., "2023 Kansas City Chiefs season")
- Coach biographical pages for cross-reference validation

Usage:
    python scripts/scrape_coaching_wikipedia.py --output data/coaching_staff_2018_2024_wikipedia.csv

Author: Claude Code
Date: November 26, 2025
"""

import argparse
import logging
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NFL team name mappings to Wikipedia page titles
TEAM_WIKIPEDIA_NAMES = {
    'ARI': 'Arizona_Cardinals',
    'ATL': 'Atlanta_Falcons',
    'BAL': 'Baltimore_Ravens',
    'BUF': 'Buffalo_Bills',
    'CAR': 'Carolina_Panthers',
    'CHI': 'Chicago_Bears',
    'CIN': 'Cincinnati_Bengals',
    'CLE': 'Cleveland_Browns',
    'DAL': 'Dallas_Cowboys',
    'DEN': 'Denver_Broncos',
    'DET': 'Detroit_Lions',
    'GB': 'Green_Bay_Packers',
    'HOU': 'Houston_Texans',
    'IND': 'Indianapolis_Colts',
    'JAX': 'Jacksonville_Jaguars',
    'KC': 'Kansas_City_Chiefs',
    'LAC': 'Los_Angeles_Chargers',
    'LAR': 'Los_Angeles_Rams',
    'LV': 'Las_Vegas_Raiders',
    'MIA': 'Miami_Dolphins',
    'MIN': 'Minnesota_Vikings',
    'NE': 'New_England_Patriots',
    'NO': 'New_Orleans_Saints',
    'NYG': 'New_York_Giants',
    'NYJ': 'New_York_Jets',
    'PHI': 'Philadelphia_Eagles',
    'PIT': 'Pittsburgh_Steelers',
    'SF': 'San_Francisco_49ers',
    'SEA': 'Seattle_Seahawks',
    'TB': 'Tampa_Bay_Buccaneers',
    'TEN': 'Tennessee_Titans',
    'WAS': 'Washington_Commanders',
}


class WikipediaCoachingScraper:
    """Scrape coaching staff data from Wikipedia."""

    def __init__(self, start_season: int = 2018, end_season: int = 2024):
        """Initialize scraper with season range."""
        self.start_season = start_season
        self.end_season = end_season
        self.base_url = "https://en.wikipedia.org/wiki"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def scrape_team_season(
        self, team_abbr: str, season: int, retry_count: int = 3
    ) -> dict[str, str] | None:
        """
        Scrape coaching staff for a single team-season from Wikipedia.

        Args:
            team_abbr: NFL team abbreviation (e.g., 'KC')
            season: Season year (e.g., 2023)
            retry_count: Number of retries on failure

        Returns:
            Dictionary with keys: team, season, head_coach, offensive_coordinator,
            defensive_coordinator, special_teams_coordinator, or None on failure
        """
        team_name = TEAM_WIKIPEDIA_NAMES[team_abbr]
        url = f"{self.base_url}/{season}_{team_name}_season"

        for attempt in range(retry_count):
            try:
                logger.info(f"Scraping {team_abbr} {season} from Wikipedia (attempt {attempt + 1}/{retry_count})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Method 1: Look for infobox with coaching staff
                coaches_from_infobox = self._extract_from_infobox(soup, team_abbr, season)

                # Method 2: Look for coaching staff section in article text (bullet lists)
                coaches_from_text = self._extract_from_text(soup, team_abbr, season)

                # Merge results - prefer text method for coordinators, infobox for head coach
                coaches_data = {
                    'team': team_abbr,
                    'season': season,
                    'head_coach': coaches_from_text.get('head_coach') or coaches_from_infobox.get('head_coach'),
                    'offensive_coordinator': coaches_from_text.get('offensive_coordinator') or coaches_from_infobox.get('offensive_coordinator'),
                    'defensive_coordinator': coaches_from_text.get('defensive_coordinator') or coaches_from_infobox.get('defensive_coordinator'),
                    'special_teams_coordinator': coaches_from_text.get('special_teams_coordinator') or coaches_from_infobox.get('special_teams_coordinator'),
                }

                if coaches_data['head_coach']:
                    logger.info(f"✓ {team_abbr} {season}: HC={coaches_data['head_coach']}, "
                               f"OC={coaches_data['offensive_coordinator']}, "
                               f"DC={coaches_data['defensive_coordinator']}")
                    time.sleep(1)  # Be respectful to Wikipedia
                    return coaches_data

                logger.warning(f"Could not find coaching staff for {team_abbr} {season}")
                return None

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Wikipedia page not found for {team_abbr} {season}")
                    return None
                logger.error(f"HTTP error for {team_abbr} {season}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(3)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {team_abbr} {season}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(3)
            except Exception as e:
                logger.error(f"Unexpected error for {team_abbr} {season}: {e}")
                return None

        return None

    def _extract_from_infobox(self, soup: BeautifulSoup, team_abbr: str, season: int) -> dict[str, str] | None:
        """
        Extract coaching staff from Wikipedia infobox.

        Wikipedia team season pages typically have an infobox with coaching staff.
        """
        infobox = soup.find('table', class_='infobox')
        if not infobox:
            return None

        coaches_data = {
            'team': team_abbr,
            'season': season,
            'head_coach': None,
            'offensive_coordinator': None,
            'defensive_coordinator': None,
            'special_teams_coordinator': None,
        }

        # Find all rows in infobox
        rows = infobox.find_all('tr')
        for row in rows:
            header = row.find('th')
            if not header:
                continue

            header_text = header.get_text(strip=True).lower()
            value_cell = row.find('td')
            if not value_cell:
                continue

            value_text = value_cell.get_text(strip=True)

            # Extract coach names
            if 'head coach' in header_text:
                coaches_data['head_coach'] = self._clean_coach_name(value_text)
            elif 'offensive coordinator' in header_text or 'off. coordinator' in header_text:
                coaches_data['offensive_coordinator'] = self._clean_coach_name(value_text)
            elif 'defensive coordinator' in header_text or 'def. coordinator' in header_text:
                coaches_data['defensive_coordinator'] = self._clean_coach_name(value_text)
            elif 'special teams' in header_text or 'st coordinator' in header_text:
                coaches_data['special_teams_coordinator'] = self._clean_coach_name(value_text)

        return coaches_data

    def _extract_from_text(self, soup: BeautifulSoup, team_abbr: str, season: int) -> dict[str, str] | None:
        """
        Extract coaching staff from article text (fallback method).

        Coordinators are typically in bulleted lists like:
        <ul><li>Offensive coordinator – <a href="...">Matt Nagy</a></li>
        """
        coaches_data = {
            'team': team_abbr,
            'season': season,
            'head_coach': None,
            'offensive_coordinator': None,
            'defensive_coordinator': None,
            'special_teams_coordinator': None,
        }

        # Find all list items containing coordinator information
        list_items = soup.find_all('li')

        for li in list_items:
            text = li.get_text()

            # Check for each coordinator type
            if 'Head coach' in text or 'head coach' in text:
                link = li.find('a')
                if link and not coaches_data['head_coach']:
                    coaches_data['head_coach'] = self._clean_coach_name(link.get_text())

            if 'Offensive coordinator' in text or 'offensive coordinator' in text:
                link = li.find('a')
                if link and not coaches_data['offensive_coordinator']:
                    coaches_data['offensive_coordinator'] = self._clean_coach_name(link.get_text())

            if 'Defensive coordinator' in text or 'defensive coordinator' in text:
                link = li.find('a')
                if link and not coaches_data['defensive_coordinator']:
                    coaches_data['defensive_coordinator'] = self._clean_coach_name(link.get_text())

            if 'Special teams coordinator' in text or 'special teams coordinator' in text:
                link = li.find('a')
                if link and not coaches_data['special_teams_coordinator']:
                    coaches_data['special_teams_coordinator'] = self._clean_coach_name(link.get_text())

        return coaches_data

    def _clean_coach_name(self, text: str) -> str | None:
        """
        Clean coach name from Wikipedia text.

        Removes brackets, citations, extra whitespace, etc.
        """
        # Remove citations like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        # Remove parenthetical content
        text = re.sub(r'\([^)]*\)', '', text)
        # Clean whitespace
        text = text.strip()
        # Take first name if multiple are listed
        if ',' in text:
            text = text.split(',')[0].strip()
        # Validate it looks like a name
        if len(text) < 3 or len(text) > 50:
            return None
        # Check for common non-name patterns
        if any(keyword in text.lower() for keyword in ['coordinator', 'coach', 'vacant', 'tbd', 'interim']):
            # Extract name if it's like "John Doe (interim)"
            match = re.match(r'^([A-Z][a-z]+ [A-Z][a-z]+)', text)
            if match:
                return match.group(1)
            return None
        return text

    def scrape_all_teams(self) -> pd.DataFrame:
        """
        Scrape coaching data for all teams across all seasons.

        Returns:
            DataFrame with columns: team, season, head_coach, offensive_coordinator,
            defensive_coordinator, special_teams_coordinator
        """
        all_data = []
        total_teams = len(TEAM_WIKIPEDIA_NAMES) * (self.end_season - self.start_season + 1)
        completed = 0

        for season in range(self.start_season, self.end_season + 1):
            for team_abbr in sorted(TEAM_WIKIPEDIA_NAMES.keys()):
                coaches_data = self.scrape_team_season(team_abbr, season)
                if coaches_data:
                    all_data.append(coaches_data)
                else:
                    # Still add row with NULLs to track missing data
                    all_data.append({
                        'team': team_abbr,
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

    def validate_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate scraped coaching data."""
        if len(df) == 0:
            return False, ["DataFrame is empty - no data scraped"]

        warnings = []

        # Check for missing head coaches
        missing_hc = df[df['head_coach'].isna()]
        if len(missing_hc) > 0:
            warnings.append(f"Missing head coach for {len(missing_hc)} team-seasons")

        # Check for missing OCs
        missing_oc = df[df['offensive_coordinator'].isna()]
        missing_oc_pct = 100 * len(missing_oc) / len(df)
        if missing_oc_pct > 10:
            warnings.append(f"Missing OC for {len(missing_oc)} team-seasons ({missing_oc_pct:.1f}%)")

        # Check for missing DCs
        missing_dc = df[df['defensive_coordinator'].isna()]
        missing_dc_pct = 100 * len(missing_dc) / len(df)
        if missing_dc_pct > 10:
            warnings.append(f"Missing DC for {len(missing_dc)} team-seasons ({missing_dc_pct:.1f}%)")

        # Overall validation
        is_valid = (
            len(missing_hc) < 10 and  # More lenient than PFR
            missing_oc_pct < 20 and
            missing_dc_pct < 20
        )

        return is_valid, warnings


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Scrape NFL coaching staff data from Wikipedia'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/coaching_staff_2018_2024_wikipedia.csv',
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
    scraper = WikipediaCoachingScraper(start_season=args.start_season, end_season=args.end_season)

    # Scrape data
    logger.info(f"Starting Wikipedia coaching data scrape ({args.start_season}-{args.end_season})")
    logger.info(f"Total team-seasons: {len(TEAM_WIKIPEDIA_NAMES) * (args.end_season - args.start_season + 1)}")

    if args.teams:
        # Scrape only specified teams
        logger.info(f"Filtering to teams: {args.teams}")
        all_data = []
        for season in range(args.start_season, args.end_season + 1):
            for team_abbr in args.teams:
                if team_abbr in TEAM_WIKIPEDIA_NAMES:
                    coaches_data = scraper.scrape_team_season(team_abbr, season)
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
    if len(df) > 0:
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
