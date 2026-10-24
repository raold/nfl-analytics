"""
NFL Team Abbreviation Mapping Utility

This module provides canonical team abbreviation handling for the NFL Analytics project.
It resolves the recurring mismatch between nflverse data (LA, SD, OAK, STL) and
standard NFL abbreviations (LAR, LAC, LV).

Usage:
    from nfl_analytics.utils.team_mapping import normalize_team, VALID_TEAMS, VALID_TEAMS_DB

    # Normalize database abbreviation to canonical form
    team = normalize_team('LA')  # Returns 'LAR'

    # Check if team is valid (accepts both forms)
    if team in VALID_TEAMS:
        ...

Created: December 9, 2025
"""

# Canonical 32 current NFL teams (standard abbreviations)
CANONICAL_TEAMS = {
    "ARI",  # Arizona Cardinals
    "ATL",  # Atlanta Falcons
    "BAL",  # Baltimore Ravens
    "BUF",  # Buffalo Bills
    "CAR",  # Carolina Panthers
    "CHI",  # Chicago Bears
    "CIN",  # Cincinnati Bengals
    "CLE",  # Cleveland Browns
    "DAL",  # Dallas Cowboys
    "DEN",  # Denver Broncos
    "DET",  # Detroit Lions
    "GB",  # Green Bay Packers
    "HOU",  # Houston Texans
    "IND",  # Indianapolis Colts
    "JAX",  # Jacksonville Jaguars
    "KC",  # Kansas City Chiefs
    "LAC",  # Los Angeles Chargers
    "LAR",  # Los Angeles Rams
    "LV",  # Las Vegas Raiders
    "MIA",  # Miami Dolphins
    "MIN",  # Minnesota Vikings
    "NE",  # New England Patriots
    "NO",  # New Orleans Saints
    "NYG",  # New York Giants
    "NYJ",  # New York Jets
    "PHI",  # Philadelphia Eagles
    "PIT",  # Pittsburgh Steelers
    "SEA",  # Seattle Seahawks
    "SF",  # San Francisco 49ers
    "TB",  # Tampa Bay Buccaneers
    "TEN",  # Tennessee Titans
    "WAS",  # Washington Commanders
}

# Historical/alternate abbreviations found in nflverse data
HISTORICAL_TEAMS = {
    "LA",  # Los Angeles Rams (nflverse uses this, not LAR)
    "SD",  # San Diego Chargers (pre-2017)
    "OAK",  # Oakland Raiders (pre-2020)
    "STL",  # St. Louis Rams (pre-2016)
}

# All valid team abbreviations (canonical + historical)
VALID_TEAMS = CANONICAL_TEAMS | HISTORICAL_TEAMS

# What the database actually uses (nflverse format)
VALID_TEAMS_DB = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",  # Rams - database uses LA, not LAR
    "LAC",  # Chargers
    "LV",  # Raiders (current)
    "OAK",  # Raiders (historical)
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SD",  # Chargers (historical)
    "SEA",
    "SF",
    "STL",  # Rams (historical)
    "TB",
    "TEN",
    "WAS",
}

# Mapping from nflverse/database abbreviations to canonical form
# Use this when the model expects canonical but data has historical
DB_TO_CANONICAL = {
    "LA": "LAR",  # Rams: database uses LA, model may expect LAR
    "SD": "LAC",  # Historical Chargers
    "OAK": "LV",  # Historical Raiders
    "STL": "LAR",  # Historical Rams
}

# Reverse mapping: canonical to database format
CANONICAL_TO_DB = {
    "LAR": "LA",  # Model uses LAR, database uses LA
    "LV": "LV",  # Same in both (post-2020)
    "LAC": "LAC",  # Same in both
}


def normalize_team(team: str, to_canonical: bool = True) -> str:
    """
    Normalize a team abbreviation.

    Args:
        team: Team abbreviation (e.g., 'LA', 'LAR', 'SD')
        to_canonical: If True, convert to canonical form (LAR, LAC, LV)
                     If False, convert to database form (LA, LAC, LV)

    Returns:
        Normalized team abbreviation

    Examples:
        >>> normalize_team('LA')
        'LAR'
        >>> normalize_team('LAR', to_canonical=False)
        'LA'
        >>> normalize_team('KC')
        'KC'
    """
    if team is None:
        return team

    team = team.upper().strip()

    if to_canonical:
        return DB_TO_CANONICAL.get(team, team)
    else:
        return CANONICAL_TO_DB.get(team, team)


def normalize_teams_in_dict(data: dict, team_keys: list[str] = None) -> dict:
    """
    Normalize all team abbreviations in a dictionary.

    Args:
        data: Dictionary potentially containing team abbreviations
        team_keys: List of keys to check for team values.
                  Defaults to ['home_team', 'away_team', 'team', 'opponent', 'posteam', 'defteam']

    Returns:
        Dictionary with normalized team abbreviations
    """
    if team_keys is None:
        team_keys = ["home_team", "away_team", "team", "opponent", "posteam", "defteam"]

    result = data.copy()
    for key in team_keys:
        if key in result and isinstance(result[key], str):
            result[key] = normalize_team(result[key])

    return result


def is_valid_team(team: str, strict: bool = False) -> bool:
    """
    Check if a team abbreviation is valid.

    Args:
        team: Team abbreviation to check
        strict: If True, only accept canonical 32-team abbreviations.
               If False (default), accept historical abbreviations too.

    Returns:
        True if valid, False otherwise
    """
    if team is None:
        return False

    team = team.upper().strip()

    if strict:
        return team in CANONICAL_TEAMS
    else:
        return team in VALID_TEAMS


def get_franchise_history(team: str) -> list[str]:
    """
    Get all historical abbreviations for a franchise.

    Args:
        team: Any team abbreviation (current or historical)

    Returns:
        List of all abbreviations used by that franchise

    Examples:
        >>> get_franchise_history('LAR')
        ['LAR', 'LA', 'STL']
        >>> get_franchise_history('LV')
        ['LV', 'OAK']
    """
    franchises = {
        "rams": ["LAR", "LA", "STL"],
        "chargers": ["LAC", "SD"],
        "raiders": ["LV", "OAK"],
    }

    team = team.upper().strip()

    for franchise, abbrevs in franchises.items():
        if team in abbrevs:
            return abbrevs

    # No history, just return the team itself
    return [team]
