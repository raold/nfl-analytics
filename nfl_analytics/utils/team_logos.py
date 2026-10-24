#!/usr/bin/env python3
"""
NFL Team Logos for Terminal Display

Display team logos as Unicode emojis in the terminal.
Works in most modern terminals that support Unicode.
"""

# Team logo mapping - Unicode emojis that work in terminals
TEAM_LOGOS = {
    # AFC East
    "BUF": "🦬",  # Buffalo Bills - Bison
    "MIA": "🐬",  # Miami Dolphins
    "NE": "🎖️",  # New England Patriots - Military medal
    "NYJ": "✈️",  # New York Jets
    # AFC North
    "BAL": "🐦‍⬛",  # Baltimore Ravens - Black bird
    "CIN": "🐅",  # Cincinnati Bengals - Tiger
    "CLE": "🟤",  # Cleveland Browns - Brown circle
    "PIT": "⚙️",  # Pittsburgh Steelers - Gear (steel)
    # AFC South
    "HOU": "🐂",  # Houston Texans - Bull
    "IND": "🐴",  # Indianapolis Colts - Horse
    "JAX": "🐆",  # Jacksonville Jaguars - Leopard/Jaguar
    "TEN": "⚔️",  # Tennessee Titans - Crossed swords
    # AFC West
    "DEN": "🐎",  # Denver Broncos - Horse
    "KC": "🏹",  # Kansas City Chiefs - Bow and arrow
    "LV": "☠️",  # Las Vegas Raiders - Skull and crossbones
    "LAC": "⚡",  # Los Angeles Chargers - Lightning bolt
    # NFC East
    "DAL": "⭐",  # Dallas Cowboys - Star
    "NYG": "🔵",  # New York Giants - Blue circle
    "PHI": "🦅",  # Philadelphia Eagles
    "WAS": "🏛️",  # Washington Commanders - Government building
    # NFC North
    "CHI": "🐻",  # Chicago Bears
    "DET": "🦁",  # Detroit Lions
    "GB": "🧀",  # Green Bay Packers - Cheese
    "MIN": "🛡️",  # Minnesota Vikings - Shield
    # NFC South
    "ATL": "🔴",  # Atlanta Falcons - Red circle
    "CAR": "🐾",  # Carolina Panthers - Paw prints
    "NO": "⚜️",  # New Orleans Saints - Fleur-de-lis
    "TB": "🏴‍☠️",  # Tampa Bay Buccaneers - Pirate flag
    # NFC West
    "ARI": "🟥",  # Arizona Cardinals - Red square
    "LA": "🐏",  # Los Angeles Rams
    "SF": "🔶",  # San Francisco 49ers - Orange diamond (gold)
    "SEA": "🌊",  # Seattle Seahawks - Wave (sea)
    # Historical teams (mapped to current)
    "OAK": "☠️",  # Oakland Raiders -> LV
    "SD": "⚡",  # San Diego Chargers -> LAC
    "STL": "🐏",  # St. Louis Rams -> LA
}

# Team colors for terminal output (ANSI color codes)
TEAM_COLORS = {
    # AFC East
    "BUF": "\033[34m",  # Blue
    "MIA": "\033[36m",  # Cyan (aqua)
    "NE": "\033[34m",  # Blue
    "NYJ": "\033[32m",  # Green
    # AFC North
    "BAL": "\033[35m",  # Purple
    "CIN": "\033[38;5;208m",  # Orange
    "CLE": "\033[38;5;94m",  # Brown
    "PIT": "\033[33m",  # Yellow
    # AFC South
    "HOU": "\033[34m",  # Blue
    "IND": "\033[34m",  # Blue
    "JAX": "\033[38;5;178m",  # Gold/Teal
    "TEN": "\033[34m",  # Blue
    # AFC West
    "DEN": "\033[38;5;208m",  # Orange
    "KC": "\033[31m",  # Red
    "LV": "\033[37m",  # Silver (white)
    "LAC": "\033[38;5;226m",  # Yellow
    # NFC East
    "DAL": "\033[34m",  # Blue
    "NYG": "\033[34m",  # Blue
    "PHI": "\033[32m",  # Green
    "WAS": "\033[38;5;88m",  # Burgundy
    # NFC North
    "CHI": "\033[34m",  # Blue
    "DET": "\033[38;5;39m",  # Honolulu blue
    "GB": "\033[32m",  # Green
    "MIN": "\033[35m",  # Purple
    # NFC South
    "ATL": "\033[31m",  # Red
    "CAR": "\033[38;5;39m",  # Panthers blue
    "NO": "\033[38;5;178m",  # Gold
    "TB": "\033[31m",  # Red
    # NFC West
    "ARI": "\033[31m",  # Red
    "LA": "\033[34m",  # Blue
    "SF": "\033[31m",  # Red
    "SEA": "\033[34m",  # Blue
}

RESET_COLOR = "\033[0m"


def get_team_logo(team: str) -> str:
    """Get the Unicode emoji logo for a team."""
    return TEAM_LOGOS.get(team.upper(), "🏈")


def get_team_color(team: str) -> str:
    """Get the ANSI color code for a team."""
    return TEAM_COLORS.get(team.upper(), "")


def format_team_with_logo(team: str, include_color: bool = True) -> str:
    """Format a team abbreviation with its logo and optional color."""
    team = team.upper()
    logo = get_team_logo(team)

    if include_color:
        color = get_team_color(team)
        return f"{logo} {color}{team}{RESET_COLOR}"
    else:
        return f"{logo} {team}"


def print_all_teams():
    """Print all teams with their logos, organized by division."""
    divisions = [
        ("AFC East", ["BUF", "MIA", "NE", "NYJ"]),
        ("AFC North", ["BAL", "CIN", "CLE", "PIT"]),
        ("AFC South", ["HOU", "IND", "JAX", "TEN"]),
        ("AFC West", ["DEN", "KC", "LV", "LAC"]),
        ("NFC East", ["DAL", "NYG", "PHI", "WAS"]),
        ("NFC North", ["CHI", "DET", "GB", "MIN"]),
        ("NFC South", ["ATL", "CAR", "NO", "TB"]),
        ("NFC West", ["ARI", "LA", "SF", "SEA"]),
    ]

    print("\n🏈 NFL TEAM LOGOS 🏈")
    print("=" * 40)

    for division, teams in divisions:
        print(f"\n{division}:")
        for team in teams:
            print(f"  {format_team_with_logo(team)}")


def format_matchup(
    home_team: str, away_team: str, score_home: int = None, score_away: int = None
) -> str:
    """Format a game matchup with logos and optional scores."""
    home = format_team_with_logo(home_team)
    away = format_team_with_logo(away_team)

    if score_home is not None and score_away is not None:
        return f"{away} ({score_away}) @ {home} ({score_home})"
    else:
        return f"{away} @ {home}"


if __name__ == "__main__":
    # Demo the functionality
    print_all_teams()

    print("\n\nSample Matchups:")
    print("=" * 40)
    print(format_matchup("BUF", "KC"))
    print(format_matchup("DAL", "PHI", 24, 31))
    print(format_matchup("GB", "CHI", 28, 21))

    print("\n\nHistorical Team Mapping:")
    print("=" * 40)
    print(
        f"OAK (Oakland Raiders) -> {format_team_with_logo('OAK')} -> {format_team_with_logo('LV')}"
    )
    print(
        f"SD (San Diego Chargers) -> {format_team_with_logo('SD')} -> {format_team_with_logo('LAC')}"
    )
    print(
        f"STL (St. Louis Rams) -> {format_team_with_logo('STL')} -> {format_team_with_logo('LA')}"
    )
