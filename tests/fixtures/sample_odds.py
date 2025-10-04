"""
Sample odds response fixture for testing.

Contains a realistic example of The Odds API historical response.
"""

SAMPLE_ODDS_RESPONSE_SINGLE_GAME = {
    "id": "abc123def456ghi789",
    "sport_key": "americanfootball_nfl",
    "sport_title": "NFL",
    "commence_time": "2023-09-07T17:00:00Z",
    "home_team": "Buffalo Bills",
    "away_team": "Arizona Cardinals",
    "bookmakers": [
        {
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": "2023-09-01T12:00:00Z",
            "markets": [
                {
                    "key": "spreads",
                    "last_update": "2023-09-01T12:00:00Z",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.91, "point": -6.5},
                        {"name": "Arizona Cardinals", "price": 1.91, "point": 6.5}
                    ]
                },
                {
                    "key": "totals",
                    "last_update": "2023-09-01T12:00:00Z",
                    "outcomes": [
                        {"name": "Over", "price": 1.87, "point": 47.5},
                        {"name": "Under", "price": 1.95, "point": 47.5}
                    ]
                },
                {
                    "key": "h2h",
                    "last_update": "2023-09-01T12:00:00Z",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.25},
                        {"name": "Arizona Cardinals", "price": 4.20}
                    ]
                }
            ]
        },
        {
            "key": "draftkings",
            "title": "DraftKings",
            "last_update": "2023-09-01T12:05:00Z",
            "markets": [
                {
                    "key": "spreads",
                    "last_update": "2023-09-01T12:05:00Z",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.87, "point": -6.0},
                        {"name": "Arizona Cardinals", "price": 1.95, "point": 6.0}
                    ]
                }
            ]
        }
    ]
}

SAMPLE_ODDS_RESPONSE_MULTI_GAME = [
    SAMPLE_ODDS_RESPONSE_SINGLE_GAME,
    {
        "id": "xyz789abc456def123",
        "sport_key": "americanfootball_nfl",
        "sport_title": "NFL",
        "commence_time": "2023-09-10T13:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Detroit Lions",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2023-09-01T12:00:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": 1.91, "point": -3.5},
                            {"name": "Detroit Lions", "price": 1.91, "point": 3.5}
                        ]
                    }
                ]
            }
        ]
    }
]
