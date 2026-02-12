"""Fetch weather from Meteostat and write to `weather` table.

Aligns DSN with docker-compose defaults and supports overrides via env vars:
- POSTGRES_HOST (default: localhost)
- POSTGRES_PORT (default: 5544)
- POSTGRES_DB   (default: devdb01)
- POSTGRES_USER (default: dro)
- POSTGRES_PASSWORD (default: sicillionbillions)
"""

import os
from datetime import timedelta

import pandas as pd
from meteostat import Hourly, Stations
from sqlalchemy import create_engine


def make_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5544")
    db = os.getenv("POSTGRES_DB", "devdb01")
    user = os.getenv("POSTGRES_USER", "dro")
    password = os.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    uri = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return create_engine(uri)


# NFL Stadium coordinates (lat, lon) - complete list for all active venues
STADIA = {
    # AFC East
    "Gillette Stadium": (42.0909, -71.2643),
    "Highmark Stadium": (42.7738, -78.7870),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "MetLife Stadium": (40.8128, -74.0742),
    # AFC North
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Paycor Stadium": (39.0954, -84.5160),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Acrisure Stadium": (40.4468, -80.0158),
    # AFC South
    "NRG Stadium": (29.6847, -95.4107),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "TIAA Bank Field": (30.3239, -81.6373),
    "Nissan Stadium": (36.1665, -86.7713),
    # AFC West
    "Empower Field at Mile High": (39.7439, -105.0201),
    "Arrowhead Stadium": (39.0489, -94.4839),
    "Allegiant Stadium": (36.0908, -115.1833),
    "SoFi Stadium": (33.9535, -118.3392),
    # NFC East
    "AT&T Stadium": (32.7473, -97.0945),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "FedExField": (38.9076, -76.8645),
    "Commanders Field": (38.9076, -76.8645),  # FedEx Field alias
    "Northwest Stadium": (38.9076, -76.8645),  # Another alias
    # NFC North
    "Soldier Field": (41.8623, -87.6167),
    "Ford Field": (42.3400, -83.0456),
    "Lambeau Field": (44.5013, -88.0622),
    "U.S. Bank Stadium": (44.9738, -93.2577),
    # NFC South
    "Mercedes-Benz Stadium": (33.7555, -84.4008),
    "Bank of America Stadium": (35.2258, -80.8530),
    "Caesars Superdome": (29.9511, -90.0812),
    "Raymond James Stadium": (27.9759, -82.5033),
    # NFC West
    "State Farm Stadium": (33.5276, -112.2626),
    "Levi's Stadium": (37.4032, -121.9698),
    "Lumen Field": (47.5952, -122.3316),
    # Historical/Alternate names
    "Sports Authority Field at Mile High": (39.7439, -105.0201),  # Old Broncos name
    "Qualcomm Stadium": (32.7831, -117.1195),  # Old Chargers (San Diego)
    "Oakland Coliseum": (37.7516, -122.2005),  # Old Raiders
    "Candlestick Park": (37.7136, -122.3864),  # Old 49ers
    "Georgia Dome": (33.7577, -84.4008),  # Old Falcons
    "Edward Jones Dome": (38.6328, -90.1884),  # Old Rams (St. Louis)
    "Los Angeles Memorial Coliseum": (34.0141, -118.2879),  # Rams/USC
    "StubHub Center": (33.8643, -118.2611),  # Chargers temp home
    "Tottenham Hotspur Stadium": (51.6042, -0.0662),  # London games
    "Wembley Stadium": (51.5560, -0.2795),  # London games
    "Estadio Azteca": (19.3030, -99.1506),  # Mexico City games
    "Allianz Arena": (48.2188, 11.6247),  # Munich games
}

# Team -> Stadium mapping (current 2024 season; historical changes tracked separately)
TEAM_STADIUM = {
    # AFC East
    "NE": "Gillette Stadium",
    "BUF": "Highmark Stadium",
    "MIA": "Hard Rock Stadium",
    "NYJ": "MetLife Stadium",
    # AFC North
    "BAL": "M&T Bank Stadium",
    "CIN": "Paycor Stadium",
    "CLE": "FirstEnergy Stadium",
    "PIT": "Acrisure Stadium",
    # AFC South
    "HOU": "NRG Stadium",
    "IND": "Lucas Oil Stadium",
    "JAX": "TIAA Bank Field",
    "TEN": "Nissan Stadium",
    # AFC West
    "DEN": "Empower Field at Mile High",
    "KC": "Arrowhead Stadium",
    "LV": "Allegiant Stadium",
    "LAC": "SoFi Stadium",
    # NFC East
    "DAL": "AT&T Stadium",
    "PHI": "Lincoln Financial Field",
    "WAS": "Northwest Stadium",
    "NYG": "MetLife Stadium",  # Giants share stadium with Jets
    # NFC North
    "CHI": "Soldier Field",
    "DET": "Ford Field",
    "GB": "Lambeau Field",
    "MIN": "U.S. Bank Stadium",
    # NFC South
    "ATL": "Mercedes-Benz Stadium",
    "CAR": "Bank of America Stadium",
    "NO": "Caesars Superdome",
    "TB": "Raymond James Stadium",
    # NFC West
    "ARI": "State Farm Stadium",
    "SF": "Levi's Stadium",
    "SEA": "Lumen Field",
    "LA": "SoFi Stadium",
    "LAR": "SoFi Stadium",
    # Historical (pre-relocations)
    "SD": "Qualcomm Stadium",  # Chargers before LAC
    "OAK": "Oakland Coliseum",  # Raiders before LV
    "STL": "Edward Jones Dome",  # Rams before LA
}


def main() -> None:
    engine = make_engine()
    games = pd.read_sql(
        """
        select game_id, home_team, kickoff
        from games
        where kickoff is not null
          and game_id not in (select game_id from weather)
          and EXTRACT(YEAR FROM kickoff) >= 2020
        order by kickoff
        """,
        engine,
    )

    print(f"Processing {len(games)} games (2020-present with reliable Meteostat coverage)...")
    rows = []
    batch_size = 100  # Write to DB every 100 games to avoid losing progress

    for idx, g in games.iterrows():
        # Map home team to stadium
        stadium = TEAM_STADIUM.get(g.home_team)
        if not stadium:
            print(f"  No stadium mapping for team: {g.home_team} (game_id: {g.game_id})")
            continue

        coords = STADIA.get(stadium)
        if not coords:
            print(f"  No coordinates for stadium: {stadium}")
            continue

        lat, lon = coords
        stations = Stations().nearby(lat, lon).fetch(1)
        if stations.empty:
            print(f"  No weather station near {stadium}")
            continue

        stid = stations.index[0]
        # Convert kickoff to timezone-naive datetime for meteostat compatibility
        kickoff_naive = pd.Timestamp(g.kickoff).tz_localize(None)
        ts = Hourly(
            stid, kickoff_naive - timedelta(hours=1), kickoff_naive + timedelta(hours=3)
        ).fetch()
        if ts.empty:
            # Skip silently for missing data (common in early years)
            continue

        wx = ts.iloc[1] if len(ts) > 1 else ts.iloc[0]

        # Helper to safely convert to float, handling NaN/NAType
        def safe_float(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        rows.append(
            {
                "game_id": g.game_id,
                "station": stid,
                "temp_c": safe_float(wx.get("temp")) if "temp" in ts.columns else None,
                "rh": safe_float(wx.get("rhum")) if "rhum" in ts.columns else None,
                "wind_kph": (
                    safe_float(wx.get("wspd")) * 3.6
                    if "wspd" in ts.columns and pd.notna(wx.get("wspd"))
                    else None
                ),
                "pressure_hpa": safe_float(wx.get("pres")) if "pres" in ts.columns else None,
                "precip_mm": safe_float(wx.get("prcp")) if "prcp" in ts.columns else None,
            }
        )

        # Batch write every 100 rows to save progress
        if len(rows) >= batch_size:
            print(
                f"  Writing batch of {len(rows)} records (processed {idx + 1}/{len(games)} games)..."
            )
            pd.DataFrame(rows).to_sql("weather", engine, if_exists="append", index=False)
            rows = []  # Clear batch
        elif (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(games)} games... ({len(rows)} pending write)")

    if rows:
        print(f"\nWriting {len(rows)} weather records to database...")
        pd.DataFrame(rows).to_sql("weather", engine, if_exists="append", index=False)
        print("Done!")
    else:
        print("No weather data collected.")


if __name__ == "__main__":
    main()
