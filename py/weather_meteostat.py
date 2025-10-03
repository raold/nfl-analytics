"""Fetch weather from Meteostat and write to `weather` table.

Aligns DSN with docker-compose defaults and supports overrides via env vars:
- POSTGRES_HOST (default: localhost)
- POSTGRES_PORT (default: 5544)
- POSTGRES_DB   (default: devdb01)
- POSTGRES_USER (default: dro)
- POSTGRES_PASSWORD (default: sicillionbillions)
"""

from datetime import timedelta
import os

import pandas as pd
from sqlalchemy import create_engine
from meteostat import Hourly, Stations


def make_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5544")
    db = os.getenv("POSTGRES_DB", "devdb01")
    user = os.getenv("POSTGRES_USER", "dro")
    password = os.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    uri = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return create_engine(uri)


# minimal stadium -> (lat, lon) mapping; expand as needed
STADIA = {
    "Lambeau Field": (44.5013, -88.0622),
    "Arrowhead Stadium": (39.0490, -94.4839),
    # ... add all
}


def main() -> None:
    engine = make_engine()
    games = pd.read_sql(
        """
        select game_id, stadium, kickoff
        from games
        where kickoff is not null
          and game_id not in (select game_id from weather)
        """,
        engine,
    )

    rows = []
    for _, g in games.iterrows():
        coords = STADIA.get(g.stadium)
        if not coords:
            continue
        lat, lon = coords
        stations = Stations().nearby(lat, lon).fetch(1)
        if stations.empty:
            continue
        stid = stations.index[0]
        ts = Hourly(stid, g.kickoff - timedelta(hours=1), g.kickoff + timedelta(hours=3)).fetch()
        if ts.empty:
            continue
        wx = ts.iloc[1] if len(ts) > 1 else ts.iloc[0]
        rows.append(
            {
                "game_id": g.game_id,
                "station": stid,
                "temp_c": float(wx["temp"]) if "temp" in wx else None,
                "rh": float(wx.get("rhum")) if "rhum" in ts.columns else None,
                "wind_kph": float(wx.get("wspd") * 3.6) if "wspd" in ts.columns else None,
                "pressure_hpa": float(wx.get("pres")) if "pres" in ts.columns else None,
                "precip_mm": float(wx.get("prcp")) if "prcp" in ts.columns else None,
            }
        )

    if rows:
        pd.DataFrame(rows).to_sql("weather", engine, if_exists="append", index=False)


if __name__ == "__main__":
    main()
