#!/usr/bin/env python3
"""
Stadium-specific weather effects analysis.

Hypothesis: Weather effects vary by stadium climate zone.
- Cold-weather stadiums (GB, CHI, BUF, DEN): home advantage in extreme cold
- Warm-weather stadiums (MIA, TB, ARI, LA): home advantage in extreme heat
- Coastal stadiums (NE, SEA, SF): wind/precipitation more impactful

Analysis:
1. Cluster stadiums by geographic region and typical climate
2. Test weather × home team interactions
3. Identify stadium-specific weather edges
"""

import os
from pathlib import Path

import pandas as pd
from scipy import stats
from sqlalchemy import create_engine


def make_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5544")
    db = os.getenv("POSTGRES_DB", "devdb01")
    user = os.getenv("POSTGRES_USER", "dro")
    password = os.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    uri = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return create_engine(uri)


# Stadium climate classifications
CLIMATE_ZONES = {
    # Cold weather (northern/mountain)
    "cold": ["GB", "CHI", "BUF", "DEN", "MIN", "CLE", "NE", "PIT"],
    # Warm weather (southern/desert)
    "warm": ["MIA", "TB", "JAX", "NO", "HOU", "ARI", "LA", "LV"],
    # Moderate (coastal/temperate)
    "moderate": ["SF", "SEA", "BAL", "WSH", "PHI", "NYG", "NYJ", "CAR", "TEN", "KC"],
    # Dome (controlled environment)
    "dome": ["ATL", "DET", "IND", "NO", "LA", "LV", "MIN"],
}

# Reverse mapping: team -> climate zone
TEAM_TO_CLIMATE = {}
for zone, teams in CLIMATE_ZONES.items():
    for team in teams:
        TEAM_TO_CLIMATE[team] = zone


def main():
    engine = make_engine()

    # Fetch games with weather and team info
    sql = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.home_score + g.away_score as total_points,
            g.home_score - g.away_score as home_margin,
            g.spread_close,
            g.total_close,
            gw.temp_c,
            gw.wind_kph,
            gw.has_precip,
            gw.precip_mm,
            gw.is_dome,
            gw.temp_extreme,
            gw.humidity
        FROM games g
        JOIN mart.game_weather gw ON g.game_id = gw.game_id
        WHERE g.season >= 2020
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND gw.temp_c IS NOT NULL
        ORDER BY g.season, g.week
    """

    df = pd.read_sql(sql, engine)
    print(f"Loaded {len(df)} games with weather data (2020-present)")

    # Add climate zone
    df["home_climate"] = df["home_team"].map(TEAM_TO_CLIMATE)
    df["away_climate"] = df["away_team"].map(TEAM_TO_CLIMATE)

    # Filter outdoor games for weather analysis
    df_outdoor = df[df["is_dome"] == 0].copy()
    print(f"Outdoor games: {len(df_outdoor)}")

    # Compute home advantage metrics
    df_outdoor["home_won"] = (df_outdoor["home_score"] > df_outdoor["away_score"]).astype(int)
    df_outdoor["home_ats"] = (df_outdoor["home_margin"] + df_outdoor["spread_close"] > 0).astype(
        int
    )

    # Temperature extremes flags
    df_outdoor["is_extreme_cold"] = (df_outdoor["temp_c"] < 0).astype(int)
    df_outdoor["is_extreme_heat"] = (df_outdoor["temp_c"] > 30).astype(int)

    print("\n" + "=" * 80)
    print("STADIUM CLIMATE ZONE ANALYSIS")
    print("=" * 80)

    # 1. Home win rate by climate zone
    print("\n1. Home Win Rate by Climate Zone:")
    print("-" * 60)
    climate_summary = (
        df_outdoor.groupby("home_climate")
        .agg(
            {
                "game_id": "count",
                "home_won": "mean",
                "home_ats": "mean",
                "temp_c": "mean",
                "wind_kph": "mean",
            }
        )
        .round(3)
    )
    climate_summary.columns = [
        "n_games",
        "home_win_rate",
        "home_ats_rate",
        "avg_temp_c",
        "avg_wind_kph",
    ]
    print(climate_summary)

    # 2. Cold-weather stadium performance in extreme cold
    print("\n2. Cold-Weather Stadiums in Extreme Cold (<0°C):")
    print("-" * 60)
    cold_stadiums = df_outdoor[df_outdoor["home_climate"] == "cold"]
    cold_extreme = cold_stadiums[cold_stadiums["is_extreme_cold"] == 1]
    cold_normal = cold_stadiums[cold_stadiums["is_extreme_cold"] == 0]

    if len(cold_extreme) > 0:
        print(
            f"Extreme cold games: n={len(cold_extreme)}, home_win_rate={cold_extreme['home_won'].mean():.3f}, home_ats={cold_extreme['home_ats'].mean():.3f}"
        )
        print(
            f"Normal temp games:  n={len(cold_normal)}, home_win_rate={cold_normal['home_won'].mean():.3f}, home_ats={cold_normal['home_ats'].mean():.3f}"
        )

        # Test if cold-weather teams have edge in extreme cold
        chi2, p_value, _, _ = stats.chi2_contingency(
            pd.crosstab(cold_stadiums["is_extreme_cold"], cold_stadiums["home_won"])
        )
        print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Cold-weather teams have significant edge in extreme cold")
        else:
            print("✗ No significant cold-weather home advantage")
    else:
        print("No extreme cold games in dataset")

    # 3. Warm-weather stadium performance in extreme heat
    print("\n3. Warm-Weather Stadiums in Extreme Heat (>30°C):")
    print("-" * 60)
    warm_stadiums = df_outdoor[df_outdoor["home_climate"] == "warm"]
    warm_extreme = warm_stadiums[warm_stadiums["is_extreme_heat"] == 1]
    warm_normal = warm_stadiums[warm_stadiums["is_extreme_heat"] == 0]

    if len(warm_extreme) > 0:
        print(
            f"Extreme heat games: n={len(warm_extreme)}, home_win_rate={warm_extreme['home_won'].mean():.3f}, home_ats={warm_extreme['home_ats'].mean():.3f}"
        )
        print(
            f"Normal temp games:  n={len(warm_normal)}, home_win_rate={warm_normal['home_won'].mean():.3f}, home_ats={warm_normal['home_ats'].mean():.3f}"
        )

        chi2, p_value, _, _ = stats.chi2_contingency(
            pd.crosstab(warm_stadiums["is_extreme_heat"], warm_stadiums["home_won"])
        )
        print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Warm-weather teams have significant edge in extreme heat")
        else:
            print("✗ No significant warm-weather home advantage")
    else:
        print("No extreme heat games in dataset")

    # 4. Climate mismatch analysis
    print("\n4. Climate Mismatch Effect (Warm Team at Cold Stadium in Winter):")
    print("-" * 60)

    # Define winter months
    df_outdoor["month"] = pd.to_datetime(df_outdoor["season"].astype(str) + "-09-01").dt.month

    # Mismatch: warm-climate team visiting cold stadium in extreme cold
    mismatch = df_outdoor[
        (df_outdoor["home_climate"] == "cold")
        & (df_outdoor["away_climate"] == "warm")
        & (df_outdoor["is_extreme_cold"] == 1)
    ]

    normal_matchup = df_outdoor[
        (df_outdoor["home_climate"] == "cold") & (df_outdoor["away_climate"] != "warm")
    ]

    if len(mismatch) > 0:
        print(
            f"Climate mismatch games: n={len(mismatch)}, home_win_rate={mismatch['home_won'].mean():.3f}"
        )
        print(
            f"Normal matchups:        n={len(normal_matchup)}, home_win_rate={normal_matchup['home_won'].mean():.3f}"
        )

        # Expected edge
        edge = mismatch["home_won"].mean() - normal_matchup["home_won"].mean()
        print(f"\nEdge from climate mismatch: {edge:.3f} ({edge*100:.1f}%)")
    else:
        print("No climate mismatch games in dataset")

    # 5. Team-specific analysis: Top cold-weather performers
    print("\n5. Top Cold-Weather Home Advantage (Teams in <5°C Games):")
    print("-" * 60)

    cold_games = df_outdoor[df_outdoor["temp_c"] < 5]
    team_cold_perf = (
        cold_games.groupby("home_team")
        .agg(
            {
                "game_id": "count",
                "home_won": "mean",
                "home_ats": "mean",
                "temp_c": "mean",
            }
        )
        .round(3)
    )
    team_cold_perf.columns = ["n_games", "win_rate", "ats_rate", "avg_temp"]
    team_cold_perf = team_cold_perf[team_cold_perf["n_games"] >= 5]  # Min 5 games
    team_cold_perf = team_cold_perf.sort_values("ats_rate", ascending=False)

    print(team_cold_perf.head(10))

    # 6. Precipitation by stadium
    print("\n6. Precipitation Games by Stadium (Top 10):")
    print("-" * 60)

    precip_games = df_outdoor[df_outdoor["has_precip"] == 1]
    stadium_precip = (
        precip_games.groupby("home_team")
        .agg(
            {
                "game_id": "count",
                "home_won": "mean",
                "temp_c": "mean",
                "precip_mm": "mean",
            }
        )
        .round(3)
    )
    stadium_precip.columns = ["n_precip_games", "win_rate", "avg_temp", "avg_precip_mm"]
    stadium_precip = stadium_precip.sort_values("n_precip_games", ascending=False)

    print(stadium_precip.head(10))

    # 7. Export results for LaTeX
    print("\n7. Exporting Results:")
    print("-" * 60)

    results = {
        "climate_summary": climate_summary.to_dict(),
        "cold_weather_edge": {
            "extreme_cold_games": int(len(cold_extreme)) if len(cold_extreme) > 0 else 0,
            "extreme_cold_win_rate": (
                float(cold_extreme["home_won"].mean()) if len(cold_extreme) > 0 else 0.0
            ),
            "normal_win_rate": float(cold_normal["home_won"].mean()),
        },
        "warm_weather_edge": {
            "extreme_heat_games": int(len(warm_extreme)) if len(warm_extreme) > 0 else 0,
            "extreme_heat_win_rate": (
                float(warm_extreme["home_won"].mean()) if len(warm_extreme) > 0 else 0.0
            ),
            "normal_win_rate": float(warm_normal["home_won"].mean()),
        },
        "climate_mismatch": {
            "n_games": int(len(mismatch)),
            "home_win_rate": float(mismatch["home_won"].mean()) if len(mismatch) > 0 else 0.0,
            "edge": float(edge) if len(mismatch) > 0 else 0.0,
        },
    }

    import json

    out_dir = Path("analysis/dissertation/figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "stadium_weather_clustering_stats.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(
        f"✅ Saved stadium clustering stats to {out_dir / 'stadium_weather_clustering_stats.json'}"
    )

    # Export team-specific cold weather table
    team_cold_perf.to_csv(out_dir / "team_cold_weather_performance.csv")
    print(
        f"✅ Saved team cold weather performance to {out_dir / 'team_cold_weather_performance.csv'}"
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total outdoor games analyzed: {len(df_outdoor)}")
    print(f"Climate zones: {list(CLIMATE_ZONES.keys())}")
    print(
        f"\nKey Finding: Stadium climate zones show {'SIGNIFICANT' if p_value < 0.05 else 'NO SIGNIFICANT'} weather-based home advantage"
    )


if __name__ == "__main__":
    main()
