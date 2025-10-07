#!/usr/bin/env python3
"""
Test temperature impact on total (over/under) predictions.

Hypothesis: Extreme temperatures (cold <0°C, heat >30°C) reduce scoring.

Analysis:
1. Split games by temperature categories (extreme cold, cold, moderate, warm, extreme heat)
2. Compare over/under hit rates and total scoring
3. Compute correlation between temperature and total points
4. Test statistical significance
5. Examine temperature × precipitation interactions
"""

import os

import pandas as pd
import numpy as np
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


def main():
    engine = make_engine()

    # Fetch games with weather and totals
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
          AND g.total_close IS NOT NULL
          AND gw.temp_c IS NOT NULL
          AND gw.is_dome = 0  -- Exclude dome games (no weather impact)
        ORDER BY g.season, g.week
    """

    df = pd.read_sql(sql, engine)
    print(f"Loaded {len(df)} outdoor games with temperature data (2020-present)")

    # Compute over/under outcomes
    df["went_over"] = (df["total_points"] > df["total_close"]).astype(int)
    df["total_error"] = df["total_points"] - df["total_close"]

    # Temperature categories
    df["temp_category"] = pd.cut(
        df["temp_c"],
        bins=[-50, 0, 10, 20, 30, 50],
        labels=["Extreme Cold (<0°C)", "Cold (0-10°C)", "Moderate (10-20°C)",
                "Warm (20-30°C)", "Extreme Heat (>30°C)"],
    )

    # Additional binary flags
    df["is_freezing"] = (df["temp_c"] < 0).astype(int)
    df["is_extreme_heat"] = (df["temp_c"] > 30).astype(int)
    df["is_extreme"] = ((df["temp_c"] < 0) | (df["temp_c"] > 30)).astype(int)

    print("\n" + "=" * 80)
    print("TEMPERATURE IMPACT ON TOTALS (OVER/UNDER)")
    print("=" * 80)

    # 1. Over/Under rates by temperature category
    print("\n1. Over/Under Hit Rates by Temperature:")
    print("-" * 60)
    summary = (
        df.groupby("temp_category")
        .agg(
            {
                "game_id": "count",
                "went_over": ["mean", "sum"],
                "temp_c": "mean",
                "total_points": "mean",
                "total_close": "mean",
                "total_error": "mean",
            }
        )
        .round(3)
    )
    summary.columns = [
        "n_games",
        "over_rate",
        "n_overs",
        "avg_temp_c",
        "avg_total_points",
        "avg_total_close",
        "avg_total_error",
    ]
    print(summary)

    # 2. Correlation between temperature and scoring
    print("\n2. Correlation Analysis:")
    print("-" * 60)
    corr_temp_total = df[["temp_c", "total_points"]].corr().iloc[0, 1]
    corr_temp_error = df[["temp_c", "total_error"]].corr().iloc[0, 1]
    corr_temp_extreme_total = df[["temp_extreme", "total_points"]].corr().iloc[0, 1]

    print(f"Correlation (temp_c, total_points):         {corr_temp_total:.4f}")
    print(f"Correlation (temp_c, total_error):          {corr_temp_error:.4f}")
    print(f"Correlation (temp_extreme, total_points):   {corr_temp_extreme_total:.4f}")

    # Pearson correlation test
    r, p_value = stats.pearsonr(df["temp_c"], df["total_points"])
    print(f"\nPearson r = {r:.4f}, p-value = {p_value:.6f}")
    if p_value < 0.05:
        print("✓ Statistically significant at α=0.05")
    else:
        print("✗ Not statistically significant at α=0.05")

    # 3. Extreme temperature vs moderate comparison
    print("\n3. Extreme Temperature (|temp - 15°C| > 15°C) vs Moderate:")
    print("-" * 60)
    moderate_temp = df[df["is_extreme"] == 0]
    extreme_temp = df[df["is_extreme"] == 1]

    print(
        f"Moderate games: n={len(moderate_temp)}, over_rate={moderate_temp['went_over'].mean():.3f}, avg_total={moderate_temp['total_points'].mean():.1f}"
    )
    print(
        f"Extreme games:  n={len(extreme_temp)}, over_rate={extreme_temp['went_over'].mean():.3f}, avg_total={extreme_temp['total_points'].mean():.1f}"
    )

    # T-test for difference in means
    if len(extreme_temp) > 0:
        t_stat, p_value = stats.ttest_ind(moderate_temp["total_points"], extreme_temp["total_points"])
        print(f"\nT-test for total_points: t={t_stat:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Significant difference in scoring between extreme/moderate temps")
        else:
            print("✗ No significant difference")

        # Chi-square test for over/under rates
        contingency = pd.crosstab(
            df["is_extreme"], df["went_over"], rownames=["is_extreme"], colnames=["went_over"]
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test for over/under: χ²={chi2:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Temperature extremes significantly affect over/under outcomes")
        else:
            print("✗ Temperature extremes do not significantly affect over/under outcomes")

    # 4. Freezing conditions analysis
    print("\n4. Freezing Conditions (<0°C) Analysis:")
    print("-" * 60)
    freezing = df[df["is_freezing"] == 1]
    non_freezing = df[df["is_freezing"] == 0]

    if len(freezing) > 0:
        print(
            f"Freezing games:     n={len(freezing)}, avg_temp={freezing['temp_c'].mean():.1f}°C, avg_total={freezing['total_points'].mean():.1f}"
        )
        print(
            f"Non-freezing games: n={len(non_freezing)}, avg_temp={non_freezing['temp_c'].mean():.1f}°C, avg_total={non_freezing['total_points'].mean():.1f}"
        )

        t_stat, p_value = stats.ttest_ind(freezing["total_points"], non_freezing["total_points"])
        print(f"\nT-test: t={t_stat:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Freezing temps significantly reduce scoring")
        else:
            print("✗ No significant freezing effect")
    else:
        print("No freezing games in dataset")

    # 5. Extreme heat analysis
    print("\n5. Extreme Heat (>30°C) Analysis:")
    print("-" * 60)
    extreme_heat = df[df["is_extreme_heat"] == 1]
    non_extreme_heat = df[df["is_extreme_heat"] == 0]

    if len(extreme_heat) > 0:
        print(
            f"Extreme heat games: n={len(extreme_heat)}, avg_temp={extreme_heat['temp_c'].mean():.1f}°C, avg_total={extreme_heat['total_points'].mean():.1f}"
        )
        print(
            f"Normal temp games:  n={len(non_extreme_heat)}, avg_temp={non_extreme_heat['temp_c'].mean():.1f}°C, avg_total={non_extreme_heat['total_points'].mean():.1f}"
        )

        t_stat, p_value = stats.ttest_ind(extreme_heat["total_points"], non_extreme_heat["total_points"])
        print(f"\nT-test: t={t_stat:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print("✓ Extreme heat significantly affects scoring")
        else:
            print("✗ No significant extreme heat effect")
    else:
        print("No extreme heat games in dataset")

    # 6. Temperature × Precipitation interaction
    print("\n6. Temperature × Precipitation Interaction:")
    print("-" * 60)
    df_precip = df[df["has_precip"] == 1]

    if len(df_precip) > 0:
        cold_precip = df_precip[df_precip["temp_c"] < 5]  # Potential snow/ice
        warm_precip = df_precip[df_precip["temp_c"] >= 5]  # Rain

        if len(cold_precip) > 0:
            print(
                f"Cold + precip (likely snow): n={len(cold_precip)}, avg_temp={cold_precip['temp_c'].mean():.1f}°C, avg_total={cold_precip['total_points'].mean():.1f}"
            )
        else:
            print("Cold + precip (likely snow): n=0")

        if len(warm_precip) > 0:
            print(
                f"Warm + precip (likely rain):  n={len(warm_precip)}, avg_temp={warm_precip['temp_c'].mean():.1f}°C, avg_total={warm_precip['total_points'].mean():.1f}"
            )
        else:
            print("Warm + precip (likely rain): n=0")

    # 7. Quadratic relationship test (U-shaped curve)
    print("\n7. Quadratic Temperature Model:")
    print("-" * 60)

    # Create temp^2 feature
    df["temp_squared"] = df["temp_c"] ** 2

    # Linear model: total_points ~ temp_c
    from scipy.stats import linregress
    slope_linear, intercept_linear, r_linear, p_linear, se_linear = linregress(
        df["temp_c"], df["total_points"]
    )

    # R-squared for quadratic model (using numpy polyfit)
    coeffs = np.polyfit(df["temp_c"], df["total_points"], 2)
    poly_fit = np.polyval(coeffs, df["temp_c"])
    ss_res = np.sum((df["total_points"] - poly_fit) ** 2)
    ss_tot = np.sum((df["total_points"] - df["total_points"].mean()) ** 2)
    r_squared_quadratic = 1 - (ss_res / ss_tot)

    print(f"Linear model R² = {r_linear**2:.4f}, p={p_linear:.6f}")
    print(f"Quadratic model R² = {r_squared_quadratic:.4f}")
    print(f"Optimal temperature (from quadratic): {-coeffs[1]/(2*coeffs[0]):.1f}°C")

    if r_squared_quadratic > r_linear**2:
        print("✓ Quadratic model fits better (U-shaped relationship)")
    else:
        print("✗ Linear model sufficient")

    # 8. Betting strategy: Bet unders in extreme temps
    print("\n8. Betting Strategy: Bet Unders in Extreme Temps:")
    print("-" * 60)
    if len(extreme_temp) > 0:
        extreme_under_wins = (extreme_temp["went_over"] == 0).sum()
        extreme_under_rate = extreme_under_wins / len(extreme_temp)
        print(
            f"Under win rate in extreme temps: {extreme_under_rate:.3f} ({extreme_under_wins}/{len(extreme_temp)})"
        )
        print(f"Expected edge vs 50-50:          {(extreme_under_rate - 0.5):.3f}")

        # ROI calculation (assume -110 odds)
        if extreme_under_rate > 0.524:  # Break-even at 52.4%
            roi = (extreme_under_rate * 1.91 - 1) * 100
            print(f"✓ Profitable strategy! ROI = {roi:.2f}%")
        else:
            print("✗ Not profitable (below break-even 52.4%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total games analyzed: {len(df)}")
    print(f"Temperature range: {df['temp_c'].min():.1f}°C to {df['temp_c'].max():.1f}°C")
    print(f"Mean temperature: {df['temp_c'].mean():.1f}°C")
    print(f"Overall over rate: {df['went_over'].mean():.3f}")
    print(f"\nKey Finding: Temperature correlation with total points = {corr_temp_total:.4f}")
    print(f"             Temp extreme correlation with total points = {corr_temp_extreme_total:.4f}")

    if abs(corr_temp_extreme_total) > abs(corr_temp_total):
        print("✓ Temperature extremes (deviation from 15°C) better predictor than raw temp")
    else:
        print("✗ Raw temperature is sufficient predictor")

    # Export results for LaTeX table generation
    results = {
        "n_games": len(df),
        "temp_range": (df["temp_c"].min(), df["temp_c"].max()),
        "temp_mean": df["temp_c"].mean(),
        "corr_temp_total": corr_temp_total,
        "corr_temp_extreme_total": corr_temp_extreme_total,
        "p_value": p_value,
        "extreme_games": len(extreme_temp),
        "freezing_games": len(freezing) if len(freezing) > 0 else 0,
        "extreme_heat_games": len(extreme_heat) if len(extreme_heat) > 0 else 0,
        "r_squared_quadratic": r_squared_quadratic,
        "optimal_temp": -coeffs[1]/(2*coeffs[0]),
    }

    # Save summary statistics
    import json
    from pathlib import Path

    out_dir = Path("analysis/dissertation/figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "temperature_impact_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved temperature impact statistics to {out_dir / 'temperature_impact_stats.json'}")


if __name__ == "__main__":
    main()
