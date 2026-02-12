"""
Test wind impact on total (over/under) predictions.

Hypothesis: High wind reduces scoring, making unders more likely.

Analysis:
1. Split games by wind speed (low <25 kph, high >=25 kph)
2. Compare over/under hit rates
3. Compute correlation between wind and total points
4. Test statistical significance
"""

import os

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
            gw.wind_kph,
            gw.temp_c,
            gw.has_precip,
            gw.is_dome
        FROM games g
        JOIN mart.game_weather gw ON g.game_id = gw.game_id
        WHERE g.season >= 2020
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND g.total_close IS NOT NULL
          AND gw.wind_kph IS NOT NULL
          AND gw.is_dome = 0  -- Exclude dome games (no weather impact)
        ORDER BY g.season, g.week
    """

    df = pd.read_sql(sql, engine)
    print(f"Loaded {len(df)} outdoor games with wind data (2020-present)")

    # Compute over/under outcomes
    df["went_over"] = (df["total_points"] > df["total_close"]).astype(int)
    df["total_error"] = df["total_points"] - df["total_close"]

    # Wind categories
    df["wind_category"] = pd.cut(
        df["wind_kph"],
        bins=[0, 25, 40, 100],
        labels=["Low (<25 kph)", "Medium (25-40 kph)", "High (>40 kph)"],
    )

    print("\n" + "=" * 80)
    print("WIND IMPACT ON TOTALS (OVER/UNDER)")
    print("=" * 80)

    # 1. Over/Under rates by wind category
    print("\n1. Over/Under Hit Rates by Wind Speed:")
    print("-" * 60)
    summary = (
        df.groupby("wind_category")
        .agg(
            {
                "game_id": "count",
                "went_over": ["mean", "sum"],
                "wind_kph": "mean",
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
        "avg_wind_kph",
        "avg_total_points",
        "avg_total_close",
        "avg_total_error",
    ]
    print(summary)

    # 2. Correlation between wind and scoring
    print("\n2. Correlation Analysis:")
    print("-" * 60)
    corr_wind_total = df[["wind_kph", "total_points"]].corr().iloc[0, 1]
    corr_wind_error = df[["wind_kph", "total_error"]].corr().iloc[0, 1]

    print(f"Correlation (wind_kph, total_points):  {corr_wind_total:.4f}")
    print(f"Correlation (wind_kph, total_error):   {corr_wind_error:.4f}")

    # Pearson correlation test
    r, p_value = stats.pearsonr(df["wind_kph"], df["total_points"])
    print(f"Pearson r = {r:.4f}, p-value = {p_value:.6f}")
    if p_value < 0.05:
        print("✓ Statistically significant at α=0.05")
    else:
        print("✗ Not statistically significant at α=0.05")

    # 3. High wind vs low wind comparison
    print("\n3. High Wind (>40 kph) vs Low Wind (<25 kph):")
    print("-" * 60)
    low_wind = df[df["wind_kph"] < 25]
    high_wind = df[df["wind_kph"] >= 40]

    print(
        f"Low wind games:  n={len(low_wind)}, over_rate={low_wind['went_over'].mean():.3f}, avg_total={low_wind['total_points'].mean():.1f}"
    )
    print(
        f"High wind games: n={len(high_wind)}, over_rate={high_wind['went_over'].mean():.3f}, avg_total={high_wind['total_points'].mean():.1f}"
    )

    # T-test for difference in means
    t_stat, p_value = stats.ttest_ind(low_wind["total_points"], high_wind["total_points"])
    print(f"\nT-test for total_points: t={t_stat:.3f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("✓ Significant difference in scoring between high/low wind")
    else:
        print("✗ No significant difference")

    # Chi-square test for over/under rates
    contingency = pd.crosstab(
        df["wind_kph"] >= 40, df["went_over"], rownames=["high_wind"], colnames=["went_over"]
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test for over/under: χ²={chi2:.3f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("✓ Wind significantly affects over/under outcomes")
    else:
        print("✗ Wind does not significantly affect over/under outcomes")

    # 4. Precipitation interaction
    print("\n4. Precipitation + Wind Interaction:")
    print("-" * 60)
    df_precip = df[df["has_precip"] == 1]
    df_no_precip = df[df["has_precip"] == 0]

    print(
        f"Games with precipitation: n={len(df_precip)}, avg_wind={df_precip['wind_kph'].mean():.1f}, avg_total={df_precip['total_points'].mean():.1f}"
    )
    print(
        f"Games without precip:     n={len(df_no_precip)}, avg_wind={df_no_precip['wind_kph'].mean():.1f}, avg_total={df_no_precip['total_points'].mean():.1f}"
    )

    # 5. Betting strategy: Fade totals in high wind
    print("\n5. Betting Strategy: Fade Totals in High Wind (>40 kph):")
    print("-" * 60)
    high_wind_under_wins = (high_wind["went_over"] == 0).sum()
    high_wind_under_rate = high_wind_under_wins / len(high_wind)
    print(
        f"Under win rate in high wind: {high_wind_under_rate:.3f} ({high_wind_under_wins}/{len(high_wind)})"
    )
    print(f"Expected edge vs 50-50:      {(high_wind_under_rate - 0.5):.3f}")

    # ROI calculation (assume -110 odds)
    if high_wind_under_rate > 0.524:  # Break-even at 52.4%
        roi = (high_wind_under_rate * 1.91 - 1) * 100
        print(f"✓ Profitable strategy! ROI = {roi:.2f}%")
    else:
        print("✗ Not profitable (below break-even 52.4%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total games analyzed: {len(df)}")
    print(f"Average wind speed: {df['wind_kph'].mean():.1f} kph")
    print(f"Overall over rate: {df['went_over'].mean():.3f}")
    print(f"\nKey Finding: Wind speed correlation with total points = {corr_wind_total:.4f}")
    if corr_wind_total < 0:
        print("✓ Negative correlation confirms hypothesis: Higher wind → Lower scoring")
    else:
        print("✗ Unexpected: Wind positively correlated with scoring")


if __name__ == "__main__":
    main()
