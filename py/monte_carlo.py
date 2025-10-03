# py/monte_carlo.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# train simple models for home/away points
train = pd.read_parquet("data/train_points.parquet")
X = train[["spread_close","epa_gap","temp_c","wind_kph","precip_mm"]].fillna(0)
y_home = train["home_score"]
y_away = train["away_score"]

m_home = GradientBoostingRegressor().fit(X, y_home)
m_away = GradientBoostingRegressor().fit(X, y_away)

def simulate_prob_row(xrow, n=100000):
    mu_h = float(m_home.predict(xrow)[0])
    mu_a = float(m_away.predict(xrow)[0])
    # gaussian approx (fast). swap for Poisson/Skellam if desired.
    sim_h = np.random.normal(mu_h, 10, n)
    sim_a = np.random.normal(mu_a, 10, n)
    spread = sim_h - sim_a
    return {
      "p_favorite_covers": float((spread > 0).mean()),
      "p_push": float((np.isclose(spread, 0, atol=0.5)).mean()),
      "mean_spread": float(spread.mean()),
      "p_total_over_45": float((sim_h + sim_a > 45).mean())
    }
