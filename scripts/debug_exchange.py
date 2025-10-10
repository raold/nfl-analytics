import pandas as pd
import numpy as np
import json

# Load results
with open('results/exchange_simulation_ev_2024.json') as f:
    results = json.load(f)

print("="*60)
print("DEBUGGING EXCHANGE SIMULATION")
print("="*60)

# Load sample data
df = pd.read_csv('data/processed/features/asof_team_features_v2.csv')
df_2024 = df[df['season'] == 2024].head(10)

print("\nSample spread conversions:")
for _, row in df_2024.iterrows():
    spread = row['spread_close']
    p_home = 1.0 / (1.0 + np.exp(spread * 0.4))
    print(f"Game: {row['away_team']} @ {row['home_team']}")
    print(f"  Spread: {spread:.1f}")
    print(f"  P(home) from spread: {p_home:.3f}")
    print(f"  Home won: {row['home_win']}")
    print()
