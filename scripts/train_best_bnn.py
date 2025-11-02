
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# Load best hyperparameters
with open('models/bayesian/bnn_optuna_results.json') as f:
    optuna_results = json.load(f)

best_params = optuna_results['best_params']
print(f"[*] Training final BNN with best params: {best_params}")

# Load data
conn = psycopg2.connect(
    host='localhost', port=5544, dbname='devdb01',
    user='dro', password='sicillionbillions'
)

query = """
SELECT
    pgs.player_id,
    pgs.season,
    pgs.week,
    pgs.rushing_yards,
    pgs.rushing_tds,
    pgs.team,
    w.temp,
    w.wind,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score
FROM mart.player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
AND pgs.season >= 2020
ORDER BY pgs.season, pgs.week
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"[+] Loaded {len(df):,} rushing performances")

# Derive is_home feature
df['is_home'] = (df['team'] == df['home_team']).astype(int)

# Parse weather columns (TEXT format, handle DOME/CLOSED ROOF/None)
df['temp_numeric'] = pd.to_numeric(df['temp'], errors='coerce').fillna(70)
df['wind_numeric'] = pd.to_numeric(df['wind'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0)

# Calculate point differential
df['point_differential'] = df.apply(
    lambda row: row['home_score'] - row['away_score'] if row['is_home'] else row['away_score'] - row['home_score'],
    axis=1
)

# Prepare features
X = df[['temp_numeric', 'wind_numeric', 'is_home', 'point_differential']].values
y_yards = df['rushing_yards'].values
y_tds = df['rushing_tds'].values

X_train, X_test, y_yards_train, y_yards_test, y_tds_train, y_tds_test = train_test_split(
    X, y_yards, y_tds, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

device = torch.device('cuda')

class BNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x)

# Train with best params
model = BNN(
    X_train_scaled.shape[1],
    best_params['hidden_dim'],
    best_params['dropout_rate']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
batch_size = best_params['batch_size']

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(np.column_stack([y_yards_train, y_tds_train])).to(device)

print("[*] Training final model with 500 epochs...")
model.train()
for epoch in range(500):
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        pred = model(batch_X)
        loss = nn.functional.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500 | Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    pred = model(X_test_tensor).cpu().numpy()

    mae_yards = mean_absolute_error(y_yards_test, pred[:, 0])
    r2_yards = r2_score(y_yards_test, pred[:, 0])
    mae_tds = mean_absolute_error(y_tds_test, pred[:, 1])

    print(f"\n[+] Final BNN Evaluation:")
    print(f"  Yards MAE: {mae_yards:.2f}")
    print(f"  Yards R�: {r2_yards:.3f}")
    print(f"  TDs MAE: {mae_tds:.3f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': best_params,
    'metrics': {
        'mae_yards': float(mae_yards),
        'r2_yards': float(r2_yards),
        'mae_tds': float(mae_tds)
    }
}, 'models/bayesian/bnn_optimized_final.pt')

print("[+] Model saved to models/bayesian/bnn_optimized_final.pt")
