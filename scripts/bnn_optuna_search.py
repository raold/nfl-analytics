
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("[*] Loading data from database...")
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

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
            nn.Linear(hidden_dim // 2, 2)  # yards + td prob
        )

    def forward(self, x):
        return self.net(x)

def objective(trial):
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Create model
    model = BNN(X_train_scaled.shape[1], hidden_dim, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    epochs = 50  # Quick training for hyperparameter search
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(np.column_stack([y_yards_train, y_tds_train])).to(device)

    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = nn.functional.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        pred = model(X_test_tensor).cpu().numpy()

        mae_yards = mean_absolute_error(y_yards_test, pred[:, 0])
        mae_tds = mean_absolute_error(y_tds_test, pred[:, 1])

        # Combined metric (lower is better)
        combined_mae = mae_yards + mae_tds * 100  # Weight TDs heavily

    return combined_mae

print("[*] Starting Optuna optimization with 100 trials...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n[+] Optimization complete!")
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.2f}")
print(f"Best params: {study.best_params}")

# Save results
output_dir = Path("models/bayesian")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "bnn_optuna_results.json", 'w') as f:
    json.dump({
        'best_trial': study.best_trial.number,
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'timestamp': str(pd.Timestamp.now())
    }, f, indent=2)

print(f"[+] Results saved to {output_dir}/bnn_optuna_results.json")
