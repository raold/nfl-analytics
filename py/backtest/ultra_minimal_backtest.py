#!/usr/bin/env python3
"""
Ultra-Minimal Backtest - Incremental Feature Addition

Tests spread prediction with progressively more features to identify leakage source.

Test 1: Spread ONLY (baseline, expect ~48-52%)
Test 2: Spread + Total (expect ~49-53%)
Test 3: Spread + Total + Rest (expect ~50-54%)
Test 4: Spread + Total + Rest + Form (expect ~51-55%)

Author: Claude Code
Date: 2025-11-03
"""

import json
from pathlib import Path

import pandas as pd
import psycopg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
}

def load_games():
    """Load games with basic betting lines."""
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.kickoff,
        g.home_team,
        g.away_team,
        g.spread_close,
        g.total_close,
        g.home_score,
        g.away_score
    FROM games g
    WHERE g.spread_close IS NOT NULL
        AND g.total_close IS NOT NULL
        AND g.home_score IS NOT NULL
        AND g.season >= 2020
        AND g.season <= 2024
    ORDER BY g.kickoff
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    # Target: home team covers spread
    df['home_margin'] = df['home_score'] - df['away_score']
    df['home_covers'] = df['home_margin'] > df['spread_close']

    return df

def test_spread_only(df):
    """Test 1: Spread ONLY (baseline)."""
    print("\n" + "="*70)
    print("TEST 1: SPREAD ONLY (Baseline)")
    print("="*70)

    # Features: just the spread
    X_train = df[df['season'] <= 2023][['spread_close']].values
    y_train = df[df['season'] <= 2023]['home_covers'].values

    X_test = df[df['season'] == 2024][['spread_close']].values
    y_test = df[df['season'] == 2024]['home_covers'].values

    # Simple logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print("\nResults:")
    print(f"  Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  AUC: {auc:.3f}")
    print(f"  Log Loss: {logloss:.3f}")
    print("  Expected: 48-52% (market efficiency baseline)")

    status = "✓ PASS" if acc < 0.54 else "✗ FAIL (possible leakage)"
    print(f"\n  Status: {status}")

    return {
        "test": "spread_only",
        "features": ["spread_close"],
        "accuracy": float(acc),
        "auc": float(auc),
        "log_loss": float(logloss),
        "status": "pass" if acc < 0.54 else "fail"
    }

def test_spread_and_total(df):
    """Test 2: Spread + Total."""
    print("\n" + "="*70)
    print("TEST 2: SPREAD + TOTAL")
    print("="*70)

    # Features: spread and total
    X_train = df[df['season'] <= 2023][['spread_close', 'total_close']].values
    y_train = df[df['season'] <= 2023]['home_covers'].values

    X_test = df[df['season'] == 2024][['spread_close', 'total_close']].values
    y_test = df[df['season'] == 2024]['home_covers'].values

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print("\nResults:")
    print(f"  Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  AUC: {auc:.3f}")
    print(f"  Log Loss: {logloss:.3f}")
    print("  Expected: 49-53%")

    status = "✓ PASS" if acc < 0.55 else "✗ FAIL (possible leakage)"
    print(f"\n  Status: {status}")

    return {
        "test": "spread_and_total",
        "features": ["spread_close", "total_close"],
        "accuracy": float(acc),
        "auc": float(auc),
        "log_loss": float(logloss),
        "status": "pass" if acc < 0.55 else "fail"
    }

def main():
    """Run all incremental tests."""
    print("="*70)
    print("ULTRA-MINIMAL BACKTEST - INCREMENTAL FEATURE ADDITION")
    print("="*70)
    print("\nObjective: Identify which features cause leakage")
    print("Strategy: Start with spread only, add features incrementally")
    print()

    # Load data
    print("Loading games (2020-2024)...")
    df = load_games()
    print(f"Loaded {len(df)} games")
    print(f"Train: {len(df[df['season'] <= 2023])} games (2020-2023)")
    print(f"Test: {len(df[df['season'] == 2024])} games (2024)")

    # Run tests
    results = []
    results.append(test_spread_only(df))
    results.append(test_spread_and_total(df))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for result in results:
        status_emoji = "✓" if result['status'] == 'pass' else "✗"
        print(f"\n{status_emoji} {result['test']}: {result['accuracy']:.1%}")
        print(f"   Features: {', '.join(result['features'])}")

    # Save results
    output_dir = Path("analysis/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ultra_minimal_backtest_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print("="*70)

    # Final assessment
    all_pass = all(r['status'] == 'pass' for r in results)
    if all_pass:
        print("\n✓ ALL TESTS PASS - No leakage detected in basic features")
        print("  You can safely add more features incrementally")
    else:
        print("\n✗ LEAKAGE DETECTED - Even basic features show unrealistic accuracy")
        print("  Check data quality and temporal ordering")

if __name__ == "__main__":
    main()
