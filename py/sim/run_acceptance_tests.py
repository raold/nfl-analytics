#!/usr/bin/env python3
"""
Run simulator acceptance tests comparing historical vs simulated metrics.

This script:
1. Loads historical and simulated metrics
2. Computes Earth Mover's Distance (EMD) for margin/total distributions
3. Computes key number mass delta
4. Computes dependence metric delta (Kendall's tau)
5. Generates acceptance test results

Output: data/sim_acceptance.csv

Usage:
    python py/sim/run_acceptance_tests.py
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def load_metrics(path: str) -> Dict:
    """Load metrics JSON file."""
    with open(path) as f:
        return json.load(f)


def pmf_to_arrays(pmf_dict: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert PMF dictionary to (values, probabilities) arrays.

    Args:
        pmf_dict: Dict mapping value -> probability

    Returns:
        (values, probabilities) as numpy arrays
    """
    items = sorted((int(k), v) for k, v in pmf_dict.items())
    values = np.array([k for k, v in items])
    probs = np.array([v for k, v in items])
    return values, probs


def compute_emd(hist_pmf: Dict, sim_pmf: Dict) -> float:
    """
    Compute Earth Mover's Distance between two PMFs.

    Args:
        hist_pmf: Historical PMF
        sim_pmf: Simulated PMF

    Returns:
        EMD (Wasserstein distance)
    """
    # Convert to arrays
    hist_vals, hist_probs = pmf_to_arrays(hist_pmf)
    sim_vals, sim_probs = pmf_to_arrays(sim_pmf)

    # Use scipy's wasserstein_distance (1D EMD)
    emd = wasserstein_distance(hist_vals, sim_vals, hist_probs, sim_probs)

    return float(emd)


def compute_key_mass_delta(
    hist_masses: Dict[int, float],
    sim_masses: Dict[int, float]
) -> Dict[int, float]:
    """
    Compute absolute difference in key number masses.

    Args:
        hist_masses: Historical key masses
        sim_masses: Simulated key masses

    Returns:
        Dict mapping key number -> absolute delta
    """
    deltas = {}
    for key in hist_masses.keys():
        hist_val = hist_masses.get(key, 0.0)
        sim_val = sim_masses.get(key, 0.0)
        deltas[key] = abs(hist_val - sim_val)

    return deltas


def compute_dependence_delta(
    hist_dep: Dict,
    sim_dep: Dict
) -> float:
    """
    Compute absolute difference in Kendall's tau.

    Args:
        hist_dep: Historical dependence metrics
        sim_dep: Simulated dependence metrics

    Returns:
        Absolute delta in Kendall's tau
    """
    hist_tau = hist_dep.get('kendall_tau', 0.0)
    sim_tau = sim_dep.get('kendall_tau', 0.0)
    return abs(hist_tau - sim_tau)


def run_acceptance_tests(
    historical_path: str = "analysis/results/historical_metrics.json",
    simulated_path: str = "analysis/results/simulated_metrics.json",
    output_path: str = "data/sim_acceptance.csv"
):
    """
    Run all acceptance tests.

    Args:
        historical_path: Path to historical metrics JSON
        simulated_path: Path to simulated metrics JSON
        output_path: Path to save results CSV
    """
    print("=" * 80)
    print("SIMULATOR ACCEPTANCE TESTS")
    print("=" * 80)

    # Load metrics
    print("\n1. Loading metrics...")
    hist = load_metrics(historical_path)
    sim = load_metrics(simulated_path)
    print(f"✅ Loaded historical ({hist['metadata']['n_games']} games)")
    print(f"✅ Loaded simulated ({sim['metadata']['n_games']} games)")

    # Test 1: Margin EMD
    print("\n2. Computing margin distribution EMD...")
    margin_emd = compute_emd(
        hist['margin_distribution']['pmf'],
        sim['margin_distribution']['pmf']
    )
    print(f"✅ Margin EMD: {margin_emd:.4f}")

    # Test 2: Total EMD
    print("\n3. Computing total distribution EMD...")
    total_emd = compute_emd(
        hist['total_distribution']['pmf'],
        sim['total_distribution']['pmf']
    )
    print(f"✅ Total EMD: {total_emd:.4f}")

    # Test 3: Key number masses
    print("\n4. Computing key number mass deltas...")
    key_deltas = compute_key_mass_delta(
        hist['key_number_masses'],
        sim['key_number_masses']
    )
    print("✅ Key mass deltas:")
    for key, delta in key_deltas.items():
        print(f"   {int(key):2d} points: {delta*100:.2f}% "
              f"(hist={hist['key_number_masses'][key]*100:.2f}%, "
              f"sim={sim['key_number_masses'][key]*100:.2f}%)")

    # Test 4: Dependence
    print("\n5. Computing dependence delta...")
    dep_delta = compute_dependence_delta(
        hist['score_dependence'],
        sim['score_dependence']
    )
    print(f"✅ Kendall's tau delta: {dep_delta:.4f}")
    print(f"   Historical: {hist['score_dependence']['kendall_tau']:.4f}")
    print(f"   Simulated: {sim['score_dependence']['kendall_tau']:.4f}")

    # Test 5: Home win rate
    print("\n6. Computing home win rate delta...")
    home_win_delta = abs(hist['home_win_rate'] - sim['home_win_rate'])
    print(f"✅ Home win rate delta: {home_win_delta*100:.2f}%")
    print(f"   Historical: {hist['home_win_rate']*100:.1f}%")
    print(f"   Simulated: {sim['home_win_rate']*100:.1f}%")

    # Compile results
    results = []

    # Add margin EMD
    results.append({
        'test': 'margin_emd',
        'metric': 'Earth Mover\'s Distance',
        'value': margin_emd,
        'threshold': 2.0,
        'pass': margin_emd < 2.0
    })

    # Add total EMD
    results.append({
        'test': 'total_emd',
        'metric': 'Earth Mover\'s Distance',
        'value': total_emd,
        'threshold': 3.0,
        'pass': total_emd < 3.0
    })

    # Add key masses
    for key, delta in key_deltas.items():
        results.append({
            'test': f'key_mass_{key}pt',
            'metric': f'{key}-point mass delta',
            'value': delta,
            'threshold': 0.05,  # 5% tolerance
            'pass': delta < 0.05
        })

    # Add dependence
    results.append({
        'test': 'kendall_tau_delta',
        'metric': 'Kendall\'s tau delta',
        'value': dep_delta,
        'threshold': 0.10,  # 0.10 tolerance
        'pass': dep_delta < 0.10
    })

    # Add home win rate
    results.append({
        'test': 'home_win_rate_delta',
        'metric': 'Home win rate delta',
        'value': home_win_delta,
        'threshold': 0.10,  # 10% tolerance
        'pass': home_win_delta < 0.10
    })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    print(f"\n7. Saving results to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("ACCEPTANCE TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests: {len(df)}")
    print(f"Passed: {df['pass'].sum()}")
    print(f"Failed: {(~df['pass']).sum()}")

    print("\nFailed tests:")
    failed = df[~df['pass']]
    if len(failed) == 0:
        print("  None - all tests passed! ✅")
    else:
        for _, row in failed.iterrows():
            print(f"  • {row['test']}: {row['value']:.4f} > {row['threshold']:.4f}")

    print("\n" + "=" * 80)

    # Overall pass/fail
    if df['pass'].all():
        print("✅ SIMULATOR ACCEPTANCE: PASSED")
    else:
        print(f"⚠️  SIMULATOR ACCEPTANCE: {(~df['pass']).sum()} TESTS FAILED")

    print("=" * 80)

    return df


if __name__ == "__main__":
    run_acceptance_tests()
