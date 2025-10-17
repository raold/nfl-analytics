#!/usr/bin/env python3
"""
Phase 1 vs Phase 2.1 Comparison Framework

Automatically compares Phase 2.1 (deeper BNN) results to Phase 1 baseline
to evaluate if architectural improvements fixed calibration.

Author: Richard Oldham
Date: October 2024
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sns.set_style("whitegrid")


class PhaseComparator:
    """Compare Phase 1 and Phase 2.1 results"""

    def __init__(self):
        self.results_dir = Path("/Users/dro/rice/nfl-analytics/experiments/calibration")
        self.phase1_results = self.load_phase1()
        self.phase21_results = self.load_phase21()

    def load_phase1(self) -> dict:
        """Load Phase 1 baseline (average across all sigma values)"""
        sigma_values = [0.5, 0.7, 1.0, 1.5]
        results = []

        for sigma in sigma_values:
            file_path = self.results_dir / f"prior_sensitivity_sigma{sigma:.1f}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    results.append(json.load(f))

        if not results:
            raise FileNotFoundError("Phase 1 results not found!")

        # Average across all sigma values (since they're all similar)
        avg_result = {
            'mae': np.mean([r['accuracy']['mae'] for r in results]),
            'rmse': np.mean([r['accuracy']['rmse'] for r in results]),
            'coverage_90': np.mean([r['calibration']['90%_coverage'] for r in results]),
            'coverage_68': np.mean([r['calibration']['68%_coverage'] for r in results]),
            'n_models': len(results)
        }

        print(f"✓ Loaded Phase 1 baseline (averaged across {len(results)} models)")
        return avg_result

    def load_phase21(self) -> dict:
        """Load Phase 2.1 deeper BNN results"""
        file_path = self.results_dir / "deeper_bnn_v2_results.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Phase 2.1 results not found at: {file_path}\n"
                "Training may not be complete yet."
            )

        with open(file_path, 'r') as f:
            result = json.load(f)

        # Normalize to match Phase 1 format
        normalized = {
            'mae': result['mae'],
            'rmse': result['rmse'],
            'coverage_90': result['coverage_90'] / 100,  # Convert from % to decimal
            'coverage_68': result['coverage_68'] / 100,
            'architecture': 'deep_bnn_v2'
        }

        print(f"✓ Loaded Phase 2.1 results")
        return normalized

    def calculate_improvements(self) -> dict:
        """Calculate improvements from Phase 1 to Phase 2.1"""
        improvements = {}

        # Absolute improvements
        improvements['coverage_90_abs'] = (
            self.phase21_results['coverage_90'] - self.phase1_results['coverage_90']
        ) * 100  # Convert to percentage points

        improvements['coverage_68_abs'] = (
            self.phase21_results['coverage_68'] - self.phase1_results['coverage_68']
        ) * 100

        improvements['mae_abs'] = (
            self.phase1_results['mae'] - self.phase21_results['mae']
        )  # Positive = improvement

        # Relative improvements
        improvements['coverage_90_rel'] = (
            improvements['coverage_90_abs'] / (90 - self.phase1_results['coverage_90'] * 100)
        ) * 100 if (90 - self.phase1_results['coverage_90'] * 100) != 0 else 0

        improvements['mae_rel'] = (
            improvements['mae_abs'] / self.phase1_results['mae']
        ) * 100

        return improvements

    def print_comparison(self):
        """Print comprehensive comparison"""
        improvements = self.calculate_improvements()

        print("\n" + "="*80)
        print("PHASE 1 vs PHASE 2.1: CALIBRATION COMPARISON")
        print("="*80)

        print("\n📊 RESULTS COMPARISON:")
        print(f"{'Metric':<25} {'Phase 1':<15} {'Phase 2.1':<15} {'Improvement':<20}")
        print("-" * 80)

        # Coverage 90%
        p1_cov90 = self.phase1_results['coverage_90'] * 100
        p21_cov90 = self.phase21_results['coverage_90'] * 100
        print(f"{'90% CI Coverage':<25} {p1_cov90:>6.1f}%{'':<8} {p21_cov90:>6.1f}%{'':<8} "
              f"{improvements['coverage_90_abs']:>+6.1f} pp{'':<8}")

        # Coverage 68%
        p1_cov68 = self.phase1_results['coverage_68'] * 100
        p21_cov68 = self.phase21_results['coverage_68'] * 100
        print(f"{'68% CI Coverage':<25} {p1_cov68:>6.1f}%{'':<8} {p21_cov68:>6.1f}%{'':<8} "
              f"{improvements['coverage_68_abs']:>+6.1f} pp{'':<8}")

        # MAE
        print(f"{'MAE (yards)':<25} {self.phase1_results['mae']:>6.2f}{'':<9} "
              f"{self.phase21_results['mae']:>6.2f}{'':<9} "
              f"{improvements['mae_abs']:>+6.2f}{'':<11}")

        # RMSE
        print(f"{'RMSE (yards)':<25} {self.phase1_results['rmse']:>6.2f}{'':<9} "
              f"{self.phase21_results['rmse']:>6.2f}{'':<9} "
              f"{self.phase21_results['rmse'] - self.phase1_results['rmse']:>+6.2f}{'':<11}")

        print("\n📈 CALIBRATION ASSESSMENT:")
        print(f"  • Phase 1 Error (90% CI): {p1_cov90 - 90:+.1f} pp (target: 90%)")
        print(f"  • Phase 2.1 Error (90% CI): {p21_cov90 - 90:+.1f} pp (target: 90%)")
        print(f"  • Gap Closed: {improvements['coverage_90_rel']:.1f}% of the way to target")

        print("\n🎯 SUCCESS CRITERIA:")
        target_met = p21_cov90 >= 75
        if target_met:
            print(f"  ✅ SUCCESS: 90% Coverage = {p21_cov90:.1f}% (target: ≥75%)")
            print(f"  → Architectural improvements worked!")
            print(f"  → Ready to proceed to Phase 2.3 (database expansion)")
        elif p21_cov90 >= 50:
            print(f"  ⚠️  PARTIAL SUCCESS: 90% Coverage = {p21_cov90:.1f}% (target: ≥75%)")
            print(f"  → Improved from {p1_cov90:.1f}% but below target")
            print(f"  → Next: Try Phase 2.2 (mixture-of-experts)")
        else:
            print(f"  ❌ INSUFFICIENT: 90% Coverage = {p21_cov90:.1f}% (target: ≥75%)")
            print(f"  → Minimal improvement from {p1_cov90:.1f}%")
            print(f"  → Next: Implement hybrid calibration (Phase 2.3 fallback)")

        return improvements

    def create_comparison_plot(self, output_path: Path):
        """Create side-by-side comparison visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Phase 1 vs Phase 2.1: Calibration Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Coverage Comparison
        ax = axes[0]
        metrics = ['90% CI', '68% CI']
        phase1_vals = [
            self.phase1_results['coverage_90'] * 100,
            self.phase1_results['coverage_68'] * 100
        ]
        phase21_vals = [
            self.phase21_results['coverage_90'] * 100,
            self.phase21_results['coverage_68'] * 100
        ]
        targets = [90, 68]

        x = np.arange(len(metrics))
        width = 0.25

        ax.bar(x - width, phase1_vals, width, label='Phase 1', alpha=0.8, color='#E74C3C')
        ax.bar(x, phase21_vals, width, label='Phase 2.1', alpha=0.8, color='#2ECC71')
        ax.bar(x + width, targets, width, label='Target', alpha=0.5, color='#3498DB')

        ax.set_ylabel('Coverage (%)', fontweight='bold')
        ax.set_title('Calibration Coverage')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Accuracy Comparison
        ax = axes[1]
        metrics = ['MAE', 'RMSE']
        phase1_vals = [self.phase1_results['mae'], self.phase1_results['rmse']]
        phase21_vals = [self.phase21_results['mae'], self.phase21_results['rmse']]

        x = np.arange(len(metrics))
        ax.bar(x - width/2, phase1_vals, width, label='Phase 1', alpha=0.8, color='#E74C3C')
        ax.bar(x + width/2, phase21_vals, width, label='Phase 2.1', alpha=0.8, color='#2ECC71')

        ax.set_ylabel('Yards', fontweight='bold')
        ax.set_title('Prediction Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Improvement Summary
        ax = axes[2]
        improvements = self.calculate_improvements()

        improvements_data = {
            '90% Coverage\n(pp)': improvements['coverage_90_abs'],
            '68% Coverage\n(pp)': improvements['coverage_68_abs'],
            'MAE\n(yards)': improvements['mae_abs']
        }

        colors = ['green' if v > 0 else 'red' for v in improvements_data.values()]
        ax.barh(list(improvements_data.keys()), list(improvements_data.values()), color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Improvement (Positive = Better)', fontweight='bold')
        ax.set_title('Phase 2.1 Improvements')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {output_path}")

    def save_latex_table(self, output_path: Path):
        """Save LaTeX comparison table"""
        improvements = self.calculate_improvements()

        p1_cov90 = self.phase1_results['coverage_90'] * 100
        p21_cov90 = self.phase21_results['coverage_90'] * 100
        p1_cov68 = self.phase1_results['coverage_68'] * 100
        p21_cov68 = self.phase21_results['coverage_68'] * 100

        latex = r"""\begin{table}[htbp]
\centering
\caption{Phase 1 vs Phase 2.1: Calibration Comparison}
\label{tab:phase_comparison}
\begin{tabular}{lccc}
\toprule
Metric & Phase 1 & Phase 2.1 & Improvement \\
\midrule
90\% CI Coverage & """
        latex += f"{p1_cov90:.1f}\\% & {p21_cov90:.1f}\\% & {improvements['coverage_90_abs']:+.1f} pp \\\\\n"
        latex += "68\\% CI Coverage & "
        latex += f"{p1_cov68:.1f}\\% & {p21_cov68:.1f}\\% & {improvements['coverage_68_abs']:+.1f} pp \\\\\n"
        latex += "MAE (yards) & "
        latex += f"{self.phase1_results['mae']:.2f} & {self.phase21_results['mae']:.2f} & "
        latex += f"{improvements['mae_abs']:+.2f} \\\\\n"
        latex += "RMSE (yards) & "
        latex += f"{self.phase1_results['rmse']:.2f} & {self.phase21_results['rmse']:.2f} & "
        latex += f"{self.phase21_results['rmse'] - self.phase1_results['rmse']:+.2f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Phase 1: Shallow BNN (2 layers, 16 units, global noise)
\item Phase 2.1: Deep BNN (4 layers, 32-16-8 units, learned noise, skip connections)
\item Target: 90\% CI coverage $\geq$ 75\% (vs 26\% baseline)
\end{tablenotes}
\end{table}
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)

        print(f"✓ LaTeX table saved to: {output_path}")


def main():
    """Run comparison when Phase 2.1 completes"""
    print("="*80)
    print("PHASE 1 vs PHASE 2.1 COMPARISON")
    print("="*80)

    try:
        comparator = PhaseComparator()

        # Print comparison
        improvements = comparator.print_comparison()

        # Create visualization
        output_dir = Path("/Users/dro/rice/nfl-analytics/analysis/phase_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        viz_path = output_dir / "phase1_vs_phase21_comparison.png"
        comparator.create_comparison_plot(viz_path)

        # Save LaTeX table
        latex_path = Path("/Users/dro/rice/nfl-analytics/analysis/dissertation/figures/out/phase_comparison_table.tex")
        comparator.save_latex_table(latex_path)

        # Save summary JSON
        summary_path = output_dir / "comparison_summary.json"
        summary = {
            'phase1': comparator.phase1_results,
            'phase21': comparator.phase21_results,
            'improvements': improvements,
            'timestamp': datetime.now().isoformat(),
            'success_criteria_met': comparator.phase21_results['coverage_90'] * 100 >= 75
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("✓ COMPARISON COMPLETE")
        print("="*80)
        print(f"\nOutputs:")
        print(f"  • Visualization: {viz_path}")
        print(f"  • LaTeX table: {latex_path}")
        print(f"  • Summary JSON: {summary_path}")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n⏳ Phase 2.1 training not complete yet. This script will run automatically when results are available.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
