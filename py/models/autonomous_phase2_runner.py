"""
Autonomous Phase 2 Runner
Monitors Phase 2.1, then automatically runs Phase 2.2, compares, and updates results

Author: Richard Oldham
Date: October 2024
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import time
import json
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousPhase2Runner:
    """Autonomously runs Phases 2.1 and 2.2, compares results"""

    def __init__(self):
        self.base_dir = Path("/Users/dro/rice/nfl-analytics")
        self.models_dir = self.base_dir / "models" / "bayesian"
        self.results_dir = self.base_dir / "experiments" / "calibration"

    def wait_for_phase21_completion(self, timeout_minutes=90):
        """
        Wait for Phase 2.1 to complete by checking for result files

        Returns:
            bool: True if completed successfully, False if timeout/error
        """
        logger.info("Waiting for Phase 2.1 (Simpler BNN) to complete...")
        logger.info(f"Timeout: {timeout_minutes} minutes")

        start_time = time.time()
        check_interval = 60  # Check every minute

        model_file = self.models_dir / "bnn_simpler_v2.pkl"
        results_file = self.results_dir / "simpler_bnn_v2_results.json"

        while True:
            elapsed = (time.time() - start_time) / 60

            if elapsed > timeout_minutes:
                logger.error(f"Timeout after {timeout_minutes} minutes")
                return False

            # Check if files exist
            if model_file.exists() and results_file.exists():
                logger.info("✓ Phase 2.1 complete!")
                logger.info(f"  Model: {model_file}")
                logger.info(f"  Results: {results_file}")

                # Load and display results
                with open(results_file) as f:
                    results = json.load(f)

                logger.info(f"\nPhase 2.1 Results:")
                logger.info(f"  90% Coverage: {results['coverage_90']:.1f}% (target: 90%)")
                logger.info(f"  MAE: {results['mae']:.1f} yards")

                return True

            # Check progress
            logger.info(f"  [{elapsed:.0f}min / {timeout_minutes}min] Waiting...")
            time.sleep(check_interval)

    def run_phase22(self):
        """Run Phase 2.2 (Mixture-of-Experts BNN)"""
        logger.info("\n" + "="*80)
        logger.info("Starting Phase 2.2: Mixture-of-Experts BNN")
        logger.info("="*80)

        script_path = self.base_dir / "py" / "models" / "bnn_mixture_experts_v2.py"

        logger.info(f"Running: {script_path}")
        result = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(self.base_dir)
        )

        if result.returncode == 0:
            logger.info("✓ Phase 2.2 complete!")
            return True
        else:
            logger.error("✗ Phase 2.2 failed!")
            logger.error(result.stderr[-1000:])  # Last 1000 chars
            return False

    def compare_results(self):
        """Compare Phase 2.1 and 2.2 results"""
        logger.info("\n" + "="*80)
        logger.info("COMPARING PHASE 2 RESULTS")
        logger.info("="*80)

        # Load results
        results_21 = json.load(open(self.results_dir / "simpler_bnn_v2_results.json"))
        results_22_file = self.results_dir / "mixture_experts_v2_results.json"

        if not results_22_file.exists():
            logger.warning("Phase 2.2 results not found, skipping comparison")
            return

        results_22 = json.load(open(results_22_file))

        # Compare
        print("\n" + "="*80)
        print("PHASE 2 COMPARISON: Simpler BNN vs Mixture-of-Experts")
        print("="*80)
        print(f"{'Metric':<20} {'Phase 1':<12} {'Phase 2.1':<12} {'Phase 2.2':<12} {'Best'}")
        print("-"*80)

        phase1_coverage = 26.0
        phase1_mae = 18.7

        metrics = [
            ("90% Coverage (%)", phase1_coverage, results_21['coverage_90'], results_22['coverage_90'], 90.0),
            ("MAE (yards)", phase1_mae, results_21['mae'], results_22['mae'], None)
        ]

        for name, p1, p21, p22, target in metrics:
            if target:  # Higher is better
                best = "2.1" if abs(p21 - target) < abs(p22 - target) else "2.2"
            else:  # Lower is better
                best = "2.1" if p21 < p22 else "2.2"

            print(f"{name:<20} {p1:<12.1f} {p21:<12.1f} {p22:<12.1f} {best}")

        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        if results_21['coverage_90'] >= 75:
            print("✓ Phase 2.1 (Simpler BNN) achieved target calibration (≥75%)")
            print(f"  Coverage: {results_21['coverage_90']:.1f}%")
            print("\nRECOMMENDATION: Use Phase 2.1 model (simpler, faster, sufficient)")

        elif results_22['coverage_90'] >= 75:
            print("✓ Phase 2.2 (Mixture-of-Experts) achieved target calibration")
            print(f"  Coverage: {results_22['coverage_90']:.1f}%")
            print("\nRECOMMENDATION: Use Phase 2.2 model (more complex but better calibrated)")

        else:
            best_cov = max(results_21['coverage_90'], results_22['coverage_90'])
            print(f"⚠️  Neither model achieved 75% target (best: {best_cov:.1f}%)")
            print("\nRECOMMENDATION: Proceed with hybrid calibration (Phase 2.3)")

        # Save comparison
        comparison = {
            'phase_1': {'coverage_90': phase1_coverage, 'mae': phase1_mae},
            'phase_2.1': results_21,
            'phase_2.2': results_22,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        comparison_file = self.results_dir / "phase2_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"\n✓ Comparison saved to: {comparison_file}")

    def run(self):
        """Main autonomous execution"""
        logger.info("="*80)
        logger.info("AUTONOMOUS PHASE 2 EXECUTION")
        logger.info("="*80)
        logger.info("\nThis script will:")
        logger.info("  1. Wait for Phase 2.1 to complete")
        logger.info("  2. Run Phase 2.2 (Mixture-of-Experts)")
        logger.info("  3. Compare results and make recommendation")
        logger.info("")

        # Step 1: Wait for Phase 2.1
        if not self.wait_for_phase21_completion(timeout_minutes=90):
            logger.error("Phase 2.1 did not complete in time")
            return False

        # Step 2: Run Phase 2.2
        if not self.run_phase22():
            logger.warning("Phase 2.2 failed, will compare Phase 2.1 only")

        # Step 3: Compare
        self.compare_results()

        logger.info("\n" + "="*80)
        logger.info("AUTONOMOUS PHASE 2 EXECUTION COMPLETE")
        logger.info("="*80)

        return True


if __name__ == "__main__":
    runner = AutonomousPhase2Runner()
    success = runner.run()
    sys.exit(0 if success else 1)
