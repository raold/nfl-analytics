#!/usr/bin/env python3
"""
International Games Feature Validation Script

Purpose:
    Validate 5 causal features extracted from international NFL games (2018-2024):
    1. Jet lag direction penalty
    2. Body clock hour
    3. Early morning penalty
    4. Altitude shock
    5. Neutral field indicator

Validation Checks:
    - Data completeness (no missing values)
    - Feature range validation (theoretical bounds)
    - Statistical distribution analysis
    - Correlation with passing performance
    - Outlier detection

Usage:
    python scripts/validate_international_features.py

Output:
    - Console report with validation results
    - Text file: results/international_features_validation.txt
    - Optional plots if matplotlib available

Author: NFL Analytics Research Team
Date: November 16, 2025
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class InternationalFeatureValidator:
    """Validates causal features from international games."""

    # Theoretical bounds for each feature
    FEATURE_BOUNDS = {
        'jet_lag_penalty': (-108, 0),  # Max 9 hours × -12 per hour
        'body_clock_hour': (-7, 17),   # UTC hours relative to home timezone
        'early_morning_penalty': (-32, 0),  # Max penalty for very early games
        'altitude_shock': (-5150, 5150),  # Denver (5280 ft) is extreme
        'altitude_penalty': (-10.45, 5.0),  # Scaled altitude shock
        'total_intl_penalty': (-150, 10),  # Sum of all penalties + bonuses
    }

    # Expected correlations (sign and rough magnitude)
    EXPECTED_CORRELATIONS = {
        'total_intl_penalty': ('negative', 'weak'),  # Lower penalty → more yards
        'jet_lag_penalty': ('negative', 'weak'),
        'early_morning_penalty': ('negative', 'weak'),
        'altitude_penalty': ('negative', 'weak'),
    }

    def __init__(self, data_path: str = "data/international_games_2018_2024.csv"):
        """Initialize validator with data path."""
        self.data_path = Path(PROJECT_ROOT) / data_path
        self.df = None
        self.validation_results = {}
        self.issues = []
        self.warnings = []

    def load_data(self) -> bool:
        """Load and validate CSV file exists."""
        if not self.data_path.exists():
            self.issues.append(f"❌ Data file not found: {self.data_path}")
            return False

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} QB performances from {self.data_path}")
            return True
        except Exception as e:
            self.issues.append(f"❌ Error loading data: {e}")
            return False

    def validate_required_columns(self) -> bool:
        """Check all required columns are present."""
        required_columns = [
            # Performance metric
            'passing_yards',
            # Causal features
            'jet_lag_penalty',
            'body_clock_hour',
            'early_morning_penalty',
            'altitude_shock',
            'altitude_penalty',
            'total_intl_penalty',
            # Metadata
            'neutral_field',
            'is_international',
            'player_name',
            'season',
            'week',
        ]

        missing = [col for col in required_columns if col not in self.df.columns]

        if missing:
            self.issues.append(f"❌ Missing required columns: {missing}")
            return False

        print(f"✅ All {len(required_columns)} required columns present")
        return True

    def check_missing_values(self) -> dict[str, int]:
        """Check for missing values in critical columns."""
        feature_cols = [
            'jet_lag_penalty', 'body_clock_hour', 'early_morning_penalty',
            'altitude_shock', 'altitude_penalty', 'total_intl_penalty'
        ]

        missing_counts = {}
        has_missing = False

        for col in feature_cols:
            missing = self.df[col].isna().sum()
            missing_counts[col] = missing

            if missing > 0:
                has_missing = True
                self.issues.append(f"❌ {col}: {missing} missing values ({missing/len(self.df)*100:.1f}%)")

        if not has_missing:
            print(f"✅ No missing values in {len(feature_cols)} feature columns")

        self.validation_results['missing_values'] = missing_counts
        return missing_counts

    def validate_feature_ranges(self) -> dict[str, dict]:
        """Validate features are within theoretical bounds."""
        range_results = {}

        for feature, (min_bound, max_bound) in self.FEATURE_BOUNDS.items():
            if feature not in self.df.columns:
                continue

            values = self.df[feature].dropna()
            actual_min = values.min()
            actual_max = values.max()

            # Check if within bounds
            within_bounds = (actual_min >= min_bound) and (actual_max <= max_bound)

            # Flag outliers
            outliers_low = (values < min_bound).sum()
            outliers_high = (values > max_bound).sum()

            range_results[feature] = {
                'theoretical_min': min_bound,
                'theoretical_max': max_bound,
                'actual_min': actual_min,
                'actual_max': actual_max,
                'within_bounds': within_bounds,
                'outliers_low': outliers_low,
                'outliers_high': outliers_high,
            }

            if within_bounds:
                print(f"✅ {feature}: [{actual_min:.2f}, {actual_max:.2f}] within [{min_bound}, {max_bound}]")
            else:
                if outliers_low > 0:
                    self.issues.append(
                        f"❌ {feature}: {outliers_low} values below {min_bound} (min: {actual_min:.2f})"
                    )
                if outliers_high > 0:
                    self.issues.append(
                        f"❌ {feature}: {outliers_high} values above {max_bound} (max: {actual_max:.2f})"
                    )

        self.validation_results['ranges'] = range_results
        return range_results

    def calculate_summary_statistics(self) -> pd.DataFrame:
        """Calculate descriptive statistics for all features."""
        feature_cols = [
            'passing_yards', 'jet_lag_penalty', 'body_clock_hour',
            'early_morning_penalty', 'altitude_shock', 'altitude_penalty',
            'total_intl_penalty'
        ]

        stats = self.df[feature_cols].describe().T
        stats['missing'] = self.df[feature_cols].isna().sum()
        stats['zeros'] = (self.df[feature_cols] == 0).sum()

        self.validation_results['summary_stats'] = stats
        return stats

    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze correlations between features and passing yards."""
        feature_cols = [
            'jet_lag_penalty', 'body_clock_hour', 'early_morning_penalty',
            'altitude_shock', 'altitude_penalty', 'total_intl_penalty'
        ]

        correlations = []

        for feature in feature_cols:
            if feature not in self.df.columns:
                continue

            # Pearson correlation
            corr = self.df['passing_yards'].corr(self.df[feature])

            # Sample size (non-missing pairs)
            n = self.df[[feature, 'passing_yards']].dropna().shape[0]

            # Simple significance test (r > 0.3 is "moderate")
            strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
            direction = 'positive' if corr > 0 else 'negative'

            correlations.append({
                'feature': feature,
                'correlation': corr,
                'direction': direction,
                'strength': strength,
                'n': n,
            })

        corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)

        # Check against expected correlations
        for _, row in corr_df.iterrows():
            feature = row['feature']
            if feature in self.EXPECTED_CORRELATIONS:
                expected_dir, expected_str = self.EXPECTED_CORRELATIONS[feature]
                if row['direction'] != expected_dir:
                    self.warnings.append(
                        f"⚠️ {feature}: Expected {expected_dir} correlation, got {row['direction']}"
                    )

        self.validation_results['correlations'] = corr_df
        return corr_df

    def detect_statistical_outliers(self) -> dict[str, list]:
        """Detect outliers using IQR method."""
        feature_cols = [
            'passing_yards', 'jet_lag_penalty', 'body_clock_hour',
            'early_morning_penalty', 'altitude_shock', 'altitude_penalty',
            'total_intl_penalty'
        ]

        outliers = {}

        for col in feature_cols:
            if col not in self.df.columns:
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_indices = self.df[outlier_mask].index.tolist()

            if len(outlier_indices) > 0:
                outliers[col] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices,
                    'values': self.df.loc[outlier_indices, col].tolist(),
                    'bounds': (lower_bound, upper_bound),
                }

                # Only warn if >20% are outliers
                if len(outlier_indices) / len(self.df) > 0.2:
                    self.warnings.append(
                        f"⚠️ {col}: {len(outlier_indices)} statistical outliers "
                        f"({len(outlier_indices)/len(self.df)*100:.1f}%)"
                    )
            else:
                outliers[col] = {'count': 0}

        self.validation_results['outliers'] = outliers
        return outliers

    def validate_neutral_field(self) -> bool:
        """Validate all games are marked as neutral field."""
        if 'neutral_field' not in self.df.columns:
            self.issues.append("❌ neutral_field column missing")
            return False

        non_neutral = (self.df['neutral_field'] != 1).sum()

        if non_neutral > 0:
            self.warnings.append(
                f"⚠️ {non_neutral} games not marked as neutral field (expected all neutral)"
            )
        else:
            print(f"✅ All {len(self.df)} games correctly marked as neutral field")

        return non_neutral == 0

    def validate_international_flag(self) -> bool:
        """Validate all games are marked as international."""
        if 'is_international' not in self.df.columns:
            self.issues.append("❌ is_international column missing")
            return False

        non_intl = (self.df['is_international'] != 1).sum()

        if non_intl > 0:
            self.warnings.append(
                f"⚠️ {non_intl} games not marked as international"
            )
        else:
            print(f"✅ All {len(self.df)} games correctly marked as international")

        return non_intl == 0

    def generate_report(self, output_path: str = "results/international_features_validation.txt") -> str:
        """Generate comprehensive validation report."""
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("INTERNATIONAL GAMES FEATURE VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data file: {self.data_path}")
        report_lines.append(f"Total QB performances: {len(self.df)}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        if 'summary_stats' in self.validation_results:
            report_lines.append(str(self.validation_results['summary_stats']))
        report_lines.append("")

        # Correlations
        report_lines.append("-" * 80)
        report_lines.append("CORRELATION ANALYSIS (with passing_yards)")
        report_lines.append("-" * 80)
        if 'correlations' in self.validation_results:
            corr_df = self.validation_results['correlations']
            for _, row in corr_df.iterrows():
                report_lines.append(
                    f"  {row['feature']:25s}: {row['correlation']:7.3f} "
                    f"({row['direction']:8s}, {row['strength']:8s}, n={row['n']})"
                )
        report_lines.append("")

        # Range validation
        report_lines.append("-" * 80)
        report_lines.append("FEATURE RANGE VALIDATION")
        report_lines.append("-" * 80)
        if 'ranges' in self.validation_results:
            for feature, bounds in self.validation_results['ranges'].items():
                status = "✅ PASS" if bounds['within_bounds'] else "❌ FAIL"
                report_lines.append(
                    f"  {feature:25s} {status}\n"
                    f"    Theoretical: [{bounds['theoretical_min']:7.2f}, {bounds['theoretical_max']:7.2f}]\n"
                    f"    Actual:      [{bounds['actual_min']:7.2f}, {bounds['actual_max']:7.2f}]\n"
                    f"    Outliers:    {bounds['outliers_low']} low, {bounds['outliers_high']} high"
                )
        report_lines.append("")

        # Statistical outliers
        report_lines.append("-" * 80)
        report_lines.append("STATISTICAL OUTLIERS (IQR method)")
        report_lines.append("-" * 80)
        if 'outliers' in self.validation_results:
            for feature, outlier_info in self.validation_results['outliers'].items():
                if outlier_info['count'] > 0:
                    report_lines.append(
                        f"  {feature:25s}: {outlier_info['count']} outliers "
                        f"(bounds: [{outlier_info['bounds'][0]:.2f}, {outlier_info['bounds'][1]:.2f}])"
                    )
                else:
                    report_lines.append(f"  {feature:25s}: No outliers")
        report_lines.append("")

        # Issues and warnings
        report_lines.append("-" * 80)
        report_lines.append("VALIDATION RESULTS")
        report_lines.append("-" * 80)

        if not self.issues:
            report_lines.append("✅ NO CRITICAL ISSUES FOUND")
        else:
            report_lines.append(f"❌ {len(self.issues)} CRITICAL ISSUES:")
            for issue in self.issues:
                report_lines.append(f"  {issue}")
        report_lines.append("")

        if not self.warnings:
            report_lines.append("✅ NO WARNINGS")
        else:
            report_lines.append(f"⚠️  {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                report_lines.append(f"  {warning}")
        report_lines.append("")

        # Overall status
        report_lines.append("=" * 80)
        if not self.issues:
            report_lines.append("OVERALL STATUS: ✅ VALIDATION PASSED")
        else:
            report_lines.append("OVERALL STATUS: ❌ VALIDATION FAILED")
        report_lines.append("=" * 80)

        # Save to file
        output_path = Path(PROJECT_ROOT) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_text = "\n".join(report_lines)
        output_path.write_text(report_text)

        print(f"\n✅ Validation report saved to: {output_path}")

        return report_text

    def run_full_validation(self) -> bool:
        """Run complete validation pipeline."""
        print("\n" + "=" * 80)
        print("INTERNATIONAL GAMES FEATURE VALIDATION")
        print("=" * 80 + "\n")

        # Step 1: Load data
        print("[1/8] Loading data...")
        if not self.load_data():
            return False
        print()

        # Step 2: Check required columns
        print("[2/8] Validating required columns...")
        if not self.validate_required_columns():
            return False
        print()

        # Step 3: Check missing values
        print("[3/8] Checking for missing values...")
        self.check_missing_values()
        print()

        # Step 4: Validate feature ranges
        print("[4/8] Validating feature ranges...")
        self.validate_feature_ranges()
        print()

        # Step 5: Calculate summary statistics
        print("[5/8] Calculating summary statistics...")
        self.calculate_summary_statistics()
        print()

        # Step 6: Analyze correlations
        print("[6/8] Analyzing correlations with passing yards...")
        self.analyze_correlations()
        print()

        # Step 7: Detect outliers
        print("[7/8] Detecting statistical outliers...")
        self.detect_statistical_outliers()
        print()

        # Step 8: Validate flags
        print("[8/8] Validating neutral_field and is_international flags...")
        self.validate_neutral_field()
        self.validate_international_flag()
        print()

        # Generate report
        print("=" * 80)
        self.generate_report()

        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total QB performances: {len(self.df)}")
        print(f"Critical issues: {len(self.issues)}")
        print(f"Warnings: {len(self.warnings)}")

        if not self.issues:
            print("\n✅ ALL VALIDATION CHECKS PASSED")
            return True
        else:
            print("\n❌ VALIDATION FAILED - See issues above")
            return False


def main():
    """Main execution function."""
    validator = InternationalFeatureValidator()

    success = validator.run_full_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
