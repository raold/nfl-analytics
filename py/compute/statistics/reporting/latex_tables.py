#!/usr/bin/env python3
"""
LaTeX Table Generator for Statistical Results.

Creates publication-ready LaTeX tables for statistical analyses,
with proper formatting for academic journals and reports.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TableStyle(Enum):
    """Predefined table styles for different contexts."""

    ACADEMIC = "academic"
    CLINICAL = "clinical"
    FINANCIAL = "financial"
    SIMPLE = "simple"


@dataclass
class ColumnFormat:
    """Formatting specification for table columns."""

    name: str
    latex_format: str  # e.g., "c", "l", "r", "p{3cm}"
    number_format: str | None = None  # e.g., ".3f", ".2%"
    bold_header: bool = True
    italic_significant: bool = False  # Italicize significant p-values


@dataclass
class StatisticalTable:
    """Configuration for statistical result tables."""

    data: pd.DataFrame
    caption: str
    label: str = ""
    position: str = "H"
    size: str = "normalsize"  # tiny, scriptsize, footnotesize, small, normalsize, large, Large
    style: TableStyle = TableStyle.ACADEMIC
    column_formats: list[ColumnFormat] | None = None
    significance_threshold: float = 0.05
    show_effect_sizes: bool = True
    show_confidence_intervals: bool = True
    notes: list[str] | None = None


class LaTeXTableGenerator:
    """
    Professional LaTeX table generator for statistical results.

    Creates publication-ready tables with proper formatting,
    significance indicators, and academic styling.
    """

    def __init__(self):
        """Initialize LaTeX table generator."""
        self.packages = [
            "booktabs",
            "array",
            "multirow",
            "longtable",
            "threeparttable",
            "dcolumn",
            "siunitx",
        ]

    def create_results_table(
        self, results: dict[str, dict[str, Any]], title: str = "Statistical Test Results"
    ) -> str:
        """
        Create a comprehensive statistical results table.

        Args:
            results: Dictionary of test results
            title: Table title/caption

        Returns:
            LaTeX table string
        """
        # Convert results to DataFrame
        rows = []
        for test_name, test_results in results.items():
            row = {
                "Test": self._format_test_name(test_name),
                "Statistic": test_results.get("statistic", np.nan),
                "P-value": test_results.get("p_value", np.nan),
                "Effect Size": test_results.get("effect_size", np.nan),
                "95% CI": self._format_ci(test_results.get("confidence_interval")),
                "Interpretation": self._get_significance_symbol(test_results.get("p_value", 1)),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Create table specification
        table_spec = StatisticalTable(
            data=df, caption=title, label="tab:statistical-results", style=TableStyle.ACADEMIC
        )

        return self.generate_table(table_spec)

    def create_descriptive_table(
        self,
        data: pd.DataFrame,
        group_var: str | None = None,
        title: str = "Descriptive Statistics",
    ) -> str:
        """
        Create descriptive statistics table.

        Args:
            data: Dataset for analysis
            group_var: Grouping variable (optional)
            title: Table title

        Returns:
            LaTeX table string
        """
        if group_var and group_var in data.columns:
            # Grouped descriptive statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != group_var]

            grouped_stats = []
            for group in data[group_var].unique():
                group_data = data[data[group_var] == group]
                for col in numeric_cols:
                    if col in group_data.columns:
                        stats_row = {
                            "Variable": col,
                            "Group": group,
                            "N": len(group_data[col].dropna()),
                            "Mean": group_data[col].mean(),
                            "SD": group_data[col].std(),
                            "Min": group_data[col].min(),
                            "Max": group_data[col].max(),
                        }
                        grouped_stats.append(stats_row)

            df = pd.DataFrame(grouped_stats)
        else:
            # Overall descriptive statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            desc_stats = []
            for col in numeric_cols:
                stats_row = {
                    "Variable": col,
                    "N": len(data[col].dropna()),
                    "Mean": data[col].mean(),
                    "SD": data[col].std(),
                    "Min": data[col].min(),
                    "Median": data[col].median(),
                    "Max": data[col].max(),
                }
                desc_stats.append(stats_row)

            df = pd.DataFrame(desc_stats)

        table_spec = StatisticalTable(
            data=df, caption=title, label="tab:descriptive-stats", style=TableStyle.ACADEMIC
        )

        return self.generate_table(table_spec)

    def create_correlation_table(
        self,
        correlation_matrix: pd.DataFrame,
        p_values: pd.DataFrame | None = None,
        title: str = "Correlation Matrix",
    ) -> str:
        """
        Create correlation matrix table with significance indicators.

        Args:
            correlation_matrix: Correlation coefficient matrix
            p_values: P-value matrix (optional)
            title: Table title

        Returns:
            LaTeX table string
        """
        # Format correlation values
        formatted_corr = correlation_matrix.copy()

        if p_values is not None:
            # Add significance stars
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    p_val = p_values.iloc[i, j]

                    if pd.isna(corr_val) or pd.isna(p_val):
                        continue

                    # Format with significance stars
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                    else:
                        stars = ""

                    formatted_corr.iloc[i, j] = f"{corr_val:.3f}{stars}"

        table_spec = StatisticalTable(
            data=formatted_corr,
            caption=title + r" (* p < 0.05, ** p < 0.01, *** p < 0.001)",
            label="tab:correlation-matrix",
            style=TableStyle.ACADEMIC,
        )

        return self.generate_table(table_spec)

    def create_regression_table(
        self, models: dict[str, dict[str, Any]], title: str = "Regression Results"
    ) -> str:
        """
        Create regression results table.

        Args:
            models: Dictionary of regression model results
            title: Table title

        Returns:
            LaTeX table string
        """
        # Extract coefficients, standard errors, and p-values
        all_variables = set()
        for model_results in models.values():
            if "coefficients" in model_results:
                all_variables.update(model_results["coefficients"].keys())

        rows = []
        for var in sorted(all_variables):
            row = {"Variable": var}

            for model_name, model_results in models.items():
                coef = model_results.get("coefficients", {}).get(var)
                se = model_results.get("standard_errors", {}).get(var)
                p_val = model_results.get("p_values", {}).get(var)

                if coef is not None and se is not None:
                    # Format coefficient with significance
                    sig_symbol = self._get_significance_symbol(p_val) if p_val else ""
                    coef_str = f"{coef:.3f}{sig_symbol}"
                    se_str = f"({se:.3f})"

                    row[f"{model_name}_coef"] = coef_str
                    row[f"{model_name}_se"] = se_str
                else:
                    row[f"{model_name}_coef"] = "—"
                    row[f"{model_name}_se"] = ""

            rows.append(row)

        # Add model statistics
        stats_rows = []
        for stat_name in ["R²", "Adjusted R²", "N", "F-statistic"]:
            row = {"Variable": stat_name}
            for model_name, model_results in models.items():
                stat_value = model_results.get("model_stats", {}).get(
                    stat_name.lower().replace("²", "_squared")
                )
                if stat_value is not None:
                    if "r_squared" in stat_name.lower():
                        row[f"{model_name}_coef"] = f"{stat_value:.3f}"
                    else:
                        row[f"{model_name}_coef"] = f"{stat_value}"
                    row[f"{model_name}_se"] = ""
                else:
                    row[f"{model_name}_coef"] = "—"
                    row[f"{model_name}_se"] = ""
            stats_rows.append(row)

        df = pd.DataFrame(rows + stats_rows)

        table_spec = StatisticalTable(
            data=df,
            caption=title + r" (* p < 0.05, ** p < 0.01, *** p < 0.001)",
            label="tab:regression-results",
            style=TableStyle.ACADEMIC,
        )

        return self.generate_table(table_spec)

    def generate_table(self, table_spec: StatisticalTable) -> str:
        """
        Generate LaTeX table from specification.

        Args:
            table_spec: Table specification

        Returns:
            Complete LaTeX table string
        """
        df = table_spec.data

        # Determine column alignment
        if table_spec.column_formats:
            col_spec = "".join([fmt.latex_format for fmt in table_spec.column_formats])
        else:
            col_spec = self._auto_detect_column_format(df)

        # Start building LaTeX
        latex_lines = []

        # Table environment
        if table_spec.style == TableStyle.ACADEMIC:
            latex_lines.extend(
                [
                    "\\begin{table}[" + table_spec.position + "]",
                    f"\\{table_spec.size}",
                    "\\centering",
                ]
            )

        # Caption and label
        if table_spec.caption:
            latex_lines.append(f"\\caption{{{table_spec.caption}}}")
        if table_spec.label:
            latex_lines.append(f"\\label{{{table_spec.label}}}")

        # Begin tabular
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        if table_spec.style == TableStyle.ACADEMIC:
            latex_lines.append("\\toprule")

        # Header row
        header_row = self._format_header_row(df.columns, table_spec)
        latex_lines.append(header_row + " \\\\")

        if table_spec.style == TableStyle.ACADEMIC:
            latex_lines.append("\\midrule")

        # Data rows
        for idx, row in df.iterrows():
            data_row = self._format_data_row(row, table_spec)
            latex_lines.append(data_row + " \\\\")

        # Footer
        if table_spec.style == TableStyle.ACADEMIC:
            latex_lines.append("\\bottomrule")

        latex_lines.append("\\end{tabular}")

        # Notes
        if table_spec.notes:
            latex_lines.append("\\begin{tablenotes}")
            latex_lines.append("\\footnotesize")
            for note in table_spec.notes:
                latex_lines.append(f"\\item {note}")
            latex_lines.append("\\end{tablenotes}")

        if table_spec.style == TableStyle.ACADEMIC:
            latex_lines.append("\\end{table}")

        return "\n".join(latex_lines)

    def _auto_detect_column_format(self, df: pd.DataFrame) -> str:
        """Auto-detect appropriate column alignment."""
        col_spec = ""

        for col in df.columns:
            if col == df.columns[0]:  # First column (usually text)
                col_spec += "l"
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_spec += "r"  # Right-align numbers
            else:
                col_spec += "c"  # Center-align text

        return col_spec

    def _format_header_row(self, columns: pd.Index, table_spec: StatisticalTable) -> str:
        """Format header row with proper LaTeX escaping."""
        headers = []

        for col in columns:
            header = str(col).replace("_", "\\_").replace("%", "\\%")
            if table_spec.style == TableStyle.ACADEMIC:
                header = f"\\textbf{{{header}}}"
            headers.append(header)

        return " & ".join(headers)

    def _format_data_row(self, row: pd.Series, table_spec: StatisticalTable) -> str:
        """Format data row with appropriate number formatting."""
        formatted_cells = []

        for col_name, value in row.items():
            if pd.isna(value):
                formatted_cells.append("—")
            elif isinstance(value, (int, float)):
                # Format numbers appropriately
                if "p-value" in col_name.lower() or "p_value" in col_name.lower():
                    formatted_cells.append(format_p_value(value))
                elif abs(value) < 0.001 and value != 0:
                    formatted_cells.append(f"{value:.2e}")
                elif abs(value) < 1:
                    formatted_cells.append(f"{value:.3f}")
                else:
                    formatted_cells.append(f"{value:.2f}")
            else:
                # String values - escape LaTeX special characters
                escaped = str(value).replace("_", "\\_").replace("%", "\\%")
                escaped = escaped.replace("&", "\\&").replace("#", "\\#")
                formatted_cells.append(escaped)

        return " & ".join(formatted_cells)

    def _format_test_name(self, test_name: str) -> str:
        """Format test names for display."""
        # Convert underscores to spaces and title case
        formatted = test_name.replace("_", " ").title()

        # Special cases
        replacements = {
            "T Test": "t-test",
            "Chi Square": "χ² test",
            "Mann Whitney": "Mann-Whitney U",
            "Wilcoxon": "Wilcoxon signed-rank",
        }

        for old, new in replacements.items():
            if old in formatted:
                formatted = formatted.replace(old, new)

        return formatted

    def _format_ci(self, ci: tuple[float, float] | None) -> str:
        """Format confidence interval."""
        if ci is None or len(ci) != 2:
            return "—"

        lower, upper = ci
        return f"({lower:.3f}, {upper:.3f})"

    def _get_significance_symbol(self, p_value: float | None) -> str:
        """Get significance symbol based on p-value."""
        if p_value is None or pd.isna(p_value):
            return ""

        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""

    def get_required_packages(self) -> list[str]:
        """Get list of required LaTeX packages."""
        return self.packages.copy()

    def create_package_imports(self) -> str:
        """Create LaTeX package import statements."""
        imports = []
        for package in self.packages:
            imports.append(f"\\usepackage{{{package}}}")

        return "\n".join(imports)


def format_p_value(p_value: float, threshold: float = 0.001) -> str:
    """
    Format p-value for publication.

    Args:
        p_value: P-value to format
        threshold: Threshold for "< 0.001" format

    Returns:
        Formatted p-value string
    """
    if pd.isna(p_value):
        return "—"

    if p_value < threshold:
        return f"< {threshold:.3f}"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.2f}"


def format_confidence_interval(ci: tuple[float, float], level: float = 0.95) -> str:
    """
    Format confidence interval for publication.

    Args:
        ci: Confidence interval tuple (lower, upper)
        level: Confidence level

    Returns:
        Formatted confidence interval string
    """
    if ci is None or len(ci) != 2:
        return "—"

    lower, upper = ci
    percentage = int(level * 100)
    return f"{percentage}\\% CI: ({lower:.3f}, {upper:.3f})"


def test():
    """Test LaTeX table generator."""
    print("=== LaTeX Table Generator Demo ===\n")

    generator = LaTeXTableGenerator()

    # Test 1: Statistical results table
    mock_results = {
        "t_test": {
            "statistic": 2.45,
            "p_value": 0.023,
            "effect_size": 0.65,
            "confidence_interval": (0.1, 1.2),
        },
        "mann_whitney": {
            "statistic": 1250,
            "p_value": 0.031,
            "effect_size": 0.58,
            "confidence_interval": (0.05, 1.1),
        },
        "chi_square": {
            "statistic": 8.34,
            "p_value": 0.004,
            "effect_size": 0.42,
            "confidence_interval": None,
        },
    }

    results_table = generator.create_results_table(
        mock_results, "Statistical Test Results for NFL Betting Strategies"
    )
    print("Statistical Results Table:")
    print(results_table[:500] + "...\n")

    # Test 2: Descriptive statistics
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "group": ["control"] * 50 + ["treatment"] * 50,
            "profit_rate": np.concatenate(
                [np.random.normal(0.02, 0.15, 50), np.random.normal(0.05, 0.18, 50)]
            ),
            "num_bets": np.random.poisson(20, 100),
            "max_drawdown": np.random.uniform(0.05, 0.30, 100),
        }
    )

    desc_table = generator.create_descriptive_table(
        sample_data, group_var="group", title="Descriptive Statistics by Treatment Group"
    )
    print("Descriptive Statistics Table:")
    print(desc_table[:500] + "...\n")

    # Test 3: Correlation matrix
    corr_data = sample_data[["profit_rate", "num_bets", "max_drawdown"]].corr()

    # Mock p-values
    p_values = pd.DataFrame(
        [[np.nan, 0.023, 0.001], [0.023, np.nan, 0.156], [0.001, 0.156, np.nan]],
        index=corr_data.index,
        columns=corr_data.columns,
    )

    corr_table = generator.create_correlation_table(
        corr_data, p_values, "Correlation Matrix of Performance Metrics"
    )
    print("Correlation Table:")
    print(corr_table[:500] + "...\n")

    # Test package requirements
    print("Required LaTeX packages:")
    for package in generator.get_required_packages():
        print(f"  \\usepackage{{{package}}}")


if __name__ == "__main__":
    test()
