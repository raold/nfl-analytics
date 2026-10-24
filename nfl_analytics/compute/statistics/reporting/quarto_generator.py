#!/usr/bin/env python3
"""
Automated Quarto Report Generator for NFL Analytics.

Creates publication-ready statistical reports with LaTeX integration,
supporting R and Python code execution for comprehensive analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


@dataclass
class QuartoSection:
    """Individual section of a Quarto document."""

    title: str
    content: str
    code_blocks: list[dict[str, str]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class QuartoDocument:
    """Complete Quarto document specification."""

    title: str
    author: str
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    format: str = "pdf"  # pdf, html, docx
    sections: list[QuartoSection] = field(default_factory=list)
    bibliography: str | None = None
    template: str = "academic"
    metadata: dict[str, Any] = field(default_factory=dict)


class QuartoGenerator:
    """
    Automated Quarto document generator for statistical reports.

    Creates publication-ready documents with integrated R/Python code,
    LaTeX tables, and academic formatting.
    """

    def __init__(self, output_dir: str = "reports", templates_dir: str | None = None):
        """
        Initialize Quarto generator.

        Args:
            output_dir: Directory for generated reports
            templates_dir: Directory containing Quarto templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup templates
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            self.templates_dir = Path(__file__).parent / "templates"
            self.templates_dir.mkdir(exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)), trim_blocks=True, lstrip_blocks=True
        )

        # Initialize default templates if they don't exist
        self._create_default_templates()

    def _create_default_templates(self):
        """Create default Quarto templates."""
        # Academic template
        academic_template = """---
title: "{{ title }}"
author: "{{ author }}"
date: "{{ date }}"
format:
  {{ format }}:
    documentclass: article
    geometry: margin=1in
    fontsize: 11pt
    colorlinks: true
    toc: true
    number-sections: true
    cite-method: natbib
{% if bibliography %}
bibliography: {{ bibliography }}
{% endif %}
{% if abstract %}
abstract: |
  {{ abstract }}
{% endif %}
{% if keywords %}
keywords: {{ keywords | join(", ") }}
{% endif %}
execute:
  echo: false
  warning: false
  message: false
---

{% for section in sections %}
# {{ section.title }}

{{ section.content }}

{% for code_block in section.code_blocks %}
```{{ "{" }}{{ code_block.language }}{{ "}" }}
{% if code_block.label %}
#| label: {{ code_block.label }}
{% endif %}
{% if code_block.caption %}
#| fig-cap: "{{ code_block.caption }}"
{% endif %}
{% if code_block.echo is defined %}
#| echo: {{ code_block.echo | lower }}
{% endif %}

{{ code_block.code }}
```

{% endfor %}

{% endfor %}

{% if references %}
# References

::: {#refs}
:::
{% endif %}
"""

        # Statistical analysis template
        stats_template = """---
title: "{{ title }}"
subtitle: "Statistical Analysis Report"
author: "{{ author }}"
date: "{{ date }}"
format:
  pdf:
    documentclass: article
    geometry: margin=1in
    fontsize: 11pt
    colorlinks: true
    toc: true
    number-sections: true
    keep-tex: true
    include-in-header:
      - text: |
          \\usepackage{booktabs}
          \\usepackage{longtable}
          \\usepackage{array}
          \\usepackage{multirow}
          \\usepackage{wrapfig}
          \\usepackage{float}
          \\usepackage{colortbl}
          \\usepackage{pdflscape}
          \\usepackage{tabu}
          \\usepackage{threeparttable}
          \\usepackage{threeparttablex}
          \\usepackage[normalem]{ulem}
          \\usepackage{makecell}
          \\usepackage{xcolor}
execute:
  echo: false
  warning: false
  message: false
  cache: true
---

```{{python}}
#| include: false
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

```{{r}}
#| include: false
library(tidyverse)
library(knitr)
library(kableExtra)
library(ggplot2)
library(broom)

# Set knitr options
opts_chunk$set(
  echo = FALSE,
  warning = FALSE,
  message = FALSE,
  fig.width = 8,
  fig.height = 6,
  dpi = 300
)
```

{% for section in sections %}
# {{ section.title }}

{{ section.content }}

{% for code_block in section.code_blocks %}
```{{ "{" }}{{ code_block.language }}{{ "}" }}
{% if code_block.label %}
#| label: {{ code_block.label }}
{% endif %}
{% if code_block.caption %}
#| fig-cap: "{{ code_block.caption }}"
{% endif %}
{% if code_block.output %}
#| output: {{ code_block.output }}
{% endif %}

{{ code_block.code }}
```

{% endfor %}

{% endfor %}
"""

        # Save templates
        with open(self.templates_dir / "academic.qmd", "w") as f:
            f.write(academic_template)

        with open(self.templates_dir / "statistical_analysis.qmd", "w") as f:
            f.write(stats_template)

    def create_statistical_report(
        self, title: str, author: str, results: dict[str, Any], data: pd.DataFrame | None = None
    ) -> QuartoDocument:
        """
        Create a comprehensive statistical analysis report.

        Args:
            title: Report title
            author: Report author
            results: Statistical results dictionary
            data: Optional dataset for analysis

        Returns:
            QuartoDocument ready for rendering
        """
        # Create abstract
        abstract = self._generate_abstract(results)

        # Create document
        doc = QuartoDocument(
            title=title,
            author=author,
            abstract=abstract,
            keywords=["statistics", "nfl", "sports betting", "analytics"],
            format="pdf",
            template="statistical_analysis",
        )

        # Add sections
        doc.sections.extend(
            [
                self._create_methods_section(results),
                self._create_results_section(results, data),
                self._create_discussion_section(results),
            ]
        )

        return doc

    def create_ab_test_report(
        self, test_results: dict[str, Any], test_config: dict[str, Any]
    ) -> QuartoDocument:
        """Create A/B test analysis report."""
        title = f"A/B Test Analysis: {test_config.get('name', 'Unnamed Test')}"
        author = "NFL Analytics Research Team"

        doc = QuartoDocument(title=title, author=author, template="statistical_analysis")

        # Executive Summary
        summary_section = QuartoSection(
            title="Executive Summary",
            content=self._generate_ab_test_summary(test_results, test_config),
        )

        # Methodology
        methods_section = QuartoSection(
            title="Methodology", content=self._generate_ab_test_methods(test_config)
        )

        # Results
        results_section = QuartoSection(
            title="Results", content=self._generate_ab_test_results(test_results)
        )

        # Add statistical analysis code
        results_section.code_blocks.append(
            {
                "language": "python",
                "label": "statistical-analysis",
                "code": self._generate_ab_test_analysis_code(test_results),
            }
        )

        # Conclusions
        conclusions_section = QuartoSection(
            title="Conclusions and Recommendations",
            content=self._generate_ab_test_conclusions(test_results, test_config),
        )

        doc.sections = [summary_section, methods_section, results_section, conclusions_section]
        return doc

    def create_power_analysis_report(
        self, power_results: dict[str, Any], design_params: dict[str, Any]
    ) -> QuartoDocument:
        """Create power analysis report."""
        title = "Statistical Power Analysis"
        author = "NFL Analytics Research Team"

        doc = QuartoDocument(title=title, author=author, template="statistical_analysis")

        # Sample Size Justification
        sample_size_section = QuartoSection(
            title="Sample Size Justification",
            content=self._generate_power_analysis_content(power_results, design_params),
        )

        # Add power curve visualization
        sample_size_section.code_blocks.append(
            {
                "language": "python",
                "label": "power-curve",
                "caption": "Statistical Power vs Effect Size",
                "code": self._generate_power_curve_code(power_results),
            }
        )

        # Add sample size table
        sample_size_section.code_blocks.append(
            {
                "language": "r",
                "label": "sample-size-table",
                "caption": "Required Sample Sizes for Different Effect Sizes",
                "code": self._generate_sample_size_table_code(power_results),
            }
        )

        doc.sections = [sample_size_section]
        return doc

    def _generate_abstract(self, results: dict[str, Any]) -> str:
        """Generate abstract from statistical results."""
        # Extract key findings
        n_tests = len(results.get("test_results", {}))
        significant_tests = sum(
            1 for test in results.get("test_results", {}).values() if test.get("p_value", 1) < 0.05
        )

        abstract = f"""
        This report presents a comprehensive statistical analysis of NFL betting strategies
        and performance metrics. We conducted {n_tests} statistical tests to evaluate
        treatment effects and model performance. Of these, {significant_tests} tests
        showed statistically significant results at α = 0.05. All analyses employed
        appropriate multiple comparison corrections and effect size calculations.
        The findings provide evidence-based recommendations for optimizing betting
        strategies and model selection in sports analytics applications.
        """

        return abstract.strip()

    def _create_methods_section(self, results: dict[str, Any]) -> QuartoSection:
        """Create methods section."""
        content = """
        ## Statistical Methods

        ### Hypothesis Testing
        We employed modern statistical methods appropriate for sports betting analytics:

        - **Permutation tests** for distribution-free hypothesis testing
        - **Bootstrap methods** for confidence interval estimation
        - **Bayesian analysis** where appropriate for decision-making contexts

        ### Multiple Comparison Correction
        Given the multiple testing nature of our analysis, we applied appropriate
        corrections:

        - **Benjamini-Hochberg FDR** control for exploratory analyses
        - **Bonferroni correction** for confirmatory testing where Type I error
          control is critical

        ### Effect Size Measures
        All significant results include appropriate effect size measures:

        - **Cohen's d** for standardized mean differences
        - **Cliff's delta** for non-parametric effect sizes
        - **Odds ratios** for binary outcomes

        ### Software
        Analyses were conducted using Python (scipy, numpy, pandas) and R
        (tidyverse, broom) with reproducible code provided in appendices.
        """

        # Add methods code block
        methods_code = """
        # Statistical testing framework
        from py.compute.statistics import (
            PermutationTest, BootstrapTest, EffectSizeCalculator,
            MultipleComparisonCorrection
        )

        # Initialize testing framework
        perm_test = PermutationTest(n_permutations=10000)
        boot_test = BootstrapTest(n_bootstrap=10000)
        effect_calc = EffectSizeCalculator()
        mc_corrector = MultipleComparisonCorrection(alpha=0.05)
        """

        section = QuartoSection(
            title="Methods",
            content=content.strip(),
            code_blocks=[
                {
                    "language": "python",
                    "label": "methods-setup",
                    "echo": True,
                    "code": methods_code.strip(),
                }
            ],
        )

        return section

    def _create_results_section(
        self, results: dict[str, Any], data: pd.DataFrame | None
    ) -> QuartoSection:
        """Create results section with tables and figures."""
        content = """
        ## Statistical Results

        ### Summary Statistics
        Table 1 presents descriptive statistics for all outcome measures.

        ### Hypothesis Testing Results
        Table 2 shows the results of all statistical tests performed, including
        p-values, effect sizes, and confidence intervals.

        ### Effect Size Analysis
        Figure 1 displays effect sizes and their confidence intervals for all
        significant comparisons.
        """

        section = QuartoSection(title="Results", content=content.strip())

        # Add summary statistics table
        if data is not None:
            section.code_blocks.append(
                {
                    "language": "python",
                    "label": "summary-stats",
                    "caption": "Descriptive Statistics",
                    "code": self._generate_summary_stats_code(data),
                }
            )

        # Add results table
        section.code_blocks.append(
            {
                "language": "r",
                "label": "results-table",
                "caption": "Statistical Test Results",
                "code": self._generate_results_table_code(results),
            }
        )

        # Add effect size plot
        section.code_blocks.append(
            {
                "language": "python",
                "label": "effect-size-plot",
                "caption": "Effect Sizes with Confidence Intervals",
                "code": self._generate_effect_size_plot_code(results),
            }
        )

        return section

    def _create_discussion_section(self, results: dict[str, Any]) -> QuartoSection:
        """Create discussion section."""
        content = """
        ## Discussion

        ### Interpretation of Results
        The statistical analysis reveals several key findings:

        1. **Primary outcomes** show statistically significant improvements
           with moderate to large effect sizes
        2. **Multiple comparison corrections** confirm robustness of findings
        3. **Confidence intervals** provide practical significance bounds

        ### Limitations
        - Sample size considerations for smaller effect detection
        - Assumption validation for parametric tests
        - Generalizability to different betting contexts

        ### Recommendations
        Based on the statistical evidence, we recommend:

        1. Implementation of validated strategies with significant effects
        2. Continued monitoring with pre-specified stopping rules
        3. Replication studies to confirm findings
        """

        return QuartoSection(title="Discussion", content=content.strip())

    def _generate_ab_test_summary(
        self, test_results: dict[str, Any], test_config: dict[str, Any]
    ) -> str:
        """Generate A/B test executive summary."""
        winner = test_results.get("winner", "No clear winner")
        confidence = test_results.get("confidence", 0)
        p_value = test_results.get("p_value", 1)

        summary = f"""
        **Test Name:** {test_config.get('name', 'Unnamed Test')}

        **Primary Finding:** {winner} with {confidence:.1%} confidence

        **Statistical Significance:** p = {p_value:.4f}

        **Recommendation:** {test_results.get('recommendation', 'Collect more data')}

        This A/B test examined the effectiveness of different betting strategies
        on key performance metrics. The analysis employed both frequentist and
        Bayesian methods to provide comprehensive evidence for decision-making.
        """

        return summary.strip()

    def _generate_ab_test_methods(self, test_config: dict[str, Any]) -> str:
        """Generate A/B test methodology section."""
        allocation = test_config.get("allocation_method", "fixed")
        alpha = test_config.get("alpha", 0.05)
        power = test_config.get("power", 0.8)

        methods = f"""
        ### Experimental Design
        - **Allocation Method:** {allocation}
        - **Significance Level:** α = {alpha}
        - **Statistical Power:** {power:.0%}

        ### Randomization
        Subjects were randomly allocated to treatment arms using {allocation}
        randomization to ensure balanced assignment and minimize selection bias.

        ### Statistical Analysis
        We employed both frequentist and Bayesian approaches:
        - **Frequentist:** Two-sample t-tests with appropriate corrections
        - **Bayesian:** Beta-binomial models with uninformative priors
        """

        return methods.strip()

    def _generate_ab_test_results(self, test_results: dict[str, Any]) -> str:
        """Generate A/B test results section."""
        arms = test_results.get("arms", {})

        results_text = "### Treatment Arm Performance\n\n"
        for arm_name, arm_data in arms.items():
            n = arm_data.get("n", 0)
            mean = arm_data.get("mean", 0)
            success_rate = arm_data.get("success_rate", 0)

            results_text += f"""
            **{arm_name}:**
            - Sample size: {n}
            - Mean outcome: {mean:.4f}
            - Success rate: {success_rate:.3f}
            """

        results_text += "\n### Statistical Inference\n"
        p_value = test_results.get("p_value")
        effect_size = test_results.get("effect_size")

        if p_value is not None:
            results_text += f"- P-value: {p_value:.4f}\n"
        if effect_size is not None:
            results_text += f"- Effect size: {effect_size:.4f}\n"

        return results_text.strip()

    def _generate_ab_test_conclusions(
        self, test_results: dict[str, Any], test_config: dict[str, Any]
    ) -> str:
        """Generate A/B test conclusions."""
        winner = test_results.get("winner")
        recommendation = test_results.get("recommendation", "")

        conclusions = f"""
        ### Primary Conclusions

        {recommendation}

        ### Implementation Recommendations

        1. **Immediate Actions:** {"Implement " + winner if winner else "Continue testing"}
        2. **Monitoring:** Set up ongoing performance tracking
        3. **Future Research:** Plan follow-up experiments

        ### Statistical Considerations

        - All tests employed appropriate multiple comparison corrections
        - Effect sizes provide practical significance beyond statistical significance
        - Confidence intervals guide implementation decisions
        """

        return conclusions.strip()

    def _generate_power_analysis_content(
        self, power_results: dict[str, Any], design_params: dict[str, Any]
    ) -> str:
        """Generate power analysis content."""
        content = f"""
        ### Study Design Parameters

        - **Primary Outcome:** {design_params.get('primary_outcome', 'Not specified')}
        - **Minimum Detectable Effect:** {design_params.get('min_effect_size', 0.1)}
        - **Statistical Power:** {design_params.get('power', 0.8):.0%}
        - **Significance Level:** α = {design_params.get('alpha', 0.05)}

        ### Sample Size Calculations

        Based on the specified parameters, the required sample size ensures
        adequate power to detect meaningful effects while controlling Type I error.

        ### Power Curve Analysis

        The power curve below shows the relationship between effect size and
        statistical power, helping to understand the trade-offs in study design.
        """

        return content.strip()

    def _generate_summary_stats_code(self, data: pd.DataFrame) -> str:
        """Generate code for summary statistics table."""
        code = """
        # Generate summary statistics
        summary_stats = data.describe()

        # Create formatted table
        print(summary_stats.round(3).to_markdown())
        """
        return code.strip()

    def _generate_results_table_code(self, results: dict[str, Any]) -> str:
        """Generate R code for results table."""
        code = """
        # Create results table
        results_df <- data.frame(
          Test = character(),
          Statistic = numeric(),
          P_Value = numeric(),
          Effect_Size = numeric(),
          CI_Lower = numeric(),
          CI_Upper = numeric(),
          stringsAsFactors = FALSE
        )

        # Format and display table
        results_df %>%
          kable(
            caption = "Statistical Test Results",
            digits = 4,
            col.names = c("Test", "Statistic", "P-value", "Effect Size", "CI Lower", "CI Upper")
          ) %>%
          kable_styling(
            bootstrap_options = c("striped", "hover", "condensed"),
            latex_options = c("striped", "hold_position")
          )
        """
        return code.strip()

    def _generate_effect_size_plot_code(self, results: dict[str, Any]) -> str:
        """Generate code for effect size plot."""
        code = """
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract effect sizes and confidence intervals
        # (This would be populated with actual results)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create forest plot of effect sizes
        y_pos = np.arange(len(effect_sizes))

        ax.errorbar(effect_sizes, y_pos, xerr=error_bars, fmt='o', capsize=5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Effect Size (Cohen\\'s d)')
        ax.set_title('Effect Sizes with 95% Confidence Intervals')

        plt.tight_layout()
        plt.show()
        """
        return code.strip()

    def _generate_ab_test_analysis_code(self, test_results: dict[str, Any]) -> str:
        """Generate analysis code for A/B test."""
        code = """
        # A/B Test Statistical Analysis
        from scipy import stats
        import numpy as np

        # Extract data for control and treatment groups
        control_data = np.array([])  # Would be populated with actual data
        treatment_data = np.array([])  # Would be populated with actual data

        # Perform statistical tests
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) +
                             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        """
        return code.strip()

    def _generate_power_curve_code(self, power_results: dict[str, Any]) -> str:
        """Generate code for power curve visualization."""
        code = """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        # Generate power curve
        effect_sizes = np.linspace(0, 1, 100)
        powers = []

        for es in effect_sizes:
            # Calculate power for each effect size
            power = stats.ttest_power(es, nobs=50, alpha=0.05)
            powers.append(power)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(effect_sizes, powers, linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
        plt.axhline(y=0.05, color='gray', linestyle=':', label='5% Type I Error')

        plt.xlabel('Effect Size (Cohen\\'s d)')
        plt.ylabel('Statistical Power')
        plt.title('Power Analysis: Effect Size vs Statistical Power')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()
        """
        return code.strip()

    def _generate_sample_size_table_code(self, power_results: dict[str, Any]) -> str:
        """Generate R code for sample size table."""
        code = """
        # Sample size calculations for different effect sizes
        library(pwr)

        effect_sizes <- c(0.2, 0.5, 0.8)
        sample_sizes <- sapply(effect_sizes, function(d) {
          result <- pwr.t.test(d = d, sig.level = 0.05, power = 0.8)
          ceiling(result$n)
        })

        sample_size_df <- data.frame(
          Effect_Size = c("Small (0.2)", "Medium (0.5)", "Large (0.8)"),
          Cohen_d = effect_sizes,
          Required_n = sample_sizes,
          Total_n = sample_sizes * 2
        )

        sample_size_df %>%
          kable(
            caption = "Required Sample Sizes by Effect Size",
            col.names = c("Effect Size", "Cohen's d", "n per group", "Total n")
          ) %>%
          kable_styling(
            bootstrap_options = c("striped", "hover"),
            latex_options = "hold_position"
          )
        """
        return code.strip()

    def render_document(self, doc: QuartoDocument, filename: str, render: bool = True) -> Path:
        """
        Render Quarto document to file.

        Args:
            doc: QuartoDocument to render
            filename: Output filename (without extension)
            render: Whether to actually render with Quarto

        Returns:
            Path to generated .qmd file
        """
        # Select template
        template = self.jinja_env.get_template(f"{doc.template}.qmd")

        # Render content
        content = template.render(
            title=doc.title,
            author=doc.author,
            date=doc.date,
            abstract=doc.abstract,
            keywords=doc.keywords,
            format=doc.format,
            sections=doc.sections,
            bibliography=doc.bibliography,
            **doc.metadata,
        )

        # Write .qmd file
        output_path = self.output_dir / f"{filename}.qmd"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated Quarto document: {output_path}")

        # Render with Quarto if requested
        if render:
            try:
                import subprocess

                result = subprocess.run(
                    ["quarto", "render", str(output_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.output_dir),
                )

                if result.returncode == 0:
                    logger.info(f"Successfully rendered document to {doc.format}")
                else:
                    logger.error(f"Quarto render failed: {result.stderr}")

            except FileNotFoundError:
                logger.warning("Quarto not found. Install Quarto to render documents.")

        return output_path

    def create_latex_table(
        self, data: pd.DataFrame, caption: str, label: str = "", format_dict: dict | None = None
    ) -> str:
        """
        Create publication-ready LaTeX table.

        Args:
            data: DataFrame to convert
            caption: Table caption
            label: LaTeX label for referencing
            format_dict: Formatting specifications

        Returns:
            LaTeX table string
        """
        # Default formatting
        if format_dict is None:
            format_dict = {
                "position": "H",
                "alignment": "c" * len(data.columns),
                "booktabs": True,
                "escape": False,
            }

        # Generate LaTeX
        latex_string = data.to_latex(
            index=False,
            escape=False,
            column_format=format_dict.get("alignment"),
            position=format_dict.get("position", "H"),
            caption=caption,
            label=f"tab:{label}" if label else None,
            **{k: v for k, v in format_dict.items() if k not in ["position", "alignment"]},
        )

        # Add booktabs formatting if requested
        if format_dict.get("booktabs", True):
            latex_string = latex_string.replace("\\hline", "\\toprule", 1)
            latex_string = latex_string.replace("\\hline", "\\bottomrule", 1)
            # Add midrule after header
            lines = latex_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().endswith("\\\\") and "toprule" not in lines[i - 1]:
                    lines.insert(i + 1, "\\midrule")
                    break
            latex_string = "\n".join(lines)

        return latex_string


def test():
    """Test the Quarto generator."""
    print("=== Quarto Generator Demo ===\n")

    # Create generator
    generator = QuartoGenerator()

    # Test statistical report
    mock_results = {
        "test_results": {
            "t_test": {"p_value": 0.023, "effect_size": 0.65, "confidence_interval": (0.1, 1.2)},
            "permutation": {
                "p_value": 0.019,
                "effect_size": 0.72,
                "confidence_interval": (0.15, 1.29),
            },
        },
        "multiple_comparisons": {
            "method": "benjamini_hochberg",
            "num_rejected": 2,
            "total_tests": 5,
        },
    }

    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "group": ["control"] * 50 + ["treatment"] * 50,
            "outcome": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(0.5, 1, 50)]),
            "baseline": np.random.normal(100, 15, 100),
        }
    )

    # Create statistical report
    doc = generator.create_statistical_report(
        title="NFL Betting Strategy Analysis",
        author="Research Team",
        results=mock_results,
        data=data,
    )

    # Render document
    output_path = generator.render_document(doc, "statistical_analysis", render=False)
    print(f"Generated report: {output_path}")

    # Test A/B test report
    ab_test_results = {
        "winner": "treatment",
        "confidence": 0.85,
        "p_value": 0.023,
        "arms": {
            "control": {"n": 100, "mean": 0.02, "success_rate": 0.15},
            "treatment": {"n": 98, "mean": 0.07, "success_rate": 0.22},
        },
        "recommendation": "Implement treatment strategy",
    }

    ab_test_config = {
        "name": "Kelly Criterion vs Fixed Betting",
        "allocation_method": "adaptive",
        "alpha": 0.05,
        "power": 0.8,
    }

    ab_doc = generator.create_ab_test_report(ab_test_results, ab_test_config)
    ab_output = generator.render_document(ab_doc, "ab_test_analysis", render=False)
    print(f"Generated A/B test report: {ab_output}")

    # Test LaTeX table
    latex_table = generator.create_latex_table(
        data.groupby("group")["outcome"].describe(),
        caption="Descriptive Statistics by Group",
        label="descriptive-stats",
    )
    print("\nGenerated LaTeX table:")
    print(latex_table[:300] + "...")


if __name__ == "__main__":
    test()
