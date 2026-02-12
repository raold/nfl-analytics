#!/usr/bin/env python3
"""
Automated Methodology Documentation Generator.

Creates comprehensive methodology sections for academic papers and reports
based on statistical analyses performed, ensuring reproducibility and
academic standards compliance.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of statistical analyses."""

    DESCRIPTIVE = "descriptive"
    HYPOTHESIS_TEST = "hypothesis_test"
    REGRESSION = "regression"
    ANOVA = "anova"
    CORRELATION = "correlation"
    SURVIVAL = "survival"
    BAYESIAN = "bayesian"
    MACHINE_LEARNING = "machine_learning"
    TIME_SERIES = "time_series"


class StudyDesign(Enum):
    """Types of study designs."""

    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_CONTROL = "case_control"
    COHORT = "cohort"


@dataclass
class MethodDescription:
    """Description of a statistical method used."""

    name: str
    category: AnalysisType
    assumptions: list[str] = field(default_factory=list)
    software: str = "Python"
    packages: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    references: list[str] = field(default_factory=list)


@dataclass
class DataDescription:
    """Description of dataset characteristics."""

    n_observations: int
    n_variables: int
    missing_data_approach: str = "complete_case"
    outlier_handling: str = "none"
    transformations: list[str] = field(default_factory=list)
    quality_checks: list[str] = field(default_factory=list)


@dataclass
class MethodologyDocument:
    """Complete methodology documentation."""

    study_design: StudyDesign
    data_description: DataDescription
    methods: list[MethodDescription]
    multiple_testing: str | None = None
    significance_level: float = 0.05
    power_analysis: str | None = None
    software_versions: dict[str, str] = field(default_factory=dict)
    reproducibility_statement: str = ""
    limitations: list[str] = field(default_factory=list)


class MethodologyDocumenter:
    """
    Automated methodology documentation generator.

    Creates comprehensive, publication-ready methodology sections
    based on analyses performed and study design.
    """

    def __init__(self):
        """Initialize methodology documenter."""
        self.standard_methods = self._load_standard_methods()
        self.templates = self._load_templates()

    def _load_standard_methods(self) -> dict[str, MethodDescription]:
        """Load standard method descriptions."""
        methods = {}

        # Hypothesis Testing Methods
        methods["permutation_test"] = MethodDescription(
            name="Permutation Test",
            category=AnalysisType.HYPOTHESIS_TEST,
            assumptions=["Exchangeability under null hypothesis", "Independent observations"],
            software="Python",
            packages=["scipy", "numpy"],
            justification="Distribution-free alternative to parametric tests, robust to non-normal distributions",
            references=[
                "Good, P. I. (2000). Permutation Tests: A Practical Guide to Resampling Methods"
            ],
        )

        methods["bootstrap_test"] = MethodDescription(
            name="Bootstrap Test",
            category=AnalysisType.HYPOTHESIS_TEST,
            assumptions=["Sample represents population", "Independent observations"],
            software="Python",
            packages=["scipy", "numpy"],
            justification="Non-parametric approach for confidence interval estimation and hypothesis testing",
            references=["Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap"],
        )

        methods["t_test"] = MethodDescription(
            name="Student's t-test",
            category=AnalysisType.HYPOTHESIS_TEST,
            assumptions=[
                "Normal distribution of data or large sample size",
                "Independent observations",
                "Homogeneity of variance (for two-sample tests)",
            ],
            software="Python",
            packages=["scipy.stats"],
            justification="Standard parametric test for comparing means",
            references=["Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25"],
        )

        methods["mann_whitney"] = MethodDescription(
            name="Mann-Whitney U Test",
            category=AnalysisType.HYPOTHESIS_TEST,
            assumptions=[
                "Independent observations",
                "Ordinal or continuous data",
                "Similar distribution shapes under null",
            ],
            software="Python",
            packages=["scipy.stats"],
            justification="Non-parametric alternative to t-test for non-normal distributions",
            references=[
                "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other"
            ],
        )

        methods["benjamini_hochberg"] = MethodDescription(
            name="Benjamini-Hochberg Procedure",
            category=AnalysisType.HYPOTHESIS_TEST,
            assumptions=["Test statistics are independent or have positive dependence"],
            software="Python",
            packages=["statsmodels"],
            justification="Controls false discovery rate in multiple testing scenarios",
            references=[
                "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing"
            ],
        )

        methods["bayesian_test"] = MethodDescription(
            name="Bayesian Hypothesis Testing",
            category=AnalysisType.BAYESIAN,
            assumptions=[
                "Prior distributions appropriately specified",
                "Model adequately represents data generating process",
            ],
            software="Python",
            packages=["pymc", "arviz"],
            justification="Provides probabilistic interpretation of evidence and incorporates prior knowledge",
            references=[
                "Kruschke, J. K. (2014). Doing Bayesian data analysis: A tutorial with R, JAGS, and Stan"
            ],
        )

        # Effect Size Methods
        methods["cohens_d"] = MethodDescription(
            name="Cohen's d",
            category=AnalysisType.DESCRIPTIVE,
            assumptions=["Approximately normal distributions", "Homogeneity of variance"],
            software="Python",
            packages=["numpy"],
            justification="Standardized effect size measure for mean differences",
            references=["Cohen, J. (1988). Statistical power analysis for the behavioral sciences"],
        )

        methods["cliffs_delta"] = MethodDescription(
            name="Cliff's delta",
            category=AnalysisType.DESCRIPTIVE,
            assumptions=["Ordinal or continuous data"],
            software="Python",
            packages=["numpy"],
            justification="Non-parametric effect size measure robust to outliers and non-normal distributions",
            references=[
                "Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions"
            ],
        )

        return methods

    def _load_templates(self) -> dict[str, str]:
        """Load methodology section templates."""
        templates = {}

        templates[
            "study_design"
        ] = """
        ## Study Design

        This {design_type} study was conducted to {study_objective}.
        {design_description}

        ### Data Collection
        {data_collection_description}

        ### Sample Size and Power
        {power_analysis_description}
        """

        templates[
            "statistical_methods"
        ] = """
        ## Statistical Analysis

        All statistical analyses were performed using {software} version {version}.
        Statistical significance was set at α = {alpha}.

        ### Descriptive Statistics
        {descriptive_methods}

        ### Inferential Statistics
        {inferential_methods}

        ### Multiple Testing Correction
        {multiple_testing_description}

        ### Effect Size Measures
        {effect_size_description}

        ### Missing Data
        {missing_data_description}

        ### Assumptions Testing
        {assumptions_description}
        """

        templates[
            "reproducibility"
        ] = """
        ## Reproducibility and Data Availability

        {reproducibility_statement}

        ### Software and Packages
        {software_details}

        ### Code Availability
        {code_availability}

        ### Data Availability
        {data_availability}
        """

        return templates

    def document_analysis(
        self,
        methods_used: list[str],
        study_config: dict[str, Any],
        data_description: DataDescription | None = None,
    ) -> MethodologyDocument:
        """
        Create methodology documentation based on methods used.

        Args:
            methods_used: List of method names that were applied
            study_config: Study configuration parameters
            data_description: Description of dataset

        Returns:
            Complete methodology documentation
        """
        # Identify method descriptions
        method_descriptions = []
        for method_name in methods_used:
            if method_name in self.standard_methods:
                method_descriptions.append(self.standard_methods[method_name])
            else:
                logger.warning(f"Unknown method: {method_name}")

        # Create study design
        study_design = StudyDesign(study_config.get("design", "experimental"))

        # Multiple testing approach
        multiple_testing = self._determine_multiple_testing(method_descriptions)

        # Power analysis description
        power_analysis = self._generate_power_analysis(study_config)

        # Software versions
        software_versions = study_config.get(
            "software_versions",
            {"Python": "3.9+", "scipy": "1.7+", "numpy": "1.21+", "pandas": "1.3+"},
        )

        doc = MethodologyDocument(
            study_design=study_design,
            data_description=data_description or DataDescription(0, 0),
            methods=method_descriptions,
            multiple_testing=multiple_testing,
            significance_level=study_config.get("alpha", 0.05),
            power_analysis=power_analysis,
            software_versions=software_versions,
            reproducibility_statement=self._generate_reproducibility_statement(),
            limitations=study_config.get("limitations", []),
        )

        return doc

    def generate_methods_section(self, doc: MethodologyDocument) -> str:
        """
        Generate complete methods section text.

        Args:
            doc: Methodology document

        Returns:
            Formatted methods section
        """
        sections = []

        # Study Design
        sections.append(self._format_study_design(doc))

        # Statistical Methods
        sections.append(self._format_statistical_methods(doc))

        # Reproducibility
        sections.append(self._format_reproducibility(doc))

        return "\n\n".join(sections)

    def _format_study_design(self, doc: MethodologyDocument) -> str:
        """Format study design section."""
        design_descriptions = {
            StudyDesign.EXPERIMENTAL: "controlled experimental design",
            StudyDesign.OBSERVATIONAL: "observational study design",
            StudyDesign.LONGITUDINAL: "longitudinal study design",
            StudyDesign.CROSS_SECTIONAL: "cross-sectional study design",
        }

        design_desc = design_descriptions.get(doc.study_design, "research design")

        section = f"""
        ## Study Design

        This study employed a {design_desc} to evaluate the effectiveness of different
        NFL betting strategies. The design was chosen to {self._get_design_justification(doc.study_design)}.

        ### Sample Characteristics
        The dataset comprised {doc.data_description.n_observations:,} observations with
        {doc.data_description.n_variables} variables. {self._format_missing_data_approach(doc.data_description)}.
        """

        if doc.power_analysis:
            section += f"\n\n### Power Analysis\n{doc.power_analysis}"

        return section.strip()

    def _format_statistical_methods(self, doc: MethodologyDocument) -> str:
        """Format statistical methods section."""
        software_list = ", ".join(
            [f"{name} ({version})" for name, version in doc.software_versions.items()]
        )

        section = f"""
        ## Statistical Analysis

        All statistical analyses were conducted using {software_list}.
        The significance level was set at α = {doc.significance_level}.

        ### Analysis Methods
        """

        # Group methods by category
        method_groups = {}
        for method in doc.methods:
            category = method.category.value
            if category not in method_groups:
                method_groups[category] = []
            method_groups[category].append(method)

        # Format each category
        for category, methods in method_groups.items():
            section += f"\n\n#### {category.replace('_', ' ').title()}\n"

            for method in methods:
                section += f"""
                **{method.name}**: {method.justification}
                """

                if method.assumptions:
                    section += " Key assumptions: " + "; ".join(method.assumptions) + "."

                if method.parameters:
                    param_str = ", ".join([f"{k}={v}" for k, v in method.parameters.items()])
                    section += f" Parameters: {param_str}."

        # Multiple testing
        if doc.multiple_testing:
            section += f"""

            ### Multiple Testing Correction
            {doc.multiple_testing}
            """

        # Effect sizes
        effect_size_methods = [
            m
            for m in doc.methods
            if "effect" in m.name.lower() or m.category == AnalysisType.DESCRIPTIVE
        ]
        if effect_size_methods:
            section += """

            ### Effect Size Measures
            Effect sizes were calculated to assess practical significance beyond statistical significance.
            """

            for method in effect_size_methods:
                if "effect" in method.name.lower() or method.name in ["Cohen's d", "Cliff's delta"]:
                    section += (
                        f" {method.name} was used as appropriate for the data type and analysis."
                    )

        return section.strip()

    def _format_reproducibility(self, doc: MethodologyDocument) -> str:
        """Format reproducibility section."""
        section = f"""
        ## Reproducibility and Transparency

        {doc.reproducibility_statement}

        ### Software Environment
        Analyses were conducted using the following software and package versions:
        """

        for software, version in doc.software_versions.items():
            section += f"\n- {software}: {version}"

        # Package dependencies
        all_packages = set()
        for method in doc.methods:
            all_packages.update(method.packages)

        if all_packages:
            section += "\n\nKey packages used: " + ", ".join(sorted(all_packages))

        section += """

        ### Code and Data Availability
        Statistical analysis code is available in the project repository with version control
        ensuring complete reproducibility. All random processes used fixed seeds for
        deterministic results.
        """

        if doc.limitations:
            section += "\n\n### Limitations\n"
            for limitation in doc.limitations:
                section += f"- {limitation}\n"

        return section.strip()

    def _determine_multiple_testing(self, methods: list[MethodDescription]) -> str | None:
        """Determine appropriate multiple testing description."""
        hypothesis_tests = [m for m in methods if m.category == AnalysisType.HYPOTHESIS_TEST]

        if len(hypothesis_tests) <= 1:
            return None

        # Check if multiple testing correction method is included
        correction_methods = ["benjamini_hochberg", "bonferroni", "holm_bonferroni"]
        has_correction = any(
            m.name.lower().replace(" ", "_").replace("-", "_") in correction_methods
            for m in methods
        )

        if has_correction:
            return (
                "Multiple testing corrections were applied to control for inflated Type I error "
                "rates when performing multiple simultaneous statistical tests."
            )
        else:
            return (
                "Multiple statistical tests were performed. Readers should interpret "
                "p-values with caution due to potential inflation of Type I error rates."
            )

    def _generate_power_analysis(self, study_config: dict[str, Any]) -> str | None:
        """Generate power analysis description."""
        if "power_analysis" not in study_config:
            return None

        power_info = study_config["power_analysis"]
        power = power_info.get("power", 0.8)
        effect_size = power_info.get("effect_size", 0.5)
        alpha = power_info.get("alpha", 0.05)

        return (
            f"Sample size calculations were based on detecting a medium effect size "
            f"(Cohen's d = {effect_size}) with {power:.0%} statistical power at "
            f"α = {alpha}, assuming a two-tailed test."
        )

    def _generate_reproducibility_statement(self) -> str:
        """Generate standard reproducibility statement."""
        return (
            "All analyses were conducted following reproducible research practices. "
            "Code is version-controlled and publicly available. Random number generators "
            "used fixed seeds to ensure deterministic results. Data processing and "
            "analysis steps are fully documented and can be independently verified."
        )

    def _get_design_justification(self, design: StudyDesign) -> str:
        """Get justification for study design choice."""
        justifications = {
            StudyDesign.EXPERIMENTAL: "allow causal inference about treatment effects while controlling for confounding variables",
            StudyDesign.OBSERVATIONAL: "examine relationships in naturalistic settings without experimental manipulation",
            StudyDesign.LONGITUDINAL: "assess changes over time and examine temporal relationships",
            StudyDesign.CROSS_SECTIONAL: "provide a snapshot of relationships at a specific time point",
        }
        return justifications.get(design, "address the research questions effectively")

    def _format_missing_data_approach(self, data_desc: DataDescription) -> str:
        """Format missing data handling approach."""
        approaches = {
            "complete_case": "Complete case analysis was employed, excluding observations with missing values",
            "multiple_imputation": "Multiple imputation was used to handle missing data",
            "listwise_deletion": "Listwise deletion was applied to handle missing observations",
            "mean_imputation": "Missing values were imputed using mean substitution",
        }

        return approaches.get(
            data_desc.missing_data_approach,
            f"Missing data were handled using {data_desc.missing_data_approach}",
        )

    def generate_citation_list(self, doc: MethodologyDocument) -> list[str]:
        """Generate list of citations for methods used."""
        citations = set()

        for method in doc.methods:
            citations.update(method.references)

        # Add standard citations
        citations.add("R Core Team (2021). R: A language and environment for statistical computing")
        citations.add("Python Software Foundation. Python Language Reference, version 3.9")

        return sorted(list(citations))

    def export_to_file(
        self, doc: MethodologyDocument, filename: str, format: str = "markdown"
    ) -> Path:
        """
        Export methodology documentation to file.

        Args:
            doc: Methodology document
            filename: Output filename
            format: Output format ("markdown", "latex", "json")

        Returns:
            Path to exported file
        """
        if format == "markdown":
            content = self.generate_methods_section(doc)
            file_path = Path(f"{filename}.md")
        elif format == "latex":
            content = self._convert_to_latex(self.generate_methods_section(doc))
            file_path = Path(f"{filename}.tex")
        elif format == "json":
            content = json.dumps(doc.__dict__, indent=2, default=str)
            file_path = Path(f"{filename}.json")
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Exported methodology documentation to {file_path}")
        return file_path

    def _convert_to_latex(self, markdown_text: str) -> str:
        """Convert markdown methodology to LaTeX format."""
        # Basic markdown to LaTeX conversion
        latex_text = markdown_text

        # Headers
        latex_text = latex_text.replace("## ", "\\section{").replace("\n", "}\n", 1)
        latex_text = latex_text.replace("### ", "\\subsection{").replace("\n", "}\n", 1)
        latex_text = latex_text.replace("#### ", "\\subsubsection{").replace("\n", "}\n", 1)

        # Bold text
        latex_text = latex_text.replace("**", "\\textbf{").replace("**", "}", 1)

        # Italic text
        latex_text = latex_text.replace("*", "\\textit{").replace("*", "}", 1)

        # Lists
        latex_text = latex_text.replace("- ", "\\item ")

        return latex_text


def generate_methods_section(analysis_log: dict[str, Any], study_config: dict[str, Any]) -> str:
    """
    Convenience function to generate methods section from analysis log.

    Args:
        analysis_log: Log of analyses performed
        study_config: Study configuration

    Returns:
        Formatted methods section
    """
    documenter = MethodologyDocumenter()

    # Extract methods used from analysis log
    methods_used = analysis_log.get("methods_used", [])

    # Create data description
    data_desc = DataDescription(
        n_observations=analysis_log.get("n_observations", 0),
        n_variables=analysis_log.get("n_variables", 0),
        missing_data_approach=analysis_log.get("missing_data_approach", "complete_case"),
    )

    # Generate documentation
    doc = documenter.document_analysis(methods_used, study_config, data_desc)

    return documenter.generate_methods_section(doc)


def test():
    """Test methodology documenter."""
    print("=== Methodology Documenter Demo ===\n")

    documenter = MethodologyDocumenter()

    # Mock analysis configuration
    methods_used = ["permutation_test", "bootstrap_test", "cohens_d", "benjamini_hochberg"]

    study_config = {
        "design": "experimental",
        "alpha": 0.05,
        "power_analysis": {"power": 0.8, "effect_size": 0.5, "alpha": 0.05},
        "software_versions": {"Python": "3.9.7", "scipy": "1.7.3", "numpy": "1.21.2"},
        "limitations": [
            "Sample limited to specific betting contexts",
            "Results may not generalize to all sports betting scenarios",
        ],
    }

    data_description = DataDescription(
        n_observations=1000,
        n_variables=15,
        missing_data_approach="complete_case",
        quality_checks=["outlier detection", "normality testing"],
    )

    # Generate documentation
    doc = documenter.document_analysis(methods_used, study_config, data_description)

    # Generate methods section
    methods_section = documenter.generate_methods_section(doc)

    print("Generated Methods Section:")
    print("=" * 50)
    print(methods_section[:1000] + "...")

    # Generate citations
    citations = documenter.generate_citation_list(doc)
    print(f"\n\nRequired Citations ({len(citations)}):")
    for i, citation in enumerate(citations[:5], 1):
        print(f"{i}. {citation}")

    # Test export
    output_path = documenter.export_to_file(doc, "methodology_example", "markdown")
    print(f"\nExported to: {output_path}")


if __name__ == "__main__":
    test()
