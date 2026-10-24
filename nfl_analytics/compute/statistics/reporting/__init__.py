"""
Reporting Module for NFL Analytics Statistics.

Provides automated report generation capabilities including Quarto integration,
LaTeX table generation, and methodology documentation.
"""

from .latex_tables import (
    LaTeXTableGenerator,
    StatisticalTable,
    format_confidence_interval,
    format_p_value,
)
from .methodology_documenter import (
    MethodDescription,
    MethodologyDocumenter,
    generate_methods_section,
)
from .quarto_generator import QuartoDocument, QuartoGenerator, QuartoSection

__all__ = [
    "QuartoGenerator",
    "QuartoDocument",
    "QuartoSection",
    "LaTeXTableGenerator",
    "StatisticalTable",
    "format_p_value",
    "format_confidence_interval",
    "MethodologyDocumenter",
    "MethodDescription",
    "generate_methods_section",
]
