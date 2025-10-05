"""
ETL Pipeline Orchestrators.

Provides end-to-end pipeline execution following Extract → Validate → Transform → Load pattern.
"""

from .base_pipeline import BasePipeline, PipelineResult

# Daily and weekly pipelines will be imported when implemented
try:
    from .daily import DailyPipeline
    from .weekly import WeeklyPipeline
    __all__ = ['BasePipeline', 'PipelineResult', 'DailyPipeline', 'WeeklyPipeline']
except ImportError:
    __all__ = ['BasePipeline', 'PipelineResult']
