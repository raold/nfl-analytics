"""
Training infrastructure for NFL semantic models.

Provides annotation pipelines and fine-tuning utilities for:
    - NER (Named Entity Recognition) on NFL corpus
    - Sentiment analysis on NFL news
    - Custom model training with LoRA
"""

from .annotation_pipeline import (
    AnnotationPipeline,
    NERAnnotation,
    SentimentAnnotation,
    NER_LABELS,
)

__all__ = [
    "AnnotationPipeline",
    "NERAnnotation",
    "SentimentAnnotation",
    "NER_LABELS",
]
