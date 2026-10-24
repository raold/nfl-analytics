"""
Semantic Stack - LLM-powered semantic understanding for NFL analytics.

This module provides NLP and LLM capabilities to enhance quantitative predictions
with semantic understanding from news, social media, and other text sources.

Architecture:
    Layer 1: Data Ingestion (ingestion/)
        - News fetching and parsing
        - Social media monitoring
        - Real-time streaming via Redis/Kafka

    Layer 2: LLM Processing Services (services/)
        - NER for players, teams, injuries
        - Sentiment analysis
        - Summarization and classification
        - RAG-powered expert reasoning

    Layer 3: Feature Integration (features/)
        - Semantic feature engineering
        - Feature store materialization
        - Ensemble integration

    Layer 4: Feedback Loop (feedback/)
        - Analyst feedback collection
        - Model performance tracking
        - Continuous improvement

Models:
    - Mistral-7B-Instruct (LoRA fine-tuned) for NER
    - RoBERTa-large (fine-tuned) for sentiment
    - all-MiniLM-L6-v2 for embeddings
    - Qdrant for vector similarity search

Target Performance:
    - Win rate: 54.5% -> 56-57% (+1.5-2.5%)
    - Props MAE: 44.25 -> 34-39 (-5 to -10 pts)
    - Latency: Maintain <500ms with pre-computation
"""

__version__ = "0.1.0"
__author__ = "NFL Analytics Team"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .services.ner_service import NERService
    from .services.sentiment_service import SentimentService
    from .ingestion.news_ingester import NewsIngester

__all__ = [
    "NERService",
    "SentimentService",
    "NewsIngester",
]
