"""
Data ingestion for semantic processing.

Real-time and batch ingestion from various sources:
    - NewsIngester: ESPN, beat reporters, official team sites
    - SocialIngester: Twitter/X (verified accounts)
    - StreamProcessor: Kafka/Redis streaming (future)
"""

from .news_ingester import (
    NewsIngester,
    NewsItem,
    NewsCategory,
    NewsPriority,
    classify_news,
    extract_teams_from_text,
)
from .social_ingester import (
    SocialIngester,
    SocialPost,
    TweetCategory,
    SourceTier,
    classify_tweet,
)

__all__ = [
    # News
    "NewsIngester",
    "NewsItem",
    "NewsCategory",
    "NewsPriority",
    "classify_news",
    "extract_teams_from_text",
    # Social
    "SocialIngester",
    "SocialPost",
    "TweetCategory",
    "SourceTier",
    "classify_tweet",
]
