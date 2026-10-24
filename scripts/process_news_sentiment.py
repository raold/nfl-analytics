#!/usr/bin/env python3
"""
Batch LLM Processing for News Sentiment Analysis.

Processes all unprocessed news articles in semantic.news_snapshots using Ollama.
Generates sentiment scores (-1.0 to +1.0) and stores them in semantic.sentiment_scores.

Each news article is analyzed per team mentioned, allowing team-specific sentiment tracking.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor

from nfl_analytics.semantic.models.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Category-based default sentiment for rule-based processing
CATEGORY_SENTIMENT = {
    "injury": -0.3,       # Injuries are generally negative
    "trade": 0.0,         # Neutral (depends on context)
    "signing": 0.2,       # Generally positive
    "release": -0.2,      # Generally negative
    "suspension": -0.4,   # Negative
    "general": 0.0,       # Neutral
    "game_preview": 0.1,  # Slight positive (anticipation)
    "game_recap": 0.0,    # Depends on outcome
}


def get_unprocessed_news(limit: int = 50) -> list[dict]:
    """Get news articles without sentiment scores."""
    query = """
    SELECT n.id, n.title, n.description, n.teams, n.category, n.published_at
    FROM semantic.news_snapshots n
    WHERE n.processed = false
    ORDER BY n.published_at DESC
    LIMIT %s
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return [dict(row) for row in cur.fetchall()]


def compute_rule_based_sentiment(news: dict) -> tuple[float, float]:
    """
    Compute sentiment using rule-based approach (fast, no LLM).

    Returns (sentiment, confidence) tuple.
    """
    title = news.get("title", "").lower()
    category = news.get("category", "general")

    # Start with category-based baseline
    sentiment = CATEGORY_SENTIMENT.get(category, 0.0)

    # Adjust based on keywords in title
    positive_keywords = ["wins", "victory", "dominant", "comeback", "hero", "breakout", "surge"]
    negative_keywords = ["injury", "out", "loss", "loses", "suspended", "benched", "struggle",
                        "disappoints", "fails", "doubt", "questionable", "acl", "torn"]

    for word in positive_keywords:
        if word in title:
            sentiment += 0.2

    for word in negative_keywords:
        if word in title:
            sentiment -= 0.2

    # Clamp to -1.0 to 1.0
    sentiment = max(-1.0, min(1.0, sentiment))

    # Confidence is lower for rule-based
    confidence = 0.5

    return round(sentiment, 2), confidence


def compute_llm_sentiment(client: OllamaClient, news: dict, team: str) -> tuple[float, float] | None:
    """
    Use LLM to assess news sentiment for a specific team.

    Returns (sentiment, confidence) tuple or None on error.
    """
    title = news.get("title", "")
    description = news.get("description", "")
    category = news.get("category", "general")

    text = f"{title}. {description}" if description else title

    prompt = f"""Analyze this NFL news article for {team} team sentiment.

News: {text}
Category: {category}

Rate sentiment on scale from -1.0 (very negative) to +1.0 (very positive):
- Injuries, suspensions, losses = negative
- Wins, signings, comebacks = positive
- Neutral news = 0.0

Respond with ONLY two numbers separated by comma: sentiment,confidence
Example: -0.4,0.8"""

    try:
        response = client.generate(prompt, temperature=0.1)
        text = response.response.strip()

        # Parse "sentiment,confidence" format
        match = re.search(r"(-?\d+\.?\d*)\s*,\s*(\d+\.?\d*)", text)
        if match:
            sentiment = float(match.group(1))
            confidence = float(match.group(2))

            # Normalize
            sentiment = max(-1.0, min(1.0, sentiment))
            confidence = max(0.0, min(1.0, confidence))

            return round(sentiment, 2), round(confidence, 2)

        # Try to extract just a single number (sentiment)
        match = re.search(r"(-?\d+\.?\d*)", text)
        if match:
            sentiment = float(match.group(1))
            sentiment = max(-1.0, min(1.0, sentiment))
            return round(sentiment, 2), 0.7

        logger.warning(f"Could not parse LLM response: {text}")
        return None

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


def insert_sentiment_score(
    news_id: int,
    team: str,
    sentiment: float,
    confidence: float,
    model_used: str,
) -> bool:
    """Insert a sentiment score into the database."""
    query = """
    INSERT INTO semantic.sentiment_scores
        (news_id, team, sentiment, confidence, model_used, prompt_version)
    VALUES (%s, %s, %s, %s, %s, 'v1')
    ON CONFLICT DO NOTHING
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (news_id, team, sentiment, confidence, model_used))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to insert sentiment: {e}")
        return False


def mark_news_processed(news_id: int) -> bool:
    """Mark a news article as processed."""
    query = """
    UPDATE semantic.news_snapshots
    SET processed = true, processed_at = NOW()
    WHERE id = %s
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (news_id,))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to mark processed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process news sentiment scores")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of news articles to process per batch (default: 50)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for nuanced sentiment (slower)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute sentiment but don't update database",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("NEWS SENTIMENT PROCESSING")
    print("=" * 60)
    print(f"Mode: {'LLM (qwen3:8b)' if args.use_llm else 'Rule-based (fast)'}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize Ollama client if using LLM
    client = None
    model_used = "rule-based"
    if args.use_llm:
        client = OllamaClient(timeout=120.0)
        if not client.is_available():
            print("ERROR: Ollama not available. Run 'ollama serve' first.")
            print("Falling back to rule-based scoring...")
            args.use_llm = False
        else:
            model_used = "qwen3:8b"
            print("Ollama connected successfully")

    total_articles = 0
    total_sentiments = 0
    start_time = time.time()

    while True:
        news_items = get_unprocessed_news(limit=args.batch_size)
        if not news_items:
            print("\nNo more unprocessed news articles.")
            break

        print(f"\nProcessing batch of {len(news_items)} articles...")

        for news in news_items:
            teams = news.get("teams") or []

            # If no teams associated, use a generic approach
            if not teams:
                teams = ["NFL"]

            for team in teams:
                # Compute sentiment
                if args.use_llm:
                    result = compute_llm_sentiment(client, news, team)
                    if result is None:
                        result = compute_rule_based_sentiment(news)
                        model_used = "rule-based-fallback"
                    else:
                        model_used = "qwen3:8b"
                else:
                    result = compute_rule_based_sentiment(news)
                    model_used = "rule-based"

                sentiment, confidence = result

                # Insert sentiment score
                if not args.dry_run:
                    insert_sentiment_score(
                        news_id=news["id"],
                        team=team,
                        sentiment=sentiment,
                        confidence=confidence,
                        model_used=model_used,
                    )
                total_sentiments += 1

            # Mark article as processed
            if not args.dry_run:
                mark_news_processed(news["id"])

            total_articles += 1

        # Rate limit for LLM
        if args.use_llm:
            time.sleep(0.5)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Articles processed: {total_articles}")
    print(f"Sentiment scores created: {total_sentiments}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print()

    if not args.dry_run:
        # Show sample results
        print("Sample sentiment scores:")
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT s.team, s.sentiment, s.confidence, s.model_used,
                           LEFT(n.title, 50) as title
                    FROM semantic.sentiment_scores s
                    JOIN semantic.news_snapshots n ON s.news_id = n.id
                    ORDER BY ABS(s.sentiment) DESC
                    LIMIT 10
                """)
                for row in cur.fetchall():
                    sign = "+" if row["sentiment"] > 0 else ""
                    print(f"  {row['team']:4} | {sign}{row['sentiment']:.2f} | {row['title']}...")

        # Verify final state
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_news,
                        COUNT(*) FILTER (WHERE processed = true) as processed
                    FROM semantic.news_snapshots
                """)
                news_row = cur.fetchone()

                cur.execute("SELECT COUNT(*) FROM semantic.sentiment_scores")
                sentiment_count = cur.fetchone()[0]

                cur.execute("SELECT AVG(sentiment) FROM semantic.sentiment_scores")
                avg_sentiment = cur.fetchone()[0]

                print(f"\nDatabase state:")
                print(f"  News articles: {news_row[0]} ({news_row[1]} processed)")
                print(f"  Sentiment scores: {sentiment_count}")
                if avg_sentiment:
                    print(f"  Average sentiment: {avg_sentiment:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
