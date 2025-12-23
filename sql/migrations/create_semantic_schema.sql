-- Semantic Data Infrastructure Schema
-- Purpose: Store news, sentiment, and source health data for semantic features
-- Author: Claude Code
-- Date: 2025-12-22

-- Create semantic schema
CREATE SCHEMA IF NOT EXISTS semantic;

-- ============================================================================
-- NEWS SNAPSHOTS
-- Raw news storage with temporal tracking and deduplication
-- ============================================================================
CREATE TABLE IF NOT EXISTS semantic.news_snapshots (
    id SERIAL PRIMARY KEY,

    -- Source identification
    source VARCHAR(50) NOT NULL,        -- 'espn_api', 'rotowire', etc.
    source_id VARCHAR(100),             -- Original ID from source
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication

    -- Content
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,

    -- Temporal fields (critical for avoiding data leakage)
    published_at TIMESTAMPTZ,           -- When article was published
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When we fetched it

    -- Structured metadata
    raw_json JSONB,                     -- Original API/scrape response
    teams TEXT[],                       -- Team abbreviations mentioned
    players TEXT[],                     -- Player names mentioned
    category VARCHAR(50),               -- 'injury', 'trade', 'coaching', etc.
    priority INTEGER DEFAULT 3,         -- 1=critical, 5=low

    -- Processing status
    processed BOOLEAN DEFAULT FALSE,    -- Has LLM processed this?
    processed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT unique_content_hash UNIQUE (content_hash)
);

-- Indexes for efficient temporal queries
CREATE INDEX IF NOT EXISTS idx_news_teams
    ON semantic.news_snapshots USING GIN(teams);
CREATE INDEX IF NOT EXISTS idx_news_players
    ON semantic.news_snapshots USING GIN(players);
CREATE INDEX IF NOT EXISTS idx_news_published
    ON semantic.news_snapshots(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source_published
    ON semantic.news_snapshots(source, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_unprocessed
    ON semantic.news_snapshots(processed)
    WHERE NOT processed;
CREATE INDEX IF NOT EXISTS idx_news_category
    ON semantic.news_snapshots(category);

COMMENT ON TABLE semantic.news_snapshots IS
    'Raw news articles from ESPN, RotoWire, and other sources. '
    'content_hash prevents duplicates across sources. '
    'published_at is used for temporal-safe feature extraction.';

-- ============================================================================
-- SENTIMENT SCORES
-- LLM-derived sentiment analysis results
-- ============================================================================
CREATE TABLE IF NOT EXISTS semantic.sentiment_scores (
    id SERIAL PRIMARY KEY,

    -- Link to source news
    news_id INTEGER REFERENCES semantic.news_snapshots(id) ON DELETE CASCADE,

    -- Team-level sentiment (may have multiple per news item)
    team VARCHAR(5) NOT NULL,           -- Team abbreviation

    -- Sentiment values
    sentiment FLOAT,                    -- -1.0 (negative) to +1.0 (positive)
    confidence FLOAT,                   -- 0.0 to 1.0
    impact_score FLOAT,                 -- Expected game impact (0.0 to 1.0)

    -- LLM metadata
    model_used VARCHAR(100),            -- 'qwen3:8b', 'mlx-qwen2.5', etc.
    prompt_version VARCHAR(20),         -- Versioning for prompt changes

    -- Timestamps
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_sentiment CHECK (sentiment >= -1.0 AND sentiment <= 1.0),
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_team
    ON semantic.sentiment_scores(team);
CREATE INDEX IF NOT EXISTS idx_sentiment_processed
    ON semantic.sentiment_scores(processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_news
    ON semantic.sentiment_scores(news_id);

COMMENT ON TABLE semantic.sentiment_scores IS
    'LLM-derived sentiment scores from news analysis. '
    'Each news item may generate multiple sentiment scores (one per team mentioned).';

-- ============================================================================
-- SOURCE METADATA
-- Health tracking for data sources
-- ============================================================================
CREATE TABLE IF NOT EXISTS semantic.source_metadata (
    source_name VARCHAR(50) PRIMARY KEY,

    -- Health tracking
    last_successful_fetch TIMESTAMPTZ,
    last_error TIMESTAMPTZ,
    last_error_message TEXT,
    error_count INTEGER DEFAULT 0,
    consecutive_errors INTEGER DEFAULT 0,
    is_healthy BOOLEAN DEFAULT TRUE,

    -- Statistics
    total_items_fetched BIGINT DEFAULT 0,
    avg_latency_ms FLOAT,

    -- Configuration
    enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 1,         -- Fetch order priority

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE semantic.source_metadata IS
    'Health and status tracking for each data source. '
    'Used to monitor reliability and implement fallback logic.';

-- Insert initial source configurations
INSERT INTO semantic.source_metadata (source_name, priority)
VALUES
    ('espn_api_news', 1),
    ('espn_api_injuries', 1),
    ('rotowire', 2)
ON CONFLICT (source_name) DO NOTHING;

-- ============================================================================
-- INJURY SNAPSHOTS
-- Dedicated table for injury data (frequently queried)
-- ============================================================================
CREATE TABLE IF NOT EXISTS semantic.injury_snapshots (
    id SERIAL PRIMARY KEY,

    -- Source identification
    source VARCHAR(50) NOT NULL,
    source_id VARCHAR(100),
    content_hash VARCHAR(64) NOT NULL,

    -- Player/team info
    athlete_id VARCHAR(100),
    athlete_name VARCHAR(200),
    team VARCHAR(5) NOT NULL,
    position VARCHAR(10),

    -- Injury details
    status VARCHAR(50) NOT NULL,        -- 'Questionable', 'Out', 'IR', etc.
    injury_type VARCHAR(100),           -- 'Knee', 'Ankle', etc.
    injury_location VARCHAR(100),
    fantasy_status VARCHAR(100),

    -- Temporal
    updated_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Raw data
    raw_json JSONB,

    -- Processing
    severity_score FLOAT,               -- LLM-assessed severity (1-5)
    processed BOOLEAN DEFAULT FALSE,

    CONSTRAINT unique_injury_hash UNIQUE (content_hash)
);

CREATE INDEX IF NOT EXISTS idx_injury_team
    ON semantic.injury_snapshots(team);
CREATE INDEX IF NOT EXISTS idx_injury_athlete
    ON semantic.injury_snapshots(athlete_id);
CREATE INDEX IF NOT EXISTS idx_injury_updated
    ON semantic.injury_snapshots(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_injury_status
    ON semantic.injury_snapshots(status);

COMMENT ON TABLE semantic.injury_snapshots IS
    'Injury data from ESPN API and other sources. '
    'Separate from news for efficient team-based queries.';

-- ============================================================================
-- AGGREGATED TEAM SENTIMENT (Materialized View)
-- Pre-computed sentiment by team for fast feature extraction
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS semantic.team_sentiment_24h AS
SELECT
    team,
    DATE_TRUNC('hour', processed_at) AS hour,
    AVG(sentiment) AS avg_sentiment,
    AVG(confidence) AS avg_confidence,
    COUNT(*) AS article_count,
    SUM(CASE WHEN sentiment > 0.2 THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN sentiment < -0.2 THEN 1 ELSE 0 END) AS negative_count
FROM semantic.sentiment_scores
WHERE processed_at >= NOW() - INTERVAL '7 days'
GROUP BY team, DATE_TRUNC('hour', processed_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_team_sentiment_24h_pk
    ON semantic.team_sentiment_24h(team, hour);

COMMENT ON MATERIALIZED VIEW semantic.team_sentiment_24h IS
    'Hourly aggregated sentiment by team for fast lookups. '
    'Refresh periodically: REFRESH MATERIALIZED VIEW CONCURRENTLY semantic.team_sentiment_24h';

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get team sentiment before a specific time (temporal safety)
CREATE OR REPLACE FUNCTION semantic.get_team_sentiment_before(
    p_team VARCHAR(5),
    p_before TIMESTAMPTZ,
    p_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    avg_sentiment FLOAT,
    article_count BIGINT,
    latest_article TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(s.sentiment)::FLOAT AS avg_sentiment,
        COUNT(*)::BIGINT AS article_count,
        MAX(n.published_at) AS latest_article
    FROM semantic.sentiment_scores s
    JOIN semantic.news_snapshots n ON s.news_id = n.id
    WHERE s.team = p_team
      AND n.published_at < p_before
      AND n.published_at >= p_before - make_interval(hours => p_hours);
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION semantic.get_team_sentiment_before IS
    'Get team sentiment using only news published BEFORE the specified time. '
    'Critical for avoiding data leakage in training/prediction.';

-- Grant permissions
GRANT USAGE ON SCHEMA semantic TO dro;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA semantic TO dro;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA semantic TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA semantic TO dro;
