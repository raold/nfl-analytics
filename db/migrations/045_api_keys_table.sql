-- Migration 045: API Keys Table
-- Purpose: Store API keys for authentication
-- Author: Claude Code
-- Date: 2025-11-15

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS auth;

-- API keys table
CREATE TABLE IF NOT EXISTS auth.api_keys (
    id SERIAL PRIMARY KEY,

    -- Key identification
    api_key VARCHAR(64) NOT NULL UNIQUE,  -- UUID4 + hash for uniqueness
    key_name VARCHAR(255) NOT NULL,  -- Descriptive name (e.g., "Production Dashboard")
    key_prefix VARCHAR(8) NOT NULL,  -- First 8 chars for identification

    -- Ownership
    owner_email VARCHAR(255) NOT NULL,
    owner_name VARCHAR(255),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason TEXT,

    -- Rate limiting
    rate_limit_per_hour INTEGER NOT NULL DEFAULT 100,
    rate_limit_per_day INTEGER NOT NULL DEFAULT 1000,

    -- Permissions
    allowed_endpoints TEXT[] DEFAULT ARRAY['/predict', '/models', '/health'],  -- NULL = all endpoints
    allowed_ip_ranges TEXT[],  -- CIDR notation (e.g., '10.0.0.0/8')

    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    total_requests BIGINT NOT NULL DEFAULT 0,

    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,  -- NULL = never expires

    -- Rotation tracking
    rotated_from_key_id INTEGER REFERENCES auth.api_keys(id),
    rotation_reason TEXT,

    -- Security
    CONSTRAINT valid_rate_limits CHECK (
        rate_limit_per_hour > 0 AND
        rate_limit_per_day >= rate_limit_per_hour
    ),
    CONSTRAINT valid_expiry CHECK (
        expires_at IS NULL OR expires_at > created_at
    )
);

-- Indexes for performance
CREATE INDEX idx_api_keys_active ON auth.api_keys(api_key) WHERE is_active = TRUE AND is_revoked = FALSE;
CREATE INDEX idx_api_keys_owner ON auth.api_keys(owner_email);
CREATE INDEX idx_api_keys_last_used ON auth.api_keys(last_used_at DESC);
CREATE INDEX idx_api_keys_prefix ON auth.api_keys(key_prefix);

-- API key usage log (for audit trail and rate limiting)
CREATE TABLE IF NOT EXISTS auth.api_key_usage (
    id BIGSERIAL PRIMARY KEY,

    -- Request identification
    api_key_id INTEGER NOT NULL REFERENCES auth.api_keys(id) ON DELETE CASCADE,
    request_id UUID NOT NULL,  -- For correlation with application logs

    -- Request details
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms FLOAT,

    -- Client information
    ip_address INET,
    user_agent TEXT,

    -- Error tracking
    error_type VARCHAR(100),
    error_message TEXT,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for usage tracking and rate limiting
CREATE INDEX idx_api_key_usage_key_time ON auth.api_key_usage(api_key_id, created_at DESC);
CREATE INDEX idx_api_key_usage_endpoint ON auth.api_key_usage(endpoint, created_at DESC);
CREATE INDEX idx_api_key_usage_status ON auth.api_key_usage(status_code, created_at DESC);

-- Partition by month for scalability (commented out for initial deployment)
-- CREATE TABLE auth.api_key_usage_template (LIKE auth.api_key_usage INCLUDING ALL);
-- ALTER TABLE auth.api_key_usage PARTITION BY RANGE (created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION auth.update_api_key_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_api_key_timestamp
    BEFORE UPDATE ON auth.api_keys
    FOR EACH ROW
    EXECUTE FUNCTION auth.update_api_key_timestamp();

-- Function to log API key usage
CREATE OR REPLACE FUNCTION auth.log_api_key_usage(
    p_api_key_id INTEGER,
    p_request_id UUID,
    p_endpoint VARCHAR,
    p_method VARCHAR,
    p_status_code INTEGER,
    p_response_time_ms FLOAT,
    p_ip_address INET,
    p_user_agent TEXT,
    p_error_type VARCHAR DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    -- Insert usage record
    INSERT INTO auth.api_key_usage (
        api_key_id, request_id, endpoint, method, status_code,
        response_time_ms, ip_address, user_agent, error_type, error_message
    ) VALUES (
        p_api_key_id, p_request_id, p_endpoint, p_method, p_status_code,
        p_response_time_ms, p_ip_address, p_user_agent, p_error_type, p_error_message
    );

    -- Update last_used_at and total_requests on api_keys table
    UPDATE auth.api_keys
    SET
        last_used_at = NOW(),
        total_requests = total_requests + 1
    WHERE id = p_api_key_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check rate limit
CREATE OR REPLACE FUNCTION auth.check_rate_limit(
    p_api_key_id INTEGER,
    p_window_hours INTEGER DEFAULT 1
) RETURNS TABLE (
    requests_in_window BIGINT,
    limit_per_window INTEGER,
    is_limited BOOLEAN
) AS $$
DECLARE
    v_limit INTEGER;
    v_count BIGINT;
BEGIN
    -- Get rate limit for this key
    SELECT
        CASE
            WHEN p_window_hours = 1 THEN rate_limit_per_hour
            WHEN p_window_hours = 24 THEN rate_limit_per_day
            ELSE rate_limit_per_hour
        END INTO v_limit
    FROM auth.api_keys
    WHERE id = p_api_key_id;

    -- Count requests in window
    SELECT COUNT(*) INTO v_count
    FROM auth.api_key_usage
    WHERE
        api_key_id = p_api_key_id AND
        created_at >= NOW() - (p_window_hours || ' hours')::INTERVAL AND
        status_code < 500;  -- Don't count server errors against limit

    RETURN QUERY SELECT v_count, v_limit, v_count >= v_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to revoke API key
CREATE OR REPLACE FUNCTION auth.revoke_api_key(
    p_api_key VARCHAR,
    p_reason TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    v_updated INTEGER;
BEGIN
    UPDATE auth.api_keys
    SET
        is_revoked = TRUE,
        is_active = FALSE,
        revoked_at = NOW(),
        revoked_reason = p_reason
    WHERE api_key = p_api_key AND is_revoked = FALSE;

    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RETURN v_updated > 0;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust based on your user setup)
GRANT USAGE ON SCHEMA auth TO dro;
GRANT SELECT, INSERT, UPDATE ON auth.api_keys TO dro;
GRANT SELECT, INSERT ON auth.api_key_usage TO dro;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA auth TO dro;

-- Insert example API key for development (NEVER use in production)
-- Key: dev_test_key_12345678901234567890123456789012
INSERT INTO auth.api_keys (
    api_key,
    key_name,
    key_prefix,
    owner_email,
    owner_name,
    rate_limit_per_hour,
    rate_limit_per_day,
    allowed_endpoints
) VALUES (
    'dev_test_key_12345678901234567890123456789012',
    'Development Test Key',
    'dev_test',
    'dev@nfl-analytics.local',
    'Development Team',
    1000,  -- Higher limit for testing
    10000,
    NULL  -- Allow all endpoints
) ON CONFLICT (api_key) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE auth.api_keys IS 'API keys for authentication and authorization';
COMMENT ON TABLE auth.api_key_usage IS 'Audit log for API key usage and rate limiting';
COMMENT ON FUNCTION auth.log_api_key_usage IS 'Log API request and update key usage statistics';
COMMENT ON FUNCTION auth.check_rate_limit IS 'Check if API key has exceeded rate limit in given window';
COMMENT ON FUNCTION auth.revoke_api_key IS 'Revoke API key with reason';

-- Verification query
SELECT
    'Migration 045 complete' AS message,
    COUNT(*) AS api_keys_created
FROM auth.api_keys;
