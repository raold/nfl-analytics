-- Migration 015: DBA Audit Log Table
-- Stores historical audit results for trend analysis and alerting

CREATE TABLE IF NOT EXISTS dba_audit_log (
  audit_id SERIAL PRIMARY KEY,
  audit_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  audit_type TEXT NOT NULL, -- 'database_overview', 'referential_integrity', 'data_quality', etc.
  table_name TEXT, -- NULL for database-wide checks
  check_name TEXT NOT NULL, -- Specific check identifier
  status TEXT NOT NULL CHECK (status IN ('PASS', 'WARNING', 'FAIL')),
  violation_count INT,
  message TEXT,
  metadata JSONB, -- Flexible storage for check-specific details
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX idx_dba_audit_log_timestamp ON dba_audit_log (audit_timestamp DESC);
CREATE INDEX idx_dba_audit_log_table ON dba_audit_log (table_name, check_name) WHERE table_name IS NOT NULL;
CREATE INDEX idx_dba_audit_log_status ON dba_audit_log (status, audit_timestamp DESC) WHERE status IN ('WARNING', 'FAIL');
CREATE INDEX idx_dba_audit_log_check ON dba_audit_log (check_name, audit_timestamp DESC);

-- Create a view for recent audit summary
CREATE OR REPLACE VIEW v_dba_audit_summary AS
SELECT
  check_name,
  table_name,
  status,
  COUNT(*) as occurrence_count,
  MAX(audit_timestamp) as last_occurred,
  AVG(violation_count) FILTER (WHERE violation_count IS NOT NULL) as avg_violations,
  MAX(violation_count) FILTER (WHERE violation_count IS NOT NULL) as max_violations
FROM dba_audit_log
WHERE audit_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY check_name, table_name, status
ORDER BY
  CASE status
    WHEN 'FAIL' THEN 1
    WHEN 'WARNING' THEN 2
    WHEN 'PASS' THEN 3
  END,
  last_occurred DESC;

COMMENT ON TABLE dba_audit_log IS
'Historical log of DBA audit checks. Used for trend analysis, alerting, and monitoring database health over time.';

COMMENT ON VIEW v_dba_audit_summary IS
'Summary of audit checks from the last 30 days, ordered by severity and recency.';
