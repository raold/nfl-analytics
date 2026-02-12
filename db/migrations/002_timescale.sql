-- Enable TimescaleDB and convert odds_history into a hypertable with policies

create extension if not exists timescaledb;

-- Create hypertable on odds_history over snapshot_at (if not already)
select create_hypertable('odds_history', 'snapshot_at', if_not_exists => true, migrate_data => true);

-- Enable compression and set segment-by to common query dimensions
alter table if exists odds_history
  set (
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'bookmaker_key,market_key'
  );

-- Add compression policy for data older than 30 days
do $$
begin
  perform add_compression_policy('odds_history', interval '30 days');
exception when others then
  -- ignore if policy already exists
  null;
end$$;

-- Add retention policy to keep 5 years of odds history
do $$
begin
  perform add_retention_policy('odds_history', interval '5 years');
exception when others then
  null;
end$$;
