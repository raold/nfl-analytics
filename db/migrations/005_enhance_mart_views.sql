-- Migration: Enhance mart views with new game metadata columns
-- Date: 2025-01-XX
-- Purpose: Add stadium, QB, coach, turnover, penalty data to mart.game_summary

-- Drop and recreate mart.game_summary with enhanced columns
DROP MATERIALIZED VIEW IF EXISTS mart.game_summary CASCADE;

CREATE MATERIALIZED VIEW mart.game_summary AS
SELECT
  g.game_id,
  g.season,
  g.week,
  g.home_team,
  g.away_team,
  g.home_score,
  g.away_score,
  g.spread_close,
  g.total_close,
  g.home_moneyline,
  g.away_moneyline,
  
  -- EPA metrics (from team_epa)
  hepa.epa_mean as home_epa_mean,
  hepa.plays as home_plays,
  aepa.epa_mean as away_epa_mean,
  aepa.plays as away_plays,
  
  -- Stadium/venue metadata (NEW)
  g.stadium,
  g.roof,
  g.surface,
  g.temp,
  g.wind,
  
  -- QB identity (NEW)
  g.home_qb_name,
  g.away_qb_name,
  
  -- Coaching (NEW)
  g.home_coach,
  g.away_coach,
  g.referee,
  
  -- Game context (NEW)
  g.game_type,
  g.overtime,
  g.away_rest,
  g.home_rest,
  
  -- Calculated stats (NEW)
  g.home_turnovers,
  g.away_turnovers,
  g.home_penalties,
  g.away_penalties,
  g.home_penalty_yards,
  g.away_penalty_yards,
  
  -- Derived metrics
  (g.home_score + g.away_score) as total_points,
  (g.home_score - g.away_score) as home_margin,
  CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win,
  CASE WHEN g.home_score + g.spread_close > g.away_score THEN 1 ELSE 0 END as home_cover,
  CASE WHEN (g.home_score + g.away_score) > g.total_close THEN 1 ELSE 0 END as over_hit,
  
  -- Turnover differential (NEW derived)
  (g.home_turnovers - g.away_turnovers) as turnover_diff,
  
  -- Penalty differential (NEW derived)
  (g.home_penalties - g.away_penalties) as penalty_diff,
  (g.home_penalty_yards - g.away_penalty_yards) as penalty_yard_diff
  
FROM games g
LEFT JOIN mart.team_epa hepa ON g.game_id = hepa.game_id AND g.home_team = hepa.posteam
LEFT JOIN mart.team_epa aepa ON g.game_id = aepa.game_id AND g.away_team = aepa.posteam;

-- Recreate index
CREATE INDEX IF NOT EXISTS mart_game_summary_idx ON mart.game_summary (season, week);
CREATE INDEX IF NOT EXISTS mart_game_summary_team_idx ON mart.game_summary (home_team, season);

-- Verify new columns
SELECT 
  'mart.game_summary' as view_name,
  COUNT(*) as column_count
FROM information_schema.columns 
WHERE table_schema='mart' AND table_name='game_summary';

-- Verify row count
SELECT 
  'mart.game_summary' as view_name,
  COUNT(*) as row_count
FROM mart.game_summary;

-- Sample verification: Show 2023 Super Bowl with enhanced data
SELECT 
  game_id,
  season,
  week,
  home_team,
  away_team,
  home_qb_name,
  away_qb_name,
  home_coach,
  away_coach,
  stadium,
  roof,
  surface,
  home_score,
  away_score,
  home_turnovers,
  away_turnovers,
  turnover_diff
FROM mart.game_summary
WHERE game_id = '2023_22_SF_KC';
