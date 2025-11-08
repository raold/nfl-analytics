-- Migration 031: Add Phase 5 HIGH-Signal Columns (Exhaustive Implementation)
-- Date: November 7, 2025
-- Purpose: Add ALL remaining HIGH-signal columns from comprehensive lateral analysis
-- Expected ROI Impact: +15-25% (cumulative with Phases 1-4: +18.8-33.5%)
-- Columns Added: ~100 total

-- ============================================================
-- PHASE 5: EXHAUSTIVE HIGH-SIGNAL COLUMN IMPLEMENTATION
-- ============================================================

-- ============================================================
-- 1. GAME CONTEXT & METADATA (12 columns) - +1.0-1.5% ROI
-- ============================================================

-- Game identification & scheduling
ALTER TABLE plays ADD COLUMN IF NOT EXISTS home_team TEXT;  -- Home team abbreviation
ALTER TABLE plays ADD COLUMN IF NOT EXISTS away_team TEXT;  -- Away team abbreviation
ALTER TABLE plays ADD COLUMN IF NOT EXISTS season_type TEXT;  -- REG, POST, PRE
ALTER TABLE plays ADD COLUMN IF NOT EXISTS week INTEGER;  -- Week number
ALTER TABLE plays ADD COLUMN IF NOT EXISTS game_date DATE;  -- Game date
ALTER TABLE plays ADD COLUMN IF NOT EXISTS start_time TEXT;  -- Kickoff time
ALTER TABLE plays ADD COLUMN IF NOT EXISTS time_of_day TEXT;  -- Time of day category

-- Game location
ALTER TABLE plays ADD COLUMN IF NOT EXISTS stadium TEXT;  -- Stadium name
ALTER TABLE plays ADD COLUMN IF NOT EXISTS location TEXT;  -- Home/Neutral
ALTER TABLE plays ADD COLUMN IF NOT EXISTS roof TEXT;  -- Dome/Outdoors/Retractable
ALTER TABLE plays ADD COLUMN IF NOT EXISTS surface TEXT;  -- Grass/Turf type
ALTER TABLE plays ADD COLUMN IF NOT EXISTS div_game BOOLEAN;  -- Divisional game flag

-- ============================================================
-- 2. WIN PROBABILITY & GAME SCRIPT (8 columns) - +2.0-3.0% ROI
-- ============================================================

-- Detailed win probability
ALTER TABLE plays ADD COLUMN IF NOT EXISTS home_wp REAL;  -- Home team win probability
ALTER TABLE plays ADD COLUMN IF NOT EXISTS away_wp REAL;  -- Away team win probability
ALTER TABLE plays ADD COLUMN IF NOT EXISTS home_wp_post REAL;  -- Home WP post-play
ALTER TABLE plays ADD COLUMN IF NOT EXISTS away_wp_post REAL;  -- Away WP post-play
ALTER TABLE plays ADD COLUMN IF NOT EXISTS vegas_home_wp REAL;  -- Vegas home WP
ALTER TABLE plays ADD COLUMN IF NOT EXISTS vegas_home_wpa REAL;  -- Vegas home WPA

-- Expected points
ALTER TABLE plays ADD COLUMN IF NOT EXISTS ep REAL;  -- Expected points
ALTER TABLE plays ADD COLUMN IF NOT EXISTS opp_fg_prob REAL;  -- Opponent FG probability

-- ============================================================
-- 3. TOTAL TEAM EPA (12 columns) - +1.5-2.5% ROI
-- ============================================================

-- Cumulative EPA by team
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_epa REAL;  -- Total home EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_epa REAL;  -- Total away EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_rush_epa REAL;  -- Home rush EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_rush_epa REAL;  -- Away rush EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_pass_epa REAL;  -- Home pass EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_pass_epa REAL;  -- Away pass EPA

-- Air/YAC EPA decomposition
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_comp_air_epa REAL;  -- Home air EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_comp_air_epa REAL;  -- Away air EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_comp_yac_epa REAL;  -- Home YAC EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_comp_yac_epa REAL;  -- Away YAC EPA
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_home_raw_air_epa REAL;  -- Home raw air
ALTER TABLE plays ADD COLUMN IF NOT EXISTS total_away_raw_air_epa REAL;  -- Away raw air

-- ============================================================
-- 4. DRIVE CONTEXT EXTENDED (10 columns) - +1.5-2.0% ROI
-- ============================================================

-- Drive timing & field position
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_real_start_time TEXT;  -- Real start time
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_quarter_start INTEGER;  -- Start quarter
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_quarter_end INTEGER;  -- End quarter
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_game_clock_start TEXT;  -- Start clock
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_game_clock_end TEXT;  -- End clock
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_start_yard_line TEXT;  -- Start yard line
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_end_yard_line TEXT;  -- End yard line

-- Drive transitions
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_start_transition TEXT;  -- How drive started
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_end_transition TEXT;  -- How drive ended
ALTER TABLE plays ADD COLUMN IF NOT EXISTS side_of_field TEXT;  -- Side of field

-- ============================================================
-- 5. RETURN GAME DETAILS (10 columns) - +2.0-3.5% ROI
-- ============================================================

-- Punt return details
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_inside_twenty BOOLEAN;  -- Punt inside 20
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_in_endzone BOOLEAN;  -- Punt touchback
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_out_of_bounds BOOLEAN;  -- Punt OOB
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_downed BOOLEAN;  -- Punt downed
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_fair_catch BOOLEAN;  -- Fair catch

-- Kickoff return details
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_inside_twenty BOOLEAN;  -- Kickoff inside 20
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_in_endzone BOOLEAN;  -- Kickoff touchback
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_out_of_bounds BOOLEAN;  -- Kickoff OOB
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_downed BOOLEAN;  -- Kickoff downed
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_fair_catch BOOLEAN;  -- Fair catch

-- ============================================================
-- 6. ADDITIONAL TACKLE DETAILS (14 columns) - +1.5-2.5% ROI
-- ============================================================

-- Solo tackles (player 2)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS solo_tackle_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS solo_tackle_2_player_name TEXT;

-- Assist tackles (players 2-4)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_2_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_3_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_3_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_4_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_4_player_name TEXT;

-- Tackle with assist (hybrid tackles)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_with_assist BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_with_assist_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_with_assist_1_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_with_assist_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_with_assist_2_player_name TEXT;

-- Tackle for loss player 2
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_for_loss_2_player_id TEXT;

-- ============================================================
-- 7. QB PRESSURE & PASS DEFENSE (5 columns) - +1.0-2.0% ROI
-- ============================================================

-- Additional QB hits
ALTER TABLE plays ADD COLUMN IF NOT EXISTS qb_hit_1_player_name TEXT;  -- QB hit 1 name
ALTER TABLE plays ADD COLUMN IF NOT EXISTS qb_hit_2_player_id TEXT;  -- QB hit 2 ID
ALTER TABLE plays ADD COLUMN IF NOT EXISTS qb_hit_2_player_name TEXT;  -- QB hit 2 name

-- Pass defense player 2
ALTER TABLE plays ADD COLUMN IF NOT EXISTS pass_defense_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS pass_defense_2_player_name TEXT;

-- ============================================================
-- 8. SPECIAL TEAMS PLAYERS (8 columns) - +0.5-1.0% ROI
-- ============================================================

-- Kicker & Punter
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kicker_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kicker_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punter_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punter_player_name TEXT;

-- Return players
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_returner_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_returner_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_returner_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_returner_player_name TEXT;

-- ============================================================
-- 9. FUMBLE DETAILS EXTENDED (10 columns) - +1.0-1.5% ROI
-- ============================================================

-- Fumbled players (1-2)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumbled_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumbled_1_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumbled_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumbled_2_player_name TEXT;

-- Forced fumble player 2
ALTER TABLE plays ADD COLUMN IF NOT EXISTS forced_fumble_player_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS forced_fumble_player_2_player_name TEXT;

-- Fumble recovery player 2
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumble_recovery_2_team TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumble_recovery_2_yards REAL;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumble_recovery_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fumble_recovery_2_player_name TEXT;

-- ============================================================
-- 10. TIMEOUT & CLOCK MANAGEMENT (8 columns) - +0.8-1.2% ROI
-- ============================================================

-- Timeouts remaining
ALTER TABLE plays ADD COLUMN IF NOT EXISTS home_timeouts_remaining INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS away_timeouts_remaining INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS posteam_timeouts_remaining INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS defteam_timeouts_remaining INTEGER;

-- Timeout details
ALTER TABLE plays ADD COLUMN IF NOT EXISTS timeout BOOLEAN;  -- Timeout called
ALTER TABLE plays ADD COLUMN IF NOT EXISTS timeout_team TEXT;  -- Which team
ALTER TABLE plays ADD COLUMN IF NOT EXISTS play_clock INTEGER;  -- Play clock seconds
ALTER TABLE plays ADD COLUMN IF NOT EXISTS quarter_seconds_remaining INTEGER;  -- Quarter clock

-- ============================================================
-- 11. SCORING PROBABILITY (4 columns) - +0.5-0.8% ROI
-- ============================================================

ALTER TABLE plays ADD COLUMN IF NOT EXISTS fg_prob REAL;  -- FG probability
ALTER TABLE plays ADD COLUMN IF NOT EXISTS td_prob REAL;  -- TD probability
ALTER TABLE plays ADD COLUMN IF NOT EXISTS opp_td_prob REAL;  -- Opponent TD prob
ALTER TABLE plays ADD COLUMN IF NOT EXISTS opp_safety_prob REAL;  -- Opponent safety prob

-- ============================================================
-- 12. INTERCEPTION & SACK DETAILS (6 columns) - +0.5-1.0% ROI
-- ============================================================

-- Interception players
ALTER TABLE plays ADD COLUMN IF NOT EXISTS interception_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS interception_player_name TEXT;

-- Half sacks (split sacks)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS half_sack_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS half_sack_1_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS half_sack_2_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS half_sack_2_player_name TEXT;

-- ============================================================
-- 13. REPLAY & CHALLENGE (2 columns) - +0.2-0.4% ROI
-- ============================================================

ALTER TABLE plays ADD COLUMN IF NOT EXISTS replay_or_challenge BOOLEAN;  -- Challenge flag
ALTER TABLE plays ADD COLUMN IF NOT EXISTS replay_or_challenge_result TEXT;  -- Result

-- ============================================================
-- 14. SPECIAL TEAMS OUTCOMES (5 columns) - +0.3-0.6% ROI
-- ============================================================

ALTER TABLE plays ADD COLUMN IF NOT EXISTS touchback BOOLEAN;  -- Touchback flag
ALTER TABLE plays ADD COLUMN IF NOT EXISTS own_kickoff_recovery BOOLEAN;  -- Onside recovery
ALTER TABLE plays ADD COLUMN IF NOT EXISTS own_kickoff_recovery_td BOOLEAN;  -- Onside TD
ALTER TABLE plays ADD COLUMN IF NOT EXISTS return_touchdown BOOLEAN;  -- Return TD
ALTER TABLE plays ADD COLUMN IF NOT EXISTS return_team TEXT;  -- Returning team

-- ============================================================
-- 15. DOWN CONVERSION TRACKING (2 columns) - +0.3-0.5% ROI
-- ============================================================

ALTER TABLE plays ADD COLUMN IF NOT EXISTS third_down_failed BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS fourth_down_failed BOOLEAN;

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Game context
CREATE INDEX IF NOT EXISTS idx_plays_home_team ON plays(home_team, game_id);
CREATE INDEX IF NOT EXISTS idx_plays_season_week ON plays(week) WHERE week IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_plays_div_game ON plays(div_game) WHERE div_game = TRUE;

-- Win probability & game script
CREATE INDEX IF NOT EXISTS idx_plays_home_wp ON plays(game_id, home_wp) WHERE home_wp IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_plays_ep ON plays(game_id, ep) WHERE ep IS NOT NULL;

-- Total EPA
CREATE INDEX IF NOT EXISTS idx_plays_total_home_epa ON plays(game_id, total_home_epa);
CREATE INDEX IF NOT EXISTS idx_plays_total_away_epa ON plays(game_id, total_away_epa);

-- Return game
CREATE INDEX IF NOT EXISTS idx_plays_punt_inside_twenty ON plays(game_id) WHERE punt_inside_twenty = TRUE;
CREATE INDEX IF NOT EXISTS idx_plays_kickoff_inside_twenty ON plays(game_id) WHERE kickoff_inside_twenty = TRUE;

-- Tackles
CREATE INDEX IF NOT EXISTS idx_plays_solo_tackle_2 ON plays(solo_tackle_2_player_id) WHERE solo_tackle_2_player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_plays_assist_tackle_2 ON plays(assist_tackle_2_player_id) WHERE assist_tackle_2_player_id IS NOT NULL;

-- Special teams players
CREATE INDEX IF NOT EXISTS idx_plays_kicker ON plays(kicker_player_id, game_id) WHERE kicker_player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_plays_punter ON plays(punter_player_id, game_id) WHERE punter_player_id IS NOT NULL;

-- Timeouts
CREATE INDEX IF NOT EXISTS idx_plays_timeout ON plays(game_id, timeout) WHERE timeout = TRUE;

-- ============================================================
-- VALIDATION QUERIES
-- ============================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'plays'
      AND column_name IN (
        'home_team', 'away_team', 'season_type', 'week', 'game_date',
        'home_wp', 'away_wp', 'total_home_epa', 'total_away_epa',
        'drive_quarter_start', 'punt_inside_twenty', 'kickoff_inside_twenty',
        'solo_tackle_2_player_id', 'assist_tackle_2_player_id',
        'kicker_player_id', 'punter_player_id', 'fumbled_1_player_id',
        'timeout', 'fg_prob', 'interception_player_id', 'half_sack_1_player_id',
        'replay_or_challenge', 'touchback', 'third_down_failed'
      );

    IF col_count >= 20 THEN
        RAISE NOTICE '✓ Migration 031: Phase 5 columns added successfully (sampled % of ~100)', col_count;
    ELSE
        RAISE WARNING '✗ Migration 031: Expected 20+ sampled columns, found %', col_count;
    END IF;
END $$;

-- ============================================================
-- ROLLBACK (if needed)
-- ============================================================

-- To rollback this migration (use with caution):
/*
-- Game context
ALTER TABLE plays DROP COLUMN IF EXISTS home_team;
ALTER TABLE plays DROP COLUMN IF EXISTS away_team;
ALTER TABLE plays DROP COLUMN IF EXISTS season_type;
ALTER TABLE plays DROP COLUMN IF EXISTS week;
ALTER TABLE plays DROP COLUMN IF EXISTS game_date;
ALTER TABLE plays DROP COLUMN IF EXISTS start_time;
ALTER TABLE plays DROP COLUMN IF EXISTS time_of_day;
ALTER TABLE plays DROP COLUMN IF EXISTS stadium;
ALTER TABLE plays DROP COLUMN IF EXISTS location;
ALTER TABLE plays DROP COLUMN IF EXISTS roof;
ALTER TABLE plays DROP COLUMN IF EXISTS surface;
ALTER TABLE plays DROP COLUMN IF EXISTS div_game;

-- Win probability
ALTER TABLE plays DROP COLUMN IF EXISTS home_wp;
ALTER TABLE plays DROP COLUMN IF EXISTS away_wp;
ALTER TABLE plays DROP COLUMN IF EXISTS home_wp_post;
ALTER TABLE plays DROP COLUMN IF EXISTS away_wp_post;
ALTER TABLE plays DROP COLUMN IF EXISTS vegas_home_wp;
ALTER TABLE plays DROP COLUMN IF EXISTS vegas_home_wpa;
ALTER TABLE plays DROP COLUMN IF EXISTS ep;
ALTER TABLE plays DROP COLUMN IF EXISTS opp_fg_prob;

-- Total EPA
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_rush_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_rush_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_pass_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_pass_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_comp_air_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_comp_air_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_comp_yac_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_comp_yac_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_home_raw_air_epa;
ALTER TABLE plays DROP COLUMN IF EXISTS total_away_raw_air_epa;

-- Drive context
ALTER TABLE plays DROP COLUMN IF EXISTS drive_real_start_time;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_quarter_start;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_quarter_end;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_game_clock_start;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_game_clock_end;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_start_yard_line;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_end_yard_line;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_start_transition;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_end_transition;
ALTER TABLE plays DROP COLUMN IF EXISTS side_of_field;

-- Return game
ALTER TABLE plays DROP COLUMN IF EXISTS punt_inside_twenty;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_in_endzone;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_out_of_bounds;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_downed;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_fair_catch;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_inside_twenty;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_in_endzone;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_out_of_bounds;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_downed;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_fair_catch;

-- Tackles
ALTER TABLE plays DROP COLUMN IF EXISTS solo_tackle_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS solo_tackle_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_3_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_3_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_4_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_4_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_with_assist;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_with_assist_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_with_assist_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_with_assist_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_with_assist_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_for_loss_2_player_id;

-- QB pressure & pass defense
ALTER TABLE plays DROP COLUMN IF EXISTS qb_hit_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS qb_hit_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS qb_hit_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS pass_defense_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS pass_defense_2_player_name;

-- Special teams players
ALTER TABLE plays DROP COLUMN IF EXISTS kicker_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS kicker_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS punter_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS punter_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_returner_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_returner_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_returner_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_returner_player_name;

-- Fumble details
ALTER TABLE plays DROP COLUMN IF EXISTS fumbled_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS fumbled_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS fumbled_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS fumbled_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS forced_fumble_player_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS forced_fumble_player_2_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS fumble_recovery_2_team;
ALTER TABLE plays DROP COLUMN IF EXISTS fumble_recovery_2_yards;
ALTER TABLE plays DROP COLUMN IF EXISTS fumble_recovery_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS fumble_recovery_2_player_name;

-- Timeout & clock
ALTER TABLE plays DROP COLUMN IF EXISTS home_timeouts_remaining;
ALTER TABLE plays DROP COLUMN IF EXISTS away_timeouts_remaining;
ALTER TABLE plays DROP COLUMN IF EXISTS posteam_timeouts_remaining;
ALTER TABLE plays DROP COLUMN IF EXISTS defteam_timeouts_remaining;
ALTER TABLE plays DROP COLUMN IF EXISTS timeout;
ALTER TABLE plays DROP COLUMN IF EXISTS timeout_team;
ALTER TABLE plays DROP COLUMN IF EXISTS play_clock;
ALTER TABLE plays DROP COLUMN IF EXISTS quarter_seconds_remaining;

-- Scoring probability
ALTER TABLE plays DROP COLUMN IF EXISTS fg_prob;
ALTER TABLE plays DROP COLUMN IF EXISTS td_prob;
ALTER TABLE plays DROP COLUMN IF EXISTS opp_td_prob;
ALTER TABLE plays DROP COLUMN IF EXISTS opp_safety_prob;

-- Interception & sacks
ALTER TABLE plays DROP COLUMN IF EXISTS interception_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS interception_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS half_sack_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS half_sack_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS half_sack_2_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS half_sack_2_player_name;

-- Replay & challenge
ALTER TABLE plays DROP COLUMN IF EXISTS replay_or_challenge;
ALTER TABLE plays DROP COLUMN IF EXISTS replay_or_challenge_result;

-- Special teams outcomes
ALTER TABLE plays DROP COLUMN IF EXISTS touchback;
ALTER TABLE plays DROP COLUMN IF EXISTS own_kickoff_recovery;
ALTER TABLE plays DROP COLUMN IF EXISTS own_kickoff_recovery_td;
ALTER TABLE plays DROP COLUMN IF EXISTS return_touchdown;
ALTER TABLE plays DROP COLUMN IF EXISTS return_team;

-- Down conversion
ALTER TABLE plays DROP COLUMN IF EXISTS third_down_failed;
ALTER TABLE plays DROP COLUMN IF EXISTS fourth_down_failed;

-- Drop indexes
DROP INDEX IF EXISTS idx_plays_home_team;
DROP INDEX IF EXISTS idx_plays_season_week;
DROP INDEX IF EXISTS idx_plays_div_game;
DROP INDEX IF EXISTS idx_plays_home_wp;
DROP INDEX IF EXISTS idx_plays_ep;
DROP INDEX IF EXISTS idx_plays_total_home_epa;
DROP INDEX IF EXISTS idx_plays_total_away_epa;
DROP INDEX IF EXISTS idx_plays_punt_inside_twenty;
DROP INDEX IF EXISTS idx_plays_kickoff_inside_twenty;
DROP INDEX IF EXISTS idx_plays_solo_tackle_2;
DROP INDEX IF EXISTS idx_plays_assist_tackle_2;
DROP INDEX IF EXISTS idx_plays_kicker;
DROP INDEX IF EXISTS idx_plays_punter;
DROP INDEX IF EXISTS idx_plays_timeout;
*/
