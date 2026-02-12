-- Migration 016: Unified Player ID Mapping View
-- Addresses DBA audit recommendation for standardized player ID access

-- Drop existing view if it exists
DROP VIEW IF EXISTS player_id_mapping CASCADE;

-- Unified Player ID Mapping View
-- Provides a single source of truth for player ID mapping across systems
CREATE OR REPLACE VIEW player_id_mapping AS
WITH player_ids AS (
  -- Base player IDs from players table
  SELECT
    player_id,
    player_name,
    position,
    NULL::TEXT as gsis_id,
    NULL::TEXT as pfr_id,
    NULL::TEXT as espn_id,
    NULL::TEXT as yahoo_id,
    NULL::TEXT as sleeper_id,
    NULL::TEXT as pff_id,
    NULL::TEXT as rotowire_id,
    NULL::TEXT as fantasy_data_id,
    NULL::TEXT as sportradar_id,
    'players' as source_table
  FROM players
  WHERE player_id IS NOT NULL

  UNION ALL

  -- IDs from rosters_weekly (most comprehensive ID mapping source)
  SELECT
    NULL as player_id,
    full_name as player_name,
    position,
    gsis_id,
    pfr_id,
    espn_id,
    yahoo_id,
    sleeper_id,
    pff_id,
    rotowire_id,
    fantasy_data_id,
    sportradar_id,
    'rosters_weekly' as source_table
  FROM rosters_weekly
  WHERE gsis_id IS NOT NULL

  UNION ALL

  -- IDs from contracts
  SELECT
    NULL as player_id,
    player as player_name,
    position,
    gsis_id,
    NULL as pfr_id,
    NULL as espn_id,
    NULL as yahoo_id,
    NULL as sleeper_id,
    NULL as pff_id,
    NULL as rotowire_id,
    NULL as fantasy_data_id,
    NULL as sportradar_id,
    'contracts' as source_table
  FROM contracts
  WHERE gsis_id IS NOT NULL

  UNION ALL

  -- PFR IDs from combine
  SELECT
    NULL as player_id,
    player_name,
    pos as position,
    NULL as gsis_id,
    pfr_id,
    NULL as espn_id,
    NULL as yahoo_id,
    NULL as sleeper_id,
    NULL as pff_id,
    NULL as rotowire_id,
    NULL as fantasy_data_id,
    NULL as sportradar_id,
    'combine' as source_table
  FROM combine
  WHERE pfr_id IS NOT NULL

  UNION ALL

  -- Draft picks with multiple ID types
  SELECT
    NULL as player_id,
    pfr_player_name as player_name,
    position,
    gsis_id,
    pfr_player_id as pfr_id,
    NULL as espn_id,
    NULL as yahoo_id,
    NULL as sleeper_id,
    NULL as pff_id,
    NULL as rotowire_id,
    NULL as fantasy_data_id,
    NULL as sportradar_id,
    'draft_picks' as source_table
  FROM draft_picks
  WHERE gsis_id IS NOT NULL OR pfr_player_id IS NOT NULL
),
aggregated AS (
  SELECT
    -- Use first non-null value for each ID type
    MAX(player_id) FILTER (WHERE player_id IS NOT NULL) as player_id,
    MAX(gsis_id) FILTER (WHERE gsis_id IS NOT NULL) as gsis_id,
    MAX(pfr_id) FILTER (WHERE pfr_id IS NOT NULL) as pfr_id,
    MAX(espn_id) FILTER (WHERE espn_id IS NOT NULL) as espn_id,
    MAX(yahoo_id) FILTER (WHERE yahoo_id IS NOT NULL) as yahoo_id,
    MAX(sleeper_id) FILTER (WHERE sleeper_id IS NOT NULL) as sleeper_id,
    MAX(pff_id) FILTER (WHERE pff_id IS NOT NULL) as pff_id,
    MAX(rotowire_id) FILTER (WHERE rotowire_id IS NOT NULL) as rotowire_id,
    MAX(fantasy_data_id) FILTER (WHERE fantasy_data_id IS NOT NULL) as fantasy_data_id,
    MAX(sportradar_id) FILTER (WHERE sportradar_id IS NOT NULL) as sportradar_id,

    -- Use most common player name (handles slight variations)
    MODE() WITHIN GROUP (ORDER BY player_name) as canonical_name,

    -- Use most common position
    MODE() WITHIN GROUP (ORDER BY position) as canonical_position,

    -- Metadata about mapping
    COUNT(DISTINCT source_table) as source_count,
    ARRAY_AGG(DISTINCT source_table ORDER BY source_table) as sources,
    COUNT(*) as total_records

  FROM player_ids
  WHERE player_id IS NOT NULL
     OR gsis_id IS NOT NULL
     OR pfr_id IS NOT NULL
     OR espn_id IS NOT NULL
  GROUP BY
    COALESCE(gsis_id, '___' || COALESCE(pfr_id, '___' || COALESCE(player_id, '___' || COALESCE(espn_id, 'NONE'))))
)
SELECT
  player_id,
  gsis_id,
  pfr_id,
  espn_id,
  yahoo_id,
  sleeper_id,
  pff_id,
  rotowire_id,
  fantasy_data_id,
  sportradar_id,
  canonical_name,
  canonical_position,
  source_count,
  sources,
  total_records,

  -- Quality score: higher is better (more ID systems matched)
  (CASE WHEN player_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN gsis_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN pfr_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN espn_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN yahoo_id IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN sleeper_id IS NOT NULL THEN 1 ELSE 0 END) as id_completeness_score

FROM aggregated
WHERE player_id IS NOT NULL
   OR gsis_id IS NOT NULL
   OR pfr_id IS NOT NULL
ORDER BY id_completeness_score DESC, source_count DESC, canonical_name;

-- Add documentation comment
COMMENT ON VIEW player_id_mapping IS
'Unified player ID mapping across all ID systems (player_id, gsis_id, pfr_id, espn_id, yahoo_id, sleeper_id, pff_id, rotowire_id, fantasy_data_id, sportradar_id).
Use this view as the single source of truth for player ID lookups and conversions between different ID systems.
The id_completeness_score indicates how many ID systems have been matched for each player (higher is better).';

-- Create a helper function to lookup player by any ID
CREATE OR REPLACE FUNCTION lookup_player_ids(
  p_player_id TEXT DEFAULT NULL,
  p_gsis_id TEXT DEFAULT NULL,
  p_pfr_id TEXT DEFAULT NULL,
  p_espn_id TEXT DEFAULT NULL
)
RETURNS TABLE (
  player_id TEXT,
  gsis_id TEXT,
  pfr_id TEXT,
  espn_id TEXT,
  canonical_name TEXT,
  canonical_position TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pim.player_id,
    pim.gsis_id,
    pim.pfr_id,
    pim.espn_id,
    pim.canonical_name,
    pim.canonical_position
  FROM player_id_mapping pim
  WHERE
    (p_player_id IS NULL OR pim.player_id = p_player_id) AND
    (p_gsis_id IS NULL OR pim.gsis_id = p_gsis_id) AND
    (p_pfr_id IS NULL OR pim.pfr_id = p_pfr_id) AND
    (p_espn_id IS NULL OR pim.espn_id = p_espn_id)
  ORDER BY pim.id_completeness_score DESC
  LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION lookup_player_ids IS
'Helper function to lookup a player by any ID system. Returns the most complete player record matching the given IDs.
Usage: SELECT * FROM lookup_player_ids(p_gsis_id => ''00-0033873'');';
