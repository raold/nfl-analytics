-- Schema of devdb01 as left by the 2026-10-24 salvage (migration 104 applied, games repaired),
-- for bootstrapping an empty database; see db/README.md. No data. Dumped read-only on
-- 2026-10-25 with pg_dump 17.7 inside the nfl-analytics-pg-1 container:
--
--   pg_dump -U dro -d devdb01 --schema-only --no-owner --no-privileges \
--     --exclude-schema=salvage --exclude-schema=_timescaledb_internal \
--     --exclude-table='public.mlb_*' --exclude-table='public.nhl_*'
--
-- Left out: the salvage schema (nflverse staging copy), Sibyl's 13 mlb_*/nhl_* tables, and
-- TimescaleDB chunks. The create_hypertable() calls at the end are not pg_dump output: a
-- schema-only dump keeps the two hypertables as plain tables.
--
-- Apply to an empty database that has (or may create) the timescaledb extension:
--   docker compose exec -T pg-1 psql -U dro -d devdb01 -v ON_ERROR_STOP=1 -f - \
--     < db/schema/devdb01_schema_2026-10-24.sql
-- (scripts/dev/init_dev.sh does this). Use psql 13.22, 14.19, 15.14, 16.10, 17.6 or newer:
-- older clients do not know the \restrict line below.

--
-- PostgreSQL database dump
--

\restrict fTfDeIRb1MIXS0kB7JJgaGUTSNzJtVEih0eq5JKR9HckAoMwLpTcNVeFQ90ibbT

-- Dumped from database version 17.7
-- Dumped by pg_dump version 17.7

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: timescaledb; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS timescaledb WITH SCHEMA public;


--
-- Name: EXTENSION timescaledb; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION timescaledb IS 'Enables scalable inserts and complex queries for time-series data (Community Edition)';


--
-- Name: mart; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA mart;


--
-- Name: metadata; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA metadata;


--
-- Name: ml; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA ml;


--
-- Name: bool_avg_state; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.bool_avg_state AS (
	total bigint,
	cnt bigint
);


--
-- Name: get_position_group(character varying); Type: FUNCTION; Schema: mart; Owner: -
--

CREATE FUNCTION mart.get_position_group(p_position character varying) RETURNS character varying
    LANGUAGE sql IMMUTABLE
    AS $$
    SELECT position_group
    FROM mart.position_groups
    WHERE position = p_position
    LIMIT 1;
$$;


--
-- Name: refresh_all_views(); Type: FUNCTION; Schema: mart; Owner: -
--

CREATE FUNCTION mart.refresh_all_views() RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;
    RAISE NOTICE 'All mart materialized views refreshed at %', NOW();
END;
$$;


--
-- Name: refresh_game_features(); Type: FUNCTION; Schema: mart; Owner: -
--

CREATE FUNCTION mart.refresh_game_features() RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
  REFRESH MATERIALIZED VIEW mart.game_features_enhanced;
  RAISE NOTICE 'Refreshed mart.game_features_enhanced';
END;
$$;


--
-- Name: FUNCTION refresh_game_features(); Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON FUNCTION mart.refresh_game_features() IS 'Refresh the game_features_enhanced materialized view after loading new data';


--
-- Name: get_dirichlet_weights(character varying); Type: FUNCTION; Schema: ml; Owner: -
--

CREATE FUNCTION ml.get_dirichlet_weights(p_stat_type character varying) RETURNS TABLE(component_name character varying, weight numeric)
    LANGUAGE plpgsql STABLE
    AS $$
DECLARE
    total_alpha NUMERIC;
BEGIN
    -- Get total alpha for normalization
    SELECT SUM(alpha) INTO total_alpha
    FROM ml.dirichlet_alpha_by_stat_type
    WHERE stat_type = p_stat_type;

    IF total_alpha IS NULL OR total_alpha = 0 THEN
        RAISE EXCEPTION 'No Dirichlet alphas found for stat_type: %', p_stat_type;
    END IF;

    -- Return normalized weights
    RETURN QUERY
    SELECT
        d.component_name,
        (d.alpha / total_alpha)::NUMERIC(8,6) as weight
    FROM ml.dirichlet_alpha_by_stat_type d
    WHERE d.stat_type = p_stat_type
    ORDER BY d.component_order;
END;
$$;


--
-- Name: log_bayesian_update(character varying, character varying, integer, integer, jsonb, jsonb, jsonb, character varying, character varying, numeric, numeric, jsonb, numeric); Type: FUNCTION; Schema: ml; Owner: -
--

CREATE FUNCTION ml.log_bayesian_update(p_update_type character varying, p_stat_type character varying, p_season integer, p_week integer, p_prior_params jsonb, p_likelihoods jsonb, p_posterior_params jsonb, p_player_id character varying DEFAULT NULL::character varying, p_game_id character varying DEFAULT NULL::character varying, p_actual_value numeric DEFAULT NULL::numeric, p_ensemble_prediction numeric DEFAULT NULL::numeric, p_component_predictions jsonb DEFAULT NULL::jsonb, p_learning_rate numeric DEFAULT 1.0) RETURNS bigint
    LANGUAGE plpgsql
    AS $$
DECLARE
    new_log_id BIGINT;
    pred_error NUMERIC;
BEGIN
    -- Compute prediction error if both values provided
    IF p_actual_value IS NOT NULL AND p_ensemble_prediction IS NOT NULL THEN
        pred_error := p_actual_value - p_ensemble_prediction;
    END IF;

    INSERT INTO ml.bayesian_update_log (
        update_type, player_id, game_id, stat_type,
        season, week,
        prior_params, likelihoods, posterior_params,
        actual_value, ensemble_prediction, prediction_error, abs_error,
        component_predictions, learning_rate_used
    ) VALUES (
        p_update_type, p_player_id, p_game_id, p_stat_type,
        p_season, p_week,
        p_prior_params, p_likelihoods, p_posterior_params,
        p_actual_value, p_ensemble_prediction, pred_error, ABS(pred_error),
        p_component_predictions, p_learning_rate
    )
    RETURNING log_id INTO new_log_id;

    RETURN new_log_id;
END;
$$;


--
-- Name: reset_dirichlet_alphas(character varying); Type: FUNCTION; Schema: ml; Owner: -
--

CREATE FUNCTION ml.reset_dirichlet_alphas(p_stat_type character varying DEFAULT NULL::character varying) RETURNS integer
    LANGUAGE plpgsql
    AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    IF p_stat_type IS NULL THEN
        UPDATE ml.dirichlet_alpha_by_stat_type
        SET alpha = prior_alpha, n_updates = 0, cumulative_likelihood = 0, updated_at = NOW();
    ELSE
        UPDATE ml.dirichlet_alpha_by_stat_type
        SET alpha = prior_alpha, n_updates = 0, cumulative_likelihood = 0, updated_at = NOW()
        WHERE stat_type = p_stat_type;
    END IF;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$;


--
-- Name: update_dirichlet_alpha(character varying, character varying, numeric, numeric); Type: FUNCTION; Schema: ml; Owner: -
--

CREATE FUNCTION ml.update_dirichlet_alpha(p_stat_type character varying, p_component_name character varying, p_likelihood numeric, p_learning_rate numeric DEFAULT 1.0) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    UPDATE ml.dirichlet_alpha_by_stat_type
    SET
        alpha = alpha + (p_likelihood * p_learning_rate),
        n_updates = n_updates + 1,
        cumulative_likelihood = cumulative_likelihood + p_likelihood,
        updated_at = NOW()
    WHERE stat_type = p_stat_type
      AND component_name = p_component_name;

    IF NOT FOUND THEN
        RAISE WARNING 'No alpha found for stat_type=%, component=%', p_stat_type, p_component_name;
    END IF;
END;
$$;


--
-- Name: update_model_timestamp(); Type: FUNCTION; Schema: ml; Owner: -
--

CREATE FUNCTION ml.update_model_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;


--
-- Name: bool_avg_ffunc(public.bool_avg_state); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bool_avg_ffunc(state public.bool_avg_state) RETURNS numeric
    LANGUAGE sql IMMUTABLE
    AS $$
  SELECT CASE WHEN state.cnt = 0 THEN NULL 
    ELSE state.total::numeric / state.cnt 
  END
$$;


--
-- Name: bool_avg_sfunc(public.bool_avg_state, boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bool_avg_sfunc(state public.bool_avg_state, val boolean) RETURNS public.bool_avg_state
    LANGUAGE sql IMMUTABLE
    AS $$
  SELECT ROW(
    state.total + CASE WHEN val THEN 1 ELSE 0 END,
    state.cnt + 1
  )::bool_avg_state
$$;


--
-- Name: bool_eq_int(boolean, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bool_eq_int(boolean, integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
  SELECT $1 = ($2 != 0)
$_$;


--
-- Name: bool_sum_sfunc(bigint, boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bool_sum_sfunc(state bigint, val boolean) RETURNS bigint
    LANGUAGE sql IMMUTABLE
    AS $$
  SELECT state + CASE WHEN val THEN 1 ELSE 0 END
$$;


--
-- Name: bool_to_int(boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.bool_to_int(boolean) RETURNS integer
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
  SELECT CASE WHEN $1 THEN 1 ELSE 0 END
$_$;


--
-- Name: get_referee_bias_effect(text, boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_referee_bias_effect(p_game_id text, p_is_home boolean) RETURNS double precision
    LANGUAGE plpgsql STABLE
    AS $$
DECLARE
    v_home_bias FLOAT;
    v_bias_effect FLOAT;
BEGIN
    SELECT home_bias
    INTO v_home_bias
    FROM referee_assignments
    WHERE game_id = p_game_id;

    IF v_home_bias IS NULL THEN
        RETURN NULL;
    END IF;

    -- Calculate bias effect: 8 yards per unit bias
    -- Positive home_bias helps home team, hurts away team
    IF p_is_home THEN
        v_bias_effect := 8.0 * v_home_bias;
    ELSE
        v_bias_effect := -8.0 * v_home_bias;
    END IF;

    RETURN v_bias_effect;
END;
$$;


--
-- Name: FUNCTION get_referee_bias_effect(p_game_id text, p_is_home boolean); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.get_referee_bias_effect(p_game_id text, p_is_home boolean) IS 'Calculate expected referee bias effect in passing yards (8 yards per unit bias)';


--
-- Name: get_scheme_change_indicator(text, integer, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_scheme_change_indicator(p_team text, p_season integer, p_week integer) RETURNS integer
    LANGUAGE sql STABLE
    AS $$
    SELECT COALESCE(oc_changed::INT, 0)
    FROM coaching_changes
    WHERE team = p_team AND season = p_season AND week = p_week;
$$;


--
-- Name: FUNCTION get_scheme_change_indicator(p_team text, p_season integer, p_week integer); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.get_scheme_change_indicator(p_team text, p_season integer, p_week integer) IS 'Returns 1 if offensive coordinator changed in this team-season-week, 0 otherwise';


--
-- Name: get_travel_fatigue_effect(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.get_travel_fatigue_effect(p_game_id text) RETURNS double precision
    LANGUAGE plpgsql STABLE
    AS $$
DECLARE
    v_distance FLOAT;
    v_time_zones INT;
    v_fatigue_effect FLOAT;
BEGIN
    SELECT travel_distance_miles, time_zones_crossed
    INTO v_distance, v_time_zones
    FROM team_travel
    WHERE game_id = p_game_id;

    IF v_distance IS NULL THEN
        RETURN NULL;
    END IF;

    -- Calculate fatigue effect
    -- Distance penalty: -0.5 yards per 1000 miles
    -- Timezone penalty: -3 yards per timezone (only if traveling East, positive time zones)
    v_fatigue_effect := -0.5 * (v_distance / 1000.0);

    IF v_time_zones > 0 THEN
        v_fatigue_effect := v_fatigue_effect - (3.0 * v_time_zones);
    END IF;

    RETURN v_fatigue_effect;
END;
$$;


--
-- Name: FUNCTION get_travel_fatigue_effect(p_game_id text); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.get_travel_fatigue_effect(p_game_id text) IS 'Calculate expected travel fatigue effect in passing yards for a game';


--
-- Name: int_eq_bool(integer, boolean); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.int_eq_bool(integer, boolean) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
  SELECT ($1 != 0) = $2
$_$;


--
-- Name: int_to_bool(integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.int_to_bool(integer) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
  SELECT $1 != 0
$_$;


--
-- Name: avg(boolean); Type: AGGREGATE; Schema: public; Owner: -
--

CREATE AGGREGATE public.avg(boolean) (
    SFUNC = public.bool_avg_sfunc,
    STYPE = public.bool_avg_state,
    INITCOND = '(0,0)',
    FINALFUNC = public.bool_avg_ffunc
);


--
-- Name: sum(boolean); Type: AGGREGATE; Schema: public; Owner: -
--

CREATE AGGREGATE public.sum(boolean) (
    SFUNC = public.bool_sum_sfunc,
    STYPE = bigint,
    INITCOND = '0'
);


--
-- Name: =; Type: OPERATOR; Schema: public; Owner: -
--

CREATE OPERATOR public.= (
    FUNCTION = public.bool_eq_int,
    LEFTARG = boolean,
    RIGHTARG = integer,
    COMMUTATOR = OPERATOR(public.=)
);


--
-- Name: =; Type: OPERATOR; Schema: public; Owner: -
--

CREATE OPERATOR public.= (
    FUNCTION = public.int_eq_bool,
    LEFTARG = integer,
    RIGHTARG = boolean,
    COMMUTATOR = OPERATOR(public.=)
);


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: model_metrics; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_metrics (
    run_id integer NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    metric_name text NOT NULL,
    metric_type text NOT NULL,
    metric_value real NOT NULL,
    epoch integer,
    step integer,
    metadata jsonb
);


--
-- Name: TABLE model_metrics; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_metrics IS 'Time-series metrics during training (TimescaleDB hypertable)';


--
-- Name: odds_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.odds_history (
    captured_at timestamp with time zone NOT NULL,
    game_id text NOT NULL,
    bookmaker text NOT NULL,
    market_type text NOT NULL,
    home_spread real,
    away_spread real,
    home_spread_price integer,
    away_spread_price integer,
    over_under real,
    over_price integer,
    under_price integer,
    home_moneyline integer,
    away_moneyline integer,
    last_update timestamp with time zone
);


--
-- Name: bayesian_player_ratings; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.bayesian_player_ratings (
    player_id character varying(50) NOT NULL,
    stat_type character varying(50) NOT NULL,
    season integer NOT NULL,
    model_version character varying(50) NOT NULL,
    rating_mean double precision NOT NULL,
    rating_sd double precision NOT NULL,
    rating_q05 double precision NOT NULL,
    rating_q50 double precision NOT NULL,
    rating_q95 double precision NOT NULL,
    position_group_mean double precision,
    team_effect double precision,
    vs_opponent_effect double precision,
    n_games_observed integer NOT NULL,
    effective_sample_size double precision,
    rhat double precision,
    trained_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


--
-- Name: games; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.games (
    game_id text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    home_team text NOT NULL,
    away_team text NOT NULL,
    kickoff timestamp with time zone,
    stadium text,
    stadium_id text,
    roof text,
    surface text,
    spread_close real,
    total_close real,
    home_moneyline real,
    away_moneyline real,
    home_spread_odds real,
    away_spread_odds real,
    over_odds real,
    under_odds real,
    home_score integer,
    away_score integer,
    away_rest integer,
    home_rest integer,
    away_qb_id text,
    home_qb_id text,
    away_qb_name text,
    home_qb_name text,
    away_coach text,
    home_coach text,
    referee text,
    game_type text,
    overtime integer,
    home_turnovers integer,
    away_turnovers integer,
    home_penalties integer,
    away_penalties integer,
    home_penalty_yards integer,
    away_penalty_yards integer,
    updated_at timestamp without time zone,
    odds_api_event_id text,
    odds_last_fetched timestamp with time zone,
    odds_snapshots_count integer DEFAULT 0
);


--
-- Name: weather; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.weather (
    game_id text NOT NULL,
    temp real,
    wind real,
    humidity real,
    conditions text,
    data_source text
);


--
-- Name: game_summary; Type: MATERIALIZED VIEW; Schema: mart; Owner: -
--

CREATE MATERIALIZED VIEW mart.game_summary AS
 SELECT g.game_id,
    g.season,
    g.week,
    g.home_team,
    g.away_team,
    g.kickoff,
    g.stadium,
    g.roof,
    g.surface,
    g.spread_close,
    g.total_close,
    g.home_moneyline,
    g.away_moneyline,
    g.home_score,
    g.away_score,
    (g.home_score - g.away_score) AS margin,
    (g.home_score + g.away_score) AS total_points,
        CASE
            WHEN (g.home_score > g.away_score) THEN 1
            WHEN (g.home_score < g.away_score) THEN 0
            ELSE NULL::integer
        END AS home_win,
        CASE
            WHEN (g.home_score IS NOT NULL) THEN true
            ELSE false
        END AS is_final,
        CASE
            WHEN ((g.home_score IS NOT NULL) AND (g.spread_close IS NOT NULL)) THEN (((g.home_score - g.away_score))::double precision - g.spread_close)
            ELSE NULL::double precision
        END AS home_cover_margin,
        CASE
            WHEN ((g.home_score IS NOT NULL) AND (g.spread_close IS NOT NULL)) THEN
            CASE
                WHEN ((((g.home_score - g.away_score))::double precision - g.spread_close) > (0)::double precision) THEN 1
                WHEN ((((g.home_score - g.away_score))::double precision - g.spread_close) < (0)::double precision) THEN 0
                ELSE NULL::integer
            END
            ELSE NULL::integer
        END AS home_covered,
        CASE
            WHEN ((g.home_score IS NOT NULL) AND (g.total_close IS NOT NULL)) THEN
            CASE
                WHEN (((g.home_score + g.away_score))::double precision > g.total_close) THEN 'OVER'::text
                WHEN (((g.home_score + g.away_score))::double precision < g.total_close) THEN 'UNDER'::text
                ELSE 'PUSH'::text
            END
            ELSE NULL::text
        END AS total_result,
    w.temp,
    w.wind,
    w.humidity,
    w.conditions,
    g.game_type,
    g.overtime,
    g.home_rest,
    g.away_rest
   FROM (public.games g
     LEFT JOIN public.weather w ON ((g.game_id = w.game_id)))
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW game_summary; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON MATERIALIZED VIEW mart.game_summary IS 'Enriched game-level summary with betting analysis. Refresh after games complete.';


--
-- Name: position_groups; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.position_groups (
    "position" character varying(10) NOT NULL,
    position_group character varying(10) NOT NULL,
    position_group_desc text,
    is_offensive boolean NOT NULL,
    typical_stats text[]
);


--
-- Name: nextgen_passing; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nextgen_passing (
    player_id text NOT NULL,
    player_display_name text,
    player_position text,
    season integer NOT NULL,
    week integer NOT NULL,
    attempts integer,
    pass_yards integer,
    pass_touchdowns integer,
    interceptions integer,
    passer_rating double precision,
    completions integer,
    completion_percentage double precision,
    avg_time_to_throw double precision,
    avg_completed_air_yards double precision,
    avg_intended_air_yards double precision,
    avg_air_yards_differential double precision,
    aggressiveness double precision,
    max_completed_air_distance double precision,
    avg_air_yards_to_sticks double precision,
    completion_percentage_above_expectation double precision,
    expected_completion_percentage double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE nextgen_passing; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.nextgen_passing IS 'NFL Next Gen Stats - QB passing metrics with player tracking data';


--
-- Name: COLUMN nextgen_passing.avg_time_to_throw; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_passing.avg_time_to_throw IS 'Average time from snap to throw in seconds - lower indicates quick release';


--
-- Name: COLUMN nextgen_passing.aggressiveness; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_passing.aggressiveness IS 'Percentage of passes thrown into tight coverage (high risk/reward)';


--
-- Name: COLUMN nextgen_passing.completion_percentage_above_expectation; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_passing.completion_percentage_above_expectation IS 'CPOE - completion % above expected based on throw difficulty';


--
-- Name: nextgen_receiving; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nextgen_receiving (
    player_id text NOT NULL,
    player_display_name text,
    player_position text,
    season integer NOT NULL,
    week integer NOT NULL,
    targets integer,
    receptions integer,
    receiving_yards integer,
    receiving_touchdowns integer,
    receiving_fumbles integer,
    receiving_air_yards integer,
    receiving_yards_after_catch integer,
    receiving_first_downs integer,
    avg_cushion double precision,
    avg_separation double precision,
    avg_intended_air_yards double precision,
    percent_share_of_intended_air_yards double precision,
    catch_percentage double precision,
    avg_yac double precision,
    avg_expected_yac double precision,
    avg_yac_above_expectation double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE nextgen_receiving; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.nextgen_receiving IS 'NFL Next Gen Stats - WR/TE/RB receiving metrics with player tracking data';


--
-- Name: COLUMN nextgen_receiving.avg_cushion; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_receiving.avg_cushion IS 'Average yards of separation at snap (press coverage vs soft)';


--
-- Name: COLUMN nextgen_receiving.avg_separation; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_receiving.avg_separation IS 'Average yards of separation from defender at catch point (higher = more open)';


--
-- Name: COLUMN nextgen_receiving.avg_yac_above_expectation; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_receiving.avg_yac_above_expectation IS 'Yards after catch above expected based on situation (key playmaker metric)';


--
-- Name: nextgen_rushing; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nextgen_rushing (
    player_id text NOT NULL,
    player_display_name text,
    player_position text,
    season integer NOT NULL,
    week integer NOT NULL,
    carries integer,
    rush_yards integer,
    rush_touchdowns integer,
    rush_first_downs integer,
    rush_fumbles integer,
    efficiency double precision,
    percent_attempts_gte_eight_defenders double precision,
    avg_time_to_los double precision,
    rush_yards_over_expected double precision,
    avg_rush_yards double precision,
    rush_yards_over_expected_per_att double precision,
    rush_pct_over_expected double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE nextgen_rushing; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.nextgen_rushing IS 'NFL Next Gen Stats - RB/QB rushing metrics with player tracking data';


--
-- Name: COLUMN nextgen_rushing.efficiency; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_rushing.efficiency IS 'Percentage of potential yards gained (100% = optimal path)';


--
-- Name: COLUMN nextgen_rushing.percent_attempts_gte_eight_defenders; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_rushing.percent_attempts_gte_eight_defenders IS 'Percentage of carries against 8+ defenders (stacked box indicator)';


--
-- Name: COLUMN nextgen_rushing.rush_yards_over_expected_per_att; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.nextgen_rushing.rush_yards_over_expected_per_att IS 'Average yards gained above expected based on blocking/defense';


--
-- Name: players; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.players (
    player_id text NOT NULL,
    player_name text,
    "position" text,
    height text,
    weight integer,
    college text,
    birth_date date,
    rookie_year integer,
    draft_club text,
    draft_number integer,
    headshot_url text,
    status text,
    entry_year integer,
    years_exp integer
);


--
-- Name: rosters_weekly; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rosters_weekly (
    season integer NOT NULL,
    week integer NOT NULL,
    game_type text NOT NULL,
    team text NOT NULL,
    gsis_id text NOT NULL,
    "position" text,
    depth_chart_position text,
    jersey_number integer,
    status text,
    full_name text,
    first_name text,
    last_name text,
    birth_date date,
    height text,
    weight integer,
    college text,
    espn_id text,
    sportradar_id text,
    yahoo_id text,
    rotowire_id text,
    pff_id text,
    pfr_id text,
    fantasy_data_id text,
    sleeper_id text,
    years_exp integer,
    headshot_url text,
    ngs_position text,
    status_description_abbr text,
    football_name text,
    esb_id text,
    gsis_it_id text,
    smart_id text,
    entry_year integer,
    rookie_year integer,
    draft_club text,
    draft_number integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: player_hierarchy; Type: MATERIALIZED VIEW; Schema: mart; Owner: -
--

CREATE MATERIALIZED VIEW mart.player_hierarchy AS
 WITH player_teams AS (
         SELECT DISTINCT ON (rw.gsis_id) rw.gsis_id AS player_id,
            rw.team,
            rw.season,
            rw.week
           FROM public.rosters_weekly rw
          WHERE ((rw.season >= 2016) AND (rw.gsis_id IS NOT NULL))
          ORDER BY rw.gsis_id, rw.season DESC, rw.week DESC
        ), player_stats_summary AS (
         SELECT all_stats.player_id,
            count(*) AS games_played,
            min(all_stats.season) AS first_season,
            max(all_stats.season) AS last_season
           FROM ( SELECT nextgen_passing.player_id,
                    nextgen_passing.season,
                    nextgen_passing.week
                   FROM public.nextgen_passing
                UNION ALL
                 SELECT nextgen_rushing.player_id,
                    nextgen_rushing.season,
                    nextgen_rushing.week
                   FROM public.nextgen_rushing
                UNION ALL
                 SELECT nextgen_receiving.player_id,
                    nextgen_receiving.season,
                    nextgen_receiving.week
                   FROM public.nextgen_receiving) all_stats
          GROUP BY all_stats.player_id
        )
 SELECT p.player_id,
    p.player_name,
    p."position",
    COALESCE(pg.position_group, 'UNK'::character varying) AS position_group,
    COALESCE(pg.is_offensive, true) AS is_offensive,
    pt.team AS current_team,
    pt.season AS current_season,
    p.rookie_year,
    p.years_exp,
    COALESCE(pss.games_played, (0)::bigint) AS games_with_stats,
    pss.first_season AS stats_first_season,
    pss.last_season AS stats_last_season,
    concat(COALESCE(pg.position_group, 'UNK'::character varying), ':', COALESCE(p."position", 'UNK'::text)) AS hierarchy_position,
    concat(COALESCE(pt.team, 'UNK'::text), ':', COALESCE(pg.position_group, 'UNK'::character varying)) AS hierarchy_team_position
   FROM (((public.players p
     LEFT JOIN mart.position_groups pg ON ((p."position" = (pg."position")::text)))
     LEFT JOIN player_teams pt ON ((p.player_id = pt.player_id)))
     LEFT JOIN player_stats_summary pss ON ((p.player_id = pss.player_id)))
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW player_hierarchy; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON MATERIALIZED VIEW mart.player_hierarchy IS 'Hierarchical player structure for Bayesian modeling. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;';


--
-- Name: player_game_stats; Type: MATERIALIZED VIEW; Schema: mart; Owner: -
--

CREATE MATERIALIZED VIEW mart.player_game_stats AS
 WITH passing_stats AS (
         SELECT nextgen_passing.player_id,
            nextgen_passing.season,
            nextgen_passing.week,
            'passing'::text AS stat_category,
            nextgen_passing.player_display_name,
            nextgen_passing.player_position,
            nextgen_passing.attempts AS stat_attempts,
            nextgen_passing.completions AS stat_completions,
            nextgen_passing.pass_yards AS stat_yards,
            nextgen_passing.pass_touchdowns AS stat_touchdowns,
            nextgen_passing.interceptions AS stat_negative,
            nextgen_passing.avg_time_to_throw,
            nextgen_passing.avg_air_yards_differential,
            nextgen_passing.completion_percentage_above_expectation AS cpoe
           FROM public.nextgen_passing
        ), rushing_stats AS (
         SELECT nextgen_rushing.player_id,
            nextgen_rushing.season,
            nextgen_rushing.week,
            'rushing'::text AS stat_category,
            nextgen_rushing.player_display_name,
            nextgen_rushing.player_position,
            nextgen_rushing.carries AS stat_attempts,
            NULL::double precision AS stat_completions,
            nextgen_rushing.rush_yards AS stat_yards,
            nextgen_rushing.rush_touchdowns AS stat_touchdowns,
            NULL::double precision AS stat_negative,
            nextgen_rushing.avg_time_to_los AS avg_time_to_throw,
            nextgen_rushing.efficiency AS avg_air_yards_differential,
            NULL::double precision AS cpoe
           FROM public.nextgen_rushing
        ), receiving_stats AS (
         SELECT nextgen_receiving.player_id,
            nextgen_receiving.season,
            nextgen_receiving.week,
            'receiving'::text AS stat_category,
            nextgen_receiving.player_display_name,
            nextgen_receiving.player_position,
            nextgen_receiving.targets AS stat_attempts,
            nextgen_receiving.receptions AS stat_completions,
            nextgen_receiving.receiving_yards AS stat_yards,
            nextgen_receiving.receiving_touchdowns AS stat_touchdowns,
            NULL::double precision AS stat_negative,
            nextgen_receiving.avg_separation AS avg_time_to_throw,
            nextgen_receiving.avg_yac_above_expectation AS avg_air_yards_differential,
            NULL::double precision AS cpoe
           FROM public.nextgen_receiving
        ), all_stats AS (
         SELECT passing_stats.player_id,
            passing_stats.season,
            passing_stats.week,
            passing_stats.stat_category,
            passing_stats.player_display_name,
            passing_stats.player_position,
            passing_stats.stat_attempts,
            passing_stats.stat_completions,
            passing_stats.stat_yards,
            passing_stats.stat_touchdowns,
            passing_stats.stat_negative,
            passing_stats.avg_time_to_throw,
            passing_stats.avg_air_yards_differential,
            passing_stats.cpoe
           FROM passing_stats
        UNION ALL
         SELECT rushing_stats.player_id,
            rushing_stats.season,
            rushing_stats.week,
            rushing_stats.stat_category,
            rushing_stats.player_display_name,
            rushing_stats.player_position,
            rushing_stats.stat_attempts,
            rushing_stats.stat_completions,
            rushing_stats.stat_yards,
            rushing_stats.stat_touchdowns,
            rushing_stats.stat_negative,
            rushing_stats.avg_time_to_throw,
            rushing_stats.avg_air_yards_differential,
            rushing_stats.cpoe
           FROM rushing_stats
        UNION ALL
         SELECT receiving_stats.player_id,
            receiving_stats.season,
            receiving_stats.week,
            receiving_stats.stat_category,
            receiving_stats.player_display_name,
            receiving_stats.player_position,
            receiving_stats.stat_attempts,
            receiving_stats.stat_completions,
            receiving_stats.stat_yards,
            receiving_stats.stat_touchdowns,
            receiving_stats.stat_negative,
            receiving_stats.avg_time_to_throw,
            receiving_stats.avg_air_yards_differential,
            receiving_stats.cpoe
           FROM receiving_stats
        )
 SELECT a.player_id,
    a.season,
    a.week,
    a.stat_category,
    a.player_display_name,
    a.player_position,
    a.stat_attempts,
    a.stat_completions,
    a.stat_yards,
    a.stat_touchdowns,
    a.stat_negative,
    a.avg_time_to_throw,
    a.avg_air_yards_differential,
    a.cpoe,
    ph.position_group,
    ph.current_team,
    ph.hierarchy_position,
    ph.hierarchy_team_position
   FROM (all_stats a
     LEFT JOIN mart.player_hierarchy ph ON ((a.player_id = ph.player_id)))
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW player_game_stats; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON MATERIALIZED VIEW mart.player_game_stats IS 'Aggregated NextGen stats at player-game level for Bayesian modeling.';


--
-- Name: team_4th_down_features; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.team_4th_down_features (
    game_id text NOT NULL,
    team text NOT NULL,
    fourth_downs integer,
    went_for_it_rate real,
    fourth_down_epa real,
    bad_decisions integer,
    avg_go_boost real,
    avg_fg_boost real
);


--
-- Name: TABLE team_4th_down_features; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON TABLE mart.team_4th_down_features IS '4th down decision quality metrics using nfl4th package';


--
-- Name: COLUMN team_4th_down_features.went_for_it_rate; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_4th_down_features.went_for_it_rate IS 'Fraction of 4th downs where team went for it (vs punt/FG)';


--
-- Name: COLUMN team_4th_down_features.fourth_down_epa; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_4th_down_features.fourth_down_epa IS 'Average EPA on 4th down plays';


--
-- Name: COLUMN team_4th_down_features.bad_decisions; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_4th_down_features.bad_decisions IS 'Count of suboptimal 4th down decisions (boost < -2)';


--
-- Name: COLUMN team_4th_down_features.avg_go_boost; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_4th_down_features.avg_go_boost IS 'Average expected points added from going for it vs alternatives';


--
-- Name: team_epa; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.team_epa (
    game_id text NOT NULL,
    posteam text NOT NULL,
    plays integer,
    epa_sum double precision,
    epa_mean double precision,
    explosive_pass double precision,
    explosive_rush double precision
);


--
-- Name: team_injury_load; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.team_injury_load (
    season integer NOT NULL,
    week integer NOT NULL,
    team text NOT NULL,
    total_injuries integer,
    players_out integer,
    players_questionable integer,
    players_doubtful integer,
    key_position_out boolean,
    qb_out boolean,
    oline_injuries integer,
    injury_severity_index real
);


--
-- Name: TABLE team_injury_load; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON TABLE mart.team_injury_load IS 'Weekly injury load metrics from official NFL injury reports';


--
-- Name: COLUMN team_injury_load.total_injuries; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_injury_load.total_injuries IS 'Total players on injury report';


--
-- Name: COLUMN team_injury_load.players_out; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_injury_load.players_out IS 'Players with Out/IR/PUP/Suspended status';


--
-- Name: COLUMN team_injury_load.key_position_out; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_injury_load.key_position_out IS 'At least one key position (QB/OL/DL/CB) is Out';


--
-- Name: COLUMN team_injury_load.qb_out; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_injury_load.qb_out IS 'Starting QB is Out';


--
-- Name: COLUMN team_injury_load.injury_severity_index; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_injury_load.injury_severity_index IS 'Weighted sum: Out/IR=3, Doubtful=2, Questionable=1';


--
-- Name: team_playoff_context; Type: TABLE; Schema: mart; Owner: -
--

CREATE TABLE mart.team_playoff_context (
    team text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    playoff_prob real,
    div_winner_prob real,
    first_seed_prob real,
    eliminated boolean,
    locked_in boolean,
    desperate boolean
);


--
-- Name: TABLE team_playoff_context; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON TABLE mart.team_playoff_context IS 'Playoff probabilities and desperation indicators from nflseedR simulation';


--
-- Name: COLUMN team_playoff_context.playoff_prob; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_playoff_context.playoff_prob IS 'Probability of making playoffs (0-1)';


--
-- Name: COLUMN team_playoff_context.eliminated; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_playoff_context.eliminated IS 'Team mathematically eliminated from playoffs (prob < 0.01)';


--
-- Name: COLUMN team_playoff_context.locked_in; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_playoff_context.locked_in IS 'Team has clinched playoff spot (prob > 0.99)';


--
-- Name: COLUMN team_playoff_context.desperate; Type: COMMENT; Schema: mart; Owner: -
--

COMMENT ON COLUMN mart.team_playoff_context.desperate IS 'Team in must-win range (prob between 0.15 and 0.60)';


--
-- Name: feature_freshness; Type: TABLE; Schema: metadata; Owner: -
--

CREATE TABLE metadata.feature_freshness (
    feature_name text NOT NULL,
    last_updated timestamp with time zone,
    row_count bigint,
    max_season integer,
    max_week integer,
    is_stale boolean,
    staleness_reason text,
    checked_at timestamp with time zone DEFAULT now()
);


--
-- Name: active_models; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.active_models AS
SELECT
    NULL::integer AS model_id,
    NULL::text AS model_name,
    NULL::text AS model_type,
    NULL::text AS model_family,
    NULL::text AS version,
    NULL::text AS model_path,
    NULL::text AS checkpoint_dir,
    NULL::text AS metadata_path,
    NULL::text AS status,
    NULL::boolean AS is_production,
    NULL::timestamp with time zone AS training_start,
    NULL::timestamp with time zone AS training_end,
    NULL::real AS training_duration_seconds,
    NULL::integer AS train_start_season,
    NULL::integer AS train_end_season,
    NULL::integer AS test_start_season,
    NULL::integer AS test_end_season,
    NULL::integer AS train_samples,
    NULL::integer AS test_samples,
    NULL::text AS hardware,
    NULL::real AS gpu_memory_gb,
    NULL::text AS git_commit_hash,
    NULL::text AS git_branch,
    NULL::timestamp with time zone AS created_at,
    NULL::timestamp with time zone AS updated_at,
    NULL::text AS description,
    NULL::text AS notes,
    NULL::bigint AS total_runs,
    NULL::timestamp with time zone AS last_training_run,
    NULL::boolean AS is_deployed;


--
-- Name: bayesian_update_log; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.bayesian_update_log (
    log_id bigint NOT NULL,
    update_type character varying(30) NOT NULL,
    player_id character varying(50),
    game_id character varying(20),
    stat_type character varying(50) NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    prior_params jsonb NOT NULL,
    likelihoods jsonb NOT NULL,
    posterior_params jsonb NOT NULL,
    actual_value numeric(10,2),
    ensemble_prediction numeric(10,2),
    prediction_error numeric(10,2),
    abs_error numeric(10,2),
    component_predictions jsonb,
    update_magnitude numeric(10,6),
    learning_rate_used numeric(6,4) DEFAULT 1.0,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT valid_update_type CHECK (((update_type)::text = ANY (ARRAY[('player_prior'::character varying)::text, ('ensemble_weight'::character varying)::text, ('dirichlet_alpha'::character varying)::text])))
);


--
-- Name: TABLE bayesian_update_log; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.bayesian_update_log IS 'Audit trail for Bayesian updates. Recommended: Archive logs older than 1 year to cold storage.';


--
-- Name: COLUMN bayesian_update_log.prior_params; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.bayesian_update_log.prior_params IS 'Prior distribution parameters as JSON (e.g., {"alpha": 10, "beta": 5} for Beta, {"mu": 250, "sigma": 30} for Normal)';


--
-- Name: COLUMN bayesian_update_log.likelihoods; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.bayesian_update_log.likelihoods IS 'Likelihood contributions by component (e.g., {"hierarchical": 0.85, "xgboost": 0.72})';


--
-- Name: COLUMN bayesian_update_log.posterior_params; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.bayesian_update_log.posterior_params IS 'Posterior distribution parameters after Bayesian update';


--
-- Name: COLUMN bayesian_update_log.update_magnitude; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.bayesian_update_log.update_magnitude IS 'Measure of how much the posterior shifted (KL divergence or similar)';


--
-- Name: bayesian_update_log_log_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.bayesian_update_log_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bayesian_update_log_log_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.bayesian_update_log_log_id_seq OWNED BY ml.bayesian_update_log.log_id;


--
-- Name: bayesian_update_stats; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.bayesian_update_stats AS
 SELECT stat_type,
    update_type,
    count(*) AS n_updates,
    avg(abs_error) AS avg_abs_error,
    percentile_cont((0.5)::double precision) WITHIN GROUP (ORDER BY ((abs_error)::double precision)) AS median_abs_error,
    percentile_cont((0.9)::double precision) WITHIN GROUP (ORDER BY ((abs_error)::double precision)) AS p90_abs_error,
    min(created_at) AS first_update,
    max(created_at) AS last_update
   FROM ml.bayesian_update_log
  WHERE (abs_error IS NOT NULL)
  GROUP BY stat_type, update_type
  ORDER BY stat_type, update_type;


--
-- Name: dirichlet_alpha_by_stat_type; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.dirichlet_alpha_by_stat_type (
    alpha_id integer NOT NULL,
    stat_type character varying(50) NOT NULL,
    component_name character varying(100) NOT NULL,
    component_order integer DEFAULT 0 NOT NULL,
    alpha numeric(12,6) NOT NULL,
    prior_alpha numeric(12,6) NOT NULL,
    current_weight numeric(8,6) GENERATED ALWAYS AS (NULL::numeric) STORED,
    n_updates integer DEFAULT 0,
    cumulative_likelihood numeric(20,10) DEFAULT 0,
    last_mae numeric(10,4),
    last_rmse numeric(10,4),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT positive_alpha CHECK ((alpha > (0)::numeric)),
    CONSTRAINT positive_prior_alpha CHECK ((prior_alpha > (0)::numeric))
);


--
-- Name: TABLE dirichlet_alpha_by_stat_type; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.dirichlet_alpha_by_stat_type IS 'Per-stat-type Dirichlet concentration parameters for adaptive ensemble stacking';


--
-- Name: COLUMN dirichlet_alpha_by_stat_type.alpha; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.alpha IS 'Current Dirichlet concentration parameter (updated via Bayesian learning)';


--
-- Name: COLUMN dirichlet_alpha_by_stat_type.prior_alpha; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.prior_alpha IS 'Original prior alpha (from grid search or expert knowledge)';


--
-- Name: COLUMN dirichlet_alpha_by_stat_type.n_updates; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.n_updates IS 'Number of Bayesian updates applied to this component';


--
-- Name: current_dirichlet_weights; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.current_dirichlet_weights AS
 WITH totals AS (
         SELECT dirichlet_alpha_by_stat_type.stat_type,
            sum(dirichlet_alpha_by_stat_type.alpha) AS total_alpha
           FROM ml.dirichlet_alpha_by_stat_type
          GROUP BY dirichlet_alpha_by_stat_type.stat_type
        )
 SELECT d.stat_type,
    d.component_name,
    d.alpha,
    d.prior_alpha,
    ((d.alpha / t.total_alpha))::numeric(8,6) AS weight,
    ((d.prior_alpha / t.total_alpha))::numeric(8,6) AS prior_weight,
    d.n_updates,
    d.updated_at
   FROM (ml.dirichlet_alpha_by_stat_type d
     JOIN totals t ON (((d.stat_type)::text = (t.stat_type)::text)))
  ORDER BY d.stat_type, d.component_order;


--
-- Name: dirichlet_alpha_by_stat_type_alpha_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.dirichlet_alpha_by_stat_type_alpha_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: dirichlet_alpha_by_stat_type_alpha_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.dirichlet_alpha_by_stat_type_alpha_id_seq OWNED BY ml.dirichlet_alpha_by_stat_type.alpha_id;


--
-- Name: ensemble_configs; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.ensemble_configs (
    ensemble_id integer NOT NULL,
    ensemble_name text NOT NULL,
    version text NOT NULL,
    model_ids integer[] NOT NULL,
    weighting_method text NOT NULL,
    model_weights jsonb,
    meta_learner_type text,
    meta_learner_params jsonb,
    performance_metrics jsonb,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE ensemble_configs; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.ensemble_configs IS 'Ensemble model configurations and weights';


--
-- Name: ensemble_configs_ensemble_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.ensemble_configs_ensemble_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ensemble_configs_ensemble_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.ensemble_configs_ensemble_id_seq OWNED BY ml.ensemble_configs.ensemble_id;


--
-- Name: game_outcome_predictions; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.game_outcome_predictions (
    id integer NOT NULL,
    prediction_run_id text NOT NULL,
    game_id text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    home_team text NOT NULL,
    away_team text NOT NULL,
    kickoff timestamp with time zone,
    pred_home_spread double precision,
    pred_total double precision,
    pred_home_win_prob double precision,
    spread_std double precision,
    total_std double precision,
    vegas_spread double precision,
    vegas_total double precision,
    actual_home_score integer,
    actual_away_score integer,
    actual_spread double precision,
    actual_total double precision,
    model_version text,
    inference_time_ms double precision,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: game_outcome_predictions_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.game_outcome_predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: game_outcome_predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.game_outcome_predictions_id_seq OWNED BY ml.game_outcome_predictions.id;


--
-- Name: hyperparameter_searches; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.hyperparameter_searches (
    search_id integer NOT NULL,
    model_type text NOT NULL,
    search_method text NOT NULL,
    search_space jsonb NOT NULL,
    num_trials integer,
    max_duration_hours real,
    best_trial_id integer,
    best_params jsonb,
    best_metric_value real,
    best_metric_name text,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    status text,
    trials jsonb,
    convergence_plot_path text,
    importance_plot_path text
);


--
-- Name: TABLE hyperparameter_searches; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.hyperparameter_searches IS 'Hyperparameter optimization search results';


--
-- Name: hyperparameter_searches_search_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.hyperparameter_searches_search_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: hyperparameter_searches_search_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.hyperparameter_searches_search_id_seq OWNED BY ml.hyperparameter_searches.search_id;


--
-- Name: stat_type_performance; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.stat_type_performance (
    performance_id integer NOT NULL,
    model_id integer NOT NULL,
    model_name character varying(100) NOT NULL,
    stat_type character varying(50) NOT NULL,
    stat_category character varying(20) DEFAULT 'player_prop'::character varying NOT NULL,
    season integer NOT NULL,
    week_start integer NOT NULL,
    week_end integer NOT NULL,
    mae numeric(10,4),
    rmse numeric(10,4),
    mape numeric(8,4),
    r_squared numeric(6,4),
    calibration_50 numeric(5,4),
    calibration_90 numeric(5,4),
    calibration_95 numeric(5,4),
    brier_score numeric(8,6),
    n_bets integer DEFAULT 0,
    n_wins integer DEFAULT 0,
    win_rate numeric(5,4),
    roi numeric(10,4),
    units_won numeric(10,2),
    n_predictions integer NOT NULL,
    n_with_outcomes integer DEFAULT 0,
    computed_at timestamp with time zone DEFAULT now(),
    notes text,
    CONSTRAINT valid_calibration CHECK ((((calibration_50 IS NULL) OR ((calibration_50 >= (0)::numeric) AND (calibration_50 <= (1)::numeric))) AND ((calibration_90 IS NULL) OR ((calibration_90 >= (0)::numeric) AND (calibration_90 <= (1)::numeric))) AND ((calibration_95 IS NULL) OR ((calibration_95 >= (0)::numeric) AND (calibration_95 <= (1)::numeric))))),
    CONSTRAINT valid_season CHECK (((season >= 2000) AND (season <= 2100))),
    CONSTRAINT valid_week_range CHECK (((week_start <= week_end) AND (week_start >= 0) AND (week_end <= 22)))
);


--
-- Name: TABLE stat_type_performance; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.stat_type_performance IS 'Per-model, per-stat-type performance metrics over time windows. Used for adaptive stacking weights.';


--
-- Name: COLUMN stat_type_performance.stat_type; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.stat_type_performance.stat_type IS 'Type of stat: passing_yards, rushing_yards, receiving_yards, spread, total, win_prob';


--
-- Name: COLUMN stat_type_performance.stat_category; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.stat_type_performance.stat_category IS 'Category: player_prop for player predictions, game_outcome for spread/total/win_prob';


--
-- Name: COLUMN stat_type_performance.calibration_90; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON COLUMN ml.stat_type_performance.calibration_90 IS 'Fraction of predictions within 90% confidence interval';


--
-- Name: latest_stat_type_performance; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.latest_stat_type_performance AS
 SELECT DISTINCT ON (model_id, stat_type) performance_id,
    model_id,
    model_name,
    stat_type,
    stat_category,
    season,
    week_start,
    week_end,
    mae,
    rmse,
    calibration_90,
    n_predictions,
    n_with_outcomes,
    win_rate,
    roi,
    computed_at
   FROM ml.stat_type_performance
  ORDER BY model_id, stat_type, computed_at DESC;


--
-- Name: model_artifacts; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_artifacts (
    artifact_id integer NOT NULL,
    model_id integer,
    run_id integer,
    artifact_type text NOT NULL,
    artifact_name text NOT NULL,
    file_path text NOT NULL,
    file_size_bytes bigint,
    epoch integer,
    is_best boolean DEFAULT false,
    storage_backend text DEFAULT 'local'::text,
    cloud_path text,
    md5_hash text,
    sha256_hash text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE model_artifacts; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_artifacts IS 'Tracking for model files, checkpoints, and artifacts';


--
-- Name: model_artifacts_artifact_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.model_artifacts_artifact_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_artifacts_artifact_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.model_artifacts_artifact_id_seq OWNED BY ml.model_artifacts.artifact_id;


--
-- Name: model_evaluations; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_evaluations (
    evaluation_id integer NOT NULL,
    model_id integer,
    evaluation_name text NOT NULL,
    evaluation_type text NOT NULL,
    evaluated_at timestamp with time zone DEFAULT now(),
    dataset_name text,
    dataset_size integer,
    date_range_start date,
    date_range_end date,
    metrics jsonb NOT NULL,
    baseline_model_id integer,
    improvement_pct real,
    p_value real,
    confidence_interval jsonb,
    notes text
);


--
-- Name: TABLE model_evaluations; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_evaluations IS 'Model evaluation results on different datasets';


--
-- Name: model_evaluations_evaluation_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.model_evaluations_evaluation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_evaluations_evaluation_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.model_evaluations_evaluation_id_seq OWNED BY ml.model_evaluations.evaluation_id;


--
-- Name: model_registry; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_registry (
    model_id integer NOT NULL,
    model_name text NOT NULL,
    model_type text NOT NULL,
    model_family text NOT NULL,
    version text NOT NULL,
    model_path text NOT NULL,
    checkpoint_dir text,
    metadata_path text,
    status text DEFAULT 'training'::text NOT NULL,
    is_production boolean DEFAULT false,
    training_start timestamp with time zone,
    training_end timestamp with time zone,
    training_duration_seconds real,
    train_start_season integer,
    train_end_season integer,
    test_start_season integer,
    test_end_season integer,
    train_samples integer,
    test_samples integer,
    hardware text,
    gpu_memory_gb real,
    git_commit_hash text,
    git_branch text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    description text,
    notes text
);


--
-- Name: TABLE model_registry; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_registry IS 'Central registry for all ML models';


--
-- Name: model_leaderboard; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.model_leaderboard AS
 SELECT mr.model_name,
    mr.model_type,
    mr.version,
    mr.status,
    (me.metrics -> 'test_auc'::text) AS test_auc,
    (me.metrics -> 'test_accuracy'::text) AS test_accuracy,
    (me.metrics -> 'mae'::text) AS mae,
    (me.metrics -> 'rmse'::text) AS rmse,
    me.evaluated_at,
    mr.is_production
   FROM (ml.model_registry mr
     JOIN ml.model_evaluations me ON ((mr.model_id = me.model_id)))
  WHERE (me.evaluation_type = 'test'::text)
  ORDER BY me.evaluated_at DESC;


--
-- Name: model_rankings_by_stat_type; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.model_rankings_by_stat_type AS
 SELECT stat_type,
    model_id,
    model_name,
    mae,
    rmse,
    calibration_90,
    n_predictions,
    rank() OVER (PARTITION BY stat_type ORDER BY mae) AS mae_rank,
    rank() OVER (PARTITION BY stat_type ORDER BY calibration_90 DESC) AS calibration_rank
   FROM ml.latest_stat_type_performance
  WHERE (n_predictions >= 10);


--
-- Name: model_registry_model_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.model_registry_model_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_registry_model_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.model_registry_model_id_seq OWNED BY ml.model_registry.model_id;


--
-- Name: player_bayesian_history; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.player_bayesian_history AS
 SELECT player_id,
    stat_type,
    count(*) AS n_updates,
    avg(abs_error) AS avg_abs_error,
    stddev(prediction_error) AS prediction_volatility,
    avg(prediction_error) AS avg_bias,
    max(created_at) AS last_update
   FROM ml.bayesian_update_log
  WHERE ((player_id IS NOT NULL) AND (abs_error IS NOT NULL))
  GROUP BY player_id, stat_type
 HAVING (count(*) >= 3);


--
-- Name: production_deployments; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.production_deployments (
    deployment_id integer NOT NULL,
    model_id integer,
    deployment_name text NOT NULL,
    environment text NOT NULL,
    deployed_at timestamp with time zone DEFAULT now(),
    undeployed_at timestamp with time zone,
    traffic_percentage real DEFAULT 100.0,
    total_predictions bigint DEFAULT 0,
    avg_latency_ms real,
    error_rate_pct real,
    monitoring_enabled boolean DEFAULT true,
    alert_threshold jsonb,
    status text DEFAULT 'active'::text NOT NULL,
    rollback_reason text,
    notes text
);


--
-- Name: TABLE production_deployments; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.production_deployments IS 'Production deployment tracking and monitoring';


--
-- Name: production_deployments_deployment_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.production_deployments_deployment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: production_deployments_deployment_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.production_deployments_deployment_id_seq OWNED BY ml.production_deployments.deployment_id;


--
-- Name: recent_bayesian_updates; Type: VIEW; Schema: ml; Owner: -
--

CREATE VIEW ml.recent_bayesian_updates AS
 SELECT log_id,
    update_type,
    stat_type,
    player_id,
    game_id,
    season,
    week,
    actual_value,
    ensemble_prediction,
    abs_error,
    created_at
   FROM ml.bayesian_update_log
  ORDER BY created_at DESC
 LIMIT 100;


--
-- Name: stat_type_performance_performance_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.stat_type_performance_performance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: stat_type_performance_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.stat_type_performance_performance_id_seq OWNED BY ml.stat_type_performance.performance_id;


--
-- Name: training_runs; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.training_runs (
    run_id integer NOT NULL,
    model_id integer,
    run_name text,
    started_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    duration_seconds real,
    status text DEFAULT 'running'::text NOT NULL,
    exit_code integer,
    hyperparameters jsonb,
    epochs integer,
    batch_size integer,
    learning_rate real,
    optimizer text,
    hardware_used text,
    gpu_utilization_avg real,
    gpu_memory_peak_gb real,
    final_metrics jsonb,
    best_epoch integer,
    checkpoint_saved boolean DEFAULT false,
    log_file_path text,
    error_message text
);


--
-- Name: TABLE training_runs; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.training_runs IS 'Individual training runs with hyperparameters and results';


--
-- Name: training_runs_run_id_seq; Type: SEQUENCE; Schema: ml; Owner: -
--

CREATE SEQUENCE ml.training_runs_run_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: training_runs_run_id_seq; Type: SEQUENCE OWNED BY; Schema: ml; Owner: -
--

ALTER SEQUENCE ml.training_runs_run_id_seq OWNED BY ml.training_runs.run_id;


--
-- Name: api_usage_tracker; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.api_usage_tracker (
    month date NOT NULL,
    api_name text NOT NULL,
    quota_limit integer DEFAULT 500,
    calls_made integer DEFAULT 0,
    last_call_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: coaching_changes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.coaching_changes (
    team text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    head_coach text,
    head_coach_fired boolean DEFAULT false,
    head_coach_tenure_weeks integer,
    oc_name text,
    oc_changed boolean DEFAULT false,
    oc_tenure_weeks integer,
    scheme text,
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT coaching_changes_scheme_check CHECK ((scheme = ANY (ARRAY['West_Coast'::text, 'Air_Raid'::text, 'Run_Heavy'::text, 'Spread'::text, 'Pro_Style'::text, 'RPO_Heavy'::text, 'Unknown'::text]))),
    CONSTRAINT coaching_changes_season_check CHECK (((season >= 2010) AND (season <= 2030))),
    CONSTRAINT coaching_changes_week_check CHECK (((week >= 1) AND (week <= 18)))
);


--
-- Name: TABLE coaching_changes; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.coaching_changes IS 'Coaching changes by team-season-week for IV causal analysis of scheme effects';


--
-- Name: COLUMN coaching_changes.head_coach_fired; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.coaching_changes.head_coach_fired IS 'Exogenous instrument for IV analysis (owner decision)';


--
-- Name: COLUMN coaching_changes.oc_changed; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.coaching_changes.oc_changed IS 'Treatment variable for IV analysis (scheme change)';


--
-- Name: COLUMN coaching_changes.scheme; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.coaching_changes.scheme IS 'Offensive scheme classification for heterogeneous treatment effects';


--
-- Name: coaching_staff; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.coaching_staff (
    id integer NOT NULL,
    team text NOT NULL,
    season integer NOT NULL,
    week integer,
    head_coach text,
    offensive_coordinator text,
    defensive_coordinator text,
    special_teams_coordinator text,
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT coaching_staff_season_check CHECK (((season >= 2002) AND (season <= 2030))),
    CONSTRAINT coaching_staff_week_check CHECK (((week IS NULL) OR ((week >= 1) AND (week <= 18))))
);


--
-- Name: coaching_staff_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.coaching_staff_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: coaching_staff_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.coaching_staff_id_seq OWNED BY public.coaching_staff.id;


--
-- Name: combine; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.combine (
    combine_id integer NOT NULL,
    season integer NOT NULL,
    draft_year integer,
    draft_team text,
    draft_round integer,
    draft_ovr integer,
    pfr_id text,
    cfb_id text,
    player_name text NOT NULL,
    pos text,
    school text,
    ht text,
    wt integer,
    forty numeric,
    bench integer,
    vertical integer,
    broad_jump integer,
    cone numeric,
    shuttle numeric,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: combine_combine_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.combine_combine_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: combine_combine_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.combine_combine_id_seq OWNED BY public.combine.combine_id;


--
-- Name: contracts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.contracts (
    otc_id text NOT NULL,
    player text NOT NULL,
    "position" text,
    team text,
    is_active boolean,
    year_signed integer,
    years integer,
    value numeric,
    apy numeric,
    guaranteed numeric,
    apy_cap_pct double precision,
    inflated_value bigint,
    inflated_apy bigint,
    inflated_guaranteed bigint,
    player_page text,
    gsis_id text,
    date_of_birth date,
    height text,
    weight integer,
    college text,
    draft_year integer,
    draft_round integer,
    draft_overall integer,
    draft_team text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: data_quality_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.data_quality_log (
    id integer NOT NULL,
    table_name text NOT NULL,
    column_name text,
    issue_type text NOT NULL,
    issue_description text,
    row_count integer,
    detected_at timestamp without time zone DEFAULT now(),
    resolved_at timestamp without time zone,
    status text DEFAULT 'open'::text
);


--
-- Name: data_quality_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.data_quality_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: data_quality_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.data_quality_log_id_seq OWNED BY public.data_quality_log.id;


--
-- Name: depth_charts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.depth_charts (
    season integer NOT NULL,
    season_type text NOT NULL,
    week integer NOT NULL,
    club_code text NOT NULL,
    gsis_id text NOT NULL,
    "position" text NOT NULL,
    depth_team text NOT NULL,
    jersey_number integer,
    last_name text,
    first_name text,
    full_name text,
    formation text NOT NULL,
    depth_position integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE depth_charts; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.depth_charts IS 'Weekly NFL depth charts - official position designations and starter status';


--
-- Name: COLUMN depth_charts.formation; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.depth_charts.formation IS 'Defensive formation (Base, Nickel, Dime) or offensive package';


--
-- Name: COLUMN depth_charts.depth_position; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.depth_charts.depth_position IS '1=starter, 2=second string, 3=third string, etc.';


--
-- Name: draft_picks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.draft_picks (
    season integer NOT NULL,
    round integer NOT NULL,
    pick integer NOT NULL,
    team text NOT NULL,
    gsis_id text,
    pfr_player_id text,
    cfb_player_id text,
    pfr_player_name text,
    hof boolean,
    "position" text,
    category text,
    side text,
    college text,
    age double precision,
    to_season integer,
    allpro integer,
    probowls integer,
    seasons_started integer,
    w_av integer,
    car_av integer,
    dr_av integer,
    games integer,
    pass_completions integer,
    pass_attempts integer,
    pass_yards integer,
    pass_tds integer,
    pass_ints integer,
    rush_atts integer,
    rush_yards integer,
    rush_tds integer,
    receptions integer,
    rec_yards integer,
    rec_tds integer,
    def_solo_tackles integer,
    def_ints integer,
    def_sacks double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: espn_qbr; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.espn_qbr (
    player_id text NOT NULL,
    season integer NOT NULL,
    game_week integer NOT NULL,
    season_type text NOT NULL,
    name_display text,
    team_abb text,
    qbr_total double precision,
    qbr_raw double precision,
    rank integer,
    qualified boolean,
    pts_added double precision,
    epa_total double precision,
    qb_plays integer,
    pass double precision,
    run double precision,
    sack double precision,
    penalty double precision,
    exp_sack double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE espn_qbr; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.espn_qbr IS 'ESPN Total QBR - Weekly QB performance ratings (2006-present)';


--
-- Name: COLUMN espn_qbr.qbr_total; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.espn_qbr.qbr_total IS 'Total QBR on 0-100 scale - accounts for situation, difficulty, and clutch factor';


--
-- Name: COLUMN espn_qbr.qualified; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.espn_qbr.qualified IS 'TRUE if QB met minimum play threshold for ranking';


--
-- Name: ftn_charting; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ftn_charting (
    nflverse_game_id text NOT NULL,
    nflverse_play_id integer NOT NULL,
    ftn_game_id text,
    season integer NOT NULL,
    week integer NOT NULL,
    ftn_play_id text,
    starting_hash text,
    qb_location text,
    n_offense_backfield integer,
    n_defense_box integer,
    is_no_huddle boolean,
    is_motion boolean,
    is_play_action boolean,
    is_screen_pass boolean,
    is_rpo boolean,
    is_trick_play boolean,
    is_qb_out_of_pocket boolean,
    is_interception_worthy boolean,
    is_throw_away boolean,
    read_thrown text,
    is_catchable_ball boolean,
    is_contested_ball boolean,
    is_created_reception boolean,
    is_drop boolean,
    is_qb_sneak boolean,
    n_blitzers integer,
    n_pass_rushers integer,
    is_qb_fault_sack boolean,
    date_pulled date,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: injuries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.injuries (
    season integer NOT NULL,
    game_type character varying(10) NOT NULL,
    team character varying(3) NOT NULL,
    week integer NOT NULL,
    gsis_id character varying(20) NOT NULL,
    "position" character varying(10),
    full_name character varying(100),
    first_name character varying(50),
    last_name character varying(50),
    report_primary_injury text,
    report_secondary_injury text,
    report_status character varying(50),
    practice_primary_injury text,
    practice_secondary_injury text,
    practice_status character varying(50),
    date_modified timestamp with time zone
);


--
-- Name: TABLE injuries; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.injuries IS 'NFL injury reports from nflreadr::load_injuries(). Schema matches R ingestion script.';


--
-- Name: international_games; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.international_games (
    id integer NOT NULL,
    game_id character varying(50) NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    kickoff timestamp with time zone NOT NULL,
    home_team character varying(3) NOT NULL,
    away_team character varying(3) NOT NULL,
    stadium character varying(100),
    home_score integer,
    away_score integer,
    total_points integer,
    spread_close numeric(5,1),
    total_close numeric(5,1),
    kickoff_hour_utc numeric(5,1),
    city character varying(50),
    player_id character varying(20),
    passing_yards integer,
    attempts integer,
    completions numeric(5,1),
    touchdowns integer,
    interceptions numeric(3,1),
    player_team character varying(3),
    position_group character varying(10),
    player_name character varying(100),
    stadium_timezone_offset integer,
    stadium_altitude_ft integer,
    stadium_lat numeric(10,6),
    stadium_lon numeric(10,6),
    team_timezone_offset integer,
    team_altitude_ft integer,
    timezone_shift integer,
    jet_lag_direction character varying(20),
    jet_lag_penalty numeric(10,2),
    body_clock_hour numeric(5,1),
    early_morning_penalty numeric(10,2),
    altitude_shock integer,
    altitude_penalty numeric(10,4),
    denver_bonus numeric(10,2),
    total_intl_penalty numeric(10,2),
    is_international integer,
    neutral_field integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE international_games; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.international_games IS 'International NFL games with jet lag, timezone, and altitude adjustment penalties for predictive modeling';


--
-- Name: COLUMN international_games.jet_lag_penalty; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.international_games.jet_lag_penalty IS 'Penalty score for timezone shift effects on player performance';


--
-- Name: COLUMN international_games.total_intl_penalty; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.international_games.total_intl_penalty IS 'Combined penalty from jet lag, altitude, and early morning effects';


--
-- Name: international_games_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.international_games_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: international_games_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.international_games_id_seq OWNED BY public.international_games.id;


--
-- Name: mv_betting_features; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.mv_betting_features AS
 WITH betting_base AS (
         SELECT g.game_id,
            g.season,
            g.week,
            g.game_type,
            g.kickoff,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            (g.home_score - g.away_score) AS actual_margin,
            g.spread_close,
            g.total_close,
            (((g.home_score - g.away_score))::double precision - g.spread_close) AS home_cover_margin,
                CASE
                    WHEN ((((g.home_score - g.away_score))::double precision - g.spread_close) > (0)::double precision) THEN 1
                    WHEN ((((g.home_score - g.away_score))::double precision - g.spread_close) < (0)::double precision) THEN 0
                    ELSE NULL::integer
                END AS home_covered,
            (((g.home_score + g.away_score))::double precision - g.total_close) AS total_margin,
                CASE
                    WHEN (((g.home_score + g.away_score))::double precision > g.total_close) THEN 1
                    WHEN (((g.home_score + g.away_score))::double precision < g.total_close) THEN 0
                    ELSE NULL::integer
                END AS over_hit
           FROM public.games g
          WHERE ((g.home_score IS NOT NULL) AND (g.spread_close IS NOT NULL))
        ), team_betting_history AS (
         SELECT bb_1.home_team AS team,
            bb_1.season,
            bb_1.week,
            bb_1.game_id,
            bb_1.kickoff,
            true AS is_home,
            bb_1.home_covered AS covered,
            bb_1.over_hit,
            avg((bb_1.home_covered)::numeric) OVER (PARTITION BY bb_1.home_team, bb_1.season ORDER BY bb_1.week ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cover_rate_l10,
            avg((bb_1.over_hit)::numeric) OVER (PARTITION BY bb_1.home_team, bb_1.season ORDER BY bb_1.week ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS over_rate_l10
           FROM betting_base bb_1
        UNION ALL
         SELECT bb_1.away_team AS team,
            bb_1.season,
            bb_1.week,
            bb_1.game_id,
            bb_1.kickoff,
            false AS is_home,
                CASE
                    WHEN (bb_1.home_covered IS NOT NULL) THEN (1 - bb_1.home_covered)
                    ELSE NULL::integer
                END AS covered,
            bb_1.over_hit,
            avg(((1 - bb_1.home_covered))::numeric) OVER (PARTITION BY bb_1.away_team, bb_1.season ORDER BY bb_1.week ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cover_rate_l10,
            avg((bb_1.over_hit)::numeric) OVER (PARTITION BY bb_1.away_team, bb_1.season ORDER BY bb_1.week ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS over_rate_l10
           FROM betting_base bb_1
        )
 SELECT bb.game_id,
    bb.season,
    bb.week,
    bb.game_type,
    bb.kickoff,
    bb.home_team,
    bb.away_team,
    bb.home_score,
    bb.away_score,
    bb.actual_margin,
    bb.spread_close,
    bb.total_close,
    bb.home_cover_margin,
    bb.home_covered,
    bb.total_margin,
    bb.over_hit,
    th_home.cover_rate_l10 AS home_cover_rate_l10,
    th_home.over_rate_l10 AS home_over_rate_l10,
    th_away.cover_rate_l10 AS away_cover_rate_l10,
    th_away.over_rate_l10 AS away_over_rate_l10,
    now() AS refreshed_at
   FROM ((betting_base bb
     LEFT JOIN team_betting_history th_home ON (((bb.game_id = th_home.game_id) AND (bb.home_team = th_home.team) AND (th_home.is_home = true))))
     LEFT JOIN team_betting_history th_away ON (((bb.game_id = th_away.game_id) AND (bb.away_team = th_away.team) AND (th_away.is_home = false))))
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW mv_betting_features; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON MATERIALIZED VIEW public.mv_betting_features IS 'Completed games only: closing spread/total (nflverse sign, positive = home favored),
ATS and over/under outcomes, and each team''s cover/over rate over its previous
(up to 10) games at the same venue within the season. No line movement or implied
probabilities. Week 1 rates are NULL.';


--
-- Name: officials; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.officials (
    game_id text NOT NULL,
    game_key text,
    official_name text NOT NULL,
    "position" text NOT NULL,
    jersey_number integer,
    official_id text,
    season integer NOT NULL,
    season_type text NOT NULL,
    week integer NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: participation; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.participation (
    nflverse_game_id text NOT NULL,
    old_game_id text,
    play_id integer NOT NULL,
    possession_team text,
    offense_formation text,
    offense_personnel text,
    defenders_in_box integer,
    defense_personnel text,
    number_of_pass_rushers integer,
    players_on_play text,
    offense_players text,
    defense_players text,
    n_offense integer,
    n_defense integer,
    ngs_air_yards double precision,
    time_to_throw double precision,
    was_pressure boolean,
    route text,
    defense_man_zone_type text,
    defense_coverage_type text,
    offense_names text,
    defense_names text,
    offense_positions text,
    defense_positions text,
    offense_numbers text,
    defense_numbers text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: pfr_defense; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pfr_defense (
    pfr_player_id text NOT NULL,
    game_id text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    game_type text,
    pfr_player_name text,
    team text,
    opponent text,
    def_targets integer,
    def_completions_allowed integer,
    def_completion_pct double precision,
    def_yards_allowed integer,
    def_yards_allowed_per_cmp double precision,
    def_yards_allowed_per_tgt double precision,
    def_receiving_td_allowed integer,
    def_passer_rating_allowed double precision,
    def_adot double precision,
    def_air_yards_completed integer,
    def_yards_after_catch integer,
    def_times_blitzed integer,
    def_times_hurried integer,
    def_times_hitqb integer,
    def_sacks double precision,
    def_pressures integer,
    def_tackles_combined integer,
    def_missed_tackles integer,
    def_missed_tackle_pct double precision,
    def_ints integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE pfr_defense; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.pfr_defense IS 'Pro Football Reference Advanced Defense Stats - Weekly player-level defensive metrics';


--
-- Name: COLUMN pfr_defense.def_passer_rating_allowed; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pfr_defense.def_passer_rating_allowed IS 'Passer rating allowed when targeted in coverage (lower is better)';


--
-- Name: COLUMN pfr_defense.def_pressures; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pfr_defense.def_pressures IS 'Total QB pressures (sacks + hits + hurries)';


--
-- Name: COLUMN pfr_defense.def_missed_tackle_pct; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pfr_defense.def_missed_tackle_pct IS 'Percentage of tackle attempts that were missed';


--
-- Name: pfr_passing; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pfr_passing (
    pfr_player_id text NOT NULL,
    game_id text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    game_type text NOT NULL,
    pfr_game_id text,
    team text,
    opponent text,
    pfr_player_name text,
    passing_drops integer,
    passing_drop_pct double precision,
    receiving_drop integer,
    receiving_drop_pct double precision,
    passing_bad_throws integer,
    passing_bad_throw_pct double precision,
    times_sacked integer,
    times_blitzed integer,
    times_hurried integer,
    times_hit integer,
    times_pressured integer,
    times_pressured_pct double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: plays; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plays (
    game_id text NOT NULL,
    play_id integer NOT NULL,
    posteam text,
    defteam text,
    quarter integer,
    time_seconds integer,
    down integer,
    ydstogo integer,
    yardline_100 integer,
    play_type text,
    yards_gained integer,
    first_down boolean,
    pass boolean,
    rush boolean,
    special boolean,
    touchdown boolean,
    pass_touchdown boolean,
    rush_touchdown boolean,
    return_touchdown boolean,
    extra_point_attempt boolean,
    two_point_attempt boolean,
    field_goal_attempt boolean,
    kickoff_attempt boolean,
    punt_attempt boolean,
    fumble boolean,
    interception boolean,
    fumble_lost boolean,
    timeout boolean,
    penalty boolean,
    penalty_yards integer,
    qb_spike boolean,
    qb_kneel boolean,
    epa real,
    wp real,
    wpa real,
    vegas_wp real,
    vegas_wpa real,
    success boolean,
    air_yards integer,
    yards_after_catch integer,
    cpoe real,
    comp_air_epa real,
    comp_yac_epa real,
    complete_pass boolean,
    incomplete_pass boolean,
    pass_length text,
    pass_location text,
    qb_hit boolean,
    qb_scramble boolean,
    run_location text,
    run_gap text,
    passer_player_id text,
    passer_player_name text,
    rusher_player_id text,
    rusher_player_name text,
    receiver_player_id text,
    receiver_player_name text,
    sack boolean,
    shotgun boolean,
    no_huddle boolean,
    qb_dropback boolean,
    posteam_score integer,
    defteam_score integer,
    score_differential integer,
    posteam_score_post integer,
    defteam_score_post integer,
    score_differential_post integer,
    game_seconds_remaining integer,
    half_seconds_remaining integer,
    drive integer,
    fixed_drive integer,
    fixed_drive_result text,
    third_down_converted boolean,
    fourth_down_converted boolean,
    extra_point_result text,
    field_goal_result text,
    kick_distance integer,
    td_team text,
    first_down_pass boolean,
    first_down_rush boolean,
    goal_to_go boolean,
    penalty_team text,
    series integer,
    series_success boolean,
    series_result text,
    passing_yards double precision,
    receiving_yards double precision,
    rushing_yards double precision,
    pass_attempt boolean,
    rush_attempt boolean,
    return_yards double precision,
    tackled_for_loss boolean,
    out_of_bounds boolean,
    aborted_play boolean,
    drive_play_count integer,
    drive_time_of_possession text,
    drive_first_downs integer,
    drive_inside20 boolean,
    drive_ended_with_score boolean,
    drive_yards_penalized integer,
    order_sequence double precision,
    temp double precision,
    wind double precision,
    weather text,
    no_score_prob double precision,
    def_wp double precision,
    game_half text,
    qb_epa double precision,
    xyac_epa double precision,
    xyac_fd boolean,
    xyac_mean_yardage double precision,
    xyac_median_yardage double precision,
    xyac_success boolean,
    two_point_conv_result text,
    play_type_nfl text
);


--
-- Name: referee_assignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.referee_assignments (
    game_id text NOT NULL,
    referee_crew text NOT NULL,
    referee_name text,
    home_penalties integer,
    away_penalties integer,
    home_penalty_yards integer,
    away_penalty_yards integer,
    home_bias double precision,
    total_penalties integer GENERATED ALWAYS AS ((COALESCE(home_penalties, 0) + COALESCE(away_penalties, 0))) STORED,
    total_penalty_yards integer GENERATED ALWAYS AS ((COALESCE(home_penalty_yards, 0) + COALESCE(away_penalty_yards, 0))) STORED,
    penalty_yards_differential integer GENERATED ALWAYS AS ((COALESCE(home_penalty_yards, 0) - COALESCE(away_penalty_yards, 0))) STORED,
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT referee_assignments_away_penalties_check CHECK ((away_penalties >= 0)),
    CONSTRAINT referee_assignments_away_penalty_yards_check CHECK ((away_penalty_yards >= 0)),
    CONSTRAINT referee_assignments_home_bias_check CHECK (((home_bias >= ('-1.0'::numeric)::double precision) AND (home_bias <= (1.0)::double precision))),
    CONSTRAINT referee_assignments_home_penalties_check CHECK ((home_penalties >= 0)),
    CONSTRAINT referee_assignments_home_penalty_yards_check CHECK ((home_penalty_yards >= 0))
);


--
-- Name: TABLE referee_assignments; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.referee_assignments IS 'Referee crew assignments and penalty bias metrics for causal analysis';


--
-- Name: COLUMN referee_assignments.home_bias; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.referee_assignments.home_bias IS 'Normalized penalty bias: (home-away)/(home+away), range [-1,1]. Positive = home favored.';


--
-- Name: COLUMN referee_assignments.penalty_yards_differential; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.referee_assignments.penalty_yards_differential IS 'Home penalty yards minus away penalty yards (positive = home penalized more)';


--
-- Name: rosters; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.rosters AS
 SELECT season,
    week,
    team,
    gsis_id AS player_id,
    "position",
    depth_chart_position,
    jersey_number,
    status,
    full_name,
    football_name,
    first_name,
    last_name,
    birth_date,
    height,
    weight,
    college,
    espn_id,
    sportradar_id,
    yahoo_id,
    rotowire_id,
    pff_id,
    pfr_id,
    fantasy_data_id,
    sleeper_id,
    years_exp,
    headshot_url,
    ngs_position,
    status_description_abbr,
    esb_id,
    gsis_it_id,
    smart_id,
    entry_year,
    rookie_year,
    draft_club,
    draft_number,
    game_type
   FROM public.rosters_weekly;


--
-- Name: rule_changes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rule_changes (
    rule_id integer NOT NULL,
    year_implemented integer NOT NULL,
    description text NOT NULL,
    affected_positions text[],
    affected_play_types text[],
    treatment_indicator text NOT NULL,
    treatment_group_definition text,
    control_group_definition text,
    expected_effect_direction text,
    expected_effect_magnitude double precision,
    notes text,
    created_at timestamp without time zone DEFAULT now(),
    CONSTRAINT rule_changes_expected_effect_direction_check CHECK ((expected_effect_direction = ANY (ARRAY['positive'::text, 'negative'::text, 'unknown'::text]))),
    CONSTRAINT rule_changes_year_implemented_check CHECK (((year_implemented >= 2000) AND (year_implemented <= 2030)))
);


--
-- Name: TABLE rule_changes; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.rule_changes IS 'NFL rule changes for Difference-in-Differences (DiD) causal analysis';


--
-- Name: COLUMN rule_changes.treatment_indicator; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.rule_changes.treatment_indicator IS 'Column name for feature engineering (e.g., did_roughing_2018)';


--
-- Name: COLUMN rule_changes.treatment_group_definition; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.rule_changes.treatment_group_definition IS 'SQL WHERE clause defining treatment group for DiD';


--
-- Name: rule_changes_rule_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.rule_changes_rule_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: rule_changes_rule_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.rule_changes_rule_id_seq OWNED BY public.rule_changes.rule_id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying(100) NOT NULL,
    description text,
    applied_at timestamp with time zone DEFAULT now()
);


--
-- Name: semantic_news; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.semantic_news (
    id integer NOT NULL,
    game_id text,
    team text,
    season integer,
    week integer,
    headline text,
    body text,
    source text,
    published_at timestamp with time zone,
    sentiment_score real,
    injury_mentions integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: semantic_news_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.semantic_news_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: semantic_news_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.semantic_news_id_seq OWNED BY public.semantic_news.id;


--
-- Name: snap_counts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.snap_counts (
    pfr_player_id text NOT NULL,
    game_id text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    game_type text,
    player text,
    "position" text,
    team text,
    opponent text,
    offense_snaps integer,
    offense_pct double precision,
    defense_snaps integer,
    defense_pct double precision,
    st_snaps integer,
    st_pct double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: TABLE snap_counts; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.snap_counts IS 'Weekly player snap counts - offensive/defensive/ST participation rates';


--
-- Name: COLUMN snap_counts.offense_pct; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.snap_counts.offense_pct IS 'Percentage of offensive snaps played (0-100)';


--
-- Name: COLUMN snap_counts.defense_pct; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.snap_counts.defense_pct IS 'Percentage of defensive snaps played (0-100)';


--
-- Name: team_injury_load; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.team_injury_load (
    season integer NOT NULL,
    week integer NOT NULL,
    team text NOT NULL,
    total_injuries integer,
    players_out integer,
    players_questionable integer,
    players_doubtful integer,
    key_position_out boolean,
    qb_out boolean,
    oline_injuries integer,
    injury_severity_index real
);


--
-- Name: team_travel; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.team_travel (
    game_id text NOT NULL,
    home_team text NOT NULL,
    away_team text NOT NULL,
    travel_distance_miles double precision,
    time_zones_crossed integer,
    away_rest_days integer,
    home_rest_days integer,
    is_thursday_game boolean DEFAULT false,
    is_monday_game boolean DEFAULT false,
    is_international boolean DEFAULT false,
    international_location text,
    international_stadium text,
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT team_travel_away_rest_days_check CHECK (((away_rest_days >= 0) AND (away_rest_days <= 21))),
    CONSTRAINT team_travel_home_rest_days_check CHECK (((home_rest_days >= 0) AND (home_rest_days <= 21))),
    CONSTRAINT team_travel_time_zones_crossed_check CHECK (((time_zones_crossed >= '-3'::integer) AND (time_zones_crossed <= 3))),
    CONSTRAINT team_travel_travel_distance_miles_check CHECK ((travel_distance_miles >= (0)::double precision))
);


--
-- Name: TABLE team_travel; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.team_travel IS 'Travel and rest data for Propensity Score Matching (PSM) analysis of home advantage';


--
-- Name: COLUMN team_travel.travel_distance_miles; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.team_travel.travel_distance_miles IS 'Distance traveled by away team (for fatigue effect isolation)';


--
-- Name: COLUMN team_travel.time_zones_crossed; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.team_travel.time_zones_crossed IS 'Time zones crossed: positive=East, negative=West (West→East harder)';


--
-- Name: trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trades (
    trade_id integer NOT NULL,
    season integer NOT NULL,
    trade_date date,
    gave text NOT NULL,
    received text NOT NULL,
    pick_season integer,
    pick_round integer,
    pick_number integer,
    conditional boolean,
    pfr_id text,
    pfr_name text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: bayesian_update_log log_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.bayesian_update_log ALTER COLUMN log_id SET DEFAULT nextval('ml.bayesian_update_log_log_id_seq'::regclass);


--
-- Name: dirichlet_alpha_by_stat_type alpha_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.dirichlet_alpha_by_stat_type ALTER COLUMN alpha_id SET DEFAULT nextval('ml.dirichlet_alpha_by_stat_type_alpha_id_seq'::regclass);


--
-- Name: ensemble_configs ensemble_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ensemble_configs ALTER COLUMN ensemble_id SET DEFAULT nextval('ml.ensemble_configs_ensemble_id_seq'::regclass);


--
-- Name: game_outcome_predictions id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.game_outcome_predictions ALTER COLUMN id SET DEFAULT nextval('ml.game_outcome_predictions_id_seq'::regclass);


--
-- Name: hyperparameter_searches search_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.hyperparameter_searches ALTER COLUMN search_id SET DEFAULT nextval('ml.hyperparameter_searches_search_id_seq'::regclass);


--
-- Name: model_artifacts artifact_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_artifacts ALTER COLUMN artifact_id SET DEFAULT nextval('ml.model_artifacts_artifact_id_seq'::regclass);


--
-- Name: model_evaluations evaluation_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_evaluations ALTER COLUMN evaluation_id SET DEFAULT nextval('ml.model_evaluations_evaluation_id_seq'::regclass);


--
-- Name: model_registry model_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry ALTER COLUMN model_id SET DEFAULT nextval('ml.model_registry_model_id_seq'::regclass);


--
-- Name: production_deployments deployment_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.production_deployments ALTER COLUMN deployment_id SET DEFAULT nextval('ml.production_deployments_deployment_id_seq'::regclass);


--
-- Name: stat_type_performance performance_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.stat_type_performance ALTER COLUMN performance_id SET DEFAULT nextval('ml.stat_type_performance_performance_id_seq'::regclass);


--
-- Name: training_runs run_id; Type: DEFAULT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_runs ALTER COLUMN run_id SET DEFAULT nextval('ml.training_runs_run_id_seq'::regclass);


--
-- Name: coaching_staff id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coaching_staff ALTER COLUMN id SET DEFAULT nextval('public.coaching_staff_id_seq'::regclass);


--
-- Name: combine combine_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine ALTER COLUMN combine_id SET DEFAULT nextval('public.combine_combine_id_seq'::regclass);


--
-- Name: data_quality_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_quality_log ALTER COLUMN id SET DEFAULT nextval('public.data_quality_log_id_seq'::regclass);


--
-- Name: international_games id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.international_games ALTER COLUMN id SET DEFAULT nextval('public.international_games_id_seq'::regclass);


--
-- Name: rule_changes rule_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_changes ALTER COLUMN rule_id SET DEFAULT nextval('public.rule_changes_rule_id_seq'::regclass);


--
-- Name: semantic_news id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.semantic_news ALTER COLUMN id SET DEFAULT nextval('public.semantic_news_id_seq'::regclass);


--
-- Name: bayesian_player_ratings bayesian_player_ratings_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.bayesian_player_ratings
    ADD CONSTRAINT bayesian_player_ratings_pkey PRIMARY KEY (player_id, stat_type, season, model_version);


--
-- Name: position_groups position_groups_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.position_groups
    ADD CONSTRAINT position_groups_pkey PRIMARY KEY ("position");


--
-- Name: team_4th_down_features team_4th_down_features_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.team_4th_down_features
    ADD CONSTRAINT team_4th_down_features_pkey PRIMARY KEY (game_id, team);


--
-- Name: team_epa team_epa_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.team_epa
    ADD CONSTRAINT team_epa_pkey PRIMARY KEY (game_id, posteam);


--
-- Name: team_injury_load team_injury_load_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.team_injury_load
    ADD CONSTRAINT team_injury_load_pkey PRIMARY KEY (season, week, team);


--
-- Name: team_playoff_context team_playoff_context_pkey; Type: CONSTRAINT; Schema: mart; Owner: -
--

ALTER TABLE ONLY mart.team_playoff_context
    ADD CONSTRAINT team_playoff_context_pkey PRIMARY KEY (team, season, week);


--
-- Name: feature_freshness feature_freshness_pkey; Type: CONSTRAINT; Schema: metadata; Owner: -
--

ALTER TABLE ONLY metadata.feature_freshness
    ADD CONSTRAINT feature_freshness_pkey PRIMARY KEY (feature_name);


--
-- Name: bayesian_update_log bayesian_update_log_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.bayesian_update_log
    ADD CONSTRAINT bayesian_update_log_pkey PRIMARY KEY (log_id);


--
-- Name: dirichlet_alpha_by_stat_type dirichlet_alpha_by_stat_type_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.dirichlet_alpha_by_stat_type
    ADD CONSTRAINT dirichlet_alpha_by_stat_type_pkey PRIMARY KEY (alpha_id);


--
-- Name: dirichlet_alpha_by_stat_type dirichlet_alpha_by_stat_type_stat_type_component_name_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.dirichlet_alpha_by_stat_type
    ADD CONSTRAINT dirichlet_alpha_by_stat_type_stat_type_component_name_key UNIQUE (stat_type, component_name);


--
-- Name: ensemble_configs ensemble_configs_ensemble_name_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ensemble_configs
    ADD CONSTRAINT ensemble_configs_ensemble_name_key UNIQUE (ensemble_name);


--
-- Name: ensemble_configs ensemble_configs_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ensemble_configs
    ADD CONSTRAINT ensemble_configs_pkey PRIMARY KEY (ensemble_id);


--
-- Name: game_outcome_predictions game_outcome_predictions_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.game_outcome_predictions
    ADD CONSTRAINT game_outcome_predictions_pkey PRIMARY KEY (id);


--
-- Name: hyperparameter_searches hyperparameter_searches_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.hyperparameter_searches
    ADD CONSTRAINT hyperparameter_searches_pkey PRIMARY KEY (search_id);


--
-- Name: model_artifacts model_artifacts_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_artifacts
    ADD CONSTRAINT model_artifacts_pkey PRIMARY KEY (artifact_id);


--
-- Name: model_evaluations model_evaluations_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_evaluations
    ADD CONSTRAINT model_evaluations_pkey PRIMARY KEY (evaluation_id);


--
-- Name: model_registry model_registry_model_name_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry
    ADD CONSTRAINT model_registry_model_name_key UNIQUE (model_name);


--
-- Name: model_registry model_registry_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry
    ADD CONSTRAINT model_registry_pkey PRIMARY KEY (model_id);


--
-- Name: production_deployments production_deployments_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.production_deployments
    ADD CONSTRAINT production_deployments_pkey PRIMARY KEY (deployment_id);


--
-- Name: stat_type_performance stat_type_performance_model_id_stat_type_season_week_start__key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.stat_type_performance
    ADD CONSTRAINT stat_type_performance_model_id_stat_type_season_week_start__key UNIQUE (model_id, stat_type, season, week_start, week_end);


--
-- Name: stat_type_performance stat_type_performance_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.stat_type_performance
    ADD CONSTRAINT stat_type_performance_pkey PRIMARY KEY (performance_id);


--
-- Name: training_runs training_runs_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_runs
    ADD CONSTRAINT training_runs_pkey PRIMARY KEY (run_id);


--
-- Name: api_usage_tracker api_usage_tracker_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.api_usage_tracker
    ADD CONSTRAINT api_usage_tracker_pkey PRIMARY KEY (month, api_name);


--
-- Name: coaching_changes coaching_changes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coaching_changes
    ADD CONSTRAINT coaching_changes_pkey PRIMARY KEY (team, season, week);


--
-- Name: coaching_staff coaching_staff_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coaching_staff
    ADD CONSTRAINT coaching_staff_pkey PRIMARY KEY (id);


--
-- Name: combine combine_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine
    ADD CONSTRAINT combine_pkey PRIMARY KEY (combine_id);


--
-- Name: combine combine_season_player_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine
    ADD CONSTRAINT combine_season_player_name_key UNIQUE (season, player_name);


--
-- Name: contracts contracts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.contracts
    ADD CONSTRAINT contracts_pkey PRIMARY KEY (otc_id);


--
-- Name: data_quality_log data_quality_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.data_quality_log
    ADD CONSTRAINT data_quality_log_pkey PRIMARY KEY (id);


--
-- Name: depth_charts depth_charts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.depth_charts
    ADD CONSTRAINT depth_charts_pkey PRIMARY KEY (gsis_id, club_code, season, season_type, week, depth_team, formation, "position");


--
-- Name: draft_picks draft_picks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.draft_picks
    ADD CONSTRAINT draft_picks_pkey PRIMARY KEY (season, round, pick);


--
-- Name: espn_qbr espn_qbr_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.espn_qbr
    ADD CONSTRAINT espn_qbr_pkey PRIMARY KEY (player_id, season, game_week, season_type);


--
-- Name: ftn_charting ftn_charting_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ftn_charting
    ADD CONSTRAINT ftn_charting_pkey PRIMARY KEY (nflverse_game_id, nflverse_play_id);


--
-- Name: games games_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.games
    ADD CONSTRAINT games_pkey PRIMARY KEY (game_id);


--
-- Name: injuries injuries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.injuries
    ADD CONSTRAINT injuries_pkey PRIMARY KEY (season, game_type, team, week, gsis_id);


--
-- Name: international_games international_games_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.international_games
    ADD CONSTRAINT international_games_pkey PRIMARY KEY (id);


--
-- Name: nextgen_passing nextgen_passing_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nextgen_passing
    ADD CONSTRAINT nextgen_passing_pkey PRIMARY KEY (player_id, season, week);


--
-- Name: nextgen_receiving nextgen_receiving_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nextgen_receiving
    ADD CONSTRAINT nextgen_receiving_pkey PRIMARY KEY (player_id, season, week);


--
-- Name: nextgen_rushing nextgen_rushing_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nextgen_rushing
    ADD CONSTRAINT nextgen_rushing_pkey PRIMARY KEY (player_id, season, week);


--
-- Name: odds_history odds_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.odds_history
    ADD CONSTRAINT odds_history_pkey PRIMARY KEY (captured_at, game_id, bookmaker, market_type);


--
-- Name: officials officials_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.officials
    ADD CONSTRAINT officials_pkey PRIMARY KEY (game_id, official_name, "position");


--
-- Name: participation participation_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.participation
    ADD CONSTRAINT participation_pkey PRIMARY KEY (nflverse_game_id, play_id);


--
-- Name: pfr_defense pfr_defense_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pfr_defense
    ADD CONSTRAINT pfr_defense_pkey PRIMARY KEY (pfr_player_id, game_id);


--
-- Name: pfr_passing pfr_passing_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pfr_passing
    ADD CONSTRAINT pfr_passing_pkey PRIMARY KEY (pfr_player_id, game_id);


--
-- Name: players players_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.players
    ADD CONSTRAINT players_pkey PRIMARY KEY (player_id);


--
-- Name: plays plays_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plays
    ADD CONSTRAINT plays_pkey PRIMARY KEY (game_id, play_id);


--
-- Name: referee_assignments referee_assignments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.referee_assignments
    ADD CONSTRAINT referee_assignments_pkey PRIMARY KEY (game_id);


--
-- Name: rosters_weekly rosters_weekly_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rosters_weekly
    ADD CONSTRAINT rosters_weekly_pkey PRIMARY KEY (season, week, game_type, team, gsis_id);


--
-- Name: rule_changes rule_changes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_changes
    ADD CONSTRAINT rule_changes_pkey PRIMARY KEY (rule_id);


--
-- Name: rule_changes rule_changes_year_implemented_description_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_changes
    ADD CONSTRAINT rule_changes_year_implemented_description_key UNIQUE (year_implemented, description);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: semantic_news semantic_news_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.semantic_news
    ADD CONSTRAINT semantic_news_pkey PRIMARY KEY (id);


--
-- Name: snap_counts snap_counts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.snap_counts
    ADD CONSTRAINT snap_counts_pkey PRIMARY KEY (pfr_player_id, game_id);


--
-- Name: team_injury_load team_injury_load_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_injury_load
    ADD CONSTRAINT team_injury_load_pkey PRIMARY KEY (season, week, team);


--
-- Name: team_travel team_travel_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_travel
    ADD CONSTRAINT team_travel_pkey PRIMARY KEY (game_id);


--
-- Name: trades trades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.trades
    ADD CONSTRAINT trades_pkey PRIMARY KEY (trade_id, gave, received);


--
-- Name: weather weather_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weather
    ADD CONSTRAINT weather_pkey PRIMARY KEY (game_id);


--
-- Name: idx_4th_down_game; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_4th_down_game ON mart.team_4th_down_features USING btree (game_id);


--
-- Name: idx_4th_down_team; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_4th_down_team ON mart.team_4th_down_features USING btree (team);


--
-- Name: idx_bayesian_player_ratings_player; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_bayesian_player_ratings_player ON mart.bayesian_player_ratings USING btree (player_id);


--
-- Name: idx_bayesian_player_ratings_season; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_bayesian_player_ratings_season ON mart.bayesian_player_ratings USING btree (season);


--
-- Name: idx_bayesian_player_ratings_stat_type; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_bayesian_player_ratings_stat_type ON mart.bayesian_player_ratings USING btree (stat_type);


--
-- Name: idx_game_summary_game_id; Type: INDEX; Schema: mart; Owner: -
--

CREATE UNIQUE INDEX idx_game_summary_game_id ON mart.game_summary USING btree (game_id);


--
-- Name: idx_game_summary_is_final; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_game_summary_is_final ON mart.game_summary USING btree (is_final);


--
-- Name: idx_game_summary_season_week; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_game_summary_season_week ON mart.game_summary USING btree (season, week);


--
-- Name: idx_game_summary_teams; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_game_summary_teams ON mart.game_summary USING btree (home_team, away_team);


--
-- Name: idx_injury_qb_out; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_injury_qb_out ON mart.team_injury_load USING btree (qb_out) WHERE (qb_out = true);


--
-- Name: idx_injury_season_week; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_injury_season_week ON mart.team_injury_load USING btree (season, week);


--
-- Name: idx_injury_team_season; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_injury_team_season ON mart.team_injury_load USING btree (team, season);


--
-- Name: idx_player_game_stats_category; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_game_stats_category ON mart.player_game_stats USING btree (stat_category);


--
-- Name: idx_player_game_stats_player_id; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_game_stats_player_id ON mart.player_game_stats USING btree (player_id);


--
-- Name: idx_player_game_stats_position_group; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_game_stats_position_group ON mart.player_game_stats USING btree (position_group);


--
-- Name: idx_player_game_stats_season_week; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_game_stats_season_week ON mart.player_game_stats USING btree (season, week);


--
-- Name: idx_player_game_stats_unique; Type: INDEX; Schema: mart; Owner: -
--

CREATE UNIQUE INDEX idx_player_game_stats_unique ON mart.player_game_stats USING btree (player_id, season, week, stat_category);


--
-- Name: idx_player_hierarchy_player_id; Type: INDEX; Schema: mart; Owner: -
--

CREATE UNIQUE INDEX idx_player_hierarchy_player_id ON mart.player_hierarchy USING btree (player_id);


--
-- Name: idx_player_hierarchy_position_group; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_hierarchy_position_group ON mart.player_hierarchy USING btree (position_group);


--
-- Name: idx_player_hierarchy_team; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_player_hierarchy_team ON mart.player_hierarchy USING btree (current_team);


--
-- Name: idx_playoff_desperate; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_playoff_desperate ON mart.team_playoff_context USING btree (desperate) WHERE (desperate = true);


--
-- Name: idx_playoff_eliminated; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_playoff_eliminated ON mart.team_playoff_context USING btree (eliminated) WHERE (eliminated = true);


--
-- Name: idx_playoff_season_week; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_playoff_season_week ON mart.team_playoff_context USING btree (season, week);


--
-- Name: idx_playoff_team_season; Type: INDEX; Schema: mart; Owner: -
--

CREATE INDEX idx_playoff_team_season ON mart.team_playoff_context USING btree (team, season);


--
-- Name: idx_feature_freshness_checked_at; Type: INDEX; Schema: metadata; Owner: -
--

CREATE INDEX idx_feature_freshness_checked_at ON metadata.feature_freshness USING btree (checked_at DESC);


--
-- Name: idx_bul_created; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_created ON ml.bayesian_update_log USING btree (created_at DESC);


--
-- Name: idx_bul_game; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_game ON ml.bayesian_update_log USING btree (game_id);


--
-- Name: idx_bul_large_errors; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_large_errors ON ml.bayesian_update_log USING btree (abs_error DESC) WHERE (abs_error > (50)::numeric);


--
-- Name: idx_bul_player; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_player ON ml.bayesian_update_log USING btree (player_id) WHERE (player_id IS NOT NULL);


--
-- Name: idx_bul_season_week; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_season_week ON ml.bayesian_update_log USING btree (season, week);


--
-- Name: idx_bul_stat_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_stat_type ON ml.bayesian_update_log USING btree (stat_type);


--
-- Name: idx_bul_update_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_bul_update_type ON ml.bayesian_update_log USING btree (update_type);


--
-- Name: idx_das_component; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_das_component ON ml.dirichlet_alpha_by_stat_type USING btree (component_name);


--
-- Name: idx_das_stat_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_das_stat_type ON ml.dirichlet_alpha_by_stat_type USING btree (stat_type);


--
-- Name: idx_das_updated; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_das_updated ON ml.dirichlet_alpha_by_stat_type USING btree (updated_at DESC);


--
-- Name: idx_gop_game_id; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_gop_game_id ON ml.game_outcome_predictions USING btree (game_id);


--
-- Name: idx_gop_run_id; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_gop_run_id ON ml.game_outcome_predictions USING btree (prediction_run_id);


--
-- Name: idx_gop_season_week; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_gop_season_week ON ml.game_outcome_predictions USING btree (season, week);


--
-- Name: idx_model_evaluations_model; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_evaluations_model ON ml.model_evaluations USING btree (model_id);


--
-- Name: idx_model_metrics_run; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_metrics_run ON ml.model_metrics USING btree (run_id, metric_name);


--
-- Name: idx_model_registry_production; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_production ON ml.model_registry USING btree (is_production);


--
-- Name: idx_model_registry_status; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_status ON ml.model_registry USING btree (status);


--
-- Name: idx_model_registry_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_type ON ml.model_registry USING btree (model_type);


--
-- Name: idx_production_deployments_active; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_production_deployments_active ON ml.production_deployments USING btree (status) WHERE (status = 'active'::text);


--
-- Name: idx_stp_computed_at; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_stp_computed_at ON ml.stat_type_performance USING btree (computed_at DESC);


--
-- Name: idx_stp_model_stat; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_stp_model_stat ON ml.stat_type_performance USING btree (model_id, stat_type);


--
-- Name: idx_stp_season_week; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_stp_season_week ON ml.stat_type_performance USING btree (season, week_start, week_end);


--
-- Name: idx_stp_stat_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_stp_stat_type ON ml.stat_type_performance USING btree (stat_type);


--
-- Name: idx_training_runs_started; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_runs_started ON ml.training_runs USING btree (started_at DESC);


--
-- Name: idx_training_runs_status; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_runs_status ON ml.training_runs USING btree (status);


--
-- Name: model_metrics_timestamp_idx; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX model_metrics_timestamp_idx ON ml.model_metrics USING btree ("timestamp" DESC);


--
-- Name: dq_log_status_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX dq_log_status_idx ON public.data_quality_log USING btree (status);


--
-- Name: dq_log_table_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX dq_log_table_idx ON public.data_quality_log USING btree (table_name);


--
-- Name: games_away_team_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX games_away_team_idx ON public.games USING btree (away_team);


--
-- Name: games_home_team_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX games_home_team_idx ON public.games USING btree (home_team);


--
-- Name: games_kickoff_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX games_kickoff_idx ON public.games USING btree (kickoff);


--
-- Name: games_season_week_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX games_season_week_idx ON public.games USING btree (season, week);


--
-- Name: idx_coaching_changes_fired; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_changes_fired ON public.coaching_changes USING btree (team, season, week) WHERE (head_coach_fired = true);


--
-- Name: idx_coaching_changes_oc_changed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_changes_oc_changed ON public.coaching_changes USING btree (team, season, week) WHERE (oc_changed = true);


--
-- Name: idx_coaching_changes_scheme; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_changes_scheme ON public.coaching_changes USING btree (scheme) WHERE (scheme IS NOT NULL);


--
-- Name: idx_coaching_changes_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_changes_season_week ON public.coaching_changes USING btree (season, week);


--
-- Name: idx_coaching_changes_team_season; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_changes_team_season ON public.coaching_changes USING btree (team, season);


--
-- Name: idx_coaching_staff_team_season; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_coaching_staff_team_season ON public.coaching_staff USING btree (team, season);


--
-- Name: idx_coaching_staff_team_season_full; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_coaching_staff_team_season_full ON public.coaching_staff USING btree (team, season) WHERE (week IS NULL);


--
-- Name: idx_coaching_staff_team_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_coaching_staff_team_season_week ON public.coaching_staff USING btree (team, season, week) WHERE (week IS NOT NULL);


--
-- Name: idx_contracts_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_contracts_active ON public.contracts USING btree (is_active, team);


--
-- Name: idx_contracts_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_contracts_player ON public.contracts USING btree (gsis_id);


--
-- Name: idx_contracts_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_contracts_team ON public.contracts USING btree (team);


--
-- Name: idx_depth_charts_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_depth_charts_player ON public.depth_charts USING btree (gsis_id);


--
-- Name: idx_depth_charts_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_depth_charts_season_week ON public.depth_charts USING btree (season, season_type, week);


--
-- Name: idx_depth_charts_starters; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_depth_charts_starters ON public.depth_charts USING btree (club_code, season, week) WHERE (depth_position = 1);


--
-- Name: idx_depth_charts_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_depth_charts_team ON public.depth_charts USING btree (club_code, season, week);


--
-- Name: idx_draft_picks_hof; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_draft_picks_hof ON public.draft_picks USING btree (hof) WHERE (hof = true);


--
-- Name: idx_draft_picks_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_draft_picks_player ON public.draft_picks USING btree (gsis_id);


--
-- Name: idx_draft_picks_position; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_draft_picks_position ON public.draft_picks USING btree ("position", season);


--
-- Name: idx_draft_picks_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_draft_picks_team ON public.draft_picks USING btree (team, season);


--
-- Name: idx_espn_qbr_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_espn_qbr_player ON public.espn_qbr USING btree (player_id);


--
-- Name: idx_espn_qbr_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_espn_qbr_season_week ON public.espn_qbr USING btree (season, game_week);


--
-- Name: idx_espn_qbr_total; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_espn_qbr_total ON public.espn_qbr USING btree (qbr_total) WHERE (qbr_total IS NOT NULL);


--
-- Name: idx_ftn_charting_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ftn_charting_game ON public.ftn_charting USING btree (nflverse_game_id);


--
-- Name: idx_ftn_charting_play_action; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ftn_charting_play_action ON public.ftn_charting USING btree (is_play_action) WHERE (is_play_action = true);


--
-- Name: idx_ftn_charting_rpo; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ftn_charting_rpo ON public.ftn_charting USING btree (is_rpo) WHERE (is_rpo = true);


--
-- Name: idx_ftn_charting_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ftn_charting_season_week ON public.ftn_charting USING btree (season, week);


--
-- Name: idx_injuries_gsis_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injuries_gsis_id ON public.injuries USING btree (gsis_id);


--
-- Name: idx_injuries_report_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injuries_report_status ON public.injuries USING btree (report_status) WHERE (report_status IS NOT NULL);


--
-- Name: idx_injuries_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injuries_season_week ON public.injuries USING btree (season, week);


--
-- Name: idx_injuries_team_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injuries_team_week ON public.injuries USING btree (team, week, season);


--
-- Name: idx_injury_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injury_season_week ON public.team_injury_load USING btree (season, week);


--
-- Name: idx_injury_team_season; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_injury_team_season ON public.team_injury_load USING btree (team, season);


--
-- Name: idx_intl_games_game_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_intl_games_game_id ON public.international_games USING btree (game_id);


--
-- Name: idx_intl_games_kickoff; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_intl_games_kickoff ON public.international_games USING btree (kickoff);


--
-- Name: idx_intl_games_player_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_intl_games_player_id ON public.international_games USING btree (player_id);


--
-- Name: idx_intl_games_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_intl_games_season_week ON public.international_games USING btree (season, week);


--
-- Name: idx_mv_betting_game_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_mv_betting_game_id ON public.mv_betting_features USING btree (game_id);


--
-- Name: idx_mv_betting_home_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mv_betting_home_team ON public.mv_betting_features USING btree (home_team, season);


--
-- Name: idx_mv_betting_kickoff; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mv_betting_kickoff ON public.mv_betting_features USING btree (kickoff);


--
-- Name: idx_mv_betting_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mv_betting_season_week ON public.mv_betting_features USING btree (season, week);


--
-- Name: idx_nextgen_passing_cpoe; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_passing_cpoe ON public.nextgen_passing USING btree (completion_percentage_above_expectation) WHERE (completion_percentage_above_expectation IS NOT NULL);


--
-- Name: idx_nextgen_passing_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_passing_player ON public.nextgen_passing USING btree (player_id);


--
-- Name: idx_nextgen_passing_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_passing_season_week ON public.nextgen_passing USING btree (season, week);


--
-- Name: idx_nextgen_receiving_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_receiving_player ON public.nextgen_receiving USING btree (player_id);


--
-- Name: idx_nextgen_receiving_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_receiving_season_week ON public.nextgen_receiving USING btree (season, week);


--
-- Name: idx_nextgen_receiving_separation; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_receiving_separation ON public.nextgen_receiving USING btree (avg_separation) WHERE (avg_separation IS NOT NULL);


--
-- Name: idx_nextgen_receiving_yac_above_exp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_receiving_yac_above_exp ON public.nextgen_receiving USING btree (avg_yac_above_expectation) WHERE (avg_yac_above_expectation IS NOT NULL);


--
-- Name: idx_nextgen_rushing_efficiency; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_rushing_efficiency ON public.nextgen_rushing USING btree (efficiency) WHERE (efficiency IS NOT NULL);


--
-- Name: idx_nextgen_rushing_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_rushing_player ON public.nextgen_rushing USING btree (player_id);


--
-- Name: idx_nextgen_rushing_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nextgen_rushing_season_week ON public.nextgen_rushing USING btree (season, week);


--
-- Name: idx_officials_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_officials_game ON public.officials USING btree (game_id);


--
-- Name: idx_officials_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_officials_name ON public.officials USING btree (official_name);


--
-- Name: idx_officials_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_officials_season_week ON public.officials USING btree (season, week);


--
-- Name: idx_participation_coverage; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_participation_coverage ON public.participation USING btree (defense_coverage_type);


--
-- Name: idx_participation_formation; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_participation_formation ON public.participation USING btree (offense_formation);


--
-- Name: idx_participation_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_participation_game ON public.participation USING btree (nflverse_game_id);


--
-- Name: idx_pfr_defense_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_defense_game ON public.pfr_defense USING btree (game_id);


--
-- Name: idx_pfr_defense_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_defense_player ON public.pfr_defense USING btree (pfr_player_id);


--
-- Name: idx_pfr_defense_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_defense_season_week ON public.pfr_defense USING btree (season, week);


--
-- Name: idx_pfr_defense_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_defense_team ON public.pfr_defense USING btree (team, season, week);


--
-- Name: idx_pfr_passing_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_passing_game ON public.pfr_passing USING btree (game_id);


--
-- Name: idx_pfr_passing_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_passing_player ON public.pfr_passing USING btree (pfr_player_id);


--
-- Name: idx_pfr_passing_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_passing_season_week ON public.pfr_passing USING btree (season, week);


--
-- Name: idx_pfr_passing_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pfr_passing_team ON public.pfr_passing USING btree (team, season);


--
-- Name: idx_referee_assignments_bias; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_referee_assignments_bias ON public.referee_assignments USING btree (home_bias DESC) WHERE (home_bias IS NOT NULL);


--
-- Name: idx_referee_assignments_crew; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_referee_assignments_crew ON public.referee_assignments USING btree (referee_crew);


--
-- Name: idx_referee_assignments_high_bias; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_referee_assignments_high_bias ON public.referee_assignments USING btree (game_id) WHERE (abs(home_bias) > (0.2)::double precision);


--
-- Name: idx_referee_assignments_referee; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_referee_assignments_referee ON public.referee_assignments USING btree (referee_name) WHERE (referee_name IS NOT NULL);


--
-- Name: idx_rosters_weekly_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rosters_weekly_player ON public.rosters_weekly USING btree (gsis_id);


--
-- Name: idx_rosters_weekly_position; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rosters_weekly_position ON public.rosters_weekly USING btree ("position", team, season);


--
-- Name: idx_rosters_weekly_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rosters_weekly_team ON public.rosters_weekly USING btree (team, season, week);


--
-- Name: idx_rule_changes_play_types; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rule_changes_play_types ON public.rule_changes USING gin (affected_play_types);


--
-- Name: idx_rule_changes_positions; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rule_changes_positions ON public.rule_changes USING gin (affected_positions);


--
-- Name: idx_rule_changes_treatment; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rule_changes_treatment ON public.rule_changes USING btree (treatment_indicator);


--
-- Name: idx_rule_changes_year; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_rule_changes_year ON public.rule_changes USING btree (year_implemented);


--
-- Name: idx_semantic_news_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_semantic_news_game ON public.semantic_news USING btree (game_id);


--
-- Name: idx_semantic_news_team_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_semantic_news_team_week ON public.semantic_news USING btree (team, season, week);


--
-- Name: idx_snap_counts_game; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_snap_counts_game ON public.snap_counts USING btree (game_id);


--
-- Name: idx_snap_counts_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_snap_counts_player ON public.snap_counts USING btree (pfr_player_id);


--
-- Name: idx_snap_counts_season_week; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_snap_counts_season_week ON public.snap_counts USING btree (season, week);


--
-- Name: idx_snap_counts_team; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_snap_counts_team ON public.snap_counts USING btree (team, season, week);


--
-- Name: idx_team_travel_away; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_away ON public.team_travel USING btree (away_team);


--
-- Name: idx_team_travel_distance; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_distance ON public.team_travel USING btree (travel_distance_miles DESC) WHERE (travel_distance_miles IS NOT NULL);


--
-- Name: idx_team_travel_home; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_home ON public.team_travel USING btree (home_team);


--
-- Name: idx_team_travel_international; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_international ON public.team_travel USING btree (game_id) WHERE (is_international = true);


--
-- Name: idx_team_travel_thursday; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_thursday ON public.team_travel USING btree (game_id) WHERE (is_thursday_game = true);


--
-- Name: idx_team_travel_time_zones; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_team_travel_time_zones ON public.team_travel USING btree (time_zones_crossed) WHERE (time_zones_crossed IS NOT NULL);


--
-- Name: idx_trades_player; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_player ON public.trades USING btree (pfr_id);


--
-- Name: idx_trades_season; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_season ON public.trades USING btree (season);


--
-- Name: idx_trades_team_gave; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_team_gave ON public.trades USING btree (gave, season);


--
-- Name: idx_trades_team_received; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trades_team_received ON public.trades USING btree (received, season);


--
-- Name: odds_history_bookmaker_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX odds_history_bookmaker_idx ON public.odds_history USING btree (bookmaker);


--
-- Name: odds_history_captured_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX odds_history_captured_at_idx ON public.odds_history USING btree (captured_at DESC);


--
-- Name: odds_history_game_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX odds_history_game_idx ON public.odds_history USING btree (game_id, captured_at DESC);


--
-- Name: players_name_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX players_name_idx ON public.players USING btree (player_name);


--
-- Name: players_position_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX players_position_idx ON public.players USING btree ("position");


--
-- Name: plays_game_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX plays_game_id_idx ON public.plays USING btree (game_id);


--
-- Name: plays_posteam_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX plays_posteam_idx ON public.plays USING btree (posteam);


--
-- Name: plays_quarter_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX plays_quarter_idx ON public.plays USING btree (quarter);


--
-- Name: weather_game_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX weather_game_id_idx ON public.weather USING btree (game_id);


--
-- Name: active_models _RETURN; Type: RULE; Schema: ml; Owner: -
--

CREATE OR REPLACE VIEW ml.active_models AS
 SELECT mr.model_id,
    mr.model_name,
    mr.model_type,
    mr.model_family,
    mr.version,
    mr.model_path,
    mr.checkpoint_dir,
    mr.metadata_path,
    mr.status,
    mr.is_production,
    mr.training_start,
    mr.training_end,
    mr.training_duration_seconds,
    mr.train_start_season,
    mr.train_end_season,
    mr.test_start_season,
    mr.test_end_season,
    mr.train_samples,
    mr.test_samples,
    mr.hardware,
    mr.gpu_memory_gb,
    mr.git_commit_hash,
    mr.git_branch,
    mr.created_at,
    mr.updated_at,
    mr.description,
    mr.notes,
    count(DISTINCT tr.run_id) AS total_runs,
    max(tr.completed_at) AS last_training_run,
    (pd.deployment_id IS NOT NULL) AS is_deployed
   FROM ((ml.model_registry mr
     LEFT JOIN ml.training_runs tr ON ((mr.model_id = tr.model_id)))
     LEFT JOIN ml.production_deployments pd ON (((mr.model_id = pd.model_id) AND (pd.status = 'active'::text))))
  WHERE (mr.status = ANY (ARRAY['training'::text, 'completed'::text, 'production'::text]))
  GROUP BY mr.model_id, pd.deployment_id;


--
-- Name: ensemble_configs ensemble_configs_update_timestamp; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER ensemble_configs_update_timestamp BEFORE UPDATE ON ml.ensemble_configs FOR EACH ROW EXECUTE FUNCTION ml.update_model_timestamp();


--
-- Name: model_registry model_registry_update_timestamp; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER model_registry_update_timestamp BEFORE UPDATE ON ml.model_registry FOR EACH ROW EXECUTE FUNCTION ml.update_model_timestamp();


--
-- Name: model_artifacts model_artifacts_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_artifacts
    ADD CONSTRAINT model_artifacts_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id) ON DELETE CASCADE;


--
-- Name: model_artifacts model_artifacts_run_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_artifacts
    ADD CONSTRAINT model_artifacts_run_id_fkey FOREIGN KEY (run_id) REFERENCES ml.training_runs(run_id) ON DELETE CASCADE;


--
-- Name: model_evaluations model_evaluations_baseline_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_evaluations
    ADD CONSTRAINT model_evaluations_baseline_model_id_fkey FOREIGN KEY (baseline_model_id) REFERENCES ml.model_registry(model_id);


--
-- Name: model_evaluations model_evaluations_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_evaluations
    ADD CONSTRAINT model_evaluations_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id) ON DELETE CASCADE;


--
-- Name: production_deployments production_deployments_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.production_deployments
    ADD CONSTRAINT production_deployments_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id) ON DELETE CASCADE;


--
-- Name: training_runs training_runs_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_runs
    ADD CONSTRAINT training_runs_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id) ON DELETE CASCADE;


--
-- Name: international_games fk_intl_games_game_id; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.international_games
    ADD CONSTRAINT fk_intl_games_game_id FOREIGN KEY (game_id) REFERENCES public.games(game_id);


--
-- Name: participation fk_participation_plays; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.participation
    ADD CONSTRAINT fk_participation_plays FOREIGN KEY (nflverse_game_id, play_id) REFERENCES public.plays(game_id, play_id) ON DELETE CASCADE;


--
-- Name: plays fk_plays_games; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plays
    ADD CONSTRAINT fk_plays_games FOREIGN KEY (game_id) REFERENCES public.games(game_id) ON DELETE CASCADE;


--
-- Name: referee_assignments fk_referee_assignments_game; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.referee_assignments
    ADD CONSTRAINT fk_referee_assignments_game FOREIGN KEY (game_id) REFERENCES public.games(game_id) ON DELETE CASCADE;


--
-- Name: team_travel fk_team_travel_game; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.team_travel
    ADD CONSTRAINT fk_team_travel_game FOREIGN KEY (game_id) REFERENCES public.games(game_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict fTfDeIRb1MIXS0kB7JJgaGUTSNzJtVEih0eq5JKR9HckAoMwLpTcNVeFQ90ibbT

--
-- TimescaleDB hypertables (added to the pg_dump output; settings as in devdb01). The dump
-- already holds every index, so create_hypertable() must not add its default ones.
--

SELECT public.create_hypertable('public.odds_history', public.by_range('captured_at', INTERVAL '30 days'), create_default_indexes => false);
SELECT public.create_hypertable('ml.model_metrics', public.by_range('timestamp', INTERVAL '7 days'), create_default_indexes => false);
