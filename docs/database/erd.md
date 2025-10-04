# Database ERD

The following entity-relationship diagram captures the core public tables and downstream mart tables/views used for analytics. A PNG export is available at `docs/database/erd.png`.

```mermaid
erDiagram
    PUBLIC_GAMES ||--o{ PUBLIC_PLAYS : game_id
    PUBLIC_GAMES ||--o| PUBLIC_WEATHER : game_id
    PUBLIC_GAMES ||--o{ PUBLIC_INJURIES : game_id
    PUBLIC_GAMES ||--o{ MART_TEAM_EPA : game_id
    PUBLIC_GAMES ||--o{ MART_TEAM_4TH_DOWN : game_id

    %% Marts and views
    PUBLIC_GAMES ||--o{ MART_GAME_SUMMARY : feeds
    MART_TEAM_EPA ||--o{ MART_GAME_SUMMARY : feeds
    PUBLIC_WEATHER ||--o{ MART_GAME_SUMMARY : feeds

    PUBLIC_GAMES ||--o{ MART_GAME_WEATHER : feeds
    PUBLIC_WEATHER ||--o{ MART_GAME_WEATHER : feeds

    PUBLIC_GAMES ||--o{ MART_GAME_FEATURES_ENH : feeds
    MART_TEAM_EPA ||--o{ MART_GAME_FEATURES_ENH : feeds
    MART_GAME_WEATHER ||--o{ MART_GAME_FEATURES_ENH : feeds
    MART_TEAM_4TH_DOWN ||--o{ MART_GAME_FEATURES_ENH : feeds
    MART_TEAM_PLAYOFF_CTX ||--o{ MART_GAME_FEATURES_ENH : feeds
    MART_TEAM_INJURY_LOAD ||--o{ MART_GAME_FEATURES_ENH : feeds

    PUBLIC_GAMES {
        text game_id PK
        int season
        int week
        text home_team
        text away_team
        timestamptz kickoff
        real spread_close
        real total_close
        real home_moneyline
        real away_moneyline
        text stadium
        text surface
    }

    PUBLIC_PLAYS {
        text game_id PK_FK
        bigint play_id PK
        text posteam
        text defteam
        int quarter
        int time_seconds
        int down
        int ydstogo
        double epa
        boolean pass
        boolean rush
    }

    PUBLIC_WEATHER {
        text game_id PK_FK
        text station
        real temp_c
        real rh
        real wind_kph
        real pressure_hpa
        real precip_mm
    }

    PUBLIC_INJURIES {
        text game_id FK
        text team
        text player_id
        text status
    }

    PUBLIC_ODDS_HISTORY {
        text event_id PK
        text bookmaker_key PK
        text market_key PK
        text outcome_name PK
        timestamptz snapshot_at PK
        timestamptz commence_time
        timestamptz market_last_update
        double outcome_price
        double outcome_point
        timestamptz fetched_at
    }

    MART_TEAM_EPA {
        text game_id PK_FK
        text posteam PK
        int plays
        double epa_sum
        double epa_mean
        double explosive_pass
        double explosive_rush
    }

    MART_TEAM_4TH_DOWN {
        text game_id PK_FK
        text team PK
        int fourth_downs
        real went_for_it_rate
        real fourth_down_epa
        int bad_decisions
        real avg_go_boost
        real avg_fg_boost
    }

    MART_TEAM_PLAYOFF_CTX {
        text team PK
        int season PK
        int week PK
        real playoff_prob
        real div_winner_prob
        real first_seed_prob
        boolean eliminated
        boolean locked_in
        boolean desperate
    }

    MART_TEAM_INJURY_LOAD {
        int season PK
        int week PK
        text team PK
        int total_injuries
        int players_out
        int players_questionable
        int players_doubtful
        boolean key_position_out
        boolean qb_out
        int oline_injuries
        real injury_severity_index
    }

    MART_GAME_SUMMARY {
        text game_id
        int season
        int week
        text home_team
        text away_team
        int home_score
        int away_score
        real spread_close
        real total_close
        real home_moneyline
        real away_moneyline
        real home_epa_mean
        real away_epa_mean
        text stadium
        text roof
        text surface
        real temp_c
        real wind_kph
    }

    MART_GAME_WEATHER {
        text game_id
        int season
        int week
        text home_team
        text away_team
        real temp_c
        real humidity
        real wind_kph
        real pressure_hpa
        real precip_mm
        real temp_extreme
        real wind_penalty
        boolean has_precip
        boolean is_dome
    }

    MART_GAME_FEATURES_ENH {
        text game_id
        int season
        int week
        text home_team
        text away_team
        real home_epa_mean
        real away_epa_mean
        real temperature
        real wind_speed
        real precipitation
        real home_4th_epa
        real away_4th_epa
        real home_injury_severity
        real away_injury_severity
        real home_playoff_prob
        real away_playoff_prob
    }
```

Notes
- PUBLIC_ODDS_HISTORY is a TimescaleDB hypertable and does not have a strict FK to PUBLIC_GAMES; joins are performed at query time using event metadata.
- Views are materialized in mart: MART_GAME_SUMMARY, MART_GAME_WEATHER, MART_GAME_FEATURES_ENH.
- Several mart tables join to games by (team, season, week) or (game_id, team) rather than strict FKs; the diagram reflects analytic joins used in views.

See also: `docs/database/schema.md` for detailed column descriptions and migration references.
