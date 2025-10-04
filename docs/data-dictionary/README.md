# NFL Analytics Data Dictionary

> **Version**: 1.0.0
> **Last Updated**: 2025-01-04
> **Purpose**: Master reference for all data standards in the NFL Analytics system

## 📚 Table of Contents

1. [Team Abbreviations](#team-abbreviations)
2. [Column Standards](#column-standards)
3. [Stadium & Venue Data](#stadium--venue-data)
4. [Time & Date Standards](#time--date-standards)
5. [Weather Standards](#weather-standards)
6. [API Mappings](#api-mappings)
7. [Translation Functions](#translation-functions)

---

## Team Abbreviations

### Canonical Team Codes (ALWAYS USE THESE)

| Logo | Canonical | Full Name | City | Conference | Division | Historical Codes |
|------|-----------|-----------|------|------------|----------|------------------|
| 🦬 | **BUF** | Buffalo Bills | Buffalo | AFC | East | - |
| 🐬 | **MIA** | Miami Dolphins | Miami | AFC | East | - |
| 🎖️ | **NE** | New England Patriots | New England | AFC | East | - |
| ✈️ | **NYJ** | New York Jets | New York | AFC | East | - |
| 🐦‍⬛ | **BAL** | Baltimore Ravens | Baltimore | AFC | North | - |
| 🐅 | **CIN** | Cincinnati Bengals | Cincinnati | AFC | North | - |
| 🟤 | **CLE** | Cleveland Browns | Cleveland | AFC | North | - |
| ⚙️ | **PIT** | Pittsburgh Steelers | Pittsburgh | AFC | North | - |
| 🐂 | **HOU** | Houston Texans | Houston | AFC | South | - |
| 🐴 | **IND** | Indianapolis Colts | Indianapolis | AFC | South | - |
| 🐆 | **JAX** | Jacksonville Jaguars | Jacksonville | AFC | South | JAC |
| ⚔️ | **TEN** | Tennessee Titans | Tennessee | AFC | South | - |
| 🐎 | **DEN** | Denver Broncos | Denver | AFC | West | - |
| 🏹 | **KC** | Kansas City Chiefs | Kansas City | AFC | West | KAN |
| ☠️ | **LV** | Las Vegas Raiders | Las Vegas | AFC | West | **OAK** (pre-2020), LVR |
| ⚡ | **LAC** | Los Angeles Chargers | Los Angeles | AFC | West | **SD** (pre-2017) |
| ⭐ | **DAL** | Dallas Cowboys | Dallas | NFC | East | - |
| 🔵 | **NYG** | New York Giants | New York | NFC | East | - |
| 🦅 | **PHI** | Philadelphia Eagles | Philadelphia | NFC | East | - |
| 🏛️ | **WAS** | Washington Commanders | Washington | NFC | East | WSH |
| 🐻 | **CHI** | Chicago Bears | Chicago | NFC | North | - |
| 🦁 | **DET** | Detroit Lions | Detroit | NFC | North | - |
| 🧀 | **GB** | Green Bay Packers | Green Bay | NFC | North | - |
| 🛡️ | **MIN** | Minnesota Vikings | Minnesota | NFC | North | - |
| 🔴 | **ATL** | Atlanta Falcons | Atlanta | NFC | South | - |
| 🐾 | **CAR** | Carolina Panthers | Carolina | NFC | South | - |
| ⚜️ | **NO** | New Orleans Saints | New Orleans | NFC | South | - |
| 🏴‍☠️ | **TB** | Tampa Bay Buccaneers | Tampa Bay | NFC | South | - |
| 🟥 | **ARI** | Arizona Cardinals | Arizona | NFC | West | AZ |
| 🐏 | **LA** | Los Angeles Rams | Los Angeles | NFC | West | **STL** (pre-2016), RAM |
| 🔶 | **SF** | San Francisco 49ers | San Francisco | NFC | West | - |
| 🌊 | **SEA** | Seattle Seahawks | Seattle | NFC | West | - |

### ⚠️ Important Relocation Notes

- **OAK → LV** (2020): Oakland Raiders became Las Vegas Raiders
- **SD → LAC** (2017): San Diego Chargers became Los Angeles Chargers
- **STL → LA** (2016): St. Louis Rams became Los Angeles Rams

**Always use canonical codes in database!** Translation functions handle historical data automatically.

---

## Column Standards

### Games Table

| Column Name | Data Type | Description | NOT These Names |
|-------------|-----------|-------------|-----------------|
| `game_id` | VARCHAR | Unique game identifier (YYYY_WW_AWAY_HOME) | - |
| `season` | INTEGER | Season year | year |
| `week` | INTEGER | Week number (1-22) | wk |
| `home_team` | VARCHAR(3) | Home team canonical abbreviation | home, home_abbr |
| `away_team` | VARCHAR(3) | Away team canonical abbreviation | away, away_abbr |
| `home_score` | INTEGER | Home team final score | home_points, pts_home |
| `away_score` | INTEGER | Away team final score | away_points, pts_away |
| `kickoff` | TIMESTAMP WITH TIME ZONE | Game start time | game_time, start_time |
| `spread_close` | FLOAT | Closing spread line | spread, line |
| `total_close` | FLOAT | Closing total line | total, over_under |

### Plays Table

| Column Name | Data Type | Description | NOT These Names |
|-------------|-----------|-------------|-----------------|
| `game_id` | VARCHAR | Foreign key to games | - |
| `play_id` | INTEGER | Play number within game | - |
| `quarter` | INTEGER | Quarter (1-6, including OT) | **qtr**, period, Q |
| `time_seconds` | INTEGER | Seconds remaining in quarter | game_seconds_remaining |
| `down` | INTEGER | Down (1-4) | - |
| `ydstogo` | INTEGER | Yards to go for first down | distance |
| `yardline_100` | INTEGER | Yards from opponent's end zone | - |
| `posteam` | VARCHAR(3) | Possession team | offense |
| `defteam` | VARCHAR(3) | Defensive team | defense |

### Weather & Stadium Fields

| Column Name | Data Type | Description | Standard Values |
|-------------|-----------|-------------|-----------------|
| `roof` | VARCHAR(20) | Roof type | `dome`, `retractable`, `outdoor` |
| `surface` | VARCHAR(20) | Playing surface | `grass`, `turf`, `fieldturf` |
| `temp_fahrenheit` | FLOAT | Temperature in °F | Always Fahrenheit |
| `wind_mph` | FLOAT | Wind speed in mph | Always mph |
| `stadium` | VARCHAR(100) | Stadium name | - |
| `stadium_id` | VARCHAR(20) | Unique stadium identifier | - |

---

## Stadium & Venue Data

### Roof Types (ONLY USE THESE)

| Canonical Value | Description | Maps From |
|-----------------|-------------|-----------|
| `dome` | Fixed dome/indoor | closed, domed, indoor |
| `retractable` | Retractable roof | - |
| `outdoor` | Open air stadium | open, outdoors, outside |

### Surface Types (ONLY USE THESE)

| Canonical Value | Description | Maps From |
|-----------------|-------------|-----------|
| `grass` | Natural grass | natural |
| `turf` | Artificial turf | artificial, astroturf |
| `fieldturf` | FieldTurf brand | - |

---

## Time & Date Standards

### Time Zones
- **Always store as**: `TIMESTAMP WITH TIME ZONE`
- **Database storage**: UTC
- **Display conversion**: Local team timezone
- **Kickoff times**: Convert to ET for display

### Date Formats
- **Database**: ISO 8601 (`YYYY-MM-DD HH:MM:SS+00`)
- **Season**: Integer year (2024, not "2024-25")
- **Week**: Integer (1-22, including playoffs)

### Game Periods
| Value | Description |
|-------|-------------|
| 1 | 1st Quarter |
| 2 | 2nd Quarter |
| 3 | 3rd Quarter |
| 4 | 4th Quarter |
| 5 | Overtime |
| 6 | 2nd Overtime (playoffs) |

**ALWAYS use `quarter`, NEVER `qtr` or `period`**

---

## Weather Standards

### Temperature
- **Column**: `temp_fahrenheit`
- **Unit**: Always Fahrenheit
- **Range**: -30 to 120 (validate)

### Wind
- **Column**: `wind_mph`
- **Unit**: Always mph
- **Format**: Speed only, no direction

### Precipitation
- **Column**: `precipitation`
- **Values**: `none`, `rain`, `snow`, `mixed`

---

## API Mappings

### nflverse API
```sql
-- Common mappings
qtr → quarter
game_seconds_remaining → time_seconds
home_points → home_score
away_points → away_score
```

### The Odds API
```sql
-- Team names are FULL names
'Buffalo Bills' → 'BUF'
'Kansas City Chiefs' → 'KC'
-- Use reference.translate_team() function
```

### ESPN API
```sql
-- Different abbreviations sometimes
period → quarter
pts_home → home_score
pts_away → away_score
```

---

## Translation Functions

### Available Database Functions

```sql
-- Translate any team abbreviation to canonical
SELECT reference.translate_team('OAK');  -- Returns 'LV'
SELECT reference.translate_team('BUF');  -- Returns 'BUF'

-- Get team full name
SELECT reference.get_team_fullname('BUF');  -- Returns 'Buffalo Bills'

-- Translate column names
SELECT reference.translate_column('plays', 'nflverse', 'qtr');  -- Returns 'quarter'

-- Standardize roof type
SELECT reference.standardize_roof('closed');  -- Returns 'dome'

-- Standardize surface type
SELECT reference.standardize_surface('artificial');  -- Returns 'turf'
```

### Python Usage

```python
# Use the translation in Python scripts
def translate_team(abbr: str) -> str:
    """Translate any team abbreviation to canonical."""
    translations = {
        'OAK': 'LV',
        'SD': 'LAC',
        'STL': 'LA',
        # ... etc
    }
    return translations.get(abbr, abbr)

# Column mappings
COLUMN_MAP = {
    'qtr': 'quarter',
    'game_seconds_remaining': 'time_seconds',
    # ... etc
}
```

### R Usage

```r
# Team translation in R
translate_team <- function(abbr) {
  translations <- c(
    "OAK" = "LV",
    "SD" = "LAC",
    "STL" = "LA"
  )
  ifelse(abbr %in% names(translations), translations[abbr], abbr)
}

# Column renaming
standardize_columns <- function(df) {
  df %>%
    rename_with(~ case_when(
      . == "qtr" ~ "quarter",
      . == "game_seconds_remaining" ~ "time_seconds",
      TRUE ~ .
    ))
}
```

---

## Validation Queries

### Check Data Quality

```sql
-- View all data quality issues
SELECT * FROM reference.data_quality_checks;

-- Check for non-canonical teams
SELECT DISTINCT home_team, away_team
FROM games
WHERE home_team NOT IN (SELECT canonical_abbr FROM reference.teams)
   OR away_team NOT IN (SELECT canonical_abbr FROM reference.teams);

-- Check for non-standard values
SELECT DISTINCT roof FROM games WHERE roof NOT IN ('dome', 'retractable', 'outdoor');
SELECT DISTINCT surface FROM games WHERE surface NOT IN ('grass', 'turf', 'fieldturf');
```

---

## Best Practices

1. **Always use canonical values** - Let translation functions handle variations
2. **Never hardcode team names** - Use the reference tables
3. **Validate on ingestion** - Transform data immediately when loading
4. **Use appropriate data types** - TIMESTAMP WITH TIME ZONE for times
5. **Document exceptions** - If you must deviate, document why

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|---------|
| 2025-01-04 | 1.0.0 | Initial data dictionary | Master Coordinator |

---

## Questions?

- Check translation functions: `SELECT * FROM reference.teams;`
- View column mappings: `SELECT * FROM reference.column_mappings;`
- Run quality checks: `SELECT * FROM reference.data_quality_checks;`