library(tidymodels); library(xgboost); library(dplyr); library(DBI); library(RPostgres)

# Connect to database (using consistent connection parameters)
con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = "devdb01",
  host     = "localhost",
  user     = "dro",
  password = "sicillionbillions",
  port     = 5544
)

# Get games data
g <- tbl(con, "games") |>
  select(game_id, season, week, home_team, away_team, home_score, away_score, spread_close, total_close) |>
  collect()

# Get EPA aggregated data (check if plays table exists)
if (dbExistsTable(con, "plays")) {
  pbp <- tbl(con, "plays")
  agg <- pbp |>
    group_by(game_id, posteam) |>
    summarise(epa_off = mean(epa, na.rm=TRUE),
              pass_rate = mean(pass, na.rm=TRUE),
              rush_rate = mean(rush, na.rm=TRUE))
} else {
  # Create empty agg if plays table doesn't exist
  agg <- data.frame(game_id = character(), posteam = character(), epa_off = numeric())
}

# Build the modeling dataset
df <- g |> mutate(spread_result = (home_score - away_score))

# Only join EPA data if it exists
if (nrow(agg |> collect()) > 0) {
  df <- df |>
    left_join(agg |> collect() |> group_by(game_id) |>
                summarise(epa_gap = diff(sort(epa_off, decreasing=TRUE))[1]), by="game_id")
} else {
  df <- df |> mutate(epa_gap = NA_real_)
}

# Join weather data if table exists
if (dbExistsTable(con, "weather")) {
  df <- df |> left_join(tbl(con,"weather") |> collect(), by="game_id")
} else {
  df <- df |> mutate(temp_c = NA_real_, wind_kph = NA_real_, precip_mm = NA_real_)
}

df <- df |>
  mutate(y = spread_result) |>
  select(y, spread_close, epa_gap, temp_c, wind_kph, precip_mm) |>
  drop_na()

set.seed(42)
split <- initial_split(df, prop = 0.8)
train <- training(split); test <- testing(split)

rec <- recipe(y ~ ., data=train) |>
  step_impute_mean(all_predictors()) |>
  step_normalize(all_predictors())

spec <- boost_tree(trees=1500, learn_rate=0.03, mtry = tune(), tree_depth = tune()) |>
  set_engine("xgboost") |>
  set_mode("regression")

wf <- workflow(rec, spec)
rs <- vfold_cv(train, v=5)
grid <- grid_latin_hypercube(mtry(c(1,5)), tree_depth(), size=15)

tuned <- tune_grid(wf, rs, grid=grid, metrics = metric_set(rmse, mae))
best <- select_best(tuned, "rmse")
final_wf <- finalize_workflow(wf, best) |> fit(train)
preds <- predict(final_wf, test) |> bind_cols(test["y"])
metrics(preds, truth=y, estimate=.pred)

# Close database connection
dbDisconnect(con)
