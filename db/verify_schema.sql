DO $$
BEGIN
  -- Ensure required columns exist on games table
  PERFORM 1 FROM information_schema.columns
   WHERE table_schema = 'public' AND table_name = 'games'
     AND column_name = 'home_moneyline';
  IF NOT FOUND THEN
    RAISE EXCEPTION 'games.home_moneyline missing';
  END IF;

  PERFORM 1 FROM information_schema.columns
   WHERE table_schema = 'public' AND table_name = 'games'
     AND column_name = 'over_odds';
  IF NOT FOUND THEN
    RAISE EXCEPTION 'games.over_odds missing';
  END IF;

  -- Ensure odds_history table exists
  PERFORM 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'odds_history';
  IF NOT FOUND THEN
    RAISE EXCEPTION 'odds_history table missing';
  END IF;
END$$;
