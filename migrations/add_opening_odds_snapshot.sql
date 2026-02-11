-- ROYALEY Migration: Add opening odds snapshot columns to predictions
-- Run this on the server BEFORE deploying the new pipeline/API code
-- docker exec royaley_db psql -U royaley -d royaley -f /tmp/migration.sql

-- Opening snapshot: captures both sides of the market at prediction time
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_line_open FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_line_open FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_odds_open INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_odds_open INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS total_open FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS over_odds_open INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS under_odds_open INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_ml_open INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_ml_open INT;

-- Also add upcoming_game_id FK if it doesn't exist (may already be there)
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS upcoming_game_id UUID REFERENCES upcoming_games(id);

-- Backfill existing predictions from upcoming_odds (opening = current for old data)
WITH consensus AS (
    SELECT 
        upcoming_game_id,
        bet_type,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_line END), AVG(home_line)) as home_line,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_line END), AVG(away_line)) as away_line,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_odds END), AVG(home_odds))::int as home_odds,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_odds END), AVG(away_odds))::int as away_odds,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN total END), AVG(total)) as total_line,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN over_odds END), AVG(over_odds))::int as over_odds,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN under_odds END), AVG(under_odds))::int as under_odds,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_ml END), AVG(home_ml))::int as home_ml,
        COALESCE(MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_ml END), AVG(away_ml))::int as away_ml
    FROM upcoming_odds
    GROUP BY upcoming_game_id, bet_type
)
UPDATE predictions p SET
    home_line_open = CASE WHEN p.bet_type = 'spread' THEN c.home_line END,
    away_line_open = CASE WHEN p.bet_type = 'spread' THEN c.away_line END,
    home_odds_open = CASE WHEN p.bet_type = 'spread' THEN c.home_odds 
                          WHEN p.bet_type = 'moneyline' THEN NULL END,
    away_odds_open = CASE WHEN p.bet_type = 'spread' THEN c.away_odds
                          WHEN p.bet_type = 'moneyline' THEN NULL END,
    total_open     = CASE WHEN p.bet_type = 'total' THEN c.total_line END,
    over_odds_open = CASE WHEN p.bet_type = 'total' THEN c.over_odds END,
    under_odds_open= CASE WHEN p.bet_type = 'total' THEN c.under_odds END,
    home_ml_open   = CASE WHEN p.bet_type = 'moneyline' THEN c.home_ml END,
    away_ml_open   = CASE WHEN p.bet_type = 'moneyline' THEN c.away_ml END
FROM consensus c
WHERE p.upcoming_game_id = c.upcoming_game_id
  AND p.bet_type = c.bet_type
  AND p.home_line_open IS NULL;

-- Verify
SELECT 
    p.bet_type,
    COUNT(*) as total,
    COUNT(p.home_line_open) as has_spread,
    COUNT(p.total_open) as has_total,
    COUNT(p.home_ml_open) as has_ml
FROM predictions p
WHERE p.upcoming_game_id IS NOT NULL
GROUP BY p.bet_type;