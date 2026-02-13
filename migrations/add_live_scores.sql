-- ROYALEY Migration: Add live score columns to upcoming_games
-- Run: cat migrations/add_live_scores.sql | docker exec -i royaley_postgres psql -U royaley -d royaley

-- Add score and live status columns
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS home_score INTEGER;
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS away_score INTEGER;
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS last_score_update TIMESTAMP;
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS completed BOOLEAN DEFAULT FALSE;

-- Verify
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'upcoming_games' AND column_name IN ('home_score','away_score','completed','last_score_update');
