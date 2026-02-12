-- ROYALEY Migration: Fix duplicate predictions
-- Adds unique constraint on (upcoming_game_id, bet_type, predicted_side)
-- Run: cat fix_duplicate_predictions.sql | docker exec -i royaley_postgres psql -U royaley -d royaley

-- 1. Remove duplicates (keep earliest created)
DELETE FROM predictions p
USING predictions p2
WHERE p.upcoming_game_id = p2.upcoming_game_id
  AND p.bet_type = p2.bet_type
  AND p.predicted_side = p2.predicted_side
  AND p.created_at > p2.created_at;

-- 2. Add unique constraint to prevent future duplicates
CREATE UNIQUE INDEX IF NOT EXISTS ix_predictions_unique_pick
  ON predictions (upcoming_game_id, bet_type, predicted_side)
  WHERE upcoming_game_id IS NOT NULL;

-- 3. Verify
SELECT 'predictions' as tbl, COUNT(*) as rows FROM predictions
UNION ALL
SELECT 'unique games', COUNT(DISTINCT upcoming_game_id) FROM predictions WHERE upcoming_game_id IS NOT NULL;