-- ROYALEY Migration: Prepare for ML model predictions
-- Clears existing market-implied predictions so pipeline can regenerate with real models
-- Run: cat migrations/prepare_ml_predictions.sql | docker exec -i royaley_postgres psql -U royaley -d royaley

-- 1. Remove existing predictions (they have edge=0% from market-implied)
TRUNCATE predictions CASCADE;
TRUNCATE prediction_results CASCADE;

-- 2. Add unique constraint (prevents duplicates on re-run)
CREATE UNIQUE INDEX IF NOT EXISTS ix_predictions_unique_pick
  ON predictions (upcoming_game_id, bet_type, predicted_side)
  WHERE upcoming_game_id IS NOT NULL;

-- 3. Verify clean state
SELECT 'predictions' as tbl, COUNT(*) as rows FROM predictions
UNION ALL
SELECT 'prediction_results', COUNT(*) FROM prediction_results
UNION ALL
SELECT 'upcoming_games', COUNT(*) FROM upcoming_games
UNION ALL
SELECT 'upcoming_odds', COUNT(*) FROM upcoming_odds;