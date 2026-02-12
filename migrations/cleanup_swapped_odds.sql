-- ROYALEY Migration: Clean corrupted odds data (home/away swapped)
-- Run BEFORE deploying fixed code, then re-run pipeline
--
-- cat cleanup_swapped_odds.sql | docker exec -i royaley_postgres psql -U royaley -d royaley

-- 1. Clear upcoming_odds (will be re-fetched with correct home/away)
TRUNCATE upcoming_odds;

-- 2. Clear predictions (opening snapshots were from corrupted odds)
TRUNCATE predictions CASCADE;

-- 3. Clear prediction_results (depends on predictions)
TRUNCATE prediction_results;

-- 4. Verify clean state
SELECT 'upcoming_odds' as tbl, COUNT(*) as rows FROM upcoming_odds
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL SELECT 'prediction_results', COUNT(*) FROM prediction_results;