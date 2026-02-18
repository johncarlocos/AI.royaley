-- ============================================================
-- ROYALEY: Fix Tennis Scores & Reset Bad Grades
-- Run: docker exec -i royaley_postgres psql -U royaley -d royaley < fix_tennis_scores.sql
-- ============================================================

BEGIN;

-- ============================================================
-- STEP 1: Delete incorrect prediction_results for games with NULL scores
-- These were graded by a buggy run without actual scores
-- ============================================================
DELETE FROM prediction_results
WHERE prediction_id IN (
    SELECT p.id FROM predictions p
    JOIN upcoming_games ug ON p.upcoming_game_id = ug.id
    JOIN sports s ON ug.sport_id = s.id
    WHERE s.code IN ('WTA', 'ATP')
      AND ug.home_score IS NULL
      AND ug.status = 'scheduled'
);

-- Report how many were deleted
DO $$
DECLARE cnt INTEGER;
BEGIN
    GET DIAGNOSTICS cnt = ROW_COUNT;
    RAISE NOTICE 'Deleted % incorrect prediction_results', cnt;
END $$;

-- ============================================================
-- STEP 2: Enter confirmed scores (total games won format)
-- Verified from WTA draw PDF, ATP results, ESPN, TennisExplorer
-- ============================================================

-- === ATP Doha R2 - Feb 18 (ALL CONFIRMED) ===

-- Tsitsipas beat Medvedev 6-3, 6-4
UPDATE upcoming_games SET home_score = 7, away_score = 12, status = 'final', completed = true, updated_at = NOW()
WHERE id = 'd62b7f7f-b6f3-4c6c-8984-9460039f14a6';

-- Khachanov beat Fucsovics 6-2, 4-6, 6-4
UPDATE upcoming_games SET home_score = 12, away_score = 16, status = 'final', completed = true, updated_at = NOW()
WHERE id = '7f5a1a24-758a-418f-8dd5-80ec94af0189';

-- Lehecka beat Bergs 6-2, 6-1
UPDATE upcoming_games SET home_score = 12, away_score = 3, status = 'final', completed = true, updated_at = NOW()
WHERE id = 'f2e293da-6a50-4d1a-9c73-dbb415509e7b';

-- Fils beat Halys 6-1, 7-6(7)
UPDATE upcoming_games SET home_score = 13, away_score = 7, status = 'final', completed = true, updated_at = NOW()
WHERE id = '7a846c06-12ee-4082-8c8d-d947834549c7';

-- Rublev beat Marozsan 6-2, 6-4
UPDATE upcoming_games SET home_score = 6, away_score = 12, status = 'final', completed = true, updated_at = NOW()
WHERE id = '89e4fb3d-1944-42e8-b10c-27752f5202ff';

-- === WTA Dubai R16 - Feb 18 (CONFIRMED) ===

-- Anisimova beat Tjen 6-1, 6-3
UPDATE upcoming_games SET home_score = 4, away_score = 12, status = 'final', completed = true, updated_at = NOW()
WHERE id = 'a9cf250a-d621-4734-99c0-9a2a7413ffab';

-- Tauson beat Linette 6-4, 6-2
UPDATE upcoming_games SET home_score = 6, away_score = 12, status = 'final', completed = true, updated_at = NOW()
WHERE id = '10577a03-1988-4604-ab70-10acf6e51e41';

-- Pegula beat Jovic 6-4, 6-2
UPDATE upcoming_games SET home_score = 6, away_score = 12, status = 'final', completed = true, updated_at = NOW()
WHERE id = 'e1b37cd1-539e-4c9d-9742-c86559b244c7';

-- Andreeva beat Cristian 7-5, 6-3
UPDATE upcoming_games SET home_score = 13, away_score = 8, status = 'final', completed = true, updated_at = NOW()
WHERE id = 'c446d8f6-1c7e-4762-99ab-1a71e84f7a4b';

-- Rybakina beat Ruzic 5-7, 6-4, 1-0 RET (Ruzic retired)
UPDATE upcoming_games SET home_score = 12, away_score = 11, status = 'final', completed = true, updated_at = NOW()
WHERE id = '51decd4b-1dc5-4742-ae7a-a9f252fe2799';

-- ============================================================
-- STEP 3: Verify what was updated
-- ============================================================
SELECT ug.id, s.code, ug.home_team_name, ug.away_team_name,
       ug.home_score, ug.away_score, ug.status,
       COUNT(p.id) as preds,
       COUNT(pr.id) as graded
FROM upcoming_games ug
JOIN sports s ON ug.sport_id = s.id
LEFT JOIN predictions p ON p.upcoming_game_id = ug.id
LEFT JOIN prediction_results pr ON pr.prediction_id = p.id AND pr.actual_result IS NOT NULL
WHERE s.code IN ('WTA', 'ATP')
  AND ug.scheduled_at < '2026-02-19 00:00:00'
GROUP BY ug.id, s.code, ug.home_team_name, ug.away_team_name,
         ug.home_score, ug.away_score, ug.status
ORDER BY ug.scheduled_at;

COMMIT;