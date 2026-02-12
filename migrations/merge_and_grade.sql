-- Fix: Transfer scores from completed duplicates to prediction-linked games
-- The pipeline and scheduler created separate game rows for the same real games.

-- Step 1: Update prediction-linked games with scores from completed duplicates
-- Match by: same sport_id + home_team_name + same scheduled_at (within 1 hour)
UPDATE upcoming_games pred_game
SET 
    home_score = score_game.home_score,
    away_score = score_game.away_score,
    status = 'completed',
    updated_at = NOW()
FROM upcoming_games score_game
WHERE score_game.status = 'completed'
  AND pred_game.status = 'scheduled'
  AND pred_game.id IN (SELECT DISTINCT upcoming_game_id FROM predictions WHERE upcoming_game_id IS NOT NULL)
  AND pred_game.sport_id = score_game.sport_id
  AND pred_game.home_team_name = score_game.home_team_name
  AND ABS(EXTRACT(EPOCH FROM (pred_game.scheduled_at - score_game.scheduled_at))) < 7200;

-- Step 2: Delete orphan completed games (duplicates with no predictions)
DELETE FROM upcoming_games 
WHERE status = 'completed' 
  AND id NOT IN (SELECT DISTINCT upcoming_game_id FROM predictions WHERE upcoming_game_id IS NOT NULL);

-- Step 3: Clear stale prediction_results (all 9 rows are pending from closing line capture)
TRUNCATE prediction_results;

-- Step 4: Grade predictions for all completed games
INSERT INTO prediction_results (prediction_id, actual_result, profit_loss, graded_at)
SELECT 
    p.id,
    -- Determine result
    CASE 
        WHEN p.bet_type = 'spread' THEN
            CASE 
                WHEN p.predicted_side ILIKE '%' || ug.home_team_name || '%' THEN
                    CASE 
                        WHEN (ug.home_score - ug.away_score) + p.line_at_prediction > 0 THEN 'win'
                        WHEN (ug.home_score - ug.away_score) + p.line_at_prediction < 0 THEN 'loss'
                        ELSE 'push'
                    END
                ELSE
                    CASE 
                        WHEN (ug.away_score - ug.home_score) + ABS(p.line_at_prediction) > 0 THEN 'win'
                        WHEN (ug.away_score - ug.home_score) + ABS(p.line_at_prediction) < 0 THEN 'loss'
                        ELSE 'push'
                    END
            END
        WHEN p.bet_type = 'total' THEN
            CASE 
                WHEN p.predicted_side ILIKE 'over%' THEN
                    CASE 
                        WHEN (ug.home_score + ug.away_score) > p.line_at_prediction THEN 'win'
                        WHEN (ug.home_score + ug.away_score) < p.line_at_prediction THEN 'loss'
                        ELSE 'push'
                    END
                ELSE
                    CASE 
                        WHEN (ug.home_score + ug.away_score) < p.line_at_prediction THEN 'win'
                        WHEN (ug.home_score + ug.away_score) > p.line_at_prediction THEN 'loss'
                        ELSE 'push'
                    END
            END
        WHEN p.bet_type = 'moneyline' THEN
            CASE 
                WHEN p.predicted_side ILIKE '%' || ug.home_team_name || '%' THEN
                    CASE 
                        WHEN ug.home_score > ug.away_score THEN 'win'
                        WHEN ug.home_score < ug.away_score THEN 'loss'
                        ELSE 'push'
                    END
                ELSE
                    CASE 
                        WHEN ug.away_score > ug.home_score THEN 'win'
                        WHEN ug.away_score < ug.home_score THEN 'loss'
                        ELSE 'push'
                    END
            END
        ELSE 'pending'
    END,
    -- Calculate profit/loss (flat $100 bet, -110 standard vig)
    CASE 
        WHEN p.bet_type IN ('spread', 'total') THEN
            -- Use the same result logic to determine P/L
            CASE
                WHEN (
                    CASE 
                        WHEN p.bet_type = 'spread' AND p.predicted_side ILIKE '%' || ug.home_team_name || '%' THEN
                            (ug.home_score - ug.away_score) + p.line_at_prediction
                        WHEN p.bet_type = 'spread' THEN
                            (ug.away_score - ug.home_score) + ABS(p.line_at_prediction)
                        WHEN p.bet_type = 'total' AND p.predicted_side ILIKE 'over%' THEN
                            (ug.home_score + ug.away_score) - p.line_at_prediction
                        ELSE
                            p.line_at_prediction - (ug.home_score + ug.away_score)
                    END
                ) > 0 THEN 
                    CASE WHEN p.odds_at_prediction IS NOT NULL AND p.odds_at_prediction > 0 
                         THEN p.odds_at_prediction * 1.0
                         WHEN p.odds_at_prediction IS NOT NULL AND p.odds_at_prediction < 0 
                         THEN 10000.0 / ABS(p.odds_at_prediction)
                         ELSE 91.0 END
                WHEN (
                    CASE 
                        WHEN p.bet_type = 'spread' AND p.predicted_side ILIKE '%' || ug.home_team_name || '%' THEN
                            (ug.home_score - ug.away_score) + p.line_at_prediction
                        WHEN p.bet_type = 'spread' THEN
                            (ug.away_score - ug.home_score) + ABS(p.line_at_prediction)
                        WHEN p.bet_type = 'total' AND p.predicted_side ILIKE 'over%' THEN
                            (ug.home_score + ug.away_score) - p.line_at_prediction
                        ELSE
                            p.line_at_prediction - (ug.home_score + ug.away_score)
                    END
                ) < 0 THEN -100.0
                ELSE 0.0
            END
        WHEN p.bet_type = 'moneyline' THEN
            CASE 
                WHEN (p.predicted_side ILIKE '%' || ug.home_team_name || '%' AND ug.home_score > ug.away_score)
                  OR (NOT p.predicted_side ILIKE '%' || ug.home_team_name || '%' AND ug.away_score > ug.home_score) THEN
                    CASE WHEN p.odds_at_prediction > 0 THEN p.odds_at_prediction * 1.0
                         WHEN p.odds_at_prediction < 0 THEN 10000.0 / ABS(p.odds_at_prediction)
                         ELSE 91.0 END
                WHEN (p.predicted_side ILIKE '%' || ug.home_team_name || '%' AND ug.home_score < ug.away_score)
                  OR (NOT p.predicted_side ILIKE '%' || ug.home_team_name || '%' AND ug.away_score < ug.home_score) THEN -100.0
                ELSE 0.0
            END
        ELSE 0.0
    END,
    NOW()
FROM predictions p
JOIN upcoming_games ug ON p.upcoming_game_id = ug.id
WHERE ug.status = 'completed'
  AND ug.home_score IS NOT NULL
  AND ug.away_score IS NOT NULL;

-- Step 5: Verify results
SELECT 'Completed games with predictions' as label, COUNT(*) as count 
FROM upcoming_games WHERE status = 'completed' AND id IN (SELECT DISTINCT upcoming_game_id FROM predictions)
UNION ALL
SELECT 'Graded predictions', COUNT(*) FROM prediction_results WHERE actual_result != 'pending'
UNION ALL
SELECT 'Wins', COUNT(*) FROM prediction_results WHERE actual_result = 'win'
UNION ALL
SELECT 'Losses', COUNT(*) FROM prediction_results WHERE actual_result = 'loss'
UNION ALL
SELECT 'Pushes', COUNT(*) FROM prediction_results WHERE actual_result = 'push'
UNION ALL
SELECT 'Total P/L', ROUND(SUM(profit_loss)::numeric, 0) FROM prediction_results;