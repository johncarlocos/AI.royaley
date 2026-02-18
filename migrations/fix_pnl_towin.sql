-- ============================================================================
-- ROYALEY: Recalculate all P/L to "to-win $100" staking model
-- 
-- Old model: flat $100 risk → wins vary ($2 to $300), losses always -$100
-- New model: to-win $100   → wins always +$100, losses vary by odds
--
-- Run: docker exec -i royaley_db psql -U royaley_user -d royaley < fix_pnl_towin.sql
-- ============================================================================

-- Preview before update
SELECT 
    pr.actual_result,
    p.odds_at_prediction as odds,
    pr.profit_loss as old_pnl,
    CASE 
        WHEN pr.actual_result = 'push' THEN 0
        WHEN pr.actual_result = 'win' THEN 100.0
        WHEN pr.actual_result = 'loss' THEN 
            CASE 
                WHEN p.odds_at_prediction > 0 THEN -ROUND(100.0 * 100.0 / p.odds_at_prediction, 2)
                WHEN p.odds_at_prediction < 0 THEN ROUND(p.odds_at_prediction::numeric, 2)
                ELSE -100.0
            END
        ELSE NULL
    END as new_pnl
FROM prediction_results pr
JOIN predictions p ON p.id = pr.prediction_id
WHERE pr.actual_result IN ('win', 'loss', 'push')
  AND p.odds_at_prediction IS NOT NULL
ORDER BY pr.graded_at DESC
LIMIT 20;

-- Apply the recalculation
UPDATE prediction_results pr
SET profit_loss = CASE 
    WHEN pr.actual_result = 'push' THEN 0
    WHEN pr.actual_result = 'win' THEN 100.0
    WHEN pr.actual_result = 'loss' THEN 
        CASE 
            WHEN p.odds_at_prediction > 0 THEN -ROUND(100.0 * 100.0 / p.odds_at_prediction, 2)
            WHEN p.odds_at_prediction < 0 THEN ROUND(p.odds_at_prediction::numeric, 2)
            ELSE -100.0
        END
    ELSE pr.profit_loss
END
FROM predictions p
WHERE p.id = pr.prediction_id
  AND pr.actual_result IN ('win', 'loss', 'push')
  AND p.odds_at_prediction IS NOT NULL;

-- Summary after update
SELECT 
    pr.actual_result,
    COUNT(*) as count,
    ROUND(AVG(pr.profit_loss)::numeric, 2) as avg_pnl,
    ROUND(SUM(pr.profit_loss)::numeric, 2) as total_pnl,
    ROUND(MIN(pr.profit_loss)::numeric, 2) as min_pnl,
    ROUND(MAX(pr.profit_loss)::numeric, 2) as max_pnl
FROM prediction_results pr
WHERE pr.actual_result IN ('win', 'loss', 'push')
GROUP BY pr.actual_result
ORDER BY pr.actual_result;