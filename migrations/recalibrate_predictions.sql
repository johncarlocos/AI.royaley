-- =============================================================================
-- ROYALEY: Recalibrate All Predictions
-- =============================================================================
-- This script applies the same probability dampening used in model_loader.py
-- to all existing predictions in the database.
--
-- Problem: Models trained with data leakage produced overconfident probabilities.
--   - Tier A (65%+) was winning only 38.3%
--   - Tier B (60-65%) was winning only 37.0%
--   - Tier C (55-60%) was winning 55.1% ← this range was honest
--   - Tier D (<55%) was winning 53.3% ← this range was honest
--
-- Fix: Apply 2/3 shrinkage toward 50%, cap at 62%, reassign tiers.
--
-- Formula: new_prob = CLAMP(0.50 + (old_prob - 0.50) * 0.667, 0.38, 0.62)
--
-- New tiers: A=58%+, B=55-58%, C=52-55%, D=<52%
--
-- Run with: docker exec -i royaley_db psql -U royaley royaley < scripts/recalibrate_predictions.sql
-- =============================================================================

BEGIN;

-- Step 0: Backup - save original values
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS probability_original FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS signal_tier_original VARCHAR(1);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS edge_original FLOAT;

UPDATE predictions
SET probability_original = probability,
    signal_tier_original = signal_tier,
    edge_original = edge
WHERE probability_original IS NULL;

-- Step 1: Recalibrate probability
-- For existing predictions, use uniform 50% shrinkage with 58% hard cap.
-- Why 58% max? Because we don't yet have enough validated data to trust
-- any model at Tier A (58%+) level. The highest old predictions (65%+)
-- were actually winning only 38% — so NO existing prediction earns Tier A.
-- 
-- Future predictions use a more targeted fix (framework clamping + weight
-- penalties) that addresses the root cause. This is just for historical data.
--
-- Formula: new = CLAMP(0.50 + (old - 0.50) * 0.50, 0.42, 0.58)
--
-- Mapping:
--   Old 78%+ → New 58% (cap)  → Tier A boundary (most get B)
--   Old 65%  → New 57.5%      → Tier B (demoted from A)
--   Old 60%  → New 55%        → Tier B (demoted from B)
--   Old 58%  → New 54%        → Tier C (preserved)
--   Old 55%  → New 52.5%      → Tier C (preserved)
--   Old 52%  → New 51%        → Tier D
--   Old 50%  → New 50%        → Tier D
UPDATE predictions
SET probability = LEAST(0.58, GREATEST(0.42, 
    0.50 + (probability_original - 0.50) * 0.50
));

-- Step 2: Reassign signal tiers with new thresholds
UPDATE predictions
SET signal_tier = CASE
    WHEN probability >= 0.58 THEN 'A'
    WHEN probability >= 0.55 THEN 'B'
    WHEN probability >= 0.52 THEN 'C'
    ELSE 'D'
END;

-- Step 3: Recalculate edge (probability - implied_probability from odds)
-- American odds to implied probability:
--   negative odds: |odds| / (|odds| + 100)
--   positive odds: 100 / (odds + 100)
UPDATE predictions
SET edge = CASE
    WHEN odds_at_prediction IS NOT NULL AND odds_at_prediction < 0 THEN
        probability - (ABS(odds_at_prediction)::float / (ABS(odds_at_prediction) + 100.0))
    WHEN odds_at_prediction IS NOT NULL AND odds_at_prediction > 0 THEN
        probability - (100.0 / (odds_at_prediction + 100.0))
    ELSE
        0.0
END;

-- Step 4: Recalculate Kelly fraction
-- kelly = (prob * decimal_odds - 1) / (decimal_odds - 1) * 0.25
-- decimal_odds: if odds < 0 → 1 + 100/|odds|, if odds > 0 → 1 + odds/100
-- Capped at 2% of bankroll, 0 for Tier D
UPDATE predictions
SET kelly_fraction = CASE
    WHEN signal_tier = 'D' THEN 0.0
    WHEN edge <= 0 THEN 0.0
    WHEN odds_at_prediction IS NULL THEN 0.0
    ELSE
        LEAST(0.02, GREATEST(0.0,
            CASE
                WHEN odds_at_prediction < 0 THEN
                    (probability * (1.0 + 100.0 / ABS(odds_at_prediction)) - 1.0) 
                    / (100.0 / ABS(odds_at_prediction)) 
                    * 0.25
                WHEN odds_at_prediction > 0 THEN
                    (probability * (1.0 + odds_at_prediction / 100.0) - 1.0) 
                    / (odds_at_prediction / 100.0) 
                    * 0.25
                ELSE 0.0
            END
        ))
END;

-- Step 5: Verify results
SELECT 
    'BEFORE' as period,
    signal_tier_original as tier,
    COUNT(*) as count,
    ROUND(AVG(probability_original)::numeric, 4) as avg_prob
FROM predictions 
WHERE signal_tier_original IS NOT NULL
GROUP BY signal_tier_original
ORDER BY signal_tier_original;

SELECT 
    'AFTER' as period,
    signal_tier as tier,
    COUNT(*) as count,
    ROUND(AVG(probability)::numeric, 4) as avg_prob,
    ROUND(AVG(edge)::numeric, 4) as avg_edge,
    ROUND(AVG(kelly_fraction)::numeric, 6) as avg_kelly
FROM predictions 
GROUP BY signal_tier
ORDER BY signal_tier;

-- Show mapping: how tiers shifted
SELECT 
    signal_tier_original as old_tier,
    signal_tier as new_tier,
    COUNT(*) as count,
    ROUND(AVG(probability_original)::numeric, 4) as avg_old_prob,
    ROUND(AVG(probability)::numeric, 4) as avg_new_prob
FROM predictions
WHERE signal_tier_original IS NOT NULL
GROUP BY signal_tier_original, signal_tier
ORDER BY signal_tier_original, signal_tier;

COMMIT;

-- =============================================================================
-- Expected result: Most old Tier A/B predictions move to C/D (correct!)
-- because their 65%+ probabilities were inflated. Tier C predictions
-- with real 55% signal should mostly stay in B/C range.
-- =============================================================================