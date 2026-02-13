BEGIN;

-- Step 0: Backup original values
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS probability_original FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS signal_tier_original VARCHAR(1);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS edge_original FLOAT;

UPDATE predictions
SET probability_original = probability,
    signal_tier_original = signal_tier::text,
    edge_original = edge
WHERE probability_original IS NULL;

-- Step 1: Recalibrate probability (50% shrinkage, 58% cap)
UPDATE predictions
SET probability = LEAST(0.58, GREATEST(0.42, 
    0.50 + (probability_original - 0.50) * 0.50
));

-- Step 2: Reassign signal tiers (cast to enum)
UPDATE predictions
SET signal_tier = (CASE
    WHEN probability >= 0.58 THEN 'A'
    WHEN probability >= 0.55 THEN 'B'
    WHEN probability >= 0.52 THEN 'C'
    ELSE 'D'
END)::signaltier;

-- Step 3: Recalculate edge
UPDATE predictions
SET edge = CASE
    WHEN odds_at_prediction IS NOT NULL AND odds_at_prediction < 0 THEN
        probability - (ABS(odds_at_prediction)::float / (ABS(odds_at_prediction) + 100.0))
    WHEN odds_at_prediction IS NOT NULL AND odds_at_prediction > 0 THEN
        probability - (100.0 / (odds_at_prediction + 100.0))
    ELSE 0.0
END;

-- Step 4: Recalculate Kelly fraction
UPDATE predictions
SET kelly_fraction = CASE
    WHEN signal_tier::text = 'D' THEN 0.0
    WHEN edge <= 0 THEN 0.0
    WHEN odds_at_prediction IS NULL THEN 0.0
    ELSE LEAST(0.02, GREATEST(0.0,
        CASE
            WHEN odds_at_prediction < 0 THEN
                (probability * (1.0 + 100.0 / ABS(odds_at_prediction)) - 1.0) 
                / (100.0 / ABS(odds_at_prediction)) * 0.25
            WHEN odds_at_prediction > 0 THEN
                (probability * (1.0 + odds_at_prediction / 100.0) - 1.0) 
                / (odds_at_prediction / 100.0) * 0.25
            ELSE 0.0
        END
    ))
END;

-- Step 5: Verify
SELECT 'BEFORE' as period, signal_tier_original as tier, COUNT(*) as count,
    ROUND(AVG(probability_original)::numeric, 4) as avg_prob
FROM predictions WHERE signal_tier_original IS NOT NULL
GROUP BY signal_tier_original ORDER BY signal_tier_original;

SELECT 'AFTER' as period, signal_tier::text as tier, COUNT(*) as count,
    ROUND(AVG(probability)::numeric, 4) as avg_prob,
    ROUND(AVG(edge)::numeric, 4) as avg_edge
FROM predictions GROUP BY signal_tier ORDER BY signal_tier;

SELECT signal_tier_original as old_tier, signal_tier::text as new_tier,
    COUNT(*) as count,
    ROUND(AVG(probability_original)::numeric, 4) as avg_old_prob,
    ROUND(AVG(probability)::numeric, 4) as avg_new_prob
FROM predictions WHERE signal_tier_original IS NOT NULL
GROUP BY signal_tier_original, signal_tier ORDER BY signal_tier_original, signal_tier;

COMMIT;