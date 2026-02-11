-- ROYALEY Migration: Fix prediction_results table
-- Run: cat fix_prediction_results.sql | docker exec -i royaley_postgres psql -U royaley -d royaley

-- Check current columns
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'prediction_results' ORDER BY ordinal_position;

-- If the table exists but is missing columns, recreate it cleanly
-- First drop if empty (safe since no graded predictions exist yet)
DROP TABLE IF EXISTS prediction_results CASCADE;

-- Recreate with correct schema
CREATE TABLE prediction_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
    actual_result VARCHAR(20) NOT NULL DEFAULT 'pending',
    closing_line FLOAT,
    closing_odds INT,
    clv FLOAT,
    profit_loss FLOAT,
    graded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ix_prediction_results_prediction ON prediction_results(prediction_id);
CREATE INDEX ix_prediction_results_result ON prediction_results(actual_result);
CREATE INDEX ix_prediction_results_graded ON prediction_results(graded_at);

-- Verify
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'prediction_results' ORDER BY ordinal_position;