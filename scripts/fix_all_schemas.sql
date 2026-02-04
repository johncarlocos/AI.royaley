-- ============================================================================
-- ROYALEY Comprehensive Schema Fix Script
-- ============================================================================
-- Run this ONCE on your current database to fix all schema mismatches.
-- Future rebuilds will use the Alembic migration instead.
--
-- Usage:
-- docker exec -it royaley_postgres psql -U royaley -d royaley -f /path/to/fix_all_schemas.sql
-- OR copy/paste into psql
-- ============================================================================

-- ============================================================================
-- 1. ADD NEW ENUM VALUES TO mlframework
-- ============================================================================

DO $$ 
BEGIN
    ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'deep_learning';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ 
BEGIN
    ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'quantum';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ 
BEGIN
    ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'meta_ensemble';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- 2. FIX ml_models TABLE
-- ============================================================================

ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) DEFAULT '';
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS performance_metrics JSONB DEFAULT '{}';
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS feature_list JSONB DEFAULT '[]';
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS hyperparameters JSONB DEFAULT '{}';
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS training_samples INTEGER DEFAULT 0;

-- Make name nullable
DO $$
BEGIN
    ALTER TABLE ml_models ALTER COLUMN name DROP NOT NULL;
EXCEPTION WHEN undefined_column THEN NULL;
END $$;

-- ============================================================================
-- 3. FIX training_runs TABLE
-- ============================================================================

ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS hyperparameters JSONB DEFAULT '{}';
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS validation_metrics JSONB DEFAULT '{}';
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS training_duration_seconds INTEGER;

-- Make columns nullable
DO $$ BEGIN ALTER TABLE training_runs ALTER COLUMN sport_id DROP NOT NULL; EXCEPTION WHEN undefined_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE training_runs ALTER COLUMN bet_type DROP NOT NULL; EXCEPTION WHEN undefined_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE training_runs ALTER COLUMN framework DROP NOT NULL; EXCEPTION WHEN undefined_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE training_runs ALTER COLUMN started_at DROP NOT NULL; EXCEPTION WHEN undefined_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE training_runs ALTER COLUMN status DROP NOT NULL; EXCEPTION WHEN undefined_column THEN NULL; END $$;

-- ============================================================================
-- 4. FIX game_features TABLE
-- ============================================================================

ALTER TABLE game_features ADD COLUMN IF NOT EXISTS feature_version VARCHAR(50) DEFAULT 'v1';
ALTER TABLE game_features ADD COLUMN IF NOT EXISTS computed_at TIMESTAMP DEFAULT NOW();

-- ============================================================================
-- 5. FIX calibration_models TABLE (handle duplicates from manual fixes)
-- ============================================================================

-- First drop any duplicate columns that were manually added
ALTER TABLE calibration_models DROP COLUMN IF EXISTS calibrator_type;
ALTER TABLE calibration_models DROP COLUMN IF EXISTS calibrator_path;

-- Rename original columns
DO $$ 
BEGIN 
    ALTER TABLE calibration_models RENAME COLUMN method TO calibrator_type;
EXCEPTION WHEN undefined_column THEN NULL;
END $$;

DO $$ 
BEGIN 
    ALTER TABLE calibration_models RENAME COLUMN calibration_path TO calibrator_path;
EXCEPTION WHEN undefined_column THEN NULL;
END $$;

-- Add columns if they still don't exist
ALTER TABLE calibration_models ADD COLUMN IF NOT EXISTS calibrator_type VARCHAR(50);
ALTER TABLE calibration_models ADD COLUMN IF NOT EXISTS calibrator_path VARCHAR(500);

-- Drop is_active (not in model)
ALTER TABLE calibration_models DROP COLUMN IF EXISTS is_active;

-- ============================================================================
-- 6. VERIFY FINAL SCHEMA
-- ============================================================================

SELECT 'ml_models columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'ml_models' 
ORDER BY ordinal_position;

SELECT 'training_runs columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'training_runs' 
ORDER BY ordinal_position;

SELECT 'calibration_models columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'calibration_models' 
ORDER BY ordinal_position;

SELECT 'game_features columns:' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'game_features' 
ORDER BY ordinal_position;

SELECT 'Schema fix complete!' as status;
