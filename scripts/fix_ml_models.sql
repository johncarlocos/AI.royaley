-- Fix ml_models schema to match SQLAlchemy model
-- Run this script directly if migrations aren't working:
-- docker exec -it royaley_postgres psql -U royaley -d royaley -f /path/to/fix_ml_models.sql
-- OR copy/paste into psql

-- Add new enum values
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

-- Add file_path column
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) DEFAULT '';

-- Add performance_metrics column
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS performance_metrics JSONB DEFAULT '{}';

-- Add feature_list column
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS feature_list JSONB DEFAULT '[]';

-- Copy data from old columns if they exist
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ml_models' AND column_name = 'model_path') THEN
        UPDATE ml_models SET file_path = COALESCE(model_path, '') WHERE file_path = '' OR file_path IS NULL;
    END IF;
END $$;

DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ml_models' AND column_name = 'feature_names') THEN
        UPDATE ml_models SET feature_list = feature_names WHERE feature_list = '[]'::jsonb OR feature_list IS NULL;
    END IF;
END $$;

-- Migrate accuracy/auc/log_loss to performance_metrics
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ml_models' AND column_name = 'accuracy') THEN
        UPDATE ml_models 
        SET performance_metrics = jsonb_build_object(
            'accuracy', COALESCE(accuracy, 0),
            'auc', COALESCE(auc, 0),
            'log_loss', COALESCE(log_loss, 0)
        )
        WHERE performance_metrics = '{}'::jsonb OR performance_metrics IS NULL;
    END IF;
END $$;

-- Verify the changes
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'ml_models' 
ORDER BY ordinal_position;
