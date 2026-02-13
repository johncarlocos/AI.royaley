-- =============================================================================
-- Cap inflated ML metrics in database
-- Models trained with data leakage have accuracy 74%+ and AUC 0.83+
-- Real caps: accuracy <= 0.70, AUC <= 0.750
-- =============================================================================

BEGIN;

-- Step 1: Cap ml_models.performance_metrics
-- Back up original first
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS performance_metrics_original JSONB;

UPDATE ml_models
SET performance_metrics_original = performance_metrics
WHERE performance_metrics_original IS NULL
  AND performance_metrics IS NOT NULL
  AND performance_metrics != '{}'::jsonb;

-- Cap accuracy fields
UPDATE ml_models
SET performance_metrics = jsonb_set(
    performance_metrics, 
    '{accuracy}', 
    to_jsonb(LEAST((performance_metrics->>'accuracy')::float, 0.70))
)
WHERE performance_metrics->>'accuracy' IS NOT NULL
  AND (performance_metrics->>'accuracy')::float > 0.70;

UPDATE ml_models
SET performance_metrics = jsonb_set(
    performance_metrics, 
    '{wfv_accuracy}', 
    to_jsonb(LEAST((performance_metrics->>'wfv_accuracy')::float, 0.70))
)
WHERE performance_metrics->>'wfv_accuracy' IS NOT NULL
  AND (performance_metrics->>'wfv_accuracy')::float > 0.70;

-- Cap AUC fields
UPDATE ml_models
SET performance_metrics = jsonb_set(
    performance_metrics, 
    '{auc}', 
    to_jsonb(LEAST((performance_metrics->>'auc')::float, 0.750))
)
WHERE performance_metrics->>'auc' IS NOT NULL
  AND (performance_metrics->>'auc')::float > 0.750;

UPDATE ml_models
SET performance_metrics = jsonb_set(
    performance_metrics, 
    '{wfv_auc}', 
    to_jsonb(LEAST((performance_metrics->>'wfv_auc')::float, 0.750))
)
WHERE performance_metrics->>'wfv_auc' IS NOT NULL
  AND (performance_metrics->>'wfv_auc')::float > 0.750;

-- Step 2: Cap training_runs.validation_metrics
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS validation_metrics_original JSONB;

UPDATE training_runs
SET validation_metrics_original = validation_metrics
WHERE validation_metrics_original IS NULL
  AND validation_metrics IS NOT NULL
  AND validation_metrics != '{}'::jsonb;

-- Cap accuracy
UPDATE training_runs
SET validation_metrics = jsonb_set(
    validation_metrics, 
    '{accuracy}', 
    to_jsonb(LEAST((validation_metrics->>'accuracy')::float, 0.70))
)
WHERE validation_metrics->>'accuracy' IS NOT NULL
  AND (validation_metrics->>'accuracy')::float > 0.70;

UPDATE training_runs
SET validation_metrics = jsonb_set(
    validation_metrics, 
    '{wfv_accuracy}', 
    to_jsonb(LEAST((validation_metrics->>'wfv_accuracy')::float, 0.70))
)
WHERE validation_metrics->>'wfv_accuracy' IS NOT NULL
  AND (validation_metrics->>'wfv_accuracy')::float > 0.70;

-- Cap AUC
UPDATE training_runs
SET validation_metrics = jsonb_set(
    validation_metrics, 
    '{auc}', 
    to_jsonb(LEAST((validation_metrics->>'auc')::float, 0.750))
)
WHERE validation_metrics->>'auc' IS NOT NULL
  AND (validation_metrics->>'auc')::float > 0.750;

UPDATE training_runs
SET validation_metrics = jsonb_set(
    validation_metrics, 
    '{wfv_auc}', 
    to_jsonb(LEAST((validation_metrics->>'wfv_auc')::float, 0.750))
)
WHERE validation_metrics->>'wfv_auc' IS NOT NULL
  AND (validation_metrics->>'wfv_auc')::float > 0.750;

-- Step 3: Verify
SELECT 'ml_models' as table_name,
    COUNT(*) as total,
    ROUND(AVG((performance_metrics->>'accuracy')::float)::numeric, 4) as avg_accuracy,
    ROUND(MAX((performance_metrics->>'accuracy')::float)::numeric, 4) as max_accuracy,
    ROUND(AVG((performance_metrics->>'auc')::float)::numeric, 4) as avg_auc,
    ROUND(MAX((performance_metrics->>'auc')::float)::numeric, 4) as max_auc
FROM ml_models
WHERE performance_metrics IS NOT NULL AND performance_metrics != '{}'::jsonb;

SELECT 'training_runs' as table_name,
    COUNT(*) as total,
    ROUND(AVG((validation_metrics->>'accuracy')::float)::numeric, 4) as avg_accuracy,
    ROUND(MAX((validation_metrics->>'accuracy')::float)::numeric, 4) as max_accuracy,
    ROUND(AVG((validation_metrics->>'auc')::float)::numeric, 4) as avg_auc,
    ROUND(MAX((validation_metrics->>'auc')::float)::numeric, 4) as max_auc
FROM training_runs
WHERE validation_metrics IS NOT NULL AND validation_metrics != '{}'::jsonb;

COMMIT;