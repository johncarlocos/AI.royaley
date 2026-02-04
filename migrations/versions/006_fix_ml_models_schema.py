"""Comprehensive ML schema fixes

Revision ID: 006_comprehensive_ml_schema_fixes
Revises: 005_create_master_stats_tables
Create Date: 2026-02-04

This migration fixes ALL schema mismatches between the SQLAlchemy models
and the database tables. This consolidates all manual fixes that were
previously applied:

1. ml_models table:
   - Add file_path, performance_metrics, feature_list columns
   - Make name column nullable

2. training_runs table:
   - Add hyperparameters, validation_metrics, error_message, training_duration_seconds
   - Make several columns nullable

3. game_features table:
   - Add feature_version, computed_at columns

4. calibration_models table:
   - Rename method -> calibrator_type
   - Rename calibration_path -> calibrator_path
   - Drop is_active column

5. mlframework enum:
   - Add deep_learning, quantum, meta_ensemble values
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = '006_comprehensive_ml_schema_fixes'
down_revision = '005_create_master_stats_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Apply all schema fixes."""
    
    connection = op.get_bind()
    
    # =========================================================================
    # 1. ADD NEW ENUM VALUES TO mlframework
    # =========================================================================
    
    # Check and add deep_learning
    op.execute("""
        DO $$ 
        BEGIN
            ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'deep_learning';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    # Check and add quantum
    op.execute("""
        DO $$ 
        BEGIN
            ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'quantum';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    # Check and add meta_ensemble
    op.execute("""
        DO $$ 
        BEGIN
            ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'meta_ensemble';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    # =========================================================================
    # 2. FIX ml_models TABLE
    # =========================================================================
    
    # Add file_path column
    op.execute("""
        ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) DEFAULT '';
    """)
    
    # Add performance_metrics column
    op.execute("""
        ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS performance_metrics JSONB DEFAULT '{}';
    """)
    
    # Add feature_list column
    op.execute("""
        ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS feature_list JSONB DEFAULT '[]';
    """)
    
    # Add hyperparameters column
    op.execute("""
        ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS hyperparameters JSONB DEFAULT '{}';
    """)
    
    # Add training_samples column
    op.execute("""
        ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS training_samples INTEGER DEFAULT 0;
    """)
    
    # Make name nullable (if it exists and has NOT NULL constraint)
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE ml_models ALTER COLUMN name DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    # Migrate data from old columns to new columns if they exist
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'ml_models' AND column_name = 'model_path') THEN
                UPDATE ml_models SET file_path = COALESCE(model_path, '') 
                WHERE file_path = '' OR file_path IS NULL;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'ml_models' AND column_name = 'feature_names') THEN
                UPDATE ml_models SET feature_list = feature_names 
                WHERE feature_list = '[]'::jsonb OR feature_list IS NULL;
            END IF;
        END $$;
    """)
    
    # Migrate individual metric columns to performance_metrics JSONB
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'ml_models' AND column_name = 'accuracy') THEN
                UPDATE ml_models 
                SET performance_metrics = jsonb_build_object(
                    'accuracy', COALESCE(accuracy, 0),
                    'auc', COALESCE(auc, 0),
                    'log_loss', COALESCE(log_loss, 0)
                )
                WHERE performance_metrics = '{}'::jsonb OR performance_metrics IS NULL;
            END IF;
        END $$;
    """)
    
    # =========================================================================
    # 3. FIX training_runs TABLE
    # =========================================================================
    
    # Add missing columns
    op.execute("""
        ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS hyperparameters JSONB DEFAULT '{}';
    """)
    
    op.execute("""
        ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS validation_metrics JSONB DEFAULT '{}';
    """)
    
    op.execute("""
        ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
    """)
    
    op.execute("""
        ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS training_duration_seconds INTEGER;
    """)
    
    # Make columns nullable that the code doesn't always provide
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE training_runs ALTER COLUMN sport_id DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE training_runs ALTER COLUMN bet_type DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE training_runs ALTER COLUMN framework DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE training_runs ALTER COLUMN started_at DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            ALTER TABLE training_runs ALTER COLUMN status DROP NOT NULL;
        EXCEPTION WHEN undefined_column THEN NULL;
        END $$;
    """)
    
    # =========================================================================
    # 4. FIX game_features TABLE
    # =========================================================================
    
    op.execute("""
        ALTER TABLE game_features ADD COLUMN IF NOT EXISTS feature_version VARCHAR(50) DEFAULT 'v1';
    """)
    
    op.execute("""
        ALTER TABLE game_features ADD COLUMN IF NOT EXISTS computed_at TIMESTAMP DEFAULT NOW();
    """)
    
    # =========================================================================
    # 5. FIX calibration_models TABLE
    # =========================================================================
    
    # Get current columns
    inspector = sa.inspect(connection)
    
    try:
        columns = [col['name'] for col in inspector.get_columns('calibration_models')]
    except:
        columns = []
    
    if columns:  # Table exists
        # Rename 'method' to 'calibrator_type' if needed
        if 'method' in columns and 'calibrator_type' not in columns:
            op.execute("""
                ALTER TABLE calibration_models RENAME COLUMN method TO calibrator_type;
            """)
        elif 'method' not in columns and 'calibrator_type' not in columns:
            op.execute("""
                ALTER TABLE calibration_models ADD COLUMN calibrator_type VARCHAR(50);
            """)
        
        # Rename 'calibration_path' to 'calibrator_path' if needed
        if 'calibration_path' in columns and 'calibrator_path' not in columns:
            op.execute("""
                ALTER TABLE calibration_models RENAME COLUMN calibration_path TO calibrator_path;
            """)
        elif 'calibration_path' not in columns and 'calibrator_path' not in columns:
            op.execute("""
                ALTER TABLE calibration_models ADD COLUMN calibrator_path VARCHAR(500);
            """)
        
        # Handle duplicate columns (if both old and new exist, drop new and rename old)
        if 'method' in columns and 'calibrator_type' in columns:
            op.execute("""
                ALTER TABLE calibration_models DROP COLUMN calibrator_type;
            """)
            op.execute("""
                ALTER TABLE calibration_models RENAME COLUMN method TO calibrator_type;
            """)
        
        if 'calibration_path' in columns and 'calibrator_path' in columns:
            op.execute("""
                ALTER TABLE calibration_models DROP COLUMN calibrator_path;
            """)
            op.execute("""
                ALTER TABLE calibration_models RENAME COLUMN calibration_path TO calibrator_path;
            """)
        
        # Drop is_active column if exists (not in model)
        if 'is_active' in columns:
            op.execute("""
                ALTER TABLE calibration_models DROP COLUMN is_active;
            """)


def downgrade():
    """Revert schema changes (partial - some changes are one-way)."""
    
    # Note: Enum values cannot be removed in PostgreSQL without recreating the type
    # So we don't try to remove deep_learning, quantum, meta_ensemble
    
    # Revert calibration_models column names
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'calibration_models' AND column_name = 'calibrator_type') THEN
                ALTER TABLE calibration_models RENAME COLUMN calibrator_type TO method;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'calibration_models' AND column_name = 'calibrator_path') THEN
                ALTER TABLE calibration_models RENAME COLUMN calibrator_path TO calibration_path;
            END IF;
        END $$;
    """)
    
    # Add back is_active
    op.execute("""
        ALTER TABLE calibration_models ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
    """)