"""Fix ml_models schema to match SQLAlchemy model

Revision ID: 006
Revises: 005
Create Date: 2026-02-04

The SQLAlchemy model uses:
- file_path (not model_path)
- performance_metrics (JSONB)
- feature_list (not feature_names)

This migration updates the database schema to match.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    # Add new values to mlframework enum if they don't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'deep_learning' AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'mlframework')) THEN
                ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'deep_learning';
            END IF;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'quantum' AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'mlframework')) THEN
                ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'quantum';
            END IF;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)
    
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'meta_ensemble' AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'mlframework')) THEN
                ALTER TYPE mlframework ADD VALUE IF NOT EXISTS 'meta_ensemble';
            END IF;
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # Add file_path column if it doesn't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'file_path'
            ) THEN
                ALTER TABLE ml_models ADD COLUMN file_path VARCHAR(500) DEFAULT '';
            END IF;
        END $$;
    """)
    
    # Add performance_metrics column if it doesn't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'performance_metrics'
            ) THEN
                ALTER TABLE ml_models ADD COLUMN performance_metrics JSONB DEFAULT '{}';
            END IF;
        END $$;
    """)
    
    # Add feature_list column if it doesn't exist
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'feature_list'
            ) THEN
                ALTER TABLE ml_models ADD COLUMN feature_list JSONB DEFAULT '[]';
            END IF;
        END $$;
    """)
    
    # Copy data from model_path to file_path if model_path exists
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'model_path'
            ) THEN
                UPDATE ml_models SET file_path = COALESCE(model_path, '') WHERE file_path = '' OR file_path IS NULL;
            END IF;
        END $$;
    """)
    
    # Copy data from feature_names to feature_list if feature_names exists
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'feature_names'
            ) THEN
                UPDATE ml_models SET feature_list = feature_names WHERE feature_list = '[]'::jsonb OR feature_list IS NULL;
            END IF;
        END $$;
    """)
    
    # Migrate accuracy/auc/log_loss to performance_metrics if they exist
    op.execute("""
        DO $$ 
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'accuracy'
            ) THEN
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


def downgrade():
    # Add back old columns if needed
    op.execute("""
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'ml_models' AND column_name = 'model_path'
            ) THEN
                ALTER TABLE ml_models ADD COLUMN model_path VARCHAR(500);
            END IF;
        END $$;
    """)
    
    # Copy data back
    op.execute("""
        UPDATE ml_models SET model_path = file_path WHERE model_path IS NULL;
    """)
