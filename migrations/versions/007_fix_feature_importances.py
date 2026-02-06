"""Fix feature_importances table schema

Revision ID: 007_fix_feature_importances
Revises: 006_comprehensive_ml_schema_fixes
Create Date: 2026-02-07

This migration fixes the feature_importances table to match the SQLAlchemy model:
- Ensures importance_score column exists (may have been named feature_value)
- Ensures importance_rank column exists
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '007_fix_feature_importances'
down_revision = '006_comprehensive_ml_schema_fixes'
branch_labels = None
depends_on = None


def upgrade():
    """Fix feature_importances table schema."""
    
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    
    # Check if table exists
    tables = inspector.get_table_names()
    
    if 'feature_importances' not in tables:
        # Create the table from scratch
        op.execute("""
            CREATE TABLE feature_importances (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
                feature_name VARCHAR(100) NOT NULL,
                importance_score FLOAT NOT NULL DEFAULT 0,
                importance_rank INTEGER NOT NULL DEFAULT 0
            );
        """)
        
        op.execute("""
            CREATE INDEX ix_feature_importances_model_id 
            ON feature_importances(model_id);
        """)
    else:
        # Table exists - fix columns
        columns = [col['name'] for col in inspector.get_columns('feature_importances')]
        
        # If feature_value exists but importance_score doesn't, rename it
        if 'feature_value' in columns and 'importance_score' not in columns:
            op.execute("""
                ALTER TABLE feature_importances 
                RENAME COLUMN feature_value TO importance_score;
            """)
        elif 'importance_score' not in columns:
            # Add importance_score if it doesn't exist
            op.execute("""
                ALTER TABLE feature_importances 
                ADD COLUMN importance_score FLOAT NOT NULL DEFAULT 0;
            """)
        
        # Add importance_rank if missing
        if 'importance_rank' not in columns:
            op.execute("""
                ALTER TABLE feature_importances 
                ADD COLUMN importance_rank INTEGER NOT NULL DEFAULT 0;
            """)
        
        # Ensure model_id foreign key exists
        if 'model_id' not in columns:
            op.execute("""
                ALTER TABLE feature_importances 
                ADD COLUMN model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE;
            """)


def downgrade():
    """Revert feature_importances changes."""
    
    # Rename importance_score back to feature_value
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'feature_importances' 
                       AND column_name = 'importance_score') THEN
                ALTER TABLE feature_importances 
                RENAME COLUMN importance_score TO feature_value;
            END IF;
        END $$;
    """)