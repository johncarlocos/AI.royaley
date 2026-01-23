"""
ROYALEY - Database Migration: Add Injury Tables
Adds injury tracking for players

Run: alembic upgrade head
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '002_add_injuries'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade():
    # Create injuries table
    op.create_table(
        'injuries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('player_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('players.id', ondelete='CASCADE'), nullable=True),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id'), nullable=False),
        sa.Column('sport_code', sa.String(10), nullable=False),
        
        # Player info (in case player not in players table)
        sa.Column('player_name', sa.String(200), nullable=False),
        sa.Column('position', sa.String(50), nullable=True),
        
        # Injury details
        sa.Column('injury_type', sa.String(200), nullable=True),
        sa.Column('body_part', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),  # Out, Doubtful, Questionable, Probable, IR
        sa.Column('status_detail', sa.String(200), nullable=True),
        
        # Impact
        sa.Column('games_missed', sa.Integer, default=0),
        sa.Column('is_starter', sa.Boolean, default=False),
        sa.Column('impact_score', sa.Float, default=0.0),  # 0-1 scale
        
        # Dates
        sa.Column('injury_date', sa.Date, nullable=True),
        sa.Column('expected_return', sa.Date, nullable=True),
        sa.Column('first_reported', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_updated', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        
        # Source
        sa.Column('source', sa.String(50), default='espn'),
        sa.Column('external_id', sa.String(100), nullable=True),
        
        # Indexes
        sa.Index('ix_injuries_team_status', 'team_id', 'status'),
        sa.Index('ix_injuries_sport_date', 'sport_code', 'last_updated'),
        sa.Index('ix_injuries_player', 'player_name'),
    )
    
    # Create game_injuries junction table (injuries affecting specific games)
    op.create_table(
        'game_injuries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('injury_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('injuries.id', ondelete='CASCADE'), nullable=False),
        sa.Column('team_side', sa.String(10), nullable=False),  # home, away
        sa.Column('impact_on_game', sa.Float, default=0.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        
        sa.UniqueConstraint('game_id', 'injury_id', name='uq_game_injuries'),
    )


def downgrade():
    op.drop_table('game_injuries')
    op.drop_table('injuries')
