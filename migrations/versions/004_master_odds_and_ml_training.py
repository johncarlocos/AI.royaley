"""
ROYALEY - Migration 004: Master Odds + ML Training Dataset
Creates 3 tables: master_odds, odds_mappings, ml_training_dataset.

NON-DESTRUCTIVE: Only adds new tables. Zero data loss risk.

Run: alembic upgrade head
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '004_master_odds_and_ml_training'
down_revision = '003_master_data_architecture'
branch_labels = None
depends_on = None


def upgrade():
    # =========================================================================
    # MASTER ODDS — One canonical row per (game × sportsbook × bet_type)
    # =========================================================================

    op.create_table(
        'master_odds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_game_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sportsbook_key', sa.String(50), nullable=False),
        sa.Column('sportsbook_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('sportsbooks.id'), nullable=True),
        sa.Column('bet_type', sa.String(30), nullable=False),
        sa.Column('period', sa.String(20), server_default='full'),

        # Spread
        sa.Column('opening_line', sa.Float, nullable=True),
        sa.Column('closing_line', sa.Float, nullable=True),
        sa.Column('opening_odds_home', sa.Integer, nullable=True),
        sa.Column('opening_odds_away', sa.Integer, nullable=True),
        sa.Column('closing_odds_home', sa.Integer, nullable=True),
        sa.Column('closing_odds_away', sa.Integer, nullable=True),

        # Total
        sa.Column('opening_total', sa.Float, nullable=True),
        sa.Column('closing_total', sa.Float, nullable=True),
        sa.Column('opening_over_odds', sa.Integer, nullable=True),
        sa.Column('opening_under_odds', sa.Integer, nullable=True),
        sa.Column('closing_over_odds', sa.Integer, nullable=True),
        sa.Column('closing_under_odds', sa.Integer, nullable=True),

        # Computed
        sa.Column('line_movement', sa.Float, nullable=True),
        sa.Column('no_vig_prob_home', sa.Float, nullable=True),

        # Metadata
        sa.Column('is_sharp', sa.Boolean, server_default='false'),
        sa.Column('num_source_records', sa.Integer, server_default='1'),
        sa.Column('first_seen_at', sa.DateTime, nullable=True),
        sa.Column('last_seen_at', sa.DateTime, nullable=True),

        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),

        sa.UniqueConstraint('master_game_id', 'sportsbook_key', 'bet_type', 'period',
                            name='uq_master_odds_game_book_type'),
    )

    op.create_index('ix_master_odds_game', 'master_odds', ['master_game_id'])
    op.create_index('ix_master_odds_book', 'master_odds', ['sportsbook_key'])
    op.create_index('ix_master_odds_sharp', 'master_odds', ['is_sharp'])

    # =========================================================================
    # ODDS MAPPINGS — Links raw odds → master_odds
    # =========================================================================

    op.create_table(
        'odds_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_odds_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_odds.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_odds_db_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index('ix_odds_map_master', 'odds_mappings', ['master_odds_id'])
    op.create_index('ix_odds_map_source', 'odds_mappings', ['source_key'])

    # =========================================================================
    # ML TRAINING DATASET — Materialized feature table for H2O / AutoGluon
    # =========================================================================

    op.create_table(
        'ml_training_dataset',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_game_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_games.id', ondelete='CASCADE'),
                  nullable=False, unique=True),

        # Identifiers
        sa.Column('sport_code', sa.String(10), nullable=False),
        sa.Column('season', sa.Integer, nullable=True),
        sa.Column('scheduled_at', sa.DateTime, nullable=False),
        sa.Column('home_team', sa.String(150), nullable=True),
        sa.Column('away_team', sa.String(150), nullable=True),

        # Targets
        sa.Column('home_score', sa.Integer, nullable=True),
        sa.Column('away_score', sa.Integer, nullable=True),
        sa.Column('home_win', sa.Integer, nullable=True),
        sa.Column('total_points', sa.Integer, nullable=True),
        sa.Column('score_margin', sa.Integer, nullable=True),

        # Odds features (from master_odds)
        sa.Column('spread_open', sa.Float, nullable=True),
        sa.Column('spread_close', sa.Float, nullable=True),
        sa.Column('spread_movement', sa.Float, nullable=True),
        sa.Column('moneyline_home', sa.Integer, nullable=True),
        sa.Column('moneyline_away', sa.Integer, nullable=True),
        sa.Column('total_open', sa.Float, nullable=True),
        sa.Column('total_close', sa.Float, nullable=True),
        sa.Column('total_movement', sa.Float, nullable=True),
        sa.Column('pinnacle_spread', sa.Float, nullable=True),
        sa.Column('pinnacle_ml_home', sa.Integer, nullable=True),
        sa.Column('pinnacle_total', sa.Float, nullable=True),
        sa.Column('num_books_with_odds', sa.Integer, nullable=True),
        sa.Column('consensus_spread', sa.Float, nullable=True),
        sa.Column('consensus_total', sa.Float, nullable=True),
        sa.Column('implied_prob_home', sa.Float, nullable=True),
        sa.Column('no_vig_prob_home', sa.Float, nullable=True),

        # Public betting
        sa.Column('public_spread_home_pct', sa.Float, nullable=True),
        sa.Column('public_ml_home_pct', sa.Float, nullable=True),
        sa.Column('public_total_over_pct', sa.Float, nullable=True),
        sa.Column('public_money_spread_home_pct', sa.Float, nullable=True),
        sa.Column('sharp_action_indicator', sa.Boolean, nullable=True),
        sa.Column('is_rlm_spread', sa.Boolean, nullable=True),

        # Weather
        sa.Column('temperature_f', sa.Float, nullable=True),
        sa.Column('wind_speed_mph', sa.Float, nullable=True),
        sa.Column('precipitation_pct', sa.Float, nullable=True),
        sa.Column('is_dome', sa.Boolean, nullable=True),
        sa.Column('humidity_pct', sa.Float, nullable=True),

        # Injuries
        sa.Column('home_injuries_out', sa.Integer, nullable=True),
        sa.Column('away_injuries_out', sa.Integer, nullable=True),
        sa.Column('home_injury_impact', sa.Float, nullable=True),
        sa.Column('away_injury_impact', sa.Float, nullable=True),
        sa.Column('home_starter_out', sa.Integer, nullable=True),
        sa.Column('away_starter_out', sa.Integer, nullable=True),

        # Context
        sa.Column('is_playoff', sa.Boolean, nullable=True),
        sa.Column('is_neutral_site', sa.Boolean, nullable=True),

        # Extensibility
        sa.Column('extra_features', postgresql.JSONB, nullable=True),

        # Metadata
        sa.Column('feature_version', sa.String(20), server_default='2.0'),
        sa.Column('computed_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index('ix_ml_training_sport_season', 'ml_training_dataset', ['sport_code', 'season'])
    op.create_index('ix_ml_training_date', 'ml_training_dataset', ['scheduled_at'])
    op.create_index('ix_ml_training_home_win', 'ml_training_dataset', ['home_win'])


def downgrade():
    op.drop_table('ml_training_dataset')
    op.drop_table('odds_mappings')
    op.drop_table('master_odds')
