"""
ROYALEY - Migration 003: Master Data Architecture
Creates 9 unification tables + adds master_*_id backfill columns.

NON-DESTRUCTIVE: Only adds new tables and nullable columns. Zero data loss risk.

Run: alembic upgrade head
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '003_master_data_architecture'
down_revision = '002_add_injuries'
branch_labels = None
depends_on = None


def upgrade():
    # =========================================================================
    # TIER 3: Infrastructure
    # =========================================================================

    op.create_table(
        'source_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('key', sa.String(50), unique=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('source_type', sa.String(30), nullable=False),
        sa.Column('priority', sa.Integer, default=50),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('provides_teams', sa.Boolean, default=False),
        sa.Column('provides_players', sa.Boolean, default=False),
        sa.Column('provides_games', sa.Boolean, default=False),
        sa.Column('provides_odds', sa.Boolean, default=False),
        sa.Column('provides_stats', sa.Boolean, default=False),
        sa.Column('sports_covered', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        'mapping_audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('entity_type', sa.String(30), nullable=False),
        sa.Column('entity_id', sa.String(100), nullable=False),
        sa.Column('action', sa.String(30), nullable=False),
        sa.Column('old_value', sa.Text, nullable=True),
        sa.Column('new_value', sa.Text, nullable=True),
        sa.Column('performed_by', sa.String(50), default='system'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_audit_entity', 'mapping_audit_log', ['entity_type', 'entity_id'])
    op.create_index('ix_audit_created', 'mapping_audit_log', ['created_at'])

    # =========================================================================
    # TIER 1: Core Master Tables
    # =========================================================================

    op.create_table(
        'master_teams',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('sport_code', sa.String(10), nullable=False, index=True),
        sa.Column('canonical_name', sa.String(150), nullable=False),
        sa.Column('short_name', sa.String(50), nullable=True),
        sa.Column('abbreviation', sa.String(10), nullable=True),
        sa.Column('city', sa.String(100), nullable=True),
        sa.Column('state', sa.String(50), nullable=True),
        sa.Column('conference', sa.String(80), nullable=True),
        sa.Column('division', sa.String(80), nullable=True),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('venues.id'), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('founded_year', sa.Integer, nullable=True),
        sa.Column('metadata_json', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('sport_code', 'canonical_name', name='uq_master_teams_sport_name'),
    )
    op.create_index('ix_master_teams_abbr', 'master_teams', ['abbreviation'])

    op.create_table(
        'master_players',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('sport_code', sa.String(10), nullable=False, index=True),
        sa.Column('master_team_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_teams.id'), nullable=True),
        sa.Column('canonical_name', sa.String(200), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('position', sa.String(50), nullable=True),
        sa.Column('birth_date', sa.Date, nullable=True),
        sa.Column('height_inches', sa.Integer, nullable=True),
        sa.Column('weight_lbs', sa.Integer, nullable=True),
        sa.Column('nationality', sa.String(50), nullable=True),
        sa.Column('hand', sa.String(5), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('metadata_json', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('sport_code', 'canonical_name', name='uq_master_players_sport_name'),
    )
    op.create_index('ix_master_players_team', 'master_players', ['master_team_id'])
    op.create_index('ix_master_players_last', 'master_players', ['last_name'])

    op.create_table(
        'master_games',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('sport_code', sa.String(10), nullable=False, index=True),
        sa.Column('season', sa.Integer, nullable=True),
        sa.Column('season_type', sa.String(20), nullable=True),
        sa.Column('scheduled_at', sa.DateTime, nullable=False),
        sa.Column('home_master_team_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_teams.id'), nullable=True),
        sa.Column('away_master_team_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_teams.id'), nullable=True),
        sa.Column('home_master_player_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_players.id'), nullable=True),
        sa.Column('away_master_player_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_players.id'), nullable=True),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('venues.id'), nullable=True),
        sa.Column('status', sa.String(20), default='scheduled'),
        sa.Column('home_score', sa.Integer, nullable=True),
        sa.Column('away_score', sa.Integer, nullable=True),
        sa.Column('score_detail', postgresql.JSONB, nullable=True),
        sa.Column('is_neutral_site', sa.Boolean, default=False),
        sa.Column('is_playoff', sa.Boolean, default=False),
        sa.Column('primary_source', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_master_games_dedup', 'master_games',
                    ['sport_code', 'scheduled_at', 'home_master_team_id', 'away_master_team_id'])
    op.create_index('ix_master_games_date', 'master_games', ['sport_code', 'scheduled_at'])
    op.create_index('ix_master_games_status', 'master_games', ['status'])

    # =========================================================================
    # TIER 2: Mapping Tables
    # =========================================================================

    op.create_table(
        'team_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_team_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_teams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_team_name', sa.String(200), nullable=False),
        sa.Column('source_external_id', sa.String(200), nullable=True),
        sa.Column('source_team_db_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('verified', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('source_key', 'source_team_name', name='uq_team_map_source_name'),
    )
    op.create_index('ix_team_map_master', 'team_mappings', ['master_team_id'])
    op.create_index('ix_team_map_source', 'team_mappings', ['source_key'])

    op.create_table(
        'player_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_player_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_players.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_player_name', sa.String(200), nullable=False),
        sa.Column('source_external_id', sa.String(200), nullable=True),
        sa.Column('source_player_db_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('verified', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('source_key', 'source_external_id', name='uq_player_map_source_ext'),
    )
    op.create_index('ix_player_map_master', 'player_mappings', ['master_player_id'])
    op.create_index('ix_player_map_source', 'player_mappings', ['source_key'])
    op.create_index('ix_player_map_name', 'player_mappings', ['source_player_name'])

    op.create_table(
        'game_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('master_game_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('master_games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_external_id', sa.String(200), nullable=True),
        sa.Column('source_game_db_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('source_key', 'source_external_id', name='uq_game_map_source_ext'),
    )
    op.create_index('ix_game_map_master', 'game_mappings', ['master_game_id'])
    op.create_index('ix_game_map_source', 'game_mappings', ['source_key'])

    op.create_table(
        'venue_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('venues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_venue_name', sa.String(200), nullable=False),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('source_key', 'source_venue_name', name='uq_venue_map_source_name'),
    )
    op.create_index('ix_venue_map_venue', 'venue_mappings', ['venue_id'])

    # =========================================================================
    # ADD master_*_id columns to EXISTING tables (all nullable, non-destructive)
    # =========================================================================

    # teams → master_team_id
    op.add_column('teams', sa.Column('master_team_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_teams_master_team_id', 'teams', ['master_team_id'])

    # players → master_player_id
    op.add_column('players', sa.Column('master_player_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_players_master_player_id', 'players', ['master_player_id'])

    # games → master_game_id
    op.add_column('games', sa.Column('master_game_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_games_master_game_id', 'games', ['master_game_id'])

    # odds → master_game_id (direct link bypasses game duplication)
    op.add_column('odds', sa.Column('master_game_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_odds_master_game_id', 'odds', ['master_game_id'])

    # player_stats → master_game_id, master_player_id
    op.add_column('player_stats', sa.Column('master_game_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('player_stats', sa.Column('master_player_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_player_stats_master_game', 'player_stats', ['master_game_id'])
    op.create_index('ix_player_stats_master_player', 'player_stats', ['master_player_id'])

    # public_betting → master_game_id
    op.add_column('public_betting', sa.Column('master_game_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_public_betting_master_game', 'public_betting', ['master_game_id'])

    # injuries → master_player_id, master_team_id
    op.add_column('injuries', sa.Column('master_player_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('injuries', sa.Column('master_team_id', postgresql.UUID(as_uuid=True), nullable=True))

    # weather_data — already has venue_id, just needs backfill (no schema change)

    # sportsbooks — fix priority/is_sharp via data script, no schema change


def downgrade():
    # Drop backfill columns from existing tables
    op.drop_index('ix_teams_master_team_id', 'teams')
    op.drop_column('teams', 'master_team_id')

    op.drop_index('ix_players_master_player_id', 'players')
    op.drop_column('players', 'master_player_id')

    op.drop_index('ix_games_master_game_id', 'games')
    op.drop_column('games', 'master_game_id')

    op.drop_index('ix_odds_master_game_id', 'odds')
    op.drop_column('odds', 'master_game_id')

    op.drop_index('ix_player_stats_master_game', 'player_stats')
    op.drop_index('ix_player_stats_master_player', 'player_stats')
    op.drop_column('player_stats', 'master_game_id')
    op.drop_column('player_stats', 'master_player_id')

    op.drop_index('ix_public_betting_master_game', 'public_betting')
    op.drop_column('public_betting', 'master_game_id')

    op.drop_column('injuries', 'master_player_id')
    op.drop_column('injuries', 'master_team_id')

    # Drop mapping tables
    op.drop_table('venue_mappings')
    op.drop_table('game_mappings')
    op.drop_table('player_mappings')
    op.drop_table('team_mappings')

    # Drop core tables
    op.drop_table('master_games')
    op.drop_table('master_players')
    op.drop_table('master_teams')

    # Drop infrastructure
    op.drop_table('mapping_audit_log')
    op.drop_table('source_registry')
