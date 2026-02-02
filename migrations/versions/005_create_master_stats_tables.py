"""Create master_player_stats and master_team_stats tables

Revision ID: 005
Revises: 004
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    # =========================================================================
    # MASTER_PLAYER_STATS - Consolidated player stats per game
    # =========================================================================
    op.create_table(
        'master_player_stats',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('master_player_id', UUID(as_uuid=True), sa.ForeignKey('master_players.id'), nullable=False),
        sa.Column('master_game_id', UUID(as_uuid=True), sa.ForeignKey('master_games.id'), nullable=False),
        sa.Column('master_team_id', UUID(as_uuid=True), sa.ForeignKey('master_teams.id'), nullable=True),
        sa.Column('sport_code', sa.String(10), nullable=False),
        
        # Common stats across sports
        sa.Column('minutes_played', sa.Float, nullable=True),
        sa.Column('points', sa.Integer, nullable=True),
        sa.Column('assists', sa.Integer, nullable=True),
        sa.Column('rebounds', sa.Integer, nullable=True),
        sa.Column('steals', sa.Integer, nullable=True),
        sa.Column('blocks', sa.Integer, nullable=True),
        sa.Column('turnovers', sa.Integer, nullable=True),
        
        # Football specific
        sa.Column('passing_yards', sa.Integer, nullable=True),
        sa.Column('passing_tds', sa.Integer, nullable=True),
        sa.Column('interceptions', sa.Integer, nullable=True),
        sa.Column('rushing_yards', sa.Integer, nullable=True),
        sa.Column('rushing_tds', sa.Integer, nullable=True),
        sa.Column('receiving_yards', sa.Integer, nullable=True),
        sa.Column('receiving_tds', sa.Integer, nullable=True),
        sa.Column('receptions', sa.Integer, nullable=True),
        sa.Column('targets', sa.Integer, nullable=True),
        sa.Column('carries', sa.Integer, nullable=True),
        
        # Baseball specific
        sa.Column('at_bats', sa.Integer, nullable=True),
        sa.Column('hits', sa.Integer, nullable=True),
        sa.Column('runs', sa.Integer, nullable=True),
        sa.Column('rbis', sa.Integer, nullable=True),
        sa.Column('home_runs', sa.Integer, nullable=True),
        sa.Column('stolen_bases', sa.Integer, nullable=True),
        sa.Column('strikeouts_batting', sa.Integer, nullable=True),
        sa.Column('walks_batting', sa.Integer, nullable=True),
        sa.Column('batting_average', sa.Float, nullable=True),
        
        # Pitching
        sa.Column('innings_pitched', sa.Float, nullable=True),
        sa.Column('earned_runs', sa.Integer, nullable=True),
        sa.Column('strikeouts_pitching', sa.Integer, nullable=True),
        sa.Column('walks_pitching', sa.Integer, nullable=True),
        sa.Column('hits_allowed', sa.Integer, nullable=True),
        sa.Column('era', sa.Float, nullable=True),
        
        # Hockey specific
        sa.Column('goals', sa.Integer, nullable=True),
        sa.Column('hockey_assists', sa.Integer, nullable=True),
        sa.Column('plus_minus', sa.Integer, nullable=True),
        sa.Column('penalty_minutes', sa.Integer, nullable=True),
        sa.Column('shots_on_goal', sa.Integer, nullable=True),
        sa.Column('saves', sa.Integer, nullable=True),
        sa.Column('goals_against', sa.Integer, nullable=True),
        
        # Tennis specific
        sa.Column('aces', sa.Integer, nullable=True),
        sa.Column('double_faults', sa.Integer, nullable=True),
        sa.Column('first_serve_pct', sa.Float, nullable=True),
        sa.Column('break_points_saved', sa.Integer, nullable=True),
        sa.Column('break_points_converted', sa.Integer, nullable=True),
        sa.Column('sets_won', sa.Integer, nullable=True),
        sa.Column('games_won', sa.Integer, nullable=True),
        
        # Full stats JSON (for sport-specific fields)
        sa.Column('stats_json', JSONB, nullable=True),
        
        # Source tracking
        sa.Column('primary_source', sa.String(50), nullable=True),
        sa.Column('num_source_records', sa.Integer, default=1),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()')),
        
        # Unique constraint
        sa.UniqueConstraint('master_player_id', 'master_game_id', name='uq_master_player_stats_player_game'),
    )
    
    # Indexes for master_player_stats
    op.create_index('ix_master_player_stats_player', 'master_player_stats', ['master_player_id'])
    op.create_index('ix_master_player_stats_game', 'master_player_stats', ['master_game_id'])
    op.create_index('ix_master_player_stats_team', 'master_player_stats', ['master_team_id'])
    op.create_index('ix_master_player_stats_sport', 'master_player_stats', ['sport_code'])
    
    # =========================================================================
    # MASTER_TEAM_STATS - Consolidated team stats per game
    # =========================================================================
    op.create_table(
        'master_team_stats',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('master_team_id', UUID(as_uuid=True), sa.ForeignKey('master_teams.id'), nullable=False),
        sa.Column('master_game_id', UUID(as_uuid=True), sa.ForeignKey('master_games.id'), nullable=False),
        sa.Column('sport_code', sa.String(10), nullable=False),
        sa.Column('is_home', sa.Boolean, nullable=True),
        
        # Universal stats
        sa.Column('points_scored', sa.Integer, nullable=True),
        sa.Column('points_allowed', sa.Integer, nullable=True),
        
        # Football stats
        sa.Column('total_yards', sa.Integer, nullable=True),
        sa.Column('passing_yards', sa.Integer, nullable=True),
        sa.Column('rushing_yards', sa.Integer, nullable=True),
        sa.Column('turnovers', sa.Integer, nullable=True),
        sa.Column('first_downs', sa.Integer, nullable=True),
        sa.Column('third_down_conv', sa.Integer, nullable=True),
        sa.Column('third_down_att', sa.Integer, nullable=True),
        sa.Column('fourth_down_conv', sa.Integer, nullable=True),
        sa.Column('fourth_down_att', sa.Integer, nullable=True),
        sa.Column('penalties', sa.Integer, nullable=True),
        sa.Column('penalty_yards', sa.Integer, nullable=True),
        sa.Column('time_of_possession', sa.Float, nullable=True),  # in minutes
        sa.Column('sacks', sa.Integer, nullable=True),
        sa.Column('interceptions', sa.Integer, nullable=True),
        sa.Column('fumbles_lost', sa.Integer, nullable=True),
        
        # Basketball stats
        sa.Column('field_goals_made', sa.Integer, nullable=True),
        sa.Column('field_goals_att', sa.Integer, nullable=True),
        sa.Column('three_pointers_made', sa.Integer, nullable=True),
        sa.Column('three_pointers_att', sa.Integer, nullable=True),
        sa.Column('free_throws_made', sa.Integer, nullable=True),
        sa.Column('free_throws_att', sa.Integer, nullable=True),
        sa.Column('rebounds', sa.Integer, nullable=True),
        sa.Column('offensive_rebounds', sa.Integer, nullable=True),
        sa.Column('defensive_rebounds', sa.Integer, nullable=True),
        sa.Column('assists', sa.Integer, nullable=True),
        sa.Column('steals', sa.Integer, nullable=True),
        sa.Column('blocks', sa.Integer, nullable=True),
        
        # Baseball stats
        sa.Column('runs', sa.Integer, nullable=True),
        sa.Column('hits', sa.Integer, nullable=True),
        sa.Column('errors', sa.Integer, nullable=True),
        sa.Column('home_runs', sa.Integer, nullable=True),
        sa.Column('strikeouts', sa.Integer, nullable=True),
        sa.Column('walks', sa.Integer, nullable=True),
        sa.Column('left_on_base', sa.Integer, nullable=True),
        
        # Hockey stats
        sa.Column('goals', sa.Integer, nullable=True),
        sa.Column('shots', sa.Integer, nullable=True),
        sa.Column('power_play_goals', sa.Integer, nullable=True),
        sa.Column('power_play_opportunities', sa.Integer, nullable=True),
        sa.Column('penalty_minutes', sa.Integer, nullable=True),
        sa.Column('faceoff_wins', sa.Integer, nullable=True),
        sa.Column('faceoff_total', sa.Integer, nullable=True),
        sa.Column('blocked_shots', sa.Integer, nullable=True),
        sa.Column('takeaways', sa.Integer, nullable=True),
        sa.Column('giveaways', sa.Integer, nullable=True),
        
        # Full stats JSON
        sa.Column('stats_json', JSONB, nullable=True),
        
        # Source tracking
        sa.Column('primary_source', sa.String(50), nullable=True),
        sa.Column('num_source_records', sa.Integer, default=1),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()')),
        
        # Unique constraint
        sa.UniqueConstraint('master_team_id', 'master_game_id', name='uq_master_team_stats_team_game'),
    )
    
    # Indexes for master_team_stats
    op.create_index('ix_master_team_stats_team', 'master_team_stats', ['master_team_id'])
    op.create_index('ix_master_team_stats_game', 'master_team_stats', ['master_game_id'])
    op.create_index('ix_master_team_stats_sport', 'master_team_stats', ['sport_code'])
    
    # =========================================================================
    # PLAYER_STATS_MAPPINGS - Links source player_stats to master
    # =========================================================================
    op.create_table(
        'player_stats_mappings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('master_player_stats_id', UUID(as_uuid=True), sa.ForeignKey('master_player_stats.id'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_player_stats_db_id', UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        
        sa.UniqueConstraint('source_key', 'source_player_stats_db_id', name='uq_player_stats_map_source'),
    )
    
    op.create_index('ix_player_stats_mappings_master', 'player_stats_mappings', ['master_player_stats_id'])
    
    # =========================================================================
    # TEAM_STATS_MAPPINGS - Links source team_stats to master
    # =========================================================================
    op.create_table(
        'team_stats_mappings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('master_team_stats_id', UUID(as_uuid=True), sa.ForeignKey('master_team_stats.id'), nullable=False),
        sa.Column('source_key', sa.String(50), nullable=False),
        sa.Column('source_team_stats_db_id', UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        
        sa.UniqueConstraint('source_key', 'source_team_stats_db_id', name='uq_team_stats_map_source'),
    )
    
    op.create_index('ix_team_stats_mappings_master', 'team_stats_mappings', ['master_team_stats_id'])


def downgrade():
    op.drop_table('team_stats_mappings')
    op.drop_table('player_stats_mappings')
    op.drop_table('master_team_stats')
    op.drop_table('master_player_stats')
