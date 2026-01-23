"""Initial schema - Create all 43 tables

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-21

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ENUM types first (using DO block to handle IF NOT EXISTS)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE userrole AS ENUM ('user', 'pro_user', 'admin', 'super_admin');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE gamestatus AS ENUM ('scheduled', 'in_progress', 'final', 'postponed', 'cancelled');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE betresult AS ENUM ('pending', 'win', 'loss', 'push', 'void');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE signaltier AS ENUM ('A', 'B', 'C', 'D');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE alertseverity AS ENUM ('info', 'warning', 'critical');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE healthstatus AS ENUM ('healthy', 'degraded', 'down');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE mlframework AS ENUM ('h2o', 'autogluon', 'sklearn', 'meta');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE taskstatus AS ENUM ('idle', 'running', 'failed', 'success');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # 1. Users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('role', postgresql.ENUM('user', 'pro_user', 'admin', 'super_admin', name='userrole', create_type=False), server_default='user'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('is_verified', sa.Boolean(), server_default='false'),
        sa.Column('two_factor_enabled', sa.Boolean(), server_default='false'),
        sa.Column('two_factor_secret', sa.String(255), nullable=True),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
    )

    # 2. Sessions table
    op.create_table('sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('is_revoked', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_sessions_token_hash', 'sessions', ['token_hash'])

    # 3. API Keys table
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('permissions', postgresql.JSONB(), server_default='{}'),
        sa.Column('rate_limit', sa.Integer(), server_default='100'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 4. User Preferences table
    op.create_table('user_preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True),
        sa.Column('timezone', sa.String(50), server_default='UTC'),
        sa.Column('notification_settings', postgresql.JSONB(), server_default='{}'),
        sa.Column('display_preferences', postgresql.JSONB(), server_default='{}'),
        sa.Column('default_sport', sa.String(10), nullable=True),
        sa.Column('odds_format', sa.String(20), server_default='american'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 5. Audit Logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('details', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'])

    # 6. Sports table
    op.create_table('sports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('code', sa.String(10), nullable=False, unique=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('api_key', sa.String(50), nullable=True),
        sa.Column('feature_count', sa.Integer(), server_default='60'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('config', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 7. Venues table
    op.create_table('venues',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('city', sa.String(100), nullable=True),
        sa.Column('state', sa.String(100), nullable=True),
        sa.Column('country', sa.String(100), server_default='USA'),
        sa.Column('timezone', sa.String(50), nullable=True),
        sa.Column('is_dome', sa.Boolean(), server_default='false'),
        sa.Column('surface', sa.String(50), nullable=True),
        sa.Column('capacity', sa.Integer(), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 8. Teams table
    op.create_table('teams',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('external_id', sa.String(50), nullable=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('abbreviation', sa.String(10), nullable=True),
        sa.Column('city', sa.String(100), nullable=True),
        sa.Column('conference', sa.String(50), nullable=True),
        sa.Column('division', sa.String(50), nullable=True),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('venues.id', ondelete='SET NULL'), nullable=True),
        sa.Column('elo_rating', sa.Float(), server_default='1500.0'),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_teams_sport_id', 'teams', ['sport_id'])
    op.create_unique_constraint('uq_teams_sport_name', 'teams', ['sport_id', 'name'])

    # 9. Players table
    op.create_table('players',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id', ondelete='SET NULL'), nullable=True),
        sa.Column('external_id', sa.String(50), nullable=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('position', sa.String(50), nullable=True),
        sa.Column('jersey_number', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), server_default='active'),
        sa.Column('injury_status', sa.String(100), nullable=True),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.Column('height_inches', sa.Integer(), nullable=True),
        sa.Column('weight_lbs', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_players_team_id', 'players', ['team_id'])

    # 10. Seasons table
    op.create_table('seasons',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('is_current', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_unique_constraint('uq_seasons_sport_year', 'seasons', ['sport_id', 'year'])

    # 11. Games table
    op.create_table('games',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('season_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('seasons.id', ondelete='SET NULL'), nullable=True),
        sa.Column('external_id', sa.String(100), nullable=True, unique=True),
        sa.Column('home_team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('away_team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('venues.id', ondelete='SET NULL'), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(), nullable=False),
        sa.Column('status', postgresql.ENUM('scheduled', 'in_progress', 'final', 'postponed', 'cancelled', name='gamestatus', create_type=False), server_default='scheduled'),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('home_rotation', sa.Integer(), nullable=True),
        sa.Column('away_rotation', sa.Integer(), nullable=True),
        sa.Column('period', sa.String(20), nullable=True),
        sa.Column('clock', sa.String(20), nullable=True),
        sa.Column('weather', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_games_sport_id', 'games', ['sport_id'])
    op.create_index('ix_games_scheduled_at', 'games', ['scheduled_at'])
    op.create_index('ix_games_status', 'games', ['status'])

    # 12. Game Features table
    op.create_table('game_features',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('features', postgresql.JSONB(), server_default='{}'),
        sa.Column('home_features', postgresql.JSONB(), server_default='{}'),
        sa.Column('away_features', postgresql.JSONB(), server_default='{}'),
        sa.Column('computed_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_game_features_game_id', 'game_features', ['game_id'])

    # 13. Team Stats table
    op.create_table('team_stats',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('season_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('seasons.id', ondelete='SET NULL'), nullable=True),
        sa.Column('games_played', sa.Integer(), server_default='0'),
        sa.Column('wins', sa.Integer(), server_default='0'),
        sa.Column('losses', sa.Integer(), server_default='0'),
        sa.Column('ties', sa.Integer(), server_default='0'),
        sa.Column('points_for', sa.Float(), server_default='0'),
        sa.Column('points_against', sa.Float(), server_default='0'),
        sa.Column('home_record', sa.String(20), nullable=True),
        sa.Column('away_record', sa.String(20), nullable=True),
        sa.Column('streak', sa.String(20), nullable=True),
        sa.Column('ats_record', sa.String(20), nullable=True),
        sa.Column('over_under_record', sa.String(20), nullable=True),
        sa.Column('advanced_stats', postgresql.JSONB(), server_default='{}'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_team_stats_team_id', 'team_stats', ['team_id'])

    # 14. Player Stats table
    op.create_table('player_stats',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('player_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('players.id', ondelete='CASCADE'), nullable=False),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=True),
        sa.Column('season_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('seasons.id', ondelete='SET NULL'), nullable=True),
        sa.Column('stat_type', sa.String(50), server_default='game'),
        sa.Column('stats', postgresql.JSONB(), server_default='{}'),
        sa.Column('minutes_played', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_player_stats_player_id', 'player_stats', ['player_id'])

    # 15. Sportsbooks table
    op.create_table('sportsbooks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('key', sa.String(50), nullable=False, unique=True),
        sa.Column('is_sharp', sa.Boolean(), server_default='false'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('priority', sa.Integer(), server_default='100'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 16. Odds table
    op.create_table('odds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sportsbook_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sportsbooks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('sportsbook_key', sa.String(50), nullable=True),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('home_line', sa.Float(), nullable=True),
        sa.Column('away_line', sa.Float(), nullable=True),
        sa.Column('home_odds', sa.Integer(), nullable=True),
        sa.Column('away_odds', sa.Integer(), nullable=True),
        sa.Column('total', sa.Float(), nullable=True),
        sa.Column('over_odds', sa.Integer(), nullable=True),
        sa.Column('under_odds', sa.Integer(), nullable=True),
        sa.Column('is_opening', sa.Boolean(), server_default='false'),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_odds_game_id', 'odds', ['game_id'])
    op.create_index('ix_odds_recorded_at', 'odds', ['recorded_at'])

    # 17. Odds Movement table
    op.create_table('odds_movements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('previous_line', sa.Float(), nullable=True),
        sa.Column('current_line', sa.Float(), nullable=True),
        sa.Column('movement', sa.Float(), nullable=True),
        sa.Column('is_steam', sa.Boolean(), server_default='false'),
        sa.Column('is_reverse', sa.Boolean(), server_default='false'),
        sa.Column('detected_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_odds_movements_game_id', 'odds_movements', ['game_id'])

    # 18. Closing Lines table
    op.create_table('closing_lines',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('spread_home', sa.Float(), nullable=True),
        sa.Column('spread_away', sa.Float(), nullable=True),
        sa.Column('total', sa.Float(), nullable=True),
        sa.Column('moneyline_home', sa.Integer(), nullable=True),
        sa.Column('moneyline_away', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(50), server_default='pinnacle'),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 19. Consensus Lines table
    op.create_table('consensus_lines',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('consensus_line', sa.Float(), nullable=True),
        sa.Column('public_bet_pct', sa.Float(), nullable=True),
        sa.Column('public_money_pct', sa.Float(), nullable=True),
        sa.Column('sharp_action', sa.String(20), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_consensus_lines_game_id', 'consensus_lines', ['game_id'])

    # 20. ML Models table
    op.create_table('ml_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('framework', postgresql.ENUM('h2o', 'autogluon', 'sklearn', 'meta', name='mlframework', create_type=False), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('is_production', sa.Boolean(), server_default='false'),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('auc', sa.Float(), nullable=True),
        sa.Column('log_loss', sa.Float(), nullable=True),
        sa.Column('hyperparameters', postgresql.JSONB(), server_default='{}'),
        sa.Column('feature_names', postgresql.JSONB(), server_default='[]'),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('promoted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_ml_models_sport_id', 'ml_models', ['sport_id'])
    op.create_unique_constraint('uq_ml_models_sport_bet_framework_version', 'ml_models', ['sport_id', 'bet_type', 'framework', 'version'])

    # 21. Predictions table
    op.create_table('predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='SET NULL'), nullable=True),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('predicted_side', sa.String(20), nullable=False),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('line_at_prediction', sa.Float(), nullable=True),
        sa.Column('odds_at_prediction', sa.Integer(), nullable=True),
        sa.Column('edge', sa.Float(), nullable=True),
        sa.Column('signal_tier', postgresql.ENUM('A', 'B', 'C', 'D', name='signaltier', create_type=False), server_default='D'),
        sa.Column('kelly_fraction', sa.Float(), nullable=True),
        sa.Column('recommended_bet', sa.Float(), nullable=True),
        sa.Column('prediction_hash', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('locked_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_predictions_game_id', 'predictions', ['game_id'])
    op.create_index('ix_predictions_signal_tier', 'predictions', ['signal_tier'])
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'])

    # 22. Prediction Results table
    op.create_table('prediction_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('predictions.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('result', postgresql.ENUM('pending', 'win', 'loss', 'push', 'void', name='betresult', create_type=False), server_default='pending'),
        sa.Column('closing_line', sa.Float(), nullable=True),
        sa.Column('clv', sa.Float(), nullable=True),
        sa.Column('actual_score_home', sa.Integer(), nullable=True),
        sa.Column('actual_score_away', sa.Integer(), nullable=True),
        sa.Column('graded_at', sa.DateTime(), nullable=True),
    )

    # 23. Player Props table
    op.create_table('player_props',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('player_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('players.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prop_type', sa.String(50), nullable=False),
        sa.Column('line', sa.Float(), nullable=False),
        sa.Column('over_odds', sa.Integer(), nullable=True),
        sa.Column('under_odds', sa.Integer(), nullable=True),
        sa.Column('predicted_value', sa.Float(), nullable=True),
        sa.Column('predicted_side', sa.String(10), nullable=True),
        sa.Column('probability', sa.Float(), nullable=True),
        sa.Column('signal_tier', postgresql.ENUM('A', 'B', 'C', 'D', name='signaltier', create_type=False), nullable=True),
        sa.Column('result', postgresql.ENUM('pending', 'win', 'loss', 'push', 'void', name='betresult', create_type=False), server_default='pending'),
        sa.Column('actual_value', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_player_props_game_id', 'player_props', ['game_id'])
    op.create_index('ix_player_props_player_id', 'player_props', ['player_id'])

    # 24. SHAP Explanations table
    op.create_table('shap_explanations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('predictions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('feature_value', sa.Float(), nullable=True),
        sa.Column('shap_value', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=True),
    )
    op.create_index('ix_shap_explanations_prediction_id', 'shap_explanations', ['prediction_id'])

    # 25. Training Runs table
    op.create_table('training_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=True),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('framework', postgresql.ENUM('h2o', 'autogluon', 'sklearn', 'meta', name='mlframework', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('idle', 'running', 'failed', 'success', name='taskstatus', create_type=False), server_default='running'),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('validation_samples', sa.Integer(), nullable=True),
        sa.Column('metrics', postgresql.JSONB(), server_default='{}'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_training_runs_sport_id', 'training_runs', ['sport_id'])

    # 26. Model Performance table
    op.create_table('model_performances',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('period_start', sa.Date(), nullable=False),
        sa.Column('period_end', sa.Date(), nullable=False),
        sa.Column('predictions_count', sa.Integer(), server_default='0'),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('auc', sa.Float(), nullable=True),
        sa.Column('avg_clv', sa.Float(), nullable=True),
        sa.Column('roi', sa.Float(), nullable=True),
        sa.Column('tier_breakdown', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_model_performances_model_id', 'model_performances', ['model_id'])

    # 27. Feature Importance table
    op.create_table('feature_importances',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('importance', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=True),
    )
    op.create_index('ix_feature_importances_model_id', 'feature_importances', ['model_id'])

    # 28. Calibration Models table
    op.create_table('calibration_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('method', sa.String(50), nullable=False),
        sa.Column('calibration_path', sa.String(500), nullable=True),
        sa.Column('ece_before', sa.Float(), nullable=True),
        sa.Column('ece_after', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 29. Bankrolls table
    op.create_table('bankrolls',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('initial_balance', sa.Numeric(12, 2), server_default='10000.00'),
        sa.Column('current_balance', sa.Numeric(12, 2), server_default='10000.00'),
        sa.Column('currency', sa.String(3), server_default='USD'),
        sa.Column('kelly_fraction', sa.Float(), server_default='0.25'),
        sa.Column('max_bet_percent', sa.Float(), server_default='0.02'),
        sa.Column('min_edge_threshold', sa.Float(), server_default='0.03'),
        sa.Column('total_wagered', sa.Numeric(12, 2), server_default='0.00'),
        sa.Column('total_won', sa.Numeric(12, 2), server_default='0.00'),
        sa.Column('total_lost', sa.Numeric(12, 2), server_default='0.00'),
        sa.Column('high_watermark', sa.Numeric(12, 2), nullable=True),
        sa.Column('low_watermark', sa.Numeric(12, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 30. Bets table
    op.create_table('bets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('predictions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='SET NULL'), nullable=True),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('side', sa.String(20), nullable=False),
        sa.Column('line', sa.Float(), nullable=True),
        sa.Column('odds', sa.Integer(), nullable=False),
        sa.Column('stake', sa.Numeric(10, 2), nullable=False),
        sa.Column('potential_payout', sa.Numeric(10, 2), nullable=True),
        sa.Column('result', postgresql.ENUM('pending', 'win', 'loss', 'push', 'void', name='betresult', create_type=False), server_default='pending'),
        sa.Column('profit_loss', sa.Numeric(10, 2), nullable=True),
        sa.Column('closing_line', sa.Float(), nullable=True),
        sa.Column('clv', sa.Float(), nullable=True),
        sa.Column('sportsbook', sa.String(100), nullable=True),
        sa.Column('placed_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('settled_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_bets_user_id', 'bets', ['user_id'])
    op.create_index('ix_bets_placed_at', 'bets', ['placed_at'])

    # 31. Bankroll Transactions table
    op.create_table('bankroll_transactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('bankroll_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('bankrolls.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('bets.id', ondelete='SET NULL'), nullable=True),
        sa.Column('transaction_type', sa.String(50), nullable=False),
        sa.Column('amount', sa.Numeric(10, 2), nullable=False),
        sa.Column('balance_after', sa.Numeric(12, 2), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_bankroll_transactions_bankroll_id', 'bankroll_transactions', ['bankroll_id'])

    # 32. System Settings table
    op.create_table('system_settings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('key', sa.String(100), nullable=False, unique=True),
        sa.Column('value', postgresql.JSONB(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 33. Scheduled Tasks table
    op.create_table('scheduled_tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('cron_expression', sa.String(100), nullable=True),
        sa.Column('interval_seconds', sa.Integer(), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), server_default='true'),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.Column('next_run_at', sa.DateTime(), nullable=True),
        sa.Column('status', postgresql.ENUM('idle', 'running', 'failed', 'success', name='taskstatus', create_type=False), server_default='idle'),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('run_count', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 34. Alerts table
    op.create_table('alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('severity', postgresql.ENUM('info', 'warning', 'critical', name='alertseverity', create_type=False), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(), nullable=True),
        sa.Column('is_read', sa.Boolean(), server_default='false'),
        sa.Column('is_resolved', sa.Boolean(), server_default='false'),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_alerts_severity', 'alerts', ['severity'])
    op.create_index('ix_alerts_created_at', 'alerts', ['created_at'])

    # 35. Data Quality Checks table
    op.create_table('data_quality_checks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('check_type', sa.String(100), nullable=False),
        sa.Column('table_name', sa.String(100), nullable=True),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', postgresql.ENUM('idle', 'running', 'failed', 'success', name='taskstatus', create_type=False), nullable=False),
        sa.Column('records_checked', sa.Integer(), nullable=True),
        sa.Column('issues_found', sa.Integer(), server_default='0'),
        sa.Column('details', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )

    # 36. System Health Snapshots table
    op.create_table('system_health_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('status', postgresql.ENUM('healthy', 'degraded', 'down', name='healthstatus', create_type=False), nullable=False),
        sa.Column('components', postgresql.JSONB(), server_default='{}'),
        sa.Column('cpu_usage', sa.Float(), nullable=True),
        sa.Column('memory_usage', sa.Float(), nullable=True),
        sa.Column('disk_usage', sa.Float(), nullable=True),
        sa.Column('api_latency_ms', sa.Float(), nullable=True),
        sa.Column('db_latency_ms', sa.Float(), nullable=True),
        sa.Column('active_connections', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_system_health_snapshots_created_at', 'system_health_snapshots', ['created_at'])

    # 37. Backtest Runs table
    op.create_table('backtest_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('sport_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sports.id', ondelete='SET NULL'), nullable=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='SET NULL'), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=False),
        sa.Column('initial_bankroll', sa.Numeric(12, 2), server_default='10000.00'),
        sa.Column('final_bankroll', sa.Numeric(12, 2), nullable=True),
        sa.Column('total_bets', sa.Integer(), server_default='0'),
        sa.Column('wins', sa.Integer(), server_default='0'),
        sa.Column('losses', sa.Integer(), server_default='0'),
        sa.Column('pushes', sa.Integer(), server_default='0'),
        sa.Column('roi', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('avg_clv', sa.Float(), nullable=True),
        sa.Column('tier_results', postgresql.JSONB(), server_default='{}'),
        sa.Column('equity_curve', postgresql.JSONB(), server_default='[]'),
        sa.Column('status', postgresql.ENUM('idle', 'running', 'failed', 'success', name='taskstatus', create_type=False), server_default='running'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )

    # 38. ELO History table
    op.create_table('elo_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('teams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=True),
        sa.Column('elo_before', sa.Float(), nullable=False),
        sa.Column('elo_after', sa.Float(), nullable=False),
        sa.Column('elo_change', sa.Float(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_elo_history_team_id', 'elo_history', ['team_id'])

    # 39. CLV Records table
    op.create_table('clv_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('bet_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('bets.id', ondelete='CASCADE'), nullable=True),
        sa.Column('prediction_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('predictions.id', ondelete='CASCADE'), nullable=True),
        sa.Column('opening_line', sa.Float(), nullable=True),
        sa.Column('bet_line', sa.Float(), nullable=True),
        sa.Column('closing_line', sa.Float(), nullable=True),
        sa.Column('clv', sa.Float(), nullable=False),
        sa.Column('clv_percentage', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_clv_records_created_at', 'clv_records', ['created_at'])

    # 40. Line Movement Alerts table
    op.create_table('line_movement_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bet_type', sa.String(50), nullable=False),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('previous_line', sa.Float(), nullable=True),
        sa.Column('current_line', sa.Float(), nullable=True),
        sa.Column('movement_size', sa.Float(), nullable=True),
        sa.Column('is_acknowledged', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_line_movement_alerts_game_id', 'line_movement_alerts', ['game_id'])

    # 41. Notifications table
    op.create_table('notifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('notification_type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('data', postgresql.JSONB(), nullable=True),
        sa.Column('is_read', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'])

    # 42. Rate Limit Logs table
    op.create_table('rate_limit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('api_keys.id', ondelete='CASCADE'), nullable=True),
        sa.Column('endpoint', sa.String(200), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('requests_count', sa.Integer(), server_default='1'),
        sa.Column('window_start', sa.DateTime(), nullable=False),
        sa.Column('is_limited', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_rate_limit_logs_window_start', 'rate_limit_logs', ['window_start'])

    # 43. Weather Data table
    op.create_table('weather_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=True),
        sa.Column('venue_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('venues.id', ondelete='CASCADE'), nullable=True),
        sa.Column('temperature_f', sa.Float(), nullable=True),
        sa.Column('feels_like_f', sa.Float(), nullable=True),
        sa.Column('humidity_pct', sa.Float(), nullable=True),
        sa.Column('wind_speed_mph', sa.Float(), nullable=True),
        sa.Column('wind_direction', sa.String(20), nullable=True),
        sa.Column('precipitation_pct', sa.Float(), nullable=True),
        sa.Column('conditions', sa.String(100), nullable=True),
        sa.Column('is_dome', sa.Boolean(), server_default='false'),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_weather_data_game_id', 'weather_data', ['game_id'])


def downgrade() -> None:
    op.drop_table('weather_data')
    op.drop_table('rate_limit_logs')
    op.drop_table('notifications')
    op.drop_table('line_movement_alerts')
    op.drop_table('clv_records')
    op.drop_table('elo_history')
    op.drop_table('backtest_runs')
    op.drop_table('system_health_snapshots')
    op.drop_table('data_quality_checks')
    op.drop_table('alerts')
    op.drop_table('scheduled_tasks')
    op.drop_table('system_settings')
    op.drop_table('bankroll_transactions')
    op.drop_table('bets')
    op.drop_table('bankrolls')
    op.drop_table('calibration_models')
    op.drop_table('feature_importances')
    op.drop_table('model_performances')
    op.drop_table('training_runs')
    op.drop_table('shap_explanations')
    op.drop_table('player_props')
    op.drop_table('prediction_results')
    op.drop_table('predictions')
    op.drop_table('ml_models')
    op.drop_table('consensus_lines')
    op.drop_table('closing_lines')
    op.drop_table('odds_movements')
    op.drop_table('odds')
    op.drop_table('sportsbooks')
    op.drop_table('player_stats')
    op.drop_table('team_stats')
    op.drop_table('game_features')
    op.drop_table('games')
    op.drop_table('seasons')
    op.drop_table('players')
    op.drop_table('teams')
    op.drop_table('venues')
    op.drop_table('sports')
    op.drop_table('audit_logs')
    op.drop_table('user_preferences')
    op.drop_table('api_keys')
    op.drop_table('sessions')
    op.drop_table('users')
    
    op.execute("DROP TYPE IF EXISTS taskstatus")
    op.execute("DROP TYPE IF EXISTS mlframework")
    op.execute("DROP TYPE IF EXISTS healthstatus")
    op.execute("DROP TYPE IF EXISTS alertseverity")
    op.execute("DROP TYPE IF EXISTS signaltier")
    op.execute("DROP TYPE IF EXISTS betresult")
    op.execute("DROP TYPE IF EXISTS gamestatus")
    op.execute("DROP TYPE IF EXISTS userrole")