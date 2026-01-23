-- ROYALEY - Database Initialization Script
-- Version: 2.0.0
-- This script runs when the PostgreSQL container is first created

-- ============================================
-- Create Extensions
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ============================================
-- Create Application User (if not exists)
-- ============================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'royaley_app') THEN
        CREATE USER royaley_app WITH PASSWORD 'app_password_change_me';
    END IF;
END
$$;

-- ============================================
-- Create Read-Only User for Reporting
-- ============================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'royaley_readonly') THEN
        CREATE USER royaley_readonly WITH PASSWORD 'readonly_password_change_me';
    END IF;
END
$$;

-- ============================================
-- Grant Permissions
-- ============================================

-- Application user permissions
GRANT CONNECT ON DATABASE royaley TO royaley_app;
GRANT USAGE ON SCHEMA public TO royaley_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO royaley_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO royaley_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO royaley_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO royaley_app;

-- Read-only user permissions
GRANT CONNECT ON DATABASE royaley TO royaley_readonly;
GRANT USAGE ON SCHEMA public TO royaley_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO royaley_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO royaley_readonly;

-- ============================================
-- Create Enum Types
-- ============================================

DO $$
BEGIN
    -- Sport enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sport_enum') THEN
        CREATE TYPE sport_enum AS ENUM (
            'NFL', 'NCAAF', 'CFL', 'NBA', 'NCAAB', 'WNBA', 'NHL', 'MLB', 'ATP', 'WTA'
        );
    END IF;
    
    -- Bet type enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'bet_type_enum') THEN
        CREATE TYPE bet_type_enum AS ENUM (
            'spread', 'moneyline', 'total', 'player_prop'
        );
    END IF;
    
    -- Signal tier enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'signal_tier_enum') THEN
        CREATE TYPE signal_tier_enum AS ENUM ('A', 'B', 'C', 'D');
    END IF;
    
    -- Prediction result enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'prediction_result_enum') THEN
        CREATE TYPE prediction_result_enum AS ENUM ('win', 'loss', 'push', 'pending', 'cancelled');
    END IF;
    
    -- User role enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role_enum') THEN
        CREATE TYPE user_role_enum AS ENUM ('user', 'premium', 'admin', 'superadmin');
    END IF;
    
    -- Alert severity enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_severity_enum') THEN
        CREATE TYPE alert_severity_enum AS ENUM ('info', 'warning', 'error', 'critical');
    END IF;
    
    -- Game status enum
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'game_status_enum') THEN
        CREATE TYPE game_status_enum AS ENUM (
            'scheduled', 'in_progress', 'final', 'postponed', 'cancelled'
        );
    END IF;
END
$$;

-- ============================================
-- Create Helper Functions
-- ============================================

-- Function to calculate ELO expected score
CREATE OR REPLACE FUNCTION calculate_elo_expected(rating_a FLOAT, rating_b FLOAT)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1.0 / (1.0 + POWER(10, (rating_b - rating_a) / 400.0));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update ELO rating
CREATE OR REPLACE FUNCTION update_elo_rating(
    current_rating FLOAT,
    expected_score FLOAT,
    actual_score FLOAT,
    k_factor FLOAT DEFAULT 32.0
)
RETURNS FLOAT AS $$
BEGIN
    RETURN current_rating + k_factor * (actual_score - expected_score);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to convert American odds to decimal
CREATE OR REPLACE FUNCTION american_to_decimal(american_odds INTEGER)
RETURNS FLOAT AS $$
BEGIN
    IF american_odds > 0 THEN
        RETURN (american_odds / 100.0) + 1;
    ELSE
        RETURN (100.0 / ABS(american_odds)) + 1;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to convert decimal odds to implied probability
CREATE OR REPLACE FUNCTION decimal_to_probability(decimal_odds FLOAT)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1.0 / decimal_odds;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate Kelly criterion
CREATE OR REPLACE FUNCTION calculate_kelly(
    probability FLOAT,
    decimal_odds FLOAT,
    kelly_fraction FLOAT DEFAULT 0.25
)
RETURNS FLOAT AS $$
DECLARE
    b FLOAT;
    q FLOAT;
    full_kelly FLOAT;
BEGIN
    b := decimal_odds - 1;
    q := 1 - probability;
    full_kelly := (b * probability - q) / b;
    
    IF full_kelly < 0 THEN
        RETURN 0;
    END IF;
    
    RETURN LEAST(full_kelly * kelly_fraction, 0.02);  -- Cap at 2%
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate CLV (Closing Line Value)
CREATE OR REPLACE FUNCTION calculate_clv(
    bet_line FLOAT,
    closing_line FLOAT,
    bet_side VARCHAR
)
RETURNS FLOAT AS $$
BEGIN
    IF bet_side IN ('home', 'over') THEN
        RETURN closing_line - bet_line;
    ELSE
        RETURN bet_line - closing_line;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update timestamp on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Create Indexes Helper Function
-- ============================================

CREATE OR REPLACE FUNCTION create_index_if_not_exists(
    index_name TEXT,
    table_name TEXT,
    column_names TEXT
)
RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = index_name) THEN
        EXECUTE format('CREATE INDEX %I ON %I (%s)', index_name, table_name, column_names);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Performance Configuration
-- ============================================

-- Set statement timeout for long-running queries
ALTER DATABASE royaley SET statement_timeout = '300s';

-- Set work_mem for complex queries
ALTER DATABASE royaley SET work_mem = '256MB';

-- Set maintenance_work_mem for maintenance operations
ALTER DATABASE royaley SET maintenance_work_mem = '512MB';

-- Enable JIT compilation
ALTER DATABASE royaley SET jit = on;

-- ============================================
-- Create Partitioning Setup (for large tables)
-- ============================================

-- This is a template for partitioning predictions by date
-- Uncomment and modify as needed:

-- CREATE TABLE predictions_partitioned (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     game_id UUID NOT NULL,
--     created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
--     -- ... other columns
-- ) PARTITION BY RANGE (created_at);

-- CREATE TABLE predictions_2024_q1 PARTITION OF predictions_partitioned
--     FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
-- CREATE TABLE predictions_2024_q2 PARTITION OF predictions_partitioned
--     FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
-- ... etc

-- ============================================
-- Initial Data: Sports Configuration
-- ============================================

-- This will be run after tables are created via Alembic migrations
-- INSERT INTO sports (code, name, api_code, active, features_count) VALUES
-- ('NFL', 'NFL Football', 'americanfootball_nfl', true, 75),
-- ('NCAAF', 'NCAA Football', 'americanfootball_ncaaf', true, 70),
-- ('CFL', 'CFL Football', 'americanfootball_cfl', true, 65),
-- ('NBA', 'NBA Basketball', 'basketball_nba', true, 80),
-- ('NCAAB', 'NCAA Basketball', 'basketball_ncaab', true, 70),
-- ('WNBA', 'WNBA Basketball', 'basketball_wnba', true, 70),
-- ('NHL', 'NHL Hockey', 'icehockey_nhl', true, 75),
-- ('MLB', 'MLB Baseball', 'baseball_mlb', true, 85),
-- ('ATP', 'ATP Tennis', 'tennis_atp', true, 60),
-- ('WTA', 'WTA Tennis', 'tennis_wta', true, 60);

-- ============================================
-- Logging
-- ============================================

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'ROYALEY database initialized successfully at %', CURRENT_TIMESTAMP;
END
$$;
