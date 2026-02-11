-- ============================================================================
-- ROYALEY - Migration: Create Live Pipeline Tables
-- Tables 58-59: upcoming_games, upcoming_odds
-- Also adds upcoming_game_id to predictions table
-- 
-- Run: docker exec -i royaley_postgres psql -U royaley -d royaley < migration.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- Table 58: upcoming_games
-- ============================================================================
CREATE TABLE IF NOT EXISTS upcoming_games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sport_id UUID NOT NULL REFERENCES sports(id) ON DELETE CASCADE,
    
    -- Odds API external ID (unique per game)
    external_id VARCHAR(200) UNIQUE NOT NULL,
    
    -- Teams (reference shared teams table)
    home_team_id UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    away_team_id UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    
    -- Team names stored directly for fast queries
    home_team_name VARCHAR(200) NOT NULL,
    away_team_name VARCHAR(200) NOT NULL,
    
    -- Schedule
    scheduled_at TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    
    -- Scores (filled after game completes)
    home_score INTEGER,
    away_score INTEGER,
    
    -- Source tracking
    source VARCHAR(50) NOT NULL DEFAULT 'odds_api',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for upcoming_games
CREATE INDEX IF NOT EXISTS ix_upcoming_games_sport ON upcoming_games(sport_id);
CREATE INDEX IF NOT EXISTS ix_upcoming_games_scheduled ON upcoming_games(scheduled_at);
CREATE INDEX IF NOT EXISTS ix_upcoming_games_status ON upcoming_games(status);
CREATE INDEX IF NOT EXISTS ix_upcoming_games_sport_scheduled ON upcoming_games(sport_id, scheduled_at);

-- ============================================================================
-- Table 59: upcoming_odds
-- ============================================================================
CREATE TABLE IF NOT EXISTS upcoming_odds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    upcoming_game_id UUID NOT NULL REFERENCES upcoming_games(id) ON DELETE CASCADE,
    
    -- Sportsbook info
    sportsbook_key VARCHAR(50) NOT NULL,
    sportsbook_name VARCHAR(100) NOT NULL,
    is_sharp BOOLEAN DEFAULT FALSE,
    
    -- Market type
    bet_type VARCHAR(50) NOT NULL,
    
    -- Spread lines
    home_line FLOAT,
    away_line FLOAT,
    home_odds INTEGER,
    away_odds INTEGER,
    
    -- Total lines
    total FLOAT,
    over_odds INTEGER,
    under_odds INTEGER,
    
    -- Moneyline odds
    home_ml INTEGER,
    away_ml INTEGER,
    
    -- Source tracking
    source VARCHAR(50) NOT NULL DEFAULT 'odds_api',
    
    recorded_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- One record per game + sportsbook + bet_type
    CONSTRAINT uq_upcoming_odds_game_book_bet 
        UNIQUE (upcoming_game_id, sportsbook_key, bet_type)
);

-- Indexes for upcoming_odds
CREATE INDEX IF NOT EXISTS ix_upcoming_odds_game ON upcoming_odds(upcoming_game_id);
CREATE INDEX IF NOT EXISTS ix_upcoming_odds_game_book_bet ON upcoming_odds(upcoming_game_id, sportsbook_key, bet_type);

-- ============================================================================
-- Add upcoming_game_id to predictions table
-- (nullable - old predictions from backtests use game_id, live pipeline uses upcoming_game_id)
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'predictions' AND column_name = 'upcoming_game_id'
    ) THEN
        ALTER TABLE predictions 
        ADD COLUMN upcoming_game_id UUID REFERENCES upcoming_games(id) ON DELETE SET NULL;
        
        CREATE INDEX ix_predictions_upcoming_game ON predictions(upcoming_game_id);
    END IF;
END $$;

-- ============================================================================
-- Make predictions.game_id nullable (live pipeline only uses upcoming_game_id)
-- ============================================================================
DO $$
BEGIN
    ALTER TABLE predictions ALTER COLUMN game_id DROP NOT NULL;
EXCEPTION
    WHEN others THEN NULL;
END $$;

COMMIT;

-- ============================================================================
-- Verify
-- ============================================================================
SELECT 'upcoming_games' as table_name, COUNT(*) as row_count FROM upcoming_games
UNION ALL
SELECT 'upcoming_odds', COUNT(*) FROM upcoming_odds
UNION ALL
SELECT 'predictions', COUNT(*) FROM predictions;
