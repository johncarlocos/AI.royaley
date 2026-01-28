-- =============================================================================
-- ROYALEY - Public Betting Tables Migration
-- Tables: public_betting, public_betting_history, sharp_money_indicators, fade_public_records
-- Run this SQL to create the public betting tables for Action Network collector
-- =============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- PUBLIC BETTING TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS public_betting (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    sport_code VARCHAR(10) NOT NULL,
    
    -- Game identification
    external_game_id VARCHAR(100),
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    game_time VARCHAR(20),
    
    -- Spread betting percentages
    spread_home_bet_pct FLOAT,
    spread_away_bet_pct FLOAT,
    spread_home_money_pct FLOAT,
    spread_away_money_pct FLOAT,
    spread_bet_count INTEGER,
    spread_line FLOAT,
    spread_opening_line FLOAT,
    
    -- Moneyline betting percentages
    ml_home_bet_pct FLOAT,
    ml_away_bet_pct FLOAT,
    ml_home_money_pct FLOAT,
    ml_away_money_pct FLOAT,
    ml_bet_count INTEGER,
    ml_home_odds INTEGER,
    ml_away_odds INTEGER,
    ml_home_opening INTEGER,
    ml_away_opening INTEGER,
    
    -- Total betting percentages
    total_over_bet_pct FLOAT,
    total_under_bet_pct FLOAT,
    total_over_money_pct FLOAT,
    total_under_money_pct FLOAT,
    total_bet_count INTEGER,
    total_line FLOAT,
    total_opening_line FLOAT,
    
    -- Sharp indicators
    is_sharp_spread BOOLEAN,
    is_sharp_ml BOOLEAN,
    is_sharp_total BOOLEAN,
    sharp_side_spread VARCHAR(20),
    sharp_side_ml VARCHAR(20),
    sharp_side_total VARCHAR(20),
    
    -- Reverse line movement indicators
    is_rlm_spread BOOLEAN,
    is_rlm_total BOOLEAN,
    
    -- Steam move detection
    is_steam_spread BOOLEAN,
    is_steam_total BOOLEAN,
    
    -- Game result
    home_score INTEGER,
    away_score INTEGER,
    game_status VARCHAR(20) DEFAULT 'scheduled',
    
    -- Metadata
    source VARCHAR(50) DEFAULT 'action_network',
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_data JSONB,
    
    -- Unique constraint
    CONSTRAINT uq_public_betting_game UNIQUE (sport_code, home_team, away_team, game_date, source)
);

-- Indexes for public_betting
CREATE INDEX IF NOT EXISTS ix_public_betting_sport_date ON public_betting(sport_code, game_date);
CREATE INDEX IF NOT EXISTS ix_public_betting_game_id ON public_betting(game_id);
CREATE INDEX IF NOT EXISTS ix_public_betting_sharp ON public_betting(is_sharp_spread, is_sharp_ml, is_sharp_total);
CREATE INDEX IF NOT EXISTS ix_public_betting_rlm ON public_betting(is_rlm_spread, is_rlm_total);

-- =============================================================================
-- PUBLIC BETTING HISTORY TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS public_betting_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    public_betting_id UUID REFERENCES public_betting(id) ON DELETE CASCADE,
    
    -- Snapshot timing
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hours_before_game FLOAT,
    
    -- Spread snapshot
    spread_home_bet_pct FLOAT,
    spread_home_money_pct FLOAT,
    spread_line FLOAT,
    spread_bet_count INTEGER,
    
    -- Moneyline snapshot
    ml_home_bet_pct FLOAT,
    ml_home_money_pct FLOAT,
    ml_home_odds INTEGER,
    ml_bet_count INTEGER,
    
    -- Total snapshot
    total_over_bet_pct FLOAT,
    total_over_money_pct FLOAT,
    total_line FLOAT,
    total_bet_count INTEGER
);

-- Index for public_betting_history
CREATE INDEX IF NOT EXISTS ix_public_betting_history_pb_snapshot ON public_betting_history(public_betting_id, snapshot_at);

-- =============================================================================
-- SHARP MONEY INDICATORS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS sharp_money_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    public_betting_id UUID REFERENCES public_betting(id) ON DELETE SET NULL,
    sport_code VARCHAR(10) NOT NULL,
    
    -- Game identification
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    
    -- Bet type
    bet_type VARCHAR(20) NOT NULL,
    
    -- Sharp indicator details
    indicator_type VARCHAR(50) NOT NULL,
    sharp_side VARCHAR(20) NOT NULL,
    confidence_score FLOAT,
    
    -- Line movement details
    line_before FLOAT,
    line_after FLOAT,
    line_movement FLOAT,
    
    -- Divergence metrics
    public_bet_pct FLOAT,
    money_pct FLOAT,
    divergence FLOAT,
    
    -- Outcome tracking
    result VARCHAR(10),
    profit_loss FLOAT,
    
    -- Metadata
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(50) DEFAULT 'action_network'
);

-- Indexes for sharp_money_indicators
CREATE INDEX IF NOT EXISTS ix_sharp_indicators_sport_date ON sharp_money_indicators(sport_code, game_date);
CREATE INDEX IF NOT EXISTS ix_sharp_indicators_type ON sharp_money_indicators(indicator_type, bet_type);

-- =============================================================================
-- FADE PUBLIC RECORDS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS fade_public_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    public_betting_id UUID REFERENCES public_betting(id) ON DELETE SET NULL,
    sport_code VARCHAR(10) NOT NULL,
    
    -- Game identification
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    game_date DATE NOT NULL,
    
    -- Bet type
    bet_type VARCHAR(20) NOT NULL,
    
    -- Public side info
    public_side VARCHAR(20) NOT NULL,
    public_bet_pct FLOAT NOT NULL,
    public_money_pct FLOAT,
    
    -- Fade configuration
    fade_threshold FLOAT NOT NULL,
    
    -- Bet details
    fade_side VARCHAR(20) NOT NULL,
    line_at_bet FLOAT,
    odds_at_bet INTEGER,
    
    -- Result tracking
    result VARCHAR(10),
    profit_loss FLOAT,
    closing_line FLOAT,
    clv FLOAT,
    
    -- Final scores
    home_score INTEGER,
    away_score INTEGER,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at TIMESTAMP
);

-- Indexes for fade_public_records
CREATE INDEX IF NOT EXISTS ix_fade_public_sport_date ON fade_public_records(sport_code, game_date);
CREATE INDEX IF NOT EXISTS ix_fade_public_result ON fade_public_records(result, sport_code);
CREATE INDEX IF NOT EXISTS ix_fade_public_threshold ON fade_public_records(fade_threshold, public_bet_pct);

-- =============================================================================
-- TRIGGER FOR UPDATED_AT
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_public_betting_updated_at ON public_betting;
CREATE TRIGGER update_public_betting_updated_at
    BEFORE UPDATE ON public_betting
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Public betting tables created successfully!';
    RAISE NOTICE 'Tables: public_betting, public_betting_history, sharp_money_indicators, fade_public_records';
END $$;