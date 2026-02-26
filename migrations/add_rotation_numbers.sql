-- Add rotation number columns to upcoming_games
-- Rotation numbers come from Action Network (e.g., 501, 502)
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS home_rotation INTEGER;
ALTER TABLE upcoming_games ADD COLUMN IF NOT EXISTS away_rotation INTEGER;

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS ix_upcoming_games_rotation ON upcoming_games (home_rotation, away_rotation);
