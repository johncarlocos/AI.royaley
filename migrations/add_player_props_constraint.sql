-- ROYALEY Migration: Add unique constraint to player_props
-- Run: docker exec royaley_db psql -U royaley -d royaley -f /tmp/migrate_props.sql

-- Add unique constraint to prevent duplicate props per game/player/type/line
ALTER TABLE player_props
    ADD CONSTRAINT uq_player_props_game_player_type_line
    UNIQUE (game_id, player_id, prop_type, line);

-- Verify
SELECT conname FROM pg_constraint WHERE conrelid = 'player_props'::regclass;