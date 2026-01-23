-- ROYALEY - Database Initialization
-- Phase 1: PostgreSQL Setup Script
-- Run automatically when Docker container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Set timezone
SET timezone = 'UTC';

-- Grant privileges to the royaley user
DO $$
BEGIN
    -- Grant privileges if the database exists
    IF EXISTS (SELECT FROM pg_database WHERE datname = 'royaley') THEN
        RAISE NOTICE 'Database royaley exists';
    END IF;
END $$;

-- Performance settings notice
DO $$
BEGIN
    RAISE NOTICE '================================================';
    RAISE NOTICE 'ROYALEY database initialized successfully';
    RAISE NOTICE 'PostgreSQL version: %', version();
    RAISE NOTICE '================================================';
END $$;
