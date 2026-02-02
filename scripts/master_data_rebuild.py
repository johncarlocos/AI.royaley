#!/usr/bin/env python3
"""
ROYALEY - Master Data Rebuild (Wrapper Script)
Single command to rebuild or sync all master data.

This script wraps the refactored services in app/services/master_data/

Usage:
    python -m scripts.master_data_rebuild [--full|--sync|--verify]
    
    # Or directly:
    ./scripts/master_data_rebuild.py --full

Options:
    --full    Full rebuild: populate + map + consolidate (default)
    --sync    Incremental sync: map new data only
    --verify  Verify current state, no changes

The orchestrator performs these steps:
    1. Populate source_registry + master_teams (pro sports)
    2. Map teams → master_teams
    3. Map players → master_players (all sports)
    4. Create master_games + game_mappings
    5. Backfill master_*_id on related tables
    6. Consolidate odds → master_odds

Files are now organized under app/services/master_data/:
    - orchestrator.py        - Main entry point
    - population_service.py  - Populate master_teams, sources
    - mapping_service.py     - Map teams, players, games
    - player_service.py      - Map all players
    - odds_service.py        - Consolidate odds
    - team_data.py          - Canonical team rosters
    - source_registry.py    - Data source definitions
"""

import sys
import os

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the orchestrator
from app.services.master_data.orchestrator import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
