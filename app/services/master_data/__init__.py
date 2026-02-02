"""
ROYALEY - Master Data Services
Canonical data management for teams, players, games, and odds.

Quick Start:
    # Full rebuild (run once or after migrations)
    python -m app.services.master_data.orchestrator --full
    
    # Incremental sync (daily operations)
    python -m app.services.master_data.orchestrator --sync
    
    # Verify current state
    python -m app.services.master_data.orchestrator --verify

Services:
    MasterDataService      - Real-time resolver for collectors
    PopulationService      - Seed master_teams, source_registry
    MappingService         - Map source → master (teams, players, games)
    PlayerMappingService   - Map all source players
    OddsConsolidationService - Deduplicate odds
    MasterDataOrchestrator - Single command orchestration
"""

from .master_data_service import MasterDataService
from .feature_extractor import MasterFeatureExtractor, GameFeatureVector
from .population_service import PopulationService, populate_master_data
from .mapping_service import MappingService, auto_map_existing_data
from .player_service import PlayerMappingService, map_all_players
from .odds_service import OddsConsolidationService, consolidate_odds
from .orchestrator import MasterDataOrchestrator

# Team data
from .team_data import (
    NFL_TEAMS, NBA_TEAMS, MLB_TEAMS, NHL_TEAMS, WNBA_TEAMS, CFL_TEAMS,
    ALL_SPORT_TEAMS, TEAM_ALIASES
)

# Source registry data
from .source_registry import (
    SOURCES, SHARP_SPORTSBOOKS, SHARP_BOOK_ALIASES, extract_source_key
)

__all__ = [
    # Services
    "MasterDataService",
    "MasterFeatureExtractor",
    "GameFeatureVector",
    "PopulationService",
    "MappingService",
    "PlayerMappingService",
    "OddsConsolidationService",
    "MasterDataOrchestrator",
    # Convenience functions
    "populate_master_data",
    "auto_map_existing_data",
    "map_all_players",
    "consolidate_odds",
    # Data
    "NFL_TEAMS", "NBA_TEAMS", "MLB_TEAMS", "NHL_TEAMS", "WNBA_TEAMS", "CFL_TEAMS",
    "ALL_SPORT_TEAMS", "TEAM_ALIASES",
    "SOURCES", "SHARP_SPORTSBOOKS", "SHARP_BOOK_ALIASES", "extract_source_key",
]