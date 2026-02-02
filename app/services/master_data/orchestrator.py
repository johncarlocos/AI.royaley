"""
ROYALEY - Master Data Orchestrator
Single command to rebuild or sync all master data.

Usage:
    python -m app.services.master_data.orchestrator [--full|--sync|--verify]

Options:
    --full    Full rebuild: populate + map + consolidate (default)
    --sync    Incremental sync: map new data only
    --verify  Verify current state, no changes
    --help    Show help

This orchestrator coordinates all master data services:
1. PopulationService - Seed master_teams, source_registry
2. MappingService - Map source → master (teams, players, games)
3. PlayerMappingService - Map all source players
4. OddsConsolidationService - Deduplicate odds
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Optional

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sqlalchemy import text
from app.core.database import db_manager

from .population_service import PopulationService
from .mapping_service import MappingService
from .player_service import PlayerMappingService
from .odds_service import OddsConsolidationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


class MasterDataOrchestrator:
    """Orchestrates all master data operations."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def full_rebuild(self) -> dict:
        """
        Full rebuild of all master data.
        
        Steps:
        1. Populate source_registry and master_teams (pro sports)
        2. Map teams → master_teams
        3. Map players → master_players (all sports)
        4. Create master_games + game_mappings
        5. Backfill master_*_id on related tables
        6. Consolidate odds → master_odds
        """
        self.start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("🚀 MASTER DATA FULL REBUILD")
        logger.info(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        await db_manager.initialize()
        
        # Step 1: Populate
        logger.info("\n📋 STEP 1: Populating source registry and master teams...")
        async with db_manager.session() as session:
            service = PopulationService(session)
            self.results["population"] = await service.populate_all()
        
        # Step 2-4: Map existing data
        logger.info("\n📋 STEP 2-4: Mapping existing data to master tables...")
        async with db_manager.session() as session:
            service = MappingService(session)
            self.results["mapping"] = await service.run_all_phases()
        
        # Step 5: Map all players (comprehensive)
        logger.info("\n📋 STEP 5: Mapping all players to master_players...")
        async with db_manager.session() as session:
            service = PlayerMappingService(session)
            self.results["players"] = await service.map_all_players()
        
        # Step 6: Consolidate odds
        logger.info("\n📋 STEP 6: Consolidating odds to master_odds...")
        async with db_manager.session() as session:
            service = OddsConsolidationService(session)
            self.results["odds"] = await service.consolidate_all()
        
        self.end_time = datetime.now()
        await self._print_summary()
        
        return self.results
    
    async def incremental_sync(self) -> dict:
        """
        Incremental sync: only map new/unmapped data.
        Faster than full rebuild for daily operations.
        """
        self.start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("🔄 MASTER DATA INCREMENTAL SYNC")
        logger.info(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        await db_manager.initialize()
        
        # Map new teams/players/games
        logger.info("\n📋 Mapping new data...")
        async with db_manager.session() as session:
            service = MappingService(session)
            self.results["mapping"] = await service.run_all_phases()
        
        # Map any new players
        logger.info("\n📋 Mapping new players...")
        async with db_manager.session() as session:
            service = PlayerMappingService(session)
            self.results["players"] = await service.map_all_players()
        
        # Consolidate new odds
        logger.info("\n📋 Consolidating new odds...")
        async with db_manager.session() as session:
            service = OddsConsolidationService(session)
            self.results["odds"] = await service.consolidate_all()
        
        self.end_time = datetime.now()
        await self._print_summary()
        
        return self.results
    
    async def verify(self) -> dict:
        """Verify current state without making changes."""
        logger.info("=" * 70)
        logger.info("🔍 MASTER DATA VERIFICATION")
        logger.info("=" * 70)
        
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            stats = {}
            
            # Count all tables
            tables = [
                ("source_registry", "sources"),
                ("master_teams", "teams"),
                ("master_players", "players"),
                ("master_games", "games"),
                ("master_odds", "odds"),
                ("team_mappings", "team_map"),
                ("player_mappings", "player_map"),
                ("game_mappings", "game_map"),
                ("odds_mappings", "odds_map"),
            ]
            
            for table, key in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[key] = result.scalar()
                except Exception:
                    stats[key] = 0
            
            # Count unmapped
            unmapped = {}
            try:
                result = await session.execute(text(
                    "SELECT COUNT(*) FROM teams WHERE master_team_id IS NULL"
                ))
                unmapped["teams"] = result.scalar()
            except Exception:
                unmapped["teams"] = 0
            
            try:
                result = await session.execute(text(
                    "SELECT COUNT(*) FROM players WHERE master_player_id IS NULL"
                ))
                unmapped["players"] = result.scalar()
            except Exception:
                unmapped["players"] = 0
            
            try:
                result = await session.execute(text(
                    "SELECT COUNT(*) FROM games WHERE master_game_id IS NULL"
                ))
                unmapped["games"] = result.scalar()
            except Exception:
                unmapped["games"] = 0
            
            try:
                result = await session.execute(text(
                    "SELECT COUNT(*) FROM odds WHERE master_game_id IS NULL"
                ))
                unmapped["odds"] = result.scalar()
            except Exception:
                unmapped["odds"] = 0
            
            # Sports breakdown
            sports_result = await session.execute(text("""
                SELECT sport_code, COUNT(*) as cnt
                FROM master_games
                GROUP BY sport_code
                ORDER BY cnt DESC
            """))
            by_sport = {r[0]: r[1] for r in sports_result.fetchall()}
        
        # Print report
        logger.info("\n📊 MASTER DATA STATUS")
        logger.info("-" * 50)
        logger.info(f"  Sources:        {stats.get('sources', 0):>10,}")
        logger.info(f"  Master Teams:   {stats.get('teams', 0):>10,}")
        logger.info(f"  Master Players: {stats.get('players', 0):>10,}")
        logger.info(f"  Master Games:   {stats.get('games', 0):>10,}")
        logger.info(f"  Master Odds:    {stats.get('odds', 0):>10,}")
        logger.info("-" * 50)
        logger.info(f"  Team Mappings:  {stats.get('team_map', 0):>10,}")
        logger.info(f"  Player Mappings:{stats.get('player_map', 0):>10,}")
        logger.info(f"  Game Mappings:  {stats.get('game_map', 0):>10,}")
        logger.info(f"  Odds Mappings:  {stats.get('odds_map', 0):>10,}")
        
        logger.info("\n⚠️  UNMAPPED RECORDS")
        logger.info("-" * 50)
        logger.info(f"  Teams:          {unmapped.get('teams', 0):>10,}")
        logger.info(f"  Players:        {unmapped.get('players', 0):>10,}")
        logger.info(f"  Games:          {unmapped.get('games', 0):>10,}")
        logger.info(f"  Odds:           {unmapped.get('odds', 0):>10,}")
        
        if by_sport:
            logger.info("\n🏈 GAMES BY SPORT")
            logger.info("-" * 50)
            for sport, count in by_sport.items():
                logger.info(f"  {sport:12s}    {count:>10,}")
        
        return {"stats": stats, "unmapped": unmapped, "by_sport": by_sport}
    
    async def _print_summary(self):
        """Print final summary."""
        duration = (self.end_time - self.start_time).total_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ MASTER DATA OPERATION COMPLETE")
        logger.info(f"   Duration: {minutes}m {seconds}s")
        logger.info("=" * 70)
        
        # Verify final state
        await self.verify()


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ROYALEY Master Data Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m app.services.master_data.orchestrator --full
    python -m app.services.master_data.orchestrator --sync
    python -m app.services.master_data.orchestrator --verify
        """
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full rebuild: populate + map + consolidate"
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Incremental sync: map new data only"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify current state, no changes"
    )
    
    args = parser.parse_args()
    
    orchestrator = MasterDataOrchestrator()
    
    if args.verify:
        await orchestrator.verify()
    elif args.sync:
        await orchestrator.incremental_sync()
    else:
        # Default to full rebuild
        await orchestrator.full_rebuild()


if __name__ == "__main__":
    asyncio.run(main())
