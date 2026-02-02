"""
ROYALEY - Population Service
Handles initial population of master_teams and source_registry tables.

This is idempotent — safe to run multiple times (uses ON CONFLICT DO NOTHING).
"""

import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .team_data import ALL_SPORT_TEAMS
from .source_registry import SOURCES, SHARP_SPORTSBOOKS

logger = logging.getLogger(__name__)


class PopulationService:
    """Populates canonical master data tables."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def populate_all(self) -> dict:
        """
        Populate all master data tables.
        Returns dict with counts for each operation.
        """
        results = {}
        
        # 1. Source registry
        results["sources"] = await self.populate_sources()
        
        # 2. Master teams (pro sports only)
        results["teams"] = await self.populate_teams()
        
        # 3. Sportsbook priorities
        results["sportsbooks"] = await self.fix_sportsbook_priorities()
        
        await self.session.commit()
        return results
    
    async def populate_sources(self) -> int:
        """Populate source_registry table."""
        logger.info("Populating source_registry...")
        
        count = 0
        for src in SOURCES:
            key, name, stype, priority, teams, players, games, odds, stats, sports = src
            
            result = await self.session.execute(text("""
                INSERT INTO source_registry (
                    id, key, name, source_type, priority, is_active,
                    provides_teams, provides_players, provides_games, 
                    provides_odds, provides_stats, sports_covered
                )
                VALUES (
                    gen_random_uuid(), :key, :name, :stype, :priority, true,
                    :teams, :players, :games, :odds, :stats, :sports
                )
                ON CONFLICT (key) DO NOTHING
            """), {
                "key": key,
                "name": name,
                "stype": stype,
                "priority": priority,
                "teams": teams,
                "players": players,
                "games": games,
                "odds": odds,
                "stats": stats,
                "sports": json.dumps(sports),
            })
            count += 1
        
        logger.info(f"  ✅ Registered {count} data sources")
        return count
    
    async def populate_teams(self) -> int:
        """Populate master_teams for all team sports."""
        logger.info("Populating master_teams...")
        
        total = 0
        for sport_code, teams_list in ALL_SPORT_TEAMS.items():
            sport_count = 0
            
            for name, abbr, city, conf, div in teams_list:
                await self.session.execute(text("""
                    INSERT INTO master_teams (
                        id, sport_code, canonical_name, abbreviation,
                        city, conference, division, is_active
                    )
                    VALUES (
                        gen_random_uuid(), :sport, :name, :abbr, 
                        :city, :conf, :div, true
                    )
                    ON CONFLICT ON CONSTRAINT uq_master_teams_sport_name DO NOTHING
                """), {
                    "sport": sport_code,
                    "name": name,
                    "abbr": abbr,
                    "city": city,
                    "conf": conf,
                    "div": div,
                })
                sport_count += 1
            
            total += sport_count
            logger.info(f"  ✅ {sport_code}: {sport_count} teams")
        
        logger.info(f"  Total master teams: {total}")
        return total
    
    async def fix_sportsbook_priorities(self) -> int:
        """Update sportsbook priorities for sharp books."""
        logger.info("Fixing sportsbook priorities...")
        
        updated = 0
        for book_key, (priority, is_sharp) in SHARP_SPORTSBOOKS.items():
            result = await self.session.execute(text("""
                UPDATE sportsbooks 
                SET is_sharp = :sharp, priority = :pri
                WHERE key = :key
            """), {"sharp": is_sharp, "pri": priority, "key": book_key})
            
            if result.rowcount > 0:
                logger.info(f"  ✅ {book_key}: priority={priority}, is_sharp={is_sharp}")
                updated += 1
        
        # Set consumer books to lower priority
        await self.session.execute(text("""
            UPDATE sportsbooks 
            SET priority = 50
            WHERE is_sharp = false AND priority = 100
        """))
        
        return updated


async def populate_master_data(session: AsyncSession) -> dict:
    """Convenience function to run population."""
    service = PopulationService(session)
    return await service.populate_all()
