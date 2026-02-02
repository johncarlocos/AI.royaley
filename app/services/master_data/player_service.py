"""
ROYALEY - Player Mapping Service
Maps source players to master_players for ALL sports.

Strategy:
1. Query all source players from `players` table (grouped by sport)
2. For each source player:
   - Check if master_player exists (by name + sport)
   - If not, create new master_player
   - Create player_mapping linking source → master
   - Update players.master_player_id
"""

import logging
from typing import Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .source_registry import extract_source_key

logger = logging.getLogger(__name__)


class PlayerMappingService:
    """Maps all source players to master_players."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def map_all_players(self) -> dict:
        """
        Map all unmapped players to master_players.
        Returns dict with counts by sport.
        """
        logger.info("=" * 70)
        logger.info("[PLAYER MAPPING] Creating master_players for all sports")
        logger.info("=" * 70)
        
        # Get sport codes
        sports_rows = await self.session.execute(text("SELECT id, code FROM sports"))
        sport_map = {str(r[0]): r[1] for r in sports_rows.fetchall()}
        
        # Count total unmapped
        count_result = await self.session.execute(text(
            "SELECT COUNT(*) FROM players WHERE master_player_id IS NULL"
        ))
        total_unmapped = count_result.scalar()
        logger.info(f"  {total_unmapped:,} unmapped source players")
        
        results = {}
        total_created = 0
        total_mapped = 0
        
        for sport_id, sport_code in sport_map.items():
            created, mapped = await self._map_sport_players(sport_id, sport_code)
            results[sport_code] = {"created": created, "mapped": mapped}
            total_created += created
            total_mapped += mapped
        
        await self.session.commit()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ PLAYER MAPPING COMPLETE")
        logger.info(f"   Master players created: {total_created:>10,}")
        logger.info(f"   Player mappings created: {total_mapped:>10,}")
        logger.info("=" * 60)
        
        return {
            "total_created": total_created,
            "total_mapped": total_mapped,
            "by_sport": results,
        }
    
    async def _map_sport_players(self, sport_id: str, sport_code: str) -> tuple:
        """Map all players for a single sport. Returns (created, mapped)."""
        
        # Get unmapped players for this sport
        result = await self.session.execute(text("""
            SELECT p.id, p.external_id, p.name, p.first_name, p.last_name, p.team_id
            FROM players p
            WHERE p.sport_id = :sid AND p.master_player_id IS NULL
        """), {"sid": sport_id})
        players = result.fetchall()
        
        if not players:
            return 0, 0
        
        created = 0
        mapped = 0
        batch_size = 1000
        
        # Build cache of existing master_players for this sport
        existing_result = await self.session.execute(text("""
            SELECT id, LOWER(canonical_name) as name FROM master_players
            WHERE sport_code = :sport
        """), {"sport": sport_code})
        existing_cache = {r[1]: str(r[0]) for r in existing_result.fetchall()}
        
        for batch_start in range(0, len(players), batch_size):
            batch = players[batch_start:batch_start + batch_size]
            
            for p in batch:
                pid, ext_id, pname, first, last, team_id = p
                
                if not pname:
                    continue
                
                source_key = extract_source_key(ext_id) if ext_id else "unknown"
                normalized_name = pname.strip().lower()
                
                # Get master_team_id from source team if available
                master_team_id = None
                if team_id:
                    team_result = await self.session.execute(text(
                        "SELECT master_team_id FROM teams WHERE id = :tid"
                    ), {"tid": str(team_id)})
                    team_row = team_result.fetchone()
                    if team_row and team_row[0]:
                        master_team_id = str(team_row[0])
                
                # Check if master_player exists
                master_player_id = existing_cache.get(normalized_name)
                
                if not master_player_id:
                    # Create new master_player
                    if not first and not last:
                        parts = pname.strip().split(None, 1)
                        first = parts[0] if parts else ""
                        last = parts[1] if len(parts) > 1 else parts[0] if parts else ""
                    
                    new_mp = await self.session.execute(text("""
                        INSERT INTO master_players (
                            id, sport_code, master_team_id, canonical_name,
                            first_name, last_name, is_active
                        )
                        VALUES (
                            gen_random_uuid(), :sport, :tid, :name,
                            :first, :last, true
                        )
                        ON CONFLICT DO NOTHING
                        RETURNING id
                    """), {
                        "sport": sport_code,
                        "tid": master_team_id,
                        "name": pname.strip(),
                        "first": first or "",
                        "last": last or "",
                    })
                    new_row = new_mp.fetchone()
                    
                    if new_row:
                        master_player_id = str(new_row[0])
                        existing_cache[normalized_name] = master_player_id
                        created += 1
                    else:
                        # Might have been created by concurrent process, try to find it
                        existing_row = await self.session.execute(text("""
                            SELECT id FROM master_players 
                            WHERE sport_code = :sport AND LOWER(canonical_name) = :name
                            LIMIT 1
                        """), {"sport": sport_code, "name": normalized_name})
                        existing = existing_row.fetchone()
                        if existing:
                            master_player_id = str(existing[0])
                            existing_cache[normalized_name] = master_player_id
                
                if master_player_id:
                    # Create player_mapping
                    await self.session.execute(text("""
                        INSERT INTO player_mappings (
                            id, master_player_id, source_key, source_player_name,
                            source_external_id, source_player_db_id, confidence, verified
                        )
                        VALUES (
                            gen_random_uuid(), :mpid, :src, :name, :ext, :pid, 1.0, true
                        )
                        ON CONFLICT DO NOTHING
                    """), {
                        "mpid": master_player_id,
                        "src": source_key,
                        "name": pname,
                        "ext": ext_id,
                        "pid": str(pid),
                    })
                    
                    # Update source player
                    await self.session.execute(text("""
                        UPDATE players SET master_player_id = :mpid WHERE id = :pid
                    """), {"mpid": master_player_id, "pid": str(pid)})
                    
                    mapped += 1
            
            await self.session.commit()
            
            if batch_start > 0 and batch_start % 5000 == 0:
                logger.info(f"     ... {sport_code}: processed {batch_start}/{len(players)}")
        
        if mapped > 0:
            logger.info(f"  ✅ {sport_code}: {created:,} created, {mapped:,} mapped")
        
        return created, mapped


async def map_all_players(session: AsyncSession) -> dict:
    """Convenience function to run player mapping."""
    service = PlayerMappingService(session)
    return await service.map_all_players()
