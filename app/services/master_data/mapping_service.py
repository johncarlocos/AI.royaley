"""
ROYALEY - Mapping Service
Maps existing source data (teams, players, games) to master tables.

Phases:
- Phase 3: Map teams → master_teams
- Phase 3b: Map tennis players → master_players  
- Phase 4: Create master_games + game_mappings
- Phase 5: Backfill master_*_id on odds/stats/betting tables
"""

import logging
import re
from collections import defaultdict
from datetime import timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .team_data import TEAM_ALIASES
from .source_registry import extract_source_key

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize a team/player name for matching."""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)  # remove punctuation
    s = re.sub(r'\s+', ' ', s)          # collapse whitespace
    return s


class MappingService:
    """Maps source data to canonical master records."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def run_all_phases(self) -> dict:
        """Run all mapping phases. Returns results dict."""
        results = {}
        
        # Phase 3: Teams
        results["teams"] = await self.phase3_map_teams()
        await self.session.commit()
        
        # Phase 3b: Tennis players
        results["tennis_players"] = await self.phase3b_map_tennis_players()
        await self.session.commit()
        
        # Phase 4: Games
        results["games"] = await self.phase4_create_master_games()
        await self.session.commit()
        
        # Phase 5: Backfill
        results["backfill"] = await self.phase5_backfill()
        await self.session.commit()
        
        return results
    
    async def phase3_map_teams(self) -> dict:
        """Map existing teams table to master_teams."""
        logger.info("=" * 70)
        logger.info("[PHASE 3] Auto-mapping existing teams → master_teams")
        logger.info("=" * 70)
        
        # Get all master teams
        master_rows = await self.session.execute(text(
            "SELECT id, sport_code, canonical_name, abbreviation, city FROM master_teams"
        ))
        master_teams = master_rows.fetchall()
        
        # Build lookup indices
        by_name = {}   # (sport_code, normalized_name) → master_team_id
        by_abbr = {}   # (sport_code, abbr) → master_team_id
        
        for mt in master_teams:
            mid, sport, cname, abbr, city = mt
            norm = normalize_name(cname)
            by_name[(sport, norm)] = mid
            if abbr:
                by_abbr[(sport, abbr.upper())] = mid
            if city:
                parts = cname.split()
                if len(parts) > 1:
                    by_name[(sport, normalize_name(parts[-1]))] = mid
        
        # Get sport_code mapping
        sports_rows = await self.session.execute(text("SELECT id, code FROM sports"))
        sport_map = {str(r[0]): r[1] for r in sports_rows.fetchall()}
        
        # Get unmapped teams
        teams_rows = await self.session.execute(text(
            "SELECT id, sport_id, external_id, name, abbreviation "
            "FROM teams WHERE master_team_id IS NULL"
        ))
        teams = teams_rows.fetchall()
        logger.info(f"  Found {len(teams)} unmapped team records")
        
        mapped = 0
        unmapped_count = 0
        batch_updates = []
        
        for team_row in teams:
            tid, sport_id, ext_id, tname, tabbr = team_row
            sport_code = sport_map.get(str(sport_id), "")
            source_key = extract_source_key(ext_id) if ext_id else "unknown"
            
            if not sport_code or not tname:
                continue
            
            # Tennis: teams are actually players
            if sport_code in ("ATP", "WTA"):
                continue
            
            master_id = None
            confidence = 0.0
            norm = normalize_name(tname)
            
            # 1. Exact name match
            master_id = by_name.get((sport_code, norm))
            if master_id:
                confidence = 1.0
            else:
                # 2. Abbreviation match
                if tabbr:
                    master_id = by_abbr.get((sport_code, tabbr.upper()))
                    if master_id:
                        confidence = 0.95
            
            if not master_id:
                # 3. Try alias table
                alias_target = TEAM_ALIASES.get(norm)
                if alias_target:
                    master_id = by_name.get((sport_code, normalize_name(alias_target)))
                    if master_id:
                        confidence = 0.9
            
            if not master_id:
                # 4. Partial match
                for (sc, mn), mid in by_name.items():
                    if sc == sport_code and norm in mn:
                        master_id = mid
                        confidence = 0.85
                        break
            
            if master_id:
                batch_updates.append((str(master_id), str(tid)))
                # Create team_mapping
                await self.session.execute(text("""
                    INSERT INTO team_mappings (
                        id, master_team_id, source_key, source_team_name,
                        source_external_id, source_team_db_id, confidence, verified
                    )
                    VALUES (
                        gen_random_uuid(), :mid, :src, :name, :ext, :tid, :conf, :ver
                    )
                    ON CONFLICT ON CONSTRAINT uq_team_map_source_name DO NOTHING
                """), {
                    "mid": str(master_id),
                    "src": source_key,
                    "name": tname,
                    "ext": ext_id,
                    "tid": str(tid),
                    "conf": confidence,
                    "ver": confidence >= 1.0,
                })
                mapped += 1
            else:
                unmapped_count += 1
        
        # Batch update teams.master_team_id
        for master_id, team_id in batch_updates:
            await self.session.execute(text(
                "UPDATE teams SET master_team_id = :mid WHERE id = :tid"
            ), {"mid": master_id, "tid": team_id})
        
        logger.info(f"  ✅ Mapped: {mapped} teams")
        logger.info(f"  ⚠️  Unmapped: {unmapped_count} teams")
        
        return {"mapped": mapped, "unmapped": unmapped_count}
    
    async def phase3b_map_tennis_players(self) -> dict:
        """Map tennis players from teams table to master_players."""
        logger.info("\n[PHASE 3b] Mapping tennis players...")
        
        # Get sport IDs for ATP/WTA
        sports_rows = await self.session.execute(text(
            "SELECT id, code FROM sports WHERE code IN ('ATP', 'WTA')"
        ))
        tennis_sports = {r[1]: str(r[0]) for r in sports_rows.fetchall()}
        
        if not tennis_sports:
            logger.info("  No tennis sports found")
            return {"mapped": 0, "created": 0}
        
        total_mapped = 0
        total_created = 0
        
        for sport_code, sport_id in tennis_sports.items():
            # Get tennis "teams" (which are players)
            result = await self.session.execute(text("""
                SELECT id, external_id, name 
                FROM teams 
                WHERE sport_id = :sid AND master_team_id IS NULL
            """), {"sid": sport_id})
            tennis_players = result.fetchall()
            
            for tp in tennis_players:
                tid, ext_id, pname = tp
                if not pname:
                    continue
                
                source_key = extract_source_key(ext_id) if ext_id else "tennis_abstract"
                
                # Check if master_player exists
                existing = await self.session.execute(text("""
                    SELECT id FROM master_players 
                    WHERE sport_code = :sport 
                    AND LOWER(canonical_name) = LOWER(:name)
                    LIMIT 1
                """), {"sport": sport_code, "name": pname.strip()})
                master = existing.fetchone()
                
                if master:
                    master_player_id = str(master[0])
                else:
                    # Create new master_player
                    parts = pname.strip().split(None, 1)
                    first = parts[0] if parts else ""
                    last = parts[1] if len(parts) > 1 else parts[0] if parts else ""
                    
                    new_mp = await self.session.execute(text("""
                        INSERT INTO master_players (
                            id, sport_code, canonical_name, first_name, last_name, is_active
                        )
                        VALUES (gen_random_uuid(), :sport, :name, :first, :last, true)
                        RETURNING id
                    """), {
                        "sport": sport_code,
                        "name": pname.strip(),
                        "first": first,
                        "last": last,
                    })
                    master_player_id = str(new_mp.fetchone()[0])
                    total_created += 1
                
                # Create player_mapping
                await self.session.execute(text("""
                    INSERT INTO player_mappings (
                        id, master_player_id, source_key, source_player_name,
                        source_external_id, source_player_db_id, confidence, verified
                    )
                    VALUES (
                        gen_random_uuid(), :mpid, :src, :name, :ext, :tid, 1.0, true
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "mpid": master_player_id,
                    "src": source_key,
                    "name": pname,
                    "ext": ext_id,
                    "tid": str(tid),
                })
                total_mapped += 1
        
        logger.info(f"  ✅ Tennis players: {total_mapped} mapped, {total_created} created")
        return {"mapped": total_mapped, "created": total_created}
    
    async def phase4_create_master_games(self) -> dict:
        """Create master_games from source games."""
        logger.info("\n" + "=" * 70)
        logger.info("[PHASE 4] Creating master_games")
        logger.info("=" * 70)
        
        # Get sport codes
        sports_rows = await self.session.execute(text("SELECT id, code FROM sports"))
        sport_map = {str(r[0]): r[1] for r in sports_rows.fetchall()}
        
        # Count games
        count_result = await self.session.execute(text(
            "SELECT COUNT(*) FROM games WHERE master_game_id IS NULL"
        ))
        total_unmapped = count_result.scalar()
        logger.info(f"  {total_unmapped:,} source games to process")
        
        total_masters = 0
        total_mapped = 0
        
        for sport_id, sport_code in sport_map.items():
            is_tennis = sport_code in ("ATP", "WTA")
            
            games_result = await self.session.execute(text("""
                SELECT g.id, g.external_id, g.home_team_id, g.away_team_id,
                       g.scheduled_at, g.venue_id, g.status, g.home_score, g.away_score
                FROM games g
                WHERE g.sport_id = :sid AND g.master_game_id IS NULL
                ORDER BY g.scheduled_at
            """), {"sid": sport_id})
            games = games_result.fetchall()
            
            if not games:
                continue
            
            sport_masters = 0
            sport_mapped = 0
            
            for g in games:
                gid, ext_id, h_team_id, a_team_id, sched, venue_id, status, h_score, a_score = g
                
                if not sched:
                    continue
                
                source_key = extract_source_key(ext_id) if ext_id else "unknown"
                
                h_master_team = None
                a_master_team = None
                h_master_player = None
                a_master_player = None
                
                if not is_tennis:
                    # Get master team IDs
                    if h_team_id:
                        h_row = await self.session.execute(text(
                            "SELECT master_team_id FROM teams WHERE id = :tid"
                        ), {"tid": str(h_team_id)})
                        h_result = h_row.fetchone()
                        h_master_team = str(h_result[0]) if h_result and h_result[0] else None
                    
                    if a_team_id:
                        a_row = await self.session.execute(text(
                            "SELECT master_team_id FROM teams WHERE id = :tid"
                        ), {"tid": str(a_team_id)})
                        a_result = a_row.fetchone()
                        a_master_team = str(a_result[0]) if a_result and a_result[0] else None
                else:
                    # Tennis: look up player mappings
                    for team_id_str, target in [(str(h_team_id), "home"), (str(a_team_id), "away")]:
                        pm_row = await self.session.execute(text("""
                            SELECT pm.master_player_id FROM player_mappings pm
                            WHERE pm.source_player_db_id = :tid LIMIT 1
                        """), {"tid": team_id_str})
                        pm = pm_row.fetchone()
                        if pm:
                            if target == "home":
                                h_master_player = str(pm[0])
                            else:
                                a_master_player = str(pm[0])
                
                # Skip if we can't identify both sides
                if not is_tennis and (not h_master_team or not a_master_team):
                    continue
                if is_tennis and (not h_master_player or not a_master_player):
                    continue
                
                # Find existing master_game (±24h window)
                window_start = sched - timedelta(hours=24)
                window_end = sched + timedelta(hours=24)
                
                if is_tennis:
                    existing_row = await self.session.execute(text("""
                        SELECT id FROM master_games
                        WHERE sport_code = :sport
                          AND scheduled_at BETWEEN :ws AND :we
                          AND home_master_player_id = :hp
                          AND away_master_player_id = :ap
                        LIMIT 1
                    """), {
                        "sport": sport_code,
                        "ws": window_start,
                        "we": window_end,
                        "hp": h_master_player,
                        "ap": a_master_player,
                    })
                else:
                    existing_row = await self.session.execute(text("""
                        SELECT id FROM master_games
                        WHERE sport_code = :sport
                          AND scheduled_at BETWEEN :ws AND :we
                          AND home_master_team_id = :ht
                          AND away_master_team_id = :at
                        LIMIT 1
                    """), {
                        "sport": sport_code,
                        "ws": window_start,
                        "we": window_end,
                        "ht": h_master_team,
                        "at": a_master_team,
                    })
                
                existing = existing_row.fetchone()
                
                if existing:
                    master_game_id = str(existing[0])
                else:
                    # Create new master_game
                    mg_row = await self.session.execute(text("""
                        INSERT INTO master_games (
                            id, sport_code, scheduled_at,
                            home_master_team_id, away_master_team_id,
                            home_master_player_id, away_master_player_id,
                            venue_id, status, home_score, away_score, primary_source
                        )
                        VALUES (
                            gen_random_uuid(), :sport, :sched,
                            :ht, :at, :hp, :ap, :vid, :status, :hs, :as_, :src
                        )
                        RETURNING id
                    """), {
                        "sport": sport_code,
                        "sched": sched,
                        "ht": h_master_team,
                        "at": a_master_team,
                        "hp": h_master_player,
                        "ap": a_master_player,
                        "vid": str(venue_id) if venue_id else None,
                        "status": status,
                        "hs": h_score,
                        "as_": a_score,
                        "src": source_key,
                    })
                    mg = mg_row.fetchone()
                    master_game_id = str(mg[0])
                    sport_masters += 1
                
                # Link source game → master_game
                await self.session.execute(text(
                    "UPDATE games SET master_game_id = :mgid WHERE id = :gid"
                ), {"mgid": master_game_id, "gid": str(gid)})
                
                # Create game_mapping
                if ext_id:
                    await self.session.execute(text("""
                        INSERT INTO game_mappings (
                            id, master_game_id, source_key, source_external_id, source_game_db_id
                        )
                        VALUES (gen_random_uuid(), :mgid, :src, :ext, :gid)
                        ON CONFLICT ON CONSTRAINT uq_game_map_source_ext DO NOTHING
                    """), {
                        "mgid": master_game_id,
                        "src": source_key,
                        "ext": ext_id,
                        "gid": str(gid),
                    })
                
                sport_mapped += 1
            
            total_masters += sport_masters
            total_mapped += sport_mapped
            
            if sport_mapped > 0:
                logger.info(f"  ✅ {sport_code}: {sport_masters} unique games, {sport_mapped} source records mapped")
            
            await self.session.commit()
        
        logger.info(f"\n  TOTAL: {total_masters} master_games created, {total_mapped} source games linked")
        return {"created": total_masters, "mapped": total_mapped}
    
    async def phase5_backfill(self) -> dict:
        """Backfill master_*_id on related tables."""
        logger.info("\n" + "=" * 70)
        logger.info("[PHASE 5] Backfilling master_*_id on related tables")
        logger.info("=" * 70)
        
        results = {}
        
        # Odds → master_game_id
        result = await self.session.execute(text("""
            UPDATE odds SET master_game_id = g.master_game_id
            FROM games g
            WHERE odds.game_id = g.id
              AND odds.master_game_id IS NULL
              AND g.master_game_id IS NOT NULL
        """))
        results["odds"] = result.rowcount
        logger.info(f"  ✅ Odds: {result.rowcount} records backfilled")
        
        # Player stats → master_game_id
        result = await self.session.execute(text("""
            UPDATE player_stats SET master_game_id = g.master_game_id
            FROM games g
            WHERE player_stats.game_id = g.id
              AND player_stats.master_game_id IS NULL
              AND g.master_game_id IS NOT NULL
        """))
        results["player_stats_game"] = result.rowcount
        logger.info(f"  ✅ Player stats (game): {result.rowcount} records backfilled")
        
        # Player stats → master_player_id
        result = await self.session.execute(text("""
            UPDATE player_stats SET master_player_id = p.master_player_id
            FROM players p
            WHERE player_stats.player_id = p.id
              AND player_stats.master_player_id IS NULL
              AND p.master_player_id IS NOT NULL
        """))
        results["player_stats_player"] = result.rowcount
        logger.info(f"  ✅ Player stats (player): {result.rowcount} records backfilled")
        
        # Players → master_player_id
        result = await self.session.execute(text("""
            UPDATE players SET master_player_id = pm.master_player_id
            FROM player_mappings pm
            WHERE players.id = pm.source_player_db_id
              AND players.master_player_id IS NULL
        """))
        results["players"] = result.rowcount
        logger.info(f"  ✅ Players: {result.rowcount} records backfilled")
        
        # Public betting → master_game_id
        result = await self.session.execute(text("""
            UPDATE public_betting pb SET master_game_id = mg.id
            FROM master_games mg
            JOIN master_teams ht ON mg.home_master_team_id = ht.id
            JOIN master_teams at_ ON mg.away_master_team_id = at_.id
            WHERE pb.master_game_id IS NULL
              AND pb.sport_code = mg.sport_code
              AND pb.game_date = DATE(mg.scheduled_at)
              AND (
                LOWER(pb.home_team) = LOWER(ht.canonical_name)
                OR LOWER(pb.home_team) = LOWER(ht.short_name)
                OR LOWER(pb.home_team) = LOWER(ht.abbreviation)
              )
              AND (
                LOWER(pb.away_team) = LOWER(at_.canonical_name)
                OR LOWER(pb.away_team) = LOWER(at_.short_name)
                OR LOWER(pb.away_team) = LOWER(at_.abbreviation)
              )
        """))
        results["public_betting"] = result.rowcount
        logger.info(f"  ✅ Public betting: {result.rowcount} records backfilled")
        
        return results


async def auto_map_existing_data(session: AsyncSession) -> dict:
    """Convenience function to run all mapping phases."""
    service = MappingService(session)
    return await service.run_all_phases()
