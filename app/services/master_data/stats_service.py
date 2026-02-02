"""
ROYALEY - Stats Consolidation Service
Consolidates player_stats and team_stats into master tables.

Similar to odds_service.py, this deduplicates stats from multiple sources
into canonical master_player_stats and master_team_stats records.

Flow:
  1. For each master_game, query all player_stats with master_player_id
  2. Group by master_player_id
  3. Merge stats from multiple sources (prioritize by source quality)
  4. INSERT into master_player_stats + create mappings
  
Same flow for team_stats → master_team_stats
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .source_registry import extract_source_key

logger = logging.getLogger(__name__)

# Source priority for stats (lower = better)
STATS_SOURCE_PRIORITY = {
    "nflfastr": 1,
    "cfbfastr": 1,
    "baseballr": 1,
    "hockeyr": 1,
    "hoopr": 1,
    "wehoop": 1,
    "espn": 5,
    "sportsdb": 10,
    "balldontlie": 8,
    "sportsipy": 15,
    "basketball_ref": 8,
    "tennis_abstract": 5,
    "unknown": 99,
}


class StatsConsolidationService:
    """Consolidates source stats into master stats tables."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def consolidate_all(self) -> dict:
        """
        Consolidate all stats into master tables.
        Returns dict with counts.
        """
        results = {}
        
        # First, backfill master_player_id on player_stats if not done
        await self._backfill_player_stats_ids()
        
        # Consolidate player stats
        results["player_stats"] = await self.consolidate_player_stats()
        
        # Consolidate team stats
        results["team_stats"] = await self.consolidate_team_stats()
        
        return results
    
    async def _backfill_player_stats_ids(self):
        """Ensure player_stats has master_player_id backfilled."""
        result = await self.session.execute(text("""
            UPDATE player_stats ps
            SET master_player_id = p.master_player_id
            FROM players p
            WHERE ps.player_id = p.id
              AND ps.master_player_id IS NULL
              AND p.master_player_id IS NOT NULL
        """))
        if result.rowcount > 0:
            logger.info(f"  Backfilled {result.rowcount:,} player_stats.master_player_id")
            await self.session.commit()
    
    async def consolidate_player_stats(self) -> dict:
        """Consolidate player_stats → master_player_stats."""
        logger.info("=" * 70)
        logger.info("[PLAYER STATS] Consolidating to master_player_stats")
        logger.info("=" * 70)
        
        # Count eligible records
        count_result = await self.session.execute(text("""
            SELECT COUNT(*) FROM player_stats 
            WHERE master_player_id IS NOT NULL 
              AND master_game_id IS NOT NULL
        """))
        total_source = count_result.scalar()
        logger.info(f"  Source records with master IDs: {total_source:,}")
        
        if total_source == 0:
            logger.warning("  No player_stats have master IDs. Run mapping first.")
            return {"source": 0, "master": 0, "mappings": 0}
        
        # Get distinct sports
        sports_result = await self.session.execute(text("""
            SELECT DISTINCT mg.sport_code
            FROM player_stats ps
            JOIN master_games mg ON ps.master_game_id = mg.id
            WHERE ps.master_player_id IS NOT NULL
        """))
        sports = [r[0] for r in sports_result.fetchall()]
        logger.info(f"  Sports: {', '.join(sports)}")
        
        total_master = 0
        total_mappings = 0
        
        for sport_code in sports:
            m, mp = await self._consolidate_player_stats_sport(sport_code)
            total_master += m
            total_mappings += mp
        
        await self.session.commit()
        
        logger.info("")
        logger.info(f"✅ PLAYER STATS: {total_master:,} master records, {total_mappings:,} mappings")
        
        return {"source": total_source, "master": total_master, "mappings": total_mappings}
    
    async def _consolidate_player_stats_sport(self, sport_code: str) -> tuple:
        """Consolidate player stats for one sport."""
        logger.info(f"\n  Processing {sport_code}...")
        
        # Get all (master_player_id, master_game_id) pairs
        pairs_result = await self.session.execute(text("""
            SELECT DISTINCT ps.master_player_id, ps.master_game_id
            FROM player_stats ps
            JOIN master_games mg ON ps.master_game_id = mg.id
            WHERE ps.master_player_id IS NOT NULL
              AND mg.sport_code = :sport
        """), {"sport": sport_code})
        pairs = pairs_result.fetchall()
        logger.info(f"    {len(pairs):,} unique (player, game) pairs")
        
        master_count = 0
        mapping_count = 0
        batch_size = 1000
        
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            
            for master_player_id, master_game_id in batch:
                m, mp = await self._consolidate_single_player_game(
                    str(master_player_id), str(master_game_id), sport_code
                )
                master_count += m
                mapping_count += mp
            
            await self.session.commit()
            
            if batch_start > 0 and batch_start % 5000 == 0:
                logger.info(f"    ... processed {batch_start:,}/{len(pairs):,}")
        
        logger.info(f"    ✅ {sport_code}: {master_count:,} master stats")
        return master_count, mapping_count
    
    async def _consolidate_single_player_game(
        self, master_player_id: str, master_game_id: str, sport_code: str
    ) -> tuple:
        """Consolidate all player_stats for one player in one game."""
        
        # Get all source stats for this player/game
        result = await self.session.execute(text("""
            SELECT ps.id, ps.stat_type, ps.stats, ps.minutes_played, ps.value,
                   p.external_id, mp.master_team_id
            FROM player_stats ps
            JOIN players p ON ps.player_id = p.id
            LEFT JOIN master_players mp ON ps.master_player_id = mp.id
            WHERE ps.master_player_id = :mpid
              AND ps.master_game_id = :mgid
        """), {"mpid": master_player_id, "mgid": master_game_id})
        rows = result.fetchall()
        
        if not rows:
            return 0, 0
        
        # Merge stats from all sources
        merged_stats = {}
        source_ids = []
        primary_source = "unknown"
        best_priority = 999
        minutes_played = None
        master_team_id = None
        
        for row in rows:
            ps_id, stat_type, stats_json, mins, value, ext_id, team_id = row
            source_key = extract_source_key(ext_id) if ext_id else "unknown"
            source_ids.append(str(ps_id))
            
            # Track best source
            priority = STATS_SOURCE_PRIORITY.get(source_key, 99)
            if priority < best_priority:
                best_priority = priority
                primary_source = source_key
            
            # Merge JSON stats
            if stats_json:
                for k, v in stats_json.items():
                    if k not in merged_stats or (v is not None and merged_stats[k] is None):
                        merged_stats[k] = v
            
            # Take non-null values
            if mins and (minutes_played is None or priority < best_priority):
                minutes_played = mins
            if team_id:
                master_team_id = str(team_id)
        
        # Extract common stats from merged JSON
        stats = self._extract_player_stats(merged_stats, sport_code)
        stats["minutes_played"] = minutes_played
        
        # Insert master_player_stats
        try:
            await self.session.execute(text("""
                INSERT INTO master_player_stats (
                    id, master_player_id, master_game_id, master_team_id, sport_code,
                    minutes_played, points, assists, rebounds, steals, blocks, turnovers,
                    passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds,
                    receiving_yards, receiving_tds, receptions, targets, carries,
                    at_bats, hits, runs, rbis, home_runs, stolen_bases,
                    goals, hockey_assists, plus_minus, shots_on_goal, saves,
                    aces, double_faults, first_serve_pct, sets_won, games_won,
                    stats_json, primary_source, num_source_records
                ) VALUES (
                    gen_random_uuid(), :mpid, :mgid, :mtid, :sport,
                    :minutes, :points, :assists, :rebounds, :steals, :blocks, :turnovers,
                    :pass_yds, :pass_tds, :ints, :rush_yds, :rush_tds,
                    :rec_yds, :rec_tds, :receptions, :targets, :carries,
                    :at_bats, :hits, :runs, :rbis, :home_runs, :stolen_bases,
                    :goals, :hockey_assists, :plus_minus, :shots, :saves,
                    :aces, :double_faults, :first_serve, :sets_won, :games_won,
                    :stats_json, :source, :num_sources
                )
                ON CONFLICT (master_player_id, master_game_id) DO UPDATE SET
                    stats_json = EXCLUDED.stats_json,
                    num_source_records = EXCLUDED.num_source_records,
                    updated_at = NOW()
                RETURNING id
            """), {
                "mpid": master_player_id,
                "mgid": master_game_id,
                "mtid": master_team_id,
                "sport": sport_code,
                "minutes": stats.get("minutes_played"),
                "points": stats.get("points"),
                "assists": stats.get("assists"),
                "rebounds": stats.get("rebounds"),
                "steals": stats.get("steals"),
                "blocks": stats.get("blocks"),
                "turnovers": stats.get("turnovers"),
                "pass_yds": stats.get("passing_yards"),
                "pass_tds": stats.get("passing_tds"),
                "ints": stats.get("interceptions"),
                "rush_yds": stats.get("rushing_yards"),
                "rush_tds": stats.get("rushing_tds"),
                "rec_yds": stats.get("receiving_yards"),
                "rec_tds": stats.get("receiving_tds"),
                "receptions": stats.get("receptions"),
                "targets": stats.get("targets"),
                "carries": stats.get("carries"),
                "at_bats": stats.get("at_bats"),
                "hits": stats.get("hits"),
                "runs": stats.get("runs"),
                "rbis": stats.get("rbis"),
                "home_runs": stats.get("home_runs"),
                "stolen_bases": stats.get("stolen_bases"),
                "goals": stats.get("goals"),
                "hockey_assists": stats.get("hockey_assists"),
                "plus_minus": stats.get("plus_minus"),
                "shots": stats.get("shots_on_goal"),
                "saves": stats.get("saves"),
                "aces": stats.get("aces"),
                "double_faults": stats.get("double_faults"),
                "first_serve": stats.get("first_serve_pct"),
                "sets_won": stats.get("sets_won"),
                "games_won": stats.get("games_won"),
                "stats_json": merged_stats if merged_stats else None,
                "source": primary_source,
                "num_sources": len(source_ids),
            })
            
            # Get the master_player_stats ID
            mps_result = await self.session.execute(text("""
                SELECT id FROM master_player_stats
                WHERE master_player_id = :mpid AND master_game_id = :mgid
            """), {"mpid": master_player_id, "mgid": master_game_id})
            mps_row = mps_result.fetchone()
            mps_id = str(mps_row[0]) if mps_row else None
            
            # Create mappings
            mapping_count = 0
            if mps_id:
                for src_id in source_ids:
                    try:
                        await self.session.execute(text("""
                            INSERT INTO player_stats_mappings (
                                id, master_player_stats_id, source_key, source_player_stats_db_id
                            ) VALUES (gen_random_uuid(), :mpsid, :src, :srcid)
                            ON CONFLICT DO NOTHING
                        """), {"mpsid": mps_id, "src": primary_source, "srcid": src_id})
                        mapping_count += 1
                    except Exception:
                        pass
            
            return 1, mapping_count
            
        except Exception as e:
            logger.debug(f"Error inserting master_player_stats: {e}")
            return 0, 0
    
    def _extract_player_stats(self, stats: dict, sport_code: str) -> dict:
        """Extract typed stats from JSON based on sport."""
        result = {}
        
        # Common mappings (different sources use different keys)
        mappings = {
            "points": ["points", "pts", "PTS"],
            "assists": ["assists", "ast", "AST"],
            "rebounds": ["rebounds", "reb", "REB", "total_rebounds"],
            "steals": ["steals", "stl", "STL"],
            "blocks": ["blocks", "blk", "BLK"],
            "turnovers": ["turnovers", "tov", "TOV", "to"],
            "passing_yards": ["passing_yards", "pass_yds", "passYards"],
            "passing_tds": ["passing_tds", "pass_td", "passTD"],
            "interceptions": ["interceptions", "int", "INT"],
            "rushing_yards": ["rushing_yards", "rush_yds", "rushYards"],
            "rushing_tds": ["rushing_tds", "rush_td", "rushTD"],
            "receiving_yards": ["receiving_yards", "rec_yds", "recYards"],
            "receiving_tds": ["receiving_tds", "rec_td", "recTD"],
            "receptions": ["receptions", "rec", "REC"],
            "targets": ["targets", "tgt", "TGT"],
            "carries": ["carries", "car", "CAR", "rush_att"],
            "at_bats": ["at_bats", "ab", "AB"],
            "hits": ["hits", "h", "H"],
            "runs": ["runs", "r", "R"],
            "rbis": ["rbis", "rbi", "RBI"],
            "home_runs": ["home_runs", "hr", "HR"],
            "stolen_bases": ["stolen_bases", "sb", "SB"],
            "goals": ["goals", "g", "G"],
            "hockey_assists": ["assists", "a", "A"],
            "plus_minus": ["plus_minus", "plusMinus", "pm"],
            "shots_on_goal": ["shots_on_goal", "sog", "SOG", "shots"],
            "saves": ["saves", "sv", "SV"],
            "aces": ["aces"],
            "double_faults": ["double_faults", "doubleFaults"],
            "first_serve_pct": ["first_serve_pct", "firstServePct"],
            "sets_won": ["sets_won", "setsWon"],
            "games_won": ["games_won", "gamesWon"],
        }
        
        for target_key, source_keys in mappings.items():
            for src_key in source_keys:
                if src_key in stats and stats[src_key] is not None:
                    try:
                        result[target_key] = int(stats[src_key]) if isinstance(stats[src_key], (int, float)) else stats[src_key]
                    except (ValueError, TypeError):
                        pass
                    break
        
        return result
    
    async def consolidate_team_stats(self) -> dict:
        """Consolidate team_stats → master_team_stats."""
        logger.info("\n" + "=" * 70)
        logger.info("[TEAM STATS] Consolidating to master_team_stats")
        logger.info("=" * 70)
        
        # Check if team_stats has master_team_id column
        col_check = await self.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'team_stats' AND column_name = 'master_team_id'
        """))
        has_master_team_id = col_check.fetchone() is not None
        
        if not has_master_team_id:
            logger.warning("  team_stats table missing master_team_id column. Adding...")
            await self.session.execute(text("""
                ALTER TABLE team_stats ADD COLUMN IF NOT EXISTS master_team_id UUID REFERENCES master_teams(id)
            """))
            await self.session.execute(text("""
                ALTER TABLE team_stats ADD COLUMN IF NOT EXISTS master_game_id UUID REFERENCES master_games(id)
            """))
            await self.session.commit()
        
        # Backfill master_team_id
        result = await self.session.execute(text("""
            UPDATE team_stats ts
            SET master_team_id = t.master_team_id
            FROM teams t
            WHERE ts.team_id = t.id
              AND ts.master_team_id IS NULL
              AND t.master_team_id IS NOT NULL
        """))
        if result.rowcount > 0:
            logger.info(f"  Backfilled {result.rowcount:,} team_stats.master_team_id")
        
        # Backfill master_game_id
        result = await self.session.execute(text("""
            UPDATE team_stats ts
            SET master_game_id = g.master_game_id
            FROM games g
            WHERE ts.game_id = g.id
              AND ts.master_game_id IS NULL
              AND g.master_game_id IS NOT NULL
        """))
        if result.rowcount > 0:
            logger.info(f"  Backfilled {result.rowcount:,} team_stats.master_game_id")
        
        await self.session.commit()
        
        # Count eligible records
        count_result = await self.session.execute(text("""
            SELECT COUNT(*) FROM team_stats 
            WHERE master_team_id IS NOT NULL 
              AND master_game_id IS NOT NULL
        """))
        total_source = count_result.scalar()
        logger.info(f"  Source records with master IDs: {total_source:,}")
        
        if total_source == 0:
            logger.warning("  No team_stats have master IDs.")
            return {"source": 0, "master": 0, "mappings": 0}
        
        # Get distinct sports
        sports_result = await self.session.execute(text("""
            SELECT DISTINCT mg.sport_code
            FROM team_stats ts
            JOIN master_games mg ON ts.master_game_id = mg.id
            WHERE ts.master_team_id IS NOT NULL
        """))
        sports = [r[0] for r in sports_result.fetchall()]
        
        total_master = 0
        total_mappings = 0
        
        for sport_code in sports:
            m, mp = await self._consolidate_team_stats_sport(sport_code)
            total_master += m
            total_mappings += mp
        
        await self.session.commit()
        
        logger.info(f"\n✅ TEAM STATS: {total_master:,} master records, {total_mappings:,} mappings")
        
        return {"source": total_source, "master": total_master, "mappings": total_mappings}
    
    async def _consolidate_team_stats_sport(self, sport_code: str) -> tuple:
        """Consolidate team stats for one sport."""
        logger.info(f"\n  Processing {sport_code}...")
        
        # Get all (master_team_id, master_game_id) pairs
        pairs_result = await self.session.execute(text("""
            SELECT DISTINCT ts.master_team_id, ts.master_game_id
            FROM team_stats ts
            JOIN master_games mg ON ts.master_game_id = mg.id
            WHERE ts.master_team_id IS NOT NULL
              AND mg.sport_code = :sport
        """), {"sport": sport_code})
        pairs = pairs_result.fetchall()
        logger.info(f"    {len(pairs):,} unique (team, game) pairs")
        
        master_count = 0
        mapping_count = 0
        batch_size = 1000
        
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            
            for master_team_id, master_game_id in batch:
                m, mp = await self._consolidate_single_team_game(
                    str(master_team_id), str(master_game_id), sport_code
                )
                master_count += m
                mapping_count += mp
            
            await self.session.commit()
        
        logger.info(f"    ✅ {sport_code}: {master_count:,} master stats")
        return master_count, mapping_count
    
    async def _consolidate_single_team_game(
        self, master_team_id: str, master_game_id: str, sport_code: str
    ) -> tuple:
        """Consolidate all team_stats for one team in one game."""
        
        # Get all source stats
        result = await self.session.execute(text("""
            SELECT ts.id, ts.stat_type, ts.stats,
                   t.external_id,
                   mg.home_master_team_id
            FROM team_stats ts
            JOIN teams t ON ts.team_id = t.id
            JOIN master_games mg ON ts.master_game_id = mg.id
            WHERE ts.master_team_id = :mtid
              AND ts.master_game_id = :mgid
        """), {"mtid": master_team_id, "mgid": master_game_id})
        rows = result.fetchall()
        
        if not rows:
            return 0, 0
        
        # Merge stats
        merged_stats = {}
        source_ids = []
        primary_source = "unknown"
        is_home = None
        
        for row in rows:
            ts_id, stat_type, stats_json, ext_id, home_team_id = row
            source_key = extract_source_key(ext_id) if ext_id else "unknown"
            source_ids.append(str(ts_id))
            
            if primary_source == "unknown":
                primary_source = source_key
            
            if stats_json:
                for k, v in stats_json.items():
                    if k not in merged_stats or (v is not None and merged_stats[k] is None):
                        merged_stats[k] = v
            
            if home_team_id:
                is_home = str(home_team_id) == master_team_id
        
        # Extract stats
        stats = self._extract_team_stats(merged_stats, sport_code)
        
        try:
            await self.session.execute(text("""
                INSERT INTO master_team_stats (
                    id, master_team_id, master_game_id, sport_code, is_home,
                    points_scored, points_allowed, total_yards, passing_yards, rushing_yards,
                    turnovers, first_downs, penalties, penalty_yards, time_of_possession,
                    field_goals_made, field_goals_att, three_pointers_made, three_pointers_att,
                    free_throws_made, free_throws_att, rebounds, assists, steals, blocks,
                    runs, hits, errors, home_runs, strikeouts, walks,
                    goals, shots, power_play_goals, penalty_minutes,
                    stats_json, primary_source, num_source_records
                ) VALUES (
                    gen_random_uuid(), :mtid, :mgid, :sport, :is_home,
                    :pts_scored, :pts_allowed, :total_yds, :pass_yds, :rush_yds,
                    :turnovers, :first_downs, :penalties, :pen_yds, :top,
                    :fgm, :fga, :tpm, :tpa, :ftm, :fta, :reb, :ast, :stl, :blk,
                    :runs, :hits, :errors, :hr, :so, :bb,
                    :goals, :shots, :ppg, :pim,
                    :stats_json, :source, :num_sources
                )
                ON CONFLICT (master_team_id, master_game_id) DO UPDATE SET
                    stats_json = EXCLUDED.stats_json,
                    num_source_records = EXCLUDED.num_source_records,
                    updated_at = NOW()
                RETURNING id
            """), {
                "mtid": master_team_id,
                "mgid": master_game_id,
                "sport": sport_code,
                "is_home": is_home,
                "pts_scored": stats.get("points_scored"),
                "pts_allowed": stats.get("points_allowed"),
                "total_yds": stats.get("total_yards"),
                "pass_yds": stats.get("passing_yards"),
                "rush_yds": stats.get("rushing_yards"),
                "turnovers": stats.get("turnovers"),
                "first_downs": stats.get("first_downs"),
                "penalties": stats.get("penalties"),
                "pen_yds": stats.get("penalty_yards"),
                "top": stats.get("time_of_possession"),
                "fgm": stats.get("field_goals_made"),
                "fga": stats.get("field_goals_att"),
                "tpm": stats.get("three_pointers_made"),
                "tpa": stats.get("three_pointers_att"),
                "ftm": stats.get("free_throws_made"),
                "fta": stats.get("free_throws_att"),
                "reb": stats.get("rebounds"),
                "ast": stats.get("assists"),
                "stl": stats.get("steals"),
                "blk": stats.get("blocks"),
                "runs": stats.get("runs"),
                "hits": stats.get("hits"),
                "errors": stats.get("errors"),
                "hr": stats.get("home_runs"),
                "so": stats.get("strikeouts"),
                "bb": stats.get("walks"),
                "goals": stats.get("goals"),
                "shots": stats.get("shots"),
                "ppg": stats.get("power_play_goals"),
                "pim": stats.get("penalty_minutes"),
                "stats_json": merged_stats if merged_stats else None,
                "source": primary_source,
                "num_sources": len(source_ids),
            })
            
            return 1, len(source_ids)
            
        except Exception as e:
            logger.debug(f"Error inserting master_team_stats: {e}")
            return 0, 0
    
    def _extract_team_stats(self, stats: dict, sport_code: str) -> dict:
        """Extract typed team stats from JSON."""
        result = {}
        
        mappings = {
            "points_scored": ["points", "pts", "score", "PTS"],
            "points_allowed": ["points_allowed", "opp_points", "oppPts"],
            "total_yards": ["total_yards", "totalYards", "yards"],
            "passing_yards": ["passing_yards", "passYards", "pass_yds"],
            "rushing_yards": ["rushing_yards", "rushYards", "rush_yds"],
            "turnovers": ["turnovers", "to", "TO"],
            "first_downs": ["first_downs", "firstDowns", "fd"],
            "penalties": ["penalties", "pen"],
            "penalty_yards": ["penalty_yards", "penYards"],
            "time_of_possession": ["time_of_possession", "top", "TOP"],
            "field_goals_made": ["fgm", "field_goals_made"],
            "field_goals_att": ["fga", "field_goals_att"],
            "three_pointers_made": ["tpm", "three_pointers_made", "3pm"],
            "three_pointers_att": ["tpa", "three_pointers_att", "3pa"],
            "free_throws_made": ["ftm", "free_throws_made"],
            "free_throws_att": ["fta", "free_throws_att"],
            "rebounds": ["rebounds", "reb", "REB"],
            "assists": ["assists", "ast", "AST"],
            "steals": ["steals", "stl", "STL"],
            "blocks": ["blocks", "blk", "BLK"],
            "runs": ["runs", "r", "R"],
            "hits": ["hits", "h", "H"],
            "errors": ["errors", "e", "E"],
            "home_runs": ["home_runs", "hr", "HR"],
            "strikeouts": ["strikeouts", "so", "SO", "k", "K"],
            "walks": ["walks", "bb", "BB"],
            "goals": ["goals", "g", "G"],
            "shots": ["shots", "sog", "SOG"],
            "power_play_goals": ["power_play_goals", "ppg", "PPG"],
            "penalty_minutes": ["penalty_minutes", "pim", "PIM"],
        }
        
        for target_key, source_keys in mappings.items():
            for src_key in source_keys:
                if src_key in stats and stats[src_key] is not None:
                    try:
                        val = stats[src_key]
                        if isinstance(val, (int, float)):
                            result[target_key] = int(val) if target_key != "time_of_possession" else float(val)
                        else:
                            result[target_key] = val
                    except (ValueError, TypeError):
                        pass
                    break
        
        return result


async def consolidate_stats(session: AsyncSession) -> dict:
    """Convenience function to run stats consolidation."""
    service = StatsConsolidationService(session)
    return await service.consolidate_all()
