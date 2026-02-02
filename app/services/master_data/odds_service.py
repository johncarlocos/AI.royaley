"""
ROYALEY - Odds Consolidation Service
Deduplicates odds across multiple collectors into one canonical record
per (master_game × sportsbook × bet_type).

Flow:
  1. For each master_game, query ALL raw odds linked via master_game_id
  2. Group by (sportsbook_key, bet_type)
  3. Sort by recorded_at to identify opening vs closing lines
  4. Compute line_movement, no_vig probability
  5. INSERT/UPDATE master_odds + create odds_mappings

Expected: ~1.4M raw odds → ~400K master_odds (3.5x dedup ratio)
"""

import logging
from datetime import datetime
from uuid import uuid4
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .source_registry import SHARP_BOOK_ALIASES

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0.0-1.0)."""
    if odds is None or odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def devig_two_way(home_odds: int, away_odds: int) -> Optional[float]:
    """Remove vig to get true probability. Returns home prob."""
    if not home_odds or not away_odds:
        return None
    p_home = american_to_implied(home_odds)
    p_away = american_to_implied(away_odds)
    total = p_home + p_away
    if total == 0:
        return None
    return round(p_home / total, 6)


class OddsConsolidationService:
    """Consolidates raw odds into master_odds with deduplication."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def consolidate_all(self) -> dict:
        """
        Consolidate all raw odds into master_odds.
        Returns dict with counts.
        """
        # Count raw odds with master_game_id
        count_result = await self.session.execute(text(
            "SELECT COUNT(*) FROM odds WHERE master_game_id IS NOT NULL"
        ))
        total_raw = count_result.scalar()
        logger.info(f"📊 Raw odds with master_game_id: {total_raw:,}")
        
        if total_raw == 0:
            logger.warning("⚠️  No odds have master_game_id yet. Run mapping first.")
            return {"raw": 0, "master": 0, "mappings": 0}
        
        # Get all distinct sports
        sport_result = await self.session.execute(text(
            "SELECT DISTINCT sport_code FROM master_games ORDER BY sport_code"
        ))
        sports = [r[0] for r in sport_result.fetchall()]
        logger.info(f"🏈 Sports to process: {', '.join(sports)}")
        
        total_master = 0
        total_mapped = 0
        
        for sport_code in sports:
            m, mp = await self._consolidate_sport(sport_code)
            total_master += m
            total_mapped += mp
        
        await self.session.commit()
        
        # Final report
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ ODDS CONSOLIDATION COMPLETE")
        logger.info(f"   Raw odds processed:    {total_raw:>12,}")
        logger.info(f"   Master odds created:   {total_master:>12,}")
        logger.info(f"   Odds mappings created: {total_mapped:>12,}")
        if total_master > 0:
            ratio = total_raw / total_master
            logger.info(f"   Dedup ratio:           {ratio:>12.1f}x")
        logger.info("=" * 60)
        
        return {
            "raw": total_raw,
            "master": total_master,
            "mappings": total_mapped,
        }
    
    async def _consolidate_sport(self, sport_code: str) -> tuple:
        """Process all odds for one sport. Returns (master_count, mapping_count)."""
        logger.info(f"\n🏟️  Processing {sport_code}...")
        
        # Get all master_games for this sport
        games_result = await self.session.execute(text("""
            SELECT mg.id
            FROM master_games mg
            WHERE mg.sport_code = :sport
            ORDER BY mg.scheduled_at
        """), {"sport": sport_code})
        game_ids = [str(r[0]) for r in games_result.fetchall()]
        logger.info(f"   {len(game_ids)} master games")
        
        master_count = 0
        mapping_count = 0
        batch_size = 500
        
        for batch_start in range(0, len(game_ids), batch_size):
            batch = game_ids[batch_start:batch_start + batch_size]
            
            for master_game_id in batch:
                m, mp = await self._consolidate_game_odds(master_game_id)
                master_count += m
                mapping_count += mp
            
            await self.session.commit()
            
            if (batch_start + batch_size) % 2000 == 0:
                logger.info(f"   ... processed {batch_start + batch_size}/{len(game_ids)} games, "
                            f"{master_count} master odds so far")
        
        logger.info(f"   ✅ {sport_code}: {master_count:,} master odds, {mapping_count:,} mappings")
        return master_count, mapping_count
    
    async def _consolidate_game_odds(self, master_game_id: str) -> tuple:
        """
        Consolidate all raw odds for a single master_game.
        Returns (master_count, mapping_count).
        """
        # Get all raw odds for this master game
        result = await self.session.execute(text("""
            SELECT o.id, o.sportsbook_key, o.bet_type,
                   o.home_line, o.away_line, o.home_odds, o.away_odds,
                   o.total, o.over_odds, o.under_odds,
                   o.is_opening, o.recorded_at,
                   s.is_sharp, s.id as sportsbook_id
            FROM odds o
            LEFT JOIN sportsbooks s ON o.sportsbook_id = s.id
            WHERE o.master_game_id = :mgid
            ORDER BY o.recorded_at ASC
        """), {"mgid": master_game_id})
        rows = result.fetchall()
        
        if not rows:
            return 0, 0
        
        # Group by (sportsbook_key, bet_type)
        groups = {}
        for r in rows:
            book_key = r[1] or "unknown"
            bet_type = r[2] or "unknown"
            key = (book_key, bet_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        master_count = 0
        mapping_count = 0
        
        for (book_key, bet_type), odds_rows in groups.items():
            # Sort by recorded_at
            odds_rows.sort(key=lambda x: x[11] or datetime.min)
            
            first = odds_rows[0]
            last = odds_rows[-1]
            
            # Identify opener
            opener = None
            for r in odds_rows:
                if r[10]:  # is_opening
                    opener = r
                    break
            if not opener:
                opener = first
            
            # Determine sharp status
            is_sharp = any(r[12] for r in odds_rows) or book_key.lower() in SHARP_BOOK_ALIASES
            sportsbook_id = first[13]
            
            # Build master_odds values
            opening_line = None
            closing_line = None
            opening_odds_home = None
            opening_odds_away = None
            closing_odds_home = None
            closing_odds_away = None
            opening_total = None
            closing_total = None
            opening_over_odds = None
            opening_under_odds = None
            closing_over_odds = None
            closing_under_odds = None
            line_movement = None
            no_vig = None
            
            if bet_type == "spread":
                opening_line = opener[3]
                closing_line = last[3]
                opening_odds_home = opener[5]
                opening_odds_away = opener[6]
                closing_odds_home = last[5]
                closing_odds_away = last[6]
                if opening_line is not None and closing_line is not None:
                    line_movement = round(closing_line - opening_line, 2)
            
            elif bet_type == "moneyline":
                opening_odds_home = opener[5]
                opening_odds_away = opener[6]
                closing_odds_home = last[5]
                closing_odds_away = last[6]
                if closing_odds_home and closing_odds_away:
                    no_vig = devig_two_way(closing_odds_home, closing_odds_away)
            
            elif bet_type == "total":
                opening_total = opener[7]
                closing_total = last[7]
                opening_over_odds = opener[8]
                opening_under_odds = opener[9]
                closing_over_odds = last[8]
                closing_under_odds = last[9]
                if opening_total is not None and closing_total is not None:
                    line_movement = round(closing_total - opening_total, 2)
            
            first_seen = opener[11]
            last_seen = last[11]
            
            # INSERT master_odds
            mo_id = uuid4()
            await self.session.execute(text("""
                INSERT INTO master_odds (
                    id, master_game_id, sportsbook_key, sportsbook_id, bet_type, period,
                    opening_line, closing_line,
                    opening_odds_home, opening_odds_away, closing_odds_home, closing_odds_away,
                    opening_total, closing_total,
                    opening_over_odds, opening_under_odds, closing_over_odds, closing_under_odds,
                    line_movement, no_vig_prob_home,
                    is_sharp, num_source_records, first_seen_at, last_seen_at
                ) VALUES (
                    :id, :mgid, :book, :sbid, :btype, 'full',
                    :ol, :cl,
                    :ooh, :ooa, :coh, :coa,
                    :ot, :ct,
                    :oov, :oun, :cov, :cun,
                    :lm, :nv,
                    :sharp, :nsr, :fs, :ls
                )
                ON CONFLICT (master_game_id, sportsbook_key, bet_type, period)
                DO UPDATE SET
                    closing_line = EXCLUDED.closing_line,
                    closing_odds_home = EXCLUDED.closing_odds_home,
                    closing_odds_away = EXCLUDED.closing_odds_away,
                    closing_total = EXCLUDED.closing_total,
                    closing_over_odds = EXCLUDED.closing_over_odds,
                    closing_under_odds = EXCLUDED.closing_under_odds,
                    line_movement = EXCLUDED.line_movement,
                    no_vig_prob_home = EXCLUDED.no_vig_prob_home,
                    num_source_records = EXCLUDED.num_source_records,
                    last_seen_at = EXCLUDED.last_seen_at,
                    updated_at = NOW()
                RETURNING id
            """), {
                "id": str(mo_id),
                "mgid": master_game_id,
                "book": book_key,
                "sbid": str(sportsbook_id) if sportsbook_id else None,
                "btype": bet_type,
                "ol": opening_line,
                "cl": closing_line,
                "ooh": opening_odds_home,
                "ooa": opening_odds_away,
                "coh": closing_odds_home,
                "coa": closing_odds_away,
                "ot": opening_total,
                "ct": closing_total,
                "oov": opening_over_odds,
                "oun": opening_under_odds,
                "cov": closing_over_odds,
                "cun": closing_under_odds,
                "lm": line_movement,
                "nv": no_vig,
                "sharp": is_sharp,
                "nsr": len(odds_rows),
                "fs": first_seen,
                "ls": last_seen,
            })
            
            # Get actual master_odds ID
            returned = (await self.session.execute(text(
                "SELECT id FROM master_odds WHERE master_game_id = :mgid "
                "AND sportsbook_key = :book AND bet_type = :bt AND period = 'full'"
            ), {"mgid": master_game_id, "book": book_key, "bt": bet_type})).fetchone()
            actual_mo_id = str(returned[0]) if returned else str(mo_id)
            
            master_count += 1
            
            # Create odds_mappings
            for r in odds_rows:
                raw_id = str(r[0])
                try:
                    await self.session.execute(text("""
                        INSERT INTO odds_mappings (id, master_odds_id, source_key, source_odds_db_id)
                        VALUES (:id, :moid, :src, :rawid)
                        ON CONFLICT DO NOTHING
                    """), {
                        "id": str(uuid4()),
                        "moid": actual_mo_id,
                        "src": book_key,
                        "rawid": raw_id,
                    })
                    mapping_count += 1
                except Exception:
                    pass
        
        return master_count, mapping_count


async def consolidate_odds(session: AsyncSession) -> dict:
    """Convenience function to run odds consolidation."""
    service = OddsConsolidationService(session)
    return await service.consolidate_all()
