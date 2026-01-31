"""
ROYALEY - Master Data Verification
Run after auto_map_existing_data.py to verify everything is linked correctly.

Run: python -m scripts.verify_master_data
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logger = logging.getLogger(__name__)


async def verify():
    await db_manager.initialize()

    async with db_manager.session() as session:
        print("\n" + "=" * 70)
        print("ROYALEY — Master Data Verification Report")
        print("=" * 70)

        # 1. Table counts
        print("\n📊 TABLE COUNTS:")
        tables = [
            "master_teams", "master_players", "master_games",
            "team_mappings", "player_mappings", "game_mappings",
            "venue_mappings", "source_registry", "mapping_audit_log",
        ]
        for t in tables:
            row = await session.execute(text(f"SELECT COUNT(*) FROM {t}"))
            print(f"  {t:30s} {row.scalar():>10,}")

        # 2. Mapping coverage
        print("\n📈 MAPPING COVERAGE:")
        row = await session.execute(text(
            "SELECT COUNT(*) FROM teams WHERE master_team_id IS NOT NULL"
        ))
        mapped_teams = row.scalar()
        row = await session.execute(text("SELECT COUNT(*) FROM teams"))
        total_teams = row.scalar()
        print(f"  Teams mapped:      {mapped_teams:,} / {total_teams:,} ({100*mapped_teams/max(total_teams,1):.1f}%)")

        row = await session.execute(text(
            "SELECT COUNT(*) FROM games WHERE master_game_id IS NOT NULL"
        ))
        mapped_games = row.scalar()
        row = await session.execute(text("SELECT COUNT(*) FROM games"))
        total_games = row.scalar()
        print(f"  Games mapped:      {mapped_games:,} / {total_games:,} ({100*mapped_games/max(total_games,1):.1f}%)")

        row = await session.execute(text(
            "SELECT COUNT(*) FROM odds WHERE master_game_id IS NOT NULL"
        ))
        mapped_odds = row.scalar()
        row = await session.execute(text("SELECT COUNT(*) FROM odds"))
        total_odds = row.scalar()
        print(f"  Odds linked:       {mapped_odds:,} / {total_odds:,} ({100*mapped_odds/max(total_odds,1):.1f}%)")

        row = await session.execute(text(
            "SELECT COUNT(*) FROM public_betting WHERE master_game_id IS NOT NULL"
        ))
        mapped_pb = row.scalar()
        row = await session.execute(text("SELECT COUNT(*) FROM public_betting"))
        total_pb = row.scalar()
        print(f"  Public betting:    {mapped_pb:,} / {total_pb:,} ({100*mapped_pb/max(total_pb,1):.1f}%)")

        # 3. Deduplication ratio
        print("\n🔗 DEDUPLICATION:")
        print(f"  Source games:      {total_games:,}")
        row = await session.execute(text("SELECT COUNT(*) FROM master_games"))
        master_count = row.scalar()
        print(f"  Master games:      {master_count:,}")
        if total_games > 0:
            ratio = total_games / max(master_count, 1)
            print(f"  Dedup ratio:       {ratio:.2f}x ({total_games - master_count:,} duplicates removed)")

        # 4. Per-sport odds linkage (the critical metric)
        print("\n🎯 ODDS LINKAGE BY SPORT:")
        rows = await session.execute(text("""
            SELECT s.code,
                   COUNT(o.id) as total_odds,
                   COUNT(o.master_game_id) as linked_odds,
                   ROUND(100.0 * COUNT(o.master_game_id) / NULLIF(COUNT(o.id), 0), 1) as pct
            FROM odds o
            JOIN games g ON o.game_id = g.id
            JOIN sports s ON g.sport_id = s.id
            GROUP BY s.code
            ORDER BY COUNT(o.id) DESC
        """))
        for row in rows.fetchall():
            sport, total, linked, pct = row
            status = "✅" if (pct or 0) > 80 else "⚠️" if (pct or 0) > 50 else "❌"
            print(f"  {status} {sport:8s} {linked:>10,} / {total:>10,} odds linked ({pct or 0}%)")

        # 5. Sportsbook priority check
        print("\n📚 SPORTSBOOK PRIORITIES:")
        rows = await session.execute(text(
            "SELECT name, key, is_sharp, priority FROM sportsbooks ORDER BY priority"
        ))
        for row in rows.fetchall():
            name, key, is_sharp, pri = row
            sharp_tag = " ⭐ SHARP" if is_sharp else ""
            print(f"  Priority {pri:3d}: {name:25s} ({key}){sharp_tag}")

        # 6. Master game feature readiness
        print("\n🤖 ML READINESS (features available per master_game):")
        row = await session.execute(text("""
            SELECT
                COUNT(*) as total_master_games,
                COUNT(CASE WHEN has_odds THEN 1 END) as with_odds,
                COUNT(CASE WHEN has_score THEN 1 END) as with_scores
            FROM (
                SELECT mg.id,
                    EXISTS(SELECT 1 FROM odds o WHERE o.master_game_id = mg.id) as has_odds,
                    (mg.home_score IS NOT NULL) as has_score
                FROM master_games mg
            ) sub
        """))
        r = row.fetchone()
        if r:
            total_mg, with_odds, with_scores = r
            print(f"  Total master games: {total_mg:,}")
            print(f"  With odds:          {with_odds:,} ({100*with_odds/max(total_mg,1):.1f}%)")
            print(f"  With final scores:  {with_scores:,} ({100*with_scores/max(total_mg,1):.1f}%)")
            trainable = min(with_odds, with_scores)
            print(f"  🏆 ML-trainable:    {trainable:,} games (have both odds + scores)")

        print("\n" + "=" * 70)
        print("Verification complete!")
        print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(verify())
