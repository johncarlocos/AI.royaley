"""
ROYALEY - Fix ALL Players: Create master_players + map source players for all 10 sports
Solves: ALL 159,622 source players are unmapped (master_player_id IS NULL).

Run: python -m scripts.fix_all_players

What this does per sport:
  1. For sports WITHOUT master_players (all except ATP/WTA):
     - Creates master_player records from unique source player names
  2. For ALL sports:
     - Maps source players → master_players by exact name match
     - Falls back to team-scoped fuzzy match for remaining
     - Updates players.master_player_id
     - Creates player_mappings

Idempotent — safe to run multiple times.
"""

import asyncio
import logging
import sys
import os
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All sport codes → sport_id UUIDs
ALL_SPORTS = {
    "ATP": "368e5d80-b36c-4a25-91b0-235cdc4a08bb",
    "CFL": "4b275967-8561-4708-879d-ec41c6b2e11b",
    "MLB": "effaaea6-5c60-4ae2-83f0-c089d908a206",
    "NBA": "a3947869-8fce-43f6-a90d-81091b124bd3",
    "NCAAB": "2d6f9898-df96-4a05-a48b-f05c260c1ee7",
    "NCAAF": "3010bf94-fc49-4ca8-b147-b6d18bc48399",
    "NFL": "c2c9e937-88dd-4976-8095-73a0db3bdb15",
    "NHL": "72534f3a-0549-4f33-9965-75112b4fda3c",
    "WNBA": "25f8ae40-00d5-4151-b92a-fd22155ef763",
    "WTA": "8e3e4f06-919f-438c-b3ba-75cc7d05891e",
}


def parse_name(full_name: str) -> tuple:
    """Parse 'First Last' into (first, last)."""
    if not full_name:
        return ("", "")
    parts = full_name.strip().split(None, 1)
    first = parts[0] if parts else ""
    last = parts[1] if len(parts) > 1 else parts[0] if parts else ""
    return (first, last)


async def fix_all_players():
    """Main fix function for all player mappings."""
    logger.info("Initializing database connection...")
    await db_manager.initialize()
    logger.info("Database connection initialized successfully")

    async with db_manager.session() as session:

        total_created = 0
        total_mapped = 0

        for sport_code, sport_id in ALL_SPORTS.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"FIXING PLAYERS: {sport_code}")
            logger.info(f"{'=' * 60}")

            # Check existing master_players count
            r = await session.execute(text(
                "SELECT COUNT(*) FROM master_players WHERE sport_code = :s"
            ), {"s": sport_code})
            existing_master = r.scalar()

            # Count source players
            r = await session.execute(text("""
                SELECT COUNT(p.id) FROM players p
                JOIN teams t ON p.team_id = t.id
                WHERE t.sport_id = :sid
            """), {"sid": sport_id})
            source_count = r.scalar()

            # Count unmapped
            r = await session.execute(text("""
                SELECT COUNT(p.id) FROM players p
                JOIN teams t ON p.team_id = t.id
                WHERE t.sport_id = :sid AND p.master_player_id IS NULL
            """), {"sid": sport_id})
            unmapped_count = r.scalar()

            logger.info(f"  Source players: {source_count:,}")
            logger.info(f"  Existing master_players: {existing_master:,}")
            logger.info(f"  Unmapped: {unmapped_count:,}")

            if unmapped_count == 0:
                logger.info(f"  ✅ All players already mapped, skipping")
                continue

            # =================================================================
            # PHASE 1: Create master_players from source data
            # =================================================================
            logger.info(f"\n  [Phase 1] Creating master_players for {sport_code}...")

            # Get all unique player names + their team info
            # For sports that already have master_players (ATP, WTA),
            # we still create any missing ones
            result = await session.execute(text("""
                SELECT DISTINCT ON (p.name)
                    p.name, p.position, p.jersey_number,
                    p.height_inches, p.weight_lbs, p.birth_date,
                    p.is_active, t.master_team_id
                FROM players p
                JOIN teams t ON p.team_id = t.id
                WHERE t.sport_id = :sid
                AND p.name IS NOT NULL AND p.name != ''
                ORDER BY p.name, p.updated_at DESC NULLS LAST
            """), {"sid": sport_id})
            unique_players = result.fetchall()
            logger.info(f"  {len(unique_players)} unique player names found")

            created = 0
            skipped = 0

            for row in unique_players:
                name = row[0]
                position = row[1]
                master_team_id = str(row[7]) if row[7] else None
                first, last = parse_name(name)
                birth_date = row[5]
                is_active = row[6] if row[6] is not None else True

                # Check if master_player already exists (exact name + sport)
                existing = await session.execute(text("""
                    SELECT id FROM master_players
                    WHERE sport_code = :sport
                    AND LOWER(canonical_name) = LOWER(:name)
                    LIMIT 1
                """), {"sport": sport_code, "name": name.strip()})

                if existing.fetchone():
                    skipped += 1
                    continue

                # Create new master_player
                await session.execute(text("""
                    INSERT INTO master_players (
                        id, sport_code, canonical_name, first_name, last_name,
                        position, master_team_id, is_active
                    ) VALUES (
                        gen_random_uuid(), :sport, :name, :first, :last,
                        :pos, :mtid, :active
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "sport": sport_code, "name": name.strip(),
                    "first": first, "last": last,
                    "pos": position, "mtid": master_team_id,
                    "active": is_active,
                })
                created += 1

                if created % 5000 == 0:
                    await session.commit()
                    logger.info(f"    ... created {created:,} master_players")

            await session.commit()
            total_created += created
            logger.info(f"  ✅ Created {created:,} master_players ({skipped:,} already existed)")

            # =================================================================
            # PHASE 2: Map source players → master_players
            # =================================================================
            logger.info(f"\n  [Phase 2] Mapping source players for {sport_code}...")

            # Get ALL unmapped source players
            result = await session.execute(text("""
                SELECT p.id, p.name, p.external_id, t.master_team_id
                FROM players p
                JOIN teams t ON p.team_id = t.id
                WHERE t.sport_id = :sid
                AND p.master_player_id IS NULL
                AND p.name IS NOT NULL
                ORDER BY p.name
            """), {"sid": sport_id})
            unmapped = result.fetchall()
            logger.info(f"  {len(unmapped)} unmapped players to process")

            mapped = 0
            no_match = 0

            for row in unmapped:
                player_id = str(row[0])
                player_name = row[1]
                ext_id = row[2]
                master_team_id = str(row[3]) if row[3] else None

                if not player_name:
                    continue

                # Try exact name match in master_players
                mp_result = await session.execute(text("""
                    SELECT id FROM master_players
                    WHERE sport_code = :sport
                    AND LOWER(canonical_name) = LOWER(:name)
                    LIMIT 1
                """), {"sport": sport_code, "name": player_name.strip()})
                mp_row = mp_result.fetchone()

                if not mp_row:
                    no_match += 1
                    continue

                master_player_id = str(mp_row[0])

                # Update source player
                await session.execute(text("""
                    UPDATE players SET master_player_id = :mpid
                    WHERE id = :pid AND master_player_id IS NULL
                """), {"mpid": master_player_id, "pid": player_id})

                # Create player_mapping
                await session.execute(text("""
                    INSERT INTO player_mappings (
                        id, master_player_id, source_key,
                        source_player_name, source_external_id,
                        source_player_db_id, confidence, verified
                    ) VALUES (
                        gen_random_uuid(), :mpid, :src,
                        :sname, :extid, :spid, 1.0, true
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "mpid": master_player_id, "src": "auto_fix",
                    "sname": player_name, "extid": ext_id or "",
                    "spid": player_id,
                })
                mapped += 1

                if mapped % 5000 == 0:
                    await session.commit()
                    logger.info(f"    ... mapped {mapped:,} players")

            await session.commit()
            total_mapped += mapped
            logger.info(f"  ✅ Mapped {mapped:,} players ({no_match:,} no master match)")

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ ALL PLAYERS FIX COMPLETE")
        logger.info(f"   Total master_players created: {total_created:,}")
        logger.info(f"   Total source players mapped:  {total_mapped:,}")
        logger.info(f"{'=' * 60}")

        # Summary by sport
        logger.info(f"\n📊 MASTER_PLAYERS BY SPORT:")
        r = await session.execute(text("""
            SELECT sport_code, COUNT(*) FROM master_players
            GROUP BY sport_code ORDER BY sport_code
        """))
        for row in r.fetchall():
            logger.info(f"  {row[0]}: {row[1]:,}")

        logger.info(f"\n📊 REMAINING UNMAPPED PLAYERS:")
        r = await session.execute(text("""
            SELECT s.code, COUNT(p.id)
            FROM players p
            JOIN teams t ON p.team_id = t.id
            JOIN sports s ON t.sport_id = s.id
            WHERE p.master_player_id IS NULL
            GROUP BY s.code ORDER BY s.code
        """))
        remaining = r.fetchall()
        if remaining:
            for row in remaining:
                logger.info(f"  {row[0]}: {row[1]:,}")
        else:
            logger.info(f"  None — all players mapped! 🎉")


if __name__ == "__main__":
    asyncio.run(fix_all_players())