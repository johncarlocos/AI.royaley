"""
ROYALEY - Fix remaining team mappings + public_betting backfill

Issues:
  1. Teams mapped: 2,953 / 27,343 (10.8%) — many source team records are unmapped
     because fix scripts only created mappings for DISTINCT names, not all source rows.
  2. Public betting: 0 / 118 — needs master_game_id backfill.

Run: python -m scripts.fix_teams_and_betting
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


async def fix_all():
    logger.info("Initializing database connection...")
    await db_manager.initialize()
    logger.info("Database connection initialized successfully")

    async with db_manager.session() as session:

        # =================================================================
        # FIX 1: Map ALL remaining source teams → master_teams
        # =================================================================
        logger.info(f"\n{'=' * 60}")
        logger.info("FIX 1: Mapping ALL unmapped source teams")
        logger.info(f"{'=' * 60}")

        total_mapped = 0
        total_no_match = 0

        for sport_code, sport_id in ALL_SPORTS.items():
            # Get all unmapped source teams for this sport
            result = await session.execute(text("""
                SELECT t.id, t.name, t.external_id
                FROM teams t
                WHERE t.sport_id = :sid
                AND t.master_team_id IS NULL
                AND t.name IS NOT NULL
            """), {"sid": sport_id})
            unmapped = result.fetchall()

            if not unmapped:
                continue

            logger.info(f"\n  {sport_code}: {len(unmapped)} unmapped source teams")
            mapped = 0
            no_match = 0

            for row in unmapped:
                team_id, team_name, ext_id = str(row[0]), row[1], row[2]

                if not team_name:
                    continue

                # Find master_team by exact name match (case-insensitive)
                mt_result = await session.execute(text("""
                    SELECT id FROM master_teams
                    WHERE sport_code = :sport
                    AND LOWER(canonical_name) = LOWER(:name)
                    LIMIT 1
                """), {"sport": sport_code, "name": team_name.strip()})
                mt_row = mt_result.fetchone()

                if not mt_row:
                    # Try abbreviation match
                    if team_name and len(team_name) <= 10:
                        mt_result = await session.execute(text("""
                            SELECT id FROM master_teams
                            WHERE sport_code = :sport
                            AND UPPER(abbreviation) = UPPER(:name)
                            LIMIT 1
                        """), {"sport": sport_code, "name": team_name.strip()})
                        mt_row = mt_result.fetchone()

                if not mt_row:
                    no_match += 1
                    continue

                master_team_id = str(mt_row[0])

                # Update source team
                await session.execute(text("""
                    UPDATE teams SET master_team_id = :mtid WHERE id = :tid
                """), {"mtid": master_team_id, "tid": team_id})

                # Create team_mapping (skip if exists)
                await session.execute(text("""
                    INSERT INTO team_mappings (id, master_team_id, source_key,
                        source_team_name, source_external_id, source_team_db_id,
                        confidence, verified)
                    VALUES (gen_random_uuid(), :mtid, :src, :sname, :extid, :stid, 1.0, true)
                    ON CONFLICT DO NOTHING
                """), {
                    "mtid": master_team_id, "src": "auto_fix_v2",
                    "sname": team_name, "extid": ext_id or "",
                    "stid": team_id,
                })
                mapped += 1

                if mapped % 2000 == 0:
                    await session.commit()

            await session.commit()
            total_mapped += mapped
            total_no_match += no_match

            if no_match > 0:
                logger.info(f"  ✅ {sport_code}: {mapped:,} mapped, {no_match:,} no master match")
            else:
                logger.info(f"  ✅ {sport_code}: {mapped:,} mapped")

        logger.info(f"\n  TOTAL: {total_mapped:,} newly mapped, {total_no_match:,} no match")

        # For the ones that had no match, these are likely teams from odds
        # sources (e.g. The Odds API) with different naming. Let's auto-create
        # master_teams for them too.
        if total_no_match > 0:
            logger.info(f"\n  Creating master_teams for {total_no_match} unmatched teams...")
            created = 0
            for sport_code, sport_id in ALL_SPORTS.items():
                result = await session.execute(text("""
                    SELECT t.id, t.name, t.abbreviation, t.city, t.conference, t.division, t.external_id
                    FROM teams t
                    WHERE t.sport_id = :sid
                    AND t.master_team_id IS NULL
                    AND t.name IS NOT NULL AND t.name != ''
                """), {"sid": sport_id})
                still_unmapped = result.fetchall()

                for row in still_unmapped:
                    team_id, name, abbr, city, conf, div, ext_id = (
                        str(row[0]), row[1], row[2], row[3], row[4], row[5], row[6]
                    )

                    # Create master_team
                    mt_result = await session.execute(text("""
                        INSERT INTO master_teams (id, sport_code, canonical_name, abbreviation,
                            city, conference, division, is_active)
                        VALUES (gen_random_uuid(), :sport, :name, :abbr, :city, :conf, :div, true)
                        ON CONFLICT ON CONSTRAINT uq_master_teams_sport_name DO NOTHING
                        RETURNING id
                    """), {
                        "sport": sport_code, "name": name.strip(),
                        "abbr": abbr or "", "city": city or "",
                        "conf": conf or "", "div": div or "",
                    })
                    mt_row = mt_result.fetchone()

                    if mt_row:
                        master_team_id = str(mt_row[0])
                        created += 1
                    else:
                        # Already exists (created by another row with same name), get it
                        existing = await session.execute(text("""
                            SELECT id FROM master_teams
                            WHERE sport_code = :sport AND LOWER(canonical_name) = LOWER(:name)
                            LIMIT 1
                        """), {"sport": sport_code, "name": name.strip()})
                        ex_row = existing.fetchone()
                        if ex_row:
                            master_team_id = str(ex_row[0])
                        else:
                            continue

                    # Map it
                    await session.execute(text("""
                        UPDATE teams SET master_team_id = :mtid WHERE id = :tid
                    """), {"mtid": master_team_id, "tid": team_id})

                    await session.execute(text("""
                        INSERT INTO team_mappings (id, master_team_id, source_key,
                            source_team_name, source_external_id, source_team_db_id,
                            confidence, verified)
                        VALUES (gen_random_uuid(), :mtid, :src, :sname, :extid, :stid, 0.9, true)
                        ON CONFLICT DO NOTHING
                    """), {
                        "mtid": master_team_id, "src": "auto_fix_v2",
                        "sname": name, "extid": ext_id or "",
                        "stid": team_id,
                    })

                if still_unmapped:
                    await session.commit()

            await session.commit()
            logger.info(f"  ✅ Created {created:,} new master_teams for previously unmatched teams")

        # Final team count
        r = await session.execute(text("SELECT COUNT(*) FROM teams WHERE master_team_id IS NULL"))
        remaining = r.scalar()
        r = await session.execute(text("SELECT COUNT(*) FROM teams"))
        total_teams = r.scalar()
        r = await session.execute(text("SELECT COUNT(*) FROM team_mappings"))
        mapping_count = r.scalar()
        logger.info(f"\n  Final: {total_teams - remaining:,} / {total_teams:,} source teams mapped ({remaining:,} remaining)")
        logger.info(f"  Team mappings: {mapping_count:,}")

        # =================================================================
        # FIX 2: Backfill public_betting.master_game_id
        # =================================================================
        logger.info(f"\n{'=' * 60}")
        logger.info("FIX 2: Backfilling public_betting master_game_id")
        logger.info(f"{'=' * 60}")

        # First check the public_betting table structure
        try:
            r = await session.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'public_betting'
                ORDER BY ordinal_position
            """))
            cols = [row[0] for row in r.fetchall()]
            logger.info(f"  public_betting columns: {', '.join(cols)}")

            if 'game_id' in cols and 'master_game_id' in cols:
                result = await session.execute(text("""
                    UPDATE public_betting SET master_game_id = g.master_game_id
                    FROM games g
                    WHERE public_betting.game_id = g.id
                    AND g.master_game_id IS NOT NULL
                    AND public_betting.master_game_id IS NULL
                """))
                backfilled = result.rowcount
                await session.commit()
                logger.info(f"  ✅ Public betting backfilled: {backfilled:,} / 118")
            elif 'game_id' in cols:
                # master_game_id column might not exist yet
                logger.info(f"  ⚠️  public_betting has game_id but no master_game_id column")
                logger.info(f"  Adding master_game_id column...")
                await session.execute(text("""
                    ALTER TABLE public_betting ADD COLUMN IF NOT EXISTS master_game_id UUID
                    REFERENCES master_games(id) ON DELETE SET NULL
                """))
                await session.commit()

                result = await session.execute(text("""
                    UPDATE public_betting SET master_game_id = g.master_game_id
                    FROM games g
                    WHERE public_betting.game_id = g.id
                    AND g.master_game_id IS NOT NULL
                    AND public_betting.master_game_id IS NULL
                """))
                backfilled = result.rowcount
                await session.commit()
                logger.info(f"  ✅ Public betting backfilled: {backfilled:,} / 118")
            else:
                logger.info(f"  ⚠️  public_betting has no game_id column — manual linking needed")
        except Exception as e:
            logger.error(f"  ❌ Public betting fix failed: {e}")
            await session.rollback()

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ FIXES COMPLETE")
        logger.info(f"{'=' * 60}")

        r = await session.execute(text("SELECT COUNT(*) FROM master_teams"))
        logger.info(f"  master_teams: {r.scalar():,}")
        r = await session.execute(text("SELECT COUNT(*) FROM team_mappings"))
        logger.info(f"  team_mappings: {r.scalar():,}")


if __name__ == "__main__":
    asyncio.run(fix_all())