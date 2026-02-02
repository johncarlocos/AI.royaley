"""
ROYALEY - Fix NCAAF/NCAAB: Auto-create master_teams + map games + backfill odds
Solves the gap where college sports had 0 master_games because no master_teams existed.

Run: python -m scripts.fix_college_sports

What this does:
  1. Reads all unique NCAAF/NCAAB teams from the `teams` table (via sport_id FK)
  2. Creates master_team records for each unique team name
  3. Creates team_mappings linking source → master
  4. Creates master_games (deduplication) for NCAAF/NCAAB
  5. Backfills master_game_id on games + odds tables
  6. Runs odds consolidation for NCAAF/NCAAB

Idempotent — safe to run multiple times.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sport IDs from the sports table
COLLEGE_SPORTS = {
    "NCAAF": "3010bf94-fc49-4ca8-b147-b6d18bc48399",
    "NCAAB": "2d6f9898-df96-4a05-a48b-f05c260c1ee7",
}

SHARP_BOOKS = {
    "pinnacle", "bookmaker", "betcris", "circa", "cris",
    "pinnacle_direct", "pinnacle_api", "circasports",
}


def american_to_implied(odds):
    if not odds or odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def devig_two_way(home_odds, away_odds):
    if not home_odds or not away_odds:
        return None
    p_home = american_to_implied(home_odds)
    p_away = american_to_implied(away_odds)
    total = p_home + p_away
    if total == 0:
        return None
    return round(p_home / total, 6)


async def fix_college():
    """Main fix function for NCAAF and NCAAB."""
    logger.info("Initializing database connection...")
    await db_manager.initialize()
    logger.info("Database connection initialized successfully")

    async with db_manager.session() as session:

        for sport_code, sport_id in COLLEGE_SPORTS.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"FIXING {sport_code} (sport_id={sport_id})")
            logger.info(f"{'=' * 60}")

            # =================================================================
            # PHASE 1: Create master_teams from source teams
            # =================================================================
            logger.info(f"\n[Phase 1] Creating master_teams for {sport_code}...")

            result = await session.execute(text("""
                SELECT DISTINCT t.name, t.abbreviation, t.city, t.conference, t.division
                FROM teams t
                WHERE t.sport_id = :sport_id
                AND t.name IS NOT NULL
                AND t.name != ''
                ORDER BY t.name
            """), {"sport_id": sport_id})
            source_teams = result.fetchall()
            logger.info(f"  Found {len(source_teams)} unique {sport_code} teams in source data")

            teams_created = 0
            teams_skipped = 0
            for row in source_teams:
                name, abbr, city, conf, div = row[0], row[1], row[2], row[3], row[4]

                if not name or len(name) < 2:
                    continue

                result = await session.execute(text("""
                    INSERT INTO master_teams (id, sport_code, canonical_name, abbreviation,
                        city, conference, division, is_active)
                    VALUES (gen_random_uuid(), :sport, :name, :abbr, :city, :conf, :div, true)
                    ON CONFLICT ON CONSTRAINT uq_master_teams_sport_name DO NOTHING
                    RETURNING id
                """), {
                    "sport": sport_code, "name": name, "abbr": abbr or "",
                    "city": city or "", "conf": conf or "", "div": div or "",
                })
                if result.fetchone():
                    teams_created += 1
                else:
                    teams_skipped += 1

            await session.commit()
            logger.info(f"  ✅ Created {teams_created} master_teams ({teams_skipped} already existed)")

            # =================================================================
            # PHASE 2: Create team_mappings (source → master)
            # =================================================================
            logger.info(f"\n[Phase 2] Creating team_mappings for {sport_code}...")

            # Get all source teams for this sport that don't have master_team_id set
            result = await session.execute(text("""
                SELECT t.id, t.name, t.external_id
                FROM teams t
                WHERE t.sport_id = :sport_id
                AND t.name IS NOT NULL
            """), {"sport_id": sport_id})
            all_source_teams = result.fetchall()
            logger.info(f"  {len(all_source_teams)} total source teams")

            mapped_count = 0
            already_mapped = 0
            unmatched = 0

            for row in all_source_teams:
                team_id, team_name, ext_id = str(row[0]), row[1], row[2]

                if not team_name:
                    continue

                # Find master_team by exact name match
                mt_result = await session.execute(text("""
                    SELECT id FROM master_teams
                    WHERE sport_code = :sport AND canonical_name = :name
                    LIMIT 1
                """), {"sport": sport_code, "name": team_name})
                mt_row = mt_result.fetchone()

                if not mt_row:
                    unmatched += 1
                    continue

                master_team_id = str(mt_row[0])

                # Update source team's master_team_id
                await session.execute(text("""
                    UPDATE teams SET master_team_id = :mtid WHERE id = :tid
                    AND (master_team_id IS NULL OR master_team_id != :mtid)
                """), {"mtid": master_team_id, "tid": team_id})

                # Create team_mapping (skip if exists)
                result = await session.execute(text("""
                    INSERT INTO team_mappings (id, master_team_id, source_key,
                        source_team_name, source_external_id, source_team_db_id,
                        confidence, verified)
                    VALUES (gen_random_uuid(), :mtid, :src, :sname, :extid, :stid,
                        1.0, true)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """), {
                    "mtid": master_team_id, "src": "auto_fix",
                    "sname": team_name, "extid": ext_id or "",
                    "stid": team_id,
                })
                if result.fetchone():
                    mapped_count += 1
                else:
                    already_mapped += 1

            await session.commit()
            logger.info(f"  ✅ Created {mapped_count} team_mappings ({already_mapped} existed, {unmatched} unmatched)")

            # =================================================================
            # PHASE 3: Create master_games (deduplication)
            # =================================================================
            logger.info(f"\n[Phase 3] Creating master_games for {sport_code}...")

            # Get all source games for this sport
            result = await session.execute(text("""
                SELECT g.id, g.external_id, g.home_team_id, g.away_team_id,
                       g.scheduled_at, g.status, g.home_score, g.away_score,
                       g.season_id, g.venue_id, g.master_game_id,
                       ht.master_team_id as home_mt_id,
                       at_.master_team_id as away_mt_id
                FROM games g
                LEFT JOIN teams ht ON g.home_team_id = ht.id
                LEFT JOIN teams at_ ON g.away_team_id = at_.id
                WHERE g.sport_id = :sport_id
                ORDER BY g.scheduled_at, g.home_team_id
            """), {"sport_id": sport_id})
            source_games = result.fetchall()
            logger.info(f"  {len(source_games)} source games to process")

            # Count how many already have master_game_id
            already_linked = sum(1 for g in source_games if g[10] is not None)
            logger.info(f"  {already_linked} already linked, {len(source_games) - already_linked} need processing")

            # Group by (home_master_team, away_master_team, date) for dedup
            game_groups = {}
            skipped_no_teams = 0
            skipped_no_date = 0

            for g in source_games:
                home_mt = str(g[11]) if g[11] else None
                away_mt = str(g[12]) if g[12] else None
                sched = g[4]

                if not home_mt or not away_mt:
                    skipped_no_teams += 1
                    continue
                if not sched:
                    skipped_no_date += 1
                    continue

                game_date = sched.date() if hasattr(sched, 'date') else sched
                key = (home_mt, away_mt, str(game_date))

                if key not in game_groups:
                    game_groups[key] = []
                game_groups[key].append(g)

            logger.info(f"  {len(game_groups)} unique games identified ({skipped_no_teams} skipped: no team mapping, {skipped_no_date} skipped: no date)")

            games_created = 0
            games_existed = 0
            games_linked = 0
            batch_count = 0

            for key, group in game_groups.items():
                home_mt, away_mt, _ = key

                # Pick best source game (prefer final/completed with scores)
                best = sorted(group, key=lambda x: (
                    1 if x[5] and x[5].lower() in ('final', 'completed') else 0,
                    1 if x[6] is not None else 0,
                    1 if x[4] is not None else 0,
                ), reverse=True)[0]

                # Check if master_game already exists for these teams on this date
                existing = await session.execute(text("""
                    SELECT id FROM master_games
                    WHERE home_master_team_id = :hmt AND away_master_team_id = :amt
                    AND DATE(scheduled_at) = DATE(:sched)
                    AND sport_code = :sport
                    LIMIT 1
                """), {"hmt": home_mt, "amt": away_mt, "sched": best[4], "sport": sport_code})
                existing_row = existing.fetchone()

                if existing_row:
                    master_game_id = str(existing_row[0])
                    games_existed += 1
                else:
                    master_game_id = str(uuid4())
                    status = best[5] or 'unknown'
                    if status.lower() in ('final', 'completed'):
                        status = 'final'

                    # Derive season year (integer)
                    sched_dt = best[4]
                    if sport_code == "NCAAF":
                        # NCAAF season spans Aug-Jan, use the fall year
                        season_int = sched_dt.year if sched_dt.month >= 6 else sched_dt.year - 1
                    else:
                        # NCAAB season spans Nov-Apr, use the spring year
                        season_int = sched_dt.year if sched_dt.month <= 6 else sched_dt.year + 1

                    await session.execute(text("""
                        INSERT INTO master_games (id, sport_code, season, scheduled_at,
                            home_master_team_id, away_master_team_id,
                            home_score, away_score, status,
                            venue_id, is_playoff, is_neutral_site, primary_source)
                        VALUES (:id, :sport, :season, :sched,
                            :hmt, :amt, :hs, :as_, :status,
                            :vid, false, false, 'auto_fix')
                        ON CONFLICT DO NOTHING
                    """), {
                        "id": master_game_id, "sport": sport_code,
                        "season": season_int, "sched": best[4],
                        "hmt": home_mt, "amt": away_mt,
                        "hs": best[6], "as_": best[7], "status": status,
                        "vid": str(best[9]) if best[9] else None,
                    })
                    games_created += 1

                # Link ALL source games in this group
                for g in group:
                    gid = str(g[0])

                    # Update games.master_game_id
                    await session.execute(text("""
                        UPDATE games SET master_game_id = :mgid
                        WHERE id = :gid AND (master_game_id IS NULL)
                    """), {"mgid": master_game_id, "gid": gid})

                    # Create game_mapping
                    await session.execute(text("""
                        INSERT INTO game_mappings (id, master_game_id, source_key,
                            source_game_db_id, source_external_id)
                        VALUES (gen_random_uuid(), :mgid, :src, :sgid, :extid)
                        ON CONFLICT DO NOTHING
                    """), {
                        "mgid": master_game_id, "src": "auto_fix",
                        "sgid": gid, "extid": g[1] or "",
                    })
                    games_linked += 1

                batch_count += 1
                if batch_count % 2000 == 0:
                    await session.commit()
                    logger.info(f"  ... {batch_count}/{len(game_groups)} groups, {games_created:,} created, {games_linked:,} linked")

            await session.commit()
            logger.info(f"  ✅ {sport_code}: {games_created:,} master_games created ({games_existed:,} existed), {games_linked:,} source games linked")

            # =================================================================
            # PHASE 4: Backfill master_game_id on odds
            # =================================================================
            logger.info(f"\n[Phase 4] Backfilling master_game_id on odds...")

            result = await session.execute(text("""
                UPDATE odds SET master_game_id = g.master_game_id
                FROM games g
                WHERE odds.game_id = g.id
                AND g.sport_id = :sport_id
                AND g.master_game_id IS NOT NULL
                AND odds.master_game_id IS NULL
            """), {"sport_id": sport_id})
            odds_backfilled = result.rowcount
            await session.commit()
            logger.info(f"  ✅ Odds backfilled: {odds_backfilled:,}")

            # Also try player_stats if table exists
            try:
                result = await session.execute(text("""
                    UPDATE player_stats SET master_game_id = g.master_game_id
                    FROM games g
                    WHERE player_stats.game_id = g.id
                    AND g.sport_id = :sport_id
                    AND g.master_game_id IS NOT NULL
                    AND (player_stats.master_game_id IS NULL)
                """), {"sport_id": sport_id})
                stats_backfilled = result.rowcount
                await session.commit()
                logger.info(f"  ✅ Player stats backfilled: {stats_backfilled:,}")
            except Exception as e:
                logger.info(f"  ⚠️  Player stats backfill skipped: {e}")
                await session.rollback()

            # =================================================================
            # PHASE 5: Consolidate odds into master_odds
            # =================================================================
            logger.info(f"\n[Phase 5] Consolidating odds for {sport_code}...")

            # Get master games that have linked odds
            games_result = await session.execute(text("""
                SELECT DISTINCT mg.id
                FROM master_games mg
                JOIN odds o ON o.master_game_id = mg.id
                WHERE mg.sport_code = :sport
                ORDER BY mg.id
            """), {"sport": sport_code})
            game_ids = [str(r[0]) for r in games_result.fetchall()]
            logger.info(f"  {len(game_ids)} master games with odds to consolidate")

            master_odds_count = 0
            mapping_count = 0

            for i, mgid in enumerate(game_ids):
                m, mp = await _consolidate_game_odds(session, mgid)
                master_odds_count += m
                mapping_count += mp

                if (i + 1) % 500 == 0:
                    await session.commit()
                    logger.info(f"  ... {i+1}/{len(game_ids)} games, {master_odds_count:,} master odds")

            await session.commit()
            logger.info(f"  ✅ {sport_code}: {master_odds_count:,} master odds, {mapping_count:,} mappings")

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        logger.info(f"\n{'=' * 60}")
        logger.info("✅ COLLEGE SPORTS FIX COMPLETE")
        logger.info(f"{'=' * 60}")

        for sport in ["NCAAF", "NCAAB"]:
            r = await session.execute(text(
                "SELECT COUNT(*) FROM master_teams WHERE sport_code = :s"
            ), {"s": sport})
            teams = r.scalar()
            r = await session.execute(text(
                "SELECT COUNT(*) FROM master_games WHERE sport_code = :s"
            ), {"s": sport})
            games = r.scalar()
            r = await session.execute(text("""
                SELECT COUNT(*) FROM master_odds mo
                JOIN master_games mg ON mo.master_game_id = mg.id
                WHERE mg.sport_code = :s
            """), {"s": sport})
            odds = r.scalar()
            logger.info(f"  {sport}: {teams:,} teams | {games:,} games | {odds:,} master odds")


async def _consolidate_game_odds(session, master_game_id: str) -> tuple:
    """Consolidate raw odds for one game into master_odds."""
    result = await session.execute(text("""
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
        odds_rows.sort(key=lambda x: x[11] or datetime.min)
        first = odds_rows[0]
        last = odds_rows[-1]

        # Find opener (is_opening=True or first record)
        opener = None
        for r in odds_rows:
            if r[10]:  # is_opening
                opener = r
                break
        if not opener:
            opener = first

        is_sharp = any(r[12] for r in odds_rows) or book_key.lower() in SHARP_BOOKS
        sportsbook_id = first[13]

        # Extract opening/closing values by bet_type
        opening_line = closing_line = None
        opening_odds_home = opening_odds_away = None
        closing_odds_home = closing_odds_away = None
        opening_total = closing_total = None
        opening_over_odds = opening_under_odds = None
        closing_over_odds = closing_under_odds = None
        line_movement = no_vig = None

        if bet_type == "spread":
            opening_line = opener[3]   # home_line
            closing_line = last[3]
            opening_odds_home = opener[5]
            opening_odds_away = opener[6]
            closing_odds_home = last[5]
            closing_odds_away = last[6]
            if opening_line is not None and closing_line is not None:
                try:
                    line_movement = round(float(closing_line) - float(opening_line), 2)
                except (TypeError, ValueError):
                    pass
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
                try:
                    line_movement = round(float(closing_total) - float(opening_total), 2)
                except (TypeError, ValueError):
                    pass

        mo_id = str(uuid4())
        await session.execute(text("""
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
                :ol, :cl, :ooh, :ooa, :coh, :coa,
                :ot, :ct, :oov, :oun, :cov, :cun,
                :lm, :nv, :sharp, :nsr, :fs, :ls
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
        """), {
            "id": mo_id, "mgid": master_game_id, "book": book_key,
            "sbid": str(sportsbook_id) if sportsbook_id else None,
            "btype": bet_type,
            "ol": opening_line, "cl": closing_line,
            "ooh": opening_odds_home, "ooa": opening_odds_away,
            "coh": closing_odds_home, "coa": closing_odds_away,
            "ot": opening_total, "ct": closing_total,
            "oov": opening_over_odds, "oun": opening_under_odds,
            "cov": closing_over_odds, "cun": closing_under_odds,
            "lm": line_movement, "nv": no_vig,
            "sharp": is_sharp, "nsr": len(odds_rows),
            "fs": opener[11], "ls": last[11],
        })

        # Get actual master_odds id (in case ON CONFLICT updated existing)
        returned = await session.execute(text("""
            SELECT id FROM master_odds
            WHERE master_game_id = :mgid AND sportsbook_key = :book
            AND bet_type = :bt AND period = 'full'
        """), {"mgid": master_game_id, "book": book_key, "bt": bet_type})
        actual_mo_row = returned.fetchone()
        actual_mo_id = str(actual_mo_row[0]) if actual_mo_row else mo_id

        master_count += 1

        # Create odds_mappings for each raw odds record
        for r in odds_rows:
            try:
                await session.execute(text("""
                    INSERT INTO odds_mappings (id, master_odds_id, source_key, source_odds_db_id)
                    VALUES (gen_random_uuid(), :moid, :src, :rawid)
                    ON CONFLICT DO NOTHING
                """), {
                    "moid": actual_mo_id,
                    "src": book_key, "rawid": str(r[0]),
                })
                mapping_count += 1
            except Exception:
                pass

    return master_count, mapping_count


if __name__ == "__main__":
    asyncio.run(fix_college())