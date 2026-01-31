"""
ROYALEY - Auto-Map Existing Data to Master Tables
Phase 3-5: Maps teams → master_teams, deduplicates games → master_games,
           backfills master_*_id on odds/stats/betting tables.

Run: python -m scripts.auto_map_existing_data

This is the heavy-lift script. Expected runtime: 10-30 minutes depending on DB size.
"""

import asyncio
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import timedelta
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, select, func, and_
from app.core.database import db_manager

logger = logging.getLogger(__name__)

# Common name aliases (source_name → canonical search terms)
TEAM_ALIASES = {
    # NFL nicknames (used by public_betting)
    "sooners": "Oklahoma Sooners", "crimson tide": "Alabama Crimson Tide",
    "golden eagles": "Southern Miss Golden Eagles", "bulldogs": None,  # ambiguous
    "tigers": None, "wildcats": None, "bears": None,  # ambiguous — need sport context
    # Common abbreviation differences
    "la lakers": "Los Angeles Lakers", "la clippers": "Los Angeles Clippers",
    "la rams": "Los Angeles Rams", "la chargers": "Los Angeles Chargers",
    "la dodgers": "Los Angeles Dodgers", "la angels": "Los Angeles Angels",
    "ny giants": "New York Giants", "ny jets": "New York Jets",
    "ny knicks": "New York Knicks", "ny mets": "New York Mets",
    "ny yankees": "New York Yankees", "ny rangers": "New York Rangers",
    "ny islanders": "New York Islanders", "sf 49ers": "San Francisco 49ers",
    "sf giants": "San Francisco Giants", "gb packers": "Green Bay Packers",
    "tb buccaneers": "Tampa Bay Buccaneers", "tb lightning": "Tampa Bay Lightning",
    "tb rays": "Tampa Bay Rays", "kc chiefs": "Kansas City Chiefs",
    "kc royals": "Kansas City Royals", "lv raiders": "Las Vegas Raiders",
    "ne patriots": "New England Patriots", "no saints": "New Orleans Saints",
    "stl cardinals": "St. Louis Cardinals", "stl blues": "St. Louis Blues",
    "washington football team": "Washington Commanders",
}


def normalize_name(name: str) -> str:
    """Normalize a team/player name for matching."""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)  # remove punctuation
    s = re.sub(r'\s+', ' ', s)           # collapse whitespace
    return s


def extract_source_key(external_id: str) -> str:
    """Extract source key from external_id prefix pattern."""
    if not external_id:
        return "unknown"
    # Patterns: "bdl_NFL_team_123", "espn_12345", "sportsdb_12345", "pinnacle_xxx"
    prefixes = [
        "bdl_", "espn_", "sportsdb_", "pinnacle_", "nflfastr_", "cfbfastr_",
        "baseballr_", "hockeyr_", "wehoop_", "hoopr_", "cfl_", "nhl_api_",
        "sportsipy_", "bref_", "cfbd_", "matchstat_", "realgm_", "ngs_",
        "kaggle_", "ta_",
    ]
    eid_lower = external_id.lower()
    for prefix in prefixes:
        if eid_lower.startswith(prefix):
            return prefix.rstrip("_")
    # If numeric only, likely ESPN
    if external_id.isdigit():
        return "espn"
    return "unknown"


async def phase3_map_teams(session):
    """Phase 3: Map existing teams table to master_teams."""
    print("\n" + "=" * 70)
    print("[PHASE 3] Auto-mapping existing teams → master_teams")
    print("=" * 70)

    # Get all master teams
    master_rows = await session.execute(text(
        "SELECT id, sport_code, canonical_name, abbreviation, city FROM master_teams"
    ))
    master_teams = master_rows.fetchall()

    # Build lookup indices
    by_name = {}   # (sport_code, normalized_name) → master_team_id
    by_abbr = {}   # (sport_code, abbr) → master_team_id
    by_city_name = {}  # (sport_code, "city name") → master_team_id

    for mt in master_teams:
        mid, sport, cname, abbr, city = mt
        norm = normalize_name(cname)
        by_name[(sport, norm)] = mid
        if abbr:
            by_abbr[(sport, abbr.upper())] = mid
        if city:
            # "Los Angeles Lakers" can match "Lakers"
            parts = cname.split()
            if len(parts) > 1:
                by_name[(sport, normalize_name(parts[-1]))] = mid  # just team name
                by_city_name[(sport, normalize_name(f"{city} {parts[-1]}"))] = mid

    # Get sport_code mapping (sport_id → sport_code)
    sports_rows = await session.execute(text("SELECT id, code FROM sports"))
    sport_map = {str(r[0]): r[1] for r in sports_rows.fetchall()}

    # Get all existing teams
    teams_rows = await session.execute(text(
        "SELECT id, sport_id, external_id, name, abbreviation FROM teams WHERE master_team_id IS NULL"
    ))
    teams = teams_rows.fetchall()
    print(f"  Found {len(teams)} unmapped team records")

    mapped = 0
    unmapped = []
    batch_updates = []

    for team_row in teams:
        tid, sport_id, ext_id, tname, tabbr = team_row
        sport_code = sport_map.get(str(sport_id), "")
        source_key = extract_source_key(ext_id) if ext_id else "unknown"

        if not sport_code or not tname:
            continue

        # Tennis: teams are actually players — skip for team mapping
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
            # 4. Partial match — team name might be just "Lakers" or "49ers"
            for (sc, mn), mid in by_name.items():
                if sc == sport_code and norm in mn:
                    master_id = mid
                    confidence = 0.85
                    break

        if master_id:
            batch_updates.append((str(master_id), str(tid)))
            # Create team_mapping
            await session.execute(text("""
                INSERT INTO team_mappings (id, master_team_id, source_key, source_team_name,
                    source_external_id, source_team_db_id, confidence, verified)
                VALUES (gen_random_uuid(), :mid, :src, :name, :ext, :tid, :conf, :ver)
                ON CONFLICT ON CONSTRAINT uq_team_map_source_name DO NOTHING
            """), {
                "mid": str(master_id), "src": source_key, "name": tname,
                "ext": ext_id, "tid": str(tid), "conf": confidence,
                "ver": confidence >= 1.0,
            })
            mapped += 1
        else:
            unmapped.append({
                "sport": sport_code, "name": tname, "abbr": tabbr,
                "external_id": ext_id, "source": source_key,
            })

    # Batch update teams.master_team_id
    for master_id, team_id in batch_updates:
        await session.execute(text(
            "UPDATE teams SET master_team_id = :mid WHERE id = :tid"
        ), {"mid": master_id, "tid": team_id})

    await session.commit()

    print(f"  ✅ Mapped: {mapped} teams")
    print(f"  ⚠️  Unmapped: {len(unmapped)} teams")

    # Write unmapped to CSV for manual review
    if unmapped:
        csv_path = os.path.join(os.path.dirname(__file__), "unmapped_teams.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sport", "name", "abbr", "external_id", "source"])
            writer.writeheader()
            writer.writerows(unmapped)
        print(f"  📄 Review file: {csv_path}")

    return mapped


async def phase3b_map_tennis_players(session):
    """Phase 3b: Map ATP/WTA 'teams' to master_players (tennis-specific fix)."""
    print("\n" + "=" * 70)
    print("[PHASE 3b] Mapping tennis 'teams' → master_players")
    print("=" * 70)

    sport_map_rows = await session.execute(text("SELECT id, code FROM sports WHERE code IN ('ATP', 'WTA')"))
    sport_map = {r[1]: str(r[0]) for r in sport_map_rows.fetchall()}

    created = 0
    for sport_code, sport_id in sport_map.items():
        # Get all teams for this tennis sport
        rows = await session.execute(text("""
            SELECT id, external_id, name, abbreviation
            FROM teams WHERE sport_id = :sid AND master_team_id IS NULL
        """), {"sid": sport_id})
        tennis_teams = rows.fetchall()
        print(f"  {sport_code}: {len(tennis_teams)} player-as-team records")

        for tt in tennis_teams:
            tid, ext_id, tname, tabbr = tt
            if not tname:
                continue

            source_key = extract_source_key(ext_id) if ext_id else "unknown"
            name_clean = tname.strip()

            # Parse name
            parts = name_clean.split(None, 1)
            first = parts[0] if parts else ""
            last = parts[1] if len(parts) > 1 else first

            # Insert into master_players (ON CONFLICT skip)
            await session.execute(text("""
                INSERT INTO master_players (id, sport_code, canonical_name, first_name, last_name, is_active)
                VALUES (gen_random_uuid(), :sport, :name, :first, :last, true)
                ON CONFLICT ON CONSTRAINT uq_master_players_sport_name DO NOTHING
            """), {"sport": sport_code, "name": name_clean, "first": first, "last": last})

            # Get the master_player_id
            mp_row = await session.execute(text("""
                SELECT id FROM master_players WHERE sport_code = :sport AND canonical_name = :name
            """), {"sport": sport_code, "name": name_clean})
            mp = mp_row.fetchone()
            if not mp:
                continue

            # Create player_mapping
            await session.execute(text("""
                INSERT INTO player_mappings (id, master_player_id, source_key, source_player_name,
                    source_external_id, source_player_db_id, confidence, verified)
                VALUES (gen_random_uuid(), :mpid, :src, :name, :ext, :tid, 0.8, false)
                ON CONFLICT ON CONSTRAINT uq_player_map_source_ext DO NOTHING
            """), {
                "mpid": str(mp[0]), "src": source_key, "name": name_clean,
                "ext": ext_id, "tid": str(tid),
            })
            created += 1

    await session.commit()
    print(f"  ✅ Created {created} tennis master_player records + mappings")


async def phase4_create_master_games(session):
    """Phase 4: Deduplicate games into master_games."""
    print("\n" + "=" * 70)
    print("[PHASE 4] Creating master_games (deduplication)")
    print("=" * 70)

    # Get sport_id → sport_code mapping
    sports_rows = await session.execute(text("SELECT id, code FROM sports"))
    sport_map = {str(r[0]): r[1] for r in sports_rows.fetchall()}

    # Process sport by sport
    total_masters = 0
    total_mapped = 0

    for sport_id_str, sport_code in sport_map.items():
        is_tennis = sport_code in ("ATP", "WTA")

        # Get all unmapped games for this sport
        games_rows = await session.execute(text("""
            SELECT g.id, g.external_id, g.home_team_id, g.away_team_id,
                   g.scheduled_at, g.status, g.home_score, g.away_score,
                   g.venue_id,
                   ht.master_team_id as home_master_team_id,
                   at.master_team_id as away_master_team_id
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.sport_id = :sid AND g.master_game_id IS NULL
            ORDER BY g.scheduled_at
        """), {"sid": sport_id_str})
        games = games_rows.fetchall()

        if not games:
            continue

        print(f"\n  {sport_code}: {len(games)} source games to process...")

        sport_masters = 0
        sport_mapped = 0

        for g in games:
            gid = str(g[0])
            ext_id = g[1]
            sched = g[4]
            status = g[5] or "scheduled"
            h_score = g[6]
            a_score = g[7]
            venue_id = str(g[8]) if g[8] else None
            h_master_team = str(g[9]) if g[9] else None
            a_master_team = str(g[10]) if g[10] else None
            source_key = extract_source_key(ext_id) if ext_id else "unknown"

            if not sched:
                continue

            # For tennis, resolve through player_mappings instead
            h_master_player = None
            a_master_player = None
            if is_tennis:
                # Look up player mapping for the "team" records
                for team_id_str, target in [(str(g[2]), "home"), (str(g[3]), "away")]:
                    pm_row = await session.execute(text("""
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

            # Try to find existing master_game (±24h, same matchup)
            window_start = sched - timedelta(hours=24)
            window_end = sched + timedelta(hours=24)

            if is_tennis:
                existing_row = await session.execute(text("""
                    SELECT id FROM master_games
                    WHERE sport_code = :sport
                      AND scheduled_at BETWEEN :ws AND :we
                      AND home_master_player_id = :hp
                      AND away_master_player_id = :ap
                    LIMIT 1
                """), {
                    "sport": sport_code, "ws": window_start, "we": window_end,
                    "hp": h_master_player, "ap": a_master_player,
                })
            else:
                existing_row = await session.execute(text("""
                    SELECT id FROM master_games
                    WHERE sport_code = :sport
                      AND scheduled_at BETWEEN :ws AND :we
                      AND home_master_team_id = :ht
                      AND away_master_team_id = :at
                    LIMIT 1
                """), {
                    "sport": sport_code, "ws": window_start, "we": window_end,
                    "ht": h_master_team, "at": a_master_team,
                })

            existing = existing_row.fetchone()

            if existing:
                master_game_id = str(existing[0])
            else:
                # Create new master_game
                mg_row = await session.execute(text("""
                    INSERT INTO master_games (id, sport_code, scheduled_at,
                        home_master_team_id, away_master_team_id,
                        home_master_player_id, away_master_player_id,
                        venue_id, status, home_score, away_score, primary_source)
                    VALUES (gen_random_uuid(), :sport, :sched,
                        :ht, :at, :hp, :ap, :vid, :status, :hs, :as_, :src)
                    RETURNING id
                """), {
                    "sport": sport_code, "sched": sched,
                    "ht": h_master_team, "at": a_master_team,
                    "hp": h_master_player, "ap": a_master_player,
                    "vid": venue_id, "status": status,
                    "hs": h_score, "as_": a_score, "src": source_key,
                })
                mg = mg_row.fetchone()
                master_game_id = str(mg[0])
                sport_masters += 1

            # Link source game → master_game
            await session.execute(text("""
                UPDATE games SET master_game_id = :mgid WHERE id = :gid
            """), {"mgid": master_game_id, "gid": gid})

            # Create game_mapping
            if ext_id:
                await session.execute(text("""
                    INSERT INTO game_mappings (id, master_game_id, source_key, source_external_id, source_game_db_id)
                    VALUES (gen_random_uuid(), :mgid, :src, :ext, :gid)
                    ON CONFLICT ON CONSTRAINT uq_game_map_source_ext DO NOTHING
                """), {"mgid": master_game_id, "src": source_key, "ext": ext_id, "gid": gid})

            sport_mapped += 1

        total_masters += sport_masters
        total_mapped += sport_mapped
        print(f"  ✅ {sport_code}: {sport_masters} unique games, {sport_mapped} source records mapped")

        # Commit per-sport to avoid huge transactions
        await session.commit()

    print(f"\n  TOTAL: {total_masters} master_games created, {total_mapped} source games linked")


async def phase5_backfill(session):
    """Phase 5: Backfill master_game_id on odds, player_stats, public_betting."""
    print("\n" + "=" * 70)
    print("[PHASE 5] Backfilling master_game_id on related tables")
    print("=" * 70)

    # Odds → master_game_id (via games.master_game_id)
    result = await session.execute(text("""
        UPDATE odds SET master_game_id = g.master_game_id
        FROM games g
        WHERE odds.game_id = g.id
          AND odds.master_game_id IS NULL
          AND g.master_game_id IS NOT NULL
    """))
    print(f"  ✅ Odds: {result.rowcount} records backfilled")
    await session.commit()

    # Player stats → master_game_id
    result = await session.execute(text("""
        UPDATE player_stats SET master_game_id = g.master_game_id
        FROM games g
        WHERE player_stats.game_id = g.id
          AND player_stats.master_game_id IS NULL
          AND g.master_game_id IS NOT NULL
    """))
    print(f"  ✅ Player stats (game): {result.rowcount} records backfilled")
    await session.commit()

    # Player stats → master_player_id
    result = await session.execute(text("""
        UPDATE player_stats SET master_player_id = p.master_player_id
        FROM players p
        WHERE player_stats.player_id = p.id
          AND player_stats.master_player_id IS NULL
          AND p.master_player_id IS NOT NULL
    """))
    print(f"  ✅ Player stats (player): {result.rowcount} records backfilled")
    await session.commit()

    # Players → master_player_id (from player_mappings)
    result = await session.execute(text("""
        UPDATE players SET master_player_id = pm.master_player_id
        FROM player_mappings pm
        WHERE players.id = pm.source_player_db_id
          AND players.master_player_id IS NULL
    """))
    print(f"  ✅ Players: {result.rowcount} records backfilled")
    await session.commit()

    # Public betting → master_game_id (match by team names + date)
    # This is trickier: public_betting has text team names, not IDs
    result = await session.execute(text("""
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
    print(f"  ✅ Public betting: {result.rowcount} records backfilled")
    await session.commit()


async def run_all():
    """Execute all phases."""
    await db_manager.initialize()

    async with db_manager.session() as session:
        await phase3_map_teams(session)
        await phase3b_map_tennis_players(session)

    async with db_manager.session() as session:
        await phase4_create_master_games(session)

    async with db_manager.session() as session:
        await phase5_backfill(session)

    # Print final summary
    async with db_manager.session() as session:
        for table in ["master_teams", "master_players", "master_games",
                       "team_mappings", "player_mappings", "game_mappings"]:
            row = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = row.scalar()
            print(f"  {table}: {count:,} rows")

    print("\n✅ All phases complete! Master data layer is ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run_all())
