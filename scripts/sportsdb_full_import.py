#!/usr/bin/env python3
"""
SportsDB Full Import - FIXED VERSION with Player Debugging
Run: docker exec royaley_api python scripts/sportsdb_full_import.py

Tables populated:
1. teams - All team rosters
2. venues - Stadiums/arenas  
3. players - Player rosters (FIXED)
4. games - Historical (10yr) + Upcoming + Past results

Order matters:
- Teams must exist before games can reference them
- Sport records must exist (already in DB)

FIXES:
- Better player response parsing (handles both "player" and "players" keys)
- Detailed logging to identify API response structure
- Skip tennis sports for player import (individual sports)
"""

import asyncio
import sys
from datetime import datetime

# Add app to path
sys.path.insert(0, '/app')

# 8 team sports (ATP/WTA are individual sports - need different handling)
TEAM_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA"]

async def main():
    print("=" * 60)
    print("SPORTSDB FULL DATABASE IMPORT - FIXED VERSION")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Sports: {', '.join(TEAM_SPORTS)}")
    print()
    
    from app.services.collectors import sportsdb_collector
    from app.core.database import db_manager
    
    await db_manager.initialize()
    
    totals = {
        "teams": 0,
        "venues": 0,
        "players": 0,
        "games_historical": 0,
        "games_upcoming": 0,
        "games_past": 0,
    }
    
    # ============================================
    # STEP 1: TEAMS
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 1: IMPORTING TEAMS")
    print("=" * 40)
    
    for sport in TEAM_SPORTS:
        try:
            data = await sportsdb_collector.collect(sport_code=sport)
            if data.success and data.data:
                async with db_manager.session() as session:
                    saved = await sportsdb_collector.save_teams_to_database(
                        data.data.get("teams", []), session
                    )
                    totals["teams"] += saved
                    print(f"✅ {sport}: {saved} teams")
        except Exception as e:
            print(f"❌ {sport}: {str(e)[:80]}")
    
    print(f"\n📊 Total teams: {totals['teams']}")
    
    # ============================================
    # STEP 2: VENUES
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 2: IMPORTING VENUES")
    print("=" * 40)
    
    for sport in TEAM_SPORTS:
        try:
            venues_data = await sportsdb_collector.collect_venues(sport_code=sport)
            if venues_data and venues_data.get("venues"):
                async with db_manager.session() as session:
                    saved = await sportsdb_collector.save_venues_to_database(
                        venues_data["venues"], session
                    )
                    totals["venues"] += saved
                    print(f"✅ {sport}: {saved} venues")
        except Exception as e:
            print(f"❌ {sport}: {str(e)[:80]}")
    
    print(f"\n📊 Total venues: {totals['venues']}")
    
    # ============================================
    # STEP 3: PLAYERS (FIXED)
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 3: IMPORTING PLAYERS (FIXED VERSION)")
    print("=" * 40)
    print("Note: This will show API response keys for debugging")
    print()
    
    for sport in TEAM_SPORTS:
        try:
            print(f"\n--- {sport} ---")
            players_data = await sportsdb_collector.collect_players(sport_code=sport)
            
            player_count = players_data.get("count", 0) if players_data else 0
            players_list = players_data.get("players", []) if players_data else []
            
            print(f"[DEBUG] {sport}: collect_players returned {player_count} players")
            
            if players_list:
                async with db_manager.session() as session:
                    saved = await sportsdb_collector.save_players_to_database(
                        players_list, session
                    )
                    totals["players"] += saved
                    print(f"✅ {sport}: {saved} players saved to database")
            else:
                print(f"⚠️ {sport}: No players to save")
                
        except Exception as e:
            import traceback
            print(f"❌ {sport}: {str(e)[:80]}")
            traceback.print_exc()
    
    print(f"\n📊 Total players: {totals['players']}")
    
    # ============================================
    # STEP 4: HISTORICAL GAMES (10 YEARS)
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 4: IMPORTING HISTORICAL GAMES (10 YEARS)")
    print("This may take 15-30 minutes...")
    print("=" * 40)
    
    for sport in TEAM_SPORTS:
        try:
            data = await sportsdb_collector.collect_historical(
                sport_code=sport,
                seasons_back=10
            )
            if data.success and data.data:
                games = data.data.get("games", [])
                async with db_manager.session() as session:
                    saved, _ = await sportsdb_collector.save_historical_to_database(
                        games, session
                    )
                    totals["games_historical"] += saved
                    print(f"✅ {sport}: {saved} historical games")
        except Exception as e:
            print(f"❌ {sport}: {str(e)[:80]}")
    
    print(f"\n📊 Total historical games: {totals['games_historical']}")
    
    # ============================================
    # STEP 5: UPCOMING GAMES (Next 20 per sport)
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 5: IMPORTING UPCOMING GAMES")
    print("=" * 40)
    
    for sport in TEAM_SPORTS:
        try:
            games = await sportsdb_collector.collect_upcoming(sport_code=sport)
            if games:
                async with db_manager.session() as session:
                    saved = await sportsdb_collector.save_games_to_database(games, session)
                    totals["games_upcoming"] += saved
                    print(f"✅ {sport}: {saved} upcoming games")
        except Exception as e:
            print(f"❌ {sport}: {str(e)[:80]}")
    
    print(f"\n📊 Total upcoming games: {totals['games_upcoming']}")
    
    # ============================================
    # STEP 6: PAST GAMES (Recent Results - Last 20 per sport)
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 6: IMPORTING PAST GAMES (RESULTS)")
    print("=" * 40)
    
    for sport in TEAM_SPORTS:
        try:
            games = await sportsdb_collector.collect_past(sport_code=sport)
            if games:
                async with db_manager.session() as session:
                    saved = await sportsdb_collector.save_games_to_database(games, session)
                    totals["games_past"] += saved
                    print(f"✅ {sport}: {saved} past games")
        except Exception as e:
            print(f"❌ {sport}: {str(e)[:80]}")
    
    print(f"\n📊 Total past games: {totals['games_past']}")
    
    # ============================================
    # STEP 7: VERIFY DATABASE COUNTS
    # ============================================
    print("\n" + "=" * 40)
    print("STEP 7: VERIFYING DATABASE")
    print("=" * 40)
    
    try:
        from sqlalchemy import text
        async with db_manager.session() as session:
            # Count teams per sport
            result = await session.execute(text("""
                SELECT s.code, COUNT(t.id) as count 
                FROM sports s 
                LEFT JOIN teams t ON t.sport_id = s.id 
                GROUP BY s.code 
                ORDER BY s.code
            """))
            print("\nTeams per sport:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")
            
            # Count players per sport (via team)
            result = await session.execute(text("""
                SELECT s.code, COUNT(p.id) as count 
                FROM sports s 
                LEFT JOIN teams t ON t.sport_id = s.id 
                LEFT JOIN players p ON p.team_id = t.id
                GROUP BY s.code 
                ORDER BY s.code
            """))
            print("\nPlayers per sport:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")
            
            # Count games per sport
            result = await session.execute(text("""
                SELECT s.code, COUNT(g.id) as count 
                FROM sports s 
                LEFT JOIN games g ON g.sport_id = s.id 
                GROUP BY s.code 
                ORDER BY s.code
            """))
            print("\nGames per sport:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")
            
            # Total counts
            result = await session.execute(text("SELECT COUNT(*) FROM teams"))
            total_teams = result.scalar()
            
            result = await session.execute(text("SELECT COUNT(*) FROM venues"))
            total_venues = result.scalar()
            
            result = await session.execute(text("SELECT COUNT(*) FROM players"))
            total_players = result.scalar()
            
            result = await session.execute(text("SELECT COUNT(*) FROM games"))
            total_games = result.scalar()
            
            print(f"\n📊 DATABASE TOTALS:")
            print(f"   Teams:   {total_teams:,}")
            print(f"   Venues:  {total_venues:,}")
            print(f"   Players: {total_players:,}")
            print(f"   Games:   {total_games:,}")
    except Exception as e:
        print(f"Verification error: {e}")
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now()}")
    print()
    print("📊 IMPORT TOTALS (this run):")
    print(f"   Teams:            {totals['teams']:,}")
    print(f"   Venues:           {totals['venues']:,}")
    print(f"   Players:          {totals['players']:,}")
    print(f"   Historical Games: {totals['games_historical']:,}")
    print(f"   Upcoming Games:   {totals['games_upcoming']:,}")
    print(f"   Past Games:       {totals['games_past']:,}")
    print()
    total_games = totals['games_historical'] + totals['games_upcoming'] + totals['games_past']
    print(f"   TOTAL GAMES:      {total_games:,}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
