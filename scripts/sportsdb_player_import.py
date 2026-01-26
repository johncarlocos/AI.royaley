#!/usr/bin/env python3
"""
SportsDB Player Import - STANDALONE SCRIPT
==========================================

Run: docker exec royaley_api python scripts/sportsdb_player_import.py

This script ONLY imports players with detailed debugging to identify
why players aren't being imported.

DEBUGGING FEATURES:
1. Shows raw API response keys
2. Tests each sport individually
3. Prints first few player records to verify data structure
4. Comprehensive error logging
"""

import asyncio
import sys
from datetime import datetime

sys.path.insert(0, '/app')

TEAM_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA"]


async def test_single_team_players():
    """Test player API with a single known team (Dallas Cowboys = 134914)."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Team Player API Test (Dallas Cowboys)")
    print("=" * 60)
    
    from app.services.collectors.collector_06_sportsdb import SportsDBCollector
    
    collector = SportsDBCollector()
    
    # Cowboys team ID
    team_id = "134914"
    
    print(f"\nCalling: /list/players/{team_id}")
    data = await collector._v2(f"/list/players/{team_id}")
    
    if data is None:
        print("❌ API returned None")
        return False
    
    print(f"\n📦 Response type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📦 Response keys: {list(data.keys())}")
        
        for key, val in data.items():
            if isinstance(val, list):
                print(f"   - {key}: {len(val)} items (LIST)")
                if val:
                    print(f"     First item keys: {list(val[0].keys()) if isinstance(val[0], dict) else 'not a dict'}")
            elif val is None:
                print(f"   - {key}: None")
            else:
                print(f"   - {key}: {type(val).__name__}")
    
    # Try to extract players using various keys
    players = None
    for key in ["player", "players", "list", "roster"]:
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, list) and len(val) > 0:
                players = val
                print(f"\n✅ Found players under key '{key}': {len(players)} players")
                break
            elif val is None:
                print(f"⚠️ Key '{key}' exists but is None")
    
    if players:
        print("\n📋 First 3 players:")
        for i, p in enumerate(players[:3]):
            print(f"   {i+1}. {p.get('strPlayer')} - {p.get('strPosition')} - #{p.get('strNumber')}")
        return True
    else:
        print("\n❌ Could not find player list in response")
        return False


async def import_players_for_sport(sport_code: str) -> int:
    """Import players for a single sport with detailed logging."""
    from app.services.collectors import sportsdb_collector
    from app.core.database import db_manager
    
    print(f"\n{'='*50}")
    print(f"IMPORTING PLAYERS: {sport_code}")
    print(f"{'='*50}")
    
    try:
        # Step 1: Collect players
        print(f"\n[1/2] Collecting players from API...")
        players_data = await sportsdb_collector.collect_players(sport_code=sport_code)
        
        if not players_data:
            print(f"❌ collect_players returned None")
            return 0
        
        player_count = players_data.get("count", 0)
        players_list = players_data.get("players", [])
        
        print(f"    Collected: {player_count} players")
        
        if not players_list:
            print(f"⚠️ No players in list")
            return 0
        
        # Show sample
        print(f"\n    Sample player data:")
        for p in players_list[:2]:
            print(f"       - {p.get('name')} ({p.get('team_name')}) - {p.get('position')}")
        
        # Step 2: Save to database
        print(f"\n[2/2] Saving to database...")
        async with db_manager.session() as session:
            saved = await sportsdb_collector.save_players_to_database(players_list, session)
            print(f"    Saved: {saved} new players")
            return saved
            
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 0


async def verify_database_players():
    """Verify player counts in database."""
    from app.core.database import db_manager
    from sqlalchemy import text
    
    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION")
    print("=" * 60)
    
    await db_manager.initialize()
    
    async with db_manager.session() as session:
        # Total players
        result = await session.execute(text("SELECT COUNT(*) FROM players"))
        total = result.scalar()
        print(f"\n📊 Total players in database: {total:,}")
        
        # Players per sport (via team)
        result = await session.execute(text("""
            SELECT s.code, COUNT(p.id) as count 
            FROM sports s 
            LEFT JOIN teams t ON t.sport_id = s.id 
            LEFT JOIN players p ON p.team_id = t.id
            GROUP BY s.code 
            ORDER BY COUNT(p.id) DESC
        """))
        print("\nPlayers per sport:")
        for row in result:
            print(f"   {row[0]}: {row[1]:,}")
        
        # Sample players
        result = await session.execute(text("""
            SELECT p.name, t.name as team, p.position, p.jersey_number
            FROM players p
            JOIN teams t ON p.team_id = t.id
            LIMIT 10
        """))
        rows = result.fetchall()
        
        if rows:
            print("\nSample players in database:")
            for row in rows:
                print(f"   - {row[0]} ({row[1]}) - {row[2]} #{row[3]}")
        else:
            print("\n⚠️ No players found in database")


async def main():
    print("=" * 60)
    print("SPORTSDB PLAYER IMPORT - DEBUG VERSION")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    
    from app.core.database import db_manager
    await db_manager.initialize()
    
    # Test 1: Single team API test
    api_works = await test_single_team_players()
    
    if not api_works:
        print("\n❌ API test failed - cannot proceed with import")
        print("   Check your API key and network connection")
        return
    
    # Import players for each sport
    print("\n" + "=" * 60)
    print("FULL PLAYER IMPORT")
    print("=" * 60)
    
    total_saved = 0
    for sport in TEAM_SPORTS:
        saved = await import_players_for_sport(sport)
        total_saved += saved
    
    print("\n" + "=" * 60)
    print(f"IMPORT COMPLETE: {total_saved:,} total players saved")
    print("=" * 60)
    
    # Verify
    await verify_database_players()


if __name__ == "__main__":
    asyncio.run(main())
