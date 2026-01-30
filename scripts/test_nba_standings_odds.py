#!/usr/bin/env python3
"""
Quick test script to collect ONLY team stats (standings) and odds for NBA.
Run with: python test_nba_standings_odds.py
"""

import asyncio
import sys
sys.path.insert(0, '/app')

from app.core.database import get_db
from app.services.collectors.collector_26_balldontlie import BallDontLieCollector

async def main():
    print("=" * 60)
    print("Testing NBA Team Stats (Standings) and Odds Collection")
    print("=" * 60)
    
    collector = BallDontLieCollector()
    
    async for session in get_db():
        try:
            # 1. Test Standings/Team Stats
            print("\n📊 Step 1: Collecting NBA standings...")
            current_year = 2025
            standings = await collector.collect_standings("NBA", season=current_year)
            print(f"   Collected: {len(standings)} standings records")
            
            if standings:
                team_stat_results = await collector.save_team_stats(standings, "NBA", session)
                print(f"   ✅ Saved: {team_stat_results.get('saved', 0)} team stats")
                print(f"   ⏭️  Skipped: {team_stat_results.get('skipped', 0)}")
            else:
                print("   ❌ No standings data returned")
            
            # 2. Test Odds
            print("\n💰 Step 2: Collecting NBA odds...")
            from datetime import datetime, timedelta
            today = datetime.now()
            dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
            
            odds = await collector.collect_odds("NBA", dates=dates, season=current_year)
            print(f"   Collected: {len(odds)} odds records")
            
            if odds:
                odds_results = await collector.save_odds(odds, "NBA", session)
                print(f"   ✅ Saved: {odds_results.get('saved', 0)} odds records")
                print(f"   ⏭️  Skipped: {odds_results.get('skipped', 0)}")
            else:
                print("   ❌ No odds data returned")
            
            print("\n" + "=" * 60)
            print("✅ Test Complete!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await collector.close()
            break

if __name__ == "__main__":
    asyncio.run(main())