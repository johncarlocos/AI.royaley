#!/usr/bin/env python3
"""
Collect 10 years of NBA historical team stats (standings) and odds from BallDontLie.
Run with: python scripts/test_nba_standings_odds.py
"""

import asyncio
import sys
sys.path.insert(0, '/app')

from datetime import datetime, timedelta
from app.core.database import get_db
from app.services.collectors.collector_26_balldontlie import BallDontLieCollectorV2

# 10 seasons: 2015-2024 (NBA season year = start year, e.g., 2024 = 2024-25 season)
SEASONS = list(range(2015, 2025))

async def main():
    print("=" * 70)
    print("NBA 10-Year Historical Data Collection: Team Stats + Odds")
    print("=" * 70)
    print(f"Seasons: {SEASONS[0]} - {SEASONS[-1]}")
    print("=" * 70)
    
    collector = BallDontLieCollectorV2()
    
    total_team_stats_saved = 0
    total_odds_saved = 0
    
    async for session in get_db():
        try:
            # ===== PART 1: TEAM STATS (STANDINGS) =====
            print("\n" + "=" * 70)
            print("PART 1: Collecting Team Stats (Standings)")
            print("=" * 70)
            
            for season in SEASONS:
                print(f"\n[Season {season}] Collecting standings...")
                try:
                    standings = await collector.collect_standings("NBA", season=season)
                    print(f"    Collected: {len(standings)} records")
                    
                    if standings:
                        result = await collector.save_team_stats(standings, "NBA", session)
                        saved = result.get("saved", 0)
                        total_team_stats_saved += saved
                        print(f"    Saved: {saved} team stats")
                    else:
                        print(f"    No standings data for season {season}")
                        
                except Exception as e:
                    print(f"    Error: {e}")
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.5)
            
            # ===== PART 2: ODDS =====
            print("\n" + "=" * 70)
            print("PART 2: Collecting Odds")
            print("=" * 70)
            
            for season in SEASONS:
                print(f"\n[Season {season}] Collecting odds...")
                try:
                    # NBA season runs Oct-June, generate date range
                    # Season 2024 = Oct 2024 to Jun 2025
                    start_date = datetime(season, 10, 1)
                    end_date = datetime(season + 1, 6, 30)
                    
                    # Generate all dates in the season
                    dates = []
                    current = start_date
                    while current <= end_date:
                        dates.append(current.strftime("%Y-%m-%d"))
                        current += timedelta(days=1)
                    
                    print(f"    Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
                    
                    # Collect odds in batches of 30 days to avoid timeout
                    season_odds = []
                    batch_size = 30
                    
                    for i in range(0, len(dates), batch_size):
                        batch_dates = dates[i:i + batch_size]
                        try:
                            odds = await collector.collect_odds("NBA", dates=batch_dates, season=season)
                            if odds:
                                season_odds.extend(odds)
                                print(f"    Batch {i//batch_size + 1}: {len(odds)} odds")
                        except Exception as e:
                            print(f"    Batch {i//batch_size + 1} error: {e}")
                        
                        await asyncio.sleep(0.3)
                    
                    print(f"    Total collected: {len(season_odds)} odds records")
                    
                    if season_odds:
                        result = await collector.save_odds(season_odds, "NBA", session)
                        saved = result.get("saved", 0)
                        total_odds_saved += saved
                        print(f"    Saved: {saved} odds records")
                    else:
                        print(f"    No odds data for season {season}")
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ===== SUMMARY =====
            print("\n" + "=" * 70)
            print("COLLECTION COMPLETE!")
            print("=" * 70)
            print(f"Total Team Stats Saved: {total_team_stats_saved}")
            print(f"Total Odds Saved: {total_odds_saved}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\nFatal Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await collector.close()
            break

if __name__ == "__main__":
    asyncio.run(main())