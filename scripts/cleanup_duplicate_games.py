#!/usr/bin/env python3
"""
ROYALEY - Cleanup Duplicate Games
==================================
Removes duplicate games from the database, keeping only one copy of each game.
Preserves the game with the most odds records attached, or the oldest if none have odds.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import db_manager


async def cleanup_duplicates():
    """Remove duplicate games, keeping the one with most associated data."""
    await db_manager.initialize()
    
    async with db_manager.session() as session:
        from sqlalchemy import text
        
        # First, find all duplicate groups
        print("Finding duplicate games...")
        result = await session.execute(text('''
            SELECT home_team_id, away_team_id, scheduled_at, COUNT(*) as cnt
            FROM games
            GROUP BY home_team_id, away_team_id, scheduled_at
            HAVING COUNT(*) > 1
        '''))
        duplicates = result.fetchall()
        print(f"Found {len(duplicates)} groups of duplicate games")
        
        if not duplicates:
            print("No duplicates found!")
            return
        
        total_deleted = 0
        
        for home_id, away_id, scheduled_at, count in duplicates:
            # Get all game IDs for this duplicate group
            result = await session.execute(text('''
                SELECT g.id, 
                       (SELECT COUNT(*) FROM odds WHERE game_id = g.id) as odds_count,
                       g.created_at
                FROM games g
                WHERE g.home_team_id = :home_id 
                  AND g.away_team_id = :away_id 
                  AND g.scheduled_at = :scheduled_at
                ORDER BY odds_count DESC, g.created_at ASC
            '''), {"home_id": home_id, "away_id": away_id, "scheduled_at": scheduled_at})
            
            games = result.fetchall()
            
            if len(games) <= 1:
                continue
            
            # Keep the first one (most odds, or oldest)
            keep_id = games[0][0]
            delete_ids = [g[0] for g in games[1:]]
            
            # Move any odds from games being deleted to the kept game
            for delete_id in delete_ids:
                await session.execute(text('''
                    UPDATE odds SET game_id = :keep_id WHERE game_id = :delete_id
                '''), {"keep_id": keep_id, "delete_id": delete_id})
                
                # Also move odds_movements
                await session.execute(text('''
                    UPDATE odds_movements SET game_id = :keep_id WHERE game_id = :delete_id
                '''), {"keep_id": keep_id, "delete_id": delete_id})
                
                # Move closing_lines
                await session.execute(text('''
                    UPDATE closing_lines SET game_id = :keep_id WHERE game_id = :delete_id
                '''), {"keep_id": keep_id, "delete_id": delete_id})
            
            # Delete the duplicate games
            placeholders = ', '.join([f':id{i}' for i in range(len(delete_ids))])
            params = {f'id{i}': str(delete_ids[i]) for i in range(len(delete_ids))}
            
            await session.execute(
                text(f'DELETE FROM games WHERE id IN ({placeholders})'),
                params
            )
            
            total_deleted += len(delete_ids)
            
            if total_deleted % 100 == 0:
                print(f"  Deleted {total_deleted} duplicates so far...")
                await session.commit()
        
        await session.commit()
        print(f"\n✅ Cleanup complete! Deleted {total_deleted} duplicate games.")
        
        # Verify
        result = await session.execute(text('SELECT COUNT(*) FROM games'))
        remaining = result.scalar()
        print(f"Remaining games: {remaining}")


if __name__ == "__main__":
    print("=" * 60)
    print("ROYALEY - Duplicate Games Cleanup")
    print("=" * 60)
    asyncio.run(cleanup_duplicates())