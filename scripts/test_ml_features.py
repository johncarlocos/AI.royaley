#!/usr/bin/env python
"""
ROYALEY - ML Feature Extraction Test Script
============================================

Test each feature dimension one by one:
    python -m scripts.test_ml_features --team
    python -m scripts.test_ml_features --context
    python -m scripts.test_ml_features --player
    python -m scripts.test_ml_features --odds
    python -m scripts.test_ml_features --situational
    python -m scripts.test_ml_features --all

Or test full extraction:
    python -m scripts.test_ml_features --full --sport NBA --limit 10
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_team_features(sport: str, limit: int):
    """Test team features only."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_team_features(sport, limit=limit)
        
        print("\n" + "=" * 70)
        print("TEAM FEATURES TEST")
        print("=" * 70)
        for r in results:
            print(f"\n📊 {r['game']} ({r['date']})")
            print(f"   Home Win% L10: {r['home_win_pct_last10']}")
            print(f"   Away Win% L10: {r['away_win_pct_last10']}")
            print(f"   Home Avg Pts L10: {r['home_avg_pts_last10']}")
            print(f"   Away Avg Pts L10: {r['away_avg_pts_last10']}")
            print(f"   Home Avg Margin L10: {r['home_avg_margin_last10']}")
            print(f"   H2H Home Wins L5: {r['h2h_home_wins_last5']}")
            print(f"   Power Rating Diff: {r['power_rating_diff']}")


async def test_game_context_features(sport: str, limit: int):
    """Test game context features only."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_game_context_features(sport, limit=limit)
        
        print("\n" + "=" * 70)
        print("GAME CONTEXT FEATURES TEST")
        print("=" * 70)
        for r in results:
            print(f"\n📅 {r['game']} ({r['date']})")
            print(f"   Home Rest Days: {r['home_rest_days']}")
            print(f"   Away Rest Days: {r['away_rest_days']}")
            print(f"   Rest Advantage: {r['rest_advantage']}")
            print(f"   Home B2B: {r['home_b2b']}")
            print(f"   Away B2B: {r['away_b2b']}")
            print(f"   Day of Week: {r['day_of_week']}")
            print(f"   Night Game: {r['is_night_game']}")


async def test_player_features(sport: str, limit: int):
    """Test player features only."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_player_features(sport, limit=limit)
        
        print("\n" + "=" * 70)
        print("PLAYER FEATURES TEST")
        print("=" * 70)
        for r in results:
            print(f"\n👤 {r['game']} ({r['date']})")
            print(f"   Home Star Pts Avg: {r['home_star_pts_avg']}")
            print(f"   Away Star Pts Avg: {r['away_star_pts_avg']}")
            print(f"   Home Injuries Out: {r['home_injuries_out']}")
            print(f"   Away Injuries Out: {r['away_injuries_out']}")


async def test_odds_features(sport: str, limit: int):
    """Test odds features only."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_odds_features(sport, limit=limit)
        
        print("\n" + "=" * 70)
        print("ODDS/MARKET FEATURES TEST")
        print("=" * 70)
        for r in results:
            print(f"\n💰 {r['game']} ({r['date']})")
            print(f"   Spread Open: {r['spread_open']}")
            print(f"   Spread Close: {r['spread_close']}")
            print(f"   Spread Movement: {r['spread_movement']}")
            print(f"   Total Close: {r['total_close']}")
            print(f"   Pinnacle Spread: {r['pinnacle_spread']}")
            print(f"   Num Books: {r['num_books']}")
            print(f"   Reverse Line Move: {r['is_rlm']}")


async def test_situational_features(sport: str, limit: int):
    """Test situational features only."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_situational_features(sport, limit=limit)
        
        print("\n" + "=" * 70)
        print("SITUATIONAL FEATURES TEST")
        print("=" * 70)
        for r in results:
            print(f"\n🎯 {r['game']} ({r['date']})")
            print(f"   Home Streak: {r['home_streak']}")
            print(f"   Away Streak: {r['away_streak']}")
            print(f"   Home Revenge: {r['home_revenge']}")
            print(f"   Away Revenge: {r['away_revenge']}")
            print(f"   Home Game #: {r['home_game_num']}")
            print(f"   Away Game #: {r['away_game_num']}")


async def test_full_extraction(sport: str, limit: int):
    """Test full feature extraction."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService, features_to_dataframe
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        
        print(f"\n🚀 Extracting ALL features for {sport} (limit={limit})...")
        features = await svc.extract_all_features(sport, limit=limit)
        
        print(f"\n✅ Extracted {len(features)} feature vectors")
        
        # Convert to DataFrame
        df = features_to_dataframe(features)
        print(f"\n📊 DataFrame shape: {df.shape}")
        print(f"\n📋 Columns ({len(df.columns)}):")
        
        # Group columns by type
        id_cols = [c for c in df.columns if 'id' in c or 'name' in c or c in ['sport_code', 'scheduled_at', 'season']]
        target_cols = [c for c in df.columns if c in ['home_win', 'total_points', 'score_margin', 'spread_result', 'over_result', 'home_score', 'away_score']]
        team_cols = [c for c in df.columns if c.startswith('home_') or c.startswith('away_') or c.startswith('h2h_') or 'power' in c]
        odds_cols = [c for c in df.columns if 'spread' in c or 'total' in c or 'money' in c or 'pinnacle' in c or 'book' in c or 'implied' in c or 'public' in c or 'sharp' in c or 'rlm' in c or 'steam' in c]
        context_cols = [c for c in df.columns if 'rest' in c or 'b2b' in c or 'night' in c or 'day' in c or 'month' in c or 'playoff' in c or 'neutral' in c or 'divisional' in c or 'conference' in c or 'rivalry' in c]
        situational_cols = [c for c in df.columns if 'streak' in c or 'revenge' in c or 'letdown' in c or 'lookahead' in c or 'game_num' in c]
        weather_cols = [c for c in df.columns if 'temperature' in c or 'wind' in c or 'precipitation' in c or 'humidity' in c or 'dome' in c]
        
        print(f"\n  🆔 ID columns ({len(id_cols)}): {id_cols[:5]}...")
        print(f"  🎯 Target columns ({len(target_cols)}): {target_cols}")
        print(f"  📊 Team features: {len([c for c in team_cols if c not in id_cols + target_cols])}")
        print(f"  💰 Odds features: {len(odds_cols)}")
        print(f"  📅 Context features: {len(context_cols)}")
        print(f"  🎯 Situational features: {len(situational_cols)}")
        print(f"  🌤️ Weather features: {len(weather_cols)}")
        
        # Show sample
        print(f"\n📈 Sample data (first row):")
        if len(df) > 0:
            row = df.iloc[0]
            sample_cols = ['home_team_name', 'away_team_name', 'home_win', 
                          'home_win_pct_last10', 'away_win_pct_last10',
                          'spread_close', 'pinnacle_spread', 'home_rest_days']
            for col in sample_cols:
                if col in df.columns:
                    print(f"   {col}: {row[col]}")
        
        # Non-null counts
        print(f"\n📉 Feature coverage (non-null %):")
        feature_groups = {
            'Team': team_cols[:5],
            'Odds': odds_cols[:5],
            'Context': context_cols[:5],
            'Situational': situational_cols[:5],
        }
        for group, cols in feature_groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                coverage = df[valid_cols].notna().mean().mean() * 100
                print(f"   {group}: {coverage:.1f}%")


async def test_all_dimensions(sport: str, limit: int):
    """Run all individual tests."""
    await test_team_features(sport, limit)
    await test_game_context_features(sport, limit)
    await test_player_features(sport, limit)
    await test_odds_features(sport, limit)
    await test_situational_features(sport, limit)


def main():
    parser = argparse.ArgumentParser(description='Test ML Feature Extraction')
    parser.add_argument('--sport', type=str, default='NBA', help='Sport code (NBA, NFL, MLB, NHL)')
    parser.add_argument('--limit', type=int, default=5, help='Number of games to test')
    
    # Test dimensions
    parser.add_argument('--team', action='store_true', help='Test team features')
    parser.add_argument('--context', action='store_true', help='Test game context features')
    parser.add_argument('--player', action='store_true', help='Test player features')
    parser.add_argument('--odds', action='store_true', help='Test odds features')
    parser.add_argument('--situational', action='store_true', help='Test situational features')
    parser.add_argument('--all', action='store_true', help='Test all dimensions')
    parser.add_argument('--full', action='store_true', help='Test full extraction')
    
    args = parser.parse_args()
    
    # Default to --all if nothing specified
    if not any([args.team, args.context, args.player, args.odds, args.situational, args.all, args.full]):
        args.all = True
    
    print(f"\n{'=' * 70}")
    print(f"ML FEATURE EXTRACTION TEST - {args.sport}")
    print(f"{'=' * 70}")
    
    if args.team:
        asyncio.run(test_team_features(args.sport, args.limit))
    elif args.context:
        asyncio.run(test_game_context_features(args.sport, args.limit))
    elif args.player:
        asyncio.run(test_player_features(args.sport, args.limit))
    elif args.odds:
        asyncio.run(test_odds_features(args.sport, args.limit))
    elif args.situational:
        asyncio.run(test_situational_features(args.sport, args.limit))
    elif args.all:
        asyncio.run(test_all_dimensions(args.sport, args.limit))
    elif args.full:
        asyncio.run(test_full_extraction(args.sport, args.limit))
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    main()
