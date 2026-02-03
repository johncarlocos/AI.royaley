#!/usr/bin/env python
"""
ROYALEY - ML Feature Extraction & CSV Export
=============================================

Generates ML features and saves to CSV files organized by sport.

OUTPUT STRUCTURE (80 files in 10 folders):
    ml_csv/
    ├── NBA/
    │   ├── ml_features_NBA_{timestamp}.csv           - Combined all ~92 features
    │   ├── ml_features_NBA_team_{timestamp}.csv      - ~24 team rolling stats
    │   ├── ml_features_NBA_game_{timestamp}.csv      - ~15 game context features
    │   ├── ml_features_NBA_player_{timestamp}.csv    - ~10 player features
    │   ├── ml_features_NBA_odds_{timestamp}.csv      - ~24 odds/market features
    │   ├── ml_features_NBA_situation_{timestamp}.csv - ~10 situational features
    │   ├── ml_features_NBA_weather_{timestamp}.csv   - ~5 weather features
    │   └── ml_features_NBA_target_{timestamp}.csv    - ~7 target variables
    ├── NFL/
    │   └── (8 files)
    ├── MLB/
    ├── NHL/
    ├── WNBA/
    ├── CFL/
    ├── NCAAF/
    ├── NCAAB/
    ├── ATP/
    └── WTA/

COMMANDS:
    # Export ALL games for ONE sport
    python -m scripts.export_ml_features --export --sport NBA
    
    # Export ALL games for ALL 10 sports (80 CSV files in 10 folders)
    python -m scripts.export_ml_features --export --sport ALL
    
    # Export with limit (for testing)
    python -m scripts.export_ml_features --export --sport NBA --limit 1000

Supported sports: NFL, NBA, MLB, NHL, WNBA, CFL, NCAAF, NCAAB, ATP, WTA
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

ALL_SPORTS = ['NFL', 'NBA', 'MLB', 'NHL', 'WNBA', 'CFL', 'NCAAF', 'NCAAB', 'ATP', 'WTA']

# =============================================================================
# COLUMN DEFINITIONS BY DIMENSION (must match MLFeatureVector in ml_features.py)
# =============================================================================

# ID columns (included in all dimension files)
ID_COLUMNS = [
    'master_game_id', 'sport_code', 'scheduled_at', 'season',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name'
]

# Target columns (~7)
TARGET_COLUMNS = [
    'home_score', 'away_score', 'home_win', 'total_points',
    'score_margin', 'spread_result', 'over_result'
]

# Team features (~24)
TEAM_COLUMNS = [
    'home_wins_last5', 'home_wins_last10', 'home_win_pct_last10',
    'home_avg_pts_last10', 'home_avg_pts_allowed_last10', 'home_avg_margin_last10',
    'home_home_win_pct', 'home_ats_record_last10', 'home_ou_over_pct_last10',
    'away_wins_last5', 'away_wins_last10', 'away_win_pct_last10',
    'away_avg_pts_last10', 'away_avg_pts_allowed_last10', 'away_avg_margin_last10',
    'away_away_win_pct', 'away_ats_record_last10', 'away_ou_over_pct_last10',
    'h2h_home_wins_last5', 'h2h_home_avg_margin', 'h2h_total_avg',
    'home_power_rating', 'away_power_rating', 'power_rating_diff'
]

# Game context features (~15)
GAME_COLUMNS = [
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    'home_is_back_to_back', 'away_is_back_to_back',
    'home_3_in_4_nights', 'away_3_in_4_nights',
    'is_divisional', 'is_conference', 'is_rivalry',
    'is_playoff', 'is_neutral_site',
    'day_of_week', 'is_night_game', 'month'
]

# Player features (~10)
PLAYER_COLUMNS = [
    'home_star_player_pts_avg', 'away_star_player_pts_avg',
    'home_top3_players_pts_avg', 'away_top3_players_pts_avg',
    'home_injuries_out', 'away_injuries_out',
    'home_injury_impact', 'away_injury_impact',
    'home_starters_out', 'away_starters_out'
]

# Odds/Market features (~24)
ODDS_COLUMNS = [
    'spread_open', 'spread_close', 'spread_movement',
    'moneyline_home_open', 'moneyline_home_close', 'moneyline_away_close',
    'total_open', 'total_close', 'total_movement',
    'pinnacle_spread', 'pinnacle_ml_home', 'pinnacle_total',
    'num_books', 'consensus_spread', 'consensus_total',
    'implied_home_prob', 'no_vig_home_prob',
    'public_spread_home_pct', 'public_ml_home_pct', 'public_total_over_pct',
    'public_money_home_pct', 'is_reverse_line_move', 'sharp_action_indicator', 'steam_move'
]

# Situational features (~10)
SITUATION_COLUMNS = [
    'home_streak', 'away_streak',
    'home_is_revenge', 'away_is_revenge',
    'home_letdown_spot', 'away_letdown_spot',
    'home_lookahead_spot', 'away_lookahead_spot',
    'home_season_game_num', 'away_season_game_num'
]

# Weather features (~5)
WEATHER_COLUMNS = [
    'temperature_f', 'wind_speed_mph', 'precipitation_pct',
    'humidity_pct', 'is_dome'
]

# Dimension mapping for CSV export
DIMENSIONS = {
    'team': TEAM_COLUMNS,
    'game': GAME_COLUMNS,
    'player': PLAYER_COLUMNS,
    'odds': ODDS_COLUMNS,
    'situation': SITUATION_COLUMNS,
    'weather': WEATHER_COLUMNS,
    'target': TARGET_COLUMNS,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_output_dir() -> str:
    """Get ml_csv output directory in project root."""
    current_dir = os.getcwd()
    
    # Try to find project root
    if os.path.exists(os.path.join(current_dir, 'app')):
        project_root = current_dir
    elif os.path.exists('/nvme0n1-disk/royaley'):
        project_root = '/nvme0n1-disk/royaley'
    else:
        project_root = current_dir
    
    output_dir = os.path.join(project_root, 'ml_csv')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def filter_columns(df, columns: List[str], include_ids: bool = True):
    """Filter DataFrame to include only specified columns that exist."""
    if include_ids:
        cols = ID_COLUMNS + columns
    else:
        cols = columns
    
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()


def calculate_coverage(df, columns: List[str]) -> float:
    """Calculate non-null coverage percentage for given columns."""
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        return 0.0
    return df[existing_cols].notna().mean().mean() * 100


# =============================================================================
# CSV EXPORT FUNCTIONS
# =============================================================================

def save_sport_csvs(df, sport: str, output_dir: str, timestamp: str) -> Dict[str, str]:
    """
    Save 8 CSV files for a single sport in sport-specific subfolder.
    
    Files created in {output_dir}/{sport}/:
        1. ml_features_{SPORT}_{timestamp}.csv (combined)
        2. ml_features_{SPORT}_team_{timestamp}.csv
        3. ml_features_{SPORT}_game_{timestamp}.csv
        4. ml_features_{SPORT}_player_{timestamp}.csv
        5. ml_features_{SPORT}_odds_{timestamp}.csv
        6. ml_features_{SPORT}_situation_{timestamp}.csv
        7. ml_features_{SPORT}_weather_{timestamp}.csv
        8. ml_features_{SPORT}_target_{timestamp}.csv
    
    Returns dict of dimension -> filepath
    """
    # Create sport-specific subfolder
    sport_dir = os.path.join(output_dir, sport)
    os.makedirs(sport_dir, exist_ok=True)
    
    files_created = {}
    
    print(f"\n   📁 Saving 8 CSV files for {sport} to {sport}/ folder...")
    
    # 1. Combined (all features)
    combined_path = os.path.join(sport_dir, f"ml_features_{sport}_{timestamp}.csv")
    df.to_csv(combined_path, index=False)
    files_created['combined'] = combined_path
    print(f"      ✅ ml_features_{sport}_{timestamp}.csv ({len(df)} rows × {len(df.columns)} cols)")
    
    # 2-8. Individual dimensions
    for dimension, columns in DIMENSIONS.items():
        dim_df = filter_columns(df, columns, include_ids=True)
        dim_path = os.path.join(sport_dir, f"ml_features_{sport}_{dimension}_{timestamp}.csv")
        dim_df.to_csv(dim_path, index=False)
        files_created[dimension] = dim_path
        
        coverage = calculate_coverage(df, columns)
        print(f"      ✅ ml_features_{sport}_{dimension}_{timestamp}.csv ({len(dim_df)} rows × {len(dim_df.columns)} cols, {coverage:.1f}% coverage)")
    
    return files_created


async def export_single_sport(sport: str, limit: int) -> tuple:
    """Extract features and save 8 CSV files for one sport."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService, features_to_dataframe
    
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 70)
    print(f"🏆 ML FEATURE EXTRACTION - {sport}")
    print("=" * 70)
    print(f"   Output folder: {output_dir}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Limit: {limit if limit else 'ALL GAMES'}")
    print(f"   Expected files: 8")
    
    await db_manager.initialize()
    
    # First, get all games
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        games = await svc._get_games(sport, completed_only=True, limit=limit)
    
    if not games:
        print(f"   ⚠️ No games found for {sport}")
        return None, {}
    
    print(f"\n   🔄 Extracting features for {len(games)} games...")
    
    # Process each game with its own session to avoid transaction corruption
    all_features = []
    error_count = 0
    
    for i, g in enumerate(games):
        try:
            async with db_manager.session() as session:
                svc = MLFeatureService(session)
                batch_features = await svc._extract_features_for_games([g], sport)
                all_features.extend(batch_features)
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only log first 5 errors
                logger.warning(f"Game {g.get('id')} failed: {e}")
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(games)} games")
    
    if (i + 1) % 100 != 0:  # Log final count if not already logged
        logger.info(f"  Processed {len(games)}/{len(games)} games")
    
    if error_count > 0:
        logger.warning(f"  ⚠️ {error_count} games had errors during extraction")
    
    if not all_features:
        print(f"   ⚠️ No features extracted for {sport}")
        return None, {}
    
    print(f"   ✅ Extracted {len(all_features)} games")
    
    df = features_to_dataframe(all_features)
    files = save_sport_csvs(df, sport, output_dir, timestamp)
    
    print(f"\n{'=' * 70}")
    print("📁 FILES CREATED")
    print("=" * 70)
    print(f"\n   Location: {output_dir}/{sport}/")
    print(f"\n   {sport} files (8 files):")
    for dim, path in files.items():
        filename = os.path.basename(path)
        print(f"      - {filename}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ COMPLETE! 8 CSV files created for {sport}")
    print("=" * 70)
    
    return df, files


async def export_all_sports(limit: int):
    """
    Export features for ALL 10 sports.
    Creates 81 CSV files total (8 per sport + 1 grand combined).
    """
    import pandas as pd
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService, features_to_dataframe
    
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 70)
    print("🏆 ML FEATURE EXTRACTION - ALL 10 SPORTS")
    print("=" * 70)
    print(f"   Output folder: {output_dir}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Limit per sport: {limit}")
    print(f"   Sports: {', '.join(ALL_SPORTS)}")
    print(f"   Expected files: 81 (8 per sport × 10 + 1 combined)")
    
    await db_manager.initialize()
    
    all_dfs = []
    all_files = {}
    summary = []
    
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        
        for sport in ALL_SPORTS:
            print(f"\n{'=' * 70}")
            print(f"📊 EXTRACTING: {sport}")
            print("=" * 70)
            
            features = await svc.extract_all_features(sport, limit=limit)
            
            if features:
                df = features_to_dataframe(features)
                all_dfs.append(df)
                
                files = save_sport_csvs(df, sport, output_dir, timestamp)
                all_files[sport] = files
                
                coverage = calculate_coverage(df, TEAM_COLUMNS + ODDS_COLUMNS)
                summary.append({
                    'sport': sport,
                    'games': len(df),
                    'columns': len(df.columns),
                    'coverage': f"{coverage:.1f}%",
                    'files': 8
                })
            else:
                print(f"   ⚠️ No games found for {sport}")
                summary.append({
                    'sport': sport,
                    'games': 0,
                    'columns': 0,
                    'coverage': "N/A",
                    'files': 0
                })
    
    # Save grand combined file (81st file)
    combined_path = None
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(output_dir, f"ml_features_ALL_SPORTS_{timestamp}.csv")
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\n{'=' * 70}")
        print(f"📦 GRAND COMBINED FILE")
        print("=" * 70)
        print(f"   ✅ ml_features_ALL_SPORTS_{timestamp}.csv")
        print(f"      Rows: {len(combined_df)}")
        print(f"      Columns: {len(combined_df.columns)}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\n{'Sport':<10} {'Games':<10} {'Columns':<10} {'Coverage':<12} {'Files':<6}")
    print("-" * 48)
    
    total_games = 0
    total_files = 0
    for s in summary:
        print(f"{s['sport']:<10} {s['games']:<10} {s['columns']:<10} {s['coverage']:<12} {s['files']:<6}")
        total_games += s['games']
        total_files += s['files']
    
    print("-" * 48)
    print(f"{'TOTAL':<10} {total_games:<10} {'-':<10} {'-':<12} {total_files + 1:<6}")
    
    # File listing
    print(f"\n{'=' * 70}")
    print("📁 ALL FILES CREATED (81 total)")
    print("=" * 70)
    print(f"\n   Location: {output_dir}/")
    
    for sport in ALL_SPORTS:
        if sport in all_files:
            print(f"\n   {sport} (8 files):")
            for dim, path in all_files[sport].items():
                filename = os.path.basename(path)
                print(f"      - {filename}")
    
    if combined_path:
        print(f"\n   GRAND COMBINED (1 file):")
        print(f"      - {os.path.basename(combined_path)}")
    
    print(f"\n{'=' * 70}")
    print(f"✅ COMPLETE! {total_files + 1} CSV files created in {output_dir}/")
    print("=" * 70)
    
    return summary


# =============================================================================
# TEST FUNCTIONS (Console output only - no CSV export)
# =============================================================================

async def test_team_features(sport: str, limit: int):
    """Test team features only (console output)."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_team_features(sport, limit=limit)
        
        print(f"\n{'=' * 70}")
        print(f"TEAM FEATURES - {sport}")
        print("=" * 70)
        
        if not results:
            print(f"   ⚠️ No games found")
            return
        
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
    """Test game context features only (console output)."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_game_context_features(sport, limit=limit)
        
        print(f"\n{'=' * 70}")
        print(f"GAME CONTEXT FEATURES - {sport}")
        print("=" * 70)
        
        if not results:
            print(f"   ⚠️ No games found")
            return
        
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
    """Test player features only (console output)."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_player_features(sport, limit=limit)
        
        print(f"\n{'=' * 70}")
        print(f"PLAYER FEATURES - {sport}")
        print("=" * 70)
        
        if not results:
            print(f"   ⚠️ No games found")
            return
        
        for r in results:
            print(f"\n👤 {r['game']} ({r['date']})")
            print(f"   Home Star Pts Avg: {r['home_star_pts_avg']}")
            print(f"   Away Star Pts Avg: {r['away_star_pts_avg']}")
            print(f"   Home Injuries Out: {r['home_injuries_out']}")
            print(f"   Away Injuries Out: {r['away_injuries_out']}")


async def test_odds_features(sport: str, limit: int):
    """Test odds features only (console output)."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_odds_features(sport, limit=limit)
        
        print(f"\n{'=' * 70}")
        print(f"ODDS/MARKET FEATURES - {sport}")
        print("=" * 70)
        
        if not results:
            print(f"   ⚠️ No games found")
            return
        
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
    """Test situational features only (console output)."""
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    await db_manager.initialize()
    async with db_manager.session() as session:
        svc = MLFeatureService(session)
        results = await svc.test_situational_features(sport, limit=limit)
        
        print(f"\n{'=' * 70}")
        print(f"SITUATIONAL FEATURES - {sport}")
        print("=" * 70)
        
        if not results:
            print(f"   ⚠️ No games found")
            return
        
        for r in results:
            print(f"\n🎯 {r['game']} ({r['date']})")
            print(f"   Home Streak: {r['home_streak']}")
            print(f"   Away Streak: {r['away_streak']}")
            print(f"   Home Revenge: {r['home_revenge']}")
            print(f"   Away Revenge: {r['away_revenge']}")
            print(f"   Home Game #: {r['home_game_num']}")
            print(f"   Away Game #: {r['away_game_num']}")


async def test_all_dimensions(sport: str, limit: int):
    """Test all dimensions for one sport (console output)."""
    await test_team_features(sport, limit)
    await test_game_context_features(sport, limit)
    await test_player_features(sport, limit)
    await test_odds_features(sport, limit)
    await test_situational_features(sport, limit)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML Feature Extraction & CSV Export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export 8 CSV files for one sport
  python -m scripts.export_ml_features --export --sport NBA --limit 1000

  # Export 81 CSV files for all 10 sports (RECOMMENDED)
  python -m scripts.export_ml_features --export --sport ALL --limit 1000

  # Test team features (console only, no CSV)
  python -m scripts.export_ml_features --team --sport NBA --limit 5
  
  # Test all dimensions (console only, no CSV)
  python -m scripts.export_ml_features --all --sport NFL --limit 5
        """
    )
    
    parser.add_argument('--sport', type=str, default='ALL',
                        help='Sport: NFL, NBA, MLB, NHL, WNBA, CFL, NCAAF, NCAAB, ATP, WTA, or ALL')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max games per sport (default: 1000)')
    
    # Export mode (saves CSV files)
    parser.add_argument('--export', action='store_true',
                        help='Export to CSV files (8 per sport, 81 total for ALL)')
    
    # Test modes (console output only)
    parser.add_argument('--team', action='store_true', help='Test team features (console)')
    parser.add_argument('--context', action='store_true', help='Test game context (console)')
    parser.add_argument('--player', action='store_true', help='Test player features (console)')
    parser.add_argument('--odds', action='store_true', help='Test odds features (console)')
    parser.add_argument('--situational', action='store_true', help='Test situational (console)')
    parser.add_argument('--all', action='store_true', help='Test all dimensions (console)')
    
    args = parser.parse_args()
    
    sport = args.sport.upper()
    
    # Export mode - save CSV files
    if args.export:
        if sport == 'ALL':
            asyncio.run(export_all_sports(args.limit))
        else:
            asyncio.run(export_single_sport(sport, args.limit))
        return
    
    # Test modes - console output only
    sports_to_test = ALL_SPORTS if sport == 'ALL' else [sport]
    
    print(f"\n{'=' * 70}")
    print(f"ML FEATURE EXTRACTION TEST (Console Only)")
    print("=" * 70)
    print(f"Sports: {', '.join(sports_to_test)}")
    print(f"Limit per sport: {args.limit}")
    print(f"\nTip: Use --export to save CSV files")
    
    if args.team:
        for s in sports_to_test:
            asyncio.run(test_team_features(s, args.limit))
    elif args.context:
        for s in sports_to_test:
            asyncio.run(test_game_context_features(s, args.limit))
    elif args.player:
        for s in sports_to_test:
            asyncio.run(test_player_features(s, args.limit))
    elif args.odds:
        for s in sports_to_test:
            asyncio.run(test_odds_features(s, args.limit))
    elif args.situational:
        for s in sports_to_test:
            asyncio.run(test_situational_features(s, args.limit))
    elif args.all:
        for s in sports_to_test:
            asyncio.run(test_all_dimensions(s, args.limit))
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 70)
        print("QUICK START:")
        print("=" * 70)
        print("\n  # Export all 81 CSV files:")
        print("  python -m scripts.export_ml_features --export --sport ALL --limit 1000")
        print("\n  # Export 8 CSV files for NBA only:")
        print("  python -m scripts.export_ml_features --export --sport NBA --limit 1000")


if __name__ == "__main__":
    main()