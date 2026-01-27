#!/usr/bin/env python3
"""
AI PRO SPORTS - Feature Validation and Build Script (CORRECTED v2)
Validates database schema and builds game_features for ML training.

SCHEMA CORRECTIONS APPLIED:
- odds table: home_line/away_line (NOT home_spread/away_spread)
- team_stats: individual rows with stat_type/value (NOT JSONB stats column)
- game_features: game_id has unique constraint for upsert
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

# Handle both direct execution and Docker execution
try:
    from sqlalchemy import text, select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    print("SQLAlchemy not found. Install with: pip install sqlalchemy[asyncio] asyncpg")
    sys.exit(1)

# Configuration
import os
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://ai_sports:ai_sports_dev@localhost:5432/ai_sports"
)

# Sport configurations with feature counts
SPORT_CONFIGS = {
    'NFL': {'feature_count': 75, 'outdoor': True},
    'NCAAF': {'feature_count': 70, 'outdoor': True},
    'CFL': {'feature_count': 65, 'outdoor': True},
    'NBA': {'feature_count': 80, 'outdoor': False},
    'NCAAB': {'feature_count': 70, 'outdoor': False},
    'WNBA': {'feature_count': 70, 'outdoor': False},
    'NHL': {'feature_count': 75, 'outdoor': False},
    'MLB': {'feature_count': 85, 'outdoor': True},
    'ATP': {'feature_count': 60, 'outdoor': True},
    'WTA': {'feature_count': 60, 'outdoor': True},
}

# Required stat types for team_stats table
REQUIRED_STAT_TYPES = [
    'wins', 'losses', 'points_per_game', 'points_allowed_per_game',
    'offensive_rating', 'defensive_rating', 'net_rating', 'pace'
]


class SchemaValidator:
    """Validates database schema matches expected structure."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.errors = []
        self.warnings = []
    
    async def validate_all(self) -> bool:
        """Run all validation checks."""
        print("\n" + "="*60)
        print("DATABASE SCHEMA VALIDATION")
        print("="*60)
        
        checks = [
            ("Games Table", self.validate_games_table),
            ("Teams Table", self.validate_teams_table),
            ("Odds Table", self.validate_odds_table),
            ("Team Stats Table", self.validate_team_stats_table),
            ("Game Features Table", self.validate_game_features_table),
            ("Weather Data Table", self.validate_weather_table),
            ("Foreign Key Relationships", self.validate_relationships),
        ]
        
        all_passed = True
        for name, check_func in checks:
            try:
                result = await check_func()
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {status}: {name}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"  ✗ ERROR: {name} - {str(e)}")
                self.errors.append(f"{name}: {str(e)}")
                all_passed = False
        
        # Print summary
        print("\n" + "-"*60)
        if self.warnings:
            print("WARNINGS:")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        
        if self.errors:
            print("ERRORS:")
            for e in self.errors:
                print(f"  ✗ {e}")
        
        print("-"*60)
        print(f"Validation {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    async def validate_games_table(self) -> bool:
        """Validate games table has required columns."""
        required_cols = [
            'id', 'sport_id', 'home_team_id', 'away_team_id',
            'home_score', 'away_score', 'status', 'scheduled_at'
        ]
        return await self._check_columns('games', required_cols)
    
    async def validate_teams_table(self) -> bool:
        """Validate teams table has required columns."""
        required_cols = [
            'id', 'sport_id', 'external_id', 'name', 'abbreviation', 'elo_rating'
        ]
        return await self._check_columns('teams', required_cols)
    
    async def validate_odds_table(self) -> bool:
        """Validate odds table - CORRECTED to use home_line/away_line."""
        required_cols = [
            'id', 'game_id', 'bet_type',
            'home_line', 'away_line',  # CORRECTED: not home_spread/away_spread
            'home_odds', 'away_odds',
            'total', 'over_odds', 'under_odds',
            'is_opening', 'recorded_at'
        ]
        return await self._check_columns('odds', required_cols)
    
    async def validate_team_stats_table(self) -> bool:
        """Validate team_stats table - CORRECTED: individual rows, not JSONB."""
        # Check columns exist
        required_cols = ['id', 'team_id', 'stat_type', 'value', 'games_played']
        if not await self._check_columns('team_stats', required_cols):
            return False
        
        # Check we have stats data
        result = await self.session.execute(
            text("SELECT COUNT(DISTINCT stat_type) FROM team_stats")
        )
        count = result.scalar() or 0
        
        if count == 0:
            self.warnings.append("team_stats table has no data")
        elif count < 5:
            self.warnings.append(f"team_stats only has {count} stat types")
        
        return True
    
    async def validate_game_features_table(self) -> bool:
        """Validate game_features table has unique constraint on game_id."""
        required_cols = ['id', 'game_id', 'features', 'feature_version', 'computed_at']
        if not await self._check_columns('game_features', required_cols):
            return False
        
        # Check for unique constraint on game_id
        result = await self.session.execute(text("""
            SELECT COUNT(*) FROM pg_indexes 
            WHERE tablename = 'game_features' 
            AND indexdef LIKE '%UNIQUE%game_id%'
        """))
        unique_count = result.scalar() or 0
        
        # Also check constraints directly
        result2 = await self.session.execute(text("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE table_name = 'game_features' 
            AND constraint_type = 'UNIQUE'
        """))
        constraint_count = result2.scalar() or 0
        
        if unique_count == 0 and constraint_count == 0:
            self.warnings.append(
                "game_features.game_id may not have UNIQUE constraint - "
                "upserts may fail. Will use INSERT with conflict check."
            )
        
        return True
    
    async def validate_weather_table(self) -> bool:
        """Validate weather_data table if it exists."""
        # Check if table exists
        result = await self.session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'weather_data'
            )
        """))
        exists = result.scalar()
        
        if not exists:
            self.warnings.append("weather_data table does not exist")
            return True  # Not critical
        
        required_cols = [
            'id', 'game_id', 'temperature_f', 'wind_speed_mph',
            'humidity_pct', 'precipitation_pct', 'is_dome'
        ]
        return await self._check_columns('weather_data', required_cols)
    
    async def validate_relationships(self) -> bool:
        """Validate foreign key relationships."""
        # Check games -> teams relationship
        result = await self.session.execute(text("""
            SELECT COUNT(*) FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            WHERE ht.id IS NULL OR at.id IS NULL
        """))
        orphan_games = result.scalar() or 0
        
        if orphan_games > 0:
            self.errors.append(f"{orphan_games} games have missing team references")
            return False
        
        return True
    
    async def _check_columns(self, table: str, columns: List[str]) -> bool:
        """Check if table has all required columns."""
        result = await self.session.execute(text(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :table
        """), {"table": table})
        
        existing_cols = {row[0] for row in result.fetchall()}
        missing = set(columns) - existing_cols
        
        if missing:
            self.errors.append(f"{table} missing columns: {missing}")
            return False
        return True


class DataReadinessChecker:
    """Checks if data is ready for feature building."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def check_readiness(self, sport_code: str = None) -> Dict[str, Any]:
        """Check data readiness for feature building."""
        print("\n" + "="*60)
        print("DATA READINESS CHECK")
        print("="*60)
        
        report = {
            'ready': True,
            'sports': {},
            'issues': []
        }
        
        # Get sports to check
        if sport_code:
            sports = [sport_code]
        else:
            result = await self.session.execute(
                text("SELECT code FROM sports WHERE is_active = true")
            )
            sports = [row[0] for row in result.fetchall()]
        
        for sport in sports:
            sport_report = await self._check_sport_readiness(sport)
            report['sports'][sport] = sport_report
            if not sport_report['ready']:
                report['ready'] = False
                report['issues'].extend(sport_report['issues'])
        
        # Print report
        for sport, data in report['sports'].items():
            status = "✓" if data['ready'] else "✗"
            print(f"\n{status} {sport}:")
            print(f"    Games: {data['total_games']} total, {data['completed_games']} completed")
            print(f"    With odds: {data['games_with_odds']}")
            print(f"    With team stats: {data['games_with_stats']}")
            print(f"    Already have features: {data['games_with_features']}")
            print(f"    Ready to build: {data['buildable_games']}")
            
            if data['issues']:
                for issue in data['issues']:
                    print(f"    ⚠ {issue}")
        
        return report
    
    async def _check_sport_readiness(self, sport_code: str) -> Dict[str, Any]:
        """Check readiness for a specific sport."""
        report = {
            'ready': True,
            'total_games': 0,
            'completed_games': 0,
            'games_with_odds': 0,
            'games_with_stats': 0,
            'games_with_features': 0,
            'buildable_games': 0,
            'issues': []
        }
        
        # Get sport_id
        result = await self.session.execute(
            text("SELECT id FROM sports WHERE code = :code"),
            {"code": sport_code}
        )
        sport_row = result.fetchone()
        if not sport_row:
            report['ready'] = False
            report['issues'].append(f"Sport {sport_code} not found in database")
            return report
        
        sport_id = sport_row[0]
        
        # Count total games
        result = await self.session.execute(
            text("SELECT COUNT(*) FROM games WHERE sport_id = :sid"),
            {"sid": sport_id}
        )
        report['total_games'] = result.scalar() or 0
        
        # Count completed games
        result = await self.session.execute(
            text("SELECT COUNT(*) FROM games WHERE sport_id = :sid AND status = 'final'"),
            {"sid": sport_id}
        )
        report['completed_games'] = result.scalar() or 0
        
        # Count games with odds (CORRECTED: check for home_line existence)
        result = await self.session.execute(text("""
            SELECT COUNT(DISTINCT g.id) FROM games g
            JOIN odds o ON g.id = o.game_id
            WHERE g.sport_id = :sid AND o.home_line IS NOT NULL
        """), {"sid": sport_id})
        report['games_with_odds'] = result.scalar() or 0
        
        # Count games with team stats (CORRECTED: query team_stats properly)
        result = await self.session.execute(text("""
            SELECT COUNT(DISTINCT g.id) FROM games g
            WHERE g.sport_id = :sid
            AND EXISTS (
                SELECT 1 FROM team_stats ts 
                WHERE ts.team_id = g.home_team_id
                AND ts.stat_type IN ('wins', 'losses', 'points_per_game')
            )
            AND EXISTS (
                SELECT 1 FROM team_stats ts 
                WHERE ts.team_id = g.away_team_id
                AND ts.stat_type IN ('wins', 'losses', 'points_per_game')
            )
        """), {"sid": sport_id})
        report['games_with_stats'] = result.scalar() or 0
        
        # Count games with features already
        result = await self.session.execute(text("""
            SELECT COUNT(*) FROM games g
            JOIN game_features gf ON g.id = gf.game_id
            WHERE g.sport_id = :sid
        """), {"sid": sport_id})
        report['games_with_features'] = result.scalar() or 0
        
        # Count buildable games (have odds, don't have features yet)
        result = await self.session.execute(text("""
            SELECT COUNT(DISTINCT g.id) FROM games g
            JOIN odds o ON g.id = o.game_id
            LEFT JOIN game_features gf ON g.id = gf.game_id
            WHERE g.sport_id = :sid 
            AND o.home_line IS NOT NULL
            AND gf.id IS NULL
        """), {"sid": sport_id})
        report['buildable_games'] = result.scalar() or 0
        
        # Check for issues
        if report['total_games'] == 0:
            report['ready'] = False
            report['issues'].append("No games found")
        
        if report['games_with_odds'] == 0:
            report['ready'] = False
            report['issues'].append("No games have odds data")
        
        if report['games_with_stats'] < report['total_games'] * 0.5:
            report['issues'].append("Less than 50% of games have team stats")
        
        return report


class FeatureBuilder:
    """Builds game_features from database data."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stats_cache = {}  # Cache team stats
    
    async def build_features(
        self,
        sport_code: str,
        limit: int = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """Build features for games."""
        print("\n" + "="*60)
        print(f"BUILDING FEATURES FOR {sport_code}")
        print("="*60)
        
        # Get sport_id
        result = await self.session.execute(
            text("SELECT id FROM sports WHERE code = :code"),
            {"code": sport_code}
        )
        sport_row = result.fetchone()
        if not sport_row:
            return {'error': f"Sport {sport_code} not found", 'built': 0, 'failed': 0}
        
        sport_id = sport_row[0]
        
        # Get games to process
        query = """
            SELECT DISTINCT g.id, g.home_team_id, g.away_team_id, g.scheduled_at,
                   g.home_score, g.away_score, g.venue_id, g.status
            FROM games g
            JOIN odds o ON g.id = o.game_id
        """
        
        if not force_rebuild:
            query += " LEFT JOIN game_features gf ON g.id = gf.game_id WHERE gf.id IS NULL AND"
        else:
            query += " WHERE"
        
        query += " g.sport_id = :sid AND o.home_line IS NOT NULL ORDER BY g.scheduled_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = await self.session.execute(text(query), {"sid": sport_id})
        games = result.fetchall()
        
        print(f"Found {len(games)} games to process")
        
        built = 0
        failed = 0
        errors = []
        
        for game in games:
            game_id = game[0]
            try:
                features = await self._build_game_features(game, sport_code)
                await self._save_features(game_id, features, force_rebuild)
                built += 1
                
                if built % 100 == 0:
                    print(f"  Progress: {built}/{len(games)} games")
                    await self.session.commit()
                    
            except Exception as e:
                failed += 1
                errors.append(f"Game {game_id}: {str(e)}")
                if failed <= 5:  # Only show first 5 errors
                    print(f"  ✗ Error on game {game_id}: {str(e)}")
        
        await self.session.commit()
        
        print(f"\nResults: {built} built, {failed} failed")
        
        return {
            'sport': sport_code,
            'built': built,
            'failed': failed,
            'errors': errors[:10]  # First 10 errors only
        }
    
    async def _build_game_features(self, game_row, sport_code: str) -> Dict[str, Any]:
        """Build features for a single game."""
        game_id, home_team_id, away_team_id, scheduled_at, home_score, away_score, venue_id, status = game_row
        
        features = {
            'meta': {
                'sport': sport_code,
                'game_id': str(game_id),
                'computed_at': datetime.utcnow().isoformat(),
                'version': '2.0'
            }
        }
        
        # Get team stats (CORRECTED: query individual rows)
        home_stats = await self._get_team_stats(home_team_id)
        away_stats = await self._get_team_stats(away_team_id)
        
        # Team performance features
        features['home_team'] = {
            'elo_rating': await self._get_elo(home_team_id),
            'wins': home_stats.get('wins', 0),
            'losses': home_stats.get('losses', 0),
            'points_per_game': home_stats.get('points_per_game', 0),
            'points_allowed': home_stats.get('points_allowed_per_game', 0),
            'offensive_rating': home_stats.get('offensive_rating', 100),
            'defensive_rating': home_stats.get('defensive_rating', 100),
            'net_rating': home_stats.get('net_rating', 0),
            'pace': home_stats.get('pace', 100),
        }
        
        features['away_team'] = {
            'elo_rating': await self._get_elo(away_team_id),
            'wins': away_stats.get('wins', 0),
            'losses': away_stats.get('losses', 0),
            'points_per_game': away_stats.get('points_per_game', 0),
            'points_allowed': away_stats.get('points_allowed_per_game', 0),
            'offensive_rating': away_stats.get('offensive_rating', 100),
            'defensive_rating': away_stats.get('defensive_rating', 100),
            'net_rating': away_stats.get('net_rating', 0),
            'pace': away_stats.get('pace', 100),
        }
        
        # Calculated differentials
        features['differentials'] = {
            'elo_diff': features['home_team']['elo_rating'] - features['away_team']['elo_rating'],
            'ppg_diff': features['home_team']['points_per_game'] - features['away_team']['points_per_game'],
            'net_rating_diff': features['home_team']['net_rating'] - features['away_team']['net_rating'],
        }
        
        # Get odds data (CORRECTED: use home_line/away_line)
        odds_data = await self._get_odds_data(game_id)
        features['odds'] = odds_data
        
        # Get weather if outdoor sport
        if SPORT_CONFIGS.get(sport_code, {}).get('outdoor', False):
            weather = await self._get_weather_data(game_id, venue_id)
            features['weather'] = weather
        
        # Rest days calculation
        features['rest'] = {
            'home_rest_days': await self._get_rest_days(home_team_id, scheduled_at),
            'away_rest_days': await self._get_rest_days(away_team_id, scheduled_at),
        }
        features['rest']['rest_advantage'] = (
            features['rest']['home_rest_days'] - features['rest']['away_rest_days']
        )
        
        # Recent form (last 5 games)
        features['form'] = {
            'home_last5_wins': await self._get_recent_wins(home_team_id, scheduled_at, 5),
            'away_last5_wins': await self._get_recent_wins(away_team_id, scheduled_at, 5),
        }
        
        # Head-to-head
        h2h = await self._get_h2h_record(home_team_id, away_team_id, scheduled_at)
        features['h2h'] = h2h
        
        return features
    
    async def _get_team_stats(self, team_id: UUID) -> Dict[str, float]:
        """Get team stats from team_stats table (CORRECTED: individual rows)."""
        # Check cache first
        cache_key = str(team_id)
        if cache_key in self.stats_cache:
            return self.stats_cache[cache_key]
        
        # Query individual stat rows and pivot them
        result = await self.session.execute(text("""
            SELECT stat_type, value FROM team_stats
            WHERE team_id = :team_id
        """), {"team_id": team_id})
        
        stats = {}
        for row in result.fetchall():
            stat_type, value = row
            stats[stat_type] = value
        
        self.stats_cache[cache_key] = stats
        return stats
    
    async def _get_elo(self, team_id: UUID) -> float:
        """Get team ELO rating."""
        result = await self.session.execute(
            text("SELECT elo_rating FROM teams WHERE id = :tid"),
            {"tid": team_id}
        )
        row = result.fetchone()
        return row[0] if row else 1500.0
    
    async def _get_odds_data(self, game_id: UUID) -> Dict[str, Any]:
        """Get odds data for game (CORRECTED: use home_line/away_line)."""
        # Get opening odds
        result = await self.session.execute(text("""
            SELECT home_line, away_line, home_odds, away_odds, total, over_odds, under_odds
            FROM odds
            WHERE game_id = :gid AND is_opening = true
            ORDER BY recorded_at ASC LIMIT 1
        """), {"gid": game_id})
        opening = result.fetchone()
        
        # Get current/latest odds
        result = await self.session.execute(text("""
            SELECT home_line, away_line, home_odds, away_odds, total, over_odds, under_odds
            FROM odds
            WHERE game_id = :gid
            ORDER BY recorded_at DESC LIMIT 1
        """), {"gid": game_id})
        current = result.fetchone()
        
        if not current:
            return {}
        
        odds_data = {
            'current_spread': current[0],  # home_line
            'current_total': current[4],
            'home_odds': current[2],
            'away_odds': current[3],
            'over_odds': current[5],
            'under_odds': current[6],
        }
        
        if opening:
            odds_data['opening_spread'] = opening[0]
            odds_data['opening_total'] = opening[4]
            odds_data['spread_movement'] = (current[0] or 0) - (opening[0] or 0)
            odds_data['total_movement'] = (current[4] or 0) - (opening[4] or 0)
        
        return odds_data
    
    async def _get_weather_data(self, game_id: UUID, venue_id: UUID) -> Dict[str, Any]:
        """Get weather data for game."""
        # Try game-specific weather first
        result = await self.session.execute(text("""
            SELECT temperature_f, wind_speed_mph, humidity_pct, precipitation_pct, is_dome
            FROM weather_data WHERE game_id = :gid
            ORDER BY recorded_at DESC LIMIT 1
        """), {"gid": game_id})
        row = result.fetchone()
        
        if row:
            return {
                'temperature': row[0],
                'wind_speed': row[1],
                'humidity': row[2],
                'precipitation': row[3],
                'is_dome': row[4]
            }
        
        # Check venue for dome
        if venue_id:
            result = await self.session.execute(
                text("SELECT is_dome FROM venues WHERE id = :vid"),
                {"vid": venue_id}
            )
            venue = result.fetchone()
            if venue and venue[0]:
                return {'is_dome': True, 'temperature': 72, 'wind_speed': 0, 'humidity': 50, 'precipitation': 0}
        
        return {'is_dome': False, 'temperature': None, 'wind_speed': None, 'humidity': None, 'precipitation': None}
    
    async def _get_rest_days(self, team_id: UUID, before_date: datetime) -> int:
        """Calculate rest days since last game."""
        result = await self.session.execute(text("""
            SELECT MAX(scheduled_at) FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
            AND scheduled_at < :before
            AND status = 'final'
        """), {"tid": team_id, "before": before_date})
        
        last_game = result.scalar()
        if last_game:
            delta = before_date - last_game
            return delta.days
        return 7  # Default to 7 if no previous game
    
    async def _get_recent_wins(self, team_id: UUID, before_date: datetime, n: int) -> int:
        """Get number of wins in last N games."""
        result = await self.session.execute(text("""
            SELECT 
                CASE 
                    WHEN home_team_id = :tid AND home_score > away_score THEN 1
                    WHEN away_team_id = :tid AND away_score > home_score THEN 1
                    ELSE 0
                END as win
            FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
            AND scheduled_at < :before
            AND status = 'final'
            ORDER BY scheduled_at DESC
            LIMIT :n
        """), {"tid": team_id, "before": before_date, "n": n})
        
        return sum(row[0] for row in result.fetchall())
    
    async def _get_h2h_record(
        self, home_team_id: UUID, away_team_id: UUID, before_date: datetime
    ) -> Dict[str, Any]:
        """Get head-to-head record."""
        result = await self.session.execute(text("""
            SELECT home_team_id, home_score, away_score
            FROM games
            WHERE ((home_team_id = :ht AND away_team_id = :at) OR 
                   (home_team_id = :at AND away_team_id = :ht))
            AND scheduled_at < :before
            AND status = 'final'
            ORDER BY scheduled_at DESC
            LIMIT 10
        """), {"ht": home_team_id, "at": away_team_id, "before": before_date})
        
        wins = losses = 0
        total_margin = 0
        
        for row in result.fetchall():
            game_home_id, home_score, away_score = row
            if game_home_id == home_team_id:
                if home_score > away_score:
                    wins += 1
                else:
                    losses += 1
                total_margin += home_score - away_score
            else:
                if away_score > home_score:
                    wins += 1
                else:
                    losses += 1
                total_margin += away_score - home_score
        
        total_games = wins + losses
        return {
            'wins': wins,
            'losses': losses,
            'win_pct': wins / total_games if total_games > 0 else 0.5,
            'avg_margin': total_margin / total_games if total_games > 0 else 0,
            'total_meetings': total_games
        }
    
    async def _save_features(self, game_id: UUID, features: Dict, force: bool = False):
        """Save features to game_features table."""
        if force:
            # Delete existing first
            await self.session.execute(
                text("DELETE FROM game_features WHERE game_id = :gid"),
                {"gid": game_id}
            )
        
        # Insert new features
        await self.session.execute(text("""
            INSERT INTO game_features (id, game_id, features, feature_version, computed_at)
            VALUES (gen_random_uuid(), :gid, :features, '2.0', NOW())
            ON CONFLICT (game_id) DO UPDATE SET
                features = EXCLUDED.features,
                feature_version = EXCLUDED.feature_version,
                computed_at = EXCLUDED.computed_at
        """), {"gid": game_id, "features": json.dumps(features)})


async def main():
    parser = argparse.ArgumentParser(description='AI PRO SPORTS Feature Builder (CORRECTED v2)')
    parser.add_argument('--validate', action='store_true', help='Validate database schema')
    parser.add_argument('--check-readiness', action='store_true', help='Check data readiness')
    parser.add_argument('--build', action='store_true', help='Build features')
    parser.add_argument('--sport', '-s', type=str, help='Sport code (e.g., NFL, NBA)')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of games to process')
    parser.add_argument('--force', '-f', action='store_true', help='Force rebuild existing features')
    parser.add_argument('--all-sports', action='store_true', help='Process all sports')
    
    args = parser.parse_args()
    
    # Create database connection
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        if args.validate:
            validator = SchemaValidator(session)
            success = await validator.validate_all()
            sys.exit(0 if success else 1)
        
        elif args.check_readiness:
            checker = DataReadinessChecker(session)
            report = await checker.check_readiness(args.sport)
            sys.exit(0 if report['ready'] else 1)
        
        elif args.build:
            builder = FeatureBuilder(session)
            
            if args.all_sports:
                sports = list(SPORT_CONFIGS.keys())
            elif args.sport:
                sports = [args.sport.upper()]
            else:
                print("Error: Specify --sport or --all-sports")
                sys.exit(1)
            
            total_built = 0
            total_failed = 0
            
            for sport in sports:
                result = await builder.build_features(
                    sport,
                    limit=args.limit,
                    force_rebuild=args.force
                )
                total_built += result.get('built', 0)
                total_failed += result.get('failed', 0)
            
            print(f"\n{'='*60}")
            print(f"TOTAL: {total_built} features built, {total_failed} failed")
            sys.exit(0 if total_failed == 0 else 1)
        
        else:
            parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())