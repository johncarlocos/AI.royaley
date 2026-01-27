#!/usr/bin/env python3
"""
AI PRO SPORTS - Feature Validation and Build Script (v5)
Uses DELETE+INSERT pattern instead of ON CONFLICT (no unique constraint needed)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Set
from uuid import UUID, uuid4

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    print("SQLAlchemy not found. Install with: pip install sqlalchemy[asyncio] asyncpg")
    sys.exit(1)

import os
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://ai_sports:ai_sports_dev@localhost:5432/ai_sports"
)

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


class SchemaValidator:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.errors = []
        self.warnings = []
    
    async def validate_all(self) -> bool:
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
        return await self._check_columns('games', ['id', 'sport_id', 'home_team_id', 'away_team_id',
                                                    'home_score', 'away_score', 'status', 'scheduled_at'])
    
    async def validate_teams_table(self) -> bool:
        return await self._check_columns('teams', ['id', 'sport_id', 'external_id', 'name', 'abbreviation', 'elo_rating'])
    
    async def validate_odds_table(self) -> bool:
        return await self._check_columns('odds', ['id', 'game_id', 'bet_type', 'home_line', 'away_line',
                                                   'home_odds', 'away_odds', 'total', 'is_opening', 'recorded_at'])
    
    async def validate_team_stats_table(self) -> bool:
        if not await self._check_columns('team_stats', ['id', 'team_id', 'stat_type', 'value']):
            return False
        result = await self.session.execute(text("SELECT COUNT(DISTINCT stat_type) FROM team_stats"))
        if (result.scalar() or 0) == 0:
            self.warnings.append("team_stats table has no data")
        return True
    
    async def validate_game_features_table(self) -> bool:
        if not await self._check_columns('game_features', ['id', 'game_id', 'features']):
            return False
        existing = await self._get_columns('game_features')
        print(f"    (Detected columns: {existing})")
        
        # Check for unique constraint on game_id
        result = await self.session.execute(text("""
            SELECT COUNT(*) FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'game_features' 
            AND c.contype = 'u'
            AND EXISTS (
                SELECT 1 FROM pg_attribute a 
                WHERE a.attrelid = t.oid AND a.attname = 'game_id'
                AND a.attnum = ANY(c.conkey)
            )
        """))
        if (result.scalar() or 0) == 0:
            self.warnings.append("game_features.game_id has no UNIQUE constraint (using DELETE+INSERT)")
        
        return True
    
    async def validate_weather_table(self) -> bool:
        result = await self.session.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'weather_data')"
        ))
        if not result.scalar():
            self.warnings.append("weather_data table does not exist")
            return True
        return await self._check_columns('weather_data', ['id', 'game_id'])
    
    async def validate_relationships(self) -> bool:
        result = await self.session.execute(text("""
            SELECT COUNT(*) FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            WHERE ht.id IS NULL OR at.id IS NULL
        """))
        orphan = result.scalar() or 0
        if orphan > 0:
            self.errors.append(f"{orphan} games have missing team references")
            return False
        return True
    
    async def _get_columns(self, table: str) -> Set[str]:
        result = await self.session.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :table"
        ), {"table": table})
        return {row[0] for row in result.fetchall()}
    
    async def _check_columns(self, table: str, columns: List[str]) -> bool:
        existing = await self._get_columns(table)
        missing = set(columns) - existing
        if missing:
            self.errors.append(f"{table} missing columns: {missing}")
            return False
        return True


class FeatureBuilder:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stats_cache = {}
        self.game_features_columns = set()
    
    async def _detect_schema(self):
        result = await self.session.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'game_features'"
        ))
        self.game_features_columns = {row[0] for row in result.fetchall()}
        print(f"  game_features columns: {self.game_features_columns}")
    
    async def build_features(self, sport_code: str, limit: int = None, force_rebuild: bool = False) -> Dict[str, Any]:
        print("\n" + "="*60)
        print(f"BUILDING FEATURES FOR {sport_code}")
        print("="*60)
        
        await self._detect_schema()
        
        result = await self.session.execute(
            text("SELECT id FROM sports WHERE code = :code"), {"code": sport_code}
        )
        sport_row = result.fetchone()
        if not sport_row:
            return {'error': f"Sport {sport_code} not found", 'built': 0, 'failed': 0}
        sport_id = sport_row[0]
        
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
        
        built = failed = 0
        errors = []
        
        for game in games:
            game_id = game[0]
            try:
                features = await self._build_game_features(game, sport_code)
                await self._save_features(game_id, features)
                await self.session.commit()
                built += 1
                if built % 50 == 0:
                    print(f"  Progress: {built}/{len(games)} games")
            except Exception as e:
                await self.session.rollback()
                failed += 1
                errors.append(f"Game {game_id}: {str(e)[:200]}")
                if failed <= 3:
                    print(f"\n  ✗ Error on game {game_id}:\n    {str(e)[:300]}")
        
        print(f"\nResults: {built} built, {failed} failed")
        return {'sport': sport_code, 'built': built, 'failed': failed, 'errors': errors[:10]}
    
    async def _build_game_features(self, game_row, sport_code: str) -> Dict[str, Any]:
        game_id, home_team_id, away_team_id, scheduled_at, home_score, away_score, venue_id, status = game_row
        
        features = {
            'meta': {
                'sport': sport_code,
                'game_id': str(game_id),
                'computed_at': datetime.utcnow().isoformat(),
                'version': '2.0'
            }
        }
        
        home_stats = await self._get_team_stats(home_team_id)
        away_stats = await self._get_team_stats(away_team_id)
        
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
        
        features['differentials'] = {
            'elo_diff': features['home_team']['elo_rating'] - features['away_team']['elo_rating'],
            'ppg_diff': features['home_team']['points_per_game'] - features['away_team']['points_per_game'],
            'net_rating_diff': features['home_team']['net_rating'] - features['away_team']['net_rating'],
        }
        
        features['odds'] = await self._get_odds_data(game_id)
        
        if SPORT_CONFIGS.get(sport_code, {}).get('outdoor', False):
            features['weather'] = await self._get_weather_data(game_id, venue_id)
        
        features['rest'] = {
            'home_rest_days': await self._get_rest_days(home_team_id, scheduled_at),
            'away_rest_days': await self._get_rest_days(away_team_id, scheduled_at),
        }
        features['rest']['rest_advantage'] = features['rest']['home_rest_days'] - features['rest']['away_rest_days']
        
        features['form'] = {
            'home_last5_wins': await self._get_recent_wins(home_team_id, scheduled_at, 5),
            'away_last5_wins': await self._get_recent_wins(away_team_id, scheduled_at, 5),
        }
        
        features['h2h'] = await self._get_h2h_record(home_team_id, away_team_id, scheduled_at)
        
        return features
    
    async def _get_team_stats(self, team_id: UUID) -> Dict[str, float]:
        cache_key = str(team_id)
        if cache_key in self.stats_cache:
            return self.stats_cache[cache_key]
        result = await self.session.execute(
            text("SELECT stat_type, value FROM team_stats WHERE team_id = :team_id"),
            {"team_id": team_id}
        )
        stats = {row[0]: row[1] for row in result.fetchall()}
        self.stats_cache[cache_key] = stats
        return stats
    
    async def _get_elo(self, team_id: UUID) -> float:
        result = await self.session.execute(
            text("SELECT elo_rating FROM teams WHERE id = :tid"), {"tid": team_id}
        )
        row = result.fetchone()
        return row[0] if row else 1500.0
    
    async def _get_odds_data(self, game_id: UUID) -> Dict[str, Any]:
        result = await self.session.execute(text("""
            SELECT home_line, away_line, home_odds, away_odds, total, over_odds, under_odds, is_opening
            FROM odds WHERE game_id = :gid ORDER BY recorded_at ASC
        """), {"gid": game_id})
        rows = result.fetchall()
        if not rows:
            return {}
        opening = next((r for r in rows if r[7]), rows[0])
        current = rows[-1]
        return {
            'current_spread': current[0], 'current_total': current[4],
            'home_odds': current[2], 'away_odds': current[3],
            'over_odds': current[5], 'under_odds': current[6],
            'opening_spread': opening[0], 'opening_total': opening[4],
            'spread_movement': (current[0] or 0) - (opening[0] or 0),
            'total_movement': (current[4] or 0) - (opening[4] or 0),
        }
    
    async def _get_weather_data(self, game_id: UUID, venue_id: UUID) -> Dict[str, Any]:
        try:
            result = await self.session.execute(text("""
                SELECT temperature_f, wind_speed_mph, humidity_pct, precipitation_pct, is_dome
                FROM weather_data WHERE game_id = :gid ORDER BY recorded_at DESC LIMIT 1
            """), {"gid": game_id})
            row = result.fetchone()
            if row:
                return {'temperature': row[0], 'wind_speed': row[1], 'humidity': row[2], 
                        'precipitation': row[3], 'is_dome': row[4]}
        except:
            pass
        if venue_id:
            try:
                result = await self.session.execute(
                    text("SELECT is_dome FROM venues WHERE id = :vid"), {"vid": venue_id}
                )
                venue = result.fetchone()
                if venue and venue[0]:
                    return {'is_dome': True, 'temperature': 72, 'wind_speed': 0, 'humidity': 50, 'precipitation': 0}
            except:
                pass
        return {'is_dome': False}
    
    async def _get_rest_days(self, team_id: UUID, before_date: datetime) -> int:
        result = await self.session.execute(text("""
            SELECT MAX(scheduled_at) FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
            AND scheduled_at < :before AND status = 'final'
        """), {"tid": team_id, "before": before_date})
        last_game = result.scalar()
        return (before_date - last_game).days if last_game else 7
    
    async def _get_recent_wins(self, team_id: UUID, before_date: datetime, n: int) -> int:
        result = await self.session.execute(text("""
            SELECT CASE WHEN home_team_id = :tid AND home_score > away_score THEN 1
                        WHEN away_team_id = :tid AND away_score > home_score THEN 1 ELSE 0 END
            FROM games WHERE (home_team_id = :tid OR away_team_id = :tid)
            AND scheduled_at < :before AND status = 'final'
            ORDER BY scheduled_at DESC LIMIT :n
        """), {"tid": team_id, "before": before_date, "n": n})
        return sum(row[0] for row in result.fetchall())
    
    async def _get_h2h_record(self, home_team_id: UUID, away_team_id: UUID, before_date: datetime) -> Dict[str, Any]:
        result = await self.session.execute(text("""
            SELECT home_team_id, home_score, away_score FROM games
            WHERE ((home_team_id = :ht AND away_team_id = :at) OR (home_team_id = :at AND away_team_id = :ht))
            AND scheduled_at < :before AND status = 'final'
            ORDER BY scheduled_at DESC LIMIT 10
        """), {"ht": home_team_id, "at": away_team_id, "before": before_date})
        wins = losses = total_margin = 0
        for row in result.fetchall():
            if row[1] is None or row[2] is None:
                continue
            if row[0] == home_team_id:
                if row[1] > row[2]: wins += 1
                else: losses += 1
                total_margin += row[1] - row[2]
            else:
                if row[2] > row[1]: wins += 1
                else: losses += 1
                total_margin += row[2] - row[1]
        total = wins + losses
        return {
            'wins': wins, 'losses': losses,
            'win_pct': wins / total if total > 0 else 0.5,
            'avg_margin': total_margin / total if total > 0 else 0,
            'total_meetings': total
        }
    
    async def _save_features(self, game_id: UUID, features: Dict):
        """Save features using DELETE + INSERT (no unique constraint needed)."""
        # Always delete first to handle both new and existing
        await self.session.execute(
            text("DELETE FROM game_features WHERE game_id = :gid"), {"gid": game_id}
        )
        
        new_id = str(uuid4())
        features_json = json.dumps(features)
        
        has_computed_at = 'computed_at' in self.game_features_columns
        has_home_features = 'home_features' in self.game_features_columns
        has_away_features = 'away_features' in self.game_features_columns
        
        if has_home_features and has_away_features and has_computed_at:
            home_features_json = json.dumps(features.get('home_team', {}))
            away_features_json = json.dumps(features.get('away_team', {}))
            await self.session.execute(text("""
                INSERT INTO game_features (id, game_id, features, home_features, away_features, computed_at)
                VALUES (:id, :gid, :features, :home_features, :away_features, NOW())
            """), {
                "id": new_id, "gid": game_id, "features": features_json,
                "home_features": home_features_json, "away_features": away_features_json
            })
        elif has_computed_at:
            await self.session.execute(text("""
                INSERT INTO game_features (id, game_id, features, computed_at)
                VALUES (:id, :gid, :features, NOW())
            """), {"id": new_id, "gid": game_id, "features": features_json})
        else:
            await self.session.execute(text("""
                INSERT INTO game_features (id, game_id, features)
                VALUES (:id, :gid, :features)
            """), {"id": new_id, "gid": game_id, "features": features_json})


async def main():
    parser = argparse.ArgumentParser(description='AI PRO SPORTS Feature Builder (v5 - DELETE+INSERT)')
    parser.add_argument('--validate', action='store_true', help='Validate database schema')
    parser.add_argument('--build', action='store_true', help='Build features')
    parser.add_argument('--sport', '-s', type=str, help='Sport code (e.g., NFL, NBA)')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of games')
    parser.add_argument('--force', '-f', action='store_true', help='Force rebuild')
    parser.add_argument('--all-sports', action='store_true', help='Process all sports')
    
    args = parser.parse_args()
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        if args.validate:
            validator = SchemaValidator(session)
            success = await validator.validate_all()
            sys.exit(0 if success else 1)
        
        elif args.build:
            builder = FeatureBuilder(session)
            
            if args.all_sports:
                sports = list(SPORT_CONFIGS.keys())
            elif args.sport:
                sports = [args.sport.upper()]
            else:
                print("Error: Specify --sport or --all-sports")
                sys.exit(1)
            
            total_built = total_failed = 0
            for sport in sports:
                result = await builder.build_features(sport, args.limit, args.force)
                total_built += result.get('built', 0)
                total_failed += result.get('failed', 0)
            
            print(f"\n{'='*60}")
            print(f"TOTAL: {total_built} features built, {total_failed} failed")
            sys.exit(0 if total_failed == 0 else 1)
        
        else:
            parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())