#!/usr/bin/env python3
"""
ROYALEY - Generate Features for ML Training
Phase 2: Transform collected data into ML-ready features

This script:
1. Reads game data from collectors (ESPN, OddsAPI, Pinnacle, etc.)
2. Computes features (ELO, team stats, H2H, momentum, weather)
3. Saves to GameFeature table for ML training
4. Optionally exports to parquet files

Usage:
    # Generate features for all games
    python generate_features.py --all
    
    # Generate features for specific sport
    python generate_features.py --sport NFL
    
    # Generate features for date range
    python generate_features.py --sport NBA --start 2024-01-01 --end 2024-12-31
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from app.core.config import settings
from app.core.database import db_manager
from app.models import (
    Sport, Team, Game, GameFeature, Odds, TeamStats, WeatherData,
    GameStatus, Injury, GameInjury
)
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class FeatureStats:
    """Statistics from feature generation."""
    sport: str
    games_processed: int = 0
    features_created: int = 0
    features_updated: int = 0
    errors: int = 0
    duration_seconds: float = 0.0


class FeatureGenerator:
    """
    Generate ML features from collected game data.
    
    Features computed:
    - ELO ratings (current and historical)
    - Team stats (win%, PPG, PPG allowed, home/away splits)
    - Head-to-head history
    - Momentum/form (recent results)
    - Rest days
    - Weather (for outdoor sports)
    - Injuries impact
    - Odds/market features (opening line, current line, movement)
    """
    
    OUTDOOR_SPORTS = ["NFL", "NCAAF", "CFL", "MLB"]
    
    def __init__(self):
        self.elo_ratings: Dict[str, Dict[str, float]] = {}  # sport -> team_id -> rating
        self.team_stats_cache: Dict[str, Dict] = {}
    
    async def generate_all_features(
        self,
        sport_code: str = None,
        start_date: date = None,
        end_date: date = None,
        overwrite: bool = False,
    ) -> List[FeatureStats]:
        """
        Generate features for all games.
        
        Args:
            sport_code: Specific sport (None = all)
            start_date: Start date filter
            end_date: End date filter
            overwrite: Overwrite existing features
            
        Returns:
            List of FeatureStats
        """
        await db_manager.initialize()
        
        stats_list = []
        
        async with db_manager.session() as session:
            # Get sports
            if sport_code:
                sports = [await self._get_sport(session, sport_code)]
                sports = [s for s in sports if s]
            else:
                result = await session.execute(select(Sport).where(Sport.is_active == True))
                sports = result.scalars().all()
            
            if not sports:
                console.print("[red]No sports found[/red]")
                return []
            
            for sport in sports:
                stats = await self._generate_sport_features(
                    session, sport, start_date, end_date, overwrite
                )
                stats_list.append(stats)
        
        return stats_list
    
    async def _generate_sport_features(
        self,
        session: AsyncSession,
        sport: Sport,
        start_date: date = None,
        end_date: date = None,
        overwrite: bool = False,
    ) -> FeatureStats:
        """Generate features for a single sport."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport.code)
        
        console.print(f"\n[cyan]Processing {sport.code}...[/cyan]")
        
        try:
            # Build query for games
            query = (
                select(Game)
                .where(Game.sport_id == sport.id)
                .order_by(Game.scheduled_at)
            )
            
            if start_date:
                query = query.where(Game.scheduled_at >= datetime.combine(start_date, datetime.min.time()))
            if end_date:
                query = query.where(Game.scheduled_at <= datetime.combine(end_date, datetime.max.time()))
            
            result = await session.execute(query)
            games = result.scalars().all()
            
            if not games:
                console.print(f"  [yellow]No games found for {sport.code}[/yellow]")
                return stats
            
            console.print(f"  [cyan]Found {len(games)} games[/cyan]")
            
            # Initialize ELO for this sport
            await self._initialize_elo(session, sport)
            
            # Process games
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Processing {sport.code}", total=len(games))
                
                batch_count = 0
                BATCH_SIZE = 50  # Commit every 50 games
                
                for game in games:
                    try:
                        features = await self._compute_game_features(
                            session, game, sport
                        )
                        
                        if features:
                            # Save or update
                            saved = await self._save_game_features(
                                session, game.id, features, overwrite
                            )
                            if saved == "created":
                                stats.features_created += 1
                            elif saved == "updated":
                                stats.features_updated += 1
                            
                            # Update ELO after game (if completed)
                            if game.status == GameStatus.FINAL and game.home_score is not None:
                                await self._update_elo_after_game(game, sport.code)
                        
                        stats.games_processed += 1
                        batch_count += 1
                        
                        # Commit in batches to prevent transaction buildup
                        if batch_count >= BATCH_SIZE:
                            try:
                                await session.commit()
                                batch_count = 0
                            except Exception as e:
                                logger.warning(f"Batch commit error: {e}")
                                await session.rollback()
                                batch_count = 0
                        
                    except Exception as e:
                        logger.warning(f"Error processing game {game.id}: {e}")
                        stats.errors += 1
                        # CRITICAL: Rollback the failed transaction to continue processing
                        try:
                            await session.rollback()
                            batch_count = 0
                        except:
                            pass
                    
                    progress.advance(task)
            
            # Commit any remaining changes
            try:
                await session.commit()
            except Exception as e:
                logger.warning(f"Final commit error: {e}")
                await session.rollback()
            
            # Save ELO ratings back to teams
            await self._save_elo_ratings(session, sport.code)
            
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(
                f"  [green]✓ {stats.features_created} created, "
                f"{stats.features_updated} updated, "
                f"{stats.errors} errors[/green]"
            )
            
        except Exception as e:
            logger.exception(f"Error generating features for {sport.code}: {e}")
            stats.errors += 1
        
        return stats
    
    async def _compute_game_features(
        self,
        session: AsyncSession,
        game: Game,
        sport: Sport,
    ) -> Optional[Dict[str, Any]]:
        """Compute all features for a single game."""
        features = {}
        
        try:
            # Get teams using query (more robust than session.get)
            home_result = await session.execute(
                select(Team).where(Team.id == game.home_team_id)
            )
            home_team = home_result.scalar_one_or_none()
            
            away_result = await session.execute(
                select(Team).where(Team.id == game.away_team_id)
            )
            away_team = away_result.scalar_one_or_none()
            
            if not home_team or not away_team:
                logger.debug(f"Missing team for game {game.id}: home={home_team is not None}, away={away_team is not None}")
                return None
            
            # 1. ELO Features
            elo_features = self._compute_elo_features(
                str(home_team.id), str(away_team.id), sport.code
            )
            features.update(elo_features)
            
            # 2. Team Stats Features
            home_stats = await self._get_team_stats(session, home_team.id, game.scheduled_at)
            away_stats = await self._get_team_stats(session, away_team.id, game.scheduled_at)
            
            features.update({
                # Home team stats
                "home_win_pct": home_stats.get("win_pct", 0.5),
                "home_games_played": home_stats.get("games", 0),
                "home_ppg": home_stats.get("ppg", 100),
                "home_ppg_allowed": home_stats.get("ppg_allowed", 100),
                "home_point_diff": home_stats.get("ppg", 100) - home_stats.get("ppg_allowed", 100),
                "home_home_win_pct": home_stats.get("home_win_pct", 0.5),
                
                # Away team stats
                "away_win_pct": away_stats.get("win_pct", 0.5),
                "away_games_played": away_stats.get("games", 0),
                "away_ppg": away_stats.get("ppg", 100),
                "away_ppg_allowed": away_stats.get("ppg_allowed", 100),
                "away_point_diff": away_stats.get("ppg", 100) - away_stats.get("ppg_allowed", 100),
                "away_away_win_pct": away_stats.get("away_win_pct", 0.5),
                
                # Differentials
                "win_pct_diff": home_stats.get("win_pct", 0.5) - away_stats.get("win_pct", 0.5),
                "ppg_diff": home_stats.get("ppg", 100) - away_stats.get("ppg", 100),
            })
            
            # 3. Momentum Features
            home_momentum = await self._get_momentum(session, home_team.id, game.scheduled_at)
            away_momentum = await self._get_momentum(session, away_team.id, game.scheduled_at)
            
            features.update({
                # Home momentum
                "home_form_last5": home_momentum.get("last5", 0.5),
                "home_form_last10": home_momentum.get("last10", 0.5),
                "home_streak": home_momentum.get("streak", 0),
                "home_rest_days": home_momentum.get("rest_days", 7),
                
                # Away momentum
                "away_form_last5": away_momentum.get("last5", 0.5),
                "away_form_last10": away_momentum.get("last10", 0.5),
                "away_streak": away_momentum.get("streak", 0),
                "away_rest_days": away_momentum.get("rest_days", 7),
                
                # Differentials
                "form_diff": home_momentum.get("last5", 0.5) - away_momentum.get("last5", 0.5),
                "rest_advantage": home_momentum.get("rest_days", 7) - away_momentum.get("rest_days", 7),
            })
            
            # 4. H2H Features
            h2h = await self._get_h2h(session, home_team.id, away_team.id, game.scheduled_at)
            features.update({
                "h2h_games": h2h.get("games", 0),
                "h2h_home_wins": h2h.get("home_wins", 0),
                "h2h_away_wins": h2h.get("away_wins", 0),
                "h2h_avg_total": h2h.get("avg_total", 200),
                "h2h_avg_margin": h2h.get("avg_margin", 0),
            })
            
            # 5. Odds Features
            odds_features = await self._get_odds_features(session, game.id)
            features.update(odds_features)
            
            # 6. Weather Features (outdoor sports only)
            if sport.code in self.OUTDOOR_SPORTS:
                weather = await self._get_weather_features(session, game.id)
                features.update(weather)
            
            # 7. Injury Features
            injury_features = await self._get_injury_features(session, game.id, home_team.id, away_team.id)
            features.update(injury_features)
            
            # 8. Sport-specific config
            sport_config = settings.get_sports_config(sport.code)
            features.update({
                "home_advantage": sport_config.get("home_advantage", 2.5),
            })
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing features for game {game.id}: {e}")
            return None
    
    def _compute_elo_features(
        self,
        home_team_id: str,
        away_team_id: str,
        sport_code: str,
    ) -> Dict[str, float]:
        """Compute ELO-based features."""
        if sport_code not in self.elo_ratings:
            self.elo_ratings[sport_code] = {}
        
        home_elo = self.elo_ratings[sport_code].get(home_team_id, 1500.0)
        away_elo = self.elo_ratings[sport_code].get(away_team_id, 1500.0)
        
        # Expected win probability
        elo_diff = home_elo - away_elo
        home_expected = 1 / (1 + 10 ** (-elo_diff / 400))
        
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "home_elo_expected": home_expected,
            "away_elo_expected": 1 - home_expected,
        }
    
    async def _update_elo_after_game(self, game: Game, sport_code: str):
        """Update ELO ratings after a game result."""
        if sport_code not in self.elo_ratings:
            return
        
        home_id = str(game.home_team_id)
        away_id = str(game.away_team_id)
        
        home_elo = self.elo_ratings[sport_code].get(home_id, 1500.0)
        away_elo = self.elo_ratings[sport_code].get(away_id, 1500.0)
        
        # K-factor from config
        sport_config = settings.get_sports_config(sport_code)
        k_factor = sport_config.get("k_factor", 32)
        
        # Expected scores
        home_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        away_expected = 1 - home_expected
        
        # Actual scores
        if game.home_score > game.away_score:
            home_actual, away_actual = 1, 0
        elif game.home_score < game.away_score:
            home_actual, away_actual = 0, 1
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Update ratings
        self.elo_ratings[sport_code][home_id] = home_elo + k_factor * (home_actual - home_expected)
        self.elo_ratings[sport_code][away_id] = away_elo + k_factor * (away_actual - away_expected)
    
    async def _get_team_stats(
        self,
        session: AsyncSession,
        team_id,
        before_date: datetime,
        n_games: int = 20,
    ) -> Dict[str, float]:
        """Get team statistics from recent games."""
        try:
            query = (
                select(Game)
                .where(
                    and_(
                        or_(
                            Game.home_team_id == team_id,
                            Game.away_team_id == team_id,
                        ),
                        Game.scheduled_at < before_date,
                        Game.status == GameStatus.FINAL,
                        Game.home_score.isnot(None),
                    )
                )
                .order_by(desc(Game.scheduled_at))
                .limit(n_games)
            )
            
            result = await session.execute(query)
            games = result.scalars().all()
            
            if not games:
                return {"win_pct": 0.5, "games": 0, "ppg": 100, "ppg_allowed": 100}
            
            wins = home_wins = away_wins = 0
            home_games = away_games = 0
            points_for = []
            points_against = []
            
            for game in games:
                is_home = game.home_team_id == team_id
                team_score = game.home_score if is_home else game.away_score
                opp_score = game.away_score if is_home else game.home_score
                
                if team_score is not None and opp_score is not None:
                    points_for.append(team_score)
                    points_against.append(opp_score)
                    
                    won = team_score > opp_score
                    if won:
                        wins += 1
                    
                    if is_home:
                        home_games += 1
                        if won:
                            home_wins += 1
                    else:
                        away_games += 1
                        if won:
                            away_wins += 1
            
            return {
                "win_pct": wins / len(games) if games else 0.5,
                "games": len(games),
                "ppg": np.mean(points_for) if points_for else 100,
                "ppg_allowed": np.mean(points_against) if points_against else 100,
                "home_win_pct": home_wins / home_games if home_games > 0 else 0.5,
                "away_win_pct": away_wins / away_games if away_games > 0 else 0.5,
            }
            
        except Exception as e:
            logger.warning(f"Error getting team stats: {e}")
            return {"win_pct": 0.5, "games": 0, "ppg": 100, "ppg_allowed": 100}
    
    async def _get_momentum(
        self,
        session: AsyncSession,
        team_id,
        before_date: datetime,
    ) -> Dict[str, float]:
        """Get team momentum/form."""
        try:
            query = (
                select(Game)
                .where(
                    and_(
                        or_(
                            Game.home_team_id == team_id,
                            Game.away_team_id == team_id,
                        ),
                        Game.scheduled_at < before_date,
                        Game.status == GameStatus.FINAL,
                    )
                )
                .order_by(desc(Game.scheduled_at))
                .limit(10)
            )
            
            result = await session.execute(query)
            games = result.scalars().all()
            
            if not games:
                return {"last5": 0.5, "last10": 0.5, "streak": 0, "rest_days": 7}
            
            results = []
            for game in games:
                is_home = game.home_team_id == team_id
                team_score = game.home_score if is_home else game.away_score
                opp_score = game.away_score if is_home else game.home_score
                
                if team_score is not None and opp_score is not None:
                    results.append(1 if team_score > opp_score else 0)
            
            # Calculate streak
            streak = 0
            if results:
                streak_val = results[0]
                for r in results:
                    if r == streak_val:
                        streak += 1 if streak_val == 1 else -1
                    else:
                        break
            
            # Rest days
            rest_days = (before_date - games[0].scheduled_at).days if games else 7
            
            return {
                "last5": np.mean(results[:5]) if len(results) >= 5 else np.mean(results) if results else 0.5,
                "last10": np.mean(results) if results else 0.5,
                "streak": streak,
                "rest_days": min(rest_days, 14),
            }
            
        except Exception as e:
            logger.warning(f"Error getting momentum: {e}")
            return {"last5": 0.5, "last10": 0.5, "streak": 0, "rest_days": 7}
    
    async def _get_h2h(
        self,
        session: AsyncSession,
        home_team_id,
        away_team_id,
        before_date: datetime,
        n_games: int = 10,
    ) -> Dict[str, float]:
        """Get head-to-head history."""
        try:
            query = (
                select(Game)
                .where(
                    and_(
                        or_(
                            and_(Game.home_team_id == home_team_id, Game.away_team_id == away_team_id),
                            and_(Game.home_team_id == away_team_id, Game.away_team_id == home_team_id),
                        ),
                        Game.scheduled_at < before_date,
                        Game.status == GameStatus.FINAL,
                    )
                )
                .order_by(desc(Game.scheduled_at))
                .limit(n_games)
            )
            
            result = await session.execute(query)
            games = result.scalars().all()
            
            if not games:
                return {"games": 0, "home_wins": 0, "away_wins": 0, "avg_total": 200, "avg_margin": 0}
            
            home_wins = 0
            totals = []
            margins = []
            
            for game in games:
                if game.home_score is not None and game.away_score is not None:
                    totals.append(game.home_score + game.away_score)
                    
                    # Track from perspective of current home team
                    if game.home_team_id == home_team_id:
                        margin = game.home_score - game.away_score
                        if margin > 0:
                            home_wins += 1
                    else:
                        margin = game.away_score - game.home_score
                        if margin > 0:
                            home_wins += 1
                    
                    margins.append(margin)
            
            return {
                "games": len(games),
                "home_wins": home_wins,
                "away_wins": len(games) - home_wins,
                "avg_total": np.mean(totals) if totals else 200,
                "avg_margin": np.mean(margins) if margins else 0,
            }
            
        except Exception as e:
            logger.warning(f"Error getting H2H: {e}")
            return {"games": 0, "home_wins": 0, "away_wins": 0, "avg_total": 200, "avg_margin": 0}
    
    async def _get_odds_features(
        self,
        session: AsyncSession,
        game_id,
    ) -> Dict[str, float]:
        """Get odds-based features."""
        try:
            query = (
                select(Odds)
                .where(Odds.game_id == game_id)
                .order_by(Odds.recorded_at)
            )
            
            result = await session.execute(query)
            odds_list = result.scalars().all()
            
            if not odds_list:
                return {
                    "opening_spread": 0, "current_spread": 0, "spread_movement": 0,
                    "opening_total": 200, "current_total": 200, "total_movement": 0,
                    "home_ml": -110, "away_ml": -110,
                }
            
            # Get opening and closing odds
            spreads = [o for o in odds_list if o.bet_type in ["spread", "spreads"]]
            totals = [o for o in odds_list if o.bet_type in ["total", "totals"]]
            moneylines = [o for o in odds_list if o.bet_type in ["h2h", "moneyline"]]
            
            opening_spread = spreads[0].home_line if spreads and spreads[0].home_line else 0
            current_spread = spreads[-1].home_line if spreads and spreads[-1].home_line else 0
            
            opening_total = totals[0].total if totals and totals[0].total else 200
            current_total = totals[-1].total if totals and totals[-1].total else 200
            
            home_ml = moneylines[-1].home_odds if moneylines and moneylines[-1].home_odds else -110
            away_ml = moneylines[-1].away_odds if moneylines and moneylines[-1].away_odds else -110
            
            return {
                "opening_spread": opening_spread,
                "current_spread": current_spread,
                "spread_movement": current_spread - opening_spread,
                "opening_total": opening_total,
                "current_total": current_total,
                "total_movement": current_total - opening_total,
                "home_ml": home_ml,
                "away_ml": away_ml,
            }
            
        except Exception as e:
            logger.warning(f"Error getting odds features: {e}")
            return {
                "opening_spread": 0, "current_spread": 0, "spread_movement": 0,
                "opening_total": 200, "current_total": 200, "total_movement": 0,
                "home_ml": -110, "away_ml": -110,
            }
    
    async def _get_weather_features(
        self,
        session: AsyncSession,
        game_id,
    ) -> Dict[str, float]:
        """Get weather features for outdoor games."""
        try:
            result = await session.execute(
                select(WeatherData).where(WeatherData.game_id == game_id)
            )
            weather = result.scalar_one_or_none()
            
            if not weather:
                return {
                    "temperature": 70, "wind_speed": 5, "precipitation_prob": 0,
                    "is_dome": 0, "weather_impact": 0,
                }
            
            temp = weather.temperature or 70
            wind = weather.wind_speed or 5
            precip = weather.precipitation_prob or 0
            is_dome = 1 if weather.weather_json and weather.weather_json.get("is_dome") else 0
            
            # Calculate weather impact score
            temp_impact = 0
            if temp < 32:
                temp_impact = (32 - temp) / 50
            elif temp > 95:
                temp_impact = (temp - 95) / 30
            
            wind_impact = min(1.0, wind / 30) if wind > 10 else 0
            precip_impact = precip / 100
            
            weather_impact = min(1.0, temp_impact + wind_impact + precip_impact)
            if is_dome:
                weather_impact = 0
            
            return {
                "temperature": temp,
                "wind_speed": wind,
                "precipitation_prob": precip,
                "is_dome": is_dome,
                "weather_impact": weather_impact,
            }
            
        except Exception as e:
            logger.warning(f"Error getting weather features: {e}")
            return {
                "temperature": 70, "wind_speed": 5, "precipitation_prob": 0,
                "is_dome": 0, "weather_impact": 0,
            }
    
    async def _get_injury_features(
        self,
        session: AsyncSession,
        game_id,
        home_team_id,
        away_team_id,
    ) -> Dict[str, float]:
        """Get injury impact features."""
        try:
            # Get injuries linked to this game
            result = await session.execute(
                select(GameInjury, Injury)
                .join(Injury, GameInjury.injury_id == Injury.id)
                .where(GameInjury.game_id == game_id)
            )
            game_injuries = result.all()
            
            home_injury_count = 0
            away_injury_count = 0
            home_injury_impact = 0.0
            away_injury_impact = 0.0
            
            for gi, injury in game_injuries:
                # Estimate injury impact based on status
                impact = 0.0
                if injury.status in ["Out", "IR"]:
                    impact = 1.0
                elif injury.status in ["Doubtful", "PUP"]:
                    impact = 0.75
                elif injury.status in ["Questionable"]:
                    impact = 0.5
                elif injury.status in ["Probable", "Day-to-Day"]:
                    impact = 0.25
                
                if injury.team_id == home_team_id:
                    home_injury_count += 1
                    home_injury_impact += impact
                elif injury.team_id == away_team_id:
                    away_injury_count += 1
                    away_injury_impact += impact
            
            return {
                "home_injury_count": home_injury_count,
                "away_injury_count": away_injury_count,
                "home_injury_impact": min(home_injury_impact, 5.0),  # Cap at 5
                "away_injury_impact": min(away_injury_impact, 5.0),
                "injury_advantage": away_injury_impact - home_injury_impact,
            }
            
        except Exception as e:
            logger.warning(f"Error getting injury features: {e}")
            return {
                "home_injury_count": 0, "away_injury_count": 0,
                "home_injury_impact": 0, "away_injury_impact": 0,
                "injury_advantage": 0,
            }
    
    async def _get_sport(self, session: AsyncSession, sport_code: str) -> Optional[Sport]:
        """Get sport by code."""
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code.upper())
        )
        return result.scalar_one_or_none()
    
    async def _initialize_elo(self, session: AsyncSession, sport: Sport):
        """Initialize ELO ratings from database."""
        result = await session.execute(
            select(Team).where(Team.sport_id == sport.id)
        )
        teams = result.scalars().all()
        
        self.elo_ratings[sport.code] = {}
        for team in teams:
            self.elo_ratings[sport.code][str(team.id)] = team.elo_rating or 1500.0
    
    async def _save_elo_ratings(self, session: AsyncSession, sport_code: str):
        """Save ELO ratings back to teams."""
        if sport_code not in self.elo_ratings:
            return
        
        from uuid import UUID
        
        for team_id, rating in self.elo_ratings[sport_code].items():
            try:
                result = await session.execute(
                    select(Team).where(Team.id == UUID(team_id))
                )
                team = result.scalar_one_or_none()
                if team:
                    team.elo_rating = rating
            except Exception as e:
                logger.warning(f"Error saving ELO for team {team_id}: {e}")
        
        try:
            await session.commit()
        except Exception as e:
            logger.warning(f"Error committing ELO ratings: {e}")
            await session.rollback()
    
    async def _save_game_features(
        self,
        session: AsyncSession,
        game_id,
        features: Dict[str, Any],
        overwrite: bool,
    ) -> str:
        """Save features to database."""
        result = await session.execute(
            select(GameFeature).where(GameFeature.game_id == game_id)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            if overwrite:
                existing.features = features
                existing.computed_at = datetime.utcnow()
                return "updated"
            return "skipped"
        else:
            gf = GameFeature(
                game_id=game_id,
                features=features,
                feature_version="2.0",
            )
            session.add(gf)
            return "created"


def print_stats(stats_list: List[FeatureStats]):
    """Print feature generation statistics."""
    table = Table(title="Feature Generation Results")
    
    table.add_column("Sport", style="cyan")
    table.add_column("Games", style="white")
    table.add_column("Created", style="green")
    table.add_column("Updated", style="yellow")
    table.add_column("Errors", style="red")
    table.add_column("Time (s)", style="magenta")
    
    total_games = total_created = total_updated = total_errors = 0
    
    for stats in stats_list:
        table.add_row(
            stats.sport,
            str(stats.games_processed),
            str(stats.features_created),
            str(stats.features_updated),
            str(stats.errors),
            f"{stats.duration_seconds:.1f}",
        )
        total_games += stats.games_processed
        total_created += stats.features_created
        total_updated += stats.features_updated
        total_errors += stats.errors
    
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_games}[/bold]",
        f"[bold]{total_created}[/bold]",
        f"[bold]{total_updated}[/bold]",
        f"[bold]{total_errors}[/bold]",
        "-",
    )
    
    console.print(table)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROYALEY Feature Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--sport", "-s", type=str, help="Sport code (NFL, NBA, etc.)")
    parser.add_argument("--all", "-a", action="store_true", help="Process all sports")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.sport and not args.all:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    
    console.print(Panel(
        f"[bold green]ROYALEY Feature Generator[/bold green]\n"
        f"Sport: {args.sport or 'ALL'}\n"
        f"Date Range: {start_date or 'All'} to {end_date or 'All'}\n"
        f"Overwrite: {args.overwrite}",
        title="Configuration"
    ))
    
    generator = FeatureGenerator()
    
    try:
        stats_list = await generator.generate_all_features(
            sport_code=args.sport,
            start_date=start_date,
            end_date=end_date,
            overwrite=args.overwrite,
        )
        
        print_stats(stats_list)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
