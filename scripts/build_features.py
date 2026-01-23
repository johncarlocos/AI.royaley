#!/usr/bin/env python3
"""
ROYALEY - Feature Store Builder
Phase 2: ML Feature Engineering Pipeline

Builds pre-computed features from database and saves to NVMe 1.
These features are used for fast ML training and inference.

Storage: /nvme1n1-disk/features/

Features computed:
- ELO ratings history
- Team statistics
- Head-to-head records
- Weather impacts (outdoor sports)
- Recent form (momentum)
- Rest days
- Home/away splits

Usage:
    # Build all features
    python build_features.py --all
    
    # Build specific sport
    python build_features.py --sport NFL
    
    # Build specific feature type
    python build_features.py --sport NBA --feature elo
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings, ALL_SPORTS, OUTDOOR_SPORTS

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class FeatureStats:
    """Statistics for feature building."""
    sport: str
    feature_type: str
    records_processed: int = 0
    features_created: int = 0
    file_size_mb: float = 0.0
    duration_seconds: float = 0.0


class FeatureBuilder:
    """
    Builds ML features from database and saves to feature store.
    
    Feature Store Structure:
    /nvme1n1-disk/features/
    ├── elo/
    │   ├── NFL_elo.parquet
    │   ├── NBA_elo.parquet
    │   └── ...
    ├── team_stats/
    │   ├── NFL_team_stats.parquet
    │   └── ...
    ├── h2h/
    │   ├── NFL_h2h.parquet
    │   └── ...
    ├── weather/
    │   ├── NFL_weather.parquet
    │   └── ...
    ├── momentum/
    │   ├── NFL_momentum.parquet
    │   └── ...
    └── combined/
        ├── NFL_features.parquet
        └── ...
    """
    
    FEATURE_TYPES = ["elo", "team_stats", "h2h", "weather", "momentum", "combined"]
    
    def __init__(self):
        self.stats: List[FeatureStats] = []
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create feature store directories."""
        base = Path(settings.FEATURES_PATH)
        
        for feature_type in self.FEATURE_TYPES:
            (base / feature_type).mkdir(parents=True, exist_ok=True)
        
        # Datasets directory
        Path(settings.DATASETS_PATH).mkdir(parents=True, exist_ok=True)
    
    async def build_all_features(
        self,
        sport_code: Optional[str] = None,
    ) -> List[FeatureStats]:
        """
        Build all features for all or specific sport.
        
        Args:
            sport_code: Optional sport to build (None = all)
            
        Returns:
            List of FeatureStats
        """
        sports = [sport_code] if sport_code else ALL_SPORTS
        
        console.print(Panel(
            f"[bold green]Building Feature Store[/bold green]\n"
            f"Sports: {', '.join(sports)}\n"
            f"Output: {settings.FEATURES_PATH}",
            title="Feature Builder"
        ))
        
        for sport in sports:
            console.print(f"\n[cyan]Processing {sport}...[/cyan]")
            
            # Build each feature type
            await self.build_elo_features(sport)
            await self.build_team_stats(sport)
            await self.build_h2h_features(sport)
            
            if sport in OUTDOOR_SPORTS:
                await self.build_weather_features(sport)
            
            await self.build_momentum_features(sport)
            await self.build_combined_features(sport)
        
        self._print_summary()
        return self.stats
    
    async def build_elo_features(self, sport_code: str) -> FeatureStats:
        """Build ELO rating history features."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="elo")
        
        console.print(f"  [yellow]Building ELO features for {sport_code}...[/yellow]")
        
        try:
            # Get game history with results
            games = await self._get_game_history(sport_code)
            
            if not games:
                console.print(f"    [red]No games found for {sport_code}[/red]")
                return stats
            
            # Calculate ELO ratings
            elo_ratings = {}
            elo_history = []
            K_FACTOR = 32
            
            for game in games:
                home_id = str(game["home_team_id"])
                away_id = str(game["away_team_id"])
                
                # Initialize ratings
                if home_id not in elo_ratings:
                    elo_ratings[home_id] = 1500.0
                if away_id not in elo_ratings:
                    elo_ratings[away_id] = 1500.0
                
                home_elo = elo_ratings[home_id]
                away_elo = elo_ratings[away_id]
                
                # Calculate expected scores
                exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
                exp_away = 1 - exp_home
                
                # Determine actual result
                if game["home_score"] is not None and game["away_score"] is not None:
                    if game["home_score"] > game["away_score"]:
                        actual_home, actual_away = 1, 0
                    elif game["home_score"] < game["away_score"]:
                        actual_home, actual_away = 0, 1
                    else:
                        actual_home, actual_away = 0.5, 0.5
                    
                    # Update ELO ratings
                    new_home_elo = home_elo + K_FACTOR * (actual_home - exp_home)
                    new_away_elo = away_elo + K_FACTOR * (actual_away - exp_away)
                    
                    elo_ratings[home_id] = new_home_elo
                    elo_ratings[away_id] = new_away_elo
                    
                    stats.records_processed += 1
                
                # Record history
                elo_history.append({
                    "game_id": str(game["id"]),
                    "game_date": game["game_date"],
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_elo_before": home_elo,
                    "away_elo_before": away_elo,
                    "home_elo_after": elo_ratings[home_id],
                    "away_elo_after": elo_ratings[away_id],
                    "elo_diff": home_elo - away_elo,
                    "home_expected": exp_home,
                })
            
            # Save to parquet
            df = pd.DataFrame(elo_history)
            output_path = Path(settings.FEATURES_PATH) / "elo" / f"{sport_code}_elo.parquet"
            df.to_parquet(output_path, index=False)
            
            stats.features_created = len(elo_history)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ ELO features: {stats.features_created} records, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building ELO features: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    async def build_team_stats(self, sport_code: str) -> FeatureStats:
        """Build team statistics features."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="team_stats")
        
        console.print(f"  [yellow]Building team stats for {sport_code}...[/yellow]")
        
        try:
            games = await self._get_game_history(sport_code, include_teams=True)
            
            if not games:
                return stats
            
            # Calculate rolling stats per team
            team_stats = {}
            stats_history = []
            
            for game in games:
                home_id = str(game["home_team_id"])
                away_id = str(game["away_team_id"])
                
                # Initialize team stats
                for team_id in [home_id, away_id]:
                    if team_id not in team_stats:
                        team_stats[team_id] = {
                            "games": 0,
                            "wins": 0,
                            "losses": 0,
                            "points_for": [],
                            "points_against": [],
                            "home_games": 0,
                            "away_games": 0,
                            "home_wins": 0,
                            "away_wins": 0,
                        }
                
                # Calculate features before game
                home_stats = team_stats[home_id]
                away_stats = team_stats[away_id]
                
                record = {
                    "game_id": str(game["id"]),
                    "game_date": game["game_date"],
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    # Home team features
                    "home_games_played": home_stats["games"],
                    "home_win_pct": home_stats["wins"] / max(1, home_stats["games"]),
                    "home_ppg": np.mean(home_stats["points_for"][-10:]) if home_stats["points_for"] else 0,
                    "home_ppg_allowed": np.mean(home_stats["points_against"][-10:]) if home_stats["points_against"] else 0,
                    "home_home_record": home_stats["home_wins"] / max(1, home_stats["home_games"]),
                    # Away team features
                    "away_games_played": away_stats["games"],
                    "away_win_pct": away_stats["wins"] / max(1, away_stats["games"]),
                    "away_ppg": np.mean(away_stats["points_for"][-10:]) if away_stats["points_for"] else 0,
                    "away_ppg_allowed": np.mean(away_stats["points_against"][-10:]) if away_stats["points_against"] else 0,
                    "away_away_record": away_stats["away_wins"] / max(1, away_stats["away_games"]),
                }
                
                stats_history.append(record)
                stats.records_processed += 1
                
                # Update team stats after game
                if game["home_score"] is not None and game["away_score"] is not None:
                    home_won = game["home_score"] > game["away_score"]
                    
                    # Home team
                    home_stats["games"] += 1
                    home_stats["home_games"] += 1
                    home_stats["points_for"].append(game["home_score"])
                    home_stats["points_against"].append(game["away_score"])
                    if home_won:
                        home_stats["wins"] += 1
                        home_stats["home_wins"] += 1
                    else:
                        home_stats["losses"] += 1
                    
                    # Away team
                    away_stats["games"] += 1
                    away_stats["away_games"] += 1
                    away_stats["points_for"].append(game["away_score"])
                    away_stats["points_against"].append(game["home_score"])
                    if not home_won:
                        away_stats["wins"] += 1
                        away_stats["away_wins"] += 1
                    else:
                        away_stats["losses"] += 1
            
            # Save
            df = pd.DataFrame(stats_history)
            output_path = Path(settings.FEATURES_PATH) / "team_stats" / f"{sport_code}_team_stats.parquet"
            df.to_parquet(output_path, index=False)
            
            stats.features_created = len(stats_history)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ Team stats: {stats.features_created} records, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building team stats: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    async def build_h2h_features(self, sport_code: str) -> FeatureStats:
        """Build head-to-head history features."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="h2h")
        
        console.print(f"  [yellow]Building H2H features for {sport_code}...[/yellow]")
        
        try:
            games = await self._get_game_history(sport_code)
            
            if not games:
                return stats
            
            # Track H2H records
            h2h_records = {}  # (team1, team2) -> history
            h2h_history = []
            
            for game in games:
                home_id = str(game["home_team_id"])
                away_id = str(game["away_team_id"])
                
                # Create consistent key (smaller id first)
                key = tuple(sorted([home_id, away_id]))
                
                if key not in h2h_records:
                    h2h_records[key] = {
                        "games": 0,
                        "team1_wins": 0,
                        "team2_wins": 0,
                        "total_points": [],
                        "margins": [],
                    }
                
                h2h = h2h_records[key]
                
                # Record features before game
                record = {
                    "game_id": str(game["id"]),
                    "game_date": game["game_date"],
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "h2h_games": h2h["games"],
                    "h2h_home_wins": h2h["team1_wins"] if home_id == key[0] else h2h["team2_wins"],
                    "h2h_away_wins": h2h["team2_wins"] if home_id == key[0] else h2h["team1_wins"],
                    "h2h_avg_total": np.mean(h2h["total_points"]) if h2h["total_points"] else 0,
                    "h2h_avg_margin": np.mean(h2h["margins"]) if h2h["margins"] else 0,
                }
                
                h2h_history.append(record)
                stats.records_processed += 1
                
                # Update H2H after game
                if game["home_score"] is not None and game["away_score"] is not None:
                    h2h["games"] += 1
                    h2h["total_points"].append(game["home_score"] + game["away_score"])
                    
                    margin = game["home_score"] - game["away_score"]
                    if home_id == key[0]:
                        h2h["margins"].append(margin)
                        if margin > 0:
                            h2h["team1_wins"] += 1
                        elif margin < 0:
                            h2h["team2_wins"] += 1
                    else:
                        h2h["margins"].append(-margin)
                        if margin > 0:
                            h2h["team2_wins"] += 1
                        elif margin < 0:
                            h2h["team1_wins"] += 1
            
            # Save
            df = pd.DataFrame(h2h_history)
            output_path = Path(settings.FEATURES_PATH) / "h2h" / f"{sport_code}_h2h.parquet"
            df.to_parquet(output_path, index=False)
            
            stats.features_created = len(h2h_history)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ H2H features: {stats.features_created} records, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building H2H features: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    async def build_weather_features(self, sport_code: str) -> FeatureStats:
        """Build weather impact features for outdoor sports."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="weather")
        
        console.print(f"  [yellow]Building weather features for {sport_code}...[/yellow]")
        
        try:
            # Get games with weather data
            games = await self._get_games_with_weather(sport_code)
            
            if not games:
                console.print(f"    [yellow]No weather data for {sport_code}[/yellow]")
                return stats
            
            weather_features = []
            
            for game in games:
                weather = game.get("weather", {})
                
                # Calculate weather features
                temp = weather.get("temperature", 70)
                wind = weather.get("wind_speed", 0)
                precip = weather.get("precipitation_prob", 0)
                
                # Temperature impact (extreme temps affect scoring)
                temp_impact = 0
                if temp < 32:
                    temp_impact = (32 - temp) / 50  # Cold impact
                elif temp > 95:
                    temp_impact = (temp - 95) / 30  # Heat impact
                
                # Wind impact (affects passing games)
                wind_impact = min(1.0, wind / 30) if wind > 10 else 0
                
                # Precipitation impact
                precip_impact = precip / 100
                
                # Combined weather impact
                total_impact = min(1.0, temp_impact + wind_impact + precip_impact)
                
                record = {
                    "game_id": str(game["id"]),
                    "game_date": game["game_date"],
                    "temperature": temp,
                    "wind_speed": wind,
                    "precipitation_prob": precip,
                    "is_dome": weather.get("is_dome", False),
                    "temp_impact": temp_impact,
                    "wind_impact": wind_impact,
                    "precip_impact": precip_impact,
                    "total_weather_impact": total_impact,
                }
                
                weather_features.append(record)
                stats.records_processed += 1
            
            # Save
            df = pd.DataFrame(weather_features)
            output_path = Path(settings.FEATURES_PATH) / "weather" / f"{sport_code}_weather.parquet"
            df.to_parquet(output_path, index=False)
            
            stats.features_created = len(weather_features)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ Weather features: {stats.features_created} records, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building weather features: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    async def build_momentum_features(self, sport_code: str) -> FeatureStats:
        """Build momentum/recent form features."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="momentum")
        
        console.print(f"  [yellow]Building momentum features for {sport_code}...[/yellow]")
        
        try:
            games = await self._get_game_history(sport_code)
            
            if not games:
                return stats
            
            # Track recent results per team
            team_history = {}  # team_id -> list of recent results
            momentum_features = []
            
            for game in games:
                home_id = str(game["home_team_id"])
                away_id = str(game["away_team_id"])
                
                # Initialize
                for team_id in [home_id, away_id]:
                    if team_id not in team_history:
                        team_history[team_id] = {
                            "results": [],  # 1=win, 0=loss, 0.5=tie
                            "margins": [],
                            "rest_days": None,
                            "last_game_date": None,
                        }
                
                home_hist = team_history[home_id]
                away_hist = team_history[away_id]
                
                # Calculate rest days
                game_date = game["game_date"]
                home_rest = (game_date - home_hist["last_game_date"]).days if home_hist["last_game_date"] else 7
                away_rest = (game_date - away_hist["last_game_date"]).days if away_hist["last_game_date"] else 7
                
                # Recent form (last 5 and 10 games)
                home_last5 = home_hist["results"][-5:] if home_hist["results"] else []
                home_last10 = home_hist["results"][-10:] if home_hist["results"] else []
                away_last5 = away_hist["results"][-5:] if away_hist["results"] else []
                away_last10 = away_hist["results"][-10:] if away_hist["results"] else []
                
                record = {
                    "game_id": str(game["id"]),
                    "game_date": game_date,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    # Home team momentum
                    "home_rest_days": min(home_rest, 14),  # Cap at 14
                    "home_form_last5": np.mean(home_last5) if home_last5 else 0.5,
                    "home_form_last10": np.mean(home_last10) if home_last10 else 0.5,
                    "home_streak": self._calculate_streak(home_hist["results"]),
                    "home_avg_margin_last5": np.mean(home_hist["margins"][-5:]) if home_hist["margins"] else 0,
                    # Away team momentum
                    "away_rest_days": min(away_rest, 14),
                    "away_form_last5": np.mean(away_last5) if away_last5 else 0.5,
                    "away_form_last10": np.mean(away_last10) if away_last10 else 0.5,
                    "away_streak": self._calculate_streak(away_hist["results"]),
                    "away_avg_margin_last5": np.mean(away_hist["margins"][-5:]) if away_hist["margins"] else 0,
                    # Rest advantage
                    "rest_advantage": home_rest - away_rest,
                    "form_advantage": (np.mean(home_last5) if home_last5 else 0.5) - (np.mean(away_last5) if away_last5 else 0.5),
                }
                
                momentum_features.append(record)
                stats.records_processed += 1
                
                # Update history after game
                if game["home_score"] is not None and game["away_score"] is not None:
                    margin = game["home_score"] - game["away_score"]
                    
                    if margin > 0:
                        home_hist["results"].append(1)
                        away_hist["results"].append(0)
                    elif margin < 0:
                        home_hist["results"].append(0)
                        away_hist["results"].append(1)
                    else:
                        home_hist["results"].append(0.5)
                        away_hist["results"].append(0.5)
                    
                    home_hist["margins"].append(margin)
                    away_hist["margins"].append(-margin)
                
                home_hist["last_game_date"] = game_date
                away_hist["last_game_date"] = game_date
            
            # Save
            df = pd.DataFrame(momentum_features)
            output_path = Path(settings.FEATURES_PATH) / "momentum" / f"{sport_code}_momentum.parquet"
            df.to_parquet(output_path, index=False)
            
            stats.features_created = len(momentum_features)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ Momentum features: {stats.features_created} records, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building momentum features: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    async def build_combined_features(self, sport_code: str) -> FeatureStats:
        """Combine all features into single dataset for training."""
        start_time = datetime.now()
        stats = FeatureStats(sport=sport_code, feature_type="combined")
        
        console.print(f"  [yellow]Building combined features for {sport_code}...[/yellow]")
        
        try:
            base_path = Path(settings.FEATURES_PATH)
            
            # Load all feature files
            elo_path = base_path / "elo" / f"{sport_code}_elo.parquet"
            team_stats_path = base_path / "team_stats" / f"{sport_code}_team_stats.parquet"
            h2h_path = base_path / "h2h" / f"{sport_code}_h2h.parquet"
            momentum_path = base_path / "momentum" / f"{sport_code}_momentum.parquet"
            weather_path = base_path / "weather" / f"{sport_code}_weather.parquet"
            
            # Start with ELO as base
            if not elo_path.exists():
                console.print(f"    [red]ELO features not found for {sport_code}[/red]")
                return stats
            
            df_combined = pd.read_parquet(elo_path)
            
            # Merge other features
            if team_stats_path.exists():
                df_team = pd.read_parquet(team_stats_path)
                df_combined = df_combined.merge(
                    df_team.drop(columns=["game_date", "home_team_id", "away_team_id"], errors="ignore"),
                    on="game_id",
                    how="left"
                )
            
            if h2h_path.exists():
                df_h2h = pd.read_parquet(h2h_path)
                df_combined = df_combined.merge(
                    df_h2h.drop(columns=["game_date", "home_team_id", "away_team_id"], errors="ignore"),
                    on="game_id",
                    how="left"
                )
            
            if momentum_path.exists():
                df_momentum = pd.read_parquet(momentum_path)
                df_combined = df_combined.merge(
                    df_momentum.drop(columns=["game_date", "home_team_id", "away_team_id"], errors="ignore"),
                    on="game_id",
                    how="left"
                )
            
            if weather_path.exists() and sport_code in OUTDOOR_SPORTS:
                df_weather = pd.read_parquet(weather_path)
                df_combined = df_combined.merge(
                    df_weather.drop(columns=["game_date"], errors="ignore"),
                    on="game_id",
                    how="left"
                )
            
            # Fill missing values
            df_combined = df_combined.fillna(0)
            
            # Save combined features
            output_path = base_path / "combined" / f"{sport_code}_features.parquet"
            df_combined.to_parquet(output_path, index=False)
            
            # Also save to datasets directory for training
            train_path = Path(settings.DATASETS_PATH) / f"{sport_code}_train.parquet"
            df_combined.to_parquet(train_path, index=False)
            
            stats.features_created = len(df_combined)
            stats.file_size_mb = output_path.stat().st_size / (1024 * 1024)
            stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            console.print(f"    [green]✓ Combined features: {stats.features_created} records, {df_combined.shape[1]} columns, {stats.file_size_mb:.2f} MB[/green]")
            
        except Exception as e:
            logger.error(f"Error building combined features: {e}")
            console.print(f"    [red]✗ Error: {e}[/red]")
        
        self.stats.append(stats)
        return stats
    
    def _calculate_streak(self, results: List[float]) -> int:
        """Calculate current win/loss streak."""
        if not results:
            return 0
        
        streak = 0
        last_result = results[-1]
        
        for result in reversed(results):
            if result == last_result:
                if last_result > 0.5:
                    streak += 1
                elif last_result < 0.5:
                    streak -= 1
            else:
                break
        
        return streak
    
    async def _get_game_history(
        self,
        sport_code: str,
        include_teams: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get game history from database."""
        try:
            from app.core.database import get_async_session
            from app.models import Game, Sport
            from sqlalchemy import select
            
            games = []
            
            async with get_async_session() as session:
                query = (
                    select(Game)
                    .join(Sport, Game.sport_id == Sport.id)
                    .where(Sport.code == sport_code)
                    .order_by(Game.game_date)
                )
                
                result = await session.execute(query)
                
                for game in result.scalars():
                    games.append({
                        "id": game.id,
                        "game_date": game.game_date,
                        "home_team_id": game.home_team_id,
                        "away_team_id": game.away_team_id,
                        "home_score": game.home_score,
                        "away_score": game.away_score,
                    })
            
            return games
            
        except ImportError:
            logger.warning("Database modules not available")
            return []
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return []
    
    async def _get_games_with_weather(self, sport_code: str) -> List[Dict[str, Any]]:
        """Get games with weather data."""
        try:
            from app.core.database import get_async_session
            from app.models import Game, Sport, WeatherData
            from sqlalchemy import select
            
            games = []
            
            async with get_async_session() as session:
                query = (
                    select(Game, WeatherData)
                    .join(Sport, Game.sport_id == Sport.id)
                    .outerjoin(WeatherData, Game.id == WeatherData.game_id)
                    .where(Sport.code == sport_code)
                    .order_by(Game.game_date)
                )
                
                result = await session.execute(query)
                
                for game, weather in result:
                    weather_data = {}
                    if weather:
                        weather_data = {
                            "temperature": weather.temperature,
                            "wind_speed": weather.wind_speed,
                            "precipitation_prob": weather.precipitation_prob,
                            "is_dome": weather.weather_json.get("is_dome", False) if weather.weather_json else False,
                        }
                    
                    games.append({
                        "id": game.id,
                        "game_date": game.game_date,
                        "weather": weather_data,
                    })
            
            return games
            
        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return []
    
    def _print_summary(self):
        """Print feature building summary."""
        table = Table(title="Feature Store Build Summary")
        table.add_column("Sport", style="cyan")
        table.add_column("Feature Type", style="blue")
        table.add_column("Records", style="green")
        table.add_column("Size (MB)", style="yellow")
        table.add_column("Time (s)", style="magenta")
        
        total_records = 0
        total_size = 0.0
        
        for stat in self.stats:
            table.add_row(
                stat.sport,
                stat.feature_type,
                str(stat.features_created),
                f"{stat.file_size_mb:.2f}",
                f"{stat.duration_seconds:.1f}"
            )
            total_records += stat.features_created
            total_size += stat.file_size_mb
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "-",
            f"[bold]{total_records}[/bold]",
            f"[bold]{total_size:.2f}[/bold]",
            "-"
        )
        
        console.print(table)
        console.print(f"\n[cyan]Feature store location:[/cyan] {settings.FEATURES_PATH}")
        console.print(f"[cyan]Training datasets:[/cyan] {settings.DATASETS_PATH}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build ML feature store"
    )
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, NBA, etc.)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build all features for all sports"
    )
    parser.add_argument(
        "--feature", "-f",
        type=str,
        choices=FeatureBuilder.FEATURE_TYPES,
        help="Specific feature type to build"
    )
    
    args = parser.parse_args()
    
    if not args.sport and not args.all:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return
    
    builder = FeatureBuilder()
    
    if args.all:
        await builder.build_all_features()
    else:
        await builder.build_all_features(sport_code=args.sport.upper())


if __name__ == "__main__":
    asyncio.run(main())
