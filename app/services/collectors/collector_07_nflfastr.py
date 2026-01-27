"""
ROYALEY - nflfastR Data Collector
Phase 1: Data Collection Services

Collects comprehensive NFL play-by-play data from nflverse/nflfastR.
Features: Play-by-play (1999-present), EPA, WPA, CPOE, air yards, next-gen stats, 75+ features.

Data Source: https://github.com/nflverse/nflverse-data/releases
Documentation: https://www.nflfastr.com/

FREE data - no API key required!

Key Data Types:
- Play-by-play: Every play with EPA, WPA, success rate (1999-present)
- Player stats: Weekly and seasonal aggregated stats
- Schedules: Full game schedules with results and betting lines
- Rosters: Player rosters by season
- NGS: Next Gen Stats (passing, rushing, receiving)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx
import pandas as pd

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus, Player, PlayerStats, TeamStats
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NFLVERSE DATA URLS (Direct GitHub Release Downloads)
# =============================================================================

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Note: Different data types have different file formats available
# - pbp: parquet, csv, rds
# - schedules: rds only (use CSV from nfldata repo as fallback)
# - player_stats: parquet, csv
# - etc.

NFLVERSE_URLS = {
    # Play-by-play (1999-present, parquet available)
    "pbp_parquet": f"{NFLVERSE_BASE}/pbp/play_by_play_{{year}}.parquet",
    "pbp_csv": f"{NFLVERSE_BASE}/pbp/play_by_play_{{year}}.csv.gz",
    
    # Schedules - use Lee Sharpe's nfldata CSV (schedules release only has rds)
    "schedules": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    
    # Player stats (per season)
    "player_stats": f"{NFLVERSE_BASE}/stats_player/player_stats_{{year}}.parquet",
    "player_stats_csv": f"{NFLVERSE_BASE}/stats_player/player_stats_{{year}}.csv",
    
    # Team stats
    "team_stats": f"{NFLVERSE_BASE}/stats_team/team_stats_{{year}}.parquet",
    
    # Rosters
    "rosters": f"{NFLVERSE_BASE}/rosters/roster_{{year}}.parquet",
    "rosters_weekly": f"{NFLVERSE_BASE}/weekly_rosters/roster_weekly_{{year}}.parquet",
    
    # Next Gen Stats
    "ngs_passing": f"{NFLVERSE_BASE}/nextgen_stats/ngs_{{year}}_passing.parquet",
    "ngs_rushing": f"{NFLVERSE_BASE}/nextgen_stats/ngs_{{year}}_rushing.parquet",
    "ngs_receiving": f"{NFLVERSE_BASE}/nextgen_stats/ngs_{{year}}_receiving.parquet",
    
    # Other data
    "teams": f"{NFLVERSE_BASE}/teams/teams.csv",
    "players": f"{NFLVERSE_BASE}/players/players.parquet",
    "officials": f"{NFLVERSE_BASE}/officials/officials.parquet",
    "combine": f"{NFLVERSE_BASE}/combine/combine.parquet",
    "draft_picks": f"{NFLVERSE_BASE}/draft_picks/draft_picks.parquet",
    "injuries": f"{NFLVERSE_BASE}/injuries/injuries_{{year}}.parquet",
    "snap_counts": f"{NFLVERSE_BASE}/snap_counts/snap_counts_{{year}}.parquet",
}


# =============================================================================
# KEY PLAY-BY-PLAY FEATURES (75+ columns)
# =============================================================================

PBP_KEY_COLUMNS = [
    # Game/Play Identifiers
    "play_id", "game_id", "old_game_id", "home_team", "away_team", 
    "season_type", "week", "posteam", "defteam", "game_date",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "play_type", "down", "ydstogo", "yardline_100", "goal_to_go",
    
    # Scores
    "total_home_score", "total_away_score", "posteam_score", "defteam_score",
    "score_differential", "posteam_score_post", "defteam_score_post",
    
    # Expected Points Added (EPA) - THE KEY METRIC
    "ep", "epa", "total_home_epa", "total_away_epa",
    "total_home_rush_epa", "total_away_rush_epa",
    "total_home_pass_epa", "total_away_pass_epa",
    "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa",
    
    # Win Probability (WP)
    "wp", "def_wp", "home_wp", "away_wp", "wpa",
    "vegas_wpa", "vegas_home_wpa", "home_wp_post", "away_wp_post",
    "vegas_wp", "vegas_home_wp",
    
    # Completion Probability (CP) & CPOE
    "cp", "cpoe",
    
    # Expected Yards After Catch (xYAC)
    "xyac_epa", "xyac_mean_yardage", "xyac_median_yardage", "xyac_success",
    
    # Air Yards
    "air_yards", "yards_after_catch",
    
    # Play Success
    "success", "first_down", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    
    # Passing
    "passer_player_id", "passer_player_name", "passing_yards",
    "pass_touchdown", "interception", "sack", "qb_hit",
    "pass_attempt", "complete_pass", "incomplete_pass",
    
    # Rushing
    "rusher_player_id", "rusher_player_name", "rushing_yards",
    "rush_touchdown", "rush_attempt",
    
    # Receiving
    "receiver_player_id", "receiver_player_name", "receiving_yards",
    
    # Drive Info
    "drive", "drive_play_count", "drive_time_of_possession",
    "drive_first_downs", "drive_inside20", "drive_ended_with_score",
    
    # Series Info
    "series", "series_success", "series_result",
    
    # Special Teams
    "kick_distance", "field_goal_attempt", "field_goal_result",
    "punt_attempt", "extra_point_attempt", "extra_point_result",
    
    # Penalties
    "penalty", "penalty_yards", "penalty_team", "penalty_type",
]

# Team abbreviation to full name mapping
NFL_TEAMS = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars",  # Old abbreviation
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "OAK": "Oakland Raiders",  # Historical
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
    "SD": "San Diego Chargers",  # Historical
    "STL": "St. Louis Rams",  # Historical
}


class NFLFastRCollector(BaseCollector):
    """
    Collector for nflfastR/nflverse NFL data.
    
    Features:
    - Play-by-play data from 1999-present
    - Expected Points Added (EPA) - gold standard metric
    - Win Probability (WP) models
    - Completion Probability (CP) & CPOE
    - Expected Yards After Catch (xYAC)
    - Air yards, next-gen stats
    - Player and team aggregated stats
    - Game schedules and results with betting lines
    
    FREE - No API key required!
    """
    
    def __init__(self):
        super().__init__(
            name="nflfastr",
            base_url="https://github.com/nflverse/nflverse-data/releases/download",
            rate_limit=30,  # Conservative for GitHub
            rate_window=60,
            timeout=300.0,  # Long timeout for large parquet files
            max_retries=3,
        )
        self.data_dir = Path(settings.MODEL_STORAGE_PATH) / "nfl_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._schedules_cache: Optional[pd.DataFrame] = None
        
    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sport_code: str = "NFL",
        collect_type: str = "schedules",
        years: List[int] = None,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect NFL data from nflfastR/nflverse.
        
        Args:
            sport_code: Must be "NFL"
            collect_type: "schedules", "pbp", "player_stats", "team_stats", "all"
            years: List of years (default: last 5 years, available 1999-present)
            
        Returns:
            CollectorResult with collected data
        """
        if sport_code != "NFL":
            return CollectorResult(
                success=False,
                error="nflfastR only supports NFL data",
                records_count=0,
            )
        
        current_year = datetime.now().year
        if years is None:
            years = list(range(current_year - 4, current_year + 1))
        
        all_data = {
            "games": [],
            "team_stats": [],
            "player_stats": [],
        }
        errors = []
        
        try:
            if collect_type in ["schedules", "all"]:
                games = await self._collect_schedules(years)
                all_data["games"] = games
                logger.info(f"[nflfastR] Collected {len(games)} games")
                
            if collect_type in ["player_stats", "all"]:
                player_stats = await self._collect_player_stats(years)
                all_data["player_stats"] = player_stats
                
            if collect_type in ["team_stats", "all"]:
                team_stats = await self._collect_team_stats(years)
                all_data["team_stats"] = team_stats
                
        except Exception as e:
            logger.error(f"[nflfastR] Collection error: {e}")
            errors.append(str(e))
        
        total_records = sum(len(v) for v in all_data.values() if isinstance(v, list))
        
        return CollectorResult(
            success=len(errors) == 0 or total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "collect_type": collect_type},
        )
    
    # =========================================================================
    # SCHEDULES / GAMES
    # =========================================================================
    
    async def _collect_schedules(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect game schedules with results and betting lines.
        
        Note: nflverse schedules release only has RDS format (not CSV/parquet).
        We try nflreadpy first, then fall back to extracting from PBP data.
        """
        games = []
        
        try:
            # First try nflreadpy if installed
            try:
                import nflreadpy as nfl
                logger.info("[nflfastR] Using nflreadpy to load schedules")
                df = nfl.load_schedules().to_pandas()
                df = df[df["season"].isin(years)]
                
                for _, row in df.iterrows():
                    game = self._parse_schedule_row(row)
                    if game:
                        games.append(game)
                
                logger.info(f"[nflfastR] Loaded {len(games)} games via nflreadpy")
                return games
                
            except ImportError:
                logger.info("[nflfastR] nflreadpy not installed")
            except Exception as e:
                logger.warning(f"[nflfastR] nflreadpy failed: {e}")
            
            # Try to extract schedules from PBP data
            logger.info("[nflfastR] Extracting schedules from PBP data...")
            
            for year in years:
                try:
                    # Only download a few columns for schedule extraction
                    url = NFLVERSE_URLS["pbp_parquet"].format(year=year)
                    cols = ["game_id", "home_team", "away_team", "game_date", "season", 
                            "week", "season_type", "total_home_score", "total_away_score"]
                    
                    df = await self._download_parquet(url, columns=cols)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Get unique games
                    game_df = df.drop_duplicates(subset=["game_id"]).copy()
                    
                    for _, row in game_df.iterrows():
                        game_id = row.get("game_id")
                        if not game_id:
                            continue
                        
                        home_team = str(row.get("home_team", ""))
                        away_team = str(row.get("away_team", ""))
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Get max scores from the game
                        home_score = int(row.get("total_home_score", 0)) if pd.notna(row.get("total_home_score")) else None
                        away_score = int(row.get("total_away_score", 0)) if pd.notna(row.get("total_away_score")) else None
                        
                        game_date = row.get("game_date")
                        if isinstance(game_date, str):
                            game_date = datetime.strptime(game_date, "%Y-%m-%d")
                        elif hasattr(game_date, 'to_pydatetime'):
                            game_date = game_date.to_pydatetime()
                        
                        games.append({
                            "sport_code": "NFL",
                            "external_id": f"nfl_{game_id}",
                            "game_id": game_id,
                            "home_team": {
                                "name": NFL_TEAMS.get(home_team, home_team),
                                "abbreviation": home_team,
                            },
                            "away_team": {
                                "name": NFL_TEAMS.get(away_team, away_team),
                                "abbreviation": away_team,
                            },
                            "game_date": game_date.isoformat() if game_date else None,
                            "status": "final",
                            "home_score": home_score,
                            "away_score": away_score,
                            "season": int(row.get("season", year)),
                            "week": int(row.get("week", 0)) if pd.notna(row.get("week")) else None,
                            "game_type": row.get("season_type", "REG"),
                        })
                    
                    logger.info(f"[nflfastR] Extracted {len(game_df)} games from {year} PBP")
                    
                except Exception as e:
                    logger.error(f"[nflfastR] Failed to extract schedules from {year}: {e}")
            
            logger.info(f"[nflfastR] Extracted total {len(games)} games from PBP data")
            
        except Exception as e:
            logger.error(f"[nflfastR] Schedule collection error: {e}")
        
        return games
    
    def _parse_schedule_row(self, row) -> Optional[Dict[str, Any]]:
        """Parse a schedule dataframe row to game dict."""
        try:
            game_id = str(row.get("game_id", ""))
            if not game_id:
                return None
            
            home_team = str(row.get("home_team", ""))
            away_team = str(row.get("away_team", ""))
            
            if not home_team or not away_team:
                return None
            
            # Parse game date
            game_date = row.get("gameday")
            if pd.isna(game_date):
                return None
            
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, "%Y-%m-%d")
            elif hasattr(game_date, 'to_pydatetime'):
                game_date = game_date.to_pydatetime()
            
            # Get scores
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            
            home_score = int(home_score) if pd.notna(home_score) else None
            away_score = int(away_score) if pd.notna(away_score) else None
            
            # Determine status
            if home_score is not None and away_score is not None:
                status = "final"
            elif game_date < datetime.now():
                status = "final"
            else:
                status = "scheduled"
            
            return {
                "sport_code": "NFL",
                "external_id": f"nfl_{game_id}",
                "game_id": game_id,
                "home_team": {
                    "name": NFL_TEAMS.get(home_team, home_team),
                    "abbreviation": home_team,
                },
                "away_team": {
                    "name": NFL_TEAMS.get(away_team, away_team),
                    "abbreviation": away_team,
                },
                "game_date": game_date.isoformat(),
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "season": int(row.get("season", 0)),
                "week": int(row.get("week", 0)) if pd.notna(row.get("week")) else None,
                "game_type": row.get("game_type", "REG"),
                "weekday": row.get("weekday"),
                "gametime": row.get("gametime"),
                "stadium": row.get("stadium"),
                "stadium_id": row.get("stadium_id"),
                "roof": row.get("roof"),
                "surface": row.get("surface"),
                "temp": row.get("temp") if pd.notna(row.get("temp")) else None,
                "wind": row.get("wind") if pd.notna(row.get("wind")) else None,
                # Betting lines
                "spread_line": float(row.get("spread_line")) if pd.notna(row.get("spread_line")) else None,
                "total_line": float(row.get("total_line")) if pd.notna(row.get("total_line")) else None,
                "home_moneyline": int(row.get("home_moneyline")) if pd.notna(row.get("home_moneyline")) else None,
                "away_moneyline": int(row.get("away_moneyline")) if pd.notna(row.get("away_moneyline")) else None,
                "result": int(row.get("result")) if pd.notna(row.get("result")) else None,
                "total": int(row.get("total")) if pd.notna(row.get("total")) else None,
                "overtime": int(row.get("overtime")) if pd.notna(row.get("overtime")) else 0,
                "div_game": int(row.get("div_game")) if pd.notna(row.get("div_game")) else 0,
            }
            
        except Exception as e:
            logger.debug(f"[nflfastR] Failed to parse schedule row: {e}")
            return None
    
    # =========================================================================
    # PLAY-BY-PLAY DATA (THE GOLD)
    # =========================================================================
    
    async def collect_pbp(
        self,
        years: List[int] = None,
        columns: List[str] = None,
        save_to_disk: bool = True,
    ) -> CollectorResult:
        """
        Collect play-by-play data with EPA, WPA, CPOE, and 75+ features.
        
        WARNING: Large files (~500MB per season compressed)
        
        Args:
            years: List of years (1999-present)
            columns: Specific columns to load (None = key columns only for memory)
            save_to_disk: Save parquet files locally for faster reuse
            
        Returns:
            CollectorResult with PBP summary info
        """
        current_year = datetime.now().year
        if years is None:
            years = [current_year]
        
        if columns is None:
            columns = PBP_KEY_COLUMNS
        
        results = []
        errors = []
        
        for year in years:
            try:
                url = NFLVERSE_URLS["pbp_parquet"].format(year=year)
                logger.info(f"[nflfastR] Downloading PBP for {year}...")
                
                # Check if cached
                cache_path = self.data_dir / f"pbp_{year}.parquet"
                
                if cache_path.exists() and not save_to_disk:
                    logger.info(f"[nflfastR] Loading cached PBP for {year}")
                    df = pd.read_parquet(cache_path, columns=columns)
                else:
                    df = await self._download_parquet(url, columns=columns)
                    
                    if df is not None and save_to_disk:
                        df.to_parquet(cache_path)
                        logger.info(f"[nflfastR] Cached PBP {year} to {cache_path}")
                
                if df is not None:
                    results.append({
                        "year": year,
                        "plays": len(df),
                        "columns": len(df.columns),
                        "size_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                    })
                    logger.info(f"[nflfastR] {year}: {len(df):,} plays loaded")
                    
            except Exception as e:
                logger.error(f"[nflfastR] PBP {year} error: {e}")
                errors.append(f"{year}: {str(e)[:50]}")
        
        total_plays = sum(r["plays"] for r in results)
        
        return CollectorResult(
            success=len(results) > 0,
            data=results,
            records_count=total_plays,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "type": "play_by_play"},
        )
    
    async def get_team_epa(
        self,
        season: int = None,
        weeks: List[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate team EPA statistics from play-by-play data.
        
        Returns dict like:
        {
            "KC": {
                "pass_epa_per_play": 0.15,
                "rush_epa_per_play": 0.08,
                "total_epa_per_play": 0.12,
                "success_rate": 0.52,
                "pass_rate": 0.58,
                "plays": 1024
            },
            ...
        }
        """
        if season is None:
            season = datetime.now().year
        
        team_epa = {}
        
        try:
            # Check cache first
            cache_path = self.data_dir / f"pbp_{season}.parquet"
            
            if cache_path.exists():
                logger.info(f"[nflfastR] Loading cached PBP for EPA calculation")
                cols = ["posteam", "play_type", "epa", "success", "week"]
                pbp = pd.read_parquet(cache_path, columns=cols)
            else:
                # Download
                url = NFLVERSE_URLS["pbp_parquet"].format(year=season)
                cols = ["posteam", "play_type", "epa", "success", "week"]
                pbp = await self._download_parquet(url, columns=cols)
            
            if pbp is None:
                return team_epa
            
            # Filter weeks if specified
            if weeks:
                pbp = pbp[pbp["week"].isin(weeks)]
            
            # Filter to pass/run plays only
            pbp = pbp[pbp["play_type"].isin(["pass", "run"])]
            pbp = pbp[pbp["posteam"].notna()]
            
            # Aggregate by team
            for team in pbp["posteam"].unique():
                team_plays = pbp[pbp["posteam"] == team]
                pass_plays = team_plays[team_plays["play_type"] == "pass"]
                rush_plays = team_plays[team_plays["play_type"] == "run"]
                
                team_epa[team] = {
                    "pass_epa_per_play": float(pass_plays["epa"].mean()) if len(pass_plays) > 0 else 0.0,
                    "rush_epa_per_play": float(rush_plays["epa"].mean()) if len(rush_plays) > 0 else 0.0,
                    "total_epa_per_play": float(team_plays["epa"].mean()) if len(team_plays) > 0 else 0.0,
                    "success_rate": float((team_plays["success"] == 1).mean()) if len(team_plays) > 0 else 0.0,
                    "pass_rate": len(pass_plays) / len(team_plays) if len(team_plays) > 0 else 0.0,
                    "plays": len(team_plays),
                }
            
            logger.info(f"[nflfastR] Calculated EPA for {len(team_epa)} teams")
            
        except Exception as e:
            logger.error(f"[nflfastR] EPA calculation error: {e}")
        
        return team_epa
    
    # =========================================================================
    # PLAYER STATS
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics."""
        player_stats = []
        
        for year in years:
            try:
                url = NFLVERSE_URLS["player_stats"].format(year=year)
                df = await self._download_parquet(url)
                
                if df is None:
                    continue
                
                for _, row in df.iterrows():
                    stats = self._parse_player_stats(row)
                    if stats:
                        player_stats.append(stats)
                
                logger.info(f"[nflfastR] {year}: {len(df)} player stat records")
                
            except Exception as e:
                logger.debug(f"[nflfastR] Player stats {year} error: {e}")
        
        return player_stats
    
    def _parse_player_stats(self, row) -> Optional[Dict[str, Any]]:
        """Parse player stats row."""
        try:
            player_id = row.get("player_id")
            if pd.isna(player_id):
                return None
            
            return {
                "player_id": player_id,
                "player_name": row.get("player_name", ""),
                "player_display_name": row.get("player_display_name", ""),
                "position": row.get("position", ""),
                "position_group": row.get("position_group", ""),
                "team": row.get("recent_team", ""),
                "season": int(row.get("season", 0)),
                "season_type": row.get("season_type", "REG"),
                "week": int(row.get("week", 0)) if pd.notna(row.get("week")) else None,
                # Passing
                "completions": int(row.get("completions", 0)) if pd.notna(row.get("completions")) else 0,
                "attempts": int(row.get("attempts", 0)) if pd.notna(row.get("attempts")) else 0,
                "passing_yards": float(row.get("passing_yards", 0)) if pd.notna(row.get("passing_yards")) else 0,
                "passing_tds": int(row.get("passing_tds", 0)) if pd.notna(row.get("passing_tds")) else 0,
                "interceptions": int(row.get("interceptions", 0)) if pd.notna(row.get("interceptions")) else 0,
                "sacks": int(row.get("sacks", 0)) if pd.notna(row.get("sacks")) else 0,
                "sack_yards": float(row.get("sack_yards", 0)) if pd.notna(row.get("sack_yards")) else 0,
                "passing_air_yards": float(row.get("passing_air_yards", 0)) if pd.notna(row.get("passing_air_yards")) else 0,
                "passing_epa": float(row.get("passing_epa", 0)) if pd.notna(row.get("passing_epa")) else 0,
                # Rushing
                "carries": int(row.get("carries", 0)) if pd.notna(row.get("carries")) else 0,
                "rushing_yards": float(row.get("rushing_yards", 0)) if pd.notna(row.get("rushing_yards")) else 0,
                "rushing_tds": int(row.get("rushing_tds", 0)) if pd.notna(row.get("rushing_tds")) else 0,
                "rushing_epa": float(row.get("rushing_epa", 0)) if pd.notna(row.get("rushing_epa")) else 0,
                # Receiving
                "receptions": int(row.get("receptions", 0)) if pd.notna(row.get("receptions")) else 0,
                "targets": int(row.get("targets", 0)) if pd.notna(row.get("targets")) else 0,
                "receiving_yards": float(row.get("receiving_yards", 0)) if pd.notna(row.get("receiving_yards")) else 0,
                "receiving_tds": int(row.get("receiving_tds", 0)) if pd.notna(row.get("receiving_tds")) else 0,
                "receiving_air_yards": float(row.get("receiving_air_yards", 0)) if pd.notna(row.get("receiving_air_yards")) else 0,
                "receiving_epa": float(row.get("receiving_epa", 0)) if pd.notna(row.get("receiving_epa")) else 0,
                # Fantasy
                "fantasy_points": float(row.get("fantasy_points", 0)) if pd.notna(row.get("fantasy_points")) else 0,
                "fantasy_points_ppr": float(row.get("fantasy_points_ppr", 0)) if pd.notna(row.get("fantasy_points_ppr")) else 0,
            }
        except Exception as e:
            return None
    
    # =========================================================================
    # TEAM STATS
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect aggregated team statistics."""
        team_stats = []
        
        for year in years:
            try:
                epa_stats = await self.get_team_epa(season=year)
                
                for team, stats in epa_stats.items():
                    team_stats.append({
                        "team": team,
                        "team_name": NFL_TEAMS.get(team, team),
                        "season": year,
                        **stats,
                    })
                
            except Exception as e:
                logger.debug(f"[nflfastR] Team stats {year} error: {e}")
        
        return team_stats
    
    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================
    
    async def collect_historical(
        self,
        years_back: int = 10,
        data_types: List[str] = None,
    ) -> CollectorResult:
        """
        Collect historical NFL data.
        
        Args:
            years_back: Number of years back (max 25+, data from 1999)
            data_types: List of data types ["schedules", "player_stats", "team_stats"]
            
        Returns:
            CollectorResult with historical data
        """
        current_year = datetime.now().year
        min_year = max(1999, current_year - years_back)
        years = list(range(min_year, current_year + 1))
        
        if data_types is None:
            data_types = ["schedules"]
        
        all_data = {
            "games": [],
            "team_stats": [],
            "player_stats": [],
        }
        errors = []
        
        logger.info(f"[nflfastR] Collecting historical data for {min_year}-{current_year}")
        
        try:
            if "schedules" in data_types:
                games = await self._collect_schedules(years)
                all_data["games"] = games
                logger.info(f"[nflfastR] Historical: {len(games)} games")
            
            if "team_stats" in data_types:
                team_stats = await self._collect_team_stats(years)
                all_data["team_stats"] = team_stats
                
            if "player_stats" in data_types:
                player_stats = await self._collect_player_stats(years)
                all_data["player_stats"] = player_stats
                
        except Exception as e:
            logger.error(f"[nflfastR] Historical collection error: {e}")
            errors.append(str(e))
        
        total_records = sum(len(v) for v in all_data.values() if isinstance(v, list))
        
        return CollectorResult(
            success=total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "data_types": data_types},
        )
    
    # =========================================================================
    # ROSTERS COLLECTION
    # =========================================================================
    
    async def collect_rosters(
        self,
        years: List[int] = None,
    ) -> CollectorResult:
        """
        Collect NFL rosters from nflverse.
        
        Args:
            years: List of years (2002-present)
            
        Returns:
            CollectorResult with roster data
        """
        current_year = datetime.now().year
        if years is None:
            years = list(range(current_year - 4, current_year + 1))
        
        all_players = []
        errors = []
        
        for year in years:
            try:
                url = NFLVERSE_URLS["rosters"].format(year=year)
                logger.info(f"[nflfastR] Downloading rosters for {year}...")
                
                df = await self._download_parquet(url)
                
                if df is None:
                    # Try CSV fallback
                    url_csv = url.replace(".parquet", ".csv")
                    df = await self._download_csv(url_csv)
                
                if df is None:
                    errors.append(f"Failed to download {year} rosters")
                    continue
                
                for _, row in df.iterrows():
                    player = self._parse_roster_row(row, year)
                    if player:
                        all_players.append(player)
                
                logger.info(f"[nflfastR] {year}: {len(df)} roster entries")
                
            except Exception as e:
                logger.error(f"[nflfastR] Rosters {year} error: {e}")
                errors.append(f"{year}: {str(e)[:50]}")
        
        return CollectorResult(
            success=len(all_players) > 0,
            data={"players": all_players},
            records_count=len(all_players),
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "type": "rosters"},
        )
    
    def _parse_roster_row(self, row, year: int) -> Optional[Dict[str, Any]]:
        """Parse a roster row to player dict."""
        try:
            player_id = row.get("gsis_id") or row.get("player_id")
            if pd.isna(player_id) or not player_id:
                return None
            
            return {
                "player_id": player_id,
                "player_name": row.get("player_name", "") or row.get("full_name", ""),
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "position": row.get("position", ""),
                "position_group": row.get("position_group", ""),
                "team": row.get("team", ""),
                "jersey_number": int(row.get("jersey_number")) if pd.notna(row.get("jersey_number")) else None,
                "height": row.get("height", ""),
                "weight": int(row.get("weight")) if pd.notna(row.get("weight")) else None,
                "birth_date": str(row.get("birth_date")) if pd.notna(row.get("birth_date")) else None,
                "college": row.get("college", ""),
                "status": row.get("status", "ACT"),
                "season": year,
            }
        except Exception as e:
            return None
    
    async def collect_all(
        self,
        years: List[int] = None,
    ) -> CollectorResult:
        """
        Collect ALL NFL data: games, rosters, player_stats, team_stats.
        
        Args:
            years: List of years (default: last 10 years)
            
        Returns:
            CollectorResult with comprehensive data
        """
        current_year = datetime.now().year
        if years is None:
            years = list(range(current_year - 9, current_year + 1))
        
        all_data = {
            "games": [],
            "players": [],
            "player_stats": [],
            "team_stats": [],
        }
        errors = []
        
        logger.info(f"[nflfastR] Collecting ALL data for {min(years)}-{max(years)}")
        
        try:
            # 1. Collect games/schedules
            logger.info("[nflfastR] Step 1/4: Collecting games...")
            games = await self._collect_schedules(years)
            all_data["games"] = games
            logger.info(f"[nflfastR] Collected {len(games)} games")
            
            # 2. Collect rosters (players)
            logger.info("[nflfastR] Step 2/4: Collecting rosters...")
            roster_result = await self.collect_rosters(years)
            if roster_result.success and roster_result.data:
                all_data["players"] = roster_result.data.get("players", [])
            logger.info(f"[nflfastR] Collected {len(all_data['players'])} roster entries")
            
            # 3. Collect player stats
            logger.info("[nflfastR] Step 3/4: Collecting player stats...")
            player_stats = await self._collect_player_stats(years)
            all_data["player_stats"] = player_stats
            logger.info(f"[nflfastR] Collected {len(player_stats)} player stat records")
            
            # 4. Collect team stats (EPA-based)
            logger.info("[nflfastR] Step 4/4: Collecting team stats...")
            team_stats = await self._collect_team_stats(years)
            all_data["team_stats"] = team_stats
            logger.info(f"[nflfastR] Collected {len(team_stats)} team stat records")
            
        except Exception as e:
            logger.error(f"[nflfastR] collect_all error: {e}")
            errors.append(str(e))
        
        total_records = sum(len(v) for v in all_data.values() if isinstance(v, list))
        
        return CollectorResult(
            success=total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "type": "all"},
        )
    
    async def save_all_to_database(
        self,
        data: Dict[str, List[Dict]],
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Save ALL collected data to database."""
        results = {
            "games": 0,
            "players": 0,
            "player_stats": 0,
            "team_stats": 0,
        }
        
        # 1. Save games
        if data.get("games"):
            results["games"] = await self._save_games(data["games"], session)
            logger.info(f"[nflfastR] Saved {results['games']} games")
        
        # 2. Save rosters (players)
        if data.get("players"):
            results["players"] = await self.save_rosters_to_database(data["players"], session)
            logger.info(f"[nflfastR] Saved {results['players']} players from rosters")
        
        # 3. Save player stats
        if data.get("player_stats"):
            saved = await self.save_players_to_database(data["player_stats"], session)
            results["player_stats"] = saved
            logger.info(f"[nflfastR] Saved {saved} player stats")
        
        # 4. Save team stats
        if data.get("team_stats"):
            results["team_stats"] = await self.save_team_stats_to_database(data["team_stats"], session)
            logger.info(f"[nflfastR] Saved {results['team_stats']} team stats")
        
        return results
    
    async def save_rosters_to_database(
        self,
        roster_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save roster data (players) to database."""
        from app.models import Player, Team, Sport
        
        saved_count = 0
        
        try:
            # Get NFL sport
            sport_result = await session.execute(
                select(Sport).where(Sport.code == "NFL")
            )
            sport = sport_result.scalar_one_or_none()
            
            if not sport:
                logger.error("[nflfastR] NFL sport not found")
                return 0
            
            # Process each roster entry
            processed_ids = set()
            
            for player_data in roster_data:
                try:
                    player_id = player_data.get("player_id")
                    if not player_id or player_id in processed_ids:
                        continue
                    
                    processed_ids.add(player_id)
                    external_id = f"nfl_{player_id}"
                    
                    # Get player name
                    player_name = (
                        player_data.get("player_name") or
                        f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip() or
                        "Unknown"
                    )
                    
                    # Get team
                    team_abbr = player_data.get("team")
                    team = None
                    if team_abbr:
                        team_result = await session.execute(
                            select(Team).where(
                                and_(
                                    Team.sport_id == sport.id,
                                    Team.abbreviation == team_abbr
                                )
                            )
                        )
                        team = team_result.scalar_one_or_none()
                    
                    # Check if player exists
                    existing = await session.execute(
                        select(Player).where(Player.external_id == external_id)
                    )
                    player = existing.scalar_one_or_none()
                    
                    if player:
                        # Update existing player
                        player.name = player_name
                        player.position = player_data.get("position")
                        player.jersey_number = player_data.get("jersey_number")
                        player.weight = player_data.get("weight")
                        player.height = player_data.get("height")
                        if team:
                            player.team_id = team.id
                        # Check status
                        status = player_data.get("status", "ACT")
                        player.is_active = status in ["ACT", "Active", "RES"]
                    else:
                        # Parse birth date
                        birth_date = None
                        birth_str = player_data.get("birth_date")
                        if birth_str and birth_str != "nan":
                            try:
                                birth_date = datetime.strptime(birth_str[:10], "%Y-%m-%d").date()
                            except:
                                pass
                        
                        # Create new player
                        status = player_data.get("status", "ACT")
                        player = Player(
                            external_id=external_id,
                            name=player_name,
                            position=player_data.get("position"),
                            jersey_number=player_data.get("jersey_number"),
                            height=player_data.get("height"),
                            weight=player_data.get("weight"),
                            birth_date=birth_date,
                            team_id=team.id if team else None,
                            is_active=status in ["ACT", "Active", "RES"],
                        )
                        session.add(player)
                        saved_count += 1
                        
                except Exception as e:
                    logger.debug(f"[nflfastR] Error saving roster player: {e}")
                    continue
            
            await session.commit()
            logger.info(f"[nflfastR] Saved {saved_count} players from rosters")
            
        except Exception as e:
            logger.error(f"[nflfastR] Error saving rosters: {e}")
        
        return saved_count
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def _download_parquet(
        self,
        url: str,
        columns: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Download and parse a parquet file from URL."""
        try:
            client = await self.get_client()
            response = await client.get(url, follow_redirects=True, timeout=300.0)
            response.raise_for_status()
            
            # Read parquet from bytes
            df = pd.read_parquet(BytesIO(response.content), columns=columns)
            
            return df
            
        except Exception as e:
            logger.error(f"[nflfastR] Download error for {url}: {e}")
            return None
    
    async def _download_csv(
        self,
        url: str,
    ) -> Optional[pd.DataFrame]:
        """Download and parse a CSV file from URL."""
        try:
            client = await self.get_client()
            response = await client.get(url, follow_redirects=True, timeout=120.0)
            response.raise_for_status()
            
            # Read CSV from bytes
            df = pd.read_csv(BytesIO(response.content))
            
            return df
            
        except Exception as e:
            logger.error(f"[nflfastR] CSV download error for {url}: {e}")
            return None
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    async def save_to_database(
        self,
        data: Dict[str, List[Dict]],
        session: AsyncSession,
    ) -> int:
        """Save collected data to database (games, players, team_stats)."""
        total_saved = 0
        
        # 1. Save games
        if data.get("games"):
            saved = await self._save_games(data["games"], session)
            total_saved += saved
            logger.info(f"[nflfastR] Saved {saved} games")
        
        # 2. Save players and player_stats
        if data.get("player_stats"):
            saved = await self.save_players_to_database(data["player_stats"], session)
            total_saved += saved
            logger.info(f"[nflfastR] Saved {saved} players")
        
        # 3. Save team_stats
        if data.get("team_stats"):
            saved = await self.save_team_stats_to_database(data["team_stats"], session)
            total_saved += saved
            logger.info(f"[nflfastR] Saved {saved} team stats")
        
        return total_saved
    
    async def _save_games(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save game records to database."""
        saved_count = 0
        
        # Get NFL sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NFL")
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            logger.error("[nflfastR] NFL sport not found in database")
            return 0
        
        for game_data in games_data:
            try:
                # Get or create teams
                home_team = await self._get_or_create_team(
                    session, sport.id, game_data["home_team"]
                )
                away_team = await self._get_or_create_team(
                    session, sport.id, game_data["away_team"]
                )
                
                if not home_team or not away_team:
                    continue
                
                external_id = game_data.get("external_id")
                
                # Check if game exists by external_id
                existing = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = existing.scalars().first()
                
                if game:
                    # Update scores
                    if game_data.get("home_score") is not None:
                        game.home_score = game_data["home_score"]
                    if game_data.get("away_score") is not None:
                        game.away_score = game_data["away_score"]
                    if game_data.get("status"):
                        game.status = GameStatus(game_data["status"])
                else:
                    # Parse date
                    game_date_str = game_data.get("game_date")
                    if not game_date_str:
                        continue
                    
                    if isinstance(game_date_str, datetime):
                        scheduled_dt = game_date_str
                    else:
                        scheduled_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                    
                    if scheduled_dt.tzinfo:
                        scheduled_dt = scheduled_dt.replace(tzinfo=None)
                    
                    # Check for duplicates by teams and date
                    date_start = scheduled_dt - timedelta(hours=12)
                    date_end = scheduled_dt + timedelta(hours=12)
                    
                    dup_check = await session.execute(
                        select(Game).where(
                            and_(
                                Game.sport_id == sport.id,
                                Game.home_team_id == home_team.id,
                                Game.away_team_id == away_team.id,
                                Game.scheduled_at >= date_start,
                                Game.scheduled_at <= date_end,
                            )
                        )
                    )
                    existing_game = dup_check.scalars().first()
                    
                    if existing_game:
                        # Update
                        if game_data.get("home_score") is not None:
                            existing_game.home_score = game_data["home_score"]
                        if game_data.get("away_score") is not None:
                            existing_game.away_score = game_data["away_score"]
                        if external_id and not existing_game.external_id:
                            existing_game.external_id = external_id
                    else:
                        # Create new
                        game = Game(
                            sport_id=sport.id,
                            external_id=external_id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            scheduled_at=scheduled_dt,
                            status=GameStatus(game_data.get("status", "scheduled")),
                            home_score=game_data.get("home_score"),
                            away_score=game_data.get("away_score"),
                        )
                        session.add(game)
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"[nflfastR] Error saving game: {e}")
                continue
        
        await session.commit()
        logger.info(f"[nflfastR] Saved {saved_count} games to database")
        
        return saved_count
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_data: Dict[str, Any],
    ) -> Optional[Team]:
        """Get or create team record."""
        team_name = team_data.get("name")
        abbreviation = team_data.get("abbreviation")
        
        if not team_name and not abbreviation:
            return None
        
        # Try by name
        if team_name:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.name == team_name,
                    )
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Try by abbreviation
        if abbreviation:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.abbreviation == abbreviation,
                    )
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Create new
        team = Team(
            sport_id=sport_id,
            external_id=f"nfl_{abbreviation}" if abbreviation else None,
            name=team_name or NFL_TEAMS.get(abbreviation, abbreviation),
            abbreviation=abbreviation,
            is_active=True,
        )
        session.add(team)
        await session.flush()
        
        return team
    
    async def save_historical_to_database(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> Tuple[int, int]:
        """Save historical games to database."""
        saved = await self._save_games(games_data, session)
        return saved, 0
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not isinstance(data, dict):
            return False
        return True
    
    # =========================================================================
    # PLAYERS & STATS SAVING
    # =========================================================================
    
    async def save_players_to_database(
        self,
        player_stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """
        Save players and their stats to database.
        
        nflfastR player_stats includes:
        - player_id, player_name, player_display_name
        - position, position_group
        - team (abbreviation)
        - Various stats: completions, passing_yards, rushing_yards, etc.
        """
        from app.models import Player, PlayerStats, Team, Sport
        from collections import defaultdict
        
        saved_players = 0
        saved_stats = 0
        
        try:
            # Get NFL sport
            sport_result = await session.execute(
                select(Sport).where(Sport.code == "NFL")
            )
            sport = sport_result.scalar_one_or_none()
            
            if not sport:
                logger.error("[nflfastR] NFL sport not found")
                return 0
            
            # Track unique players we've processed
            processed_players = set()
            
            for stat_row in player_stats_data:
                try:
                    player_id = stat_row.get("player_id") or stat_row.get("gsis_id")
                    if not player_id or player_id in processed_players:
                        continue
                    
                    processed_players.add(player_id)
                    external_id = f"nfl_{player_id}"
                    
                    # Get player name
                    player_name = (
                        stat_row.get("player_display_name") or 
                        stat_row.get("player_name") or
                        stat_row.get("athlete_name") or
                        "Unknown"
                    )
                    
                    # Get team
                    team_abbr = stat_row.get("recent_team") or stat_row.get("team")
                    team = None
                    if team_abbr:
                        team_result = await session.execute(
                            select(Team).where(
                                and_(
                                    Team.sport_id == sport.id,
                                    Team.abbreviation == team_abbr
                                )
                            )
                        )
                        team = team_result.scalar_one_or_none()
                    
                    # Check if player exists
                    existing = await session.execute(
                        select(Player).where(Player.external_id == external_id)
                    )
                    player = existing.scalar_one_or_none()
                    
                    if player:
                        # Update
                        player.name = player_name
                        player.position = stat_row.get("position") or stat_row.get("position_group")
                        if team:
                            player.team_id = team.id
                        player.is_active = True
                    else:
                        # Create new player
                        player = Player(
                            external_id=external_id,
                            name=player_name,
                            position=stat_row.get("position") or stat_row.get("position_group"),
                            team_id=team.id if team else None,
                            is_active=True,
                        )
                        session.add(player)
                        await session.flush()  # Get player.id
                        saved_players += 1
                    
                    # Save individual stats
                    stat_types = [
                        "completions", "attempts", "passing_yards", "passing_tds", 
                        "interceptions", "sacks", "sack_yards", "sack_fumbles",
                        "rushing_yards", "rushing_tds", "rushing_fumbles",
                        "receptions", "targets", "receiving_yards", "receiving_tds",
                        "fantasy_points", "fantasy_points_ppr"
                    ]
                    
                    for stat_type in stat_types:
                        value = stat_row.get(stat_type)
                        if value is not None:
                            try:
                                # Skip NaN values
                                if pd.isna(value):
                                    continue
                                stat_record = PlayerStats(
                                    player_id=player.id,
                                    stat_type=stat_type,
                                    value=float(value),
                                )
                                session.add(stat_record)
                                saved_stats += 1
                            except:
                                pass
                            
                except Exception as e:
                    logger.debug(f"[nflfastR] Error saving player: {e}")
                    continue
            
            await session.commit()
            logger.info(f"[nflfastR] Saved {saved_players} players, {saved_stats} stats")
            
        except Exception as e:
            logger.error(f"[nflfastR] Error saving players: {e}")
        
        return saved_players
    
    async def save_team_stats_to_database(
        self,
        team_stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save team aggregate statistics to database."""
        from app.models import Team, TeamStats, Sport
        from collections import defaultdict
        
        saved_count = 0
        
        try:
            # Get NFL sport
            sport_result = await session.execute(
                select(Sport).where(Sport.code == "NFL")
            )
            sport = sport_result.scalar_one_or_none()
            
            if not sport:
                return 0
            
            # Aggregate stats by team
            team_aggregates = defaultdict(lambda: defaultdict(float))
            team_games = defaultdict(int)
            
            for stat_row in team_stats_data:
                team_abbr = stat_row.get("recent_team") or stat_row.get("team")
                if not team_abbr:
                    continue
                
                # Aggregate various stats
                for stat_type in ["passing_yards", "rushing_yards", "receiving_yards", "points"]:
                    value = stat_row.get(stat_type)
                    if value is not None:
                        try:
                            if not pd.isna(value):
                                team_aggregates[team_abbr][stat_type] += float(value)
                        except:
                            pass
                
                team_games[team_abbr] += 1
            
            # Save team stats
            for team_abbr, stats in team_aggregates.items():
                team_result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.abbreviation == team_abbr
                        )
                    )
                )
                team = team_result.scalar_one_or_none()
                
                if not team:
                    continue
                
                games_played = team_games.get(team_abbr, 1)
                
                for stat_type, total_value in stats.items():
                    try:
                        # Check if exists
                        existing = await session.execute(
                            select(TeamStats).where(
                                and_(
                                    TeamStats.team_id == team.id,
                                    TeamStats.stat_type == stat_type,
                                )
                            )
                        )
                        team_stat = existing.scalar_one_or_none()
                        
                        if team_stat:
                            team_stat.value = total_value
                            team_stat.games_played = games_played
                            team_stat.computed_at = datetime.utcnow()
                        else:
                            team_stat = TeamStats(
                                team_id=team.id,
                                stat_type=stat_type,
                                value=total_value,
                                games_played=games_played,
                            )
                            session.add(team_stat)
                            saved_count += 1
                            
                    except Exception as e:
                        logger.debug(f"[nflfastR] Error saving team stat: {e}")
            
            await session.commit()
            logger.info(f"[nflfastR] Saved {saved_count} team stats")
            
        except Exception as e:
            logger.error(f"[nflfastR] Error saving team stats: {e}")
        
        return saved_count


# =============================================================================
# SINGLETON & REGISTRATION
# =============================================================================

nflfastr_collector = NFLFastRCollector()

try:
    collector_manager.register("nflfastr", nflfastr_collector)
except:
    pass  # Already registered