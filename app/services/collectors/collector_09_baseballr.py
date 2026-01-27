"""
ROYALEY - baseballR Data Collector
Phase 1: Data Collection Services

Collects comprehensive MLB data using pybaseball library and MLB Stats API.
Features: Statcast data, player stats, team stats, game schedules, 85+ features.

Data Sources:
- pybaseball: https://github.com/jldbc/pybaseball (Python port of baseballr)
- MLB Stats API: https://statsapi.mlb.com
- Baseball Savant: https://baseballsavant.mlb.com

FREE data - no API key required!

Key Data Types:
- Statcast: Pitch-level data with exit velocity, launch angle, spin rate
- Player Stats: Batting, pitching, fielding stats
- Team Stats: Team performance metrics
- Game Schedules: Full schedules with results
- FanGraphs: Advanced metrics (wRC+, FIP, WAR)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx
import pandas as pd
import numpy as np

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MLB STATS API URLS
# =============================================================================

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

MLB_API_URLS = {
    "schedule": f"{MLB_API_BASE}/schedule",
    "teams": f"{MLB_API_BASE}/teams",
    "team": f"{MLB_API_BASE}/teams/{{team_id}}",
    "roster": f"{MLB_API_BASE}/teams/{{team_id}}/roster",
    "player": f"{MLB_API_BASE}/people/{{player_id}}",
    "player_stats": f"{MLB_API_BASE}/people/{{player_id}}/stats",
    "standings": f"{MLB_API_BASE}/standings",
    "game": f"{MLB_API_BASE}/game/{{game_pk}}/feed/live",
    "game_boxscore": f"{MLB_API_BASE}/game/{{game_pk}}/boxscore",
}

# Baseball Savant / Statcast
SAVANT_BASE = "https://baseballsavant.mlb.com"
SAVANT_URLS = {
    "statcast_search": f"{SAVANT_BASE}/statcast_search/csv",
    "leaderboard": f"{SAVANT_BASE}/leaderboard",
}


# =============================================================================
# MLB TEAMS
# =============================================================================

MLB_TEAMS = {
    # American League East
    108: {"name": "Los Angeles Angels", "abbr": "LAA", "division": "AL West"},
    109: {"name": "Arizona Diamondbacks", "abbr": "ARI", "division": "NL West"},
    110: {"name": "Baltimore Orioles", "abbr": "BAL", "division": "AL East"},
    111: {"name": "Boston Red Sox", "abbr": "BOS", "division": "AL East"},
    112: {"name": "Chicago Cubs", "abbr": "CHC", "division": "NL Central"},
    113: {"name": "Cincinnati Reds", "abbr": "CIN", "division": "NL Central"},
    114: {"name": "Cleveland Guardians", "abbr": "CLE", "division": "AL Central"},
    115: {"name": "Colorado Rockies", "abbr": "COL", "division": "NL West"},
    116: {"name": "Detroit Tigers", "abbr": "DET", "division": "AL Central"},
    117: {"name": "Houston Astros", "abbr": "HOU", "division": "AL West"},
    118: {"name": "Kansas City Royals", "abbr": "KC", "division": "AL Central"},
    119: {"name": "Los Angeles Dodgers", "abbr": "LAD", "division": "NL West"},
    120: {"name": "Washington Nationals", "abbr": "WSH", "division": "NL East"},
    121: {"name": "New York Mets", "abbr": "NYM", "division": "NL East"},
    133: {"name": "Oakland Athletics", "abbr": "OAK", "division": "AL West"},
    134: {"name": "Pittsburgh Pirates", "abbr": "PIT", "division": "NL Central"},
    135: {"name": "San Diego Padres", "abbr": "SD", "division": "NL West"},
    136: {"name": "Seattle Mariners", "abbr": "SEA", "division": "AL West"},
    137: {"name": "San Francisco Giants", "abbr": "SF", "division": "NL West"},
    138: {"name": "St. Louis Cardinals", "abbr": "STL", "division": "NL Central"},
    139: {"name": "Tampa Bay Rays", "abbr": "TB", "division": "AL East"},
    140: {"name": "Texas Rangers", "abbr": "TEX", "division": "AL West"},
    141: {"name": "Toronto Blue Jays", "abbr": "TOR", "division": "AL East"},
    142: {"name": "Minnesota Twins", "abbr": "MIN", "division": "AL Central"},
    143: {"name": "Philadelphia Phillies", "abbr": "PHI", "division": "NL East"},
    144: {"name": "Atlanta Braves", "abbr": "ATL", "division": "NL East"},
    145: {"name": "Chicago White Sox", "abbr": "CWS", "division": "AL Central"},
    146: {"name": "Miami Marlins", "abbr": "MIA", "division": "NL East"},
    147: {"name": "New York Yankees", "abbr": "NYY", "division": "AL East"},
    158: {"name": "Milwaukee Brewers", "abbr": "MIL", "division": "NL Central"},
}

# Abbreviation to team ID mapping
MLB_ABBR_TO_ID = {v["abbr"]: k for k, v in MLB_TEAMS.items()}


# =============================================================================
# KEY STATCAST FEATURES
# =============================================================================

STATCAST_KEY_COLUMNS = [
    # Identifiers
    "game_pk", "game_date", "player_name", "batter", "pitcher",
    "events", "description",
    # Pitch Data
    "pitch_type", "release_speed", "release_spin_rate",
    "release_extension", "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z",
    "zone", "spin_axis",
    # Batted Ball Data
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    # Count/Situation
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b",
    # Teams
    "home_team", "away_team", "bat_score", "fld_score", "post_bat_score",
    # Results
    "hit_location", "bb_type", "hc_x", "hc_y",
    "delta_home_win_exp", "delta_run_exp",
]


class BaseballRCollector(BaseCollector):
    """
    Collector for MLB data using pybaseball library and MLB Stats API.
    
    Features:
    - Statcast data (pitch-level with 85+ features)
    - Player batting/pitching stats
    - Team standings and stats
    - Game schedules and results
    - FanGraphs advanced metrics
    
    FREE - No API key required!
    """
    
    def __init__(self):
        super().__init__(
            name="baseballr",
            base_url="https://statsapi.mlb.com/api/v1",
            rate_limit=30,
            rate_window=60,
            timeout=180.0,
            max_retries=3,
        )
        self.data_dir = Path(settings.MODEL_STORAGE_PATH) / "mlb_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pybaseball_available = self._check_pybaseball()
        
    def _check_pybaseball(self) -> bool:
        """Check if pybaseball is installed."""
        try:
            import pybaseball
            return True
        except ImportError:
            logger.info("[baseballR] pybaseball not installed, using MLB Stats API only")
            return False
    
    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sport_code: str = "MLB",
        collect_type: str = "schedules",
        years: List[int] = None,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect MLB data.
        
        Args:
            sport_code: Must be "MLB"
            collect_type: "schedules", "teams", "rosters", "player_stats", 
                          "team_stats", "statcast", "all"
            years: List of years (default: last 5 years)
            
        Returns:
            CollectorResult with collected data
        """
        if sport_code.upper() != "MLB":
            return CollectorResult(
                success=False,
                error="baseballR only supports MLB data",
                records_count=0,
            )
        
        current_year = datetime.now().year
        if years is None:
            years = list(range(current_year - 4, current_year + 1))
        
        all_data = {
            "games": [],
            "teams": [],
            "rosters": [],
            "player_stats": [],
            "team_stats": [],
            "statcast": [],
        }
        errors = []
        
        try:
            if collect_type in ["teams", "all"]:
                teams = await self._collect_teams()
                all_data["teams"] = teams
                logger.info(f"[baseballR] Collected {len(teams)} teams")
                
            if collect_type in ["schedules", "all"]:
                games = await self._collect_schedules(years)
                all_data["games"] = games
                logger.info(f"[baseballR] Collected {len(games)} games")
                
            if collect_type in ["rosters", "all"]:
                rosters = await self._collect_rosters()
                all_data["rosters"] = rosters
                logger.info(f"[baseballR] Collected {len(rosters)} roster entries")
                
            if collect_type in ["player_stats", "all"]:
                player_stats = await self._collect_player_stats(years)
                all_data["player_stats"] = player_stats
                logger.info(f"[baseballR] Collected {len(player_stats)} player stats")
                
            if collect_type in ["team_stats", "all"]:
                team_stats = await self._collect_team_stats(years)
                all_data["team_stats"] = team_stats
                logger.info(f"[baseballR] Collected {len(team_stats)} team stats")
                
            if collect_type == "statcast":
                # Statcast is heavy - only collect if explicitly requested
                statcast = await self._collect_statcast(years)
                all_data["statcast"] = statcast
                logger.info(f"[baseballR] Collected {len(statcast)} statcast records")
                
        except Exception as e:
            logger.error(f"[baseballR] Collection error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
    # TEAMS
    # =========================================================================
    
    async def _collect_teams(self) -> List[Dict[str, Any]]:
        """Collect MLB team information."""
        teams = []
        
        try:
            client = await self.get_client()
            
            # Get all MLB teams
            url = f"{MLB_API_BASE}/teams?sportId=1"
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            data = response.json()
            
            for team in data.get("teams", []):
                team_id = team.get("id")
                if team_id not in MLB_TEAMS:
                    continue
                
                teams.append({
                    "mlb_id": team_id,
                    "name": team.get("name"),
                    "abbreviation": team.get("abbreviation"),
                    "city": team.get("locationName"),
                    "venue": team.get("venue", {}).get("name"),
                    "venue_id": team.get("venue", {}).get("id"),
                    "division": team.get("division", {}).get("name"),
                    "league": team.get("league", {}).get("name"),
                    "first_year": team.get("firstYearOfPlay"),
                })
            
            logger.info(f"[baseballR] Loaded {len(teams)} MLB teams")
            
        except Exception as e:
            logger.error(f"[baseballR] Teams collection error: {e}")
        
        return teams
    
    # =========================================================================
    # SCHEDULES / GAMES
    # =========================================================================
    
    async def _collect_schedules(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect game schedules and results."""
        games = []
        
        for year in years:
            try:
                # MLB season typically runs March-October
                start_date = f"{year}-03-01"
                end_date = f"{year}-11-15"
                
                client = await self.get_client()
                url = (f"{MLB_API_BASE}/schedule"
                       f"?sportId=1"
                       f"&startDate={start_date}"
                       f"&endDate={end_date}"
                       f"&gameType=R,F,D,L,W,C,P"  # Regular, Finals, Div, League, Wild Card, Championship, Postseason
                       f"&hydrate=team,venue")
                
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()
                
                data = response.json()
                
                for date_entry in data.get("dates", []):
                    for game in date_entry.get("games", []):
                        parsed = self._parse_game(game, year)
                        if parsed:
                            games.append(parsed)
                
                logger.info(f"[baseballR] {year}: {len([g for g in games if g.get('season') == year])} games")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"[baseballR] Schedule {year} error: {e}")
        
        logger.info(f"[baseballR] Total {len(games)} games collected")
        return games
    
    def _parse_game(self, game: Dict, year: int) -> Optional[Dict[str, Any]]:
        """Parse MLB API game data."""
        try:
            game_pk = game.get("gamePk")
            if not game_pk:
                return None
            
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            
            home_team = home.get("team", {})
            away_team = away.get("team", {})
            
            # Get game date
            game_date_str = game.get("gameDate")
            if game_date_str:
                game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                game_date = game_date.replace(tzinfo=None)
            else:
                game_date = datetime(year, 4, 1)
            
            # Get status
            status = game.get("status", {}).get("detailedState", "Scheduled")
            if status in ["Final", "Game Over", "Completed Early"]:
                game_status = "final"
            elif status in ["In Progress", "Warmup", "Pre-Game"]:
                game_status = "in_progress"
            else:
                game_status = "scheduled"
            
            # Get scores
            home_score = home.get("score")
            away_score = away.get("score")
            
            # Get venue
            venue = game.get("venue", {})
            
            return {
                "sport_code": "MLB",
                "external_id": f"mlb_{game_pk}",
                "game_pk": game_pk,
                "home_team": {
                    "name": home_team.get("name"),
                    "abbreviation": home_team.get("abbreviation"),
                    "mlb_id": home_team.get("id"),
                },
                "away_team": {
                    "name": away_team.get("name"),
                    "abbreviation": away_team.get("abbreviation"),
                    "mlb_id": away_team.get("id"),
                },
                "game_date": game_date.isoformat(),
                "status": game_status,
                "home_score": int(home_score) if home_score is not None else None,
                "away_score": int(away_score) if away_score is not None else None,
                "season": year,
                "game_type": game.get("gameType"),
                "venue": venue.get("name"),
                "venue_id": venue.get("id"),
                "day_night": game.get("dayNight"),
                "series_description": game.get("seriesDescription"),
            }
            
        except Exception as e:
            logger.debug(f"[baseballR] Game parse error: {e}")
            return None
    
    # =========================================================================
    # ROSTERS
    # =========================================================================
    
    async def _collect_rosters(self) -> List[Dict[str, Any]]:
        """Collect current rosters for all teams."""
        rosters = []
        
        client = await self.get_client()
        
        for team_id, team_info in MLB_TEAMS.items():
            try:
                url = f"{MLB_API_BASE}/teams/{team_id}/roster?rosterType=active"
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                
                for player in data.get("roster", []):
                    person = player.get("person", {})
                    position = player.get("position", {})
                    
                    rosters.append({
                        "mlb_id": person.get("id"),
                        "name": person.get("fullName"),
                        "team": team_info["abbr"],
                        "team_id": team_id,
                        "position": position.get("abbreviation"),
                        "position_name": position.get("name"),
                        "jersey_number": player.get("jerseyNumber"),
                        "status": player.get("status", {}).get("description"),
                    })
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.debug(f"[baseballR] Roster {team_info['abbr']} error: {e}")
        
        logger.info(f"[baseballR] Collected {len(rosters)} roster entries")
        return rosters
    
    # =========================================================================
    # PLAYER STATS
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics using pybaseball or MLB API."""
        player_stats = []
        
        if self._pybaseball_available:
            player_stats = await self._collect_player_stats_pybaseball(years)
        else:
            player_stats = await self._collect_player_stats_api(years)
        
        return player_stats
    
    async def _collect_player_stats_pybaseball(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player stats using pybaseball library."""
        player_stats = []
        
        try:
            import pybaseball
            
            # Cache to avoid duplicate API calls
            pybaseball.cache.enable()
            
            for year in years:
                try:
                    # Get batting stats
                    logger.info(f"[baseballR] Collecting batting stats for {year}...")
                    batting = pybaseball.batting_stats(year, qual=1)
                    
                    if batting is not None and len(batting) > 0:
                        for _, row in batting.iterrows():
                            stats = {
                                "player_name": row.get("Name"),
                                "player_id": row.get("IDfg"),
                                "team": row.get("Team"),
                                "season": year,
                                "stat_type": "batting",
                                "games": row.get("G"),
                                "at_bats": row.get("AB"),
                                "hits": row.get("H"),
                                "doubles": row.get("2B"),
                                "triples": row.get("3B"),
                                "home_runs": row.get("HR"),
                                "runs": row.get("R"),
                                "rbi": row.get("RBI"),
                                "walks": row.get("BB"),
                                "strikeouts": row.get("SO"),
                                "stolen_bases": row.get("SB"),
                                "batting_avg": row.get("AVG"),
                                "obp": row.get("OBP"),
                                "slg": row.get("SLG"),
                                "ops": row.get("OPS"),
                                "woba": row.get("wOBA"),
                                "wrc_plus": row.get("wRC+"),
                                "war": row.get("WAR"),
                            }
                            player_stats.append(stats)
                        
                        logger.info(f"[baseballR] {year}: {len(batting)} batting stats")
                    
                    # Get pitching stats
                    logger.info(f"[baseballR] Collecting pitching stats for {year}...")
                    pitching = pybaseball.pitching_stats(year, qual=1)
                    
                    if pitching is not None and len(pitching) > 0:
                        for _, row in pitching.iterrows():
                            stats = {
                                "player_name": row.get("Name"),
                                "player_id": row.get("IDfg"),
                                "team": row.get("Team"),
                                "season": year,
                                "stat_type": "pitching",
                                "games": row.get("G"),
                                "games_started": row.get("GS"),
                                "wins": row.get("W"),
                                "losses": row.get("L"),
                                "saves": row.get("SV"),
                                "innings_pitched": row.get("IP"),
                                "hits_allowed": row.get("H"),
                                "runs_allowed": row.get("R"),
                                "earned_runs": row.get("ER"),
                                "walks": row.get("BB"),
                                "strikeouts": row.get("SO"),
                                "home_runs_allowed": row.get("HR"),
                                "era": row.get("ERA"),
                                "whip": row.get("WHIP"),
                                "fip": row.get("FIP"),
                                "xfip": row.get("xFIP"),
                                "war": row.get("WAR"),
                                "k_per_9": row.get("K/9"),
                                "bb_per_9": row.get("BB/9"),
                            }
                            player_stats.append(stats)
                        
                        logger.info(f"[baseballR] {year}: {len(pitching)} pitching stats")
                    
                except Exception as e:
                    logger.error(f"[baseballR] pybaseball {year} error: {e}")
                    
        except Exception as e:
            logger.error(f"[baseballR] pybaseball collection error: {e}")
        
        return player_stats
    
    async def _collect_player_stats_api(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player stats using MLB Stats API (fallback)."""
        player_stats = []
        
        client = await self.get_client()
        
        # Get stats for each team's roster
        for team_id, team_info in MLB_TEAMS.items():
            for year in years:
                try:
                    # Get team stats for the season
                    url = (f"{MLB_API_BASE}/teams/{team_id}/stats"
                           f"?season={year}"
                           f"&group=hitting,pitching"
                           f"&stats=season")
                    
                    response = await client.get(url, timeout=30.0)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    for stat_group in data.get("stats", []):
                        group_type = stat_group.get("group", {}).get("displayName")
                        
                        for split in stat_group.get("splits", []):
                            player = split.get("player", {})
                            stats = split.get("stat", {})
                            
                            if not player:
                                continue
                            
                            stat_record = {
                                "player_name": player.get("fullName"),
                                "player_id": player.get("id"),
                                "team": team_info["abbr"],
                                "season": year,
                                "stat_type": "batting" if group_type == "hitting" else "pitching",
                            }
                            stat_record.update(stats)
                            player_stats.append(stat_record)
                    
                    # Rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.debug(f"[baseballR] API stats {team_info['abbr']} {year} error: {e}")
        
        return player_stats
    
    # =========================================================================
    # TEAM STATS
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team statistics and standings."""
        team_stats = []
        
        if self._pybaseball_available:
            team_stats = await self._collect_team_stats_pybaseball(years)
        else:
            team_stats = await self._collect_team_stats_api(years)
        
        return team_stats
    
    async def _collect_team_stats_pybaseball(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team stats using pybaseball."""
        team_stats = []
        
        try:
            import pybaseball
            
            for year in years:
                try:
                    # Get team batting stats
                    logger.info(f"[baseballR] Collecting team batting for {year}...")
                    batting = pybaseball.team_batting(year)
                    
                    if batting is not None and len(batting) > 0:
                        for _, row in batting.iterrows():
                            team_name = row.get("Team", row.get("Tm", "Unknown"))
                            
                            team_stats.append({
                                "team": team_name,
                                "season": year,
                                "stat_category": "batting",
                                "games": row.get("G"),
                                "runs": row.get("R"),
                                "hits": row.get("H"),
                                "doubles": row.get("2B"),
                                "triples": row.get("3B"),
                                "home_runs": row.get("HR"),
                                "rbi": row.get("RBI"),
                                "walks": row.get("BB"),
                                "strikeouts": row.get("SO"),
                                "batting_avg": row.get("AVG"),
                                "obp": row.get("OBP"),
                                "slg": row.get("SLG"),
                                "ops": row.get("OPS"),
                            })
                    
                    # Get team pitching stats
                    logger.info(f"[baseballR] Collecting team pitching for {year}...")
                    pitching = pybaseball.team_pitching(year)
                    
                    if pitching is not None and len(pitching) > 0:
                        for _, row in pitching.iterrows():
                            team_name = row.get("Team", row.get("Tm", "Unknown"))
                            
                            team_stats.append({
                                "team": team_name,
                                "season": year,
                                "stat_category": "pitching",
                                "games": row.get("G"),
                                "wins": row.get("W"),
                                "losses": row.get("L"),
                                "era": row.get("ERA"),
                                "innings_pitched": row.get("IP"),
                                "hits_allowed": row.get("H"),
                                "runs_allowed": row.get("R"),
                                "earned_runs": row.get("ER"),
                                "walks": row.get("BB"),
                                "strikeouts": row.get("SO"),
                                "whip": row.get("WHIP"),
                            })
                    
                    logger.info(f"[baseballR] {year}: team stats collected")
                    
                except Exception as e:
                    logger.error(f"[baseballR] Team stats {year} error: {e}")
                    
        except Exception as e:
            logger.error(f"[baseballR] pybaseball team stats error: {e}")
        
        return team_stats
    
    async def _collect_team_stats_api(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team stats using MLB Stats API."""
        team_stats = []
        
        client = await self.get_client()
        
        for year in years:
            try:
                # Get standings which include W/L records
                url = f"{MLB_API_BASE}/standings?leagueId=103,104&season={year}"
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                
                for record in data.get("records", []):
                    division = record.get("division", {}).get("name")
                    
                    for team_record in record.get("teamRecords", []):
                        team = team_record.get("team", {})
                        
                        team_stats.append({
                            "team": team.get("name"),
                            "team_abbr": MLB_TEAMS.get(team.get("id"), {}).get("abbr"),
                            "season": year,
                            "division": division,
                            "wins": team_record.get("wins"),
                            "losses": team_record.get("losses"),
                            "win_pct": team_record.get("winningPercentage"),
                            "games_back": team_record.get("gamesBack"),
                            "runs_scored": team_record.get("runsScored"),
                            "runs_allowed": team_record.get("runsAllowed"),
                            "run_differential": team_record.get("runDifferential"),
                            "home_wins": team_record.get("records", {}).get("splitRecords", [{}])[0].get("wins") if team_record.get("records") else None,
                            "home_losses": team_record.get("records", {}).get("splitRecords", [{}])[0].get("losses") if team_record.get("records") else None,
                        })
                
                logger.info(f"[baseballR] {year}: standings collected")
                
                # Rate limiting
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"[baseballR] Standings {year} error: {e}")
        
        return team_stats
    
    # =========================================================================
    # STATCAST DATA
    # =========================================================================
    
    async def _collect_statcast(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect Statcast pitch-level data.
        
        WARNING: This is large data (~1M+ rows per season)
        Only call when explicitly requested.
        """
        statcast_data = []
        
        if not self._pybaseball_available:
            logger.warning("[baseballR] Statcast requires pybaseball library")
            return statcast_data
        
        try:
            import pybaseball
            
            for year in years:
                try:
                    # Statcast data is available from 2015+
                    if year < 2015:
                        continue
                    
                    logger.info(f"[baseballR] Collecting Statcast for {year} (this may take a while)...")
                    
                    # Get season date range
                    start_date = f"{year}-03-28"
                    end_date = f"{year}-10-05"
                    
                    df = pybaseball.statcast(start_dt=start_date, end_dt=end_date)
                    
                    if df is not None and len(df) > 0:
                        # Convert to records (only key columns to save memory)
                        cols_to_keep = [c for c in STATCAST_KEY_COLUMNS if c in df.columns]
                        df_subset = df[cols_to_keep]
                        
                        statcast_data.extend(df_subset.to_dict('records'))
                        logger.info(f"[baseballR] {year}: {len(df):,} Statcast pitches")
                    
                except Exception as e:
                    logger.error(f"[baseballR] Statcast {year} error: {e}")
                    
        except Exception as e:
            logger.error(f"[baseballR] Statcast collection error: {e}")
        
        return statcast_data
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    async def save_to_database(
        self,
        data: Dict[str, List[Dict]],
        session: AsyncSession,
    ) -> int:
        """Save collected data to database."""
        total_saved = 0
        
        if data.get("teams"):
            saved = await self._save_teams(data["teams"], session)
            total_saved += saved
        
        if data.get("games"):
            saved = await self._save_games(data["games"], session)
            total_saved += saved
        
        if data.get("rosters"):
            saved = await self.save_rosters_to_database(data["rosters"], session)
            total_saved += saved
        
        if data.get("player_stats"):
            saved = await self.save_players_to_database(data["player_stats"], session)
            total_saved += saved
        
        if data.get("team_stats"):
            saved = await self.save_team_stats_to_database(data["team_stats"], session)
            total_saved += saved
        
        return total_saved
    
    async def _save_teams(
        self,
        teams_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save team records to database."""
        saved_count = 0
        
        # Get MLB sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "MLB")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[baseballR] MLB sport not found in database")
            return 0
        
        for team_data in teams_data:
            try:
                abbr = team_data.get("abbreviation")
                name = team_data.get("name")
                
                if not abbr and not name:
                    continue
                
                # Check if team exists
                existing = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            or_(
                                Team.abbreviation == abbr,
                                Team.name == name,
                            )
                        )
                    )
                )
                team = existing.scalars().first()
                
                if team:
                    # Update
                    team.name = name
                    team.abbreviation = abbr
                    team.city = team_data.get("city")
                    team.conference = team_data.get("league")
                    team.division = team_data.get("division")
                else:
                    # Create new
                    team = Team(
                        sport_id=sport.id,
                        external_id=f"mlb_{team_data.get('mlb_id')}",
                        name=name,
                        abbreviation=abbr,
                        city=team_data.get("city"),
                        conference=team_data.get("league"),
                        division=team_data.get("division"),
                        is_active=True,
                    )
                    session.add(team)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[baseballR] Error saving team: {e}")
        
        logger.info(f"[baseballR] Saved {saved_count} teams")
        return saved_count
    
    async def _save_games(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save game records to database."""
        saved_count = 0
        
        # Get MLB sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "MLB")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[baseballR] MLB sport not found")
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
                
                # Check if game exists
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
                    if isinstance(game_date_str, datetime):
                        scheduled_dt = game_date_str
                    else:
                        scheduled_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                    
                    if scheduled_dt.tzinfo:
                        scheduled_dt = scheduled_dt.replace(tzinfo=None)
                    
                    # Check for duplicates
                    date_start = scheduled_dt - timedelta(hours=6)
                    date_end = scheduled_dt + timedelta(hours=6)
                    
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
                        # Update existing
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
                logger.debug(f"[baseballR] Error saving game: {e}")
        
        logger.info(f"[baseballR] Saved {saved_count} games")
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
        
        # Try by abbreviation first
        if abbreviation:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.abbreviation == abbreviation,
                    )
                )
            )
            team = result.scalars().first()
            if team:
                return team
        
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
            team = result.scalars().first()
            if team:
                return team
        
        # Get division from MLB_TEAMS
        mlb_id = team_data.get("mlb_id")
        division = None
        if mlb_id and mlb_id in MLB_TEAMS:
            division = MLB_TEAMS[mlb_id].get("division")
        
        # Create new team
        team = Team(
            sport_id=sport_id,
            external_id=f"mlb_{mlb_id}" if mlb_id else None,
            name=team_name,
            abbreviation=abbreviation,
            division=division,
            is_active=True,
        )
        session.add(team)
        await session.flush()
        
        return team
    
    async def save_rosters_to_database(
        self,
        rosters_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save player rosters to database."""
        from app.models import Player, Team, Sport
        
        saved_count = 0
        batch_count = 0
        
        # Get MLB sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "MLB")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[baseballR] MLB sport not found")
            return 0
        
        for roster_entry in rosters_data:
            player_id = roster_entry.get("mlb_id")
            if not player_id:
                continue
            
            external_id = f"mlb_{player_id}"
            player_name = roster_entry.get("name", "Unknown")
            
            # Get team
            team_abbr = roster_entry.get("team")
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
                team = team_result.scalars().first()
            
            # Check if player exists
            existing = await session.execute(
                select(Player).where(Player.external_id == external_id)
            )
            player = existing.scalars().first()
            
            if player:
                # Update existing player
                player.name = player_name
                player.position = roster_entry.get("position")
                player.jersey_number = roster_entry.get("jersey_number")
                if team:
                    player.team_id = team.id
                player.is_active = True
            else:
                # Create new player
                player = Player(
                    external_id=external_id,
                    name=player_name,
                    position=roster_entry.get("position"),
                    jersey_number=roster_entry.get("jersey_number"),
                    team_id=team.id if team else None,
                    is_active=True,
                )
                session.add(player)
                saved_count += 1
            
            # Flush in batches
            batch_count += 1
            if batch_count >= 500:
                await session.flush()
                logger.info(f"[baseballR] Flushed batch, {saved_count} new players so far")
                batch_count = 0
        
        logger.info(f"[baseballR] Saved {saved_count} players from rosters")
        return saved_count
    
    async def save_players_to_database(
        self,
        player_stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save MLB players and their stats to database."""
        from app.models import Player, PlayerStats, Team, Sport
        
        saved_players = 0
        saved_stats = 0
        
        # Get MLB sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "MLB")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[baseballR] MLB sport not found")
            return 0
        
        # Track unique players
        processed_players = set()
        
        for stat_row in player_stats_data:
            player_id = stat_row.get("player_id")
            player_name = stat_row.get("player_name")
            
            if not player_id or player_id in processed_players:
                continue
            
            processed_players.add(player_id)
            external_id = f"mlb_{player_id}"
            
            # Get team
            team_abbr = stat_row.get("team")
            team = None
            if team_abbr:
                team_result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            or_(
                                Team.abbreviation == team_abbr,
                                Team.name.ilike(f"%{team_abbr}%")
                            )
                        )
                    )
                )
                team = team_result.scalars().first()
            
            # Check if player exists
            existing = await session.execute(
                select(Player).where(Player.external_id == external_id)
            )
            player = existing.scalars().first()
            
            is_new_player = player is None
            
            if player:
                # Update
                player.name = player_name
                if team:
                    player.team_id = team.id
                player.is_active = True
            else:
                # Create new player
                player = Player(
                    external_id=external_id,
                    name=player_name,
                    team_id=team.id if team else None,
                    is_active=True,
                )
                session.add(player)
                await session.flush()
                saved_players += 1
            
            # Only save stats for new players
            if is_new_player:
                stat_types = [
                    "games", "at_bats", "hits", "doubles", "triples", "home_runs",
                    "runs", "rbi", "walks", "strikeouts", "stolen_bases",
                    "batting_avg", "obp", "slg", "ops", "woba", "wrc_plus", "war",
                    "games_started", "wins", "losses", "saves", "innings_pitched",
                    "hits_allowed", "runs_allowed", "earned_runs", "era", "whip",
                    "fip", "xfip", "k_per_9", "bb_per_9"
                ]
                
                for stat_type in stat_types:
                    value = stat_row.get(stat_type)
                    if value is not None:
                        try:
                            if not pd.isna(value):
                                season = stat_row.get("season", 0)
                                full_stat_type = f"{stat_type}_{season}"
                                
                                stat_record = PlayerStats(
                                    player_id=player.id,
                                    stat_type=full_stat_type,
                                    value=float(value),
                                )
                                session.add(stat_record)
                                saved_stats += 1
                        except (ValueError, TypeError):
                            pass
        
        logger.info(f"[baseballR] Saved {saved_players} players, {saved_stats} stats")
        return saved_players
    
    async def save_team_stats_to_database(
        self,
        team_stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save MLB team statistics to database."""
        from app.models import Team, TeamStats, Sport
        
        saved_count = 0
        
        # Get MLB sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "MLB")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[baseballR] MLB sport not found")
            return 0
        
        # Stat types to save
        stat_types = [
            "wins", "losses", "win_pct", "runs_scored", "runs_allowed",
            "run_differential", "games", "batting_avg", "obp", "slg", "ops",
            "era", "whip", "home_runs", "strikeouts"
        ]
        
        for stat_row in team_stats_data:
            team_name = stat_row.get("team") or stat_row.get("team_abbr")
            if not team_name:
                continue
            
            season = stat_row.get("season", 0)
            
            # Find team
            team_result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport.id,
                        or_(
                            Team.name.ilike(f"%{team_name}%"),
                            Team.abbreviation == team_name
                        )
                    )
                )
            )
            team = team_result.scalars().first()
            
            if not team:
                continue
            
            # Save each stat type
            for stat_type in stat_types:
                value = stat_row.get(stat_type)
                if value is None:
                    continue
                
                try:
                    if pd.isna(value):
                        continue
                    
                    # Make stat_type unique by season
                    full_stat_type = f"{stat_type}_{season}"
                    
                    # Check if exists
                    existing = await session.execute(
                        select(TeamStats).where(
                            and_(
                                TeamStats.team_id == team.id,
                                TeamStats.stat_type == full_stat_type,
                            )
                        )
                    )
                    team_stat = existing.scalars().first()
                    
                    if team_stat:
                        team_stat.value = float(value)
                        team_stat.computed_at = datetime.utcnow()
                    else:
                        team_stat = TeamStats(
                            team_id=team.id,
                            stat_type=full_stat_type,
                            value=float(value),
                        )
                        session.add(team_stat)
                        saved_count += 1
                        
                except (ValueError, TypeError):
                    pass
        
        logger.info(f"[baseballR] Saved {saved_count} team stats")
        return saved_count
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not isinstance(data, dict):
            return False
        return True


# =============================================================================
# SINGLETON & REGISTRATION
# =============================================================================

baseballr_collector = BaseballRCollector()

try:
    collector_manager.register("baseballr", baseballr_collector)
except:
    pass  # Already registered