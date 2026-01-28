"""
ROYALEY - CFL Official API Data Collector
Phase 1: Data Collection Services

Collects comprehensive CFL (Canadian Football League) data from the official CFL API.
Features: Games, rosters, player stats, team stats, standings, play-by-play.

Data Sources:
- CFL Official API: https://api.cfl.ca/

API Key Required - Request from tech@cfl.ca

Key Data Types:
- Teams: All 9 CFL teams with divisions
- Games: Full game schedules with results (2004-present)
- Rosters: Player rosters by season
- Player Stats: Passing, rushing, receiving, defense, special teams
- Team Stats: Wins, losses, standings data
- Play-by-play: Game events (optional)

Tables Filled:
- sports (CFL entry)
- teams (9 CFL teams)
- games (10+ years of games)
- players (all CFL players)
- player_stats (seasonal stats)
- team_stats (seasonal stats)
- venues (CFL stadiums)
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus, Player, PlayerStats, TeamStats, Venue
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CFL API CONFIGURATION
# =============================================================================

CFL_API_BASE = "https://api.cfl.ca/v1"

# CFL API Endpoints
CFL_ENDPOINTS = {
    "games": f"{CFL_API_BASE}/games",
    "standings": f"{CFL_API_BASE}/standings",
    "players": f"{CFL_API_BASE}/players",
    "teams": f"{CFL_API_BASE}/teams",
    "leaders": f"{CFL_API_BASE}/leaders",
}

# =============================================================================
# CFL TEAMS (9 Teams - 2 Divisions)
# =============================================================================

CFL_TEAMS = {
    # West Division
    "BC": {
        "id": "BC",
        "name": "BC Lions",
        "city": "Vancouver",
        "division": "West",
        "venue": "BC Place",
        "venue_city": "Vancouver",
        "abbreviation": "BC",
    },
    "CGY": {
        "id": "CGY",
        "name": "Calgary Stampeders",
        "city": "Calgary",
        "division": "West",
        "venue": "McMahon Stadium",
        "venue_city": "Calgary",
        "abbreviation": "CGY",
    },
    "EDM": {
        "id": "EDM",
        "name": "Edmonton Elks",
        "city": "Edmonton",
        "division": "West",
        "venue": "Commonwealth Stadium",
        "venue_city": "Edmonton",
        "abbreviation": "EDM",
    },
    "SSK": {
        "id": "SSK",
        "name": "Saskatchewan Roughriders",
        "city": "Regina",
        "division": "West",
        "venue": "Mosaic Stadium",
        "venue_city": "Regina",
        "abbreviation": "SSK",
    },
    "WPG": {
        "id": "WPG",
        "name": "Winnipeg Blue Bombers",
        "city": "Winnipeg",
        "division": "West",
        "venue": "IG Field",
        "venue_city": "Winnipeg",
        "abbreviation": "WPG",
    },
    
    # East Division
    "HAM": {
        "id": "HAM",
        "name": "Hamilton Tiger-Cats",
        "city": "Hamilton",
        "division": "East",
        "venue": "Tim Hortons Field",
        "venue_city": "Hamilton",
        "abbreviation": "HAM",
    },
    "MTL": {
        "id": "MTL",
        "name": "Montreal Alouettes",
        "city": "Montreal",
        "division": "East",
        "venue": "Percival Molson Memorial Stadium",
        "venue_city": "Montreal",
        "abbreviation": "MTL",
    },
    "OTT": {
        "id": "OTT",
        "name": "Ottawa Redblacks",
        "city": "Ottawa",
        "division": "East",
        "venue": "TD Place Stadium",
        "venue_city": "Ottawa",
        "abbreviation": "OTT",
    },
    "TOR": {
        "id": "TOR",
        "name": "Toronto Argonauts",
        "city": "Toronto",
        "division": "East",
        "venue": "BMO Field",
        "venue_city": "Toronto",
        "abbreviation": "TOR",
    },
}

# Abbreviation mapping for API responses
TEAM_ABBR_MAP = {
    "BC": "BC",
    "CGY": "CGY", 
    "CAL": "CGY",
    "EDM": "EDM",
    "SSK": "SSK",
    "SAK": "SSK",
    "WPG": "WPG",
    "WIN": "WPG",
    "HAM": "HAM",
    "MTL": "MTL",
    "MON": "MTL",
    "OTT": "OTT",
    "TOR": "TOR",
}

# CFL Player stat categories
CFL_STAT_CATEGORIES = [
    "passing",
    "rushing", 
    "receiving",
    "defence",
    "field_goals",
    "punts",
    "punt_returns",
    "kick_returns",
    "converts",
]


# =============================================================================
# CFL COLLECTOR CLASS
# =============================================================================

class CFLCollector(BaseCollector):
    """
    Collector for CFL data using the official CFL API.
    
    Features:
    - Game schedules and results (2004-present)
    - Team information and standings
    - Player rosters and statistics
    - Team statistics
    - Play-by-play data (optional)
    
    Requires API key from tech@cfl.ca
    """
    
    def __init__(self):
        super().__init__(
            name="cfl",
            base_url=CFL_API_BASE,
            rate_limit=60,
            rate_window=60,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._api_key = getattr(settings, 'CFL_API_KEY', None) or ""
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with API key."""
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": "ROYALEY Sports Analytics/1.0",
                "Accept": "application/json",
            }
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers=headers,
            )
        return self._client
    
    def _get_auth_params(self) -> Dict[str, str]:
        """Get authentication parameters for API requests."""
        if self._api_key:
            return {"key": self._api_key}
        return {}
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        years: List[int] = None,
        collect_type: str = "all"
    ) -> CollectorResult:
        """
        Collect CFL data from official API.
        
        Args:
            years: List of seasons to collect (e.g., [2023, 2024])
            collect_type: Type of data to collect:
                - "all": Teams, games, rosters, player stats, team stats
                - "teams": Only teams
                - "games": Only games/schedules
                - "rosters": Only rosters
                - "player_stats": Only player statistics
                - "team_stats": Only team statistics
        
        Returns:
            CollectorResult with collected data
        """
        if years is None:
            current_year = datetime.now().year
            years = [current_year]
        
        logger.info(f"[CFL] Collecting CFL data for seasons: {years}")
        
        # Check API key
        if not self._api_key:
            logger.warning("[CFL] No API key configured. Set CFL_API_KEY in settings.")
            logger.warning("[CFL] Using fallback team data only.")
        
        data = {
            "teams": [],
            "games": [],
            "rosters": [],
            "player_stats": [],
            "team_stats": [],
            "venues": [],
        }
        total_records = 0
        
        try:
            # Collect teams (works without API for basic data)
            if collect_type in ["all", "teams"]:
                teams = await self._collect_teams()
                data["teams"] = teams
                total_records += len(teams)
                logger.info(f"[CFL] Collected {len(teams)} teams")
            
            # Collect venues
            if collect_type in ["all", "venues"]:
                venues = await self._collect_venues()
                data["venues"] = venues
                total_records += len(venues)
                logger.info(f"[CFL] Collected {len(venues)} venues")
            
            # Collect games (requires API key)
            if collect_type in ["all", "games"]:
                games = await self._collect_games(years)
                data["games"] = games
                total_records += len(games)
                logger.info(f"[CFL] Collected {len(games)} games")
            
            # Collect standings/team stats (requires API key)
            if collect_type in ["all", "team_stats"]:
                team_stats = await self._collect_team_stats(years)
                data["team_stats"] = team_stats
                total_records += len(team_stats)
                logger.info(f"[CFL] Collected {len(team_stats)} team stats")
            
            # Collect players/rosters (requires API key)
            if collect_type in ["all", "rosters"]:
                rosters = await self._collect_rosters(years)
                data["rosters"] = rosters
                total_records += len(rosters)
                logger.info(f"[CFL] Collected {len(rosters)} roster entries")
            
            # Collect player stats (requires API key)
            if collect_type in ["all", "player_stats"]:
                player_stats = await self._collect_player_stats(years)
                data["player_stats"] = player_stats
                total_records += len(player_stats)
                logger.info(f"[CFL] Collected {len(player_stats)} player stats")
            
            logger.info(f"[CFL] Total records collected: {total_records}")
            
            return CollectorResult(
                success=True,
                data=data,
                records_count=total_records,
            )
            
        except Exception as e:
            logger.error(f"[CFL] Collection error: {e}")
            import traceback
            traceback.print_exc()
            return CollectorResult(
                success=False,
                data=data,
                records_count=total_records,
                error=str(e)
            )
        finally:
            await self.close()

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self) -> List[Dict[str, Any]]:
        """Collect all CFL teams."""
        teams = []
        
        # Use predefined team data (always available)
        for team_id, team_info in CFL_TEAMS.items():
            teams.append({
                "external_id": f"cfl_{team_id}",
                "cfl_id": team_id,
                "name": team_info["name"],
                "abbreviation": team_info["abbreviation"],
                "city": team_info["city"],
                "division": team_info["division"],
                "conference": team_info["division"],  # CFL uses divisions
                "venue_name": team_info["venue"],
                "venue_city": team_info["venue_city"],
                "league": "CFL",
                "is_active": True,
            })
        
        # Try to get additional data from API if key available
        if self._api_key:
            try:
                client = await self.get_client()
                params = self._get_auth_params()
                
                response = await client.get(CFL_ENDPOINTS["teams"], params=params, timeout=30.0)
                
                if response.status_code == 200:
                    api_data = response.json()
                    api_teams = api_data.get("data", [])
                    
                    # Merge API data with predefined data
                    for api_team in api_teams:
                        abbr = api_team.get("abbreviation", "")
                        normalized_abbr = TEAM_ABBR_MAP.get(abbr, abbr)
                        
                        # Find matching team
                        for team in teams:
                            if team["abbreviation"] == normalized_abbr:
                                # Update with API data
                                if api_team.get("team_id"):
                                    team["cfl_api_id"] = api_team["team_id"]
                                if api_team.get("full_name"):
                                    team["full_name"] = api_team["full_name"]
                                if api_team.get("venue"):
                                    venue = api_team["venue"]
                                    if isinstance(venue, dict):
                                        team["venue_name"] = venue.get("name", team["venue_name"])
                                break
                    
                    logger.info(f"[CFL] Enhanced team data from API")
                else:
                    logger.warning(f"[CFL] Teams API returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"[CFL] Could not fetch teams from API: {e}")
        
        logger.info(f"[CFL] Loaded {len(teams)} CFL teams")
        return teams

    # =========================================================================
    # VENUES COLLECTION
    # =========================================================================
    
    async def _collect_venues(self) -> List[Dict[str, Any]]:
        """Collect CFL venue information."""
        venues = []
        seen_venues = set()
        
        for team_id, team_info in CFL_TEAMS.items():
            venue_name = team_info["venue"]
            if venue_name in seen_venues:
                continue
            seen_venues.add(venue_name)
            
            venues.append({
                "name": venue_name,
                "city": team_info["venue_city"],
                "state": "",  # Canadian provinces
                "country": "Canada",
                "is_dome": venue_name == "BC Place",  # BC Place is a dome
                "surface": "turf",
                "team_abbr": team_info["abbreviation"],
            })
        
        return venues

    # =========================================================================
    # GAMES COLLECTION
    # =========================================================================
    
    async def _collect_games(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect games for specified seasons."""
        all_games = []
        
        if not self._api_key:
            logger.warning("[CFL] No API key - cannot collect games")
            return all_games
        
        try:
            client = await self.get_client()
            params = self._get_auth_params()
            
            for year in years:
                year_games = []
                
                try:
                    # CFL API: /games/{season}
                    url = f"{CFL_ENDPOINTS['games']}/{year}"
                    response = await client.get(url, params=params, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        games_data = data.get("data", [])
                        
                        for game in games_data:
                            game_id = game.get("game_id", "")
                            if not game_id:
                                continue
                            
                            # Parse teams
                            team1 = game.get("team_1", {})
                            team2 = game.get("team_2", {})
                            
                            # Determine home/away
                            # CFL API: is_home field or venue-based detection
                            if team1.get("is_home") or game.get("venue", {}).get("name", "") in str(CFL_TEAMS.get(team1.get("abbreviation", ""), {}).get("venue", "")):
                                home_team = team1
                                away_team = team2
                            else:
                                home_team = team2
                                away_team = team1
                            
                            # Normalize team abbreviations
                            home_abbr = TEAM_ABBR_MAP.get(home_team.get("abbreviation", ""), home_team.get("abbreviation", ""))
                            away_abbr = TEAM_ABBR_MAP.get(away_team.get("abbreviation", ""), away_team.get("abbreviation", ""))
                            
                            # Parse date
                            date_str = game.get("date_start", "")
                            try:
                                if date_str:
                                    # Handle various date formats
                                    if "T" in date_str:
                                        scheduled_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                                        if scheduled_at.tzinfo:
                                            scheduled_at = scheduled_at.replace(tzinfo=None)
                                    else:
                                        scheduled_at = datetime.strptime(date_str[:10], "%Y-%m-%d")
                                else:
                                    scheduled_at = datetime(year, 6, 1)  # Default to June
                            except:
                                scheduled_at = datetime(year, 6, 1)
                            
                            # Determine game status
                            event_status = game.get("event_status", {})
                            status_id = event_status.get("event_status_id", 1)
                            
                            if status_id == 4 or event_status.get("is_active") == False:
                                game_status = "final"
                            elif status_id == 2 or event_status.get("is_active") == True:
                                game_status = "in_progress"
                            elif status_id == 5:
                                game_status = "postponed"
                            elif status_id == 6:
                                game_status = "cancelled"
                            else:
                                game_status = "scheduled"
                            
                            # Get scores
                            home_score = home_team.get("score")
                            away_score = away_team.get("score")
                            
                            # Get venue
                            venue = game.get("venue", {})
                            venue_name = venue.get("name", "") if isinstance(venue, dict) else ""
                            
                            game_record = {
                                "external_id": f"cfl_{game_id}",
                                "cfl_id": game_id,
                                "league": "CFL",
                                "season": year,
                                "week": game.get("week", 0),
                                "game_type": game.get("game_type", {}).get("name", "Regular Season") if isinstance(game.get("game_type"), dict) else "Regular Season",
                                "scheduled_at": scheduled_at,
                                "status": game_status,
                                "home_team_abbr": home_abbr,
                                "home_team_name": home_team.get("team_name", ""),
                                "away_team_abbr": away_abbr,
                                "away_team_name": away_team.get("team_name", ""),
                                "home_score": int(home_score) if home_score is not None else None,
                                "away_score": int(away_score) if away_score is not None else None,
                                "venue_name": venue_name,
                                "attendance": game.get("attendance", 0),
                                "weather": game.get("weather", {}),
                            }
                            
                            year_games.append(game_record)
                        
                        all_games.extend(year_games)
                        logger.info(f"[CFL] {year}: {len(year_games)} games")
                        
                    elif response.status_code == 401:
                        logger.error("[CFL] API key invalid or expired")
                        break
                    else:
                        logger.warning(f"[CFL] Games API returned {response.status_code} for {year}")
                    
                    await asyncio.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"[CFL] Error collecting games for {year}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[CFL] Error collecting games: {e}")
        
        logger.info(f"[CFL] Total {len(all_games)} games collected")
        return all_games

    # =========================================================================
    # TEAM STATS / STANDINGS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team standings/stats for specified seasons."""
        all_stats = []
        
        if not self._api_key:
            logger.warning("[CFL] No API key - cannot collect standings")
            return all_stats
        
        try:
            client = await self.get_client()
            params = self._get_auth_params()
            
            for year in years:
                try:
                    # CFL API: /standings/{season}
                    url = f"{CFL_ENDPOINTS['standings']}/{year}"
                    response = await client.get(url, params=params, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        standings_data = data.get("data", [])
                        
                        # Standings may be organized by division
                        for division in standings_data:
                            division_name = division.get("division", {}).get("name", "Unknown") if isinstance(division.get("division"), dict) else "Unknown"
                            teams_standing = division.get("teams", division.get("standings", []))
                            
                            for team_standing in teams_standing:
                                team_abbr = team_standing.get("abbreviation", "")
                                normalized_abbr = TEAM_ABBR_MAP.get(team_abbr, team_abbr)
                                
                                stat_record = {
                                    "league": "CFL",
                                    "season": year,
                                    "team_abbr": normalized_abbr,
                                    "team_name": team_standing.get("team_name", team_standing.get("name", "")),
                                    "division": division_name,
                                    "wins": team_standing.get("wins", 0),
                                    "losses": team_standing.get("losses", 0),
                                    "ties": team_standing.get("ties", 0),
                                    "points_for": team_standing.get("points_for", 0),
                                    "points_against": team_standing.get("points_against", 0),
                                    "home_wins": team_standing.get("home_wins", 0),
                                    "home_losses": team_standing.get("home_losses", 0),
                                    "away_wins": team_standing.get("away_wins", 0),
                                    "away_losses": team_standing.get("away_losses", 0),
                                    "streak": team_standing.get("streak", ""),
                                    "division_rank": team_standing.get("division_rank", 0),
                                }
                                all_stats.append(stat_record)
                        
                        logger.info(f"[CFL] {year}: {len([s for s in all_stats if s['season'] == year])} team stats")
                        
                    elif response.status_code == 401:
                        logger.error("[CFL] API key invalid")
                        break
                    else:
                        logger.warning(f"[CFL] Standings API returned {response.status_code} for {year}")
                    
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"[CFL] Error collecting standings for {year}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[CFL] Error collecting team stats: {e}")
        
        logger.info(f"[CFL] Total {len(all_stats)} team stats")
        return all_stats

    # =========================================================================
    # ROSTERS COLLECTION
    # =========================================================================
    
    async def _collect_rosters(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player rosters for all teams."""
        all_rosters = []
        seen_players = set()
        
        if not self._api_key:
            logger.warning("[CFL] No API key - cannot collect rosters")
            return all_rosters
        
        try:
            client = await self.get_client()
            params = self._get_auth_params()
            
            for year in years:
                year_count = 0
                
                # CFL API may have team-specific roster endpoints
                # Try /players with season filter
                try:
                    params_with_season = {**params, "season": year}
                    response = await client.get(CFL_ENDPOINTS["players"], params=params_with_season, timeout=60.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        players_data = data.get("data", [])
                        
                        for player in players_data:
                            player_id = player.get("cfl_central_id", player.get("player_id", ""))
                            if not player_id:
                                continue
                            
                            unique_key = f"{player_id}_{year}"
                            if unique_key in seen_players:
                                continue
                            seen_players.add(unique_key)
                            
                            team = player.get("team", {})
                            team_abbr = team.get("abbreviation", "") if isinstance(team, dict) else ""
                            normalized_abbr = TEAM_ABBR_MAP.get(team_abbr, team_abbr)
                            
                            # Parse birth date
                            birth_date = player.get("birth_date", "")
                            
                            roster_entry = {
                                "external_id": f"cfl_{player_id}",
                                "cfl_id": player_id,
                                "league": "CFL",
                                "season": year,
                                "team_abbr": normalized_abbr,
                                "team_name": team.get("name", team.get("team_name", "")) if isinstance(team, dict) else "",
                                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                                "first_name": player.get("first_name", ""),
                                "last_name": player.get("last_name", ""),
                                "position": player.get("position", {}).get("abbreviation", "") if isinstance(player.get("position"), dict) else player.get("position", ""),
                                "jersey_number": player.get("uniform_number", player.get("jersey", "")),
                                "height": player.get("height", ""),
                                "weight": player.get("weight", ""),
                                "birth_date": birth_date,
                                "birth_place": player.get("birth_city", ""),
                                "college": player.get("school", {}).get("name", "") if isinstance(player.get("school"), dict) else player.get("college", ""),
                                "nationality": player.get("nationality", ""),
                                "is_canadian": player.get("is_national", False),
                                "rookie_year": player.get("rookie_year", ""),
                            }
                            
                            all_rosters.append(roster_entry)
                            year_count += 1
                        
                        logger.info(f"[CFL] {year}: {year_count} roster entries")
                        
                    elif response.status_code == 401:
                        logger.error("[CFL] API key invalid")
                        break
                    else:
                        logger.warning(f"[CFL] Players API returned {response.status_code} for {year}")
                    
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.warning(f"[CFL] Error collecting rosters for {year}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[CFL] Error collecting rosters: {e}")
        
        logger.info(f"[CFL] Total {len(all_rosters)} roster entries")
        return all_rosters

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics."""
        all_stats = []
        
        if not self._api_key:
            logger.warning("[CFL] No API key - cannot collect player stats")
            return all_stats
        
        try:
            client = await self.get_client()
            params = self._get_auth_params()
            
            for year in years:
                year_stats = []
                
                # Try to get league leaders which include stats
                for category in CFL_STAT_CATEGORIES:
                    try:
                        url = f"{CFL_ENDPOINTS['leaders']}/{year}/category/{category}"
                        response = await client.get(url, params=params, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            leaders_data = data.get("data", [])
                            
                            for leader in leaders_data:
                                player = leader.get("player", {})
                                player_id = player.get("cfl_central_id", player.get("player_id", ""))
                                
                                if not player_id:
                                    continue
                                
                                stat_record = {
                                    "external_id": f"cfl_{player_id}_{year}_{category}",
                                    "player_id": f"cfl_{player_id}",
                                    "league": "CFL",
                                    "season": year,
                                    "stat_category": category,
                                    "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                                    "team_abbr": TEAM_ABBR_MAP.get(player.get("team", {}).get("abbreviation", ""), ""),
                                    "games_played": leader.get("games_played", 0),
                                    "stat_value": leader.get("value", leader.get("yards", leader.get("touchdowns", 0))),
                                    "rank": leader.get("rank", 0),
                                }
                                
                                # Add category-specific stats
                                if category == "passing":
                                    stat_record.update({
                                        "pass_attempts": leader.get("pass_attempts", 0),
                                        "pass_completions": leader.get("pass_completions", 0),
                                        "pass_yards": leader.get("pass_yards", 0),
                                        "pass_touchdowns": leader.get("pass_touchdowns", 0),
                                        "interceptions": leader.get("pass_interceptions", 0),
                                        "passer_rating": leader.get("passer_rating", 0),
                                    })
                                elif category == "rushing":
                                    stat_record.update({
                                        "rush_attempts": leader.get("rush_attempts", 0),
                                        "rush_yards": leader.get("rush_yards", 0),
                                        "rush_touchdowns": leader.get("rush_touchdowns", 0),
                                        "rush_avg": leader.get("rush_average", 0),
                                    })
                                elif category == "receiving":
                                    stat_record.update({
                                        "receptions": leader.get("receptions", 0),
                                        "receiving_yards": leader.get("receiving_yards", 0),
                                        "receiving_touchdowns": leader.get("receiving_touchdowns", 0),
                                        "targets": leader.get("targets", 0),
                                    })
                                elif category == "defence":
                                    stat_record.update({
                                        "tackles": leader.get("tackles", 0),
                                        "sacks": leader.get("sacks", 0),
                                        "interceptions_def": leader.get("interceptions", 0),
                                        "forced_fumbles": leader.get("forced_fumbles", 0),
                                    })
                                
                                year_stats.append(stat_record)
                        
                        await asyncio.sleep(0.15)
                        
                    except Exception as e:
                        logger.debug(f"[CFL] Error fetching {category} stats for {year}: {e}")
                        continue
                
                all_stats.extend(year_stats)
                logger.info(f"[CFL] {year}: {len(year_stats)} player stats")
            
        except Exception as e:
            logger.error(f"[CFL] Error collecting player stats: {e}")
        
        logger.info(f"[CFL] Total {len(all_stats)} player stats")
        return all_stats

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Ensure sport exists
            sport = await self._ensure_sport(session)
            if sport:
                logger.info(f"[CFL] Sport 'CFL' ready (ID: {sport.id})")
            
            # Save venues
            if data.get("venues"):
                saved = await self._save_venues(session, data["venues"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} venues")
            
            # Save teams
            if data.get("teams"):
                saved = await self._save_teams(session, data["teams"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} teams")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(session, data["games"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} games")
            
            # Save rosters/players
            if data.get("rosters"):
                saved = await self._save_rosters(session, data["rosters"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} players")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self._save_team_stats(session, data["team_stats"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} team stats")
            
            # Save player stats
            if data.get("player_stats"):
                saved = await self._save_player_stats(session, data["player_stats"])
                total_saved += saved
                logger.info(f"[CFL] Saved {saved} player stats")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[CFL] Error saving to database: {e}")
            import traceback
            traceback.print_exc()
            await session.rollback()
            raise
        
        return total_saved
    
    async def _ensure_sport(self, session: AsyncSession) -> Optional[Sport]:
        """Ensure CFL sport exists in database."""
        try:
            result = await session.execute(
                select(Sport).where(Sport.code == "CFL")
            )
            sport = result.scalar_one_or_none()
            
            if not sport:
                sport = Sport(
                    code="CFL",
                    name="Canadian Football League",
                    is_active=True,
                    config={"collector": "CFL", "source": "CFL Official API"}
                )
                session.add(sport)
                await session.flush()
                logger.info("[CFL] Created sport: CFL")
            
            return sport
            
        except Exception as e:
            logger.error(f"[CFL] Error ensuring sport: {e}")
            return None
    
    async def _save_venues(self, session: AsyncSession, venues: List[Dict[str, Any]]) -> int:
        """Save venue data to database."""
        saved = 0
        
        for venue_data in venues:
            try:
                venue_name = venue_data.get("name", "")
                if not venue_name:
                    continue
                
                # Check if venue exists
                result = await session.execute(
                    select(Venue).where(Venue.name == venue_name)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.city = venue_data.get("city", existing.city)
                    existing.country = venue_data.get("country", existing.country)
                    existing.is_dome = venue_data.get("is_dome", existing.is_dome)
                    existing.surface = venue_data.get("surface", existing.surface)
                else:
                    # Create new
                    venue = Venue(
                        name=venue_name,
                        city=venue_data.get("city", ""),
                        state=venue_data.get("state", ""),
                        country=venue_data.get("country", "Canada"),
                        is_dome=venue_data.get("is_dome", False),
                        surface=venue_data.get("surface", "turf"),
                    )
                    session.add(venue)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving venue: {e}")
                continue
        
        await session.flush()
        return saved
    
    async def _save_teams(self, session: AsyncSession, teams: List[Dict[str, Any]]) -> int:
        """Save teams to database with proper duplicate handling."""
        saved = 0
        updated = 0
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == "CFL")
        )
        sport = result.scalar_one_or_none()
        if not sport:
            return 0
        
        for team_data in teams:
            try:
                external_id = team_data.get("external_id", "")
                team_name = team_data.get("name", "")
                
                # Check by external_id first
                result = await session.execute(
                    select(Team).where(Team.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Also check by (sport_id, name)
                if not existing and team_name:
                    result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == team_name
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.external_id = external_id
                    existing.abbreviation = team_data.get("abbreviation", existing.abbreviation)
                    existing.city = team_data.get("city", existing.city)
                    existing.conference = team_data.get("conference", existing.conference)
                    existing.division = team_data.get("division", existing.division)
                    updated += 1
                else:
                    # Create new
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_name,
                        abbreviation=team_data.get("abbreviation", ""),
                        city=team_data.get("city", ""),
                        conference=team_data.get("conference", ""),
                        division=team_data.get("division", ""),
                        is_active=team_data.get("is_active", True),
                    )
                    session.add(team)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving team: {e}")
                continue
        
        await session.flush()
        logger.info(f"[CFL] Teams: {saved} new, {updated} updated")
        return saved + updated
    
    async def _save_games(self, session: AsyncSession, games: List[Dict[str, Any]]) -> int:
        """Save games to database with proper duplicate handling."""
        saved = 0
        updated = 0
        skipped = 0
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == "CFL")
        )
        sport = result.scalar_one_or_none()
        if not sport:
            return 0
        
        for game_data in games:
            try:
                external_id = game_data.get("external_id", "")
                
                # Find home team
                home_abbr = game_data.get("home_team_abbr", "")
                home_external_id = f"cfl_{home_abbr}"
                
                result = await session.execute(
                    select(Team).where(Team.external_id == home_external_id)
                )
                home_team = result.scalar_one_or_none()
                
                # Fallback by name
                if not home_team and game_data.get("home_team_name"):
                    result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == game_data.get("home_team_name")
                            )
                        )
                    )
                    home_team = result.scalar_one_or_none()
                
                # Find away team
                away_abbr = game_data.get("away_team_abbr", "")
                away_external_id = f"cfl_{away_abbr}"
                
                result = await session.execute(
                    select(Team).where(Team.external_id == away_external_id)
                )
                away_team = result.scalar_one_or_none()
                
                # Fallback by name
                if not away_team and game_data.get("away_team_name"):
                    result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == game_data.get("away_team_name")
                            )
                        )
                    )
                    away_team = result.scalar_one_or_none()
                
                if not home_team or not away_team:
                    skipped += 1
                    continue
                
                # Check if game exists
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Map status
                status_map = {
                    "final": GameStatus.FINAL,
                    "in_progress": GameStatus.IN_PROGRESS,
                    "scheduled": GameStatus.SCHEDULED,
                    "postponed": GameStatus.POSTPONED,
                    "cancelled": GameStatus.CANCELLED,
                }
                game_status = status_map.get(game_data.get("status", "scheduled"), GameStatus.SCHEDULED)
                
                if existing:
                    # Update scores
                    existing.home_score = game_data.get("home_score")
                    existing.away_score = game_data.get("away_score")
                    existing.status = game_status
                    updated += 1
                else:
                    # Create new game
                    game = Game(
                        sport_id=sport.id,
                        external_id=external_id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=game_data.get("scheduled_at", datetime.now()),
                        status=game_status,
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                    )
                    session.add(game)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving game: {e}")
                continue
        
        await session.flush()
        logger.info(f"[CFL] Games: {saved} new, {updated} updated, {skipped} skipped")
        return saved + updated
    
    async def _save_rosters(self, session: AsyncSession, rosters: List[Dict[str, Any]]) -> int:
        """Save roster/player data to database."""
        saved = 0
        updated = 0
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == "CFL")
        )
        sport = result.scalar_one_or_none()
        if not sport:
            return 0
        
        for roster_data in rosters:
            try:
                external_id = roster_data.get("external_id", "")
                
                # Check if player exists
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Find team
                team_abbr = roster_data.get("team_abbr", "")
                team_external_id = f"cfl_{team_abbr}"
                
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_external_id)
                )
                team = team_result.scalar_one_or_none()
                
                if existing:
                    # Update player info
                    existing.name = roster_data.get("player_name", existing.name)
                    existing.position = roster_data.get("position", existing.position)
                    if team:
                        existing.team_id = team.id
                    existing.height = roster_data.get("height", existing.height)
                    try:
                        jersey = roster_data.get("jersey_number")
                        if jersey:
                            existing.jersey_number = int(jersey)
                    except:
                        pass
                    try:
                        weight = roster_data.get("weight")
                        if weight:
                            existing.weight = int(weight)
                    except:
                        pass
                    updated += 1
                else:
                    # Create new player
                    player = Player(
                        external_id=external_id,
                        name=roster_data.get("player_name", ""),
                        position=roster_data.get("position", ""),
                        team_id=team.id if team else None,
                        height=roster_data.get("height", ""),
                        is_active=True,
                    )
                    
                    try:
                        jersey = roster_data.get("jersey_number")
                        if jersey:
                            player.jersey_number = int(jersey)
                    except:
                        pass
                    
                    try:
                        weight = roster_data.get("weight")
                        if weight:
                            player.weight = int(weight)
                    except:
                        pass
                    
                    session.add(player)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving player: {e}")
                continue
        
        await session.flush()
        logger.info(f"[CFL] Players: {saved} new, {updated} updated")
        return saved + updated
    
    async def _save_team_stats(self, session: AsyncSession, team_stats: List[Dict[str, Any]]) -> int:
        """Save team statistics to database."""
        saved = 0
        
        for stat_data in team_stats:
            try:
                team_abbr = stat_data.get("team_abbr", "")
                team_external_id = f"cfl_{team_abbr}"
                
                # Find team
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_external_id)
                )
                team = team_result.scalar_one_or_none()
                
                if not team:
                    continue
                
                season = stat_data.get("season", datetime.now().year)
                stat_type = f"CFL_{season}_standings"
                
                # Check existing
                result = await session.execute(
                    select(TeamStats).where(
                        and_(
                            TeamStats.team_id == team.id,
                            TeamStats.stat_type == stat_type
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                wins = stat_data.get("wins", 0)
                losses = stat_data.get("losses", 0)
                
                if existing:
                    existing.value = float(wins)
                    existing.games_played = wins + losses + stat_data.get("ties", 0)
                else:
                    team_stat = TeamStats(
                        team_id=team.id,
                        stat_type=stat_type,
                        value=float(wins),
                        games_played=wins + losses + stat_data.get("ties", 0),
                    )
                    session.add(team_stat)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving team stat: {e}")
                continue
        
        await session.flush()
        return saved
    
    async def _save_player_stats(self, session: AsyncSession, player_stats: List[Dict[str, Any]]) -> int:
        """Save player statistics to database."""
        saved = 0
        
        for stat_data in player_stats:
            try:
                player_external_id = stat_data.get("player_id", "")
                
                # Find player
                player_result = await session.execute(
                    select(Player).where(Player.external_id == player_external_id)
                )
                player = player_result.scalar_one_or_none()
                
                if not player:
                    continue
                
                season = stat_data.get("season", datetime.now().year)
                category = stat_data.get("stat_category", "general")
                stat_type = f"CFL_{season}_{category}"
                
                # Check existing
                result = await session.execute(
                    select(PlayerStats).where(
                        and_(
                            PlayerStats.player_id == player.id,
                            PlayerStats.stat_type == stat_type
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                stat_value = stat_data.get("stat_value", 0)
                
                if existing:
                    existing.value = float(stat_value)
                else:
                    player_stat = PlayerStats(
                        player_id=player.id,
                        stat_type=stat_type,
                        value=float(stat_value),
                    )
                    session.add(player_stat)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFL] Error saving player stat: {e}")
                continue
        
        await session.flush()
        return saved

    # =========================================================================
    # VALIDATION METHOD (Required by BaseCollector)
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """Validate collected CFL data."""
        if data is None:
            return False
        if not isinstance(data, dict):
            return False
        
        # Check that we have at least some data
        has_teams = len(data.get("teams", [])) > 0
        has_games = len(data.get("games", [])) > 0
        
        return has_teams or has_games


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

cfl_collector = CFLCollector()

# Register with collector manager
collector_manager.register(cfl_collector)
logger.info("Registered collector: CFL")