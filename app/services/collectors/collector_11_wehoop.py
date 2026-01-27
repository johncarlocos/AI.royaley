"""
ROYALEY - wehoop Data Collector
Phase 1: Data Collection Services

Collects comprehensive WNBA data from ESPN and WNBA Stats APIs.
Features: Play-by-play, box scores, standings, player stats, team stats.

Data Sources:
- ESPN API: https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/
- WNBA Stats API: https://stats.wnba.com/stats/

FREE data - no API key required!

Key Data Types:
- Teams: All 12 WNBA teams with conferences
- Schedules: Full game schedules with results (2016-present)
- Rosters: Player rosters by season
- Player Stats: Comprehensive basketball statistics
- Team Stats: Team season statistics

Tables Filled:
- sports (WNBA entry)
- teams (12 WNBA teams)
- games (10 years of games)
- players (all WNBA players)
- player_stats (seasonal stats)
- team_stats (seasonal stats)
- venues (WNBA arenas)
"""

import asyncio
import logging
from datetime import datetime, timedelta
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
# WNBA API CONFIGURATION
# =============================================================================

# ESPN API endpoints
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_TEAMS = f"{ESPN_BASE}/teams"
ESPN_STANDINGS = f"{ESPN_BASE}/standings"

# WNBA Stats API endpoints (similar to NBA Stats API)
WNBA_STATS_BASE = "https://stats.wnba.com/stats"

# Headers for WNBA Stats API (requires specific headers)
WNBA_STATS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Host": "stats.wnba.com",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# WNBA Teams with IDs and info (ESPN Team IDs)
WNBA_TEAMS = {
    # Eastern Conference
    3: {"abbr": "ATL", "name": "Atlanta Dream", "city": "Atlanta", "state": "GA", "conference": "Eastern"},
    4: {"abbr": "CHI", "name": "Chicago Sky", "city": "Chicago", "state": "IL", "conference": "Eastern"},
    5: {"abbr": "CONN", "name": "Connecticut Sun", "city": "Uncasville", "state": "CT", "conference": "Eastern"},
    6: {"abbr": "IND", "name": "Indiana Fever", "city": "Indianapolis", "state": "IN", "conference": "Eastern"},
    9: {"abbr": "NY", "name": "New York Liberty", "city": "Brooklyn", "state": "NY", "conference": "Eastern"},
    14: {"abbr": "WSH", "name": "Washington Mystics", "city": "Washington", "state": "DC", "conference": "Eastern"},
    
    # Western Conference
    16: {"abbr": "DAL", "name": "Dallas Wings", "city": "Arlington", "state": "TX", "conference": "Western"},
    8: {"abbr": "LV", "name": "Las Vegas Aces", "city": "Las Vegas", "state": "NV", "conference": "Western"},
    17: {"abbr": "LA", "name": "Los Angeles Sparks", "city": "Los Angeles", "state": "CA", "conference": "Western"},
    11: {"abbr": "MIN", "name": "Minnesota Lynx", "city": "Minneapolis", "state": "MN", "conference": "Western"},
    18: {"abbr": "PHX", "name": "Phoenix Mercury", "city": "Phoenix", "state": "AZ", "conference": "Western"},
    19: {"abbr": "SEA", "name": "Seattle Storm", "city": "Seattle", "state": "WA", "conference": "Western"},
    
    # Expansion teams - Golden State Valkyries (2025)
    20: {"abbr": "GS", "name": "Golden State Valkyries", "city": "San Francisco", "state": "CA", "conference": "Western"},
}

# Alternative abbreviation mapping
ABBR_TO_ID = {info["abbr"]: team_id for team_id, info in WNBA_TEAMS.items()}

# WNBA Venues
WNBA_VENUES = {
    "ATL": {"name": "Gateway Center Arena", "city": "College Park", "state": "GA", "capacity": 3500},
    "CHI": {"name": "Wintrust Arena", "city": "Chicago", "state": "IL", "capacity": 10387},
    "CONN": {"name": "Mohegan Sun Arena", "city": "Uncasville", "state": "CT", "capacity": 9323},
    "IND": {"name": "Gainbridge Fieldhouse", "city": "Indianapolis", "state": "IN", "capacity": 17923},
    "NY": {"name": "Barclays Center", "city": "Brooklyn", "state": "NY", "capacity": 17732},
    "WSH": {"name": "Entertainment & Sports Arena", "city": "Washington", "state": "DC", "capacity": 4200},
    "DAL": {"name": "College Park Center", "city": "Arlington", "state": "TX", "capacity": 7000},
    "LV": {"name": "Michelob ULTRA Arena", "city": "Las Vegas", "state": "NV", "capacity": 12000},
    "LA": {"name": "Crypto.com Arena", "city": "Los Angeles", "state": "CA", "capacity": 18997},
    "MIN": {"name": "Target Center", "city": "Minneapolis", "state": "MN", "capacity": 18978},
    "PHX": {"name": "Footprint Center", "city": "Phoenix", "state": "AZ", "capacity": 17071},
    "SEA": {"name": "Climate Pledge Arena", "city": "Seattle", "state": "WA", "capacity": 18100},
    "GS": {"name": "Chase Center", "city": "San Francisco", "state": "CA", "capacity": 18064},
}


# =============================================================================
# WNBA PLAYER STATS COLUMNS
# =============================================================================

PLAYER_STATS_COLUMNS = [
    # Basic Stats
    "games_played", "minutes", "points", "rebounds", "assists",
    "steals", "blocks", "turnovers", "fouls",
    
    # Shooting
    "field_goals_made", "field_goals_attempted", "field_goal_pct",
    "three_pointers_made", "three_pointers_attempted", "three_point_pct",
    "free_throws_made", "free_throws_attempted", "free_throw_pct",
    
    # Rebounds
    "offensive_rebounds", "defensive_rebounds",
    
    # Advanced
    "plus_minus", "efficiency", "double_doubles", "triple_doubles",
]

TEAM_STATS_COLUMNS = [
    # Record
    "wins", "losses", "win_pct", "home_wins", "home_losses",
    "away_wins", "away_losses", "conference_wins", "conference_losses",
    
    # Scoring
    "points_per_game", "points_against_per_game", "point_differential",
    
    # Shooting
    "field_goal_pct", "three_point_pct", "free_throw_pct",
    
    # Other
    "rebounds_per_game", "assists_per_game", "turnovers_per_game",
    "steals_per_game", "blocks_per_game",
]


# =============================================================================
# WEHOOP COLLECTOR CLASS
# =============================================================================

class WehoopCollector(BaseCollector):
    """
    Collector for WNBA data using ESPN and WNBA Stats APIs.
    
    Features:
    - Game schedules and results (2016-present)
    - Team information and standings
    - Player statistics
    - Team statistics
    - Rosters
    - No API key required (FREE)
    """
    
    def __init__(self):
        super().__init__(
            name="wehoop",
            base_url=ESPN_BASE,
            rate_limit=60,  # requests per minute
            rate_window=60,
        )
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                }
            )
        return self._client
    
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
        Collect WNBA data.
        
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
        try:
            data = {}
            total_records = 0
            
            # Default to recent seasons
            if years is None:
                current_year = datetime.now().year
                # WNBA season runs May-October
                if datetime.now().month >= 5:
                    years = [current_year]
                else:
                    years = [current_year - 1]
            
            logger.info(f"[wehoop] Collecting WNBA data for seasons: {years}")
            
            # Collect teams
            if collect_type in ["all", "teams"]:
                teams = await self._collect_teams()
                data["teams"] = teams
                total_records += len(teams)
                logger.info(f"[wehoop] Collected {len(teams)} teams")
            
            # Collect games/schedules
            if collect_type in ["all", "games"]:
                games = await self._collect_games(years)
                data["games"] = games
                total_records += len(games)
                logger.info(f"[wehoop] Collected {len(games)} games")
            
            # Collect rosters
            if collect_type in ["all", "rosters"]:
                rosters = await self._collect_rosters(years)
                data["rosters"] = rosters
                total_records += len(rosters)
                logger.info(f"[wehoop] Collected {len(rosters)} roster entries")
            
            # Collect player stats
            if collect_type in ["all", "player_stats"]:
                player_stats = await self._collect_player_stats(years)
                data["player_stats"] = player_stats
                total_records += len(player_stats)
                logger.info(f"[wehoop] Collected {len(player_stats)} player stats")
            
            # Collect team stats
            if collect_type in ["all", "team_stats"]:
                team_stats = await self._collect_team_stats(years)
                data["team_stats"] = team_stats
                total_records += len(team_stats)
                logger.info(f"[wehoop] Collected {len(team_stats)} team stats")
            
            return CollectorResult(
                success=True,
                records_count=total_records,
                data=data,
            )
            
        except Exception as e:
            logger.error(f"[wehoop] Collection error: {e}")
            return CollectorResult(
                success=False,
                records_count=0,
                errors=[str(e)],
            )

    async def validate(self, data: Any) -> bool:
        """Validate collected WNBA data."""
        if data is None:
            return False
        
        if not isinstance(data, dict):
            return False
        
        valid_keys = ["teams", "games", "rosters", "player_stats", "team_stats"]
        has_data = any(
            key in data and data[key] and len(data[key]) > 0
            for key in valid_keys
        )
        
        return has_data

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self) -> List[Dict[str, Any]]:
        """Collect all WNBA teams from ESPN API."""
        teams = []
        
        try:
            client = await self.get_client()
            
            # Get teams from ESPN API
            url = f"{ESPN_TEAMS}?limit=50"
            response = await client.get(url, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                espn_teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
                
                for team_data in espn_teams:
                    team = team_data.get("team", {})
                    team_id = team.get("id")
                    abbr = team.get("abbreviation", "")
                    name = team.get("displayName", "")
                    
                    # Get additional info from our mapping
                    mapped_info = WNBA_TEAMS.get(int(team_id), {}) if team_id else {}
                    
                    teams.append({
                        "espn_id": team_id,
                        "abbr": abbr or mapped_info.get("abbr", ""),
                        "name": name or mapped_info.get("name", ""),
                        "city": team.get("location", mapped_info.get("city", "")),
                        "conference": mapped_info.get("conference", ""),
                        "logo_url": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                    })
                
                logger.info(f"[wehoop] Loaded {len(teams)} WNBA teams from ESPN")
            
            # Fallback to static mapping if ESPN fails
            if not teams:
                for team_id, info in WNBA_TEAMS.items():
                    teams.append({
                        "espn_id": str(team_id),
                        "abbr": info["abbr"],
                        "name": info["name"],
                        "city": info["city"],
                        "conference": info["conference"],
                        "logo_url": "",
                    })
                logger.info(f"[wehoop] Loaded {len(teams)} WNBA teams from static mapping")
        
        except Exception as e:
            logger.error(f"[wehoop] Error collecting teams: {e}")
            # Use static mapping as fallback
            for team_id, info in WNBA_TEAMS.items():
                teams.append({
                    "espn_id": str(team_id),
                    "abbr": info["abbr"],
                    "name": info["name"],
                    "city": info["city"],
                    "conference": info["conference"],
                    "logo_url": "",
                })
        
        return teams

    # =========================================================================
    # GAMES COLLECTION
    # =========================================================================
    
    async def _collect_games(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect games for specified seasons from ESPN API."""
        all_games = []
        
        try:
            client = await self.get_client()
            
            for year in years:
                season_games = []
                
                # WNBA season: May to October
                start_date = datetime(year, 5, 1)
                end_date = datetime(year, 10, 31)
                
                # Collect week by week
                current_date = start_date
                while current_date <= end_date:
                    try:
                        # Format date for ESPN API
                        date_str = current_date.strftime("%Y%m%d")
                        url = f"{ESPN_SCOREBOARD}?dates={date_str}"
                        
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            events = data.get("events", [])
                            
                            for event in events:
                                game = self._parse_espn_game(event, year)
                                if game:
                                    season_games.append(game)
                        
                        # Move to next week
                        current_date += timedelta(days=7)
                        await asyncio.sleep(0.1)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"[wehoop] Error collecting games for {current_date}: {e}")
                        current_date += timedelta(days=7)
                        continue
                
                # Also try scoreboard for the whole season (historical)
                try:
                    url = f"{ESPN_SCOREBOARD}?season={year}&seasontype=2&limit=500"
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get("events", [])
                        
                        for event in events:
                            game = self._parse_espn_game(event, year)
                            if game and game["game_id"] not in [g["game_id"] for g in season_games]:
                                season_games.append(game)
                except Exception as e:
                    logger.warning(f"[wehoop] Error collecting season {year} games: {e}")
                
                logger.info(f"[wehoop] {year}: {len(season_games)} games")
                all_games.extend(season_games)
            
            logger.info(f"[wehoop] Total {len(all_games)} games collected")
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting games: {e}")
        
        return all_games
    
    def _parse_espn_game(self, event: Dict, year: int) -> Optional[Dict[str, Any]]:
        """Parse ESPN event into game data."""
        try:
            game_id = event.get("id")
            competitions = event.get("competitions", [{}])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                return None
            
            # Find home and away teams
            home_team = None
            away_team = None
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team = comp
                else:
                    away_team = comp
            
            if not home_team or not away_team:
                return None
            
            # Parse date
            date_str = event.get("date", "")
            try:
                scheduled_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                # Convert to naive UTC
                if scheduled_at.tzinfo is not None:
                    scheduled_at = scheduled_at.replace(tzinfo=None)
            except:
                scheduled_at = datetime.utcnow()
            
            # Parse status
            status_data = event.get("status", {})
            status_type = status_data.get("type", {}).get("name", "STATUS_SCHEDULED")
            
            status_map = {
                "STATUS_FINAL": "final",
                "STATUS_IN_PROGRESS": "in_progress",
                "STATUS_SCHEDULED": "scheduled",
                "STATUS_POSTPONED": "postponed",
                "STATUS_CANCELED": "cancelled",
            }
            status = status_map.get(status_type, "scheduled")
            
            # Get scores
            home_score = int(home_team.get("score", 0) or 0) if status == "final" else None
            away_score = int(away_team.get("score", 0) or 0) if status == "final" else None
            
            # Get venue info
            venue_data = competition.get("venue", {})
            
            return {
                "game_id": game_id,
                "season": year,
                "scheduled_at": scheduled_at.isoformat(),
                "status": status,
                "home_team_id": home_team.get("team", {}).get("id"),
                "home_team_abbr": home_team.get("team", {}).get("abbreviation", ""),
                "home_team_name": home_team.get("team", {}).get("displayName", ""),
                "away_team_id": away_team.get("team", {}).get("id"),
                "away_team_abbr": away_team.get("team", {}).get("abbreviation", ""),
                "away_team_name": away_team.get("team", {}).get("displayName", ""),
                "home_score": home_score,
                "away_score": away_score,
                "venue_name": venue_data.get("fullName", ""),
                "venue_city": venue_data.get("address", {}).get("city", ""),
                "venue_state": venue_data.get("address", {}).get("state", ""),
            }
        
        except Exception as e:
            logger.warning(f"[wehoop] Error parsing game: {e}")
            return None

    # =========================================================================
    # ROSTERS COLLECTION
    # =========================================================================
    
    async def _collect_rosters(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect rosters for all teams across specified seasons."""
        all_rosters = []
        
        try:
            client = await self.get_client()
            
            for year in years:
                for team_id, team_info in WNBA_TEAMS.items():
                    try:
                        # ESPN roster endpoint
                        url = f"{ESPN_TEAMS}/{team_id}/roster?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            athletes = data.get("athletes", [])
                            
                            for category in athletes:
                                for player in category.get("items", []):
                                    roster_entry = {
                                        "season": year,
                                        "team_id": team_id,
                                        "team_abbr": team_info["abbr"],
                                        "player_id": player.get("id"),
                                        "player_name": player.get("displayName", ""),
                                        "position": player.get("position", {}).get("abbreviation", ""),
                                        "jersey_number": player.get("jersey", ""),
                                        "height": player.get("height", ""),
                                        "weight": player.get("weight", ""),
                                        "birth_date": player.get("dateOfBirth", ""),
                                        "experience": player.get("experience", {}).get("years", 0),
                                    }
                                    all_rosters.append(roster_entry)
                        
                        await asyncio.sleep(0.1)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"[wehoop] Error collecting roster for {team_info['abbr']} {year}: {e}")
                        continue
                
                logger.info(f"[wehoop] {year}: Collected rosters")
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting rosters: {e}")
        
        logger.info(f"[wehoop] Collected {len(all_rosters)} roster entries")
        return all_rosters

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics for specified seasons."""
        all_stats = []
        
        try:
            client = await self.get_client()
            
            for year in years:
                try:
                    # Try ESPN leaders/statistics endpoint
                    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/leaders?season={year}"
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        categories = data.get("leaders", [])
                        
                        # Collect unique players from leaders
                        players_seen = set()
                        
                        for category in categories:
                            cat_name = category.get("name", "")
                            leaders = category.get("leaders", [])
                            
                            for leader in leaders:
                                athlete = leader.get("athlete", {})
                                player_id = athlete.get("id")
                                
                                if player_id and player_id not in players_seen:
                                    players_seen.add(player_id)
                                    
                                    # Get detailed stats for this player
                                    player_stats = await self._get_player_season_stats(
                                        client, player_id, year
                                    )
                                    if player_stats:
                                        all_stats.append(player_stats)
                    
                    logger.info(f"[wehoop] {year}: Collected player stats")
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error collecting player stats for {year}: {e}")
                    continue
                
                await asyncio.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting player stats: {e}")
        
        logger.info(f"[wehoop] Collected {len(all_stats)} player stats records")
        return all_stats
    
    async def _get_player_season_stats(
        self, 
        client: httpx.AsyncClient, 
        player_id: str, 
        year: int
    ) -> Optional[Dict[str, Any]]:
        """Get detailed season stats for a player."""
        try:
            url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/wnba/athletes/{player_id}/stats?season={year}"
            response = await client.get(url, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get("splits", {}).get("categories", [])
                
                player_stats = {
                    "player_id": player_id,
                    "season": year,
                    "player_name": data.get("athlete", {}).get("displayName", ""),
                    "team_abbr": data.get("athlete", {}).get("team", {}).get("abbreviation", ""),
                }
                
                # Parse stats categories
                for category in stats:
                    cat_stats = category.get("stats", [])
                    for stat in cat_stats:
                        stat_name = stat.get("name", "").lower().replace(" ", "_")
                        stat_value = stat.get("value", 0)
                        player_stats[stat_name] = stat_value
                
                return player_stats
            
        except Exception as e:
            logger.debug(f"[wehoop] Could not get stats for player {player_id}: {e}")
        
        return None

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team statistics for specified seasons."""
        all_stats = []
        
        try:
            client = await self.get_client()
            
            for year in years:
                try:
                    # Get standings which include team stats
                    url = f"{ESPN_STANDINGS}?season={year}"
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        conferences = data.get("children", [])
                        
                        for conference in conferences:
                            conf_name = conference.get("name", "")
                            standings = conference.get("standings", {}).get("entries", [])
                            
                            for entry in standings:
                                team = entry.get("team", {})
                                team_id = team.get("id")
                                
                                # Parse stats
                                stats_list = entry.get("stats", [])
                                stats_dict = {}
                                for stat in stats_list:
                                    stat_name = stat.get("name", "").lower().replace(" ", "_")
                                    stat_value = stat.get("value", 0)
                                    stats_dict[stat_name] = stat_value
                                
                                team_stats = {
                                    "team_id": team_id,
                                    "team_abbr": team.get("abbreviation", ""),
                                    "team_name": team.get("displayName", ""),
                                    "season": year,
                                    "conference": conf_name,
                                    **stats_dict
                                }
                                all_stats.append(team_stats)
                    
                    logger.info(f"[wehoop] {year}: Collected team stats")
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error collecting team stats for {year}: {e}")
                    continue
                
                await asyncio.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting team stats: {e}")
        
        logger.info(f"[wehoop] Collected {len(all_stats)} team stats records")
        return all_stats

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Save teams first
            if data.get("teams"):
                saved = await self.save_teams_to_database(data["teams"], session)
                total_saved += saved
                logger.info(f"[wehoop] Saved {saved} teams")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(data["games"], session)
                total_saved += saved
                logger.info(f"[wehoop] Saved {saved} games")
            
            # Save rosters/players
            if data.get("rosters"):
                saved = await self.save_rosters_to_database(data["rosters"], session)
                total_saved += saved
                logger.info(f"[wehoop] Saved {saved} roster entries")
            
            # Save player stats
            if data.get("player_stats"):
                saved = await self.save_player_stats_to_database(data["player_stats"], session)
                total_saved += saved
                logger.info(f"[wehoop] Saved {saved} player stats")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self.save_team_stats_to_database(data["team_stats"], session)
                total_saved += saved
                logger.info(f"[wehoop] Saved {saved} team stats")
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving to database: {e}")
            raise
        
        return total_saved
    
    async def save_teams_to_database(
        self, 
        teams: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> int:
        """Save teams to database."""
        saved_count = 0
        
        try:
            # Get or create WNBA sport
            sport = await self._get_or_create_sport(session, "WNBA", "Women's National Basketball Association")
            
            # Pre-load existing teams
            existing_by_external = {}
            existing_by_name = {}
            existing_result = await session.execute(
                select(Team).where(Team.sport_id == sport.id)
            )
            for team in existing_result.scalars().all():
                if team.external_id:
                    existing_by_external[team.external_id] = team
                if team.name:
                    existing_by_name[team.name] = team
            
            for team_data in teams:
                try:
                    external_id = f"wnba_{team_data.get('espn_id', team_data.get('abbr', ''))}"
                    team_name = team_data.get("name", "")
                    
                    # Check if team exists
                    team = existing_by_external.get(external_id) or existing_by_name.get(team_name)
                    
                    if team:
                        # Update existing
                        team.abbreviation = team_data.get("abbr", team.abbreviation)
                        team.city = team_data.get("city", team.city)
                        team.conference = team_data.get("conference", team.conference)
                        team.logo_url = team_data.get("logo_url", team.logo_url)
                        if not team.external_id:
                            team.external_id = external_id
                    else:
                        # Create new
                        team = Team(
                            sport_id=sport.id,
                            external_id=external_id,
                            name=team_name,
                            abbreviation=team_data.get("abbr", ""),
                            city=team_data.get("city", ""),
                            conference=team_data.get("conference", ""),
                            logo_url=team_data.get("logo_url", ""),
                            is_active=True,
                        )
                        session.add(team)
                        existing_by_external[external_id] = team
                        existing_by_name[team_name] = team
                        saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error saving team {team_data.get('name')}: {e}")
                    continue
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving teams: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    async def _save_games(
        self, 
        games: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> int:
        """Save games to database."""
        saved_count = 0
        
        try:
            # Get sport
            sport = await self._get_or_create_sport(session, "WNBA", "Women's National Basketball Association")
            
            # Get all teams for this sport
            teams_result = await session.execute(
                select(Team).where(Team.sport_id == sport.id)
            )
            teams = {t.abbreviation: t for t in teams_result.scalars().all()}
            
            # Get existing games
            existing_result = await session.execute(
                select(Game.external_id).where(Game.sport_id == sport.id)
            )
            existing_external_ids = {r[0] for r in existing_result.fetchall()}
            
            new_games = []
            
            for game_data in games:
                try:
                    game_id = game_data.get("game_id")
                    external_id = f"wnba_{game_id}"
                    
                    if external_id in existing_external_ids:
                        continue
                    
                    # Get teams
                    home_abbr = game_data.get("home_team_abbr", "")
                    away_abbr = game_data.get("away_team_abbr", "")
                    home_team = teams.get(home_abbr)
                    away_team = teams.get(away_abbr)
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Parse scheduled time
                    scheduled_str = game_data.get("scheduled_at")
                    if scheduled_str:
                        try:
                            scheduled_at = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
                            if scheduled_at.tzinfo is not None:
                                scheduled_at = scheduled_at.replace(tzinfo=None)
                        except:
                            scheduled_at = datetime.utcnow()
                    else:
                        scheduled_at = datetime.utcnow()
                    
                    # Map status
                    status_map = {
                        "final": GameStatus.FINAL,
                        "in_progress": GameStatus.IN_PROGRESS,
                        "postponed": GameStatus.POSTPONED,
                        "scheduled": GameStatus.SCHEDULED,
                        "cancelled": GameStatus.CANCELLED,
                    }
                    status = status_map.get(game_data.get("status", "scheduled"), GameStatus.SCHEDULED)
                    
                    game = Game(
                        sport_id=sport.id,
                        external_id=external_id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=scheduled_at,
                        status=status,
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                    )
                    new_games.append(game)
                    existing_external_ids.add(external_id)
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error processing game {game_data.get('game_id')}: {e}")
                    continue
            
            if new_games:
                session.add_all(new_games)
                await session.commit()
                logger.info(f"[wehoop] Saved {len(new_games)} new games")
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving games: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    async def save_rosters_to_database(
        self, 
        rosters: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> int:
        """Save roster entries as players to database."""
        saved_count = 0
        
        try:
            # Get all teams
            teams_result = await session.execute(select(Team))
            teams = {t.abbreviation: t for t in teams_result.scalars().all()}
            
            # Get existing players
            existing_result = await session.execute(select(Player.external_id))
            existing_ids = {r[0] for r in existing_result.fetchall()}
            
            new_players = []
            
            for roster_entry in rosters:
                try:
                    player_id = roster_entry.get("player_id")
                    external_id = f"wnba_player_{player_id}"
                    
                    if external_id in existing_ids:
                        continue
                    
                    team_abbr = roster_entry.get("team_abbr", "")
                    team = teams.get(team_abbr)
                    
                    # Parse birth date
                    birth_date = None
                    birth_str = roster_entry.get("birth_date", "")
                    if birth_str:
                        try:
                            birth_date = datetime.fromisoformat(birth_str.replace("Z", "+00:00")).date()
                        except:
                            pass
                    
                    player = Player(
                        external_id=external_id,
                        team_id=team.id if team else None,
                        name=roster_entry.get("player_name", ""),
                        position=roster_entry.get("position", ""),
                        jersey_number=int(roster_entry.get("jersey_number", 0) or 0) if roster_entry.get("jersey_number") else None,
                        height=roster_entry.get("height", ""),
                        weight=int(roster_entry.get("weight", 0) or 0) if roster_entry.get("weight") else None,
                        birth_date=birth_date,
                        is_active=True,
                    )
                    new_players.append(player)
                    existing_ids.add(external_id)
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error saving player {roster_entry.get('player_name')}: {e}")
                    continue
            
            if new_players:
                session.add_all(new_players)
                await session.commit()
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving rosters: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    async def save_player_stats_to_database(
        self, 
        player_stats: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> int:
        """Save player statistics to database."""
        saved_count = 0
        
        try:
            # Get all players
            players_result = await session.execute(select(Player))
            players = {}
            for p in players_result.scalars().all():
                if p.external_id:
                    # Extract player ID from external_id
                    pid = p.external_id.replace("wnba_player_", "")
                    players[pid] = p
            
            for stats in player_stats:
                try:
                    player_id = str(stats.get("player_id", ""))
                    player = players.get(player_id)
                    
                    if not player:
                        continue
                    
                    season = stats.get("season")
                    
                    # Check if stats already exist for this player/season
                    existing = await session.execute(
                        select(PlayerStats).where(
                            and_(
                                PlayerStats.player_id == player.id,
                                PlayerStats.stat_type == f"season_{season}"
                            )
                        )
                    )
                    existing_stats = existing.scalars().first()
                    
                    # Prepare stats value (JSON)
                    stats_value = {k: v for k, v in stats.items() 
                                   if k not in ["player_id", "season", "player_name", "team_abbr"]}
                    
                    if existing_stats:
                        # Update
                        existing_stats.value = sum(v for v in stats_value.values() if isinstance(v, (int, float)))
                    else:
                        # Create
                        player_stat = PlayerStats(
                            player_id=player.id,
                            stat_type=f"season_{season}",
                            value=sum(v for v in stats_value.values() if isinstance(v, (int, float))),
                            games_played=stats.get("games_played", 0) or 0,
                        )
                        session.add(player_stat)
                        saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error saving player stats: {e}")
                    continue
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving player stats: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    async def save_team_stats_to_database(
        self, 
        team_stats: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> int:
        """Save team statistics to database."""
        saved_count = 0
        
        try:
            # Get all teams
            teams_result = await session.execute(select(Team))
            teams = {t.abbreviation: t for t in teams_result.scalars().all()}
            
            for stats in team_stats:
                try:
                    team_abbr = stats.get("team_abbr", "")
                    team = teams.get(team_abbr)
                    
                    if not team:
                        continue
                    
                    season = stats.get("season")
                    stat_type = f"season_{season}"
                    
                    # Check if stats already exist
                    existing = await session.execute(
                        select(TeamStats).where(
                            and_(
                                TeamStats.team_id == team.id,
                                TeamStats.stat_type == stat_type
                            )
                        )
                    )
                    existing_stats = existing.scalars().first()
                    
                    # Calculate aggregate value
                    wins = stats.get("wins", 0) or 0
                    losses = stats.get("losses", 0) or 0
                    games_played = wins + losses
                    
                    if existing_stats:
                        # Update
                        existing_stats.value = wins / games_played if games_played > 0 else 0
                        existing_stats.games_played = games_played
                    else:
                        # Create
                        team_stat = TeamStats(
                            team_id=team.id,
                            stat_type=stat_type,
                            value=wins / games_played if games_played > 0 else 0,
                            games_played=games_played,
                        )
                        session.add(team_stat)
                        saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error saving team stats: {e}")
                    continue
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[wehoop] Error saving team stats: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    async def _get_or_create_sport(
        self, 
        session: AsyncSession, 
        code: str, 
        name: str
    ) -> Sport:
        """Get or create sport entry."""
        result = await session.execute(
            select(Sport).where(Sport.code == code)
        )
        sport = result.scalars().first()
        
        if not sport:
            sport = Sport(
                code=code,
                name=name,
                is_active=True,
            )
            session.add(sport)
            await session.flush()
        
        return sport


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

wehoop_collector = WehoopCollector()

# Register with collector manager
try:
    collector_manager.register(wehoop_collector)
    logger.info("Registered collector: wehoop")
except Exception as e:
    logger.warning(f"Could not register wehoop collector: {e}")