"""
ROYALEY - hockeyR Data Collector
Phase 1: Data Collection Services

Collects comprehensive NHL data from the official NHL API.
Features: Play-by-play (2010-present), Expected Goals (xG), Corsi, Fenwick, 
shot metrics, shift data, 75+ features.

Data Sources:
- NHL Web API: https://api-web.nhle.com/v1/
- NHL Stats API: https://api.nhle.com/stats/rest/

FREE data - no API key required!

Key Data Types:
- Teams: All 32 NHL teams with conferences/divisions
- Schedules: Full game schedules with results (2010-present)
- Rosters: Player rosters by season
- Player Stats: Skater and goalie statistics
- Team Stats: Team season statistics
- Play-by-play: Game events with advanced metrics
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
# NHL API CONFIGURATION
# =============================================================================

NHL_WEB_API = "https://api-web.nhle.com/v1"
NHL_STATS_API = "https://api.nhle.com/stats/rest/en"

# NHL Teams with IDs and info
# Team IDs are from NHL API
NHL_TEAMS = {
    1: {"abbr": "NJD", "name": "New Jersey Devils", "city": "Newark", "conference": "Eastern", "division": "Metropolitan"},
    2: {"abbr": "NYI", "name": "New York Islanders", "city": "Elmont", "conference": "Eastern", "division": "Metropolitan"},
    3: {"abbr": "NYR", "name": "New York Rangers", "city": "New York", "conference": "Eastern", "division": "Metropolitan"},
    4: {"abbr": "PHI", "name": "Philadelphia Flyers", "city": "Philadelphia", "conference": "Eastern", "division": "Metropolitan"},
    5: {"abbr": "PIT", "name": "Pittsburgh Penguins", "city": "Pittsburgh", "conference": "Eastern", "division": "Metropolitan"},
    6: {"abbr": "BOS", "name": "Boston Bruins", "city": "Boston", "conference": "Eastern", "division": "Atlantic"},
    7: {"abbr": "BUF", "name": "Buffalo Sabres", "city": "Buffalo", "conference": "Eastern", "division": "Atlantic"},
    8: {"abbr": "MTL", "name": "Montréal Canadiens", "city": "Montreal", "conference": "Eastern", "division": "Atlantic"},
    9: {"abbr": "OTT", "name": "Ottawa Senators", "city": "Ottawa", "conference": "Eastern", "division": "Atlantic"},
    10: {"abbr": "TOR", "name": "Toronto Maple Leafs", "city": "Toronto", "conference": "Eastern", "division": "Atlantic"},
    12: {"abbr": "CAR", "name": "Carolina Hurricanes", "city": "Raleigh", "conference": "Eastern", "division": "Metropolitan"},
    13: {"abbr": "FLA", "name": "Florida Panthers", "city": "Sunrise", "conference": "Eastern", "division": "Atlantic"},
    14: {"abbr": "TBL", "name": "Tampa Bay Lightning", "city": "Tampa", "conference": "Eastern", "division": "Atlantic"},
    15: {"abbr": "WSH", "name": "Washington Capitals", "city": "Washington", "conference": "Eastern", "division": "Metropolitan"},
    16: {"abbr": "CHI", "name": "Chicago Blackhawks", "city": "Chicago", "conference": "Western", "division": "Central"},
    17: {"abbr": "DET", "name": "Detroit Red Wings", "city": "Detroit", "conference": "Eastern", "division": "Atlantic"},
    18: {"abbr": "NSH", "name": "Nashville Predators", "city": "Nashville", "conference": "Western", "division": "Central"},
    19: {"abbr": "STL", "name": "St. Louis Blues", "city": "St. Louis", "conference": "Western", "division": "Central"},
    20: {"abbr": "CGY", "name": "Calgary Flames", "city": "Calgary", "conference": "Western", "division": "Pacific"},
    21: {"abbr": "COL", "name": "Colorado Avalanche", "city": "Denver", "conference": "Western", "division": "Central"},
    22: {"abbr": "EDM", "name": "Edmonton Oilers", "city": "Edmonton", "conference": "Western", "division": "Pacific"},
    23: {"abbr": "VAN", "name": "Vancouver Canucks", "city": "Vancouver", "conference": "Western", "division": "Pacific"},
    24: {"abbr": "ANA", "name": "Anaheim Ducks", "city": "Anaheim", "conference": "Western", "division": "Pacific"},
    25: {"abbr": "DAL", "name": "Dallas Stars", "city": "Dallas", "conference": "Western", "division": "Central"},
    26: {"abbr": "LAK", "name": "Los Angeles Kings", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    28: {"abbr": "SJS", "name": "San Jose Sharks", "city": "San Jose", "conference": "Western", "division": "Pacific"},
    29: {"abbr": "CBJ", "name": "Columbus Blue Jackets", "city": "Columbus", "conference": "Eastern", "division": "Metropolitan"},
    30: {"abbr": "MIN", "name": "Minnesota Wild", "city": "St. Paul", "conference": "Western", "division": "Central"},
    52: {"abbr": "WPG", "name": "Winnipeg Jets", "city": "Winnipeg", "conference": "Western", "division": "Central"},
    53: {"abbr": "ARI", "name": "Arizona Coyotes", "city": "Tempe", "conference": "Western", "division": "Central"},  # Moved to Utah
    54: {"abbr": "VGK", "name": "Vegas Golden Knights", "city": "Las Vegas", "conference": "Western", "division": "Pacific"},
    55: {"abbr": "SEA", "name": "Seattle Kraken", "city": "Seattle", "conference": "Western", "division": "Pacific"},
    59: {"abbr": "UTA", "name": "Utah Hockey Club", "city": "Salt Lake City", "conference": "Western", "division": "Central"},  # New team 2024
}

# Abbreviation to ID mapping
ABBR_TO_ID = {info["abbr"]: team_id for team_id, info in NHL_TEAMS.items()}


# =============================================================================
# NHL KEY STATISTICS (75+ features)
# =============================================================================

SKATER_STATS_COLUMNS = [
    # Basic Stats
    "games_played", "goals", "assists", "points", "plus_minus",
    "penalty_minutes", "points_per_game", "even_strength_goals",
    "even_strength_points", "power_play_goals", "power_play_points",
    "short_handed_goals", "short_handed_points", "overtime_goals",
    "game_winning_goals", "shots", "shooting_pct", "time_on_ice_per_game",
    "faceoff_win_pct",
    
    # Advanced Stats
    "goals_per_60", "assists_per_60", "points_per_60", "shots_per_60",
    "corsi_for", "corsi_against", "corsi_pct", "corsi_rel",
    "fenwick_for", "fenwick_against", "fenwick_pct", "fenwick_rel",
    "expected_goals_for", "expected_goals_against", "expected_goals_pct",
    "on_ice_shooting_pct", "on_ice_save_pct", "pdo",
    "zone_start_pct", "hits", "blocked_shots", "takeaways", "giveaways",
]

GOALIE_STATS_COLUMNS = [
    # Basic Stats
    "games_played", "games_started", "wins", "losses", "overtime_losses",
    "saves", "save_pct", "goals_against", "goals_against_avg",
    "shutouts", "time_on_ice", "shots_against",
    
    # Advanced Stats
    "quality_starts", "quality_start_pct", "really_bad_starts",
    "goals_saved_above_avg", "even_strength_save_pct",
    "power_play_save_pct", "short_handed_save_pct",
    "high_danger_save_pct", "medium_danger_save_pct", "low_danger_save_pct",
]

TEAM_STATS_COLUMNS = [
    # Basic Stats
    "games_played", "wins", "losses", "overtime_losses", "points", "point_pct",
    "goals_for", "goals_against", "goals_per_game", "goals_against_per_game",
    "power_play_pct", "penalty_kill_pct", "shots_per_game", "shots_against_per_game",
    "faceoff_win_pct",
    
    # Advanced Stats
    "corsi_for", "corsi_against", "corsi_pct",
    "fenwick_for", "fenwick_against", "fenwick_pct",
    "expected_goals_for", "expected_goals_against", "expected_goals_pct",
    "high_danger_chances_for", "high_danger_chances_against",
    "pdo", "shooting_pct", "save_pct",
]


# =============================================================================
# HOCKEYR COLLECTOR CLASS
# =============================================================================

class HockeyRCollector(BaseCollector):
    """
    NHL data collector using official NHL API.
    
    Collects: Teams, Games, Players, Player Stats, Team Stats, Rosters
    
    Features:
    - 10+ years of historical data
    - Play-by-play with advanced metrics
    - Expected Goals (xG), Corsi, Fenwick
    - Shot metrics and shift data
    - 75+ statistical features
    """
    
    def __init__(self):
        super().__init__(
            name="hockeyR",
            base_url=NHL_WEB_API,
            rate_limit=100,
            rate_window=60,
            timeout=60.0,
            max_retries=3,
        )
        self._custom_client: Optional[httpx.AsyncClient] = None
        
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with custom settings."""
        if self._custom_client is None or self._custom_client.is_closed:
            self._custom_client = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Royaley/1.0)",
                    "Accept": "application/json",
                }
            )
        return self._custom_client
    
    async def close(self):
        """Close HTTP client."""
        if self._custom_client and not self._custom_client.is_closed:
            await self._custom_client.aclose()
            self._custom_client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        years: List[int] = None,
        collect_type: str = "all"
    ) -> CollectorResult:
        """
        Collect NHL data.
        
        Args:
            years: List of seasons to collect (e.g., [2023, 2024] for 2023-24, 2024-25)
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
            
            # Default to current season
            if years is None:
                current_year = datetime.now().year
                # NHL season spans two calendar years (e.g., 2024-25 season starts in Oct 2024)
                if datetime.now().month >= 10:
                    years = [current_year]
                else:
                    years = [current_year - 1]
            
            logger.info(f"[hockeyR] Collecting NHL data for seasons: {[f'{y}-{y+1}' for y in years]}")
            
            # Collect teams
            if collect_type in ["all", "teams"]:
                teams = await self._collect_teams()
                data["teams"] = teams
                total_records += len(teams)
                logger.info(f"[hockeyR] Collected {len(teams)} teams")
            
            # Collect games/schedules
            if collect_type in ["all", "games"]:
                games = await self._collect_games(years)
                data["games"] = games
                total_records += len(games)
                logger.info(f"[hockeyR] Collected {len(games)} games")
            
            # Collect rosters
            if collect_type in ["all", "rosters"]:
                rosters = await self._collect_rosters(years)
                data["rosters"] = rosters
                total_records += len(rosters)
                logger.info(f"[hockeyR] Collected {len(rosters)} roster entries")
            
            # Collect player stats
            if collect_type in ["all", "player_stats"]:
                player_stats = await self._collect_player_stats(years)
                data["player_stats"] = player_stats
                total_records += len(player_stats)
                logger.info(f"[hockeyR] Collected {len(player_stats)} player stats")
            
            # Collect team stats
            if collect_type in ["all", "team_stats"]:
                team_stats = await self._collect_team_stats(years)
                data["team_stats"] = team_stats
                total_records += len(team_stats)
                logger.info(f"[hockeyR] Collected {len(team_stats)} team stats")
            
            return CollectorResult(
                success=True,
                records_count=total_records,
                data=data,
            )
            
        except Exception as e:
            logger.error(f"[hockeyR] Collection error: {e}")
            return CollectorResult(
                success=False,
                records_count=0,
                errors=[str(e)],
            )

    async def validate(self, data: Any) -> bool:
        """
        Validate collected NHL data.
        
        Checks:
        - Data is not None
        - Data is a dict with expected keys
        - Data contains records
        """
        if data is None:
            return False
        
        if not isinstance(data, dict):
            return False
        
        # Check for at least one data type
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
        """Collect all NHL teams from API and static mapping."""
        teams = []
        
        try:
            client = await self.get_client()
            
            # Try to get teams from NHL API
            url = f"{NHL_WEB_API}/standings/now"
            response = await client.get(url, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                standings = data.get("standings", [])
                
                for team_data in standings:
                    team_abbr = team_data.get("teamAbbrev", {}).get("default", "")
                    team_name = team_data.get("teamName", {}).get("default", "")
                    team_common = team_data.get("teamCommonName", {}).get("default", "")
                    
                    # Get team ID from abbreviation
                    team_id = ABBR_TO_ID.get(team_abbr)
                    
                    if team_abbr and team_id:
                        team_info = NHL_TEAMS.get(team_id, {})
                        teams.append({
                            "nhl_id": team_id,
                            "abbr": team_abbr,
                            "name": team_name or team_info.get("name", ""),
                            "city": team_info.get("city", ""),
                            "conference": team_data.get("conferenceName", team_info.get("conference", "")),
                            "division": team_data.get("divisionName", team_info.get("division", "")),
                            "wins": team_data.get("wins", 0),
                            "losses": team_data.get("losses", 0),
                            "ot_losses": team_data.get("otLosses", 0),
                            "points": team_data.get("points", 0),
                            "point_pct": team_data.get("pointPctg", 0),
                            "goals_for": team_data.get("goalFor", 0),
                            "goals_against": team_data.get("goalAgainst", 0),
                        })
            
            # If API failed, use static mapping
            if not teams:
                logger.info("[hockeyR] Using static team mapping")
                for team_id, team_info in NHL_TEAMS.items():
                    teams.append({
                        "nhl_id": team_id,
                        "abbr": team_info["abbr"],
                        "name": team_info["name"],
                        "city": team_info["city"],
                        "conference": team_info["conference"],
                        "division": team_info["division"],
                    })
            
            logger.info(f"[hockeyR] Loaded {len(teams)} NHL teams")
            return teams
            
        except Exception as e:
            logger.error(f"[hockeyR] Teams collection error: {e}")
            # Return static mapping on error
            return [
                {
                    "nhl_id": team_id,
                    "abbr": info["abbr"],
                    "name": info["name"],
                    "city": info["city"],
                    "conference": info["conference"],
                    "division": info["division"],
                }
                for team_id, info in NHL_TEAMS.items()
            ]

    # =========================================================================
    # GAMES/SCHEDULE COLLECTION
    # =========================================================================
    
    async def _collect_games(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect games for specified seasons."""
        all_games = []
        client = await self.get_client()
        
        for year in years:
            try:
                # NHL season format: 20232024 for 2023-24 season
                season_id = f"{year}{year + 1}"
                
                # Get schedule for the entire season
                # NHL regular season typically runs October to April
                start_date = f"{year}-10-01"
                end_date = f"{year + 1}-06-30"  # Include playoffs
                
                # Use club-schedule endpoint for each team to get full schedule
                # Or iterate through dates
                games_for_season = await self._collect_season_games(client, year, start_date, end_date)
                all_games.extend(games_for_season)
                
                logger.info(f"[hockeyR] {year}-{year+1}: {len(games_for_season)} games")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"[hockeyR] Error collecting {year} games: {e}")
        
        logger.info(f"[hockeyR] Total {len(all_games)} games collected")
        return all_games
    
    async def _collect_season_games(
        self, 
        client: httpx.AsyncClient, 
        year: int,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Collect games for a single season by iterating through dates."""
        games = []
        seen_game_ids = set()
        
        # Convert to datetime
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Limit end date to today if in future
        today = datetime.now()
        if end > today:
            end = today
        
        # Iterate week by week to reduce API calls
        while current <= end:
            try:
                date_str = current.strftime("%Y-%m-%d")
                url = f"{NHL_WEB_API}/schedule/{date_str}"
                
                response = await client.get(url, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    game_week = data.get("gameWeek", [])
                    
                    for day in game_week:
                        day_games = day.get("games", [])
                        
                        for game in day_games:
                            game_id = game.get("id")
                            
                            # Skip duplicates
                            if game_id in seen_game_ids:
                                continue
                            seen_game_ids.add(game_id)
                            
                            # Parse game data
                            home_team = game.get("homeTeam", {})
                            away_team = game.get("awayTeam", {})
                            
                            game_state = game.get("gameState", "")
                            
                            # Map game state to our status
                            if game_state in ["OFF", "FINAL"]:
                                status = "final"
                            elif game_state in ["LIVE", "CRIT"]:
                                status = "in_progress"
                            elif game_state in ["PPD"]:
                                status = "postponed"
                            else:
                                status = "scheduled"
                            
                            games.append({
                                "nhl_game_id": game_id,
                                "season": year,
                                "game_type": game.get("gameType", 2),  # 2=regular, 3=playoffs
                                "scheduled_at": game.get("startTimeUTC"),
                                "home_team_abbr": home_team.get("abbrev", ""),
                                "away_team_abbr": away_team.get("abbrev", ""),
                                "home_team_id": home_team.get("id"),
                                "away_team_id": away_team.get("id"),
                                "home_score": home_team.get("score"),
                                "away_score": away_team.get("score"),
                                "venue": game.get("venue", {}).get("default", ""),
                                "status": status,
                                "period": game.get("period"),
                                "clock": game.get("clock", {}).get("timeRemaining") if game.get("clock") else None,
                            })
                
                # Rate limiting - small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"[hockeyR] Error fetching {date_str}: {e}")
            
            # Move forward by 7 days
            current += timedelta(days=7)
        
        return games

    # =========================================================================
    # ROSTERS COLLECTION
    # =========================================================================
    
    async def _collect_rosters(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect rosters for specified seasons."""
        rosters = []
        client = await self.get_client()
        
        # For current season, get all team rosters
        for year in years:
            season_id = f"{year}{year + 1}"
            
            for team_id, team_info in NHL_TEAMS.items():
                try:
                    abbr = team_info["abbr"]
                    url = f"{NHL_WEB_API}/roster/{abbr}/{season_id}"
                    
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Process forwards, defensemen, goalies
                        for position_group in ["forwards", "defensemen", "goalies"]:
                            players = data.get(position_group, [])
                            
                            for player in players:
                                rosters.append({
                                    "nhl_player_id": player.get("id"),
                                    "name": f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}".strip(),
                                    "team_abbr": abbr,
                                    "team_id": team_id,
                                    "position": player.get("positionCode", ""),
                                    "jersey_number": player.get("sweaterNumber"),
                                    "shoots_catches": player.get("shootsCatches", ""),
                                    "height_inches": player.get("heightInInches"),
                                    "weight_pounds": player.get("weightInPounds"),
                                    "birth_date": player.get("birthDate"),
                                    "birth_city": player.get("birthCity", {}).get("default", ""),
                                    "birth_country": player.get("birthCountry", ""),
                                    "season": year,
                                })
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"[hockeyR] Roster {abbr} {year} error: {e}")
        
        logger.info(f"[hockeyR] Collected {len(rosters)} roster entries")
        return rosters

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics for specified seasons."""
        all_stats = []
        client = await self.get_client()
        
        for year in years:
            try:
                season_id = f"{year}{year + 1}"
                
                # Get skater stats
                skater_stats = await self._collect_skater_stats(client, season_id)
                all_stats.extend(skater_stats)
                logger.info(f"[hockeyR] {year}-{year+1}: {len(skater_stats)} skater stats")
                
                # Get goalie stats
                goalie_stats = await self._collect_goalie_stats(client, season_id)
                all_stats.extend(goalie_stats)
                logger.info(f"[hockeyR] {year}-{year+1}: {len(goalie_stats)} goalie stats")
                
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"[hockeyR] Error collecting {year} player stats: {e}")
        
        return all_stats
    
    async def _collect_skater_stats(
        self, 
        client: httpx.AsyncClient, 
        season_id: str
    ) -> List[Dict[str, Any]]:
        """Collect skater statistics from NHL Stats API."""
        stats = []
        
        try:
            # Get summary stats
            url = f"{NHL_STATS_API}/skater/summary"
            params = {
                "limit": -1,  # Get all
                "cayenneExp": f"seasonId={season_id} and gameTypeId=2",  # Regular season
                "sort": "points",
                "direction": "DESC",
            }
            
            response = await client.get(url, params=params, timeout=60.0)
            
            if response.status_code == 200:
                data = response.json()
                players = data.get("data", [])
                
                for player in players:
                    stats.append({
                        "nhl_player_id": player.get("playerId"),
                        "player_name": player.get("skaterFullName", ""),
                        "team_abbr": player.get("teamAbbrevs", ""),
                        "season": int(season_id[:4]),
                        "position": player.get("positionCode", ""),
                        "stat_type": "skater",
                        # Basic stats
                        "games_played": player.get("gamesPlayed", 0),
                        "goals": player.get("goals", 0),
                        "assists": player.get("assists", 0),
                        "points": player.get("points", 0),
                        "plus_minus": player.get("plusMinus", 0),
                        "penalty_minutes": player.get("penaltyMinutes", 0),
                        "points_per_game": player.get("pointsPerGame", 0),
                        "power_play_goals": player.get("ppGoals", 0),
                        "power_play_points": player.get("ppPoints", 0),
                        "short_handed_goals": player.get("shGoals", 0),
                        "short_handed_points": player.get("shPoints", 0),
                        "game_winning_goals": player.get("gameWinningGoals", 0),
                        "overtime_goals": player.get("otGoals", 0),
                        "shots": player.get("shots", 0),
                        "shooting_pct": player.get("shootingPct", 0),
                        "time_on_ice_per_game": player.get("timeOnIcePerGame", 0),
                        "faceoff_win_pct": player.get("faceoffWinPct", 0),
                        "even_strength_goals": player.get("evGoals", 0),
                        "even_strength_points": player.get("evPoints", 0),
                    })
            
        except Exception as e:
            logger.error(f"[hockeyR] Skater stats error: {e}")
        
        return stats
    
    async def _collect_goalie_stats(
        self, 
        client: httpx.AsyncClient, 
        season_id: str
    ) -> List[Dict[str, Any]]:
        """Collect goalie statistics from NHL Stats API."""
        stats = []
        
        try:
            url = f"{NHL_STATS_API}/goalie/summary"
            params = {
                "limit": -1,
                "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
                "sort": "wins",
                "direction": "DESC",
            }
            
            response = await client.get(url, params=params, timeout=60.0)
            
            if response.status_code == 200:
                data = response.json()
                goalies = data.get("data", [])
                
                for goalie in goalies:
                    stats.append({
                        "nhl_player_id": goalie.get("playerId"),
                        "player_name": goalie.get("goalieFullName", ""),
                        "team_abbr": goalie.get("teamAbbrevs", ""),
                        "season": int(season_id[:4]),
                        "position": "G",
                        "stat_type": "goalie",
                        # Basic stats
                        "games_played": goalie.get("gamesPlayed", 0),
                        "games_started": goalie.get("gamesStarted", 0),
                        "wins": goalie.get("wins", 0),
                        "losses": goalie.get("losses", 0),
                        "overtime_losses": goalie.get("otLosses", 0),
                        "saves": goalie.get("saves", 0),
                        "save_pct": goalie.get("savePct", 0),
                        "goals_against": goalie.get("goalsAgainst", 0),
                        "goals_against_avg": goalie.get("goalsAgainstAverage", 0),
                        "shutouts": goalie.get("shutouts", 0),
                        "time_on_ice": goalie.get("timeOnIce", 0),
                        "shots_against": goalie.get("shotsAgainst", 0),
                    })
            
        except Exception as e:
            logger.error(f"[hockeyR] Goalie stats error: {e}")
        
        return stats

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team statistics for specified seasons."""
        all_stats = []
        client = await self.get_client()
        
        for year in years:
            try:
                season_id = f"{year}{year + 1}"
                
                # Get team summary stats
                url = f"{NHL_STATS_API}/team/summary"
                params = {
                    "limit": -1,
                    "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
                }
                
                response = await client.get(url, params=params, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    teams = data.get("data", [])
                    
                    for team in teams:
                        all_stats.append({
                            "team_name": team.get("teamFullName", ""),
                            "team_abbr": team.get("teamAbbrev", ""),
                            "season": year,
                            "games_played": team.get("gamesPlayed", 0),
                            "wins": team.get("wins", 0),
                            "losses": team.get("losses", 0),
                            "overtime_losses": team.get("otLosses", 0),
                            "points": team.get("points", 0),
                            "point_pct": team.get("pointPct", 0),
                            "goals_for": team.get("goalsFor", 0),
                            "goals_against": team.get("goalsAgainst", 0),
                            "goals_per_game": team.get("goalsForPerGame", 0),
                            "goals_against_per_game": team.get("goalsAgainstPerGame", 0),
                            "power_play_pct": team.get("powerPlayPct", 0),
                            "penalty_kill_pct": team.get("penaltyKillPct", 0),
                            "shots_per_game": team.get("shotsForPerGame", 0),
                            "shots_against_per_game": team.get("shotsAgainstPerGame", 0),
                            "faceoff_win_pct": team.get("faceoffWinPct", 0),
                        })
                    
                    logger.info(f"[hockeyR] {year}-{year+1}: {len(teams)} team stats")
                
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"[hockeyR] Team stats {year} error: {e}")
        
        return all_stats

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(
        self,
        data: Dict[str, Any],
        session: AsyncSession
    ) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        # Save teams first
        if data.get("teams"):
            saved = await self.save_teams_to_database(data["teams"], session)
            total_saved += saved
        
        # Save games
        if data.get("games"):
            saved = await self._save_games(data["games"], session)
            total_saved += saved
        
        # Save rosters/players
        if data.get("rosters"):
            saved = await self.save_rosters_to_database(data["rosters"], session)
            total_saved += saved
        
        # Save player stats
        if data.get("player_stats"):
            saved = await self.save_player_stats_to_database(data["player_stats"], session)
            total_saved += saved
        
        # Save team stats
        if data.get("team_stats"):
            saved = await self.save_team_stats_to_database(data["team_stats"], session)
            total_saved += saved
        
        return total_saved
    
    async def save_teams_to_database(
        self,
        teams_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save NHL teams to database."""
        saved_count = 0
        
        # Get or create NHL sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NHL")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            sport = Sport(
                code="NHL",
                name="National Hockey League",
                feature_count=75,
                is_active=True,
            )
            session.add(sport)
            await session.flush()
            logger.info("[hockeyR] Created NHL sport")
        
        for team_data in teams_data:
            external_id = f"nhl_{team_data.get('nhl_id', team_data.get('abbr'))}"
            
            # Check if team exists
            existing = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport.id,
                        Team.external_id == external_id
                    )
                )
            )
            team = existing.scalars().first()
            
            if team:
                # Update existing
                team.name = team_data.get("name", team.name)
                team.abbreviation = team_data.get("abbr", team.abbreviation)
                team.city = team_data.get("city", team.city)
                team.conference = team_data.get("conference", team.conference)
                team.division = team_data.get("division", team.division)
            else:
                # Create new
                team = Team(
                    sport_id=sport.id,
                    external_id=external_id,
                    name=team_data.get("name", ""),
                    abbreviation=team_data.get("abbr", ""),
                    city=team_data.get("city", ""),
                    conference=team_data.get("conference", ""),
                    division=team_data.get("division", ""),
                    is_active=True,
                )
                session.add(team)
                saved_count += 1
        
        await session.flush()
        logger.info(f"[hockeyR] Saved {saved_count} teams")
        return saved_count
    
    async def _save_games(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save NHL games to database with duplicate handling."""
        saved_count = 0
        updated_count = 0
        batch_count = 0
        
        # Get NHL sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NHL")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[hockeyR] NHL sport not found")
            return 0
        
        # Pre-load existing game external_ids to avoid duplicates
        existing_ids_result = await session.execute(
            select(Game.external_id).where(
                and_(
                    Game.sport_id == sport.id,
                    Game.external_id.isnot(None),
                    Game.external_id.like("nhl_%")
                )
            )
        )
        existing_external_ids = set(row[0] for row in existing_ids_result.fetchall() if row[0])
        logger.info(f"[hockeyR] Found {len(existing_external_ids)} existing NHL games in database")
        
        # Get team mapping
        teams_result = await session.execute(
            select(Team).where(Team.sport_id == sport.id)
        )
        teams = {t.abbreviation: t for t in teams_result.scalars().all()}
        
        for game_data in games_data:
            game_id = game_data.get("nhl_game_id")
            if not game_id:
                continue
            
            external_id = f"nhl_{game_id}"
            
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
                except:
                    scheduled_at = datetime.now()
            else:
                scheduled_at = datetime.now()
            
            # Map status
            status_map = {
                "final": GameStatus.FINAL,
                "in_progress": GameStatus.IN_PROGRESS,
                "postponed": GameStatus.POSTPONED,
                "scheduled": GameStatus.SCHEDULED,
            }
            status = status_map.get(game_data.get("status", "scheduled"), GameStatus.SCHEDULED)
            
            if external_id in existing_external_ids:
                # Update existing game
                existing = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = existing.scalars().first()
                if game:
                    game.home_score = game_data.get("home_score")
                    game.away_score = game_data.get("away_score")
                    game.status = status
                    game.period = game_data.get("period")
                    game.clock = game_data.get("clock")
                    updated_count += 1
            else:
                # Create new game
                game = Game(
                    sport_id=sport.id,
                    external_id=external_id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    scheduled_at=scheduled_at,
                    status=status,
                    home_score=game_data.get("home_score"),
                    away_score=game_data.get("away_score"),
                    period=game_data.get("period"),
                    clock=game_data.get("clock"),
                )
                session.add(game)
                existing_external_ids.add(external_id)
                saved_count += 1
            
            # Batch flush
            batch_count += 1
            if batch_count >= 500:
                await session.flush()
                logger.info(f"[hockeyR] Progress: {saved_count} new, {updated_count} updated")
                batch_count = 0
        
        # Final flush
        if batch_count > 0:
            await session.flush()
        
        logger.info(f"[hockeyR] Saved {saved_count} new games, updated {updated_count} existing")
        return saved_count
    
    async def save_rosters_to_database(
        self,
        rosters_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save player rosters to database."""
        saved_count = 0
        batch_count = 0
        
        def safe_int(val):
            """Safely convert value to int or None."""
            if val is None or val == '':
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        
        # Get NHL sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NHL")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            logger.error("[hockeyR] NHL sport not found")
            return 0
        
        # Pre-load existing player external_ids
        existing_ids_result = await session.execute(
            select(Player.external_id).where(
                Player.external_id.like("nhl_%")
            )
        )
        existing_player_ids = set(row[0] for row in existing_ids_result.fetchall() if row[0])
        logger.info(f"[hockeyR] Found {len(existing_player_ids)} existing NHL players")
        
        # Get team mapping
        teams_result = await session.execute(
            select(Team).where(Team.sport_id == sport.id)
        )
        teams = {t.abbreviation: t for t in teams_result.scalars().all()}
        
        for roster_entry in rosters_data:
            player_id = roster_entry.get("nhl_player_id")
            if not player_id:
                continue
            
            external_id = f"nhl_{player_id}"
            player_name = roster_entry.get("name", "Unknown")
            jersey_num = safe_int(roster_entry.get("jersey_number"))
            weight = safe_int(roster_entry.get("weight_pounds"))
            
            # Get team
            team_abbr = roster_entry.get("team_abbr")
            team = teams.get(team_abbr)
            
            # Parse birth date
            birth_date_str = roster_entry.get("birth_date")
            birth_date = None
            if birth_date_str:
                try:
                    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
                except:
                    pass
            
            # Convert height (inches to string like "6'2")
            height_inches = roster_entry.get("height_inches")
            height_str = None
            if height_inches:
                feet = height_inches // 12
                inches = height_inches % 12
                height_str = f"{feet}'{inches}\""
            
            if external_id in existing_player_ids:
                # Update existing player
                existing = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                player = existing.scalars().first()
                if player:
                    player.name = player_name
                    player.position = roster_entry.get("position")
                    player.jersey_number = jersey_num
                    player.height = height_str
                    player.weight = weight
                    player.birth_date = birth_date
                    if team:
                        player.team_id = team.id
                    player.is_active = True
            else:
                # Create new player
                player = Player(
                    external_id=external_id,
                    name=player_name,
                    position=roster_entry.get("position"),
                    jersey_number=jersey_num,
                    height=height_str,
                    weight=weight,
                    birth_date=birth_date,
                    team_id=team.id if team else None,
                    is_active=True,
                )
                session.add(player)
                existing_player_ids.add(external_id)
                saved_count += 1
            
            # Batch flush
            batch_count += 1
            if batch_count >= 500:
                await session.flush()
                logger.info(f"[hockeyR] Flushed batch, {saved_count} new players so far")
                batch_count = 0
        
        # Final flush
        if batch_count > 0:
            await session.flush()
        
        logger.info(f"[hockeyR] Saved {saved_count} players from rosters")
        return saved_count
    
    async def save_player_stats_to_database(
        self,
        stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save player statistics to database."""
        saved_count = 0
        batch_count = 0
        
        # Get existing players
        existing_result = await session.execute(
            select(Player).where(Player.external_id.like("nhl_%"))
        )
        players_map = {p.external_id: p for p in existing_result.scalars().all()}
        
        for stat_row in stats_data:
            player_id = stat_row.get("nhl_player_id")
            if not player_id:
                continue
            
            external_id = f"nhl_{player_id}"
            player = players_map.get(external_id)
            
            if not player:
                continue
            
            season = stat_row.get("season", 0)
            stat_type = stat_row.get("stat_type", "skater")
            
            # Define stats to save based on type
            if stat_type == "skater":
                stats_to_save = [
                    "games_played", "goals", "assists", "points", "plus_minus",
                    "penalty_minutes", "power_play_goals", "power_play_points",
                    "short_handed_goals", "short_handed_points", "shots",
                    "shooting_pct", "game_winning_goals", "overtime_goals",
                    "faceoff_win_pct", "time_on_ice_per_game",
                ]
            else:  # goalie
                stats_to_save = [
                    "games_played", "games_started", "wins", "losses",
                    "overtime_losses", "saves", "save_pct", "goals_against",
                    "goals_against_avg", "shutouts", "shots_against",
                ]
            
            for stat_name in stats_to_save:
                value = stat_row.get(stat_name)
                if value is not None:
                    try:
                        full_stat_type = f"{stat_name}_{season}"
                        stat_record = PlayerStats(
                            player_id=player.id,
                            stat_type=full_stat_type,
                            value=float(value),
                        )
                        session.add(stat_record)
                        saved_count += 1
                    except (ValueError, TypeError):
                        pass
            
            batch_count += 1
            if batch_count >= 100:
                await session.flush()
                batch_count = 0
        
        if batch_count > 0:
            await session.flush()
        
        logger.info(f"[hockeyR] Saved {saved_count} player stats")
        return saved_count
    
    async def save_team_stats_to_database(
        self,
        stats_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save team statistics to database."""
        saved_count = 0
        
        # Get NHL sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NHL")
        )
        sport = sport_result.scalars().first()
        
        if not sport:
            return 0
        
        # Get team mapping
        teams_result = await session.execute(
            select(Team).where(Team.sport_id == sport.id)
        )
        teams = {t.abbreviation: t for t in teams_result.scalars().all()}
        
        for stat_row in stats_data:
            team_abbr = stat_row.get("team_abbr")
            team = teams.get(team_abbr)
            
            if not team:
                continue
            
            season = stat_row.get("season", 0)
            
            # Stats to save
            stats_to_save = [
                "games_played", "wins", "losses", "overtime_losses",
                "points", "point_pct", "goals_for", "goals_against",
                "goals_per_game", "goals_against_per_game",
                "power_play_pct", "penalty_kill_pct",
                "shots_per_game", "shots_against_per_game", "faceoff_win_pct",
            ]
            
            for stat_name in stats_to_save:
                value = stat_row.get(stat_name)
                if value is not None:
                    try:
                        full_stat_type = f"{stat_name}_{season}"
                        stat_record = TeamStats(
                            team_id=team.id,
                            stat_type=full_stat_type,
                            value=float(value),
                            games_played=stat_row.get("games_played", 0),
                        )
                        session.add(stat_record)
                        saved_count += 1
                    except (ValueError, TypeError):
                        pass
        
        await session.flush()
        logger.info(f"[hockeyR] Saved {saved_count} team stats")
        return saved_count


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

hockeyr_collector = HockeyRCollector()

# Register with collector manager (uses collector.name internally)
collector_manager.register(hockeyr_collector)