"""
ROYALEY - hoopR Data Collector
Phase 1: Data Collection Services

Collects comprehensive NBA and NCAAB data from ESPN APIs and NBA Stats API.
Features: Games, rosters, player stats, team stats, standings.

Data Sources:
- ESPN API: https://site.api.espn.com/apis/site/v2/sports/basketball/
- NBA Stats API: https://stats.nba.com/stats/

FREE data - no API key required!

Key Data Types:
- Teams: All 30 NBA teams + 350+ NCAAB teams
- Games: Full game schedules with results (10+ years)
- Rosters: Player rosters by season
- Player Stats: Points, rebounds, assists, etc.
- Team Stats: Wins, losses, standings data

Tables Filled:
- sports (NBA, NCAAB entries)
- teams (30 NBA + 350+ NCAAB teams)
- games (10 years of games)
- players (all players)
- player_stats (seasonal stats)
- team_stats (seasonal stats)
- venues (arenas)
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
# ESPN API CONFIGURATION
# =============================================================================

ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NCAAB_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

ESPN_ENDPOINTS = {
    "NBA": {
        "base": ESPN_NBA_BASE,
        "teams": f"{ESPN_NBA_BASE}/teams",
        "scoreboard": f"{ESPN_NBA_BASE}/scoreboard",
        "standings": f"{ESPN_NBA_BASE}/standings",
        "sport_code": "NBA",
    },
    "NCAAB": {
        "base": ESPN_NCAAB_BASE,
        "teams": f"{ESPN_NCAAB_BASE}/teams",
        "scoreboard": f"{ESPN_NCAAB_BASE}/scoreboard",
        "standings": f"{ESPN_NCAAB_BASE}/standings",
        "sport_code": "NCAAB",
    }
}

# =============================================================================
# NBA TEAMS (30 Teams)
# =============================================================================

NBA_TEAMS = {
    # Atlantic Division (Eastern Conference)
    "1": {"abbr": "BOS", "name": "Boston Celtics", "city": "Boston", "conference": "Eastern", "division": "Atlantic"},
    "17": {"abbr": "BKN", "name": "Brooklyn Nets", "city": "Brooklyn", "conference": "Eastern", "division": "Atlantic"},
    "18": {"abbr": "NYK", "name": "New York Knicks", "city": "New York", "conference": "Eastern", "division": "Atlantic"},
    "19": {"abbr": "PHI", "name": "Philadelphia 76ers", "city": "Philadelphia", "conference": "Eastern", "division": "Atlantic"},
    "28": {"abbr": "TOR", "name": "Toronto Raptors", "city": "Toronto", "conference": "Eastern", "division": "Atlantic"},
    
    # Central Division (Eastern Conference)
    "4": {"abbr": "CHI", "name": "Chicago Bulls", "city": "Chicago", "conference": "Eastern", "division": "Central"},
    "5": {"abbr": "CLE", "name": "Cleveland Cavaliers", "city": "Cleveland", "conference": "Eastern", "division": "Central"},
    "8": {"abbr": "DET", "name": "Detroit Pistons", "city": "Detroit", "conference": "Eastern", "division": "Central"},
    "11": {"abbr": "IND", "name": "Indiana Pacers", "city": "Indianapolis", "conference": "Eastern", "division": "Central"},
    "15": {"abbr": "MIL", "name": "Milwaukee Bucks", "city": "Milwaukee", "conference": "Eastern", "division": "Central"},
    
    # Southeast Division (Eastern Conference)
    "1610612737": {"abbr": "ATL", "name": "Atlanta Hawks", "city": "Atlanta", "conference": "Eastern", "division": "Southeast"},
    "3": {"abbr": "CHA", "name": "Charlotte Hornets", "city": "Charlotte", "conference": "Eastern", "division": "Southeast"},
    "14": {"abbr": "MIA", "name": "Miami Heat", "city": "Miami", "conference": "Eastern", "division": "Southeast"},
    "19": {"abbr": "ORL", "name": "Orlando Magic", "city": "Orlando", "conference": "Eastern", "division": "Southeast"},
    "27": {"abbr": "WAS", "name": "Washington Wizards", "city": "Washington", "conference": "Eastern", "division": "Southeast"},
    
    # Northwest Division (Western Conference)
    "7": {"abbr": "DEN", "name": "Denver Nuggets", "city": "Denver", "conference": "Western", "division": "Northwest"},
    "16": {"abbr": "MIN", "name": "Minnesota Timberwolves", "city": "Minneapolis", "conference": "Western", "division": "Northwest"},
    "22": {"abbr": "OKC", "name": "Oklahoma City Thunder", "city": "Oklahoma City", "conference": "Western", "division": "Northwest"},
    "21": {"abbr": "POR", "name": "Portland Trail Blazers", "city": "Portland", "conference": "Western", "division": "Northwest"},
    "26": {"abbr": "UTA", "name": "Utah Jazz", "city": "Salt Lake City", "conference": "Western", "division": "Northwest"},
    
    # Pacific Division (Western Conference)
    "9": {"abbr": "GSW", "name": "Golden State Warriors", "city": "San Francisco", "conference": "Western", "division": "Pacific"},
    "12": {"abbr": "LAC", "name": "Los Angeles Clippers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "13": {"abbr": "LAL", "name": "Los Angeles Lakers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "23": {"abbr": "PHX", "name": "Phoenix Suns", "city": "Phoenix", "conference": "Western", "division": "Pacific"},
    "23": {"abbr": "SAC", "name": "Sacramento Kings", "city": "Sacramento", "conference": "Western", "division": "Pacific"},
    
    # Southwest Division (Western Conference)
    "6": {"abbr": "DAL", "name": "Dallas Mavericks", "city": "Dallas", "conference": "Western", "division": "Southwest"},
    "10": {"abbr": "HOU", "name": "Houston Rockets", "city": "Houston", "conference": "Western", "division": "Southwest"},
    "20": {"abbr": "MEM", "name": "Memphis Grizzlies", "city": "Memphis", "conference": "Western", "division": "Southwest"},
    "3": {"abbr": "NOP", "name": "New Orleans Pelicans", "city": "New Orleans", "conference": "Western", "division": "Southwest"},
    "24": {"abbr": "SAS", "name": "San Antonio Spurs", "city": "San Antonio", "conference": "Western", "division": "Southwest"},
}

# =============================================================================
# NCAAB CONFERENCES (Major Conferences)
# =============================================================================

NCAAB_MAJOR_CONFERENCES = [
    {"id": "1", "name": "ACC", "full_name": "Atlantic Coast Conference"},
    {"id": "4", "name": "Big 12", "full_name": "Big 12 Conference"},
    {"id": "5", "name": "Big East", "full_name": "Big East Conference"},
    {"id": "7", "name": "Big Ten", "full_name": "Big Ten Conference"},
    {"id": "8", "name": "Pac-12", "full_name": "Pac-12 Conference"},
    {"id": "9", "name": "SEC", "full_name": "Southeastern Conference"},
    {"id": "12", "name": "Mountain West", "full_name": "Mountain West Conference"},
    {"id": "3", "name": "AAC", "full_name": "American Athletic Conference"},
    {"id": "18", "name": "WCC", "full_name": "West Coast Conference"},
    {"id": "46", "name": "Atlantic 10", "full_name": "Atlantic 10 Conference"},
]


# =============================================================================
# HOOPR COLLECTOR CLASS
# =============================================================================

class HoopRCollector(BaseCollector):
    """
    Collector for NBA and NCAAB data using ESPN API.
    
    Features:
    - Game schedules and results (10+ years)
    - Team information and standings
    - Player rosters and statistics
    - Team statistics
    - No API key required (FREE)
    
    Supports:
    - NBA (30 teams)
    - NCAAB (350+ teams from major conferences)
    """
    
    def __init__(self):
        super().__init__(
            name="hoopr",
            base_url=ESPN_NBA_BASE,
            rate_limit=60,
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
        leagues: List[str] = None,
        collect_type: str = "all"
    ) -> CollectorResult:
        """
        Collect basketball data from ESPN APIs.
        
        Args:
            years: List of seasons to collect (e.g., [2023, 2024])
            leagues: List of leagues ["NBA", "NCAAB"] or None for all
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
        
        if leagues is None:
            leagues = ["NBA", "NCAAB"]
        
        logger.info(f"[hoopR] Collecting data for leagues: {leagues}, seasons: {years}")
        
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
            for league in leagues:
                logger.info(f"[hoopR] Processing {league}...")
                
                # Collect teams
                if collect_type in ["all", "teams"]:
                    teams = await self._collect_teams(league)
                    data["teams"].extend(teams)
                    total_records += len(teams)
                    logger.info(f"[hoopR] {league}: Collected {len(teams)} teams")
                
                # Collect games
                if collect_type in ["all", "games"]:
                    games = await self._collect_games(league, years)
                    data["games"].extend(games)
                    total_records += len(games)
                    logger.info(f"[hoopR] {league}: Collected {len(games)} games")
                
                # Collect rosters
                if collect_type in ["all", "rosters"]:
                    rosters = await self._collect_rosters(league, years)
                    data["rosters"].extend(rosters)
                    total_records += len(rosters)
                    logger.info(f"[hoopR] {league}: Collected {len(rosters)} roster entries")
                
                # Collect team stats
                if collect_type in ["all", "team_stats"]:
                    team_stats = await self._collect_team_stats(league, years)
                    data["team_stats"].extend(team_stats)
                    total_records += len(team_stats)
                    logger.info(f"[hoopR] {league}: Collected {len(team_stats)} team stats")
            
            logger.info(f"[hoopR] Total records collected: {total_records}")
            
            return CollectorResult(
                success=True,
                data=data,
                records_count=total_records,
            )
            
        except Exception as e:
            logger.error(f"[hoopR] Collection error: {e}")
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
    
    async def _collect_teams(self, league: str) -> List[Dict[str, Any]]:
        """Collect all teams for a league."""
        teams = []
        
        try:
            client = await self.get_client()
            endpoint = ESPN_ENDPOINTS[league]["teams"]
            
            response = await client.get(f"{endpoint}?limit=500", timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse ESPN teams response
                sports = data.get("sports", [])
                for sport in sports:
                    for league_data in sport.get("leagues", []):
                        for team_entry in league_data.get("teams", []):
                            team = team_entry.get("team", team_entry)
                            
                            team_id = str(team.get("id", ""))
                            if not team_id:
                                continue
                            
                            # Get location/venue info
                            location = team.get("location", "")
                            venue_info = team.get("venue", {}) if isinstance(team.get("venue"), dict) else {}
                            
                            teams.append({
                                "external_id": f"espn_{league.lower()}_{team_id}",
                                "espn_id": team_id,
                                "name": team.get("displayName", team.get("name", "")),
                                "abbreviation": team.get("abbreviation", ""),
                                "city": location or team.get("shortDisplayName", ""),
                                "conference": team.get("groups", {}).get("parent", {}).get("name", "") if isinstance(team.get("groups"), dict) else "",
                                "division": "",
                                "logo_url": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                                "color": team.get("color", ""),
                                "alternate_color": team.get("alternateColor", ""),
                                "league": league,
                                "venue_name": venue_info.get("fullName", venue_info.get("name", "")),
                                "venue_city": venue_info.get("address", {}).get("city", "") if isinstance(venue_info.get("address"), dict) else "",
                                "is_active": team.get("isActive", True),
                            })
                
                logger.info(f"[hoopR] Loaded {len(teams)} {league} teams from ESPN")
                
        except Exception as e:
            logger.error(f"[hoopR] Error collecting {league} teams: {e}")
            import traceback
            traceback.print_exc()
        
        return teams

    # =========================================================================
    # GAMES COLLECTION
    # =========================================================================
    
    async def _collect_games(self, league: str, years: List[int]) -> List[Dict[str, Any]]:
        """Collect games for specified seasons using team schedules."""
        all_games = []
        seen_game_ids = set()
        
        try:
            client = await self.get_client()
            base_url = ESPN_ENDPOINTS[league]["base"]
            
            # First get all teams
            teams_response = await client.get(f"{ESPN_ENDPOINTS[league]['teams']}?limit=500", timeout=30.0)
            teams_list = []
            
            if teams_response.status_code == 200:
                teams_data = teams_response.json()
                sports = teams_data.get("sports", [])
                for sport in sports:
                    for league_data in sport.get("leagues", []):
                        for team_entry in league_data.get("teams", []):
                            team = team_entry.get("team", team_entry)
                            team_id = team.get("id")
                            if team_id:
                                teams_list.append({
                                    "id": str(team_id),
                                    "abbr": team.get("abbreviation", ""),
                                    "name": team.get("displayName", "")
                                })
            
            # For NCAAB, limit to top teams to avoid overwhelming API
            if league == "NCAAB":
                # Get top 100 teams from rankings or limit scope
                teams_list = teams_list[:100]
                logger.info(f"[hoopR] NCAAB: Limited to top {len(teams_list)} teams")
            
            logger.info(f"[hoopR] {league}: Collecting games for {len(teams_list)} teams")
            
            for year in years:
                year_count = 0
                
                for team in teams_list:
                    team_id = team["id"]
                    
                    try:
                        # Get team schedule
                        url = f"{base_url}/teams/{team_id}/schedule?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            events = data.get("events", [])
                            
                            for event in events:
                                game_id = str(event.get("id", ""))
                                
                                if not game_id or game_id in seen_game_ids:
                                    continue
                                seen_game_ids.add(game_id)
                                
                                # Parse competition
                                competitions = event.get("competitions", [{}])
                                if not competitions:
                                    continue
                                    
                                competition = competitions[0]
                                competitors = competition.get("competitors", [])
                                
                                if len(competitors) < 2:
                                    continue
                                
                                # Identify home/away teams
                                home_team = None
                                away_team = None
                                for comp in competitors:
                                    if comp.get("homeAway") == "home":
                                        home_team = comp
                                    else:
                                        away_team = comp
                                
                                if not home_team or not away_team:
                                    continue
                                
                                # Parse game time
                                game_date_str = event.get("date", "")
                                try:
                                    if game_date_str:
                                        # Handle ISO format with Z
                                        game_date_str = game_date_str.replace("Z", "+00:00")
                                        scheduled_at = datetime.fromisoformat(game_date_str)
                                        # Remove timezone for database compatibility
                                        if scheduled_at.tzinfo is not None:
                                            scheduled_at = scheduled_at.replace(tzinfo=None)
                                    else:
                                        scheduled_at = datetime(year, 1, 1)
                                except:
                                    scheduled_at = datetime(year, 1, 1)
                                
                                # Determine game status
                                status_type = event.get("status", {}).get("type", {})
                                status_name = status_type.get("name", "STATUS_SCHEDULED")
                                
                                if status_name in ["STATUS_FINAL", "STATUS_FULL_TIME"]:
                                    game_status = "final"
                                elif status_name in ["STATUS_IN_PROGRESS", "STATUS_HALFTIME"]:
                                    game_status = "in_progress"
                                elif status_name in ["STATUS_POSTPONED"]:
                                    game_status = "postponed"
                                elif status_name in ["STATUS_CANCELED", "STATUS_CANCELLED"]:
                                    game_status = "cancelled"
                                else:
                                    game_status = "scheduled"
                                
                                # Get scores
                                home_score = None
                                away_score = None
                                try:
                                    home_score = int(home_team.get("score", {}).get("value", 0)) if isinstance(home_team.get("score"), dict) else int(home_team.get("score", 0) or 0)
                                    away_score = int(away_team.get("score", {}).get("value", 0)) if isinstance(away_team.get("score"), dict) else int(away_team.get("score", 0) or 0)
                                except:
                                    pass
                                
                                # Get venue
                                venue = competition.get("venue", {})
                                venue_name = venue.get("fullName", venue.get("name", "")) if isinstance(venue, dict) else ""
                                venue_city = venue.get("address", {}).get("city", "") if isinstance(venue, dict) and isinstance(venue.get("address"), dict) else ""
                                
                                game_record = {
                                    "external_id": f"espn_{league.lower()}_{game_id}",
                                    "espn_id": game_id,
                                    "league": league,
                                    "season": year,
                                    "scheduled_at": scheduled_at,
                                    "status": game_status,
                                    "home_team_id": str(home_team.get("team", {}).get("id", home_team.get("id", ""))),
                                    "home_team_abbr": home_team.get("team", {}).get("abbreviation", ""),
                                    "home_team_name": home_team.get("team", {}).get("displayName", ""),
                                    "away_team_id": str(away_team.get("team", {}).get("id", away_team.get("id", ""))),
                                    "away_team_abbr": away_team.get("team", {}).get("abbreviation", ""),
                                    "away_team_name": away_team.get("team", {}).get("displayName", ""),
                                    "home_score": home_score,
                                    "away_score": away_score,
                                    "venue_name": venue_name,
                                    "venue_city": venue_city,
                                    "attendance": competition.get("attendance", 0),
                                    "neutral_site": competition.get("neutralSite", False),
                                }
                                
                                all_games.append(game_record)
                                year_count += 1
                        
                        await asyncio.sleep(0.05)  # Rate limiting
                        
                    except Exception as e:
                        logger.debug(f"[hoopR] Error getting schedule for team {team_id}: {e}")
                        continue
                
                logger.info(f"[hoopR] {league} {year}: {year_count} games collected")
            
        except Exception as e:
            logger.error(f"[hoopR] Error collecting {league} games: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"[hoopR] {league}: Total {len(all_games)} unique games collected")
        return all_games

    # =========================================================================
    # ROSTERS COLLECTION
    # =========================================================================
    
    async def _collect_rosters(self, league: str, years: List[int]) -> List[Dict[str, Any]]:
        """Collect rosters for all teams."""
        all_rosters = []
        seen_players = set()
        debug_logged = False
        
        try:
            client = await self.get_client()
            base_url = ESPN_ENDPOINTS[league]["base"]
            
            # Get teams first
            teams_response = await client.get(f"{ESPN_ENDPOINTS[league]['teams']}?limit=500", timeout=30.0)
            teams_list = []
            
            if teams_response.status_code == 200:
                teams_data = teams_response.json()
                sports = teams_data.get("sports", [])
                for sport in sports:
                    for league_data in sport.get("leagues", []):
                        for team_entry in league_data.get("teams", []):
                            team = team_entry.get("team", team_entry)
                            team_id = team.get("id")
                            if team_id:
                                teams_list.append({
                                    "id": str(team_id),
                                    "abbr": team.get("abbreviation", ""),
                                })
            
            # Limit NCAAB teams
            if league == "NCAAB":
                teams_list = teams_list[:100]
            
            for year in years:
                year_count = 0
                
                for team in teams_list:
                    team_id = team["id"]
                    
                    try:
                        url = f"{base_url}/teams/{team_id}/roster?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Debug: log structure on first successful response
                            if not debug_logged and year == years[0]:
                                import json
                                logger.info(f"[hoopR] DEBUG {league} roster keys: {list(data.keys())}")
                                debug_logged = True
                            
                            players = []
                            
                            # Structure 1: athletes[] with position groups containing items[]
                            athletes = data.get("athletes", [])
                            for group in athletes:
                                if isinstance(group, dict):
                                    # Check for items array
                                    items = group.get("items", [])
                                    if items:
                                        players.extend(items)
                                    # Also check if group itself has player data
                                    if group.get("id") and group.get("displayName"):
                                        players.append(group)
                                elif isinstance(group, list):
                                    players.extend(group)
                            
                            # Structure 2: Direct athletes array with players
                            if not players and athletes:
                                for athlete in athletes:
                                    if isinstance(athlete, dict) and athlete.get("id"):
                                        players.append(athlete)
                            
                            # Structure 3: roster[] array
                            roster_list = data.get("roster", [])
                            if roster_list:
                                for entry in roster_list:
                                    if isinstance(entry, dict):
                                        if entry.get("id"):
                                            players.append(entry)
                                        # Check for nested athlete
                                        athlete = entry.get("athlete", {})
                                        if athlete and athlete.get("id"):
                                            players.append(athlete)
                            
                            # Structure 4: team.athletes[]
                            team_data = data.get("team", {})
                            if team_data:
                                team_athletes = team_data.get("athletes", [])
                                for athlete in team_athletes:
                                    if isinstance(athlete, dict) and athlete.get("id"):
                                        players.append(athlete)
                            
                            # Process found players
                            for player in players:
                                player_id = str(player.get("id", ""))
                                if not player_id:
                                    continue
                                
                                unique_key = f"{player_id}_{year}"
                                if unique_key in seen_players:
                                    continue
                                seen_players.add(unique_key)
                                
                                # Parse position
                                position = ""
                                if isinstance(player.get("position"), dict):
                                    position = player.get("position", {}).get("abbreviation", "")
                                else:
                                    position = str(player.get("position", ""))
                                
                                roster_entry = {
                                    "external_id": f"espn_{league.lower()}_{player_id}",
                                    "espn_id": player_id,
                                    "league": league,
                                    "season": year,
                                    "team_id": team_id,
                                    "team_abbr": team["abbr"],
                                    "player_name": player.get("displayName", player.get("fullName", "")),
                                    "position": position,
                                    "jersey_number": player.get("jersey", ""),
                                    "height": player.get("displayHeight", ""),
                                    "weight": player.get("displayWeight", ""),
                                    "birth_date": player.get("dateOfBirth", ""),
                                    "age": player.get("age", 0),
                                    "birth_place": player.get("birthPlace", {}).get("city", "") if isinstance(player.get("birthPlace"), dict) else "",
                                    "experience": player.get("experience", {}).get("years", 0) if isinstance(player.get("experience"), dict) else 0,
                                    "college": player.get("college", {}).get("name", "") if isinstance(player.get("college"), dict) else "",
                                    "headshot_url": player.get("headshot", {}).get("href", "") if isinstance(player.get("headshot"), dict) else "",
                                }
                                
                                all_rosters.append(roster_entry)
                                year_count += 1
                        
                        await asyncio.sleep(0.05)
                        
                    except Exception as e:
                        logger.debug(f"[hoopR] Error getting roster for team {team_id}: {e}")
                        continue
                
                logger.info(f"[hoopR] {league} {year}: {year_count} roster entries")
            
        except Exception as e:
            logger.error(f"[hoopR] Error collecting {league} rosters: {e}")
        
        logger.info(f"[hoopR] {league}: Total {len(all_rosters)} roster entries")
        return all_rosters

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, league: str, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team statistics from standings."""
        all_stats = []
        debug_logged = False
        
        try:
            client = await self.get_client()
            standings_url = ESPN_ENDPOINTS[league]["standings"]
            
            for year in years:
                year_stats = []
                
                try:
                    url = f"{standings_url}?season={year}"
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Debug: log structure on first call
                        if not debug_logged:
                            import json
                            logger.info(f"[hoopR] DEBUG {league} standings keys: {list(data.keys())}")
                            debug_logged = True
                        
                        # Try multiple possible structures
                        entries = []
                        
                        # Structure 1: children -> standings -> entries
                        children = data.get("children", [])
                        for child in children:
                            conf_name = child.get("name", child.get("abbreviation", ""))
                            standings = child.get("standings", {})
                            child_entries = standings.get("entries", [])
                            for entry in child_entries:
                                entry["_conference"] = conf_name
                            entries.extend(child_entries)
                        
                        # Structure 2: standings.entries (dict)
                        if not entries:
                            standings = data.get("standings", {})
                            if isinstance(standings, dict):
                                entries = standings.get("entries", [])
                            elif isinstance(standings, list):
                                entries = standings
                        
                        # Structure 3: groups -> entries
                        if not entries:
                            groups = data.get("groups", [])
                            for group in groups:
                                conf_name = group.get("name", group.get("abbreviation", ""))
                                group_entries = group.get("entries", group.get("standings", {}).get("entries", []))
                                for entry in group_entries:
                                    entry["_conference"] = conf_name
                                entries.extend(group_entries)
                        
                        # Structure 4: leagues -> groups -> entries (NCAAB specific)
                        if not entries:
                            leagues = data.get("leagues", [])
                            for lg in leagues:
                                lg_groups = lg.get("groups", [])
                                for group in lg_groups:
                                    conf_name = group.get("name", group.get("abbreviation", ""))
                                    group_entries = group.get("entries", [])
                                    for entry in group_entries:
                                        entry["_conference"] = conf_name
                                    entries.extend(group_entries)
                        
                        # Structure 5: Direct entries at root
                        if not entries:
                            entries = data.get("entries", [])
                        
                        for entry in entries:
                            team = entry.get("team", {})
                            team_id = str(team.get("id", ""))
                            team_abbr = team.get("abbreviation", "")
                            
                            if not team_id and not team_abbr:
                                continue
                            
                            # Parse stats
                            stats_list = entry.get("stats", [])
                            stats_dict = {"wins": 0, "losses": 0}
                            
                            for stat in stats_list:
                                if isinstance(stat, dict):
                                    stat_name = stat.get("name", stat.get("abbreviation", ""))
                                    stat_value = stat.get("value", stat.get("displayValue", 0))
                                    
                                    if stat_name:
                                        clean_name = str(stat_name).lower().replace(" ", "_").replace("-", "_")
                                        try:
                                            stats_dict[clean_name] = float(stat_value) if stat_value else 0
                                        except:
                                            stats_dict[clean_name] = 0
                            
                            team_stat = {
                                "league": league,
                                "season": year,
                                "team_id": team_id,
                                "team_abbr": team_abbr,
                                "team_name": team.get("displayName", team.get("name", "")),
                                "conference": entry.get("_conference", ""),
                                **stats_dict
                            }
                            year_stats.append(team_stat)
                        
                        all_stats.extend(year_stats)
                        logger.info(f"[hoopR] {league} {year}: {len(year_stats)} team stats")
                    
                except Exception as e:
                    logger.warning(f"[hoopR] Error collecting {league} team stats for {year}: {e}")
                    continue
                
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"[hoopR] Error collecting {league} team stats: {e}")
        
        logger.info(f"[hoopR] {league}: Total {len(all_stats)} team stats")
        return all_stats

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Ensure sports exist
            for league in ["NBA", "NCAAB"]:
                sport = await self._ensure_sport(session, league)
                if sport:
                    logger.info(f"[hoopR] Sport '{league}' ready (ID: {sport.id})")
            
            # Save teams
            if data.get("teams"):
                saved = await self._save_teams(session, data["teams"])
                total_saved += saved
                logger.info(f"[hoopR] Saved {saved} teams")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(session, data["games"])
                total_saved += saved
                logger.info(f"[hoopR] Saved {saved} games")
            
            # Save rosters/players
            if data.get("rosters"):
                saved = await self._save_rosters(session, data["rosters"])
                total_saved += saved
                logger.info(f"[hoopR] Saved {saved} roster entries")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self._save_team_stats(session, data["team_stats"])
                total_saved += saved
                logger.info(f"[hoopR] Saved {saved} team stats")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[hoopR] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _ensure_sport(self, session: AsyncSession, sport_code: str) -> Optional[Sport]:
        """Ensure sport exists in database."""
        try:
            result = await session.execute(
                select(Sport).where(Sport.code == sport_code)
            )
            sport = result.scalar_one_or_none()
            
            if not sport:
                sport_names = {
                    "NBA": "National Basketball Association",
                    "NCAAB": "NCAA Men's Basketball",
                }
                sport = Sport(
                    code=sport_code,
                    name=sport_names.get(sport_code, sport_code),
                    is_active=True,
                    config={"collector": "hoopR", "source": "ESPN"}
                )
                session.add(sport)
                await session.flush()
                logger.info(f"[hoopR] Created sport: {sport_code}")
            
            return sport
            
        except Exception as e:
            logger.error(f"[hoopR] Error ensuring sport {sport_code}: {e}")
            return None
    
    async def _save_teams(self, session: AsyncSession, teams: List[Dict[str, Any]]) -> int:
        """Save teams to database with proper duplicate handling."""
        saved = 0
        updated = 0
        
        for team_data in teams:
            try:
                league = team_data.get("league", "NBA")
                
                # Get sport
                result = await session.execute(
                    select(Sport).where(Sport.code == league)
                )
                sport = result.scalar_one_or_none()
                if not sport:
                    continue
                
                external_id = team_data.get("external_id", "")
                team_name = team_data.get("name", "")
                
                # Check if team exists by external_id
                result = await session.execute(
                    select(Team).where(Team.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Also check by (sport_id, name) - the unique constraint
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
                    # Update existing team
                    existing.external_id = external_id  # Update external_id if found by name
                    existing.abbreviation = team_data.get("abbreviation", existing.abbreviation)
                    existing.city = team_data.get("city", existing.city)
                    existing.conference = team_data.get("conference", existing.conference)
                    existing.division = team_data.get("division", existing.division)
                    existing.logo_url = team_data.get("logo_url", existing.logo_url)
                    updated += 1
                else:
                    # Create new team
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_name,
                        abbreviation=team_data.get("abbreviation", ""),
                        city=team_data.get("city", ""),
                        conference=team_data.get("conference", ""),
                        division=team_data.get("division", ""),
                        logo_url=team_data.get("logo_url", ""),
                        is_active=team_data.get("is_active", True),
                    )
                    session.add(team)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[hoopR] Error saving team: {e}")
                continue
        
        await session.flush()
        logger.info(f"[hoopR] Teams: {saved} new, {updated} updated")
        return saved + updated
    
    async def _save_games(self, session: AsyncSession, games: List[Dict[str, Any]]) -> int:
        """Save games to database with proper duplicate handling."""
        saved = 0
        updated = 0
        skipped = 0
        
        for game_data in games:
            try:
                league = game_data.get("league", "NBA")
                
                # Get sport
                result = await session.execute(
                    select(Sport).where(Sport.code == league)
                )
                sport = result.scalar_one_or_none()
                if not sport:
                    continue
                
                # Find home team - try external_id first, then by name
                home_team_espn_id = f"espn_{league.lower()}_{game_data.get('home_team_id', '')}"
                home_result = await session.execute(
                    select(Team).where(Team.external_id == home_team_espn_id)
                )
                home_team = home_result.scalar_one_or_none()
                
                # Fallback: find by name
                if not home_team and game_data.get("home_team_name"):
                    home_result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == game_data.get("home_team_name")
                            )
                        )
                    )
                    home_team = home_result.scalar_one_or_none()
                
                # Find away team - try external_id first, then by name
                away_team_espn_id = f"espn_{league.lower()}_{game_data.get('away_team_id', '')}"
                away_result = await session.execute(
                    select(Team).where(Team.external_id == away_team_espn_id)
                )
                away_team = away_result.scalar_one_or_none()
                
                # Fallback: find by name
                if not away_team and game_data.get("away_team_name"):
                    away_result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == game_data.get("away_team_name")
                            )
                        )
                    )
                    away_team = away_result.scalar_one_or_none()
                
                if not home_team or not away_team:
                    skipped += 1
                    continue
                
                # Check if game exists
                external_id = game_data.get("external_id", "")
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
                logger.debug(f"[hoopR] Error saving game: {e}")
                continue
        
        await session.flush()
        logger.info(f"[hoopR] Games: {saved} new, {updated} updated, {skipped} skipped (no teams)")
        return saved + updated
    
    async def _save_rosters(self, session: AsyncSession, rosters: List[Dict[str, Any]]) -> int:
        """Save roster/player data to database."""
        saved = 0
        
        for roster_data in rosters:
            try:
                external_id = roster_data.get("external_id", "")
                
                # Check if player exists
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Find team
                league = roster_data.get("league", "NBA")
                team_espn_id = f"espn_{league.lower()}_{roster_data.get('team_id', '')}"
                
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_espn_id)
                )
                team = team_result.scalar_one_or_none()
                
                if existing:
                    # Update player info
                    existing.name = roster_data.get("player_name", existing.name)
                    existing.position = roster_data.get("position", existing.position)
                    if team:
                        existing.team_id = team.id
                    try:
                        existing.jersey_number = int(roster_data.get("jersey_number", 0) or 0)
                    except:
                        pass
                    existing.height = roster_data.get("height", existing.height)
                    try:
                        existing.weight = int(roster_data.get("weight", 0) or 0) if roster_data.get("weight") else None
                    except:
                        pass
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
                        player.jersey_number = int(roster_data.get("jersey_number", 0) or 0)
                    except:
                        pass
                    
                    try:
                        if roster_data.get("weight"):
                            player.weight = int(roster_data.get("weight", 0))
                    except:
                        pass
                    
                    session.add(player)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[hoopR] Error saving player: {e}")
                continue
        
        await session.flush()
        return saved
    
    async def _save_team_stats(self, session: AsyncSession, team_stats: List[Dict[str, Any]]) -> int:
        """Save team statistics to database."""
        saved = 0
        
        for stat_data in team_stats:
            try:
                league = stat_data.get("league", "NBA")
                team_espn_id = f"espn_{league.lower()}_{stat_data.get('team_id', '')}"
                
                # Find team
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_espn_id)
                )
                team = team_result.scalar_one_or_none()
                
                if not team:
                    continue
                
                season = stat_data.get("season", datetime.now().year)
                stat_type = f"{league}_{season}_standings"
                
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
                
                # Calculate value (use wins as primary stat)
                wins = stat_data.get("wins", stat_data.get("overall_wins", 0))
                
                if existing:
                    existing.value = float(wins)
                    existing.games_played = int(stat_data.get("games_played", wins + stat_data.get("losses", 0)))
                else:
                    team_stat = TeamStats(
                        team_id=team.id,
                        stat_type=stat_type,
                        value=float(wins),
                        games_played=int(stat_data.get("games_played", wins + stat_data.get("losses", 0))),
                    )
                    session.add(team_stat)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[hoopR] Error saving team stat: {e}")
                continue
        
        await session.flush()
        return saved

    # =========================================================================
    # VALIDATION METHOD (Required by BaseCollector)
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """Validate collected basketball data."""
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

hoopr_collector = HoopRCollector()

# Register with collector manager
collector_manager.register(hoopr_collector)
logger.info("Registered collector: hoopR")