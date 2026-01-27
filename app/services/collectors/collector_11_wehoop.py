"""
ROYALEY - wehoop Data Collector
Phase 1: Data Collection Services

Collects comprehensive WNBA data from ESPN APIs.
Features: Games, rosters, player stats, team stats, standings.

Data Sources:
- ESPN API: https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/

FREE data - no API key required!

Key Data Types:
- Teams: All 12+ WNBA teams with conferences
- Games: Full game schedules with results (2016-present)
- Rosters: Player rosters by season
- Player Stats: Points, rebounds, assists, etc.
- Team Stats: Wins, losses, standings data

Tables Filled:
- sports (WNBA entry)
- teams (12+ WNBA teams)
- games (10 years of games)
- players (all WNBA players)
- player_stats (seasonal stats)
- team_stats (seasonal stats)
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
# ESPN API CONFIGURATION
# =============================================================================

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_TEAMS = f"{ESPN_BASE}/teams"
ESPN_STANDINGS = f"{ESPN_BASE}/standings"

# WNBA Teams with ESPN IDs
WNBA_TEAMS = {
    # Eastern Conference
    "3": {"abbr": "ATL", "name": "Atlanta Dream", "city": "Atlanta", "conference": "Eastern"},
    "4": {"abbr": "CHI", "name": "Chicago Sky", "city": "Chicago", "conference": "Eastern"},
    "5": {"abbr": "CONN", "name": "Connecticut Sun", "city": "Uncasville", "conference": "Eastern"},
    "6": {"abbr": "IND", "name": "Indiana Fever", "city": "Indianapolis", "conference": "Eastern"},
    "9": {"abbr": "NYL", "name": "New York Liberty", "city": "Brooklyn", "conference": "Eastern"},
    "14": {"abbr": "WSH", "name": "Washington Mystics", "city": "Washington", "conference": "Eastern"},
    
    # Western Conference
    "16": {"abbr": "DAL", "name": "Dallas Wings", "city": "Arlington", "conference": "Western"},
    "8": {"abbr": "LVA", "name": "Las Vegas Aces", "city": "Las Vegas", "conference": "Western"},
    "17": {"abbr": "LA", "name": "Los Angeles Sparks", "city": "Los Angeles", "conference": "Western"},
    "11": {"abbr": "MIN", "name": "Minnesota Lynx", "city": "Minneapolis", "conference": "Western"},
    "18": {"abbr": "PHO", "name": "Phoenix Mercury", "city": "Phoenix", "conference": "Western"},
    "19": {"abbr": "SEA", "name": "Seattle Storm", "city": "Seattle", "conference": "Western"},
    
    # Expansion (2025)
    "20": {"abbr": "GS", "name": "Golden State Valkyries", "city": "San Francisco", "conference": "Western"},
}


# =============================================================================
# WEHOOP COLLECTOR CLASS
# =============================================================================

class WehoopCollector(BaseCollector):
    """
    Collector for WNBA data using ESPN API.
    
    Features:
    - Game schedules and results (2016-present)
    - Team information and standings
    - Player rosters and statistics
    - Team statistics
    - No API key required (FREE)
    """
    
    def __init__(self):
        super().__init__(
            name="wehoop",
            base_url=ESPN_BASE,
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
                if datetime.now().month >= 5:
                    years = [current_year]
                else:
                    years = [current_year - 1]
            
            logger.info(f"[wehoop] Collecting WNBA data for seasons: {years}")
            
            # Collect teams FIRST (needed for other collections)
            if collect_type in ["all", "teams"]:
                teams = await self._collect_teams()
                data["teams"] = teams
                total_records += len(teams)
                logger.info(f"[wehoop] Collected {len(teams)} teams")
            
            # Collect games using team schedules (more complete than scoreboard)
            if collect_type in ["all", "games"]:
                games = await self._collect_games_via_schedules(years)
                data["games"] = games
                total_records += len(games)
                logger.info(f"[wehoop] Collected {len(games)} games")
            
            # Collect rosters
            if collect_type in ["all", "rosters"]:
                rosters = await self._collect_rosters(years)
                data["rosters"] = rosters
                total_records += len(rosters)
                logger.info(f"[wehoop] Collected {len(rosters)} roster entries")
            
            # Collect player stats from rosters with stats
            if collect_type in ["all", "player_stats"]:
                player_stats = await self._collect_player_stats(years)
                data["player_stats"] = player_stats
                total_records += len(player_stats)
                logger.info(f"[wehoop] Collected {len(player_stats)} player stats")
            
            # Collect team stats from standings
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
            import traceback
            traceback.print_exc()
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
                
                # ESPN returns nested structure
                sports = data.get("sports", [])
                if sports:
                    leagues = sports[0].get("leagues", [])
                    if leagues:
                        espn_teams = leagues[0].get("teams", [])
                        
                        for team_data in espn_teams:
                            team = team_data.get("team", {})
                            team_id = str(team.get("id", ""))
                            abbr = team.get("abbreviation", "")
                            name = team.get("displayName", "")
                            location = team.get("location", "")
                            
                            # Get additional info from our mapping
                            mapped_info = WNBA_TEAMS.get(team_id, {})
                            
                            teams.append({
                                "espn_id": team_id,
                                "abbr": abbr or mapped_info.get("abbr", ""),
                                "name": name or mapped_info.get("name", ""),
                                "city": location or mapped_info.get("city", ""),
                                "conference": mapped_info.get("conference", ""),
                                "logo_url": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                            })
                        
                        logger.info(f"[wehoop] Loaded {len(teams)} WNBA teams from ESPN")
            
            # Fallback to static mapping if ESPN fails
            if not teams:
                logger.warning("[wehoop] ESPN teams API failed, using static mapping")
                for team_id, info in WNBA_TEAMS.items():
                    teams.append({
                        "espn_id": team_id,
                        "abbr": info["abbr"],
                        "name": info["name"],
                        "city": info["city"],
                        "conference": info["conference"],
                        "logo_url": "",
                    })
        
        except Exception as e:
            logger.error(f"[wehoop] Error collecting teams: {e}")
            # Use static mapping as fallback
            for team_id, info in WNBA_TEAMS.items():
                teams.append({
                    "espn_id": team_id,
                    "abbr": info["abbr"],
                    "name": info["name"],
                    "city": info["city"],
                    "conference": info["conference"],
                    "logo_url": "",
                })
        
        return teams

    # =========================================================================
    # GAMES COLLECTION (via Team Schedules - MORE COMPLETE)
    # =========================================================================
    
    async def _collect_games_via_schedules(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect games using team schedule endpoints (more complete than scoreboard)."""
        all_games = {}  # Use dict to dedupe by game_id
        
        try:
            client = await self.get_client()
            
            for year in years:
                logger.info(f"[wehoop] Collecting {year} season games...")
                
                # Get schedule for each team
                for team_id in WNBA_TEAMS.keys():
                    try:
                        url = f"{ESPN_BASE}/teams/{team_id}/schedule?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            events = data.get("events", [])
                            
                            for event in events:
                                game = self._parse_schedule_event(event, year)
                                if game and game["game_id"] not in all_games:
                                    all_games[game["game_id"]] = game
                        
                        await asyncio.sleep(0.05)  # Rate limiting
                        
                    except Exception as e:
                        logger.debug(f"[wehoop] Error getting schedule for team {team_id}: {e}")
                        continue
                
                logger.info(f"[wehoop] {year}: {len([g for g in all_games.values() if g.get('season') == year])} games so far")
            
            games_list = list(all_games.values())
            logger.info(f"[wehoop] Total {len(games_list)} unique games collected")
            return games_list
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting games: {e}")
            return list(all_games.values())
    
    def _parse_schedule_event(self, event: Dict, year: int) -> Optional[Dict[str, Any]]:
        """Parse ESPN schedule event into game data."""
        try:
            game_id = event.get("id")
            if not game_id:
                return None
            
            competitions = event.get("competitions", [])
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
                if scheduled_at.tzinfo is not None:
                    scheduled_at = scheduled_at.replace(tzinfo=None)
            except:
                return None
            
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
            home_score = None
            away_score = None
            if status == "final":
                try:
                    home_score = int(home_team.get("score", {}).get("value", 0) or home_team.get("score", 0) or 0)
                    away_score = int(away_team.get("score", {}).get("value", 0) or away_team.get("score", 0) or 0)
                except:
                    home_score = None
                    away_score = None
            
            # Get team info
            home_team_data = home_team.get("team", {})
            away_team_data = away_team.get("team", {})
            
            return {
                "game_id": game_id,
                "season": year,
                "scheduled_at": scheduled_at.isoformat(),
                "status": status,
                "home_team_id": home_team_data.get("id"),
                "home_team_abbr": home_team_data.get("abbreviation", ""),
                "home_team_name": home_team_data.get("displayName", ""),
                "away_team_id": away_team_data.get("id"),
                "away_team_abbr": away_team_data.get("abbreviation", ""),
                "away_team_name": away_team_data.get("displayName", ""),
                "home_score": home_score,
                "away_score": away_score,
            }
        
        except Exception as e:
            logger.debug(f"[wehoop] Error parsing game: {e}")
            return None

    # =========================================================================
    # ROSTERS COLLECTION (FIXED)
    # =========================================================================
    
    async def _collect_rosters(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect rosters for all teams across specified seasons."""
        all_rosters = []
        seen_players = set()  # Track unique players
        
        try:
            client = await self.get_client()
            
            for year in years:
                year_count = 0
                
                for team_id, team_info in WNBA_TEAMS.items():
                    try:
                        url = f"{ESPN_BASE}/teams/{team_id}/roster?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # ESPN returns athletes in groups (by position)
                            athletes_groups = data.get("athletes", [])
                            
                            for group in athletes_groups:
                                # Each group has "items" array
                                items = group.get("items", [])
                                
                                for player in items:
                                    player_id = player.get("id")
                                    if not player_id:
                                        continue
                                    
                                    # Create unique key for deduplication
                                    unique_key = f"{player_id}_{year}"
                                    if unique_key in seen_players:
                                        continue
                                    seen_players.add(unique_key)
                                    
                                    roster_entry = {
                                        "season": year,
                                        "team_id": team_id,
                                        "team_abbr": team_info["abbr"],
                                        "player_id": player_id,
                                        "player_name": player.get("displayName", ""),
                                        "position": player.get("position", {}).get("abbreviation", "") if isinstance(player.get("position"), dict) else player.get("position", ""),
                                        "jersey_number": player.get("jersey", ""),
                                        "height": player.get("displayHeight", ""),
                                        "weight": player.get("displayWeight", ""),
                                        "birth_date": player.get("dateOfBirth", ""),
                                        "age": player.get("age", 0),
                                        "experience": player.get("experience", {}).get("years", 0) if isinstance(player.get("experience"), dict) else 0,
                                    }
                                    all_rosters.append(roster_entry)
                                    year_count += 1
                        
                        await asyncio.sleep(0.05)  # Rate limiting
                        
                    except Exception as e:
                        logger.debug(f"[wehoop] Error collecting roster for {team_info['abbr']} {year}: {e}")
                        continue
                
                logger.info(f"[wehoop] {year}: {year_count} roster entries")
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting rosters: {e}")
        
        logger.info(f"[wehoop] Total {len(all_rosters)} roster entries collected")
        return all_rosters

    # =========================================================================
    # PLAYER STATS COLLECTION (from roster stats endpoint)
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics for specified seasons."""
        all_stats = []
        seen_stats = set()
        
        try:
            client = await self.get_client()
            
            for year in years:
                year_count = 0
                
                for team_id, team_info in WNBA_TEAMS.items():
                    try:
                        # Get roster with statistics
                        url = f"{ESPN_BASE}/teams/{team_id}/roster?season={year}"
                        response = await client.get(url, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Get athletes from all groups
                            for group in data.get("athletes", []):
                                for player in group.get("items", []):
                                    player_id = player.get("id")
                                    if not player_id:
                                        continue
                                    
                                    unique_key = f"{player_id}_{year}"
                                    if unique_key in seen_stats:
                                        continue
                                    seen_stats.add(unique_key)
                                    
                                    # Try to get player statistics
                                    stats = player.get("statistics", {})
                                    if not stats:
                                        # Try alternate stats location
                                        stats = player.get("stats", {})
                                    
                                    # Also try to get from player stats endpoint
                                    player_stats = await self._get_player_stats_detail(client, player_id, year)
                                    
                                    stat_record = {
                                        "player_id": player_id,
                                        "player_name": player.get("displayName", ""),
                                        "team_abbr": team_info["abbr"],
                                        "season": year,
                                        **player_stats
                                    }
                                    
                                    # Only add if we have some stats
                                    if any(v for k, v in stat_record.items() if k not in ["player_id", "player_name", "team_abbr", "season"]):
                                        all_stats.append(stat_record)
                                        year_count += 1
                        
                        await asyncio.sleep(0.05)
                        
                    except Exception as e:
                        logger.debug(f"[wehoop] Error collecting stats for team {team_id}: {e}")
                        continue
                
                logger.info(f"[wehoop] {year}: {year_count} player stats records")
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting player stats: {e}")
        
        logger.info(f"[wehoop] Total {len(all_stats)} player stats records")
        return all_stats
    
    async def _get_player_stats_detail(
        self, 
        client: httpx.AsyncClient, 
        player_id: str, 
        year: int
    ) -> Dict[str, Any]:
        """Get detailed stats for a player from the statistics endpoint."""
        stats = {}
        
        try:
            url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/wnba/athletes/{player_id}/stats?season={year}"
            response = await client.get(url, timeout=15.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse splits/categories
                splits = data.get("splits", {})
                categories = splits.get("categories", [])
                
                for category in categories:
                    cat_stats = category.get("stats", [])
                    for stat in cat_stats:
                        stat_name = stat.get("name", "").lower().replace(" ", "_").replace("/", "_per_")
                        stat_value = stat.get("value", 0)
                        if stat_name and stat_value is not None:
                            stats[stat_name] = stat_value
                
        except Exception as e:
            logger.debug(f"[wehoop] Could not get detailed stats for player {player_id}: {e}")
        
        return stats

    # =========================================================================
    # TEAM STATS COLLECTION (from Standings - FIXED)
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect team statistics from standings endpoint."""
        all_stats = []
        
        try:
            client = await self.get_client()
            
            for year in years:
                try:
                    url = f"{ESPN_STANDINGS}?season={year}"
                    response = await client.get(url, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Parse standings structure: children -> standings -> entries
                        children = data.get("children", [])
                        
                        for conference in children:
                            conf_name = conference.get("name", "")
                            
                            standings = conference.get("standings", {})
                            entries = standings.get("entries", [])
                            
                            for entry in entries:
                                team = entry.get("team", {})
                                team_id = str(team.get("id", ""))
                                team_abbr = team.get("abbreviation", "")
                                
                                # Parse all stats
                                stats_list = entry.get("stats", [])
                                stats_dict = {}
                                
                                for stat in stats_list:
                                    stat_name = stat.get("name", "")
                                    stat_value = stat.get("value", 0)
                                    stat_display = stat.get("displayValue", "")
                                    
                                    if stat_name:
                                        # Clean stat name
                                        clean_name = stat_name.lower().replace(" ", "_").replace("-", "_")
                                        stats_dict[clean_name] = stat_value
                                        if stat_display:
                                            stats_dict[f"{clean_name}_display"] = stat_display
                                
                                team_stats = {
                                    "team_id": team_id,
                                    "team_abbr": team_abbr,
                                    "team_name": team.get("displayName", ""),
                                    "season": year,
                                    "conference": conf_name,
                                    **stats_dict
                                }
                                all_stats.append(team_stats)
                    
                    logger.info(f"[wehoop] {year}: Collected team stats for {len([s for s in all_stats if s['season'] == year])} teams")
                    
                except Exception as e:
                    logger.warning(f"[wehoop] Error collecting team stats for {year}: {e}")
                    continue
                
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"[wehoop] Error collecting team stats: {e}")
        
        logger.info(f"[wehoop] Total {len(all_stats)} team stats records")
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
            
            # Get all teams for this sport - build lookup by abbreviation AND name
            teams_result = await session.execute(
                select(Team).where(Team.sport_id == sport.id)
            )
            teams_by_abbr = {}
            teams_by_name = {}
            for t in teams_result.scalars().all():
                teams_by_abbr[t.abbreviation] = t
                teams_by_name[t.name] = t
            
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
                    
                    # Get teams by abbreviation or name
                    home_abbr = game_data.get("home_team_abbr", "")
                    away_abbr = game_data.get("away_team_abbr", "")
                    home_name = game_data.get("home_team_name", "")
                    away_name = game_data.get("away_team_name", "")
                    
                    home_team = teams_by_abbr.get(home_abbr) or teams_by_name.get(home_name)
                    away_team = teams_by_abbr.get(away_abbr) or teams_by_name.get(away_name)
                    
                    if not home_team or not away_team:
                        logger.debug(f"[wehoop] Teams not found: {home_abbr}/{home_name} vs {away_abbr}/{away_name}")
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
                    logger.debug(f"[wehoop] Error processing game {game_data.get('game_id')}: {e}")
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
                    
                    # Parse jersey number
                    jersey = roster_entry.get("jersey_number", "")
                    jersey_num = None
                    if jersey:
                        try:
                            jersey_num = int(jersey)
                        except:
                            pass
                    
                    # Parse weight
                    weight_str = roster_entry.get("weight", "")
                    weight = None
                    if weight_str:
                        try:
                            # Format might be "150 lbs" or just "150"
                            weight = int(str(weight_str).replace("lbs", "").strip().split()[0])
                        except:
                            pass
                    
                    player = Player(
                        external_id=external_id,
                        team_id=team.id if team else None,
                        name=roster_entry.get("player_name", ""),
                        position=roster_entry.get("position", ""),
                        jersey_number=jersey_num,
                        height=roster_entry.get("height", ""),
                        weight=weight,
                        birth_date=birth_date,
                        is_active=True,
                    )
                    new_players.append(player)
                    existing_ids.add(external_id)
                    saved_count += 1
                    
                except Exception as e:
                    logger.debug(f"[wehoop] Error saving player {roster_entry.get('player_name')}: {e}")
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
                    pid = p.external_id.replace("wnba_player_", "")
                    players[pid] = p
            
            for stats in player_stats:
                try:
                    player_id = str(stats.get("player_id", ""))
                    player = players.get(player_id)
                    
                    if not player:
                        continue
                    
                    season = stats.get("season")
                    stat_type = f"wnba_season_{season}"
                    
                    # Check if stats already exist
                    existing = await session.execute(
                        select(PlayerStats).where(
                            and_(
                                PlayerStats.player_id == player.id,
                                PlayerStats.stat_type == stat_type
                            )
                        )
                    )
                    existing_stats = existing.scalars().first()
                    
                    # Get numeric value (points if available)
                    value = stats.get("points", stats.get("pts", 0)) or 0
                    games = stats.get("games_played", stats.get("gp", 0)) or 0
                    
                    if existing_stats:
                        existing_stats.value = float(value)
                        existing_stats.games_played = int(games)
                    else:
                        player_stat = PlayerStats(
                            player_id=player.id,
                            stat_type=stat_type,
                            value=float(value),
                            games_played=int(games),
                        )
                        session.add(player_stat)
                        saved_count += 1
                    
                except Exception as e:
                    logger.debug(f"[wehoop] Error saving player stats: {e}")
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
            teams_by_abbr = {t.abbreviation: t for t in teams_result.scalars().all()}
            teams_by_name = {t.name: t for t in teams_result.scalars().all()}
            
            for stats in team_stats:
                try:
                    team_abbr = stats.get("team_abbr", "")
                    team_name = stats.get("team_name", "")
                    team = teams_by_abbr.get(team_abbr) or teams_by_name.get(team_name)
                    
                    if not team:
                        continue
                    
                    season = stats.get("season")
                    stat_type = f"wnba_season_{season}"
                    
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
                    
                    # Get win percentage
                    wins = stats.get("wins", 0) or 0
                    losses = stats.get("losses", 0) or 0
                    games_played = wins + losses
                    win_pct = wins / games_played if games_played > 0 else 0
                    
                    if existing_stats:
                        existing_stats.value = win_pct
                        existing_stats.games_played = games_played
                    else:
                        team_stat = TeamStats(
                            team_id=team.id,
                            stat_type=stat_type,
                            value=win_pct,
                            games_played=games_played,
                        )
                        session.add(team_stat)
                        saved_count += 1
                    
                except Exception as e:
                    logger.debug(f"[wehoop] Error saving team stats: {e}")
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