"""
ROYALEY - NHL Official API Collector
Phase 1: Data Collection Services

Collects NHL EDGE tracking stats from the Official NHL API.
Supplements hockeyR with cutting-edge puck and player tracking metrics.

Data Sources:
- NHL Web API: https://api-web.nhle.com/v1/
- NHL EDGE Endpoints: /v1/edge/...

Key Data Types:
- Shot Speed (max, average) per player
- Skating Speed (max, average, bursts) per player
- Distance Traveled per player/team
- Zone Time Percentages (OZ, DZ, NZ)
- Shot Location Data
- Goalie Save Location Data

FREE data - no API key required!

Tables Filled (using existing tables):
- players - NHL player records
- player_stats - EDGE stats (shot_speed_max, skating_speed_max, etc.)
- team_stats - Team EDGE metrics
- teams - NHL team records (via hockeyR)
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
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

# NHL Teams with IDs
NHL_TEAMS = {
    1: {"abbr": "NJD", "name": "New Jersey Devils"},
    2: {"abbr": "NYI", "name": "New York Islanders"},
    3: {"abbr": "NYR", "name": "New York Rangers"},
    4: {"abbr": "PHI", "name": "Philadelphia Flyers"},
    5: {"abbr": "PIT", "name": "Pittsburgh Penguins"},
    6: {"abbr": "BOS", "name": "Boston Bruins"},
    7: {"abbr": "BUF", "name": "Buffalo Sabres"},
    8: {"abbr": "MTL", "name": "Montréal Canadiens"},
    9: {"abbr": "OTT", "name": "Ottawa Senators"},
    10: {"abbr": "TOR", "name": "Toronto Maple Leafs"},
    12: {"abbr": "CAR", "name": "Carolina Hurricanes"},
    13: {"abbr": "FLA", "name": "Florida Panthers"},
    14: {"abbr": "TBL", "name": "Tampa Bay Lightning"},
    15: {"abbr": "WSH", "name": "Washington Capitals"},
    16: {"abbr": "CHI", "name": "Chicago Blackhawks"},
    17: {"abbr": "DET", "name": "Detroit Red Wings"},
    18: {"abbr": "NSH", "name": "Nashville Predators"},
    19: {"abbr": "STL", "name": "St. Louis Blues"},
    20: {"abbr": "CGY", "name": "Calgary Flames"},
    21: {"abbr": "COL", "name": "Colorado Avalanche"},
    22: {"abbr": "EDM", "name": "Edmonton Oilers"},
    23: {"abbr": "VAN", "name": "Vancouver Canucks"},
    24: {"abbr": "ANA", "name": "Anaheim Ducks"},
    25: {"abbr": "DAL", "name": "Dallas Stars"},
    26: {"abbr": "LAK", "name": "Los Angeles Kings"},
    28: {"abbr": "SJS", "name": "San Jose Sharks"},
    29: {"abbr": "CBJ", "name": "Columbus Blue Jackets"},
    30: {"abbr": "MIN", "name": "Minnesota Wild"},
    52: {"abbr": "WPG", "name": "Winnipeg Jets"},
    53: {"abbr": "ARI", "name": "Arizona Coyotes"},
    54: {"abbr": "VGK", "name": "Vegas Golden Knights"},
    55: {"abbr": "SEA", "name": "Seattle Kraken"},
    59: {"abbr": "UTA", "name": "Utah Hockey Club"},
}

ABBR_TO_ID = {info["abbr"]: team_id for team_id, info in NHL_TEAMS.items()}

# EDGE stats categories
EDGE_SKATER_POSITIONS = ["F", "D"]  # Forwards, Defensemen
EDGE_SORT_OPTIONS = {
    "shot_speed": ["maxShotSpeed", "avgShotSpeed"],
    "skating_speed": ["maxSkatingSpeed", "avgSkatingSpeed"],
    "distance": ["distanceTotal", "distancePerGame"],
    "zone_time": ["offensiveZonePct", "defensiveZonePct"],
}


# =============================================================================
# NHL OFFICIAL API COLLECTOR CLASS
# =============================================================================

class NHLOfficialAPICollector(BaseCollector):
    """
    NHL Official API collector for EDGE tracking stats.
    
    Collects:
    - Skater EDGE stats (shot speed, skating speed, distance, zone time)
    - Goalie EDGE stats (save locations, 5v5 performance)
    - Team EDGE stats (aggregate metrics)
    - Leaders/rankings for EDGE categories
    
    Features:
    - 10+ years of historical data (EDGE available from ~2021)
    - Comprehensive tracking metrics
    - Supplements hockeyR with cutting-edge data
    """
    
    def __init__(self):
        super().__init__(
            name="nhl_official_api",
            base_url=NHL_WEB_API,
            rate_limit=60,
            rate_window=60,
            timeout=60.0,
            max_retries=3,
        )
        self._custom_client: Optional[httpx.AsyncClient] = None
        
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
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
        years_back: int = 10,
        collect_type: str = "all",
        game_type: int = 2  # 2=regular, 3=playoffs
    ) -> CollectorResult:
        """
        Collect NHL EDGE data.
        
        Args:
            years_back: Number of years to collect (default: 10)
            collect_type: Type of data:
                - "all": All EDGE data
                - "skaters": Skater EDGE stats only
                - "goalies": Goalie EDGE stats only
                - "teams": Team EDGE stats only
                - "leaders": EDGE leaders only
            game_type: 2=regular season, 3=playoffs
        
        Returns:
            CollectorResult with collected data
        """
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NHL season format: 20232024 for 2023-24 season
        # Season starts in October, so if month < 10, we're in previous season
        if current_month < 10:
            end_season_year = current_year
        else:
            end_season_year = current_year + 1
        
        # Generate season list (EDGE data available from ~2021)
        seasons = []
        for i in range(years_back):
            season_end = end_season_year - i
            season_start = season_end - 1
            # EDGE tracking started ~2021
            if season_start >= 2015:  # Go back to 2015-16 for basic stats
                seasons.append(f"{season_start}{season_end}")
        
        logger.info(f"[NHL API] Collecting EDGE data for seasons: {seasons[:5]}... ({len(seasons)} total)")
        logger.info(f"[NHL API] Collection type: {collect_type}, Game type: {game_type}")
        
        data = {
            "skater_stats": [],
            "goalie_stats": [],
            "team_stats": [],
            "leaders": [],
            "game_tracking": [],
        }
        total_records = 0
        
        try:
            client = await self.get_client()
            
            for season in seasons:
                logger.info(f"[NHL API] Processing season {season}...")
                
                try:
                    # Collect skater EDGE stats
                    if collect_type in ["all", "skaters"]:
                        skaters = await self._collect_skater_edge_stats(client, season, game_type)
                        data["skater_stats"].extend(skaters)
                        total_records += len(skaters)
                        logger.info(f"[NHL API] {season}: {len(skaters)} skater EDGE records")
                    
                    # Collect goalie EDGE stats
                    if collect_type in ["all", "goalies"]:
                        goalies = await self._collect_goalie_edge_stats(client, season, game_type)
                        data["goalie_stats"].extend(goalies)
                        total_records += len(goalies)
                        logger.info(f"[NHL API] {season}: {len(goalies)} goalie EDGE records")
                    
                    # Collect team EDGE stats
                    if collect_type in ["all", "teams"]:
                        teams = await self._collect_team_edge_stats(client, season, game_type)
                        data["team_stats"].extend(teams)
                        total_records += len(teams)
                        logger.info(f"[NHL API] {season}: {len(teams)} team EDGE records")
                    
                    # Collect EDGE leaders
                    if collect_type in ["all", "leaders"]:
                        leaders = await self._collect_edge_leaders(client, season, game_type)
                        data["leaders"].extend(leaders)
                        total_records += len(leaders)
                        logger.info(f"[NHL API] {season}: {len(leaders)} leader records")
                    
                    # Rate limiting between seasons
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"[NHL API] Error collecting season {season}: {e}")
                    continue
            
            logger.info(f"[NHL API] Total records collected: {total_records}")
            
            return CollectorResult(
                success=True,
                data=data,
                records_count=total_records,
            )
            
        except Exception as e:
            logger.error(f"[NHL API] Collection error: {e}")
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
    # VALIDATE METHOD (Required by BaseCollector)
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """
        Validate collected NHL EDGE data.
        
        Args:
            data: CollectorResult or dict with collected data
            
        Returns:
            True if data is valid
        """
        if data is None:
            return False
        
        # If it's a CollectorResult, check success
        if hasattr(data, 'success'):
            return data.success
        
        # If it's a dict, check for expected keys
        if isinstance(data, dict):
            expected_keys = ["skater_stats", "goalie_stats", "team_stats"]
            has_data = any(data.get(key) for key in expected_keys)
            return has_data
        
        return False

    # =========================================================================
    # SKATER EDGE STATS
    # =========================================================================
    
    async def _collect_skater_edge_stats(
        self,
        client: httpx.AsyncClient,
        season: str,
        game_type: int
    ) -> List[Dict[str, Any]]:
        """Collect skater EDGE stats for a season."""
        skaters = []
        
        # Get player list from stats API
        try:
            # Get all skaters for the season
            stats_url = f"{NHL_STATS_API}/skater/summary?limit=-1&cayenneExp=seasonId={season} and gameTypeId={game_type}"
            response = await client.get(stats_url)
            
            if response.status_code != 200:
                logger.warning(f"[NHL API] Failed to get skater list: {response.status_code}")
                return skaters
            
            player_data = response.json()
            player_list = player_data.get("data", [])
            
            logger.debug(f"[NHL API] Found {len(player_list)} skaters for {season}")
            
            # Process each player (limit to avoid excessive API calls)
            for idx, player in enumerate(player_list[:500]):  # Top 500 players
                player_id = player.get("playerId")
                if not player_id:
                    continue
                
                try:
                    # Get EDGE detail for player
                    edge_data = await self._get_skater_edge_detail(client, player_id, season, game_type)
                    
                    if edge_data:
                        edge_data["player_name"] = player.get("skaterFullName", "Unknown")
                        edge_data["team_abbr"] = player.get("teamAbbrevs", "")
                        edge_data["position"] = player.get("positionCode", "")
                        edge_data["games_played"] = player.get("gamesPlayed", 0)
                        skaters.append(edge_data)
                    
                    # Rate limiting
                    if idx % 50 == 0 and idx > 0:
                        await asyncio.sleep(1)
                        logger.debug(f"[NHL API] Processed {idx}/{len(player_list[:500])} skaters")
                        
                except Exception as e:
                    logger.debug(f"[NHL API] Error getting EDGE for player {player_id}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"[NHL API] Error collecting skater EDGE stats: {e}")
        
        return skaters
    
    async def _get_skater_edge_detail(
        self,
        client: httpx.AsyncClient,
        player_id: int,
        season: str,
        game_type: int
    ) -> Optional[Dict[str, Any]]:
        """Get detailed EDGE stats for a skater."""
        edge_data = {
            "nhl_player_id": player_id,
            "season": season,
            "game_type": game_type,
        }
        
        # Get shot speed detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-shot-speed-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["shot_speed_max"] = data.get("maxShotSpeed")
                edge_data["shot_speed_avg"] = data.get("avgShotSpeed")
                edge_data["total_shots"] = data.get("totalShots")
                edge_data["shots_on_goal"] = data.get("shotsOnGoal")
        except:
            pass
        
        # Get skating speed detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-skating-speed-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["skating_speed_max"] = data.get("maxSkatingSpeed")
                edge_data["skating_speed_avg"] = data.get("avgSkatingSpeed")
                edge_data["speed_bursts_22plus"] = data.get("speedBursts22Plus")
                edge_data["speed_bursts_20_22"] = data.get("speedBursts20To22")
                edge_data["speed_bursts_18_20"] = data.get("speedBursts18To20")
        except:
            pass
        
        # Get skating distance detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-skating-distance-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["distance_total"] = data.get("distanceTotal")
                edge_data["distance_per_game"] = data.get("distancePerGame")
        except:
            pass
        
        # Get zone time (from skater detail)
        try:
            url = f"{NHL_WEB_API}/edge/skater-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                zone_time = data.get("zoneTime", {})
                edge_data["offensive_zone_pct"] = zone_time.get("offensiveZonePct")
                edge_data["defensive_zone_pct"] = zone_time.get("defensiveZonePct")
                edge_data["neutral_zone_pct"] = zone_time.get("neutralZonePct")
                edge_data["offensive_zone_time"] = zone_time.get("offensiveZoneTime")
                edge_data["defensive_zone_time"] = zone_time.get("defensiveZoneTime")
                edge_data["neutral_zone_time"] = zone_time.get("neutralZoneTime")
                edge_data["raw_data"] = data
        except:
            pass
        
        # Get shot location detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-shot-location-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                shots = data.get("shotsByZone", {})
                edge_data["shots_low"] = shots.get("low", {}).get("shots")
                edge_data["shots_mid"] = shots.get("mid", {}).get("shots")
                edge_data["shots_high"] = shots.get("high", {}).get("shots")
                edge_data["goals_low"] = shots.get("low", {}).get("goals")
                edge_data["goals_mid"] = shots.get("mid", {}).get("goals")
                edge_data["goals_high"] = shots.get("high", {}).get("goals")
        except:
            pass
        
        # Only return if we have some EDGE data
        if any(edge_data.get(k) for k in ["shot_speed_max", "skating_speed_max", "distance_total"]):
            return edge_data
        
        return None

    # =========================================================================
    # GOALIE EDGE STATS
    # =========================================================================
    
    async def _collect_goalie_edge_stats(
        self,
        client: httpx.AsyncClient,
        season: str,
        game_type: int
    ) -> List[Dict[str, Any]]:
        """Collect goalie EDGE stats for a season."""
        goalies = []
        
        try:
            # Get all goalies for the season
            stats_url = f"{NHL_STATS_API}/goalie/summary?limit=-1&cayenneExp=seasonId={season} and gameTypeId={game_type}"
            response = await client.get(stats_url)
            
            if response.status_code != 200:
                return goalies
            
            goalie_data = response.json()
            goalie_list = goalie_data.get("data", [])
            
            for goalie in goalie_list[:100]:  # Top 100 goalies
                player_id = goalie.get("playerId")
                if not player_id:
                    continue
                
                try:
                    edge_data = await self._get_goalie_edge_detail(client, player_id, season, game_type)
                    
                    if edge_data:
                        edge_data["player_name"] = goalie.get("goalieFullName", "Unknown")
                        edge_data["team_abbr"] = goalie.get("teamAbbrevs", "")
                        edge_data["games_played"] = goalie.get("gamesPlayed", 0)
                        edge_data["save_pct"] = goalie.get("savePct")
                        edge_data["goals_against_avg"] = goalie.get("goalsAgainstAverage")
                        goalies.append(edge_data)
                        
                except Exception as e:
                    logger.debug(f"[NHL API] Error getting goalie EDGE for {player_id}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"[NHL API] Error collecting goalie EDGE stats: {e}")
        
        return goalies
    
    async def _get_goalie_edge_detail(
        self,
        client: httpx.AsyncClient,
        player_id: int,
        season: str,
        game_type: int
    ) -> Optional[Dict[str, Any]]:
        """Get detailed EDGE stats for a goalie."""
        edge_data = {
            "nhl_player_id": player_id,
            "season": season,
            "game_type": game_type,
        }
        
        # Get goalie shot location detail
        try:
            url = f"{NHL_WEB_API}/edge/goalie-shot-location-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Save locations
                saves = data.get("savesByZone", {})
                edge_data["saves_low"] = saves.get("low", {}).get("saves")
                edge_data["saves_mid"] = saves.get("mid", {}).get("saves")
                edge_data["saves_high"] = saves.get("high", {}).get("saves")
                edge_data["save_pct_low"] = saves.get("low", {}).get("savePct")
                edge_data["save_pct_mid"] = saves.get("mid", {}).get("savePct")
                edge_data["save_pct_high"] = saves.get("high", {}).get("savePct")
                
                # Shots against
                shots = data.get("shotsByZone", {})
                edge_data["shots_against_low"] = shots.get("low", {}).get("shots")
                edge_data["shots_against_mid"] = shots.get("mid", {}).get("shots")
                edge_data["shots_against_high"] = shots.get("high", {}).get("shots")
                
                # Goals against
                edge_data["goals_against_low"] = shots.get("low", {}).get("goals")
                edge_data["goals_against_mid"] = shots.get("mid", {}).get("goals")
                edge_data["goals_against_high"] = shots.get("high", {}).get("goals")
                
                edge_data["raw_data"] = data
        except:
            pass
        
        # Get goalie 5v5 detail
        try:
            url = f"{NHL_WEB_API}/edge/goalie-5v5-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["save_pct_5v5"] = data.get("savePct5v5")
                edge_data["goals_against_5v5"] = data.get("goalsAgainst5v5")
                edge_data["shots_against_5v5"] = data.get("shotsAgainst5v5")
        except:
            pass
        
        # Get high danger stats
        try:
            url = f"{NHL_WEB_API}/edge/goalie-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                hd = data.get("highDanger", {})
                edge_data["high_danger_saves"] = hd.get("saves")
                edge_data["high_danger_save_pct"] = hd.get("savePct")
                edge_data["high_danger_goals_against"] = hd.get("goalsAgainst")
                edge_data["total_saves"] = data.get("totalSaves")
                edge_data["total_shots_against"] = data.get("totalShotsAgainst")
        except:
            pass
        
        # Only return if we have some EDGE data
        if any(edge_data.get(k) for k in ["saves_low", "save_pct_5v5", "high_danger_saves"]):
            return edge_data
        
        return None

    # =========================================================================
    # TEAM EDGE STATS
    # =========================================================================
    
    async def _collect_team_edge_stats(
        self,
        client: httpx.AsyncClient,
        season: str,
        game_type: int
    ) -> List[Dict[str, Any]]:
        """Collect team EDGE stats for a season."""
        teams = []
        
        for team_id, team_info in NHL_TEAMS.items():
            try:
                edge_data = await self._get_team_edge_detail(client, team_id, season, game_type)
                
                if edge_data:
                    edge_data["nhl_team_id"] = team_id
                    edge_data["team_name"] = team_info["name"]
                    edge_data["team_abbr"] = team_info["abbr"]
                    teams.append(edge_data)
                
                await asyncio.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"[NHL API] Error getting team EDGE for {team_id}: {e}")
                continue
        
        return teams
    
    async def _get_team_edge_detail(
        self,
        client: httpx.AsyncClient,
        team_id: int,
        season: str,
        game_type: int
    ) -> Optional[Dict[str, Any]]:
        """Get detailed EDGE stats for a team."""
        edge_data = {
            "season": season,
            "game_type": game_type,
        }
        
        # Get team detail
        try:
            url = f"{NHL_WEB_API}/edge/team-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["raw_data"] = data
        except:
            pass
        
        # Get team skating distance
        try:
            url = f"{NHL_WEB_API}/edge/team-skating-distance-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                edge_data["distance_total"] = data.get("distanceTotal")
                edge_data["distance_per_game"] = data.get("distancePerGame")
        except:
            pass
        
        # Get team shot location
        try:
            url = f"{NHL_WEB_API}/edge/team-shot-location-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                shots = data.get("shotsByZone", {})
                edge_data["shots_low"] = shots.get("low", {}).get("shots")
                edge_data["shots_mid"] = shots.get("mid", {}).get("shots")
                edge_data["shots_high"] = shots.get("high", {}).get("shots")
                edge_data["goals_low"] = shots.get("low", {}).get("goals")
                edge_data["goals_mid"] = shots.get("mid", {}).get("goals")
                edge_data["goals_high"] = shots.get("high", {}).get("goals")
                edge_data["shooting_pct_low"] = shots.get("low", {}).get("shootingPct")
                edge_data["shooting_pct_mid"] = shots.get("mid", {}).get("shootingPct")
                edge_data["shooting_pct_high"] = shots.get("high", {}).get("shootingPct")
        except:
            pass
        
        # Get team zone time from leaders endpoint
        try:
            url = f"{NHL_WEB_API}/edge/team-zone-time-top-10/all/offensiveZonePct/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                for team in data.get("data", []):
                    if team.get("teamId") == team_id:
                        edge_data["offensive_zone_pct"] = team.get("offensiveZonePct")
                        edge_data["defensive_zone_pct"] = team.get("defensiveZonePct")
                        edge_data["neutral_zone_pct"] = team.get("neutralZonePct")
                        break
        except:
            pass
        
        # Only return if we have some data
        if any(edge_data.get(k) for k in ["distance_total", "shots_low", "offensive_zone_pct"]):
            return edge_data
        
        return None

    # =========================================================================
    # EDGE LEADERS
    # =========================================================================
    
    async def _collect_edge_leaders(
        self,
        client: httpx.AsyncClient,
        season: str,
        game_type: int
    ) -> List[Dict[str, Any]]:
        """Collect EDGE leaders for various categories."""
        leaders = []
        
        # Shot speed leaders
        for sort_by in ["maxShotSpeed", "avgShotSpeed"]:
            for position in ["F", "D"]:
                try:
                    url = f"{NHL_WEB_API}/edge/skater-shot-speed-top-10/{position}/{sort_by}/{season}/{game_type}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        for rank, player in enumerate(data.get("data", [])[:10], 1):
                            leaders.append({
                                "category": "shot_speed",
                                "metric": sort_by,
                                "season": season,
                                "game_type": game_type,
                                "nhl_player_id": player.get("playerId"),
                                "player_name": player.get("skaterFullName", "Unknown"),
                                "team_abbr": player.get("teamAbbrev"),
                                "position": position,
                                "rank": rank,
                                "value": player.get(sort_by, 0),
                            })
                except:
                    pass
        
        # Skating speed leaders
        for sort_by in ["maxSkatingSpeed", "avgSkatingSpeed"]:
            for position in ["F", "D"]:
                try:
                    url = f"{NHL_WEB_API}/edge/skater-speed-top-10/{position}/{sort_by}/{season}/{game_type}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        for rank, player in enumerate(data.get("data", [])[:10], 1):
                            leaders.append({
                                "category": "skating_speed",
                                "metric": sort_by,
                                "season": season,
                                "game_type": game_type,
                                "nhl_player_id": player.get("playerId"),
                                "player_name": player.get("skaterFullName", "Unknown"),
                                "team_abbr": player.get("teamAbbrev"),
                                "position": position,
                                "rank": rank,
                                "value": player.get(sort_by, 0),
                            })
                except:
                    pass
        
        # Distance leaders
        for sort_by in ["distanceTotal", "distancePerGame"]:
            for position in ["F", "D"]:
                try:
                    url = f"{NHL_WEB_API}/edge/skater-distance-top-10/{position}/all/{sort_by}/{season}/{game_type}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        for rank, player in enumerate(data.get("data", [])[:10], 1):
                            leaders.append({
                                "category": "distance",
                                "metric": sort_by,
                                "season": season,
                                "game_type": game_type,
                                "nhl_player_id": player.get("playerId"),
                                "player_name": player.get("skaterFullName", "Unknown"),
                                "team_abbr": player.get("teamAbbrev"),
                                "position": position,
                                "rank": rank,
                                "value": player.get(sort_by, 0),
                            })
                except:
                    pass
        
        return leaders

    # =========================================================================
    # DATABASE SAVE METHODS (Using Existing Tables)
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to existing database tables."""
        total_saved = 0
        
        try:
            from app.models import Player, PlayerStats, Team, TeamStats, Sport, Season
            
            # Get or create NHL sport
            nhl_sport = await self._get_or_create_nhl_sport(session)
            
            # Save skater stats
            if data.get("skater_stats"):
                saved = await self._save_skater_stats_to_existing(session, data["skater_stats"], nhl_sport)
                total_saved += saved
                logger.info(f"[NHL API] Saved {saved} skater EDGE stat records")
            
            # Save goalie stats
            if data.get("goalie_stats"):
                saved = await self._save_goalie_stats_to_existing(session, data["goalie_stats"], nhl_sport)
                total_saved += saved
                logger.info(f"[NHL API] Saved {saved} goalie EDGE stat records")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self._save_team_stats_to_existing(session, data["team_stats"], nhl_sport)
                total_saved += saved
                logger.info(f"[NHL API] Saved {saved} team EDGE stat records")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[NHL API] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _get_or_create_nhl_sport(self, session: AsyncSession):
        """Get or create NHL sport record."""
        from app.models import Sport
        
        result = await session.execute(
            select(Sport).where(Sport.code == "NHL")
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                code="NHL",
                name="National Hockey League",
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        return sport
    
    async def _get_or_create_season(self, session: AsyncSession, sport_id, season_str: str):
        """Get or create season record for NHL season like '20242025'."""
        from app.models import Season
        from datetime import date
        
        # Parse season string (e.g., "20242025" -> 2024-2025 season)
        start_year = int(season_str[:4])
        end_year = int(season_str[4:])
        
        result = await session.execute(
            select(Season).where(
                and_(
                    Season.sport_id == sport_id,
                    Season.year == start_year
                )
            )
        )
        season = result.scalar_one_or_none()
        
        if not season:
            season = Season(
                sport_id=sport_id,
                year=start_year,
                name=f"{start_year}-{end_year % 100:02d}",
                start_date=date(start_year, 10, 1),
                end_date=date(end_year, 6, 30),
                is_current=(end_year == datetime.now().year or 
                           (end_year == datetime.now().year + 1 and datetime.now().month >= 10))
            )
            session.add(season)
            await session.flush()
        
        return season
    
    async def _get_or_create_player(self, session: AsyncSession, nhl_player_id: int, player_name: str, 
                                     team_abbr: str = None, position: str = None):
        """Get or create player record."""
        from app.models import Player, Team
        
        external_id = f"nhl_{nhl_player_id}"
        
        result = await session.execute(
            select(Player).where(Player.external_id == external_id)
        )
        player = result.scalar_one_or_none()
        
        if not player:
            # Try to find team
            team_id = None
            if team_abbr:
                team_result = await session.execute(
                    select(Team).where(Team.abbreviation == team_abbr)
                )
                team = team_result.scalar_one_or_none()
                if team:
                    team_id = team.id
            
            player = Player(
                external_id=external_id,
                name=player_name,
                team_id=team_id,
                position=position,
                is_active=True
            )
            session.add(player)
            await session.flush()
        else:
            # Update if needed
            if position and player.position != position:
                player.position = position
        
        return player
    
    async def _save_skater_stats_to_existing(
        self,
        session: AsyncSession,
        records: List[Dict[str, Any]],
        sport
    ) -> int:
        """Save skater EDGE stats to existing player_stats table."""
        from app.models import PlayerStats
        
        saved = 0
        
        # EDGE stat types to save
        edge_stat_types = [
            ("shot_speed_max", "edge_shot_speed_max"),
            ("shot_speed_avg", "edge_shot_speed_avg"),
            ("skating_speed_max", "edge_skating_speed_max"),
            ("skating_speed_avg", "edge_skating_speed_avg"),
            ("speed_bursts_22plus", "edge_speed_bursts_22plus"),
            ("speed_bursts_20_22", "edge_speed_bursts_20_22"),
            ("speed_bursts_18_20", "edge_speed_bursts_18_20"),
            ("distance_total", "edge_distance_total"),
            ("distance_per_game", "edge_distance_per_game"),
            ("offensive_zone_pct", "edge_offensive_zone_pct"),
            ("defensive_zone_pct", "edge_defensive_zone_pct"),
            ("neutral_zone_pct", "edge_neutral_zone_pct"),
            ("offensive_zone_time", "edge_offensive_zone_time"),
            ("defensive_zone_time", "edge_defensive_zone_time"),
            ("neutral_zone_time", "edge_neutral_zone_time"),
            ("shots_low", "edge_shots_low"),
            ("shots_mid", "edge_shots_mid"),
            ("shots_high", "edge_shots_high"),
            ("goals_low", "edge_goals_low"),
            ("goals_mid", "edge_goals_mid"),
            ("goals_high", "edge_goals_high"),
        ]
        
        for record in records:
            try:
                nhl_player_id = record.get("nhl_player_id")
                player_name = record.get("player_name", "Unknown")
                season_str = record.get("season")
                
                if not nhl_player_id or not season_str:
                    continue
                
                # Get or create player
                player = await self._get_or_create_player(
                    session, nhl_player_id, player_name,
                    record.get("team_abbr"), record.get("position")
                )
                
                # Get or create season
                season = await self._get_or_create_season(session, sport.id, season_str)
                
                # Save each EDGE stat
                for data_key, stat_type in edge_stat_types:
                    value = record.get(data_key)
                    if value is not None:
                        # Check for existing
                        result = await session.execute(
                            select(PlayerStats).where(
                                and_(
                                    PlayerStats.player_id == player.id,
                                    PlayerStats.season_id == season.id,
                                    PlayerStats.stat_type == stat_type
                                )
                            )
                        )
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            existing.value = float(value)
                        else:
                            stat = PlayerStats(
                                player_id=player.id,
                                season_id=season.id,
                                stat_type=stat_type,
                                value=float(value)
                            )
                            session.add(stat)
                        
                        saved += 1
                
            except Exception as e:
                logger.debug(f"[NHL API] Error saving skater stat: {e}")
                continue
        
        await session.flush()
        return saved
    
    async def _save_goalie_stats_to_existing(
        self,
        session: AsyncSession,
        records: List[Dict[str, Any]],
        sport
    ) -> int:
        """Save goalie EDGE stats to existing player_stats table."""
        from app.models import PlayerStats
        
        saved = 0
        
        # Goalie EDGE stat types
        goalie_stat_types = [
            ("saves_low", "edge_saves_low"),
            ("saves_mid", "edge_saves_mid"),
            ("saves_high", "edge_saves_high"),
            ("save_pct_low", "edge_save_pct_low"),
            ("save_pct_mid", "edge_save_pct_mid"),
            ("save_pct_high", "edge_save_pct_high"),
            ("shots_against_low", "edge_shots_against_low"),
            ("shots_against_mid", "edge_shots_against_mid"),
            ("shots_against_high", "edge_shots_against_high"),
            ("goals_against_low", "edge_goals_against_low"),
            ("goals_against_mid", "edge_goals_against_mid"),
            ("goals_against_high", "edge_goals_against_high"),
            ("save_pct_5v5", "edge_save_pct_5v5"),
            ("goals_against_5v5", "edge_goals_against_5v5"),
            ("shots_against_5v5", "edge_shots_against_5v5"),
            ("high_danger_saves", "edge_high_danger_saves"),
            ("high_danger_save_pct", "edge_high_danger_save_pct"),
            ("high_danger_goals_against", "edge_high_danger_goals_against"),
            ("total_saves", "edge_total_saves"),
            ("total_shots_against", "edge_total_shots_against"),
            ("save_pct", "save_pct"),
            ("goals_against_avg", "goals_against_avg"),
        ]
        
        for record in records:
            try:
                nhl_player_id = record.get("nhl_player_id")
                player_name = record.get("player_name", "Unknown")
                season_str = record.get("season")
                
                if not nhl_player_id or not season_str:
                    continue
                
                # Get or create player (goalies)
                player = await self._get_or_create_player(
                    session, nhl_player_id, player_name,
                    record.get("team_abbr"), "G"
                )
                
                # Get or create season
                season = await self._get_or_create_season(session, sport.id, season_str)
                
                # Save each EDGE stat
                for data_key, stat_type in goalie_stat_types:
                    value = record.get(data_key)
                    if value is not None:
                        result = await session.execute(
                            select(PlayerStats).where(
                                and_(
                                    PlayerStats.player_id == player.id,
                                    PlayerStats.season_id == season.id,
                                    PlayerStats.stat_type == stat_type
                                )
                            )
                        )
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            existing.value = float(value)
                        else:
                            stat = PlayerStats(
                                player_id=player.id,
                                season_id=season.id,
                                stat_type=stat_type,
                                value=float(value)
                            )
                            session.add(stat)
                        
                        saved += 1
                
            except Exception as e:
                logger.debug(f"[NHL API] Error saving goalie stat: {e}")
                continue
        
        await session.flush()
        return saved
    
    async def _save_team_stats_to_existing(
        self,
        session: AsyncSession,
        records: List[Dict[str, Any]],
        sport
    ) -> int:
        """Save team EDGE stats to existing team_stats table."""
        from app.models import TeamStats, Team
        
        saved = 0
        
        # Team EDGE stat types
        team_stat_types = [
            ("distance_total", "edge_distance_total"),
            ("distance_per_game", "edge_distance_per_game"),
            ("offensive_zone_pct", "edge_offensive_zone_pct"),
            ("defensive_zone_pct", "edge_defensive_zone_pct"),
            ("neutral_zone_pct", "edge_neutral_zone_pct"),
            ("avg_shot_speed", "edge_avg_shot_speed"),
            ("max_shot_speed", "edge_max_shot_speed"),
            ("shots_low", "edge_shots_low"),
            ("shots_mid", "edge_shots_mid"),
            ("shots_high", "edge_shots_high"),
            ("goals_low", "edge_goals_low"),
            ("goals_mid", "edge_goals_mid"),
            ("goals_high", "edge_goals_high"),
            ("shooting_pct_low", "edge_shooting_pct_low"),
            ("shooting_pct_mid", "edge_shooting_pct_mid"),
            ("shooting_pct_high", "edge_shooting_pct_high"),
        ]
        
        for record in records:
            try:
                team_abbr = record.get("team_abbr")
                season_str = record.get("season")
                
                if not team_abbr or not season_str:
                    continue
                
                # Find team
                result = await session.execute(
                    select(Team).where(Team.abbreviation == team_abbr)
                )
                team = result.scalar_one_or_none()
                
                if not team:
                    logger.debug(f"[NHL API] Team not found: {team_abbr}")
                    continue
                
                # Get or create season
                season = await self._get_or_create_season(session, sport.id, season_str)
                
                # Save each EDGE stat
                for data_key, stat_type in team_stat_types:
                    value = record.get(data_key)
                    if value is not None:
                        result = await session.execute(
                            select(TeamStats).where(
                                and_(
                                    TeamStats.team_id == team.id,
                                    TeamStats.season_id == season.id,
                                    TeamStats.stat_type == stat_type
                                )
                            )
                        )
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            existing.value = float(value)
                        else:
                            stat = TeamStats(
                                team_id=team.id,
                                season_id=season.id,
                                stat_type=stat_type,
                                value=float(value),
                                games_played=record.get("games_played", 0)
                            )
                            session.add(stat)
                        
                        saved += 1
                
            except Exception as e:
                logger.debug(f"[NHL API] Error saving team stat: {e}")
                continue
        
        await session.flush()
        return saved


# =============================================================================
# CREATE SINGLETON INSTANCE
# =============================================================================

nhl_official_api_collector = NHLOfficialAPICollector()

# Register with collector manager
collector_manager.register_collector(nhl_official_api_collector)
logger.info("Registered collector: NHL Official API")