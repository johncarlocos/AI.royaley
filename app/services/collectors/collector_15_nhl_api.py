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
        # Season starts in October, so if month < 10, we're in previous season's finish
        # Current date is Jan 2026, so we're in 2025-26 season = 20252026
        # But NHL API data might lag, so let's also try previous season
        if current_month < 7:  # Before July = still in previous season or just ended
            end_season_year = current_year
        else:
            end_season_year = current_year + 1
        
        # Generate season list (EDGE data available from ~2021)
        seasons = []
        for i in range(years_back):
            season_end = end_season_year - i
            season_start = season_end - 1
            # EDGE tracking started in 2021-22 season
            if season_start >= 2021:
                seasons.append(f"{season_start}{season_end}")
        
        # If no EDGE seasons, fall back to basic stats seasons
        if not seasons:
            for i in range(years_back):
                season_end = end_season_year - i
                season_start = season_end - 1
                if season_start >= 2015:
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
        
        has_any_data = False
        
        # Get shot speed detail (most reliable EDGE endpoint)
        try:
            url = f"{NHL_WEB_API}/edge/skater-shot-speed-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Debug: log first response structure
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Shot speed keys for {player_id}: {list(data.keys())[:15]}")
                
                # Try multiple key patterns - NHL API uses various formats
                edge_data["shot_speed_max"] = (
                    data.get("maxShotSpeed") or 
                    data.get("shotSpeedMax") or
                    data.get("max") or
                    (data.get("shotSpeed", {}).get("max") if isinstance(data.get("shotSpeed"), dict) else None)
                )
                edge_data["shot_speed_avg"] = (
                    data.get("avgShotSpeed") or 
                    data.get("shotSpeedAvg") or
                    data.get("avg") or
                    (data.get("shotSpeed", {}).get("avg") if isinstance(data.get("shotSpeed"), dict) else None)
                )
                edge_data["total_shots"] = data.get("totalShots") or data.get("shots")
                
                # Check all keys for any shot/speed data
                for key, value in data.items():
                    if value and isinstance(value, (int, float)):
                        if "max" in key.lower() and "shot" in key.lower() and not edge_data.get("shot_speed_max"):
                            edge_data["shot_speed_max"] = value
                            has_any_data = True
                        elif "avg" in key.lower() and "shot" in key.lower() and not edge_data.get("shot_speed_avg"):
                            edge_data["shot_speed_avg"] = value
                            has_any_data = True
                
                if edge_data.get("shot_speed_max") or edge_data.get("shot_speed_avg"):
                    has_any_data = True
                    
                # Store raw data for analysis
                edge_data["raw_shot_speed"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Shot speed error for {player_id}: {e}")
        
        # Get skating speed detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-skating-speed-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Debug: log structure
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Skating speed keys for {player_id}: {list(data.keys())[:15]}")
                
                # Try multiple key patterns
                edge_data["skating_speed_max"] = (
                    data.get("maxSkatingSpeed") or 
                    data.get("skatingSpeedMax") or
                    data.get("max") or
                    (data.get("skatingSpeed", {}).get("max") if isinstance(data.get("skatingSpeed"), dict) else None)
                )
                edge_data["skating_speed_avg"] = (
                    data.get("avgSkatingSpeed") or 
                    data.get("skatingSpeedAvg") or
                    data.get("avg") or
                    (data.get("skatingSpeed", {}).get("avg") if isinstance(data.get("skatingSpeed"), dict) else None)
                )
                edge_data["speed_bursts_22plus"] = data.get("speedBursts22Plus") or data.get("bursts22Plus")
                edge_data["speed_bursts_20_22"] = data.get("speedBursts20To22") or data.get("bursts20To22")
                edge_data["speed_bursts_18_20"] = data.get("speedBursts18To20") or data.get("bursts18To20")
                
                # Scan all keys for skating speed data
                for key, value in data.items():
                    if value and isinstance(value, (int, float)):
                        if "max" in key.lower() and "skat" in key.lower() and not edge_data.get("skating_speed_max"):
                            edge_data["skating_speed_max"] = value
                            has_any_data = True
                        elif "avg" in key.lower() and "skat" in key.lower() and not edge_data.get("skating_speed_avg"):
                            edge_data["skating_speed_avg"] = value
                            has_any_data = True
                
                if edge_data.get("skating_speed_max") or edge_data.get("skating_speed_avg"):
                    has_any_data = True
                    
                edge_data["raw_skating_speed"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Skating speed error for {player_id}: {e}")
        
        # Get skating distance detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-skating-distance-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Distance keys for {player_id}: {list(data.keys())[:15]}")
                
                edge_data["distance_total"] = (
                    data.get("distanceTotal") or 
                    data.get("totalDistance") or
                    data.get("total") or
                    (data.get("distance", {}).get("total") if isinstance(data.get("distance"), dict) else None)
                )
                edge_data["distance_per_game"] = (
                    data.get("distancePerGame") or 
                    data.get("perGame") or
                    data.get("avg") or
                    (data.get("distance", {}).get("perGame") if isinstance(data.get("distance"), dict) else None)
                )
                
                if edge_data.get("distance_total") or edge_data.get("distance_per_game"):
                    has_any_data = True
                    
                edge_data["raw_distance"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Distance error for {player_id}: {e}")
        
        # Get zone time from skater detail
        try:
            url = f"{NHL_WEB_API}/edge/skater-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Skater detail keys for {player_id}: {list(data.keys())[:15]}")
                
                # Zone time - try multiple structures
                zone_time = data.get("zoneTime", data.get("zoneTimeBreakdown", data.get("zones", {})))
                if zone_time and isinstance(zone_time, dict):
                    edge_data["offensive_zone_pct"] = zone_time.get("offensiveZonePct") or zone_time.get("ozPct") or zone_time.get("oz")
                    edge_data["defensive_zone_pct"] = zone_time.get("defensiveZonePct") or zone_time.get("dzPct") or zone_time.get("dz")
                    edge_data["neutral_zone_pct"] = zone_time.get("neutralZonePct") or zone_time.get("nzPct") or zone_time.get("nz")
                    if any([edge_data.get("offensive_zone_pct"), edge_data.get("defensive_zone_pct")]):
                        has_any_data = True
                
                # Also check top-level for zone data
                edge_data["offensive_zone_pct"] = edge_data.get("offensive_zone_pct") or data.get("offensiveZonePct") or data.get("ozPct")
                edge_data["defensive_zone_pct"] = edge_data.get("defensive_zone_pct") or data.get("defensiveZonePct") or data.get("dzPct")
                
                edge_data["raw_data"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Skater detail error for {player_id}: {e}")
        
        # Return data if we got anything useful
        if has_any_data:
            return edge_data
        
        return None
        
        # Return data if we got anything
        if has_any_data:
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
        
        has_any_data = False
        
        # Get goalie shot location detail (primary endpoint)
        try:
            url = f"{NHL_WEB_API}/edge/goalie-shot-location-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Debug: log first response structure
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Goalie shot location keys for {player_id}: {list(data.keys())[:15]}")
                    # Also log sample value types
                    for k, v in list(data.items())[:5]:
                        logger.info(f"[NHL API DEBUG] {k}: {type(v).__name__} = {v if not isinstance(v, dict) else list(v.keys())[:5]}")
                
                # Try nested structure first
                saves = data.get("savesByZone", data.get("saves", {}))
                if saves and isinstance(saves, dict):
                    # Nested: {"low": {"saves": 100, "savePct": 0.92}, ...}
                    if isinstance(saves.get("low"), dict):
                        edge_data["saves_low"] = saves["low"].get("saves")
                        edge_data["save_pct_low"] = saves["low"].get("savePct")
                        has_any_data = True
                    if isinstance(saves.get("mid"), dict):
                        edge_data["saves_mid"] = saves["mid"].get("saves")
                        edge_data["save_pct_mid"] = saves["mid"].get("savePct")
                    if isinstance(saves.get("high"), dict):
                        edge_data["saves_high"] = saves["high"].get("saves")
                        edge_data["save_pct_high"] = saves["high"].get("savePct")
                
                # Try flat structure: {"lowSaves": 100, "lowSavePct": 0.92, ...}
                if not has_any_data:
                    edge_data["saves_low"] = data.get("lowSaves") or data.get("savesLow")
                    edge_data["saves_mid"] = data.get("midSaves") or data.get("savesMid") or data.get("middleSaves")
                    edge_data["saves_high"] = data.get("highSaves") or data.get("savesHigh")
                    edge_data["save_pct_low"] = data.get("lowSavePct") or data.get("savePctLow")
                    edge_data["save_pct_mid"] = data.get("midSavePct") or data.get("savePctMid") or data.get("middleSavePct")
                    edge_data["save_pct_high"] = data.get("highSavePct") or data.get("savePctHigh")
                    if any([edge_data.get("saves_low"), edge_data.get("saves_mid"), edge_data.get("saves_high")]):
                        has_any_data = True
                
                # Scan all keys for save data
                for key, value in data.items():
                    if value and isinstance(value, (int, float)):
                        key_lower = key.lower()
                        if "save" in key_lower:
                            has_any_data = True  # Found some save data
                
                edge_data["raw_shot_location"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Goalie shot location error for {player_id}: {e}")
        
        # Get goalie 5v5 detail
        try:
            url = f"{NHL_WEB_API}/edge/goalie-5v5-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Goalie 5v5 keys for {player_id}: {list(data.keys())[:15]}")
                
                edge_data["save_pct_5v5"] = (
                    data.get("savePct5v5") or 
                    data.get("savePct") or 
                    data.get("fiveOnFiveSavePct") or
                    data.get("5v5SavePct")
                )
                edge_data["goals_against_5v5"] = (
                    data.get("goalsAgainst5v5") or 
                    data.get("goalsAgainst") or
                    data.get("5v5GoalsAgainst")
                )
                edge_data["shots_against_5v5"] = (
                    data.get("shotsAgainst5v5") or 
                    data.get("shotsAgainst") or
                    data.get("5v5ShotsAgainst")
                )
                if edge_data.get("save_pct_5v5"):
                    has_any_data = True
                    
                edge_data["raw_5v5"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Goalie 5v5 error for {player_id}: {e}")
        
        # Get goalie detail (high danger stats)
        try:
            url = f"{NHL_WEB_API}/edge/goalie-detail/{player_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if player_id % 500 == 0:
                    logger.info(f"[NHL API DEBUG] Goalie detail keys for {player_id}: {list(data.keys())[:15]}")
                
                # High danger - try nested and flat
                hd = data.get("highDanger", data.get("highDangerShots", {}))
                if hd and isinstance(hd, dict):
                    edge_data["high_danger_saves"] = hd.get("saves")
                    edge_data["high_danger_save_pct"] = hd.get("savePct") or hd.get("savePercentage")
                    edge_data["high_danger_goals_against"] = hd.get("goalsAgainst") or hd.get("goals")
                    if edge_data.get("high_danger_saves"):
                        has_any_data = True
                
                # Flat high danger
                edge_data["high_danger_saves"] = edge_data.get("high_danger_saves") or data.get("highDangerSaves")
                edge_data["high_danger_save_pct"] = edge_data.get("high_danger_save_pct") or data.get("highDangerSavePct")
                
                # Total saves
                edge_data["total_saves"] = data.get("totalSaves") or data.get("saves")
                edge_data["total_shots_against"] = data.get("totalShotsAgainst") or data.get("shotsAgainst")
                
                if edge_data.get("total_saves") or edge_data.get("high_danger_saves"):
                    has_any_data = True
                    
                edge_data["raw_detail"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Goalie detail error for {player_id}: {e}")
        
        if has_any_data:
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
        
        has_any_data = False
        
        # Get team detail (main endpoint)
        try:
            url = f"{NHL_WEB_API}/edge/team-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Debug: log first team's structure
                if team_id == 1:
                    logger.info(f"[NHL API DEBUG] Team detail keys for {team_id}: {list(data.keys())[:15]}")
                    for k, v in list(data.items())[:5]:
                        logger.info(f"[NHL API DEBUG] Team {k}: {type(v).__name__} = {v if not isinstance(v, dict) else list(v.keys())[:5] if isinstance(v, dict) else v}")
                
                edge_data["raw_data"] = data
                
                # Try to extract any useful team data
                for key, value in data.items():
                    if value and isinstance(value, (int, float)):
                        has_any_data = True
                        key_lower = key.lower()
                        if "distance" in key_lower and "total" in key_lower:
                            edge_data["distance_total"] = value
                        elif "distance" in key_lower and "game" in key_lower:
                            edge_data["distance_per_game"] = value
                        elif "oz" in key_lower or "offensive" in key_lower:
                            edge_data["offensive_zone_pct"] = value
                        elif "dz" in key_lower or "defensive" in key_lower:
                            edge_data["defensive_zone_pct"] = value
        except Exception as e:
            logger.debug(f"[NHL API] Team detail error for {team_id}: {e}")
        
        # Get team skating distance
        try:
            url = f"{NHL_WEB_API}/edge/team-skating-distance-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if team_id == 1:
                    logger.info(f"[NHL API DEBUG] Team distance keys for {team_id}: {list(data.keys())[:15]}")
                
                # Try multiple key patterns
                edge_data["distance_total"] = (
                    data.get("distanceTotal") or 
                    data.get("totalDistance") or
                    data.get("total")
                )
                edge_data["distance_per_game"] = (
                    data.get("distancePerGame") or 
                    data.get("avgDistance") or
                    data.get("perGame")
                )
                
                if edge_data.get("distance_total") or edge_data.get("distance_per_game"):
                    has_any_data = True
                    
                edge_data["raw_distance"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Team distance error for {team_id}: {e}")
        
        # Get team shot location
        try:
            url = f"{NHL_WEB_API}/edge/team-shot-location-detail/{team_id}/{season}/{game_type}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                
                if team_id == 1:
                    logger.info(f"[NHL API DEBUG] Team shot location keys for {team_id}: {list(data.keys())[:15]}")
                
                # Try nested structure
                shots = data.get("shotsByZone", data.get("shots", {}))
                if shots and isinstance(shots, dict):
                    if isinstance(shots.get("low"), dict):
                        edge_data["shots_low"] = shots["low"].get("shots")
                        edge_data["goals_low"] = shots["low"].get("goals")
                        edge_data["shooting_pct_low"] = shots["low"].get("shootingPct")
                        has_any_data = True
                    if isinstance(shots.get("mid"), dict):
                        edge_data["shots_mid"] = shots["mid"].get("shots")
                        edge_data["goals_mid"] = shots["mid"].get("goals")
                        edge_data["shooting_pct_mid"] = shots["mid"].get("shootingPct")
                    if isinstance(shots.get("high"), dict):
                        edge_data["shots_high"] = shots["high"].get("shots")
                        edge_data["goals_high"] = shots["high"].get("goals")
                        edge_data["shooting_pct_high"] = shots["high"].get("shootingPct")
                
                # Try flat structure
                if not has_any_data:
                    edge_data["shots_low"] = data.get("lowShots") or data.get("shotsLow")
                    edge_data["shots_mid"] = data.get("midShots") or data.get("shotsMid")
                    edge_data["shots_high"] = data.get("highShots") or data.get("shotsHigh")
                    if any([edge_data.get("shots_low"), edge_data.get("shots_mid"), edge_data.get("shots_high")]):
                        has_any_data = True
                
                edge_data["raw_shot_location"] = data
        except Exception as e:
            logger.debug(f"[NHL API] Team shot location error for {team_id}: {e}")
        
        # Only return if we have some data
        if has_any_data or edge_data.get("raw_data"):
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
collector_manager.register(nhl_official_api_collector)
logger.info("Registered collector: NHL Official API")