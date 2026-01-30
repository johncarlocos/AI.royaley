"""
ROYALEY - TheSportsDB Collector (V2 Premium) - PLAYER FIX
============================================================

V2 API Endpoints (Premium $295):
================================
Base URL: https://www.thesportsdb.com/api/v2/json
Auth: X-API-KEY header with key 688655

TEAMS & VENUES:
- /list/teams/{leagueId}          → minimal team data
- /lookup/team/{teamId}           → FULL team data + stadium

SCHEDULE (GAMES):
- /schedule/next/league/{leagueId}     → next 20 upcoming
- /schedule/previous/league/{leagueId} → last 20 results
- /schedule/league/{leagueId}/{season} → FULL season (3000 limit)
- /schedule/full/team/{teamId}         → full team schedule (250 limit)

SEASONS:
- /list/seasons/{leagueId}        → available seasons for league

PLAYERS:
- /list/players/{teamId}          → team roster (returns "player" or "players" key)
- /lookup/player/{playerId}       → player details

STANDINGS:
- /lookup/table/{leagueId}/{season} → league standings

LIVESCORES:
- /livescore/{leagueId}           → live scores

Season Format:
- NFL/MLB/NCAAF: "2024" (calendar year)
- NBA/NHL/NCAAB: "2024-2025" (split season)

FIX: Updated _get_list to handle both "player" and "players" keys
FIX: Added better error handling and logging for player collection
"""

import asyncio
import logging
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)

# League IDs - ALL 10 SPORTS (matching database)
SPORTSDB_LEAGUE_IDS = {
    # Major US Pro Leagues
    "NFL": 4391,
    "NBA": 4387,
    "NHL": 4380,
    "MLB": 4424,
    # College Sports
    "NCAAF": 4479,
    "NCAAB": 4607,
    # Additional Leagues
    "CFL": 4405,
    "WNBA": 4516,
    # Tennis (individual sport - no teams)
    "ATP": 4464,
    "WTA": 4517,
}

# Season format by sport
SPLIT_SEASON_SPORTS = ["NBA", "NHL", "NCAAB", "NCAAF"]  # Use "2024-2025" format
CALENDAR_YEAR_SPORTS = ["NFL", "MLB", "CFL", "WNBA", "ATP", "WTA"]  # Use "2024" format

# All sports for ML training
ML_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]

STATUS_MAP = {
    "NS": "scheduled", "1H": "in_progress", "2H": "in_progress",
    "HT": "in_progress", "FT": "final", "AOT": "final",
    "Q1": "in_progress", "Q2": "in_progress", "Q3": "in_progress", "Q4": "in_progress",
    "OT": "in_progress", "Match Finished": "final", "Finished": "final",
    "Not Started": "scheduled", "": "scheduled",
}


class SportsDBCollector(BaseCollector):
    """TheSportsDB V2 API Collector - Complete Implementation with Player Fix."""
    
    def __init__(self):
        super().__init__(
            name="sportsdb",
            base_url="https://www.thesportsdb.com/api/v2/json",
            rate_limit=120, rate_window=60, timeout=30.0, max_retries=3,
        )
        self.api_key = getattr(settings, 'SPORTSDB_API_KEY', None) or "688655"
        logger.info(f"[SportsDB] V2 Premium | Key: {self.api_key}")
        print(f"[SportsDB] V2 Premium | Key: {self.api_key}")
    
    def _headers(self) -> Dict[str, str]:
        return {"Accept": "application/json", "X-API-KEY": self.api_key}
    
    async def _v2(self, endpoint: str) -> Optional[Any]:
        """V2 API request with detailed response logging."""
        url = f"{self.base_url}{endpoint}"
        logger.info(f"[SportsDB] 🌐 {url}")
        print(f"[SportsDB] 🌐 {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        # Check for error message
                        if "Message" in data and len(data) == 1:
                            logger.warning(f"[SportsDB] ⚠️ API Message: {data.get('Message')}")
                            print(f"[SportsDB] ⚠️ API Message: {data.get('Message')}")
                            return None
                        # Log all keys for debugging
                        info = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()}
                        logger.info(f"[SportsDB] ✅ 200 | Keys: {list(data.keys())} | {info}")
                        print(f"[SportsDB] ✅ 200 | Keys: {list(data.keys())} | {info}")
                    return data
                else:
                    logger.warning(f"[SportsDB] ❌ {resp.status_code}")
                    print(f"[SportsDB] ❌ {resp.status_code}: {resp.text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"[SportsDB] ❌ {e}")
            print(f"[SportsDB] ❌ {e}")
            return None
    
    def _get_list(self, data: Any, *keys) -> List[Dict]:
        """Extract list from V2 response - FIXED to handle both 'player' and 'players' keys."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Check all possible keys including singular versions
            all_possible_keys = list(keys) + [
                "schedule", "events", "list", "lookup", "teams", "livescore", 
                "table", "players", "player", "seasons", "roster"
            ]
            for k in all_possible_keys:
                if k in data:
                    val = data[k]
                    if isinstance(val, list):
                        return val
                    elif val is None:
                        # API returned null - this is normal for empty rosters
                        logger.debug(f"[SportsDB] Key '{k}' is null")
                        return []
        return []
    
    def _get_season_format(self, sport_code: str, year: int) -> str:
        """Get correct season format for sport."""
        if sport_code in SPLIT_SEASON_SPORTS:
            return f"{year}-{year+1}"
        return str(year)  # Calendar year sports

    # =========================================================================
    # REQUIRED ABSTRACT METHODS
    # =========================================================================
    
    async def collect(self, sport_code: str = None, collect_type: str = "all", **kwargs) -> CollectorResult:
        """Main collection method."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_data = {"teams": [], "games": [], "venues": [], "players": [], "livescores": []}
        
        for sport in sports:
            if sport not in SPORTSDB_LEAGUE_IDS:
                continue
            try:
                if collect_type in ("all", "teams"):
                    teams = await self._collect_teams(sport)
                    all_data["teams"].extend(teams)
                if collect_type in ("all", "games", "schedule"):
                    games = await self._collect_schedule(sport)
                    all_data["games"].extend(games)
                if collect_type in ("all", "livescores"):
                    live = await self._collect_livescores(sport)
                    all_data["livescores"].extend(live)
            except Exception as e:
                logger.error(f"[SportsDB] {sport} error: {e}")
        
        total = sum(len(v) for v in all_data.values())
        return CollectorResult(success=total > 0, data=all_data, records_count=total)
    
    async def validate(self, data: Any) -> bool:
        return bool(data) and (len(data) > 0 if isinstance(data, (dict, list)) else True)

    # =========================================================================
    # SEASONS - Get available seasons for a league
    # =========================================================================
    
    async def get_available_seasons(self, sport_code: str) -> List[str]:
        """Get list of available seasons for a league (most recent first)."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/list/seasons/{league_id}")
        seasons = self._get_list(data, "seasons")
        
        result = [s.get("strSeason") for s in seasons if s.get("strSeason")]
        # Sort descending (most recent first)
        result = sorted(result, reverse=True)
        logger.info(f"[SportsDB] {sport_code}: {len(result)} seasons available")
        print(f"[SportsDB] {sport_code}: {len(result)} seasons available")
        return result

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self, sport_code: str, include_details: bool = False) -> List[Dict]:
        """Collect teams - with optional full details including city."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/list/teams/{league_id}")
        teams = self._get_list(data, "list", "teams")
        
        result = []
        for t in teams:
            team_data = {
                "external_id": f"sportsdb_{t.get('idTeam')}",
                "sportsdb_id": t.get("idTeam"),
                "name": t.get("strTeam"),
                "abbreviation": (t.get("strTeamShort") or "")[:10],
                "country": t.get("strCountry"),
                "logo_url": t.get("strBadge"),
                "sport_code": sport_code,
            }
            
            # Get full details including city if requested
            if include_details:
                details = await self._get_team_details(t.get("idTeam"))
                if details:
                    # Extract city from strLocation (format: "City, State" or just "City")
                    loc = details.get("strLocation") or ""
                    city = loc.split(",")[0].strip() if loc else None
                    
                    team_data.update({
                        "city": city,
                        "stadium": details.get("strStadium"),
                        "conference": details.get("strDivision"),  # Sometimes division is conference
                        "division": details.get("strDivision"),
                    })
                await asyncio.sleep(0.1)  # Rate limit
            
            result.append(team_data)
        
        logger.info(f"[SportsDB] {sport_code}: {len(result)} teams")
        print(f"[SportsDB] {sport_code}: {len(result)} teams")
        return result
    
    async def collect_teams_with_details(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect teams with full details including city (for weather)."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_teams = []
        
        for sport in sports:
            teams = await self._collect_teams(sport, include_details=True)
            all_teams.extend(teams)
        
        return {"teams": all_teams, "count": len(all_teams)}
    
    async def collect_all_livescores(self) -> CollectorResult:
        """Collect live scores from all sports."""
        all_live = []
        for sport in ML_SPORTS:
            try:
                live = await self._collect_livescores(sport)
                all_live.extend(live)
            except Exception as e:
                logger.warning(f"[SportsDB] {sport} livescore error: {e}")
        
        return CollectorResult(
            success=True,
            data={"livescores": all_live},
            records_count=len(all_live)
        )
    
    async def collect_standings(self, sport_code: str = None, season: str = None) -> Dict[str, Any]:
        """Collect standings for all sports."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_standings = []
        
        # Default to current season
        current_year = datetime.now().year
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # Get season format
            if season:
                season_str = season
            else:
                season_str = self._get_season_format(sport, current_year)
            
            try:
                data = await self._v2(f"/lookup/table/{league_id}/{season_str}")
                standings = self._get_list(data, "table", "standings")
                
                for s in standings:
                    all_standings.append({
                        "team_name": s.get("strTeam"),
                        "team_id": s.get("idTeam"),
                        "rank": s.get("intRank"),
                        "played": s.get("intPlayed"),
                        "wins": s.get("intWin"),
                        "losses": s.get("intLoss"),
                        "draws": s.get("intDraw"),
                        "points": s.get("intPoints"),
                        "goals_for": s.get("intGoalsFor"),
                        "goals_against": s.get("intGoalsAgainst"),
                        "sport_code": sport,
                        "season": season_str,
                    })
                
                print(f"[SportsDB] {sport} {season_str}: {len(standings)} standings")
            except Exception as e:
                logger.warning(f"[SportsDB] {sport} standings error: {e}")
        
        return {"standings": all_standings, "count": len(all_standings)}
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """General save method - dispatches to specific save methods."""
        total_saved = 0
        
        # Save teams
        if "teams" in data and data["teams"]:
            saved = await self.save_teams_to_database(data["teams"], session)
            total_saved += saved
        
        # Save games
        if "games" in data and data["games"]:
            saved = await self.save_games_to_database(data["games"], session)
            total_saved += saved
        
        # Save venues
        if "venues" in data and data["venues"]:
            saved = await self.save_venues_to_database(data["venues"], session)
            total_saved += saved
        
        # Save players
        if "players" in data and data["players"]:
            saved = await self.save_players_to_database(data["players"], session)
            total_saved += saved
        
        # Update livescores
        if "livescores" in data and data["livescores"]:
            await self._update_livescores(data, session)
        
        return total_saved
    
    async def _get_team_details(self, team_id: str) -> Optional[Dict]:
        """Get FULL team details including stadium."""
        data = await self._v2(f"/lookup/team/{team_id}")
        teams = self._get_list(data, "lookup", "teams")
        return teams[0] if teams else None

    # =========================================================================
    # VENUES COLLECTION
    # =========================================================================
    
    # Comprehensive venue/city coordinates cache (expanded for all sports)
    CITY_COORDS = {
        # === US Major Cities ===
        "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298), "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740), "Philadelphia": (39.9526, -75.1652),
        "San Antonio": (29.4241, -98.4936), "San Diego": (32.7157, -117.1611),
        "Dallas": (32.7767, -96.7970), "San Francisco": (37.7749, -122.4194),
        "Denver": (39.7392, -104.9903), "Seattle": (47.6062, -122.3321),
        "Boston": (42.3601, -71.0589), "Atlanta": (33.7490, -84.3880),
        "Miami": (25.7617, -80.1918), "Minneapolis": (44.9778, -93.2650),
        "Cleveland": (41.4993, -81.6944), "Detroit": (42.3314, -83.0458),
        "Tampa": (27.9506, -82.4572), "Baltimore": (39.2904, -76.6122),
        "Pittsburgh": (40.4406, -79.9959), "Charlotte": (35.2271, -80.8431),
        "Indianapolis": (39.7684, -86.1581), "Nashville": (36.1627, -86.7816),
        "New Orleans": (29.9511, -90.0715), "Las Vegas": (36.1699, -115.1398),
        "Kansas City": (39.0997, -94.5786), "Cincinnati": (39.1031, -84.5120),
        "Green Bay": (44.5192, -88.0198), "Jacksonville": (30.3322, -81.6557),
        "Buffalo": (42.8864, -78.8784), "Oakland": (37.8044, -122.2712),
        "Milwaukee": (43.0389, -87.9065), "Sacramento": (38.5816, -121.4944),
        "Portland": (45.5152, -122.6784), "Memphis": (35.1495, -90.0490),
        "Louisville": (38.2527, -85.7585), "St. Louis": (38.6270, -90.1994),
        "Orlando": (28.5383, -81.3792), "Salt Lake City": (40.7608, -111.8910),
        "Raleigh": (35.7796, -78.6382), "Oklahoma City": (35.4676, -97.5164),
        
        # === NFL Stadium Cities ===
        "Arlington": (32.7357, -97.1081), "Foxborough": (42.0654, -71.2481),
        "Glendale": (33.5387, -112.1860), "Inglewood": (33.9617, -118.3531),
        "East Rutherford": (40.8128, -74.0742), "Landover": (38.9076, -76.8645),
        "Orchard Park": (42.7738, -78.7870), "Paradise": (36.0908, -115.1836),
        "Santa Clara": (37.4030, -121.9700),
        
        # === College Football Cities ===
        "Tuscaloosa": (33.2098, -87.5692), "Auburn": (32.6099, -85.4808),
        "Clemson": (34.6834, -82.8374), "Columbus": (39.9612, -82.9988),
        "Ann Arbor": (42.2808, -83.7430), "Austin": (30.2672, -97.7431),
        "Norman": (35.2226, -97.4395), "Baton Rouge": (30.4515, -91.1871),
        "Gainesville": (29.6516, -82.3248), "Knoxville": (35.9606, -83.9207),
        "Madison": (43.0731, -89.4012), "Eugene": (44.0521, -123.0868),
        "State College": (40.7934, -77.8600), "South Bend": (41.6764, -86.2520),
        "College Station": (30.6280, -96.3344), "Athens": (33.9519, -83.3576),
        "Tallahassee": (30.4383, -84.2807), "Lincoln": (40.8258, -96.6852),
        "Boulder": (40.0150, -105.2705), "Ames": (42.0308, -93.6319),
        "Corvallis": (44.5646, -123.2620), "Pullman": (46.7298, -117.1817),
        "Stillwater": (36.1156, -97.0584), "Morgantown": (39.6295, -79.9559),
        "Blacksburg": (37.2296, -80.4139), "Charlottesville": (38.0293, -78.4767),
        "Durham": (35.9940, -78.8986), "Chapel Hill": (35.9132, -79.0558),
        "Waco": (31.5493, -97.1467), "Provo": (40.2338, -111.6585),
        "Lubbock": (33.5779, -101.8552), "Fort Worth": (32.7555, -97.3308),
        "Pasadena": (34.1478, -118.1445), "Tempe": (33.4255, -111.9400),
        
        # === Canadian Cities (CFL) ===
        "Toronto": (43.6532, -79.3832), "Montreal": (45.5017, -73.5673),
        "Vancouver": (49.2827, -123.1207), "Calgary": (51.0447, -114.0719),
        "Edmonton": (53.5461, -113.4938), "Ottawa": (45.4215, -75.6972),
        "Winnipeg": (49.8951, -97.1384), "Hamilton": (43.2557, -79.8711),
        "Regina": (50.4452, -104.6189), "Saskatchewan": (52.1332, -106.6700),
        
        # === Tennis ATP/WTA Venues ===
        # Grand Slams
        "Melbourne": (-37.8136, 144.9631),  # Australian Open
        "Paris": (48.8566, 2.3522),  # French Open - Roland Garros
        "London": (51.5074, -0.1278),  # Wimbledon
        "Flushing": (40.7498, -73.8463),  # US Open - Flushing Meadows
        "Wimbledon": (51.4340, -0.2143),
        
        # ATP/WTA Masters & Major Tournaments
        "Indian Wells": (33.7238, -116.3106), "Key Biscayne": (25.6940, -80.1669),
        "Monte Carlo": (43.7384, 7.4246), "Monaco": (43.7384, 7.4246),
        "Madrid": (40.4168, -3.7038), "Rome": (41.9028, 12.4964),
        "Toronto": (43.6532, -79.3832), "Montreal": (45.5017, -73.5673),
        "Cincinnati": (39.1031, -84.5120), "Shanghai": (31.2304, 121.4737),
        "Beijing": (39.9042, 116.4074), "Tokyo": (35.6762, 139.6503),
        "Dubai": (25.2048, 55.2708), "Doha": (25.2854, 51.5310),
        "Miami Gardens": (25.9579, -80.2389), "Brisbane": (-27.4698, 153.0251),
        "Sydney": (-33.8688, 151.2093), "Adelaide": (-34.9285, 138.6007),
        "Auckland": (-36.8509, 174.7645), "Buenos Aires": (-34.6037, -58.3816),
        "Rio de Janeiro": (-22.9068, -43.1729), "Sao Paulo": (-23.5505, -46.6333),
        "Acapulco": (16.8531, -99.8237), "Santiago": (-33.4489, -70.6693),
        "Barcelona": (41.3874, 2.1686), "Munich": (48.1351, 11.5820),
        "Stuttgart": (48.7758, 9.1829), "Hamburg": (53.5511, 9.9937),
        "Halle": (52.0799, 8.0324), "Queen's Club": (51.4875, -0.2141),
        "Eastbourne": (50.7684, 0.2904), "s-Hertogenbosch": (51.6978, 5.3037),
        "Bastad": (56.4267, 12.8558), "Gstaad": (46.4750, 7.2873),
        "Atlanta": (33.7490, -84.3880), "Washington": (38.9072, -77.0369),
        "Los Cabos": (22.8905, -109.9167), "Winston-Salem": (36.0999, -80.2442),
        "St. Petersburg": (59.9343, 30.3351), "Metz": (49.1193, 6.1757),
        "Astana": (51.1605, 71.4704), "Zhuhai": (22.2710, 113.5767),
        "Chengdu": (30.5728, 104.0668), "Sofia": (42.6977, 23.3219),
        "Antwerp": (51.2194, 4.4025), "Moscow": (55.7558, 37.6173),
        "Vienna": (48.2082, 16.3738), "Basel": (47.5596, 7.5886),
        "Stockholm": (59.3293, 18.0686), "Turin": (45.0703, 7.6869),
        "Malaga": (36.7213, -4.4214),
        
        # WTA Specific
        "Charleston": (32.7765, -79.9311), "Birmingham": (52.4862, -1.8904),
        "Nottingham": (52.9548, -1.1581), "San Jose": (37.3382, -121.8863),
        "Guadalajara": (20.6597, -103.3496), "Osaka": (34.6937, 135.5023),
        "Seoul": (37.5665, 126.9780), "Wuhan": (30.5928, 114.3055),
        "Zhengzhou": (34.7466, 113.6254), "Shenzhen": (22.5431, 114.0579),
        "Luxembourg": (49.6116, 6.1319), "Linz": (48.3069, 14.2858),
        "Guadalajara": (20.6597, -103.3496), "Portoroz": (45.5095, 13.5906),
        "Parma": (44.8015, 10.3279), "Cluj-Napoca": (46.7712, 23.6236),
        "Strasbourg": (48.5734, 7.7521), "Rabat": (34.0209, -6.8416),
        "Istanbul": (41.0082, 28.9784), "Prague": (50.0755, 14.4378),
        "Palermo": (38.1157, 13.3615), "Bogota": (4.7110, -74.0721),
        "Tashkent": (41.2995, 69.2401), "Nanchang": (28.6820, 115.8579),
        "Hua Hin": (12.5684, 99.9577), "Hobart": (-42.8821, 147.3272),
        
        # === International Cities ===
        "Berlin": (52.5200, 13.4050), "Frankfurt": (50.1109, 8.6821),
        "Amsterdam": (52.3676, 4.9041), "Brussels": (50.8503, 4.3517),
        "Zurich": (47.3769, 8.5417), "Geneva": (46.2044, 6.1432),
        "Milan": (45.4642, 9.1900), "Florence": (43.7696, 11.2558),
        "Naples": (40.8518, 14.2681), "Venice": (45.4408, 12.3155),
        "Lisbon": (38.7223, -9.1393), "Dublin": (53.3498, -6.2603),
        "Manchester": (53.4808, -2.2426), "Edinburgh": (55.9533, -3.1883),
        "Warsaw": (52.2297, 21.0122), "Budapest": (47.4979, 19.0402),
        "Athens": (37.9838, 23.7275), "Cairo": (30.0444, 31.2357),
        "Cape Town": (-33.9249, 18.4241), "Johannesburg": (-26.2041, 28.0473),
        "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
        "Singapore": (1.3521, 103.8198), "Hong Kong": (22.3193, 114.1694),
        "Taipei": (25.0330, 121.5654), "Bangkok": (13.7563, 100.5018),
        "Kuala Lumpur": (3.1390, 101.6869), "Jakarta": (-6.2088, 106.8456),
    }
    
    # Stadium name to coordinates (specific venues)
    STADIUM_COORDS = {
        # NFL Stadiums
        "AT&T Stadium": (32.7473, -97.0945),
        "Allegiant Stadium": (36.0909, -115.1833),
        "Arrowhead Stadium": (39.0489, -94.4839),
        "Bank of America Stadium": (35.2258, -80.8528),
        "Caesars Superdome": (29.9511, -90.0812),
        "Empower Field at Mile High": (39.7439, -105.0201),
        "Ford Field": (42.3400, -83.0456),
        "Gillette Stadium": (42.0909, -71.2643),
        "Hard Rock Stadium": (25.9580, -80.2389),
        "Highmark Stadium": (42.7738, -78.7870),
        "Huntington Bank Field": (41.5061, -81.6995),
        "Levi's Stadium": (37.4033, -121.9694),
        "Lincoln Financial Field": (39.9008, -75.1675),
        "Lucas Oil Stadium": (39.7601, -86.1639),
        "Lumen Field": (47.5952, -122.3316),
        "M&T Bank Stadium": (39.2780, -76.6227),
        "Mercedes-Benz Stadium": (33.7554, -84.4010),
        "MetLife Stadium": (40.8128, -74.0742),
        "Nissan Stadium": (36.1665, -86.7713),
        "NRG Stadium": (29.6847, -95.4107),
        "Paycor Stadium": (39.0954, -84.5160),
        "Raymond James Stadium": (27.9759, -82.5033),
        "SoFi Stadium": (33.9535, -118.3392),
        "Soldier Field": (41.8623, -87.6167),
        "State Farm Stadium": (33.5276, -112.2626),
        "TIAA Bank Field": (30.3239, -81.6373),
        "U.S. Bank Stadium": (44.9736, -93.2575),
        "FedExField": (38.9076, -76.8645),
        "Northwest Stadium": (38.9076, -76.8645),
        
        # MLB Stadiums
        "American Family Field": (43.0280, -87.9712),
        "Angel Stadium": (33.8003, -117.8827),
        "Busch Stadium": (38.6226, -90.1928),
        "Chase Field": (33.4453, -112.0667),
        "Citi Field": (40.7571, -73.8458),
        "Citizens Bank Park": (39.9061, -75.1665),
        "Comerica Park": (42.3390, -83.0485),
        "Coors Field": (39.7559, -104.9942),
        "Dodger Stadium": (34.0739, -118.2400),
        "Fenway Park": (42.3467, -71.0972),
        "Globe Life Field": (32.7473, -97.0819),
        "Great American Ball Park": (39.0974, -84.5082),
        "Guaranteed Rate Field": (41.8299, -87.6338),
        "Kauffman Stadium": (39.0517, -94.4803),
        "loanDepot park": (25.7781, -80.2196),
        "Minute Maid Park": (29.7573, -95.3555),
        "Nationals Park": (38.8730, -77.0074),
        "Oakland Coliseum": (37.7516, -122.2005),
        "Oracle Park": (37.7786, -122.3893),
        "Oriole Park at Camden Yards": (39.2838, -76.6217),
        "Petco Park": (32.7076, -117.1570),
        "PNC Park": (40.4469, -80.0057),
        "Progressive Field": (41.4962, -81.6852),
        "Rogers Centre": (43.6414, -79.3894),
        "T-Mobile Park": (47.5914, -122.3325),
        "Target Field": (44.9817, -93.2776),
        "Tropicana Field": (27.7682, -82.6534),
        "Truist Park": (33.8908, -84.4678),
        "Wrigley Field": (41.9484, -87.6553),
        "Yankee Stadium": (40.8296, -73.9262),
        
        # Tennis Venues
        "Melbourne Park": (-37.8215, 144.9783),
        "Roland Garros": (48.8469, 2.2528),
        "All England Club": (51.4340, -0.2143),
        "USTA Billie Jean King National Tennis Center": (40.7498, -73.8463),
        "Indian Wells Tennis Garden": (33.7238, -116.3106),
        "Crandon Park Tennis Center": (25.7022, -80.1516),
        "Foro Italico": (41.9283, 12.4575),
        "Caja Mágica": (40.3722, -3.6878),
        "Monte-Carlo Country Club": (43.7515, 7.4420),
        "Stade Roland Garros": (48.8469, 2.2528),
        "Rod Laver Arena": (-37.8215, 144.9783),
    }

    
    def _get_city_coords(self, city: str, state: str = None, stadium_name: str = None) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for a city/stadium from cache or geocoding."""
        # Try stadium name first (most accurate)
        if stadium_name:
            # Check exact stadium match
            if stadium_name in self.STADIUM_COORDS:
                return self.STADIUM_COORDS[stadium_name]
            # Try partial stadium match
            for cached_stadium, coords in self.STADIUM_COORDS.items():
                if cached_stadium.lower() in stadium_name.lower() or stadium_name.lower() in cached_stadium.lower():
                    return coords
        
        if not city:
            return None, None
        
        # Try exact city match
        if city in self.CITY_COORDS:
            return self.CITY_COORDS[city]
        
        # Try partial city match
        for cached_city, coords in self.CITY_COORDS.items():
            if cached_city.lower() in city.lower() or city.lower() in cached_city.lower():
                return coords
        
        # Try with state (e.g., "Columbus, Ohio" vs "Columbus, Georgia")
        if state:
            city_state = f"{city}, {state}"
            for cached_city, coords in self.CITY_COORDS.items():
                if cached_city.lower() == city.lower():
                    return coords
        
        return None, None
    
    async def _geocode_location(self, city: str, state: str = None, country: str = "USA") -> Tuple[Optional[float], Optional[float]]:
        """Geocode a location using Nominatim (free, no API key)."""
        if not city:
            return None, None
        
        try:
            query = city
            if state:
                query = f"{city}, {state}"
            if country:
                query = f"{query}, {country}"
            
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": query,
                "format": "json",
                "limit": 1,
            }
            headers = {
                "User-Agent": "Royaley Sports Analytics/1.0"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        lat = float(data[0].get("lat", 0))
                        lon = float(data[0].get("lon", 0))
                        if lat and lon:
                            logger.info(f"[SportsDB] Geocoded {query}: ({lat}, {lon})")
                            return lat, lon
        except Exception as e:
            logger.debug(f"[SportsDB] Geocoding failed for {city}: {e}")
        
        return None, None
    
    async def collect_venues(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect venues via /lookup/team for each team, including lat/lon.
        
        Enhanced version with:
        - Stadium coordinate lookup
        - City coordinate fallback  
        - Nominatim geocoding fallback
        - SportsDB API lat/lon extraction
        """
        sports = [sport_code] if sport_code else ML_SPORTS
        all_venues = []
        team_cities = []  # Track team-city mappings for updating teams
        geocoded_count = 0
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._get_list(data, "list", "teams")
            
            print(f"[SportsDB] {sport}: Found {len(teams)} teams, fetching stadium details...")
            logger.info(f"[SportsDB] {sport}: Found {len(teams)} teams")
            
            seen = set()
            venue_count = 0
            
            for t in teams:
                team_id = t.get("idTeam")
                if not team_id:
                    continue
                
                details = await self._get_team_details(team_id)
                if not details:
                    await asyncio.sleep(0.1)
                    continue
                
                # Extract location info
                loc = details.get("strLocation") or ""
                city = loc.split(",")[0].strip() if loc else None
                state = loc.split(",")[1].strip() if loc and "," in loc else None
                country = details.get("strCountry") or "USA"
                
                # Track team-city mapping
                team_name = details.get("strTeam")
                if team_name and city:
                    team_cities.append({
                        "team_name": team_name,
                        "city": city,
                        "sport_code": sport,
                    })
                
                # Get stadium name
                name = details.get("strStadium")
                if not name or name in seen:
                    await asyncio.sleep(0.1)
                    continue
                seen.add(name)
                
                # Get capacity
                cap = None
                cap_str = details.get("intStadiumCapacity")
                if cap_str:
                    try:
                        cap = int(str(cap_str).replace(",", ""))
                    except:
                        pass
                
                # Try to get lat/lon from multiple sources
                lat, lon = None, None
                
                # 1. Check SportsDB API fields first (strMapLat, strMapLong, etc.)
                api_lat = details.get("strStadiumLat") or details.get("strMapLat") or details.get("strLat")
                api_lon = details.get("strStadiumLon") or details.get("strMapLon") or details.get("strLon") or details.get("strMapLong") or details.get("strLong")
                if api_lat and api_lon:
                    try:
                        lat = float(api_lat)
                        lon = float(api_lon)
                        logger.debug(f"[SportsDB] Got coords from API for {name}: ({lat}, {lon})")
                    except:
                        pass
                
                # 2. Try stadium name lookup
                if not lat or not lon:
                    lat, lon = self._get_city_coords(city, state, stadium_name=name)
                
                # 3. Try city lookup
                if not lat or not lon:
                    lat, lon = self._get_city_coords(city, state)
                
                # 4. Geocoding fallback (rate limited)
                if not lat or not lon:
                    lat, lon = await self._geocode_location(city, state, country)
                    if lat and lon:
                        geocoded_count += 1
                        # Add to cache for future use
                        if city and city not in self.CITY_COORDS:
                            self.CITY_COORDS[city] = (lat, lon)
                    await asyncio.sleep(1.0)  # Rate limit geocoding (1 req/sec)
                else:
                    await asyncio.sleep(0.15)  # Normal rate limit
                
                venue_data = {
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": country,
                    "capacity": cap,
                    "is_dome": self._is_dome(name),
                    "latitude": lat,
                    "longitude": lon,
                    "team_name": team_name,
                    "external_id": f"sportsdb_venue_{details.get('idVenue') or team_id}",
                    "sport_code": sport,
                    "sportsdb_team_id": team_id,
                }
                
                all_venues.append(venue_data)
                venue_count += 1
                
                # Log progress
                coords_status = f"({lat:.4f}, {lon:.4f})" if lat and lon else "NO COORDS"
                logger.debug(f"[SportsDB] {sport}: {name} in {city} - {coords_status}")
            
            logger.info(f"[SportsDB] {sport}: {venue_count} venues collected")
            print(f"[SportsDB] {sport}: {venue_count} venues collected")
        
        # Summary stats
        with_coords = sum(1 for v in all_venues if v.get("latitude") and v.get("longitude"))
        print(f"[SportsDB] Total: {len(all_venues)} venues, {with_coords} with coordinates, {geocoded_count} geocoded")
        logger.info(f"[SportsDB] Total: {len(all_venues)} venues, {with_coords} with coordinates, {geocoded_count} geocoded")
        
        return {"venues": all_venues, "team_cities": team_cities, "count": len(all_venues), "with_coords": with_coords}
    
    def _is_dome(self, name: str) -> bool:
        check = (name or "").lower()
        return any(d in check for d in ["dome", "arena", "center", "centre", "garden", "indoor", "fieldhouse", "forum", "palace"])

    # =========================================================================
    # PLAYERS COLLECTION - FIXED
    # =========================================================================
    
    async def collect_players(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect players via /list/players/{teamId} - FIXED version."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_players = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # Skip tennis (individual sport)
            if sport in ["ATP", "WTA"]:
                print(f"[SportsDB] {sport}: Skipping (individual sport, no team rosters)")
                continue
            
            # Get teams first
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._get_list(data, "list", "teams")
            
            print(f"[SportsDB] {sport}: Fetching players for {len(teams)} teams...")
            
            sport_player_count = 0
            
            for t in teams:
                team_id = t.get("idTeam")
                team_name = t.get("strTeam")
                if not team_id:
                    continue
                
                player_data = await self._v2(f"/list/players/{team_id}")
                
                # DEBUG: Log raw response structure
                if player_data:
                    print(f"[SportsDB] {team_name}: Response keys = {list(player_data.keys()) if isinstance(player_data, dict) else 'list'}")
                
                # FIXED: Use _get_list which now handles both "player" and "players" keys
                players = self._get_list(player_data, "player", "players", "list")
                
                if not players:
                    print(f"[SportsDB] {team_name}: No players found")
                    continue
                
                print(f"[SportsDB] {team_name}: {len(players)} players")
                
                for p in players:
                    player_entry = {
                        "external_id": f"sportsdb_{p.get('idPlayer')}",
                        "sportsdb_id": p.get("idPlayer"),
                        "name": p.get("strPlayer"),
                        "team_name": team_name,
                        "team_id": team_id,
                        "position": p.get("strPosition"),
                        "number": p.get("strNumber"),
                        "nationality": p.get("strNationality"),
                        "birth_date": p.get("dateBorn"),
                        "height": p.get("strHeight"),
                        "weight": p.get("strWeight"),
                        "photo_url": p.get("strCutout") or p.get("strThumb"),
                        "sport_code": sport,
                    }
                    all_players.append(player_entry)
                    sport_player_count += 1
                
                await asyncio.sleep(0.15)
            
            logger.info(f"[SportsDB] {sport}: {sport_player_count} players collected")
            print(f"[SportsDB] {sport}: {sport_player_count} players total")
        
        print(f"[SportsDB] TOTAL: {len(all_players)} players across all sports")
        return {"players": all_players, "count": len(all_players)}

    # =========================================================================
    # SCHEDULE / GAMES - Current
    # =========================================================================
    
    async def _collect_schedule(self, sport_code: str) -> List[Dict]:
        """Collect upcoming games."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/schedule/next/league/{league_id}")
        events = self._get_list(data, "schedule", "events")
        
        games = [self._parse_event(e, sport_code) for e in events]
        games = [g for g in games if g]
        
        logger.info(f"[SportsDB] {sport_code}: {len(games)} upcoming games")
        return games
    
    async def _collect_past_games(self, sport_code: str) -> List[Dict]:
        """Collect recent results."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/schedule/previous/league/{league_id}")
        events = self._get_list(data, "schedule", "events")
        
        games = [self._parse_event(e, sport_code) for e in events]
        return [g for g in games if g]
    
    async def _collect_livescores(self, sport_code: str) -> List[Dict]:
        """Collect live scores."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/livescore/{league_id}")
        events = self._get_list(data, "livescore", "events")
        
        result = []
        for e in events:
            result.append({
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "home_team": e.get("strHomeTeam"),
                "away_team": e.get("strAwayTeam"),
                "home_score": self._safe_int(e.get("intHomeScore")),
                "away_score": self._safe_int(e.get("intAwayScore")),
                "status": STATUS_MAP.get(e.get("strStatus"), "in_progress"),
                "sport_code": sport_code,
            })
        return result
    
    def _parse_event(self, e: Dict, sport_code: str) -> Optional[Dict]:
        """Parse event to game dict."""
        try:
            date_str = e.get("dateEvent", "")
            if not date_str:
                return None
            
            time_str = e.get("strTime", "00:00:00")
            if not time_str:
                time_str = "00:00:00"
            
            # Handle time format
            time_str = time_str.replace("+00:00", "").strip()
            if len(time_str) == 5:  # HH:MM
                time_str += ":00"
            
            try:
                scheduled = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except:
                scheduled = datetime.strptime(date_str, "%Y-%m-%d")
            
            return {
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "home_team": e.get("strHomeTeam"),
                "away_team": e.get("strAwayTeam"),
                "scheduled_time": scheduled,
                "home_score": self._safe_int(e.get("intHomeScore")),
                "away_score": self._safe_int(e.get("intAwayScore")),
                "venue": e.get("strVenue"),
                "round": e.get("intRound"),
                "season": e.get("strSeason"),
                "sport_code": sport_code,
            }
        except Exception as ex:
            logger.debug(f"[SportsDB] Parse error: {ex}")
            return None
    
    def _safe_int(self, val) -> Optional[int]:
        if val is None or val == "":
            return None
        try:
            return int(val)
        except:
            return None

    # =========================================================================
    # HISTORICAL DATA (Multi-season)
    # =========================================================================
    
    async def collect_historical(self, sport_code: str, seasons_back: int = 10) -> CollectorResult:
        """Collect historical games for multiple seasons."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return CollectorResult(success=False, data={"games": []}, records_count=0)
        
        # Get available seasons
        seasons = await self.get_available_seasons(sport_code)
        if not seasons:
            print(f"[SportsDB] {sport_code}: No seasons found")
            return CollectorResult(success=False, data={"games": []}, records_count=0)
        
        # Limit to requested number of seasons
        seasons = seasons[:seasons_back]
        print(f"[SportsDB] {sport_code}: Collecting {len(seasons)} seasons: {seasons}")
        
        all_games = []
        for season in seasons:
            try:
                data = await self._v2(f"/schedule/league/{league_id}/{season}")
                events = self._get_list(data, "schedule", "events")
                
                games = [self._parse_event(e, sport_code) for e in events]
                games = [g for g in games if g]
                all_games.extend(games)
                
                print(f"[SportsDB] {sport_code} {season}: {len(games)} games")
                await asyncio.sleep(0.5)  # Be nice to the API
            except Exception as e:
                print(f"[SportsDB] {sport_code} {season}: Error - {e}")
        
        print(f"[SportsDB] {sport_code}: {len(all_games)} total historical games")
        return CollectorResult(success=len(all_games) > 0, data={"games": all_games}, records_count=len(all_games))

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_venues_to_database(self, venues: List[Dict], session: AsyncSession) -> int:
        """Save venues with lat/lon and also update team cities."""
        try:
            from app.models import Venue
        except ImportError:
            try:
                from app.models.models import Venue
            except:
                logger.error("[SportsDB] Venue model not found")
                return 0
        
        saved = 0
        updated = 0
        for v in venues:
            try:
                name = v.get("name")
                if not name:
                    continue
                
                existing = await session.execute(select(Venue).where(Venue.name == name))
                venue = existing.scalar_one_or_none()
                
                if venue:
                    venue.city = v.get("city") or venue.city
                    venue.state = v.get("state") or venue.state
                    venue.country = v.get("country") or venue.country
                    venue.capacity = v.get("capacity") or venue.capacity
                    venue.is_dome = v.get("is_dome", venue.is_dome)
                    # Update lat/lon if provided
                    if v.get("latitude"):
                        venue.latitude = v.get("latitude")
                    if v.get("longitude"):
                        venue.longitude = v.get("longitude")
                    updated += 1
                else:
                    venue = Venue(
                        name=name[:200],
                        city=(v.get("city") or "")[:100] or None,
                        state=(v.get("state") or "")[:50] or None,
                        country=(v.get("country") or "USA")[:50],
                        capacity=v.get("capacity"),
                        is_dome=v.get("is_dome", False),
                        latitude=v.get("latitude"),
                        longitude=v.get("longitude"),
                    )
                    session.add(venue)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Venue save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} new venues, updated {updated}")
        print(f"[SportsDB] Saved {saved} new venues, updated {updated}")
        return saved + updated
    
    async def update_team_cities_from_venues(self, team_cities: List[Dict], session: AsyncSession) -> int:
        """Update team cities from venue data."""
        updated = 0
        for tc in team_cities:
            try:
                team_name = tc.get("team_name")
                city = tc.get("city")
                sport_code = tc.get("sport_code")
                
                if not team_name or not city:
                    continue
                
                # Get sport
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                # Find team and update city
                team_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
                )
                team = team_result.scalar_one_or_none()
                
                if team and not team.city:
                    team.city = city
                    updated += 1
                    
            except Exception as e:
                logger.debug(f"[SportsDB] Team city update error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Updated {updated} team cities")
        print(f"[SportsDB] Updated {updated} team cities")
        return updated
    
    async def link_venues_to_teams(self, session: AsyncSession) -> int:
        """Link venues to teams based on team name matching."""
        try:
            from app.models import Venue, Team
        except ImportError:
            from app.models.models import Venue, Team
        
        linked = 0
        try:
            # Get all venues with team data
            venues_result = await session.execute(select(Venue))
            venues = venues_result.scalars().all()
            
            # Create venue lookup by various keys
            venue_lookup = {}
            for v in venues:
                venue_lookup[v.name.lower()] = v
                if v.city:
                    venue_lookup[v.city.lower()] = v
            
            # Get all teams without venue_id
            teams_result = await session.execute(select(Team).where(Team.venue_id == None))
            teams = teams_result.scalars().all()
            
            for team in teams:
                # Try to match by team city
                if team.city:
                    city_key = team.city.lower()
                    if city_key in venue_lookup:
                        team.venue_id = venue_lookup[city_key].id
                        linked += 1
                        continue
                
                # Try to match by team name (e.g., "Dallas Cowboys" -> "AT&T Stadium" in Arlington/Dallas)
                team_city = team.name.split()[0].lower() if team.name else None
                if team_city and team_city in venue_lookup:
                    team.venue_id = venue_lookup[team_city].id
                    linked += 1
            
            await session.commit()
            logger.info(f"[SportsDB] Linked {linked} teams to venues")
            print(f"[SportsDB] Linked {linked} teams to venues")
        except Exception as e:
            logger.warning(f"[SportsDB] Link venues to teams error: {e}")
        
        return linked
    
    async def link_venues_to_games(self, session: AsyncSession) -> int:
        """Link venues to games based on home team's venue."""
        try:
            from app.models import Venue, Team
        except ImportError:
            from app.models.models import Venue, Team
        
        linked = 0
        try:
            # Get all games without venue_id
            games_result = await session.execute(
                select(Game).where(Game.venue_id == None)
            )
            games = games_result.scalars().all()
            
            if not games:
                logger.info("[SportsDB] No games without venue_id found")
                return 0
            
            # Build team -> venue lookup
            teams_result = await session.execute(
                select(Team).where(Team.venue_id != None)
            )
            teams = teams_result.scalars().all()
            
            team_venue_map = {}
            for team in teams:
                team_venue_map[team.id] = team.venue_id
                if team.name:
                    team_venue_map[team.name.lower()] = team.venue_id
            
            # Also build venue lookup by city for fallback
            venues_result = await session.execute(
                select(Venue).where(and_(Venue.latitude != None, Venue.longitude != None))
            )
            venues = venues_result.scalars().all()
            
            venue_by_city = {}
            for v in venues:
                if v.city:
                    venue_by_city[v.city.lower()] = v.id
            
            for game in games:
                venue_id = None
                
                # Try home team's venue
                if game.home_team_id and game.home_team_id in team_venue_map:
                    venue_id = team_venue_map[game.home_team_id]
                
                # Try by home team name lookup
                if not venue_id and hasattr(game, 'home_team') and game.home_team:
                    team_name = game.home_team.name.lower() if hasattr(game.home_team, 'name') else None
                    if team_name and team_name in team_venue_map:
                        venue_id = team_venue_map[team_name]
                
                if venue_id:
                    game.venue_id = venue_id
                    linked += 1
            
            await session.commit()
            logger.info(f"[SportsDB] Linked {linked} games to venues")
            print(f"[SportsDB] Linked {linked} games to venues")
        except Exception as e:
            logger.warning(f"[SportsDB] Link venues to games error: {e}")
        
        return linked
    
    async def save_players_to_database(self, players: List[Dict], session: AsyncSession) -> int:
        """Save players - FIXED with better error handling."""
        try:
            from app.models import Player
        except ImportError:
            try:
                from app.models.models import Player
            except:
                logger.error("[SportsDB] Player model not found")
                print("[SportsDB] ❌ Player model not found!")
                return 0
        
        saved = 0
        skipped_no_ext_id = 0
        skipped_no_sport = 0
        skipped_error = 0
        
        for p in players:
            try:
                ext_id = p.get("external_id")
                if not ext_id:
                    skipped_no_ext_id += 1
                    continue
                
                existing = await session.execute(select(Player).where(Player.external_id == ext_id))
                player = existing.scalar_one_or_none()
                
                if not player:
                    # Get team
                    sport_code = p.get("sport_code")
                    sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                    sport = sport_result.scalar_one_or_none()
                    if not sport:
                        skipped_no_sport += 1
                        continue
                    
                    team_result = await session.execute(
                        select(Team).where(and_(Team.sport_id == sport.id, Team.name == p.get("team_name")))
                    )
                    team = team_result.scalar_one_or_none()
                    
                    # Parse jersey number
                    jersey = None
                    if p.get("number"):
                        try:
                            jersey = int(str(p.get("number")).strip())
                        except:
                            pass
                    
                    player = Player(
                        external_id=ext_id,
                        name=p.get("name", "Unknown")[:200],
                        team_id=team.id if team else None,
                        position=p.get("position"),
                        jersey_number=jersey,
                    )
                    session.add(player)
                    saved += 1
            except Exception as e:
                skipped_error += 1
                logger.debug(f"[SportsDB] Player save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} new players (skipped: {skipped_no_ext_id} no ext_id, {skipped_no_sport} no sport, {skipped_error} errors)")
        print(f"[SportsDB] Saved {saved} new players (skipped: {skipped_no_ext_id} no ext_id, {skipped_no_sport} no sport, {skipped_error} errors)")
        return saved
    
    async def save_teams_to_database(self, teams: List[Dict], session: AsyncSession) -> int:
        """Save teams with city data."""
        saved = 0
        updated = 0
        for t in teams:
            try:
                sport_code = t.get("sport_code")
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                name = t.get("name")
                if not name:
                    continue
                
                existing = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == name))
                )
                team = existing.scalar_one_or_none()
                
                if team:
                    # Update existing team
                    team.abbreviation = t.get("abbreviation") or team.abbreviation
                    team.logo_url = t.get("logo_url") or team.logo_url
                    # Update city if provided and not already set
                    if t.get("city") and not team.city:
                        team.city = t.get("city")
                    # Update conference/division if provided
                    if t.get("conference"):
                        team.conference = t.get("conference")
                    if t.get("division"):
                        team.division = t.get("division")
                    updated += 1
                else:
                    team = Team(
                        external_id=t.get("external_id") or f"sportsdb_{t.get('sportsdb_id')}",
                        name=name,
                        abbreviation=(t.get("abbreviation") or "UNK")[:10],
                        sport_id=sport.id,
                        logo_url=t.get("logo_url"),
                        city=t.get("city"),
                        conference=t.get("conference"),
                        division=t.get("division"),
                    )
                    session.add(team)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Team save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} new teams, updated {updated}")
        print(f"[SportsDB] Saved {saved} new teams, updated {updated}")
        return saved + updated
    
    async def save_games_to_database(self, games: List[Dict], session: AsyncSession) -> int:
        """Save games - auto-creates teams, handles duplicates properly."""
        saved = 0
        updated = 0
        skipped_no_sport = 0
        skipped_no_team = 0
        teams_created = 0
        
        for g in games:
            try:
                sport_code = g.get("sport_code")
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    skipped_no_sport += 1
                    continue
                
                home_name = g.get("home_team", "")
                away_name = g.get("away_team", "")
                
                if not home_name or not away_name:
                    skipped_no_team += 1
                    continue
                
                # Get or create home team
                home_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == home_name))
                )
                home_team = home_result.scalar_one_or_none()
                
                if not home_team:
                    # Try partial match (contains)
                    home_result = await session.execute(
                        select(Team).where(and_(Team.sport_id == sport.id, Team.name.ilike(f"%{home_name}%"))).limit(1)
                    )
                    home_team = home_result.scalar_one_or_none()
                
                if not home_team:
                    # Auto-create team
                    home_team = Team(
                        name=home_name,
                        sport_id=sport.id,
                        external_id=f"sportsdb_{sport_code}_{home_name.replace(' ', '_').lower()}",
                    )
                    session.add(home_team)
                    await session.flush()
                    teams_created += 1
                
                # Get or create away team
                away_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == away_name))
                )
                away_team = away_result.scalar_one_or_none()
                
                if not away_team:
                    # Try partial match (contains)
                    away_result = await session.execute(
                        select(Team).where(and_(Team.sport_id == sport.id, Team.name.ilike(f"%{away_name}%"))).limit(1)
                    )
                    away_team = away_result.scalar_one_or_none()
                
                if not away_team:
                    # Auto-create team
                    away_team = Team(
                        name=away_name,
                        sport_id=sport.id,
                        external_id=f"sportsdb_{sport_code}_{away_name.replace(' ', '_').lower()}",
                    )
                    session.add(away_team)
                    await session.flush()
                    teams_created += 1
                
                if not home_team or not away_team:
                    skipped_no_team += 1
                    continue
                
                ext_id = g.get("external_id")
                
                # Check if game exists
                existing = await session.execute(select(Game).where(Game.external_id == ext_id))
                game = existing.scalar_one_or_none()
                
                if game:
                    # Update existing game
                    game.home_score = g.get("home_score")
                    game.away_score = g.get("away_score")
                    game.scheduled_at = g.get("scheduled_time")
                    updated += 1
                else:
                    # Create new game
                    try:
                        game = Game(
                            external_id=ext_id,
                            sport_id=sport.id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            scheduled_at=g.get("scheduled_time"),
                            home_score=g.get("home_score"),
                            away_score=g.get("away_score"),
                        )
                        session.add(game)
                        await session.flush()
                        saved += 1
                    except Exception as insert_err:
                        # Handle race condition - game was inserted by another process
                        await session.rollback()
                        # Re-fetch and update
                        existing = await session.execute(select(Game).where(Game.external_id == ext_id))
                        game = existing.scalar_one_or_none()
                        if game:
                            game.home_score = g.get("home_score")
                            game.away_score = g.get("away_score")
                            updated += 1
                        
            except Exception as e:
                logger.warning(f"[SportsDB] Game save error: {e}")
        
        try:
            await session.commit()
        except Exception as commit_err:
            logger.error(f"[SportsDB] Commit error: {commit_err}")
            await session.rollback()
        
        logger.info(f"[SportsDB] Saved {saved} new, updated {updated} (teams created: {teams_created}, skipped: {skipped_no_sport} no sport, {skipped_no_team} no team)")
        print(f"[SportsDB] Saved {saved} new, updated {updated}, created {teams_created} teams")
        return saved + updated
    
    async def save_historical_to_database(self, games: List[Dict], session: AsyncSession) -> Tuple[int, int]:
        """Save historical games."""
        saved = await self.save_games_to_database(games, session)
        return saved, 0
    
    async def _update_livescores(self, data: Dict, session: AsyncSession) -> bool:
        """Update live scores."""
        for s in data.get("livescores", []):
            try:
                ext_id = s.get("external_id")
                result = await session.execute(select(Game).where(Game.external_id == ext_id))
                game = result.scalar_one_or_none()
                if game:
                    game.home_score = s.get("home_score")
                    game.away_score = s.get("away_score")
            except:
                pass
        await session.commit()
        return True
    
    async def has_data_available(self, sport_code: str = None) -> bool:
        """Check API availability."""
        data = await self._v2(f"/list/teams/{SPORTSDB_LEAGUE_IDS.get('NFL', 4391)}")
        return bool(data)
    
    async def collect_past_games(self, sport_code: str, days_back: int = 30) -> List[Dict]:
        return await self._collect_past_games(sport_code)
    
    async def collect_upcoming(self, sport_code: str) -> List[Dict]:
        """Collect upcoming games for a sport."""
        return await self._collect_schedule(sport_code)
    
    async def collect_past(self, sport_code: str) -> List[Dict]:
        """Collect recent results for a sport."""
        return await self._collect_past_games(sport_code)


# Singleton
sportsdb_collector = SportsDBCollector()

try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass