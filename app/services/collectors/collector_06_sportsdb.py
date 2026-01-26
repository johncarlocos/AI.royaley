"""
ROYALEY - TheSportsDB Collector (V2 Premium)
Phase 1: Data Collection Services

PREMIUM USER FIX ($295 subscription):
- Uses V2 API with X-API-KEY header authentication
- API Key: 688655 (user's premium key)
- Base URL: https://www.thesportsdb.com/api/v2/json

V2 API Endpoints (Premium):
- /list/teams/{league_id} - List all teams in league
- /lookup/team/{team_id} - Lookup team details
- /schedule/league/{league_id} - League schedule
- /schedule/next/league/{league_id} - Upcoming events
- /schedule/previous/league/{league_id} - Past events
- /livescore/{league_id} - Live scores
"""

import asyncio
import logging
import httpx
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

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


# TheSportsDB League IDs for supported sports
SPORTSDB_LEAGUE_IDS = {
    "NFL": 4391,
    "NBA": 4387,
    "NHL": 4380,
    "MLB": 4424,
    "NCAAF": 4479,
    "NCAAB": 4607,
    "WNBA": 4962,
    "CFL": 4406,
    "ATP": 4464,
    "WTA": 4465,
}

# Sport names in TheSportsDB
SPORTSDB_SPORT_NAMES = {
    "NFL": "American Football",
    "NCAAF": "American Football",
    "CFL": "American Football",
    "NBA": "Basketball",
    "NCAAB": "Basketball",
    "WNBA": "Basketball",
    "NHL": "Ice Hockey",
    "MLB": "Baseball",
    "ATP": "Tennis",
    "WTA": "Tennis",
}

# Game status mapping
STATUS_MAP = {
    "NS": "scheduled",
    "1H": "in_progress",
    "2H": "in_progress",
    "HT": "in_progress",
    "Q1": "in_progress",
    "Q2": "in_progress",
    "Q3": "in_progress",
    "Q4": "in_progress",
    "OT": "in_progress",
    "BT": "in_progress",
    "PT": "in_progress",
    "ET": "in_progress",
    "FT": "final",
    "AOT": "final",
    "AET": "final",
    "AP": "final",
    "CANC": "cancelled",
    "PST": "postponed",
    "ABD": "cancelled",
    "AWD": "final",
    "WO": "final",
    "Match Finished": "final",
    "Not Started": "scheduled",
    "": "scheduled",
}


class SportsDBCollector(BaseCollector):
    """
    Collector for TheSportsDB API - V2 Premium Version.
    
    Uses V2 API with X-API-KEY header authentication.
    Premium features: Livescores, video highlights, full access.
    """
    
    def __init__(self):
        super().__init__(
            name="sportsdb",
            base_url="https://www.thesportsdb.com/api/v2/json",
            rate_limit=120,  # Business tier: 120/min
            rate_window=60,
            timeout=30.0,
            max_retries=3,
        )
        # User's premium API key
        self.api_key = settings.SPORTSDB_API_KEY or "688655"
        # V1 base URL for fallback
        self.v1_base_url = f"https://www.thesportsdb.com/api/v1/json/{self.api_key}"
        logger.info(f"[SportsDB] Initialized with premium API key: {self.api_key}")
        
    def _get_headers(self) -> Dict[str, str]:
        """Get V2 API headers with X-API-KEY authentication."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }
    
    async def _request_v2(self, endpoint: str) -> Optional[Dict]:
        """
        Make a V2 API request with header authentication.
        
        Args:
            endpoint: V2 endpoint path (e.g., "/list/teams/4391")
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        logger.info(f"[SportsDB] 🌐 V2 request: {url}")
        print(f"[SportsDB] 🌐 V2 request: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                logger.info(f"[SportsDB] 📥 V2 Response: {response.status_code}")
                print(f"[SportsDB] 📥 V2 Response: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.error(f"[SportsDB] V2 API 401 Unauthorized - check API key")
                    return None
                elif response.status_code == 429:
                    logger.warning(f"[SportsDB] V2 API rate limited")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.warning(f"[SportsDB] V2 API returned {response.status_code}: {response.text[:200]}")
                    return None
                    
        except Exception as e:
            logger.error(f"[SportsDB] V2 API error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _request_v1(self, endpoint: str) -> Optional[Dict]:
        """
        Make a V1 API request (fallback).
        
        Args:
            endpoint: V1 endpoint path (e.g., "lookup_all_teams.php?id=4391")
        """
        url = f"{self.v1_base_url}/{endpoint}"
        
        logger.info(f"[SportsDB] 🌐 V1 fallback: {url}")
        print(f"[SportsDB] 🌐 V1 fallback: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                logger.info(f"[SportsDB] 📥 V1 Response: {response.status_code}")
                print(f"[SportsDB] 📥 V1 Response: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"[SportsDB] V1 API returned {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"[SportsDB] V1 API error: {e}")
            return None

    async def collect(
        self,
        sport_code: str = None,
        collect_type: str = "all",
        days_ahead: int = 7,
        **kwargs,
    ) -> CollectorResult:
        """Collect data from TheSportsDB."""
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(SPORTSDB_LEAGUE_IDS.keys())
        )
        
        all_data = {
            "games": [],
            "teams": [],
            "livescores": [],
        }
        
        for sport in sports_to_collect:
            if sport not in SPORTSDB_LEAGUE_IDS:
                continue
                
            try:
                if collect_type in ("all", "schedule", "games"):
                    games = await self._collect_schedule(sport, days_ahead)
                    all_data["games"].extend(games)
                    
                if collect_type in ("all", "teams"):
                    teams = await self._collect_teams(sport)
                    all_data["teams"].extend(teams)
                    
                if collect_type in ("all", "livescores"):
                    live = await self._collect_livescores(sport)
                    all_data["livescores"].extend(live)
                    
            except Exception as e:
                logger.error(f"[SportsDB] Error collecting {sport}: {e}")
        
        total = len(all_data["games"]) + len(all_data["teams"]) + len(all_data["livescores"])
        
        return CollectorResult(
            source="sportsdb",
            success=total > 0,
            data=all_data,
            records_count=total,
        )

    # =========================================================================
    # TEAMS COLLECTION (for Venues)
    # =========================================================================
    
    async def _collect_teams(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect team information using V2 API."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        teams = []
        
        # Try V2 API first: /list/teams/{league_id}
        data = await self._request_v2(f"/list/teams/{league_id}")
        
        if data and data.get("teams"):
            for team_data in data["teams"]:
                team = self._parse_team(team_data, sport_code)
                if team:
                    teams.append(team)
            logger.info(f"[SportsDB] V2: Collected {len(teams)} teams for {sport_code}")
        else:
            # Fallback to V1
            data = await self._request_v1(f"lookup_all_teams.php?id={league_id}")
            if data and data.get("teams"):
                for team_data in data["teams"]:
                    team = self._parse_team(team_data, sport_code)
                    if team:
                        teams.append(team)
                logger.info(f"[SportsDB] V1: Collected {len(teams)} teams for {sport_code}")
        
        return teams
    
    def _parse_team(self, team_data: Dict, sport_code: str) -> Optional[Dict[str, Any]]:
        """Parse team data from API response."""
        try:
            return {
                "external_id": f"sportsdb_{team_data.get('idTeam')}",
                "name": team_data.get("strTeam"),
                "abbreviation": (team_data.get("strTeamShort") or "")[:10],
                "city": (team_data.get("strStadiumLocation") or "").split(",")[0].strip() or None,
                "conference": team_data.get("strDivision"),
                "division": team_data.get("strDivision"),
                "logo_url": team_data.get("strTeamBadge") or team_data.get("strBadge"),
                "sport_code": sport_code,
                "stadium": team_data.get("strStadium"),
            }
        except Exception as e:
            logger.debug(f"[SportsDB] Error parsing team: {e}")
            return None

    # =========================================================================
    # SCHEDULE COLLECTION
    # =========================================================================
    
    async def _collect_schedule(self, sport_code: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Collect upcoming game schedule."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        # Try V2 API first: /schedule/next/league/{league_id}
        data = await self._request_v2(f"/schedule/next/league/{league_id}")
        
        if data:
            events = data.get("schedule") or data.get("events") or []
            for event in events:
                game = self._parse_event(event, sport_code)
                if game:
                    games.append(game)
            logger.info(f"[SportsDB] V2: Collected {len(games)} upcoming games for {sport_code}")
        else:
            # Fallback to V1
            data = await self._request_v1(f"eventsnextleague.php?id={league_id}")
            if data and data.get("events"):
                for event in data["events"]:
                    game = self._parse_event(event, sport_code)
                    if game:
                        games.append(game)
                logger.info(f"[SportsDB] V1: Collected {len(games)} upcoming games for {sport_code}")
        
        return games
    
    async def _collect_results(self, sport_code: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Collect recent game results."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        # Try V2 API first
        data = await self._request_v2(f"/schedule/previous/league/{league_id}")
        
        if data:
            events = data.get("schedule") or data.get("events") or []
            for event in events[:limit]:
                game = self._parse_event(event, sport_code)
                if game:
                    games.append(game)
            logger.info(f"[SportsDB] V2: Collected {len(games)} past games for {sport_code}")
        else:
            # Fallback to V1
            data = await self._request_v1(f"eventspastleague.php?id={league_id}")
            if data and data.get("events"):
                for event in data["events"][:limit]:
                    game = self._parse_event(event, sport_code)
                    if game:
                        games.append(game)
                logger.info(f"[SportsDB] V1: Collected {len(games)} past games for {sport_code}")
        
        return games
    
    def _parse_event(self, event: Dict, sport_code: str) -> Optional[Dict[str, Any]]:
        """Parse event data."""
        try:
            date_str = event.get("dateEvent", "")
            time_str = event.get("strTime", "00:00:00") or "00:00:00"
            
            if not date_str:
                return None
            
            try:
                if "+" in time_str:
                    time_str = time_str.split("+")[0]
                time_str = time_str.replace("Z", "").strip() or "00:00:00"
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                scheduled_time = dt.replace(tzinfo=timezone.utc)
            except:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    scheduled_time = dt.replace(tzinfo=timezone.utc)
                except:
                    return None
            
            home_score = None
            away_score = None
            if event.get("intHomeScore") not in (None, "", "null"):
                try:
                    home_score = int(event["intHomeScore"])
                except:
                    pass
            if event.get("intAwayScore") not in (None, "", "null"):
                try:
                    away_score = int(event["intAwayScore"])
                except:
                    pass
            
            status_str = event.get("strStatus", "") or ""
            status = STATUS_MAP.get(status_str, "scheduled")
            
            return {
                "external_id": f"sportsdb_{event.get('idEvent')}",
                "sport_code": sport_code,
                "home_team": event.get("strHomeTeam"),
                "away_team": event.get("strAwayTeam"),
                "scheduled_time": scheduled_time,
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "venue": event.get("strVenue"),
                "season": event.get("strSeason"),
                "round": event.get("intRound"),
            }
        except Exception as e:
            logger.debug(f"[SportsDB] Error parsing event: {e}")
            return None

    # =========================================================================
    # LIVESCORES
    # =========================================================================
    
    async def _collect_livescores(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect live scores (Premium V2 only)."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        livescores = []
        
        # V2 only for livescores
        data = await self._request_v2(f"/livescore/{league_id}")
        
        if data and data.get("events"):
            for event in data["events"]:
                score = {
                    "external_id": f"sportsdb_{event.get('idEvent')}",
                    "sport_code": sport_code,
                    "home_team": event.get("strHomeTeam"),
                    "away_team": event.get("strAwayTeam"),
                    "home_score": int(event.get("intHomeScore", 0) or 0),
                    "away_score": int(event.get("intAwayScore", 0) or 0),
                    "status": event.get("strStatus", ""),
                    "progress": event.get("strProgress", ""),
                    "updated_at": datetime.now(timezone.utc),
                }
                livescores.append(score)
            logger.info(f"[SportsDB] Collected {len(livescores)} live scores for {sport_code}")
        
        return livescores
    
    async def collect_all_livescores(self) -> CollectorResult:
        """Collect live scores for all sports."""
        all_livescores = []
        
        for sport_code in SPORTSDB_LEAGUE_IDS.keys():
            scores = await self._collect_livescores(sport_code)
            all_livescores.extend(scores)
            await asyncio.sleep(0.2)
        
        return CollectorResult(
            source="sportsdb_live",
            success=True,
            data={"livescores": all_livescores},
            records_count=len(all_livescores),
        )

    # =========================================================================
    # VENUES COLLECTION
    # =========================================================================
    
    async def collect_venues(self, sport_code: str) -> Dict[str, Any]:
        """
        Collect venue/stadium information.
        
        Venues come from team data (strStadium field).
        """
        venues = []
        
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            logger.warning(f"[SportsDB] Unknown sport code: {sport_code}")
            return {"venues": [], "sport_code": sport_code}
        
        # Get teams (venues are part of team data)
        # Try V2 first
        data = await self._request_v2(f"/list/teams/{league_id}")
        
        if not data or not data.get("teams"):
            # Fallback to V1
            data = await self._request_v1(f"lookup_all_teams.php?id={league_id}")
        
        if data and data.get("teams"):
            teams = data["teams"]
            logger.info(f"[SportsDB] Found {len(teams)} teams for {sport_code}")
            
            seen_venues = set()
            
            for team in teams:
                venue_name = team.get("strStadium")
                if not venue_name or venue_name in seen_venues:
                    continue
                
                seen_venues.add(venue_name)
                
                # Parse capacity
                capacity = None
                capacity_str = team.get("intStadiumCapacity")
                if capacity_str:
                    try:
                        capacity = int(str(capacity_str).replace(",", ""))
                    except:
                        pass
                
                # Parse location
                location = team.get("strStadiumLocation", "") or ""
                city = location.split(",")[0].strip() if location else None
                state = location.split(",")[1].strip() if location and "," in location else None
                
                # Parse coordinates
                latitude = None
                longitude = None
                try:
                    if team.get("strStadiumLatitude"):
                        latitude = float(team["strStadiumLatitude"])
                    if team.get("strStadiumLongitude"):
                        longitude = float(team["strStadiumLongitude"])
                except:
                    pass
                
                # Check if dome
                stadium_desc = team.get("strStadiumDescription", "") or ""
                is_dome = self._is_dome(venue_name, stadium_desc)
                
                venues.append({
                    "name": venue_name,
                    "city": city,
                    "state": state,
                    "country": team.get("strCountry", "USA"),
                    "capacity": capacity,
                    "latitude": latitude,
                    "longitude": longitude,
                    "is_dome": is_dome,
                    "team_name": team.get("strTeam"),
                    "external_id": f"sportsdb_{team.get('idTeam')}",
                    "sport_code": sport_code,
                })
            
            logger.info(f"[SportsDB] Collected {len(venues)} venues for {sport_code}")
        else:
            logger.warning(f"[SportsDB] No team data returned for {sport_code}")
        
        return {"venues": venues, "sport_code": sport_code}
    
    def _is_dome(self, venue_name: str, description: str = "") -> bool:
        """Determine if venue is a dome/indoor stadium."""
        known_domes = [
            "dome", "superdome", "metrodome", "alamodome",
            "carrier dome", "lucas oil", "at&t stadium",
            "u.s. bank stadium", "sofi stadium", "allegiant stadium",
            "mercedes-benz stadium", "nrg stadium", "ford field",
            "caesars superdome", "state farm stadium",
            "arena", "center", "garden",
        ]
        
        venue_lower = (venue_name or "").lower()
        desc_lower = (description or "").lower()
        
        for dome in known_domes:
            if dome in venue_lower or dome in desc_lower:
                return True
        
        if "retractable" in venue_lower or "retractable" in desc_lower:
            return True
        
        return False
    
    async def save_venues_to_database(
        self,
        venues_data: List[Dict[str, Any]],
        session: AsyncSession
    ) -> int:
        """Save venues to database."""
        try:
            from app.models import Venue
        except ImportError:
            try:
                from app.models.venue_models import Venue
            except ImportError:
                logger.error("[SportsDB] Venue model not found")
                return 0
        
        saved_count = 0
        
        for venue_data in venues_data:
            try:
                venue_name = venue_data.get("name")
                if not venue_name:
                    continue
                
                existing = await session.execute(
                    select(Venue).where(Venue.name == venue_name)
                )
                venue = existing.scalar_one_or_none()
                
                if venue:
                    venue.city = venue_data.get("city") or venue.city
                    venue.state = venue_data.get("state") or venue.state
                    venue.country = venue_data.get("country") or venue.country
                    venue.capacity = venue_data.get("capacity") or venue.capacity
                    venue.is_dome = venue_data.get("is_dome", venue.is_dome)
                    if venue_data.get("latitude"):
                        venue.latitude = venue_data["latitude"]
                    if venue_data.get("longitude"):
                        venue.longitude = venue_data["longitude"]
                else:
                    venue = Venue(
                        name=venue_name[:200],
                        city=(venue_data.get("city") or "")[:100] or None,
                        state=(venue_data.get("state") or "")[:50] or None,
                        country=(venue_data.get("country") or "USA")[:50],
                        capacity=venue_data.get("capacity"),
                        latitude=venue_data.get("latitude"),
                        longitude=venue_data.get("longitude"),
                        is_dome=venue_data.get("is_dome", False),
                    )
                    session.add(venue)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[SportsDB] Error saving venue: {e}")
                continue
        
        try:
            await session.commit()
            logger.info(f"[SportsDB] Saved {saved_count} new venues")
        except Exception as e:
            logger.error(f"[SportsDB] Error committing venues: {e}")
            await session.rollback()
            
        return saved_count

    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================
    
    async def collect_historical(
        self,
        sport_code: str,
        seasons_back: int = 10,
        **kwargs,
    ) -> CollectorResult:
        """Collect historical game data."""
        all_games = []
        current_year = datetime.now().year
        
        for year_offset in range(seasons_back):
            season_year = current_year - year_offset
            
            if sport_code in ["NFL", "NCAAF", "MLB"]:
                season = f"{season_year}"
            elif sport_code in ["NBA", "NHL", "NCAAB"]:
                season = f"{season_year - 1}-{season_year}"
            else:
                season = f"{season_year}"
            
            games = await self._collect_season_games(sport_code, season)
            all_games.extend(games)
            await asyncio.sleep(0.5)
        
        return CollectorResult(
            source="sportsdb_history",
            success=len(all_games) > 0,
            data={"games": all_games},
            records_count=len(all_games),
        )
    
    async def _collect_season_games(self, sport_code: str, season: str) -> List[Dict[str, Any]]:
        """Collect all games for a season."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        # Try V2 first
        data = await self._request_v2(f"/schedule/league/{league_id}/season/{season}")
        
        if not data or not data.get("schedule"):
            # Fallback to V1
            data = await self._request_v1(f"eventsseason.php?id={league_id}&s={season}")
        
        if data:
            events = data.get("schedule") or data.get("events") or []
            for event in events:
                game = self._parse_event(event, sport_code)
                if game:
                    games.append(game)
            logger.info(f"[SportsDB] Collected {len(games)} games for {sport_code} season {season}")
        
        return games
    
    async def collect_past_games(self, sport_code: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Collect recently completed games."""
        return await self._collect_results(sport_code, limit=50)

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_games_to_database(self, games: List[Dict[str, Any]], session: AsyncSession) -> int:
        """Save games to database."""
        saved_count = 0
        
        for game_data in games:
            try:
                sport_code = game_data.get("sport_code")
                
                sport_result = await session.execute(
                    select(Sport).where(Sport.code == sport_code)
                )
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                home_team_name = game_data.get("home_team", "")
                away_team_name = game_data.get("away_team", "")
                
                home_result = await session.execute(
                    select(Team).where(
                        and_(Team.sport_id == sport.id, Team.name.ilike(f"%{home_team_name}%"))
                    )
                )
                home_team = home_result.scalar_one_or_none()
                
                away_result = await session.execute(
                    select(Team).where(
                        and_(Team.sport_id == sport.id, Team.name.ilike(f"%{away_team_name}%"))
                    )
                )
                away_team = away_result.scalar_one_or_none()
                
                if not home_team or not away_team:
                    continue
                
                external_id = game_data.get("external_id")
                existing = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = existing.scalar_one_or_none()
                
                status_str = game_data.get("status", "scheduled")
                status_result = await session.execute(
                    select(GameStatus).where(GameStatus.name == status_str)
                )
                status = status_result.scalar_one_or_none()
                
                if game:
                    if status:
                        game.status_id = status.id
                    game.home_score = game_data.get("home_score")
                    game.away_score = game_data.get("away_score")
                else:
                    game = Game(
                        external_id=external_id,
                        sport_id=sport.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_time=game_data.get("scheduled_time"),
                        status_id=status.id if status else None,
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                        season=game_data.get("season"),
                    )
                    session.add(game)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[SportsDB] Error saving game: {e}")
                continue
        
        try:
            await session.commit()
            logger.info(f"[SportsDB] Saved {saved_count} games")
        except Exception as e:
            logger.error(f"[SportsDB] Error committing games: {e}")
            await session.rollback()
            
        return saved_count
    
    async def save_teams_to_database(self, teams: List[Dict[str, Any]], session: AsyncSession) -> int:
        """Save teams to database."""
        saved_count = 0
        
        for team_data in teams:
            try:
                sport_code = team_data.get("sport_code")
                
                sport_result = await session.execute(
                    select(Sport).where(Sport.code == sport_code)
                )
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                team_name = team_data.get("name")
                existing = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
                )
                team = existing.scalar_one_or_none()
                
                if team:
                    team.abbreviation = team_data.get("abbreviation") or team.abbreviation
                    team.city = team_data.get("city") or team.city
                    team.conference = team_data.get("conference") or team.conference
                    team.division = team_data.get("division") or team.division
                    team.logo_url = team_data.get("logo_url") or team.logo_url
                else:
                    team = Team(
                        name=team_name,
                        abbreviation=(team_data.get("abbreviation") or "")[:10],
                        sport_id=sport.id,
                        city=team_data.get("city"),
                        conference=team_data.get("conference"),
                        division=team_data.get("division"),
                        logo_url=team_data.get("logo_url"),
                        external_id=team_data.get("external_id"),
                    )
                    session.add(team)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[SportsDB] Error saving team: {e}")
                continue
        
        try:
            await session.commit()
            logger.info(f"[SportsDB] Saved {saved_count} teams")
        except Exception as e:
            logger.error(f"[SportsDB] Error committing teams: {e}")
            await session.rollback()
            
        return saved_count
    
    async def save_historical_to_database(self, games: List[Dict[str, Any]], session: AsyncSession) -> Tuple[int, int]:
        """Save historical games."""
        saved = await self.save_games_to_database(games, session)
        return saved, 0
    
    async def _update_livescores(self, data: Dict[str, Any], session: AsyncSession) -> bool:
        """Update live scores in database."""
        livescores = data.get("livescores", [])
        
        for score_data in livescores:
            try:
                external_id = score_data.get("external_id")
                game_result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = game_result.scalar_one_or_none()
                
                if game:
                    game.home_score = score_data.get("home_score")
                    game.away_score = score_data.get("away_score")
            except:
                pass
        
        try:
            await session.commit()
            return True
        except:
            await session.rollback()
            return False
    
    async def has_data_available(self, sport_code: str = None) -> bool:
        """Check if data is available."""
        sports = [sport_code] if sport_code else list(SPORTSDB_LEAGUE_IDS.keys())[:1]
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if league_id:
                data = await self._request_v2(f"/schedule/next/league/{league_id}")
                if data:
                    return True
        return False
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if isinstance(data, dict):
            return len(data) > 0
        if isinstance(data, list):
            return len(data) > 0
        return True


# Create singleton instance
sportsdb_collector = SportsDBCollector()

# Register with collector manager
try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass