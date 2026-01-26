"""
ROYALEY - TheSportsDB Collector (V2 Premium) - COMPLETE FIX
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
- /list/players/{teamId}          → team roster

STANDINGS:
- /lookup/table/{leagueId}/{season} → league standings

LIVESCORES:
- /livescore/{leagueId}           → live scores

Season Format:
- NFL/MLB/NCAAF: "2024" (calendar year)
- NBA/NHL/NCAAB: "2024-2025" (split season)
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
    """TheSportsDB V2 API Collector - Complete Implementation."""
    
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
        """V2 API request."""
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
                        info = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()}
                        logger.info(f"[SportsDB] ✅ 200 | {info}")
                        print(f"[SportsDB] ✅ 200 | {info}")
                    return data
                else:
                    logger.warning(f"[SportsDB] ❌ {resp.status_code}")
                    print(f"[SportsDB] ❌ {resp.status_code}: {resp.text[:100]}")
                    return None
        except Exception as e:
            logger.error(f"[SportsDB] ❌ {e}")
            print(f"[SportsDB] ❌ {e}")
            return None
    
    def _get_list(self, data: Any, *keys) -> List[Dict]:
        """Extract list from V2 response."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in list(keys) + ["schedule", "events", "list", "lookup", "teams", "livescore", "table", "players", "seasons"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
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
    
    async def _collect_teams(self, sport_code: str) -> List[Dict]:
        """Collect teams - minimal data from /list."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/list/teams/{league_id}")
        teams = self._get_list(data, "list", "teams")
        
        result = []
        for t in teams:
            result.append({
                "external_id": f"sportsdb_{t.get('idTeam')}",
                "sportsdb_id": t.get("idTeam"),
                "name": t.get("strTeam"),
                "abbreviation": (t.get("strTeamShort") or "")[:10],
                "country": t.get("strCountry"),
                "logo_url": t.get("strBadge"),
                "sport_code": sport_code,
            })
        
        logger.info(f"[SportsDB] {sport_code}: {len(result)} teams")
        print(f"[SportsDB] {sport_code}: {len(result)} teams")
        return result
    
    async def _get_team_details(self, team_id: str) -> Optional[Dict]:
        """Get FULL team details including stadium."""
        data = await self._v2(f"/lookup/team/{team_id}")
        teams = self._get_list(data, "lookup", "teams")
        return teams[0] if teams else None

    # =========================================================================
    # VENUES COLLECTION
    # =========================================================================
    
    async def collect_venues(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect venues via /lookup/team for each team."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_venues = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._get_list(data, "list", "teams")
            
            print(f"[SportsDB] {sport}: Found {len(teams)} teams, fetching stadium details...")
            
            seen = set()
            venue_count = 0
            
            for t in teams:
                team_id = t.get("idTeam")
                if not team_id:
                    continue
                
                details = await self._get_team_details(team_id)
                if not details:
                    continue
                
                name = details.get("strStadium")
                if not name or name in seen:
                    continue
                seen.add(name)
                
                cap = None
                cap_str = details.get("intStadiumCapacity")
                if cap_str:
                    try:
                        cap = int(str(cap_str).replace(",", ""))
                    except:
                        pass
                
                loc = details.get("strLocation") or ""
                city = loc.split(",")[0].strip() if loc else None
                state = loc.split(",")[1].strip() if loc and "," in loc else None
                
                all_venues.append({
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": details.get("strCountry", "USA"),
                    "capacity": cap,
                    "is_dome": self._is_dome(name),
                    "team_name": details.get("strTeam"),
                    "external_id": f"sportsdb_venue_{details.get('idVenue') or team_id}",
                    "sport_code": sport,
                })
                venue_count += 1
                await asyncio.sleep(0.15)
            
            logger.info(f"[SportsDB] {sport}: {venue_count} venues")
            print(f"[SportsDB] {sport}: {venue_count} venues")
        
        return {"venues": all_venues, "count": len(all_venues)}
    
    def _is_dome(self, name: str) -> bool:
        check = (name or "").lower()
        return any(d in check for d in ["dome", "arena", "center", "centre", "garden", "indoor", "fieldhouse", "forum", "palace"])

    # =========================================================================
    # PLAYERS COLLECTION
    # =========================================================================
    
    async def collect_players(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect players via /list/players/{teamId}."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_players = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # Get teams first
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._get_list(data, "list", "teams")
            
            print(f"[SportsDB] {sport}: Fetching players for {len(teams)} teams...")
            
            for t in teams:
                team_id = t.get("idTeam")
                team_name = t.get("strTeam")
                if not team_id:
                    continue
                
                player_data = await self._v2(f"/list/players/{team_id}")
                players = self._get_list(player_data, "players", "list")
                
                for p in players:
                    all_players.append({
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
                    })
                
                await asyncio.sleep(0.15)
            
            logger.info(f"[SportsDB] {sport}: {len([p for p in all_players if p['sport_code'] == sport])} players")
        
        print(f"[SportsDB] Total: {len(all_players)} players")
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
    
    def _parse_event(self, e: Dict, sport_code: str) -> Optional[Dict]:
        """Parse event to game dict."""
        try:
            date_str = e.get("dateEvent", "")
            if not date_str:
                return None
            
            time_str = (e.get("strTime") or "00:00:00").split("+")[0].replace("Z", "").strip() or "00:00:00"
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Keep as naive datetime (no timezone) for database compatibility
            
            home_score = away_score = None
            if e.get("intHomeScore") not in (None, "", "null"):
                try:
                    home_score = int(e["intHomeScore"])
                except:
                    pass
            if e.get("intAwayScore") not in (None, "", "null"):
                try:
                    away_score = int(e["intAwayScore"])
                except:
                    pass
            
            return {
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sport_code": sport_code,
                "home_team": e.get("strHomeTeam"),
                "away_team": e.get("strAwayTeam"),
                "home_team_id": e.get("idHomeTeam"),
                "away_team_id": e.get("idAwayTeam"),
                "scheduled_time": dt,  # naive datetime
                "status": STATUS_MAP.get(e.get("strStatus", ""), "scheduled"),
                "home_score": home_score,
                "away_score": away_score,
                "venue": e.get("strVenue"),
                "season": e.get("strSeason"),
            }
        except Exception as ex:
            logger.debug(f"[SportsDB] Parse error: {ex}")
            return None

    # =========================================================================
    # HISTORICAL GAMES - Full Season Schedule
    # =========================================================================
    
    async def collect_historical(self, sport_code: str = None, seasons_back: int = 10, **kwargs) -> CollectorResult:
        """Collect historical games using /schedule/league/{id}/{season}."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_games = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # First get available seasons
            available = await self.get_available_seasons(sport)
            if not available:
                # Fall back to generating season strings
                year = datetime.now().year
                available = [self._get_season_format(sport, year - i) for i in range(seasons_back)]
            
            # Limit to requested seasons
            seasons_to_fetch = available[:seasons_back]
            
            for season in seasons_to_fetch:
                # V2 endpoint: /schedule/league/{leagueId}/{season}
                data = await self._v2(f"/schedule/league/{league_id}/{season}")
                events = self._get_list(data, "schedule", "events")
                
                games = [self._parse_event(e, sport) for e in events]
                games = [g for g in games if g]
                all_games.extend(games)
                
                logger.info(f"[SportsDB] {sport} {season}: {len(games)} games")
                print(f"[SportsDB] {sport} {season}: {len(games)} games")
                await asyncio.sleep(0.3)
        
        return CollectorResult(
            success=len(all_games) > 0,
            data={"games": all_games},
            records_count=len(all_games),
        )

    # =========================================================================
    # STANDINGS / TABLE
    # =========================================================================
    
    async def collect_standings(self, sport_code: str = None, season: str = None) -> Dict[str, Any]:
        """Collect league standings."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_standings = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # Get current season if not specified
            sport_season = season
            if not sport_season:
                year = datetime.now().year
                month = datetime.now().month
                
                # Determine current season based on sport and month
                if sport in SPLIT_SEASON_SPORTS:
                    # NBA/NHL/NCAAB/NCAAF: season spans two years
                    if month <= 6:  # Jan-Jun = second half of season
                        sport_season = f"{year-1}-{year}"
                    else:  # Jul-Dec = first half of new season
                        sport_season = f"{year}-{year+1}"
                else:
                    # NFL/MLB/MLS/WNBA/UFC/CFL: single year seasons
                    if sport == "NFL" and month <= 2:  # NFL playoffs Jan-Feb
                        sport_season = str(year - 1)
                    elif sport == "WNBA" and month < 5:  # WNBA starts May
                        sport_season = str(year - 1)
                    elif sport == "MLB" and month < 4:  # MLB starts April
                        sport_season = str(year - 1)
                    elif sport == "MLS" and month < 3:  # MLS starts March
                        sport_season = str(year - 1)
                    elif sport == "CFL" and month < 6:  # CFL starts June
                        sport_season = str(year - 1)
                    else:
                        sport_season = str(year)
            
            data = await self._v2(f"/lookup/table/{league_id}/{sport_season}")
            table = self._get_list(data, "table", "standings")
            
            for entry in table:
                all_standings.append({
                    "sport_code": sport,
                    "season": sport_season,
                    "team_name": entry.get("strTeam"),
                    "team_id": entry.get("idTeam"),
                    "rank": entry.get("intRank"),
                    "wins": entry.get("intWin"),
                    "losses": entry.get("intLoss"),
                    "draws": entry.get("intDraw"),
                    "points": entry.get("intPoints"),
                    "games_played": entry.get("intPlayed"),
                    "goals_for": entry.get("intGoalsFor"),
                    "goals_against": entry.get("intGoalsAgainst"),
                    "goal_diff": entry.get("intGoalDifference"),
                })
            
            logger.info(f"[SportsDB] {sport} {sport_season}: {len(table)} standings entries")
            print(f"[SportsDB] {sport} {sport_season}: {len(table)} standings entries")
        
        return {"standings": all_standings, "count": len(all_standings)}

    # =========================================================================
    # LIVESCORES
    # =========================================================================
    
    async def _collect_livescores(self, sport_code: str) -> List[Dict]:
        """Collect live scores."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/livescore/{league_id}")
        events = self._get_list(data, "livescore", "events")
        
        scores = []
        for e in events:
            scores.append({
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sport_code": sport_code,
                "home_team": e.get("strHomeTeam"),
                "away_team": e.get("strAwayTeam"),
                "home_score": int(e.get("intHomeScore", 0) or 0),
                "away_score": int(e.get("intAwayScore", 0) or 0),
                "status": e.get("strStatus", ""),
            })
        return scores
    
    async def collect_all_livescores(self) -> CollectorResult:
        """Collect all live scores."""
        all_scores = []
        for sport in ML_SPORTS:
            scores = await self._collect_livescores(sport)
            all_scores.extend(scores)
            await asyncio.sleep(0.1)
        return CollectorResult(success=True, data={"livescores": all_scores}, records_count=len(all_scores))

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict, session: AsyncSession) -> int:
        """Save collected data."""
        total = 0
        if data.get("teams"):
            total += await self.save_teams_to_database(data["teams"], session)
        if data.get("games"):
            total += await self.save_games_to_database(data["games"], session)
        return total
    
    async def save_venues_to_database(self, venues: List[Dict], session: AsyncSession) -> int:
        """Save venues."""
        try:
            from app.models import Venue
        except ImportError:
            try:
                from app.models.models import Venue
            except:
                logger.error("[SportsDB] Venue model not found")
                return 0
        
        saved = 0
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
                else:
                    venue = Venue(
                        name=name[:200],
                        city=(v.get("city") or "")[:100] or None,
                        state=(v.get("state") or "")[:50] or None,
                        country=(v.get("country") or "USA")[:50],
                        capacity=v.get("capacity"),
                        is_dome=v.get("is_dome", False),
                    )
                    session.add(venue)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Venue save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} new venues")
        print(f"[SportsDB] Saved {saved} new venues")
        return saved
    
    async def save_players_to_database(self, players: List[Dict], session: AsyncSession) -> int:
        """Save players."""
        try:
            from app.models import Player
        except ImportError:
            try:
                from app.models.models import Player
            except:
                logger.error("[SportsDB] Player model not found")
                return 0
        
        saved = 0
        for p in players:
            try:
                ext_id = p.get("external_id")
                if not ext_id:
                    continue
                
                existing = await session.execute(select(Player).where(Player.external_id == ext_id))
                player = existing.scalar_one_or_none()
                
                if not player:
                    # Get team
                    sport_code = p.get("sport_code")
                    sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                    sport = sport_result.scalar_one_or_none()
                    if not sport:
                        continue
                    
                    team_result = await session.execute(
                        select(Team).where(and_(Team.sport_id == sport.id, Team.name == p.get("team_name")))
                    )
                    team = team_result.scalar_one_or_none()
                    
                    player = Player(
                        external_id=ext_id,
                        name=p.get("name", "Unknown")[:200],
                        team_id=team.id if team else None,
                        position=p.get("position"),
                        jersey_number=p.get("number"),
                    )
                    session.add(player)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Player save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} new players")
        print(f"[SportsDB] Saved {saved} new players")
        return saved
    
    async def save_teams_to_database(self, teams: List[Dict], session: AsyncSession) -> int:
        """Save teams."""
        saved = 0
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
                    team.abbreviation = t.get("abbreviation") or team.abbreviation
                    team.logo_url = t.get("logo_url") or team.logo_url
                else:
                    team = Team(
                        external_id=t.get("external_id") or f"sportsdb_{t.get('sportsdb_id')}",
                        name=name,
                        abbreviation=(t.get("abbreviation") or "UNK")[:10],
                        sport_id=sport.id,
                        logo_url=t.get("logo_url"),
                    )
                    session.add(team)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Team save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} teams")
        return saved
    
    async def save_games_to_database(self, games: List[Dict], session: AsyncSession) -> int:
        """Save games - auto-creates teams if they don't exist."""
        saved = 0
        skipped_no_sport = 0
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
                
                # Get or create home team
                home_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == home_name))
                )
                home_team = home_result.scalar_one_or_none()
                
                if not home_team and home_name:
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
                
                if not away_team and away_name:
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
                    continue
                
                ext_id = g.get("external_id")
                existing = await session.execute(select(Game).where(Game.external_id == ext_id))
                game = existing.scalar_one_or_none()
                
                if game:
                    game.home_score = g.get("home_score")
                    game.away_score = g.get("away_score")
                else:
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
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Game save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} games, created {teams_created} teams (skipped: {skipped_no_sport} no sport)")
        print(f"[SportsDB] Saved {saved} games, created {teams_created} teams")
        return saved
    
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


# Singleton
sportsdb_collector = SportsDBCollector()

try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass