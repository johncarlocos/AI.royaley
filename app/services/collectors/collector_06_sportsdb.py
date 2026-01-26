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
