"""
ROYALEY - TheSportsDB Collector (V2 Premium) - FIXED
=====================================================

V2 API Structure:
- /list/teams/{league_id}     → returns {'list': [...]}  (minimal data, NO stadium)
- /lookup/team/{team_id}      → returns {'lookup': [...]} (FULL data with stadium)
- /schedule/next/league/{id}  → returns {'schedule': [...]}
- /schedule/previous/league   → returns {'schedule': [...]}
- /livescore/{league_id}      → returns {'livescore': [...]}

Premium API ($295):
- Base URL: https://www.thesportsdb.com/api/v2/json
- Auth: X-API-KEY header
- Key: 688655
"""

import asyncio
import logging
import httpx
from datetime import datetime, timezone
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

# League IDs
SPORTSDB_LEAGUE_IDS = {
    "NFL": 4391, "NBA": 4387, "NHL": 4380, "MLB": 4424,
    "NCAAF": 4479, "NCAAB": 4607, "MLS": 4346,
}

ML_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB"]

STATUS_MAP = {
    "NS": "scheduled", "1H": "in_progress", "2H": "in_progress",
    "HT": "in_progress", "FT": "final", "AOT": "final",
    "Match Finished": "final", "Finished": "final",
    "Not Started": "scheduled", "": "scheduled",
}


class SportsDBCollector(BaseCollector):
    """TheSportsDB V2 API Collector - FIXED."""
    
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
        """Extract list from V2 response - tries multiple keys."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try specified keys + common V2 keys
            for k in list(keys) + ["list", "lookup", "teams", "schedule", "events", "livescore", "table", "players"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
        return []

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
        return CollectorResult(source="sportsdb", success=total > 0, data=all_data, records_count=total)
    
    async def validate(self, data: Any) -> bool:
        return bool(data) and (len(data) > 0 if isinstance(data, (dict, list)) else True)

    # =========================================================================
    # TEAMS COLLECTION - Uses /list then /lookup for full details
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
        """Get FULL team details including stadium from /lookup/team."""
        data = await self._v2(f"/lookup/team/{team_id}")
        teams = self._get_list(data, "lookup", "teams")
        return teams[0] if teams else None

    # =========================================================================
    # VENUES COLLECTION - Must use /lookup/team for stadium data
    # =========================================================================
    
    async def collect_venues(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect venues - requires /lookup/team for each team."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_venues = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            # Step 1: Get team IDs from /list
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._get_list(data, "list", "teams")
            
            print(f"[SportsDB] {sport}: Found {len(teams)} teams, fetching stadium details...")
            
            seen = set()
            venue_count = 0
            
            # Step 2: For each team, get full details with /lookup
            for t in teams:
                team_id = t.get("idTeam")
                if not team_id:
                    continue
                
                # Get full team details
                details = await self._get_team_details(team_id)
                if not details:
                    continue
                
                # Extract stadium
                name = details.get("strStadium")
                if not name or name in seen:
                    continue
                seen.add(name)
                
                # Parse capacity
                cap = None
                cap_str = details.get("intStadiumCapacity")
                if cap_str:
                    try:
                        cap = int(str(cap_str).replace(",", ""))
                    except:
                        pass
                
                # Parse location (e.g., "Orchard Park, New York")
                loc = details.get("strLocation") or ""
                city = loc.split(",")[0].strip() if loc else None
                state = loc.split(",")[1].strip() if loc and "," in loc else None
                
                all_venues.append({
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": details.get("strCountry", "USA"),
                    "capacity": cap,
                    "latitude": None,  # V2 doesn't have coordinates
                    "longitude": None,
                    "is_dome": self._is_dome(name),
                    "team_name": details.get("strTeam"),
                    "external_id": f"sportsdb_venue_{details.get('idVenue') or team_id}",
                    "sport_code": sport,
                })
                venue_count += 1
                
                # Rate limit - be gentle
                await asyncio.sleep(0.15)
            
            logger.info(f"[SportsDB] {sport}: {venue_count} venues")
            print(f"[SportsDB] {sport}: {venue_count} venues")
        
        return {"venues": all_venues, "count": len(all_venues)}
    
    def _is_dome(self, name: str) -> bool:
        """Check if indoor venue."""
        check = (name or "").lower()
        return any(d in check for d in ["dome", "arena", "center", "centre", "garden", "indoor", "fieldhouse", "forum", "palace"])

    # =========================================================================
    # SCHEDULE / GAMES
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
                "scheduled_time": dt.replace(tzinfo=timezone.utc),
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
    # HISTORICAL
    # =========================================================================
    
    async def collect_historical(self, sport_code: str = None, seasons_back: int = 10, **kwargs) -> CollectorResult:
        """Collect historical games."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_games = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            year = datetime.now().year
            for offset in range(seasons_back):
                y = year - offset
                season = f"{y-1}-{y}" if sport in ["NBA", "NHL", "NCAAB"] else str(y)
                
                data = await self._v2(f"/schedule/league/{league_id}/season/{season}")
                events = self._get_list(data, "schedule", "events")
                
                games = [self._parse_event(e, sport) for e in events]
                games = [g for g in games if g]
                all_games.extend(games)
                
                logger.info(f"[SportsDB] {sport} {season}: {len(games)} games")
                await asyncio.sleep(0.3)
        
        return CollectorResult(
            source="sportsdb_history",
            success=len(all_games) > 0,
            data={"games": all_games},
            records_count=len(all_games),
        )

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
        return CollectorResult(source="sportsdb_live", success=True, data={"livescores": all_scores}, records_count=len(all_scores))

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
        """Save games."""
        saved = 0
        for g in games:
            try:
                sport_code = g.get("sport_code")
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                home_name = g.get("home_team", "")
                away_name = g.get("away_team", "")
                
                home_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name.ilike(f"%{home_name}%")))
                )
                home_team = home_result.scalar_one_or_none()
                
                away_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name.ilike(f"%{away_name}%")))
                )
                away_team = away_result.scalar_one_or_none()
                
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
        logger.info(f"[SportsDB] Saved {saved} games")
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