"""
ROYALEY - TheSportsDB Collector (V2 ONLY - Premium)
COMPLETE ML DATA COLLECTION

Premium subscription ($295) - V2 API ONLY
- Base URL: https://www.thesportsdb.com/api/v2/json
- Authentication: X-API-KEY header
- API Key: 688655

COLLECTS ALL DATA FOR ML TRAINING:
- Teams (with logos, stadiums)
- Venues (stadiums with capacity, coordinates, dome status)
- Games (schedules, results, historical)
- Live Scores (real-time)
- Players (rosters)
- Standings/Tables
- Season data (10+ years historical)
"""

import asyncio
import logging
import httpx
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# TheSportsDB League IDs - All supported leagues
SPORTSDB_LEAGUE_IDS = {
    # Major US Sports
    "NFL": 4391,
    "NBA": 4387,
    "NHL": 4380,
    "MLB": 4424,
    "MLS": 4346,
    # College Sports
    "NCAAF": 4479,
    "NCAAB": 4607,
    # Other
    "WNBA": 4962,
    "CFL": 4406,
    "XFL": 4989,
    # International
    "EPL": 4328,      # English Premier League
    "LALIGA": 4335,   # Spanish La Liga
    "BUNDESLIGA": 4331,
    "SERIEA": 4332,
    "LIGUE1": 4334,
}

# Sports we focus on for ML
ML_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB"]

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
    "FT": "final",
    "AOT": "final",
    "AET": "final",
    "AP": "final",
    "CANC": "cancelled",
    "PST": "postponed",
    "ABD": "cancelled",
    "Match Finished": "final",
    "Not Started": "scheduled",
    "Finished": "final",
    "": "scheduled",
}


class SportsDBCollector(BaseCollector):
    """
    TheSportsDB V2 API Collector (Premium Only).
    Complete ML Data Collection.
    """
    
    def __init__(self):
        super().__init__(
            name="sportsdb",
            base_url="https://www.thesportsdb.com/api/v2/json",
            rate_limit=120,
            rate_window=60,
            timeout=30.0,
            max_retries=3,
        )
        self.api_key = settings.SPORTSDB_API_KEY or "688655"
        logger.info(f"[SportsDB] V2 Premium initialized | Key: {self.api_key}")
        print(f"[SportsDB] V2 Premium initialized | Key: {self.api_key}")
        
    def _headers(self) -> Dict[str, str]:
        """V2 API headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }
    
    async def _v2(self, endpoint: str) -> Optional[Any]:
        """Make V2 API request."""
        url = f"{self.base_url}{endpoint}"
        
        logger.info(f"[SportsDB] 🌐 {url}")
        print(f"[SportsDB] 🌐 {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Debug output
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        counts = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()}
                        logger.info(f"[SportsDB] ✅ 200 | Keys: {counts}")
                        print(f"[SportsDB] ✅ 200 | Keys: {counts}")
                    elif isinstance(data, list):
                        logger.info(f"[SportsDB] ✅ 200 | List[{len(data)}]")
                        print(f"[SportsDB] ✅ 200 | List[{len(data)}]")
                    return data
                else:
                    logger.warning(f"[SportsDB] ❌ {resp.status_code}")
                    print(f"[SportsDB] ❌ {resp.status_code}: {resp.text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"[SportsDB] ❌ Error: {e}")
            print(f"[SportsDB] ❌ Error: {e}")
            return None

    def _extract_list(self, data: Any, *keys) -> List[Dict]:
        """Extract list from response trying multiple keys."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try specified keys first
            for k in keys:
                if k in data and isinstance(data[k], list):
                    return data[k]
            # Try common keys
            for k in ["teams", "team", "events", "event", "schedule", "players", "player", 
                      "table", "standings", "data", "results", "list"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
            # Single key dict
            if len(data) == 1:
                val = list(data.values())[0]
                if isinstance(val, list):
                    return val
        return []

    # =========================================================================
    # MAIN COLLECT - Required abstract method
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
        """Validate data."""
        return bool(data) and (len(data) > 0 if isinstance(data, (dict, list)) else True)

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self, sport_code: str) -> List[Dict]:
        """Collect all teams for a league."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/list/teams/{league_id}")
        teams = self._extract_list(data, "teams")
        
        result = []
        for t in teams:
            result.append({
                "external_id": f"sportsdb_{t.get('idTeam')}",
                "sportsdb_id": t.get("idTeam"),
                "name": t.get("strTeam"),
                "full_name": t.get("strTeamLong") or t.get("strTeam"),
                "abbreviation": (t.get("strTeamShort") or "")[:10],
                "city": (t.get("strStadiumLocation") or "").split(",")[0].strip() or None,
                "country": t.get("strCountry"),
                "conference": t.get("strLeague2") or t.get("strDivision"),
                "division": t.get("strDivision"),
                "logo_url": t.get("strBadge") or t.get("strTeamBadge"),
                "banner_url": t.get("strBanner"),
                "jersey_url": t.get("strTeamJersey"),
                "sport_code": sport_code,
                "league_id": league_id,
                # Stadium info (for venues)
                "stadium": t.get("strStadium"),
                "stadium_capacity": t.get("intStadiumCapacity"),
                "stadium_location": t.get("strStadiumLocation"),
                "stadium_lat": t.get("strStadiumLatitude"),
                "stadium_lon": t.get("strStadiumLongitude"),
                "stadium_desc": t.get("strStadiumDescription"),
                # Social
                "website": t.get("strWebsite"),
                "facebook": t.get("strFacebook"),
                "twitter": t.get("strTwitter"),
                "instagram": t.get("strInstagram"),
                # Founded
                "formed_year": t.get("intFormedYear"),
            })
        
        logger.info(f"[SportsDB] {sport_code}: {len(result)} teams")
        print(f"[SportsDB] {sport_code}: {len(result)} teams")
        return result

    # =========================================================================
    # VENUES COLLECTION (from team data)
    # =========================================================================
    
    async def collect_venues(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect all venues for ML sports."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_venues = []
        
        for sport in sports:
            league_id = SPORTSDB_LEAGUE_IDS.get(sport)
            if not league_id:
                continue
            
            data = await self._v2(f"/list/teams/{league_id}")
            teams = self._extract_list(data, "teams")
            
            seen = set()
            for t in teams:
                name = t.get("strStadium")
                if not name or name in seen:
                    continue
                seen.add(name)
                
                # Parse capacity
                cap = None
                cap_str = t.get("intStadiumCapacity")
                if cap_str:
                    try:
                        cap = int(str(cap_str).replace(",", ""))
                    except:
                        pass
                
                # Parse location
                loc = t.get("strStadiumLocation", "") or ""
                city = loc.split(",")[0].strip() if loc else None
                state = loc.split(",")[1].strip() if loc and "," in loc else None
                
                # Coordinates
                lat = lon = None
                try:
                    if t.get("strStadiumLatitude"):
                        lat = float(t["strStadiumLatitude"])
                    if t.get("strStadiumLongitude"):
                        lon = float(t["strStadiumLongitude"])
                except:
                    pass
                
                all_venues.append({
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": t.get("strCountry", "USA"),
                    "capacity": cap,
                    "latitude": lat,
                    "longitude": lon,
                    "is_dome": self._is_dome(name, t.get("strStadiumDescription", "")),
                    "team_name": t.get("strTeam"),
                    "external_id": f"sportsdb_{t.get('idTeam')}",
                    "sport_code": sport,
                })
            
            logger.info(f"[SportsDB] {sport}: {len(seen)} venues")
            print(f"[SportsDB] {sport}: {len(seen)} venues")
            await asyncio.sleep(0.1)
        
        return {"venues": all_venues, "count": len(all_venues)}
    
    def _is_dome(self, name: str, desc: str = "") -> bool:
        """Check if indoor venue."""
        check = ((name or "") + " " + (desc or "")).lower()
        return any(d in check for d in ["dome", "arena", "center", "garden", "indoor", "retractable", "fieldhouse"])

    # =========================================================================
    # SCHEDULE / GAMES
    # =========================================================================
    
    async def _collect_schedule(self, sport_code: str, next_days: int = 14) -> List[Dict]:
        """Collect upcoming schedule."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/schedule/next/league/{league_id}")
        events = self._extract_list(data, "schedule", "events")
        
        games = [g for g in (self._parse_event(e, sport_code) for e in events) if g]
        logger.info(f"[SportsDB] {sport_code}: {len(games)} upcoming games")
        return games
    
    async def _collect_past_games(self, sport_code: str) -> List[Dict]:
        """Collect recent results."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/schedule/previous/league/{league_id}")
        events = self._extract_list(data, "schedule", "events")
        
        games = [g for g in (self._parse_event(e, sport_code) for e in events) if g]
        logger.info(f"[SportsDB] {sport_code}: {len(games)} past games")
        return games
    
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
            hs = e.get("intHomeScore")
            if hs not in (None, "", "null"):
                try:
                    home_score = int(hs)
                except:
                    pass
            aws = e.get("intAwayScore")
            if aws not in (None, "", "null"):
                try:
                    away_score = int(aws)
                except:
                    pass
            
            return {
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sportsdb_id": e.get("idEvent"),
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
                "round": e.get("intRound"),
                "week": e.get("intRound"),  # For NFL
                "spectators": e.get("intSpectators"),
                "video_url": e.get("strVideo"),
                "thumb_url": e.get("strThumb"),
            }
        except Exception as ex:
            logger.debug(f"[SportsDB] Parse error: {ex}")
            return None

    # =========================================================================
    # HISTORICAL DATA - For ML Training (10+ years)
    # =========================================================================
    
    async def collect_historical(self, sport_code: str = None, seasons_back: int = 10, **kwargs) -> CollectorResult:
        """Collect historical data for ML training."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_games = []
        
        for sport in sports:
            games = await self._collect_all_seasons(sport, seasons_back)
            all_games.extend(games)
        
        logger.info(f"[SportsDB] Historical total: {len(all_games)} games")
        print(f"[SportsDB] Historical total: {len(all_games)} games")
        
        return CollectorResult(
            source="sportsdb_history",
            success=len(all_games) > 0,
            data={"games": all_games},
            records_count=len(all_games),
        )
    
    async def _collect_all_seasons(self, sport_code: str, seasons_back: int = 10) -> List[Dict]:
        """Collect all seasons for a sport."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        all_games = []
        current_year = datetime.now().year
        
        for offset in range(seasons_back):
            year = current_year - offset
            
            # Season format varies by sport
            if sport_code in ["NBA", "NHL", "NCAAB"]:
                season = f"{year - 1}-{year}"
            else:  # NFL, MLB, NCAAF
                season = str(year)
            
            logger.info(f"[SportsDB] Fetching {sport_code} {season}...")
            print(f"[SportsDB] Fetching {sport_code} {season}...")
            
            data = await self._v2(f"/schedule/league/{league_id}/season/{season}")
            events = self._extract_list(data, "schedule", "events")
            
            games = [g for g in (self._parse_event(e, sport_code) for e in events) if g]
            all_games.extend(games)
            
            logger.info(f"[SportsDB] {sport_code} {season}: {len(games)} games")
            print(f"[SportsDB] {sport_code} {season}: {len(games)} games")
            
            await asyncio.sleep(0.3)  # Rate limit respect
        
        return all_games

    # =========================================================================
    # LIVESCORES (Premium Feature)
    # =========================================================================
    
    async def _collect_livescores(self, sport_code: str) -> List[Dict]:
        """Collect live scores."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        data = await self._v2(f"/livescore/{league_id}")
        events = self._extract_list(data, "events", "livescore")
        
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
                "progress": e.get("strProgress", ""),
                "updated_at": datetime.now(timezone.utc),
            })
        
        logger.info(f"[SportsDB] {sport_code}: {len(scores)} live scores")
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
    # PLAYERS
    # =========================================================================
    
    async def collect_players(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect players for all teams."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_players = []
        
        for sport in sports:
            teams = await self._collect_teams(sport)
            
            for team in teams[:5]:  # Limit to avoid rate limits during testing
                team_id = team.get("sportsdb_id")
                if not team_id:
                    continue
                
                data = await self._v2(f"/lookup/team/players/{team_id}")
                players = self._extract_list(data, "players", "player")
                
                for p in players:
                    all_players.append({
                        "external_id": f"sportsdb_{p.get('idPlayer')}",
                        "sportsdb_id": p.get("idPlayer"),
                        "name": p.get("strPlayer"),
                        "team_id": team_id,
                        "team_name": team.get("name"),
                        "sport_code": sport,
                        "position": p.get("strPosition"),
                        "number": p.get("strNumber"),
                        "nationality": p.get("strNationality"),
                        "birthdate": p.get("dateBorn"),
                        "height": p.get("strHeight"),
                        "weight": p.get("strWeight"),
                        "thumb_url": p.get("strThumb"),
                        "cutout_url": p.get("strCutout"),
                    })
                
                await asyncio.sleep(0.2)
            
            logger.info(f"[SportsDB] {sport}: {len(all_players)} players")
        
        return {"players": all_players, "count": len(all_players)}

    # =========================================================================
    # STANDINGS / TABLES
    # =========================================================================
    
    async def collect_standings(self, sport_code: str, season: str = None) -> Dict[str, Any]:
        """Collect league standings/table."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return {"standings": [], "sport_code": sport_code}
        
        if not season:
            year = datetime.now().year
            if sport_code in ["NBA", "NHL", "NCAAB"]:
                season = f"{year - 1}-{year}"
            else:
                season = str(year)
        
        data = await self._v2(f"/lookup/table/{league_id}/season/{season}")
        table = self._extract_list(data, "table", "standings")
        
        standings = []
        for row in table:
            standings.append({
                "team_id": row.get("idTeam"),
                "team_name": row.get("strTeam"),
                "sport_code": sport_code,
                "season": season,
                "rank": row.get("intRank"),
                "played": row.get("intPlayed"),
                "wins": row.get("intWin"),
                "losses": row.get("intLoss"),
                "draws": row.get("intDraw"),
                "goals_for": row.get("intGoalsFor"),
                "goals_against": row.get("intGoalsAgainst"),
                "goal_diff": row.get("intGoalDifference"),
                "points": row.get("intPoints"),
            })
        
        logger.info(f"[SportsDB] {sport_code} {season}: {len(standings)} standings")
        return {"standings": standings, "sport_code": sport_code, "season": season}

    # =========================================================================
    # FULL ML DATA COLLECTION
    # =========================================================================
    
    async def collect_all_ml_data(self, seasons_back: int = 10) -> Dict[str, Any]:
        """
        Collect ALL data needed for ML training.
        This is the master method for populating the database.
        """
        logger.info("[SportsDB] ========== FULL ML DATA COLLECTION ==========")
        print("[SportsDB] ========== FULL ML DATA COLLECTION ==========")
        
        results = {
            "teams": [],
            "venues": [],
            "games_upcoming": [],
            "games_historical": [],
            "livescores": [],
            "totals": {},
        }
        
        # 1. Teams & Venues (from same API call)
        logger.info("[SportsDB] Step 1: Collecting Teams & Venues...")
        print("[SportsDB] Step 1: Collecting Teams & Venues...")
        
        for sport in ML_SPORTS:
            teams = await self._collect_teams(sport)
            results["teams"].extend(teams)
            await asyncio.sleep(0.2)
        
        venues_data = await self.collect_venues()
        results["venues"] = venues_data.get("venues", [])
        
        # 2. Upcoming Games
        logger.info("[SportsDB] Step 2: Collecting Upcoming Games...")
        print("[SportsDB] Step 2: Collecting Upcoming Games...")
        
        for sport in ML_SPORTS:
            games = await self._collect_schedule(sport)
            results["games_upcoming"].extend(games)
            await asyncio.sleep(0.2)
        
        # 3. Past/Recent Games
        logger.info("[SportsDB] Step 3: Collecting Recent Results...")
        print("[SportsDB] Step 3: Collecting Recent Results...")
        
        for sport in ML_SPORTS:
            games = await self._collect_past_games(sport)
            results["games_upcoming"].extend(games)
            await asyncio.sleep(0.2)
        
        # 4. Historical Games (10 years)
        logger.info(f"[SportsDB] Step 4: Collecting {seasons_back} Years Historical...")
        print(f"[SportsDB] Step 4: Collecting {seasons_back} Years Historical...")
        
        historical = await self.collect_historical(seasons_back=seasons_back)
        results["games_historical"] = historical.data.get("games", [])
        
        # 5. Live Scores (if any games in progress)
        logger.info("[SportsDB] Step 5: Collecting Live Scores...")
        print("[SportsDB] Step 5: Collecting Live Scores...")
        
        live_result = await self.collect_all_livescores()
        results["livescores"] = live_result.data.get("livescores", [])
        
        # Totals
        results["totals"] = {
            "teams": len(results["teams"]),
            "venues": len(results["venues"]),
            "games_upcoming": len(results["games_upcoming"]),
            "games_historical": len(results["games_historical"]),
            "livescores": len(results["livescores"]),
            "total_games": len(results["games_upcoming"]) + len(results["games_historical"]),
        }
        
        logger.info(f"[SportsDB] ========== COLLECTION COMPLETE ==========")
        logger.info(f"[SportsDB] Totals: {results['totals']}")
        print(f"[SportsDB] ========== COLLECTION COMPLETE ==========")
        print(f"[SportsDB] Totals: {results['totals']}")
        
        return results

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_venues_to_database(self, venues_data: List[Dict], session: AsyncSession) -> int:
        """Save venues to database."""
        try:
            from app.models import Venue
        except ImportError:
            try:
                from app.models.venue_models import Venue
            except:
                logger.error("[SportsDB] Venue model not found")
                return 0
        
        saved = 0
        for v in venues_data:
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
                    if v.get("latitude"):
                        venue.latitude = v["latitude"]
                    if v.get("longitude"):
                        venue.longitude = v["longitude"]
                else:
                    venue = Venue(
                        name=name[:200],
                        city=(v.get("city") or "")[:100] or None,
                        state=(v.get("state") or "")[:50] or None,
                        country=(v.get("country") or "USA")[:50],
                        capacity=v.get("capacity"),
                        latitude=v.get("latitude"),
                        longitude=v.get("longitude"),
                        is_dome=v.get("is_dome", False),
                    )
                    session.add(venue)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Venue error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} venues")
        return saved
    
    async def save_teams_to_database(self, teams: List[Dict], session: AsyncSession) -> int:
        """Save teams to database."""
        saved = 0
        for t in teams:
            try:
                sport_code = t.get("sport_code")
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                name = t.get("name")
                existing = await session.execute(select(Team).where(and_(Team.sport_id == sport.id, Team.name == name)))
                team = existing.scalar_one_or_none()
                
                if team:
                    team.abbreviation = t.get("abbreviation") or team.abbreviation
                    team.city = t.get("city") or team.city
                    team.logo_url = t.get("logo_url") or team.logo_url
                    team.conference = t.get("conference") or team.conference
                    team.division = t.get("division") or team.division
                else:
                    team = Team(
                        name=name,
                        abbreviation=(t.get("abbreviation") or "")[:10],
                        sport_id=sport.id,
                        city=t.get("city"),
                        conference=t.get("conference"),
                        division=t.get("division"),
                        logo_url=t.get("logo_url"),
                        external_id=t.get("external_id"),
                    )
                    session.add(team)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Team error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} teams")
        return saved
    
    async def save_games_to_database(self, games: List[Dict], session: AsyncSession) -> int:
        """Save games to database."""
        saved = 0
        for g in games:
            try:
                sport_code = g.get("sport_code")
                sport_result = await session.execute(select(Sport).where(Sport.code == sport_code))
                sport = sport_result.scalar_one_or_none()
                if not sport:
                    continue
                
                # Find teams
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
                
                status_result = await session.execute(select(GameStatus).where(GameStatus.name == g.get("status", "scheduled")))
                status = status_result.scalar_one_or_none()
                
                if game:
                    if status:
                        game.status_id = status.id
                    game.home_score = g.get("home_score")
                    game.away_score = g.get("away_score")
                else:
                    game = Game(
                        external_id=ext_id,
                        sport_id=sport.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_time=g.get("scheduled_time"),
                        status_id=status.id if status else None,
                        home_score=g.get("home_score"),
                        away_score=g.get("away_score"),
                        season=g.get("season"),
                    )
                    session.add(game)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Game error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} games")
        return saved
    
    async def save_historical_to_database(self, games: List[Dict], session: AsyncSession) -> Tuple[int, int]:
        """Save historical games."""
        saved = await self.save_games_to_database(games, session)
        return saved, 0
    
    async def _update_livescores(self, data: Dict, session: AsyncSession) -> bool:
        """Update live scores in DB."""
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
        """Check if API is working."""
        sports = [sport_code] if sport_code else ["NFL"]
        for s in sports:
            lid = SPORTSDB_LEAGUE_IDS.get(s)
            if lid:
                data = await self._v2(f"/list/teams/{lid}")
                if data:
                    return True
        return False
    
    async def collect_past_games(self, sport_code: str, days_back: int = 30) -> List[Dict]:
        """Public method for past games."""
        return await self._collect_past_games(sport_code)


# Singleton
sportsdb_collector = SportsDBCollector()

try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass