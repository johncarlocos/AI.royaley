"""
ROYALEY - TheSportsDB Collector (V2 Premium)
============================================

ONE COMMAND fills ALL tables:
    python scripts/master_import.py --source sportsdb_all

Tables filled:
    - teams       (from /list/teams)
    - venues      (from team data - strStadium)
    - games       (from /schedule endpoints)
    - players     (from /lookup/team/players)
    - team_stats  (from /lookup/table - standings)

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
from app.models import Sport, Team, Game, Venue, Player, TeamStats, Season
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)

# League IDs
LEAGUE_IDS = {
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
    """TheSportsDB V2 API Collector."""
    
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
        """V2 API request with header auth."""
        url = f"{self.base_url}{endpoint}"
        logger.info(f"[SportsDB] 🌐 {url}")
        print(f"[SportsDB] 🌐 {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Debug
                    if isinstance(data, dict):
                        info = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()}
                        logger.info(f"[SportsDB] ✅ 200 | {info}")
                        print(f"[SportsDB] ✅ 200 | {info}")
                    elif isinstance(data, list):
                        logger.info(f"[SportsDB] ✅ 200 | List[{len(data)}]")
                        print(f"[SportsDB] ✅ 200 | List[{len(data)}]")
                    return data
                else:
                    logger.warning(f"[SportsDB] ❌ {resp.status_code}: {resp.text[:100]}")
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
            for k in list(keys) + ["teams", "events", "schedule", "players", "table"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
            if len(data) == 1:
                v = list(data.values())[0]
                if isinstance(v, list):
                    return v
        return []

    # =========================================================================
    # REQUIRED ABSTRACT METHODS
    # =========================================================================
    
    async def collect(self, sport_code: str = None, **kwargs) -> CollectorResult:
        """Main collect - teams and upcoming games."""
        sports = [sport_code] if sport_code else ML_SPORTS
        data = {"teams": [], "games": []}
        
        for sport in sports:
            if sport not in LEAGUE_IDS:
                continue
            teams = await self._fetch_teams(sport)
            data["teams"].extend(teams)
            games = await self._fetch_schedule(sport)
            data["games"].extend(games)
        
        total = len(data["teams"]) + len(data["games"])
        return CollectorResult(source="sportsdb", success=total > 0, data=data, records_count=total)
    
    async def validate(self, data: Any) -> bool:
        return bool(data) and (len(data) > 0 if isinstance(data, (dict, list)) else True)

    # =========================================================================
    # DATA FETCHERS
    # =========================================================================
    
    async def _fetch_teams(self, sport: str) -> List[Dict]:
        """Fetch teams from V2 API."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/list/teams/{lid}")
        teams = self._get_list(data, "teams")
        
        result = []
        for t in teams:
            result.append({
                "external_id": f"sportsdb_{t.get('idTeam')}",
                "sportsdb_id": t.get("idTeam"),
                "name": t.get("strTeam"),
                "abbreviation": (t.get("strTeamShort") or "")[:10],
                "city": (t.get("strStadiumLocation") or "").split(",")[0].strip() or None,
                "country": t.get("strCountry"),
                "conference": t.get("strLeague2") or t.get("strDivision"),
                "division": t.get("strDivision"),
                "logo_url": t.get("strBadge") or t.get("strTeamBadge"),
                "sport_code": sport,
                # Stadium info for venues
                "stadium": t.get("strStadium"),
                "stadium_capacity": t.get("intStadiumCapacity"),
                "stadium_location": t.get("strStadiumLocation"),
                "stadium_lat": t.get("strStadiumLatitude"),
                "stadium_lon": t.get("strStadiumLongitude"),
                "stadium_desc": t.get("strStadiumDescription"),
            })
        
        logger.info(f"[SportsDB] {sport}: {len(result)} teams")
        print(f"[SportsDB] {sport}: {len(result)} teams")
        return result
    
    async def _fetch_schedule(self, sport: str) -> List[Dict]:
        """Fetch upcoming games."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/schedule/next/league/{lid}")
        events = self._get_list(data, "schedule", "events")
        
        games = [self._parse_event(e, sport) for e in events]
        games = [g for g in games if g]
        
        logger.info(f"[SportsDB] {sport}: {len(games)} upcoming games")
        return games
    
    async def _fetch_past_games(self, sport: str) -> List[Dict]:
        """Fetch recent results."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/schedule/previous/league/{lid}")
        events = self._get_list(data, "schedule", "events")
        
        games = [self._parse_event(e, sport) for e in events]
        return [g for g in games if g]
    
    async def _fetch_season(self, sport: str, season: str) -> List[Dict]:
        """Fetch games for a season."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/schedule/league/{lid}/season/{season}")
        events = self._get_list(data, "schedule", "events")
        
        games = [self._parse_event(e, sport) for e in events]
        games = [g for g in games if g]
        
        logger.info(f"[SportsDB] {sport} {season}: {len(games)} games")
        return games
    
    async def _fetch_players(self, team_id: str) -> List[Dict]:
        """Fetch players for a team."""
        data = await self._v2(f"/lookup/team/players/{team_id}")
        return self._get_list(data, "players", "player")
    
    async def _fetch_standings(self, sport: str, season: str) -> List[Dict]:
        """Fetch league standings."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/lookup/table/{lid}/season/{season}")
        return self._get_list(data, "table", "standings")
    
    async def _fetch_livescores(self, sport: str) -> List[Dict]:
        """Fetch live scores."""
        lid = LEAGUE_IDS.get(sport)
        if not lid:
            return []
        
        data = await self._v2(f"/livescore/{lid}")
        events = self._get_list(data, "events", "livescore")
        
        scores = []
        for e in events:
            scores.append({
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sport_code": sport,
                "home_team": e.get("strHomeTeam"),
                "away_team": e.get("strAwayTeam"),
                "home_score": int(e.get("intHomeScore", 0) or 0),
                "away_score": int(e.get("intAwayScore", 0) or 0),
                "status": e.get("strStatus", ""),
            })
        return scores
    
    def _parse_event(self, e: Dict, sport: str) -> Optional[Dict]:
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
                try: home_score = int(e["intHomeScore"])
                except: pass
            if e.get("intAwayScore") not in (None, "", "null"):
                try: away_score = int(e["intAwayScore"])
                except: pass
            
            return {
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sport_code": sport,
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
    # PUBLIC COLLECTION METHODS
    # =========================================================================
    
    async def collect_venues(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect venues from team data."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_venues = []
        
        for sport in sports:
            teams = await self._fetch_teams(sport)
            
            seen = set()
            for t in teams:
                name = t.get("stadium")
                if not name or name in seen:
                    continue
                seen.add(name)
                
                # Parse capacity
                cap = None
                cap_str = t.get("stadium_capacity")
                if cap_str:
                    try: cap = int(str(cap_str).replace(",", ""))
                    except: pass
                
                # Parse location
                loc = t.get("stadium_location", "") or ""
                city = loc.split(",")[0].strip() if loc else None
                state = loc.split(",")[1].strip() if loc and "," in loc else None
                
                # Coordinates
                lat = lon = None
                try:
                    if t.get("stadium_lat"): lat = float(t["stadium_lat"])
                    if t.get("stadium_lon"): lon = float(t["stadium_lon"])
                except: pass
                
                all_venues.append({
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": t.get("country", "USA"),
                    "capacity": cap,
                    "latitude": lat,
                    "longitude": lon,
                    "is_dome": self._is_dome(name, t.get("stadium_desc", "")),
                    "team_name": t.get("name"),
                    "sport_code": sport,
                })
            
            logger.info(f"[SportsDB] {sport}: {len(seen)} venues")
            print(f"[SportsDB] {sport}: {len(seen)} venues")
            await asyncio.sleep(0.1)
        
        return {"venues": all_venues, "count": len(all_venues)}
    
    def _is_dome(self, name: str, desc: str = "") -> bool:
        check = ((name or "") + " " + (desc or "")).lower()
        return any(d in check for d in ["dome", "arena", "center", "garden", "indoor", "retractable"])
    
    async def collect_players(self, sport_code: str = None) -> Dict[str, Any]:
        """Collect players for all teams."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_players = []
        
        for sport in sports:
            teams = await self._fetch_teams(sport)
            
            for team in teams:
                team_id = team.get("sportsdb_id")
                if not team_id:
                    continue
                
                players_raw = await self._fetch_players(team_id)
                
                for p in players_raw:
                    all_players.append({
                        "external_id": f"sportsdb_{p.get('idPlayer')}",
                        "name": p.get("strPlayer"),
                        "team_name": team.get("name"),
                        "sport_code": sport,
                        "position": p.get("strPosition"),
                        "number": p.get("strNumber"),
                        "nationality": p.get("strNationality"),
                        "birthdate": p.get("dateBorn"),
                        "height": p.get("strHeight"),
                        "weight": p.get("strWeight"),
                    })
                
                await asyncio.sleep(0.15)  # Rate limit
            
            logger.info(f"[SportsDB] {sport}: {len(all_players)} players total")
            print(f"[SportsDB] {sport}: {len(all_players)} players total")
        
        return {"players": all_players, "count": len(all_players)}
    
    async def collect_standings(self, sport_code: str, season: str = None) -> Dict[str, Any]:
        """Collect league standings for team_stats."""
        if not season:
            year = datetime.now().year
            if sport_code in ["NBA", "NHL", "NCAAB"]:
                season = f"{year - 1}-{year}"
            else:
                season = str(year)
        
        table = await self._fetch_standings(sport_code, season)
        
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
                "points": row.get("intPoints"),
            })
        
        logger.info(f"[SportsDB] {sport_code} {season}: {len(standings)} standings")
        return {"standings": standings, "sport_code": sport_code, "season": season}
    
    async def collect_historical(self, sport_code: str = None, seasons_back: int = 10, **kwargs) -> CollectorResult:
        """Collect historical games for ML training."""
        sports = [sport_code] if sport_code else ML_SPORTS
        all_games = []
        
        for sport in sports:
            year = datetime.now().year
            for offset in range(seasons_back):
                y = year - offset
                season = f"{y-1}-{y}" if sport in ["NBA", "NHL", "NCAAB"] else str(y)
                
                games = await self._fetch_season(sport, season)
                all_games.extend(games)
                await asyncio.sleep(0.3)
        
        logger.info(f"[SportsDB] Historical: {len(all_games)} total games")
        print(f"[SportsDB] Historical: {len(all_games)} total games")
        
        return CollectorResult(
            source="sportsdb_history",
            success=len(all_games) > 0,
            data={"games": all_games},
            records_count=len(all_games),
        )
    
    async def collect_all_livescores(self) -> CollectorResult:
        """Collect all live scores."""
        all_scores = []
        for sport in ML_SPORTS:
            scores = await self._fetch_livescores(sport)
            all_scores.extend(scores)
            await asyncio.sleep(0.1)
        return CollectorResult(source="sportsdb_live", success=True, data={"livescores": all_scores}, records_count=len(all_scores))

    # =========================================================================
    # MASTER COLLECTION - ONE COMMAND FILLS ALL TABLES
    # =========================================================================
    
    async def collect_all(self, seasons_back: int = 5) -> Dict[str, Any]:
        """
        MASTER METHOD - Collect ALL data for ML training.
        
        Fills: teams, venues, games, players, team_stats
        """
        logger.info("[SportsDB] ========== COLLECTING ALL DATA ==========")
        print("[SportsDB] ========== COLLECTING ALL DATA ==========")
        
        results = {
            "teams": [],
            "venues": [],
            "games": [],
            "players": [],
            "standings": [],
        }
        
        # 1. Teams (also gives us venue data)
        print("[SportsDB] Step 1/5: Teams...")
        for sport in ML_SPORTS:
            teams = await self._fetch_teams(sport)
            results["teams"].extend(teams)
            await asyncio.sleep(0.2)
        
        # 2. Venues (from team data)
        print("[SportsDB] Step 2/5: Venues...")
        venues_data = await self.collect_venues()
        results["venues"] = venues_data.get("venues", [])
        
        # 3. Games - upcoming + recent + historical
        print("[SportsDB] Step 3/5: Games...")
        for sport in ML_SPORTS:
            # Upcoming
            games = await self._fetch_schedule(sport)
            results["games"].extend(games)
            # Recent
            past = await self._fetch_past_games(sport)
            results["games"].extend(past)
            await asyncio.sleep(0.2)
        
        # Historical (fewer seasons for speed)
        print(f"[SportsDB] Step 3b/5: Historical ({seasons_back} seasons)...")
        historical = await self.collect_historical(seasons_back=seasons_back)
        results["games"].extend(historical.data.get("games", []))
        
        # 4. Players
        print("[SportsDB] Step 4/5: Players...")
        players_data = await self.collect_players()
        results["players"] = players_data.get("players", [])
        
        # 5. Standings (current season)
        print("[SportsDB] Step 5/5: Standings...")
        for sport in ML_SPORTS:
            try:
                standings = await self.collect_standings(sport)
                results["standings"].extend(standings.get("standings", []))
            except:
                pass
            await asyncio.sleep(0.2)
        
        totals = {k: len(v) for k, v in results.items()}
        logger.info(f"[SportsDB] ========== COMPLETE: {totals} ==========")
        print(f"[SportsDB] ========== COMPLETE: {totals} ==========")
        
        return results

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict, session: AsyncSession) -> int:
        """Save all collected data to database."""
        total = 0
        
        if data.get("teams"):
            total += await self.save_teams_to_database(data["teams"], session)
        if data.get("venues"):
            total += await self.save_venues_to_database(data["venues"], session)
        if data.get("games"):
            total += await self.save_games_to_database(data["games"], session)
        if data.get("players"):
            total += await self.save_players_to_database(data["players"], session)
        if data.get("standings"):
            total += await self.save_standings_to_database(data["standings"], session)
        
        return total
    
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
                    team.city = t.get("city") or team.city
                    team.logo_url = t.get("logo_url") or team.logo_url
                    team.conference = t.get("conference") or team.conference
                    team.division = t.get("division") or team.division
                else:
                    team = Team(
                        external_id=t.get("external_id") or f"sportsdb_{t.get('sportsdb_id')}",
                        name=name,
                        abbreviation=(t.get("abbreviation") or "UNK")[:10],
                        sport_id=sport.id,
                        city=t.get("city"),
                        conference=t.get("conference"),
                        division=t.get("division"),
                        logo_url=t.get("logo_url"),
                    )
                    session.add(team)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Team save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} teams")
        return saved
    
    async def save_venues_to_database(self, venues: List[Dict], session: AsyncSession) -> int:
        """Save venues."""
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
                    if v.get("latitude"): venue.latitude = v["latitude"]
                    if v.get("longitude"): venue.longitude = v["longitude"]
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
                logger.debug(f"[SportsDB] Venue save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} venues")
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
    
    async def save_players_to_database(self, players: List[Dict], session: AsyncSession) -> int:
        """Save players."""
        saved = 0
        for p in players:
            try:
                ext_id = p.get("external_id")
                if not ext_id:
                    continue
                
                existing = await session.execute(select(Player).where(Player.external_id == ext_id))
                player = existing.scalar_one_or_none()
                
                if player:
                    player.position = p.get("position") or player.position
                else:
                    # Find team
                    team = None
                    team_name = p.get("team_name")
                    if team_name:
                        team_result = await session.execute(
                            select(Team).where(Team.name.ilike(f"%{team_name}%"))
                        )
                        team = team_result.scalar_one_or_none()
                    
                    player = Player(
                        external_id=ext_id,
                        name=p.get("name", "Unknown")[:200],
                        team_id=team.id if team else None,
                        position=(p.get("position") or "")[:50] or None,
                        jersey_number=int(p["number"]) if p.get("number") and p["number"].isdigit() else None,
                    )
                    session.add(player)
                    saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Player save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} players")
        return saved
    
    async def save_standings_to_database(self, standings: List[Dict], session: AsyncSession) -> int:
        """Save standings to team_stats."""
        saved = 0
        for s in standings:
            try:
                team_name = s.get("team_name")
                if not team_name:
                    continue
                
                team_result = await session.execute(
                    select(Team).where(Team.name.ilike(f"%{team_name}%"))
                )
                team = team_result.scalar_one_or_none()
                if not team:
                    continue
                
                # Save wins as team_stat
                for stat_type, value in [("wins", s.get("wins")), ("losses", s.get("losses")), ("rank", s.get("rank"))]:
                    if value is None:
                        continue
                    
                    existing = await session.execute(
                        select(TeamStats).where(and_(
                            TeamStats.team_id == team.id,
                            TeamStats.stat_type == stat_type
                        ))
                    )
                    stat = existing.scalar_one_or_none()
                    
                    if stat:
                        stat.value = float(value)
                        stat.games_played = s.get("played") or 0
                    else:
                        stat = TeamStats(
                            team_id=team.id,
                            stat_type=stat_type,
                            value=float(value),
                            games_played=s.get("played") or 0,
                        )
                        session.add(stat)
                        saved += 1
            except Exception as e:
                logger.debug(f"[SportsDB] Standing save error: {e}")
        
        await session.commit()
        logger.info(f"[SportsDB] Saved {saved} team stats")
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
        data = await self._v2(f"/list/teams/{LEAGUE_IDS.get('NFL', 4391)}")
        return bool(data)
    
    async def collect_past_games(self, sport_code: str, days_back: int = 30) -> List[Dict]:
        return await self._fetch_past_games(sport_code)


# Singleton
sportsdb_collector = SportsDBCollector()

try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass
