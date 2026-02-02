"""
ROYALEY - CFL Data Collector (TheSportsDB V2 + The Odds API)
=============================================================

Data Sources:
- TheSportsDB V2 Premium API (Key: 688655, League ID: 4405)
  → Teams, Games (2008-2025), Players, Team Stats (Standings)
- The Odds API (Sport key: americanfootball_cfl)
  → Moneyline, Spread, Totals odds (2022-2026)

CFL Season Format: Calendar year (e.g., "2024")
CFL Structure: 9 teams, 2 divisions (East/West), 21-game regular season

Data Types Collected (7):
1. Teams      → TheSportsDB /list/teams/4405 + /lookup/team/{id}
2. Games      → TheSportsDB /schedule/league/4405/{season}
3. Players    → TheSportsDB /list/players/{teamId}
4. Player Stats → TheSportsDB (limited - no per-game CFL stats available)
5. Team Stats   → TheSportsDB /lookup/table/4405/{season} (standings)
6. Injuries     → Not available from TheSportsDB (placeholder for future source)
7. Odds         → The Odds API /v4/sports/americanfootball_cfl/odds-history

Tables Filled: sports, teams, games, players, player_stats, team_stats, odds
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Sport, Team, Player, Game, GameStatus,
    TeamStats, PlayerStats, Odds, Sportsbook, Venue,
)
from app.models.injury_models import Injury
from app.services.collectors.base_collector import (
    BaseCollector, CollectorResult, collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# TheSportsDB V2 Premium
SPORTSDB_BASE_URL = "https://www.thesportsdb.com/api/v2/json"
SPORTSDB_API_KEY = "688655"
CFL_LEAGUE_ID = 4405

# The Odds API
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
CFL_ODDS_SPORT_KEY = "americanfootball_cfl"

# CFL season range available on TheSportsDB
CFL_FIRST_SEASON = 2008
CFL_CURRENT_SEASON = 2025

# CFL Teams - hardcoded reference for name matching & fallback
CFL_TEAMS = {
    "BC": {"name": "BC Lions", "city": "Vancouver", "division": "West", "venue": "BC Place", "abbr": "BC"},
    "CGY": {"name": "Calgary Stampeders", "city": "Calgary", "division": "West", "venue": "McMahon Stadium", "abbr": "CGY"},
    "EDM": {"name": "Edmonton Elks", "city": "Edmonton", "division": "West", "venue": "Commonwealth Stadium", "abbr": "EDM"},
    "SSK": {"name": "Saskatchewan Roughriders", "city": "Regina", "division": "West", "venue": "Mosaic Stadium", "abbr": "SSK"},
    "WPG": {"name": "Winnipeg Blue Bombers", "city": "Winnipeg", "division": "West", "venue": "IG Field", "abbr": "WPG"},
    "HAM": {"name": "Hamilton Tiger-Cats", "city": "Hamilton", "division": "East", "venue": "Tim Hortons Field", "abbr": "HAM"},
    "MTL": {"name": "Montreal Alouettes", "city": "Montreal", "division": "East", "venue": "Percival Molson Memorial Stadium", "abbr": "MTL"},
    "OTT": {"name": "Ottawa Redblacks", "city": "Ottawa", "division": "East", "venue": "TD Place Stadium", "abbr": "OTT"},
    "TOR": {"name": "Toronto Argonauts", "city": "Toronto", "division": "East", "venue": "BMO Field", "abbr": "TOR"},
}

# Name normalization for matching across data sources
# Maps various name forms → canonical CFL team abbreviation
CFL_NAME_MAP = {
    # Standard names
    "bc lions": "BC",
    "calgary stampeders": "CGY",
    "edmonton elks": "EDM",
    "edmonton eskimos": "EDM",  # Historical name
    "saskatchewan roughriders": "SSK",
    "winnipeg blue bombers": "WPG",
    "hamilton tiger-cats": "HAM",
    "hamilton tigercats": "HAM",
    "montreal alouettes": "MTL",
    "ottawa redblacks": "OTT",
    "ottawa renegades": "OTT",  # Historical name
    "toronto argonauts": "TOR",
    # Short names (The Odds API sometimes uses these)
    "bc": "BC",
    "calgary": "CGY",
    "edmonton": "EDM",
    "saskatchewan": "SSK",
    "winnipeg": "WPG",
    "hamilton": "HAM",
    "montreal": "MTL",
    "ottawa": "OTT",
    "toronto": "TOR",
}

# Game status mapping from TheSportsDB
STATUS_MAP = {
    "Match Finished": "final",
    "Finished": "final",
    "FT": "final",
    "AOT": "final",
    "After OT": "final",
    "Not Started": "scheduled",
    "NS": "scheduled",
    "": "scheduled",
    "1H": "in_progress",
    "2H": "in_progress",
    "HT": "in_progress",
    "Q1": "in_progress",
    "Q2": "in_progress",
    "Q3": "in_progress",
    "Q4": "in_progress",
    "OT": "in_progress",
    "Postponed": "postponed",
    "Cancelled": "cancelled",
    "Abandoned": "cancelled",
}


# =============================================================================
# CFL COLLECTOR CLASS
# =============================================================================

class CFLCollector(BaseCollector):
    """
    CFL Data Collector using TheSportsDB V2 Premium + The Odds API.
    
    Collects 7 data types for CFL:
    1. Teams (9 active + historical)
    2. Games (2008-2025, ~3,000 games)
    3. Players (current rosters, ~500+ players)
    4. Player Stats (limited - TheSportsDB has no per-game CFL stats)
    5. Team Stats (standings: W/L/T/PF/PA per season)
    6. Injuries (not available from TheSportsDB - placeholder)
    7. Odds (The Odds API, 2022-2026, moneyline/spread/totals)
    """

    def __init__(self):
        super().__init__(
            name="cfl",
            base_url=SPORTSDB_BASE_URL,
            rate_limit=120,
            rate_window=60,
            timeout=30.0,
            max_retries=3,
        )
        self.sportsdb_key = SPORTSDB_API_KEY
        self.odds_api_key = ODDS_API_KEY
        logger.info(f"[CFL] TheSportsDB V2 | Key: {self.sportsdb_key} | League: {CFL_LEAGUE_ID}")
        print(f"[CFL] TheSportsDB V2 | Key: {self.sportsdb_key} | League: {CFL_LEAGUE_ID}")

    # =========================================================================
    # HTTP HELPERS
    # =========================================================================

    def _sportsdb_headers(self) -> Dict[str, str]:
        """Headers for TheSportsDB V2 Premium API."""
        return {"Accept": "application/json", "X-API-KEY": self.sportsdb_key}

    async def _sportsdb_get(self, endpoint: str) -> Optional[Any]:
        """Make a TheSportsDB V2 API request with logging."""
        url = f"{SPORTSDB_BASE_URL}{endpoint}"
        logger.info(f"[CFL] 🌐 {url}")
        print(f"[CFL] 🌐 {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._sportsdb_headers())

                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        if "Message" in data and len(data) == 1:
                            logger.warning(f"[CFL] ⚠️ API Message: {data.get('Message')}")
                            print(f"[CFL] ⚠️ API Message: {data.get('Message')}")
                            return None
                        info = {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()}
                        logger.info(f"[CFL] ✅ 200 | {info}")
                        print(f"[CFL] ✅ 200 | {info}")
                    return data
                else:
                    logger.warning(f"[CFL] ❌ {resp.status_code}: {resp.text[:200]}")
                    print(f"[CFL] ❌ {resp.status_code}")
                    return None
        except Exception as e:
            logger.error(f"[CFL] ❌ Request error: {e}")
            print(f"[CFL] ❌ {e}")
            return None

    async def _odds_api_get(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Make a The Odds API request."""
        url = f"{ODDS_API_BASE_URL}{endpoint}"
        if params is None:
            params = {}
        params["apiKey"] = self.odds_api_key

        logger.info(f"[CFL] 🎰 {url}")
        print(f"[CFL] 🎰 Odds API: {endpoint}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    print(f"[CFL] ✅ Odds API 200 | {len(data) if isinstance(data, list) else 1} items | Remaining: {remaining}")
                    return data
                elif resp.status_code == 422:
                    logger.info(f"[CFL] Odds API: No data available for this request")
                    return []
                else:
                    logger.warning(f"[CFL] ❌ Odds API {resp.status_code}: {resp.text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"[CFL] ❌ Odds API error: {e}")
            return None

    def _extract_list(self, data: Any, *keys) -> List[Dict]:
        """Extract list from TheSportsDB V2 response (handles various key names)."""
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            all_keys = list(keys) + [
                "schedule", "events", "list", "lookup", "teams",
                "table", "players", "player", "seasons", "roster",
            ]
            for k in all_keys:
                if k in data:
                    val = data[k]
                    if isinstance(val, list):
                        return val
                    elif val is None:
                        return []
        return []

    def _safe_int(self, val) -> Optional[int]:
        """Safely convert to int."""
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _normalize_team_name(self, name: str) -> Optional[str]:
        """Normalize a CFL team name to abbreviation."""
        if not name:
            return None
        return CFL_NAME_MAP.get(name.lower().strip())

    async def _ensure_sport(self, session: AsyncSession) -> Optional[Sport]:
        """Ensure CFL sport exists in database."""
        result = await session.execute(select(Sport).where(Sport.code == "CFL"))
        sport = result.scalar_one_or_none()

        if not sport:
            sport = Sport(
                code="CFL",
                name="Canadian Football League",
                is_active=True,
                config={"collector": "sportsdb_v2", "source": "TheSportsDB + The Odds API"},
            )
            session.add(sport)
            await session.flush()
            logger.info("[CFL] Created sport: CFL")
            print("[CFL] ✅ Created sport: CFL")

        return sport

    # =========================================================================
    # 1. TEAMS - TheSportsDB /list/teams/4405 + /lookup/team/{id}
    # =========================================================================

    async def collect_teams(self) -> List[Dict]:
        """
        Collect all CFL teams from TheSportsDB.
        
        Returns list of team dicts with: name, city, division, logo, stadium info.
        Also fetches full details (city, stadium) via /lookup/team for each team.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 1: COLLECTING TEAMS")
        print(f"{'='*60}")

        data = await self._sportsdb_get(f"/list/teams/{CFL_LEAGUE_ID}")
        raw_teams = self._extract_list(data, "list", "teams")

        if not raw_teams:
            print("[CFL] ⚠️ No teams from API, using hardcoded fallback")
            return self._fallback_teams()

        teams = []
        for t in raw_teams:
            team_id = t.get("idTeam")
            team_name = t.get("strTeam", "")
            abbr_lookup = self._normalize_team_name(team_name)

            team_data = {
                "sportsdb_id": team_id,
                "external_id": f"sportsdb_{team_id}",
                "name": team_name,
                "abbreviation": (t.get("strTeamShort") or abbr_lookup or "")[:10],
                "country": t.get("strCountry", "Canada"),
                "logo_url": t.get("strBadge"),
                "sport_code": "CFL",
            }

            # Fetch full details (city, stadium, division)
            if team_id:
                details = await self._sportsdb_get(f"/lookup/team/{team_id}")
                detail_list = self._extract_list(details, "lookup", "teams")
                if detail_list:
                    d = detail_list[0]
                    loc = d.get("strLocation") or ""
                    city = loc.split(",")[0].strip() if loc else None

                    # Try to find matching hardcoded team for division
                    division = None
                    if abbr_lookup and abbr_lookup in CFL_TEAMS:
                        division = CFL_TEAMS[abbr_lookup]["division"]

                    team_data.update({
                        "city": city,
                        "stadium": d.get("strStadium"),
                        "conference": division or d.get("strDivision"),
                        "division": division or d.get("strDivision"),
                        "description": (d.get("strDescriptionEN") or "")[:500] if d.get("strDescriptionEN") else None,
                    })

                await asyncio.sleep(0.15)  # Rate limit

            teams.append(team_data)
            print(f"[CFL]   ✅ {team_name} ({team_data.get('city', '?')}) - {team_data.get('division', '?')}")

        print(f"[CFL] Teams collected: {len(teams)}")
        return teams

    def _fallback_teams(self) -> List[Dict]:
        """Hardcoded CFL team data as fallback."""
        teams = []
        for abbr, info in CFL_TEAMS.items():
            teams.append({
                "external_id": f"cfl_{abbr}",
                "name": info["name"],
                "abbreviation": abbr,
                "city": info["city"],
                "division": info["division"],
                "conference": info["division"],
                "sport_code": "CFL",
            })
        return teams

    async def save_teams(self, teams: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL teams to database. Updates existing, creates new."""
        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0, "updated": 0}

        saved = 0
        updated = 0

        for t in teams:
            try:
                name = t.get("name", "").strip()
                if not name:
                    continue

                # Check by name first (handles existing teams with cfl_ external_ids)
                result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == name))
                )
                team = result.scalar_one_or_none()

                # Also check by sportsdb external_id
                if not team and t.get("external_id"):
                    result = await session.execute(
                        select(Team).where(Team.external_id == t["external_id"])
                    )
                    team = result.scalar_one_or_none()

                if team:
                    # Update existing
                    team.abbreviation = t.get("abbreviation") or team.abbreviation
                    team.logo_url = t.get("logo_url") or team.logo_url
                    if t.get("city") and not team.city:
                        team.city = t["city"]
                    if t.get("conference"):
                        team.conference = t["conference"]
                    if t.get("division"):
                        team.division = t["division"]
                    updated += 1
                else:
                    # Create new
                    team = Team(
                        external_id=t.get("external_id", f"cfl_{t.get('abbreviation', 'UNK')}"),
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
                logger.debug(f"[CFL] Team save error: {e}")

        await session.commit()
        print(f"[CFL] Teams: {saved} new, {updated} updated")
        return {"saved": saved, "updated": updated}

    # =========================================================================
    # 2. GAMES - TheSportsDB /schedule/league/4405/{season}
    # =========================================================================

    async def collect_games(self, seasons_back: int = 17) -> List[Dict]:
        """
        Collect CFL games from TheSportsDB for multiple seasons.
        
        Args:
            seasons_back: Number of seasons to collect (default 17 = 2008-2025)
        
        Returns list of game dicts with: teams, scores, dates, venues, status.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 2: COLLECTING GAMES ({seasons_back} seasons)")
        print(f"{'='*60}")

        # Get available seasons
        season_data = await self._sportsdb_get(f"/list/seasons/{CFL_LEAGUE_ID}")
        available_seasons = self._extract_list(season_data, "seasons")
        season_list = sorted(
            [s.get("strSeason") for s in available_seasons if s.get("strSeason")],
            reverse=True,
        )

        if not season_list:
            # Fallback: generate season strings
            current_year = datetime.now().year
            season_list = [str(y) for y in range(current_year, CFL_FIRST_SEASON - 1, -1)]

        # Limit to requested number
        season_list = season_list[:seasons_back]
        print(f"[CFL] Collecting seasons: {season_list}")

        all_games = []
        for season in season_list:
            try:
                data = await self._sportsdb_get(f"/schedule/league/{CFL_LEAGUE_ID}/{season}")
                events = self._extract_list(data, "schedule", "events")

                season_games = []
                for e in events:
                    game = self._parse_event(e)
                    if game:
                        season_games.append(game)

                all_games.extend(season_games)
                print(f"[CFL]   {season}: {len(season_games)} games")
                await asyncio.sleep(0.5)  # Be nice to API

            except Exception as e:
                logger.error(f"[CFL] Error collecting {season}: {e}")
                print(f"[CFL]   {season}: ❌ Error - {e}")

        print(f"[CFL] Total games collected: {len(all_games)}")
        return all_games

    def _parse_event(self, e: Dict) -> Optional[Dict]:
        """Parse a TheSportsDB event into a game dict."""
        try:
            date_str = e.get("dateEvent", "")
            if not date_str:
                return None

            time_str = e.get("strTime", "00:00:00") or "00:00:00"
            time_str = time_str.replace("+00:00", "").strip()
            if len(time_str) == 5:
                time_str += ":00"

            try:
                scheduled = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                scheduled = datetime.strptime(date_str, "%Y-%m-%d")

            status_raw = e.get("strStatus") or ""
            status = STATUS_MAP.get(status_raw, "scheduled")

            return {
                "external_id": f"sportsdb_{e.get('idEvent')}",
                "sportsdb_id": e.get("idEvent"),
                "home_team": e.get("strHomeTeam", "").strip(),
                "away_team": e.get("strAwayTeam", "").strip(),
                "scheduled_time": scheduled,
                "home_score": self._safe_int(e.get("intHomeScore")),
                "away_score": self._safe_int(e.get("intAwayScore")),
                "venue": e.get("strVenue"),
                "round": e.get("intRound"),
                "season": e.get("strSeason"),
                "status": status,
                "sport_code": "CFL",
            }

        except Exception as ex:
            logger.debug(f"[CFL] Parse error: {ex}")
            return None

    async def save_games(self, games: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL games to database. Auto-creates teams if needed."""
        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0, "updated": 0}

        saved = 0
        updated = 0
        skipped = 0
        teams_created = 0

        for g in games:
            try:
                home_name = g.get("home_team", "")
                away_name = g.get("away_team", "")
                if not home_name or not away_name:
                    skipped += 1
                    continue

                # Find or create home team
                home_team = await self._find_or_create_team(home_name, sport, session)
                if not home_team:
                    skipped += 1
                    continue

                # Find or create away team
                away_team = await self._find_or_create_team(away_name, sport, session)
                if not away_team:
                    skipped += 1
                    continue

                ext_id = g.get("external_id")

                # Check if game exists
                existing = await session.execute(select(Game).where(Game.external_id == ext_id))
                game = existing.scalar_one_or_none()

                if game:
                    game.home_score = g.get("home_score")
                    game.away_score = g.get("away_score")
                    game.scheduled_at = g.get("scheduled_time")
                    updated += 1
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
                logger.debug(f"[CFL] Game save error: {e}")
                skipped += 1

        try:
            await session.commit()
        except Exception as e:
            logger.error(f"[CFL] Commit error: {e}")
            await session.rollback()

        print(f"[CFL] Games: {saved} new, {updated} updated, {skipped} skipped")
        return {"saved": saved, "updated": updated, "skipped": skipped}

    async def _find_or_create_team(
        self, team_name: str, sport: Sport, session: AsyncSession
    ) -> Optional[Team]:
        """Find a CFL team by name (multiple matching strategies), or create it."""
        # 1. Exact name match
        result = await session.execute(
            select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
        )
        team = result.scalar_one_or_none()
        if team:
            return team

        # 2. ILIKE partial match
        result = await session.execute(
            select(Team).where(
                and_(Team.sport_id == sport.id, Team.name.ilike(f"%{team_name}%"))
            ).limit(1)
        )
        team = result.scalar_one_or_none()
        if team:
            return team

        # 3. Reverse partial match (team_name contains DB team name)
        all_teams_result = await session.execute(
            select(Team).where(Team.sport_id == sport.id)
        )
        all_teams = all_teams_result.scalars().all()
        for t in all_teams:
            if t.name and (t.name.lower() in team_name.lower() or team_name.lower() in t.name.lower()):
                return t

        # 4. Abbreviation-based lookup
        abbr = self._normalize_team_name(team_name)
        if abbr:
            for t in all_teams:
                if t.abbreviation and t.abbreviation.upper() == abbr.upper():
                    return t

        # 5. Auto-create team
        new_ext = f"sportsdb_CFL_{team_name.replace(' ', '_').lower()}"
        new_team = Team(
            name=team_name,
            sport_id=sport.id,
            external_id=new_ext,
            abbreviation=(abbr or team_name[:3].upper())[:10],
        )
        # Enrich from hardcoded data
        if abbr and abbr in CFL_TEAMS:
            info = CFL_TEAMS[abbr]
            new_team.city = info["city"]
            new_team.division = info["division"]
            new_team.conference = info["division"]

        session.add(new_team)
        await session.flush()
        print(f"[CFL]   ➕ Auto-created team: {team_name}")
        return new_team

    # =========================================================================
    # 3. PLAYERS - TheSportsDB /list/players/{teamId}
    # =========================================================================

    async def collect_players(self) -> List[Dict]:
        """
        Collect CFL players from TheSportsDB by fetching rosters for each team.
        
        Returns list of player dicts with: name, team, position, nationality, etc.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 3: COLLECTING PLAYERS")
        print(f"{'='*60}")

        # First get all CFL teams from API
        data = await self._sportsdb_get(f"/list/teams/{CFL_LEAGUE_ID}")
        teams = self._extract_list(data, "list", "teams")

        if not teams:
            print("[CFL] ⚠️ No teams found, cannot collect players")
            return []

        all_players = []
        for t in teams:
            team_id = t.get("idTeam")
            team_name = t.get("strTeam", "Unknown")

            if not team_id:
                continue

            player_data = await self._sportsdb_get(f"/list/players/{team_id}")
            players = self._extract_list(player_data, "player", "players", "list")

            if not players:
                print(f"[CFL]   {team_name}: 0 players (empty roster)")
                await asyncio.sleep(0.15)
                continue

            for p in players:
                player_entry = {
                    "external_id": f"sportsdb_{p.get('idPlayer')}",
                    "sportsdb_id": p.get("idPlayer"),
                    "name": p.get("strPlayer", "Unknown"),
                    "team_name": team_name,
                    "team_sportsdb_id": team_id,
                    "position": p.get("strPosition"),
                    "number": p.get("strNumber"),
                    "nationality": p.get("strNationality"),
                    "birth_date": p.get("dateBorn"),
                    "height": p.get("strHeight"),
                    "weight": p.get("strWeight"),
                    "photo_url": p.get("strCutout") or p.get("strThumb"),
                    "description": (p.get("strDescriptionEN") or "")[:500] or None,
                    "sport_code": "CFL",
                }
                all_players.append(player_entry)

            print(f"[CFL]   {team_name}: {len(players)} players")
            await asyncio.sleep(0.15)

        print(f"[CFL] Total players collected: {len(all_players)}")
        return all_players

    async def save_players(self, players: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL players to database."""
        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0, "updated": 0}

        saved = 0
        updated = 0

        for p in players:
            try:
                ext_id = p.get("external_id")
                if not ext_id:
                    continue

                # Check if player exists by external_id
                result = await session.execute(select(Player).where(Player.external_id == ext_id))
                player = result.scalar_one_or_none()

                if player:
                    # Update existing
                    player.name = p.get("name") or player.name
                    player.position = p.get("position") or player.position
                    # Update team assignment
                    team_name = p.get("team_name")
                    if team_name:
                        team_result = await session.execute(
                            select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
                        )
                        team = team_result.scalar_one_or_none()
                        if team:
                            player.team_id = team.id
                    updated += 1
                else:
                    # Find team
                    team = None
                    team_name = p.get("team_name")
                    if team_name:
                        team_result = await session.execute(
                            select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
                        )
                        team = team_result.scalar_one_or_none()
                        # Try partial match
                        if not team:
                            team_result = await session.execute(
                                select(Team).where(
                                    and_(Team.sport_id == sport.id, Team.name.ilike(f"%{team_name}%"))
                                ).limit(1)
                            )
                            team = team_result.scalar_one_or_none()

                    # Parse jersey number
                    jersey = None
                    if p.get("number"):
                        try:
                            jersey = int(str(p["number"]).strip())
                        except (ValueError, TypeError):
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
                logger.debug(f"[CFL] Player save error: {e}")

        await session.commit()
        print(f"[CFL] Players: {saved} new, {updated} updated")
        return {"saved": saved, "updated": updated}

    # =========================================================================
    # 4. PLAYER STATS - TheSportsDB (Limited)
    # =========================================================================

    async def collect_player_stats(self) -> List[Dict]:
        """
        Collect CFL player stats.
        
        LIMITATION: TheSportsDB does not provide per-game CFL player statistics
        (no passing/rushing/receiving/defense stats). This method returns an empty
        list and logs the limitation.
        
        Future: Can be extended to scrape from footballdb.com or StatsCrew.com
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 4: COLLECTING PLAYER STATS")
        print(f"{'='*60}")
        print("[CFL] ⚠️ TheSportsDB does NOT provide per-game CFL player statistics")
        print("[CFL]    Future sources: footballdb.com, StatsCrew.com scraping")
        print("[CFL]    Player stats: 0 collected (source limitation)")

        logger.info("[CFL] Player stats not available from TheSportsDB for CFL")
        return []

    async def save_player_stats(self, stats: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL player stats to database."""
        if not stats:
            return {"saved": 0}

        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0}

        saved = 0
        for stat in stats:
            try:
                player_ext_id = stat.get("player_external_id")
                if not player_ext_id:
                    continue

                result = await session.execute(
                    select(Player).where(Player.external_id == player_ext_id)
                )
                player = result.scalar_one_or_none()
                if not player:
                    continue

                stat_type = stat.get("stat_type", "general")
                game_id = stat.get("game_id")

                player_stat = PlayerStats(
                    player_id=player.id,
                    game_id=game_id,
                    stat_type=stat_type,
                    value=float(stat.get("value", 0)),
                )
                session.add(player_stat)
                saved += 1

            except Exception as e:
                logger.debug(f"[CFL] Player stat save error: {e}")

        if saved > 0:
            await session.commit()
        print(f"[CFL] Player stats: {saved} saved")
        return {"saved": saved}

    # =========================================================================
    # 5. TEAM STATS - TheSportsDB /lookup/table/4405/{season}
    # =========================================================================

    async def collect_team_stats(self, seasons_back: int = 17) -> List[Dict]:
        """
        Collect CFL team stats (standings) from TheSportsDB.
        
        Returns: W, L, D, GF (points for), GA (points against), etc. per team per season.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 5: COLLECTING TEAM STATS (STANDINGS)")
        print(f"{'='*60}")

        current_year = datetime.now().year
        start_year = max(CFL_FIRST_SEASON, current_year - seasons_back + 1)

        all_stats = []
        for year in range(start_year, current_year + 1):
            season_str = str(year)
            try:
                data = await self._sportsdb_get(f"/lookup/table/{CFL_LEAGUE_ID}/{season_str}")
                standings = self._extract_list(data, "table", "standings")

                if not standings:
                    print(f"[CFL]   {year}: No standings data")
                    await asyncio.sleep(0.3)
                    continue

                season_count = 0
                for s in standings:
                    team_name = s.get("strTeam", "")
                    if not team_name:
                        continue

                    # Each standing entry becomes multiple TeamStats records
                    # (one per stat type for ML flexibility)
                    base = {
                        "team_name": team_name,
                        "team_sportsdb_id": s.get("idTeam"),
                        "season": year,
                        "sport_code": "CFL",
                    }

                    stat_fields = {
                        "wins": self._safe_int(s.get("intWin")) or 0,
                        "losses": self._safe_int(s.get("intLoss")) or 0,
                        "draws": self._safe_int(s.get("intDraw")) or 0,
                        "played": self._safe_int(s.get("intPlayed")) or 0,
                        "points_for": self._safe_int(s.get("intGoalsFor")) or 0,
                        "points_against": self._safe_int(s.get("intGoalsAgainst")) or 0,
                        "goal_difference": self._safe_int(s.get("intGoalDifference")) or 0,
                        "standings_points": self._safe_int(s.get("intPoints")) or 0,
                        "rank": self._safe_int(s.get("intRank")) or 0,
                    }

                    for stat_name, stat_value in stat_fields.items():
                        all_stats.append({
                            **base,
                            "stat_type": f"CFL_{year}_{stat_name}",
                            "stat_name": stat_name,
                            "value": float(stat_value),
                            "games_played": stat_fields["played"],
                        })
                        season_count += 1

                print(f"[CFL]   {year}: {len(standings)} teams, {season_count} stat records")
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"[CFL] Error collecting standings for {year}: {e}")
                print(f"[CFL]   {year}: ❌ Error - {e}")

        print(f"[CFL] Total team stats collected: {len(all_stats)}")
        return all_stats

    async def save_team_stats(self, stats: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL team stats (standings) to database."""
        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0}

        saved = 0
        skipped = 0

        for stat in stats:
            try:
                team_name = stat.get("team_name", "")
                if not team_name:
                    skipped += 1
                    continue

                # Find team
                team_result = await session.execute(
                    select(Team).where(and_(Team.sport_id == sport.id, Team.name == team_name))
                )
                team = team_result.scalar_one_or_none()

                if not team:
                    # Try partial match
                    team_result = await session.execute(
                        select(Team).where(
                            and_(Team.sport_id == sport.id, Team.name.ilike(f"%{team_name}%"))
                        ).limit(1)
                    )
                    team = team_result.scalar_one_or_none()

                if not team:
                    skipped += 1
                    continue

                stat_type = stat.get("stat_type", "")

                # Check if stat already exists
                existing = await session.execute(
                    select(TeamStats).where(
                        and_(TeamStats.team_id == team.id, TeamStats.stat_type == stat_type)
                    )
                )
                existing_stat = existing.scalar_one_or_none()

                if existing_stat:
                    existing_stat.value = stat.get("value", 0.0)
                    existing_stat.games_played = stat.get("games_played", 0)
                else:
                    team_stat = TeamStats(
                        team_id=team.id,
                        stat_type=stat_type,
                        value=stat.get("value", 0.0),
                        games_played=stat.get("games_played", 0),
                    )
                    session.add(team_stat)
                    saved += 1

            except Exception as e:
                logger.debug(f"[CFL] Team stat save error: {e}")
                skipped += 1

        await session.commit()
        print(f"[CFL] Team stats: {saved} saved, {skipped} skipped")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # 6. INJURIES - Not available from TheSportsDB
    # =========================================================================

    async def collect_injuries(self) -> List[Dict]:
        """
        Collect CFL injuries.
        
        LIMITATION: TheSportsDB does not provide CFL injury data.
        This method returns an empty list and logs the limitation.
        
        Future: Can be extended to scrape from CFL.ca/injuries or TSN CFL reports.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 6: COLLECTING INJURIES")
        print(f"{'='*60}")
        print("[CFL] ⚠️ TheSportsDB does NOT provide CFL injury data")
        print("[CFL]    Future sources: CFL.ca, TSN, Sportsnet CFL injury reports")
        print("[CFL]    Injuries: 0 collected (source limitation)")

        logger.info("[CFL] Injury data not available from TheSportsDB for CFL")
        return []

    async def save_injuries(self, injuries: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL injuries to database."""
        if not injuries:
            return {"saved": 0, "updated": 0}

        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0, "updated": 0}

        saved = 0
        for inj in injuries:
            try:
                ext_id = inj.get("external_id")
                if not ext_id:
                    continue

                # Check existing
                result = await session.execute(
                    select(Injury).where(Injury.external_id == ext_id)
                )
                existing = result.scalar_one_or_none()
                if existing:
                    continue

                # Find team
                team_name = inj.get("team_name", "")
                team_result = await session.execute(
                    select(Team).where(
                        and_(Team.sport_id == sport.id, Team.name.ilike(f"%{team_name}%"))
                    ).limit(1)
                )
                team = team_result.scalar_one_or_none()
                if not team:
                    continue

                injury = Injury(
                    external_id=ext_id,
                    team_id=team.id,
                    sport_code="CFL",
                    player_name=inj.get("player_name", ""),
                    position=inj.get("position"),
                    injury_type=inj.get("injury_type"),
                    status=inj.get("status", "Unknown"),
                    source="thesportsdb",
                )
                session.add(injury)
                saved += 1

            except Exception as e:
                logger.debug(f"[CFL] Injury save error: {e}")

        if saved > 0:
            await session.commit()
        print(f"[CFL] Injuries: {saved} saved")
        return {"saved": saved, "updated": 0}

    # =========================================================================
    # 7. ODDS - The Odds API /v4/sports/americanfootball_cfl/odds
    # =========================================================================

    async def collect_odds(self, days_back: int = 30) -> List[Dict]:
        """
        Collect CFL odds from The Odds API.
        
        Collects moneyline (h2h), spreads, and totals from available bookmakers.
        Uses /odds endpoint for upcoming games, /odds-history for historical.
        
        Args:
            days_back: Number of days back for historical odds (default 30)
        
        Returns list of odds dicts ready for database save.
        """
        print(f"\n{'='*60}")
        print(f"[CFL] STEP 7: COLLECTING ODDS (The Odds API)")
        print(f"{'='*60}")

        if not self.odds_api_key:
            print("[CFL] ⚠️ No ODDS_API_KEY set - cannot collect odds")
            print("[CFL]    Set ODDS_API_KEY environment variable")
            return []

        all_odds = []

        # 1. Upcoming odds (live/upcoming games)
        print("[CFL] Fetching upcoming CFL odds...")
        upcoming = await self._odds_api_get(
            f"/sports/{CFL_ODDS_SPORT_KEY}/odds",
            params={
                "regions": "us,us2",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
        )

        if upcoming and isinstance(upcoming, list):
            for event in upcoming:
                odds_records = self._parse_odds_event(event)
                all_odds.extend(odds_records)
            print(f"[CFL]   Upcoming: {len(upcoming)} events → {len(all_odds)} odds records")
        else:
            print("[CFL]   Upcoming: No active CFL events (may be off-season)")

        # 2. Historical odds (recent past)
        print(f"[CFL] Fetching historical CFL odds ({days_back} days back)...")
        today = datetime.now()

        for days_ago in range(1, days_back + 1, 3):  # Sample every 3 days to conserve API calls
            target_date = today - timedelta(days=days_ago)
            date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")

            hist = await self._odds_api_get(
                f"/sports/{CFL_ODDS_SPORT_KEY}/odds-history",
                params={
                    "regions": "us,us2",
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "american",
                    "date": date_str,
                },
            )

            if hist and isinstance(hist, dict):
                events = hist.get("data", [])
                for event in events:
                    odds_records = self._parse_odds_event(event, recorded_at=target_date)
                    all_odds.extend(odds_records)

                if events:
                    print(f"[CFL]   {target_date.strftime('%Y-%m-%d')}: {len(events)} events")

            await asyncio.sleep(0.2)  # Rate limit

        print(f"[CFL] Total odds collected: {len(all_odds)}")
        return all_odds

    def _parse_odds_event(self, event: Dict, recorded_at: datetime = None) -> List[Dict]:
        """Parse a single Odds API event into multiple odds records."""
        records = []

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence_time_str = event.get("commence_time", "")

        try:
            if commence_time_str:
                commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                commence_time = commence_time.replace(tzinfo=None)
            else:
                commence_time = None
        except (ValueError, TypeError):
            commence_time = None

        bookmakers = event.get("bookmakers", [])

        for bookie in bookmakers:
            bookie_key = bookie.get("key", "unknown")
            markets = bookie.get("markets", [])

            for market in markets:
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h":
                    # Moneyline
                    home_odds = None
                    away_odds = None
                    for o in outcomes:
                        if o.get("name") == home_team:
                            home_odds = self._safe_int(o.get("price"))
                        elif o.get("name") == away_team:
                            away_odds = self._safe_int(o.get("price"))

                    if home_odds is not None or away_odds is not None:
                        records.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "sportsbook_key": bookie_key,
                            "bet_type": "moneyline",
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "recorded_at": recorded_at or datetime.utcnow(),
                        })

                elif market_key == "spreads":
                    # Point spread
                    home_line = None
                    away_line = None
                    home_odds = None
                    away_odds = None
                    for o in outcomes:
                        if o.get("name") == home_team:
                            home_line = o.get("point")
                            home_odds = self._safe_int(o.get("price"))
                        elif o.get("name") == away_team:
                            away_line = o.get("point")
                            away_odds = self._safe_int(o.get("price"))

                    records.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "sportsbook_key": bookie_key,
                        "bet_type": "spread",
                        "home_line": home_line,
                        "away_line": away_line,
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                        "recorded_at": recorded_at or datetime.utcnow(),
                    })

                elif market_key == "totals":
                    # Over/Under
                    total = None
                    over_odds = None
                    under_odds = None
                    for o in outcomes:
                        if o.get("name") == "Over":
                            total = o.get("point")
                            over_odds = self._safe_int(o.get("price"))
                        elif o.get("name") == "Under":
                            under_odds = self._safe_int(o.get("price"))

                    records.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "sportsbook_key": bookie_key,
                        "bet_type": "total",
                        "total": total,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "recorded_at": recorded_at or datetime.utcnow(),
                    })

        return records

    async def save_odds(self, odds: List[Dict], session: AsyncSession) -> Dict[str, int]:
        """Save CFL odds to database. Matches games by team names + date."""
        sport = await self._ensure_sport(session)
        if not sport:
            return {"saved": 0}

        saved = 0
        unmatched = 0

        for odd in odds:
            try:
                home_name = odd.get("home_team", "")
                away_name = odd.get("away_team", "")
                commence = odd.get("commence_time")

                if not home_name or not away_name or not commence:
                    unmatched += 1
                    continue

                # Find matching game (team names + date ±2 days)
                game = await self._match_game(
                    home_name, away_name, commence, sport, session
                )

                if not game:
                    unmatched += 1
                    continue

                # Create odds record
                odds_record = Odds(
                    game_id=game.id,
                    sportsbook_key=odd.get("sportsbook_key"),
                    bet_type=odd.get("bet_type", "moneyline"),
                    home_line=odd.get("home_line"),
                    away_line=odd.get("away_line"),
                    home_odds=odd.get("home_odds"),
                    away_odds=odd.get("away_odds"),
                    total=odd.get("total"),
                    over_odds=odd.get("over_odds"),
                    under_odds=odd.get("under_odds"),
                    recorded_at=odd.get("recorded_at", datetime.utcnow()),
                )
                session.add(odds_record)
                saved += 1

            except Exception as e:
                logger.debug(f"[CFL] Odds save error: {e}")
                unmatched += 1

        await session.commit()
        print(f"[CFL] Odds: {saved} saved, {unmatched} unmatched")
        return {"saved": saved, "unmatched": unmatched}

    async def _match_game(
        self,
        home_name: str,
        away_name: str,
        commence_time: datetime,
        sport: Sport,
        session: AsyncSession,
    ) -> Optional[Game]:
        """Match an odds event to a database game by team names + date (±2 days)."""
        date_start = commence_time - timedelta(days=2)
        date_end = commence_time + timedelta(days=2)

        # Get CFL teams that match the names
        home_team = await self._find_team_by_name(home_name, sport, session)
        away_team = await self._find_team_by_name(away_name, sport, session)

        if not home_team or not away_team:
            return None

        # Search for game with these teams in date range
        result = await session.execute(
            select(Game).where(
                and_(
                    Game.sport_id == sport.id,
                    Game.home_team_id == home_team.id,
                    Game.away_team_id == away_team.id,
                    Game.scheduled_at >= date_start,
                    Game.scheduled_at <= date_end,
                )
            ).limit(1)
        )
        game = result.scalar_one_or_none()

        # Try reversed home/away (neutral site games)
        if not game:
            result = await session.execute(
                select(Game).where(
                    and_(
                        Game.sport_id == sport.id,
                        Game.home_team_id == away_team.id,
                        Game.away_team_id == home_team.id,
                        Game.scheduled_at >= date_start,
                        Game.scheduled_at <= date_end,
                    )
                ).limit(1)
            )
            game = result.scalar_one_or_none()

        return game

    async def _find_team_by_name(
        self, name: str, sport: Sport, session: AsyncSession
    ) -> Optional[Team]:
        """Find a CFL team by name (exact, partial, or abbreviation lookup)."""
        # Exact match
        result = await session.execute(
            select(Team).where(and_(Team.sport_id == sport.id, Team.name == name))
        )
        team = result.scalar_one_or_none()
        if team:
            return team

        # ILIKE partial match
        result = await session.execute(
            select(Team).where(
                and_(Team.sport_id == sport.id, Team.name.ilike(f"%{name}%"))
            ).limit(1)
        )
        team = result.scalar_one_or_none()
        if team:
            return team

        # Try via name map → abbreviation → DB lookup
        abbr = self._normalize_team_name(name)
        if abbr:
            result = await session.execute(
                select(Team).where(
                    and_(Team.sport_id == sport.id, Team.abbreviation == abbr)
                ).limit(1)
            )
            team = result.scalar_one_or_none()
            if team:
                return team

        return None

    # =========================================================================
    # MAIN COLLECT METHOD (Orchestrates all 7 steps)
    # =========================================================================

    async def collect(self, **kwargs) -> CollectorResult:
        """
        Main collection entry point. Runs all 7 data collection steps in order.
        
        Keyword Args:
            collect_type: str - "all", "teams", "games", "players", 
                                "player_stats", "team_stats", "injuries", "odds"
            seasons_back: int - Number of seasons for games/team_stats (default 17)
            odds_days_back: int - Days back for odds history (default 30)
        
        Returns: CollectorResult with summary
        """
        collect_type = kwargs.get("collect_type", "all")
        seasons_back = kwargs.get("seasons_back", 17)
        odds_days_back = kwargs.get("odds_days_back", 30)

        print(f"\n{'='*70}")
        print(f"  ROYALEY CFL COLLECTOR - TheSportsDB V2 + The Odds API")
        print(f"  Type: {collect_type} | Seasons: {seasons_back} | Odds Days: {odds_days_back}")
        print(f"{'='*70}")

        data = {
            "teams": [], "games": [], "players": [],
            "player_stats": [], "team_stats": [],
            "injuries": [], "odds": [],
        }
        total = 0

        try:
            if collect_type in ("all", "teams"):
                data["teams"] = await self.collect_teams()
                total += len(data["teams"])

            if collect_type in ("all", "games"):
                data["games"] = await self.collect_games(seasons_back=seasons_back)
                total += len(data["games"])

            if collect_type in ("all", "players"):
                data["players"] = await self.collect_players()
                total += len(data["players"])

            if collect_type in ("all", "player_stats"):
                data["player_stats"] = await self.collect_player_stats()
                total += len(data["player_stats"])

            if collect_type in ("all", "team_stats"):
                data["team_stats"] = await self.collect_team_stats(seasons_back=seasons_back)
                total += len(data["team_stats"])

            if collect_type in ("all", "injuries"):
                data["injuries"] = await self.collect_injuries()
                total += len(data["injuries"])

            if collect_type in ("all", "odds"):
                data["odds"] = await self.collect_odds(days_back=odds_days_back)
                total += len(data["odds"])

        except Exception as e:
            logger.error(f"[CFL] Collection error: {e}")
            print(f"[CFL] ❌ Collection error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"  CFL COLLECTION SUMMARY")
        print(f"  Teams: {len(data['teams']):,}")
        print(f"  Games: {len(data['games']):,}")
        print(f"  Players: {len(data['players']):,}")
        print(f"  Player Stats: {len(data['player_stats']):,} (limited)")
        print(f"  Team Stats: {len(data['team_stats']):,}")
        print(f"  Injuries: {len(data['injuries']):,} (not available)")
        print(f"  Odds: {len(data['odds']):,}")
        print(f"  TOTAL: {total:,} records")
        print(f"{'='*70}")

        return CollectorResult(
            success=total > 0,
            data=data,
            records_count=total,
        )

    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected CFL data to database (all 7 types)."""
        total_saved = 0

        try:
            # Ensure sport exists first
            await self._ensure_sport(session)

            # 1. Teams (must be first - games/players reference teams)
            if data.get("teams"):
                result = await self.save_teams(data["teams"], session)
                total_saved += result.get("saved", 0) + result.get("updated", 0)

            # 2. Games (must be before odds - odds reference games)
            if data.get("games"):
                result = await self.save_games(data["games"], session)
                total_saved += result.get("saved", 0) + result.get("updated", 0)

            # 3. Players
            if data.get("players"):
                result = await self.save_players(data["players"], session)
                total_saved += result.get("saved", 0) + result.get("updated", 0)

            # 4. Player Stats
            if data.get("player_stats"):
                result = await self.save_player_stats(data["player_stats"], session)
                total_saved += result.get("saved", 0)

            # 5. Team Stats
            if data.get("team_stats"):
                result = await self.save_team_stats(data["team_stats"], session)
                total_saved += result.get("saved", 0)

            # 6. Injuries
            if data.get("injuries"):
                result = await self.save_injuries(data["injuries"], session)
                total_saved += result.get("saved", 0)

            # 7. Odds (last - needs games to exist for matching)
            if data.get("odds"):
                result = await self.save_odds(data["odds"], session)
                total_saved += result.get("saved", 0)

            print(f"[CFL] ✅ Total saved to database: {total_saved:,}")

        except Exception as e:
            logger.error(f"[CFL] Database save error: {e}")
            print(f"[CFL] ❌ Save error: {e}")
            await session.rollback()
            raise

        return total_saved

    # =========================================================================
    # REQUIRED ABSTRACT METHODS (BaseCollector)
    # =========================================================================

    async def validate(self, data: Any) -> bool:
        """Validate collected CFL data."""
        if not data or not isinstance(data, dict):
            return False
        return any(len(data.get(k, [])) > 0 for k in ["teams", "games", "players"])


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

cfl_collector = CFLCollector()

try:
    collector_manager.register(cfl_collector)
except Exception:
    pass