"""
ROYALEY - TheSportsDB Collector
Phase 1: Data Collection Services

Collects comprehensive sports data from TheSportsDB API.
Features: Games, scores, teams, livescores, historical results.

API Documentation: https://www.thesportsdb.com/documentation
API Tier: Premium ($295/mo) - Full V2 API access with livescores

V2 API Base URL: https://www.thesportsdb.com/api/v2/json
V1 API Base URL: https://www.thesportsdb.com/api/v1/json/{API_KEY}

Authentication:
- V2: X-API-KEY header
- V1: API key in URL path
"""

import asyncio
import logging
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
    "ATP": 4464,   # ATP Tour
    "WTA": 4465,   # WTA Tour
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
    "NS": "scheduled",      # Not Started
    "1H": "in_progress",    # First Half
    "2H": "in_progress",    # Second Half
    "HT": "in_progress",    # Half Time
    "Q1": "in_progress",    # Quarter 1
    "Q2": "in_progress",    # Quarter 2
    "Q3": "in_progress",    # Quarter 3
    "Q4": "in_progress",    # Quarter 4
    "OT": "in_progress",    # Overtime
    "BT": "in_progress",    # Break Time
    "PT": "in_progress",    # Penalty Time
    "ET": "in_progress",    # Extra Time
    "FT": "final",          # Full Time / Finished
    "AOT": "final",         # After Over Time
    "AET": "final",         # After Extra Time
    "AP": "final",          # After Penalties
    "CANC": "cancelled",    # Cancelled
    "PST": "postponed",     # Postponed
    "ABD": "cancelled",     # Abandoned
    "AWD": "final",         # Awarded
    "WO": "final",          # Walkover
    "Match Finished": "final",
    "Not Started": "scheduled",
    "": "scheduled",
}


class SportsDBCollector(BaseCollector):
    """
    Collector for TheSportsDB API.
    
    Features:
    - Games/Events schedule (current, upcoming, past)
    - Livescores (real-time updates)
    - Team information
    - Historical results (by season)
    - League standings
    
    Premium features available with $9+/mo subscription.
    """
    
    def __init__(self):
        # V2 API base URL (premium)
        super().__init__(
            name="sportsdb",
            base_url="https://www.thesportsdb.com/api/v2/json",
            rate_limit=120,  # Business tier limit
            rate_window=60,
            timeout=30.0,
            max_retries=3,
        )
        self.api_key = settings.SPORTSDB_API_KEY or "688655"
        self.v1_base_url = f"https://www.thesportsdb.com/api/v1/json/{self.api_key}"
        
    def _get_headers(self) -> Dict[str, str]:
        """Get V2 API headers with authentication."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }
    
    async def collect(
        self,
        sport_code: str = None,
        collect_type: str = "all",
        days_ahead: int = 7,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect data from TheSportsDB.
        
        Args:
            sport_code: Optional sport code (NFL, NBA, etc.)
            collect_type: Type of data (schedule, livescores, teams, all)
            days_ahead: Days ahead for schedule
            
        Returns:
            CollectorResult with collected data
        """
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(SPORTSDB_LEAGUE_IDS.keys())
        )
        
        all_data = {
            "games": [],
            "livescores": [],
            "teams": [],
        }
        errors = []
        
        for sport in sports_to_collect:
            if sport not in SPORTSDB_LEAGUE_IDS:
                logger.warning(f"[SportsDB] Unknown sport: {sport}")
                continue
                
            try:
                if collect_type in ["schedule", "all"]:
                    # Get upcoming games
                    games = await self._collect_schedule(sport, days_ahead)
                    all_data["games"].extend(games)
                    
                    # Get recent results
                    results = await self._collect_results(sport)
                    all_data["games"].extend(results)
                
                if collect_type in ["livescores", "all"]:
                    livescores = await self._collect_livescores(sport)
                    all_data["livescores"].extend(livescores)
                
                if collect_type in ["teams", "all"]:
                    teams = await self._collect_teams(sport)
                    all_data["teams"].extend(teams)
                    
            except Exception as e:
                logger.error(f"[SportsDB] Error collecting {sport}: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        total_records = sum(len(v) for v in all_data.values())
        
        return CollectorResult(
            success=len(errors) == 0 or total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={
                "sports_collected": sports_to_collect,
                "collect_type": collect_type,
            },
        )
    
    # =========================================================================
    # SCHEDULE & EVENTS
    # =========================================================================
    
    async def _collect_schedule(
        self,
        sport_code: str,
        days_ahead: int = 7,
    ) -> List[Dict[str, Any]]:
        """Collect upcoming game schedule using V2 API."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        try:
            # V2 API: Next events for league
            endpoint = f"/schedule/next/league/{league_id}"
            
            data = await self.get(endpoint)
            events = data.get("schedule") or data.get("events") or []
            
            logger.info(f"[SportsDB] {sport_code}: Fetched {len(events)} upcoming events")
            
            for event in events:
                game = self._parse_event(event, sport_code)
                if game:
                    games.append(game)
                    
        except Exception as e:
            logger.error(f"[SportsDB] Failed to get schedule for {sport_code}: {e}")
            
            # Fallback to V1 API
            try:
                games = await self._collect_schedule_v1(sport_code)
            except Exception as e2:
                logger.error(f"[SportsDB] V1 fallback also failed: {e2}")
        
        return games
    
    async def _collect_schedule_v1(self, sport_code: str) -> List[Dict[str, Any]]:
        """Fallback: Collect schedule using V1 API."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        try:
            # V1 API endpoint
            url = f"{self.v1_base_url}/eventsnextleague.php?id={league_id}"
            
            client = await self.get_client()
            response = await client.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()
            
            events = data.get("events") or []
            logger.info(f"[SportsDB V1] {sport_code}: Fetched {len(events)} events")
            
            for event in events:
                game = self._parse_event(event, sport_code)
                if game:
                    games.append(game)
                    
        except Exception as e:
            logger.error(f"[SportsDB V1] Schedule error: {e}")
        
        return games
    
    async def _collect_results(
        self,
        sport_code: str,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Collect recent game results."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        try:
            # V2 API: Previous events for league
            endpoint = f"/schedule/previous/league/{league_id}"
            
            data = await self.get(endpoint)
            events = data.get("schedule") or data.get("events") or []
            
            logger.info(f"[SportsDB] {sport_code}: Fetched {len(events)} recent results")
            
            for event in events:
                game = self._parse_event(event, sport_code, is_result=True)
                if game:
                    games.append(game)
                    
        except Exception as e:
            logger.error(f"[SportsDB] Failed to get results for {sport_code}: {e}")
            
            # Fallback to V1 API
            try:
                url = f"{self.v1_base_url}/eventspastleague.php?id={league_id}"
                client = await self.get_client()
                response = await client.get(url, headers={"Accept": "application/json"})
                response.raise_for_status()
                data = response.json()
                
                events = data.get("events") or []
                for event in events:
                    game = self._parse_event(event, sport_code, is_result=True)
                    if game:
                        games.append(game)
            except Exception as e2:
                logger.error(f"[SportsDB V1] Results fallback failed: {e2}")
        
        return games
    
    # =========================================================================
    # LIVESCORES
    # =========================================================================
    
    async def _collect_livescores(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect live scores (premium feature)."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        sport_name = SPORTSDB_SPORT_NAMES.get(sport_code)
        
        if not league_id or not sport_name:
            return []
        
        livescores = []
        
        try:
            # V2 API: Livescores by league
            endpoint = f"/livescore/{league_id}"
            
            data = await self.get(endpoint)
            events = data.get("livescore") or data.get("events") or []
            
            if events:
                logger.info(f"[SportsDB] {sport_code}: {len(events)} live games")
            
            for event in events:
                live = self._parse_livescore(event, sport_code)
                if live:
                    livescores.append(live)
                    
        except Exception as e:
            # Livescores may not be available for all sports/times
            logger.debug(f"[SportsDB] No livescores for {sport_code}: {e}")
        
        return livescores
    
    async def collect_all_livescores(self) -> CollectorResult:
        """Collect all current livescores across all sports."""
        all_livescores = []
        errors = []
        
        try:
            # V2 API: All livescores
            endpoint = "/livescore/all"
            
            data = await self.get(endpoint)
            events = data.get("livescore") or data.get("events") or []
            
            logger.info(f"[SportsDB] Fetched {len(events)} total livescores")
            
            for event in events:
                # Determine sport from event data
                sport_code = self._get_sport_from_event(event)
                live = self._parse_livescore(event, sport_code or "UNKNOWN")
                if live:
                    all_livescores.append(live)
                    
        except Exception as e:
            logger.error(f"[SportsDB] Failed to get all livescores: {e}")
            errors.append(str(e))
        
        return CollectorResult(
            success=len(all_livescores) > 0,
            data=all_livescores,
            records_count=len(all_livescores),
            error="; ".join(errors) if errors else None,
        )
    
    # =========================================================================
    # TEAMS
    # =========================================================================
    
    async def _collect_teams(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect team information for a league."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        teams = []
        
        try:
            # V2 API: Teams by league
            endpoint = f"/list/teams/{league_id}"
            
            data = await self.get(endpoint)
            teams_data = data.get("teams") or []
            
            logger.info(f"[SportsDB] {sport_code}: Fetched {len(teams_data)} teams")
            
            for team_item in teams_data:
                team = self._parse_team(team_item, sport_code)
                if team:
                    teams.append(team)
                    
        except Exception as e:
            logger.error(f"[SportsDB] Failed to get teams for {sport_code}: {e}")
            
            # Fallback to V1 API
            try:
                url = f"{self.v1_base_url}/lookup_all_teams.php?id={league_id}"
                client = await self.get_client()
                response = await client.get(url, headers={"Accept": "application/json"})
                response.raise_for_status()
                data = response.json()
                
                teams_data = data.get("teams") or []
                for team_item in teams_data:
                    team = self._parse_team(team_item, sport_code)
                    if team:
                        teams.append(team)
            except Exception as e2:
                logger.error(f"[SportsDB V1] Teams fallback failed: {e2}")
        
        return teams
    
    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================
    
    async def collect_historical(
        self,
        sport_code: str = None,
        season: str = None,
        seasons_back: int = 5,
    ) -> CollectorResult:
        """
        Collect historical game results by season.
        
        Args:
            sport_code: Sport to collect (or all if None)
            season: Specific season (e.g., "2024-2025") or None for recent
            seasons_back: Number of seasons to collect if season is None
            
        Returns:
            CollectorResult with historical games
        """
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(SPORTSDB_LEAGUE_IDS.keys())
        )
        
        all_games = []
        errors = []
        
        for sport in sports_to_collect:
            if sport not in SPORTSDB_LEAGUE_IDS:
                continue
                
            try:
                if season:
                    # Collect specific season
                    games = await self._collect_season_events(sport, season)
                    all_games.extend(games)
                else:
                    # Collect multiple seasons
                    seasons = self._get_recent_seasons(sport, seasons_back)
                    for s in seasons:
                        try:
                            games = await self._collect_season_events(sport, s)
                            all_games.extend(games)
                            await asyncio.sleep(0.5)  # Rate limit
                        except Exception as e:
                            logger.warning(f"[SportsDB] Season {s} error: {e}")
                            
            except Exception as e:
                logger.error(f"[SportsDB] Historical error for {sport}: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        logger.info(f"[SportsDB Historical] Total: {len(all_games)} games collected")
        
        return CollectorResult(
            success=len(all_games) > 0,
            data=all_games,
            records_count=len(all_games),
            error="; ".join(errors) if errors else None,
            metadata={"seasons_collected": seasons_back},
        )
    
    async def _collect_season_events(
        self,
        sport_code: str,
        season: str,
    ) -> List[Dict[str, Any]]:
        """Collect all events for a specific season using V1 API."""
        league_id = SPORTSDB_LEAGUE_IDS.get(sport_code)
        if not league_id:
            return []
        
        games = []
        
        try:
            # V1 API: Events by season (more reliable for historical)
            url = f"{self.v1_base_url}/eventsseason.php?id={league_id}&s={season}"
            
            client = await self.get_client()
            response = await client.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()
            
            events = data.get("events") or []
            logger.info(f"[SportsDB] {sport_code} {season}: {len(events)} events")
            
            for event in events:
                game = self._parse_event(event, sport_code, is_result=True)
                if game:
                    games.append(game)
                    
        except Exception as e:
            logger.error(f"[SportsDB] Season {season} error: {e}")
        
        return games
    
    def _get_recent_seasons(self, sport_code: str, count: int = 5) -> List[str]:
        """Generate list of recent season strings."""
        seasons = []
        current_year = datetime.now().year
        
        # Different sports have different season formats
        if sport_code in ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL"]:
            # These sports span calendar years (e.g., 2024-2025)
            for i in range(count):
                year = current_year - i
                seasons.append(f"{year}-{year+1}")
        else:
            # MLB and others use single year
            for i in range(count):
                year = current_year - i
                seasons.append(str(year))
        
        return seasons
    
    # =========================================================================
    # PARSING METHODS
    # =========================================================================
    
    def _parse_event(
        self,
        event: Dict[str, Any],
        sport_code: str,
        is_result: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Parse TheSportsDB event to normalized game record."""
        try:
            event_id = event.get("idEvent")
            if not event_id:
                return None
            
            # Get teams
            home_team_name = event.get("strHomeTeam") or event.get("strTeam1")
            away_team_name = event.get("strAwayTeam") or event.get("strTeam2")
            
            if not home_team_name or not away_team_name:
                return None
            
            # Get scores
            home_score = event.get("intHomeScore")
            away_score = event.get("intAwayScore")
            
            # Parse scores to int
            try:
                home_score = int(home_score) if home_score and home_score != "" else None
            except (ValueError, TypeError):
                home_score = None
            try:
                away_score = int(away_score) if away_score and away_score != "" else None
            except (ValueError, TypeError):
                away_score = None
            
            # Get status
            status_raw = event.get("strStatus") or event.get("strProgress") or ""
            status = STATUS_MAP.get(status_raw, "scheduled")
            
            # If we have scores and it's a result, mark as final
            if is_result and home_score is not None and away_score is not None:
                status = "final"
            
            # Parse date
            date_str = event.get("dateEvent") or event.get("strDate")
            time_str = event.get("strTime") or event.get("strTimeLocal") or "00:00"
            
            if date_str:
                try:
                    # Handle different date formats
                    if "T" in str(date_str):
                        game_date = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                    else:
                        # Combine date and time
                        datetime_str = f"{date_str} {time_str}"
                        game_date = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                except Exception:
                    game_date = datetime.now()
            else:
                game_date = datetime.now()
            
            # Get venue
            venue_name = event.get("strVenue") or ""
            
            return {
                "sport_code": sport_code,
                "external_id": f"sportsdb_{event_id}",
                "name": event.get("strEvent") or f"{away_team_name} @ {home_team_name}",
                "home_team": {
                    "id": event.get("idHomeTeam"),
                    "name": home_team_name,
                    "abbreviation": self._get_abbreviation(home_team_name),
                },
                "away_team": {
                    "id": event.get("idAwayTeam"),
                    "name": away_team_name,
                    "abbreviation": self._get_abbreviation(away_team_name),
                },
                "venue": {
                    "name": venue_name,
                },
                "game_date": game_date.isoformat(),
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "season": event.get("strSeason"),
                "round": event.get("intRound"),
            }
            
        except Exception as e:
            logger.warning(f"[SportsDB] Failed to parse event: {e}")
            return None
    
    def _parse_livescore(
        self,
        event: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse livescore event."""
        try:
            event_id = event.get("idEvent") or event.get("idLiveScore")
            if not event_id:
                return None
            
            home_team = event.get("strHomeTeam")
            away_team = event.get("strAwayTeam")
            
            if not home_team or not away_team:
                return None
            
            # Get scores
            home_score = event.get("intHomeScore")
            away_score = event.get("intAwayScore")
            
            try:
                home_score = int(home_score) if home_score else 0
            except (ValueError, TypeError):
                home_score = 0
            try:
                away_score = int(away_score) if away_score else 0
            except (ValueError, TypeError):
                away_score = 0
            
            # Get progress/status
            progress = event.get("strProgress") or event.get("strStatus") or ""
            status = STATUS_MAP.get(progress, "in_progress")
            
            return {
                "sport_code": sport_code,
                "external_id": f"sportsdb_{event_id}",
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "status": status,
                "progress": progress,
                "clock": event.get("strEventTime") or "",
                "league": event.get("strLeague"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.warning(f"[SportsDB] Failed to parse livescore: {e}")
            return None
    
    def _parse_team(
        self,
        team: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse team data to normalized record."""
        try:
            team_id = team.get("idTeam")
            team_name = team.get("strTeam")
            
            if not team_name:
                return None
            
            return {
                "sport_code": sport_code,
                "external_id": f"sportsdb_{team_id}" if team_id else None,
                "name": team_name,
                "abbreviation": team.get("strTeamShort") or self._get_abbreviation(team_name),
                "city": team.get("strStadiumLocation"),
                "stadium": team.get("strStadium"),
                "logo_url": team.get("strBadge") or team.get("strTeamBadge"),
                "banner_url": team.get("strTeamBanner"),
                "description": team.get("strDescriptionEN"),
                "founded": team.get("intFormedYear"),
                "website": team.get("strWebsite"),
            }
            
        except Exception as e:
            logger.warning(f"[SportsDB] Failed to parse team: {e}")
            return None
    
    def _get_abbreviation(self, team_name: str) -> str:
        """Generate team abbreviation from name."""
        if not team_name:
            return "UNK"
        
        # Common abbreviations
        abbrevs = {
            "New England Patriots": "NE",
            "Kansas City Chiefs": "KC",
            "San Francisco 49ers": "SF",
            "Los Angeles Lakers": "LAL",
            "Los Angeles Clippers": "LAC",
            "Los Angeles Rams": "LAR",
            "Los Angeles Chargers": "LAC",
            "Los Angeles Dodgers": "LAD",
            "Los Angeles Angels": "LAA",
            "New York Yankees": "NYY",
            "New York Mets": "NYM",
            "New York Giants": "NYG",
            "New York Jets": "NYJ",
            "New York Knicks": "NYK",
            "New York Rangers": "NYR",
            "New York Islanders": "NYI",
        }
        
        if team_name in abbrevs:
            return abbrevs[team_name]
        
        # Split and take first letter of significant words
        words = team_name.split()
        if len(words) >= 2:
            return (words[0][0] + words[-1][0]).upper()
        return team_name[:3].upper()
    
    def _get_sport_from_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Determine sport code from event data."""
        sport = event.get("strSport")
        league = event.get("strLeague")
        
        if sport == "American Football" or "NFL" in str(league):
            return "NFL"
        elif sport == "Basketball" or "NBA" in str(league):
            return "NBA"
        elif sport == "Ice Hockey" or "NHL" in str(league):
            return "NHL"
        elif sport == "Baseball" or "MLB" in str(league):
            return "MLB"
        
        return None
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    async def save_to_database(
        self,
        data: Dict[str, List[Dict]],
        session: AsyncSession,
    ) -> int:
        """
        Save collected data to database.
        
        Args:
            data: Collected data with games, livescores, teams
            session: Database session
            
        Returns:
            Total records saved/updated
        """
        total_saved = 0
        
        # Save teams first
        if data.get("teams"):
            saved = await self._save_teams(data["teams"], session)
            total_saved += saved
        
        # Save games
        if data.get("games"):
            saved = await self._save_games(data["games"], session)
            total_saved += saved
        
        # Update from livescores
        if data.get("livescores"):
            updated = await self._update_livescores(data["livescores"], session)
            total_saved += updated
        
        return total_saved
    
    async def _save_teams(
        self,
        teams_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save team records to database."""
        saved_count = 0
        
        for team_data in teams_data:
            try:
                sport_code = team_data["sport_code"]
                
                # Get sport
                sport_result = await session.execute(
                    select(Sport).where(Sport.code == sport_code)
                )
                sport = sport_result.scalar_one_or_none()
                
                if not sport:
                    continue
                
                team_name = team_data.get("name")
                if not team_name:
                    continue
                
                # Check if team exists
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.name == team_name,
                        )
                    )
                )
                team = result.scalar_one_or_none()
                
                if not team:
                    # Create new team
                    team = Team(
                        sport_id=sport.id,
                        external_id=team_data.get("external_id"),
                        name=team_name,
                        abbreviation=team_data.get("abbreviation"),
                        is_active=True,
                    )
                    session.add(team)
                    saved_count += 1
                else:
                    # Update existing
                    if team_data.get("external_id") and not team.external_id:
                        team.external_id = team_data["external_id"]
                    if team_data.get("abbreviation"):
                        team.abbreviation = team_data["abbreviation"]
                        
            except Exception as e:
                logger.error(f"[SportsDB] Error saving team: {e}")
                continue
        
        await session.commit()
        return saved_count
    
    async def _save_games(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save game records to database."""
        saved_count = 0
        
        for game_data in games_data:
            try:
                sport_code = game_data["sport_code"]
                
                # Get sport
                sport_result = await session.execute(
                    select(Sport).where(Sport.code == sport_code)
                )
                sport = sport_result.scalar_one_or_none()
                
                if not sport:
                    continue
                
                # Get or create teams
                home_team = await self._get_or_create_team(
                    session,
                    sport.id,
                    game_data["home_team"],
                )
                away_team = await self._get_or_create_team(
                    session,
                    sport.id,
                    game_data["away_team"],
                )
                
                if not home_team or not away_team:
                    continue
                
                # Check if game exists by external_id
                external_id = game_data.get("external_id")
                existing = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = existing.scalars().first()
                
                if game:
                    # Update existing game
                    if game_data.get("home_score") is not None:
                        game.home_score = game_data["home_score"]
                    if game_data.get("away_score") is not None:
                        game.away_score = game_data["away_score"]
                    if game_data.get("status"):
                        game.status = GameStatus(game_data["status"])
                else:
                    # Parse scheduled date
                    game_date_str = game_data.get("game_date")
                    if game_date_str:
                        try:
                            if isinstance(game_date_str, datetime):
                                scheduled_dt = game_date_str
                            else:
                                scheduled_dt = datetime.fromisoformat(
                                    game_date_str.replace("Z", "+00:00")
                                )
                            # Remove timezone info for naive datetime
                            if scheduled_dt.tzinfo is not None:
                                scheduled_dt = scheduled_dt.replace(tzinfo=None)
                        except Exception:
                            scheduled_dt = datetime.now()
                    else:
                        scheduled_dt = datetime.now()
                    
                    # Check for duplicate by teams and date (within 12 hours)
                    date_start = scheduled_dt - timedelta(hours=12)
                    date_end = scheduled_dt + timedelta(hours=12)
                    
                    dup_check = await session.execute(
                        select(Game).where(
                            and_(
                                Game.sport_id == sport.id,
                                Game.home_team_id == home_team.id,
                                Game.away_team_id == away_team.id,
                                Game.scheduled_at >= date_start,
                                Game.scheduled_at <= date_end,
                            )
                        )
                    )
                    existing_game = dup_check.scalars().first()
                    
                    if existing_game:
                        # Update existing
                        if game_data.get("home_score") is not None:
                            existing_game.home_score = game_data["home_score"]
                        if game_data.get("away_score") is not None:
                            existing_game.away_score = game_data["away_score"]
                        if game_data.get("status"):
                            existing_game.status = GameStatus(game_data["status"])
                        if external_id and not existing_game.external_id:
                            existing_game.external_id = external_id
                    else:
                        # Create new game
                        game = Game(
                            sport_id=sport.id,
                            external_id=external_id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            scheduled_at=scheduled_dt,
                            status=GameStatus(game_data.get("status", "scheduled")),
                            home_score=game_data.get("home_score"),
                            away_score=game_data.get("away_score"),
                        )
                        session.add(game)
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"[SportsDB] Error saving game: {e}")
                continue
        
        await session.commit()
        return saved_count
    
    async def _update_livescores(
        self,
        livescores: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Update games with livescore data."""
        updated_count = 0
        
        for live in livescores:
            try:
                external_id = live.get("external_id")
                if not external_id:
                    continue
                
                # Find game by external_id
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = result.scalars().first()
                
                if game:
                    game.home_score = live.get("home_score")
                    game.away_score = live.get("away_score")
                    
                    status = live.get("status", "in_progress")
                    game.status = GameStatus(status)
                    
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"[SportsDB] Error updating livescore: {e}")
                continue
        
        await session.commit()
        return updated_count
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_data: Dict[str, Any],
    ) -> Optional[Team]:
        """Get or create team record."""
        team_name = team_data.get("name")
        if not team_name:
            return None
        
        # Try by name first
        result = await session.execute(
            select(Team).where(
                and_(
                    Team.sport_id == sport_id,
                    Team.name == team_name,
                )
            )
        )
        team = result.scalar_one_or_none()
        
        if team:
            return team
        
        # Try by abbreviation
        abbreviation = team_data.get("abbreviation")
        if abbreviation:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.abbreviation == abbreviation,
                    )
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Create new team
        external_id = team_data.get("id")
        team = Team(
            sport_id=sport_id,
            external_id=f"sportsdb_{external_id}" if external_id else abbreviation,
            name=team_name,
            abbreviation=abbreviation or team_name[:3].upper(),
            is_active=True,
        )
        session.add(team)
        await session.flush()
        
        return team
    
    async def save_historical_to_database(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> Tuple[int, int]:
        """
        Save historical game data to database.
        
        Returns:
            Tuple of (saved_count, updated_count)
        """
        saved_count = 0
        updated_count = 0
        
        for game_data in games_data:
            try:
                result = await self._save_single_game(game_data, session)
                if result == "saved":
                    saved_count += 1
                elif result == "updated":
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"[SportsDB] Error saving historical game: {e}")
                continue
        
        await session.commit()
        logger.info(f"[SportsDB Historical] Saved: {saved_count}, Updated: {updated_count}")
        
        return saved_count, updated_count
    
    async def _save_single_game(
        self,
        game_data: Dict[str, Any],
        session: AsyncSession,
    ) -> Optional[str]:
        """Save a single game record. Returns 'saved', 'updated', or None."""
        sport_code = game_data.get("sport_code")
        
        # Get sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            return None
        
        # Get or create teams
        home_team = await self._get_or_create_team(
            session, sport.id, game_data["home_team"]
        )
        away_team = await self._get_or_create_team(
            session, sport.id, game_data["away_team"]
        )
        
        if not home_team or not away_team:
            return None
        
        # Parse date
        game_date_str = game_data.get("game_date")
        if game_date_str:
            try:
                if isinstance(game_date_str, datetime):
                    scheduled_dt = game_date_str
                else:
                    scheduled_dt = datetime.fromisoformat(
                        game_date_str.replace("Z", "+00:00")
                    )
                if scheduled_dt.tzinfo is not None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=None)
            except Exception:
                scheduled_dt = datetime.now()
        else:
            return None
        
        # Check for existing game
        external_id = game_data.get("external_id")
        
        # First check by external_id
        if external_id:
            existing = await session.execute(
                select(Game).where(Game.external_id == external_id)
            )
            game = existing.scalars().first()
            
            if game:
                # Update scores
                if game_data.get("home_score") is not None:
                    game.home_score = game_data["home_score"]
                if game_data.get("away_score") is not None:
                    game.away_score = game_data["away_score"]
                if game_data.get("status"):
                    game.status = GameStatus(game_data["status"])
                return "updated"
        
        # Check by teams and date
        date_start = scheduled_dt - timedelta(hours=12)
        date_end = scheduled_dt + timedelta(hours=12)
        
        existing = await session.execute(
            select(Game).where(
                and_(
                    Game.sport_id == sport.id,
                    Game.home_team_id == home_team.id,
                    Game.away_team_id == away_team.id,
                    Game.scheduled_at >= date_start,
                    Game.scheduled_at <= date_end,
                )
            )
        )
        game = existing.scalars().first()
        
        if game:
            # Update existing
            if game_data.get("home_score") is not None:
                game.home_score = game_data["home_score"]
            if game_data.get("away_score") is not None:
                game.away_score = game_data["away_score"]
            if game_data.get("status"):
                game.status = GameStatus(game_data["status"])
            if external_id and not game.external_id:
                game.external_id = external_id
            return "updated"
        
        # Create new game
        game = Game(
            sport_id=sport.id,
            external_id=external_id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            scheduled_at=scheduled_dt,
            status=GameStatus(game_data.get("status", "final")),
            home_score=game_data.get("home_score"),
            away_score=game_data.get("away_score"),
        )
        session.add(game)
        
        return "saved"
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not isinstance(data, dict):
            return False
        
        # Check for at least some valid data
        games = data.get("games", [])
        teams = data.get("teams", [])
        livescores = data.get("livescores", [])
        
        return len(games) > 0 or len(teams) > 0 or len(livescores) > 0


# Create singleton instance
sportsdb_collector = SportsDBCollector()

# Register with collector manager
try:
    collector_manager.register("sportsdb", sportsdb_collector)
except:
    pass  # Already registered
