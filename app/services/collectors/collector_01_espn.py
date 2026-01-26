"""
ROYALEY - ESPN Data Collector
Phase 1: Data Collection Services

Collects game schedules, scores, team information, injuries, and players from ESPN public API.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Sport, Team, Game, GameStatus, Player
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# ESPN API sport mappings
ESPN_SPORT_PATHS = {
    "NFL": {"sport": "football", "league": "nfl"},
    "NCAAF": {"sport": "football", "league": "college-football"},
    "CFL": {"sport": "football", "league": "cfl"},
    "NBA": {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "mens-college-basketball"},
    "WNBA": {"sport": "basketball", "league": "wnba"},
    "NHL": {"sport": "hockey", "league": "nhl"},
    "MLB": {"sport": "baseball", "league": "mlb"},
    "ATP": {"sport": "tennis", "league": "atp"},
    "WTA": {"sport": "tennis", "league": "wta"},
}


class ESPNCollector(BaseCollector):
    """
    Collector for ESPN public API.
    
    Features:
    - Game schedules and results
    - Team information
    - Scores and standings
    - Injuries
    - Players
    - No API key required
    """
    
    def __init__(self):
        super().__init__(
            name="espn",
            base_url="https://site.api.espn.com/apis/site/v2/sports",
            rate_limit=120,  # requests per minute
            rate_window=60,
        )
    
    async def collect(
        self,
        sport_code: str = None,
        collect_type: str = "all",
        days_ahead: int = 7,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect data from ESPN.
        
        Args:
            sport_code: Optional sport code (collects all if None)
            collect_type: Type of data (schedule, scores, teams, all)
            days_ahead: Number of days ahead for schedule
            
        Returns:
            CollectorResult with collected data
        """
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(ESPN_SPORT_PATHS.keys())
        )
        
        all_data = {
            "games": [],
            "scores": [],
            "teams": [],
        }
        errors = []
        
        for sport in sports_to_collect:
            if sport not in ESPN_SPORT_PATHS:
                continue
                
            try:
                if collect_type in ["schedule", "all"]:
                    games = await self._collect_schedule(sport, days_ahead)
                    all_data["games"].extend(games)
                
                if collect_type in ["scores", "all"]:
                    scores = await self._collect_scores(sport)
                    all_data["scores"].extend(scores)
                
                if collect_type in ["teams", "all"]:
                    teams = await self._collect_teams(sport)
                    all_data["teams"].extend(teams)
                    
            except Exception as e:
                logger.error(f"Error collecting {sport} from ESPN: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        total_records = sum(len(v) for v in all_data.values())
        
        return CollectorResult(
            success=len(errors) == 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={
                "sports_collected": sports_to_collect,
                "collect_type": collect_type,
            },
        )
    
    async def _collect_schedule(
        self,
        sport_code: str,
        days_ahead: int = 7,
    ) -> List[Dict[str, Any]]:
        """Collect upcoming game schedule."""
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            return []
        
        endpoint = f"/{sport_path['sport']}/{sport_path['league']}/scoreboard"
        games = []
        
        # Collect for each day
        for day_offset in range(days_ahead):
            date = datetime.utcnow() + timedelta(days=day_offset)
            date_str = date.strftime("%Y%m%d")
            
            try:
                data = await self.get(endpoint, params={"dates": date_str})
                events = data.get("events", [])
                
                for event in events:
                    game = self._parse_event(event, sport_code)
                    if game:
                        games.append(game)
                        
            except Exception as e:
                logger.warning(f"Failed to get schedule for {date_str}: {e}")
        
        return games
    
    async def _collect_scores(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect current and recent scores."""
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            return []
        
        endpoint = f"/{sport_path['sport']}/{sport_path['league']}/scoreboard"
        
        try:
            data = await self.get(endpoint)
            events = data.get("events", [])
            
            scores = []
            for event in events:
                score = self._parse_score(event, sport_code)
                if score:
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to collect scores for {sport_code}: {e}")
            return []
    
    async def _collect_teams(self, sport_code: str) -> List[Dict[str, Any]]:
        """Collect team information."""
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            return []
        
        endpoint = f"/{sport_path['sport']}/{sport_path['league']}/teams"
        
        try:
            data = await self.get(endpoint, params={"limit": 500})
            teams_data = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            
            teams = []
            for team_item in teams_data:
                team = self._parse_team(team_item.get("team", {}), sport_code)
                if team:
                    teams.append(team)
            
            return teams
            
        except Exception as e:
            logger.error(f"Failed to collect teams for {sport_code}: {e}")
            return []
    
    def _parse_event(
        self,
        event: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse ESPN event to normalized game record."""
        try:
            event_id = event.get("id")
            name = event.get("name", "")
            
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) != 2:
                return None
            
            # Find home and away teams
            home_team = None
            away_team = None
            
            for competitor in competitors:
                if competitor.get("homeAway") == "home":
                    home_team = competitor
                else:
                    away_team = competitor
            
            if not home_team or not away_team:
                return None
            
            # Parse venue
            venue = competition.get("venue", {})
            
            # Parse status
            status_data = competition.get("status", {})
            status = self._map_status(status_data.get("type", {}).get("name", ""))
            
            return {
                "sport_code": sport_code,
                "external_id": event_id,
                "name": name,
                "home_team": {
                    "id": home_team.get("team", {}).get("id"),
                    "name": home_team.get("team", {}).get("displayName"),
                    "abbreviation": home_team.get("team", {}).get("abbreviation"),
                },
                "away_team": {
                    "id": away_team.get("team", {}).get("id"),
                    "name": away_team.get("team", {}).get("displayName"),
                    "abbreviation": away_team.get("team", {}).get("abbreviation"),
                },
                "venue": {
                    "id": venue.get("id"),
                    "name": venue.get("fullName"),
                    "city": venue.get("address", {}).get("city"),
                    "state": venue.get("address", {}).get("state"),
                },
                "game_date": event.get("date"),
                "status": status,
                "broadcast": self._get_broadcast(competition),
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None
    
    def _parse_score(
        self,
        event: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse ESPN event to score record."""
        try:
            event_id = event.get("id")
            
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            home_score = None
            away_score = None
            
            for competitor in competitors:
                score = competitor.get("score")
                if competitor.get("homeAway") == "home":
                    home_score = int(score) if score else None
                else:
                    away_score = int(score) if score else None
            
            status_data = competition.get("status", {})
            status = self._map_status(status_data.get("type", {}).get("name", ""))
            
            return {
                "sport_code": sport_code,
                "external_id": event_id,
                "home_score": home_score,
                "away_score": away_score,
                "status": status,
                "period": status_data.get("period"),
                "clock": status_data.get("displayClock"),
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse score: {e}")
            return None
    
    def _parse_team(
        self,
        team: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse ESPN team to normalized record."""
        try:
            return {
                "sport_code": sport_code,
                "external_id": team.get("id"),
                "name": team.get("displayName"),
                "abbreviation": team.get("abbreviation"),
                "city": team.get("location"),
                "logo_url": team.get("logos", [{}])[0].get("href") if team.get("logos") else None,
            }
        except Exception as e:
            logger.warning(f"Failed to parse team: {e}")
            return None
    
    def _map_status(self, espn_status: str) -> str:
        """Map ESPN status to internal status."""
        status_map = {
            "STATUS_SCHEDULED": "scheduled",
            "STATUS_IN_PROGRESS": "in_progress",
            "STATUS_HALFTIME": "in_progress",
            "STATUS_FINAL": "final",
            "STATUS_POSTPONED": "postponed",
            "STATUS_CANCELED": "cancelled",
            "STATUS_DELAYED": "scheduled",
        }
        return status_map.get(espn_status, "scheduled")
    
    def _get_broadcast(self, competition: Dict[str, Any]) -> Optional[str]:
        """Extract broadcast information."""
        broadcasts = competition.get("broadcasts", [])
        if broadcasts:
            names = broadcasts[0].get("names", [])
            if names:
                return names[0]
        return None
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not isinstance(data, dict):
            return False
        
        for game in data.get("games", []):
            if not all([
                game.get("external_id"),
                game.get("home_team"),
                game.get("away_team"),
                game.get("game_date"),
            ]):
                return False
        
        return True
    
    async def save_games_to_database(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """
        Save collected games to database.
        
        Args:
            games_data: List of parsed game records
            session: Database session
            
        Returns:
            Number of records saved
        """
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
                
                # Check if game exists
                existing = await session.execute(
                    select(Game).where(
                        Game.sport_id == sport.id,
                        Game.external_id == game_data["external_id"],
                    )
                )
                game = existing.scalar_one_or_none()
                
                if game:
                    # Update existing
                    game.status = GameStatus(game_data["status"])
                else:
                    # Create new
                    scheduled_dt = datetime.fromisoformat(
                        game_data["game_date"].replace("Z", "+00:00")
                    )
                    # Convert to naive datetime (remove timezone info)
                    if scheduled_dt.tzinfo is not None:
                        scheduled_dt = scheduled_dt.replace(tzinfo=None)
                    
                    game = Game(
                        sport_id=sport.id,
                        external_id=game_data["external_id"],
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=scheduled_dt,
                        status=GameStatus(game_data["status"]),
                    )
                    session.add(game)
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving game: {e}")
                await session.rollback()
                continue
        
        await session.commit()
        return saved_count
    
    async def save_scores_to_database(
        self,
        scores_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Update game scores in database."""
        updated_count = 0
        
        for score_data in scores_data:
            try:
                result = await session.execute(
                    select(Game).where(Game.external_id == score_data["external_id"])
                )
                game = result.scalar_one_or_none()
                
                if game:
                    game.home_score = score_data.get("home_score")
                    game.away_score = score_data.get("away_score")
                    game.status = GameStatus(score_data["status"])
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Error updating score: {e}")
                continue
        
        await session.commit()
        return updated_count
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_data: Dict[str, Any],
    ) -> Team:
        """Get or create team record."""
        external_id = team_data.get("id")
        team_name = team_data.get("name")
        abbreviation = team_data.get("abbreviation")
        
        # First try by name (this is the unique constraint)
        if team_name:
            result = await session.execute(
                select(Team).where(
                    Team.sport_id == sport_id,
                    Team.name == team_name,
                )
            )
            team = result.scalar_one_or_none()
            if team:
                # Update external_id if different
                if external_id and team.external_id != external_id:
                    team.external_id = external_id
                return team
        
        # Try by external_id
        if external_id:
            result = await session.execute(
                select(Team).where(
                    Team.sport_id == sport_id,
                    Team.external_id == external_id,
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Try by abbreviation
        if abbreviation:
            result = await session.execute(
                select(Team).where(
                    Team.sport_id == sport_id,
                    Team.abbreviation == abbreviation,
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Create new team
        team = Team(
            sport_id=sport_id,
            external_id=external_id or abbreviation,
            name=team_name,
            abbreviation=abbreviation,
            elo_rating=1500.0,
        )
        session.add(team)
        await session.flush()
        
        return team
    
    async def get_standings(self, sport_code: str) -> List[Dict[str, Any]]:
        """Get current standings for a sport."""
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            return []
        
        endpoint = f"/{sport_path['sport']}/{sport_path['league']}/standings"
        
        try:
            data = await self.get(endpoint)
            return data.get("standings", [])
        except Exception as e:
            logger.error(f"Failed to get standings for {sport_code}: {e}")
            return []
    
    async def collect_historical(
        self,
        sport_code: str = None,
        days_back: int = 365,
        batch_size: int = 7,
    ) -> CollectorResult:
        """
        Collect historical game data from ESPN.
        
        Fetches past game schedules, scores, and results for ML training.
        ESPN API allows fetching any past date via the scoreboard endpoint.
        
        Args:
            sport_code: Specific sport (NFL, NBA, etc.) or None for all
            days_back: Number of days to fetch (default 365)
            batch_size: Days per batch for progress logging
            
        Returns:
            CollectorResult with historical games
        """
        sports = [sport_code.upper()] if sport_code else ["NFL", "NBA", "NHL", "MLB"]
        all_games = []
        errors = []
        
        for sport in sports:
            sport_path = ESPN_SPORT_PATHS.get(sport)
            if not sport_path:
                continue
                
            try:
                games = await self._fetch_historical_games(sport, sport_path, days_back)
                all_games.extend(games)
                logger.info(f"[ESPN Historical] {sport}: {len(games)} games")
            except Exception as e:
                errors.append(f"{sport}: {str(e)}")
                logger.error(f"[ESPN Historical] Error collecting {sport}: {e}")
        
        return CollectorResult(
            success=len(all_games) > 0,
            data={"games": all_games, "teams": [], "scores": []},
            records_count=len(all_games),
            error="; ".join(errors) if errors else None,
            metadata={"type": "historical_games", "sports": sports, "days_back": days_back}
        )
    
    async def _fetch_historical_games(
        self,
        sport_code: str,
        sport_path: Dict[str, str],
        days_back: int,
    ) -> List[Dict[str, Any]]:
        """Fetch historical games for a specific sport."""
        all_games = []
        endpoint = f"/{sport_path['sport']}/{sport_path['league']}/scoreboard"
        
        logger.info(f"[ESPN Historical] Fetching {sport_code} games for past {days_back} days...")
        
        for day_offset in range(1, days_back + 1):
            target_date = datetime.utcnow() - timedelta(days=day_offset)
            date_str = target_date.strftime("%Y%m%d")
            
            try:
                data = await self.get(endpoint, params={"dates": date_str})
                
                events = data.get("events", [])
                for event in events:
                    parsed = self._parse_event(event, sport_code)
                    if parsed:
                        # Also parse scores for historical data
                        score_data = self._parse_score(event, sport_code)
                        if score_data:
                            parsed["home_score"] = score_data.get("home_score")
                            parsed["away_score"] = score_data.get("away_score")
                            # Update status from score data (more accurate for completed games)
                            parsed["status"] = score_data.get("status", parsed["status"])
                        all_games.append(parsed)
                
                # Progress logging every 30 days
                if day_offset % 30 == 0:
                    logger.info(f"[ESPN Historical] {sport_code}: {day_offset}/{days_back} days, {len(all_games)} games found")
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"[ESPN Historical] No data for {date_str}: {e}")
                continue
        
        return all_games
    
    async def save_historical_to_database(
        self,
        games_data: List[Dict[str, Any]],
        sport_code: str,
        session,
    ) -> Tuple[int, int]:
        """Save historical games to database."""
        from app.models.models import Game, Team, Sport, GameStatus
        from sqlalchemy import select
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        if not sport:
            logger.error(f"Sport {sport_code} not found in database")
            return 0, 0
        
        saved = 0
        updated = 0
        
        for game_data in games_data:
            try:
                # Check if game exists
                result = await session.execute(
                    select(Game).where(
                        Game.external_id == game_data["external_id"],
                        Game.sport_id == sport.id
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update scores if available
                    if game_data.get("home_score") is not None:
                        existing.home_score = game_data["home_score"]
                        existing.away_score = game_data["away_score"]
                        existing.status = GameStatus(game_data["status"])
                        updated += 1
                else:
                    # Extract team data - handle both nested dict and flat formats
                    home_team_data = game_data.get("home_team", {})
                    away_team_data = game_data.get("away_team", {})
                    
                    # Handle nested format from _parse_event
                    if isinstance(home_team_data, dict):
                        home_team_info = {
                            "id": home_team_data.get("id"),
                            "name": home_team_data.get("name"),
                            "abbreviation": home_team_data.get("abbreviation")
                        }
                    else:
                        # Handle flat format
                        home_team_info = {
                            "id": game_data.get("home_team_id"),
                            "name": home_team_data,
                            "abbreviation": game_data.get("home_abbreviation")
                        }
                    
                    if isinstance(away_team_data, dict):
                        away_team_info = {
                            "id": away_team_data.get("id"),
                            "name": away_team_data.get("name"),
                            "abbreviation": away_team_data.get("abbreviation")
                        }
                    else:
                        away_team_info = {
                            "id": game_data.get("away_team_id"),
                            "name": away_team_data,
                            "abbreviation": game_data.get("away_abbreviation")
                        }
                    
                    # Get or create teams
                    home_team = await self._get_or_create_team(
                        session, sport.id, home_team_info
                    )
                    away_team = await self._get_or_create_team(
                        session, sport.id, away_team_info
                    )
                    
                    # Parse datetime
                    scheduled_dt = datetime.fromisoformat(
                        game_data["game_date"].replace("Z", "+00:00")
                    )
                    if scheduled_dt.tzinfo is not None:
                        scheduled_dt = scheduled_dt.replace(tzinfo=None)
                    
                    # Create game
                    game = Game(
                        sport_id=sport.id,
                        external_id=game_data["external_id"],
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=scheduled_dt,
                        status=GameStatus(game_data["status"]),
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                    )
                    session.add(game)
                    saved += 1
                
                # Commit every 100 records
                if (saved + updated) % 100 == 0:
                    await session.commit()
                    
            except Exception as e:
                logger.debug(f"Skipping game {game_data.get('external_id')}: {e}")
                await session.rollback()
                continue
        
        await session.commit()
        return saved, updated

    # =========================================================================
    # INJURIES COLLECTION
    # =========================================================================
    
    async def collect_injuries(self, sport_code: str) -> Dict[str, Any]:
        """
        Collect injury reports from ESPN.
        
        ESPN provides injury data at:
        - https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/injuries
        
        Response structure:
        {
            "injuries": [
                {
                    "id": "22",
                    "displayName": "Arizona Cardinals",
                    "injuries": [
                        {
                            "athlete": {"displayName": "...", "position": {...}, "team": {...}},
                            "status": "Injured Reserve",
                            "type": {"description": "Injured Reserve"},
                            "details": {"type": "Knee - MCL", "returnDate": "..."}
                        }
                    ]
                }
            ]
        }
        """
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            logger.warning(f"[ESPN] Injuries not supported for {sport_code}")
            return {"injuries": []}
        
        injuries = []
        
        try:
            # Try injuries endpoint
            endpoint = f"/{sport_path['sport']}/{sport_path['league']}/injuries"
            
            try:
                data = await self.get(endpoint)
                
                # Parse team injuries - ESPN wraps in "injuries" at top level
                # Each item is a team with nested "injuries" list
                teams_with_injuries = data.get("injuries", [])
                
                for team_data in teams_with_injuries:
                    # Team info is at team level
                    team_name = team_data.get("displayName", "")
                    team_id = team_data.get("id", "")
                    
                    # Injuries are nested within each team
                    team_injuries = team_data.get("injuries", [])
                    
                    for injury in team_injuries:
                        athlete = injury.get("athlete", {})
                        athlete_team = athlete.get("team", {})
                        injury_type = injury.get("type", {})
                        injury_details = injury.get("details", {})
                        
                        # Get team abbreviation from athlete's team info
                        team_abbr = athlete_team.get("abbreviation", "")
                        if not team_abbr and team_name:
                            # Fallback - extract from team name
                            team_abbr = team_id
                        
                        injuries.append({
                            "player_name": athlete.get("displayName", ""),
                            "player_id": athlete.get("id"),
                            "position": athlete.get("position", {}).get("abbreviation", ""),
                            "team_name": team_name or athlete_team.get("displayName", ""),
                            "team_abbr": team_abbr,
                            "injury_type": injury_details.get("type", "") or injury_type.get("description", ""),
                            "status": injury.get("status", "") or injury_type.get("description", ""),
                            "details": injury_details.get("detail", ""),
                            "return_date": injury_details.get("returnDate"),
                            "is_starter": athlete.get("starter", False),
                            "short_comment": injury.get("shortComment", ""),
                        })
                
                logger.info(f"[ESPN] Collected {len(injuries)} injuries for {sport_code}")
                
            except Exception as e:
                logger.warning(f"[ESPN] Injuries endpoint error: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            logger.error(f"[ESPN] Error collecting injuries: {e}")
        
        return {"injuries": injuries, "sport_code": sport_code}
    
    async def save_injuries_to_database(
        self, 
        injuries_data: List[Dict[str, Any]], 
        sport_code: str,
        session: AsyncSession
    ) -> int:
        """Save injuries to database."""
        # Import Injury model - handle both possible locations
        try:
            from app.models.injury_models import Injury
        except ImportError:
            try:
                from app.models import Injury
            except ImportError:
                logger.error("[ESPN] Injury model not found")
                return 0
        
        saved_count = 0
        
        # Get sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            logger.error(f"[ESPN] Sport {sport_code} not found")
            return 0
        
        for injury_data in injuries_data:
            try:
                player_name = injury_data.get("player_name", "")
                if not player_name:
                    continue
                
                # Find team by abbreviation or name
                team_abbr = injury_data.get("team_abbr", "")
                team_name = injury_data.get("team_name", "")
                
                team = None
                if team_abbr:
                    team_result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.abbreviation == team_abbr
                            )
                        )
                    )
                    team = team_result.scalar_one_or_none()
                
                if not team and team_name:
                    team_result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name.ilike(f"%{team_name}%")
                            )
                        )
                    )
                    team = team_result.scalar_one_or_none()
                
                if not team:
                    logger.debug(f"[ESPN] Team not found for injury: {team_abbr} / {team_name}")
                    continue
                
                # Check if injury already exists for this player on this team
                existing = await session.execute(
                    select(Injury).where(
                        and_(
                            Injury.team_id == team.id,
                            Injury.player_name == player_name,
                        )
                    )
                )
                injury = existing.scalar_one_or_none()
                
                status = injury_data.get("status", "Unknown")
                injury_type = injury_data.get("injury_type", "")
                
                if injury:
                    # Update existing injury
                    injury.status = status
                    injury.injury_type = injury_type
                    injury.status_detail = injury_data.get("details") or injury_data.get("short_comment")
                    injury.is_starter = injury_data.get("is_starter", False)
                    injury.position = injury_data.get("position") or injury.position
                    # last_updated auto-updates via onupdate
                else:
                    # Create new injury
                    injury = Injury(
                        team_id=team.id,
                        sport_code=sport_code,
                        player_name=player_name,
                        position=injury_data.get("position"),
                        injury_type=injury_type,
                        status=status,
                        status_detail=injury_data.get("details") or injury_data.get("short_comment"),
                        is_starter=injury_data.get("is_starter", False),
                        source="espn",
                        external_id=str(injury_data.get("player_id")) if injury_data.get("player_id") else None,
                    )
                    session.add(injury)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[ESPN] Error saving injury for {injury_data.get('player_name')}: {e}")
                continue
        
        try:
            await session.commit()
            logger.info(f"[ESPN] Saved {saved_count} new injuries for {sport_code}")
        except Exception as e:
            logger.error(f"[ESPN] Error committing injuries: {e}")
            await session.rollback()
            
        return saved_count

    # =========================================================================
    # PLAYERS COLLECTION
    # =========================================================================
    
    async def collect_players(self, sport_code: str) -> Dict[str, Any]:
        """
        Collect player rosters from ESPN.
        
        ESPN provides roster data at team endpoints.
        """
        sport_path = ESPN_SPORT_PATHS.get(sport_code)
        if not sport_path:
            return {"players": []}
        
        players = []
        
        try:
            # Get teams first
            endpoint = f"/{sport_path['sport']}/{sport_path['league']}/teams"
            data = await self.get(endpoint, params={"limit": 500})
            
            teams_data = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            
            for team_item in teams_data:
                team = team_item.get("team", {})
                team_id = team.get("id")
                team_abbr = team.get("abbreviation", "")
                
                if not team_id:
                    continue
                
                # Get roster for each team
                roster_endpoint = f"/{sport_path['sport']}/{sport_path['league']}/teams/{team_id}/roster"
                
                try:
                    roster_data = await self.get(roster_endpoint)
                    
                    for group in roster_data.get("athletes", []):
                        for athlete in group.get("items", []):
                            players.append({
                                "external_id": f"espn_{athlete.get('id')}",
                                "name": athlete.get("displayName", ""),
                                "position": athlete.get("position", {}).get("abbreviation", ""),
                                "jersey_number": athlete.get("jersey"),
                                "team_abbr": team_abbr,
                                "height": athlete.get("displayHeight"),
                                "weight": athlete.get("displayWeight"),
                                "birth_date": athlete.get("dateOfBirth"),
                                "is_active": athlete.get("active", True),
                            })
                            
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"[ESPN] Error getting roster for team {team_id}: {e}")
                    continue
            
            logger.info(f"[ESPN] Collected {len(players)} players for {sport_code}")
            
        except Exception as e:
            logger.error(f"[ESPN] Error collecting players: {e}")
        
        return {"players": players, "sport_code": sport_code}
    
    async def save_players_to_database(
        self,
        players_data: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession
    ) -> int:
        """Save players to database."""
        saved_count = 0
        
        # Get sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            return 0
        
        for player_data in players_data:
            try:
                # Find team
                team_result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.abbreviation == player_data.get("team_abbr")
                        )
                    )
                )
                team = team_result.scalar_one_or_none()
                
                external_id = player_data.get("external_id")
                if not external_id:
                    continue
                
                # Check if player exists
                existing = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                player = existing.scalar_one_or_none()
                
                if player:
                    # Update
                    player.name = player_data.get("name", player.name)
                    player.position = player_data.get("position", player.position)
                    player.team_id = team.id if team else player.team_id
                    player.is_active = player_data.get("is_active", True)
                else:
                    # Parse birth date
                    birth_date = None
                    if player_data.get("birth_date"):
                        try:
                            birth_date = datetime.fromisoformat(
                                player_data["birth_date"].replace("Z", "+00:00")
                            ).date()
                        except:
                            pass
                    
                    # Parse jersey number
                    jersey = None
                    if player_data.get("jersey_number"):
                        try:
                            jersey = int(player_data["jersey_number"])
                        except:
                            pass
                    
                    # Parse weight
                    weight = None
                    if player_data.get("weight"):
                        try:
                            weight_str = str(player_data["weight"]).replace(" lbs", "").strip()
                            weight = int(weight_str)
                        except:
                            pass
                    
                    player = Player(
                        external_id=external_id,
                        name=player_data.get("name", ""),
                        position=player_data.get("position"),
                        jersey_number=jersey,
                        team_id=team.id if team else None,
                        height=player_data.get("height"),
                        weight=weight,
                        birth_date=birth_date,
                        is_active=player_data.get("is_active", True),
                    )
                    session.add(player)
                    saved_count += 1
                    
            except Exception as e:
                logger.debug(f"[ESPN] Error saving player: {e}")
                continue
        
        await session.commit()
        logger.info(f"[ESPN] Saved {saved_count} players")
        return saved_count


# Create and register collector instance
espn_collector = ESPNCollector()
collector_manager.register(espn_collector)
