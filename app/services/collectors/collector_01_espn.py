"""
ROYALEY - ESPN Data Collector
Phase 1: Data Collection Services

Collects game schedules, scores, and team information from ESPN public API.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Sport, Team, Game, GameStatus
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
                    game = Game(
                        sport_id=sport.id,
                        external_id=game_data["external_id"],
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=datetime.fromisoformat(
                            game_data["game_date"].replace("Z", "+00:00")
                        ),
                        status=GameStatus(game_data["status"]),
                        broadcast=game_data.get("broadcast"),
                    )
                    session.add(game)
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving game: {e}")
                await session.rollback()  # Reset transaction state
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


# Create and register collector instance
espn_collector = ESPNCollector()
collector_manager.register(espn_collector)
