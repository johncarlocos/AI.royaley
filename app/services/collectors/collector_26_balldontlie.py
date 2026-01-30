"""
BallDontLie API Collector - V2 FIXED
=====================================
Comprehensive multi-sport data collector from BallDontLie API.
$299/month plan - Full access to all sports and endpoints.

Supports 9 sports:
- NBA, NFL, MLB, NHL (Team Sports)
- WNBA, NCAAF, NCAAB (Team Sports)
- ATP, WTA (Tennis - Individual Sport)

FIXES:
- Correct API endpoint paths per sport
- NHL uses box_scores (no /stats endpoint)
- Odds use v1 endpoint (not v2)
- Proper injury handling with nested teams array
- Tennis pseudo-teams for game linking
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from rich.console import Console
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Game, Team, Player, Sport, PlayerStats, TeamStats, 
    Odds, Sportsbook, GameStatus
)
from app.models.injury_models import Injury
from app.services.collectors.base_collector import (
    BaseCollector, CollectorResult
)

logger = logging.getLogger(__name__)
console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
BASE_URL = "https://api.balldontlie.io"

# Sport configurations with CORRECT API paths
# Based on official BallDontLie API documentation
SPORT_CONFIG = {
    "NBA": {
        "code": "NBA",
        "name": "National Basketball Association",
        "is_tennis": False,
        "endpoints": {
            "teams": "/nba/v1/teams",
            "players": "/nba/v1/players",
            "games": "/nba/v1/games",
            "stats": "/nba/v1/stats",  # NBA has stats endpoint
            "box_scores": "/nba/v1/box_scores",
            "standings": "/nba/v1/standings",
            "injuries": "/nba/v1/player_injuries",
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "odds_uses_dates": True,  # Uses dates[] parameter
        "standings_requires_season": True,
    },
    "NFL": {
        "code": "NFL",
        "name": "National Football League",
        "is_tennis": False,
        "endpoints": {
            "teams": "/nfl/v1/teams",
            "players": "/nfl/v1/players",
            "games": "/nfl/v1/games",
            "stats": "/nfl/v1/stats",  # NFL has stats endpoint
            "box_scores": "/nfl/v1/box_scores",
            "standings": "/nfl/v1/standings",
            "injuries": "/nfl/v1/player_injuries",
            "odds": "/nfl/v1/odds",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "odds_uses_dates": False,  # NFL uses season+week, NOT dates[]!
        "odds_requires_week": True,
        "standings_requires_season": True,
    },
    "MLB": {
        "code": "MLB",
        "name": "Major League Baseball",
        "is_tennis": False,
        "endpoints": {
            "teams": "/mlb/v1/teams",
            "players": "/mlb/v1/players",
            "games": "/mlb/v1/games",
            "stats": "/mlb/v1/stats",  # MLB has stats endpoint
            "box_scores": "/mlb/v1/box_scores",
            "standings": "/mlb/v1/standings",
            "injuries": "/mlb/v1/player_injuries",
            "odds": "/mlb/v1/odds",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "odds_uses_dates": True,
        "standings_requires_season": True,
    },
    "NHL": {
        "code": "NHL",
        "name": "National Hockey League",
        "is_tennis": False,
        "endpoints": {
            "teams": "/nhl/v1/teams",
            "players": "/nhl/v1/players",
            "games": "/nhl/v1/games",
            "stats": None,  # NHL does NOT have /stats endpoint!
            "box_scores": "/nhl/v1/box_scores",  # Use this for player stats
            "standings": "/nhl/v1/standings",
            "injuries": "/nhl/v1/player_injuries",
            "odds": "/nhl/v1/odds",  # v1 not v2!
        },
        "season_start": 2015,
        "has_stats_endpoint": False,  # NHL uses box_scores instead
        "odds_uses_dates": True,
        "standings_requires_season": False,  # NHL standings works without season
    },
    "WNBA": {
        "code": "WNBA",
        "name": "Women's National Basketball Association",
        "is_tennis": False,
        "endpoints": {
            "teams": "/wnba/v1/teams",
            "players": "/wnba/v1/players",
            "games": "/wnba/v1/games",
            "stats": "/wnba/v1/stats",
            "box_scores": "/wnba/v1/box_scores",
            "standings": "/wnba/v1/standings",
            "injuries": "/wnba/v1/player_injuries",
            "odds": "/wnba/v1/odds",
        },
        "season_start": 2018,
        "has_stats_endpoint": True,
        "odds_uses_dates": True,
        "standings_requires_season": True,
    },
    "NCAAF": {
        "code": "NCAAF",
        "name": "NCAA Football",
        "is_tennis": False,
        "endpoints": {
            "teams": "/ncaaf/v1/teams",
            "players": "/ncaaf/v1/players",
            "games": "/ncaaf/v1/games",
            "stats": "/ncaaf/v1/stats",
            "box_scores": "/ncaaf/v1/box_scores",
            "standings": "/ncaaf/v1/standings",
            "injuries": "/ncaaf/v1/player_injuries",
            "odds": "/ncaaf/v1/odds",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "odds_uses_dates": False,  # NCAAF uses season+week like NFL
        "odds_requires_week": True,
        "standings_requires_season": True,
    },
    "NCAAB": {
        "code": "NCAAB",
        "name": "NCAA Basketball",
        "is_tennis": False,
        "endpoints": {
            "teams": "/ncaab/v1/teams",
            "players": "/ncaab/v1/players",
            "games": "/ncaab/v1/games",
            "stats": "/ncaab/v1/stats",
            "box_scores": "/ncaab/v1/box_scores",
            "standings": "/ncaab/v1/standings",
            "injuries": "/ncaab/v1/player_injuries",
            "odds": "/ncaab/v1/odds",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "odds_uses_dates": True,
        "standings_requires_season": True,
    },
    "ATP": {
        "code": "ATP",
        "name": "ATP Tennis (Men's)",
        "is_tennis": True,
        "endpoints": {
            "players": "/atp/v1/players",
            "matches": "/atp/v1/matches",
            "tournaments": "/atp/v1/tournaments",
            "rankings": "/atp/v1/rankings",
            "odds": "/atp/v1/odds",
        },
        "season_start": 2017,
        "has_stats_endpoint": False,
        "odds_uses_dates": True,
        "standings_requires_season": False,
    },
    "WTA": {
        "code": "WTA",
        "name": "WTA Tennis (Women's)",
        "is_tennis": True,
        "endpoints": {
            "players": "/wta/v1/players",
            "matches": "/wta/v1/matches",
            "tournaments": "/wta/v1/tournaments",
            "rankings": "/wta/v1/rankings",
            "odds": "/wta/v1/odds",
        },
        "season_start": 2017,
        "has_stats_endpoint": False,
        "odds_uses_dates": True,
        "standings_requires_season": False,
    },
}

SUPPORTED_SPORTS = list(SPORT_CONFIG.keys())

# Stat field mappings per sport (for stats endpoint)
STAT_FIELDS = {
    "NBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min", "fgm", "fga", 
            "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "pf"],
    "NFL": ["passing_yards", "passing_touchdowns", "rushing_yards", "rushing_touchdowns", 
            "receiving_yards", "receiving_touchdowns", "receptions", "passing_interceptions", 
            "fumbles", "total_tackles", "sacks"],
    "MLB": ["hits", "at_bats", "runs", "rbi", "hr", "bb", "k", "avg", "era"],
    "NHL": ["goals", "assists", "points", "plus_minus", "penalty_minutes", "shots_on_goal", 
            "hits", "blocked_shots", "takeaways", "giveaways", "time_on_ice"],
    "WNBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min"],
    "NCAAF": ["passing_yards", "rushing_yards", "receiving_yards", "passing_touchdowns",
              "rushing_touchdowns", "receiving_touchdowns"],
    "NCAAB": ["pts", "reb", "ast", "stl", "blk", "min"],
    "ATP": [],
    "WTA": [],
}


# =============================================================================
# BALLDONTLIE COLLECTOR V2
# =============================================================================

class BallDontLieCollectorV2(BaseCollector):
    """
    BallDontLie API Collector V2 - Fixed version.
    
    Properly handles:
    - All 9 sports including tennis
    - Sport-specific endpoint paths
    - NHL box_scores (not stats)
    - v1 odds endpoints (not v2)
    - Injuries with nested teams array
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="balldontlie",
            base_url=BASE_URL,
            rate_limit=100,
            rate_window=60,
            timeout=60.0,
            max_retries=3,
        )
        self.api_key = api_key or API_KEY
        self.client: Optional[httpx.AsyncClient] = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        if self.api_key:
            logger.info(f"[BallDontLie] API Key configured: {self.api_key[:8]}...")
            console.print(f"[green][BallDontLie] API Key configured: {self.api_key[:8]}...[/green]")
        else:
            logger.warning("[BallDontLie] No API key configured!")
            console.print("[red][BallDontLie] No API key configured![/red]")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers={
                    "Authorization": self.api_key,
                    "Accept": "application/json",
                },
                timeout=60.0,
            )
        return self.client
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if isinstance(data, list):
            return len(data) > 0
        if isinstance(data, dict):
            return bool(data)
        return True
    
    async def _api_request(
        self, 
        endpoint: str, 
        params: Dict = None,
        sport_code: str = ""
    ) -> Optional[Dict]:
        """Make API request with error handling."""
        client = await self._get_client()
        
        try:
            logger.info(f"[BallDontLie] 🌐 {sport_code}: {BASE_URL}{endpoint}")
            console.print(f"[cyan][BallDontLie] 🌐 {sport_code}: {endpoint}[/cyan]")
            
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Log result count
            if "data" in data:
                count = len(data["data"]) if isinstance(data["data"], list) else 1
                logger.info(f"[BallDontLie] ✅ {sport_code}: {count} items")
                console.print(f"[green][BallDontLie] ✅ {count} items[/green]")
            
            return data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"[BallDontLie] Rate limited, waiting 60s...")
                console.print("[yellow][BallDontLie] Rate limited, waiting 60s...[/yellow]")
                await asyncio.sleep(60)
                return await self._api_request(endpoint, params, sport_code)
            else:
                logger.error(f"[BallDontLie] HTTP Error {e.response.status_code}: {e}")
                return None
        except Exception as e:
            logger.error(f"[BallDontLie] Request error: {e}")
            return None
    
    async def _paginated_request(
        self, 
        endpoint: str, 
        sport_code: str,
        params: Dict = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """Make paginated API request."""
        all_data = []
        params = params or {}
        params["per_page"] = 100
        cursor = None
        page = 0
        
        while page < max_pages:
            if cursor:
                params["cursor"] = cursor
            
            await asyncio.sleep(self.rate_limit_delay)
            response = await self._api_request(endpoint, params, sport_code)
            
            if not response or "data" not in response:
                break
            
            data = response["data"]
            if not data:
                break
            
            all_data.extend(data)
            
            # Check for next page
            meta = response.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            
            page += 1
        
        return all_data

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def collect_teams(self, sport_code: str) -> List[Dict]:
        """Collect teams for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("teams")
        if not endpoint:
            return []
        
        console.print(f"[bold blue]🏀 Collecting {sport_code} teams...[/bold blue]")
        return await self._paginated_request(endpoint, sport_code, max_pages=5)
    
    async def save_teams(
        self, 
        teams: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save teams to database."""
        if not teams:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        # Get or create sport
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                id=uuid4(),
                code=sport_code,
                name=SPORT_CONFIG[sport_code]["name"],
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        saved = 0
        updated = 0
        skipped = 0
        
        for team_data in teams:
            try:
                team_id = team_data.get("id")
                if not team_id:
                    skipped += 1
                    continue
                
                external_id = f"bdl_{sport_code}_{team_id}"
                
                # Check if exists
                result = await session.execute(
                    select(Team).where(Team.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get team name fields (API varies by sport)
                full_name = team_data.get("full_name") or team_data.get("name") or ""
                name = team_data.get("name") or team_data.get("short_name") or full_name
                abbr = team_data.get("abbreviation") or team_data.get("tricode") or name[:3].upper()
                city = team_data.get("city") or team_data.get("location") or ""
                conference = team_data.get("conference") or team_data.get("conference_name") or ""
                division = team_data.get("division") or team_data.get("division_name") or ""
                
                if existing:
                    existing.name = name
                    existing.full_name = full_name
                    existing.abbreviation = abbr
                    existing.city = city
                    existing.conference = conference
                    existing.division = division
                    updated += 1
                else:
                    team = Team(
                        id=uuid4(),
                        external_id=external_id,
                        sport_id=sport.id,
                        name=name,
                        full_name=full_name,
                        abbreviation=abbr,
                        city=city,
                        conference=conference,
                        division=division,
                        is_active=True,
                    )
                    session.add(team)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving team: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Teams: {saved} saved, {updated} updated, {skipped} skipped[/green]")
        return {"saved": saved, "updated": updated, "skipped": skipped}

    # =========================================================================
    # PLAYERS COLLECTION
    # =========================================================================
    
    async def collect_players(self, sport_code: str) -> List[Dict]:
        """Collect players for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("players")
        if not endpoint:
            return []
        
        console.print(f"[bold blue]👤 Collecting {sport_code} players...[/bold blue]")
        return await self._paginated_request(endpoint, sport_code, max_pages=200)
    
    async def save_players(
        self, 
        players: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save players to database."""
        if not players:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        saved = 0
        updated = 0
        skipped = 0
        
        for player_data in players:
            try:
                player_id = player_data.get("id")
                if not player_id:
                    skipped += 1
                    continue
                
                external_id = f"bdl_{sport_code}_{player_id}"
                
                # Check if exists
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get team if available - handle both simple object and array format
                team_id = None
                team_data = player_data.get("team")
                teams_array = player_data.get("teams", [])
                
                if team_data and team_data.get("id"):
                    # Simple team object
                    team_ext_id = f"bdl_{sport_code}_{team_data['id']}"
                    result = await session.execute(
                        select(Team).where(Team.external_id == team_ext_id)
                    )
                    team = result.scalar_one_or_none()
                    if team:
                        team_id = team.id
                elif teams_array and len(teams_array) > 0:
                    # Array of teams (NHL format) - use most recent
                    latest_team = teams_array[0]
                    if latest_team.get("id"):
                        team_ext_id = f"bdl_{sport_code}_{latest_team['id']}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == team_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            team_id = team.id
                
                # Player fields
                first_name = player_data.get("first_name", "")
                last_name = player_data.get("last_name", "")
                full_name = player_data.get("full_name") or f"{first_name} {last_name}".strip()
                position = player_data.get("position") or player_data.get("position_code") or ""
                
                # Height/weight handling
                # Height is stored as VARCHAR
                height = player_data.get("height") or player_data.get("height_in_inches") or ""
                if isinstance(height, int):
                    height = str(height)
                
                # Weight is stored as INTEGER in database
                weight_raw = player_data.get("weight") or player_data.get("weight_in_pounds")
                weight = None
                if weight_raw is not None:
                    try:
                        weight = int(weight_raw)
                    except (ValueError, TypeError):
                        weight = None
                
                jersey = player_data.get("jersey_number") or player_data.get("sweater_number") or ""
                if isinstance(jersey, int):
                    jersey = jersey
                else:
                    try:
                        jersey = int(jersey) if jersey else None
                    except:
                        jersey = None
                
                if existing:
                    existing.team_id = team_id
                    existing.name = full_name
                    existing.position = position
                    existing.height = height
                    existing.weight = weight
                    if jersey:
                        existing.jersey_number = jersey
                    updated += 1
                else:
                    player = Player(
                        id=uuid4(),
                        external_id=external_id,
                        team_id=team_id,
                        name=full_name,
                        position=position,
                        height=height,
                        weight=weight,
                        jersey_number=jersey,
                        is_active=True,
                    )
                    session.add(player)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving player: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Players: {saved} saved, {updated} updated, {skipped} skipped[/green]")
        return {"saved": saved, "updated": updated, "skipped": skipped}

    # =========================================================================
    # TENNIS PSEUDO-TEAMS (for ATP/WTA)
    # =========================================================================
    
    async def create_tennis_pseudo_teams(
        self, 
        players: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Create pseudo-teams for tennis players."""
        if sport_code not in ["ATP", "WTA"]:
            return {"saved": 0}
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                id=uuid4(),
                code=sport_code,
                name=SPORT_CONFIG[sport_code]["name"],
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        saved = 0
        
        for player_data in players:
            player_id = player_data.get("id")
            if not player_id:
                continue
            
            external_id = f"bdl_{sport_code}_player_{player_id}"
            
            # Check if pseudo-team exists
            result = await session.execute(
                select(Team).where(Team.external_id == external_id)
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                first_name = player_data.get("first_name", "")
                last_name = player_data.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                country = player_data.get("country") or player_data.get("citizenship") or ""
                
                team = Team(
                    id=uuid4(),
                    external_id=external_id,
                    sport_id=sport.id,
                    name=full_name,
                    full_name=full_name,
                    abbreviation=last_name[:3].upper() if last_name else "TEN",
                    city=country,  # Store country as city for tennis
                    is_active=True,
                )
                session.add(team)
                saved += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Pseudo-Teams: {saved} created[/green]")
        return {"saved": saved}

    # =========================================================================
    # GAMES COLLECTION
    # =========================================================================
    
    async def collect_games(
        self, 
        sport_code: str, 
        season: int = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """Collect games for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        # Tennis uses 'matches' endpoint
        endpoint = config["endpoints"].get("games") or config["endpoints"].get("matches")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        
        console.print(f"[bold blue]🏟️ Collecting {sport_code} games ({season or 'current'})...[/bold blue]")
        games = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(games)} games collected[/green]")
        return games
    
    async def save_games(
        self, 
        games: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save games to database."""
        if not games:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        config = SPORT_CONFIG.get(sport_code)
        is_tennis = config.get("is_tennis", False)
        
        # Get sport
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        saved = 0
        updated = 0
        skipped = 0
        
        for game_data in games:
            try:
                game_id = game_data.get("id")
                if not game_id:
                    skipped += 1
                    continue
                
                external_id = f"bdl_{sport_code}_{game_id}"
                
                # Check if exists
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get teams
                if is_tennis:
                    # Tennis: player_1 and player_2
                    player1 = game_data.get("player_1") or game_data.get("home_player") or {}
                    player2 = game_data.get("player_2") or game_data.get("away_player") or {}
                    
                    home_ext_id = f"bdl_{sport_code}_player_{player1.get('id')}" if player1.get('id') else None
                    away_ext_id = f"bdl_{sport_code}_player_{player2.get('id')}" if player2.get('id') else None
                else:
                    # Team sports
                    home_team = game_data.get("home_team") or {}
                    away_team = game_data.get("visitor_team") or game_data.get("away_team") or {}
                    
                    home_ext_id = f"bdl_{sport_code}_{home_team.get('id')}" if home_team.get('id') else None
                    away_ext_id = f"bdl_{sport_code}_{away_team.get('id')}" if away_team.get('id') else None
                
                # Look up teams
                home_team_id = None
                away_team_id = None
                
                if home_ext_id:
                    result = await session.execute(
                        select(Team).where(Team.external_id == home_ext_id)
                    )
                    home = result.scalar_one_or_none()
                    if home:
                        home_team_id = home.id
                
                if away_ext_id:
                    result = await session.execute(
                        select(Team).where(Team.external_id == away_ext_id)
                    )
                    away = result.scalar_one_or_none()
                    if away:
                        away_team_id = away.id
                
                # Skip if no teams
                if not home_team_id or not away_team_id:
                    skipped += 1
                    continue
                
                # Parse date
                date_str = game_data.get("date") or game_data.get("datetime") or game_data.get("game_date") or game_data.get("start_time_utc")
                scheduled_at = None
                if date_str:
                    try:
                        if "T" in str(date_str):
                            scheduled_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        else:
                            scheduled_at = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
                    except:
                        pass
                
                # Scores
                home_score = game_data.get("home_team_score") or game_data.get("home_score") or 0
                away_score = game_data.get("visitor_team_score") or game_data.get("away_score") or 0
                
                # Status
                status_str = game_data.get("status") or game_data.get("game_state") or ""
                if "final" in str(status_str).lower() or status_str == "OFF":
                    status = GameStatus.FINAL
                elif "progress" in str(status_str).lower() or status_str == "LIVE":
                    status = GameStatus.IN_PROGRESS
                else:
                    status = GameStatus.SCHEDULED
                
                season = game_data.get("season")
                
                if existing:
                    existing.home_score = home_score
                    existing.away_score = away_score
                    existing.status = status
                    if scheduled_at:
                        existing.scheduled_at = scheduled_at
                    updated += 1
                else:
                    game = Game(
                        id=uuid4(),
                        external_id=external_id,
                        sport_id=sport.id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        scheduled_at=scheduled_at or datetime.utcnow(),
                        status=status,
                        home_score=home_score,
                        away_score=away_score,
                        # Note: season_id requires lookup - skip for now
                    )
                    session.add(game)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving game: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Games: {saved} saved, {updated} updated, {skipped} skipped[/green]")
        return {"saved": saved, "updated": updated, "skipped": skipped}

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def collect_player_stats(
        self, 
        sport_code: str, 
        season: int = None,
        game_ids: List[int] = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """Collect player stats for a sport using /stats endpoint."""
        config = SPORT_CONFIG.get(sport_code)
        if not config or not config.get("has_stats_endpoint"):
            console.print(f"[yellow]⚠️ {sport_code} does not have /stats endpoint, skipping...[/yellow]")
            return []
        
        endpoint = config["endpoints"].get("stats")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        if game_ids:
            for gid in game_ids[:10]:  # Limit to 10 game IDs
                params["game_ids[]"] = gid
        
        console.print(f"[bold blue]📊 Collecting {sport_code} player stats ({season or 'current'})...[/bold blue]")
        stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(stats)} stat records collected[/green]")
        return stats

    async def collect_box_scores(
        self, 
        sport_code: str, 
        season: int = None,
        dates: List[str] = None,
        game_ids: List[int] = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """Collect box scores (used for NHL and as alternative for other sports)."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("box_scores")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        if dates:
            for d in dates[:10]:
                params["dates[]"] = d
        if game_ids:
            for gid in game_ids[:10]:
                params["game_ids[]"] = gid
        
        console.print(f"[bold blue]📊 Collecting {sport_code} box scores ({season or 'current'})...[/bold blue]")
        box_scores = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(box_scores)} box score records collected[/green]")
        return box_scores
    
    async def save_player_stats(
        self, 
        stats: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save player stats to database."""
        if not stats:
            return {"saved": 0, "skipped": 0}
        
        stat_fields = STAT_FIELDS.get(sport_code, ["pts", "reb", "ast"])
        
        saved = 0
        skipped = 0
        
        for stat_data in stats:
            try:
                # Get player - handle both 'player' object and player_id
                player_api_id = None
                if stat_data.get("player"):
                    player_api_id = stat_data["player"].get("id")
                elif stat_data.get("player_id"):
                    player_api_id = stat_data["player_id"]
                
                if not player_api_id:
                    skipped += 1
                    continue
                
                player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                result = await session.execute(
                    select(Player).where(Player.external_id == player_ext_id)
                )
                player = result.scalar_one_or_none()
                
                if not player:
                    skipped += 1
                    continue
                
                # Get game if available
                game_id = None
                game_api_id = None
                if stat_data.get("game"):
                    game_api_id = stat_data["game"].get("id")
                elif stat_data.get("game_id"):
                    game_api_id = stat_data["game_id"]
                
                if game_api_id:
                    game_ext_id = f"bdl_{sport_code}_{game_api_id}"
                    result = await session.execute(
                        select(Game).where(Game.external_id == game_ext_id)
                    )
                    game = result.scalar_one_or_none()
                    if game:
                        game_id = game.id
                
                # Save each stat type
                for stat_type in stat_fields:
                    value = stat_data.get(stat_type)
                    if value is not None:
                        try:
                            # Handle string values like time_on_ice "15:30"
                            if isinstance(value, str) and ":" in value:
                                # Convert time string to minutes
                                parts = value.split(":")
                                value = float(parts[0]) + float(parts[1])/60
                            else:
                                value = float(value)
                        except:
                            continue
                        
                        stat = PlayerStats(
                            id=uuid4(),
                            player_id=player.id,
                            game_id=game_id,
                            stat_type=stat_type,
                            value=value,
                        )
                        session.add(stat)
                        saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving stat: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Player Stats: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # INJURIES COLLECTION - FIXED
    # =========================================================================
    
    async def collect_injuries(self, sport_code: str) -> List[Dict]:
        """Collect injuries for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config or config.get("is_tennis"):
            return []
        
        endpoint = config["endpoints"].get("injuries")
        if not endpoint:
            return []
        
        console.print(f"[bold blue]🏥 Collecting {sport_code} injuries...[/bold blue]")
        injuries = await self._paginated_request(endpoint, sport_code, max_pages=5)
        console.print(f"[green]✅ {sport_code}: {len(injuries)} injuries collected[/green]")
        return injuries
    
    async def save_injuries(
        self, 
        injuries: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save injuries to database - FIXED to handle nested teams array."""
        if not injuries:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        saved = 0
        updated = 0
        skipped = 0
        
        for injury_data in injuries:
            try:
                # Get player info
                player_info = injury_data.get("player") or {}
                player_api_id = player_info.get("id")
                
                if not player_api_id:
                    skipped += 1
                    continue
                
                # Create external_id for this injury
                external_id = f"bdl_{sport_code}_inj_{player_api_id}"
                
                # Check if already exists
                result = await session.execute(
                    select(Injury).where(Injury.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get team - HANDLE BOTH FORMATS:
                # 1. Simple team object: player_info.get("team")
                # 2. Array of teams: player_info.get("teams") - NHL format
                team_id = None
                team_api_id = None
                
                # Try simple team object first
                team_obj = player_info.get("team") or injury_data.get("team")
                if team_obj and team_obj.get("id"):
                    team_api_id = team_obj["id"]
                
                # Try teams array (NHL format)
                if not team_api_id:
                    teams_array = player_info.get("teams", [])
                    if teams_array and len(teams_array) > 0:
                        # Get most recent team (usually first in array)
                        team_api_id = teams_array[0].get("id")
                
                if team_api_id:
                    team_ext_id = f"bdl_{sport_code}_{team_api_id}"
                    result = await session.execute(
                        select(Team).where(Team.external_id == team_ext_id)
                    )
                    team = result.scalar_one_or_none()
                    if team:
                        team_id = team.id
                
                # If still no team, try to find from player record
                if not team_id:
                    player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                    result = await session.execute(
                        select(Player).where(Player.external_id == player_ext_id)
                    )
                    player = result.scalar_one_or_none()
                    if player and player.team_id:
                        team_id = player.team_id
                
                if not team_id:
                    logger.warning(f"[BallDontLie] No team found for injury: {player_info.get('full_name', 'Unknown')}")
                    skipped += 1
                    continue
                
                # Get player_id (optional link)
                player_id = None
                player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                result = await session.execute(
                    select(Player).where(Player.external_id == player_ext_id)
                )
                player = result.scalar_one_or_none()
                if player:
                    player_id = player.id
                
                # Player name
                first_name = player_info.get("first_name", "")
                last_name = player_info.get("last_name", "")
                full_name = player_info.get("full_name") or f"{first_name} {last_name}".strip() or "Unknown"
                
                # Injury details
                status = injury_data.get("status") or injury_data.get("status_abbreviation") or "Unknown"
                injury_type = injury_data.get("injury_type") or injury_data.get("description") or injury_data.get("comment") or ""
                comment = injury_data.get("comment") or ""
                return_date = injury_data.get("return_date")
                
                if existing:
                    # Update existing injury
                    existing.status = status
                    existing.injury_type = str(injury_type)[:200] if injury_type else None
                    existing.player_name = full_name
                    updated += 1
                else:
                    injury = Injury(
                        id=uuid4(),
                        player_id=player_id,
                        team_id=team_id,
                        sport_code=sport_code,
                        player_name=full_name,
                        position=player_info.get("position") or player_info.get("position_code"),
                        injury_type=str(injury_type)[:200] if injury_type else None,
                        status=status,
                        source="balldontlie",
                        external_id=external_id,
                    )
                    session.add(injury)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving injury: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Injuries: {saved} saved, {updated} updated, {skipped} skipped[/green]")
        return {"saved": saved, "updated": updated, "skipped": skipped}

    # =========================================================================
    # STANDINGS COLLECTION
    # =========================================================================
    
    async def collect_standings(self, sport_code: str, season: int = None) -> List[Dict]:
        """Collect standings for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config or config.get("is_tennis"):
            return []
        
        endpoint = config["endpoints"].get("standings")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]📈 Collecting {sport_code} standings...[/bold blue]")
        standings = await self._paginated_request(endpoint, sport_code, params, max_pages=10)
        console.print(f"[green]✅ {sport_code}: {len(standings)} standings records collected[/green]")
        return standings
    
    async def save_team_stats(
        self, 
        standings: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save standings as team stats."""
        if not standings:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for standing in standings:
            try:
                team_data = standing.get("team") or {}
                team_api_id = team_data.get("id")
                
                if not team_api_id:
                    skipped += 1
                    continue
                
                team_ext_id = f"bdl_{sport_code}_{team_api_id}"
                result = await session.execute(
                    select(Team).where(Team.external_id == team_ext_id)
                )
                team = result.scalar_one_or_none()
                
                if not team:
                    skipped += 1
                    continue
                
                season = standing.get("season")
                
                # Save various stat types from standings
                stat_types = [
                    ("wins", standing.get("wins")),
                    ("losses", standing.get("losses")),
                    ("ot_losses", standing.get("ot_losses")),
                    ("win_pct", standing.get("win_percentage") or standing.get("points_pctg")),
                    ("points", standing.get("points")),
                    ("goals_for", standing.get("goals_for")),
                    ("goals_against", standing.get("goals_against")),
                    ("goal_differential", standing.get("goal_differential")),
                    ("points_for", standing.get("points_for")),
                    ("points_against", standing.get("points_against")),
                    ("games_played", standing.get("games_played")),
                ]
                
                for stat_type, value in stat_types:
                    if value is not None:
                        try:
                            value = float(value)
                        except:
                            continue
                        
                        stat = TeamStats(
                            id=uuid4(),
                            team_id=team.id,
                            stat_type=stat_type,
                            value=value,
                            games_played=standing.get("games_played") or 0,
                            # Note: season_id requires lookup - skip for now
                        )
                        session.add(stat)
                        saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving team stat: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Team Stats: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # ODDS COLLECTION - FIXED
    # =========================================================================
    
    async def collect_odds(
        self, 
        sport_code: str, 
        dates: List[str] = None,
        game_ids: List[int] = None,
        season: int = None,
        week: int = None,
        max_pages: int = 10
    ) -> List[Dict]:
        """Collect odds for a sport.
        
        Different sports have different parameter requirements:
        - NFL/NCAAF: requires (season AND week) OR game_ids[]
        - Other sports: requires dates[] OR game_ids[]
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("odds")
        if not endpoint:
            return []
        
        params = {}
        
        # Check if this sport requires season+week (NFL, NCAAF)
        requires_week = config.get("odds_requires_week", False)
        uses_dates = config.get("odds_uses_dates", True)
        
        if requires_week:
            # NFL/NCAAF: use season + week
            if game_ids:
                for gid in game_ids[:10]:
                    params["game_ids[]"] = gid
            else:
                # Default to current season and recent weeks
                current_year = datetime.now().year
                s = season or current_year
                # Try multiple weeks (current and recent)
                weeks_to_try = [week] if week else list(range(1, 19))  # NFL has 18 weeks
                
                all_odds = []
                for w in weeks_to_try[-4:]:  # Last 4 weeks
                    params = {"season": s, "week": w}
                    console.print(f"[bold blue]💰 Collecting {sport_code} odds (season={s}, week={w})...[/bold blue]")
                    try:
                        odds = await self._paginated_request(endpoint, sport_code, params, max_pages=2)
                        all_odds.extend(odds)
                        if odds:
                            console.print(f"[green]✅ {sport_code} week {w}: {len(odds)} odds[/green]")
                    except Exception as e:
                        logger.warning(f"[BallDontLie] No odds for {sport_code} week {w}: {e}")
                
                console.print(f"[green]✅ {sport_code}: {len(all_odds)} total odds records collected[/green]")
                return all_odds
        else:
            # Other sports: use dates[] or game_ids[]
            if game_ids:
                for gid in game_ids[:10]:
                    params["game_ids[]"] = gid
            elif dates:
                for d in dates[:10]:
                    params["dates[]"] = d
            else:
                # Default to recent dates
                today = datetime.now()
                recent_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                params["dates[]"] = recent_date
        
        console.print(f"[bold blue]💰 Collecting {sport_code} odds...[/bold blue]")
        odds = await self._paginated_request(endpoint, sport_code, params, max_pages=max_pages)
        console.print(f"[green]✅ {sport_code}: {len(odds)} odds records collected[/green]")
        return odds
    
    async def save_odds(
        self, 
        odds_data: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save odds to database - creates separate records per bet_type."""
        if not odds_data:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for odd in odds_data:
            try:
                game_api_id = odd.get("game_id")
                if not game_api_id:
                    skipped += 1
                    continue
                
                game_ext_id = f"bdl_{sport_code}_{game_api_id}"
                result = await session.execute(
                    select(Game).where(Game.external_id == game_ext_id)
                )
                game = result.scalar_one_or_none()
                
                if not game:
                    skipped += 1
                    continue
                
                # Get or create sportsbook
                vendor = odd.get("vendor", "unknown")
                vendor_key = vendor.lower().replace(" ", "_").replace("-", "_")
                result = await session.execute(
                    select(Sportsbook).where(Sportsbook.key == vendor_key)
                )
                sportsbook = result.scalar_one_or_none()
                
                if not sportsbook:
                    sportsbook = Sportsbook(
                        id=uuid4(),
                        name=vendor,
                        key=vendor_key,
                        is_active=True,
                    )
                    session.add(sportsbook)
                    await session.flush()
                
                # Parse values with proper type conversion
                def safe_float(val):
                    if val is None:
                        return None
                    try:
                        return float(val)
                    except:
                        return None
                
                def safe_int(val):
                    if val is None:
                        return None
                    try:
                        return int(float(val))
                    except:
                        return None
                
                # 1. Create SPREAD odds record if spread data exists
                spread_home = safe_float(odd.get("spread_home_value"))
                spread_away = safe_float(odd.get("spread_away_value"))
                spread_home_odds = safe_int(odd.get("spread_home_odds"))
                spread_away_odds = safe_int(odd.get("spread_away_odds"))
                
                if spread_home is not None or spread_away is not None:
                    spread_record = Odds(
                        id=uuid4(),
                        game_id=game.id,
                        sportsbook_id=sportsbook.id,
                        sportsbook_key=vendor_key,
                        bet_type="spread",
                        home_line=spread_home,
                        away_line=spread_away,
                        home_odds=spread_home_odds,
                        away_odds=spread_away_odds,
                    )
                    session.add(spread_record)
                    saved += 1
                
                # 2. Create TOTAL odds record if total data exists
                total_line = safe_float(odd.get("total_value"))
                over_odds = safe_int(odd.get("total_over_odds"))
                under_odds = safe_int(odd.get("total_under_odds"))
                
                if total_line is not None:
                    total_record = Odds(
                        id=uuid4(),
                        game_id=game.id,
                        sportsbook_id=sportsbook.id,
                        sportsbook_key=vendor_key,
                        bet_type="total",
                        total=total_line,
                        over_odds=over_odds,
                        under_odds=under_odds,
                    )
                    session.add(total_record)
                    saved += 1
                
                # 3. Create MONEYLINE odds record if moneyline data exists
                ml_home = safe_int(odd.get("moneyline_home_odds"))
                ml_away = safe_int(odd.get("moneyline_away_odds"))
                
                if ml_home is not None or ml_away is not None:
                    ml_record = Odds(
                        id=uuid4(),
                        game_id=game.id,
                        sportsbook_id=sportsbook.id,
                        sportsbook_key=vendor_key,
                        bet_type="moneyline",
                        home_odds=ml_home,
                        away_odds=ml_away,
                    )
                    session.add(ml_record)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving odds: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Odds: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================
    
    async def collect_sport(
        self, 
        sport_code: str, 
        years: int = 10
    ) -> Dict[str, Any]:
        """Collect all data for a single sport."""
        from app.core.database import db_manager
        
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            console.print(f"[red]❌ Unknown sport: {sport_code}[/red]")
            return {}
        
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]COLLECTING {sport_code} DATA ({years} years)[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        
        is_tennis = config.get("is_tennis", False)
        has_stats = config.get("has_stats_endpoint", True)
        current_year = datetime.now().year
        season_start = config.get("season_start", 2015)
        start_year = max(season_start, current_year - years)
        
        results = {
            "sport": sport_code,
            "teams": {"saved": 0, "updated": 0},
            "players": {"saved": 0, "updated": 0},
            "games": {"saved": 0, "updated": 0},
            "player_stats": {"saved": 0},
            "injuries": {"saved": 0, "updated": 0},
            "team_stats": {"saved": 0},
            "odds": {"saved": 0},
        }
        
        try:
            async with db_manager.session() as session:
                # 1. Teams (or pseudo-teams for tennis)
                if not is_tennis:
                    console.print(f"\n[bold]📊 Step 1: Collecting {sport_code} teams...[/bold]")
                    teams = await self.collect_teams(sport_code)
                    team_results = await self.save_teams(teams, sport_code, session)
                    results["teams"] = team_results
                
                # 2. Players
                console.print(f"\n[bold]📊 Step 2: Collecting {sport_code} players...[/bold]")
                players = await self.collect_players(sport_code)
                
                if is_tennis:
                    # Create pseudo-teams from players
                    await self.create_tennis_pseudo_teams(players, sport_code, session)
                
                player_results = await self.save_players(players, sport_code, session)
                results["players"] = player_results
                
                # 3. Games (historical)
                console.print(f"\n[bold]📊 Step 3: Collecting {sport_code} games...[/bold]")
                for year in range(start_year, current_year + 1):
                    games = await self.collect_games(sport_code, season=year)
                    game_results = await self.save_games(games, sport_code, session)
                    results["games"]["saved"] += game_results.get("saved", 0)
                    results["games"]["updated"] += game_results.get("updated", 0)
                
                # 4. Player Stats
                console.print(f"\n[bold]📊 Step 4: Collecting {sport_code} player stats...[/bold]")
                if has_stats and not is_tennis:
                    # Use /stats endpoint
                    for year in range(start_year, current_year + 1):
                        stats = await self.collect_player_stats(sport_code, season=year)
                        stat_results = await self.save_player_stats(stats, sport_code, session)
                        results["player_stats"]["saved"] += stat_results.get("saved", 0)
                elif not is_tennis:
                    # Use box_scores (for NHL and fallback)
                    # Get recent game dates for box scores
                    today = datetime.now()
                    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
                    box_scores = await self.collect_box_scores(sport_code, dates=dates)
                    stat_results = await self.save_player_stats(box_scores, sport_code, session)
                    results["player_stats"]["saved"] += stat_results.get("saved", 0)
                
                # 5. Injuries
                if not is_tennis:
                    console.print(f"\n[bold]📊 Step 5: Collecting {sport_code} injuries...[/bold]")
                    injuries = await self.collect_injuries(sport_code)
                    injury_results = await self.save_injuries(injuries, sport_code, session)
                    results["injuries"] = injury_results
                
                # 6. Standings/Team Stats
                if not is_tennis:
                    console.print(f"\n[bold]📊 Step 6: Collecting {sport_code} standings...[/bold]")
                    # Pass current season - some APIs require it (NBA)
                    standings = await self.collect_standings(sport_code, season=current_year)
                    team_stat_results = await self.save_team_stats(standings, sport_code, session)
                    results["team_stats"] = team_stat_results
                
                # 7. Odds
                console.print(f"\n[bold]📊 Step 7: Collecting {sport_code} odds...[/bold]")
                # Get recent dates for odds (for sports that use dates)
                today = datetime.now()
                dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                # Pass season for NFL/NCAAF which require it
                odds = await self.collect_odds(sport_code, dates=dates, season=current_year)
                odds_results = await self.save_odds(odds, sport_code, session)
                results["odds"] = odds_results
        
        except Exception as e:
            logger.error(f"[BallDontLie V2] Error collecting {sport_code}: {e}")
            console.print(f"[red]❌ Error: {e}[/red]")
        
        finally:
            await self.close()
        
        # Summary
        total_saved = (
            results["teams"].get("saved", 0) +
            results["players"].get("saved", 0) +
            results["games"].get("saved", 0) +
            results["player_stats"].get("saved", 0) +
            results["injuries"].get("saved", 0) +
            results["team_stats"].get("saved", 0) +
            results["odds"].get("saved", 0)
        )
        
        console.print(f"\n[bold green]✅ {sport_code} Collection Complete[/bold green]")
        console.print(f"  Teams: {results['teams'].get('saved', 0)} saved, {results['teams'].get('updated', 0)} updated")
        console.print(f"  Players: {results['players'].get('saved', 0)} saved, {results['players'].get('updated', 0)} updated")
        console.print(f"  Games: {results['games'].get('saved', 0)} saved, {results['games'].get('updated', 0)} updated")
        console.print(f"  Player Stats: {results['player_stats'].get('saved', 0)} saved")
        console.print(f"  Injuries: {results['injuries'].get('saved', 0)} saved, {results['injuries'].get('updated', 0)} updated")
        console.print(f"  Team Stats: {results['team_stats'].get('saved', 0)} saved")
        console.print(f"  Odds: {results['odds'].get('saved', 0)} saved")
        console.print(f"  [bold green]✓ {sport_code}: {total_saved:,} records saved[/bold green]")
        
        return results
    
    async def collect(self, **kwargs) -> CollectorResult:
        """Main collection entry point."""
        result = CollectorResult(source="balldontlie_v2")
        
        sports = kwargs.get("sports", SUPPORTED_SPORTS)
        years = kwargs.get("years", 10)
        
        for sport_code in sports:
            try:
                sport_results = await self.collect_sport(sport_code, years=years)
                
                total = sum([
                    sport_results.get("teams", {}).get("saved", 0),
                    sport_results.get("players", {}).get("saved", 0),
                    sport_results.get("games", {}).get("saved", 0),
                    sport_results.get("player_stats", {}).get("saved", 0),
                    sport_results.get("injuries", {}).get("saved", 0),
                    sport_results.get("team_stats", {}).get("saved", 0),
                    sport_results.get("odds", {}).get("saved", 0),
                ])
                
                result.records += total
                result.success = True
            
            except Exception as e:
                logger.error(f"[BallDontLie V2] Error collecting {sport_code}: {e}")
                result.errors.append(f"{sport_code}: {str(e)}")
        
        return result


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

async def collect_sport(sport_code: str, years: int = 10) -> Dict[str, int]:
    """Standalone function to collect a single sport."""
    collector = BallDontLieCollectorV2()
    return await collector.collect_sport(sport_code, years)

async def collect_all_sports(years: int = 10) -> Dict[str, Dict]:
    """Collect all supported sports."""
    collector = BallDontLieCollectorV2()
    results = {}
    
    for sport_code in SUPPORTED_SPORTS:
        results[sport_code] = await collector.collect_sport(sport_code, years)
    
    return results