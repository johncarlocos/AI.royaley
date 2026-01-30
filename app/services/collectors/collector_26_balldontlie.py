"""
ROYALEY - BallDontLie Data Collector
Phase 1: Data Collection Services

Comprehensive multi-sport data collector from BallDontLie API.
$299/month plan - Full access to all sports and endpoints.

Supported Sports (9):
- NBA: Basketball
- NFL: Football  
- MLB: Baseball
- NHL: Hockey
- WNBA: Women's Basketball
- NCAAF: College Football
- NCAAB: College Basketball
- ATP: Men's Tennis
- WTA: Women's Tennis

API Documentation:
- NBA: https://docs.balldontlie.io/#nba-api
- NFL: https://nfl.balldontlie.io/#nfl-api
- MLB: https://mlb.balldontlie.io/#mlb-api
- NHL: https://nhl.balldontlie.io/#nhl-api
- WNBA: https://wnba.balldontlie.io/#wnba-api
- NCAAF: https://ncaaf.balldontlie.io/#ncaaf-api
- NCAAB: https://ncaab.balldontlie.io/#ncaab-api
- ATP: https://atp.balldontlie.io/#atp-api
- WTA: https://wta.balldontlie.io/#wta-api

Data Collected:
- Teams: Full team information, conferences, divisions
- Players: Player details, positions, physical attributes
- Games: Schedules, scores, status
- Player Stats: Game-by-game and season statistics
- Team Stats: Standings, win/loss records
- Injuries: Player injury status and details
- Seasons: Season information

Tables Filled:
- sports
- teams
- players
- games
- player_stats
- team_stats
- injuries
- venues
- seasons
"""

import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import httpx
from rich.console import Console
from rich.progress import Progress, TaskID
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import (
    Sport, Team, Game, GameStatus, Player, PlayerStats, TeamStats, 
    Venue, Season, Injury
)
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# BALLDONTLIE API CONFIGURATION
# =============================================================================

# API Key (from settings or direct)
BALLDONTLIE_API_KEY = getattr(settings, 'BALLDONTLIE_API_KEY', None) or "a43e51d8-eba5-4ccd-b9dd-340d89d5e0c8"

# Base URLs for each sport
# API paths based on official documentation:
# NBA: https://docs.balldontlie.io/ - uses /v1/ prefix
# Other sports: use /{sport}/v1/ prefix (nfl, mlb, nhl, wnba, ncaaf, ncaab, atp, wta)
SPORT_CONFIG = {
    "NBA": {
        "base_url": "https://api.balldontlie.io",
        "code": "NBA",
        "name": "National Basketball Association",
        "endpoints": {
            "teams": "/v1/teams",
            "players": "/v1/players",
            "active_players": "/v1/players/active",
            "games": "/v1/games",
            "stats": "/v1/stats",
            "season_averages": "/v1/season_averages/general",
            "advanced_stats": "/v1/stats/advanced",
            "box_scores": "/v1/box_scores",
            "box_scores_live": "/v1/box_scores/live",
            "standings": "/v1/standings",
            "injuries": "/v1/player_injuries",
            "leaders": "/v1/leaders",
            "odds": "/v2/odds",
            "player_props": "/v2/odds/player_props",
        },
        "season_start_year": 2015,  # 10 years back - NBA data goes back to 1946
    },
    "NFL": {
        "base_url": "https://api.balldontlie.io",
        "code": "NFL",
        "name": "National Football League",
        "endpoints": {
            "teams": "/nfl/v1/teams",
            "players": "/nfl/v1/players",
            "games": "/nfl/v1/games",
            "stats": "/nfl/v1/stats",
            "season_stats": "/nfl/v1/season_stats",
            "advanced_rushing": "/nfl/v1/stats/advanced/rushing",
            "advanced_passing": "/nfl/v1/stats/advanced/passing",
            "advanced_receiving": "/nfl/v1/stats/advanced/receiving",
            "injuries": "/nfl/v1/player_injuries",
            "standings": "/nfl/v1/standings",
            "weeks": "/nfl/v1/weeks",
            "odds": "/nfl/v2/odds",
            "player_props": "/nfl/v2/odds/player_props",
        },
        "season_start_year": 2015,
    },
    "MLB": {
        "base_url": "https://api.balldontlie.io",
        "code": "MLB",
        "name": "Major League Baseball",
        "endpoints": {
            "teams": "/mlb/v1/teams",
            "players": "/mlb/v1/players",
            "games": "/mlb/v1/games",
            "stats": "/mlb/v1/stats",
            "season_stats": "/mlb/v1/season_stats",
            "team_season_stats": "/mlb/v1/team_season_stats",
            "injuries": "/mlb/v1/player_injuries",
            "standings": "/mlb/v1/standings",
            "odds": "/mlb/v2/odds",
            "player_props": "/mlb/v2/odds/player_props",
        },
        "season_start_year": 2015,
    },
    "NHL": {
        "base_url": "https://api.balldontlie.io",
        "code": "NHL",
        "name": "National Hockey League",
        "endpoints": {
            "teams": "/nhl/v1/teams",
            "players": "/nhl/v1/players",
            "games": "/nhl/v1/games",
            "stats": "/nhl/v1/stats",
            "season_stats": "/nhl/v1/season_stats",
            "team_stats_leaders": "/nhl/v1/team_stats/leaders",
            "injuries": "/nhl/v1/player_injuries",
            "standings": "/nhl/v1/standings",
            "odds": "/nhl/v2/odds",
            "player_props": "/nhl/v2/odds/player_props",
        },
        "season_start_year": 2015,
    },
    "WNBA": {
        "base_url": "https://api.balldontlie.io",
        "code": "WNBA",
        "name": "Women's National Basketball Association",
        "endpoints": {
            "teams": "/wnba/v1/teams",
            "players": "/wnba/v1/players",
            "active_players": "/wnba/v1/players/active",
            "games": "/wnba/v1/games",
            "stats": "/wnba/v1/stats",
            "season_averages": "/wnba/v1/season_averages/general",
            "injuries": "/wnba/v1/player_injuries",
            "standings": "/wnba/v1/standings",
            "odds": "/wnba/v2/odds",
            "player_props": "/wnba/v2/odds/player_props",
        },
        "season_start_year": 2018,
    },
    "NCAAF": {
        "base_url": "https://api.balldontlie.io",
        "code": "NCAAF",
        "name": "NCAA Football",
        "endpoints": {
            "teams": "/ncaaf/v1/teams",
            "players": "/ncaaf/v1/players",
            "games": "/ncaaf/v1/games",
            "stats": "/ncaaf/v1/stats",
            "season_stats": "/ncaaf/v1/season_stats",
            "injuries": "/ncaaf/v1/player_injuries",
            "standings": "/ncaaf/v1/standings",
            "odds": "/ncaaf/v2/odds",
            "player_props": "/ncaaf/v2/odds/player_props",
        },
        "season_start_year": 2015,
    },
    "NCAAB": {
        "base_url": "https://api.balldontlie.io",
        "code": "NCAAB",
        "name": "NCAA Basketball",
        "endpoints": {
            "teams": "/ncaab/v1/teams",
            "players": "/ncaab/v1/players",
            "active_players": "/ncaab/v1/players/active",
            "games": "/ncaab/v1/games",
            "stats": "/ncaab/v1/stats",
            "season_averages": "/ncaab/v1/season_averages/general",
            "injuries": "/ncaab/v1/player_injuries",
            "standings": "/ncaab/v1/standings",
            "odds": "/ncaab/v2/odds",
            "player_props": "/ncaab/v2/odds/player_props",
        },
        "season_start_year": 2015,
    },
    "ATP": {
        "base_url": "https://api.balldontlie.io",
        "code": "ATP",
        "name": "ATP Tennis (Men's)",
        "endpoints": {
            "players": "/atp/v1/players",
            "tournaments": "/atp/v1/tournaments",
            "matches": "/atp/v1/matches",
            "rankings": "/atp/v1/rankings",
            "head_to_head": "/atp/v1/head_to_head",
            "odds": "/atp/v2/odds",
        },
        "season_start_year": 2015,
    },
    "WTA": {
        "base_url": "https://api.balldontlie.io",
        "code": "WTA",
        "name": "WTA Tennis (Women's)",
        "endpoints": {
            "players": "/wta/v1/players",
            "tournaments": "/wta/v1/tournaments",
            "matches": "/wta/v1/matches",
            "rankings": "/wta/v1/rankings",
            "head_to_head": "/wta/v1/head_to_head",
            "odds": "/wta/v2/odds",
        },
        "season_start_year": 2015,
    },
}

# List of all supported sports
SUPPORTED_SPORTS = list(SPORT_CONFIG.keys())


# =============================================================================
# BALLDONTLIE COLLECTOR CLASS
# =============================================================================

class BallDontLieCollector(BaseCollector):
    """
    Comprehensive multi-sport data collector from BallDontLie API.
    
    Features:
    - All 9 sports supported (NBA, NFL, MLB, NHL, WNBA, NCAAF, NCAAB, ATP, WTA)
    - 10 years historical data
    - Teams, players, games, stats, injuries
    - Automatic pagination handling
    - Rate limiting and retry logic
    
    $299/month plan with full API access.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="balldontlie",
            base_url="https://api.balldontlie.io",
            rate_limit=300,  # Premium plan allows higher rate
            rate_window=60,
            timeout=60.0,
            max_retries=5,
        )
        self.api_key = api_key or BALLDONTLIE_API_KEY
        self._client: Optional[httpx.AsyncClient] = None
        
        if self.api_key:
            logger.info(f"[BallDontLie] API Key configured: {self.api_key[:8]}...")
            console.print(f"[green][BallDontLie] API Key configured: {self.api_key[:8]}...[/green]")
        else:
            logger.warning("[BallDontLie] No API key configured!")
            console.print("[yellow][BallDontLie] ⚠️ No API key configured![/yellow]")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key authentication."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "Royaley/1.0 (Sports Prediction Platform)",
        }
        if self.api_key:
            headers["Authorization"] = self.api_key
        return headers
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                headers=self._get_headers(),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _api_request(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        sport_code: str = "NBA",
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request with pagination support.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            sport_code: Sport code for logging
            
        Returns:
            API response data or None on error
        """
        client = await self.get_client()
        params = params or {}
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"[BallDontLie] 🌐 {sport_code}: {url}")
        console.print(f"[cyan][BallDontLie] 🌐 {sport_code}: {endpoint}[/cyan]")
        
        try:
            # Rate limiting
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self.rate_limiter.add_request()
            
            response = await client.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"[BallDontLie] Rate limited, waiting {retry_after}s")
                console.print(f"[yellow][BallDontLie] ⏳ Rate limited, waiting {retry_after}s[/yellow]")
                await asyncio.sleep(retry_after)
                return await self._api_request(endpoint, params, sport_code)
            
            if response.status_code == 401:
                logger.error("[BallDontLie] ❌ Invalid API key!")
                console.print("[red][BallDontLie] ❌ Invalid API key![/red]")
                return None
            
            if response.status_code == 404:
                logger.warning(f"[BallDontLie] Endpoint not found: {endpoint}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Log success
            if isinstance(data, dict):
                data_key = "data" if "data" in data else list(data.keys())[0] if data else "empty"
                count = len(data.get("data", data.get(data_key, []))) if isinstance(data.get("data", data.get(data_key, [])), list) else 1
                logger.info(f"[BallDontLie] ✅ {sport_code}: {count} items")
                console.print(f"[green][BallDontLie] ✅ {count} items[/green]")
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[BallDontLie] HTTP {e.response.status_code}: {e}")
            console.print(f"[red][BallDontLie] ❌ HTTP {e.response.status_code}[/red]")
            return None
        except Exception as e:
            logger.error(f"[BallDontLie] Error: {e}")
            console.print(f"[red][BallDontLie] ❌ Error: {str(e)[:50]}[/red]")
            return None
    
    async def _paginated_request(
        self,
        endpoint: str,
        sport_code: str,
        params: Dict[str, Any] = None,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Make paginated API request, collecting all pages.
        
        Args:
            endpoint: API endpoint
            sport_code: Sport code
            params: Base query parameters
            max_pages: Maximum pages to fetch
            
        Returns:
            List of all items across all pages
        """
        all_items = []
        params = params or {}
        params["per_page"] = 100  # Max per page
        
        cursor = None
        page = 0
        
        while page < max_pages:
            if cursor:
                params["cursor"] = cursor
            
            data = await self._api_request(endpoint, params, sport_code)
            
            if not data:
                break
            
            # Handle different response formats
            items = data.get("data", [])
            if isinstance(items, list):
                all_items.extend(items)
            elif isinstance(items, dict):
                all_items.append(items)
            
            # Check for next page
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            
            if not cursor:
                break
            
            page += 1
            
            # Small delay between pages
            await asyncio.sleep(0.1)
        
        return all_items
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if isinstance(data, list):
            return len(data) > 0
        if isinstance(data, dict):
            return bool(data.get("data") or len(data) > 0)
        return bool(data)
    
    # =========================================================================
    # TEAM COLLECTION
    # =========================================================================
    
    async def collect_teams(self, sport_code: str) -> List[Dict[str, Any]]:
        """
        Collect all teams for a sport.
        
        Args:
            sport_code: Sport code (NBA, NFL, etc.)
            
        Returns:
            List of team dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            logger.warning(f"[BallDontLie] Unknown sport: {sport_code}")
            return []
        
        endpoint = config["endpoints"].get("teams")
        if not endpoint:
            logger.warning(f"[BallDontLie] No teams endpoint for {sport_code}")
            return []
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} teams...[/bold blue]")
        
        teams = await self._paginated_request(endpoint, sport_code)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(teams)} teams collected[/green]")
        return teams
    
    async def save_teams_to_database(
        self,
        teams: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save teams to database.
        
        Args:
            teams: List of team data from API
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of teams saved
        """
        if not teams:
            return 0
        
        # Get or create sport
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return 0
        
        saved = 0
        for team_data in teams:
            try:
                external_id = f"bdl_{sport_code}_{team_data.get('id', '')}"
                
                # Check if exists
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == external_id
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.name = team_data.get("full_name") or team_data.get("name", "")
                    existing.abbreviation = team_data.get("abbreviation", "")[:10]
                    existing.city = team_data.get("city", "")
                    existing.conference = team_data.get("conference", "")
                    existing.division = team_data.get("division", "")
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new
                    team = Team(
                        id=uuid4(),
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_data.get("full_name") or team_data.get("name", ""),
                        abbreviation=team_data.get("abbreviation", "UNK")[:10],
                        city=team_data.get("city", ""),
                        conference=team_data.get("conference", ""),
                        division=team_data.get("division", ""),
                        elo_rating=1500.0,
                        is_active=True,
                    )
                    session.add(team)
                    saved += 1
                
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving team: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} new teams[/green]")
        return saved
    
    # =========================================================================
    # PLAYER COLLECTION
    # =========================================================================
    
    async def collect_players(
        self,
        sport_code: str,
        team_id: int = None,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Collect all players for a sport.
        
        Args:
            sport_code: Sport code
            team_id: Optional team filter
            max_pages: Maximum pages to fetch
            
        Returns:
            List of player dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("players")
        if not endpoint:
            return []
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} players...[/bold blue]")
        
        params = {}
        if team_id:
            params["team_ids[]"] = team_id
        
        players = await self._paginated_request(endpoint, sport_code, params, max_pages)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(players)} players collected[/green]")
        return players
    
    async def save_players_to_database(
        self,
        players: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save players to database.
        
        Args:
            players: List of player data
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of players saved
        """
        if not players:
            return 0
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return 0
        
        saved = 0
        for player_data in players:
            try:
                external_id = f"bdl_{sport_code}_{player_data.get('id', '')}"
                
                # Check if exists
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get team if available
                team_id = None
                if player_data.get("team"):
                    team_ext_id = f"bdl_{sport_code}_{player_data['team'].get('id', '')}"
                    team_result = await session.execute(
                        select(Team).where(Team.external_id == team_ext_id)
                    )
                    team = team_result.scalar_one_or_none()
                    if team:
                        team_id = team.id
                
                # Parse height
                height = None
                if player_data.get("height"):
                    height = player_data.get("height")
                elif player_data.get("height_feet") and player_data.get("height_inches"):
                    height = f"{player_data['height_feet']}'{player_data['height_inches']}\""
                
                # Parse birth date
                birth_date = None
                if player_data.get("birth_date"):
                    try:
                        birth_date = datetime.strptime(
                            player_data["birth_date"], "%Y-%m-%d"
                        ).date()
                    except:
                        pass
                
                if existing:
                    # Update
                    existing.name = f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip()
                    existing.position = player_data.get("position", "")
                    existing.height = height
                    existing.weight = player_data.get("weight_pounds") or player_data.get("weight")
                    existing.jersey_number = player_data.get("jersey_number")
                    existing.team_id = team_id
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new
                    player = Player(
                        id=uuid4(),
                        external_id=external_id,
                        team_id=team_id,
                        name=f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip(),
                        position=player_data.get("position", ""),
                        jersey_number=player_data.get("jersey_number"),
                        birth_date=birth_date,
                        height=height,
                        weight=player_data.get("weight_pounds") or player_data.get("weight"),
                        is_active=True,
                    )
                    session.add(player)
                    saved += 1
                
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving player: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} new players[/green]")
        return saved
    
    # =========================================================================
    # GAME COLLECTION
    # =========================================================================
    
    async def collect_games(
        self,
        sport_code: str,
        season: int = None,
        start_date: str = None,
        end_date: str = None,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Collect games for a sport and season.
        
        Args:
            sport_code: Sport code
            season: Season year
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_pages: Maximum pages
            
        Returns:
            List of game dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        # Tennis uses "matches" endpoint
        endpoint = config["endpoints"].get("games") or config["endpoints"].get("matches")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} games{f' ({season})' if season else ''}...[/bold blue]")
        
        games = await self._paginated_request(endpoint, sport_code, params, max_pages)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(games)} games collected[/green]")
        return games
    
    async def save_games_to_database(
        self,
        games: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save games to database.
        
        Args:
            games: List of game data
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of games saved
        """
        if not games:
            return 0
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return 0
        
        saved = 0
        for game_data in games:
            try:
                external_id = f"bdl_{sport_code}_{game_data.get('id', '')}"
                
                # Check if exists
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get teams
                home_team_id = None
                away_team_id = None
                
                home_team_data = game_data.get("home_team", {})
                away_team_data = game_data.get("visitor_team", {}) or game_data.get("away_team", {})
                
                if home_team_data:
                    home_ext_id = f"bdl_{sport_code}_{home_team_data.get('id', '')}"
                    home_result = await session.execute(
                        select(Team).where(Team.external_id == home_ext_id)
                    )
                    home_team = home_result.scalar_one_or_none()
                    if home_team:
                        home_team_id = home_team.id
                
                if away_team_data:
                    away_ext_id = f"bdl_{sport_code}_{away_team_data.get('id', '')}"
                    away_result = await session.execute(
                        select(Team).where(Team.external_id == away_ext_id)
                    )
                    away_team = away_result.scalar_one_or_none()
                    if away_team:
                        away_team_id = away_team.id
                
                if not home_team_id or not away_team_id:
                    continue
                
                # Parse date
                scheduled_at = datetime.utcnow()
                if game_data.get("date"):
                    try:
                        scheduled_at = datetime.strptime(
                            game_data["date"][:19], "%Y-%m-%dT%H:%M:%S"
                        )
                    except:
                        try:
                            scheduled_at = datetime.strptime(
                                game_data["date"], "%Y-%m-%d"
                            )
                        except:
                            pass
                
                # Determine status
                status = GameStatus.SCHEDULED
                if game_data.get("status") == "Final":
                    status = GameStatus.FINAL
                elif game_data.get("status") in ["In Progress", "Live"]:
                    status = GameStatus.IN_PROGRESS
                elif game_data.get("status") == "Postponed":
                    status = GameStatus.POSTPONED
                
                if existing:
                    # Update
                    existing.home_score = game_data.get("home_team_score")
                    existing.away_score = game_data.get("visitor_team_score") or game_data.get("away_team_score")
                    existing.status = status
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new
                    game = Game(
                        id=uuid4(),
                        sport_id=sport.id,
                        external_id=external_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        scheduled_at=scheduled_at,
                        status=status,
                        home_score=game_data.get("home_team_score"),
                        away_score=game_data.get("visitor_team_score") or game_data.get("away_team_score"),
                    )
                    session.add(game)
                    saved += 1
                
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving game: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} new games[/green]")
        return saved
    
    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def collect_player_stats(
        self,
        sport_code: str,
        season: int = None,
        player_id: int = None,
        game_id: int = None,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Collect player stats for a sport.
        
        Args:
            sport_code: Sport code
            season: Season year
            player_id: Optional player filter
            game_id: Optional game filter
            max_pages: Maximum pages
            
        Returns:
            List of stat dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("stats") or config["endpoints"].get("season_stats")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        if player_id:
            params["player_ids[]"] = player_id
        if game_id:
            params["game_ids[]"] = game_id
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} player stats{f' ({season})' if season else ''}...[/bold blue]")
        
        stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(stats)} stat records collected[/green]")
        return stats
    
    async def save_player_stats_to_database(
        self,
        stats: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save player stats to database.
        
        Args:
            stats: List of stat data
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of stats saved
        """
        if not stats:
            return 0
        
        saved = 0
        
        # Sport-specific stat mappings
        stat_mappings = {
            "NBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min", "fgm", "fga", "fg_pct", 
                    "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "pf"],
            "NFL": ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", 
                    "receiving_yards", "receiving_tds", "receptions", "interceptions", 
                    "fumbles", "tackles", "sacks"],
            "MLB": ["hits", "at_bats", "runs", "rbi", "home_runs", "stolen_bases", 
                    "batting_avg", "era", "strikeouts", "wins", "losses"],
            "NHL": ["goals", "assists", "points", "plus_minus", "pim", "shots", 
                    "hits", "blocks", "takeaways", "giveaways"],
            "WNBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min"],
            "NCAAF": ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns"],
            "NCAAB": ["pts", "reb", "ast", "stl", "blk", "min"],
        }
        
        stat_fields = stat_mappings.get(sport_code, ["pts", "reb", "ast"])
        
        for stat_data in stats:
            try:
                # Get player
                player_api_id = stat_data.get("player", {}).get("id")
                if not player_api_id:
                    continue
                
                player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                player_result = await session.execute(
                    select(Player).where(Player.external_id == player_ext_id)
                )
                player = player_result.scalar_one_or_none()
                
                if not player:
                    continue
                
                # Get game if available
                game_id = None
                if stat_data.get("game", {}).get("id"):
                    game_ext_id = f"bdl_{sport_code}_{stat_data['game']['id']}"
                    game_result = await session.execute(
                        select(Game).where(Game.external_id == game_ext_id)
                    )
                    game = game_result.scalar_one_or_none()
                    if game:
                        game_id = game.id
                
                # Save each stat type
                for stat_type in stat_fields:
                    value = stat_data.get(stat_type)
                    if value is not None:
                        try:
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
                logger.error(f"[BallDontLie] Error saving player stat: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} player stats[/green]")
        return saved
    
    # =========================================================================
    # INJURY COLLECTION
    # =========================================================================
    
    async def collect_injuries(self, sport_code: str) -> List[Dict[str, Any]]:
        """
        Collect current injuries for a sport.
        
        Args:
            sport_code: Sport code
            
        Returns:
            List of injury dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("injuries")
        if not endpoint:
            console.print(f"[yellow][BallDontLie] No injuries endpoint for {sport_code}[/yellow]")
            return []
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} injuries...[/bold blue]")
        
        injuries = await self._paginated_request(endpoint, sport_code)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(injuries)} injuries collected[/green]")
        return injuries
    
    async def save_injuries_to_database(
        self,
        injuries: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save injuries to database.
        
        Args:
            injuries: List of injury data
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of injuries saved
        """
        if not injuries:
            return 0
        
        saved = 0
        for injury_data in injuries:
            try:
                # Get player
                player_api_id = injury_data.get("player", {}).get("id")
                player_name = injury_data.get("player", {}).get("first_name", "") + " " + \
                              injury_data.get("player", {}).get("last_name", "")
                
                player = None
                player_id = None
                if player_api_id:
                    player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                    player_result = await session.execute(
                        select(Player).where(Player.external_id == player_ext_id)
                    )
                    player = player_result.scalar_one_or_none()
                    if player:
                        player_id = player.id
                
                # Get team
                team_api_id = injury_data.get("team", {}).get("id")
                if not team_api_id:
                    continue
                
                team_ext_id = f"bdl_{sport_code}_{team_api_id}"
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_ext_id)
                )
                team = team_result.scalar_one_or_none()
                if not team:
                    continue
                
                # Parse status
                status = injury_data.get("status", "Unknown")
                injury_desc = injury_data.get("comment") or injury_data.get("description", "")
                
                # Create or update injury
                injury = Injury(
                    id=uuid4(),
                    player_id=player_id,
                    team_id=team.id,
                    sport_code=sport_code,
                    player_name=player_name.strip() or "Unknown",
                    position=injury_data.get("player", {}).get("position"),
                    injury_type=injury_desc[:200] if injury_desc else None,
                    status=status,
                    source="balldontlie",
                    external_id=f"bdl_{sport_code}_inj_{injury_data.get('id', '')}",
                )
                session.add(injury)
                saved += 1
                
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving injury: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} injuries[/green]")
        return saved
    
    # =========================================================================
    # STANDINGS / TEAM STATS
    # =========================================================================
    
    async def collect_standings(
        self,
        sport_code: str,
        season: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect standings (team stats) for a sport.
        
        Args:
            sport_code: Sport code
            season: Season year
            
        Returns:
            List of standings dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("standings")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} standings{f' ({season})' if season else ''}...[/bold blue]")
        
        standings = await self._paginated_request(endpoint, sport_code, params)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(standings)} standings records collected[/green]")
        return standings
    
    async def save_team_stats_to_database(
        self,
        standings: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save team stats from standings data.
        
        Args:
            standings: List of standings data
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of stats saved
        """
        if not standings:
            return 0
        
        saved = 0
        
        # Stat types to extract from standings
        stat_types = ["wins", "losses", "ties", "win_pct", "points_for", "points_against",
                      "home_wins", "home_losses", "away_wins", "away_losses",
                      "streak", "last_10", "games_back"]
        
        for standing in standings:
            try:
                # Get team
                team_api_id = standing.get("team", {}).get("id")
                if not team_api_id:
                    continue
                
                team_ext_id = f"bdl_{sport_code}_{team_api_id}"
                team_result = await session.execute(
                    select(Team).where(Team.external_id == team_ext_id)
                )
                team = team_result.scalar_one_or_none()
                if not team:
                    continue
                
                # Save each stat
                for stat_type in stat_types:
                    value = standing.get(stat_type)
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
                            games_played=standing.get("games_played", 0),
                        )
                        session.add(stat)
                        saved += 1
                
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving team stat: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} team stats[/green]")
        return saved
    
    # =========================================================================
    # ODDS COLLECTION (V2 Endpoints)
    # =========================================================================
    
    async def collect_odds(
        self,
        sport_code: str,
        dates: List[str] = None,
        game_ids: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect betting odds for a sport (V2 endpoint).
        
        Available vendors: betmgm, fanduel, draftkings, bet365, caesars, espnbet,
                          ballybet, betway, betparx, betrivers, rebet, polymarket, kalshi
        
        Args:
            sport_code: Sport code (NBA, NFL, etc.)
            dates: List of dates in YYYY-MM-DD format
            game_ids: List of game IDs
            
        Returns:
            List of odds dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("odds")
        if not endpoint:
            console.print(f"[yellow][BallDontLie] No odds endpoint for {sport_code}[/yellow]")
            return []
        
        params = {}
        if dates:
            params["dates[]"] = dates
        if game_ids:
            params["game_ids[]"] = game_ids
        
        # Need at least one of dates or game_ids
        if not dates and not game_ids:
            # Default to today
            from datetime import date as dt_date
            params["dates[]"] = dt_date.today().isoformat()
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} odds...[/bold blue]")
        
        odds = await self._paginated_request(endpoint, sport_code, params)
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(odds)} odds records collected[/green]")
        return odds
    
    async def collect_player_props(
        self,
        sport_code: str,
        game_id: int,
        player_id: int = None,
        prop_type: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect player props for a specific game (V2 endpoint).
        
        Player props are LIVE and updated in real-time. Historical data is not stored.
        
        Supported prop types: points, rebounds, assists, threes, steals, blocks,
                             points_rebounds, points_assists, points_rebounds_assists,
                             rebounds_assists, double_double, triple_double
        
        Available vendors: draftkings, betway, betrivers, ballybet, betparx,
                          caesars, fanduel, rebet
        
        Args:
            sport_code: Sport code (NBA, NFL, MLB, etc.)
            game_id: Game ID (required)
            player_id: Optional player ID filter
            prop_type: Optional prop type filter
            
        Returns:
            List of player prop dictionaries
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("player_props")
        if not endpoint:
            console.print(f"[yellow][BallDontLie] No player props endpoint for {sport_code}[/yellow]")
            return []
        
        params = {"game_id": game_id}
        if player_id:
            params["player_id"] = player_id
        if prop_type:
            params["prop_type"] = prop_type
        
        console.print(f"[bold blue][BallDontLie] Collecting {sport_code} player props for game {game_id}...[/bold blue]")
        
        # Player props returns all props in single response (no pagination)
        data = await self._api_request(endpoint, params, sport_code)
        
        props = data.get("data", []) if data else []
        
        console.print(f"[green][BallDontLie] {sport_code}: {len(props)} player props collected[/green]")
        return props
    
    async def save_odds_to_database(
        self,
        odds: List[Dict[str, Any]],
        sport_code: str,
        session: AsyncSession,
    ) -> int:
        """
        Save odds to database.
        
        Saves to odds table with vendor as sportsbook.
        
        Args:
            odds: List of odds data from API
            sport_code: Sport code
            session: Database session
            
        Returns:
            Number of odds saved
        """
        if not odds:
            return 0
        
        from app.models import Odds  # Import here to avoid circular import
        
        saved = 0
        for odds_data in odds:
            try:
                # Get game
                game_api_id = odds_data.get("game_id")
                if not game_api_id:
                    continue
                
                game_ext_id = f"bdl_{sport_code}_{game_api_id}"
                game_result = await session.execute(
                    select(Game).where(Game.external_id == game_ext_id)
                )
                game = game_result.scalar_one_or_none()
                if not game:
                    continue
                
                vendor = odds_data.get("vendor", "unknown")
                external_id = f"bdl_{sport_code}_odds_{odds_data.get('id', '')}"
                
                # Check if exists
                existing_result = await session.execute(
                    select(Odds).where(Odds.external_id == external_id)
                )
                existing = existing_result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.spread_home = self._parse_spread(odds_data.get("spread_home_value"))
                    existing.spread_away = self._parse_spread(odds_data.get("spread_away_value"))
                    existing.spread_home_odds = odds_data.get("spread_home_odds")
                    existing.spread_away_odds = odds_data.get("spread_away_odds")
                    existing.moneyline_home = odds_data.get("moneyline_home_odds")
                    existing.moneyline_away = odds_data.get("moneyline_away_odds")
                    existing.total = self._parse_total(odds_data.get("total_value"))
                    existing.over_odds = odds_data.get("total_over_odds")
                    existing.under_odds = odds_data.get("total_under_odds")
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new
                    from app.models import Odds
                    odds_record = Odds(
                        id=uuid4(),
                        game_id=game.id,
                        sportsbook_name=vendor,
                        external_id=external_id,
                        spread_home=self._parse_spread(odds_data.get("spread_home_value")),
                        spread_away=self._parse_spread(odds_data.get("spread_away_value")),
                        spread_home_odds=odds_data.get("spread_home_odds"),
                        spread_away_odds=odds_data.get("spread_away_odds"),
                        moneyline_home=odds_data.get("moneyline_home_odds"),
                        moneyline_away=odds_data.get("moneyline_away_odds"),
                        total=self._parse_total(odds_data.get("total_value")),
                        over_odds=odds_data.get("total_over_odds"),
                        under_odds=odds_data.get("total_under_odds"),
                        source="balldontlie",
                    )
                    session.add(odds_record)
                    saved += 1
                    
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving odds: {e}")
        
        await session.commit()
        console.print(f"[green][BallDontLie] {sport_code}: Saved {saved} new odds[/green]")
        return saved
    
    def _parse_spread(self, value: str) -> Optional[float]:
        """Parse spread value from string."""
        if value is None:
            return None
        try:
            return float(value)
        except:
            return None
    
    def _parse_total(self, value: str) -> Optional[float]:
        """Parse total value from string."""
        if value is None:
            return None
        try:
            return float(value)
        except:
            return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def _get_or_create_sport(
        self,
        sport_code: str,
        session: AsyncSession,
    ) -> Optional[Sport]:
        """Get or create sport record."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return None
        
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                id=uuid4(),
                code=sport_code,
                name=config["name"],
                is_active=True,
            )
            session.add(sport)
            await session.commit()
            await session.refresh(sport)
            console.print(f"[green][BallDontLie] Created sport: {sport_code}[/green]")
        
        return sport
    
    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sport_code: str = None,
        sports: List[str] = None,
        seasons: int = 10,
        collect_teams: bool = True,
        collect_players: bool = True,
        collect_games: bool = True,
        collect_stats: bool = True,
        collect_injuries: bool = True,
        collect_standings: bool = True,
    ) -> CollectorResult:
        """
        Main collection method - collect all data for specified sports.
        
        Args:
            sport_code: Single sport to collect
            sports: List of sports to collect
            seasons: Number of seasons to collect (default 10)
            collect_teams: Whether to collect teams
            collect_players: Whether to collect players
            collect_games: Whether to collect games
            collect_stats: Whether to collect player stats
            collect_injuries: Whether to collect injuries
            collect_standings: Whether to collect standings
            
        Returns:
            CollectorResult with counts and status
        """
        result = CollectorResult(success=False, data={})
        total_records = 0
        
        # Determine sports to collect
        if sport_code:
            sports_to_collect = [sport_code]
        elif sports:
            sports_to_collect = sports
        else:
            sports_to_collect = SUPPORTED_SPORTS
        
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]BALLDONTLIE DATA COLLECTION[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        console.print(f"[cyan]Sports: {', '.join(sports_to_collect)}[/cyan]")
        console.print(f"[cyan]Seasons: {seasons} years[/cyan]")
        console.print(f"[cyan]API Key: {self.api_key[:8]}...[/cyan]")
        console.print(f"[bold green]{'='*60}[/bold green]\n")
        
        current_year = datetime.now().year
        
        try:
            from app.core.database import db_manager
            await db_manager.initialize()
            
            for sport in sports_to_collect:
                config = SPORT_CONFIG.get(sport)
                if not config:
                    console.print(f"[yellow]⚠️ Unknown sport: {sport}[/yellow]")
                    continue
                
                console.print(f"\n[bold blue]{'='*40}[/bold blue]")
                console.print(f"[bold blue]{sport} - {config['name']}[/bold blue]")
                console.print(f"[bold blue]{'='*40}[/bold blue]\n")
                
                async with db_manager.session() as session:
                    # 1. Collect Teams
                    if collect_teams:
                        teams = await self.collect_teams(sport)
                        if teams:
                            saved = await self.save_teams_to_database(teams, sport, session)
                            total_records += saved
                    
                    # 2. Collect Players
                    if collect_players:
                        players = await self.collect_players(sport)
                        if players:
                            saved = await self.save_players_to_database(players, sport, session)
                            total_records += saved
                    
                    # 3. Collect Games (historical)
                    if collect_games:
                        start_year = config.get("season_start_year", current_year - seasons)
                        for year in range(start_year, current_year + 1):
                            games = await self.collect_games(sport, season=year)
                            if games:
                                saved = await self.save_games_to_database(games, sport, session)
                                total_records += saved
                            await asyncio.sleep(0.2)  # Rate limit between seasons
                    
                    # 4. Collect Player Stats (historical)
                    if collect_stats:
                        start_year = config.get("season_start_year", current_year - seasons)
                        for year in range(start_year, current_year + 1):
                            stats = await self.collect_player_stats(sport, season=year)
                            if stats:
                                saved = await self.save_player_stats_to_database(stats, sport, session)
                                total_records += saved
                            await asyncio.sleep(0.2)
                    
                    # 5. Collect Injuries
                    if collect_injuries:
                        injuries = await self.collect_injuries(sport)
                        if injuries:
                            saved = await self.save_injuries_to_database(injuries, sport, session)
                            total_records += saved
                    
                    # 6. Collect Standings / Team Stats
                    if collect_standings:
                        standings = await self.collect_standings(sport)
                        if standings:
                            saved = await self.save_team_stats_to_database(standings, sport, session)
                            total_records += saved
                
                console.print(f"[green]✅ {sport} collection complete[/green]")
            
            result.success = True
            result.records_count = total_records
            
        except Exception as e:
            logger.error(f"[BallDontLie] Collection error: {e}")
            console.print(f"[red]❌ Error: {e}[/red]")
            result.error = str(e)
        finally:
            await self.close()
        
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]COLLECTION COMPLETE - {total_records} total records[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]\n")
        
        return result
    
    # =========================================================================
    # COMPREHENSIVE HISTORICAL COLLECTION
    # =========================================================================
    
    async def collect_full_history(
        self,
        sports: List[str] = None,
        years: int = 10,
    ) -> Dict[str, int]:
        """
        Collect full historical data for all sports (10 years).
        
        Args:
            sports: List of sports to collect (default: all 9)
            years: Number of years to collect
            
        Returns:
            Dictionary with counts per data type
        """
        sports = sports or SUPPORTED_SPORTS
        
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]BALLDONTLIE - FULL 10-YEAR HISTORICAL IMPORT[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[cyan]Sports: {', '.join(sports)}[/cyan]")
        console.print(f"[cyan]Years: {years}[/cyan]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]\n")
        
        totals = {
            "teams": 0,
            "players": 0,
            "games": 0,
            "player_stats": 0,
            "team_stats": 0,
            "injuries": 0,
        }
        
        result = await self.collect(
            sports=sports,
            seasons=years,
            collect_teams=True,
            collect_players=True,
            collect_games=True,
            collect_stats=True,
            collect_injuries=True,
            collect_standings=True,
        )
        
        totals["total"] = result.records_count
        
        return totals


# =============================================================================
# COLLECTOR INSTANCE AND REGISTRATION
# =============================================================================

# Create singleton instance
balldontlie_collector = BallDontLieCollector()

# Register with collector manager
try:
    collector_manager.register(balldontlie_collector)
    logger.info("Registered collector: BallDontLie")
except Exception as e:
    logger.warning(f"Could not register BallDontLie collector: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def collect_balldontlie(
    sport_code: str = None,
    sports: List[str] = None,
    seasons: int = 10,
) -> CollectorResult:
    """
    Convenience function to collect BallDontLie data.
    
    Args:
        sport_code: Single sport code
        sports: List of sport codes
        seasons: Number of seasons
        
    Returns:
        CollectorResult
    """
    return await balldontlie_collector.collect(
        sport_code=sport_code,
        sports=sports,
        seasons=seasons,
    )


async def collect_balldontlie_full_history(years: int = 10) -> Dict[str, int]:
    """
    Convenience function for full historical collection.
    
    Args:
        years: Number of years to collect
        
    Returns:
        Dictionary with counts
    """
    return await balldontlie_collector.collect_full_history(years=years)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BallDontLie Data Collector")
    parser.add_argument("--sport", type=str, help="Single sport to collect")
    parser.add_argument("--sports", nargs="+", help="List of sports to collect")
    parser.add_argument("--seasons", type=int, default=10, help="Number of seasons (default: 10)")
    parser.add_argument("--full", action="store_true", help="Full 10-year historical import")
    parser.add_argument("--teams-only", action="store_true", help="Only collect teams")
    parser.add_argument("--players-only", action="store_true", help="Only collect players")
    parser.add_argument("--games-only", action="store_true", help="Only collect games")
    parser.add_argument("--injuries-only", action="store_true", help="Only collect injuries")
    
    args = parser.parse_args()
    
    async def main():
        if args.full:
            result = await collect_balldontlie_full_history(years=args.seasons)
            console.print(f"\n[bold green]Results: {result}[/bold green]")
        else:
            result = await balldontlie_collector.collect(
                sport_code=args.sport,
                sports=args.sports,
                seasons=args.seasons,
                collect_teams=args.teams_only or not any([args.players_only, args.games_only, args.injuries_only]),
                collect_players=args.players_only or not any([args.teams_only, args.games_only, args.injuries_only]),
                collect_games=args.games_only or not any([args.teams_only, args.players_only, args.injuries_only]),
                collect_stats=not any([args.teams_only, args.players_only, args.games_only, args.injuries_only]),
                collect_injuries=args.injuries_only or not any([args.teams_only, args.players_only, args.games_only]),
                collect_standings=not any([args.teams_only, args.players_only, args.games_only, args.injuries_only]),
            )
            console.print(f"\n[bold green]Records collected: {result.records_count}[/bold green]")
    
    asyncio.run(main())