"""
BallDontLie API Collector - V2 FIXED
=====================================
Comprehensive multi-sport data collector from BallDontLie API.
$299/month plan - Full access to all sports and endpoints.

Supports 9 sports:
- NBA, NFL, MLB, NHL (Team Sports)
- WNBA, NCAAF, NCAAB (Team Sports)
- ATP, WTA (Tennis - Individual Sport)

FIXES in V2:
- Properly handles tennis (ATP/WTA) matches with player_1/player_2
- Creates pseudo-teams for tennis players
- Properly saves all data types
- Better error handling and logging
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

# Sport configurations
SPORT_CONFIG = {
    "NBA": {
        "code": "NBA",
        "name": "National Basketball Association",
        "is_tennis": False,
        "endpoints": {
            "teams": "/v1/teams",
            "players": "/v1/players",
            "games": "/v1/games",
            "stats": "/v1/stats",
            "standings": "/v1/standings",
            "injuries": "/v1/player_injuries",
            "odds": "/v2/odds",
        },
        "season_start": 2015,
    },
    "NFL": {
        "code": "NFL",
        "name": "National Football League",
        "is_tennis": False,
        "endpoints": {
            "teams": "/nfl/v1/teams",
            "players": "/nfl/v1/players",
            "games": "/nfl/v1/games",
            "stats": "/nfl/v1/stats",
            "standings": "/nfl/v1/standings",
            "injuries": "/nfl/v1/player_injuries",
            "odds": "/nfl/v2/odds",
        },
        "season_start": 2015,
    },
    "MLB": {
        "code": "MLB",
        "name": "Major League Baseball",
        "is_tennis": False,
        "endpoints": {
            "teams": "/mlb/v1/teams",
            "players": "/mlb/v1/players",
            "games": "/mlb/v1/games",
            "stats": "/mlb/v1/stats",
            "standings": "/mlb/v1/standings",
            "injuries": "/mlb/v1/player_injuries",
            "odds": "/mlb/v2/odds",
        },
        "season_start": 2015,
    },
    "NHL": {
        "code": "NHL",
        "name": "National Hockey League",
        "is_tennis": False,
        "endpoints": {
            "teams": "/nhl/v1/teams",
            "players": "/nhl/v1/players",
            "games": "/nhl/v1/games",
            "stats": "/nhl/v1/stats",
            "standings": "/nhl/v1/standings",
            "injuries": "/nhl/v1/player_injuries",
            "odds": "/nhl/v2/odds",
        },
        "season_start": 2015,
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
            "standings": "/wnba/v1/standings",
            "injuries": "/wnba/v1/player_injuries",
            "odds": "/wnba/v2/odds",
        },
        "season_start": 2018,
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
            "standings": "/ncaaf/v1/standings",
            "injuries": "/ncaaf/v1/player_injuries",
            "odds": "/ncaaf/v2/odds",
        },
        "season_start": 2015,
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
            "standings": "/ncaab/v1/standings",
            "injuries": "/ncaab/v1/player_injuries",
            "odds": "/ncaab/v2/odds",
        },
        "season_start": 2015,
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
            "odds": "/atp/v2/odds",
        },
        "season_start": 2017,
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
            "odds": "/wta/v2/odds",
        },
        "season_start": 2017,
    },
}

SUPPORTED_SPORTS = list(SPORT_CONFIG.keys())

# Stat field mappings per sport
STAT_FIELDS = {
    "NBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min", "fgm", "fga", 
            "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "pf"],
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
    "ATP": ["aces", "double_faults", "first_serve_pct", "break_points_saved"],
    "WTA": ["aces", "double_faults", "first_serve_pct", "break_points_saved"],
}


# =============================================================================
# BALLDONTLIE COLLECTOR V2
# =============================================================================

class BallDontLieCollectorV2(BaseCollector):
    """
    BallDontLie API Collector V2 - Fixed version.
    
    Properly handles:
    - All 9 sports including tennis
    - Team sports (NBA, NFL, etc.) and individual sports (ATP, WTA)
    - All data types: teams, players, games, stats, injuries, standings, odds
    """
    
    def __init__(self, api_key: str = None):
        super().__init__()
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
        """
        Validate collected data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
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
            if isinstance(data, dict) and "data" in data:
                count = len(data["data"])
                logger.info(f"[BallDontLie] ✅ {sport_code}: {count} items")
                console.print(f"[green][BallDontLie] ✅ {count} items[/green]")
            
            return data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                logger.warning(f"[BallDontLie] Rate limited, waiting 60s...")
                await asyncio.sleep(60)
                return await self._api_request(endpoint, params, sport_code)
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
        """Fetch all pages of data."""
        all_data = []
        params = params or {}
        params["per_page"] = 100
        cursor = None
        pages = 0
        
        while pages < max_pages:
            if cursor:
                params["cursor"] = cursor
            
            data = await self._api_request(endpoint, params, sport_code)
            if not data:
                break
            
            items = data.get("data", [])
            if not items:
                break
            
            all_data.extend(items)
            pages += 1
            
            # Check for next page
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            
            await asyncio.sleep(self.rate_limit_delay)
        
        return all_data
    
    # =========================================================================
    # GET OR CREATE SPORT
    # =========================================================================
    
    async def _get_or_create_sport(self, sport_code: str, session: AsyncSession) -> Optional[Sport]:
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
            await session.flush()
        
        return sport
    
    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def collect_teams(self, sport_code: str) -> List[Dict]:
        """Collect teams for a sport (team sports only)."""
        config = SPORT_CONFIG.get(sport_code)
        if not config or config.get("is_tennis"):
            return []  # Tennis has no teams
        
        endpoint = config["endpoints"].get("teams")
        if not endpoint:
            return []
        
        console.print(f"[bold blue]📋 Collecting {sport_code} teams...[/bold blue]")
        teams = await self._paginated_request(endpoint, sport_code)
        console.print(f"[green]✅ {sport_code}: {len(teams)} teams collected[/green]")
        return teams
    
    async def save_teams(
        self, 
        teams: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save teams to database."""
        if not teams:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        saved = 0
        updated = 0
        skipped = 0
        
        for team_data in teams:
            try:
                external_id = f"bdl_{sport_code}_{team_data.get('id', '')}"
                team_name = team_data.get("full_name") or team_data.get("name", "")
                
                if not team_name or not team_name.strip():
                    skipped += 1
                    continue
                
                # Check if exists
                result = await session.execute(
                    select(Team).where(Team.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.name = team_name
                    existing.abbreviation = str(team_data.get("abbreviation", ""))[:10] or existing.abbreviation
                    existing.city = str(team_data.get("city", "")) or existing.city
                    existing.conference = str(team_data.get("conference", "")) or existing.conference
                    existing.division = str(team_data.get("division", "")) or existing.division
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                else:
                    # Create new
                    team = Team(
                        id=uuid4(),
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_name,
                        abbreviation=str(team_data.get("abbreviation", "UNK"))[:10],
                        city=str(team_data.get("city", "")),
                        conference=str(team_data.get("conference", "")),
                        division=str(team_data.get("division", "")),
                        elo_rating=1500.0,
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
    # TENNIS: CREATE PSEUDO-TEAMS FROM PLAYERS
    # =========================================================================
    
    async def create_tennis_pseudo_teams(
        self, 
        players: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """
        Create pseudo-teams for tennis players.
        In tennis, each player is treated as their own "team" for game linking.
        """
        if sport_code not in ["ATP", "WTA"]:
            return {"saved": 0, "updated": 0}
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return {"saved": 0, "updated": 0}
        
        saved = 0
        updated = 0
        
        for player_data in players:
            try:
                player_id = player_data.get("id")
                if not player_id:
                    continue
                
                # Create team external_id from player
                external_id = f"bdl_{sport_code}_player_{player_id}"
                
                first_name = player_data.get("first_name", "")
                last_name = player_data.get("last_name", "")
                player_name = f"{first_name} {last_name}".strip()
                
                if not player_name:
                    continue
                
                # Country as "city" for tennis players
                country = player_data.get("country", "")
                
                # Check if pseudo-team exists
                result = await session.execute(
                    select(Team).where(Team.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.name = player_name
                    existing.city = country
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                else:
                    # Create pseudo-team
                    team = Team(
                        id=uuid4(),
                        sport_id=sport.id,
                        external_id=external_id,
                        name=player_name,
                        abbreviation=last_name[:10] if last_name else "UNK",
                        city=country,
                        conference="",
                        division="",
                        elo_rating=1500.0,
                        is_active=True,
                    )
                    session.add(team)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error creating tennis pseudo-team: {e}")
        
        await session.commit()
        console.print(f"[green]🎾 {sport_code} Pseudo-Teams: {saved} saved, {updated} updated[/green]")
        return {"saved": saved, "updated": updated}
    
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
        players = await self._paginated_request(endpoint, sport_code, max_pages=200)
        console.print(f"[green]✅ {sport_code}: {len(players)} players collected[/green]")
        return players
    
    async def save_players(
        self, 
        players: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save players to database."""
        if not players:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        config = SPORT_CONFIG.get(sport_code)
        is_tennis = config.get("is_tennis", False)
        
        saved = 0
        updated = 0
        skipped = 0
        
        for player_data in players:
            try:
                external_id = f"bdl_{sport_code}_{player_data.get('id', '')}"
                
                # Handle name
                first_name = player_data.get("first_name", "")
                last_name = player_data.get("last_name", "")
                player_name = f"{first_name} {last_name}".strip()
                
                if not player_name:
                    skipped += 1
                    continue
                
                # Find team (for team sports)
                team_id = None
                if not is_tennis and player_data.get("team"):
                    team_api_id = player_data["team"].get("id")
                    if team_api_id:
                        team_ext_id = f"bdl_{sport_code}_{team_api_id}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == team_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            team_id = team.id
                
                # Parse height
                height = None
                if player_data.get("height"):
                    height = str(player_data["height"])
                elif player_data.get("height_feet") and player_data.get("height_inches"):
                    height = f"{player_data['height_feet']}'{player_data['height_inches']}\""
                
                # Parse weight
                weight = None
                weight_val = player_data.get("weight_pounds") or player_data.get("weight")
                if weight_val:
                    try:
                        weight = int(weight_val)
                    except:
                        pass
                
                # Parse jersey
                jersey = None
                jersey_val = player_data.get("jersey_number")
                if jersey_val is not None:
                    try:
                        jersey = int(jersey_val)
                    except:
                        pass
                
                # Parse birth date
                birth_date = None
                if player_data.get("birth_date"):
                    try:
                        birth_date = datetime.strptime(player_data["birth_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                # Check if exists
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.name = player_name
                    existing.position = str(player_data.get("position", "")) or existing.position
                    existing.height = height or existing.height
                    existing.weight = weight or existing.weight
                    existing.jersey_number = jersey if jersey is not None else existing.jersey_number
                    existing.team_id = team_id or existing.team_id
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                else:
                    player = Player(
                        id=uuid4(),
                        external_id=external_id,
                        team_id=team_id,
                        name=player_name,
                        position=str(player_data.get("position", "")),
                        height=height,
                        weight=weight,
                        jersey_number=jersey,
                        birth_date=birth_date,
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
    # GAMES COLLECTION
    # =========================================================================
    
    async def collect_games(
        self, 
        sport_code: str, 
        season: int = None,
        max_pages: int = 100
    ) -> List[Dict]:
        """Collect games/matches for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        # Tennis uses "matches", team sports use "games"
        is_tennis = config.get("is_tennis", False)
        endpoint = config["endpoints"].get("matches" if is_tennis else "games")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        
        console.print(f"[bold blue]🏆 Collecting {sport_code} {'matches' if is_tennis else 'games'} ({season or 'current'})...[/bold blue]")
        games = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(games)} {'matches' if is_tennis else 'games'} collected[/green]")
        return games
    
    async def save_games(
        self, 
        games: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save games to database. Handles both team sports and tennis."""
        if not games:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        sport = await self._get_or_create_sport(sport_code, session)
        if not sport:
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        config = SPORT_CONFIG.get(sport_code)
        is_tennis = config.get("is_tennis", False)
        
        saved = 0
        updated = 0
        skipped = 0
        
        for game_data in games:
            try:
                external_id = f"bdl_{sport_code}_{game_data.get('id', '')}"
                
                # Check if exists
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                # Get home/away team IDs
                home_team_id = None
                away_team_id = None
                
                if is_tennis:
                    # Tennis: player_1 = home, player_2 = away
                    player_1 = game_data.get("player_1", {}) or {}
                    player_2 = game_data.get("player_2", {}) or {}
                    
                    if player_1.get("id"):
                        p1_ext_id = f"bdl_{sport_code}_player_{player_1['id']}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == p1_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            home_team_id = team.id
                    
                    if player_2.get("id"):
                        p2_ext_id = f"bdl_{sport_code}_player_{player_2['id']}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == p2_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            away_team_id = team.id
                else:
                    # Team sports: home_team, visitor_team/away_team
                    home_team_data = game_data.get("home_team", {}) or {}
                    away_team_data = game_data.get("visitor_team", {}) or game_data.get("away_team", {}) or {}
                    
                    if home_team_data.get("id"):
                        home_ext_id = f"bdl_{sport_code}_{home_team_data['id']}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == home_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            home_team_id = team.id
                    
                    if away_team_data.get("id"):
                        away_ext_id = f"bdl_{sport_code}_{away_team_data['id']}"
                        result = await session.execute(
                            select(Team).where(Team.external_id == away_ext_id)
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            away_team_id = team.id
                
                # Skip if no teams found
                if not home_team_id or not away_team_id:
                    skipped += 1
                    continue
                
                # Parse date
                scheduled_at = datetime.utcnow()
                date_str = game_data.get("date") or game_data.get("start_time")
                if date_str:
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"]:
                        try:
                            scheduled_at = datetime.strptime(date_str[:19], fmt[:len(date_str[:19])])
                            break
                        except:
                            continue
                
                # Determine status
                status = GameStatus.SCHEDULED
                status_str = str(game_data.get("status", "")).lower()
                if status_str in ["final", "finished", "completed"]:
                    status = GameStatus.FINAL
                elif status_str in ["in_progress", "in progress", "live"]:
                    status = GameStatus.IN_PROGRESS
                elif status_str in ["postponed"]:
                    status = GameStatus.POSTPONED
                
                # Get scores
                if is_tennis:
                    # Tennis scores
                    home_score = game_data.get("player_1_score")
                    away_score = game_data.get("player_2_score")
                else:
                    # Team sport scores
                    home_score = game_data.get("home_team_score")
                    away_score = game_data.get("visitor_team_score") or game_data.get("away_team_score")
                
                # Convert scores to int
                home_score = int(home_score) if home_score is not None else None
                away_score = int(away_score) if away_score is not None else None
                
                if existing:
                    existing.home_score = home_score if home_score is not None else existing.home_score
                    existing.away_score = away_score if away_score is not None else existing.away_score
                    existing.status = status
                    existing.updated_at = datetime.utcnow()
                    updated += 1
                else:
                    game = Game(
                        id=uuid4(),
                        sport_id=sport.id,
                        external_id=external_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        scheduled_at=scheduled_at,
                        status=status,
                        home_score=home_score,
                        away_score=away_score,
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
        max_pages: int = 100
    ) -> List[Dict]:
        """Collect player stats for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("stats")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["seasons[]"] = season
        
        console.print(f"[bold blue]📊 Collecting {sport_code} player stats ({season or 'current'})...[/bold blue]")
        stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(stats)} stat records collected[/green]")
        return stats
    
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
                # Get player
                player_api_id = None
                if stat_data.get("player"):
                    player_api_id = stat_data["player"].get("id")
                
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
                if stat_data.get("game", {}).get("id"):
                    game_ext_id = f"bdl_{sport_code}_{stat_data['game']['id']}"
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
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Player Stats: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}
    
    # =========================================================================
    # INJURIES COLLECTION
    # =========================================================================
    
    async def collect_injuries(self, sport_code: str) -> List[Dict]:
        """Collect injuries for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("injuries")
        if not endpoint:
            console.print(f"[yellow]⚠️ No injuries endpoint for {sport_code}[/yellow]")
            return []
        
        console.print(f"[bold blue]🏥 Collecting {sport_code} injuries...[/bold blue]")
        injuries = await self._paginated_request(endpoint, sport_code)
        console.print(f"[green]✅ {sport_code}: {len(injuries)} injuries collected[/green]")
        return injuries
    
    async def save_injuries(
        self, 
        injuries: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save injuries to database."""
        if not injuries:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for injury_data in injuries:
            try:
                # Get team
                team_api_id = injury_data.get("team", {}).get("id")
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
                
                # Get player (optional)
                player_id = None
                player_api_id = injury_data.get("player", {}).get("id")
                if player_api_id:
                    player_ext_id = f"bdl_{sport_code}_{player_api_id}"
                    result = await session.execute(
                        select(Player).where(Player.external_id == player_ext_id)
                    )
                    player = result.scalar_one_or_none()
                    if player:
                        player_id = player.id
                
                # Player name
                first_name = injury_data.get("player", {}).get("first_name", "")
                last_name = injury_data.get("player", {}).get("last_name", "")
                player_name = f"{first_name} {last_name}".strip() or "Unknown"
                
                injury = Injury(
                    id=uuid4(),
                    player_id=player_id,
                    team_id=team.id,
                    sport_code=sport_code,
                    player_name=player_name,
                    position=injury_data.get("player", {}).get("position"),
                    injury_type=str(injury_data.get("comment") or injury_data.get("description", ""))[:200],
                    status=injury_data.get("status", "Unknown"),
                    source="balldontlie",
                    external_id=f"bdl_{sport_code}_inj_{injury_data.get('id', '')}",
                )
                session.add(injury)
                saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving injury: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Injuries: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}
    
    # =========================================================================
    # STANDINGS / TEAM STATS COLLECTION
    # =========================================================================
    
    async def collect_standings(self, sport_code: str, season: int = None) -> List[Dict]:
        """Collect standings for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("standings")
        if not endpoint:
            console.print(f"[yellow]⚠️ No standings endpoint for {sport_code}[/yellow]")
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]🏅 Collecting {sport_code} standings...[/bold blue]")
        standings = await self._paginated_request(endpoint, sport_code, params)
        console.print(f"[green]✅ {sport_code}: {len(standings)} standings records collected[/green]")
        return standings
    
    async def save_team_stats(
        self, 
        standings: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save team stats from standings."""
        if not standings:
            return {"saved": 0, "skipped": 0}
        
        stat_types = ["wins", "losses", "ties", "win_pct", "points_for", "points_against",
                      "home_wins", "home_losses", "away_wins", "away_losses",
                      "streak", "last_10", "games_back"]
        
        saved = 0
        skipped = 0
        
        for standing in standings:
            try:
                team_api_id = standing.get("team", {}).get("id")
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
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Team Stats: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}
    
    # =========================================================================
    # ODDS COLLECTION
    # =========================================================================
    
    async def collect_odds(
        self, 
        sport_code: str, 
        game_ids: List[int] = None,
        dates: List[str] = None
    ) -> List[Dict]:
        """Collect betting odds for a sport."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("odds")
        if not endpoint:
            console.print(f"[yellow]⚠️ No odds endpoint for {sport_code}[/yellow]")
            return []
        
        params = {}
        if game_ids:
            params["game_ids[]"] = game_ids
        if dates:
            params["dates[]"] = dates
        
        console.print(f"[bold blue]💰 Collecting {sport_code} odds...[/bold blue]")
        odds = await self._paginated_request(endpoint, sport_code, params, max_pages=50)
        console.print(f"[green]✅ {sport_code}: {len(odds)} odds records collected[/green]")
        return odds
    
    async def _get_or_create_sportsbook(
        self, 
        name: str, 
        session: AsyncSession
    ) -> Optional[Sportsbook]:
        """Get or create sportsbook record."""
        result = await session.execute(
            select(Sportsbook).where(Sportsbook.name == name)
        )
        sportsbook = result.scalar_one_or_none()
        
        if not sportsbook:
            sportsbook = Sportsbook(
                id=uuid4(),
                name=name,
                api_key=name.lower().replace(" ", "_"),
                is_active=True,
            )
            session.add(sportsbook)
            await session.flush()
        
        return sportsbook
    
    async def save_odds(
        self, 
        odds_data: List[Dict], 
        sport_code: str, 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Save odds to database."""
        if not odds_data:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for odds_record in odds_data:
            try:
                # Get game
                game_api_id = odds_record.get("game", {}).get("id") or odds_record.get("game_id")
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
                
                # Process each vendor/sportsbook
                vendors = odds_record.get("odds", []) or [odds_record]
                
                for vendor_data in vendors:
                    vendor_name = vendor_data.get("vendor") or vendor_data.get("sportsbook") or "unknown"
                    
                    # Get or create sportsbook
                    sportsbook = await self._get_or_create_sportsbook(vendor_name, session)
                    if not sportsbook:
                        continue
                    
                    # Extract spread data
                    spread_data = vendor_data.get("spread", {}) or {}
                    spread_home = spread_data.get("home_line") or spread_data.get("home")
                    spread_away = spread_data.get("away_line") or spread_data.get("away")
                    spread_home_odds = spread_data.get("home_odds")
                    spread_away_odds = spread_data.get("away_odds")
                    
                    # Extract total data
                    total_data = vendor_data.get("total", {}) or {}
                    total_line = total_data.get("line") or total_data.get("total")
                    over_odds = total_data.get("over_odds") or total_data.get("over")
                    under_odds = total_data.get("under_odds") or total_data.get("under")
                    
                    # Extract moneyline data
                    ml_data = vendor_data.get("moneyline", {}) or {}
                    ml_home = ml_data.get("home") or ml_data.get("home_odds")
                    ml_away = ml_data.get("away") or ml_data.get("away_odds")
                    
                    # Create odds record
                    odds = Odds(
                        id=uuid4(),
                        game_id=game.id,
                        sportsbook_id=sportsbook.id,
                        spread_home=float(spread_home) if spread_home is not None else None,
                        spread_away=float(spread_away) if spread_away is not None else None,
                        spread_home_odds=int(spread_home_odds) if spread_home_odds is not None else None,
                        spread_away_odds=int(spread_away_odds) if spread_away_odds is not None else None,
                        total_line=float(total_line) if total_line is not None else None,
                        over_odds=int(over_odds) if over_odds is not None else None,
                        under_odds=int(under_odds) if under_odds is not None else None,
                        moneyline_home=int(ml_home) if ml_home is not None else None,
                        moneyline_away=int(ml_away) if ml_away is not None else None,
                    )
                    session.add(odds)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving odds: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Odds: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}
    
    # =========================================================================
    # MAIN COLLECTION METHOD - PER SPORT
    # =========================================================================
    
    async def collect_sport(
        self,
        sport_code: str,
        years: int = 10,
        collect_teams: bool = True,
        collect_players: bool = True,
        collect_games: bool = True,
        collect_stats: bool = True,
        collect_injuries: bool = True,
        collect_standings: bool = True,
        collect_odds: bool = True,
    ) -> Dict[str, int]:
        """
        Collect all data for a single sport.
        
        Args:
            sport_code: Sport code (NBA, NFL, ATP, etc.)
            years: Number of years of historical data
            collect_*: Flags to enable/disable data types
            
        Returns:
            Dictionary with counts per data type
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            console.print(f"[red]❌ Unknown sport: {sport_code}[/red]")
            return {}
        
        is_tennis = config.get("is_tennis", False)
        current_year = datetime.now().year
        start_year = config.get("season_start", current_year - years)
        
        results = {
            "teams": {"saved": 0, "updated": 0},
            "players": {"saved": 0, "updated": 0},
            "games": {"saved": 0, "updated": 0},
            "player_stats": {"saved": 0},
            "injuries": {"saved": 0},
            "team_stats": {"saved": 0},
            "odds": {"saved": 0},
        }
        
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]{sport_code} - {config['name']}[/bold cyan]")
        console.print(f"[bold cyan]Seasons: {start_year} to {current_year}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
        
        try:
            from app.core.database import db_manager
            await db_manager.initialize()
            
            async with db_manager.session() as session:
                # 1. Teams (or pseudo-teams for tennis)
                if collect_teams:
                    if is_tennis:
                        # Tennis: collect players first, then create pseudo-teams
                        players = await self.collect_players(sport_code)
                        if players:
                            results["players"] = await self.save_players(players, sport_code, session)
                            results["teams"] = await self.create_tennis_pseudo_teams(players, sport_code, session)
                    else:
                        teams = await self.collect_teams(sport_code)
                        if teams:
                            results["teams"] = await self.save_teams(teams, sport_code, session)
                
                # 2. Players (for team sports - tennis already done above)
                if collect_players and not is_tennis:
                    players = await self.collect_players(sport_code)
                    if players:
                        results["players"] = await self.save_players(players, sport_code, session)
                
                # 3. Games - Historical
                if collect_games:
                    for year in range(start_year, current_year + 1):
                        games = await self.collect_games(sport_code, season=year)
                        if games:
                            year_results = await self.save_games(games, sport_code, session)
                            results["games"]["saved"] += year_results.get("saved", 0)
                            results["games"]["updated"] += year_results.get("updated", 0)
                        await asyncio.sleep(0.2)
                
                # 4. Player Stats - Historical (skip for tennis)
                if collect_stats and not is_tennis:
                    for year in range(start_year, current_year + 1):
                        stats = await self.collect_player_stats(sport_code, season=year)
                        if stats:
                            year_results = await self.save_player_stats(stats, sport_code, session)
                            results["player_stats"]["saved"] += year_results.get("saved", 0)
                        await asyncio.sleep(0.2)
                
                # 5. Injuries
                if collect_injuries and not is_tennis:
                    injuries = await self.collect_injuries(sport_code)
                    if injuries:
                        results["injuries"] = await self.save_injuries(injuries, sport_code, session)
                
                # 6. Standings / Team Stats
                if collect_standings and not is_tennis:
                    standings = await self.collect_standings(sport_code)
                    if standings:
                        results["team_stats"] = await self.save_team_stats(standings, sport_code, session)
                
                # 7. Odds (current/recent only - API doesn't return historical odds)
                if collect_odds:
                    odds_data = await self.collect_odds(sport_code)
                    if odds_data:
                        results["odds"] = await self.save_odds(odds_data, sport_code, session)
        
        except Exception as e:
            logger.error(f"[BallDontLie] Error collecting {sport_code}: {e}")
            console.print(f"[red]❌ Error: {e}[/red]")
        
        # Print summary
        console.print(f"\n[bold green]✅ {sport_code} Collection Complete[/bold green]")
        console.print(f"  Teams: {results['teams'].get('saved', 0)} saved, {results['teams'].get('updated', 0)} updated")
        console.print(f"  Players: {results['players'].get('saved', 0)} saved, {results['players'].get('updated', 0)} updated")
        console.print(f"  Games: {results['games'].get('saved', 0)} saved, {results['games'].get('updated', 0)} updated")
        console.print(f"  Player Stats: {results['player_stats'].get('saved', 0)} saved")
        console.print(f"  Injuries: {results['injuries'].get('saved', 0)} saved")
        console.print(f"  Team Stats: {results['team_stats'].get('saved', 0)} saved")
        console.print(f"  Odds: {results['odds'].get('saved', 0)} saved")
        
        return results
    
    # =========================================================================
    # STANDARD COLLECTOR INTERFACE
    # =========================================================================
    
    async def collect(
        self,
        sport_code: str = None,
        sports: List[str] = None,
        seasons: int = 10,
        **kwargs
    ) -> CollectorResult:
        """Main collection method."""
        result = CollectorResult(success=False, data={})
        total = 0
        
        if sport_code:
            sports_list = [sport_code]
        elif sports:
            sports_list = sports
        else:
            sports_list = SUPPORTED_SPORTS
        
        try:
            for sport in sports_list:
                sport_results = await self.collect_sport(sport, years=seasons, **kwargs)
                for key, val in sport_results.items():
                    if isinstance(val, dict):
                        total += val.get("saved", 0)
            
            result.success = True
            result.records_count = total
            
        except Exception as e:
            result.error = str(e)
        finally:
            await self.close()
        
        return result


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

# Global collector instance
_collector: Optional[BallDontLieCollectorV2] = None

def get_collector() -> BallDontLieCollectorV2:
    global _collector
    if not _collector:
        _collector = BallDontLieCollectorV2()
    return _collector


async def collect_sport(sport_code: str, years: int = 10) -> Dict[str, int]:
    """Collect all data for a single sport."""
    collector = get_collector()
    return await collector.collect_sport(sport_code, years=years)


async def collect_all_sports(years: int = 10) -> Dict[str, Dict]:
    """Collect all data for all 9 sports."""
    collector = get_collector()
    results = {}
    
    for sport in SUPPORTED_SPORTS:
        results[sport] = await collector.collect_sport(sport, years=years)
        await asyncio.sleep(1)  # Brief pause between sports
    
    return results