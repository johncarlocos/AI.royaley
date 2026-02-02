"""
BallDontLie API Collector - V2 UPDATED
=======================================
Comprehensive multi-sport data collector from BallDontLie API.
$299/month plan - Full access to all sports and endpoints.

Supports 9 sports:
- NBA, NFL, MLB, NHL (Team Sports)
- WNBA, NCAAF, NCAAB (Team Sports)
- ATP, WTA (Tennis - Individual Sport)

UPDATES (from manual Docker import sessions):
- Team stats endpoints per sport (NBA/NCAAB/WNBA/NFL/MLB)
- ATP match_stats endpoint (/atp/v1/match_stats)
- ATP/WTA rankings collection
- Tennis historical odds from The Odds API (5yr backfill, 2022+)
- WNBA injury date parsing (expected_return)
- Correct endpoint paths verified against live API
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from rich.console import Console
from sqlalchemy import select, and_, func, or_, text
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
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))

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
            # DISCOVERED: Season averages per team (25 fields)
            "team_season_averages": "/nba/v1/team_season_averages/general",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "has_team_stats_endpoint": True,
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
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
            # DISCOVERED: Per-game team stats (30 fields, paginated)
            "team_stats": "/nfl/v1/team_stats",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "has_team_stats_endpoint": True,
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
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
            # DISCOVERED: Season stats per team (36 fields batting+pitching+fielding)
            "team_season_stats": "/mlb/v1/teams/season_stats",
        },
        "season_start": 2015,
        "has_stats_endpoint": True,
        "has_team_stats_endpoint": True,
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
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
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
            "stats": None,  # WNBA does NOT have /stats endpoint (404)
            "player_stats": "/wnba/v1/player_stats",  # FIXED: Use /player_stats endpoint
            "box_scores": "/wnba/v1/box_scores",
            "standings": "/wnba/v1/standings",
            "injuries": "/wnba/v1/player_injuries",
            "odds": "/v2/odds",
            # DISCOVERED: Season averages per team (18 fields)
            "team_season_stats": "/wnba/v1/team_season_stats",
        },
        "season_start": 2008,
        "has_stats_endpoint": False,
        "has_player_stats_endpoint": True,  # NEW: /player_stats works
        "has_team_stats_endpoint": True,
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
            "stats": None,  # NCAAF does NOT have /stats endpoint (404)
            "player_stats": "/ncaaf/v1/player_stats",  # FIXED: Use /player_stats endpoint
            "box_scores": "/ncaaf/v1/box_scores",
            "standings": None,  # NCAAF standings returns 400
            "injuries": None,  # NCAAF does NOT have injuries endpoint (404)
            "odds": "/v2/odds",
        },
        "season_start": 2004,
        "has_stats_endpoint": False,
        "has_player_stats_endpoint": True,  # NEW: /player_stats works
        "standings_requires_season": False,
    },
    "NCAAB": {
        "code": "NCAAB",
        "name": "NCAA Basketball",
        "is_tennis": False,
        "endpoints": {
            "teams": "/ncaab/v1/teams",
            "players": "/ncaab/v1/players",
            "games": "/ncaab/v1/games",
            "stats": None,  # NCAAB does NOT have /stats endpoint (404)
            "player_stats": "/ncaab/v1/player_stats",  # FIXED: Use /player_stats endpoint
            "box_scores": "/ncaab/v1/box_scores",
            "standings": None,  # NCAAB standings returns 400
            "injuries": None,  # NCAAB does NOT have injuries endpoint (404)
            "odds": "/v2/odds",
            # DISCOVERED: Per-game team stats (17 fields, paginated)
            "team_stats": "/ncaab/v1/team_stats",
        },
        "season_start": 2002,
        "has_stats_endpoint": False,
        "has_player_stats_endpoint": True,  # NEW: /player_stats works
        "has_team_stats_endpoint": True,
        "standings_requires_season": False,
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
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
            # DISCOVERED: Match-level stats (14 fields, filter set_number==0)
            "match_stats": "/atp/v1/match_stats",
        },
        "season_start": 2017,
        "has_stats_endpoint": False,
        "has_match_stats": True,
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
            "odds": "/v2/odds",  # FIXED: unified endpoint per BDL docs
            # WTA does NOT have match_stats endpoint (404)
        },
        "season_start": 2017,
        "has_stats_endpoint": False,
        "has_match_stats": False,
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
    "WNBA": ["pts", "reb", "ast", "stl", "blk", "turnover", "min", "fgm", "fga",
             "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "pf", "plus_minus"],
    "NCAAF": ["passing_completions", "passing_attempts", "passing_yards", "passing_touchdowns",
              "passing_interceptions", "passing_qbr", "passing_rating",
              "rushing_attempts", "rushing_yards", "rushing_touchdowns", "rushing_long",
              "receptions", "receiving_yards", "receiving_touchdowns", "receiving_targets", "receiving_long",
              "total_tackles", "solo_tackles", "tackles_for_loss", "sacks", "interceptions", "passes_defended"],
    "NCAAB": ["pts", "reb", "ast", "stl", "blk", "turnover", "min", "fgm", "fga",
              "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "pf"],
    "ATP": [],
    "WTA": [],
}

# Team stats field mappings per sport (DISCOVERED via manual testing)
TEAM_STATS_FIELDS = {
    "NBA": [
        "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb",
        "pf", "pfd", "min", "w", "l", "gp", "w_pct", "blka",
    ],
    "NCAAB": [
        "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
        "ast", "stl", "blk", "turnovers", "fouls",
    ],
    "WNBA": [
        "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
        "ast", "stl", "blk", "turnover", "pts", "games_played",
    ],
    "NFL": [
        "first_downs", "first_downs_passing", "first_downs_rushing", "first_downs_penalty",
        "third_down_conversions", "third_down_attempts", "fourth_down_conversions", "fourth_down_attempts",
        "total_offensive_plays", "total_yards", "yards_per_play", "total_drives",
        "net_passing_yards", "passing_completions", "passing_attempts", "yards_per_pass",
        "sacks", "sack_yards_lost", "rushing_yards", "rushing_attempts", "yards_per_rush_attempt",
        "red_zone_scores", "red_zone_attempts", "penalties", "penalty_yards",
        "turnovers", "fumbles_lost", "interceptions_thrown", "defensive_touchdowns", "possession_time",
    ],
    "MLB": [
        "gp", "batting_ab", "batting_r", "batting_h", "batting_2b", "batting_3b",
        "batting_hr", "batting_rbi", "batting_tb", "batting_bb", "batting_so",
        "batting_sb", "batting_avg", "batting_obp", "batting_slg", "batting_ops",
        "pitching_w", "pitching_l", "pitching_era", "pitching_sv", "pitching_cg",
        "pitching_sho", "pitching_qs", "pitching_ip", "pitching_h", "pitching_er",
        "pitching_hr", "pitching_bb", "pitching_k", "pitching_oba", "pitching_whip",
        "fielding_e", "fielding_fp", "fielding_tc", "fielding_po", "fielding_a",
    ],
}

# ATP match_stats fields (set_number == 0 = match totals)
TENNIS_MATCH_STAT_FIELDS = [
    "serve_rating", "aces", "double_faults", "first_serve_pct",
    "first_serve_points_won_pct", "second_serve_points_won_pct",
    "break_points_saved_pct", "return_rating", "first_return_won_pct",
    "second_return_won_pct", "break_points_converted_pct",
    "total_service_points_won_pct", "total_return_points_won_pct",
    "total_points_won_pct",
]

# Tennis tournament date ranges for The Odds API historical backfill
ATP_TOURNAMENT_DATES = [
    ("tennis_atp_aus_open_singles", "01-10", "01-30"),
    ("tennis_atp_qatar_open", "01-15", "01-28"),
    ("tennis_atp_dubai", "02-20", "03-05"),
    ("tennis_atp_indian_wells", "03-05", "03-20"),
    ("tennis_atp_miami_open", "03-19", "04-02"),
    ("tennis_atp_monte_carlo_masters", "04-06", "04-16"),
    ("tennis_atp_madrid_open", "04-25", "05-08"),
    ("tennis_atp_italian_open", "05-08", "05-22"),
    ("tennis_atp_french_open", "05-22", "06-12"),
    ("tennis_atp_wimbledon", "06-27", "07-16"),
    ("tennis_atp_canadian_open", "08-01", "08-14"),
    ("tennis_atp_cincinnati_open", "08-11", "08-22"),
    ("tennis_atp_us_open", "08-22", "09-10"),
    ("tennis_atp_china_open", "09-28", "10-10"),
    ("tennis_atp_shanghai_masters", "10-02", "10-16"),
    ("tennis_atp_paris_masters", "10-25", "11-05"),
]

WTA_TOURNAMENT_DATES = [
    ("tennis_wta_aus_open_singles", "01-10", "01-30"),
    ("tennis_wta_qatar_open", "02-06", "02-18"),
    ("tennis_wta_dubai", "02-16", "02-26"),
    ("tennis_wta_indian_wells", "03-05", "03-20"),
    ("tennis_wta_miami_open", "03-19", "04-02"),
    ("tennis_wta_madrid_open", "04-25", "05-08"),
    ("tennis_wta_italian_open", "05-08", "05-22"),
    ("tennis_wta_french_open", "05-22", "06-12"),
    ("tennis_wta_wimbledon", "06-27", "07-16"),
    ("tennis_wta_canadian_open", "08-01", "08-14"),
    ("tennis_wta_cincinnati_open", "08-11", "08-22"),
    ("tennis_wta_us_open", "08-22", "09-10"),
    ("tennis_wta_china_open", "09-25", "10-08"),
    ("tennis_wta_wuhan_open", "10-04", "10-15"),
]


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
        self.odds_api_key = THE_ODDS_API_KEY
        self.client: Optional[httpx.AsyncClient] = None
        self.odds_client: Optional[httpx.AsyncClient] = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        if self.api_key:
            logger.info(f"[BallDontLie] API Key configured: {self.api_key[:8]}...")
            console.print(f"[green][BallDontLie] API Key configured: {self.api_key[:8]}...[/green]")
        else:
            logger.warning("[BallDontLie] No API key configured!")
            console.print("[red][BallDontLie] No API key configured![/red]")
        
        if self.odds_api_key:
            logger.info(f"[BallDontLie] The Odds API key configured: {self.odds_api_key[:8]}...")
        else:
            logger.info("[BallDontLie] No Odds API key - tennis odds will be skipped")
    
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
        """Close HTTP clients."""
        if self.client:
            await self.client.aclose()
            self.client = None
        if self.odds_client:
            await self.odds_client.aclose()
            self.odds_client = None
    
    async def _get_odds_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for The Odds API."""
        if not self.odds_client:
            self.odds_client = httpx.AsyncClient(
                base_url="https://api.the-odds-api.com",
                timeout=30.0,
            )
        return self.odds_client
    
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
                    existing.abbreviation = abbr
                    existing.city = city
                    updated += 1
                else:
                    team = Team(
                        id=uuid4(),
                        external_id=external_id,
                        sport_id=sport.id,
                        name=name,
                        abbreviation=abbr,
                        city=city,
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
                
                # Tennis fallback: link player to their pseudo-team
                if team_id is None and sport_code in ("ATP", "WTA"):
                    # Try by pseudo-team external_id pattern
                    pseudo_ext_id = f"bdl_{sport_code}_player_{player_id}"
                    result = await session.execute(
                        select(Team).where(Team.external_id == pseudo_ext_id)
                    )
                    team = result.scalar_one_or_none()
                    if team:
                        team_id = team.id
                    else:
                        # Fallback: match by name
                        first_name = player_data.get("first_name", "")
                        last_name = player_data.get("last_name", "")
                        player_full_name = f"{first_name} {last_name}".strip()
                        if player_full_name:
                            result = await session.execute(
                                select(Team).where(
                                    and_(
                                        Team.sport_id == (
                                            select(Sport.id).where(Sport.code == sport_code).scalar_subquery()
                                        ),
                                        Team.name == player_full_name
                                    )
                                )
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
        """Create pseudo-teams for tennis players using raw SQL to avoid session conflicts.
        
        Uses INSERT ... WHERE NOT EXISTS to skip duplicates without breaking the session.
        All inserts are batched and committed once at the end.
        """
        from sqlalchemy import text
        
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
            await session.flush()  # flush only, don't commit yet
        
        saved = 0
        skipped = 0
        
        for player_data in players:
            try:
                player_id = player_data.get("id")
                if not player_id:
                    skipped += 1
                    continue
                
                external_id = f"bdl_{sport_code}_player_{player_id}"
                first_name = player_data.get("first_name", "")
                last_name = player_data.get("last_name", "")
                full_name = f"{first_name} {last_name}".strip()
                
                if not full_name:
                    skipped += 1
                    continue
                
                country = player_data.get("country") or player_data.get("citizenship") or ""
                abbr = last_name[:3].upper() if last_name else "TEN"
                
                # Use raw SQL with NOT EXISTS to skip duplicates without session conflicts
                result = await session.execute(
                    text(
                        """INSERT INTO teams (id, sport_id, external_id, name, abbreviation, city, is_active, created_at, updated_at)
                        SELECT gen_random_uuid(), s.id, 
                               CAST(:ext_id AS VARCHAR), CAST(:name AS VARCHAR), 
                               CAST(:abbr AS VARCHAR), CAST(:country AS VARCHAR), 
                               true, now(), now()
                        FROM sports s WHERE s.code = CAST(:sport AS VARCHAR)
                        AND NOT EXISTS (
                            SELECT 1 FROM teams t WHERE t.sport_id = s.id 
                            AND (t.external_id = CAST(:ext_id AS VARCHAR) OR t.name = CAST(:name AS VARCHAR))
                        )"""
                    ),
                    {"ext_id": external_id, "name": full_name, "abbr": abbr, "country": country, "sport": sport_code}
                )
                if result.rowcount > 0:
                    saved += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"[BallDontLie] Error creating pseudo-team for {full_name}: {e}")
                skipped += 1
        
        # Single commit at the end
        await session.commit()
        
        console.print(f"[green]💾 {sport_code} Pseudo-Teams: {saved} created, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

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
                            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            # Convert to naive datetime (remove timezone) for database
                            scheduled_at = dt.replace(tzinfo=None)
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

    async def collect_player_stats_endpoint(
        self, 
        sport_code: str, 
        season: int = None,
        max_pages: int = 500
    ) -> List[Dict]:
        """Collect player stats using /player_stats endpoint (NCAAB, NCAAF, WNBA).
        
        This is different from /stats - it uses 'season' param (not 'seasons[]')
        and is available for sports that don't have the /stats endpoint.
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("player_stats")
        if not endpoint:
            console.print(f"[yellow]⚠️ {sport_code} does not have /player_stats endpoint[/yellow]")
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]📊 Collecting {sport_code} player_stats ({season or 'all'})...[/bold blue]")
        stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        console.print(f"[green]✅ {sport_code}: {len(stats)} player_stats records collected[/green]")
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
    # TEAM STATS COLLECTION - NEW (discovered endpoints)
    # =========================================================================

    async def collect_team_stats_endpoint(
        self,
        sport_code: str,
        season: int = None,
        max_pages: int = 100,
    ) -> List[Dict]:
        """Collect team stats from sport-specific endpoints.
        
        Discovered endpoints:
        - NBA: /nba/v1/team_season_averages/general (season averages, 25 fields)
        - NCAAB: /ncaab/v1/team_stats (per-game, paginated, 17 fields)
        - WNBA: /wnba/v1/team_season_stats (season averages, 18 fields)
        - NFL: /nfl/v1/team_stats (per-game, paginated, 30 fields)
        - MLB: /mlb/v1/teams/season_stats (season stats, 36 fields)
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config or not config.get("has_team_stats_endpoint"):
            return []
        
        # Pick the correct endpoint key per sport
        endpoint = None
        for key in ["team_season_averages", "team_stats", "team_season_stats"]:
            endpoint = config["endpoints"].get(key)
            if endpoint:
                break
        
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]📊 Collecting {sport_code} team stats ({season or 'current'})...[/bold blue]")
        
        # NFL and NCAAB return per-game stats (paginated)
        # NBA, WNBA, MLB return season averages (not paginated same way)
        if sport_code in ("NFL", "NCAAB"):
            stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        else:
            response = await self._api_request(endpoint, params, sport_code)
            stats = response.get("data", []) if response else []
        
        console.print(f"[green]✅ {sport_code}: {len(stats)} team stat records collected[/green]")
        return stats
    
    async def save_team_stats_from_endpoint(
        self,
        stats_data: List[Dict],
        sport_code: str,
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Save team stats from sport-specific endpoints."""
        if not stats_data:
            return {"saved": 0, "skipped": 0}
        
        fields = TEAM_STATS_FIELDS.get(sport_code, [])
        if not fields:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for record in stats_data:
            try:
                team_data = record.get("team") or {}
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
                
                gp = record.get("gp") or record.get("games_played") or 0
                try:
                    gp = int(gp)
                except:
                    gp = 0
                
                for stat_type in fields:
                    value = record.get(stat_type)
                    if value is not None:
                        try:
                            if isinstance(value, str) and ":" in value:
                                parts = value.split(":")
                                value = float(parts[0]) + float(parts[1]) / 60
                            else:
                                value = float(value)
                        except:
                            continue
                        
                        stat = TeamStats(
                            id=uuid4(),
                            team_id=team.id,
                            stat_type=stat_type,
                            value=value,
                            games_played=gp,
                        )
                        session.add(stat)
                        saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving team stat: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Team Stats (endpoint): {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # TENNIS MATCH STATS - NEW (ATP only, WTA returns 404)
    # =========================================================================

    async def collect_tennis_match_stats(
        self,
        sport_code: str,
        season: int = None,
        max_pages: int = 500,
    ) -> List[Dict]:
        """Collect ATP match_stats. Only ATP has this endpoint; WTA returns 404."""
        config = SPORT_CONFIG.get(sport_code)
        if not config or not config.get("has_match_stats"):
            return []
        
        endpoint = config["endpoints"].get("match_stats")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]🎾 Collecting {sport_code} match stats ({season or 'current'})...[/bold blue]")
        all_stats = await self._paginated_request(endpoint, sport_code, params, max_pages)
        
        # Filter to match totals only (set_number == 0)
        match_totals = [s for s in all_stats if s.get("set_number") == 0]
        console.print(f"[green]✅ {sport_code}: {len(match_totals)} match stat totals (from {len(all_stats)} raw)[/green]")
        return match_totals
    
    async def save_tennis_match_stats(
        self,
        stats_data: List[Dict],
        sport_code: str,
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Save tennis match stats as PlayerStats records."""
        if not stats_data:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for record in stats_data:
            try:
                player_data = record.get("player") or {}
                player_api_id = player_data.get("id") or record.get("player_id")
                
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
                
                game_id = None
                match_api_id = record.get("match_id") or record.get("game_id")
                if match_api_id:
                    game_ext_id = f"bdl_{sport_code}_{match_api_id}"
                    result = await session.execute(
                        select(Game).where(Game.external_id == game_ext_id)
                    )
                    game = result.scalar_one_or_none()
                    if game:
                        game_id = game.id
                
                for stat_type in TENNIS_MATCH_STAT_FIELDS:
                    value = record.get(stat_type)
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
                logger.error(f"[BallDontLie] Error saving tennis match stat: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Match Stats: {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # TENNIS RANKINGS - NEW (ATP and WTA)
    # =========================================================================

    async def collect_tennis_rankings(
        self,
        sport_code: str,
        season: int = None,
    ) -> List[Dict]:
        """Collect ATP/WTA rankings."""
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("rankings")
        if not endpoint:
            return []
        
        params = {}
        if season:
            params["season"] = season
        
        console.print(f"[bold blue]🏆 Collecting {sport_code} rankings ({season or 'current'})...[/bold blue]")
        rankings = await self._paginated_request(endpoint, sport_code, params, max_pages=20)
        console.print(f"[green]✅ {sport_code}: {len(rankings)} rankings collected[/green]")
        return rankings
    
    async def save_tennis_rankings(
        self,
        rankings: List[Dict],
        sport_code: str,
        season: int,
        session: AsyncSession,
    ) -> Dict[str, int]:
        """Save rankings as PlayerStats with stat_types ranking_{year} and ranking_points_{year}."""
        if not rankings:
            return {"saved": 0, "skipped": 0}
        
        saved = 0
        skipped = 0
        
        for record in rankings:
            try:
                player_data = record.get("player") or {}
                player_api_id = player_data.get("id") or record.get("player_id")
                
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
                
                rank = record.get("rank") or record.get("ranking")
                points = record.get("points") or record.get("ranking_points")
                
                if rank is not None:
                    stat = PlayerStats(
                        id=uuid4(),
                        player_id=player.id,
                        stat_type=f"ranking_{season}",
                        value=float(rank),
                    )
                    session.add(stat)
                    saved += 1
                
                if points is not None:
                    stat = PlayerStats(
                        id=uuid4(),
                        player_id=player.id,
                        stat_type=f"ranking_points_{season}",
                        value=float(points),
                    )
                    session.add(stat)
                    saved += 1
            
            except Exception as e:
                logger.error(f"[BallDontLie] Error saving ranking: {e}")
                skipped += 1
        
        await session.commit()
        console.print(f"[green]💾 {sport_code} Rankings ({season}): {saved} saved, {skipped} skipped[/green]")
        return {"saved": saved, "skipped": skipped}

    # =========================================================================
    # TENNIS HISTORICAL ODDS - NEW (The Odds API, 5yr backfill)
    # =========================================================================

    async def _find_tennis_game(
        self,
        session: AsyncSession,
        sport_code: str,
        home_name: str,
        away_name: str,
        event_date: datetime,
    ) -> Optional[str]:
        """Match an Odds API event to a BDL game by player names + date."""
        result = await session.execute(
            text('''
                SELECT g.id FROM games g
                JOIN sports s ON g.sport_id = s.id
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE s.code = :sport
                AND (
                    (ht.name ILIKE :home AND at.name ILIKE :away) OR
                    (ht.name ILIKE :away AND at.name ILIKE :home)
                )
                AND g.scheduled_at BETWEEN :d1 AND :d2
                AND ht.name NOT LIKE '%%(Games)%%'
                LIMIT 1
            '''),
            {
                'sport': sport_code,
                'home': f'%{home_name}%',
                'away': f'%{away_name}%',
                'd1': event_date - timedelta(days=2),
                'd2': event_date + timedelta(days=2),
            }
        )
        row = result.fetchone()
        return row[0] if row else None

    async def collect_tennis_historical_odds(
        self,
        sport_code: str,
        years_back: int = 5,
    ) -> Dict[str, int]:
        """Collect historical tennis odds from The Odds API.
        
        Uses /v4/historical/sports/{tournament}/odds endpoint.
        Data available from 2022 onwards. Matches events to BDL games by player name + date.
        """
        from app.core.database import db_manager
        
        if not self.odds_api_key:
            console.print(f"[yellow]⚠️ No Odds API key - skipping tennis odds[/yellow]")
            return {"saved": 0, "no_game": 0}
        
        tournaments = ATP_TOURNAMENT_DATES if sport_code == "ATP" else WTA_TOURNAMENT_DATES
        current_year = datetime.now().year
        start_year = max(2022, current_year - years_back)
        
        grand_saved = 0
        grand_no_game = 0
        
        console.print(f"\n[bold magenta]🎾 Collecting {sport_code} historical odds ({start_year}-{current_year})[/bold magenta]")
        
        odds_client = await self._get_odds_client()
        
        for year in range(start_year, current_year + 1):
            for tourney_key, mm_start, mm_end in tournaments:
                try:
                    start = datetime.strptime(f"{year}-{mm_start}", "%Y-%m-%d")
                    end = datetime.strptime(f"{year}-{mm_end}", "%Y-%m-%d")
                except:
                    continue
                
                if start > datetime.now():
                    continue
                
                console.print(f"[cyan]  {tourney_key} {year}...[/cyan]")
                seen_events = set()
                current = start
                
                while current <= end:
                    date_str = current.strftime("%Y-%m-%dT12:00:00Z")
                    try:
                        r = await odds_client.get(
                            f"/v4/historical/sports/{tourney_key}/odds",
                            params={
                                "apiKey": self.odds_api_key,
                                "regions": "us",
                                "markets": "h2h",
                                "oddsFormat": "american",
                                "date": date_str,
                            }
                        )
                        
                        if r.status_code != 200:
                            current += timedelta(days=1)
                            continue
                        
                        data = r.json().get("data", [])
                        new_events = [e for e in data if e["id"] not in seen_events]
                        
                        if new_events:
                            async with db_manager.session() as session:
                                for event in new_events:
                                    seen_events.add(event["id"])
                                    home = event.get("home_team", "")
                                    away = event.get("away_team", "")
                                    commence = datetime.fromisoformat(
                                        event["commence_time"].replace("Z", "+00:00")
                                    ).replace(tzinfo=None)
                                    
                                    game_id = await self._find_tennis_game(
                                        session, sport_code, home, away, commence
                                    )
                                    if not game_id:
                                        grand_no_game += 1
                                        continue
                                    
                                    for bm in event.get("bookmakers", []):
                                        bm_key = bm.get("key", "")
                                        for market in bm.get("markets", []):
                                            if market["key"] != "h2h":
                                                continue
                                            outcomes = {
                                                o["name"]: o["price"]
                                                for o in market.get("outcomes", [])
                                            }
                                            home_odds = outcomes.get(home)
                                            away_odds = outcomes.get(away)
                                            if home_odds and away_odds:
                                                odds_record = Odds(
                                                    id=uuid4(),
                                                    game_id=game_id,
                                                    sportsbook_key=bm_key,
                                                    bet_type="h2h",
                                                    home_odds=int(home_odds),
                                                    away_odds=int(away_odds),
                                                    recorded_at=commence,
                                                )
                                                session.add(odds_record)
                                                grand_saved += 1
                                
                                await session.commit()
                    
                    except Exception as e:
                        logger.debug(f"[Tennis Odds] Error on {date_str}: {e}")
                    
                    current += timedelta(days=1)
                    await asyncio.sleep(0.05)
                
                console.print(f"    Saved: {grand_saved} | No match: {grand_no_game}")
        
        console.print(f"[green]💾 {sport_code} Historical Odds: {grand_saved} saved, {grand_no_game} unmatched[/green]")
        return {"saved": grand_saved, "no_game": grand_no_game}

    # =========================================================================
    # INJURIES COLLECTION - FIXED
    # =========================================================================
    
    @staticmethod
    def _parse_return_date(date_str: str) -> Optional[datetime]:
        """Parse expected return date from injury data.
        
        Handles formats like "May 1", "Oct 12", "2026-01-15", etc.
        For month-day strings, infers year: months before June → next year, else current year.
        """
        if not date_str:
            return None
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            pass
        
        # Try "Month Day" format (e.g., "May 1", "Oct 12")
        for fmt in ("%B %d", "%b %d"):
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                now = datetime.now()
                year = now.year + 1 if dt.month < 6 else now.year
                return dt.replace(year=year)
            except:
                continue
        
        return None

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
        """Save injuries to database - FIXED to handle nested teams array + WNBA date parsing."""
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
                expected_return_str = injury_data.get("expected_return") or injury_data.get("return_date") or ""
                
                # Build status_detail from comment + expected_return (truncate for varchar(200))
                status_detail = comment
                if expected_return_str:
                    status_detail = f"{comment} | Expected: {expected_return_str}".strip(" |")
                status_detail = str(status_detail)[:200] if status_detail else None
                
                # Parse expected_return date (handles "May 1", "Oct 12" WNBA format)
                expected_return = self._parse_return_date(expected_return_str) if expected_return_str else None
                
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
                        status_detail=status_detail,
                        source="balldontlie",
                        external_id=external_id,
                    )
                    # Set expected_return if the model supports it
                    if hasattr(injury, 'expected_return') and expected_return:
                        injury.expected_return = expected_return.date() if hasattr(expected_return, 'date') else expected_return
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
        
        The unified /v2/odds endpoint uses dates[] parameter for all sports.
        """
        config = SPORT_CONFIG.get(sport_code)
        if not config:
            return []
        
        endpoint = config["endpoints"].get("odds")
        if not endpoint:
            return []
        
        params = {}
        
        # Unified /v2/odds endpoint uses dates[] for ALL sports
        if game_ids:
            for gid in game_ids[:10]:
                params["game_ids[]"] = gid
        elif dates:
            for d in dates[:10]:
                params["dates[]"] = d
        else:
            # Default to recent dates (last 7 days)
            today = datetime.now()
            all_odds = []
            for i in range(7):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                params = {"dates[]": date}
                console.print(f"[bold blue]💰 Collecting {sport_code} odds ({date})...[/bold blue]")
                try:
                    odds = await self._paginated_request(endpoint, sport_code, params, max_pages=2)
                    if odds:
                        all_odds.extend(odds)
                        console.print(f"[green]✅ {sport_code} {date}: {len(odds)} odds[/green]")
                except Exception as e:
                    logger.warning(f"[BallDontLie] No odds for {sport_code} {date}: {e}")
            
            console.print(f"[green]✅ {sport_code}: {len(all_odds)} total odds records collected[/green]")
            return all_odds
        
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
        has_team_stats = config.get("has_team_stats_endpoint", False)
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
            "tennis_match_stats": {"saved": 0},
            "tennis_rankings": {"saved": 0},
            "tennis_odds": {"saved": 0},
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
                has_player_stats = config.get("has_player_stats_endpoint", False)
                
                if is_tennis:
                    # ATP: match_stats + rankings; WTA: rankings only
                    if config.get("has_match_stats"):
                        console.print(f"\n[bold]🎾 Step 4a: Collecting {sport_code} match stats...[/bold]")
                        for year in range(start_year, current_year + 1):
                            match_stats = await self.collect_tennis_match_stats(sport_code, season=year)
                            ms_results = await self.save_tennis_match_stats(match_stats, sport_code, session)
                            results["tennis_match_stats"]["saved"] += ms_results.get("saved", 0)
                    
                    console.print(f"\n[bold]🏆 Step 4b: Collecting {sport_code} rankings...[/bold]")
                    for year in range(start_year, current_year + 1):
                        rankings = await self.collect_tennis_rankings(sport_code, season=year)
                        rk_results = await self.save_tennis_rankings(rankings, sport_code, year, session)
                        results["tennis_rankings"]["saved"] += rk_results.get("saved", 0)
                
                elif has_stats and not is_tennis:
                    # Use /stats endpoint (NBA, NFL, MLB)
                    for year in range(start_year, current_year + 1):
                        stats = await self.collect_player_stats(sport_code, season=year)
                        stat_results = await self.save_player_stats(stats, sport_code, session)
                        results["player_stats"]["saved"] += stat_results.get("saved", 0)
                elif has_player_stats and not is_tennis:
                    # Use /player_stats endpoint (NCAAB, NCAAF, WNBA)
                    for year in range(start_year, current_year + 1):
                        stats = await self.collect_player_stats_endpoint(sport_code, season=year)
                        stat_results = await self.save_player_stats(stats, sport_code, session)
                        results["player_stats"]["saved"] += stat_results.get("saved", 0)
                elif not is_tennis and not has_stats and not has_player_stats:
                    # Use box_scores as last fallback (for NHL)
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
                    console.print(f"\n[bold]📊 Step 6: Collecting {sport_code} team stats...[/bold]")
                    if has_team_stats:
                        # Use discovered team stats endpoints (NBA/NCAAB/WNBA/NFL/MLB)
                        for year in range(start_year, current_year + 1):
                            ts_data = await self.collect_team_stats_endpoint(sport_code, season=year)
                            ts_results = await self.save_team_stats_from_endpoint(ts_data, sport_code, session)
                            results["team_stats"]["saved"] += ts_results.get("saved", 0)
                    else:
                        # Fallback to standings (NHL, etc.)
                        standings = await self.collect_standings(sport_code, season=current_year)
                        team_stat_results = await self.save_team_stats(standings, sport_code, session)
                        results["team_stats"] = team_stat_results
                
                # 7. Odds
                console.print(f"\n[bold]📊 Step 7: Collecting {sport_code} odds...[/bold]")
                if is_tennis:
                    # Tennis odds from The Odds API historical endpoint (5yr backfill)
                    tennis_odds_results = await self.collect_tennis_historical_odds(
                        sport_code, years_back=min(years, 5)
                    )
                    results["tennis_odds"] = tennis_odds_results
                else:
                    # Team sport odds from BDL /v2/odds
                    today = datetime.now()
                    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
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
            results["odds"].get("saved", 0) +
            results["tennis_match_stats"].get("saved", 0) +
            results["tennis_rankings"].get("saved", 0) +
            results["tennis_odds"].get("saved", 0)
        )
        
        console.print(f"\n[bold green]✅ {sport_code} Collection Complete[/bold green]")
        console.print(f"  Teams: {results['teams'].get('saved', 0)} saved, {results['teams'].get('updated', 0)} updated")
        console.print(f"  Players: {results['players'].get('saved', 0)} saved, {results['players'].get('updated', 0)} updated")
        console.print(f"  Games: {results['games'].get('saved', 0)} saved, {results['games'].get('updated', 0)} updated")
        console.print(f"  Player Stats: {results['player_stats'].get('saved', 0)} saved")
        if is_tennis:
            console.print(f"  Match Stats: {results['tennis_match_stats'].get('saved', 0)} saved")
            console.print(f"  Rankings: {results['tennis_rankings'].get('saved', 0)} saved")
            console.print(f"  Tennis Odds: {results['tennis_odds'].get('saved', 0)} saved")
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
                    sport_results.get("tennis_match_stats", {}).get("saved", 0),
                    sport_results.get("tennis_rankings", {}).get("saved", 0),
                    sport_results.get("tennis_odds", {}).get("saved", 0),
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