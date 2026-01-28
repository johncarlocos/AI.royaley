"""
ROYALEY - Sportsipy Data Collector
Phase 1: Data Collection Services

Collects comprehensive sports data from Sports-Reference sites via sportsipy package.
- Baseball-Reference (MLB)
- Basketball-Reference (NBA, NCAAB)
- Pro-Football-Reference (NFL)
- College Football Reference (NCAAF)  
- Hockey-Reference (NHL)

Data Sources:
- https://www.baseball-reference.com/
- https://www.basketball-reference.com/
- https://www.pro-football-reference.com/
- https://www.sports-reference.com/cfb/
- https://www.hockey-reference.com/

FREE data - uses sportsipy Python package!

pip install sportsipy

Tables Filled:
- sports - Sport definitions
- teams - Team info with conference/division
- players - Player info
- seasons - Season definitions
- games - Game records with scores
- team_stats - Team statistics by season
- player_stats - Player statistics
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import time
import random

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus, Player, PlayerStats, TeamStats, Season
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SPORT CONFIGURATIONS
# =============================================================================

SPORT_CONFIGS = {
    "MLB": {
        "code": "MLB",
        "name": "Major League Baseball",
        "module": "sportsipy.mlb",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 3,  # March
        "season_end_month": 10,   # October
    },
    "NBA": {
        "code": "NBA", 
        "name": "National Basketball Association",
        "module": "sportsipy.nba",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 10,  # October
        "season_end_month": 6,     # June
    },
    "NFL": {
        "code": "NFL",
        "name": "National Football League", 
        "module": "sportsipy.nfl",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 9,   # September
        "season_end_month": 2,     # February
    },
    "NHL": {
        "code": "NHL",
        "name": "National Hockey League",
        "module": "sportsipy.nhl",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 10,  # October
        "season_end_month": 6,     # June
    },
    "NCAAF": {
        "code": "NCAAF",
        "name": "NCAA Football",
        "module": "sportsipy.ncaaf",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 8,   # August
        "season_end_month": 1,     # January
    },
    "NCAAB": {
        "code": "NCAAB",
        "name": "NCAA Basketball",
        "module": "sportsipy.ncaab",
        "teams_class": "Teams",
        "schedule_class": "Schedule",
        "boxscore_class": "Boxscore",
        "roster_class": "Roster",
        "season_start_month": 11,  # November
        "season_end_month": 4,     # April
    },
}

# Team stat types by sport
TEAM_STAT_TYPES = {
    "MLB": [
        "wins", "losses", "win_percentage", "runs_scored", "runs_allowed",
        "batting_average", "on_base_percentage", "slugging_percentage", 
        "on_base_plus_slugging", "earned_run_average", "strikeouts",
        "walks", "home_runs", "stolen_bases", "fielding_percentage"
    ],
    "NBA": [
        "wins", "losses", "win_percentage", "points_per_game", "points_allowed_per_game",
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
        "rebounds_per_game", "assists_per_game", "steals_per_game", "blocks_per_game",
        "turnovers_per_game", "offensive_rating", "defensive_rating", "net_rating"
    ],
    "NFL": [
        "wins", "losses", "ties", "win_percentage", "points_for", "points_against",
        "yards_per_game", "yards_allowed_per_game", "passing_yards", "rushing_yards",
        "turnovers", "takeaways", "sacks", "first_downs", "third_down_percentage"
    ],
    "NHL": [
        "wins", "losses", "overtime_losses", "points", "goals_for", "goals_against",
        "goal_differential", "shots_on_goal", "power_play_percentage",
        "penalty_kill_percentage", "save_percentage", "goals_per_game"
    ],
    "NCAAF": [
        "wins", "losses", "win_percentage", "points_for", "points_against",
        "yards_per_game", "yards_allowed_per_game", "passing_yards", "rushing_yards",
        "turnovers", "sacks"
    ],
    "NCAAB": [
        "wins", "losses", "win_percentage", "points_per_game", "points_allowed_per_game",
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
        "rebounds_per_game", "assists_per_game", "turnovers_per_game"
    ],
}

# Player stat types by sport
PLAYER_STAT_TYPES = {
    "MLB": {
        "batting": ["games", "at_bats", "runs", "hits", "doubles", "triples", "home_runs",
                   "runs_batted_in", "stolen_bases", "batting_average", "on_base_percentage",
                   "slugging_percentage", "on_base_plus_slugging", "walks", "strikeouts"],
        "pitching": ["games", "games_started", "wins", "losses", "saves", "innings_pitched",
                    "hits_allowed", "runs_allowed", "earned_runs", "earned_run_average",
                    "strikeouts", "walks", "whip", "batting_average_against"]
    },
    "NBA": {
        "stats": ["games_played", "minutes_played", "points", "field_goals", "field_goal_percentage",
                 "three_pointers", "three_point_percentage", "free_throws", "free_throw_percentage",
                 "rebounds", "assists", "steals", "blocks", "turnovers", "personal_fouls",
                 "player_efficiency_rating", "true_shooting_percentage"]
    },
    "NFL": {
        "passing": ["games", "completions", "attempts", "passing_yards", "passing_touchdowns",
                   "interceptions", "passer_rating", "yards_per_attempt"],
        "rushing": ["games", "rush_attempts", "rush_yards", "rush_touchdowns", "yards_per_carry"],
        "receiving": ["games", "receptions", "receiving_yards", "receiving_touchdowns",
                     "yards_per_reception", "targets"]
    },
    "NHL": {
        "skater": ["games_played", "goals", "assists", "points", "plus_minus", "penalty_minutes",
                  "power_play_goals", "power_play_assists", "shots_on_goal", "shooting_percentage",
                  "time_on_ice", "blocks", "hits"],
        "goalie": ["games_played", "wins", "losses", "overtime_losses", "saves", "goals_against",
                  "save_percentage", "goals_against_average", "shutouts"]
    },
    "NCAAF": {
        "passing": ["games", "completions", "attempts", "passing_yards", "passing_touchdowns", "interceptions"],
        "rushing": ["games", "rush_attempts", "rush_yards", "rush_touchdowns"],
        "receiving": ["games", "receptions", "receiving_yards", "receiving_touchdowns"]
    },
    "NCAAB": {
        "stats": ["games_played", "minutes_played", "points", "field_goal_percentage",
                 "three_point_percentage", "free_throw_percentage", "rebounds", "assists",
                 "steals", "blocks", "turnovers"]
    },
}


# =============================================================================
# SPORTSIPY COLLECTOR CLASS
# =============================================================================

class SportsipyCollector(BaseCollector):
    """Collector for Sports-Reference data via sportsipy package."""
    
    name = "sportsipy"
    
    def __init__(self):
        super().__init__(
            name="sportsipy",
            base_url="https://www.sports-reference.com",
            rate_limit=0.5,  # 2 requests per second max (be polite)
        )
        self._sportsipy_available = None
        self._modules = {}
    
    def _check_sportsipy(self) -> bool:
        """Check if sportsipy is installed."""
        if self._sportsipy_available is None:
            try:
                import sportsipy
                self._sportsipy_available = True
                logger.info("[Sportsipy] Package available")
            except ImportError:
                self._sportsipy_available = False
                logger.warning("[Sportsipy] Package not installed. Install with: pip install sportsipy")
        return self._sportsipy_available
    
    def _get_module(self, sport_code: str):
        """Get sportsipy module for a sport."""
        if sport_code not in self._modules:
            config = SPORT_CONFIGS.get(sport_code)
            if not config:
                return None
            
            try:
                module_name = config["module"]
                # Import the specific sport module
                if sport_code == "MLB":
                    from sportsipy.mlb import teams as mlb_teams
                    self._modules[sport_code] = mlb_teams
                elif sport_code == "NBA":
                    from sportsipy.nba import teams as nba_teams
                    self._modules[sport_code] = nba_teams
                elif sport_code == "NFL":
                    from sportsipy.nfl import teams as nfl_teams
                    self._modules[sport_code] = nfl_teams
                elif sport_code == "NHL":
                    from sportsipy.nhl import teams as nhl_teams
                    self._modules[sport_code] = nhl_teams
                elif sport_code == "NCAAF":
                    from sportsipy.ncaaf import teams as ncaaf_teams
                    self._modules[sport_code] = ncaaf_teams
                elif sport_code == "NCAAB":
                    from sportsipy.ncaab import teams as ncaab_teams
                    self._modules[sport_code] = ncaab_teams
                    
                logger.info(f"[Sportsipy] Loaded module for {sport_code}")
            except ImportError as e:
                logger.error(f"[Sportsipy] Failed to import module for {sport_code}: {e}")
                return None
        
        return self._modules.get(sport_code)

    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if hasattr(data, 'success'):
            return data.success
        if isinstance(data, dict):
            return bool(data)
        return False

    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sports: List[str] = None,
        years_back: int = 10,
        collect_type: str = "all",
    ) -> CollectorResult:
        """
        Collect data from Sports-Reference via sportsipy.
        
        Args:
            sports: List of sport codes (MLB, NBA, NFL, NHL, NCAAF, NCAAB)
            years_back: Number of years to collect (default: 10)
            collect_type: Type of data to collect:
                - "all": All data types
                - "teams": Teams only
                - "players": Players and rosters only
                - "games": Games/schedules only
                - "stats": Team and player stats only
        
        Returns:
            CollectorResult with collected data
        """
        if not self._check_sportsipy():
            return CollectorResult(
                success=False,
                data=None,
                records=0,
                error="sportsipy package not installed. Install with: pip install sportsipy"
            )
        
        if sports is None:
            sports = list(SPORT_CONFIGS.keys())
        
        current_year = datetime.now().year
        
        data = {
            "teams": [],
            "players": [],
            "games": [],
            "team_stats": [],
            "player_stats": [],
            "seasons": [],
        }
        total_records = 0
        errors = []
        
        for sport_code in sports:
            if sport_code not in SPORT_CONFIGS:
                logger.warning(f"[Sportsipy] Unknown sport: {sport_code}")
                continue
            
            config = SPORT_CONFIGS[sport_code]
            logger.info(f"[Sportsipy] Collecting {sport_code} data ({years_back} years)...")
            
            try:
                # Determine season range
                start_year = current_year - years_back
                end_year = current_year
                
                for year in range(start_year, end_year + 1):
                    logger.info(f"[Sportsipy] {sport_code} {year}...")
                    
                    try:
                        # Rate limiting - be polite to Sports-Reference
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                        
                        # Collect teams for this season
                        if collect_type in ["all", "teams"]:
                            teams = await self._collect_teams(sport_code, year)
                            data["teams"].extend(teams)
                            total_records += len(teams)
                            logger.info(f"[Sportsipy] {sport_code} {year}: {len(teams)} teams")
                        
                        # Collect team stats
                        if collect_type in ["all", "stats"]:
                            team_stats = await self._collect_team_stats(sport_code, year)
                            data["team_stats"].extend(team_stats)
                            total_records += len(team_stats)
                            logger.info(f"[Sportsipy] {sport_code} {year}: {len(team_stats)} team stats")
                        
                        # Collect games/schedules
                        if collect_type in ["all", "games"]:
                            games = await self._collect_games(sport_code, year)
                            data["games"].extend(games)
                            total_records += len(games)
                            logger.info(f"[Sportsipy] {sport_code} {year}: {len(games)} games")
                        
                        # Collect players/rosters (less frequently - expensive)
                        if collect_type in ["all", "players"] and year >= current_year - 2:
                            players = await self._collect_players(sport_code, year)
                            data["players"].extend(players)
                            total_records += len(players)
                            logger.info(f"[Sportsipy] {sport_code} {year}: {len(players)} players")
                        
                        # Add season record
                        season_data = {
                            "sport_code": sport_code,
                            "year": year,
                            "name": f"{year}-{str(year + 1)[-2:]}" if config["season_start_month"] >= 8 else str(year),
                        }
                        data["seasons"].append(season_data)
                        
                    except Exception as e:
                        logger.warning(f"[Sportsipy] Error collecting {sport_code} {year}: {e}")
                        errors.append(f"{sport_code} {year}: {str(e)[:50]}")
                        continue
                
            except Exception as e:
                logger.error(f"[Sportsipy] Error collecting {sport_code}: {e}")
                errors.append(f"{sport_code}: {str(e)[:100]}")
        
        logger.info(f"[Sportsipy] Total records collected: {total_records}")
        
        return CollectorResult(
            success=total_records > 0,
            data=data,
            records=total_records,
            error="; ".join(errors[:5]) if errors else None
        )

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self, sport_code: str, year: int) -> List[Dict[str, Any]]:
        """Collect teams for a sport and season."""
        teams = []
        
        try:
            module = self._get_module(sport_code)
            if not module:
                return teams
            
            # Run in thread to not block async
            loop = asyncio.get_event_loop()
            teams_obj = await loop.run_in_executor(
                None, 
                lambda: module.Teams(year=str(year))
            )
            
            for team in teams_obj:
                try:
                    team_data = {
                        "sport_code": sport_code,
                        "year": year,
                        "external_id": f"{sport_code}_{team.abbreviation}",
                        "abbreviation": team.abbreviation,
                        "name": getattr(team, 'name', team.abbreviation),
                    }
                    
                    # Try to get additional info
                    if hasattr(team, 'conference'):
                        team_data["conference"] = team.conference
                    if hasattr(team, 'division'):
                        team_data["division"] = team.division
                    
                    teams.append(team_data)
                    
                except Exception as e:
                    logger.debug(f"[Sportsipy] Error parsing team: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"[Sportsipy] Error collecting {sport_code} teams {year}: {e}")
        
        return teams

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, sport_code: str, year: int) -> List[Dict[str, Any]]:
        """Collect team statistics for a sport and season."""
        stats = []
        
        try:
            module = self._get_module(sport_code)
            if not module:
                return stats
            
            # Rate limiting
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            loop = asyncio.get_event_loop()
            teams_obj = await loop.run_in_executor(
                None,
                lambda: module.Teams(year=str(year))
            )
            
            stat_types = TEAM_STAT_TYPES.get(sport_code, [])
            
            for team in teams_obj:
                team_abbr = team.abbreviation
                
                for stat_type in stat_types:
                    try:
                        # Try to get the stat value
                        value = getattr(team, stat_type, None)
                        
                        if value is not None:
                            # Convert to float if possible
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                continue
                            
                            stat_data = {
                                "sport_code": sport_code,
                                "team_abbr": team_abbr,
                                "year": year,
                                "stat_type": f"sportsref_{stat_type}",
                                "value": value,
                            }
                            stats.append(stat_data)
                            
                    except Exception as e:
                        logger.debug(f"[Sportsipy] Error getting {stat_type} for {team_abbr}: {e}")
                        continue
                
                # Rate limiting between teams
                await asyncio.sleep(0.1)
                        
        except Exception as e:
            logger.warning(f"[Sportsipy] Error collecting {sport_code} team stats {year}: {e}")
        
        return stats

    # =========================================================================
    # GAMES/SCHEDULE COLLECTION
    # =========================================================================
    
    async def _collect_games(self, sport_code: str, year: int) -> List[Dict[str, Any]]:
        """Collect games/schedules for a sport and season."""
        games = []
        seen_game_ids = set()
        
        try:
            module = self._get_module(sport_code)
            if not module:
                return games
            
            # Rate limiting
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            loop = asyncio.get_event_loop()
            teams_obj = await loop.run_in_executor(
                None,
                lambda: module.Teams(year=str(year))
            )
            
            for team in teams_obj:
                team_abbr = team.abbreviation
                
                try:
                    # Get team schedule
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                    
                    schedule = await loop.run_in_executor(
                        None,
                        lambda t=team: t.schedule
                    )
                    
                    if schedule is None:
                        continue
                    
                    for game in schedule:
                        try:
                            # Create unique game ID
                            game_date = getattr(game, 'date', None) or getattr(game, 'datetime', None)
                            if game_date is None:
                                continue
                            
                            # Get opponent
                            opponent = getattr(game, 'opponent_abbr', None) or getattr(game, 'opponent_name', '')
                            
                            # Determine home/away
                            location = getattr(game, 'location', '')
                            is_home = location != '@' if location else True
                            
                            if is_home:
                                home_team = team_abbr
                                away_team = opponent
                            else:
                                home_team = opponent
                                away_team = team_abbr
                            
                            # Create unique game ID to avoid duplicates
                            if isinstance(game_date, datetime):
                                date_str = game_date.strftime("%Y%m%d")
                            else:
                                date_str = str(game_date).replace("-", "")[:8]
                            
                            # Sort teams alphabetically for consistent ID
                            teams_sorted = sorted([home_team, away_team])
                            game_id = f"{sport_code}_{date_str}_{teams_sorted[0]}_{teams_sorted[1]}"
                            
                            if game_id in seen_game_ids:
                                continue
                            seen_game_ids.add(game_id)
                            
                            # Get scores
                            home_score = None
                            away_score = None
                            
                            if is_home:
                                home_score = getattr(game, 'points_for', None) or getattr(game, 'runs', None) or getattr(game, 'goals', None)
                                away_score = getattr(game, 'points_against', None) or getattr(game, 'runs_against', None) or getattr(game, 'goals_against', None)
                            else:
                                away_score = getattr(game, 'points_for', None) or getattr(game, 'runs', None) or getattr(game, 'goals', None)
                                home_score = getattr(game, 'points_against', None) or getattr(game, 'runs_against', None) or getattr(game, 'goals_against', None)
                            
                            # Determine game status
                            status = "final"
                            if home_score is None or away_score is None:
                                status = "scheduled"
                            
                            game_data = {
                                "sport_code": sport_code,
                                "external_id": game_id,
                                "year": year,
                                "home_team_abbr": home_team,
                                "away_team_abbr": away_team,
                                "scheduled_at": game_date if isinstance(game_date, datetime) else datetime.strptime(str(game_date)[:10], "%Y-%m-%d"),
                                "home_score": int(home_score) if home_score is not None else None,
                                "away_score": int(away_score) if away_score is not None else None,
                                "status": status,
                            }
                            games.append(game_data)
                            
                        except Exception as e:
                            logger.debug(f"[Sportsipy] Error parsing game: {e}")
                            continue
                    
                except Exception as e:
                    logger.debug(f"[Sportsipy] Error getting schedule for {team_abbr}: {e}")
                    continue
                        
        except Exception as e:
            logger.warning(f"[Sportsipy] Error collecting {sport_code} games {year}: {e}")
        
        return games

    # =========================================================================
    # PLAYERS/ROSTER COLLECTION
    # =========================================================================
    
    async def _collect_players(self, sport_code: str, year: int) -> List[Dict[str, Any]]:
        """Collect players/rosters for a sport and season."""
        players = []
        seen_player_ids = set()
        
        try:
            module = self._get_module(sport_code)
            if not module:
                return players
            
            # Rate limiting
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            loop = asyncio.get_event_loop()
            teams_obj = await loop.run_in_executor(
                None,
                lambda: module.Teams(year=str(year))
            )
            
            for team in teams_obj:
                team_abbr = team.abbreviation
                
                try:
                    # Get team roster - this is expensive
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    
                    roster = await loop.run_in_executor(
                        None,
                        lambda t=team: t.roster
                    )
                    
                    if roster is None:
                        continue
                    
                    for player in roster.players:
                        try:
                            player_id = getattr(player, 'player_id', None) or getattr(player, 'name', '')
                            
                            if not player_id or player_id in seen_player_ids:
                                continue
                            seen_player_ids.add(player_id)
                            
                            player_data = {
                                "sport_code": sport_code,
                                "external_id": f"sportsref_{sport_code}_{player_id}",
                                "team_abbr": team_abbr,
                                "year": year,
                                "name": getattr(player, 'name', player_id),
                                "position": getattr(player, 'position', None),
                                "jersey_number": getattr(player, 'jersey_number', None),
                            }
                            
                            # Try to get additional info
                            if hasattr(player, 'height'):
                                player_data["height"] = str(player.height)
                            if hasattr(player, 'weight'):
                                try:
                                    player_data["weight"] = int(player.weight)
                                except:
                                    pass
                            
                            players.append(player_data)
                            
                        except Exception as e:
                            logger.debug(f"[Sportsipy] Error parsing player: {e}")
                            continue
                    
                except Exception as e:
                    logger.debug(f"[Sportsipy] Error getting roster for {team_abbr}: {e}")
                    continue
                        
        except Exception as e:
            logger.warning(f"[Sportsipy] Error collecting {sport_code} players {year}: {e}")
        
        return players

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Save seasons first
            if data.get("seasons"):
                saved = await self._save_seasons(session, data["seasons"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} seasons")
            
            # Save teams
            if data.get("teams"):
                saved = await self._save_teams(session, data["teams"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} teams")
            
            # Save players
            if data.get("players"):
                saved = await self._save_players(session, data["players"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} players")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(session, data["games"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} games")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self._save_team_stats(session, data["team_stats"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} team stats")
            
            # Save player stats
            if data.get("player_stats"):
                saved = await self._save_player_stats(session, data["player_stats"])
                total_saved += saved
                logger.info(f"[Sportsipy] Saved {saved} player stats")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[Sportsipy] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _get_or_create_sport(self, session: AsyncSession, sport_code: str) -> Sport:
        """Get or create sport record."""
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            config = SPORT_CONFIGS.get(sport_code, {})
            sport = Sport(
                code=sport_code,
                name=config.get("name", sport_code),
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        return sport
    
    async def _get_or_create_season(
        self, 
        session: AsyncSession, 
        sport_id: UUID, 
        year: int,
        sport_code: str
    ) -> Season:
        """Get or create season record."""
        result = await session.execute(
            select(Season).where(
                and_(
                    Season.sport_id == sport_id,
                    Season.year == year
                )
            )
        )
        season = result.scalar_one_or_none()
        
        if not season:
            config = SPORT_CONFIGS.get(sport_code, {})
            start_month = config.get("season_start_month", 1)
            end_month = config.get("season_end_month", 12)
            
            # Determine season dates
            if start_month >= 8:  # Fall sports
                start_date = date(year, start_month, 1)
                end_date = date(year + 1, end_month, 28)
                name = f"{year}-{str(year + 1)[-2:]}"
            else:
                start_date = date(year, start_month, 1)
                end_date = date(year, end_month, 28)
                name = str(year)
            
            season = Season(
                sport_id=sport_id,
                year=year,
                name=name,
                start_date=start_date,
                end_date=end_date,
                is_current=(year == datetime.now().year)
            )
            session.add(season)
            await session.flush()
        
        return season
    
    async def _save_seasons(self, session: AsyncSession, seasons: List[Dict]) -> int:
        """Save season records."""
        saved = 0
        
        for season_data in seasons:
            try:
                sport = await self._get_or_create_sport(session, season_data["sport_code"])
                await self._get_or_create_season(
                    session, sport.id, season_data["year"], season_data["sport_code"]
                )
                saved += 1
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving season: {e}")
        
        await session.flush()
        return saved
    
    async def _save_teams(self, session: AsyncSession, teams: List[Dict]) -> int:
        """Save team records."""
        saved = 0
        
        for team_data in teams:
            try:
                sport = await self._get_or_create_sport(session, team_data["sport_code"])
                external_id = team_data["external_id"]
                
                # Check for existing team
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
                    # Update if needed
                    if team_data.get("conference"):
                        existing.conference = team_data["conference"]
                    if team_data.get("division"):
                        existing.division = team_data["division"]
                else:
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_data["name"],
                        abbreviation=team_data["abbreviation"],
                        conference=team_data.get("conference"),
                        division=team_data.get("division"),
                        is_active=True
                    )
                    session.add(team)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving team: {e}")
        
        await session.flush()
        return saved
    
    async def _save_players(self, session: AsyncSession, players: List[Dict]) -> int:
        """Save player records."""
        saved = 0
        
        for player_data in players:
            try:
                external_id = player_data["external_id"]
                
                # Check for existing player
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update if needed
                    if player_data.get("position"):
                        existing.position = player_data["position"]
                    if player_data.get("jersey_number"):
                        existing.jersey_number = player_data["jersey_number"]
                else:
                    # Find team
                    team_id = None
                    if player_data.get("team_abbr"):
                        sport = await self._get_or_create_sport(session, player_data["sport_code"])
                        team_ext_id = f"{player_data['sport_code']}_{player_data['team_abbr']}"
                        result = await session.execute(
                            select(Team).where(
                                and_(
                                    Team.sport_id == sport.id,
                                    Team.external_id == team_ext_id
                                )
                            )
                        )
                        team = result.scalar_one_or_none()
                        if team:
                            team_id = team.id
                    
                    player = Player(
                        external_id=external_id,
                        team_id=team_id,
                        name=player_data["name"],
                        position=player_data.get("position"),
                        jersey_number=player_data.get("jersey_number"),
                        height=player_data.get("height"),
                        weight=player_data.get("weight"),
                        is_active=True
                    )
                    session.add(player)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving player: {e}")
        
        await session.flush()
        return saved
    
    async def _save_games(self, session: AsyncSession, games: List[Dict]) -> int:
        """Save game records."""
        saved = 0
        
        for game_data in games:
            try:
                sport = await self._get_or_create_sport(session, game_data["sport_code"])
                season = await self._get_or_create_season(
                    session, sport.id, game_data["year"], game_data["sport_code"]
                )
                
                external_id = game_data["external_id"]
                
                # Find teams
                home_ext_id = f"{game_data['sport_code']}_{game_data['home_team_abbr']}"
                away_ext_id = f"{game_data['sport_code']}_{game_data['away_team_abbr']}"
                
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == home_ext_id
                        )
                    )
                )
                home_team = result.scalar_one_or_none()
                
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == away_ext_id
                        )
                    )
                )
                away_team = result.scalar_one_or_none()
                
                if not home_team or not away_team:
                    continue
                
                # Check for existing game
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update scores if available
                    if game_data.get("home_score") is not None:
                        existing.home_score = game_data["home_score"]
                    if game_data.get("away_score") is not None:
                        existing.away_score = game_data["away_score"]
                    if game_data.get("status") == "final":
                        existing.status = GameStatus.FINAL
                else:
                    status = GameStatus.FINAL if game_data.get("status") == "final" else GameStatus.SCHEDULED
                    
                    game = Game(
                        sport_id=sport.id,
                        season_id=season.id,
                        external_id=external_id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=game_data["scheduled_at"],
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                        status=status
                    )
                    session.add(game)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving game: {e}")
        
        await session.flush()
        return saved
    
    async def _save_team_stats(self, session: AsyncSession, stats: List[Dict]) -> int:
        """Save team statistics."""
        saved = 0
        
        for stat_data in stats:
            try:
                sport = await self._get_or_create_sport(session, stat_data["sport_code"])
                season = await self._get_or_create_season(
                    session, sport.id, stat_data["year"], stat_data["sport_code"]
                )
                
                # Find team
                team_ext_id = f"{stat_data['sport_code']}_{stat_data['team_abbr']}"
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == team_ext_id
                        )
                    )
                )
                team = result.scalar_one_or_none()
                
                if not team:
                    continue
                
                # Check for existing stat
                result = await session.execute(
                    select(TeamStats).where(
                        and_(
                            TeamStats.team_id == team.id,
                            TeamStats.season_id == season.id,
                            TeamStats.stat_type == stat_data["stat_type"]
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = stat_data["value"]
                else:
                    stat = TeamStats(
                        team_id=team.id,
                        season_id=season.id,
                        stat_type=stat_data["stat_type"],
                        value=stat_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving team stat: {e}")
        
        await session.flush()
        return saved
    
    async def _save_player_stats(self, session: AsyncSession, stats: List[Dict]) -> int:
        """Save player statistics."""
        saved = 0
        
        for stat_data in stats:
            try:
                # Find player
                result = await session.execute(
                    select(Player).where(Player.external_id == stat_data["player_external_id"])
                )
                player = result.scalar_one_or_none()
                
                if not player:
                    continue
                
                # Get season
                sport = await self._get_or_create_sport(session, stat_data["sport_code"])
                season = await self._get_or_create_season(
                    session, sport.id, stat_data["year"], stat_data["sport_code"]
                )
                
                # Check for existing stat
                result = await session.execute(
                    select(PlayerStats).where(
                        and_(
                            PlayerStats.player_id == player.id,
                            PlayerStats.season_id == season.id,
                            PlayerStats.stat_type == stat_data["stat_type"]
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = stat_data["value"]
                else:
                    stat = PlayerStats(
                        player_id=player.id,
                        season_id=season.id,
                        stat_type=stat_data["stat_type"],
                        value=stat_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Sportsipy] Error saving player stat: {e}")
        
        await session.flush()
        return saved


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

sportsipy_collector = SportsipyCollector()

# Register with collector manager
collector_manager.register(sportsipy_collector)
logger.info("Registered collector: Sportsipy")