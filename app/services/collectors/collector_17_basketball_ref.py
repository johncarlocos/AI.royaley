"""
ROYALEY - Basketball Reference Scraper Collector
Phase 1: Data Collection Services

Collects comprehensive NBA data from Basketball-Reference via basketball-reference-scraper.
- Team rosters and stats
- Player stats and career data
- Game schedules and box scores
- Injury reports
- Shot charts and play-by-play

Data Source:
- https://www.basketball-reference.com/

FREE data - uses basketball-reference-scraper Python package!

pip install basketball-reference-scraper

NOTE: This package requires Selenium + Chrome browser. 
It may not work in Docker containers without Chrome installed.
For reliable NBA data, use the hoopr collector instead.

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
# NBA TEAM CONFIGURATION
# =============================================================================

NBA_TEAMS = {
    "ATL": {"name": "Atlanta Hawks", "city": "Atlanta", "conference": "Eastern", "division": "Southeast"},
    "BOS": {"name": "Boston Celtics", "city": "Boston", "conference": "Eastern", "division": "Atlantic"},
    "BRK": {"name": "Brooklyn Nets", "city": "Brooklyn", "conference": "Eastern", "division": "Atlantic"},
    "CHO": {"name": "Charlotte Hornets", "city": "Charlotte", "conference": "Eastern", "division": "Southeast"},
    "CHI": {"name": "Chicago Bulls", "city": "Chicago", "conference": "Eastern", "division": "Central"},
    "CLE": {"name": "Cleveland Cavaliers", "city": "Cleveland", "conference": "Eastern", "division": "Central"},
    "DAL": {"name": "Dallas Mavericks", "city": "Dallas", "conference": "Western", "division": "Southwest"},
    "DEN": {"name": "Denver Nuggets", "city": "Denver", "conference": "Western", "division": "Northwest"},
    "DET": {"name": "Detroit Pistons", "city": "Detroit", "conference": "Eastern", "division": "Central"},
    "GSW": {"name": "Golden State Warriors", "city": "San Francisco", "conference": "Western", "division": "Pacific"},
    "HOU": {"name": "Houston Rockets", "city": "Houston", "conference": "Western", "division": "Southwest"},
    "IND": {"name": "Indiana Pacers", "city": "Indianapolis", "conference": "Eastern", "division": "Central"},
    "LAC": {"name": "Los Angeles Clippers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "LAL": {"name": "Los Angeles Lakers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "MEM": {"name": "Memphis Grizzlies", "city": "Memphis", "conference": "Western", "division": "Southwest"},
    "MIA": {"name": "Miami Heat", "city": "Miami", "conference": "Eastern", "division": "Southeast"},
    "MIL": {"name": "Milwaukee Bucks", "city": "Milwaukee", "conference": "Eastern", "division": "Central"},
    "MIN": {"name": "Minnesota Timberwolves", "city": "Minneapolis", "conference": "Western", "division": "Northwest"},
    "NOP": {"name": "New Orleans Pelicans", "city": "New Orleans", "conference": "Western", "division": "Southwest"},
    "NYK": {"name": "New York Knicks", "city": "New York", "conference": "Eastern", "division": "Atlantic"},
    "OKC": {"name": "Oklahoma City Thunder", "city": "Oklahoma City", "conference": "Western", "division": "Northwest"},
    "ORL": {"name": "Orlando Magic", "city": "Orlando", "conference": "Eastern", "division": "Southeast"},
    "PHI": {"name": "Philadelphia 76ers", "city": "Philadelphia", "conference": "Eastern", "division": "Atlantic"},
    "PHO": {"name": "Phoenix Suns", "city": "Phoenix", "conference": "Western", "division": "Pacific"},
    "POR": {"name": "Portland Trail Blazers", "city": "Portland", "conference": "Western", "division": "Northwest"},
    "SAC": {"name": "Sacramento Kings", "city": "Sacramento", "conference": "Western", "division": "Pacific"},
    "SAS": {"name": "San Antonio Spurs", "city": "San Antonio", "conference": "Western", "division": "Southwest"},
    "TOR": {"name": "Toronto Raptors", "city": "Toronto", "conference": "Eastern", "division": "Atlantic"},
    "UTA": {"name": "Utah Jazz", "city": "Salt Lake City", "conference": "Western", "division": "Northwest"},
    "WAS": {"name": "Washington Wizards", "city": "Washington", "conference": "Eastern", "division": "Southeast"},
}

# Team stat types to collect
TEAM_STAT_TYPES = [
    "wins", "losses", "win_pct", "points_per_game", "opponent_points_per_game",
    "field_goal_pct", "three_point_pct", "free_throw_pct",
    "rebounds_per_game", "assists_per_game", "steals_per_game", "blocks_per_game",
    "turnovers_per_game", "offensive_rating", "defensive_rating", "net_rating",
    "pace", "true_shooting_pct", "effective_fg_pct"
]

# Player stat types to collect
PLAYER_STAT_TYPES = [
    "games_played", "games_started", "minutes_per_game",
    "points_per_game", "rebounds_per_game", "assists_per_game",
    "steals_per_game", "blocks_per_game", "turnovers_per_game",
    "field_goal_pct", "three_point_pct", "free_throw_pct",
    "offensive_rebounds", "defensive_rebounds", "personal_fouls",
    "player_efficiency_rating", "true_shooting_pct", "usage_rate"
]


# =============================================================================
# BASKETBALL REFERENCE COLLECTOR CLASS
# =============================================================================

class BasketballRefCollector(BaseCollector):
    """Collector for NBA data via basketball-reference-scraper package."""
    
    name = "basketball_ref"
    
    def __init__(self):
        super().__init__(
            name="basketball_ref",
            base_url="https://www.basketball-reference.com",
            rate_limit=0.5,  # Be polite - 2 requests per second max
        )
        self._package_available = None
        self._modules = {}
    
    def _check_package(self) -> bool:
        """Check if basketball-reference-scraper is installed and working."""
        if self._package_available is None:
            try:
                # Try importing without triggering Selenium
                import basketball_reference_scraper
                self._package_available = True
                logger.info("[BasketballRef] Package available")
            except ImportError as e:
                self._package_available = False
                logger.warning(f"[BasketballRef] Package not installed: {e}")
            except Exception as e:
                # Selenium/Chrome errors
                self._package_available = False
                logger.warning(f"[BasketballRef] Package error (likely Selenium/Chrome issue): {e}")
        return self._package_available

    def _safe_import_teams(self):
        """Safely import teams module."""
        try:
            from basketball_reference_scraper.teams import get_roster, get_team_stats, get_team_misc
            return {"get_roster": get_roster, "get_team_stats": get_team_stats, "get_team_misc": get_team_misc}
        except Exception as e:
            logger.error(f"[BasketballRef] Failed to import teams module: {e}")
            return None

    def _safe_import_seasons(self):
        """Safely import seasons module."""
        try:
            from basketball_reference_scraper.seasons import get_schedule, get_standings
            return {"get_schedule": get_schedule, "get_standings": get_standings}
        except Exception as e:
            logger.error(f"[BasketballRef] Failed to import seasons module: {e}")
            return None

    def _safe_import_players(self):
        """Safely import players module."""
        try:
            from basketball_reference_scraper.players import get_stats, get_game_logs
            return {"get_stats": get_stats, "get_game_logs": get_game_logs}
        except Exception as e:
            logger.error(f"[BasketballRef] Failed to import players module: {e}")
            return None

    def _safe_import_box_scores(self):
        """Safely import box scores module."""
        try:
            from basketball_reference_scraper.box_scores import get_box_scores
            return {"get_box_scores": get_box_scores}
        except Exception as e:
            logger.error(f"[BasketballRef] Failed to import box_scores module: {e}")
            return None

    def _safe_import_injury(self):
        """Safely import injury module."""
        try:
            from basketball_reference_scraper.injury_report import get_injury_report
            return {"get_injury_report": get_injury_report}
        except Exception as e:
            logger.error(f"[BasketballRef] Failed to import injury module: {e}")
            return None

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
        years_back: int = 10,
        collect_type: str = "all",
    ) -> CollectorResult:
        """
        Collect NBA data from Basketball-Reference.
        
        Args:
            years_back: Number of years to collect (default: 10)
            collect_type: Type of data to collect:
                - "all": All data types
                - "teams": Teams and rosters only
                - "players": Player stats only
                - "games": Schedules and box scores only
                - "stats": Team and player stats only
                - "injuries": Current injury report only
        
        Returns:
            CollectorResult with collected data
        """
        # Test if package works (Selenium might fail)
        teams_module = self._safe_import_teams()
        if teams_module is None:
            return CollectorResult(
                success=False,
                data=None,
                records_count=0,
                error="basketball-reference-scraper package not working. Requires Selenium + Chrome. Use hoopr collector instead."
            )
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        data = {
            "teams": [],
            "players": [],
            "games": [],
            "team_stats": [],
            "player_stats": [],
            "seasons": [],
            "injuries": [],
        }
        total_records = 0
        errors = []
        
        # NBA season uses END year (2024 = 2023-24 season)
        # Season runs Oct-June, so latest complete is current year if after June
        if current_month < 7:
            latest_year = current_year - 1
        else:
            latest_year = current_year
        
        start_year = latest_year - years_back + 1
        end_year = latest_year
        
        logger.info(f"[BasketballRef] Collecting NBA data for years {start_year} to {end_year}")
        
        # Collect current injuries (doesn't need year)
        if collect_type in ["all", "injuries"]:
            try:
                injuries = await self._collect_injuries()
                data["injuries"].extend(injuries)
                total_records += len(injuries)
                logger.info(f"[BasketballRef] Collected {len(injuries)} injury reports")
            except Exception as e:
                logger.warning(f"[BasketballRef] Error collecting injuries: {e}")
                errors.append(f"injuries: {str(e)[:50]}")
        
        for year in range(start_year, end_year + 1):
            logger.info(f"[BasketballRef] NBA {year}...")
            
            try:
                # Rate limiting - be polite to Basketball-Reference
                await asyncio.sleep(random.uniform(2.0, 4.0))
                
                # Collect teams/rosters
                if collect_type in ["all", "teams"]:
                    teams = await self._collect_teams(year)
                    data["teams"].extend(teams)
                    total_records += len(teams)
                    logger.info(f"[BasketballRef] NBA {year}: {len(teams)} teams")
                
                # Collect team stats
                if collect_type in ["all", "stats"]:
                    team_stats = await self._collect_team_stats(year)
                    data["team_stats"].extend(team_stats)
                    total_records += len(team_stats)
                    logger.info(f"[BasketballRef] NBA {year}: {len(team_stats)} team stats")
                
                # Collect schedule/games
                if collect_type in ["all", "games"]:
                    games = await self._collect_games(year)
                    data["games"].extend(games)
                    total_records += len(games)
                    logger.info(f"[BasketballRef] NBA {year}: {len(games)} games")
                
                # Collect players (only recent years - expensive)
                if collect_type in ["all", "players"] and year >= current_year - 2:
                    players = await self._collect_players(year)
                    data["players"].extend(players)
                    total_records += len(players)
                    logger.info(f"[BasketballRef] NBA {year}: {len(players)} players")
                
                # Add season record
                season_data = {
                    "sport_code": "NBA",
                    "year": year,
                    "name": f"{year-1}-{str(year)[-2:]}",  # e.g., "2023-24"
                }
                data["seasons"].append(season_data)
                
            except Exception as e:
                logger.warning(f"[BasketballRef] Error collecting NBA {year}: {e}")
                errors.append(f"NBA {year}: {str(e)[:50]}")
                continue
        
        logger.info(f"[BasketballRef] Total records collected: {total_records}")
        
        return CollectorResult(
            success=total_records > 0,
            data=data,
            records_count=total_records,
            error="; ".join(errors[:5]) if errors else None
        )

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self, year: int) -> List[Dict[str, Any]]:
        """Collect team rosters for a season."""
        teams = []
        
        teams_module = self._safe_import_teams()
        if not teams_module:
            return teams
        
        get_roster = teams_module.get("get_roster")
        
        for abbr, info in NBA_TEAMS.items():
            try:
                # Rate limiting between teams
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Get roster
                loop = asyncio.get_event_loop()
                roster_df = await loop.run_in_executor(
                    None,
                    lambda a=abbr, y=year: get_roster(a, y)
                )
                
                team_data = {
                    "sport_code": "NBA",
                    "year": year,
                    "external_id": f"NBA_{abbr}",
                    "abbreviation": abbr,
                    "name": info["name"],
                    "city": info["city"],
                    "conference": info["conference"],
                    "division": info["division"],
                    "roster_count": len(roster_df) if roster_df is not None else 0,
                }
                teams.append(team_data)
                
            except Exception as e:
                logger.debug(f"[BasketballRef] Error getting roster for {abbr}: {e}")
                # Still add team even without roster
                teams.append({
                    "sport_code": "NBA",
                    "year": year,
                    "external_id": f"NBA_{abbr}",
                    "abbreviation": abbr,
                    "name": info["name"],
                    "city": info["city"],
                    "conference": info["conference"],
                    "division": info["division"],
                })
        
        return teams

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, year: int) -> List[Dict[str, Any]]:
        """Collect team statistics for a season."""
        stats = []
        
        teams_module = self._safe_import_teams()
        if not teams_module:
            return stats
        
        get_team_stats = teams_module.get("get_team_stats")
        
        for abbr in NBA_TEAMS.keys():
            try:
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                loop = asyncio.get_event_loop()
                stats_df = await loop.run_in_executor(
                    None,
                    lambda a=abbr, y=year: get_team_stats(a, y, data_format='PER_GAME')
                )
                
                if stats_df is not None and not stats_df.empty:
                    # Convert DataFrame row to stats
                    for col in stats_df.columns:
                        try:
                            value = float(stats_df[col].iloc[-1])  # Get latest/total
                            stat_data = {
                                "sport_code": "NBA",
                                "team_abbr": abbr,
                                "year": year,
                                "stat_type": f"bref_{col.lower().replace(' ', '_')}",
                                "value": value,
                            }
                            stats.append(stat_data)
                        except (ValueError, TypeError):
                            continue
                
            except Exception as e:
                logger.debug(f"[BasketballRef] Error getting stats for {abbr}: {e}")
                continue
        
        return stats

    # =========================================================================
    # GAMES/SCHEDULE COLLECTION
    # =========================================================================
    
    async def _collect_games(self, year: int) -> List[Dict[str, Any]]:
        """Collect game schedules for a season."""
        games = []
        
        seasons_module = self._safe_import_seasons()
        if not seasons_module:
            return games
        
        get_schedule = seasons_module.get("get_schedule")
        
        try:
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            loop = asyncio.get_event_loop()
            schedule_df = await loop.run_in_executor(
                None,
                lambda y=year: get_schedule(y)
            )
            
            if schedule_df is not None and not schedule_df.empty:
                for idx, row in schedule_df.iterrows():
                    try:
                        # Parse game data
                        game_date = row.get('DATE', row.get('Date', None))
                        home_team = row.get('HOME', row.get('Home', None))
                        away_team = row.get('VISITOR', row.get('Visitor', row.get('Away', None)))
                        home_pts = row.get('HOME_PTS', row.get('Home_PTS', row.get('PTS', None)))
                        away_pts = row.get('VISITOR_PTS', row.get('Visitor_PTS', row.get('PTS.1', None)))
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Create game ID
                        if isinstance(game_date, str):
                            date_str = game_date.replace("-", "")[:8]
                        elif hasattr(game_date, 'strftime'):
                            date_str = game_date.strftime("%Y%m%d")
                        else:
                            date_str = str(idx)
                        
                        teams_sorted = sorted([str(home_team)[:3], str(away_team)[:3]])
                        game_id = f"NBA_{date_str}_{teams_sorted[0]}_{teams_sorted[1]}"
                        
                        # Determine status
                        status = "final" if home_pts is not None else "scheduled"
                        
                        game_data = {
                            "sport_code": "NBA",
                            "external_id": game_id,
                            "year": year,
                            "home_team_abbr": str(home_team)[:3].upper(),
                            "away_team_abbr": str(away_team)[:3].upper(),
                            "scheduled_at": game_date if isinstance(game_date, datetime) else datetime.now(),
                            "home_score": int(home_pts) if home_pts is not None else None,
                            "away_score": int(away_pts) if away_pts is not None else None,
                            "status": status,
                        }
                        games.append(game_data)
                        
                    except Exception as e:
                        logger.debug(f"[BasketballRef] Error parsing game: {e}")
                        continue
            
        except Exception as e:
            logger.warning(f"[BasketballRef] Error getting schedule for {year}: {e}")
        
        return games

    # =========================================================================
    # PLAYERS COLLECTION
    # =========================================================================
    
    async def _collect_players(self, year: int) -> List[Dict[str, Any]]:
        """Collect player data for a season."""
        players = []
        seen_players = set()
        
        teams_module = self._safe_import_teams()
        if not teams_module:
            return players
        
        get_roster = teams_module.get("get_roster")
        
        for abbr in NBA_TEAMS.keys():
            try:
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                loop = asyncio.get_event_loop()
                roster_df = await loop.run_in_executor(
                    None,
                    lambda a=abbr, y=year: get_roster(a, y)
                )
                
                if roster_df is not None and not roster_df.empty:
                    for idx, row in roster_df.iterrows():
                        try:
                            player_name = row.get('PLAYER', row.get('Player', str(idx)))
                            
                            if player_name in seen_players:
                                continue
                            seen_players.add(player_name)
                            
                            player_data = {
                                "sport_code": "NBA",
                                "external_id": f"bref_NBA_{player_name.replace(' ', '_').lower()}",
                                "team_abbr": abbr,
                                "year": year,
                                "name": player_name,
                                "position": row.get('POS', row.get('Pos', None)),
                                "jersey_number": row.get('NUMBER', row.get('No.', None)),
                                "height": row.get('HEIGHT', row.get('Ht', None)),
                                "weight": row.get('WEIGHT', row.get('Wt', None)),
                                "birth_date": row.get('BIRTH_DATE', row.get('Birth Date', None)),
                            }
                            players.append(player_data)
                            
                        except Exception as e:
                            logger.debug(f"[BasketballRef] Error parsing player: {e}")
                            continue
                
            except Exception as e:
                logger.debug(f"[BasketballRef] Error getting roster for {abbr}: {e}")
                continue
        
        return players

    # =========================================================================
    # INJURY REPORT COLLECTION
    # =========================================================================
    
    async def _collect_injuries(self) -> List[Dict[str, Any]]:
        """Collect current NBA injury report."""
        injuries = []
        
        injury_module = self._safe_import_injury()
        if not injury_module:
            return injuries
        
        get_injury_report = injury_module.get("get_injury_report")
        
        try:
            loop = asyncio.get_event_loop()
            injury_df = await loop.run_in_executor(
                None,
                get_injury_report
            )
            
            if injury_df is not None and not injury_df.empty:
                for idx, row in injury_df.iterrows():
                    try:
                        injury_data = {
                            "sport_code": "NBA",
                            "player_name": row.get('PLAYER', row.get('Player', '')),
                            "team": row.get('TEAM', row.get('Team', '')),
                            "date": row.get('DATE', row.get('Date', datetime.now())),
                            "injury": row.get('INJURY', row.get('Description', '')),
                            "status": row.get('STATUS', row.get('Status', '')),
                        }
                        injuries.append(injury_data)
                    except Exception as e:
                        logger.debug(f"[BasketballRef] Error parsing injury: {e}")
                        continue
            
        except Exception as e:
            logger.warning(f"[BasketballRef] Error getting injury report: {e}")
        
        return injuries

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
                logger.info(f"[BasketballRef] Saved {saved} seasons")
            
            # Save teams
            if data.get("teams"):
                saved = await self._save_teams(session, data["teams"])
                total_saved += saved
                logger.info(f"[BasketballRef] Saved {saved} teams")
            
            # Save players
            if data.get("players"):
                saved = await self._save_players(session, data["players"])
                total_saved += saved
                logger.info(f"[BasketballRef] Saved {saved} players")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(session, data["games"])
                total_saved += saved
                logger.info(f"[BasketballRef] Saved {saved} games")
            
            # Save team stats
            if data.get("team_stats"):
                saved = await self._save_team_stats(session, data["team_stats"])
                total_saved += saved
                logger.info(f"[BasketballRef] Saved {saved} team stats")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[BasketballRef] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _get_or_create_sport(self, session: AsyncSession) -> Sport:
        """Get or create NBA sport record."""
        result = await session.execute(
            select(Sport).where(Sport.code == "NBA")
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                code="NBA",
                name="National Basketball Association",
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        return sport
    
    async def _get_or_create_season(
        self, 
        session: AsyncSession, 
        sport_id: UUID, 
        year: int
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
            # NBA season runs Oct-June
            start_date = date(year - 1, 10, 1)
            end_date = date(year, 6, 30)
            name = f"{year-1}-{str(year)[-2:]}"
            
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
                sport = await self._get_or_create_sport(session)
                await self._get_or_create_season(session, sport.id, season_data["year"])
                saved += 1
            except Exception as e:
                logger.debug(f"[BasketballRef] Error saving season: {e}")
        
        await session.flush()
        return saved
    
    async def _save_teams(self, session: AsyncSession, teams: List[Dict]) -> int:
        """Save team records."""
        saved = 0
        
        for team_data in teams:
            try:
                sport = await self._get_or_create_sport(session)
                external_id = team_data["external_id"]
                
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
                    existing.conference = team_data.get("conference")
                    existing.division = team_data.get("division")
                else:
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_data["name"],
                        abbreviation=team_data["abbreviation"],
                        city=team_data.get("city"),
                        conference=team_data.get("conference"),
                        division=team_data.get("division"),
                        is_active=True
                    )
                    session.add(team)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[BasketballRef] Error saving team: {e}")
        
        await session.flush()
        return saved
    
    async def _save_players(self, session: AsyncSession, players: List[Dict]) -> int:
        """Save player records."""
        saved = 0
        
        for player_data in players:
            try:
                external_id = player_data["external_id"]
                
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    if player_data.get("position"):
                        existing.position = player_data["position"]
                else:
                    # Find team
                    team_id = None
                    if player_data.get("team_abbr"):
                        sport = await self._get_or_create_sport(session)
                        team_ext_id = f"NBA_{player_data['team_abbr']}"
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
                logger.debug(f"[BasketballRef] Error saving player: {e}")
        
        await session.flush()
        return saved
    
    async def _save_games(self, session: AsyncSession, games: List[Dict]) -> int:
        """Save game records."""
        saved = 0
        
        for game_data in games:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, game_data["year"])
                
                external_id = game_data["external_id"]
                
                # Find teams
                home_ext_id = f"NBA_{game_data['home_team_abbr']}"
                away_ext_id = f"NBA_{game_data['away_team_abbr']}"
                
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
                
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
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
                logger.debug(f"[BasketballRef] Error saving game: {e}")
        
        await session.flush()
        return saved
    
    async def _save_team_stats(self, session: AsyncSession, stats: List[Dict]) -> int:
        """Save team statistics."""
        saved = 0
        
        for stat_data in stats:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, stat_data["year"])
                
                team_ext_id = f"NBA_{stat_data['team_abbr']}"
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
                logger.debug(f"[BasketballRef] Error saving team stat: {e}")
        
        await session.flush()
        return saved


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

basketball_ref_collector = BasketballRefCollector()

# Register with collector manager
collector_manager.register(basketball_ref_collector)
logger.info("Registered collector: Basketball Reference")