"""
ROYALEY - NFL Next Gen Stats Collector
Phase 1: Data Collection Services

Collects NFL player tracking data from Next Gen Stats via nfl_data_py library.
Features: Time-to-throw, air yards, separation, completion probability, route charts.

Data Source: NFL Next Gen Stats (via nflverse/nfl_data_py)
Documentation: https://github.com/nflverse/nfl_data_py

FREE data - no API key required!
Data available from 2016-present (NFL tracking system started in 2016)

Key Data Types:
- Passing: Time-to-throw, air yards, completion probability, aggressiveness
- Rushing: Expected rushing yards, rush yards over expected, efficiency
- Receiving: Separation, cushion, target share, YAC above expectation
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Sport, Team, Season, Player, PlayerStats, TeamStats
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Next Gen Stats data available from 2016 (when tracking system was installed)
NGS_START_YEAR = 2016
NGS_CURRENT_YEAR = datetime.now().year

# Stat types available in NGS
NGS_STAT_TYPES = ["passing", "rushing", "receiving"]

# NFL Teams mapping (abbreviation -> full name)
NFL_TEAMS = {
    "ARI": {"name": "Arizona Cardinals", "city": "Phoenix", "conference": "NFC", "division": "NFC West"},
    "ATL": {"name": "Atlanta Falcons", "city": "Atlanta", "conference": "NFC", "division": "NFC South"},
    "BAL": {"name": "Baltimore Ravens", "city": "Baltimore", "conference": "AFC", "division": "AFC North"},
    "BUF": {"name": "Buffalo Bills", "city": "Buffalo", "conference": "AFC", "division": "AFC East"},
    "CAR": {"name": "Carolina Panthers", "city": "Charlotte", "conference": "NFC", "division": "NFC South"},
    "CHI": {"name": "Chicago Bears", "city": "Chicago", "conference": "NFC", "division": "NFC North"},
    "CIN": {"name": "Cincinnati Bengals", "city": "Cincinnati", "conference": "AFC", "division": "AFC North"},
    "CLE": {"name": "Cleveland Browns", "city": "Cleveland", "conference": "AFC", "division": "AFC North"},
    "DAL": {"name": "Dallas Cowboys", "city": "Dallas", "conference": "NFC", "division": "NFC East"},
    "DEN": {"name": "Denver Broncos", "city": "Denver", "conference": "AFC", "division": "AFC West"},
    "DET": {"name": "Detroit Lions", "city": "Detroit", "conference": "NFC", "division": "NFC North"},
    "GB": {"name": "Green Bay Packers", "city": "Green Bay", "conference": "NFC", "division": "NFC North"},
    "HOU": {"name": "Houston Texans", "city": "Houston", "conference": "AFC", "division": "AFC South"},
    "IND": {"name": "Indianapolis Colts", "city": "Indianapolis", "conference": "AFC", "division": "AFC South"},
    "JAX": {"name": "Jacksonville Jaguars", "city": "Jacksonville", "conference": "AFC", "division": "AFC South"},
    "KC": {"name": "Kansas City Chiefs", "city": "Kansas City", "conference": "AFC", "division": "AFC West"},
    "LA": {"name": "Los Angeles Rams", "city": "Los Angeles", "conference": "NFC", "division": "NFC West"},
    "LAC": {"name": "Los Angeles Chargers", "city": "Los Angeles", "conference": "AFC", "division": "AFC West"},
    "LV": {"name": "Las Vegas Raiders", "city": "Las Vegas", "conference": "AFC", "division": "AFC West"},
    "MIA": {"name": "Miami Dolphins", "city": "Miami", "conference": "AFC", "division": "AFC East"},
    "MIN": {"name": "Minnesota Vikings", "city": "Minneapolis", "conference": "NFC", "division": "NFC North"},
    "NE": {"name": "New England Patriots", "city": "Foxborough", "conference": "AFC", "division": "AFC East"},
    "NO": {"name": "New Orleans Saints", "city": "New Orleans", "conference": "NFC", "division": "NFC South"},
    "NYG": {"name": "New York Giants", "city": "East Rutherford", "conference": "NFC", "division": "NFC East"},
    "NYJ": {"name": "New York Jets", "city": "East Rutherford", "conference": "AFC", "division": "AFC East"},
    "PHI": {"name": "Philadelphia Eagles", "city": "Philadelphia", "conference": "NFC", "division": "NFC East"},
    "PIT": {"name": "Pittsburgh Steelers", "city": "Pittsburgh", "conference": "AFC", "division": "AFC North"},
    "SEA": {"name": "Seattle Seahawks", "city": "Seattle", "conference": "NFC", "division": "NFC West"},
    "SF": {"name": "San Francisco 49ers", "city": "San Francisco", "conference": "NFC", "division": "NFC West"},
    "TB": {"name": "Tampa Bay Buccaneers", "city": "Tampa", "conference": "NFC", "division": "NFC South"},
    "TEN": {"name": "Tennessee Titans", "city": "Nashville", "conference": "AFC", "division": "AFC South"},
    "WAS": {"name": "Washington Commanders", "city": "Landover", "conference": "NFC", "division": "NFC East"},
    # Historical team names
    "OAK": {"name": "Oakland Raiders", "city": "Oakland", "conference": "AFC", "division": "AFC West"},
    "SD": {"name": "San Diego Chargers", "city": "San Diego", "conference": "AFC", "division": "AFC West"},
    "STL": {"name": "St. Louis Rams", "city": "St. Louis", "conference": "NFC", "division": "NFC West"},
}

# NGS Passing Stat Columns
NGS_PASSING_STATS = [
    "attempts",
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "passer_rating",
    "completions",
    "completion_percentage",
    "expected_completion_percentage",
    "completion_percentage_above_expectation",  # CPOE
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "avg_air_yards_differential",
    "aggressiveness",
    "max_completed_air_distance",
    "avg_air_yards_to_sticks",
]

# NGS Rushing Stat Columns
NGS_RUSHING_STATS = [
    "rush_attempts",
    "rush_yards",
    "avg_rush_yards",
    "rush_touchdowns",
    "efficiency",
    "percent_attempts_gte_eight_defenders",
    "avg_time_to_los",
    "expected_rush_yards",
    "rush_yards_over_expected",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
]

# NGS Receiving Stat Columns
NGS_RECEIVING_STATS = [
    "targets",
    "receptions",
    "yards",
    "rec_touchdowns",
    "avg_cushion",
    "avg_separation",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "catch_percentage",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
]


# =============================================================================
# NEXT GEN STATS COLLECTOR
# =============================================================================

class NextGenStatsCollector(BaseCollector):
    """
    NFL Next Gen Stats collector using nfl_data_py library.
    
    Collects player tracking data including:
    - Passing: Time-to-throw, completion probability, air yards, aggressiveness
    - Rushing: Expected rush yards, efficiency, time to LOS
    - Receiving: Separation, cushion, YAC above expectation
    
    Data available from 2016-present.
    """
    
    def __init__(self):
        super().__init__(
            name="nfl_nextgen_stats",
            base_url="https://github.com/nflverse",  # Data comes from nflverse
            rate_limit=60,
            rate_window=60,
            timeout=120.0,
            max_retries=3,
        )
        self._nfl_data_py = None
        logger.info("Registered collector: NFL Next Gen Stats")
    
    def _get_nfl_data_py(self):
        """Lazy load nfl_data_py module."""
        if self._nfl_data_py is None:
            try:
                import nfl_data_py as nfl
                self._nfl_data_py = nfl
                logger.info("[NGS] nfl_data_py loaded successfully")
            except ImportError:
                logger.error("[NGS] nfl_data_py not installed. Install with: pip install nfl_data_py")
                raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py")
        return self._nfl_data_py
    
    async def validate(self) -> bool:
        """Validate that nfl_data_py is available."""
        try:
            self._get_nfl_data_py()
            logger.info("[NGS] Validation passed - nfl_data_py available")
            return True
        except ImportError:
            logger.error("[NGS] Validation failed - nfl_data_py not available")
            return False
    
    # =========================================================================
    # DATA FETCHING METHODS
    # =========================================================================
    
    async def fetch_ngs_passing(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch NGS passing data for specified years.
        
        Columns include:
        - avg_time_to_throw: Average time from snap to throw
        - avg_completed_air_yards: Average air yards on completions
        - avg_intended_air_yards: Average intended air yards
        - completion_percentage_above_expectation: CPOE
        - aggressiveness: Percentage of tight-window throws
        """
        nfl = self._get_nfl_data_py()
        
        try:
            logger.info(f"[NGS] Fetching passing data for years: {years}")
            df = nfl.import_ngs_data(stat_type="passing", years=years)
            logger.info(f"[NGS] Retrieved {len(df)} passing records")
            return df
        except Exception as e:
            logger.error(f"[NGS] Error fetching passing data: {e}")
            return pd.DataFrame()
    
    async def fetch_ngs_rushing(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch NGS rushing data for specified years.
        
        Columns include:
        - expected_rush_yards: Expected rushing yards based on defensive alignment
        - rush_yards_over_expected: RYOE
        - efficiency: Percentage of runs with positive EPA
        - avg_time_to_los: Average time to reach line of scrimmage
        """
        nfl = self._get_nfl_data_py()
        
        try:
            logger.info(f"[NGS] Fetching rushing data for years: {years}")
            df = nfl.import_ngs_data(stat_type="rushing", years=years)
            logger.info(f"[NGS] Retrieved {len(df)} rushing records")
            return df
        except Exception as e:
            logger.error(f"[NGS] Error fetching rushing data: {e}")
            return pd.DataFrame()
    
    async def fetch_ngs_receiving(self, years: List[int]) -> pd.DataFrame:
        """
        Fetch NGS receiving data for specified years.
        
        Columns include:
        - avg_separation: Average yards of separation at catch point
        - avg_cushion: Average yards of cushion at snap
        - avg_yac_above_expectation: YAC above expected
        - percent_share_of_intended_air_yards: Target share
        """
        nfl = self._get_nfl_data_py()
        
        try:
            logger.info(f"[NGS] Fetching receiving data for years: {years}")
            df = nfl.import_ngs_data(stat_type="receiving", years=years)
            logger.info(f"[NGS] Retrieved {len(df)} receiving records")
            return df
        except Exception as e:
            logger.error(f"[NGS] Error fetching receiving data: {e}")
            return pd.DataFrame()
    
    async def fetch_all_ngs_data(self, years: List[int]) -> Dict[str, pd.DataFrame]:
        """Fetch all NGS data types for specified years."""
        return {
            "passing": await self.fetch_ngs_passing(years),
            "rushing": await self.fetch_ngs_rushing(years),
            "receiving": await self.fetch_ngs_receiving(years),
        }
    
    # =========================================================================
    # DATA PARSING METHODS
    # =========================================================================
    
    def _parse_passing_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse NGS passing dataframe into standardized format."""
        records = []
        
        for _, row in df.iterrows():
            try:
                player_id = row.get("player_gsis_id") or row.get("player_id")
                player_name = row.get("player_display_name") or row.get("player_name")
                team_abbr = row.get("team_abbr") or row.get("recent_team")
                season = row.get("season")
                week = row.get("week")
                
                if not player_name or not season:
                    continue
                
                # Build stats dictionary
                stats = {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_abbr": team_abbr,
                    "season": int(season),
                    "week": int(week) if pd.notna(week) else None,
                    "stat_category": "ngs_passing",
                    "stats": {}
                }
                
                # Extract all available passing stats
                for col in NGS_PASSING_STATS:
                    if col in row and pd.notna(row[col]):
                        stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else row[col]
                
                # Also capture any additional columns
                for col in df.columns:
                    if col not in ["player_gsis_id", "player_id", "player_display_name", "player_name", 
                                   "team_abbr", "recent_team", "season", "week", "season_type", "player_position"]:
                        if pd.notna(row[col]) and col not in stats["stats"]:
                            try:
                                stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])
                            except:
                                pass
                
                if stats["stats"]:
                    records.append(stats)
                    
            except Exception as e:
                logger.debug(f"[NGS] Error parsing passing row: {e}")
                continue
        
        return records
    
    def _parse_rushing_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse NGS rushing dataframe into standardized format."""
        records = []
        
        for _, row in df.iterrows():
            try:
                player_id = row.get("player_gsis_id") or row.get("player_id")
                player_name = row.get("player_display_name") or row.get("player_name")
                team_abbr = row.get("team_abbr") or row.get("recent_team")
                season = row.get("season")
                week = row.get("week")
                
                if not player_name or not season:
                    continue
                
                stats = {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_abbr": team_abbr,
                    "season": int(season),
                    "week": int(week) if pd.notna(week) else None,
                    "stat_category": "ngs_rushing",
                    "stats": {}
                }
                
                # Extract all available rushing stats
                for col in NGS_RUSHING_STATS:
                    if col in row and pd.notna(row[col]):
                        stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else row[col]
                
                # Capture additional columns
                for col in df.columns:
                    if col not in ["player_gsis_id", "player_id", "player_display_name", "player_name", 
                                   "team_abbr", "recent_team", "season", "week", "season_type", "player_position"]:
                        if pd.notna(row[col]) and col not in stats["stats"]:
                            try:
                                stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])
                            except:
                                pass
                
                if stats["stats"]:
                    records.append(stats)
                    
            except Exception as e:
                logger.debug(f"[NGS] Error parsing rushing row: {e}")
                continue
        
        return records
    
    def _parse_receiving_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse NGS receiving dataframe into standardized format."""
        records = []
        
        for _, row in df.iterrows():
            try:
                player_id = row.get("player_gsis_id") or row.get("player_id")
                player_name = row.get("player_display_name") or row.get("player_name")
                team_abbr = row.get("team_abbr") or row.get("recent_team")
                season = row.get("season")
                week = row.get("week")
                
                if not player_name or not season:
                    continue
                
                stats = {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_abbr": team_abbr,
                    "season": int(season),
                    "week": int(week) if pd.notna(week) else None,
                    "stat_category": "ngs_receiving",
                    "stats": {}
                }
                
                # Extract all available receiving stats
                for col in NGS_RECEIVING_STATS:
                    if col in row and pd.notna(row[col]):
                        stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else row[col]
                
                # Capture additional columns
                for col in df.columns:
                    if col not in ["player_gsis_id", "player_id", "player_display_name", "player_name", 
                                   "team_abbr", "recent_team", "season", "week", "season_type", "player_position"]:
                        if pd.notna(row[col]) and col not in stats["stats"]:
                            try:
                                stats["stats"][col] = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])
                            except:
                                pass
                
                if stats["stats"]:
                    records.append(stats)
                    
            except Exception as e:
                logger.debug(f"[NGS] Error parsing receiving row: {e}")
                continue
        
        return records
    
    # =========================================================================
    # MAIN COLLECTION METHODS
    # =========================================================================
    
    async def collect(
        self,
        years: List[int] = None,
        stat_type: str = "all",
    ) -> CollectorResult:
        """
        Main collection method for NFL Next Gen Stats.
        
        Args:
            years: List of years to collect (default: current year)
            stat_type: Type of stats to collect ('all', 'passing', 'rushing', 'receiving')
            
        Returns:
            CollectorResult with collected NGS data
        """
        if years is None:
            current_year = datetime.now().year
            years = [current_year]
        
        # Filter to valid years (NGS started in 2016)
        years = [y for y in years if y >= NGS_START_YEAR]
        
        if not years:
            return CollectorResult(
                success=False,
                error=f"No valid years specified. NGS data available from {NGS_START_YEAR}."
            )
        
        logger.info(f"[NGS] Collecting {stat_type} stats for years: {years}")
        
        all_data = {
            "passing": [],
            "rushing": [],
            "receiving": [],
            "teams": list(NFL_TEAMS.items()),
        }
        
        try:
            # Validate nfl_data_py is available
            if not await self.validate():
                return CollectorResult(
                    success=False,
                    error="nfl_data_py library not available"
                )
            
            # Collect based on stat_type
            if stat_type in ["all", "passing"]:
                passing_df = await self.fetch_ngs_passing(years)
                if len(passing_df) > 0:
                    all_data["passing"] = self._parse_passing_stats(passing_df)
                    logger.info(f"[NGS] Parsed {len(all_data['passing'])} passing records")
            
            if stat_type in ["all", "rushing"]:
                rushing_df = await self.fetch_ngs_rushing(years)
                if len(rushing_df) > 0:
                    all_data["rushing"] = self._parse_rushing_stats(rushing_df)
                    logger.info(f"[NGS] Parsed {len(all_data['rushing'])} rushing records")
            
            if stat_type in ["all", "receiving"]:
                receiving_df = await self.fetch_ngs_receiving(years)
                if len(receiving_df) > 0:
                    all_data["receiving"] = self._parse_receiving_stats(receiving_df)
                    logger.info(f"[NGS] Parsed {len(all_data['receiving'])} receiving records")
            
            total_records = (
                len(all_data["passing"]) +
                len(all_data["rushing"]) +
                len(all_data["receiving"])
            )
            
            logger.info(f"[NGS] Collection complete: {total_records} total records")
            
            return CollectorResult(
                success=True,
                data=all_data,
                records_count=total_records,
                metadata={
                    "years": years,
                    "stat_type": stat_type,
                    "passing_count": len(all_data["passing"]),
                    "rushing_count": len(all_data["rushing"]),
                    "receiving_count": len(all_data["receiving"]),
                }
            )
            
        except Exception as e:
            logger.error(f"[NGS] Collection error: {e}")
            import traceback
            traceback.print_exc()
            return CollectorResult(
                success=False,
                error=str(e)
            )
    
    async def collect_history(self, years_back: int = 10) -> CollectorResult:
        """
        Collect historical NGS data.
        
        Args:
            years_back: Number of years of history (max 10, data starts 2016)
            
        Returns:
            CollectorResult with historical data
        """
        current_year = datetime.now().year
        start_year = max(NGS_START_YEAR, current_year - years_back)
        years = list(range(start_year, current_year + 1))
        
        logger.info(f"[NGS] Collecting historical data: {start_year} to {current_year}")
        
        return await self.collect(years=years, stat_type="all")
    
    async def collect_current_season(self) -> CollectorResult:
        """Collect NGS data for current/most recent season."""
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NFL season runs Sep-Feb
        # If we're in Jan-Aug, the most recent completed season is previous year
        # If we're in Sep-Dec, current season is this year
        if current_month <= 8:  # Jan-Aug: use previous year's season
            season_year = current_year - 1
        else:  # Sep-Dec: use current year's season
            season_year = current_year
        
        logger.info(f"[NGS] Collecting current/recent season: {season_year}")
        return await self.collect(years=[season_year], stat_type="all")
    
    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """
        Save collected NGS data to database.
        
        Args:
            data: Dictionary with passing, rushing, receiving data
            session: Database session
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        error_count = 0
        
        try:
            logger.info("[NGS] Starting database save...")
            
            # Get or create NFL sport
            result = await session.execute(
                select(Sport).where(Sport.code == "NFL")
            )
            sport = result.scalar_one_or_none()
            
            if not sport:
                sport = Sport(
                    code="NFL",
                    name="NFL Football",
                    api_key="football_nfl",
                    feature_count=75,
                    is_active=True
                )
                session.add(sport)
                await session.flush()
            
            sport_id = sport.id
            
            # Cache for teams, seasons, players
            team_cache: Dict[str, UUID] = {}
            season_cache: Dict[int, UUID] = {}
            player_cache: Dict[str, UUID] = {}
            
            # Process all stat categories
            all_stats = []
            all_stats.extend([(s, "passing") for s in data.get("passing", [])])
            all_stats.extend([(s, "rushing") for s in data.get("rushing", [])])
            all_stats.extend([(s, "receiving") for s in data.get("receiving", [])])
            
            total_stats = len(all_stats)
            logger.info(f"[NGS] Processing {total_stats} stat records...")
            
            for idx, (stat_data, stat_category) in enumerate(all_stats):
                if idx > 0 and idx % 500 == 0:
                    logger.info(f"[NGS] Progress: {idx}/{total_stats} ({saved_count} saved, {error_count} errors)")
                
                try:
                    season_year = stat_data.get("season")
                    team_abbr = stat_data.get("team_abbr")
                    player_name = stat_data.get("player_name")
                    player_ext_id = stat_data.get("player_id")
                    week = stat_data.get("week")
                    stats = stat_data.get("stats", {})
                    
                    if not all([season_year, player_name]):
                        continue
                    
                    # Get or create season
                    if season_year not in season_cache:
                        result = await session.execute(
                            select(Season).where(
                                and_(Season.sport_id == sport_id, Season.year == season_year)
                            )
                        )
                        season = result.scalar_one_or_none()
                        
                        if not season:
                            # NFL season: Sep to Feb
                            start_date = date(season_year, 9, 1)
                            end_date = date(season_year + 1, 2, 15)
                            
                            season = Season(
                                sport_id=sport_id,
                                year=season_year,
                                name=f"{season_year} NFL Season",
                                start_date=start_date,
                                end_date=end_date,
                                is_current=(season_year >= datetime.now().year)
                            )
                            session.add(season)
                            await session.flush()
                        
                        season_cache[season_year] = season.id
                    
                    season_id = season_cache[season_year]
                    
                    # Get or create team
                    team_id = None
                    if team_abbr and team_abbr not in team_cache:
                        team_info = NFL_TEAMS.get(team_abbr)
                        if team_info:
                            # First try to find by name (unique constraint is on sport_id + name)
                            result = await session.execute(
                                select(Team).where(
                                    and_(Team.sport_id == sport_id, Team.name == team_info["name"])
                                )
                            )
                            team = result.scalar_one_or_none()
                            
                            if not team:
                                # Also try by external_id
                                external_id = f"nfl_{team_abbr.lower()}"
                                result = await session.execute(
                                    select(Team).where(
                                        and_(Team.sport_id == sport_id, Team.external_id == external_id)
                                    )
                                )
                                team = result.scalar_one_or_none()
                            
                            if not team:
                                # Also try by abbreviation
                                result = await session.execute(
                                    select(Team).where(
                                        and_(Team.sport_id == sport_id, Team.abbreviation == team_abbr)
                                    )
                                )
                                team = result.scalar_one_or_none()
                            
                            if not team:
                                # Create new team
                                external_id = f"nfl_{team_abbr.lower()}"
                                team = Team(
                                    sport_id=sport_id,
                                    external_id=external_id,
                                    name=team_info["name"],
                                    abbreviation=team_abbr,
                                    city=team_info.get("city"),
                                    conference=team_info.get("conference"),
                                    division=team_info.get("division"),
                                    is_active=True
                                )
                                session.add(team)
                                await session.flush()
                            
                            team_cache[team_abbr] = team.id
                    
                    if team_abbr:
                        team_id = team_cache.get(team_abbr)
                    
                    # Get or create player
                    player_key = f"{player_name}_{team_abbr or 'FA'}"
                    
                    if player_key not in player_cache:
                        # Try to find existing player
                        if player_ext_id:
                            ext_id = f"nfl_gsis_{player_ext_id}"
                            result = await session.execute(
                                select(Player).where(Player.external_id == ext_id)
                            )
                        else:
                            # Search by name
                            result = await session.execute(
                                select(Player).where(
                                    and_(
                                        Player.name == player_name,
                                        Player.team_id == team_id
                                    )
                                )
                            )
                        
                        player = result.scalar_one_or_none()
                        
                        if not player:
                            ext_id = f"nfl_gsis_{player_ext_id}" if player_ext_id else f"nfl_ngs_{player_name.replace(' ', '_').lower()}"
                            player = Player(
                                team_id=team_id,
                                external_id=ext_id,
                                name=player_name,
                                is_active=True
                            )
                            session.add(player)
                            await session.flush()
                        
                        player_cache[player_key] = player.id
                    
                    player_id = player_cache[player_key]
                    
                    # Save each stat as a PlayerStats record
                    stat_type_base = f"ngs_{stat_category}"
                    
                    for stat_name, stat_value in stats.items():
                        if stat_value is None:
                            continue
                        
                        try:
                            stat_value_float = float(stat_value) if isinstance(stat_value, (int, float)) else None
                            if stat_value_float is None:
                                continue
                            
                            # Create unique stat type with week if available
                            if week:
                                full_stat_type = f"{stat_type_base}_{stat_name}_w{week}"
                            else:
                                full_stat_type = f"{stat_type_base}_{stat_name}"
                            
                            # Check for existing stat
                            result = await session.execute(
                                select(PlayerStats).where(
                                    and_(
                                        PlayerStats.player_id == player_id,
                                        PlayerStats.season_id == season_id,
                                        PlayerStats.stat_type == full_stat_type
                                    )
                                )
                            )
                            existing_stat = result.scalar_one_or_none()
                            
                            if existing_stat:
                                existing_stat.value = stat_value_float
                            else:
                                stat = PlayerStats(
                                    player_id=player_id,
                                    season_id=season_id,
                                    stat_type=full_stat_type,
                                    value=stat_value_float
                                )
                                session.add(stat)
                            
                            saved_count += 1
                            
                        except Exception as e:
                            logger.debug(f"[NGS] Error saving stat {stat_name}: {e}")
                            continue
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(f"[NGS] Error saving record for {stat_data.get('player_name')}: {e}")
                    continue
            
            await session.commit()
            logger.info(f"[NGS] Saved {saved_count} records to database ({error_count} errors)")
            
        except Exception as e:
            logger.error(f"[NGS] Database save error: {e}")
            import traceback
            traceback.print_exc()
            await session.rollback()
            raise
        
        return saved_count


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

nextgenstats_collector = NextGenStatsCollector()

# Register with collector manager
try:
    collector_manager.register(nextgenstats_collector)
except Exception as e:
    logger.debug(f"Could not register NextGenStats collector: {e}")