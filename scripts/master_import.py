#!/usr/bin/env python3
"""
ROYALEY - Master Data Import (Enhanced)
========================================

ONE COMMAND to import all data for ML training.

Usage:
    python scripts/master_import.py --full              # ALL data for ML training
    python scripts/master_import.py --current           # Current data (default)
    python scripts/master_import.py --historical        # Historical only
    python scripts/master_import.py --source pinnacle   # Specific source
    python scripts/master_import.py --source injuries   # Injuries from ESPN
    python scripts/master_import.py --source weather    # Weather for upcoming games
    python scripts/master_import.py --source players    # Players/stats from nflfastR/cfbfastR
    python scripts/master_import.py --sport NFL         # Specific sport
    python scripts/master_import.py --daemon            # Run continuously
    python scripts/master_import.py --status            # Show status

Tables filled by each source:
    espn          → games, teams, injuries, players
    odds_api      → odds, sportsbooks, games, teams
    pinnacle      → odds, odds_movements, closing_lines, sportsbooks
    weather       → weather_data
    sportsdb      → games, teams, venues
    nflfastr      → games, teams, players, player_stats, team_stats
    cfbfastr      → games, teams, players, player_stats, team_stats
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, field

# MUST come before app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            import re
            text = str(args[0]) if args else ""
            print(re.sub(r'\[.*?\]', '', text))

console = Console()
shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    console.print("\n[yellow]Shutting down...[/yellow]")
    shutdown_flag = True


@dataclass
class ImportResult:
    source: str
    success: bool = False
    records: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# IMPORT FUNCTIONS
# =============================================================================

async def import_espn(sports: List[str] = None) -> ImportResult:
    """Import games/scores/teams from ESPN."""
    result = ImportResult(source="espn")
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        for sport in sports:
            try:
                data = await espn_collector.collect(sport_code=sport)
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        # Save teams first (before games, since games reference teams)
                        if data.data.get("teams"):
                            await espn_collector.save_teams_to_database(data.data["teams"], session)
                        if data.data.get("games"):
                            await espn_collector.save_games_to_database(data.data["games"], session)
                        if data.data.get("scores"):
                            await espn_collector.save_scores_to_database(data.data["scores"], session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_espn_injuries(sports: List[str] = None) -> ImportResult:
    """Import injuries from ESPN."""
    result = ImportResult(source="injuries")
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                injuries_data = await espn_collector.collect_injuries(sport_code=sport)
                if injuries_data and injuries_data.get("injuries"):
                    async with db_manager.session() as session:
                        saved = await espn_collector.save_injuries_to_database(
                            injuries_data["injuries"], sport, session
                        )
                        result.records += saved
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        
        result.success = result.records > 0 or len(result.errors) == 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_espn_players(sports: List[str] = None) -> ImportResult:
    """Import players from ESPN."""
    result = ImportResult(source="espn_players")
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                players_data = await espn_collector.collect_players(sport_code=sport)
                if players_data and players_data.get("players"):
                    async with db_manager.session() as session:
                        saved = await espn_collector.save_players_to_database(
                            players_data["players"], sport, session
                        )
                        result.records += saved
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        
        result.success = result.records > 0 or len(result.errors) == 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_odds_api(sports: List[str] = None) -> ImportResult:
    """Import odds from TheOddsAPI."""
    result = ImportResult(source="odds_api")
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await odds_collector.collect(sport_code=sport)
                if data.success:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await odds_collector.save_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_pinnacle(sports: List[str] = None) -> ImportResult:
    """Import Pinnacle odds with line movements and closing lines."""
    result = ImportResult(source="pinnacle")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                data = await pinnacle_collector.collect(sport_code=sport)
                if data.success:
                    result.records += data.records_count
                    async with db_manager.session() as session:
                        # Save odds
                        await pinnacle_collector.save_to_database(data.data, session)
                        # Track line movements
                        movements = await pinnacle_collector.track_line_movements(data.data, session)
                        result.records += movements
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_pinnacle_history(sports: List[str] = None, pages: int = 50) -> ImportResult:
    """Import historical Pinnacle game results."""
    result = ImportResult(source="pinnacle_history")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                data = await pinnacle_collector.collect_historical(
                    sport_code=sport, 
                    max_pages=pages
                )
                if data.success:
                    result.records += data.records_count
                    async with db_manager.session() as session:
                        await pinnacle_collector.save_historical_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_pinnacle_closing_lines(sports: List[str] = None) -> ImportResult:
    """Capture closing lines from Pinnacle for completed games."""
    result = ImportResult(source="closing_lines")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                async with db_manager.session() as session:
                    saved = await pinnacle_collector.capture_closing_lines(sport, session)
                    result.records += saved
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_espn_history(sports: List[str] = None, days: int = 30) -> ImportResult:
    """Import ESPN historical games."""
    result = ImportResult(source="espn_history")
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        for sport in sports:
            try:
                data = await espn_collector.collect_historical(
                    sport_code=sport, 
                    days_back=days
                )
                if data.success:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await espn_collector.save_historical_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_odds_api_history(sports: List[str] = None, days: int = 30) -> ImportResult:
    """Import OddsAPI historical odds."""
    result = ImportResult(source="odds_api_history")
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                data = await odds_collector.collect_historical(
                    sport_code=sport,
                    days_back=days
                )
                if data.success:
                    result.records += data.records_count
                    async with db_manager.session() as session:
                        await odds_collector.save_historical_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_weather(sports: List[str] = None, days: int = 7) -> ImportResult:
    """Import weather for upcoming outdoor games."""
    result = ImportResult(source="weather")
    try:
        from app.services.collectors.collector_05_weather import WeatherCollector
        from app.core.database import db_manager
        
        # Only outdoor sports need weather
        outdoor_sports = ["NFL", "MLB", "NCAAF"]
        if sports:
            outdoor_sports = [s for s in sports if s in outdoor_sports]
        
        if not outdoor_sports:
            result.success = True
            return result
        
        await db_manager.initialize()
        
        async with WeatherCollector() as collector:
            for sport in outdoor_sports:
                try:
                    weather_result = await collector.collect_for_upcoming_games(
                        sport_code=sport,
                        days_ahead=days
                    )
                    if weather_result:
                        result.records += weather_result.get("saved", 0)
                except Exception as e:
                    result.errors.append(f"{sport}: {str(e)[:50]}")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb(sports: List[str] = None) -> ImportResult:
    """Import from TheSportsDB."""
    result = ImportResult(source="sportsdb")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        for sport in sports:
            try:
                data = await sportsdb_collector.collect(sport_code=sport)
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await sportsdb_collector.save_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0 or len(result.errors) == 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_history(sports: List[str] = None, seasons: int = 10) -> ImportResult:
    """Import TheSportsDB historical data."""
    result = ImportResult(source="sportsdb_history")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        for sport in sports:
            try:
                data = await sportsdb_collector.collect_historical(
                    sport_code=sport,
                    seasons_back=seasons
                )
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        saved, updated = await sportsdb_collector.save_historical_to_database(
                            data.data.get("games", []), session
                        )
                        result.records = saved + updated
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_livescores() -> ImportResult:
    """Import live scores from TheSportsDB."""
    result = ImportResult(source="sportsdb_live")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        data = await sportsdb_collector.collect_all_livescores()
        if data.success:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await sportsdb_collector._update_livescores(data.data, session)
        result.success = True
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_venues(sports: List[str] = None) -> ImportResult:
    """Import venues from TheSportsDB."""
    result = ImportResult(source="venues")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        await db_manager.initialize()
        
        for sport in sports:
            try:
                venues_data = await sportsdb_collector.collect_venues(sport_code=sport)
                if venues_data and venues_data.get("venues"):
                    async with db_manager.session() as session:
                        saved = await sportsdb_collector.save_venues_to_database(
                            venues_data["venues"], session
                        )
                        result.records += saved
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_players(sports: List[str] = None) -> ImportResult:
    """Import players from TheSportsDB."""
    result = ImportResult(source="sportsdb_players")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        await db_manager.initialize()
        
        players_data = await sportsdb_collector.collect_players()
        if players_data and players_data.get("players"):
            async with db_manager.session() as session:
                saved = await sportsdb_collector.save_players_to_database(
                    players_data["players"], session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_standings(sports: List[str] = None, season: str = None) -> ImportResult:
    """Import standings from TheSportsDB."""
    result = ImportResult(source="sportsdb_standings")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        standings_data = await sportsdb_collector.collect_standings(season=season)
        if standings_data and standings_data.get("standings"):
            print(f"[SportsDB] Retrieved {len(standings_data['standings'])} standings entries")
            result.records = len(standings_data["standings"])
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb_seasons(sports: List[str] = None) -> ImportResult:
    """List available seasons from TheSportsDB."""
    result = ImportResult(source="sportsdb_seasons")
    try:
        from app.services.collectors import sportsdb_collector
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        
        for sport in sports:
            seasons = await sportsdb_collector.get_available_seasons(sport)
            print(f"[SportsDB] {sport}: {seasons[:5]}...")
            result.records += len(seasons)
        
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nflfastr(sports: List[str] = None) -> ImportResult:
    """Import NFL current season from nflfastR."""
    result = ImportResult(source="nflfastr")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        
        current_year = datetime.now().year
        years = [current_year - 1, current_year] if datetime.now().month < 9 else [current_year]
        
        data = await nflfastr_collector.collect(years=years)
        if data.success:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await nflfastr_collector.save_to_database(data.data, session)
        result.success = result.records > 0 or data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nflfastr_history(years_back: int = 10) -> ImportResult:
    """Import NFL historical data from nflfastR."""
    result = ImportResult(source="nflfastr_history")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await nflfastr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await nflfastr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nflfastr_pbp(years_back: int = 3) -> ImportResult:
    """Import NFL play-by-play data."""
    result = ImportResult(source="nflfastr_pbp")
    try:
        from app.services.collectors import nflfastr_collector
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await nflfastr_collector.collect_pbp(years=years)
        if data.success:
            result.records = data.records_count
        result.success = data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nflfastr_players() -> ImportResult:
    """Import NFL players and stats from nflfastR."""
    result = ImportResult(source="nfl_players")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        # Collect player stats
        current_year = datetime.now().year
        years = list(range(current_year - 3, current_year + 1))
        
        data = await nflfastr_collector.collect(years=years, collect_type="player_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await nflfastr_collector.save_players_to_database(
                    data.data.get("player_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr(sports: List[str] = None) -> ImportResult:
    """Import NCAAF current season from cfbfastR."""
    result = ImportResult(source="cfbfastr")
    try:
        from app.services.collectors import cfbfastr_collector
        from app.core.database import db_manager
        
        current_year = datetime.now().year
        years = [current_year - 1, current_year]
        
        data = await cfbfastr_collector.collect(years=years)
        if data.success:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await cfbfastr_collector.save_to_database(data.data, session)
        result.success = result.records > 0 or data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr_history(years_back: int = 10) -> ImportResult:
    """Import NCAAF historical data from cfbfastR."""
    result = ImportResult(source="cfbfastr_history")
    try:
        from app.services.collectors import cfbfastr_collector
        from app.core.database import db_manager
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await cfbfastr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await cfbfastr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr_pbp(years_back: int = 3) -> ImportResult:
    """Import NCAAF play-by-play data."""
    result = ImportResult(source="cfbfastr_pbp")
    try:
        from app.services.collectors import cfbfastr_collector
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await cfbfastr_collector.collect(years=years, collect_type="pbp")
        if data.success:
            result.records = data.records_count
        result.success = data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr_sp(years_back: int = 5) -> ImportResult:
    """Import NCAAF SP+ ratings."""
    result = ImportResult(source="cfbfastr_sp")
    try:
        from app.services.collectors import cfbfastr_collector
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await cfbfastr_collector.collect(years=years, collect_type="sp_ratings")
        if data.success:
            result.records = data.records_count
        result.success = data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr_recruiting(years_back: int = 5) -> ImportResult:
    """Import NCAAF recruiting data."""
    result = ImportResult(source="cfbfastr_recruiting")
    try:
        from app.services.collectors import cfbfastr_collector
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await cfbfastr_collector.collect(years=years, collect_type="recruiting")
        if data.success:
            result.records = data.records_count
        result.success = data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbfastr_players() -> ImportResult:
    """Import NCAAF players and stats from cfbfastR."""
    result = ImportResult(source="ncaaf_players")
    try:
        from app.services.collectors import cfbfastr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 3, current_year + 1))
        
        data = await cfbfastr_collector.collect(years=years, collect_type="player_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbfastr_collector.save_players_to_database(
                    data.data.get("player_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# SOURCE MAPPING
# =============================================================================

IMPORT_MAP = {
    # Current data
    "espn": import_espn,
    "odds_api": import_odds_api,
    "pinnacle": import_pinnacle,
    "weather": import_weather,
    "sportsdb": import_sportsdb,
    "nflfastr": import_nflfastr,
    "cfbfastr": import_cfbfastr,
    
    # Historical data
    "pinnacle_history": import_pinnacle_history,
    "espn_history": import_espn_history,
    "odds_api_history": import_odds_api_history,
    "sportsdb_history": import_sportsdb_history,
    "nflfastr_history": import_nflfastr_history,
    "cfbfastr_history": import_cfbfastr_history,
    
    # Specialized data
    "injuries": import_espn_injuries,
    "players": import_espn_players,
    "nfl_players": import_nflfastr_players,
    "ncaaf_players": import_cfbfastr_players,
    "venues": import_sportsdb_venues,
    "sportsdb_players": import_sportsdb_players,
    "sportsdb_standings": import_sportsdb_standings,
    "sportsdb_seasons": import_sportsdb_seasons,
    "closing_lines": import_pinnacle_closing_lines,
    
    # Live data
    "sportsdb_live": import_sportsdb_livescores,
    
    # Play-by-play
    "nflfastr_pbp": import_nflfastr_pbp,
    "cfbfastr_pbp": import_cfbfastr_pbp,
    "cfbfastr_sp": import_cfbfastr_sp,
    "cfbfastr_recruiting": import_cfbfastr_recruiting,
}

# Source groups
CURRENT_SOURCES = ["espn", "odds_api", "pinnacle", "weather", "sportsdb", "nflfastr", "cfbfastr"]
HISTORICAL_SOURCES = ["pinnacle_history", "espn_history", "odds_api_history", "sportsdb_history", "nflfastr_history", "cfbfastr_history"]
PLAYER_SOURCES = ["injuries", "players", "nfl_players", "ncaaf_players"]
SPECIALIZED_SOURCES = ["venues", "closing_lines", "sportsdb_players", "sportsdb_standings", "sportsdb_seasons"]

# Full ML training data - everything needed
FULL_ML_SOURCES = (
    HISTORICAL_SOURCES + 
    CURRENT_SOURCES + 
    PLAYER_SOURCES + 
    SPECIALIZED_SOURCES
)

ALL_SOURCES = list(IMPORT_MAP.keys())


# =============================================================================
# MAIN IMPORT RUNNER
# =============================================================================

async def run_import(sources: List[str], sports: List[str] = None, pages: int = 50, days: int = 30, seasons: int = 10):
    """Run data import for specified sources."""
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold]ROYALEY DATA IMPORT[/bold]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Sources: {', '.join(sources)}")
    console.print(f"Sports: {', '.join(sports) if sports else 'ALL'}\n")
    
    total_records = 0
    total_errors = 0
    
    for source in sources:
        if shutdown_flag:
            break
            
        console.print(f"[cyan]📊 {source}...[/cyan]", end=" ")
        
        func = IMPORT_MAP.get(source)
        if not func:
            console.print("[yellow]unknown[/yellow]")
            continue
        
        try:
            # Call with appropriate parameters based on source
            if source == "pinnacle_history":
                result = await func(sports=sports, pages=pages)
            elif source in ["espn_history", "odds_api_history"]:
                result = await func(sports=sports, days=days)
            elif source == "sportsdb_history":
                result = await func(sports=sports, seasons=seasons)
            elif source in ["nflfastr_history", "cfbfastr_history"]:
                result = await func(years_back=seasons)
            elif source in ["sportsdb_live", "nflfastr_pbp", "cfbfastr_pbp", 
                           "cfbfastr_sp", "cfbfastr_recruiting", "closing_lines",
                           "nfl_players", "ncaaf_players"]:
                result = await func()
            elif source == "weather":
                result = await func(sports=sports, days=7)
            else:
                result = await func(sports=sports)
            
            total_records += result.records
            total_errors += len(result.errors)
            
            if result.success:
                console.print(f"[green]✅ {result.records} records[/green]")
            else:
                console.print(f"[red]❌ {result.errors[0] if result.errors else 'failed'}[/red]")
        except Exception as e:
            console.print(f"[red]❌ {str(e)[:50]}[/red]")
            total_errors += 1
    
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold]COMPLETE[/bold] - Records: {total_records}, Errors: {total_errors}")


async def daemon_mode(interval: int, sources: List[str], sports: List[str]):
    """Run imports continuously at specified interval."""
    global shutdown_flag
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    console.print(f"\n[bold green]🚀 Daemon Mode - Every {interval} min[/bold green]")
    
    run_count = 0
    while not shutdown_flag:
        run_count += 1
        console.print(f"\n[cyan]--- Run #{run_count} ---[/cyan]")
        
        try:
            await run_import(sources, sports)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        if shutdown_flag:
            break
            
        console.print(f"\n[dim]Next in {interval} min...[/dim]")
        for _ in range(interval * 60):
            if shutdown_flag:
                break
            await asyncio.sleep(1)
    
    console.print("\n[green]Stopped.[/green]")


def show_status():
    """Show status of all data collectors."""
    console.print("\n[bold]ROYALEY DATA COLLECTORS - ENHANCED[/bold]")
    console.print("=" * 60)
    
    console.print("\n[green]✅ CURRENT DATA (--current):[/green]")
    console.print("  • espn          → games, teams")
    console.print("  • odds_api      → odds, sportsbooks")
    console.print("  • pinnacle      → odds, odds_movements, closing_lines")
    console.print("  • weather       → weather_data")
    console.print("  • sportsdb      → games, teams")
    console.print("  • nflfastr      → games, teams (NFL)")
    console.print("  • cfbfastr      → games, teams (NCAAF)")
    
    console.print("\n[green]✅ HISTORICAL DATA (--historical):[/green]")
    console.print("  • pinnacle_history  → games (archived)")
    console.print("  • espn_history      → games (past N days)")
    console.print("  • odds_api_history  → odds (past N days)")
    console.print("  • sportsdb_history  → games (past N seasons)")
    console.print("  • nflfastr_history  → games (1999-present)")
    console.print("  • cfbfastr_history  → games (2002-present)")
    
    console.print("\n[cyan]⚡ SPECIALIZED DATA:[/cyan]")
    console.print("  • injuries      → injuries (from ESPN)")
    console.print("  • players       → players (from ESPN)")
    console.print("  • nfl_players   → players, player_stats (from nflfastR)")
    console.print("  • ncaaf_players → players, player_stats (from cfbfastR)")
    console.print("  • venues        → venues (from TheSportsDB)")
    console.print("  • sportsdb_players   → players (from TheSportsDB)")
    console.print("  • sportsdb_standings → standings (from TheSportsDB)")
    console.print("  • sportsdb_seasons   → list available seasons")
    console.print("  • closing_lines → closing_lines (from Pinnacle)")
    
    console.print("\n[cyan]⚡ PLAY-BY-PLAY:[/cyan]")
    console.print("  • nflfastr_pbp      → PBP + EPA/WPA/CPOE")
    console.print("  • cfbfastr_pbp      → PBP + EPA")
    console.print("  • cfbfastr_sp       → SP+ ratings")
    console.print("  • cfbfastr_recruiting → Recruiting rankings")
    
    console.print("\n[cyan]⚡ LIVESCORES:[/cyan]")
    console.print("  • sportsdb_live     → Real-time scores")
    
    console.print("\n[bold yellow]🚀 FULL ML TRAINING (--full):[/bold yellow]")
    console.print("  Imports ALL data needed for ML model training:")
    console.print("  → Historical games, odds, players, injuries, venues, weather")
    console.print("  → Closing lines for CLV calculation")
    console.print("  → Player/team stats for feature engineering")
    
    console.print("\n[bold]TABLES FILLED:[/bold]")
    console.print("  games, teams, sports, odds, sportsbooks, odds_movements,")
    console.print("  closing_lines, weather_data, injuries, players, player_stats,")
    console.print("  team_stats, venues")


def main():
    parser = argparse.ArgumentParser(description="ROYALEY Master Data Import")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="ALL data for ML training")
    mode.add_argument("--all", action="store_true", help="All sources")
    mode.add_argument("--current", action="store_true", help="Current only (default)")
    mode.add_argument("--historical", action="store_true", help="Historical only")
    mode.add_argument("--daemon", action="store_true", help="Run continuously")
    mode.add_argument("--status", action="store_true", help="Show status")
    
    parser.add_argument("--source", "-s", help="Specific source")
    parser.add_argument("--sport", help="Specific sport")
    parser.add_argument("--sports", help="Comma-separated sports")
    parser.add_argument("--pages", "-p", type=int, default=50, help="Pinnacle history pages")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days back for history")
    parser.add_argument("--seasons", type=int, default=10, help="Seasons back (default: 10)")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Daemon interval (min)")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    # Determine sources
    if args.source:
        sources = [args.source]
    elif args.full:
        sources = FULL_ML_SOURCES
    elif args.all:
        sources = ALL_SOURCES
    elif args.historical:
        sources = HISTORICAL_SOURCES
    else:
        sources = CURRENT_SOURCES
    
    # Determine sports
    sports = None
    if args.sport:
        sports = [args.sport.upper()]
    elif args.sports:
        sports = [s.strip().upper() for s in args.sports.split(",")]
    
    # Run
    if args.daemon:
        asyncio.run(daemon_mode(args.interval, sources, sports))
    else:
        asyncio.run(run_import(sources, sports, args.pages, args.days, args.seasons))


if __name__ == "__main__":
    main()