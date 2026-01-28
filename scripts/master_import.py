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
    nflfastr      → games, teams, players, player_stats, team_stats (NFL)
    cfbfastr      → games, teams, players, player_stats, team_stats (NCAAF)
    baseballr     → games, teams, players, player_stats, team_stats (MLB)
    hockeyr       → games, teams, players, player_stats, team_stats (NHL)
    wehoop        → games, teams, players, player_stats, team_stats (WNBA)
"""

import asyncio
import argparse
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, field

# Setup logging for this module
logger = logging.getLogger(__name__)

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
    """Import odds from TheOddsAPI for all 10 sports.
    
    Note: Tennis (ATP/WTA) uses tournament-specific endpoints and is
    collected separately using collect_tennis() method.
    """
    result = ImportResult(source="odds_api")
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        # Main sports supported by OddsAPI (excludes tennis - handled separately)
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        sports_to_collect = sports or main_sports
        
        # Collect main sports
        for sport in sports_to_collect:
            try:
                data = await odds_collector.collect(sport_code=sport)
                if data.success:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        # Save odds to database
                        await odds_collector.save_to_database(data.data, session)
                        # Track line movements
                        movements = await odds_collector.track_line_movements(data.data, session)
                        result.records += movements
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        
        # Collect tennis (tournament-specific endpoints)
        if not sports or "ATP" in sports or "WTA" in sports:
            try:
                tennis_data = await odds_collector.collect_tennis()
                if tennis_data.success and tennis_data.records_count > 0:
                    result.records += tennis_data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await odds_collector.save_to_database(tennis_data.data, session)
                        movements = await odds_collector.track_line_movements(tennis_data.data, session)
                        result.records += movements
                    logger.info(f"[Tennis] Collected {tennis_data.records_count} records from active tournaments")
                else:
                    logger.info("[Tennis] No active tournaments found")
            except Exception as e:
                result.errors.append(f"Tennis: {str(e)[:50]}")
        
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_pinnacle(sports: List[str] = None) -> ImportResult:
    """Import Pinnacle odds with line movements and closing lines.
    
    Fills: odds, odds_movements, sportsbooks tables
    Sports: NFL, NBA, NHL, MLB, NCAAF, NCAAB, WNBA, CFL, ATP, WTA
    """
    result = ImportResult(source="pinnacle")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        # All 10 supported sports
        ALL_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL", "ATP", "WTA"]
        sports_to_collect = sports if sports else ALL_SPORTS
        
        await db_manager.initialize()
        
        logger.info(f"[Pinnacle] Starting collection for {len(sports_to_collect)} sports")
        
        for sport in sports_to_collect:
            try:
                data = await pinnacle_collector.collect(sport_code=sport)
                if data.success:
                    result.records += data.records_count
                    async with db_manager.session() as session:
                        # Save odds to database
                        saved = await pinnacle_collector.save_to_database(data.data, session)
                        logger.info(f"[Pinnacle] {sport}: Saved {saved} odds records")
                        
                        # Track line movements
                        movements = await pinnacle_collector.track_line_movements(data.data, session)
                        result.records += movements
                        logger.info(f"[Pinnacle] {sport}: Tracked {movements} line movements")
                else:
                    logger.warning(f"[Pinnacle] {sport}: No data collected")
            except Exception as e:
                error_msg = f"{sport}: {str(e)[:50]}"
                result.errors.append(error_msg)
                logger.error(f"[Pinnacle] {error_msg}")
        
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        logger.error(f"[Pinnacle] Fatal error: {e}")
    return result


async def import_pinnacle_history(sports: List[str] = None, pages: int = 100) -> ImportResult:
    """Import historical Pinnacle game results (scores/outcomes only, NOT historical odds).
    
    Fills: games table with historical results for ML training labels
    Sports: NFL, NBA, NHL, MLB, NCAAF, NCAAB, WNBA, CFL, ATP, WTA
    
    IMPORTANT: This provides game RESULTS for ML training, not historical odds.
    Historical odds can only be built by running collect() continuously over time.
    """
    result = ImportResult(source="pinnacle_history")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        # All 10 supported sports
        ALL_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL", "ATP", "WTA"]
        sports_to_collect = sports if sports else ALL_SPORTS
        
        await db_manager.initialize()
        
        logger.info(f"[Pinnacle History] Starting collection for {len(sports_to_collect)} sports (max {pages} pages each)")
        
        for sport in sports_to_collect:
            try:
                data = await pinnacle_collector.collect_historical(
                    sport_code=sport, 
                    max_pages=pages
                )
                if data.success:
                    result.records += data.records_count
                    async with db_manager.session() as session:
                        saved, updated = await pinnacle_collector.save_historical_to_database(
                            data.data, sport, session
                        )
                        logger.info(f"[Pinnacle History] {sport}: {saved} new, {updated} updated games")
                else:
                    logger.warning(f"[Pinnacle History] {sport}: No data collected")
            except Exception as e:
                error_msg = f"{sport}: {str(e)[:50]}"
                result.errors.append(error_msg)
                logger.error(f"[Pinnacle History] {error_msg}")
        
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        logger.error(f"[Pinnacle History] Fatal error: {e}")
    return result


async def import_pinnacle_closing_lines(sports: List[str] = None) -> ImportResult:
    """Capture closing lines from Pinnacle for games that have started.
    
    Fills: closing_lines table (benchmark for CLV calculation)
    Sports: NFL, NBA, NHL, MLB, NCAAF, NCAAB, WNBA, CFL, ATP, WTA
    
    Run this periodically to capture final lines before game starts.
    """
    result = ImportResult(source="closing_lines")
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        # All 10 supported sports
        ALL_SPORTS = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL", "ATP", "WTA"]
        sports_to_collect = sports if sports else ALL_SPORTS
        
        await db_manager.initialize()
        
        logger.info(f"[Closing Lines] Capturing closing lines for {len(sports_to_collect)} sports")
        
        for sport in sports_to_collect:
            try:
                async with db_manager.session() as session:
                    saved = await pinnacle_collector.capture_closing_lines(sport, session)
                    result.records += saved
                    if saved > 0:
                        logger.info(f"[Closing Lines] {sport}: Captured {saved} closing lines")
            except Exception as e:
                error_msg = f"{sport}: {str(e)[:50]}"
                result.errors.append(error_msg)
                logger.error(f"[Closing Lines] {error_msg}")
        
        result.success = result.records >= 0  # Success even if no new lines captured
    except Exception as e:
        result.errors.append(str(e)[:100])
        logger.error(f"[Closing Lines] Fatal error: {e}")
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
    """Import OddsAPI historical odds (requires paid subscription $119+/month).
    
    Note: Tennis historical data uses tournament-specific endpoints and 
    is not currently supported in this function.
    """
    result = ImportResult(source="odds_api_history")
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        # Main sports (excludes tennis - requires tournament-specific keys)
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        sports_to_collect = sports if sports else main_sports
        # Filter out tennis if specified (not supported for historical)
        sports_to_collect = [s for s in sports_to_collect if s not in ["ATP", "WTA"]]
        
        await db_manager.initialize()
        
        for sport in sports_to_collect:
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
    """Import weather for upcoming outdoor games.
    
    Fills: weather_data table
    Outdoor Sports: NFL, NCAAF, CFL, MLB, ATP, WTA
    
    Note: Indoor sports (NBA, NHL, NCAAB, WNBA) don't need weather data.
    """
    result = ImportResult(source="weather")
    try:
        from app.services.collectors.collector_05_weather import WeatherCollector
        from app.core.database import db_manager
        
        # Outdoor sports that need weather data
        OUTDOOR_SPORTS = ["NFL", "NCAAF", "CFL", "MLB", "ATP", "WTA"]
        
        if sports:
            # Filter to only outdoor sports
            outdoor_sports = [s for s in sports if s.upper() in OUTDOOR_SPORTS]
        else:
            outdoor_sports = OUTDOOR_SPORTS
        
        if not outdoor_sports:
            logger.info("[Weather] No outdoor sports specified, skipping")
            result.success = True
            return result
        
        await db_manager.initialize()
        
        logger.info(f"[Weather] Collecting weather for {len(outdoor_sports)} outdoor sports")
        
        async with WeatherCollector() as collector:
            for sport in outdoor_sports:
                try:
                    weather_stats = await collector.collect_for_upcoming_games(
                        sport_code=sport,
                        days_ahead=days
                    )
                    # weather_stats is CollectorStats object
                    if weather_stats:
                        result.records += weather_stats.weather_saved
                        logger.info(f"[Weather] {sport}: {weather_stats.weather_saved} records saved")
                except Exception as e:
                    error_msg = f"{sport}: {str(e)[:50]}"
                    result.errors.append(error_msg)
                    logger.error(f"[Weather] {error_msg}")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        logger.error(f"[Weather] Fatal error: {e}")
    return result


async def import_weather_history(sports: List[str] = None, days: int = 365) -> ImportResult:
    """Import historical weather for past games using Open-Meteo (FREE).
    
    Fills: weather_data table with historical weather
    Outdoor Sports: NFL, NCAAF, CFL, MLB, ATP, WTA
    
    Uses Open-Meteo API (free, no key required) with data from 1940 to present.
    """
    result = ImportResult(source="weather_history")
    try:
        from app.services.collectors.collector_05_weather import WeatherCollector
        from app.core.database import db_manager
        
        # Outdoor sports that need weather data
        OUTDOOR_SPORTS = ["NFL", "NCAAF", "CFL", "MLB", "ATP", "WTA"]
        
        if sports:
            # Filter to only outdoor sports
            outdoor_sports = [s for s in sports if s.upper() in OUTDOOR_SPORTS]
        else:
            outdoor_sports = OUTDOOR_SPORTS
        
        if not outdoor_sports:
            logger.info("[Weather History] No outdoor sports specified, skipping")
            result.success = True
            return result
        
        await db_manager.initialize()
        
        logger.info(f"[Weather History] Collecting historical weather for {len(outdoor_sports)} sports")
        logger.info(f"[Weather History] Looking back {days} days (~{days//365} years)")
        logger.info(f"[Weather History] Using Open-Meteo API (FREE, no key required)")
        
        async with WeatherCollector() as collector:
            for sport in outdoor_sports:
                try:
                    weather_stats = await collector.collect_historical_for_games(
                        sport_code=sport,
                        days_back=days
                    )
                    # weather_stats is CollectorStats object
                    if weather_stats:
                        result.records += weather_stats.weather_saved
                        logger.info(f"[Weather History] {sport}: {weather_stats.weather_saved} records saved")
                except Exception as e:
                    error_msg = f"{sport}: {str(e)[:50]}"
                    result.errors.append(error_msg)
                    logger.error(f"[Weather History] {error_msg}")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        logger.error(f"[Weather History] Fatal error: {e}")
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
    """Import venues from TheSportsDB and update team cities.
    
    Fills: venues table (with lat/lon for weather)
    Also updates: teams table (city field)
    """
    result = ImportResult(source="venues")
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        # All 10 sports (matching database)
        sports = sports or ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "CFL", "WNBA", "ATP", "WTA"]
        await db_manager.initialize()
        
        all_team_cities = []
        
        for sport in sports:
            try:
                venues_data = await sportsdb_collector.collect_venues(sport_code=sport)
                if venues_data:
                    # Save venues
                    if venues_data.get("venues"):
                        async with db_manager.session() as session:
                            saved = await sportsdb_collector.save_venues_to_database(
                                venues_data["venues"], session
                            )
                            result.records += saved
                    
                    # Collect team cities for later update
                    if venues_data.get("team_cities"):
                        all_team_cities.extend(venues_data["team_cities"])
                        
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        
        # Update team cities from collected data
        if all_team_cities:
            try:
                async with db_manager.session() as session:
                    updated = await sportsdb_collector.update_team_cities_from_venues(
                        all_team_cities, session
                    )
                    logger.info(f"[Venues] Updated {updated} team cities")
            except Exception as e:
                logger.warning(f"[Venues] Team city update error: {e}")
        
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
    """Import NFL historical data from nflfastR (games, players, player_stats, team_stats)."""
    result = ImportResult(source="nflfastr_history")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        from rich.console import Console
        
        console = Console()
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        console.print(f"[bold blue]Collecting NFL data for {min(years)}-{max(years)} ({len(years)} seasons)...[/]")
        
        # Use collect_all for comprehensive data
        data = await nflfastr_collector.collect_all(years=years)
        
        save_results = {
            "games": 0,
            "players": 0,
            "player_stats": 0,
            "team_stats": 0,
        }
        
        if data.success and data.data:
            result.records = data.records_count
            await db_manager.initialize()
            
            # Save each type in a separate session to prevent cascading rollback
            # 1. Games
            if data.data.get("games"):
                try:
                    async with db_manager.session() as session:
                        save_results["games"] = await nflfastr_collector._save_games(data.data["games"], session)
                        console.print(f"[green]✅ Games saved: {save_results['games']}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Games save error: {str(e)[:50]}[/]")
            
            # 2. Players (rosters)
            if data.data.get("players"):
                try:
                    async with db_manager.session() as session:
                        save_results["players"] = await nflfastr_collector.save_rosters_to_database(data.data["players"], session)
                        console.print(f"[green]✅ Players saved: {save_results['players']}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Players save error: {str(e)[:50]}[/]")
            
            # 3. Player stats
            if data.data.get("player_stats"):
                try:
                    async with db_manager.session() as session:
                        save_results["player_stats"] = await nflfastr_collector.save_players_to_database(data.data["player_stats"], session)
                        console.print(f"[green]✅ Player stats saved: {save_results['player_stats']}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Player stats save error: {str(e)[:50]}[/]")
            
            # 4. Team stats
            if data.data.get("team_stats"):
                try:
                    async with db_manager.session() as session:
                        save_results["team_stats"] = await nflfastr_collector.save_team_stats_to_database(data.data["team_stats"], session)
                        console.print(f"[green]✅ Team stats saved: {save_results['team_stats']}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Team stats save error: {str(e)[:50]}[/]")
            
            result.records = sum(save_results.values())
        
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
    """Import NFL players (rosters + stats) from nflfastR."""
    result = ImportResult(source="nfl_players")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        from rich.console import Console
        
        console = Console()
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 9, current_year + 1))  # 10 years of rosters
        
        console.print(f"[bold blue]Collecting NFL rosters and player stats for {min(years)}-{max(years)}...[/]")
        
        total_saved = 0
        
        # 1. Collect and save rosters (player base info)
        roster_data = await nflfastr_collector.collect_rosters(years=years)
        if roster_data.success and roster_data.data:
            async with db_manager.session() as session:
                saved = await nflfastr_collector.save_rosters_to_database(
                    roster_data.data.get("players", []), session
                )
                total_saved += saved
                console.print(f"[green]✅ Players from rosters: {saved}[/]")
        
        # 2. Collect and save player stats
        stats_data = await nflfastr_collector.collect(years=years, collect_type="player_stats")
        if stats_data.success and stats_data.data:
            async with db_manager.session() as session:
                saved = await nflfastr_collector.save_players_to_database(
                    stats_data.data.get("player_stats", []), session
                )
                total_saved += saved
                console.print(f"[green]✅ Player stats records: {saved}[/]")
        
        result.records = total_saved
        result.success = result.records > 0
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


async def import_nflfastr_team_stats() -> ImportResult:
    """Import NFL team stats (EPA, success rates) from nflfastR."""
    result = ImportResult(source="nfl_team_stats")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        from rich.console import Console
        
        console = Console()
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 9, current_year + 1))
        
        console.print(f"[bold blue]Collecting NFL team stats for {min(years)}-{max(years)}...[/]")
        
        data = await nflfastr_collector.collect(years=years, collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await nflfastr_collector.save_team_stats_to_database(
                    data.data.get("team_stats", []), session
                )
                result.records = saved
                console.print(f"[green]✅ Team stats saved: {saved}[/]")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nflfastr_rosters() -> ImportResult:
    """Import NFL rosters from nflfastR."""
    result = ImportResult(source="nfl_rosters")
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        from rich.console import Console
        
        console = Console()
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 9, current_year + 1))
        
        console.print(f"[bold blue]Collecting NFL rosters for {min(years)}-{max(years)}...[/]")
        
        data = await nflfastr_collector.collect_rosters(years=years)
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await nflfastr_collector.save_rosters_to_database(
                    data.data.get("players", []), session
                )
                result.records = saved
                console.print(f"[green]✅ Players from rosters: {saved}[/]")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# BASEBALLR (MLB Stats API + pybaseball)
# =============================================================================

async def import_baseballr(sports: List[str] = None) -> ImportResult:
    """Import MLB current data from baseballR/MLB Stats API."""
    result = ImportResult(source="baseballr")
    try:
        from app.services.collectors import baseballr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 1, current_year + 1))
        
        data = await baseballr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await baseballr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_baseballr_history(years_back: int = 10) -> ImportResult:
    """Import MLB historical data from baseballR/MLB Stats API."""
    result = ImportResult(source="baseballr_history")
    try:
        from app.services.collectors import baseballr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back, current_year + 1))
        
        data = await baseballr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await baseballr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_baseballr_players() -> ImportResult:
    """Import MLB players and stats from baseballR."""
    result = ImportResult(source="mlb_players")
    try:
        from app.services.collectors import baseballr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 3, current_year + 1))
        
        data = await baseballr_collector.collect(years=years, collect_type="player_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await baseballr_collector.save_players_to_database(
                    data.data.get("player_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_baseballr_rosters() -> ImportResult:
    """Import MLB rosters from baseballR/MLB Stats API."""
    result = ImportResult(source="mlb_rosters")
    try:
        from app.services.collectors import baseballr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await baseballr_collector.collect(collect_type="rosters")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await baseballr_collector.save_rosters_to_database(
                    data.data.get("rosters", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_baseballr_team_stats() -> ImportResult:
    """Import MLB team stats from baseballR."""
    result = ImportResult(source="mlb_team_stats")
    try:
        from app.services.collectors import baseballr_collector
        from app.core.database import db_manager
        from rich.console import Console
        
        console = Console()
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 9, current_year + 1))
        
        console.print(f"[bold blue]Collecting MLB team stats for {min(years)}-{max(years)}...[/]")
        
        data = await baseballr_collector.collect(years=years, collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await baseballr_collector.save_team_stats_to_database(
                    data.data.get("team_stats", []), session
                )
                result.records = saved
                console.print(f"[green]✅ Team stats saved: {saved}[/]")
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# HOCKEYR IMPORTS (NHL)
# =============================================================================

async def import_hockeyr(sports: List[str] = None) -> ImportResult:
    """Import NHL current season data from hockeyR/NHL API."""
    result = ImportResult(source="hockeyr")
    try:
        from app.services.collectors import hockeyr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        # NHL season spans two calendar years (Oct-June)
        if datetime.now().month >= 10:
            years = [current_year]
        else:
            years = [current_year - 1]
        
        data = await hockeyr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await hockeyr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_hockeyr_history(years_back: int = 10) -> ImportResult:
    """Import NHL historical data from hockeyR/NHL API (10 years)."""
    result = ImportResult(source="hockeyr_history")
    try:
        from app.services.collectors import hockeyr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        # NHL season format: 2015 means 2015-16 season
        if datetime.now().month >= 10:
            end_year = current_year
        else:
            end_year = current_year - 1
        
        years = list(range(end_year - years_back + 1, end_year + 1))
        
        logger.info(f"[hockeyR] Collecting {len(years)} seasons: {min(years)}-{min(years)+1} to {max(years)}-{max(years)+1}")
        
        data = await hockeyr_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await hockeyr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_hockeyr_players() -> ImportResult:
    """Import NHL players and stats from hockeyR/NHL API."""
    result = ImportResult(source="nhl_players")
    try:
        from app.services.collectors import hockeyr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 10:
            years = list(range(current_year - 2, current_year + 1))
        else:
            years = list(range(current_year - 3, current_year))
        
        data = await hockeyr_collector.collect(years=years, collect_type="player_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await hockeyr_collector.save_player_stats_to_database(
                    data.data.get("player_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_hockeyr_rosters() -> ImportResult:
    """Import NHL rosters from hockeyR/NHL API."""
    result = ImportResult(source="nhl_rosters")
    try:
        from app.services.collectors import hockeyr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 10:
            years = [current_year]
        else:
            years = [current_year - 1]
        
        data = await hockeyr_collector.collect(years=years, collect_type="rosters")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await hockeyr_collector.save_rosters_to_database(
                    data.data.get("rosters", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_hockeyr_team_stats() -> ImportResult:
    """Import NHL team stats from hockeyR/NHL API."""
    result = ImportResult(source="nhl_team_stats")
    try:
        from app.services.collectors import hockeyr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 10:
            years = list(range(current_year - 9, current_year + 1))
        else:
            years = list(range(current_year - 10, current_year))
        
        logger.info(f"[hockeyR] Collecting NHL team stats for {min(years)}-{max(years)+1}...")
        
        data = await hockeyr_collector.collect(years=years, collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await hockeyr_collector.save_team_stats_to_database(
                    data.data.get("team_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# WEHOOP IMPORTS (WNBA)
# =============================================================================

async def import_wehoop(sports: List[str] = None) -> ImportResult:
    """Import WNBA current season data from wehoop/ESPN API."""
    result = ImportResult(source="wehoop")
    try:
        from app.services.collectors import wehoop_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        # WNBA season runs May-October
        if datetime.now().month >= 5:
            years = [current_year]
        else:
            years = [current_year - 1]
        
        data = await wehoop_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await wehoop_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_wehoop_history(years_back: int = 10) -> ImportResult:
    """Import WNBA historical data from wehoop/ESPN API (10 years)."""
    result = ImportResult(source="wehoop_history")
    try:
        from app.services.collectors import wehoop_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        # WNBA season format: single calendar year
        if datetime.now().month >= 5:
            end_year = current_year
        else:
            end_year = current_year - 1
        
        years = list(range(end_year - years_back + 1, end_year + 1))
        
        logger.info(f"[wehoop] Collecting {len(years)} seasons: {min(years)} to {max(years)}")
        
        data = await wehoop_collector.collect(years=years, collect_type="all")
        if data.success:
            result.records = data.records_count
            async with db_manager.session() as session:
                await wehoop_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_wehoop_players() -> ImportResult:
    """Import WNBA players and stats from wehoop/ESPN API."""
    result = ImportResult(source="wnba_players")
    try:
        from app.services.collectors import wehoop_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 5:
            years = list(range(current_year - 2, current_year + 1))
        else:
            years = list(range(current_year - 3, current_year))
        
        data = await wehoop_collector.collect(years=years, collect_type="player_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await wehoop_collector.save_player_stats_to_database(
                    data.data.get("player_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_wehoop_rosters() -> ImportResult:
    """Import WNBA rosters from wehoop/ESPN API."""
    result = ImportResult(source="wnba_rosters")
    try:
        from app.services.collectors import wehoop_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 5:
            years = [current_year]
        else:
            years = [current_year - 1]
        
        data = await wehoop_collector.collect(years=years, collect_type="rosters")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await wehoop_collector.save_rosters_to_database(
                    data.data.get("rosters", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_wehoop_team_stats() -> ImportResult:
    """Import WNBA team stats from wehoop/ESPN API."""
    result = ImportResult(source="wnba_team_stats")
    try:
        from app.services.collectors import wehoop_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        if datetime.now().month >= 5:
            years = list(range(current_year - 9, current_year + 1))
        else:
            years = list(range(current_year - 10, current_year))
        
        logger.info(f"[wehoop] Collecting WNBA team stats for {min(years)}-{max(years)}...")
        
        data = await wehoop_collector.collect(years=years, collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await wehoop_collector.save_team_stats_to_database(
                    data.data.get("team_stats", []), session
                )
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# HOOPR IMPORTS (NBA + NCAAB)
# =============================================================================

async def import_hoopr(sports: List[str] = None) -> ImportResult:
    """Import NBA and NCAAB current season data from hoopR/ESPN API."""
    result = ImportResult(source="hoopr")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = [current_year]
        leagues = ["NBA", "NCAAB"]
        
        data = await hoopr_collector.collect(years=years, leagues=leagues, collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                await hoopr_collector.save_to_database(data.data, session)
                result.records = data.records_count
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        import traceback
        traceback.print_exc()
    return result


async def import_hoopr_history(years_back: int = 10) -> ImportResult:
    """Import NBA and NCAAB historical data from hoopR/ESPN API (10 years)."""
    result = ImportResult(source="hoopr_history")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        # NBA season spans two years (Oct-Jun), so use calendar year
        years = list(range(current_year - years_back + 1, current_year + 1))
        leagues = ["NBA", "NCAAB"]
        
        logger.info(f"[hoopR] Collecting {len(years)} seasons: {min(years)} to {max(years)}")
        
        data = await hoopr_collector.collect(years=years, leagues=leagues, collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                await hoopr_collector.save_to_database(data.data, session)
                result.records = data.records_count
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        import traceback
        traceback.print_exc()
    return result


async def import_hoopr_nba(years_back: int = 10) -> ImportResult:
    """Import NBA only data from hoopR/ESPN API (10 years)."""
    result = ImportResult(source="hoopr_nba")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back + 1, current_year + 1))
        
        logger.info(f"[hoopR] Collecting NBA data for {len(years)} seasons: {min(years)} to {max(years)}")
        
        data = await hoopr_collector.collect(years=years, leagues=["NBA"], collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                await hoopr_collector.save_to_database(data.data, session)
                result.records = data.records_count
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        import traceback
        traceback.print_exc()
    return result


async def import_hoopr_ncaab(years_back: int = 10) -> ImportResult:
    """Import NCAAB only data from hoopR/ESPN API (10 years)."""
    result = ImportResult(source="hoopr_ncaab")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back + 1, current_year + 1))
        
        logger.info(f"[hoopR] Collecting NCAAB data for {len(years)} seasons: {min(years)} to {max(years)}")
        
        data = await hoopr_collector.collect(years=years, leagues=["NCAAB"], collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                await hoopr_collector.save_to_database(data.data, session)
                result.records = data.records_count
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
        import traceback
        traceback.print_exc()
    return result


async def import_hoopr_players() -> ImportResult:
    """Import NBA and NCAAB players from hoopR/ESPN API."""
    result = ImportResult(source="nba_players")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 4, current_year + 1))  # 5 years of players
        
        data = await hoopr_collector.collect(years=years, leagues=["NBA", "NCAAB"], collect_type="rosters")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await hoopr_collector._save_rosters(session, data.data.get("rosters", []))
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_hoopr_team_stats() -> ImportResult:
    """Import NBA and NCAAB team stats from hoopR/ESPN API."""
    result = ImportResult(source="nba_team_stats")
    try:
        from app.services.collectors import hoopr_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - 9, current_year + 1))  # 10 years
        
        data = await hoopr_collector.collect(years=years, leagues=["NBA", "NCAAB"], collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await hoopr_collector._save_team_stats(session, data.data.get("team_stats", []))
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# CFL OFFICIAL API IMPORT FUNCTIONS
# =============================================================================

async def import_cfl() -> ImportResult:
    """Import current CFL data from CFL Official API."""
    result = ImportResult(source="cfl")
    try:
        from app.services.collectors import cfl_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        
        data = await cfl_collector.collect(years=[current_year], collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfl_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfl_history(years_back: int = 10) -> ImportResult:
    """Import historical CFL data (10 years) from CFL Official API."""
    result = ImportResult(source="cfl_history")
    try:
        from app.services.collectors import cfl_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back + 1, current_year + 1))
        
        logging.info(f"[CFL] Collecting {len(years)} seasons: {years[0]} to {years[-1]}")
        
        data = await cfl_collector.collect(years=years, collect_type="all")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfl_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfl_teams() -> ImportResult:
    """Import CFL teams only (works without API key)."""
    result = ImportResult(source="cfl_teams")
    try:
        from app.services.collectors import cfl_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfl_collector.collect(years=[], collect_type="teams")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfl_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfl_rosters(years_back: int = 5) -> ImportResult:
    """Import CFL player rosters."""
    result = ImportResult(source="cfl_rosters")
    try:
        from app.services.collectors import cfl_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back + 1, current_year + 1))
        
        data = await cfl_collector.collect(years=years, collect_type="rosters")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfl_collector._save_rosters(session, data.data.get("rosters", []))
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfl_standings(years_back: int = 10) -> ImportResult:
    """Import CFL standings/team stats."""
    result = ImportResult(source="cfl_standings")
    try:
        from app.services.collectors import cfl_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        current_year = datetime.now().year
        years = list(range(current_year - years_back + 1, current_year + 1))
        
        data = await cfl_collector.collect(years=years, collect_type="team_stats")
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfl_collector._save_team_stats(session, data.data.get("team_stats", []))
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# ACTION NETWORK IMPORTS
# =============================================================================

async def import_action_network() -> ImportResult:
    """Import current public betting data from Action Network."""
    result = ImportResult(source="action_network")
    try:
        from app.services.collectors import action_network_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await action_network_collector.collect(
            sports=["NFL", "NCAAF", "NBA", "NCAAB", "NHL", "MLB"],
            days_back=0,
            collect_type="current"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await action_network_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_action_network_history(days_back: int = 30) -> ImportResult:
    """Import historical public betting data from Action Network."""
    result = ImportResult(source="action_network_history")
    try:
        from app.services.collectors import action_network_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[ActionNetwork] Collecting {days_back} days of historical data")
        
        data = await action_network_collector.collect(
            sports=["NFL", "NCAAF", "NBA", "NCAAB", "NHL", "MLB"],
            days_back=days_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await action_network_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_public_betting(sports: List[str] = None, days_back: int = 7) -> ImportResult:
    """Import public betting data for specific sports."""
    result = ImportResult(source="public_betting")
    try:
        from app.services.collectors import action_network_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        if sports is None:
            sports = ["NFL", "NCAAF", "NBA", "NCAAB", "NHL", "MLB"]
        
        data = await action_network_collector.collect(
            sports=sports,
            days_back=days_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await action_network_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sharp_money() -> ImportResult:
    """Import and detect sharp money indicators from current public betting."""
    result = ImportResult(source="sharp_money")
    try:
        from app.services.collectors import action_network_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await action_network_collector.collect(
            sports=["NFL", "NCAAF", "NBA", "NCAAB", "NHL", "MLB"],
            days_back=3,
            collect_type="current"
        )
        if data.success and data.data:
            result.records = len(data.data.get("sharp_indicators", []))
            async with db_manager.session() as session:
                await action_network_collector.save_to_database(data.data, session)
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# NHL OFFICIAL API IMPORTS
# =============================================================================

async def import_nhl_api() -> ImportResult:
    """Import current NHL EDGE stats from NHL Official API."""
    result = ImportResult(source="nhl_api")
    try:
        from app.services.collectors import nhl_official_api_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        # Current season only
        data = await nhl_official_api_collector.collect(
            years_back=1,
            collect_type="all",
            game_type=2  # Regular season
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await nhl_official_api_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_nhl_api_history(years_back: int = 10) -> ImportResult:
    """Import historical NHL EDGE stats (10 years)."""
    result = ImportResult(source="nhl_api_history")
    try:
        from app.services.collectors import nhl_official_api_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[NHL API] Collecting {years_back} years of EDGE data")
        
        data = await nhl_official_api_collector.collect(
            years_back=years_back,
            collect_type="all",
            game_type=2  # Regular season
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await nhl_official_api_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# SPORTSIPY IMPORTS (Sports-Reference Scraper)
# =============================================================================

async def import_sportsipy() -> ImportResult:
    """Import current season data from Sports-Reference via sportsipy (all 6 sports)."""
    result = ImportResult(source="sportsipy")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        # Current season only, all sports
        data = await sportsipy_collector.collect(
            sports=["MLB", "NBA", "NFL", "NHL", "NCAAF", "NCAAB"],
            years_back=1,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_history(years_back: int = 10) -> ImportResult:
    """Import 10 years historical data from Sports-Reference (all 6 sports)."""
    result = ImportResult(source="sportsipy_history")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[Sportsipy] Collecting {years_back} years of historical data for all sports")
        
        data = await sportsipy_collector.collect(
            sports=["MLB", "NBA", "NFL", "NHL", "NCAAF", "NCAAB"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# BASKETBALL REFERENCE IMPORTS (NBA Scraper)
# =============================================================================

async def import_basketball_ref() -> ImportResult:
    """Import current NBA data from Basketball-Reference."""
    result = ImportResult(source="basketball_ref")
    try:
        from app.services.collectors import basketball_ref_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await basketball_ref_collector.collect(
            years_back=1,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await basketball_ref_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_basketball_ref_history(years_back: int = 10) -> ImportResult:
    """Import 10 years NBA data from Basketball-Reference."""
    result = ImportResult(source="basketball_ref_history")
    try:
        from app.services.collectors import basketball_ref_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[BasketballRef] Collecting {years_back} years of NBA data")
        
        data = await basketball_ref_collector.collect(
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await basketball_ref_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_basketball_ref_teams() -> ImportResult:
    """Import NBA teams from Basketball-Reference."""
    result = ImportResult(source="basketball_ref_teams")
    try:
        from app.services.collectors import basketball_ref_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await basketball_ref_collector.collect(
            years_back=1,
            collect_type="teams"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await basketball_ref_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_basketball_ref_injuries() -> ImportResult:
    """Import current NBA injury report from Basketball-Reference."""
    result = ImportResult(source="basketball_ref_injuries")
    try:
        from app.services.collectors import basketball_ref_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await basketball_ref_collector.collect(
            years_back=1,
            collect_type="injuries"
        )
        if data.success and data.data:
            # Injuries are returned but not saved to DB (would need injury model)
            result.records = len(data.data.get("injuries", []))
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# COLLEGE FOOTBALL DATA API IMPORTS (NCAAF Advanced)
# =============================================================================

async def import_cfbd() -> ImportResult:
    """Import current NCAAF data from College Football Data API."""
    result = ImportResult(source="cfbd")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=1,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_history(years_back: int = 10) -> ImportResult:
    """Import 10 years NCAAF data from College Football Data API."""
    result = ImportResult(source="cfbd_history")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[CFBD] Collecting {years_back} years of NCAAF data")
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_teams() -> ImportResult:
    """Import NCAAF teams from College Football Data API."""
    result = ImportResult(source="cfbd_teams")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=1,
            collect_type="teams"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_games(years_back: int = 10) -> ImportResult:
    """Import NCAAF games from College Football Data API."""
    result = ImportResult(source="cfbd_games")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="games"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_stats(years_back: int = 10) -> ImportResult:
    """Import NCAAF team/player stats from College Football Data API."""
    result = ImportResult(source="cfbd_stats")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="stats"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_ratings(years_back: int = 10) -> ImportResult:
    """Import NCAAF ratings (SP+, SRS, talent) from College Football Data API."""
    result = ImportResult(source="cfbd_ratings")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="ratings"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_recruiting(years_back: int = 10) -> ImportResult:
    """Import NCAAF recruiting from College Football Data API."""
    result = ImportResult(source="cfbd_recruiting")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="recruiting"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_cfbd_lines(years_back: int = 10) -> ImportResult:
    """Import NCAAF betting lines from College Football Data API."""
    result = ImportResult(source="cfbd_lines")
    try:
        from app.services.collectors import cfbd_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await cfbd_collector.collect(
            years_back=years_back,
            collect_type="lines"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await cfbd_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# MATCHSTAT TENNIS API IMPORTS (ATP/WTA)
# =============================================================================

async def import_matchstat() -> ImportResult:
    """Import current ATP/WTA tennis data from Matchstat API."""
    result = ImportResult(source="matchstat")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=1,
            collect_type="all",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_history(years_back: int = 10) -> ImportResult:
    """Import 10 years ATP/WTA tennis data from Matchstat API."""
    result = ImportResult(source="matchstat_history")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        logging.info(f"[Matchstat] Collecting {years_back} years of ATP/WTA data")
        
        data = await matchstat_collector.collect(
            years_back=years_back,
            collect_type="all",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
        if data.error:
            result.errors.append(data.error[:100])
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_rankings() -> ImportResult:
    """Import current ATP/WTA rankings from Matchstat API."""
    result = ImportResult(source="matchstat_rankings")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=1,
            collect_type="rankings",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_players() -> ImportResult:
    """Import ATP/WTA player profiles from Matchstat API."""
    result = ImportResult(source="matchstat_players")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=1,
            collect_type="players",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_matches(years_back: int = 10) -> ImportResult:
    """Import ATP/WTA match results from Matchstat API."""
    result = ImportResult(source="matchstat_matches")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=years_back,
            collect_type="matches",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_stats() -> ImportResult:
    """Import ATP/WTA player stats from Matchstat API."""
    result = ImportResult(source="matchstat_stats")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=1,
            collect_type="stats",
            tours=["ATP", "WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_atp(years_back: int = 10) -> ImportResult:
    """Import ATP only data from Matchstat API."""
    result = ImportResult(source="matchstat_atp")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=years_back,
            collect_type="all",
            tours=["ATP"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_matchstat_wta(years_back: int = 10) -> ImportResult:
    """Import WTA only data from Matchstat API."""
    result = ImportResult(source="matchstat_wta")
    try:
        from app.services.collectors import matchstat_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await matchstat_collector.collect(
            years_back=years_back,
            collect_type="all",
            tours=["WTA"]
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await matchstat_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_mlb(years_back: int = 10) -> ImportResult:
    """Import MLB data from Sports-Reference."""
    result = ImportResult(source="sportsipy_mlb")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["MLB"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_nba(years_back: int = 10) -> ImportResult:
    """Import NBA data from Sports-Reference."""
    result = ImportResult(source="sportsipy_nba")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["NBA"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_nfl(years_back: int = 10) -> ImportResult:
    """Import NFL data from Sports-Reference."""
    result = ImportResult(source="sportsipy_nfl")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["NFL"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_nhl(years_back: int = 10) -> ImportResult:
    """Import NHL data from Sports-Reference."""
    result = ImportResult(source="sportsipy_nhl")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["NHL"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_ncaaf(years_back: int = 10) -> ImportResult:
    """Import NCAAF data from Sports-Reference."""
    result = ImportResult(source="sportsipy_ncaaf")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["NCAAF"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_ncaab(years_back: int = 10) -> ImportResult:
    """Import NCAAB data from Sports-Reference."""
    result = ImportResult(source="sportsipy_ncaab")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["NCAAB"],
            years_back=years_back,
            collect_type="all"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_teams(sports: List[str] = None) -> ImportResult:
    """Import teams only from Sports-Reference."""
    result = ImportResult(source="sportsipy_teams")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        if sports is None:
            sports = ["MLB", "NBA", "NFL", "NHL", "NCAAF", "NCAAB"]
        
        data = await sportsipy_collector.collect(
            sports=sports,
            years_back=1,
            collect_type="teams"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
                result.records = saved
        
        result.success = result.records >= 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsipy_stats(years_back: int = 10) -> ImportResult:
    """Import team stats only from Sports-Reference."""
    result = ImportResult(source="sportsipy_stats")
    try:
        from app.services.collectors import sportsipy_collector
        from app.core.database import db_manager
        
        await db_manager.initialize()
        
        data = await sportsipy_collector.collect(
            sports=["MLB", "NBA", "NFL", "NHL", "NCAAF", "NCAAB"],
            years_back=years_back,
            collect_type="stats"
        )
        if data.success and data.data:
            async with db_manager.session() as session:
                saved = await sportsipy_collector.save_to_database(data.data, session)
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
    "baseballr": import_baseballr,
    "hockeyr": import_hockeyr,
    "wehoop": import_wehoop,
    "hoopr": import_hoopr,
    "cfl": import_cfl,
    "action_network": import_action_network,
    "public_betting": import_public_betting,
    "sharp_money": import_sharp_money,
    "nhl_api": import_nhl_api,
    "sportsipy": import_sportsipy,
    "basketball_ref": import_basketball_ref,
    "cfbd": import_cfbd,
    "matchstat": import_matchstat,
    
    # Historical data
    "pinnacle_history": import_pinnacle_history,
    "espn_history": import_espn_history,
    "odds_api_history": import_odds_api_history,
    "sportsdb_history": import_sportsdb_history,
    "nflfastr_history": import_nflfastr_history,
    "cfbfastr_history": import_cfbfastr_history,
    "baseballr_history": import_baseballr_history,
    "hockeyr_history": import_hockeyr_history,
    "wehoop_history": import_wehoop_history,
    "hoopr_history": import_hoopr_history,
    "cfl_history": import_cfl_history,
    "action_network_history": import_action_network_history,
    "nhl_api_history": import_nhl_api_history,
    "weather_history": import_weather_history,
    "sportsipy_history": import_sportsipy_history,
    "basketball_ref_history": import_basketball_ref_history,
    "cfbd_history": import_cfbd_history,
    "matchstat_history": import_matchstat_history,
    
    # Specialized data
    "injuries": import_espn_injuries,
    "players": import_espn_players,
    "nfl_players": import_nflfastr_players,
    "nfl_rosters": import_nflfastr_rosters,
    "nfl_team_stats": import_nflfastr_team_stats,
    "ncaaf_players": import_cfbfastr_players,
    "mlb_players": import_baseballr_players,
    "mlb_rosters": import_baseballr_rosters,
    "mlb_team_stats": import_baseballr_team_stats,
    "nhl_players": import_hockeyr_players,
    "nhl_rosters": import_hockeyr_rosters,
    "nhl_team_stats": import_hockeyr_team_stats,
    "wnba_players": import_wehoop_players,
    "wnba_rosters": import_wehoop_rosters,
    "wnba_team_stats": import_wehoop_team_stats,
    "nba_players": import_hoopr_players,
    "nba_team_stats": import_hoopr_team_stats,
    "hoopr_nba": import_hoopr_nba,
    "hoopr_ncaab": import_hoopr_ncaab,
    "cfl_teams": import_cfl_teams,
    "cfl_rosters": import_cfl_rosters,
    "cfl_standings": import_cfl_standings,
    "venues": import_sportsdb_venues,
    "sportsdb_players": import_sportsdb_players,
    "sportsdb_standings": import_sportsdb_standings,
    "sportsdb_seasons": import_sportsdb_seasons,
    "closing_lines": import_pinnacle_closing_lines,
    
    # Sportsipy individual sports
    "sportsipy_mlb": import_sportsipy_mlb,
    "sportsipy_nba": import_sportsipy_nba,
    "sportsipy_nfl": import_sportsipy_nfl,
    "sportsipy_nhl": import_sportsipy_nhl,
    "sportsipy_ncaaf": import_sportsipy_ncaaf,
    "sportsipy_ncaab": import_sportsipy_ncaab,
    "sportsipy_teams": import_sportsipy_teams,
    "sportsipy_stats": import_sportsipy_stats,
    
    # Basketball Reference
    "basketball_ref_teams": import_basketball_ref_teams,
    "basketball_ref_injuries": import_basketball_ref_injuries,
    
    # College Football Data
    "cfbd_teams": import_cfbd_teams,
    "cfbd_games": import_cfbd_games,
    "cfbd_stats": import_cfbd_stats,
    "cfbd_ratings": import_cfbd_ratings,
    "cfbd_recruiting": import_cfbd_recruiting,
    "cfbd_lines": import_cfbd_lines,
    
    # Matchstat Tennis
    "matchstat_rankings": import_matchstat_rankings,
    "matchstat_players": import_matchstat_players,
    "matchstat_matches": import_matchstat_matches,
    "matchstat_stats": import_matchstat_stats,
    "matchstat_atp": import_matchstat_atp,
    "matchstat_wta": import_matchstat_wta,
    
    # Live data
    "sportsdb_live": import_sportsdb_livescores,
    
    # Play-by-play
    "nflfastr_pbp": import_nflfastr_pbp,
    "cfbfastr_pbp": import_cfbfastr_pbp,
    "cfbfastr_sp": import_cfbfastr_sp,
    "cfbfastr_recruiting": import_cfbfastr_recruiting,
}

# Source groups
CURRENT_SOURCES = ["espn", "odds_api", "pinnacle", "weather", "sportsdb", "nflfastr", "cfbfastr", "baseballr", "hockeyr", "wehoop", "hoopr", "cfl", "action_network", "nhl_api", "sportsipy", "basketball_ref", "cfbd", "matchstat"]
HISTORICAL_SOURCES = ["pinnacle_history", "espn_history", "odds_api_history", "sportsdb_history", "nflfastr_history", "cfbfastr_history", "baseballr_history", "hockeyr_history", "wehoop_history", "hoopr_history", "cfl_history", "action_network_history", "nhl_api_history", "weather_history", "sportsipy_history", "basketball_ref_history", "cfbd_history", "matchstat_history"]
PLAYER_SOURCES = ["injuries", "players", "nfl_players", "ncaaf_players", "mlb_players", "nhl_players", "wnba_players", "nba_players", "cfl_rosters", "matchstat_players"]
SPECIALIZED_SOURCES = ["venues", "closing_lines", "sportsdb_players", "sportsdb_standings", "sportsdb_seasons", "mlb_rosters", "mlb_team_stats", "nhl_rosters", "nhl_team_stats", "wnba_rosters", "wnba_team_stats", "nba_team_stats", "hoopr_nba", "hoopr_ncaab", "cfl_teams", "cfl_standings", "sportsipy_mlb", "sportsipy_nba", "sportsipy_nfl", "sportsipy_nhl", "sportsipy_ncaaf", "sportsipy_ncaab", "sportsipy_teams", "sportsipy_stats", "basketball_ref_teams", "basketball_ref_injuries", "cfbd_teams", "cfbd_games", "cfbd_stats", "cfbd_ratings", "cfbd_recruiting", "cfbd_lines", "matchstat_rankings", "matchstat_matches", "matchstat_stats", "matchstat_atp", "matchstat_wta"]

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

async def run_import(sources: List[str], sports: List[str] = None, pages: int = 100, days: int = 30, seasons: int = 10):
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
            elif source in ["nflfastr_history", "cfbfastr_history", "baseballr_history", "hockeyr_history", "wehoop_history", "hoopr_history", "hoopr_nba", "hoopr_ncaab", "cfl_history", "cfl_rosters", "cfl_standings", "nhl_api_history"]:
                result = await func(years_back=seasons)
            elif source in ["sportsipy_history", "sportsipy_mlb", "sportsipy_nba", "sportsipy_nfl", "sportsipy_nhl", "sportsipy_ncaaf", "sportsipy_ncaab", "sportsipy_stats", "basketball_ref_history", "cfbd_history", "cfbd_games", "cfbd_stats", "cfbd_ratings", "cfbd_recruiting", "cfbd_lines", "matchstat_history", "matchstat_matches", "matchstat_atp", "matchstat_wta"]:
                result = await func(years_back=seasons)
            elif source == "action_network_history":
                result = await func(days_back=days)
            elif source in ["public_betting"]:
                result = await func(sports=sports, days_back=days)
            elif source == "weather_history":
                result = await func(sports=sports, days=days)
            elif source in ["sportsdb_live", "nflfastr_pbp", "cfbfastr_pbp", 
                           "cfbfastr_sp", "cfbfastr_recruiting", "closing_lines",
                           "nfl_players", "ncaaf_players", "mlb_players", 
                           "mlb_rosters", "mlb_team_stats",
                           "nhl_players", "nhl_rosters", "nhl_team_stats",
                           "wnba_players", "wnba_rosters", "wnba_team_stats",
                           "nba_players", "nba_team_stats",
                           "cfl_teams",
                           "action_network", "sharp_money", "nhl_api",
                           "sportsipy", "sportsipy_teams",
                           "basketball_ref", "basketball_ref_teams", "basketball_ref_injuries",
                           "cfbd", "cfbd_teams",
                           "matchstat", "matchstat_rankings", "matchstat_players", "matchstat_stats"]:
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
    console.print("  • baseballr     → games, teams (MLB)")
    console.print("  • hockeyr       → games, teams (NHL)")
    console.print("  • wehoop        → games, teams (WNBA)")
    console.print("  • hoopr         → games, teams (NBA, NCAAB)")
    
    console.print("\n[green]✅ HISTORICAL DATA (--historical):[/green]")
    console.print("  • pinnacle_history  → games (archived)")
    console.print("  • espn_history      → games (past N days)")
    console.print("  • odds_api_history  → odds (past N days)")
    console.print("  • sportsdb_history  → games (past N seasons)")
    console.print("  • nflfastr_history  → games (1999-present)")
    console.print("  • cfbfastr_history  → games (2002-present)")
    console.print("  • baseballr_history → games (2016-present)")
    console.print("  • hockeyr_history   → games (2016-present)")
    console.print("  • wehoop_history    → games (2016-present)")
    console.print("  • hoopr_history     → games (2016-present, NBA + NCAAB)")
    console.print("  • hoopr_nba         → games (2016-present, NBA only)")
    console.print("  • hoopr_ncaab       → games (2016-present, NCAAB only)")
    
    console.print("\n[cyan]⚡ SPECIALIZED DATA:[/cyan]")
    console.print("  • injuries      → injuries (from ESPN)")
    console.print("  • players       → players (from ESPN)")
    console.print("  • nfl_players   → players, player_stats (from nflfastR)")
    console.print("  • ncaaf_players → players, player_stats (from cfbfastR)")
    console.print("  • mlb_players   → players, player_stats (from baseballR)")
    console.print("  • nhl_players   → players, player_stats (from hockeyR)")
    console.print("  • wnba_players  → players, player_stats (from wehoop)")
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
    parser.add_argument("--pages", "-p", type=int, default=100, help="Pinnacle history pages per sport (100 events/page)")
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