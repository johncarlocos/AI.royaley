#!/usr/bin/env python3
"""
ROYALEY - Master Data Import
=============================

ONE COMMAND to import all data.

Usage:
    python scripts/master_import.py --current          # Current data (default)
    python scripts/master_import.py --historical       # Historical only
    python scripts/master_import.py --all              # Both
    python scripts/master_import.py --source pinnacle  # Specific source
    python scripts/master_import.py --sport NFL        # Specific sport
    python scripts/master_import.py --daemon           # Run continuously
    python scripts/master_import.py --status           # Show status
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import List
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
    success: bool
    records: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# IMPORT FUNCTIONS
# =============================================================================

async def import_espn(sports: List[str] = None) -> ImportResult:
    """Import games/scores from ESPN."""
    result = ImportResult(source="espn", success=False)
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await espn_collector.collect(sport_code=sport)
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        # ESPN returns {games: [], scores: [], teams: []}
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


async def import_odds_api(sports: List[str] = None) -> ImportResult:
    """Import odds from TheOddsAPI."""
    result = ImportResult(source="odds_api", success=False)
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await odds_collector.collect(sport_code=sport)
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await odds_collector.save_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await odds_collector.close()
        except:
            pass
    return result


async def import_pinnacle(sports: List[str] = None) -> ImportResult:
    """Import CLV lines from Pinnacle."""
    result = ImportResult(source="pinnacle", success=False)
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await pinnacle_collector.collect(sport_code=sport)
                if data.success and data.data:
                    result.records += data.records_count
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await pinnacle_collector.save_to_database(data.data, session)
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await pinnacle_collector.close()
        except:
            pass
    return result


async def import_pinnacle_history(sports: List[str] = None, pages: int = 50) -> ImportResult:
    """Import historical results from Pinnacle archive."""
    result = ImportResult(source="pinnacle_history", success=False)
    try:
        from app.services.collectors import pinnacle_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await pinnacle_collector.collect_historical(sport_code=sport, max_pages=pages)
                if data.success and data.data:
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        saved, updated = await pinnacle_collector.save_historical_to_database(
                            data.data, sport, session
                        )
                        result.records += saved + updated
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await pinnacle_collector.close()
        except:
            pass
    return result


async def import_espn_history(sports: List[str] = None, days: int = 365) -> ImportResult:
    """Import historical games from ESPN."""
    result = ImportResult(source="espn_history", success=False)
    try:
        from app.services.collectors import espn_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await espn_collector.collect_historical(sport_code=sport, days_back=days)
                if data.success and data.data:
                    games = data.data.get("games", [])
                    if games:
                        await db_manager.initialize()
                        async with db_manager.session() as session:
                            saved, updated = await espn_collector.save_historical_to_database(
                                games, sport, session
                            )
                            result.records += saved + updated
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await espn_collector.close()
        except:
            pass
    return result


async def import_odds_api_history(sports: List[str] = None, days: int = 30) -> ImportResult:
    """Import historical odds from OddsAPI (requires paid subscription)."""
    result = ImportResult(source="odds_api_history", success=False)
    try:
        from app.services.collectors import odds_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await odds_collector.collect_historical(sport_code=sport, days_back=days)
                if data.success and data.data:
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        saved, updated = await odds_collector.save_historical_to_database(
                            data.data, session
                        )
                        result.records += saved + updated
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await odds_collector.close()
        except:
            pass
    return result


async def import_weather(sports: List[str] = None) -> ImportResult:
    """Import weather data for outdoor games."""
    result = ImportResult(source="weather", success=False)
    try:
        from app.services.collectors.collector_05_weather import WeatherCollector
        
        sports = sports or ["NFL", "MLB"]
        for sport in sports:
            try:
                async with WeatherCollector() as collector:
                    stats = await collector.collect_for_upcoming_games(sport_code=sport, days_ahead=7)
                    result.records += stats.weather_fetched
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


async def import_sportsdb(sports: List[str] = None) -> ImportResult:
    """Import games/scores/livescores from TheSportsDB."""
    result = ImportResult(source="sportsdb", success=False)
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
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
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await sportsdb_collector.close()
        except:
            pass
    return result


async def import_sportsdb_history(sports: List[str] = None, seasons: int = 10) -> ImportResult:
    """Import historical game results from TheSportsDB by season (default: 10 years)."""
    result = ImportResult(source="sportsdb_history", success=False)
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        sports = sports or ["NFL", "NBA", "NHL", "MLB"]
        for sport in sports:
            try:
                data = await sportsdb_collector.collect_historical(
                    sport_code=sport, 
                    seasons_back=seasons
                )
                if data.success and data.data:
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        saved, updated = await sportsdb_collector.save_historical_to_database(
                            data.data, session
                        )
                        result.records += saved + updated
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await sportsdb_collector.close()
        except:
            pass
    return result


async def import_sportsdb_livescores() -> ImportResult:
    """Import all current livescores from TheSportsDB."""
    result = ImportResult(source="sportsdb_live", success=False)
    try:
        from app.services.collectors import sportsdb_collector
        from app.core.database import db_manager
        
        data = await sportsdb_collector.collect_all_livescores()
        if data.success and data.data:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await sportsdb_collector._update_livescores(data.data, session)
        result.success = result.records > 0 or data.success
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await sportsdb_collector.close()
        except:
            pass
    return result


async def import_nflfastr(sports: List[str] = None) -> ImportResult:
    """Import NFL games/schedules from nflfastR (FREE - 1999-present)."""
    result = ImportResult(source="nflfastr", success=False)
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        
        # nflfastR is NFL only
        data = await nflfastr_collector.collect(sport_code="NFL", collect_type="schedules")
        if data.success and data.data:
            result.records = data.records_count
            await db_manager.initialize()
            async with db_manager.session() as session:
                await nflfastr_collector.save_to_database(data.data, session)
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await nflfastr_collector.close()
        except:
            pass
    return result


async def import_nflfastr_history(years_back: int = 10) -> ImportResult:
    """Import historical NFL data from nflfastR (1999-present, FREE)."""
    result = ImportResult(source="nflfastr_history", success=False)
    try:
        from app.services.collectors import nflfastr_collector
        from app.core.database import db_manager
        
        data = await nflfastr_collector.collect_historical(
            years_back=years_back,
            data_types=["schedules"]
        )
        if data.success and data.data:
            await db_manager.initialize()
            async with db_manager.session() as session:
                saved, updated = await nflfastr_collector.save_historical_to_database(
                    data.data.get("games", []), session
                )
                result.records = saved + updated
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await nflfastr_collector.close()
        except:
            pass
    return result


async def import_nflfastr_pbp(years: List[int] = None) -> ImportResult:
    """Import NFL play-by-play data with EPA, WPA, CPOE (large files!)."""
    result = ImportResult(source="nflfastr_pbp", success=False)
    try:
        from app.services.collectors import nflfastr_collector
        from datetime import datetime
        
        if years is None:
            # Default to 2025 since current year data may not be available
            years = [2025]
        
        data = await nflfastr_collector.collect_pbp(years=years, save_to_disk=True)
        if data.success:
            result.records = data.records_count
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    finally:
        try:
            await nflfastr_collector.close()
        except:
            pass
    return result


# =============================================================================
# SOURCE MAPPING
# =============================================================================

IMPORT_MAP = {
    "espn": import_espn,
    "odds_api": import_odds_api,
    "pinnacle": import_pinnacle,
    "pinnacle_history": import_pinnacle_history,
    "espn_history": import_espn_history,
    "odds_api_history": import_odds_api_history,
    "weather": import_weather,
    "sportsdb": import_sportsdb,
    "sportsdb_history": import_sportsdb_history,
    "sportsdb_live": import_sportsdb_livescores,
    "nflfastr": import_nflfastr,
    "nflfastr_history": import_nflfastr_history,
    "nflfastr_pbp": import_nflfastr_pbp,
}

CURRENT_SOURCES = ["espn", "odds_api", "pinnacle", "weather", "sportsdb", "nflfastr"]
HISTORICAL_SOURCES = ["pinnacle_history", "espn_history", "odds_api_history", "sportsdb_history", "nflfastr_history"]
ALL_SOURCES = CURRENT_SOURCES + HISTORICAL_SOURCES


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
            if source == "pinnacle_history":
                result = await func(sports=sports, pages=pages)
            elif source == "espn_history":
                result = await func(sports=sports, days=days)
            elif source == "odds_api_history":
                result = await func(sports=sports, days=days)
            elif source == "sportsdb_history":
                result = await func(sports=sports, seasons=seasons)
            elif source == "sportsdb_live":
                result = await func()
            elif source == "nflfastr_history":
                result = await func(years_back=seasons)
            elif source == "nflfastr_pbp":
                result = await func()
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
    console.print("\n[bold]ROYALEY DATA COLLECTORS[/bold]")
    console.print("=" * 50)
    console.print("\n[green]✅ CURRENT DATA:[/green]")
    console.print("  • ESPN          - Games/scores (FREE)")
    console.print("  • OddsAPI       - 40+ books ($59/mo)")
    console.print("  • Pinnacle      - CLV lines ($10/mo)")
    console.print("  • Weather       - OpenWeatherMap (FREE)")
    console.print("  • SportsDB      - Games/scores/livescores ($295/mo)")
    console.print("  • nflfastr      - NFL schedules/results (FREE)")
    console.print("\n[green]✅ HISTORICAL DATA:[/green]")
    console.print("  • pinnacle_history  - Game results (~100/page)")
    console.print("  • espn_history      - Historical games (FREE)")
    console.print("  • odds_api_history  - Historical odds ($119/mo)")
    console.print("  • sportsdb_history  - Historical by season ($295/mo)")
    console.print("  • nflfastr_history  - NFL 1999-present (FREE)")
    console.print("\n[cyan]⚡ ADVANCED NFL DATA:[/cyan]")
    console.print("  • nflfastr_pbp      - Play-by-play + EPA/WPA/CPOE (FREE)")
    console.print("\n[cyan]⚡ LIVESCORES:[/cyan]")
    console.print("  • sportsdb_live     - Real-time scores ($295/mo)")
    console.print("\n[yellow]⏳ PENDING:[/yellow]")
    console.print("  • Basketball-Ref - NBA stats")
    console.print("  • Hockey-Ref    - NHL stats")
    console.print("  • Statcast      - MLB pitch data")


def main():
    parser = argparse.ArgumentParser(description="ROYALEY Master Data Import")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="All data")
    mode.add_argument("--current", action="store_true", help="Current only (default)")
    mode.add_argument("--historical", action="store_true", help="Historical only")
    mode.add_argument("--daemon", action="store_true", help="Run continuously")
    mode.add_argument("--status", action="store_true", help="Show status")
    
    parser.add_argument("--source", "-s", help="Specific source")
    parser.add_argument("--sport", help="Specific sport")
    parser.add_argument("--sports", help="Comma-separated sports")
    parser.add_argument("--pages", "-p", type=int, default=50, help="Pinnacle history pages (100 events/page)")
    parser.add_argument("--days", "-d", type=int, default=30, help="Days back for ESPN/OddsAPI history")
    parser.add_argument("--seasons", type=int, default=10, help="Seasons back for SportsDB history (default: 10)")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Daemon interval (min)")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    # Determine sources
    if args.source:
        sources = [args.source]
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
