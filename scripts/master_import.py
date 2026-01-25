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
    """Import injuries/lineups from ESPN."""
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
                        await espn_collector.save_to_database(data.data, session)
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
    """Import sharp odds from Pinnacle."""
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


async def import_weather(sports: List[str] = None) -> ImportResult:
    """Import weather for outdoor games."""
    result = ImportResult(source="weather", success=False)
    try:
        from app.services.collectors import WeatherCollector
        
        sports = sports or ["NFL", "MLB"]
        for sport in sports:
            try:
                async with WeatherCollector() as collector:
                    stats = await collector.collect_for_upcoming_games(
                        sport_code=sport, 
                        days_ahead=7
                    )
                    result.records += stats.weather_fetched
            except Exception as e:
                result.errors.append(f"{sport}: {str(e)[:50]}")
        result.success = result.records > 0
    except Exception as e:
        result.errors.append(str(e)[:100])
    return result


# =============================================================================
# SOURCE MAPPING
# =============================================================================

IMPORT_MAP = {
    "espn": import_espn,
    "odds_api": import_odds_api,
    "pinnacle": import_pinnacle,
    "pinnacle_history": import_pinnacle_history,
    "weather": import_weather,
}

CURRENT_SOURCES = ["espn", "odds_api", "pinnacle", "weather"]
HISTORICAL_SOURCES = ["pinnacle_history"]
ALL_SOURCES = CURRENT_SOURCES + HISTORICAL_SOURCES


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

async def run_import(sources: List[str], sports: List[str] = None, pages: int = 50):
    """Run imports from specified sources."""
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
            console.print("[yellow]Cancelled[/yellow]")
            break
        
        console.print(f"[cyan]📊 {source}...[/cyan]", end=" ")
        
        func = IMPORT_MAP.get(source)
        if not func:
            console.print("[yellow]unknown[/yellow]")
            continue
        
        try:
            if source == "pinnacle_history":
                result = await func(sports=sports, pages=pages)
            else:
                result = await func(sports=sports)
            
            total_records += result.records
            total_errors += len(result.errors)
            
            if result.success:
                console.print(f"[green]✅ {result.records} records[/green]")
            else:
                err_msg = result.errors[0] if result.errors else "failed"
                console.print(f"[red]❌ {err_msg}[/red]")
        except Exception as e:
            console.print(f"[red]❌ {str(e)[:50]}[/red]")
            total_errors += 1
    
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold]COMPLETE[/bold] - Records: {total_records}, Errors: {total_errors}")


async def daemon_mode(interval: int, sources: List[str], sports: List[str]):
    """Run imports continuously."""
    global shutdown_flag
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    console.print(f"\n[bold green]🚀 Daemon Mode - Every {interval} min[/bold green]")
    console.print("Press Ctrl+C to stop\n")
    
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
    """Show collector status."""
    console.print("\n[bold]ROYALEY DATA COLLECTORS[/bold]")
    console.print("=" * 50)
    console.print("\n[green]✅ IMPLEMENTED:[/green]")
    console.print("  collector_01_espn.py      - Injuries (FREE)")
    console.print("  collector_02_odds_api.py  - 40+ books ($59/mo)")
    console.print("  collector_03_pinnacle.py  - CLV lines ($10/mo)")
    console.print("  collector_04_tennis.py    - Tennis stats")
    console.print("  collector_05_weather.py   - Weather (FREE)")
    console.print("\n[yellow]⏳ PENDING:[/yellow]")
    console.print("  nflfastR, cfbfastR, baseballr, hockeyR")
    console.print("  TheSportsDB, BallDontLie, Google Distance")


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
    parser.add_argument("--pages", "-p", type=int, default=50, help="History pages")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Daemon interval")
    
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
    
    if args.daemon:
        asyncio.run(daemon_mode(args.interval, sources, sports))
    else:
        asyncio.run(run_import(sources, sports, args.pages))


if __name__ == "__main__":
    main()
