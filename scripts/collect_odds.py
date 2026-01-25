#!/usr/bin/env python3
"""
ROYALEY - Scheduled Odds Collection

Runs on a schedule to collect current odds from all sources.
This builds up historical odds data over time for ML training.

Recommended schedule: Every 30 minutes

Usage:
    # Run once manually
    python scripts/collect_odds.py
    
    # Run continuously (every 30 min)
    python scripts/collect_odds.py --daemon --interval 30
    
    # Specific sports only
    python scripts/collect_odds.py --sports NFL,NBA,NHL

Cron example (add to crontab):
    */30 * * * * cd /nvme0n1-disk/royaley && docker exec royaley_api python scripts/collect_odds.py >> /var/log/royaley/odds_collection.log 2>&1
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    console.print("\n[yellow]Shutdown signal received. Finishing current collection...[/yellow]")
    shutdown_flag = True


async def collect_pinnacle_odds(sports: List[str] = None, save: bool = True) -> dict:
    """Collect current odds from Pinnacle."""
    from app.services.collectors.pinnacle_collector import pinnacle_collector
    from app.core.database import db_manager
    
    results = {
        "source": "pinnacle",
        "success": False,
        "records": 0,
        "sports": [],
        "errors": [],
    }
    
    try:
        if sports:
            for sport in sports:
                result = await pinnacle_collector.collect(sport_code=sport)
                if result.success and result.data:
                    results["records"] += result.records_count
                    results["sports"].append(sport)
                    
                    if save:
                        await db_manager.initialize()
                        async with db_manager.session() as session:
                            await pinnacle_collector.save_to_database(result.data, session)
                            
                if result.error:
                    results["errors"].append(f"{sport}: {result.error}")
        else:
            result = await pinnacle_collector.collect()
            if result.success and result.data:
                results["records"] = result.records_count
                results["sports"] = result.metadata.get("successful_sports", [])
                
                if save:
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await pinnacle_collector.save_to_database(result.data, session)
            
            if result.error:
                results["errors"].append(result.error)
        
        results["success"] = results["records"] > 0
        
    except Exception as e:
        results["errors"].append(str(e))
    
    finally:
        await pinnacle_collector.close()
    
    return results


async def collect_odds_api(sports: List[str] = None, save: bool = True) -> dict:
    """Collect current odds from TheOddsAPI (40+ sportsbooks)."""
    from app.services.collectors.odds_collector import odds_collector
    from app.core.database import db_manager
    
    results = {
        "source": "odds_api",
        "success": False,
        "records": 0,
        "sports": [],
        "errors": [],
    }
    
    try:
        if sports:
            for sport in sports:
                result = await odds_collector.collect(sport_code=sport)
                if result.success and result.data:
                    results["records"] += result.records_count
                    results["sports"].append(sport)
                    
                    if save:
                        await db_manager.initialize()
                        async with db_manager.session() as session:
                            await odds_collector.save_to_database(result.data, session)
                            
                if result.error:
                    results["errors"].append(f"{sport}: {result.error}")
        else:
            result = await odds_collector.collect()
            if result.success and result.data:
                results["records"] = result.records_count
                results["sports"] = result.metadata.get("successful_sports", [])
                
                if save:
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        await odds_collector.save_to_database(result.data, session)
            
            if result.error:
                results["errors"].append(result.error)
        
        results["success"] = results["records"] > 0
        
    except Exception as e:
        results["errors"].append(str(e))
    
    finally:
        await odds_collector.close()
    
    return results


async def run_collection(
    sources: List[str] = None,
    sports: List[str] = None,
    save: bool = True,
) -> dict:
    """Run odds collection from all sources."""
    sources = sources or ["pinnacle", "odds_api"]
    timestamp = datetime.utcnow().isoformat()
    
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold]ODDS COLLECTION - {timestamp}[/bold]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")
    console.print(f"Sources: {', '.join(sources)}")
    console.print(f"Sports: {', '.join(sports) if sports else 'ALL'}")
    console.print()
    
    all_results = {
        "timestamp": timestamp,
        "sources": {},
        "total_records": 0,
        "total_errors": 0,
    }
    
    for source in sources:
        console.print(f"[cyan]📊 Collecting from {source}...[/cyan]")
        
        if source == "pinnacle":
            result = await collect_pinnacle_odds(sports, save)
        elif source == "odds_api":
            result = await collect_odds_api(sports, save)
        else:
            console.print(f"[yellow]Unknown source: {source}[/yellow]")
            continue
        
        all_results["sources"][source] = result
        all_results["total_records"] += result["records"]
        all_results["total_errors"] += len(result["errors"])
        
        if result["success"]:
            console.print(f"[green]  ✅ {source}: {result['records']} records from {', '.join(result['sports'])}[/green]")
        else:
            console.print(f"[red]  ❌ {source}: {result['errors']}[/red]")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total records: {all_results['total_records']}")
    console.print(f"  Errors: {all_results['total_errors']}")
    
    return all_results


async def daemon_mode(
    interval_minutes: int = 30,
    sources: List[str] = None,
    sports: List[str] = None,
    save: bool = True,
):
    """Run collection on a schedule."""
    global shutdown_flag
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    console.print(f"\n[bold green]Starting Daemon Mode[/bold green]")
    console.print(f"Interval: {interval_minutes} minutes")
    console.print(f"Press Ctrl+C to stop")
    console.print()
    
    run_count = 0
    
    while not shutdown_flag:
        run_count += 1
        console.print(f"\n[bold cyan]Run #{run_count}[/bold cyan]")
        
        try:
            await run_collection(sources, sports, save)
        except Exception as e:
            console.print(f"[red]Collection error: {e}[/red]")
        
        if shutdown_flag:
            break
        
        # Wait for next interval
        console.print(f"\n[dim]Next collection in {interval_minutes} minutes...[/dim]")
        
        for _ in range(interval_minutes * 60):
            if shutdown_flag:
                break
            await asyncio.sleep(1)
    
    console.print("\n[green]Daemon stopped gracefully.[/green]")


async def capture_closing_lines():
    """Capture closing lines for games about to start."""
    from app.services.collectors.pinnacle_collector import pinnacle_collector
    from app.core.database import db_manager
    
    console.print("\n[cyan]📸 Capturing closing lines...[/cyan]")
    
    try:
        await db_manager.initialize()
        async with db_manager.session() as session:
            captured = await pinnacle_collector.capture_closing_lines(session)
            console.print(f"[green]✅ Captured {captured} closing lines[/green]")
            return captured
    except Exception as e:
        console.print(f"[red]Error capturing closing lines: {e}[/red]")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Collect odds from all sources")
    parser.add_argument("--sources", type=str, help="Comma-separated sources (pinnacle,odds_api)")
    parser.add_argument("--sports", type=str, help="Comma-separated sports (NFL,NBA,NHL)")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run continuously")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Minutes between collections (daemon mode)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    parser.add_argument("--closing-lines", action="store_true", help="Capture closing lines only")
    
    args = parser.parse_args()
    
    sources = args.sources.split(",") if args.sources else None
    sports = args.sports.split(",") if args.sports else None
    save = not args.no_save
    
    if args.closing_lines:
        asyncio.run(capture_closing_lines())
    elif args.daemon:
        asyncio.run(daemon_mode(args.interval, sources, sports, save))
    else:
        asyncio.run(run_collection(sources, sports, save))


if __name__ == "__main__":
    main()
