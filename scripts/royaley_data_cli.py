#!/usr/bin/env python3
"""
ROYALEY - Master Data Operations CLI
Complete command-line interface for all data operations.

Usage:
    # Show system status
    python royaley_data_cli.py status
    
    # Load historical data
    python royaley_data_cli.py load-historical --sport NFL --years 5
    python royaley_data_cli.py load-historical --all --years 10
    
    # Collect live data
    python royaley_data_cli.py collect-odds
    python royaley_data_cli.py collect-injuries --all
    python royaley_data_cli.py collect-weather --upcoming
    
    # Build features
    python royaley_data_cli.py build-features --all
    
    # Backup operations
    python royaley_data_cli.py backup --full
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()


# Configuration
ALL_SPORTS = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL", "MLB", "ATP", "WTA"]
OUTDOOR_SPORTS = ["NFL", "NCAAF", "CFL", "MLB", "ATP", "WTA"]

STORAGE_PATHS = {
    "nvme0": "/nvme0n1-disk/royaley",
    "nvme1": "/nvme1n1-disk",
    "hdd": "/sda-disk",
}


def cmd_status():
    """Show system status."""
    if HAS_RICH:
        console.print(Panel("[bold blue]ROYALEY System Status[/bold blue]", title="Status"))
    else:
        console.print("=== ROYALEY System Status ===")
    
    # Storage status
    if HAS_RICH:
        storage_table = Table(title="Storage")
        storage_table.add_column("Disk", style="cyan")
        storage_table.add_column("Mount", style="green")
        storage_table.add_column("Used", style="yellow")
        storage_table.add_column("Available", style="blue")
        storage_table.add_column("Status", style="magenta")
    else:
        console.print("\nStorage:")
    
    for name, path in STORAGE_PATHS.items():
        if os.path.exists(path):
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            used = total - free
            
            if HAS_RICH:
                storage_table.add_row(
                    name.upper(),
                    path,
                    f"{used / (1024**3):.1f} GB",
                    f"{free / (1024**3):.1f} GB",
                    "[green]✓ Mounted[/green]"
                )
            else:
                console.print(f"  {name.upper()}: {path} - Used: {used / (1024**3):.1f} GB, Free: {free / (1024**3):.1f} GB")
        else:
            if HAS_RICH:
                storage_table.add_row(
                    name.upper(),
                    path,
                    "-",
                    "-",
                    "[red]✗ Not Found[/red]"
                )
            else:
                console.print(f"  {name.upper()}: {path} - NOT FOUND")
    
    if HAS_RICH:
        console.print(storage_table)
    
    # Check containers
    console.print("\n[bold]Docker Containers:[/bold]" if HAS_RICH else "\nDocker Containers:")
    os.system("docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep royaley || echo 'No containers or Docker not running'")
    
    # Database check
    console.print("\n[bold]Database Tables:[/bold]" if HAS_RICH else "\nDatabase Tables:")
    os.system("docker exec royaley_postgres psql -U royaley -d royaley -c \"SELECT relname as table, n_live_tup as rows FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 10;\" 2>/dev/null || echo 'Database not accessible'")


async def cmd_load_historical(args):
    """Load historical data."""
    from scripts.load_historical_data import ESPNHistoricalLoader
    
    sports = ALL_SPORTS if args.all else [args.sport.upper()] if args.sport else None
    
    if not sports:
        console.print("[red]Specify --sport or --all[/red]" if HAS_RICH else "Error: Specify --sport or --all")
        return
    
    if HAS_RICH:
        console.print(Panel(
            f"[bold blue]Loading Historical Data[/bold blue]\n"
            f"Sports: {', '.join(sports)}\n"
            f"Years: {args.years}",
            title="Historical Data Load"
        ))
    else:
        console.print(f"Loading historical data for {', '.join(sports)} ({args.years} years)")
    
    async with ESPNHistoricalLoader() as loader:
        if args.all:
            await loader.load_all_sports(years=args.years, save_to_db=not args.no_db)
        else:
            for sport in sports:
                await loader.load_sport(sport, years=args.years, save_to_db=not args.no_db)


async def cmd_collect_injuries(args):
    """Collect injury data."""
    from scripts.injury_collector import InjuryCollector
    
    if HAS_RICH:
        console.print(Panel("[bold green]Collecting Injury Data[/bold green]", title="Injury Collection"))
    else:
        console.print("=== Collecting Injury Data ===")
    
    async with InjuryCollector() as collector:
        if args.all:
            await collector.collect_all_sports()
        elif args.sport:
            await collector.collect_sport(args.sport.upper())
        else:
            console.print("[yellow]Specify --sport or --all[/yellow]" if HAS_RICH else "Specify --sport or --all")


async def cmd_collect_weather(args):
    """Collect weather data."""
    from scripts.weather_collector import WeatherCollector
    
    api_key = os.environ.get("WEATHER_API_KEY", "")
    
    if not api_key:
        console.print("[red]WEATHER_API_KEY not set in environment![/red]" if HAS_RICH else "Error: WEATHER_API_KEY not set")
        console.print("Add to .env file: WEATHER_API_KEY=your_openweathermap_key")
        return
    
    if HAS_RICH:
        console.print(Panel(
            f"[bold cyan]Collecting Weather Data[/bold cyan]\n"
            f"Days ahead: {args.days}",
            title="Weather Collection"
        ))
    else:
        console.print(f"=== Collecting Weather Data ({args.days} days ahead) ===")
    
    async with WeatherCollector(api_key=api_key) as collector:
        await collector.collect_for_upcoming_games(
            sport_code=args.sport,
            days_ahead=args.days
        )


async def cmd_collect_odds():
    """Collect odds data."""
    console.print("[cyan]Collecting odds data...[/cyan]" if HAS_RICH else "Collecting odds data...")
    
    import subprocess
    result = subprocess.run(
        ["docker", "exec", "royaley_api", "python", "-m", "app.cli.admin", "data", "collect-odds"],
        capture_output=True,
        text=True
    )
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"[red]{result.stderr}[/red]" if HAS_RICH else result.stderr)


async def cmd_build_features(args):
    """Build feature store."""
    from scripts.build_features import FeatureBuilder
    
    if HAS_RICH:
        console.print(Panel("[bold magenta]Building Feature Store[/bold magenta]", title="Feature Builder"))
    else:
        console.print("=== Building Feature Store ===")
    
    builder = FeatureBuilder()
    
    if args.all:
        await builder.build_all_features()
    elif args.sport:
        await builder.build_all_features(sport_code=args.sport.upper())
    else:
        console.print("[yellow]Specify --sport or --all[/yellow]" if HAS_RICH else "Specify --sport or --all")


async def cmd_backup(args):
    """Run backup operations."""
    from scripts.backup_manager import BackupManager
    
    manager = BackupManager()
    
    if args.full:
        console.print("[cyan]Running full backup...[/cyan]" if HAS_RICH else "Running full backup...")
        await manager.full_backup()
    elif args.database:
        console.print("[cyan]Backing up database...[/cyan]" if HAS_RICH else "Backing up database...")
        await manager.backup_database()
    elif args.models:
        console.print("[cyan]Backing up models...[/cyan]" if HAS_RICH else "Backing up models...")
        await manager.backup_models()
    elif args.list:
        backups = manager.list_backups()
        console.print("\n[bold]Available Backups:[/bold]" if HAS_RICH else "\nAvailable Backups:")
        for backup_type, items in backups.items():
            if items:
                console.print(f"\n{backup_type.upper()}:")
                for item in items[:5]:
                    console.print(f"  • {item.name}")
    else:
        console.print("[yellow]Specify --full, --database, --models, or --list[/yellow]" if HAS_RICH else "Specify an option")


def cmd_setup_directories():
    """Create all necessary directories."""
    console.print("[cyan]Creating directory structure...[/cyan]" if HAS_RICH else "Creating directory structure...")
    
    directories = [
        # NVMe 0
        "/nvme0n1-disk/royaley/models",
        "/nvme0n1-disk/royaley/logs",
        "/nvme0n1-disk/royaley/data",
        # NVMe 1
        "/nvme1n1-disk/features/elo",
        "/nvme1n1-disk/features/team_stats",
        "/nvme1n1-disk/features/h2h",
        "/nvme1n1-disk/features/weather",
        "/nvme1n1-disk/features/injuries",
        "/nvme1n1-disk/features/momentum",
        "/nvme1n1-disk/features/combined",
        "/nvme1n1-disk/datasets",
        "/nvme1n1-disk/ml-training",
        "/nvme1n1-disk/checkpoints",
        # HDD
        "/sda-disk/raw-data/espn",
        "/sda-disk/raw-data/odds-api",
        "/sda-disk/raw-data/weather",
        "/sda-disk/raw-data/injuries",
        "/sda-disk/backups/database",
        "/sda-disk/backups/models",
        "/sda-disk/backups/configs",
        "/sda-disk/archives/daily",
        "/sda-disk/archives/weekly",
        "/sda-disk/archives/monthly",
    ]
    
    created = 0
    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            created += 1
        except Exception as e:
            console.print(f"[red]Failed to create {dir_path}: {e}[/red]" if HAS_RICH else f"Failed: {dir_path}")
    
    console.print(f"[green]Created {created} directories[/green]" if HAS_RICH else f"Created {created} directories")


def main():
    parser = argparse.ArgumentParser(
        description="ROYALEY Data Operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  royaley_data_cli.py status                           # Show system status
  royaley_data_cli.py setup                            # Create directories
  royaley_data_cli.py load-historical --all --years 5  # Load 5 years of data
  royaley_data_cli.py collect-injuries --all           # Collect all injuries
  royaley_data_cli.py collect-weather --upcoming       # Collect weather
  royaley_data_cli.py build-features --all             # Build all features
  royaley_data_cli.py backup --full                    # Full system backup
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # Setup command
    subparsers.add_parser("setup", help="Create directory structure")
    
    # Load historical command
    load_parser = subparsers.add_parser("load-historical", help="Load historical data")
    load_parser.add_argument("--sport", "-s", help="Sport code")
    load_parser.add_argument("--all", "-a", action="store_true", help="All sports")
    load_parser.add_argument("--years", "-y", type=int, default=10, help="Years to load")
    load_parser.add_argument("--no-db", action="store_true", help="Skip database save")
    
    # Collect injuries command
    injuries_parser = subparsers.add_parser("collect-injuries", help="Collect injury data")
    injuries_parser.add_argument("--sport", "-s", help="Sport code")
    injuries_parser.add_argument("--all", "-a", action="store_true", help="All sports")
    
    # Collect weather command
    weather_parser = subparsers.add_parser("collect-weather", help="Collect weather data")
    weather_parser.add_argument("--sport", "-s", help="Sport code")
    weather_parser.add_argument("--upcoming", action="store_true", help="For upcoming games")
    weather_parser.add_argument("--days", type=int, default=7, help="Days ahead")
    
    # Collect odds command
    subparsers.add_parser("collect-odds", help="Collect odds data")
    
    # Build features command
    features_parser = subparsers.add_parser("build-features", help="Build feature store")
    features_parser.add_argument("--sport", "-s", help="Sport code")
    features_parser.add_argument("--all", "-a", action="store_true", help="All sports")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup operations")
    backup_parser.add_argument("--full", action="store_true", help="Full backup")
    backup_parser.add_argument("--database", action="store_true", help="Database only")
    backup_parser.add_argument("--models", action="store_true", help="Models only")
    backup_parser.add_argument("--list", action="store_true", help="List backups")
    
    args = parser.parse_args()
    
    if args.command == "status":
        cmd_status()
    elif args.command == "setup":
        cmd_setup_directories()
    elif args.command == "load-historical":
        asyncio.run(cmd_load_historical(args))
    elif args.command == "collect-injuries":
        asyncio.run(cmd_collect_injuries(args))
    elif args.command == "collect-weather":
        if not args.upcoming:
            console.print("[yellow]Use --upcoming to collect weather for upcoming games[/yellow]" if HAS_RICH else "Use --upcoming")
        else:
            asyncio.run(cmd_collect_weather(args))
    elif args.command == "collect-odds":
        asyncio.run(cmd_collect_odds())
    elif args.command == "build-features":
        asyncio.run(cmd_build_features(args))
    elif args.command == "backup":
        asyncio.run(cmd_backup(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
