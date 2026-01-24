#!/usr/bin/env python3
"""
ROYALEY - Historical Data Loader
Loads historical game data from ESPN API and saves to database.

Features:
- Loads 10 years of historical data
- Saves raw JSON to HDD (/app/raw-data/espn/)
- Processes and saves to PostgreSQL
- Rate limiting and error handling
- Progress tracking

Usage:
    python load_historical_data.py --sport NFL --years 5
    python load_historical_data.py --all --years 10
    python load_historical_data.py --sport NBA --start 2020-01-01 --end 2024-12-31
"""

import asyncio
import argparse
import json
import gzip
import logging
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import aiohttp
import aiofiles

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            print(*[str(a).replace('[', '').replace(']', '') for a in args])

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ESPN API configuration
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

ESPN_SPORT_PATHS = {
    "NFL": {"sport": "football", "league": "nfl"},
    "NCAAF": {"sport": "football", "league": "college-football"},
    "CFL": {"sport": "football", "league": "cfl"},
    "NBA": {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "mens-college-basketball"},
    "WNBA": {"sport": "basketball", "league": "wnba"},
    "NHL": {"sport": "hockey", "league": "nhl"},
    "MLB": {"sport": "baseball", "league": "mlb"},
    "ATP": {"sport": "tennis", "league": "atp"},
    "WTA": {"sport": "tennis", "league": "wta"},
}

ALL_SPORTS = list(ESPN_SPORT_PATHS.keys())

# Season date ranges
SEASON_DATES = {
    "NFL": {"start_month": 9, "end_month": 2, "crosses_year": True},
    "NCAAF": {"start_month": 8, "end_month": 1, "crosses_year": True},
    "CFL": {"start_month": 6, "end_month": 11, "crosses_year": False},
    "NBA": {"start_month": 10, "end_month": 6, "crosses_year": True},
    "NCAAB": {"start_month": 11, "end_month": 4, "crosses_year": True},
    "WNBA": {"start_month": 5, "end_month": 10, "crosses_year": False},
    "NHL": {"start_month": 10, "end_month": 6, "crosses_year": True},
    "MLB": {"start_month": 3, "end_month": 10, "crosses_year": False},
    "ATP": {"start_month": 1, "end_month": 11, "crosses_year": False},
    "WTA": {"start_month": 1, "end_month": 11, "crosses_year": False},
}

RAW_DATA_PATH = "/app/raw-data/espn"


@dataclass
class LoaderStats:
    """Statistics for data loading."""
    sport: str
    dates_processed: int = 0
    games_found: int = 0
    games_saved: int = 0
    teams_created: int = 0
    raw_files_created: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> str:
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return str(delta).split('.')[0]
        return "N/A"


class ESPNHistoricalLoader:
    """
    Loads historical data from ESPN API.
    
    Data Flow:
    1. Fetch scoreboard data from ESPN
    2. Save raw JSON to HDD (immutable archive)
    3. Parse and save to PostgreSQL
    """
    
    RATE_LIMIT_DELAY = 0.1  # seconds between requests
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats: Dict[str, LoaderStats] = {}
        
        # Ensure directory exists
        Path(RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Royaley/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_sport(
        self,
        sport_code: str,
        years: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_to_db: bool = True,
    ) -> LoaderStats:
        """Load historical data for a specific sport."""
        if sport_code not in ESPN_SPORT_PATHS:
            console.print(f"[red]Unknown sport: {sport_code}[/red]")
            return LoaderStats(sport=sport_code, errors=[f"Unknown sport: {sport_code}"])
        
        stats = LoaderStats(sport=sport_code, start_time=datetime.now())
        self.stats[sport_code] = stats
        
        sport_path = ESPN_SPORT_PATHS[sport_code]
        season_info = SEASON_DATES.get(sport_code, {"start_month": 1, "end_month": 12, "crosses_year": False})
        
        if HAS_RICH:
            console.print(Panel(
                f"[bold blue]Loading {sport_code} Historical Data[/bold blue]\n"
                f"Years: {years} | Save to DB: {save_to_db}",
                title="Historical Data Loader"
            ))
        else:
            console.print(f"=== Loading {sport_code} Historical Data ({years} years) ===")
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=years * 365)
        
        # Generate dates to fetch
        dates_to_fetch = self._generate_dates(start_date, end_date, season_info)
        
        console.print(f"[cyan]Date range: {start_date.date()} to {end_date.date()}[/cyan]")
        console.print(f"[cyan]Days to fetch: {len(dates_to_fetch)}[/cyan]")
        
        all_games = []
        teams_cache = {}
        
        # Process dates
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[green]Fetching {sport_code}...", total=len(dates_to_fetch))
                
                for fetch_date in dates_to_fetch:
                    games = await self._fetch_and_process_date(
                        sport_path, sport_code, fetch_date, stats, teams_cache
                    )
                    all_games.extend(games)
                    progress.update(task, advance=1)
        else:
            for i, fetch_date in enumerate(dates_to_fetch):
                games = await self._fetch_and_process_date(
                    sport_path, sport_code, fetch_date, stats, teams_cache
                )
                all_games.extend(games)
                
                if i % 100 == 0:
                    console.print(f"  Processed {i}/{len(dates_to_fetch)} dates, {stats.games_found} games found")
        
        # Save to database
        if save_to_db and all_games:
            console.print("[yellow]Saving to database...[/yellow]")
            await self._save_to_database(sport_code, all_games, teams_cache, stats)
        
        stats.end_time = datetime.now()
        self._print_summary(stats)
        
        return stats
    
    async def load_all_sports(self, years: int = 10, save_to_db: bool = True) -> Dict[str, LoaderStats]:
        """Load historical data for all sports."""
        if HAS_RICH:
            console.print(Panel(
                f"[bold green]Loading ALL Sports Historical Data[/bold green]\n"
                f"Sports: {', '.join(ALL_SPORTS)}\n"
                f"Years: {years}",
                title="Full Historical Data Load"
            ))
        else:
            console.print(f"=== Loading ALL Sports ({years} years) ===")
        
        results = {}
        
        for sport in ALL_SPORTS:
            try:
                stats = await self.load_sport(sport, years, save_to_db=save_to_db)
                results[sport] = stats
            except Exception as e:
                console.print(f"[red]Failed to load {sport}: {e}[/red]")
                results[sport] = LoaderStats(sport=sport, errors=[str(e)])
        
        self._print_final_summary(results)
        return results
    
    def _generate_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        season_info: Dict,
    ) -> List[datetime]:
        """Generate list of dates within sport's season."""
        dates = []
        current = start_date
        
        while current <= end_date:
            month = current.month
            start_month = season_info["start_month"]
            end_month = season_info["end_month"]
            crosses_year = season_info["crosses_year"]
            
            in_season = False
            if crosses_year:
                in_season = month >= start_month or month <= end_month
            else:
                in_season = start_month <= month <= end_month
            
            if in_season:
                dates.append(current)
            
            current += timedelta(days=1)
        
        return dates
    
    async def _fetch_and_process_date(
        self,
        sport_path: Dict[str, str],
        sport_code: str,
        fetch_date: datetime,
        stats: LoaderStats,
        teams_cache: Dict,
    ) -> List[Dict]:
        """Fetch and process data for a single date."""
        try:
            await asyncio.sleep(self.RATE_LIMIT_DELAY)
            
            # Fetch from ESPN
            date_str = fetch_date.strftime("%Y%m%d")
            url = f"{ESPN_BASE_URL}/{sport_path['sport']}/{sport_path['league']}/scoreboard"
            params = {"dates": date_str}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 429:  # Rate limited
                    await asyncio.sleep(5)
                    return await self._fetch_and_process_date(
                        sport_path, sport_code, fetch_date, stats, teams_cache
                    )
                
                if response.status != 200:
                    return []
                
                data = await response.json()
            
            # Parse events
            events = data.get("events", [])
            games = []
            
            for event in events:
                try:
                    game = self._parse_event(event, sport_code, fetch_date, teams_cache)
                    if game:
                        games.append(game)
                        stats.games_found += 1
                except Exception as e:
                    logger.debug(f"Error parsing event: {e}")
            
            # Save raw data
            if games:
                await self._save_raw_data(sport_code, fetch_date, data)
                stats.raw_files_created += 1
            
            stats.dates_processed += 1
            return games
            
        except Exception as e:
            stats.errors.append(f"{fetch_date.date()}: {str(e)[:50]}")
            return []
    
    def _parse_event(
        self,
        event: Dict,
        sport_code: str,
        fetch_date: datetime,
        teams_cache: Dict,
    ) -> Optional[Dict]:
        """Parse ESPN event into game data."""
        event_id = event.get("id")
        
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        
        competition = competitions[0]
        competitors = competition.get("competitors", [])
        
        if len(competitors) != 2:
            return None
        
        # Find home and away
        home_data = None
        away_data = None
        
        for comp in competitors:
            team = comp.get("team", {})
            team_info = {
                "id": str(team.get("id", "")),
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "city": team.get("location", ""),
                "logo": team.get("logo", ""),
            }
            
            # Cache team
            if team_info["id"]:
                teams_cache[team_info["id"]] = team_info
            
            if comp.get("homeAway") == "home":
                home_data = {**team_info, "score": comp.get("score"), "winner": comp.get("winner")}
            else:
                away_data = {**team_info, "score": comp.get("score"), "winner": comp.get("winner")}
        
        if not home_data or not away_data:
            return None
        
        # Parse game date
        game_date_str = event.get("date", "")
        try:
            game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            game_date = fetch_date
        
        # Parse status
        status_data = competition.get("status", {}).get("type", {})
        
        # Parse venue
        venue = competition.get("venue", {})
        
        return {
            "external_id": str(event_id),
            "sport_code": sport_code,
            "name": event.get("name", ""),
            "game_date": game_date,
            "home_team": home_data,
            "away_team": away_data,
            "home_score": int(home_data.get("score", 0)) if home_data.get("score") else None,
            "away_score": int(away_data.get("score", 0)) if away_data.get("score") else None,
            "status": status_data.get("name", ""),
            "status_completed": status_data.get("completed", False),
            "venue": {
                "name": venue.get("fullName", ""),
                "city": venue.get("address", {}).get("city", ""),
                "state": venue.get("address", {}).get("state", ""),
                "indoor": venue.get("indoor", False),
            },
            "attendance": competition.get("attendance"),
        }
    
    async def _save_raw_data(self, sport_code: str, fetch_date: datetime, data: Dict):
        """Save raw JSON to HDD."""
        year = fetch_date.strftime("%Y")
        month = fetch_date.strftime("%m")
        day = fetch_date.strftime("%d")
        
        dir_path = Path(RAW_DATA_PATH) / sport_code / year / month
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{day}.json.gz"
        
        async with aiofiles.open(file_path, 'wb') as f:
            json_str = json.dumps(data, indent=2, default=str)
            compressed = gzip.compress(json_str.encode('utf-8'))
            await f.write(compressed)
    
    async def _save_to_database(
        self,
        sport_code: str,
        games: List[Dict],
        teams_cache: Dict,
        stats: LoaderStats,
    ):
        """Save games to PostgreSQL."""
        try:
            from app.core.database import db_manager
            from app.models import Sport, Team, Game
            from app.models.models import GameStatus
            from sqlalchemy import select
            
            await db_manager.initialize()
            
            # PHASE 1: Save teams first (separate transaction)
            team_db_cache = {}
            async with db_manager.session() as session:
                # Get or create sport
                result = await session.execute(
                    select(Sport).where(Sport.code == sport_code)
                )
                sport = result.scalar_one_or_none()
                
                if not sport:
                    sport = Sport(
                        id=uuid4(),
                        code=sport_code,
                        name=self._get_sport_name(sport_code),
                        is_active=True
                    )
                    session.add(sport)
                    await session.flush()
                
                sport_id = sport.id
                
                # Create teams
                for team_id, team_data in teams_cache.items():
                    result = await session.execute(
                        select(Team).where(
                            Team.sport_id == sport_id,
                            Team.external_id == team_id
                        )
                    )
                    team = result.scalar_one_or_none()
                    
                    if not team:
                        team = Team(
                            id=uuid4(),
                            sport_id=sport_id,
                            external_id=team_id,
                            name=team_data.get("name", "Unknown"),
                            abbreviation=team_data.get("abbreviation", "UNK"),
                            city=team_data.get("city", ""),
                            logo_url=team_data.get("logo", ""),
                            elo_rating=1500.0,
                            is_active=True
                        )
                        session.add(team)
                        await session.flush()
                        stats.teams_created += 1
                    
                    team_db_cache[team_id] = team.id  # Store ID, not object
                
                await session.commit()
                logger.info(f"Teams saved: {stats.teams_created}")
            
            # PHASE 2: Save games (separate transaction, batch by batch)
            async with db_manager.session() as session:
                for game_data in games:
                    try:
                        home_team_id = team_db_cache.get(game_data["home_team"]["id"])
                        away_team_id = team_db_cache.get(game_data["away_team"]["id"])
                        
                        if not home_team_id or not away_team_id:
                            continue
                        
                        # Check if exists
                        result = await session.execute(
                            select(Game).where(Game.external_id == game_data["external_id"])
                        )
                        if result.scalar_one_or_none():
                            continue
                        
                        # Map status
                        status = self._map_status(game_data["status"])
                        
                        game = Game(
                            id=uuid4(),
                            sport_id=sport_id,
                            external_id=game_data["external_id"],
                            home_team_id=home_team_id,
                            away_team_id=away_team_id,
                            game_date=game_data["game_date"],
                            home_score=game_data.get("home_score"),
                            away_score=game_data.get("away_score"),
                            status=status,
                            broadcast=game_data.get("broadcast"),
                        )
                        session.add(game)
                        stats.games_saved += 1
                        
                    except Exception as e:
                        logger.debug(f"Error saving game: {e}")
                
                await session.commit()
                logger.info(f"Games saved: {stats.games_saved}")
                
        except ImportError:
            logger.warning("Database modules not available - skipping database save")
            console.print("[yellow]Database modules not available - only raw JSON saved[/yellow]")
        except Exception as e:
            logger.error(f"Database error: {e}")
            stats.errors.append(f"DB error: {str(e)[:50]}")
    
    def _map_status(self, espn_status: str):
        """Map ESPN status to GameStatus enum."""
        from app.models.models import GameStatus
        
        status_map = {
            "STATUS_SCHEDULED": GameStatus.SCHEDULED,
            "STATUS_IN_PROGRESS": GameStatus.IN_PROGRESS,
            "STATUS_HALFTIME": GameStatus.IN_PROGRESS,
            "STATUS_FINAL": GameStatus.FINAL,
            "STATUS_POSTPONED": GameStatus.POSTPONED,
            "STATUS_CANCELED": GameStatus.CANCELLED,
        }
        return status_map.get(espn_status, GameStatus.SCHEDULED)
    
    def _get_sport_name(self, code: str) -> str:
        """Get full sport name."""
        names = {
            "NFL": "National Football League",
            "NCAAF": "NCAA Football",
            "CFL": "Canadian Football League",
            "NBA": "National Basketball Association",
            "NCAAB": "NCAA Basketball",
            "WNBA": "Women's NBA",
            "NHL": "National Hockey League",
            "MLB": "Major League Baseball",
            "ATP": "ATP Tennis",
            "WTA": "WTA Tennis",
        }
        return names.get(code, code)
    
    def _print_summary(self, stats: LoaderStats):
        """Print summary for single sport."""
        if not HAS_RICH:
            console.print(f"\n{stats.sport} Summary:")
            console.print(f"  Dates processed: {stats.dates_processed}")
            console.print(f"  Games found: {stats.games_found}")
            console.print(f"  Games saved: {stats.games_saved}")
            console.print(f"  Duration: {stats.duration}")
            return
        
        table = Table(title=f"{stats.sport} Loading Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Dates Processed", str(stats.dates_processed))
        table.add_row("Games Found", str(stats.games_found))
        table.add_row("Games Saved", str(stats.games_saved))
        table.add_row("Teams Created", str(stats.teams_created))
        table.add_row("Raw Files Created", str(stats.raw_files_created))
        table.add_row("Errors", str(len(stats.errors)))
        table.add_row("Duration", stats.duration)
        
        console.print(table)
    
    def _print_final_summary(self, results: Dict[str, LoaderStats]):
        """Print final summary for all sports."""
        if not HAS_RICH:
            console.print("\n=== Final Summary ===")
            total_found = sum(s.games_found for s in results.values())
            total_saved = sum(s.games_saved for s in results.values())
            console.print(f"Total games found: {total_found}")
            console.print(f"Total games saved: {total_saved}")
            return
        
        table = Table(title="Historical Data Load - Final Summary")
        table.add_column("Sport", style="cyan")
        table.add_column("Games Found", style="blue")
        table.add_column("Games Saved", style="green")
        table.add_column("Raw Files", style="yellow")
        table.add_column("Errors", style="red")
        table.add_column("Duration", style="magenta")
        
        total_found = 0
        total_saved = 0
        total_files = 0
        
        for sport, stats in results.items():
            table.add_row(
                sport,
                str(stats.games_found),
                str(stats.games_saved),
                str(stats.raw_files_created),
                str(len(stats.errors)),
                stats.duration
            )
            total_found += stats.games_found
            total_saved += stats.games_saved
            total_files += stats.raw_files_created
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_found}[/bold]",
            f"[bold]{total_saved}[/bold]",
            f"[bold]{total_files}[/bold]",
            "-",
            "-"
        )
        
        console.print(table)
        console.print(f"\n[cyan]Raw data saved to:[/cyan] {RAW_DATA_PATH}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load historical sports data from ESPN")
    parser.add_argument("--sport", "-s", type=str, help="Sport code (NFL, NBA, etc.)")
    parser.add_argument("--all", "-a", action="store_true", help="Load all sports")
    parser.add_argument("--years", "-y", type=int, default=10, help="Years to load (default: 10)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-db", action="store_true", help="Skip database save")
    parser.add_argument("--list-sports", action="store_true", help="List available sports")
    
    args = parser.parse_args()
    
    if args.list_sports:
        console.print("[bold]Available Sports:[/bold]")
        for sport in ALL_SPORTS:
            console.print(f"  • {sport}")
        return
    
    if not args.sport and not args.all:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    async with ESPNHistoricalLoader() as loader:
        if args.all:
            await loader.load_all_sports(years=args.years, save_to_db=not args.no_db)
        else:
            await loader.load_sport(
                args.sport.upper(),
                years=args.years,
                start_date=start_date,
                end_date=end_date,
                save_to_db=not args.no_db
            )


if __name__ == "__main__":
    asyncio.run(main())
