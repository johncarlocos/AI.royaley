#!/usr/bin/env python3
"""
ROYALEY - Pinnacle Historical Data Import

Imports historical game results from Pinnacle archive for ML training.
This provides the OUTCOME labels (who won, scores, margins).

For historical ODDS data, you need TheOddsAPI ($119/mo plan).

Usage:
    python scripts/import_pinnacle_history.py --sport NFL
    python scripts/import_pinnacle_history.py --sport NBA --pages 50
    python scripts/import_pinnacle_history.py --all --pages 100
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

# Pinnacle sport IDs
PINNACLE_SPORTS = {
    "NFL": 7,
    "NCAAF": 7,
    "CFL": 7,
    "NBA": 3,
    "NCAAB": 3,
    "WNBA": 3,
    "NHL": 4,
    "MLB": 9,
    "ATP": 2,
    "WTA": 2,
}

# League filters to separate NFL from CFL/NCAAF etc
LEAGUE_FILTERS = {
    "NFL": ["NFL", "National Football League"],
    "NCAAF": ["NCAA", "College Football", "NCAAF", "FBS"],
    "CFL": ["Canadian Football", "CFL"],
    "NBA": ["NBA", "National Basketball Association"],
    "NCAAB": ["NCAA", "College Basketball", "NCAAB"],
    "WNBA": ["WNBA"],
    "NHL": ["NHL", "National Hockey League"],
    "MLB": ["MLB", "Major League Baseball"],
    "ATP": ["ATP"],
    "WTA": ["WTA"],
}


class PinnacleHistoryImporter:
    """Import historical results from Pinnacle archive."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pinnacle-odds.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com",
        }
    
    async def fetch_archive_page(
        self, 
        sport_id: int, 
        page_num: int,
        client: httpx.AsyncClient,
    ) -> List[Dict[str, Any]]:
        """Fetch a single page of archive data."""
        try:
            r = await client.get(
                f"{self.base_url}/kit/v1/archive",
                params={"sport_id": sport_id, "page_num": page_num},
                headers=self.headers,
                timeout=30,
            )
            
            if r.status_code != 200:
                return []
            
            data = r.json()
            return data.get("events", [])
            
        except Exception as e:
            console.print(f"[red]Error fetching page {page_num}: {e}[/red]")
            return []
    
    async def fetch_all_history(
        self,
        sport_code: str,
        max_pages: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch all historical events for a sport."""
        sport_id = PINNACLE_SPORTS.get(sport_code)
        if not sport_id:
            console.print(f"[red]Unknown sport: {sport_code}[/red]")
            return []
        
        league_filters = LEAGUE_FILTERS.get(sport_code, [])
        all_events = []
        
        async with httpx.AsyncClient() as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Fetching {sport_code} history...", total=max_pages)
                
                for page in range(1, max_pages + 1):
                    events = await self.fetch_archive_page(sport_id, page, client)
                    
                    if not events:
                        progress.update(task, completed=max_pages)
                        break
                    
                    # Filter by league if needed
                    if league_filters:
                        filtered = [
                            e for e in events
                            if any(lf.lower() in e.get("league_name", "").lower() for lf in league_filters)
                        ]
                        all_events.extend(filtered)
                    else:
                        all_events.extend(events)
                    
                    progress.update(task, advance=1)
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.2)
        
        return all_events
    
    def parse_event_result(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse an event into a normalized result record."""
        event_id = event.get("event_id")
        home_team = event.get("home", "")
        away_team = event.get("away", "")
        starts = event.get("starts", "")
        league_name = event.get("league_name", "")
        
        if not all([event_id, home_team, away_team, starts]):
            return None
        
        # Get period results (full game is number=0)
        period_results = event.get("period_results", [])
        full_game_result = None
        
        for pr in period_results:
            if pr.get("number") == 0 and pr.get("status") == 1:  # status 1 = settled
                full_game_result = pr
                break
        
        if not full_game_result:
            return None
        
        home_score = full_game_result.get("team_1_score")
        away_score = full_game_result.get("team_2_score")
        
        if home_score is None or away_score is None:
            return None
        
        # Determine winner
        if home_score > away_score:
            winner = "home"
        elif away_score > home_score:
            winner = "away"
        else:
            winner = "draw"
        
        return {
            "external_id": str(event_id),
            "home_team": home_team,
            "away_team": away_team,
            "starts": starts,
            "league_name": league_name,
            "home_score": home_score,
            "away_score": away_score,
            "total_score": home_score + away_score,
            "margin": home_score - away_score,
            "winner": winner,
            "settled_at": full_game_result.get("settled_at"),
        }
    
    async def save_to_database(
        self,
        results: List[Dict[str, Any]],
        sport_code: str,
    ) -> int:
        """Save historical results to database."""
        from app.core.database import db_manager
        from app.models import Game, Sport, Team, GameStatus
        from sqlalchemy import select, and_
        from datetime import datetime, timedelta
        
        saved = 0
        updated = 0
        
        await db_manager.initialize()
        
        async with db_manager.session() as session:
            # Get or create sport
            sport_result = await session.execute(
                select(Sport).where(Sport.code == sport_code)
            )
            sport = sport_result.scalar_one_or_none()
            
            if not sport:
                sport_names = {
                    "NBA": "National Basketball Association",
                    "NFL": "National Football League",
                    "NCAAF": "NCAA Football",
                    "NCAAB": "NCAA Basketball",
                    "NHL": "National Hockey League",
                    "MLB": "Major League Baseball",
                    "WNBA": "Women's National Basketball Association",
                    "CFL": "Canadian Football League",
                    "ATP": "ATP Tennis",
                    "WTA": "WTA Tennis",
                }
                sport = Sport(
                    code=sport_code,
                    name=sport_names.get(sport_code, sport_code),
                    is_active=True,
                )
                session.add(sport)
                await session.flush()
            
            for result in results:
                try:
                    # Get or create teams
                    home_team = await self._get_or_create_team(
                        session, sport.id, result["home_team"]
                    )
                    away_team = await self._get_or_create_team(
                        session, sport.id, result["away_team"]
                    )
                    
                    # Parse date
                    game_date = datetime.fromisoformat(result["starts"].replace("Z", "+00:00"))
                    game_date = datetime(
                        game_date.year, game_date.month, game_date.day,
                        game_date.hour, game_date.minute, game_date.second
                    )
                    
                    # Check if game exists
                    existing = await session.execute(
                        select(Game).where(Game.external_id == result["external_id"])
                    )
                    game = existing.scalar_one_or_none()
                    
                    if game:
                        # Update with results
                        game.home_score = result["home_score"]
                        game.away_score = result["away_score"]
                        game.status = GameStatus.FINAL
                        updated += 1
                    else:
                        # Create new game with results
                        game = Game(
                            sport_id=sport.id,
                            external_id=result["external_id"],
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            scheduled_at=game_date,
                            home_score=result["home_score"],
                            away_score=result["away_score"],
                            status=GameStatus.FINAL,
                        )
                        session.add(game)
                        saved += 1
                    
                    if (saved + updated) % 100 == 0:
                        await session.flush()
                        
                except Exception as e:
                    console.print(f"[dim]Error saving {result['home_team']} vs {result['away_team']}: {e}[/dim]")
                    continue
            
            await session.commit()
        
        return saved, updated
    
    async def _get_or_create_team(self, session, sport_id, team_name: str):
        """Get or create team record."""
        from app.models import Team
        from sqlalchemy import select, and_
        
        result = await session.execute(
            select(Team).where(
                and_(
                    Team.sport_id == sport_id,
                    Team.name == team_name,
                )
            )
        )
        team = result.scalar_one_or_none()
        
        if not team:
            abbreviation = team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()
            team = Team(
                sport_id=sport_id,
                external_id=f"pinnacle_{team_name.lower().replace(' ', '_')}",
                name=team_name,
                abbreviation=abbreviation,
                is_active=True,
            )
            session.add(team)
            await session.flush()
        
        return team


async def main():
    parser = argparse.ArgumentParser(description="Import Pinnacle historical data")
    parser.add_argument("--sport", "-s", type=str, help="Sport code (NFL, NBA, etc.)")
    parser.add_argument("--all", action="store_true", help="Import all sports")
    parser.add_argument("--pages", "-p", type=int, default=50, help="Max pages per sport")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--dry-run", action="store_true", help="Show data without saving")
    
    args = parser.parse_args()
    
    # Get API key
    from app.core.config import settings
    api_key = settings.RAPIDAPI_KEY
    
    if not api_key:
        console.print("[red]RAPIDAPI_KEY not configured![/red]")
        return
    
    importer = PinnacleHistoryImporter(api_key)
    
    # Determine sports to import
    if args.all:
        sports = ["NFL", "NBA", "NHL", "MLB"]  # Main US sports
    elif args.sport:
        sports = [args.sport.upper()]
    else:
        console.print("[yellow]Specify --sport or --all[/yellow]")
        return
    
    console.print(f"\n[bold blue]Pinnacle Historical Import[/bold blue]")
    console.print(f"Sports: {', '.join(sports)}")
    console.print(f"Max pages: {args.pages}")
    console.print()
    
    total_events = 0
    total_saved = 0
    total_updated = 0
    
    for sport_code in sports:
        console.print(f"\n[cyan]{'='*50}[/cyan]")
        console.print(f"[bold]{sport_code}[/bold]")
        console.print(f"[cyan]{'='*50}[/cyan]")
        
        # Fetch historical events
        events = await importer.fetch_all_history(sport_code, args.pages)
        console.print(f"[green]Fetched {len(events)} raw events[/green]")
        
        # Parse results
        results = []
        for event in events:
            result = importer.parse_event_result(event)
            if result:
                results.append(result)
        
        console.print(f"[green]Parsed {len(results)} completed games with scores[/green]")
        total_events += len(results)
        
        if results:
            # Show sample
            table = Table(title=f"Sample {sport_code} Results")
            table.add_column("Date", style="cyan")
            table.add_column("Home", style="white")
            table.add_column("Away", style="white")
            table.add_column("Score", style="green")
            table.add_column("Winner", style="yellow")
            
            for r in results[:5]:
                date_str = r["starts"][:10] if r["starts"] else ""
                table.add_row(
                    date_str,
                    r["home_team"][:20],
                    r["away_team"][:20],
                    f"{r['home_score']}-{r['away_score']}",
                    r["winner"],
                )
            
            console.print(table)
            
            # Show date range
            dates = [r["starts"] for r in results if r["starts"]]
            if dates:
                console.print(f"[dim]Date range: {min(dates)[:10]} to {max(dates)[:10]}[/dim]")
        
        # Save if requested
        if args.save and results:
            console.print("\n[cyan]💾 Saving to database...[/cyan]")
            saved, updated = await importer.save_to_database(results, sport_code)
            console.print(f"[green]✅ Saved {saved} new, updated {updated} existing[/green]")
            total_saved += saved
            total_updated += updated
    
    # Summary
    console.print(f"\n[bold green]{'='*50}[/bold green]")
    console.print(f"[bold]SUMMARY[/bold]")
    console.print(f"[bold green]{'='*50}[/bold green]")
    console.print(f"Total events parsed: {total_events}")
    if args.save:
        console.print(f"Total saved: {total_saved}")
        console.print(f"Total updated: {total_updated}")


if __name__ == "__main__":
    asyncio.run(main())
