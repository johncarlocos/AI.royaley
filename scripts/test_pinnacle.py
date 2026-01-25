#!/usr/bin/env python3
"""
ROYALEY - Pinnacle Odds Collector Test Script

Tests the Pinnacle odds collector via RapidAPI.

Usage:
    python scripts/test_pinnacle.py
    python scripts/test_pinnacle.py --sport NFL
    python scripts/test_pinnacle.py --save  # Save to database
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def test_pinnacle_collection(sport_code: str = None, save_to_db: bool = False):
    """Test Pinnacle odds collection."""
    from app.services.collectors.pinnacle_collector import pinnacle_collector
    from app.core.config import settings
    
    console.print(Panel.fit(
        "[bold blue]Pinnacle Odds Collector Test[/bold blue]\n"
        f"RapidAPI Key: {settings.RAPIDAPI_KEY[:20]}...\n"
        f"Sport: {sport_code or 'ALL'}",
        title="🎯 Pinnacle Test"
    ))
    
    if not settings.RAPIDAPI_KEY:
        console.print("[red]❌ RAPIDAPI_KEY not configured in .env![/red]")
        console.print("[yellow]Add: RAPIDAPI_KEY=your_key_here[/yellow]")
        return
    
    # Test collection
    console.print("\n[cyan]📊 Collecting Pinnacle odds...[/cyan]")
    
    try:
        result = await pinnacle_collector.collect(sport_code=sport_code)
        
        if result.success:
            console.print(f"[green]✅ Collection successful![/green]")
            console.print(f"[green]   Records collected: {result.records_count}[/green]")
            
            if result.data:
                # Show sample data
                table = Table(title="Sample Pinnacle Odds")
                table.add_column("Sport", style="cyan")
                table.add_column("Game", style="white")
                table.add_column("Type", style="yellow")
                table.add_column("Home Odds", style="green")
                table.add_column("Away Odds", style="green")
                table.add_column("Line/Total", style="magenta")
                
                for record in result.data[:10]:
                    line_value = ""
                    if record.get("home_line"):
                        line_value = f"{record['home_line']:+.1f}"
                    elif record.get("total"):
                        line_value = f"O/U {record['total']}"
                    
                    table.add_row(
                        record.get("sport_code", ""),
                        f"{record.get('away_team', '')[:15]} @ {record.get('home_team', '')[:15]}",
                        record.get("bet_type", ""),
                        str(record.get("home_odds", "")) if record.get("home_odds") else "",
                        str(record.get("away_odds", "")) if record.get("away_odds") else "",
                        line_value,
                    )
                
                console.print(table)
                
                if result.records_count > 10:
                    console.print(f"[dim]... and {result.records_count - 10} more records[/dim]")
                
                # Save to database if requested
                if save_to_db and result.data:
                    console.print("\n[cyan]💾 Saving to database...[/cyan]")
                    
                    from app.core.database import db_manager
                    
                    await db_manager.initialize()
                    async with db_manager.session() as session:
                        saved = await pinnacle_collector.save_to_database(result.data, session)
                        console.print(f"[green]✅ Saved {saved} records to database[/green]")
            else:
                console.print("[yellow]⚠️ No data returned[/yellow]")
        else:
            console.print(f"[red]❌ Collection failed: {result.error}[/red]")
            
        # Show metadata
        if result.metadata:
            console.print("\n[dim]Metadata:[/dim]")
            for key, value in result.metadata.items():
                console.print(f"[dim]  {key}: {value}[/dim]")
                
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    finally:
        await pinnacle_collector.close()


async def test_sports_list():
    """Test getting available sports."""
    from app.services.collectors.pinnacle_collector import pinnacle_collector
    
    console.print("\n[cyan]📋 Fetching available sports...[/cyan]")
    
    try:
        sports = await pinnacle_collector.get_sports_list()
        
        if sports:
            table = Table(title="Available Sports")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="white")
            
            for sport in sports[:20]:
                if isinstance(sport, dict):
                    table.add_row(
                        str(sport.get("id", "")),
                        sport.get("name", str(sport)),
                    )
                else:
                    table.add_row("", str(sport))
            
            console.print(table)
        else:
            console.print("[yellow]No sports data returned[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error fetching sports: {e}[/red]")
    
    finally:
        await pinnacle_collector.close()


async def test_raw_api_call():
    """Test raw API call to debug endpoints."""
    import httpx
    from app.core.config import settings
    
    console.print("\n[cyan]🔧 Testing raw API endpoints...[/cyan]")
    console.print(f"[dim]API Key: {settings.RAPIDAPI_KEY[:20]}...[/dim]")
    
    headers = {
        "X-RapidAPI-Key": settings.RAPIDAPI_KEY,
        "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com",
    }
    
    # Correct endpoints from documentation
    endpoints = [
        "/kit/v1/sports",
        "/kit/v1/betting-status",
        "/kit/v1/leagues?sport_id=4",  # Basketball
        "/kit/v1/markets?sport_id=4&is_have_odds=true&event_type=prematch",  # NBA odds
        "/kit/v1/markets?sport_id=15&is_have_odds=true&event_type=prematch",  # NFL odds
    ]
    
    async with httpx.AsyncClient(base_url="https://pinnacle-odds.p.rapidapi.com", timeout=30) as client:
        for endpoint in endpoints:
            try:
                console.print(f"\n[dim]Testing: {endpoint}[/dim]")
                response = await client.get(endpoint, headers=headers)
                
                status_color = "green" if response.status_code == 200 else "red"
                console.print(f"[{status_color}]  Status: {response.status_code}[/{status_color}]")
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        console.print(f"[green]  Response: List with {len(data)} items[/green]")
                        if data:
                            console.print(f"[dim]  Sample: {str(data[0])[:300]}...[/dim]")
                    elif isinstance(data, dict):
                        console.print(f"[green]  Response: Dict with keys: {list(data.keys())[:10]}[/green]")
                        # Show events count if present
                        if "events" in data:
                            console.print(f"[green]  Events: {len(data['events'])} events found[/green]")
                            if data["events"]:
                                console.print(f"[dim]  First event: {str(data['events'][0])[:300]}...[/dim]")
                else:
                    console.print(f"[dim]  Response: {response.text[:200]}[/dim]")
                    
            except Exception as e:
                console.print(f"[red]  Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Test Pinnacle Odds Collector")
    parser.add_argument("--sport", "-s", type=str, help="Sport code (NFL, NBA, etc.)")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--sports-list", action="store_true", help="List available sports")
    parser.add_argument("--debug", action="store_true", help="Debug API endpoints")
    
    args = parser.parse_args()
    
    if args.debug:
        asyncio.run(test_raw_api_call())
    elif args.sports_list:
        asyncio.run(test_sports_list())
    else:
        asyncio.run(test_pinnacle_collection(
            sport_code=args.sport,
            save_to_db=args.save,
        ))


if __name__ == "__main__":
    main()
