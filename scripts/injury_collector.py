"""
ROYALEY - Injury Data Collector
Collects injury reports from ESPN API (FREE)

ESPN provides free injury data through their API endpoints.
This collector fetches injuries for all teams and saves to database.

Features:
- Collects from ESPN's free injury API
- Saves raw JSON to HDD for archival
- Updates injury database table
- Calculates impact scores based on position/role
- Links injuries to upcoming games

Usage:
    # Collect injuries for all sports
    python injury_collector.py --all
    
    # Collect for specific sport
    python injury_collector.py --sport NFL
"""

import asyncio
import argparse
import json
import gzip
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from uuid import uuid4
import aiohttp
import aiofiles

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# ESPN Sport mappings
ESPN_SPORT_PATHS = {
    "NFL": {"sport": "football", "league": "nfl"},
    "NCAAF": {"sport": "football", "league": "college-football"},
    "CFL": {"sport": "football", "league": "cfl"},
    "NBA": {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "mens-college-basketball"},
    "WNBA": {"sport": "basketball", "league": "wnba"},
    "NHL": {"sport": "hockey", "league": "nhl"},
    "MLB": {"sport": "baseball", "league": "mlb"},
}

# Position impact scores (how much a position affects team performance)
POSITION_IMPACT = {
    # NFL
    "QB": 1.0, "RB": 0.6, "WR": 0.5, "TE": 0.4, "OL": 0.3, "OT": 0.35, "OG": 0.25, "C": 0.3,
    "DL": 0.3, "DE": 0.35, "DT": 0.25, "LB": 0.4, "MLB": 0.45, "OLB": 0.35,
    "DB": 0.35, "CB": 0.4, "S": 0.35, "FS": 0.35, "SS": 0.35, "K": 0.2, "P": 0.1,
    # NBA
    "PG": 0.7, "SG": 0.5, "SF": 0.5, "PF": 0.5, "C": 0.6, "G": 0.6, "F": 0.5,
    # MLB
    "SP": 0.8, "RP": 0.3, "CP": 0.4, "C": 0.4, "1B": 0.3, "2B": 0.35, "3B": 0.35,
    "SS": 0.4, "LF": 0.3, "CF": 0.35, "RF": 0.3, "DH": 0.25, "P": 0.5,
    # NHL
    "G": 0.9, "D": 0.4, "LW": 0.4, "RW": 0.4, "C": 0.5, "W": 0.4,
}

# Status severity (how likely to miss game)
STATUS_SEVERITY = {
    "Out": 1.0,
    "IR": 1.0,
    "Injured Reserve": 1.0,
    "Suspended": 1.0,
    "Doubtful": 0.8,
    "Questionable": 0.5,
    "Probable": 0.2,
    "Day-To-Day": 0.6,
    "Day-to-Day": 0.6,
    "DTD": 0.6,
    "Unknown": 0.3,
}


@dataclass
class InjuryData:
    """Parsed injury data."""
    player_name: str
    team_id: str
    team_name: str
    sport_code: str
    position: str
    injury_type: str
    body_part: str
    status: str
    status_detail: str
    is_starter: bool = False
    external_id: str = ""
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score based on position and status."""
        position_impact = POSITION_IMPACT.get(self.position, 0.3)
        status_severity = STATUS_SEVERITY.get(self.status, 0.3)
        starter_bonus = 0.2 if self.is_starter else 0.0
        return min(1.0, (position_impact + starter_bonus) * status_severity)


@dataclass
class CollectorStats:
    """Statistics for collection run."""
    sport: str = ""
    teams_processed: int = 0
    injuries_found: int = 0
    injuries_saved: int = 0
    raw_archived: bool = False
    errors: List[str] = field(default_factory=list)


class InjuryCollector:
    """
    Collects injury data from ESPN API.
    
    ESPN Injury API Endpoints:
    - Team injuries: /sports/{sport}/{league}/teams/{team_id}/injuries
    - All injuries: /sports/{sport}/{league}/injuries
    
    All responses are archived to HDD before processing.
    """
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
    RAW_DATA_PATH = "/app/raw-data/injuries"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats: Dict[str, CollectorStats] = {}
        
        # Ensure directories exist
        Path(self.RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Royaley/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_all_sports(self) -> Dict[str, CollectorStats]:
        """Collect injuries for all sports."""
        if HAS_RICH:
            console.print(Panel(
                "[bold green]Injury Data Collection[/bold green]\n"
                f"Sports: {', '.join(ESPN_SPORT_PATHS.keys())}",
                title="Injury Collector"
            ))
        else:
            console.print("=== Injury Data Collection ===")
        
        for sport_code in ESPN_SPORT_PATHS.keys():
            try:
                self.stats[sport_code] = await self.collect_sport(sport_code)
            except Exception as e:
                logger.error(f"Error collecting {sport_code}: {e}")
                self.stats[sport_code] = CollectorStats(
                    sport=sport_code, 
                    errors=[str(e)]
                )
        
        self._print_summary()
        return self.stats
    
    async def collect_sport(self, sport_code: str) -> CollectorStats:
        """Collect injuries for a specific sport."""
        stats = CollectorStats(sport=sport_code)
        
        if sport_code not in ESPN_SPORT_PATHS:
            stats.errors.append(f"Unknown sport: {sport_code}")
            return stats
        
        sport_path = ESPN_SPORT_PATHS[sport_code]
        console.print(f"[cyan]Collecting {sport_code} injuries...[/cyan]")
        
        try:
            # Get all teams first
            teams = await self._get_teams(sport_path)
            stats.teams_processed = len(teams)
            
            all_injuries = []
            
            # Collect injuries for each team
            for team in teams:
                try:
                    team_injuries = await self._get_team_injuries(
                        sport_path, team, sport_code
                    )
                    all_injuries.extend(team_injuries)
                    await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error getting injuries for {team.get('name')}: {e}")
            
            stats.injuries_found = len(all_injuries)
            
            # Archive raw data
            if all_injuries:
                await self._archive_raw_data(sport_code, all_injuries)
                stats.raw_archived = True
            
            # Save to database
            for injury in all_injuries:
                try:
                    saved = await self._save_injury_to_db(injury)
                    if saved:
                        stats.injuries_saved += 1
                except Exception as e:
                    stats.errors.append(f"Save error: {str(e)[:50]}")
            
            console.print(
                f"  [green]✓[/green] {sport_code}: {stats.injuries_found} injuries, "
                f"{stats.injuries_saved} saved"
            )
            
        except Exception as e:
            stats.errors.append(str(e))
            logger.error(f"Error collecting {sport_code}: {e}")
        
        return stats
    
    async def _get_teams(self, sport_path: Dict[str, str]) -> List[Dict]:
        """Get all teams for a sport."""
        url = f"{self.BASE_URL}/{sport_path['sport']}/{sport_path['league']}/teams"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                teams = []
                for sport in data.get("sports", []):
                    for league in sport.get("leagues", []):
                        for team_entry in league.get("teams", []):
                            team = team_entry.get("team", {})
                            teams.append({
                                "id": team.get("id"),
                                "name": team.get("displayName"),
                                "abbreviation": team.get("abbreviation"),
                            })
                
                return teams
                
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []
    
    async def _get_team_injuries(
        self, 
        sport_path: Dict[str, str], 
        team: Dict,
        sport_code: str
    ) -> List[InjuryData]:
        """Get injuries for a specific team."""
        team_id = team.get("id")
        if not team_id:
            return []
        
        # Try the team roster endpoint which includes injury info
        url = f"{self.BASE_URL}/{sport_path['sport']}/{sport_path['league']}/teams/{team_id}/roster"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                return self._parse_roster_injuries(data, team, sport_code)
                
        except Exception as e:
            logger.debug(f"Error fetching roster for {team.get('name')}: {e}")
            return []
    
    def _parse_roster_injuries(
        self, 
        data: Dict, 
        team: Dict,
        sport_code: str
    ) -> List[InjuryData]:
        """Parse injuries from roster data."""
        injuries = []
        
        # ESPN roster structure varies by sport
        athletes = []
        
        # Try different roster structures
        for group in data.get("athletes", []):
            if isinstance(group, dict):
                athletes.extend(group.get("items", []))
            elif isinstance(group, list):
                athletes.extend(group)
        
        # Also check direct roster
        if "roster" in data:
            athletes.extend(data.get("roster", []))
        
        for athlete in athletes:
            try:
                # Check if player has injury
                injuries_data = athlete.get("injuries", [])
                
                if not injuries_data:
                    # Check status
                    status = athlete.get("status", {})
                    if status.get("type") in ["injury", "injured"]:
                        injuries_data = [{
                            "type": {"name": status.get("name", "Unknown")},
                            "details": {"type": status.get("name", "")},
                            "status": status.get("abbreviation", "Unknown")
                        }]
                
                for injury_info in injuries_data:
                    injury_type = injury_info.get("type", {})
                    details = injury_info.get("details", {})
                    
                    # Parse injury data
                    injury = InjuryData(
                        player_name=athlete.get("displayName", athlete.get("fullName", "Unknown")),
                        team_id=str(team.get("id", "")),
                        team_name=team.get("name", ""),
                        sport_code=sport_code,
                        position=athlete.get("position", {}).get("abbreviation", ""),
                        injury_type=injury_type.get("name", "") or injury_type.get("description", ""),
                        body_part=details.get("location", "") or details.get("type", ""),
                        status=self._normalize_status(injury_info.get("status", "Unknown")),
                        status_detail=injury_info.get("longComment", injury_info.get("shortComment", "")),
                        is_starter=athlete.get("starter", False),
                        external_id=str(athlete.get("id", "")),
                    )
                    
                    injuries.append(injury)
                    
            except Exception as e:
                logger.debug(f"Error parsing athlete: {e}")
                continue
        
        return injuries
    
    def _normalize_status(self, status: str) -> str:
        """Normalize injury status string."""
        status_lower = str(status).lower().strip()
        
        status_map = {
            "o": "Out",
            "out": "Out",
            "ir": "IR",
            "injured reserve": "IR",
            "d": "Doubtful",
            "doubtful": "Doubtful",
            "q": "Questionable",
            "questionable": "Questionable",
            "p": "Probable",
            "probable": "Probable",
            "day-to-day": "Day-To-Day",
            "dtd": "Day-To-Day",
            "suspended": "Suspended",
            "sus": "Suspended",
        }
        
        for key, value in status_map.items():
            if key in status_lower:
                return value
        
        return status.title() if status else "Unknown"
    
    async def _archive_raw_data(self, sport_code: str, injuries: List[InjuryData]):
        """Archive raw injury data to HDD."""
        now = datetime.utcnow()
        dir_path = Path(self.RAW_DATA_PATH) / sport_code / now.strftime("%Y/%m")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{now.strftime('%d_%H%M%S')}.json.gz"
        file_path = dir_path / file_name
        
        data = {
            "sport": sport_code,
            "collected_at": now.isoformat(),
            "injury_count": len(injuries),
            "injuries": [
                {
                    "player_name": inj.player_name,
                    "team_id": inj.team_id,
                    "team_name": inj.team_name,
                    "position": inj.position,
                    "injury_type": inj.injury_type,
                    "body_part": inj.body_part,
                    "status": inj.status,
                    "status_detail": inj.status_detail,
                    "is_starter": inj.is_starter,
                    "impact_score": inj.impact_score,
                    "external_id": inj.external_id,
                }
                for inj in injuries
            ]
        }
        
        async with aiofiles.open(file_path, 'wb') as f:
            json_str = json.dumps(data, indent=2)
            compressed = gzip.compress(json_str.encode('utf-8'))
            await f.write(compressed)
        
        logger.info(f"Archived {len(injuries)} injuries to {file_path}")
    
    async def _save_injury_to_db(self, injury: InjuryData) -> bool:
        """Save injury to database."""
        try:
            from app.core.database import get_async_session
            from app.models.injury_models import Injury
            from app.models import Team
            from sqlalchemy import select, and_
            
            async with get_async_session() as session:
                # Find team
                result = await session.execute(
                    select(Team).where(Team.external_id == injury.team_id)
                )
                team = result.scalar_one_or_none()
                
                if not team:
                    # Try by name
                    result = await session.execute(
                        select(Team).where(Team.name.ilike(f"%{injury.team_name}%"))
                    )
                    team = result.scalar_one_or_none()
                
                if not team:
                    logger.warning(f"Team not found: {injury.team_name}")
                    return False
                
                # Check if injury already exists
                result = await session.execute(
                    select(Injury).where(
                        and_(
                            Injury.player_name == injury.player_name,
                            Injury.team_id == team.id,
                            Injury.sport_code == injury.sport_code,
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing
                    existing.injury_type = injury.injury_type
                    existing.body_part = injury.body_part
                    existing.status = injury.status
                    existing.status_detail = injury.status_detail
                    existing.is_starter = injury.is_starter
                    existing.impact_score = injury.impact_score
                    existing.last_updated = datetime.utcnow()
                else:
                    # Create new
                    db_injury = Injury(
                        id=uuid4(),
                        team_id=team.id,
                        sport_code=injury.sport_code,
                        player_name=injury.player_name,
                        position=injury.position,
                        injury_type=injury.injury_type,
                        body_part=injury.body_part,
                        status=injury.status,
                        status_detail=injury.status_detail,
                        is_starter=injury.is_starter,
                        impact_score=injury.impact_score,
                        source="espn",
                        external_id=injury.external_id,
                    )
                    session.add(db_injury)
                
                await session.commit()
                return True
                
        except ImportError:
            logger.warning("Database modules not available")
            return False
        except Exception as e:
            logger.error(f"Error saving injury: {e}")
            return False
    
    def _print_summary(self):
        """Print collection summary."""
        if not HAS_RICH:
            console.print("\n=== Injury Collection Summary ===")
            for sport, stats in self.stats.items():
                console.print(f"{sport}: {stats.injuries_found} found, {stats.injuries_saved} saved")
            return
        
        table = Table(title="Injury Collection Summary")
        table.add_column("Sport", style="cyan")
        table.add_column("Teams", style="blue")
        table.add_column("Injuries Found", style="yellow")
        table.add_column("Saved", style="green")
        table.add_column("Archived", style="magenta")
        table.add_column("Errors", style="red")
        
        total_found = 0
        total_saved = 0
        
        for sport, stats in self.stats.items():
            table.add_row(
                sport,
                str(stats.teams_processed),
                str(stats.injuries_found),
                str(stats.injuries_saved),
                "✓" if stats.raw_archived else "-",
                str(len(stats.errors)) if stats.errors else "-"
            )
            total_found += stats.injuries_found
            total_saved += stats.injuries_saved
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "-",
            f"[bold]{total_found}[/bold]",
            f"[bold]{total_saved}[/bold]",
            "-",
            "-"
        )
        
        console.print(table)
        console.print(f"\n[cyan]Raw data archived to:[/cyan] {self.RAW_DATA_PATH}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect injury data from ESPN"
    )
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, NBA, etc.)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Collect for all sports"
    )
    parser.add_argument(
        "--list-sports",
        action="store_true",
        help="List available sports"
    )
    
    args = parser.parse_args()
    
    if args.list_sports:
        console.print("[bold]Available Sports:[/bold]")
        for sport in ESPN_SPORT_PATHS.keys():
            console.print(f"  • {sport}")
        return
    
    if not args.sport and not args.all:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return
    
    async with InjuryCollector() as collector:
        if args.all:
            await collector.collect_all_sports()
        else:
            await collector.collect_sport(args.sport.upper())


if __name__ == "__main__":
    asyncio.run(main())
