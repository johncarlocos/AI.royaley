"""
ROYALEY - RealGM/NBA Salary Data Collector
Phase 1: Data Collection Services

Collector 20: RealGM-style NBA salary and contract data collection.

Since RealGM blocks automated scraping (403 errors), this collector uses
ESPN's publicly available salary data as an alternative source, which provides:
- Player salaries by season (1999-2000 to present)
- Team salary totals
- Contract details

Data Source:
- Primary: ESPN NBA Salaries (https://www.espn.com/nba/salaries)
- Alternative: ESPN Team Rosters for salary data

FREE data - no API key required!

Tables Filled:
- sports - Sport definitions (NBA)
- teams - NBA team info
- players - Player info with salary data
- player_stats - Player salary stats by season (stat_type='salary')
- seasons - Season definitions

Usage:
    from app.services.collectors import realgm_collector
    
    # Collect current season salaries
    result = await realgm_collector.collect(years=[2025])
    
    # Collect 10 years of salary history
    result = await realgm_collector.collect(years=list(range(2016, 2026)))
    
    # Save to database
    await realgm_collector.save_to_database(result.data, session)
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from bs4 import BeautifulSoup

import httpx
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Sport, Team, Player, PlayerStats, Season
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ESPN SALARY URL CONFIGURATION
# =============================================================================

ESPN_SALARY_BASE = "https://www.espn.com/nba/salaries/_/year/{year}"
ESPN_SALARY_PAGE = "https://www.espn.com/nba/salaries/_/year/{year}/page/{page}"
ESPN_TEAM_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"

# Rate limiting
REQUEST_DELAY = 1.0  # 1 second between requests to be respectful


# =============================================================================
# NBA TEAM MAPPING
# =============================================================================

NBA_TEAMS = {
    "ATL": {"id": "1", "name": "Atlanta Hawks", "city": "Atlanta", "conference": "Eastern", "division": "Southeast"},
    "BOS": {"id": "2", "name": "Boston Celtics", "city": "Boston", "conference": "Eastern", "division": "Atlantic"},
    "BKN": {"id": "17", "name": "Brooklyn Nets", "city": "Brooklyn", "conference": "Eastern", "division": "Atlantic"},
    "CHA": {"id": "30", "name": "Charlotte Hornets", "city": "Charlotte", "conference": "Eastern", "division": "Southeast"},
    "CHI": {"id": "4", "name": "Chicago Bulls", "city": "Chicago", "conference": "Eastern", "division": "Central"},
    "CLE": {"id": "5", "name": "Cleveland Cavaliers", "city": "Cleveland", "conference": "Eastern", "division": "Central"},
    "DAL": {"id": "6", "name": "Dallas Mavericks", "city": "Dallas", "conference": "Western", "division": "Southwest"},
    "DEN": {"id": "7", "name": "Denver Nuggets", "city": "Denver", "conference": "Western", "division": "Northwest"},
    "DET": {"id": "8", "name": "Detroit Pistons", "city": "Detroit", "conference": "Eastern", "division": "Central"},
    "GS": {"id": "9", "name": "Golden State Warriors", "city": "San Francisco", "conference": "Western", "division": "Pacific"},
    "GSW": {"id": "9", "name": "Golden State Warriors", "city": "San Francisco", "conference": "Western", "division": "Pacific"},
    "HOU": {"id": "10", "name": "Houston Rockets", "city": "Houston", "conference": "Western", "division": "Southwest"},
    "IND": {"id": "11", "name": "Indiana Pacers", "city": "Indianapolis", "conference": "Eastern", "division": "Central"},
    "LAC": {"id": "12", "name": "LA Clippers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "LAL": {"id": "13", "name": "Los Angeles Lakers", "city": "Los Angeles", "conference": "Western", "division": "Pacific"},
    "MEM": {"id": "29", "name": "Memphis Grizzlies", "city": "Memphis", "conference": "Western", "division": "Southwest"},
    "MIA": {"id": "14", "name": "Miami Heat", "city": "Miami", "conference": "Eastern", "division": "Southeast"},
    "MIL": {"id": "15", "name": "Milwaukee Bucks", "city": "Milwaukee", "conference": "Eastern", "division": "Central"},
    "MIN": {"id": "16", "name": "Minnesota Timberwolves", "city": "Minneapolis", "conference": "Western", "division": "Northwest"},
    "NO": {"id": "3", "name": "New Orleans Pelicans", "city": "New Orleans", "conference": "Western", "division": "Southwest"},
    "NOP": {"id": "3", "name": "New Orleans Pelicans", "city": "New Orleans", "conference": "Western", "division": "Southwest"},
    "NY": {"id": "18", "name": "New York Knicks", "city": "New York", "conference": "Eastern", "division": "Atlantic"},
    "NYK": {"id": "18", "name": "New York Knicks", "city": "New York", "conference": "Eastern", "division": "Atlantic"},
    "OKC": {"id": "25", "name": "Oklahoma City Thunder", "city": "Oklahoma City", "conference": "Western", "division": "Northwest"},
    "ORL": {"id": "19", "name": "Orlando Magic", "city": "Orlando", "conference": "Eastern", "division": "Southeast"},
    "PHI": {"id": "20", "name": "Philadelphia 76ers", "city": "Philadelphia", "conference": "Eastern", "division": "Atlantic"},
    "PHX": {"id": "21", "name": "Phoenix Suns", "city": "Phoenix", "conference": "Western", "division": "Pacific"},
    "POR": {"id": "22", "name": "Portland Trail Blazers", "city": "Portland", "conference": "Western", "division": "Northwest"},
    "SA": {"id": "24", "name": "San Antonio Spurs", "city": "San Antonio", "conference": "Western", "division": "Southwest"},
    "SAS": {"id": "24", "name": "San Antonio Spurs", "city": "San Antonio", "conference": "Western", "division": "Southwest"},
    "SAC": {"id": "23", "name": "Sacramento Kings", "city": "Sacramento", "conference": "Western", "division": "Pacific"},
    "TOR": {"id": "28", "name": "Toronto Raptors", "city": "Toronto", "conference": "Eastern", "division": "Atlantic"},
    "UTA": {"id": "26", "name": "Utah Jazz", "city": "Salt Lake City", "conference": "Western", "division": "Northwest"},
    "UTAH": {"id": "26", "name": "Utah Jazz", "city": "Salt Lake City", "conference": "Western", "division": "Northwest"},
    "WAS": {"id": "27", "name": "Washington Wizards", "city": "Washington", "conference": "Eastern", "division": "Southeast"},
    "WSH": {"id": "27", "name": "Washington Wizards", "city": "Washington", "conference": "Eastern", "division": "Southeast"},
}

# Team name variations to abbreviation mapping
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


class RealGMCollector(BaseCollector):
    """
    Collector for NBA salary data (RealGM-style).
    
    Uses ESPN salary pages as primary data source since RealGM 
    blocks automated access.
    """
    
    def __init__(self, db_session=None):
        super().__init__(
            name="realgm",
            base_url="https://www.espn.com",
            rate_limit=30,  # 30 requests per minute
            rate_window=60,
            timeout=30.0,
            max_retries=3
        )
        self.db_session = db_session
        logger.info("Registered collector: RealGM NBA Salaries")
    
    def validate(self) -> bool:
        """Validate collector configuration"""
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers to mimic browser"""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML page with rate limiting"""
        await asyncio.sleep(REQUEST_DELAY)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_headers(), follow_redirects=True)
                
                if response.status_code == 200:
                    return response.text
                else:
                    logger.warning(f"[RealGM] HTTP {response.status_code} for {url}")
                    return None
        except Exception as e:
            logger.error(f"[RealGM] Error fetching {url}: {e}")
            return None
    
    def _parse_salary(self, salary_str: str) -> Optional[int]:
        """Parse salary string to integer (e.g., '$59,606,817' -> 59606817)"""
        if not salary_str:
            return None
        
        # Remove $ and commas
        cleaned = salary_str.replace("$", "").replace(",", "").strip()
        
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    def _parse_salary_page(self, html: str, year: int) -> List[Dict[str, Any]]:
        """Parse ESPN salary page HTML to extract player salaries"""
        salaries = []
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Find all table rows with salary data
            # ESPN uses tables with class "tablehead"
            tables = soup.find_all("table")
            
            for table in tables:
                rows = table.find_all("tr")
                
                for row in rows:
                    cells = row.find_all("td")
                    
                    # Typical format: Rank, Name, Team, Salary
                    if len(cells) >= 4:
                        try:
                            # Extract rank
                            rank_text = cells[0].get_text(strip=True)
                            if not rank_text.isdigit():
                                continue
                            
                            # Extract player name
                            name_cell = cells[1]
                            name_link = name_cell.find("a")
                            if name_link:
                                player_name = name_link.get_text(strip=True)
                                # Try to get player ID from link
                                href = name_link.get("href", "")
                                player_id_match = re.search(r"/id/(\d+)/", href)
                                player_id = player_id_match.group(1) if player_id_match else None
                            else:
                                player_name = name_cell.get_text(strip=True)
                                player_id = None
                            
                            # Extract position from name (e.g., "Stephen Curry, G")
                            position = None
                            if ", " in player_name:
                                parts = player_name.rsplit(", ", 1)
                                player_name = parts[0]
                                position = parts[1] if len(parts) > 1 else None
                            
                            # Extract team
                            team_cell = cells[2]
                            team_link = team_cell.find("a")
                            if team_link:
                                team_name = team_link.get_text(strip=True)
                            else:
                                team_name = team_cell.get_text(strip=True)
                            
                            # Get team abbreviation
                            team_abbr = TEAM_NAME_TO_ABBR.get(team_name)
                            
                            # Extract salary
                            salary_text = cells[3].get_text(strip=True)
                            salary = self._parse_salary(salary_text)
                            
                            if player_name and salary:
                                salaries.append({
                                    "rank": int(rank_text),
                                    "player_name": player_name,
                                    "player_id": player_id,
                                    "position": position,
                                    "team_name": team_name,
                                    "team_abbr": team_abbr,
                                    "salary": salary,
                                    "season_year": year,
                                    "season_name": f"{year-1}-{str(year)[2:]}",  # e.g., "2024-25"
                                })
                        except Exception as e:
                            logger.debug(f"[RealGM] Error parsing row: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"[RealGM] Error parsing salary page: {e}")
        
        return salaries
    
    async def fetch_salaries_for_year(self, year: int) -> List[Dict[str, Any]]:
        """Fetch all salary data for a specific season year"""
        all_salaries = []
        page = 1
        max_pages = 15  # ESPN has ~460 players, 40 per page = ~12 pages
        
        logger.info(f"[RealGM] Fetching salaries for {year-1}-{str(year)[2:]} season...")
        
        while page <= max_pages:
            if page == 1:
                url = ESPN_SALARY_BASE.format(year=year)
            else:
                url = ESPN_SALARY_PAGE.format(year=year, page=page)
            
            html = await self._fetch_page(url)
            
            if not html:
                break
            
            salaries = self._parse_salary_page(html, year)
            
            if not salaries:
                break
            
            all_salaries.extend(salaries)
            logger.info(f"[RealGM] Page {page}: {len(salaries)} players (total: {len(all_salaries)})")
            
            # If we got less than expected, probably last page
            if len(salaries) < 35:
                break
            
            page += 1
        
        logger.info(f"[RealGM] Total for {year}: {len(all_salaries)} player salaries")
        return all_salaries
    
    async def fetch_team_rosters(self) -> List[Dict[str, Any]]:
        """Fetch current rosters with salary info from ESPN API"""
        all_players = []
        
        for abbr, team_info in NBA_TEAMS.items():
            if abbr != team_info.get("name", "").split()[0]:  # Skip duplicates
                if abbr not in ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", 
                               "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
                               "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", 
                               "TOR", "UTA", "WAS"]:
                    continue
            
            team_id = team_info["id"]
            url = ESPN_TEAM_ROSTER.format(team_id=team_id)
            
            try:
                await asyncio.sleep(REQUEST_DELAY)
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        athletes = data.get("athletes", [])
                        
                        for group in athletes:
                            items = group.get("items", [])
                            for player in items:
                                contract = player.get("contract", {})
                                salary = contract.get("salary")
                                
                                player_data = {
                                    "player_id": player.get("id"),
                                    "player_name": player.get("displayName", player.get("fullName")),
                                    "position": player.get("position", {}).get("abbreviation"),
                                    "jersey": player.get("jersey"),
                                    "team_abbr": abbr,
                                    "team_name": team_info["name"],
                                    "salary": salary,
                                    "height": player.get("displayHeight"),
                                    "weight": player.get("displayWeight"),
                                    "age": player.get("age"),
                                    "birthdate": player.get("dateOfBirth"),
                                    "college": player.get("college", {}).get("name") if player.get("college") else None,
                                    "experience": player.get("experience", {}).get("years") if player.get("experience") else None,
                                }
                                all_players.append(player_data)
                        
                        logger.info(f"[RealGM] {team_info['name']}: {len(items) if items else 0} players")
                    else:
                        logger.warning(f"[RealGM] Failed to fetch roster for {team_info['name']}: HTTP {response.status_code}")
                        
            except Exception as e:
                logger.error(f"[RealGM] Error fetching roster for {team_info['name']}: {e}")
        
        logger.info(f"[RealGM] Total roster players: {len(all_players)}")
        return all_players
    
    async def collect(
        self,
        years: List[int] = None,
        collect_type: str = "all"
    ) -> CollectorResult:
        """
        Main collection method for NBA salary data.
        
        Args:
            years: List of years to collect (e.g., [2024, 2025] for 2023-24 and 2024-25 seasons)
            collect_type: Type of collection ('all', 'salaries', 'rosters')
            
        Returns:
            CollectorResult with collected data
        """
        if years is None:
            current_year = datetime.now().year
            # NBA season spans two calendar years, use later year
            if datetime.now().month >= 10:  # Oct-Dec = current season
                years = [current_year + 1]
            else:  # Jan-Sep = previous season
                years = [current_year]
        
        logger.info(f"[RealGM] Collecting NBA salary data for years: {years}, type: {collect_type}")
        
        all_data = {
            "salaries": [],
            "rosters": [],
            "teams": list(NBA_TEAMS.values()),
        }
        total_records = 0
        
        try:
            # Collect salary data by year
            if collect_type in ["all", "salaries"]:
                for year in years:
                    salaries = await self.fetch_salaries_for_year(year)
                    all_data["salaries"].extend(salaries)
                    total_records += len(salaries)
            
            # Collect current rosters (has detailed player info)
            if collect_type in ["all", "rosters"]:
                rosters = await self.fetch_team_rosters()
                all_data["rosters"] = rosters
                total_records += len(rosters)
            
            logger.info(f"[RealGM] Collection complete: {total_records} records")
            
            return CollectorResult(
                success=True,
                data=all_data,
                records_count=total_records,
                metadata={"years": years, "collect_type": collect_type}
            )
            
        except Exception as e:
            logger.error(f"[RealGM] Collection error: {e}")
            return CollectorResult(
                success=False,
                error=str(e),
                records_count=total_records,
                data=all_data if total_records > 0 else None
            )
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """
        Save collected salary data to database.
        
        Args:
            data: Dictionary with salaries, rosters, teams
            session: Database session
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        
        try:
            # Get or create NBA sport
            result = await session.execute(
                select(Sport).where(Sport.code == "NBA")
            )
            sport = result.scalar_one_or_none()
            
            if not sport:
                sport = Sport(
                    code="NBA",
                    name="NBA Basketball",
                    api_key="basketball_nba",
                    feature_count=80,
                    is_active=True
                )
                session.add(sport)
                await session.flush()
            
            sport_id = sport.id
            
            # Cache for teams and seasons
            team_cache: Dict[str, UUID] = {}
            season_cache: Dict[int, UUID] = {}
            player_cache: Dict[str, UUID] = {}
            
            # Process salary data
            salaries = data.get("salaries", [])
            
            for salary_data in salaries:
                try:
                    year = salary_data.get("season_year")
                    team_abbr = salary_data.get("team_abbr")
                    player_name = salary_data.get("player_name")
                    salary = salary_data.get("salary")
                    
                    if not all([year, player_name, salary]):
                        continue
                    
                    # Get or create season
                    if year not in season_cache:
                        season_name = f"{year-1}-{str(year)[2:]}"
                        result = await session.execute(
                            select(Season).where(
                                and_(Season.sport_id == sport_id, Season.year == year)
                            )
                        )
                        season = result.scalar_one_or_none()
                        
                        if not season:
                            season = Season(
                                sport_id=sport_id,
                                year=year,
                                name=season_name,
                                is_active=(year >= datetime.now().year)
                            )
                            session.add(season)
                            await session.flush()
                        
                        season_cache[year] = season.id
                    
                    season_id = season_cache[year]
                    
                    # Get or create team
                    team_id = None
                    if team_abbr and team_abbr not in team_cache:
                        team_info = NBA_TEAMS.get(team_abbr)
                        if team_info:
                            external_id = f"espn_nba_{team_info['id']}"
                            result = await session.execute(
                                select(Team).where(
                                    and_(Team.sport_id == sport_id, Team.external_id == external_id)
                                )
                            )
                            team = result.scalar_one_or_none()
                            
                            if not team:
                                team = Team(
                                    sport_id=sport_id,
                                    external_id=external_id,
                                    name=team_info["name"],
                                    abbreviation=team_abbr,
                                    city=team_info.get("city"),
                                    conference=team_info.get("conference"),
                                    division=team_info.get("division"),
                                    is_active=True
                                )
                                session.add(team)
                                await session.flush()
                            
                            team_cache[team_abbr] = team.id
                    
                    if team_abbr:
                        team_id = team_cache.get(team_abbr)
                    
                    # Get or create player
                    player_key = f"{player_name}_{team_abbr or 'FA'}"
                    external_player_id = salary_data.get("player_id")
                    
                    if player_key not in player_cache:
                        # Try to find existing player
                        if external_player_id:
                            ext_id = f"espn_nba_{external_player_id}"
                            result = await session.execute(
                                select(Player).where(Player.external_id == ext_id)
                            )
                        else:
                            # Search by name
                            result = await session.execute(
                                select(Player).where(
                                    and_(
                                        Player.name == player_name,
                                        Player.team_id == team_id
                                    )
                                )
                            )
                        
                        player = result.scalar_one_or_none()
                        
                        if not player:
                            ext_id = f"espn_nba_{external_player_id}" if external_player_id else f"realgm_{player_name.replace(' ', '_').lower()}"
                            player = Player(
                                team_id=team_id,
                                external_id=ext_id,
                                name=player_name,
                                position=salary_data.get("position"),
                                is_active=True
                            )
                            session.add(player)
                            await session.flush()
                        
                        player_cache[player_key] = player.id
                    
                    player_id = player_cache[player_key]
                    
                    # Create or update salary stat
                    result = await session.execute(
                        select(PlayerStats).where(
                            and_(
                                PlayerStats.player_id == player_id,
                                PlayerStats.season_id == season_id,
                                PlayerStats.stat_type == "salary"
                            )
                        )
                    )
                    existing_stat = result.scalar_one_or_none()
                    
                    if existing_stat:
                        existing_stat.value = float(salary)
                    else:
                        stat = PlayerStats(
                            player_id=player_id,
                            season_id=season_id,
                            stat_type="salary",
                            value=float(salary)
                        )
                        session.add(stat)
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.debug(f"[RealGM] Error saving salary for {salary_data.get('player_name')}: {e}")
                    continue
            
            # Process roster data (additional player details)
            rosters = data.get("rosters", [])
            
            for roster_data in rosters:
                try:
                    player_name = roster_data.get("player_name")
                    team_abbr = roster_data.get("team_abbr")
                    
                    if not player_name:
                        continue
                    
                    # Get team_id
                    team_id = team_cache.get(team_abbr)
                    
                    # Find or create player
                    external_player_id = roster_data.get("player_id")
                    if external_player_id:
                        ext_id = f"espn_nba_{external_player_id}"
                        result = await session.execute(
                            select(Player).where(Player.external_id == ext_id)
                        )
                        player = result.scalar_one_or_none()
                        
                        if not player:
                            player = Player(
                                team_id=team_id,
                                external_id=ext_id,
                                name=player_name,
                                position=roster_data.get("position"),
                                jersey_number=int(roster_data.get("jersey")) if roster_data.get("jersey") else None,
                                height=roster_data.get("height"),
                                weight=int(roster_data.get("weight").replace(" lbs", "")) if roster_data.get("weight") else None,
                                is_active=True
                            )
                            session.add(player)
                            saved_count += 1
                        else:
                            # Update existing player with roster details
                            if roster_data.get("position"):
                                player.position = roster_data.get("position")
                            if roster_data.get("jersey"):
                                player.jersey_number = int(roster_data.get("jersey"))
                            if roster_data.get("height"):
                                player.height = roster_data.get("height")
                            if roster_data.get("weight"):
                                try:
                                    player.weight = int(roster_data.get("weight").replace(" lbs", ""))
                                except:
                                    pass
                            player.team_id = team_id
                        
                except Exception as e:
                    logger.debug(f"[RealGM] Error saving roster player {roster_data.get('player_name')}: {e}")
                    continue
            
            await session.commit()
            logger.info(f"[RealGM] Saved {saved_count} records to database")
            
        except Exception as e:
            logger.error(f"[RealGM] Database save error: {e}")
            await session.rollback()
            raise
        
        return saved_count
    
    # =================================================================
    # CONVENIENCE METHODS
    # =================================================================
    
    async def collect_current_season(self) -> CollectorResult:
        """Collect salary data for current NBA season only"""
        current_year = datetime.now().year
        if datetime.now().month >= 10:
            year = current_year + 1
        else:
            year = current_year
        return await self.collect(years=[year], collect_type="all")
    
    async def collect_history(self, years_back: int = 10) -> CollectorResult:
        """Collect historical salary data for specified number of years"""
        current_year = datetime.now().year
        if datetime.now().month >= 10:
            end_year = current_year + 1
        else:
            end_year = current_year
        
        start_year = end_year - years_back + 1
        years = list(range(start_year, end_year + 1))
        
        logger.info(f"[RealGM] Collecting {years_back} years of salary history: {start_year}-{end_year}")
        return await self.collect(years=years, collect_type="salaries")
    
    async def collect_salaries_only(self, years: List[int] = None) -> CollectorResult:
        """Collect only salary data (no rosters)"""
        return await self.collect(years=years, collect_type="salaries")
    
    async def collect_rosters_only(self) -> CollectorResult:
        """Collect only current roster data"""
        return await self.collect(collect_type="rosters")


# Singleton instance for import
realgm_collector = RealGMCollector()