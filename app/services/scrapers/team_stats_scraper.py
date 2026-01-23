"""
Team Statistics Web Scraper
===========================
Scraper for collecting team statistics from sports websites.

CONFIGURATION:
1. Add your target URLs to the URLS list below
2. Update the CSS selectors in parse_page() to match the website structure
3. Run the scraper
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup

from .base_scraper import (
    BaseWebScraper, 
    ScraperConfig, 
    ScrapedData,
    extract_text,
    extract_texts,
    extract_number
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your team stats URLs here
    # Example: "https://www.example-sports.com/nfl/teams/statistics"
]

BASE_URL = ""

# ============================================================================


class TeamStatsScraper(BaseWebScraper):
    """
    Scraper for collecting team statistics.
    
    Extracts:
    - Team name and record
    - Offensive statistics
    - Defensive statistics
    - Rankings
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="team_stats_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=3600,  # Stats don't change often
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse team statistics page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        teams = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        # Find stats table rows
        team_rows = soup.select("table tbody tr, .team-row, .stats-row")
        
        for row in team_rows:
            try:
                team_data = self._parse_team_row(row)
                if team_data:
                    teams.append(team_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "team_count": len(teams),
            "teams": teams
        }
    
    def _parse_team_row(self, row: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single team statistics row."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        # Get all cells in the row
        cells = row.select("td")
        
        if len(cells) < 2:
            return None
        
        team_name = extract_text(row, ".team-name, td:first-child a, .team")
        
        if not team_name:
            return None
        
        # Extract statistics - customize based on available columns
        return {
            "team_name": team_name,
            "wins": extract_number(extract_text(row, ".wins, td:nth-child(2)")),
            "losses": extract_number(extract_text(row, ".losses, td:nth-child(3)")),
            "points_per_game": extract_number(extract_text(row, ".ppg, td:nth-child(4)")),
            "points_allowed": extract_number(extract_text(row, ".papg, td:nth-child(5)")),
            "offensive_rating": extract_number(extract_text(row, ".ortg")),
            "defensive_rating": extract_number(extract_text(row, ".drtg")),
        }
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_team_stats(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape team statistics."""
    scraper = TeamStatsScraper(urls=urls)
    return await scraper.scrape_all()
