"""
Standings Web Scraper
=====================
Scraper for collecting team standings from sports websites.

CONFIGURATION:
1. Add your target URLs to the URLS list below
2. Update the CSS selectors to match the website structure
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
    extract_number
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your standings URLs here
    # Example: "https://www.example-sports.com/nfl/standings"
]

BASE_URL = ""

# ============================================================================


class StandingsScraper(BaseWebScraper):
    """
    Scraper for collecting team standings.
    
    Extracts:
    - Team name
    - Wins, losses, ties
    - Win percentage
    - Division/conference
    - Games behind
    - Streak
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="standings_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=3600,
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse standings page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        standings = []
        current_division = ""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        # Find division headers and team rows
        elements = soup.select(
            ".standings-table tr, .division-header, .team-row, "
            "table tbody tr, [data-testid='standings-row']"
        )
        
        for el in elements:
            try:
                # Check if this is a division/conference header
                if el.select_one(".division, .conference, th"):
                    div_text = extract_text(el, ".division, .conference, th")
                    if div_text:
                        current_division = div_text
                    continue
                
                team_data = self._parse_team_row(el, current_division)
                if team_data:
                    standings.append(team_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "team_count": len(standings),
            "standings": standings
        }
    
    def _parse_team_row(self, row: BeautifulSoup, division: str) -> Optional[Dict[str, Any]]:
        """Parse a single standings row."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        team_name = extract_text(row, ".team-name, .team, td:first-child a")
        
        if not team_name:
            return None
        
        wins = extract_number(extract_text(row, ".wins, .w, td:nth-child(2)"))
        losses = extract_number(extract_text(row, ".losses, .l, td:nth-child(3)"))
        ties = extract_number(extract_text(row, ".ties, .t"))
        
        win_pct = extract_number(extract_text(row, ".pct, .win-pct"))
        games_behind = extract_text(row, ".gb, .games-behind")
        streak = extract_text(row, ".streak, .strk")
        
        # Calculate win percentage if not provided
        if win_pct == 0 and (wins + losses) > 0:
            win_pct = round(wins / (wins + losses), 3)
        
        return {
            "team_name": team_name,
            "division": division,
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
            "win_pct": win_pct,
            "games_behind": games_behind,
            "streak": streak
        }
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB", "wnba": "WNBA"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_standings(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape standings."""
    scraper = StandingsScraper(urls=urls)
    return await scraper.scrape_all()
