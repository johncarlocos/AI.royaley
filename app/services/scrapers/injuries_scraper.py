"""
Injuries Web Scraper
====================
Scraper for collecting injury reports from sports websites.

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
    extract_text
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your injury report URLs here
    # Example: "https://www.example-sports.com/nfl/injuries"
]

BASE_URL = ""

# ============================================================================


class InjuriesScraper(BaseWebScraper):
    """
    Scraper for collecting injury reports.
    
    Extracts:
    - Player name
    - Team
    - Injury type
    - Status (Out, Doubtful, Questionable, Probable)
    - Expected return date
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="injuries_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=1800,  # 30 min - injuries update throughout day
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse injury report page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        injuries = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        # Try multiple common patterns
        injury_rows = soup.select(
            ".injury-row, .player-injury, table tbody tr, "
            "[data-testid='injury-row'], .injury-report-item"
        )
        
        for row in injury_rows:
            try:
                injury_data = self._parse_injury_row(row)
                if injury_data:
                    injuries.append(injury_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "injury_count": len(injuries),
            "injuries": injuries
        }
    
    def _parse_injury_row(self, row: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single injury row."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        player_name = extract_text(row, ".player-name, .player, td:first-child")
        
        if not player_name:
            return None
        
        team = extract_text(row, ".team, .player-team")
        injury = extract_text(row, ".injury, .injury-type, .injury-description")
        status = extract_text(row, ".status, .injury-status")
        
        # Normalize status
        status_lower = status.lower() if status else ""
        normalized_status = "Unknown"
        
        if "out" in status_lower:
            normalized_status = "Out"
        elif "doubtful" in status_lower:
            normalized_status = "Doubtful"
        elif "questionable" in status_lower:
            normalized_status = "Questionable"
        elif "probable" in status_lower:
            normalized_status = "Probable"
        elif "day-to-day" in status_lower or "dtd" in status_lower:
            normalized_status = "Day-to-Day"
        elif "ir" in status_lower or "injured reserve" in status_lower:
            normalized_status = "IR"
        
        return {
            "player_name": player_name,
            "team": team,
            "injury": injury,
            "status": normalized_status,
            "status_raw": status,
            "is_out": normalized_status in ["Out", "IR"],
            "is_questionable": normalized_status in ["Questionable", "Doubtful", "Day-to-Day"]
        }
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_injuries(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape injury reports."""
    scraper = InjuriesScraper(urls=urls)
    return await scraper.scrape_all()
