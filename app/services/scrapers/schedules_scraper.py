"""
Schedules Web Scraper
=====================
Scraper for collecting game schedules from sports websites.

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
    extract_attr
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your schedule URLs here
    # Example: "https://www.example-sports.com/nfl/schedule"
]

BASE_URL = ""

# ============================================================================


class SchedulesScraper(BaseWebScraper):
    """
    Scraper for collecting game schedules.
    
    Extracts:
    - Home and away teams
    - Game date and time
    - Venue/location
    - TV broadcast info
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="schedules_scraper",
            scraper_version="1.0.0",
            requests_per_minute=25,
            min_delay_seconds=1.5,
            max_delay_seconds=3.0,
            cache_ttl_seconds=3600,  # Schedules don't change often
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse schedule page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        games = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        game_elements = soup.select(
            ".schedule-row, .game-row, .event, table tbody tr, "
            "[data-testid='game'], .matchup"
        )
        
        for game_el in game_elements:
            try:
                game_data = self._parse_game(game_el)
                if game_data:
                    games.append(game_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "game_count": len(games),
            "games": games
        }
    
    def _parse_game(self, game_el: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single game schedule element."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        home_team = extract_text(game_el, ".home-team, .team-home, [data-team='home']")
        away_team = extract_text(game_el, ".away-team, .team-away, [data-team='away']")
        
        if not home_team and not away_team:
            return None
        
        game_date = extract_text(game_el, ".date, .game-date, time")
        game_time = extract_text(game_el, ".time, .game-time, .start-time")
        venue = extract_text(game_el, ".venue, .location, .stadium")
        tv_network = extract_text(game_el, ".tv, .broadcast, .network")
        
        # Try to get datetime attribute
        datetime_attr = extract_attr(game_el, "time, [datetime]", "datetime")
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "game_date": game_date,
            "game_time": game_time,
            "datetime_iso": datetime_attr,
            "venue": venue,
            "tv_network": tv_network
        }
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB", "wnba": "WNBA",
                  "atp": "ATP", "wta": "WTA", "cfl": "CFL"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_schedules(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape schedules."""
    scraper = SchedulesScraper(urls=urls)
    return await scraper.scrape_all()
