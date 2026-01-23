"""
Odds Web Scraper
================
Scraper for collecting odds data from sportsbook websites.

CONFIGURATION:
1. Add your target URLs to the URLS list below
2. Update the CSS selectors in parse_page() to match the website structure
3. Run the scraper

Example URLs (replace with actual URLs):
- https://example-sportsbook.com/nfl/odds
- https://example-sportsbook.com/nba/odds
"""

import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup

from .base_scraper import (
    BaseWebScraper, 
    ScraperConfig, 
    ScrapedData,
    extract_text,
    extract_texts,
    extract_number,
    extract_odds
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your sportsbook odds URLs here
    # Example: "https://www.example-sportsbook.com/sports/nfl/odds"
    # Example: "https://www.example-sportsbook.com/sports/nba/odds"
]

BASE_URL = ""  # Set your base URL here, e.g., "https://www.example-sportsbook.com"

# ============================================================================


class OddsScraper(BaseWebScraper):
    """
    Scraper for collecting odds from sportsbook websites.
    
    Extracts:
    - Game matchups (home team, away team)
    - Spread odds
    - Moneyline odds  
    - Total (over/under) odds
    - Game date/time
    
    Usage:
        scraper = OddsScraper()
        results = await scraper.scrape_all()
        
        for result in results:
            if result.success:
                print(result.data)
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        """
        Initialize the odds scraper.
        
        Args:
            urls: Optional list of URLs. If None, uses URLS constant.
        """
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="odds_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,  # Be respectful to the server
            min_delay_seconds=2.0,
            max_delay_seconds=5.0,
            cache_ttl_seconds=60,  # Odds change frequently
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse odds page and extract betting lines.
        
        CUSTOMIZE THIS METHOD:
        Update the CSS selectors to match your target website's HTML structure.
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary with extracted odds data
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        games = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW TO MATCH YOUR TARGET WEBSITE
        # ================================================================
        
        # Example: Find all game containers
        # game_elements = soup.select(".game-container")
        game_elements = soup.select(".game, .event, .matchup, [data-testid='event']")
        
        for game_el in game_elements:
            try:
                game_data = self._parse_game(game_el)
                if game_data:
                    games.append(game_data)
            except Exception as e:
                # Log error but continue with other games
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "game_count": len(games),
            "games": games
        }
    
    def _parse_game(self, game_el: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """
        Parse a single game element.
        
        CUSTOMIZE THIS METHOD for your target website.
        """
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        # Example selectors - update for your target site
        home_team = extract_text(game_el, ".home-team, .team-home, [data-team='home']")
        away_team = extract_text(game_el, ".away-team, .team-away, [data-team='away']")
        
        # Game time
        game_time = extract_text(game_el, ".game-time, .event-time, .start-time")
        
        # Spread
        home_spread = extract_text(game_el, ".home-spread .line, [data-type='spread'] .home")
        away_spread = extract_text(game_el, ".away-spread .line, [data-type='spread'] .away")
        home_spread_odds = extract_odds(extract_text(game_el, ".home-spread .odds"))
        away_spread_odds = extract_odds(extract_text(game_el, ".away-spread .odds"))
        
        # Moneyline
        home_ml = extract_odds(extract_text(game_el, ".home-ml, [data-type='moneyline'] .home"))
        away_ml = extract_odds(extract_text(game_el, ".away-ml, [data-type='moneyline'] .away"))
        
        # Total
        total_line = extract_text(game_el, ".total-line, [data-type='total'] .line")
        over_odds = extract_odds(extract_text(game_el, ".over-odds, [data-type='total'] .over"))
        under_odds = extract_odds(extract_text(game_el, ".under-odds, [data-type='total'] .under"))
        
        # Only return if we have team names
        if not home_team and not away_team:
            return None
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "game_time": game_time,
            "spread": {
                "home_line": extract_number(home_spread),
                "away_line": extract_number(away_spread),
                "home_odds": home_spread_odds,
                "away_odds": away_spread_odds
            },
            "moneyline": {
                "home": home_ml,
                "away": away_ml
            },
            "total": {
                "line": extract_number(total_line),
                "over_odds": over_odds,
                "under_odds": under_odds
            }
        }
    
    def _detect_sport(self, url: str) -> str:
        """Detect sport from URL."""
        url_lower = url.lower()
        if "nfl" in url_lower or "football" in url_lower:
            return "NFL"
        elif "nba" in url_lower or "basketball" in url_lower:
            return "NBA"
        elif "mlb" in url_lower or "baseball" in url_lower:
            return "MLB"
        elif "nhl" in url_lower or "hockey" in url_lower:
            return "NHL"
        elif "ncaaf" in url_lower or "college-football" in url_lower:
            return "NCAAF"
        elif "ncaab" in url_lower or "college-basketball" in url_lower:
            return "NCAAB"
        return "UNKNOWN"


# Convenience function
async def scrape_odds(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """
    Convenience function to scrape odds.
    
    Args:
        urls: Optional list of URLs to scrape
        
    Returns:
        List of ScrapedData objects
    """
    scraper = OddsScraper(urls=urls)
    return await scraper.scrape_all()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Add your URLs here
        test_urls = [
            # "https://example.com/odds/nfl"
        ]
        
        if not test_urls:
            print("Please add URLs to test_urls list")
            return
        
        results = await scrape_odds(test_urls)
        
        for result in results:
            print(f"\nURL: {result.url}")
            print(f"Success: {result.success}")
            if result.success:
                print(f"Games found: {result.data.get('game_count', 0)}")
            else:
                print(f"Error: {result.error_message}")
    
    asyncio.run(main())
