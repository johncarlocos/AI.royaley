"""
Scores & Results Web Scraper
============================
Scraper for collecting game scores and results from sports websites.

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
    extract_number
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your scores/results URLs here
    # Example: "https://www.example-sports.com/scores/nfl"
    # Example: "https://www.example-sports.com/scores/nba"
]

BASE_URL = ""  # Set your base URL here

# ============================================================================


class ScoresScraper(BaseWebScraper):
    """
    Scraper for collecting game scores and results.
    
    Extracts:
    - Teams playing
    - Final scores
    - Game status (final, in progress, scheduled)
    - Quarter/period scores
    - Game date
    
    Usage:
        scraper = ScoresScraper()
        results = await scraper.scrape_all()
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="scores_scraper",
            scraper_version="1.0.0",
            requests_per_minute=30,
            min_delay_seconds=1.0,
            max_delay_seconds=3.0,
            cache_ttl_seconds=30,  # Scores update frequently during games
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse scores page and extract game results.
        
        CUSTOMIZE THE SELECTORS BELOW for your target website.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        games = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        game_elements = soup.select(".scoreboard, .game-card, .score-container, [data-testid='scoreboard']")
        
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
        """Parse a single game score element."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        home_team = extract_text(game_el, ".home-team, .team-home")
        away_team = extract_text(game_el, ".away-team, .team-away")
        
        home_score = extract_number(extract_text(game_el, ".home-score, .score-home"))
        away_score = extract_number(extract_text(game_el, ".away-score, .score-away"))
        
        status = extract_text(game_el, ".game-status, .status, .game-state")
        game_time = extract_text(game_el, ".game-time, .game-clock")
        
        if not home_team and not away_team:
            return None
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "status": status or "unknown",
            "game_time": game_time,
            "is_final": "final" in status.lower() if status else False
        }
    
    def _detect_sport(self, url: str) -> str:
        """Detect sport from URL."""
        url_lower = url.lower()
        sports = {
            "nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
            "ncaaf": "NCAAF", "ncaab": "NCAAB", "wnba": "WNBA",
            "atp": "ATP", "wta": "WTA", "cfl": "CFL"
        }
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_scores(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape scores."""
    scraper = ScoresScraper(urls=urls)
    return await scraper.scrape_all()
