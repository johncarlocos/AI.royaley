"""
Line Movements Web Scraper
==========================
Scraper for collecting line movement data from sportsbooks.

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
    extract_number,
    extract_odds
)


# ============================================================================
# CONFIGURATION - ADD YOUR URLS HERE
# ============================================================================

URLS = [
    # Add your line movement URLs here
    # Example: "https://www.example-sportsbook.com/line-movement/nfl"
]

BASE_URL = ""

# ============================================================================


class LineMovementsScraper(BaseWebScraper):
    """
    Scraper for collecting line movement data.
    
    Extracts:
    - Game matchup
    - Opening line
    - Current line
    - Line movement direction and magnitude
    - Timestamps
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="line_movements_scraper",
            scraper_version="1.0.0",
            requests_per_minute=15,
            min_delay_seconds=2.0,
            max_delay_seconds=5.0,
            cache_ttl_seconds=120,  # Lines move frequently
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse line movements page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        movements = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        game_elements = soup.select(
            ".line-movement-row, .game-lines, .movement-container, "
            "[data-testid='line-movement'], table tbody tr"
        )
        
        for game_el in game_elements:
            try:
                movement_data = self._parse_movement(game_el)
                if movement_data:
                    movements.append(movement_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "movement_count": len(movements),
            "movements": movements
        }
    
    def _parse_movement(self, game_el: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single line movement element."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        home_team = extract_text(game_el, ".home-team, .team-home")
        away_team = extract_text(game_el, ".away-team, .team-away")
        
        if not home_team and not away_team:
            return None
        
        # Opening lines
        opening_spread = extract_number(extract_text(game_el, ".opening-spread, .open-spread"))
        opening_total = extract_number(extract_text(game_el, ".opening-total, .open-total"))
        opening_home_ml = extract_odds(extract_text(game_el, ".opening-home-ml"))
        opening_away_ml = extract_odds(extract_text(game_el, ".opening-away-ml"))
        
        # Current lines
        current_spread = extract_number(extract_text(game_el, ".current-spread, .curr-spread"))
        current_total = extract_number(extract_text(game_el, ".current-total, .curr-total"))
        current_home_ml = extract_odds(extract_text(game_el, ".current-home-ml"))
        current_away_ml = extract_odds(extract_text(game_el, ".current-away-ml"))
        
        # Calculate movements
        spread_movement = current_spread - opening_spread if opening_spread else 0
        total_movement = current_total - opening_total if opening_total else 0
        
        # Public betting percentages (if available)
        public_spread = extract_number(extract_text(game_el, ".public-spread, .bet-pct"))
        public_money = extract_number(extract_text(game_el, ".money-pct, .sharp-money"))
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "spread": {
                "opening": opening_spread,
                "current": current_spread,
                "movement": spread_movement,
                "direction": "favorite" if spread_movement < 0 else "underdog" if spread_movement > 0 else "none"
            },
            "total": {
                "opening": opening_total,
                "current": current_total,
                "movement": total_movement,
                "direction": "over" if total_movement > 0 else "under" if total_movement < 0 else "none"
            },
            "moneyline": {
                "opening_home": opening_home_ml,
                "opening_away": opening_away_ml,
                "current_home": current_home_ml,
                "current_away": current_away_ml
            },
            "public_betting": {
                "spread_pct": public_spread,
                "money_pct": public_money
            },
            "steam_move": abs(spread_movement) >= 1.5 or abs(total_movement) >= 2.0,
            "reverse_line_movement": self._detect_rlm(spread_movement, public_spread)
        }
    
    def _detect_rlm(self, spread_movement: float, public_pct: float) -> bool:
        """Detect reverse line movement."""
        if not public_pct:
            return False
        # RLM: line moves against public betting
        # If public is >60% on one side but line moves other way
        if public_pct > 60 and spread_movement > 0:
            return True
        if public_pct < 40 and spread_movement < 0:
            return True
        return False
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_line_movements(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape line movements."""
    scraper = LineMovementsScraper(urls=urls)
    return await scraper.scrape_all()
