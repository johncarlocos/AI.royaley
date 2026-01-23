"""
Player Props Web Scraper
========================
Scraper for collecting player prop betting lines from sportsbooks.

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
    # Add your player props URLs here
    # Example: "https://www.example-sportsbook.com/props/nba"
]

BASE_URL = ""

# ============================================================================


class PlayerPropsScraper(BaseWebScraper):
    """
    Scraper for collecting player prop lines.
    
    Extracts:
    - Player name and team
    - Prop type (points, rebounds, assists, etc.)
    - Line value
    - Over/under odds
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="player_props_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=300,  # Props update frequently
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse player props page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        props = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        prop_elements = soup.select(
            ".prop-row, .player-prop, .prop-bet, "
            "[data-testid='prop'], table tbody tr"
        )
        
        for prop_el in prop_elements:
            try:
                prop_data = self._parse_prop(prop_el)
                if prop_data:
                    props.append(prop_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "prop_count": len(props),
            "props": props
        }
    
    def _parse_prop(self, prop_el: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single player prop element."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        player_name = extract_text(prop_el, ".player-name, .player, [data-player]")
        
        if not player_name:
            return None
        
        team = extract_text(prop_el, ".team, .player-team")
        prop_type = extract_text(prop_el, ".prop-type, .market, .bet-type")
        line = extract_number(extract_text(prop_el, ".line, .prop-line, .value"))
        
        over_odds = extract_odds(extract_text(prop_el, ".over-odds, .over .odds"))
        under_odds = extract_odds(extract_text(prop_el, ".under-odds, .under .odds"))
        
        # Normalize prop type
        prop_type_normalized = self._normalize_prop_type(prop_type)
        
        return {
            "player_name": player_name,
            "team": team,
            "prop_type": prop_type_normalized,
            "prop_type_raw": prop_type,
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds
        }
    
    def _normalize_prop_type(self, prop_type: str) -> str:
        """Normalize prop type to standard format."""
        if not prop_type:
            return "UNKNOWN"
        
        prop_lower = prop_type.lower()
        
        # Basketball
        if "point" in prop_lower:
            return "POINTS"
        elif "rebound" in prop_lower:
            return "REBOUNDS"
        elif "assist" in prop_lower:
            return "ASSISTS"
        elif "three" in prop_lower or "3pt" in prop_lower:
            return "THREES"
        elif "steal" in prop_lower:
            return "STEALS"
        elif "block" in prop_lower:
            return "BLOCKS"
        elif "pra" in prop_lower:
            return "PRA"  # Points + Rebounds + Assists
        
        # Football
        elif "passing" in prop_lower and "yard" in prop_lower:
            return "PASSING_YARDS"
        elif "rushing" in prop_lower and "yard" in prop_lower:
            return "RUSHING_YARDS"
        elif "receiving" in prop_lower and "yard" in prop_lower:
            return "RECEIVING_YARDS"
        elif "reception" in prop_lower:
            return "RECEPTIONS"
        elif "touchdown" in prop_lower:
            return "TOUCHDOWNS"
        
        # Baseball
        elif "strikeout" in prop_lower:
            return "STRIKEOUTS"
        elif "hit" in prop_lower:
            return "HITS"
        elif "rbi" in prop_lower:
            return "RBIS"
        elif "total base" in prop_lower:
            return "TOTAL_BASES"
        
        # Hockey
        elif "goal" in prop_lower:
            return "GOALS"
        elif "shot" in prop_lower:
            return "SHOTS"
        elif "save" in prop_lower:
            return "SAVES"
        
        return prop_type.upper().replace(" ", "_")
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB", "wnba": "WNBA"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_player_props(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape player props."""
    scraper = PlayerPropsScraper(urls=urls)
    return await scraper.scrape_all()
