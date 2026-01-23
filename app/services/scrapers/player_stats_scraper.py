"""
Player Statistics Web Scraper
=============================
Scraper for collecting player statistics from sports websites.

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
    # Add your player stats URLs here
    # Example: "https://www.example-sports.com/nba/players/stats"
]

BASE_URL = ""

# ============================================================================


class PlayerStatsScraper(BaseWebScraper):
    """
    Scraper for collecting player statistics.
    
    Extracts:
    - Player name and team
    - Games played
    - Points/goals/yards per game
    - Other sport-specific stats
    """
    
    def __init__(self, urls: Optional[List[str]] = None):
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name="player_stats_scraper",
            scraper_version="1.0.0",
            requests_per_minute=20,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=3600,
        )
        super().__init__(config)
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse player statistics page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        players = []
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        player_rows = soup.select("table tbody tr, .player-row, .stats-row")
        
        for row in player_rows:
            try:
                player_data = self._parse_player_row(row, url)
                if player_data:
                    players.append(player_data)
            except Exception:
                pass
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "sport": self._detect_sport(url),
            "player_count": len(players),
            "players": players
        }
    
    def _parse_player_row(self, row: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Parse a single player statistics row."""
        
        # ================================================================
        # CUSTOMIZE SELECTORS BELOW
        # ================================================================
        
        player_name = extract_text(row, ".player-name, td:first-child a, .player")
        team = extract_text(row, ".team, .player-team")
        
        if not player_name:
            return None
        
        sport = self._detect_sport(url)
        
        # Base stats
        stats = {
            "player_name": player_name,
            "team": team,
            "games_played": extract_number(extract_text(row, ".gp, td:nth-child(2)")),
        }
        
        # Sport-specific stats
        if sport in ["NBA", "NCAAB", "WNBA"]:
            stats.update({
                "points_per_game": extract_number(extract_text(row, ".ppg")),
                "rebounds_per_game": extract_number(extract_text(row, ".rpg")),
                "assists_per_game": extract_number(extract_text(row, ".apg")),
                "steals_per_game": extract_number(extract_text(row, ".spg")),
                "blocks_per_game": extract_number(extract_text(row, ".bpg")),
                "field_goal_pct": extract_number(extract_text(row, ".fgp")),
                "three_point_pct": extract_number(extract_text(row, ".tpp")),
            })
        elif sport in ["NFL", "NCAAF"]:
            stats.update({
                "passing_yards": extract_number(extract_text(row, ".pass-yds")),
                "rushing_yards": extract_number(extract_text(row, ".rush-yds")),
                "receiving_yards": extract_number(extract_text(row, ".rec-yds")),
                "touchdowns": extract_number(extract_text(row, ".td")),
            })
        elif sport == "MLB":
            stats.update({
                "batting_average": extract_number(extract_text(row, ".avg")),
                "home_runs": extract_number(extract_text(row, ".hr")),
                "rbi": extract_number(extract_text(row, ".rbi")),
                "era": extract_number(extract_text(row, ".era")),
            })
        elif sport == "NHL":
            stats.update({
                "goals": extract_number(extract_text(row, ".g")),
                "assists": extract_number(extract_text(row, ".a")),
                "points": extract_number(extract_text(row, ".pts")),
                "plus_minus": extract_number(extract_text(row, ".pm")),
            })
        
        return stats
    
    def _detect_sport(self, url: str) -> str:
        url_lower = url.lower()
        sports = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL",
                  "ncaaf": "NCAAF", "ncaab": "NCAAB", "wnba": "WNBA"}
        for key, value in sports.items():
            if key in url_lower:
                return value
        return "UNKNOWN"


async def scrape_player_stats(urls: Optional[List[str]] = None) -> List[ScrapedData]:
    """Convenience function to scrape player statistics."""
    scraper = PlayerStatsScraper(urls=urls)
    return await scraper.scrape_all()
