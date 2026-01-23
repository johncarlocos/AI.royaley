"""
Web Scrapers Module
===================

This module provides 10 ready-to-use web scrapers for collecting sports data.
Each scraper is a placeholder that you can configure with your target URLs.

Available Scrapers:
1. OddsScraper - Collect odds from sportsbooks
2. ScoresScraper - Collect game scores and results
3. TeamStatsScraper - Collect team statistics
4. PlayerStatsScraper - Collect player statistics
5. InjuriesScraper - Collect injury reports
6. SchedulesScraper - Collect game schedules
7. StandingsScraper - Collect team standings
8. PlayerPropsScraper - Collect player prop lines
9. LineMovementsScraper - Collect line movement data
10. GenericScraper - Flexible scraper for any website

Quick Start:
    from app.services.scrapers import OddsScraper
    
    # Create scraper with your URLs
    scraper = OddsScraper(urls=[
        "https://example-sportsbook.com/nfl/odds",
        "https://example-sportsbook.com/nba/odds"
    ])
    
    # Scrape all URLs
    results = await scraper.scrape_all()
    
    # Process results
    for result in results:
        if result.success:
            print(result.data)

Configuration:
    Each scraper can be configured with:
    - urls: List of URLs to scrape
    - Custom rate limiting settings
    - Proxy support
    - Custom headers
    - Cache settings
    
    See base_scraper.py for ScraperConfig options.

Customization:
    To customize a scraper for your target website:
    1. Inherit from the appropriate scraper class
    2. Override the parse_page() method
    3. Update the CSS selectors to match your target site's HTML
"""

# Base classes and utilities
from .base_scraper import (
    BaseWebScraper,
    ScraperConfig,
    ScrapedData,
    ScraperStatus,
    extract_text,
    extract_texts,
    extract_attr,
    extract_number,
    extract_odds,
)

# Scraper implementations
from .odds_scraper import OddsScraper, scrape_odds
from .scores_scraper import ScoresScraper, scrape_scores
from .team_stats_scraper import TeamStatsScraper, scrape_team_stats
from .player_stats_scraper import PlayerStatsScraper, scrape_player_stats
from .injuries_scraper import InjuriesScraper, scrape_injuries
from .schedules_scraper import SchedulesScraper, scrape_schedules
from .standings_scraper import StandingsScraper, scrape_standings
from .player_props_scraper import PlayerPropsScraper, scrape_player_props
from .line_movements_scraper import LineMovementsScraper, scrape_line_movements
from .generic_scraper import GenericScraper, TableScraper, scrape_generic, scrape_tables


__all__ = [
    # Base classes
    "BaseWebScraper",
    "ScraperConfig",
    "ScrapedData",
    "ScraperStatus",
    
    # Utility functions
    "extract_text",
    "extract_texts",
    "extract_attr",
    "extract_number",
    "extract_odds",
    
    # Scraper classes
    "OddsScraper",
    "ScoresScraper",
    "TeamStatsScraper",
    "PlayerStatsScraper",
    "InjuriesScraper",
    "SchedulesScraper",
    "StandingsScraper",
    "PlayerPropsScraper",
    "LineMovementsScraper",
    "GenericScraper",
    "TableScraper",
    
    # Convenience functions
    "scrape_odds",
    "scrape_scores",
    "scrape_team_stats",
    "scrape_player_stats",
    "scrape_injuries",
    "scrape_schedules",
    "scrape_standings",
    "scrape_player_props",
    "scrape_line_movements",
    "scrape_generic",
    "scrape_tables",
]


# Scraper registry for dynamic access
SCRAPER_REGISTRY = {
    "odds": OddsScraper,
    "scores": ScoresScraper,
    "team_stats": TeamStatsScraper,
    "player_stats": PlayerStatsScraper,
    "injuries": InjuriesScraper,
    "schedules": SchedulesScraper,
    "standings": StandingsScraper,
    "player_props": PlayerPropsScraper,
    "line_movements": LineMovementsScraper,
    "generic": GenericScraper,
    "table": TableScraper,
}


def get_scraper(scraper_type: str, **kwargs):
    """
    Get a scraper instance by type.
    
    Args:
        scraper_type: One of 'odds', 'scores', 'team_stats', 'player_stats',
                      'injuries', 'schedules', 'standings', 'player_props',
                      'line_movements', 'generic', 'table'
        **kwargs: Arguments to pass to the scraper constructor
        
    Returns:
        Scraper instance
        
    Example:
        scraper = get_scraper("odds", urls=["https://example.com/odds"])
    """
    if scraper_type not in SCRAPER_REGISTRY:
        raise ValueError(
            f"Unknown scraper type: {scraper_type}. "
            f"Available types: {list(SCRAPER_REGISTRY.keys())}"
        )
    return SCRAPER_REGISTRY[scraper_type](**kwargs)
