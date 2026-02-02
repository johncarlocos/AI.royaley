"""
ROYALEY - Source Registry Data
Contains definitions for all 27 data sources/collectors.

Format: (key, name, source_type, priority, provides_teams, provides_players, 
         provides_games, provides_odds, provides_stats, sports_covered)
"""

# =============================================================================
# DATA SOURCES (27 collectors)
# =============================================================================

SOURCES = [
    # key, name, source_type, priority, teams, players, games, odds, stats, sports
    ("espn", "ESPN API", "api", 5, True, True, True, False, True, 
     ["NFL", "NBA", "MLB", "NHL", "WNBA", "NCAAF", "NCAAB"]),
    
    ("odds_api", "The Odds API", "api", 3, False, False, True, True, False, 
     ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "WNBA", "ATP", "WTA", "CFL"]),
    
    ("pinnacle", "Pinnacle", "api", 1, False, False, True, True, False, 
     ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "ATP", "WTA"]),
    
    ("tennis_abstract", "Tennis Abstract", "scraper", 10, False, True, True, False, True, 
     ["ATP", "WTA"]),
    
    ("weather_openweather", "OpenWeatherMap", "api", 5, False, False, False, False, False, 
     ["NFL", "MLB", "NCAAF", "CFL"]),
    
    ("sportsdb", "TheSportsDB", "api", 15, True, True, True, False, False, 
     ["NFL", "NBA", "MLB", "NHL", "WNBA", "CFL"]),
    
    ("nflfastr", "nflfastR", "file", 2, True, True, True, False, True, 
     ["NFL"]),
    
    ("cfbfastr", "cfbfastR", "file", 2, True, True, True, False, True, 
     ["NCAAF"]),
    
    ("baseballr", "baseballR", "file", 2, True, True, True, False, True, 
     ["MLB"]),
    
    ("hockeyr", "hockeyR", "file", 2, True, True, True, False, True, 
     ["NHL"]),
    
    ("wehoop", "wehoop", "file", 2, True, True, True, False, True, 
     ["WNBA"]),
    
    ("hoopr", "hoopR", "file", 2, True, True, True, False, True, 
     ["NBA"]),
    
    ("cfl_api", "CFL API", "api", 5, True, True, True, False, True, 
     ["CFL"]),
    
    ("action_network", "Action Network", "scraper", 8, False, False, False, False, False, 
     ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]),
    
    ("nhl_api", "NHL Official API", "api", 3, True, True, True, False, True, 
     ["NHL"]),
    
    ("sportsipy", "Sportsipy", "scraper", 20, True, True, True, False, True, 
     ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]),
    
    ("basketball_ref", "Basketball Reference", "scraper", 8, True, True, True, False, True, 
     ["NBA", "WNBA"]),
    
    ("cfbd", "CollegeFootballData", "api", 4, True, True, True, False, True, 
     ["NCAAF"]),
    
    ("matchstat", "MatchStat", "scraper", 12, False, True, True, False, True, 
     ["ATP", "WTA"]),
    
    ("realgm", "RealGM", "scraper", 15, True, True, False, False, True, 
     ["NBA", "WNBA"]),
    
    ("nextgenstats", "Next Gen Stats", "api", 6, False, True, False, False, True, 
     ["NFL"]),
    
    ("kaggle", "Kaggle Datasets", "file", 25, True, True, True, False, True, 
     ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]),
    
    ("balldontlie", "BallDontLie", "api", 8, True, True, True, False, True, 
     ["NFL", "NBA", "MLB", "NCAAF", "NCAAB", "ATP", "WTA"]),
    
    ("polymarket", "Polymarket", "api", 30, False, False, False, True, False, 
     ["NFL", "NBA", "MLB"]),
    
    ("kalshi", "Kalshi", "api", 30, False, False, False, True, False, 
     ["NFL", "NBA", "MLB"]),
    
    ("weatherstack", "Weatherstack", "api", 10, False, False, False, False, False, 
     ["NFL", "MLB", "NCAAF"]),
]

# =============================================================================
# SHARP SPORTSBOOKS (for CLV tracking)
# =============================================================================

SHARP_SPORTSBOOKS = {
    "pinnacle": (1, True),
    "bookmaker": (5, True),
    "betcris": (8, True),
    "circa": (10, True),
}

# Alternative names/keys for sharp books
SHARP_BOOK_ALIASES = {
    "pinnacle", "bookmaker", "betcris", "circa", "cris",
    "pinnacle_direct", "pinnacle_api",
}

# =============================================================================
# SOURCE KEY EXTRACTION PATTERNS
# =============================================================================

SOURCE_PREFIXES = [
    "bdl_", "espn_", "sportsdb_", "pinnacle_", "nflfastr_", "cfbfastr_",
    "baseballr_", "hockeyr_", "wehoop_", "hoopr_", "cfl_", "nhl_api_",
    "sportsipy_", "bref_", "cfbd_", "matchstat_", "realgm_", "ngs_",
    "kaggle_", "ta_",
]


def extract_source_key(external_id: str) -> str:
    """Extract source key from external_id prefix pattern."""
    if not external_id:
        return "unknown"
    
    eid_lower = external_id.lower()
    for prefix in SOURCE_PREFIXES:
        if eid_lower.startswith(prefix):
            return prefix.rstrip("_")
    
    # If numeric only, likely ESPN
    if external_id.isdigit():
        return "espn"
    
    return "unknown"
