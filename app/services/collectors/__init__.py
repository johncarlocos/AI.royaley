"""
ROYALEY - Data Collectors Package

ALL collector logic lives here. Scripts are just thin wrappers.

IMPLEMENTED COLLECTORS:
    collector_01_espn.py      - ESPN (Injuries, lineups) - FREE
    collector_02_odds_api.py  - TheOddsAPI (40+ books) - $59/mo
    collector_03_pinnacle.py  - Pinnacle (CLV benchmark) - $10/mo
    collector_04_tennis.py    - Tennis stats (class only, no singleton)
    collector_05_weather.py   - OpenWeatherMap - FREE
    collector_06_sportsdb.py  - TheSportsDB (Games, scores, livescores) - $295/mo
    collector_07_nflfastr.py  - nflfastR (NFL PBP, EPA, WPA, CPOE) - FREE
    collector_08_cfbfastr.py  - cfbfastR (NCAAF PBP, EPA, SP+, Recruiting) - FREE
    collector_09_baseballr.py - baseballR (MLB Statcast, FanGraphs, 85+ features) - FREE
    collector_10_hockeyr.py   - hockeyR (NHL xG, Corsi, Fenwick, 75+ features) - FREE
    collector_11_wehoop.py    - wehoop (WNBA PBP, box scores, player stats) - FREE
"""

from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorManager,
    CollectorResult,
    RateLimiter,
    RetryStrategy,
    collector_manager,
)
from app.services.collectors.collector_01_espn import ESPNCollector, espn_collector
from app.services.collectors.collector_02_odds_api import OddsCollector, odds_collector
from app.services.collectors.collector_03_pinnacle import PinnacleCollector, pinnacle_collector
from app.services.collectors.collector_04_tennis import TennisCollector
from app.services.collectors.collector_05_weather import WeatherCollector, weather_collector
from app.services.collectors.collector_06_sportsdb import SportsDBCollector, sportsdb_collector
from app.services.collectors.collector_07_nflfastr import NFLFastRCollector, nflfastr_collector
from app.services.collectors.collector_08_cfbfastr import CFBFastRCollector, cfbfastr_collector
from app.services.collectors.collector_09_baseballr import BaseballRCollector, baseballr_collector
from app.services.collectors.collector_10_hockeyr import HockeyRCollector, hockeyr_collector
from app.services.collectors.collector_11_wehoop import WehoopCollector, wehoop_collector

__all__ = [
    "BaseCollector",
    "CollectorManager",
    "CollectorResult",
    "RateLimiter",
    "RetryStrategy",
    "collector_manager",
    "ESPNCollector",
    "espn_collector",
    "OddsCollector",
    "odds_collector",
    "PinnacleCollector",
    "pinnacle_collector",
    "TennisCollector",
    "WeatherCollector",
    "weather_collector",
    "SportsDBCollector",
    "sportsdb_collector",
    "NFLFastRCollector",
    "nflfastr_collector",
    "CFBFastRCollector",
    "cfbfastr_collector",
    "BaseballRCollector",
    "baseballr_collector",
    "HockeyRCollector",
    "hockeyr_collector",
    "WehoopCollector",
    "wehoop_collector",
]