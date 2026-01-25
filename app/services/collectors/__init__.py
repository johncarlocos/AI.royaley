"""
ROYALEY - Data Collectors Package

ALL collector logic lives here. Scripts are just thin wrappers.

IMPLEMENTED COLLECTORS:
    collector_01_espn.py      - ESPN (Injuries, lineups) - FREE
    collector_02_odds_api.py  - TheOddsAPI (40+ books) - $59/mo
    collector_03_pinnacle.py  - Pinnacle (CLV benchmark) - $10/mo
    collector_04_tennis.py    - Tennis stats
    collector_05_weather.py   - OpenWeatherMap - FREE
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
    "tennis_collector",
    "WeatherCollector",
    "weather_collector",
]
