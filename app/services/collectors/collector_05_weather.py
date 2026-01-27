"""
ROYALEY - Weather Data Collector
Collects weather data for outdoor sports games using OpenWeatherMap API.

Matches existing WeatherData model columns:
- temperature_f, feels_like_f, humidity_pct
- wind_speed_mph, wind_direction
- precipitation_pct, conditions
- is_dome

Usage:
    # Collect weather for upcoming games
    python weather_collector.py --upcoming --days 7
    
    # Collect for specific sport
    python weather_collector.py --sport NFL --upcoming
"""

import asyncio
import argparse
import json
import gzip
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from uuid import UUID
import aiohttp
import aiofiles

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Outdoor sports that need weather data
OUTDOOR_SPORTS = ["NFL", "NCAAF", "CFL", "MLB", "ATP", "WTA"]

# Known dome/indoor stadiums
DOME_STADIUMS = {
    # NFL
    "AT&T Stadium": True,
    "Caesars Superdome": True,
    "Ford Field": True,
    "Lucas Oil Stadium": True,
    "Mercedes-Benz Stadium": True,
    "State Farm Stadium": True,
    "U.S. Bank Stadium": True,
    "Allegiant Stadium": True,
    "SoFi Stadium": True,  # Technically open but covered
    # MLB
    "Tropicana Field": True,
    "T-Mobile Park": True,
    "Rogers Centre": True,
    "Minute Maid Park": True,
    "Chase Field": True,
    "Globe Life Field": True,
    "loanDepot park": True,
    "American Family Field": True,
}

# City coordinates cache (for faster lookups)
CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Francisco": (37.7749, -122.4194),
    "Denver": (39.7392, -104.9903),
    "Seattle": (47.6062, -122.3321),
    "Boston": (42.3601, -71.0589),
    "Atlanta": (33.7490, -84.3880),
    "Miami": (25.7617, -80.1918),
    "Minneapolis": (44.9778, -93.2650),
    "Cleveland": (41.4993, -81.6944),
    "Detroit": (42.3314, -83.0458),
    "Tampa": (27.9506, -82.4572),
    "Baltimore": (39.2904, -76.6122),
    "Pittsburgh": (40.4406, -79.9959),
    "Charlotte": (35.2271, -80.8431),
    "Indianapolis": (39.7684, -86.1581),
    "Nashville": (36.1627, -86.7816),
    "New Orleans": (29.9511, -90.0715),
    "Las Vegas": (36.1699, -115.1398),
    "Kansas City": (39.0997, -94.5786),
    "Cincinnati": (39.1031, -84.5120),
    "Green Bay": (44.5192, -88.0198),
    "Jacksonville": (30.3322, -81.6557),
    "Buffalo": (42.8864, -78.8784),
    "Oakland": (37.8044, -122.2712),
    "Milwaukee": (43.0389, -87.9065),
    "Toronto": (43.6532, -79.3832),
    "Montreal": (45.5017, -73.5673),
    "Vancouver": (49.2827, -123.1207),
    "Calgary": (51.0447, -114.0719),
    "Edmonton": (53.5461, -113.4938),
    "Ottawa": (45.4215, -75.6972),
    "Winnipeg": (49.8951, -97.1384),
}


@dataclass
class WeatherResult:
    """Weather data matching database model."""
    game_id: str
    temperature_f: float
    feels_like_f: float
    humidity_pct: float
    wind_speed_mph: float
    wind_direction: str
    precipitation_pct: float
    conditions: str
    is_dome: bool
    
    # Additional data for raw archive
    visibility_miles: float = 10.0
    pressure_inhg: float = 30.0
    cloud_cover_pct: int = 0
    

@dataclass 
class CollectorStats:
    """Collection statistics."""
    games_processed: int = 0
    weather_fetched: int = 0
    weather_saved: int = 0
    dome_games: int = 0
    api_calls: int = 0
    errors: List[str] = field(default_factory=list)


class WeatherCollector:
    """
    Weather collector using OpenWeatherMap API.
    
    API: https://api.openweathermap.org/data/2.5/weather
    Free tier: 1000 calls/day
    
    Saves to:
    - Database: weather_data table
    - Archive: /app/raw-data/weather/
    """
    
    OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"
    GEOCODE_URL = "http://api.openweathermap.org/geo/1.0/direct"
    RAW_DATA_PATH = "/app/raw-data/weather"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY", "")
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = CollectorStats()
        self._coord_cache: Dict[str, Tuple[float, float]] = dict(CITY_COORDS)
        
        # Ensure directory exists
        Path(self.RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _validate_api_key(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            console.print("[red]❌ Weather API key not configured![/red]")
            console.print("[yellow]Add to .env file:[/yellow]")
            console.print("  WEATHER_API_KEY=your_openweathermap_key")
            console.print("\n[cyan]Get free API key from:[/cyan]")
            console.print("  https://openweathermap.org/api")
            return False
        return True
    
    async def collect_for_upcoming_games(
        self,
        sport_code: Optional[str] = None,
        days_ahead: int = 7,
    ) -> CollectorStats:
        """
        Collect weather for upcoming games.
        
        Args:
            sport_code: Optional specific sport
            days_ahead: Number of days to look ahead
            
        Returns:
            CollectorStats
        """
        if not self._validate_api_key():
            return self.stats
        
        sports = [sport_code.upper()] if sport_code else OUTDOOR_SPORTS
        
        if HAS_RICH:
            console.print(Panel(
                f"[bold blue]Weather Collection - Upcoming Games[/bold blue]\n"
                f"Sports: {', '.join(sports)}\n"
                f"Days ahead: {days_ahead}",
                title="Weather Collector"
            ))
        else:
            console.print(f"=== Weather Collection for {', '.join(sports)} ===")
        
        # Get upcoming games
        games = await self._get_upcoming_games(sports, days_ahead)
        
        console.print(f"[cyan]Found {len(games)} upcoming outdoor games[/cyan]")
        
        if not games:
            console.print("[yellow]No games found[/yellow]")
            return self.stats
        
        for game in games:
            try:
                weather = await self._fetch_weather_for_game(game)
                
                if weather:
                    await self._save_weather(weather)
                    self.stats.weather_fetched += 1
                
                self.stats.games_processed += 1
                
                # Rate limiting (1000/day = ~1/minute, but let's be safe)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.stats.errors.append(str(e)[:100])
                logger.error(f"Error for game {game.get('id')}: {e}")
        
        self._print_summary()
        return self.stats
    
    async def _get_upcoming_games(
        self,
        sports: List[str],
        days_ahead: int,
    ) -> List[Dict[str, Any]]:
        """Get upcoming games from database."""
        try:
            from app.core.database import db_manager
            from app.models import Game, Sport, Team, Venue
            from app.models.models import GameStatus
            from sqlalchemy import select, and_
            
            games = []
            now = datetime.utcnow()
            end_date = now + timedelta(days=days_ahead)
            
            await db_manager.initialize()
            async with db_manager.session() as session:
                query = (
                    select(Game, Sport, Team, Venue)
                    .join(Sport, Game.sport_id == Sport.id)
                    .join(Team, Game.home_team_id == Team.id)
                    .outerjoin(Venue, Game.venue_id == Venue.id)
                    .where(
                        and_(
                            Sport.code.in_(sports),
                            Game.scheduled_at >= now,
                            Game.scheduled_at <= end_date,
                            Game.status == GameStatus.SCHEDULED,
                        )
                    )
                )
                
                result = await session.execute(query)
                
                for game, sport, team, venue in result:
                    games.append({
                        "id": str(game.id),
                        "sport_code": sport.code,
                        "game_date": game.scheduled_at,
                        "venue_name": venue.name if venue else None,
                        "city": venue.city if venue else team.city,
                        "state": venue.state if venue else None,
                        "is_dome": venue.is_dome if venue else False,
                        "latitude": venue.latitude if venue else None,
                        "longitude": venue.longitude if venue else None,
                    })
            
            return games
            
        except ImportError:
            logger.warning("Database modules not available")
            return []
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return []
    
    async def _fetch_weather_for_game(
        self,
        game: Dict[str, Any],
    ) -> Optional[WeatherResult]:
        """Fetch weather for a specific game."""
        venue = game.get("venue_name", "")
        city = game.get("city", "")
        
        # Check if dome stadium
        is_dome = game.get("is_dome", False) or DOME_STADIUMS.get(venue, False)
        
        if is_dome:
            self.stats.dome_games += 1
            # Return neutral weather for dome games
            return WeatherResult(
                game_id=game["id"],
                temperature_f=72.0,
                feels_like_f=72.0,
                humidity_pct=50.0,
                wind_speed_mph=0.0,
                wind_direction="N/A",
                precipitation_pct=0.0,
                conditions="Indoor",
                is_dome=True,
            )
        
        # Get coordinates
        lat = game.get("latitude")
        lon = game.get("longitude")
        
        if not lat or not lon:
            lat, lon = await self._get_coordinates(city)
        
        if lat is None:
            logger.warning(f"Could not geocode: {city}")
            return None
        
        # Fetch weather
        return await self._fetch_openweathermap(game, lat, lon)
    
    async def _get_coordinates(self, city: str) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for a city."""
        if not city:
            return None, None
        
        # Check cache
        if city in self._coord_cache:
            return self._coord_cache[city]
        
        # Use geocoding API
        params = {
            "q": f"{city},US",
            "limit": 1,
            "appid": self.api_key,
        }
        
        try:
            self.stats.api_calls += 1
            async with self.session.get(self.GEOCODE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        lat, lon = data[0].get("lat"), data[0].get("lon")
                        self._coord_cache[city] = (lat, lon)
                        return lat, lon
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
        
        return None, None
    
    async def _fetch_openweathermap(
        self,
        game: Dict[str, Any],
        lat: float,
        lon: float,
    ) -> Optional[WeatherResult]:
        """Fetch weather from OpenWeatherMap."""
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial",  # Fahrenheit
        }
        
        try:
            self.stats.api_calls += 1
            
            async with self.session.get(self.OPENWEATHERMAP_URL, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Weather API error {response.status}: {error_text}")
                    return None
                
                data = await response.json()
            
            # Archive raw data
            await self._archive_raw_data(game["id"], data)
            
            # Parse response
            main = data.get("main", {})
            wind = data.get("wind", {})
            weather = data.get("weather", [{}])[0]
            clouds = data.get("clouds", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})
            
            # Calculate precipitation probability
            precip_pct = 0.0
            if rain.get("1h", 0) > 0 or snow.get("1h", 0) > 0:
                precip_pct = min(100, (rain.get("1h", 0) + snow.get("1h", 0)) * 10)
            elif clouds.get("all", 0) > 80:
                precip_pct = 30.0
            
            # Convert wind direction degrees to cardinal
            wind_deg = wind.get("deg", 0)
            wind_dir = self._degrees_to_cardinal(wind_deg)
            
            return WeatherResult(
                game_id=game["id"],
                temperature_f=main.get("temp", 70.0),
                feels_like_f=main.get("feels_like", 70.0),
                humidity_pct=main.get("humidity", 50.0),
                wind_speed_mph=wind.get("speed", 0.0),
                wind_direction=wind_dir,
                precipitation_pct=precip_pct,
                conditions=weather.get("main", "Clear"),
                is_dome=False,
                visibility_miles=data.get("visibility", 10000) / 1609.34,
                pressure_inhg=main.get("pressure", 1013) * 0.02953,
                cloud_cover_pct=clouds.get("all", 0),
            )
            
        except Exception as e:
            logger.error(f"OpenWeatherMap error: {e}")
            return None
    
    def _degrees_to_cardinal(self, degrees: int) -> str:
        """Convert wind degrees to cardinal direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(degrees / 22.5) % 16
        return directions[idx]
    
    async def _archive_raw_data(self, game_id: str, data: Dict[str, Any]):
        """Archive raw weather response."""
        now = datetime.utcnow()
        dir_path = Path(self.RAW_DATA_PATH) / now.strftime("%Y/%m/%d")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{game_id}.json.gz"
        
        archive_data = {
            "game_id": game_id,
            "fetched_at": now.isoformat(),
            "data": data,
        }
        
        async with aiofiles.open(file_path, 'wb') as f:
            json_str = json.dumps(archive_data, indent=2)
            compressed = gzip.compress(json_str.encode('utf-8'))
            await f.write(compressed)
    
    async def _save_weather(self, weather: WeatherResult):
        """Save weather to database."""
        try:
            from app.core.database import db_manager
            from app.models.models import WeatherData
            from sqlalchemy import select
            
            await db_manager.initialize()
            async with db_manager.session() as session:
                # Check if exists
                result = await session.execute(
                    select(WeatherData).where(
                        WeatherData.game_id == UUID(weather.game_id)
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update
                    existing.temperature_f = weather.temperature_f
                    existing.feels_like_f = weather.feels_like_f
                    existing.humidity_pct = weather.humidity_pct
                    existing.wind_speed_mph = weather.wind_speed_mph
                    existing.wind_direction = weather.wind_direction
                    existing.precipitation_pct = weather.precipitation_pct
                    existing.conditions = weather.conditions
                    existing.is_dome = weather.is_dome
                    existing.recorded_at = datetime.utcnow()
                else:
                    # Create new
                    weather_record = WeatherData(
                        game_id=UUID(weather.game_id),
                        temperature_f=weather.temperature_f,
                        feels_like_f=weather.feels_like_f,
                        humidity_pct=weather.humidity_pct,
                        wind_speed_mph=weather.wind_speed_mph,
                        wind_direction=weather.wind_direction,
                        precipitation_pct=weather.precipitation_pct,
                        conditions=weather.conditions,
                        is_dome=weather.is_dome,
                    )
                    session.add(weather_record)
                
                await session.commit()
                self.stats.weather_saved += 1
                
        except ImportError:
            logger.warning("Database modules not available")
        except Exception as e:
            logger.error(f"Error saving weather: {e}")
            self.stats.errors.append(str(e)[:100])
    
    def _print_summary(self):
        """Print collection summary."""
        if not HAS_RICH:
            console.print(f"\n=== Weather Collection Summary ===")
            console.print(f"Games processed: {self.stats.games_processed}")
            console.print(f"Weather fetched: {self.stats.weather_fetched}")
            console.print(f"Dome games: {self.stats.dome_games}")
            console.print(f"API calls: {self.stats.api_calls}")
            return
        
        table = Table(title="Weather Collection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Games Processed", str(self.stats.games_processed))
        table.add_row("Weather Fetched", str(self.stats.weather_fetched))
        table.add_row("Weather Saved", str(self.stats.weather_saved))
        table.add_row("Dome Games (skipped)", str(self.stats.dome_games))
        table.add_row("API Calls Used", str(self.stats.api_calls))
        table.add_row("Errors", str(len(self.stats.errors)))
        
        console.print(table)
        console.print(f"\n[cyan]Raw data archived to:[/cyan] {self.RAW_DATA_PATH}")

    # =========================================================================
    # HISTORICAL WEATHER COLLECTION (Open-Meteo API - FREE)
    # =========================================================================
    
    OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    async def collect_historical_for_games(
        self,
        sport_code: Optional[str] = None,
        days_back: int = 365,
    ) -> CollectorStats:
        """
        Collect historical weather for past games using Open-Meteo API (FREE).
        
        Open-Meteo provides historical weather data from 1940 to present.
        No API key required!
        
        Args:
            sport_code: Optional specific sport
            days_back: Number of days to look back (max ~10 years)
            
        Returns:
            CollectorStats
        """
        sports = [sport_code.upper()] if sport_code else OUTDOOR_SPORTS
        
        console.print(f"[bold blue]Historical Weather Collection[/bold blue]")
        console.print(f"Sports: {', '.join(sports)}")
        console.print(f"Days back: {days_back}")
        console.print(f"[green]Using Open-Meteo API (FREE, no key required)[/green]")
        
        # Get past games without weather data
        games = await self._get_past_games_without_weather(sports, days_back)
        
        console.print(f"[cyan]Found {len(games)} past games needing weather[/cyan]")
        
        if not games:
            console.print("[yellow]No games found needing historical weather[/yellow]")
            return self.stats
        
        for game in games:
            try:
                weather = await self._fetch_historical_weather(game)
                
                if weather:
                    await self._save_weather(weather)
                    self.stats.weather_fetched += 1
                
                self.stats.games_processed += 1
                
                # Open-Meteo rate limit: 10,000/day, be nice
                await asyncio.sleep(0.2)
                
            except Exception as e:
                self.stats.errors.append(str(e)[:100])
                logger.error(f"Error for game {game.get('id')}: {e}")
        
        self._print_summary()
        return self.stats
    
    async def _get_past_games_without_weather(
        self,
        sports: List[str],
        days_back: int,
    ) -> List[Dict[str, Any]]:
        """Get past games that don't have weather data."""
        try:
            from app.core.database import db_manager
            from app.models import Game, Sport, Team, Venue, WeatherData
            from app.models.models import GameStatus
            from sqlalchemy import select, and_, not_, exists
            from sqlalchemy.orm import aliased
            
            games = []
            now = datetime.utcnow()
            start_date = now - timedelta(days=days_back)
            
            await db_manager.initialize()
            async with db_manager.session() as session:
                # Subquery to check if weather exists for game
                weather_exists = (
                    select(WeatherData.id)
                    .where(WeatherData.game_id == Game.id)
                    .exists()
                )
                
                query = (
                    select(Game, Sport, Team, Venue)
                    .join(Sport, Game.sport_id == Sport.id)
                    .join(Team, Game.home_team_id == Team.id)
                    .outerjoin(Venue, Game.venue_id == Venue.id)
                    .where(
                        and_(
                            Sport.code.in_(sports),
                            Game.scheduled_at >= start_date,
                            Game.scheduled_at <= now,
                            Game.status == GameStatus.FINAL,
                            ~weather_exists,  # No weather data exists
                        )
                    )
                    .order_by(Game.scheduled_at.desc())
                    .limit(1000)  # Process in batches
                )
                
                result = await session.execute(query)
                
                for game, sport, team, venue in result:
                    games.append({
                        "id": str(game.id),
                        "sport_code": sport.code,
                        "game_date": game.scheduled_at,
                        "venue_name": venue.name if venue else None,
                        "city": venue.city if venue else team.city,
                        "state": venue.state if venue else None,
                        "is_dome": venue.is_dome if venue else False,
                        "latitude": venue.latitude if venue else None,
                        "longitude": venue.longitude if venue else None,
                    })
            
            return games
            
        except ImportError:
            logger.warning("Database modules not available")
            return []
        except Exception as e:
            logger.error(f"Error fetching past games: {e}")
            return []
    
    async def _fetch_historical_weather(
        self,
        game: Dict[str, Any],
    ) -> Optional[WeatherResult]:
        """Fetch historical weather from Open-Meteo (FREE)."""
        venue = game.get("venue_name", "")
        city = game.get("city", "")
        game_date = game.get("game_date")
        
        if not game_date:
            return None
        
        # Check if dome stadium
        is_dome = game.get("is_dome", False) or DOME_STADIUMS.get(venue, False)
        
        if is_dome:
            self.stats.dome_games += 1
            # Return neutral weather for dome games
            return WeatherResult(
                game_id=game["id"],
                temperature_f=72.0,
                feels_like_f=72.0,
                humidity_pct=50.0,
                wind_speed_mph=0.0,
                wind_direction="N/A",
                precipitation_pct=0.0,
                conditions="Indoor",
                is_dome=True,
            )
        
        # Get coordinates
        lat = game.get("latitude")
        lon = game.get("longitude")
        
        if not lat or not lon:
            lat, lon = await self._get_coordinates(city)
        
        if lat is None:
            logger.warning(f"Could not geocode: {city}")
            return None
        
        # Format date for API
        date_str = game_date.strftime("%Y-%m-%d")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        }
        
        try:
            self.stats.api_calls += 1
            
            async with self.session.get(self.OPEN_METEO_HISTORICAL_URL, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Open-Meteo API error {response.status}: {error_text}")
                    return None
                
                data = await response.json()
            
            # Parse hourly data - get values around game time (assume 7pm local = ~0:00 UTC next day)
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            
            if not times:
                return None
            
            # Find index closest to game time
            game_hour = game_date.hour
            idx = min(game_hour, len(times) - 1)
            
            # Get weather values
            temp = hourly.get("temperature_2m", [70])[idx] if hourly.get("temperature_2m") else 70
            humidity = hourly.get("relative_humidity_2m", [50])[idx] if hourly.get("relative_humidity_2m") else 50
            precip = hourly.get("precipitation", [0])[idx] if hourly.get("precipitation") else 0
            wind_speed = hourly.get("wind_speed_10m", [0])[idx] if hourly.get("wind_speed_10m") else 0
            wind_dir = hourly.get("wind_direction_10m", [0])[idx] if hourly.get("wind_direction_10m") else 0
            weather_code = hourly.get("weather_code", [0])[idx] if hourly.get("weather_code") else 0
            
            # Convert weather code to conditions
            conditions = self._weather_code_to_conditions(weather_code)
            
            # Calculate precipitation probability from actual precipitation
            precip_pct = min(100, precip * 20) if precip > 0 else 0
            
            return WeatherResult(
                game_id=game["id"],
                temperature_f=temp,
                feels_like_f=temp,  # Open-Meteo doesn't have feels_like in archive
                humidity_pct=humidity,
                wind_speed_mph=wind_speed,
                wind_direction=self._degrees_to_cardinal(int(wind_dir)),
                precipitation_pct=precip_pct,
                conditions=conditions,
                is_dome=False,
            )
            
        except Exception as e:
            logger.error(f"Open-Meteo error: {e}")
            return None
    
    def _weather_code_to_conditions(self, code: int) -> str:
        """Convert WMO weather code to condition string."""
        # WMO Weather interpretation codes
        # https://open-meteo.com/en/docs
        codes = {
            0: "Clear",
            1: "Mostly Clear",
            2: "Partly Cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Fog",
            51: "Light Drizzle",
            53: "Drizzle",
            55: "Heavy Drizzle",
            56: "Freezing Drizzle",
            57: "Freezing Drizzle",
            61: "Light Rain",
            63: "Rain",
            65: "Heavy Rain",
            66: "Freezing Rain",
            67: "Freezing Rain",
            71: "Light Snow",
            73: "Snow",
            75: "Heavy Snow",
            77: "Snow Grains",
            80: "Light Showers",
            81: "Showers",
            82: "Heavy Showers",
            85: "Snow Showers",
            86: "Heavy Snow Showers",
            95: "Thunderstorm",
            96: "Thunderstorm with Hail",
            99: "Thunderstorm with Heavy Hail",
        }
        return codes.get(code, "Unknown")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect weather data for outdoor sports games"
    )
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, MLB, etc.)"
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Collect for upcoming games"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days ahead to collect"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenWeatherMap API key (or set WEATHER_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    if not args.upcoming:
        parser.print_help()
        console.print("\n[yellow]Use --upcoming to collect weather for upcoming games[/yellow]")
        return
    
    async with WeatherCollector(api_key=args.api_key) as collector:
        await collector.collect_for_upcoming_games(
            sport_code=args.sport,
            days_ahead=args.days
        )


# Create singleton instance for import
weather_collector = WeatherCollector()


if __name__ == "__main__":
    asyncio.run(main())