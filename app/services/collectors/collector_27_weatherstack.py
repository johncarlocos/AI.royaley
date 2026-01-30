"""
ROYALEY - Weatherstack Weather Data Collector (Collector #27)
Backup weather provider for outdoor sports games using Weatherstack API.

Features:
- Current weather for upcoming games
- Historical weather (back to 2015) for past games
- Supports outdoor sports: NFL, NCAAF, CFL, MLB, ATP, WTA
- Dome stadium detection (neutral weather for indoor games)
- Coordinate caching for performance
- Raw data archiving (gzipped JSON)
- Seamless fallback from OpenWeatherMap

API Documentation: https://weatherstack.com/documentation
Plan: Standard ($9.99/month) - 50,000 requests/month
- Current weather endpoint
- Historical weather endpoint (back to 2015)

Database Table: weather_data
Columns: game_id, venue_id, temperature_f, feels_like_f, humidity_pct,
         wind_speed_mph, wind_direction, precipitation_pct, conditions, is_dome

Usage:
    # Collect weather for upcoming games
    python -m app.services.collectors.collector_27_weatherstack --upcoming --days 7
    
    # Collect historical weather for past games
    python -m app.services.collectors.collector_27_weatherstack --historical --days 365
    
    # Collect for specific sport
    python -m app.services.collectors.collector_27_weatherstack --sport NFL --upcoming
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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

# City coordinates cache (for faster lookups without API calls)
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
    "Saskatchewan": (52.1579, -106.6702),
    "Hamilton": (43.2557, -79.8711),
    "Arlington": (32.7357, -97.1081),
    "Glendale": (33.5387, -112.1860),
    "Foxborough": (42.0929, -71.2646),
    "East Rutherford": (40.8128, -74.0742),
    "Landover": (38.9076, -76.8645),
    "Orchard Park": (42.7738, -78.7870),
    "Paradise": (36.0909, -115.1833),  # Las Vegas Raiders
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
    uv_index: int = 0
    weather_code: int = 0
    

@dataclass 
class CollectorStats:
    """Collection statistics."""
    games_processed: int = 0
    weather_fetched: int = 0
    weather_saved: int = 0
    dome_games: int = 0
    api_calls: int = 0
    api_errors: int = 0
    db_errors: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def duration_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.games_processed == 0:
            return 0.0
        return (self.weather_fetched / self.games_processed) * 100


class WeatherstackCollector:
    """
    Weather collector using Weatherstack API (Backup provider).
    
    API: https://api.weatherstack.com/
    Standard Plan ($9.99/month): 50,000 requests/month
    
    Features:
    - Current weather endpoint
    - Historical weather (back to 2015)
    - Auto-detect location from coordinates or city name
    
    Saves to:
    - Database: weather_data table
    - Archive: /app/raw-data/weatherstack/
    """
    
    # Weatherstack API Endpoints
    BASE_URL = "https://api.weatherstack.com"
    CURRENT_URL = f"{BASE_URL}/current"
    HISTORICAL_URL = f"{BASE_URL}/historical"
    AUTOCOMPLETE_URL = f"{BASE_URL}/autocomplete"
    
    # Archive path
    RAW_DATA_PATH = "/app/raw-data/weatherstack"
    
    def __init__(self, api_key: str = None):
        """
        Initialize Weatherstack collector.
        
        Args:
            api_key: Weatherstack API key. Falls back to WEATHERSTACK_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("WEATHERSTACK_API_KEY", "")
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = CollectorStats()
        self._coord_cache: Dict[str, Tuple[float, float]] = dict(CITY_COORDS)
        self._city_name_cache: Dict[Tuple[float, float], str] = {}
        
        # Ensure archive directory exists
        Path(self.RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _validate_api_key(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            console.print("[red]❌ Weatherstack API key not configured![/red]")
            console.print("[yellow]Add to .env file:[/yellow]")
            console.print("  WEATHERSTACK_API_KEY=your_weatherstack_api_key")
            console.print("\n[cyan]Get API key from:[/cyan]")
            console.print("  https://weatherstack.com/signup/free")
            return False
        return True
    
    # =========================================================================
    # CURRENT WEATHER COLLECTION (Upcoming Games)
    # =========================================================================
    
    async def collect_for_upcoming_games(
        self,
        sport_code: Optional[str] = None,
        days_ahead: int = 7,
    ) -> CollectorStats:
        """
        Collect current weather for upcoming games.
        
        Args:
            sport_code: Optional specific sport code (NFL, MLB, etc.)
            days_ahead: Number of days to look ahead for games
            
        Returns:
            CollectorStats with collection results
        """
        if not self._validate_api_key():
            return self.stats
        
        self.stats = CollectorStats()
        sports = [sport_code.upper()] if sport_code else OUTDOOR_SPORTS
        
        if HAS_RICH:
            console.print(Panel(
                f"[bold blue]🌤️ Weatherstack Collection - Upcoming Games[/bold blue]\n"
                f"Sports: {', '.join(sports)}\n"
                f"Days ahead: {days_ahead}\n"
                f"API: Weatherstack (Backup Provider)",
                title="Weatherstack Collector"
            ))
        else:
            console.print(f"=== Weatherstack Collection for {', '.join(sports)} ===")
        
        # Get upcoming games from database
        games = await self._get_upcoming_games(sports, days_ahead)
        
        console.print(f"[cyan]Found {len(games)} upcoming outdoor games[/cyan]")
        
        if not games:
            console.print("[yellow]No upcoming games found[/yellow]")
            return self.stats
        
        # Process each game
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching weather...", total=len(games))
                
                for game in games:
                    try:
                        weather = await self._fetch_current_weather(game)
                        
                        if weather:
                            await self._save_weather(weather)
                            self.stats.weather_fetched += 1
                        
                        self.stats.games_processed += 1
                        progress.update(task, advance=1)
                        
                        # Rate limiting (50,000/month ≈ 1.6k/day, be conservative)
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        self.stats.errors.append(str(e)[:100])
                        logger.error(f"Error for game {game.get('id')}: {e}")
        else:
            for i, game in enumerate(games):
                try:
                    console.print(f"Processing game {i+1}/{len(games)}...")
                    weather = await self._fetch_current_weather(game)
                    
                    if weather:
                        await self._save_weather(weather)
                        self.stats.weather_fetched += 1
                    
                    self.stats.games_processed += 1
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.stats.errors.append(str(e)[:100])
                    logger.error(f"Error for game {game.get('id')}: {e}")
        
        self._print_summary()
        return self.stats
    
    async def _fetch_current_weather(
        self,
        game: Dict[str, Any],
    ) -> Optional[WeatherResult]:
        """
        Fetch current weather for a game from Weatherstack API.
        
        Args:
            game: Game info dict with location data
            
        Returns:
            WeatherResult or None if failed
        """
        venue = game.get("venue_name", "")
        city = game.get("city", "")
        
        # Check if dome stadium (return neutral indoor weather)
        is_dome = game.get("is_dome", False) or DOME_STADIUMS.get(venue, False)
        
        if is_dome:
            self.stats.dome_games += 1
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
        
        # Build query - prefer coordinates, fallback to city name
        lat = game.get("latitude")
        lon = game.get("longitude")
        
        if lat and lon:
            query = f"{lat},{lon}"
        elif city:
            # Check coordinate cache
            if city in self._coord_cache:
                lat, lon = self._coord_cache[city]
                query = f"{lat},{lon}"
            else:
                query = city
        else:
            logger.warning(f"No location data for game {game.get('id')}")
            return None
        
        # Build API parameters
        params = {
            "access_key": self.api_key,
            "query": query,
            "units": "f",  # Fahrenheit
        }
        
        try:
            self.stats.api_calls += 1
            
            async with self.session.get(self.CURRENT_URL, params=params) as response:
                data = await response.json()
                
                # Check for API errors
                if "error" in data:
                    error_info = data.get("error", {})
                    error_code = error_info.get("code", "Unknown")
                    error_msg = error_info.get("info", "Unknown error")
                    logger.warning(f"Weatherstack API error {error_code}: {error_msg}")
                    self.stats.api_errors += 1
                    return None
                
                if response.status != 200:
                    logger.warning(f"Weatherstack HTTP error {response.status}")
                    self.stats.api_errors += 1
                    return None
            
            # Archive raw data
            await self._archive_raw_data(game["id"], "current", data)
            
            # Parse response
            current = data.get("current", {})
            location = data.get("location", {})
            
            if not current:
                logger.warning(f"No current weather data in response for game {game['id']}")
                return None
            
            # Cache coordinates from response
            if location.get("lat") and location.get("lon"):
                loc_name = location.get("name", city)
                self._coord_cache[loc_name] = (
                    float(location["lat"]),
                    float(location["lon"])
                )
            
            # Convert wind direction degrees to cardinal if only degree is given
            wind_dir = current.get("wind_dir", "N")
            
            # Calculate precipitation probability from precipitation amount
            precip = current.get("precip", 0) or 0
            precip_pct = min(100.0, float(precip) * 20) if precip > 0 else 0.0
            
            # Get weather description
            descriptions = current.get("weather_descriptions", ["Clear"])
            conditions = descriptions[0] if descriptions else "Clear"
            
            return WeatherResult(
                game_id=game["id"],
                temperature_f=float(current.get("temperature", 70)),
                feels_like_f=float(current.get("feelslike", current.get("temperature", 70))),
                humidity_pct=float(current.get("humidity", 50)),
                wind_speed_mph=self._kmh_to_mph(float(current.get("wind_speed", 0))),
                wind_direction=wind_dir,
                precipitation_pct=precip_pct,
                conditions=conditions,
                is_dome=False,
                visibility_miles=self._km_to_miles(float(current.get("visibility", 16))),
                pressure_inhg=self._mb_to_inhg(float(current.get("pressure", 1013))),
                cloud_cover_pct=int(current.get("cloudcover", 0)),
                uv_index=int(current.get("uv_index", 0)),
                weather_code=int(current.get("weather_code", 0)),
            )
            
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching weather: {e}")
            self.stats.api_errors += 1
            return None
        except Exception as e:
            logger.error(f"Weatherstack error: {e}")
            self.stats.api_errors += 1
            return None
    
    # =========================================================================
    # HISTORICAL WEATHER COLLECTION (Past Games)
    # =========================================================================
    
    async def collect_historical_for_games(
        self,
        sport_code: Optional[str] = None,
        days_back: int = 365,
    ) -> CollectorStats:
        """
        Collect historical weather for past games using Weatherstack Historical API.
        
        Weatherstack Standard plan supports historical data back to 2015.
        
        Args:
            sport_code: Optional specific sport code
            days_back: Number of days to look back (max ~10 years)
            
        Returns:
            CollectorStats with collection results
        """
        if not self._validate_api_key():
            return self.stats
        
        self.stats = CollectorStats()
        sports = [sport_code.upper()] if sport_code else OUTDOOR_SPORTS
        
        if HAS_RICH:
            console.print(Panel(
                f"[bold blue]📊 Weatherstack Historical Weather Collection[/bold blue]\n"
                f"Sports: {', '.join(sports)}\n"
                f"Days back: {days_back}\n"
                f"[yellow]Note: Historical endpoint requires Standard+ plan[/yellow]",
                title="Weatherstack Historical"
            ))
        else:
            console.print(f"=== Weatherstack Historical Collection ({days_back} days) ===")
        
        # Get past games without weather data
        games = await self._get_past_games_without_weather(sports, days_back)
        
        console.print(f"[cyan]Found {len(games)} past games needing weather[/cyan]")
        
        if not games:
            console.print("[yellow]No games found needing historical weather[/yellow]")
            return self.stats
        
        # Process each game
        for i, game in enumerate(games):
            try:
                if HAS_RICH and i % 10 == 0:
                    console.print(f"[dim]Processing game {i+1}/{len(games)}...[/dim]")
                
                weather = await self._fetch_historical_weather(game)
                
                if weather:
                    await self._save_weather(weather)
                    self.stats.weather_fetched += 1
                
                self.stats.games_processed += 1
                
                # Rate limiting (be conservative with historical calls)
                await asyncio.sleep(0.3)
                
            except Exception as e:
                self.stats.errors.append(str(e)[:100])
                logger.error(f"Error for game {game.get('id')}: {e}")
        
        self._print_summary()
        return self.stats
    
    async def _fetch_historical_weather(
        self,
        game: Dict[str, Any],
    ) -> Optional[WeatherResult]:
        """
        Fetch historical weather for a past game.
        
        Args:
            game: Game info dict with location and date
            
        Returns:
            WeatherResult or None if failed
        """
        venue = game.get("venue_name", "")
        city = game.get("city", "")
        game_date = game.get("game_date")
        
        if not game_date:
            logger.warning(f"No game date for game {game.get('id')}")
            return None
        
        # Check if dome stadium
        is_dome = game.get("is_dome", False) or DOME_STADIUMS.get(venue, False)
        
        if is_dome:
            self.stats.dome_games += 1
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
        
        # Build query
        lat = game.get("latitude")
        lon = game.get("longitude")
        
        if lat and lon:
            query = f"{lat},{lon}"
        elif city:
            if city in self._coord_cache:
                lat, lon = self._coord_cache[city]
                query = f"{lat},{lon}"
            else:
                query = city
        else:
            return None
        
        # Format date for API
        date_str = game_date.strftime("%Y-%m-%d")
        
        # Build API parameters
        params = {
            "access_key": self.api_key,
            "query": query,
            "historical_date": date_str,
            "hourly": "1",  # Get hourly data for game time precision
            "units": "f",
        }
        
        try:
            self.stats.api_calls += 1
            
            async with self.session.get(self.HISTORICAL_URL, params=params) as response:
                data = await response.json()
                
                # Check for API errors
                if "error" in data:
                    error_info = data.get("error", {})
                    error_code = error_info.get("code", "Unknown")
                    error_msg = error_info.get("info", "Unknown error")
                    logger.warning(f"Weatherstack Historical API error {error_code}: {error_msg}")
                    self.stats.api_errors += 1
                    return None
            
            # Archive raw data
            await self._archive_raw_data(game["id"], "historical", data)
            
            # Parse historical response
            historical = data.get("historical", {})
            date_data = historical.get(date_str, {})
            
            if not date_data:
                logger.warning(f"No historical data for {date_str}")
                return None
            
            # Get hourly data closest to game time
            hourly = date_data.get("hourly", [])
            game_hour = game_date.hour
            
            if hourly:
                # Find closest hour to game time
                # Weatherstack hourly time format: "0", "300", "600", etc. (minutes since midnight)
                target_minutes = game_hour * 100  # Approximate
                closest_hour = min(hourly, key=lambda h: abs(int(h.get("time", 0)) - target_minutes))
                hour_data = closest_hour
            else:
                # Use daily averages
                hour_data = {
                    "temperature": date_data.get("avgtemp", 70),
                    "humidity": 50,
                    "wind_speed": 0,
                    "wind_dir": "N",
                    "precip": 0,
                    "weather_descriptions": ["Clear"],
                }
            
            # Extract values
            temp = float(hour_data.get("temperature", date_data.get("avgtemp", 70)))
            humidity = float(hour_data.get("humidity", 50))
            wind_speed = float(hour_data.get("wind_speed", 0))
            wind_dir = hour_data.get("wind_dir", "N")
            precip = float(hour_data.get("precip", 0))
            
            # Get conditions
            descriptions = hour_data.get("weather_descriptions", ["Clear"])
            conditions = descriptions[0] if descriptions else "Clear"
            
            # Calculate precipitation probability
            precip_pct = min(100.0, precip * 20) if precip > 0 else 0.0
            
            return WeatherResult(
                game_id=game["id"],
                temperature_f=temp,
                feels_like_f=float(hour_data.get("feelslike", temp)),
                humidity_pct=humidity,
                wind_speed_mph=self._kmh_to_mph(wind_speed),
                wind_direction=wind_dir,
                precipitation_pct=precip_pct,
                conditions=conditions,
                is_dome=False,
                cloud_cover_pct=int(hour_data.get("cloudcover", 0)),
                uv_index=int(date_data.get("uv_index", hour_data.get("uv_index", 0))),
                weather_code=int(hour_data.get("weather_code", 0)),
            )
            
        except Exception as e:
            logger.error(f"Historical weather error: {e}")
            self.stats.api_errors += 1
            return None
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
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
    
    async def _get_past_games_without_weather(
        self,
        sports: List[str],
        days_back: int,
    ) -> List[Dict[str, Any]]:
        """Get past games that don't have weather data."""
        try:
            from app.core.database import db_manager
            from app.models import Game, Sport, Team, Venue
            from app.models.models import WeatherData, GameStatus
            from sqlalchemy import select, and_
            
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
                    # Update existing record
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
                    # Create new record
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
            self.stats.db_errors += 1
            self.stats.errors.append(f"DB: {str(e)[:50]}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _kmh_to_mph(self, kmh: float) -> float:
        """Convert km/h to mph."""
        return kmh * 0.621371
    
    def _km_to_miles(self, km: float) -> float:
        """Convert kilometers to miles."""
        return km * 0.621371
    
    def _mb_to_inhg(self, mb: float) -> float:
        """Convert millibars to inches of mercury."""
        return mb * 0.02953
    
    def _degrees_to_cardinal(self, degrees: int) -> str:
        """Convert wind degrees to cardinal direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(degrees / 22.5) % 16
        return directions[idx]
    
    async def _archive_raw_data(
        self,
        game_id: str,
        data_type: str,
        data: Dict[str, Any]
    ):
        """Archive raw weather response."""
        now = datetime.utcnow()
        dir_path = Path(self.RAW_DATA_PATH) / data_type / now.strftime("%Y/%m/%d")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{game_id}.json.gz"
        
        archive_data = {
            "game_id": game_id,
            "fetched_at": now.isoformat(),
            "source": "weatherstack",
            "endpoint": data_type,
            "data": data,
        }
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                json_str = json.dumps(archive_data, indent=2)
                compressed = gzip.compress(json_str.encode('utf-8'))
                await f.write(compressed)
        except Exception as e:
            logger.warning(f"Failed to archive raw data: {e}")
    
    def _print_summary(self):
        """Print collection summary."""
        if not HAS_RICH:
            console.print(f"\n=== Weatherstack Collection Summary ===")
            console.print(f"Duration: {self.stats.duration_seconds:.1f}s")
            console.print(f"Games processed: {self.stats.games_processed}")
            console.print(f"Weather fetched: {self.stats.weather_fetched}")
            console.print(f"Weather saved: {self.stats.weather_saved}")
            console.print(f"Dome games: {self.stats.dome_games}")
            console.print(f"API calls: {self.stats.api_calls}")
            console.print(f"API errors: {self.stats.api_errors}")
            console.print(f"Success rate: {self.stats.success_rate:.1f}%")
            return
        
        table = Table(title="🌤️ Weatherstack Collection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Duration", f"{self.stats.duration_seconds:.1f}s")
        table.add_row("Games Processed", str(self.stats.games_processed))
        table.add_row("Weather Fetched", str(self.stats.weather_fetched))
        table.add_row("Weather Saved", str(self.stats.weather_saved))
        table.add_row("Dome Games (indoor)", str(self.stats.dome_games))
        table.add_row("API Calls", str(self.stats.api_calls))
        table.add_row("API Errors", str(self.stats.api_errors))
        table.add_row("DB Errors", str(self.stats.db_errors))
        table.add_row("Success Rate", f"{self.stats.success_rate:.1f}%")
        
        console.print(table)
        console.print(f"\n[cyan]Raw data archived to:[/cyan] {self.RAW_DATA_PATH}")
        
        if self.stats.errors:
            console.print(f"\n[yellow]Errors ({len(self.stats.errors)}):[/yellow]")
            for err in self.stats.errors[:5]:
                console.print(f"  - {err}")
            if len(self.stats.errors) > 5:
                console.print(f"  ... and {len(self.stats.errors) - 5} more")


# =============================================================================
# CLI MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Collect weather data using Weatherstack API (backup provider)"
    )
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, MLB, NCAAF, CFL, ATP, WTA)"
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Collect current weather for upcoming games"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Collect historical weather for past games"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days ahead (upcoming) or back (historical)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Weatherstack API key (or set WEATHERSTACK_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    if not args.upcoming and not args.historical:
        parser.print_help()
        console.print("\n[yellow]Use --upcoming or --historical to collect weather[/yellow]")
        console.print("\n[cyan]Examples:[/cyan]")
        console.print("  # Upcoming games (next 7 days)")
        console.print("  python -m app.services.collectors.collector_27_weatherstack --upcoming")
        console.print("\n  # Historical (past year)")
        console.print("  python -m app.services.collectors.collector_27_weatherstack --historical --days 365")
        console.print("\n  # Specific sport")
        console.print("  python -m app.services.collectors.collector_27_weatherstack --sport NFL --upcoming")
        return
    
    async with WeatherstackCollector(api_key=args.api_key) as collector:
        if args.upcoming:
            await collector.collect_for_upcoming_games(
                sport_code=args.sport,
                days_ahead=args.days
            )
        elif args.historical:
            await collector.collect_historical_for_games(
                sport_code=args.sport,
                days_back=args.days
            )


# Create singleton instance for import
weatherstack_collector = WeatherstackCollector()


if __name__ == "__main__":
    asyncio.run(main())
