"""
ROYALEY - TheOddsAPI Collector
Phase 1: Data Collection Services

Collects real-time odds from 40+ sportsbooks via TheOddsAPI.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings, ODDS_API_SPORT_KEYS, ODDS_API_TENNIS_TOURNAMENTS
from app.models import Game, Odds, Sportsbook, OddsMovement, Sport, Team, GameStatus
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


class OddsCollector(BaseCollector):
    """
    Collector for TheOddsAPI.
    
    Features:
    - Real-time odds from 40+ sportsbooks
    - Support for spreads, moneylines, and totals
    - Line movement detection
    - Rate limit tracking
    """
    
    MARKETS = ["spreads", "h2h", "totals"]  # spread, moneyline, total
    MARKET_TYPE_MAP = {
        "spreads": "spread",
        "h2h": "moneyline",
        "totals": "total",
    }
    
    def __init__(self):
        super().__init__(
            name="odds_api",
            base_url=settings.ODDS_API_BASE_URL,
            rate_limit=500,  # Monthly limit tracked separately
            rate_window=3600,
        )
        self.api_key = settings.ODDS_API_KEY
        self._requests_used = 0
        self._requests_remaining = 500
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Accept": "application/json",
        }
    
    async def collect(
        self,
        sport_code: str = None,
        markets: List[str] = None,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect odds for specified sports.
        
        Args:
            sport_code: Optional sport code (collects all if None)
            markets: List of markets to collect (spreads, h2h, totals)
            
        Returns:
            CollectorResult with odds data
            
        Note: Tennis (ATP/WTA) requires tournament-specific API calls.
              Use collect_tennis() method for tennis odds.
        """
        if not self.api_key:
            return CollectorResult(
                success=False,
                error="TheOddsAPI key not configured",
            )
        
        # Use main sport keys (excludes tennis which needs tournament-specific calls)
        sports_to_collect = (
            [sport_code] if sport_code and sport_code in ODDS_API_SPORT_KEYS
            else list(ODDS_API_SPORT_KEYS.keys())
        )
        
        markets = markets or self.MARKETS
        all_odds = []
        errors = []
        successful_sports = []
        
        logger.info(f"Starting collection for {len(sports_to_collect)} sport(s): {sports_to_collect}")
        
        # Collect each sport individually and combine results
        for sport in sports_to_collect:
            try:
                logger.info(f"Collecting {sport} odds data (markets: {markets})")
                odds_data = await self._collect_sport_odds(sport, markets)
                all_odds.extend(odds_data)
                successful_sports.append(sport)
                logger.info(f"Successfully collected {len(odds_data)} odds records for {sport}")
            except Exception as e:
                logger.error(f"Error collecting {sport} odds: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        # Return success if we collected data for at least one sport
        # This allows partial success when collecting all sports
        logger.info(f"Collection complete: {len(successful_sports)}/{len(sports_to_collect)} sports succeeded, {len(all_odds)} total odds records collected")
        
        return CollectorResult(
            success=len(successful_sports) > 0,
            data=all_odds,
            records_count=len(all_odds),
            error="; ".join(errors) if errors else None,
            metadata={
                "sports_collected": sports_to_collect,
                "successful_sports": successful_sports,
                "failed_sports": [sport for sport in sports_to_collect if sport not in successful_sports],
                "markets": markets,
                "requests_remaining": self._requests_remaining,
            },
        )
    
    async def collect_tennis(
        self,
        tour: str = None,  # "ATP", "WTA", or None for both
        **kwargs,
    ) -> CollectorResult:
        """
        Collect tennis odds from tournament-specific endpoints.
        
        Tennis in TheOddsAPI uses tournament-specific sport keys like
        'tennis_atp_french_open', 'tennis_wta_wimbledon', etc.
        This method tries all known tournaments and returns data from
        those currently active.
        
        Args:
            tour: Optional - "ATP", "WTA", or None for both tours
            
        Returns:
            CollectorResult with tennis odds data
        """
        if not self.api_key:
            return CollectorResult(
                success=False,
                error="TheOddsAPI key not configured",
            )
        
        # Determine which tours to collect
        if tour:
            tours = [tour.upper()]
        else:
            tours = list(ODDS_API_TENNIS_TOURNAMENTS.keys())
        
        all_odds = []
        active_tournaments = []
        errors = []
        
        logger.info(f"Starting tennis collection for tours: {tours}")
        
        for tour_name in tours:
            tournaments = ODDS_API_TENNIS_TOURNAMENTS.get(tour_name, [])
            
            for tournament_key in tournaments:
                try:
                    params = {
                        "apiKey": self.api_key,
                        "regions": "us",
                        "markets": "h2h",  # Tennis only supports h2h (moneyline)
                        "oddsFormat": "american",
                    }
                    
                    data = await self.get(f"/sports/{tournament_key}/odds", params=params)
                    
                    if data and len(data) > 0:
                        logger.info(f"[Tennis] Found {len(data)} events for {tournament_key}")
                        parsed = self._parse_odds_response(data, tour_name)
                        all_odds.extend(parsed)
                        active_tournaments.append(tournament_key)
                        
                except Exception as e:
                    # 404 is expected for tournaments not in season - don't log as error
                    if "404" not in str(e):
                        logger.debug(f"Tennis tournament {tournament_key}: {e}")
        
        logger.info(f"Tennis collection complete: {len(active_tournaments)} active tournaments, {len(all_odds)} odds records")
        
        return CollectorResult(
            success=True,  # Success even with 0 results (no tournaments active)
            data=all_odds,
            records_count=len(all_odds),
            error="; ".join(errors) if errors else None,
            metadata={
                "tours": tours,
                "active_tournaments": active_tournaments,
                "requests_remaining": self._requests_remaining,
            },
        )
    
    async def _collect_sport_odds(
        self,
        sport_code: str,
        markets: List[str],
    ) -> List[Dict[str, Any]]:
        """Collect odds for a single sport."""
        api_sport_key = ODDS_API_SPORT_KEYS.get(sport_code)
        if not api_sport_key:
            logger.warning(f"Unknown sport code: {sport_code}")
            return []
        
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        
        try:
            logger.info(f"[OddsCollector] Fetching odds for {sport_code} (API key: {api_sport_key})")
            logger.info(f"[OddsCollector] Request params: {params}")
            print(f"[OddsCollector] 🎲 Fetching odds for {sport_code} from TheOddsAPI...")
            print(f"[OddsCollector] URL: /sports/{api_sport_key}/odds")
            print(f"[OddsCollector] Params: {params}")
            
            data = await self.get(f"/sports/{api_sport_key}/odds", params=params)
            
            logger.info(f"[OddsCollector] ✅ Received {len(data)} events from TheOddsAPI for {sport_code}")
            print(f"[OddsCollector] ✅ Successfully fetched {len(data)} events from TheOddsAPI")
            if data:
                print(f"[OddsCollector] First event sample: {data[0] if len(data) > 0 else 'No data'}")
            
            parsed = self._parse_odds_response(data, sport_code)
            logger.info(f"[OddsCollector] ✅ Parsed {len(parsed)} odds records for {sport_code}")
            print(f"[OddsCollector] ✅ Parsed {len(parsed)} odds records")
            
            return parsed
        except Exception as e:
            logger.error(f"Failed to fetch {sport_code} odds: {e}")
            print(f"[OddsCollector] ❌ Error fetching odds: {e}")
            raise
    
    def _parse_odds_response(
        self,
        data: List[Dict[str, Any]],
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """Parse TheOddsAPI response into normalized odds records."""
        parsed_odds = []
        
        for event in data:
            event_id = event.get("id")
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            commence_time = event.get("commence_time")
            
            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker.get("key")
                book_name = bookmaker.get("title")
                last_update = bookmaker.get("last_update")
                
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    market_type = self.MARKET_TYPE_MAP.get(market_key, market_key)
                    
                    for outcome in market.get("outcomes", []):
                        outcome_name = outcome.get("name")
                        if outcome_name == home_team:
                            selection = "home"
                        elif outcome_name == away_team:
                            selection = "away"
                        elif outcome_name.lower() == "over":
                            selection = "over"
                        elif outcome_name.lower() == "under":
                            selection = "under"
                        else:
                            selection = outcome_name.lower()
                        
                        odds_record = {
                            "sport_code": sport_code,
                            "external_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "sportsbook_key": book_key,
                            "sportsbook_name": book_name,
                            "market_type": market_type,
                            "selection": selection,
                            "price": outcome.get("price"),
                            "line": outcome.get("point"),
                            "last_update": last_update,
                        }
                        parsed_odds.append(odds_record)
        
        return parsed_odds
    
    async def validate(self, data: Any) -> bool:
        """Validate odds data."""
        if not isinstance(data, list):
            return False
        
        for record in data:
            if not all([
                record.get("external_id"),
                record.get("sportsbook_key"),
                record.get("market_type"),
                record.get("selection"),
                record.get("price") is not None,
            ]):
                return False
            
            price = record.get("price")
            if price < -10000 or price > 10000:
                logger.warning(f"Suspicious odds value: {price}")
                return False
            
            if record.get("market_type") not in ["spread", "moneyline", "total"]:
                return False
            
            if record.get("selection") not in ["home", "away", "over", "under"]:
                return False
        
        return True
    
    async def save_to_database(
        self,
        odds_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """
        Save collected odds to database.
        
        The new Odds schema stores one row per (game, sportsbook, bet_type) with
        all lines and odds combined. This method groups the parsed records accordingly.
        
        Args:
            odds_data: List of parsed odds records
            session: Database session
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        sportsbook_cache: Dict[str, UUID] = {}
        
        # Group records by (external_id, sportsbook_key, market_type)
        grouped: Dict[tuple, Dict[str, Any]] = {}
        
        for record in odds_data:
            key = (record["external_id"], record["sportsbook_key"], record["market_type"])
            
            if key not in grouped:
                grouped[key] = {
                    "sport_code": record.get("sport_code"),
                    "external_id": record["external_id"],
                    "home_team": record.get("home_team"),
                    "away_team": record.get("away_team"),
                    "commence_time": record.get("commence_time"),
                    "sportsbook_key": record["sportsbook_key"],
                    "sportsbook_name": record.get("sportsbook_name"),
                    "bet_type": record["market_type"],
                    "home_line": None,
                    "away_line": None,
                    "home_odds": None,
                    "away_odds": None,
                    "total": None,
                    "over_odds": None,
                    "under_odds": None,
                }
            
            selection = record.get("selection")
            price = record.get("price")
            line = record.get("line")
            
            if selection == "home":
                grouped[key]["home_odds"] = price
                if line is not None:
                    grouped[key]["home_line"] = line
            elif selection == "away":
                grouped[key]["away_odds"] = price
                if line is not None:
                    grouped[key]["away_line"] = line
            elif selection == "over":
                grouped[key]["over_odds"] = price
                if line is not None:
                    grouped[key]["total"] = line
            elif selection == "under":
                grouped[key]["under_odds"] = price
                if line is not None and grouped[key]["total"] is None:
                    grouped[key]["total"] = line
        
        # Now save each grouped record
        for record in grouped.values():
            savepoint = await session.begin_nested()
            try:
                book_key = record["sportsbook_key"]
                if book_key not in sportsbook_cache:
                    sportsbook = await self._get_or_create_sportsbook(
                        session,
                        book_key,
                        record.get("sportsbook_name", book_key),
                    )
                    sportsbook_cache[book_key] = sportsbook.id
                
                sportsbook_id = sportsbook_cache[book_key]
                
                # Find or create game
                game_result = await session.execute(
                    select(Game).where(Game.external_id == record["external_id"])
                )
                game = game_result.scalar_one_or_none()
                
                if not game:
                    game = await self._create_game_from_odds_record(
                        session,
                        record,
                        record.get("sport_code")
                    )
                    if not game:
                        logger.warning(f"Could not create game for {record.get('external_id')}")
                        await savepoint.rollback()
                        continue
                
                # Create new odds record
                new_odds = Odds(
                    game_id=game.id,
                    sportsbook_id=sportsbook_id,
                    sportsbook_key=book_key,
                    bet_type=record["bet_type"],
                    home_line=record.get("home_line"),
                    away_line=record.get("away_line"),
                    home_odds=record.get("home_odds"),
                    away_odds=record.get("away_odds"),
                    total=record.get("total"),
                    over_odds=record.get("over_odds"),
                    under_odds=record.get("under_odds"),
                    is_opening=False,
                )
                session.add(new_odds)
                await savepoint.commit()
                saved_count += 1
                
            except Exception as e:
                await savepoint.rollback()
                logger.error(f"Error saving odds record: {e}", exc_info=True)
                continue
        
        await session.commit()
        return saved_count
    
    async def _create_game_from_odds_record(
        self,
        session: AsyncSession,
        record: Dict[str, Any],
        sport_code: str,
    ) -> Optional[Game]:
        """Create a game record from odds data if it doesn't exist."""
        try:
            # Get or create sport
            sport = await self._get_or_create_sport(session, sport_code)
            
            if not sport:
                logger.warning(f"Could not create sport {sport_code}")
                return None
            
            # Get or create teams
            home_team_name = record.get("home_team", "")
            away_team_name = record.get("away_team", "")
            
            if not home_team_name or not away_team_name:
                return None
            
            home_team = await self._get_or_create_team(session, sport.id, home_team_name)
            away_team = await self._get_or_create_team(session, sport.id, away_team_name)
            
            # Parse game date and ensure it's timezone-naive UTC
            commence_time = record.get("commence_time")
            if isinstance(commence_time, str):
                # Handle ISO format strings - use fromisoformat for better UTC handling
                try:
                    # Replace 'Z' with '+00:00' for UTC (fromisoformat doesn't handle 'Z' directly in Python < 3.11)
                    if commence_time.endswith('Z'):
                        dt_str = commence_time[:-1] + '+00:00'
                        parsed_dt = datetime.fromisoformat(dt_str)
                    else:
                        parsed_dt = datetime.fromisoformat(commence_time)
                except (ValueError, AttributeError):
                    # Fallback to dateutil parser for non-ISO formats
                    from dateutil import parser
                    parsed_dt = parser.parse(commence_time)
                
                # Convert to UTC if timezone-aware, then create a new naive datetime
                if parsed_dt.tzinfo is not None:
                    # Convert to UTC and extract the components to create a new naive datetime
                    utc_dt = parsed_dt.astimezone(timezone.utc)
                    game_date = datetime(
                        utc_dt.year, utc_dt.month, utc_dt.day,
                        utc_dt.hour, utc_dt.minute, utc_dt.second,
                        utc_dt.microsecond
                    )
                else:
                    # Already naive, create new datetime object to ensure it's truly naive
                    game_date = datetime(
                        parsed_dt.year, parsed_dt.month, parsed_dt.day,
                        parsed_dt.hour, parsed_dt.minute, parsed_dt.second,
                        parsed_dt.microsecond
                    )
            elif isinstance(commence_time, datetime):
                # If it's already a datetime object, convert to naive UTC
                if commence_time.tzinfo is not None:
                    # Convert to UTC and extract components to create new naive datetime
                    utc_dt = commence_time.astimezone(timezone.utc)
                    game_date = datetime(
                        utc_dt.year, utc_dt.month, utc_dt.day,
                        utc_dt.hour, utc_dt.minute, utc_dt.second,
                        utc_dt.microsecond
                    )
                else:
                    # Already naive, create new datetime to ensure it's truly naive
                    game_date = datetime(
                        commence_time.year, commence_time.month, commence_time.day,
                        commence_time.hour, commence_time.minute, commence_time.second,
                        commence_time.microsecond
                    )
            else:
                # Default to naive UTC datetime
                now_utc = datetime.now(timezone.utc)
                game_date = datetime(
                    now_utc.year, now_utc.month, now_utc.day,
                    now_utc.hour, now_utc.minute, now_utc.second,
                    now_utc.microsecond
                )
            
            # Create game
            game = Game(
                sport_id=sport.id,
                external_id=record["external_id"],
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                scheduled_at=game_date,
                status=GameStatus.SCHEDULED,
            )
            session.add(game)
            await session.flush()  # Flush to get the ID
            
            logger.info(f"Created game {game.id} for {away_team_name} @ {home_team_name}")
            return game
            
        except Exception as e:
            logger.error(f"Error creating game from odds record: {e}")
            return None
    
    async def _get_or_create_sport(
        self,
        session: AsyncSession,
        sport_code: str,
    ) -> Optional[Sport]:
        """Get or create a sport record."""
        result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = result.scalar_one_or_none()
        
        if sport:
            return sport
        
        # Create new sport
        sport_names = {
            "NBA": "National Basketball Association",
            "NFL": "National Football League",
            "NCAAF": "NCAA Football",
            "NCAAB": "NCAA Basketball",
            "NHL": "National Hockey League",
            "MLB": "Major League Baseball",
            "WNBA": "Women's National Basketball Association",
            "CFL": "Canadian Football League",
            "ATP": "ATP Tennis",
            "WTA": "WTA Tennis",
        }
        
        sport = Sport(
            code=sport_code,
            name=sport_names.get(sport_code, sport_code),
            is_active=True,
        )
        session.add(sport)
        await session.flush()
        logger.info(f"Created sport {sport_code}")
        return sport
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_name: str,
    ) -> Team:
        """Get or create a team record."""
        import hashlib
        
        # Try to find by name
        result = await session.execute(
            select(Team).where(
                Team.sport_id == sport_id,
                Team.name == team_name
            )
        )
        team = result.scalar_one_or_none()
        
        if team:
            return team
        
        # Create new team with short external_id (max 50 chars for VARCHAR(50) column)
        # Format: first 8 chars of sport_id + _ + hash of team_name (12 chars)
        sport_prefix = str(sport_id)[:8]
        team_hash = hashlib.md5(team_name.lower().encode()).hexdigest()[:12]
        external_id = f"{sport_prefix}_{team_hash}"  # Total: 8 + 1 + 12 = 21 chars
        
        abbreviation = team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()
        team = Team(
            sport_id=sport_id,
            external_id=external_id,
            name=team_name,
            abbreviation=abbreviation,
            is_active=True,
        )
        session.add(team)
        await session.flush()
        return team
    
    async def _get_or_create_sportsbook(
        self,
        session: AsyncSession,
        key: str,
        name: str,
    ) -> Sportsbook:
        """Get or create sportsbook record."""
        result = await session.execute(
            select(Sportsbook).where(Sportsbook.key == key)
        )
        sportsbook = result.scalar_one_or_none()
        
        if not sportsbook:
            sharp_books = ["pinnacle", "betcris", "bookmaker"]
            is_sharp = key.lower() in sharp_books
            
            sportsbook = Sportsbook(
                name=name,
                key=key,
                is_sharp=is_sharp,
            )
            session.add(sportsbook)
            await session.flush()
        
        return sportsbook
    
    def _calculate_movement_size(
        self,
        old_line: Optional[float],
        new_line: Optional[float],
    ) -> Optional[float]:
        """Calculate line movement size."""
        if old_line is None or new_line is None:
            return None
        return abs(new_line - old_line)
    
    async def get_best_odds(
        self,
        game_id: UUID,
        market_type: str,
        selection: str,
        session: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """Get best available odds for a selection."""
        result = await session.execute(
            select(Odds, Sportsbook)
            .join(Sportsbook)
            .where(
                Odds.game_id == game_id,
                Odds.market_type == market_type,
                Odds.selection == selection,
                Odds.is_current == True,
            )
            .order_by(Odds.price.desc())
        )
        
        best = result.first()
        if not best:
            return None
        
        odds, sportsbook = best
        return {
            "sportsbook": sportsbook.name,
            "price": odds.price,
            "line": odds.line,
        }
    
    async def get_sports_list(self) -> List[Dict[str, Any]]:
        """Get list of available sports from TheOddsAPI."""
        params = {"apiKey": self.api_key}
        return await self.get("/sports", params=params)
    
    async def get_events(self, sport_code: str) -> List[Dict[str, Any]]:
        """Get upcoming events for a sport."""
        api_sport_key = ODDS_API_SPORT_KEYS.get(sport_code)
        if not api_sport_key:
            return []
        
        params = {"apiKey": self.api_key}
        return await self.get(f"/sports/{api_sport_key}/events", params=params)
    
    async def collect_historical(
        self,
        sport_code: str = None,
        days_back: int = 30,
        markets: List[str] = None,
    ) -> CollectorResult:
        """
        Collect historical odds data from TheOddsAPI.
        
        This provides HISTORICAL ODDS for ML training - critical for CLV analysis.
        Requires paid OddsAPI subscription ($119+/month).
        
        Args:
            sport_code: Specific sport (NFL, NBA, etc.) or None for all
            days_back: Number of days to fetch (default 30)
            markets: Markets to fetch (default: spreads, h2h, totals)
            
        Returns:
            CollectorResult with historical odds records
        """
        # Support all 10 sports
        sports = [sport_code.upper()] if sport_code else ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL", "ATP", "WTA"]
        markets = markets or ["spreads", "h2h", "totals"]
        
        # Tennis only supports h2h (moneyline)
        TENNIS_SPORTS = ["ATP", "WTA"]
        
        all_odds = []
        errors = []
        
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                continue
            
            # Use only h2h for tennis
            sport_markets = ["h2h"] if sport in TENNIS_SPORTS else markets
                
            try:
                odds = await self._fetch_historical_odds(
                    api_sport_key, sport, days_back, sport_markets
                )
                all_odds.extend(odds)
                logger.info(f"[OddsAPI Historical] {sport}: {len(odds)} odds records")
            except Exception as e:
                errors.append(f"{sport}: {str(e)}")
                logger.error(f"[OddsAPI Historical] Error collecting {sport}: {e}")
        
        return CollectorResult(
            success=len(all_odds) > 0,
            data=all_odds,
            records_count=len(all_odds),
            error="; ".join(errors) if errors else None,
            metadata={"type": "historical_odds", "sports": sports, "days_back": days_back}
        )

    async def collect_historical_comprehensive(
        self,
        sport_code: str = None,
        years_back: int = 5,
        save_to_db: bool = True,
    ) -> CollectorResult:
        """
        Collect COMPREHENSIVE historical odds data from TheOddsAPI.
        
        Optimized for the $119/mo Mega plan (10,000 requests/month):
        - Samples every 3rd day to stay within limits
        - Tracks progress for resumability
        - Saves to database incrementally
        
        For 5 years of data across 8 sports:
        - 5 years = 1,825 days
        - Sample every 3 days = ~608 days
        - 8 sports × 608 = ~4,864 requests (within 10k limit)
        
        Args:
            sport_code: Specific sport or None for all
            years_back: Years of historical data (default 5, max ~7 for OddsAPI)
            save_to_db: Save to database incrementally
            
        Returns:
            CollectorResult with statistics
        """
        from datetime import datetime, timedelta
        from rich.console import Console
        from rich.progress import Progress, TaskID
        
        console = Console()
        
        # Sports to collect (exclude tennis - different endpoints)
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        sports = [sport_code.upper()] if sport_code else main_sports
        sports = [s for s in sports if s in main_sports]
        
        markets = ["spreads", "h2h", "totals"]
        
        # Calculate date range
        days_back = years_back * 365
        sample_interval = 3  # Every 3rd day to save API calls
        
        total_records = 0
        total_movements = 0
        errors = []
        
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold]📊 COMPREHENSIVE HISTORICAL ODDS COLLECTION[/bold]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")
        console.print(f"Sports: {', '.join(sports)}")
        console.print(f"Years back: {years_back} ({days_back} days)")
        console.print(f"Sample interval: Every {sample_interval} days")
        console.print(f"Estimated API calls: {len(sports) * (days_back // sample_interval)}")
        console.print()
        
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                continue
            
            sport_records = 0
            console.print(f"[cyan]📈 {sport}...[/cyan]")
            
            # Fetch in batches
            for day_offset in range(0, days_back, sample_interval):
                target_date = datetime.utcnow() - timedelta(days=day_offset + 1)
                date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")
                
                try:
                    params = {
                        "apiKey": self.api_key,
                        "regions": "us",
                        "markets": ",".join(markets),
                        "oddsFormat": "american",
                        "date": date_str,
                    }
                    
                    # Historical endpoint
                    endpoint = f"/historical/sports/{api_sport_key}/odds"
                    
                    data = await self.get(endpoint, params=params)
                    
                    if data and isinstance(data, dict):
                        events = data.get("data", [])
                        timestamp = data.get("timestamp")
                        
                        batch_records = []
                        for event in events:
                            parsed = self._parse_historical_event(event, sport, timestamp)
                            batch_records.extend(parsed)
                        
                        sport_records += len(batch_records)
                        
                        # Save incrementally if enabled
                        if save_to_db and batch_records:
                            from app.core.database import db_manager
                            await db_manager.initialize()
                            async with db_manager.session() as session:
                                saved, _ = await self.save_historical_to_database(batch_records, session)
                                total_records += saved
                    
                    # Progress every 30 days
                    if day_offset % 90 == 0 and day_offset > 0:
                        console.print(f"    {day_offset}/{days_back} days, {sport_records} records...")
                    
                    # Rate limiting - respect API limits
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    if "401" in str(e) or "403" in str(e):
                        errors.append(f"{sport}: API key issue - {str(e)[:30]}")
                        console.print(f"  [red]❌ API key issue for {sport}[/red]")
                        break
                    # Continue on other errors
                    continue
            
            console.print(f"  [green]✅ {sport}: {sport_records} odds records[/green]")
        
        console.print(f"\n[bold green]Total: {total_records} odds records saved[/bold green]")
        
        return CollectorResult(
            success=total_records > 0,
            data={"total_records": total_records},
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={"type": "historical_odds_comprehensive", "years_back": years_back}
        )

    async def collect_odds_movements(
        self,
        sport_code: str = None,
        hours_back: int = 24,
    ) -> CollectorResult:
        """
        Track line movements by comparing current odds to previous snapshots.
        
        Fills: odds_movements table
        
        Line movements are critical for:
        - Sharp money detection
        - Steam move identification
        - Reverse line movement analysis
        
        Args:
            sport_code: Specific sport or None for all
            hours_back: How far back to look for movements (default 24)
            
        Returns:
            CollectorResult with movement data
        """
        from datetime import datetime, timedelta
        
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        sports = [sport_code.upper()] if sport_code else main_sports
        
        all_movements = []
        errors = []
        
        logger.info(f"[OddsAPI] Collecting odds movements for {len(sports)} sports...")
        
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                continue
            
            try:
                # Get current odds
                params = {
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "spreads,h2h,totals",
                    "oddsFormat": "american",
                }
                
                current_data = await self.get(f"/sports/{api_sport_key}/odds", params=params)
                
                if not current_data:
                    continue
                
                # Get historical snapshot from X hours ago
                past_time = datetime.utcnow() - timedelta(hours=hours_back)
                past_date_str = past_time.strftime("%Y-%m-%dT%H:00:00Z")
                
                params["date"] = past_date_str
                endpoint = f"/historical/sports/{api_sport_key}/odds"
                
                try:
                    past_data = await self.get(endpoint, params=params)
                    past_events = past_data.get("data", []) if past_data else []
                except:
                    past_events = []
                
                # Compare and detect movements
                movements = self._detect_line_movements(
                    current_data, past_events, sport, hours_back
                )
                all_movements.extend(movements)
                
                logger.info(f"[OddsAPI Movements] {sport}: {len(movements)} movements detected")
                
                await asyncio.sleep(0.3)
                
            except Exception as e:
                errors.append(f"{sport}: {str(e)[:50]}")
                logger.error(f"[OddsAPI Movements] Error for {sport}: {e}")
        
        return CollectorResult(
            success=len(all_movements) > 0,
            data=all_movements,
            records_count=len(all_movements),
            error="; ".join(errors) if errors else None,
            metadata={"type": "odds_movements", "hours_back": hours_back}
        )

    def _detect_line_movements(
        self,
        current_events: List[Dict],
        past_events: List[Dict],
        sport_code: str,
        hours_back: int,
    ) -> List[Dict[str, Any]]:
        """Detect line movements between two snapshots."""
        movements = []
        
        # Index past events by event_id
        past_by_id = {}
        for event in past_events:
            event_id = event.get("id")
            if event_id:
                past_by_id[event_id] = event
        
        for current_event in current_events:
            event_id = current_event.get("id")
            if not event_id or event_id not in past_by_id:
                continue
            
            past_event = past_by_id[event_id]
            home_team = current_event.get("home_team")
            away_team = current_event.get("away_team")
            commence_time = current_event.get("commence_time")
            
            # Index past bookmaker odds
            past_odds = {}
            for book in past_event.get("bookmakers", []):
                book_key = book.get("key")
                for market in book.get("markets", []):
                    market_key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        key = f"{book_key}:{market_key}:{outcome.get('name')}"
                        past_odds[key] = {
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        }
            
            # Compare current to past
            for book in current_event.get("bookmakers", []):
                book_key = book.get("key")
                book_title = book.get("title")
                
                for market in book.get("markets", []):
                    market_key = market.get("key")
                    
                    for outcome in market.get("outcomes", []):
                        selection = outcome.get("name")
                        current_price = outcome.get("price")
                        current_point = outcome.get("point")
                        
                        key = f"{book_key}:{market_key}:{selection}"
                        
                        if key in past_odds:
                            past_price = past_odds[key].get("price")
                            past_point = past_odds[key].get("point")
                            
                            # Detect movement
                            price_change = None
                            line_change = None
                            
                            if past_price and current_price:
                                price_change = current_price - past_price
                            
                            if past_point is not None and current_point is not None:
                                line_change = current_point - past_point
                            
                            # Only record if there's actual movement
                            if price_change or line_change:
                                movements.append({
                                    "sport_code": sport_code,
                                    "event_id": event_id,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "commence_time": commence_time,
                                    "sportsbook": book_key,
                                    "sportsbook_title": book_title,
                                    "market": market_key,
                                    "selection": selection,
                                    "previous_odds": past_price,
                                    "current_odds": current_price,
                                    "odds_change": price_change,
                                    "previous_line": past_point,
                                    "current_line": current_point,
                                    "line_change": line_change,
                                    "hours_elapsed": hours_back,
                                    "detected_at": datetime.utcnow().isoformat(),
                                })
        
        return movements

    async def save_movements_to_database(
        self,
        movements_data: List[Dict[str, Any]],
        session,
    ) -> int:
        """Save odds movements to database."""
        from app.models.models import OddsMovement, Game, Sportsbook, Sport
        
        saved = 0
        
        for movement in movements_data:
            try:
                sport_code = movement.get("sport_code")
                home_team = movement.get("home_team")
                away_team = movement.get("away_team")
                commence_time = movement.get("commence_time")
                
                # Find game
                game_id = await self._find_game_id(
                    session, sport_code, home_team, away_team, commence_time
                )
                
                if not game_id:
                    continue
                
                # Get or create sportsbook
                sportsbook_key = movement.get("sportsbook")
                sportsbook = await self._get_or_create_sportsbook(
                    session, sportsbook_key, movement.get("sportsbook_title", sportsbook_key)
                )
                
                # Map market to bet_type
                market = movement.get("market")
                bet_type_map = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
                bet_type = bet_type_map.get(market, market)
                
                # Create movement record
                odds_movement = OddsMovement(
                    game_id=game_id,
                    sportsbook_id=sportsbook.id,
                    bet_type=bet_type,
                    previous_line=movement.get("previous_line"),
                    current_line=movement.get("current_line"),
                    line_change=movement.get("line_change"),
                    previous_odds=movement.get("previous_odds"),
                    current_odds=movement.get("current_odds"),
                    odds_change=movement.get("odds_change"),
                    recorded_at=datetime.utcnow(),
                )
                session.add(odds_movement)
                saved += 1
                
                if saved % 50 == 0:
                    await session.commit()
                    
            except Exception as e:
                logger.debug(f"Error saving movement: {e}")
                continue
        
        await session.commit()
        logger.info(f"[OddsAPI] Saved {saved} odds movements")
        return saved
    
    async def _fetch_historical_odds(
        self,
        api_sport_key: str,
        sport_code: str,
        days_back: int,
        markets: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch historical odds for a specific sport."""
        from datetime import datetime, timedelta
        
        all_odds = []
        
        for day_offset in range(days_back):
            target_date = datetime.utcnow() - timedelta(days=day_offset + 1)
            date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")
            
            try:
                params = {
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": ",".join(markets),
                    "oddsFormat": "american",
                    "date": date_str,
                }
                
                # Historical endpoint
                endpoint = f"/historical/sports/{api_sport_key}/odds"
                
                logger.info(f"[OddsAPI Historical] Fetching {sport_code} odds for {target_date.strftime('%Y-%m-%d')}")
                
                data = await self.get(endpoint, params=params)
                
                if data and isinstance(data, dict):
                    events = data.get("data", [])
                    timestamp = data.get("timestamp")
                    
                    for event in events:
                        parsed = self._parse_historical_event(event, sport_code, timestamp)
                        all_odds.extend(parsed)
                
                # Rate limiting - be gentle with historical API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"[OddsAPI Historical] Failed for {target_date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        return all_odds
    
    def _parse_historical_event(
        self, 
        event: Dict[str, Any], 
        sport_code: str,
        snapshot_time: str = None
    ) -> List[Dict[str, Any]]:
        """Parse a historical event into odds records."""
        odds_records = []
        
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key")
            book_title = bookmaker.get("title")
            last_update = bookmaker.get("last_update")
            
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                
                for outcome in market.get("outcomes", []):
                    record = {
                        "sport_code": sport_code,
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "snapshot_time": snapshot_time,
                        "sportsbook": book_key,
                        "sportsbook_title": book_title,
                        "market": market_key,
                        "selection": outcome.get("name"),
                        "odds": outcome.get("price"),
                        "point": outcome.get("point"),
                        "last_update": last_update,
                        "is_historical": True,
                    }
                    odds_records.append(record)
        
        return odds_records
    
    async def save_historical_to_database(
        self,
        odds_data: List[Dict[str, Any]],
        session,
    ) -> Tuple[int, int]:
        """
        Save historical odds to database.
        
        The Odds table requires:
        - game_id (FK to games)
        - bet_type (spread, moneyline, total)
        - home_line/away_line, home_odds/away_odds, total/over_odds/under_odds
        
        We need to:
        1. Group raw data by event+sportsbook+market
        2. Find matching game by team names and time
        3. Create properly structured Odds records
        """
        from app.models.models import Odds, Sportsbook, Game, Team, Sport
        from sqlalchemy import and_, func
        from collections import defaultdict
        
        saved = 0
        updated = 0
        skipped = 0
        
        # Step 1: Group odds by event_id + sportsbook + market
        # This combines home/away/over/under into single records
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        for record in odds_data:
            event_id = record.get("event_id")
            sportsbook_key = record.get("sportsbook")
            market = record.get("market")  # h2h, spreads, totals
            selection = record.get("selection")  # team name or Over/Under
            home_team = record.get("home_team")
            away_team = record.get("away_team")
            
            # Store metadata on the group
            key = (event_id, sportsbook_key, market)
            grouped[key]["meta"] = {
                "sport_code": record.get("sport_code"),
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": record.get("commence_time"),
                "sportsbook": sportsbook_key,
                "sportsbook_title": record.get("sportsbook_title"),
                "last_update": record.get("last_update"),
                "market": market,
            }
            
            # Determine selection type
            if selection == home_team:
                grouped[key]["home"] = {
                    "odds": record.get("odds"),
                    "point": record.get("point"),
                }
            elif selection == away_team:
                grouped[key]["away"] = {
                    "odds": record.get("odds"),
                    "point": record.get("point"),
                }
            elif selection and selection.lower() == "over":
                grouped[key]["over"] = {
                    "odds": record.get("odds"),
                    "point": record.get("point"),
                }
            elif selection and selection.lower() == "under":
                grouped[key]["under"] = {
                    "odds": record.get("odds"),
                    "point": record.get("point"),
                }
        
        # Step 2: Process each grouped record
        # Cache for game lookups
        game_cache = {}
        sportsbook_cache = {}
        
        for key, data in grouped.items():
            try:
                meta = data.get("meta", {})
                if not meta:
                    skipped += 1
                    continue
                
                home_team_name = meta.get("home_team")
                away_team_name = meta.get("away_team")
                sport_code = meta.get("sport_code")
                commence_time_str = meta.get("commence_time")
                sportsbook_key = meta.get("sportsbook")
                market = meta.get("market")
                
                if not all([home_team_name, away_team_name, sport_code]):
                    skipped += 1
                    continue
                
                # Find game by teams and approximate time
                cache_key = f"{sport_code}:{home_team_name}:{away_team_name}:{commence_time_str}"
                
                if cache_key in game_cache:
                    game_id = game_cache[cache_key]
                else:
                    game_id = await self._find_game_id(
                        session, sport_code, home_team_name, away_team_name, commence_time_str
                    )
                    game_cache[cache_key] = game_id
                
                if not game_id:
                    skipped += 1
                    continue
                
                # Get or create sportsbook
                if sportsbook_key in sportsbook_cache:
                    sportsbook_id = sportsbook_cache[sportsbook_key]
                else:
                    sportsbook = await self._get_or_create_sportsbook(
                        session, sportsbook_key, meta.get("sportsbook_title", sportsbook_key)
                    )
                    sportsbook_id = sportsbook.id
                    sportsbook_cache[sportsbook_key] = sportsbook_id
                
                # Map market to bet_type
                bet_type_map = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
                bet_type = bet_type_map.get(market, market)
                
                # Parse recorded_at timestamp
                last_update = meta.get("last_update")
                if last_update:
                    try:
                        recorded_at = datetime.fromisoformat(last_update.replace("Z", "+00:00")).replace(tzinfo=None)
                    except:
                        recorded_at = datetime.utcnow()
                else:
                    recorded_at = datetime.utcnow()
                
                # Build Odds record based on bet_type
                odds_kwargs = {
                    "game_id": game_id,
                    "sportsbook_id": sportsbook_id,
                    "sportsbook_key": sportsbook_key,
                    "bet_type": bet_type,
                    "is_opening": False,
                    "recorded_at": recorded_at,
                }
                
                if bet_type == "moneyline":
                    home_data = data.get("home", {})
                    away_data = data.get("away", {})
                    odds_kwargs["home_odds"] = int(home_data.get("odds")) if home_data.get("odds") else None
                    odds_kwargs["away_odds"] = int(away_data.get("odds")) if away_data.get("odds") else None
                    
                elif bet_type == "spread":
                    home_data = data.get("home", {})
                    away_data = data.get("away", {})
                    odds_kwargs["home_line"] = float(home_data.get("point")) if home_data.get("point") else None
                    odds_kwargs["away_line"] = float(away_data.get("point")) if away_data.get("point") else None
                    odds_kwargs["home_odds"] = int(home_data.get("odds")) if home_data.get("odds") else None
                    odds_kwargs["away_odds"] = int(away_data.get("odds")) if away_data.get("odds") else None
                    
                elif bet_type == "total":
                    over_data = data.get("over", {})
                    under_data = data.get("under", {})
                    odds_kwargs["total"] = float(over_data.get("point")) if over_data.get("point") else None
                    odds_kwargs["over_odds"] = int(over_data.get("odds")) if over_data.get("odds") else None
                    odds_kwargs["under_odds"] = int(under_data.get("odds")) if under_data.get("odds") else None
                
                # Create and add the odds record
                odds_record = Odds(**odds_kwargs)
                session.add(odds_record)
                saved += 1
                
                # Commit every 100 records
                if saved % 100 == 0:
                    await session.commit()
                    
            except Exception as e:
                logger.debug(f"Skipping record due to error: {e}")
                await session.rollback()
                skipped += 1
                continue
        
        # Final commit
        try:
            await session.commit()
        except Exception as e:
            logger.error(f"Final commit error: {e}")
            await session.rollback()
        
        logger.info(f"[OddsAPI Historical] Saved: {saved}, Skipped: {skipped}")
        return saved, updated
    
    async def _find_game_id(
        self,
        session,
        sport_code: str,
        home_team_name: str,
        away_team_name: str,
        commence_time_str: str,
    ) -> Optional[UUID]:
        """
        Find a game by sport, teams, and approximate time.
        
        Uses flexible team name matching since API names may differ from DB names.
        """
        from app.models.models import Game, Team, Sport
        from sqlalchemy import and_, or_, func
        from datetime import timedelta
        
        try:
            # Parse commence time
            if commence_time_str:
                commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00")).replace(tzinfo=None)
            else:
                return None
            
            # Time window: +/- 24 hours to account for timezone differences
            time_start = commence_time - timedelta(hours=24)
            time_end = commence_time + timedelta(hours=24)
            
            # Find sport
            sport_result = await session.execute(
                select(Sport.id).where(Sport.code == sport_code)
            )
            sport_id = sport_result.scalar_one_or_none()
            if not sport_id:
                return None
            
            # Normalize team names for matching
            home_normalized = self._normalize_team_name(home_team_name)
            away_normalized = self._normalize_team_name(away_team_name)
            
            # Get all games in the time window - use .all() to properly fetch results
            games_result = await session.execute(
                select(Game.id, Game.home_team_id, Game.away_team_id)
                .where(
                    and_(
                        Game.sport_id == sport_id,
                        Game.scheduled_at >= time_start,
                        Game.scheduled_at <= time_end,
                    )
                )
            )
            games_in_window = games_result.all()
            
            for game_row in games_in_window:
                game_id, home_team_id, away_team_id = game_row
                
                # Get team names
                home_team_result = await session.execute(
                    select(Team.name).where(Team.id == home_team_id)
                )
                db_home_name = home_team_result.scalar_one_or_none()
                
                away_team_result = await session.execute(
                    select(Team.name).where(Team.id == away_team_id)
                )
                db_away_name = away_team_result.scalar_one_or_none()
                
                if db_home_name and db_away_name:
                    db_home_normalized = self._normalize_team_name(db_home_name)
                    db_away_normalized = self._normalize_team_name(db_away_name)
                    
                    # Match if both teams match (allowing partial matches)
                    if (self._teams_match(home_normalized, db_home_normalized) and
                        self._teams_match(away_normalized, db_away_normalized)):
                        return game_id
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding game: {e}")
            return None
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        if not name:
            return ""
        # Remove common suffixes, lowercase, strip whitespace
        normalized = name.lower().strip()
        # Remove "City" suffix for matching (e.g., "Kansas City Chiefs" -> "chiefs")
        words = normalized.split()
        if len(words) > 1:
            # Return just the team name (last word typically)
            return words[-1]
        return normalized
    
    def _teams_match(self, name1: str, name2: str) -> bool:
        """Check if two team names match (allowing partial matches)."""
        if not name1 or not name2:
            return False
        # Exact match
        if name1 == name2:
            return True
        # One contains the other
        if name1 in name2 or name2 in name1:
            return True
        return False
    
    async def track_line_movements(
        self,
        odds_data: List[Dict[str, Any]],
        session,
    ) -> int:
        """
        Track line movements by comparing current odds to previous odds.
        
        A line movement is detected when the spread, total, or moneyline changes
        from the previous snapshot.
        
        Args:
            odds_data: List of current odds records
            session: Database session
            
        Returns:
            Number of movements detected and saved
        """
        from app.models.models import OddsMovement, Odds, Game
        from sqlalchemy import select, and_, desc
        
        movements_saved = 0
        
        # Group odds by game for comparison
        game_odds = {}
        for record in odds_data:
            external_id = record.get("external_id")
            if external_id:
                if external_id not in game_odds:
                    game_odds[external_id] = []
                game_odds[external_id].append(record)
        
        for external_id, records in game_odds.items():
            try:
                # Find the game
                game_result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = game_result.scalar_one_or_none()
                
                if not game:
                    continue
                
                # Get the latest odds for this game
                prev_odds_result = await session.execute(
                    select(Odds)
                    .where(Odds.game_id == game.id)
                    .order_by(desc(Odds.recorded_at))
                    .limit(1)
                )
                prev_odds = prev_odds_result.scalar_one_or_none()
                
                if not prev_odds:
                    continue
                
                # Group current records by market type
                for record in records:
                    market_type = record.get("market_type")
                    selection = record.get("selection")
                    line = record.get("line")
                    
                    if not market_type or line is None:
                        continue
                    
                    # Compare with previous odds
                    previous_line = None
                    movement = None
                    
                    if market_type == "spread":
                        if selection == "home" and prev_odds.home_line is not None:
                            previous_line = prev_odds.home_line
                            if line != previous_line:
                                movement = line - previous_line
                        elif selection == "away" and prev_odds.away_line is not None:
                            previous_line = prev_odds.away_line
                            if line != previous_line:
                                movement = line - previous_line
                    
                    elif market_type == "total":
                        if prev_odds.total is not None:
                            previous_line = prev_odds.total
                            if line != previous_line:
                                movement = line - previous_line
                    
                    # If movement detected, record it
                    if movement is not None and abs(movement) >= 0.5:
                        # Detect steam move (rapid movement in one direction)
                        is_steam = abs(movement) >= 2.0
                        
                        # Detect reverse movement (movement against public betting)
                        is_reverse = False  # Would need public betting data
                        
                        movement_record = OddsMovement(
                            game_id=game.id,
                            bet_type=market_type,
                            previous_line=previous_line,
                            current_line=line,
                            movement=movement,
                            is_steam=is_steam,
                            is_reverse=is_reverse,
                        )
                        session.add(movement_record)
                        movements_saved += 1
                
            except Exception as e:
                logger.debug(f"Error tracking movement for {external_id}: {e}")
                continue
        
        try:
            await session.commit()
        except Exception as e:
            logger.error(f"Error committing movements: {e}")
            await session.rollback()
        
        logger.info(f"[OddsAPI] Tracked {movements_saved} line movements")
        return movements_saved

    # =========================================================================
    # COMPREHENSIVE 5-YEAR HISTORICAL DATA COLLECTION
    # =========================================================================
    
    async def collect_full_historical(
        self,
        sport_code: str = None,
        years_back: int = 5,
        sample_interval: int = 1,
        include_tennis: bool = True,
    ) -> CollectorResult:
        """
        Collect MAXIMUM historical odds data from TheOddsAPI.
        
        IMPORTANT: OddsAPI historical data only goes back to 2020 (~5 years max).
        10-year historical data is NOT available from OddsAPI.
        
        $119/mo Mega Plan Strategy:
        - 10,000 requests/month
        - Each request costs 10 per region per market (3 markets = 30)
        - Daily collection for 8 sports × 30 days = 240 requests/month
        - For 5 years: sample every 3-7 days to stay within limits
        
        Args:
            sport_code: Specific sport or None for all 10 sports
            years_back: Years back (max 5, OddsAPI limit)
            sample_interval: Days between samples (1=daily, 3=every 3rd day)
            include_tennis: Include ATP/WTA tournaments
            
        Returns:
            CollectorResult with comprehensive odds data
        """
        from datetime import datetime, timedelta
        from rich.console import Console
        
        console = Console()
        
        # Limit to OddsAPI's available history
        years_back = min(years_back, 5)
        days_back = years_back * 365
        
        # Main sports
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        
        if sport_code:
            sports = [sport_code.upper()]
        else:
            sports = main_sports.copy()
        
        markets = ["spreads", "h2h", "totals"]
        
        total_odds = 0
        total_movements = 0
        all_errors = []
        sport_stats = {}
        
        console.print(f"\n[bold blue]{'='*70}[/bold blue]")
        console.print(f"[bold]📊 ODDSAPI COMPREHENSIVE HISTORICAL DATA COLLECTION[/bold]")
        console.print(f"[bold blue]{'='*70}[/bold blue]")
        console.print(f"[yellow]⚠️  NOTE: OddsAPI only has data back to 2020 (~5 years max)[/yellow]")
        console.print(f"Sports: {', '.join(sports)}")
        console.print(f"Years: {years_back} ({days_back} days)")
        console.print(f"Sample interval: Every {sample_interval} day(s)")
        console.print(f"Estimated API requests: ~{len(sports) * (days_back // sample_interval)}")
        console.print()
        
        # Collect main sports
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                console.print(f"  [yellow]⚠️ {sport}: No API key mapping[/yellow]")
                continue
            
            console.print(f"[cyan]📈 {sport}...[/cyan]")
            sport_odds = 0
            sport_movements = 0
            
            try:
                # Collect by season chunks for efficiency
                for day_offset in range(0, days_back, sample_interval):
                    target_date = datetime.utcnow() - timedelta(days=day_offset + 1)
                    date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")
                    
                    try:
                        params = {
                            "apiKey": self.api_key,
                            "regions": "us",
                            "markets": ",".join(markets),
                            "oddsFormat": "american",
                            "date": date_str,
                        }
                        
                        endpoint = f"/historical/sports/{api_sport_key}/odds"
                        data = await self.get(endpoint, params=params)
                        
                        if data and isinstance(data, dict):
                            events = data.get("data", [])
                            timestamp = data.get("timestamp")
                            
                            for event in events:
                                parsed = self._parse_historical_event(event, sport, timestamp)
                                sport_odds += len(parsed)
                                
                                # Save to database
                                from app.core.database import db_manager
                                await db_manager.initialize()
                                async with db_manager.session() as session:
                                    saved, _ = await self.save_historical_to_database(parsed, session)
                        
                        # Progress update every 90 days
                        if day_offset % 90 == 0 and day_offset > 0:
                            console.print(f"    Progress: {day_offset}/{days_back} days, {sport_odds} odds...")
                        
                        # Rate limiting
                        await asyncio.sleep(0.2)
                        
                    except Exception as e:
                        if "401" in str(e) or "403" in str(e) or "429" in str(e):
                            console.print(f"  [red]❌ API limit/auth error: {str(e)[:50]}[/red]")
                            all_errors.append(f"{sport}: {str(e)[:50]}")
                            break
                        continue
                
                sport_stats[sport] = {"odds": sport_odds, "movements": sport_movements}
                total_odds += sport_odds
                console.print(f"  [green]✅ {sport}: {sport_odds:,} odds records[/green]")
                
            except Exception as e:
                all_errors.append(f"{sport}: {str(e)[:50]}")
                console.print(f"  [red]❌ {sport}: Error - {str(e)[:40]}[/red]")
        
        # Collect tennis tournaments if enabled
        if include_tennis and not sport_code:
            console.print(f"\n[cyan]🎾 Tennis Tournaments...[/cyan]")
            tennis_odds = await self._collect_tennis_historical(years_back, sample_interval)
            total_odds += tennis_odds
            sport_stats["Tennis"] = {"odds": tennis_odds, "movements": 0}
            console.print(f"  [green]✅ Tennis: {tennis_odds:,} odds records[/green]")
        
        # Summary
        console.print(f"\n[bold blue]{'='*70}[/bold blue]")
        console.print(f"[bold green]📊 COLLECTION COMPLETE[/bold green]")
        console.print(f"[bold]Total Odds Records: {total_odds:,}[/bold]")
        for sport, stats in sport_stats.items():
            console.print(f"  {sport}: {stats['odds']:,} odds")
        if all_errors:
            console.print(f"[yellow]Errors: {len(all_errors)}[/yellow]")
        console.print(f"[bold blue]{'='*70}[/bold blue]\n")
        
        return CollectorResult(
            success=total_odds > 0,
            data={"total_odds": total_odds, "by_sport": sport_stats},
            records_count=total_odds,
            error="; ".join(all_errors) if all_errors else None,
            metadata={"type": "full_historical", "years_back": years_back}
        )

    async def _collect_tennis_historical(
        self,
        years_back: int = 5,
        sample_interval: int = 7,
    ) -> int:
        """Collect historical tennis odds from major tournaments."""
        from datetime import datetime, timedelta
        
        total_odds = 0
        days_back = years_back * 365
        
        # Combine ATP and WTA tournaments
        all_tournaments = []
        for tour_list in ODDS_API_TENNIS_TOURNAMENTS.values():
            all_tournaments.extend(tour_list)
        
        for tournament_key in all_tournaments:
            try:
                for day_offset in range(0, days_back, sample_interval):
                    target_date = datetime.utcnow() - timedelta(days=day_offset + 1)
                    date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")
                    
                    params = {
                        "apiKey": self.api_key,
                        "regions": "us",
                        "markets": "h2h",  # Tennis only has moneyline
                        "oddsFormat": "american",
                        "date": date_str,
                    }
                    
                    endpoint = f"/historical/sports/{tournament_key}/odds"
                    
                    try:
                        data = await self.get(endpoint, params=params)
                        
                        if data and isinstance(data, dict):
                            events = data.get("data", [])
                            timestamp = data.get("timestamp")
                            
                            sport_code = "ATP" if "atp" in tournament_key else "WTA"
                            
                            for event in events:
                                parsed = self._parse_historical_event(event, sport_code, timestamp)
                                total_odds += len(parsed)
                                
                                # Save to database
                                from app.core.database import db_manager
                                await db_manager.initialize()
                                async with db_manager.session() as session:
                                    await self.save_historical_to_database(parsed, session)
                        
                        await asyncio.sleep(0.2)
                        
                    except Exception as e:
                        # Tournament might not have data for this date
                        continue
                        
            except Exception as e:
                logger.debug(f"Tennis tournament {tournament_key} error: {e}")
                continue
        
        return total_odds

    async def collect_movements_historical(
        self,
        sport_code: str = None,
        days_back: int = 30,
        years_back: int = None,
    ) -> CollectorResult:
        """
        Collect historical line movements by comparing odds snapshots.
        
        Automatically creates games if they don't exist, so no pre-import required.
        
        OddsAPI Historical Limits:
        - Data available from ~June 2020 onwards (~5 years max)
        - Each API call costs quota
        - For maximum data, use years_back=5 or days_back=1825
        
        Args:
            sport_code: Specific sport or None for all
            days_back: Days of history to analyze (default 30)
            years_back: Years of history (overrides days_back if set, max 5)
            
        Returns:
            CollectorResult with movement data
        """
        from datetime import datetime, timedelta
        from rich.console import Console
        
        console = Console()
        
        # Calculate days from years if specified
        if years_back:
            # OddsAPI only has data from ~June 2020, cap at 5 years
            years_back = min(years_back, 5)
            days_back = years_back * 365
            console.print(f"[yellow]⚠️ OddsAPI historical data limited to ~June 2020 onwards (max ~5 years)[/yellow]")
        
        # All supported sports
        main_sports = ["NFL", "NBA", "NHL", "MLB", "NCAAF", "NCAAB", "WNBA", "CFL"]
        sports = [sport_code.upper()] if sport_code else main_sports
        
        total_movements = 0
        total_games_created = 0
        all_errors = []
        api_calls = 0
        
        console.print(f"\n[bold cyan]📉 COLLECTING MAXIMUM LINE MOVEMENTS[/bold cyan]")
        console.print(f"Sports: {', '.join(sports)}")
        console.print(f"Days back: {days_back} ({days_back // 365} years)")
        console.print(f"[dim]Games will be auto-created if they don't exist[/dim]")
        console.print(f"[dim]Estimated API calls: ~{days_back * len(sports) * 2}[/dim]")
        console.print()
        
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                console.print(f"[yellow]⚠️ {sport}: No API key mapping, skipping[/yellow]")
                continue
            
            console.print(f"[cyan]📈 {sport}...[/cyan]")
            sport_movements = 0
            sport_games_created = 0
            sport_api_calls = 0
            
            # Accumulate movements for batch save
            accumulated_movements = []
            
            try:
                # Process day by day
                for day_offset in range(0, days_back):
                    target_date = datetime.utcnow() - timedelta(days=day_offset + 1)
                    
                    # Skip dates before OddsAPI historical data began (~June 2020)
                    if target_date < datetime(2020, 6, 1):
                        continue
                    
                    # Get TWO snapshots per day (morning/evening) - faster than 3
                    snapshot_times = ["10:00:00", "21:00:00"]
                    day_snapshots = []
                    
                    for snap_time in snapshot_times:
                        date_str = target_date.strftime(f"%Y-%m-%dT{snap_time}Z")
                        
                        try:
                            params = {
                                "apiKey": self.api_key,
                                "regions": "us,us2,eu",  # Main regions
                                "markets": "spreads,h2h,totals",
                                "oddsFormat": "american",
                                "date": date_str,
                            }
                            data = await self.get(f"/historical/sports/{api_sport_key}/odds", params=params)
                            sport_api_calls += 1
                            api_calls += 1
                            
                            if data and data.get("data"):
                                day_snapshots.append({
                                    "time": date_str,
                                    "events": data.get("data", [])
                                })
                            
                            await asyncio.sleep(0.15)  # Rate limiting
                            
                        except Exception as e:
                            logger.debug(f"Snapshot error {date_str}: {e}")
                            continue
                    
                    # Compare snapshots for movements
                    if len(day_snapshots) >= 2:
                        movements = self._compare_snapshots_for_movements(
                            day_snapshots[0]["events"], 
                            day_snapshots[1]["events"], 
                            sport, 
                            datetime.fromisoformat(day_snapshots[0]["time"].replace("Z", "+00:00")),
                            datetime.fromisoformat(day_snapshots[1]["time"].replace("Z", "+00:00"))
                        )
                        accumulated_movements.extend(movements)
                    
                    # Save batch every 7 days OR at end
                    if (day_offset + 1) % 7 == 0 or day_offset == days_back - 1:
                        if accumulated_movements:
                            from app.core.database import db_manager
                            await db_manager.initialize()
                            async with db_manager.session() as session:
                                saved, created = await self._save_movements_with_games(
                                    accumulated_movements, session, sport
                                )
                                sport_movements += saved
                                sport_games_created += created
                            accumulated_movements = []  # Reset
                        
                        # Progress update
                        console.print(f"    [dim]{sport}: {day_offset + 1}/{days_back} days, {sport_movements} movements, {sport_api_calls} calls[/dim]")
                
                total_movements += sport_movements
                total_games_created += sport_games_created
                console.print(f"  [green]✅ {sport}: {sport_movements} movements, {sport_games_created} games ({sport_api_calls} API calls)[/green]")
                
            except Exception as e:
                all_errors.append(f"{sport}: {str(e)[:50]}")
                console.print(f"  [red]❌ {sport}: {str(e)[:50]}[/red]")
        
        console.print(f"\n[bold green]═══════════════════════════════════════════════════[/bold green]")
        console.print(f"[bold green]TOTAL: {total_movements} movements, {total_games_created} games created[/bold green]")
        console.print(f"[bold green]API Calls Used: {api_calls}[/bold green]")
        console.print(f"[bold green]═══════════════════════════════════════════════════[/bold green]")
        
        return CollectorResult(
            success=total_movements > 0 or len(all_errors) == 0,
            data={"total_movements": total_movements, "games_created": total_games_created, "api_calls": api_calls},
            records_count=total_movements,
            error="; ".join(all_errors) if all_errors else None,
            metadata={"type": "movements_historical", "days_back": days_back, "years_back": years_back}
        )

    def _compare_snapshots_for_movements(
        self,
        events1: List[Dict],
        events2: List[Dict],
        sport_code: str,
        date1: datetime,
        date2: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Compare two snapshots and detect ALL line movements.
        
        Captures movements with very low threshold for maximum data.
        """
        movements = []
        
        # Index events by ID for fast lookup
        events1_by_id = {e.get("id"): e for e in events1}
        
        for event2 in events2:
            event_id = event2.get("id")
            if event_id not in events1_by_id:
                continue
            
            event1 = events1_by_id[event_id]
            home_team = event2.get("home_team")
            away_team = event2.get("away_team")
            commence_time = event2.get("commence_time")
            
            # Index all bookmaker odds from event1
            odds1_index = {}
            for book in event1.get("bookmakers", []):
                book_key = book.get("key")
                for market in book.get("markets", []):
                    market_key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        key = f"{book_key}:{market_key}:{outcome.get('name')}"
                        odds1_index[key] = {
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        }
            
            # Compare with event2 odds
            for book in event2.get("bookmakers", []):
                book_key = book.get("key")
                book_title = book.get("title")
                
                for market in book.get("markets", []):
                    market_key = market.get("key")
                    
                    for outcome in market.get("outcomes", []):
                        selection = outcome.get("name")
                        current_price = outcome.get("price")
                        current_point = outcome.get("point")
                        
                        key = f"{book_key}:{market_key}:{selection}"
                        
                        if key in odds1_index:
                            prev = odds1_index[key]
                            prev_price = prev.get("price")
                            prev_point = prev.get("point")
                            
                            # Calculate changes
                            price_change = None
                            line_change = None
                            
                            if prev_price and current_price:
                                price_change = current_price - prev_price
                            
                            if prev_point is not None and current_point is not None:
                                line_change = current_point - prev_point
                            
                            # LOW THRESHOLD for maximum data capture:
                            # Any odds change >= 2 points OR any line change >= 0.5
                            has_movement = False
                            if price_change and abs(price_change) >= 2:
                                has_movement = True
                            if line_change and abs(line_change) >= 0.5:
                                has_movement = True
                            
                            if has_movement:
                                movements.append({
                                    "sport_code": sport_code,
                                    "event_id": event_id,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "commence_time": commence_time,
                                    "sportsbook": book_key,
                                    "sportsbook_title": book_title,
                                    "market": market_key,
                                    "selection": selection,
                                    "previous_odds": prev_price,
                                    "current_odds": current_price,
                                    "odds_change": price_change,
                                    "previous_line": prev_point,
                                    "current_line": current_point,
                                    "line_change": line_change,
                                    "snapshot_time_1": date1.isoformat() if hasattr(date1, 'isoformat') else str(date1),
                                    "snapshot_time_2": date2.isoformat() if hasattr(date2, 'isoformat') else str(date2),
                                })
        
        return movements

    async def _save_movements_with_games(
        self,
        movements: List[Dict[str, Any]],
        session,
        sport_code: str,
    ) -> Tuple[int, int]:
        """
        Save movements to database with FAST batch processing.
        
        Optimized approach:
        1. Pre-fetch/create all unique games at once
        2. Bulk insert movements
        3. Minimal database roundtrips
        
        Returns:
            Tuple of (movements_saved, games_created)
        """
        if not movements:
            return 0, 0
        
        saved = 0
        games_created = 0
        
        # Step 1: Get or create sport ONCE
        sport = await self._get_or_create_sport(session, sport_code)
        if not sport:
            logger.warning(f"Could not get/create sport {sport_code}")
            return 0, 0
        
        # Step 2: Collect unique events and teams
        unique_events = {}  # event_id -> event_data
        unique_teams = set()
        
        for mov in movements:
            event_id = mov.get("event_id")
            if event_id not in unique_events:
                unique_events[event_id] = {
                    "external_id": event_id,
                    "home_team": mov.get("home_team"),
                    "away_team": mov.get("away_team"),
                    "commence_time": mov.get("commence_time"),
                }
                unique_teams.add(mov.get("home_team"))
                unique_teams.add(mov.get("away_team"))
        
        # Step 3: Batch create/fetch all teams at once
        team_cache = {}
        for team_name in unique_teams:
            if team_name:
                team = await self._get_or_create_team(session, sport.id, team_name)
                if team:
                    team_cache[team_name] = team.id
        
        await session.commit()
        
        # Step 4: Batch create/fetch all games at once
        game_cache = {}  # event_id -> game_id
        
        for event_id, event_data in unique_events.items():
            home_team = event_data.get("home_team")
            away_team = event_data.get("away_team")
            
            home_team_id = team_cache.get(home_team)
            away_team_id = team_cache.get(away_team)
            
            if not home_team_id or not away_team_id:
                continue
            
            # Try to find existing game by external_id first (fastest)
            from sqlalchemy import select
            result = await session.execute(
                select(Game.id).where(Game.external_id == event_id)
            )
            existing_game = result.scalar_one_or_none()
            
            if existing_game:
                game_cache[event_id] = existing_game
            else:
                # Create new game
                try:
                    commence_time = event_data.get("commence_time")
                    if isinstance(commence_time, str):
                        from dateutil.parser import parse as parse_date
                        commence_time = parse_date(commence_time).replace(tzinfo=None)
                    
                    game = Game(
                        sport_id=sport.id,
                        external_id=event_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        scheduled_at=commence_time,
                        status=GameStatus.SCHEDULED,
                    )
                    session.add(game)
                    await session.flush()
                    game_cache[event_id] = game.id
                    games_created += 1
                except Exception as e:
                    logger.debug(f"Error creating game {event_id}: {e}")
                    continue
        
        await session.commit()
        
        # Step 5: Bulk insert movements
        movement_records = []
        for mov in movements:
            event_id = mov.get("event_id")
            game_id = game_cache.get(event_id)
            
            if not game_id:
                continue
            
            market = mov.get("market")
            bet_type_map = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
            bet_type = bet_type_map.get(market, market)
            
            line_change = mov.get("line_change")
            if line_change is None and mov.get("odds_change"):
                line_change = mov.get("odds_change") / 10.0
            
            is_steam = bool(line_change and abs(line_change) >= 1.5)
            
            movement_records.append(OddsMovement(
                game_id=game_id,
                bet_type=bet_type,
                previous_line=mov.get("previous_line"),
                current_line=mov.get("current_line"),
                movement=line_change,
                is_steam=is_steam,
                is_reverse=False,
                detected_at=datetime.utcnow(),
            ))
        
        # Bulk add all movements
        session.add_all(movement_records)
        await session.commit()
        saved = len(movement_records)
        
        return saved, games_created

    async def collect_movements_full_history(
        self,
        years: int = 5,
    ) -> CollectorResult:
        """
        Collect MAXIMUM historical line movements from OddsAPI.
        
        OddsAPI has data from ~June 2020, so max ~5 years of historical data.
        This collects ALL available movements across ALL sports.
        
        ⚠️ WARNING: This uses significant API quota!
        Estimated: ~30,000+ API calls for 5 years across 8 sports
        
        Args:
            years: Years of history (max 5, limited by OddsAPI)
            
        Returns:
            CollectorResult with movement data
        """
        from rich.console import Console
        console = Console()
        
        # Cap at 5 years (OddsAPI limit - data from June 2020)
        years = min(years, 5)
        
        console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]🏆 MAXIMUM HISTORICAL MOVEMENTS COLLECTION[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[yellow]Collecting {years} years of movements across ALL sports[/yellow]")
        console.print(f"[yellow]⚠️ OddsAPI only has data from ~June 2020[/yellow]")
        console.print(f"[dim]Estimated API calls: ~{years * 365 * 8 * 3} (may vary by sport season)[/dim]")
        console.print()
        
        return await self.collect_movements_historical(
            sport_code=None,
            years_back=years
        )


# Create and register collector instance
odds_collector = OddsCollector()
collector_manager.register(odds_collector)