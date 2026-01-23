"""
LOYALEY - TheOddsAPI Collector
Phase 1: Data Collection Services

Collects real-time odds from 40+ sportsbooks via TheOddsAPI.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings, ODDS_API_SPORT_KEYS
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
        """
        if not self.api_key:
            return CollectorResult(
                success=False,
                error="TheOddsAPI key not configured",
            )
        
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(ODDS_API_SPORT_KEYS.keys())
        )
        
        markets = markets or self.MARKETS
        all_odds = []
        errors = []
        successful_sports = []
        
        # Tennis sports only support h2h (moneyline) market
        TENNIS_SPORTS = ["ATP", "WTA"]
        
        logger.info(f"Starting collection for {len(sports_to_collect)} sport(s): {sports_to_collect}")
        
        # Collect each sport individually and combine results
        for sport in sports_to_collect:
            try:
                # Use only h2h market for tennis sports
                sport_markets = markets if sport not in TENNIS_SPORTS else ["h2h"]
                
                logger.info(f"Collecting {sport} odds data (markets: {sport_markets})")
                odds_data = await self._collect_sport_odds(sport, sport_markets)
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
        
        Args:
            odds_data: List of parsed odds records
            session: Database session
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        sportsbook_cache: Dict[str, UUID] = {}
        
        for record in odds_data:
            # Use savepoint to isolate each record so one failure doesn't abort the entire transaction
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
                
                game_result = await session.execute(
                    select(Game).where(Game.external_id == record["external_id"])
                )
                game = game_result.scalar_one_or_none()
                
                # Create game if it doesn't exist
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
                
                existing_odds = await session.execute(
                    select(Odds).where(
                        Odds.game_id == game.id,
                        Odds.sportsbook_id == sportsbook_id,
                        Odds.market_type == record["market_type"],
                        Odds.selection == record["selection"],
                        Odds.is_current == True,
                    )
                )
                existing = existing_odds.scalar_one_or_none()
                
                # If existing odds exist and data hasn't changed, skip saving
                if existing:
                    # Compare price and line (handle None values)
                    new_line = record.get("line")
                    new_price = record.get("price")
                    
                    # Check if data has changed
                    line_changed = (
                        (existing.line is None) != (new_line is None) or
                        (existing.line is not None and new_line is not None and existing.line != new_line)
                    )
                    price_changed = existing.price != new_price
                    
                    if not line_changed and not price_changed:
                        # Data hasn't changed, skip saving
                        await savepoint.rollback()
                        continue
                    
                    # Data has changed, create movement record and mark old as not current
                    movement = OddsMovement(
                        game_id=game.id,
                        sportsbook_id=sportsbook_id,
                        market_type=record["market_type"],
                        old_line=existing.line,
                        new_line=new_line,
                        old_price=existing.price,
                        new_price=new_price,
                        movement_size=self._calculate_movement_size(
                            existing.line, new_line
                        ),
                    )
                    session.add(movement)
                    existing.is_current = False
                
                # Create new odds record (either new or changed)
                new_odds = Odds(
                    game_id=game.id,
                    sportsbook_id=sportsbook_id,
                    market_type=record["market_type"],
                    selection=record["selection"],
                    price=record["price"],
                    line=record.get("line"),
                    is_current=True,
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
                game_date=game_date,
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
        
        # Create new team
        abbreviation = team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()
        team = Team(
            sport_id=sport_id,
            external_id=f"{sport_id}_{team_name.lower().replace(' ', '_')}",
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
            select(Sportsbook).where(Sportsbook.api_key == key)
        )
        sportsbook = result.scalar_one_or_none()
        
        if not sportsbook:
            sharp_books = ["pinnacle", "betcris", "bookmaker"]
            is_sharp = key.lower() in sharp_books
            
            sportsbook = Sportsbook(
                name=name,
                api_key=key,
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


# Create and register collector instance
odds_collector = OddsCollector()
collector_manager.register(odds_collector)
