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
        sports = [sport_code.upper()] if sport_code else ["NFL", "NBA", "NHL", "MLB"]
        markets = markets or ["spreads", "h2h", "totals"]
        all_odds = []
        errors = []
        
        for sport in sports:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport)
            if not api_sport_key:
                continue
                
            try:
                odds = await self._fetch_historical_odds(
                    api_sport_key, sport, days_back, markets
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
        """Save historical odds to database."""
        from app.models.models import Odds, Sportsbook
        
        saved = 0
        updated = 0
        
        for record in odds_data:
            try:
                # Get or create sportsbook
                sportsbook = await self._get_or_create_sportsbook(
                    session,
                    record["sportsbook"],
                    record.get("sportsbook_title", record["sportsbook"])
                )
                
                # Create odds record
                odds = Odds(
                    sportsbook_id=sportsbook.id,
                    external_id=f"{record['event_id']}_{record['sportsbook']}_{record['market']}_{record['selection']}",
                    market_type=record["market"],
                    selection=record["selection"],
                    odds_value=float(record["odds"]) if record["odds"] else None,
                    point=float(record["point"]) if record.get("point") else None,
                    is_live=False,
                    recorded_at=datetime.fromisoformat(record["last_update"].replace("Z", "+00:00")).replace(tzinfo=None) if record.get("last_update") else datetime.utcnow(),
                )
                session.add(odds)
                saved += 1
                
                # Commit every 100 records
                if saved % 100 == 0:
                    await session.commit()
                    
            except Exception as e:
                logger.debug(f"Skipping duplicate/error: {e}")
                await session.rollback()
                continue
        
        await session.commit()
        return saved, updated


# Create and register collector instance
odds_collector = OddsCollector()
collector_manager.register(odds_collector)
