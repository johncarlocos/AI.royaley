"""
Collector #25: Kalshi Prediction Markets
CFTC-regulated prediction market data for sports events

API Documentation: https://docs.kalshi.com/
Base URL: https://api.elections.kalshi.com/trade-api/v2

Data Available:
- Events: Real-world occurrences that can be traded on
- Markets: Binary outcome contracts within events
- Series: Groups of related events
- Trades: Historical trade data
- Candlesticks: Price history (OHLC)

Sports Categories:
- NFL, NBA, MLB, NHL, MLS, College Football, College Basketball
- Golf, Tennis, MMA/UFC, Soccer/Football

Tables Created:
- kalshi_series: Event series (e.g., "NFL Championship")
- kalshi_events: Individual events (e.g., "Super Bowl LX Winner")
- kalshi_markets: Tradeable contracts with prices
- kalshi_prices: Historical price snapshots
- kalshi_trades: Trade history for volume analysis

FREE API - No authentication required for public endpoints
Rate Limit: ~100 requests/minute
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Any, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.collectors.base_collector import BaseCollector, CollectorResult

logger = logging.getLogger(__name__)


@dataclass
class KalshiSeries:
    """Kalshi series (group of related events)"""
    ticker: str
    title: str
    category: str
    tags: List[str] = field(default_factory=list)
    frequency: Optional[str] = None
    

@dataclass
class KalshiEvent:
    """Kalshi event data structure"""
    event_ticker: str
    series_ticker: str
    title: str
    sub_title: Optional[str] = None
    category: Optional[str] = None
    strike_date: Optional[datetime] = None
    mutually_exclusive: bool = True


@dataclass 
class KalshiMarket:
    """Kalshi market (contract) data structure"""
    ticker: str
    event_ticker: str
    title: str
    subtitle: Optional[str] = None
    status: str = "open"
    yes_bid: Optional[int] = None  # in cents
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    result: Optional[str] = None  # yes/no/null
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    created_time: Optional[datetime] = None


class KalshiCollector(BaseCollector):
    """
    Kalshi prediction market data collector.
    
    Collects sports-related event contracts from Kalshi's
    CFTC-regulated prediction market exchange.
    """
    
    COLLECTOR_NAME = "kalshi"
    COLLECTOR_ID = 25
    
    # API Configuration
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    # Sports-related categories and series prefixes
    SPORTS_CATEGORIES = ["Sports"]
    
    SPORTS_SERIES_PREFIXES = [
        "KX",  # General sports
        "INFL",  # NFL
        "INBA",  # NBA  
        "IMLB",  # MLB
        "INHL",  # NHL
        "IMLS",  # MLS
        "INCAAF",  # College Football
        "INCAAB",  # College Basketball
        "IPGA",  # Golf/PGA
        "IUFC",  # UFC/MMA
        "ITENNIS",  # Tennis
        "NFL",  # NFL direct
        "NBA",  # NBA direct
        "MLB",  # MLB direct
        "NHL",  # NHL direct
        "CFB",  # College Football
        "CBB",  # College Basketball
        "SUPERBOWL",
        "WORLDSERIES",
        "STANLEY",
        "FINALS",
        "MARCH",  # March Madness
    ]
    
    # Map Kalshi categories to our sport codes
    SPORT_MAPPING = {
        "nfl": "NFL",
        "football": "NFL", 
        "pro football": "NFL",
        "nba": "NBA",
        "basketball": "NBA",
        "mlb": "MLB",
        "baseball": "MLB",
        "nhl": "NHL",
        "hockey": "NHL",
        "mls": "MLS",
        "soccer": "SOCCER",
        "ncaaf": "NCAAF",
        "college football": "NCAAF",
        "ncaab": "NCAAB",
        "college basketball": "NCAAB",
        "march madness": "NCAAB",
        "pga": "PGA",
        "golf": "PGA",
        "ufc": "UFC",
        "mma": "UFC",
        "atp": "ATP",
        "wta": "WTA",
        "tennis": "ATP",
    }
    
    def __init__(self, db: AsyncSession, config: Optional[Dict] = None):
        super().__init__(
            name="kalshi",
            base_url=self.BASE_URL,
            rate_limit=60,
            rate_window=60,
            timeout=30.0,
            max_retries=3
        )
        self.db = db
        self.config = config or {}
        self._tables_checked = False
        
    async def collect(self, **kwargs) -> CollectorResult:
        """Main collection method - required abstract implementation"""
        return await self._collect_impl(**kwargs)
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data - required abstract implementation"""
        if data is None:
            return False
        if isinstance(data, dict):
            return any(k in data for k in ["series", "events", "markets", "prices"])
        return True
    
    async def collect_all(self, years_back: int = 10) -> Dict[str, Any]:
        """Main entry point for data collection"""
        result = await self.collect(years_back=years_back)
        return result.data if result.data else {
            "series": 0,
            "events": 0,
            "markets": 0,
            "prices": 0,
            "trades": 0,
            "errors": [result.error] if result.error else []
        }
    
    async def _ensure_tables(self):
        """Ensure Kalshi tables exist"""
        if self._tables_checked:
            return
            
        try:
            # Check if tables exist
            result = await self.db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'kalshi_series'
                )
            """))
            exists = result.scalar()
            
            if not exists:
                logger.info("[Kalshi] Creating tables...")
                await self._create_tables()
                
            self._tables_checked = True
            
        except Exception as e:
            logger.error(f"[Kalshi] Table check error: {e}")
            raise
    
    async def _create_tables(self):
        """Create Kalshi-specific tables"""
        
        # Series table
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS kalshi_series (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                ticker VARCHAR(50) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                category VARCHAR(100),
                tags JSONB DEFAULT '[]',
                frequency VARCHAR(50),
                sport_code VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Events table  
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS kalshi_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_ticker VARCHAR(100) UNIQUE NOT NULL,
                series_ticker VARCHAR(50),
                title TEXT NOT NULL,
                sub_title TEXT,
                category VARCHAR(100),
                sport_code VARCHAR(20),
                strike_date TIMESTAMP,
                mutually_exclusive BOOLEAN DEFAULT true,
                game_id UUID REFERENCES games(id),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Markets (contracts) table
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS kalshi_markets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                ticker VARCHAR(100) UNIQUE NOT NULL,
                event_ticker VARCHAR(100) REFERENCES kalshi_events(event_ticker),
                title TEXT NOT NULL,
                subtitle TEXT,
                status VARCHAR(20) DEFAULT 'open',
                yes_bid INTEGER,
                yes_ask INTEGER,
                no_bid INTEGER,
                no_ask INTEGER,
                last_price INTEGER,
                volume BIGINT DEFAULT 0,
                volume_24h BIGINT DEFAULT 0,
                open_interest BIGINT DEFAULT 0,
                liquidity BIGINT DEFAULT 0,
                result VARCHAR(10),
                open_time TIMESTAMP,
                close_time TIMESTAMP,
                expiration_time TIMESTAMP,
                rules_primary TEXT,
                rules_secondary TEXT,
                sport_code VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Price history table
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS kalshi_prices (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                market_ticker VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                yes_bid INTEGER,
                yes_ask INTEGER,
                no_bid INTEGER,
                no_ask INTEGER,
                last_price INTEGER,
                volume BIGINT DEFAULT 0,
                open_interest BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(market_ticker, timestamp)
            )
        """))
        
        # Trades table
        await self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS kalshi_trades (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                trade_id VARCHAR(100) UNIQUE,
                market_ticker VARCHAR(100) NOT NULL,
                price INTEGER NOT NULL,
                count INTEGER DEFAULT 1,
                taker_side VARCHAR(10),
                created_time TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create indexes (must execute separately for asyncpg)
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_events_series ON kalshi_events(series_ticker)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_events_sport ON kalshi_events(sport_code)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_markets_event ON kalshi_markets(event_ticker)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_markets_sport ON kalshi_markets(sport_code)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_prices_market ON kalshi_prices(market_ticker)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_prices_time ON kalshi_prices(timestamp)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_trades_market ON kalshi_trades(market_ticker)"))
        await self.db.execute(text("CREATE INDEX IF NOT EXISTS idx_kalshi_trades_time ON kalshi_trades(created_time)"))
        
        await self.db.commit()
        logger.info("[Kalshi] Tables created successfully")
    
    async def _collect_impl(self, **kwargs) -> CollectorResult:
        """Implementation of data collection"""
        years_back = kwargs.get('years_back', 10)
        
        await self._ensure_tables()
        
        stats = {
            "series": 0,
            "events": 0,
            "markets": 0,
            "prices": 0,
            "trades": 0,
            "errors": []
        }
        
        try:
            # Step 1: Collect series
            logger.info("[Kalshi] Collecting series...")
            series_count = await self._collect_series()
            stats["series"] = series_count
            
            # Step 2: Collect events (with nested markets)
            logger.info("[Kalshi] Collecting events and markets...")
            event_count, market_count = await self._collect_events_and_markets(years_back)
            stats["events"] = event_count
            stats["markets"] = market_count
            
            # Step 3: Collect historical prices (candlesticks)
            logger.info("[Kalshi] Collecting price history...")
            price_count = await self._collect_price_history(years_back)
            stats["prices"] = price_count
            
            # Step 4: Collect trade history
            logger.info("[Kalshi] Collecting trades...")
            trade_count = await self._collect_trades()
            stats["trades"] = trade_count
            
            await self.db.commit()
            
            logger.info(f"[Kalshi] Collection complete: {stats}")
            
            return CollectorResult(
                success=len(stats["errors"]) == 0,
                data=stats,
                records_count=sum([stats["series"], stats["events"], stats["markets"], stats["prices"], stats["trades"]]),
                error="; ".join(stats["errors"]) if stats["errors"] else None
            )
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"[Kalshi] Collection error: {e}")
            stats["errors"].append(str(e))
            return CollectorResult(
                success=False,
                data=stats,
                records_count=0,
                error=str(e)
            )
    
    async def _collect_series(self) -> int:
        """Collect series (groups of related events)"""
        count = 0
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                cursor = None
                
                while True:
                    params = {"limit": 200}
                    if cursor:
                        params["cursor"] = cursor
                        
                    response = await client.get(
                        f"{self.BASE_URL}/series",
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    series_list = data.get("series", [])
                    
                    for series in series_list:
                        ticker = series.get("ticker", "")
                        title = series.get("title", "")
                        category = series.get("category", "")
                        tags = series.get("tags") or []  # Handle None
                        
                        # Check if sports-related
                        sport_code = self._detect_sport(title, category, ticker, tags)
                        
                        if sport_code or self._is_sports_series(ticker, title, category, tags):
                            try:
                                await self._save_series(
                                    ticker=ticker,
                                    title=title,
                                    category=category,
                                    tags=tags,
                                    frequency=series.get("frequency"),
                                    sport_code=sport_code
                                )
                                count += 1
                            except Exception as e:
                                logger.warning(f"[Kalshi] Series save error for {ticker}: {e}")
                                await self.db.rollback()
                                continue
                    
                    # Commit after each page
                    await self.db.commit()
                    
                    cursor = data.get("cursor", "")
                    if not cursor or not series_list:
                        break
                        
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"[Kalshi] Series collection error: {e}")
            
        return count
    
    async def _collect_events_and_markets(self, years_back: int = 10) -> Tuple[int, int]:
        """Collect events with nested markets"""
        event_count = 0
        market_count = 0
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * years_back)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Collect by status: open, closed, settled
                for status in ["open", "closed", "settled"]:
                    cursor = None
                    
                    while True:
                        params = {
                            "limit": 200,
                            "with_nested_markets": "true",
                            "status": status
                        }
                        if cursor:
                            params["cursor"] = cursor
                            
                        response = await client.get(
                            f"{self.BASE_URL}/events",
                            params=params
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        events = data.get("events", [])
                        
                        for event in events:
                            # Check if sports-related
                            title = event.get("title") or ""
                            category = event.get("category") or ""
                            series_ticker = event.get("series_ticker") or ""
                            
                            sport_code = self._detect_sport(title, category, series_ticker, [])
                            
                            if not sport_code and not self._is_sports_event(event):
                                continue
                            
                            # Save event
                            event_ticker = event.get("event_ticker") or ""
                            try:
                                await self._save_event(
                                    event_ticker=event_ticker,
                                    series_ticker=series_ticker,
                                    title=title,
                                    sub_title=event.get("sub_title"),
                                    category=category,
                                    sport_code=sport_code,
                                    strike_date=event.get("strike_date"),
                                    mutually_exclusive=event.get("mutually_exclusive", True)
                                )
                                event_count += 1
                            except Exception as e:
                                logger.warning(f"[Kalshi] Event save error for {event_ticker}: {e}")
                                await self.db.rollback()
                                continue
                            
                            # Save nested markets
                            markets = event.get("markets") or []
                            for market in markets:
                                try:
                                    await self._save_market(market, event_ticker, sport_code)
                                    market_count += 1
                                except Exception as e:
                                    logger.warning(f"[Kalshi] Market save error: {e}")
                                    await self.db.rollback()
                                    continue
                        
                        cursor = data.get("cursor", "")
                        if not cursor or not events:
                            break
                            
                        await asyncio.sleep(0.3)
                        
                    # Commit after each status batch
                    await self.db.commit()
                    logger.info(f"[Kalshi] Collected {status} events: {event_count}")
                    
        except Exception as e:
            logger.error(f"[Kalshi] Events collection error: {e}")
            
        return event_count, market_count
    
    async def _collect_price_history(self, years_back: int = 10) -> int:
        """Collect historical price data using candlesticks API"""
        count = 0
        
        try:
            # Get all sports market tickers
            result = await self.db.execute(text("""
                SELECT ticker FROM kalshi_markets 
                WHERE sport_code IS NOT NULL
                ORDER BY volume DESC
                LIMIT 1000
            """))
            tickers = [row[0] for row in result.fetchall()]
            
            if not tickers:
                logger.info("[Kalshi] No sports markets found for price history")
                return 0
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    try:
                        # Get candlestick data
                        params = {
                            "ticker": ticker,
                            "period_interval": 1440  # Daily candles
                        }
                        
                        response = await client.get(
                            f"{self.BASE_URL}/markets/{ticker}/candlesticks",
                            params=params
                        )
                        
                        if response.status_code != 200:
                            continue
                            
                        data = response.json()
                        candles = data.get("candlesticks", [])
                        
                        for candle in candles:
                            ts = candle.get("end_period_ts")
                            if ts:
                                timestamp = datetime.fromtimestamp(ts)
                                await self._save_price(
                                    market_ticker=ticker,
                                    timestamp=timestamp,
                                    yes_bid=candle.get("yes_bid"),
                                    yes_ask=candle.get("yes_ask"),
                                    last_price=candle.get("close"),
                                    volume=candle.get("volume", 0),
                                    open_interest=candle.get("open_interest", 0)
                                )
                                count += 1
                        
                        await asyncio.sleep(0.2)
                        
                    except Exception as e:
                        logger.warning(f"[Kalshi] Price history error for {ticker}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"[Kalshi] Price history collection error: {e}")
            
        return count
    
    async def _collect_trades(self) -> int:
        """Collect historical trade data"""
        count = 0
        
        try:
            # Get sports market tickers
            result = await self.db.execute(text("""
                SELECT ticker FROM kalshi_markets 
                WHERE sport_code IS NOT NULL
                ORDER BY volume DESC
                LIMIT 500
            """))
            tickers = [row[0] for row in result.fetchall()]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    try:
                        cursor = None
                        ticker_count = 0
                        
                        while ticker_count < 10000:  # Limit per ticker
                            params = {"ticker": ticker, "limit": 1000}
                            if cursor:
                                params["cursor"] = cursor
                                
                            response = await client.get(
                                f"{self.BASE_URL}/markets/trades",
                                params=params
                            )
                            
                            if response.status_code != 200:
                                break
                                
                            data = response.json()
                            trades = data.get("trades", [])
                            
                            for trade in trades:
                                await self._save_trade(
                                    trade_id=trade.get("trade_id"),
                                    market_ticker=trade.get("ticker", ticker),
                                    price=trade.get("yes_price", 0),
                                    count=trade.get("count", 1),
                                    taker_side=trade.get("taker_side"),
                                    created_time=trade.get("created_time")
                                )
                                count += 1
                                ticker_count += 1
                            
                            cursor = data.get("cursor", "")
                            if not cursor or not trades:
                                break
                                
                            await asyncio.sleep(0.3)
                            
                    except Exception as e:
                        logger.warning(f"[Kalshi] Trade collection error for {ticker}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"[Kalshi] Trade collection error: {e}")
            
        return count
    
    async def _save_series(self, ticker: str, title: str, category: str,
                          tags: List[str], frequency: Optional[str], 
                          sport_code: Optional[str]):
        """Save or update series"""
        import json
        # Handle None tags
        tags = tags or []
        await self.db.execute(text("""
            INSERT INTO kalshi_series (ticker, title, category, tags, frequency, sport_code, updated_at)
            VALUES (:ticker, :title, :category, :tags, :frequency, :sport_code, NOW())
            ON CONFLICT (ticker) DO UPDATE SET
                title = :title,
                category = :category,
                tags = :tags,
                frequency = :frequency,
                sport_code = :sport_code,
                updated_at = NOW()
        """), {
            "ticker": ticker,
            "title": title,
            "category": category,
            "tags": json.dumps(tags),
            "frequency": frequency,
            "sport_code": sport_code
        })
    
    async def _save_event(self, event_ticker: str, series_ticker: str, title: str,
                         sub_title: Optional[str], category: str, sport_code: Optional[str],
                         strike_date: Optional[str], mutually_exclusive: bool):
        """Save or update event"""
        # Parse strike_date and remove timezone
        strike_dt = None
        if strike_date:
            try:
                strike_dt = datetime.fromisoformat(strike_date.replace("Z", "+00:00"))
                if strike_dt.tzinfo is not None:
                    strike_dt = strike_dt.replace(tzinfo=None)
            except:
                pass
                
        await self.db.execute(text("""
            INSERT INTO kalshi_events (event_ticker, series_ticker, title, sub_title, 
                                       category, sport_code, strike_date, mutually_exclusive, updated_at)
            VALUES (:event_ticker, :series_ticker, :title, :sub_title, 
                    :category, :sport_code, :strike_date, :mutually_exclusive, NOW())
            ON CONFLICT (event_ticker) DO UPDATE SET
                series_ticker = :series_ticker,
                title = :title,
                sub_title = :sub_title,
                category = :category,
                sport_code = :sport_code,
                strike_date = :strike_date,
                mutually_exclusive = :mutually_exclusive,
                updated_at = NOW()
        """), {
            "event_ticker": event_ticker,
            "series_ticker": series_ticker,
            "title": title,
            "sub_title": sub_title,
            "category": category,
            "sport_code": sport_code,
            "strike_date": strike_dt,
            "mutually_exclusive": mutually_exclusive
        })
    
    async def _save_market(self, market: Dict, event_ticker: str, sport_code: Optional[str]):
        """Save or update market"""
        ticker = market.get("ticker", "")
        
        # Parse timestamps
        open_time = self._parse_timestamp(market.get("open_time"))
        close_time = self._parse_timestamp(market.get("close_time"))
        expiration_time = self._parse_timestamp(market.get("expiration_time"))
        
        await self.db.execute(text("""
            INSERT INTO kalshi_markets (
                ticker, event_ticker, title, subtitle, status,
                yes_bid, yes_ask, no_bid, no_ask, last_price,
                volume, volume_24h, open_interest, liquidity, result,
                open_time, close_time, expiration_time,
                rules_primary, rules_secondary, sport_code, updated_at
            )
            VALUES (
                :ticker, :event_ticker, :title, :subtitle, :status,
                :yes_bid, :yes_ask, :no_bid, :no_ask, :last_price,
                :volume, :volume_24h, :open_interest, :liquidity, :result,
                :open_time, :close_time, :expiration_time,
                :rules_primary, :rules_secondary, :sport_code, NOW()
            )
            ON CONFLICT (ticker) DO UPDATE SET
                event_ticker = :event_ticker,
                title = :title,
                subtitle = :subtitle,
                status = :status,
                yes_bid = :yes_bid,
                yes_ask = :yes_ask,
                no_bid = :no_bid,
                no_ask = :no_ask,
                last_price = :last_price,
                volume = :volume,
                volume_24h = :volume_24h,
                open_interest = :open_interest,
                liquidity = :liquidity,
                result = :result,
                open_time = :open_time,
                close_time = :close_time,
                expiration_time = :expiration_time,
                rules_primary = :rules_primary,
                rules_secondary = :rules_secondary,
                sport_code = :sport_code,
                updated_at = NOW()
        """), {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": market.get("title", ""),
            "subtitle": market.get("subtitle"),
            "status": market.get("status", "open"),
            "yes_bid": market.get("yes_bid"),
            "yes_ask": market.get("yes_ask"),
            "no_bid": market.get("no_bid"),
            "no_ask": market.get("no_ask"),
            "last_price": market.get("last_price"),
            "volume": market.get("volume", 0),
            "volume_24h": market.get("volume_24h", 0),
            "open_interest": market.get("open_interest", 0),
            "liquidity": market.get("liquidity", 0),
            "result": market.get("result"),
            "open_time": open_time,
            "close_time": close_time,
            "expiration_time": expiration_time,
            "rules_primary": market.get("rules_primary"),
            "rules_secondary": market.get("rules_secondary"),
            "sport_code": sport_code
        })
    
    async def _save_price(self, market_ticker: str, timestamp: datetime,
                         yes_bid: Optional[int], yes_ask: Optional[int],
                         last_price: Optional[int], volume: int, open_interest: int):
        """Save price snapshot"""
        await self.db.execute(text("""
            INSERT INTO kalshi_prices (market_ticker, timestamp, yes_bid, yes_ask, 
                                       last_price, volume, open_interest)
            VALUES (:market_ticker, :timestamp, :yes_bid, :yes_ask,
                    :last_price, :volume, :open_interest)
            ON CONFLICT (market_ticker, timestamp) DO UPDATE SET
                yes_bid = :yes_bid,
                yes_ask = :yes_ask,
                last_price = :last_price,
                volume = :volume,
                open_interest = :open_interest
        """), {
            "market_ticker": market_ticker,
            "timestamp": timestamp,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": last_price,
            "volume": volume,
            "open_interest": open_interest
        })
    
    async def _save_trade(self, trade_id: Optional[str], market_ticker: str,
                         price: int, count: int, taker_side: Optional[str],
                         created_time: Optional[str]):
        """Save trade record"""
        created_dt = self._parse_timestamp(created_time) or datetime.utcnow()
        
        # Generate trade_id if not provided
        if not trade_id:
            trade_id = f"{market_ticker}_{created_dt.timestamp()}_{price}"
            
        try:
            await self.db.execute(text("""
                INSERT INTO kalshi_trades (trade_id, market_ticker, price, count, 
                                           taker_side, created_time)
                VALUES (:trade_id, :market_ticker, :price, :count, 
                        :taker_side, :created_time)
                ON CONFLICT (trade_id) DO NOTHING
            """), {
                "trade_id": trade_id,
                "market_ticker": market_ticker,
                "price": price,
                "count": count,
                "taker_side": taker_side,
                "created_time": created_dt
            })
        except Exception as e:
            # Ignore duplicate errors
            pass
    
    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp and return timezone-naive datetime"""
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Convert to naive datetime (remove timezone)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except:
            return None
    
    def _detect_sport(self, title: str, category: str, series_ticker: str, 
                     tags: List[str]) -> Optional[str]:
        """Detect sport code from event metadata"""
        # Handle None values
        tags = tags or []
        title = title or ""
        category = category or ""
        series_ticker = series_ticker or ""
        
        text = f"{title} {category} {series_ticker} {' '.join(tags)}".lower()
        
        for keyword, sport_code in self.SPORT_MAPPING.items():
            if keyword in text:
                return sport_code
                
        return None
    
    def _is_sports_series(self, ticker: str, title: str, category: str, 
                         tags: List[str]) -> bool:
        """Check if series is sports-related"""
        # Handle None values
        ticker = ticker or ""
        title = title or ""
        category = category or ""
        tags = tags or []
        
        # Check category
        if category and category.lower() in ["sports", "sport"]:
            return True
            
        # Check tags
        sports_tags = {"sports", "nfl", "nba", "mlb", "nhl", "football", 
                      "basketball", "baseball", "hockey", "soccer", "golf",
                      "tennis", "mma", "ufc"}
        if tags:
            if any(t.lower() in sports_tags for t in tags if t):
                return True
        
        # Check ticker prefix
        ticker_upper = ticker.upper()
        for prefix in self.SPORTS_SERIES_PREFIXES:
            if ticker_upper.startswith(prefix):
                return True
                
        # Check title keywords
        title_lower = title.lower()
        sports_keywords = ["super bowl", "world series", "stanley cup", 
                         "nba finals", "march madness", "championship",
                         "playoff", "nfl", "nba", "mlb", "nhl", "pga",
                         "masters", "wimbledon", "us open"]
        return any(kw in title_lower for kw in sports_keywords)
    
    def _is_sports_event(self, event: Dict) -> bool:
        """Check if event is sports-related"""
        title = (event.get("title") or "").lower()
        category = (event.get("category") or "").lower()
        series_ticker = (event.get("series_ticker") or "").upper()
        
        # Category check
        if "sport" in category:
            return True
            
        # Series prefix check
        for prefix in self.SPORTS_SERIES_PREFIXES:
            if series_ticker.startswith(prefix):
                return True
        
        # Title keyword check
        sports_keywords = [
            "super bowl", "nfl", "nba", "mlb", "nhl", "mls",
            "world series", "stanley cup", "finals", "playoff",
            "championship", "march madness", "ncaa", "college",
            "pga", "masters", "golf", "tennis", "wimbledon",
            "ufc", "mma", "boxing"
        ]
        return any(kw in title for kw in sports_keywords)


# Factory function
def get_collector(db: AsyncSession, config: Optional[Dict] = None) -> KalshiCollector:
    return KalshiCollector(db, config)